"""Semi-synthetic injection of a pump (burst / stealth) or a wash-like pingpong into real trades.

Everything random comes from `np.random.default_rng(sc.seed)`, so an injection is a pure function
of `(trades, sc, base_quote_per_s)`. The input frame is never modified.

Pump model (`pump_burst`, `pump_stealth`): synthetic taker-buy trades during
`[t_start, t_peak)` add `vol_mult * base` quote per second in expectation (clusters of
`rush_size` fills sharing one millisecond for bursts, single fills for stealth); every real trade
with `ts >= t_start` has its price multiplied by `m(t)`: linear `1 -> 1 + impact` over the
duration, then `1 + impact * exp(-(t - t_peak) / dump_tau)` (`dump_tau <= 0`: back to 1 at once).
Prices are rounded to the symbol's observed tick (never below one tick).

Wash model (`wash_pingpong`): pairs of a taker buy then a taker sell 1-20 ms later, equal round
quantity, at the current real price; real trades are untouched.
"""

import hashlib
import math

import numpy as np
import polars as pl

from idsai.contracts.frames import LABELS, TRADES, conform
from idsai.detection.simulator.guard import assert_injection_allowed
from idsai.detection.simulator.scenarios import Scenario

DAY_US = 86_400 * 1_000_000
FALLBACK_TICK = 1e-8
QUOTE_SIGMA = 0.5  # lognormal sigma of synthetic pump trade sizes
MAX_SYNTH_TRADES = 100_000  # cap on synthetic pump fills (mean trade size grows beyond it)
PAIR_MIN_GAP_US = 1_000
PAIR_MAX_GAP_US = 20_000
PAIR_SLOT_US = PAIR_MAX_GAP_US + 1_000  # one pair never overlaps its neighbour
ROUND_MANTISSAS = (1.0, 2.0, 5.0)


def inject(
    trades: pl.DataFrame, sc: Scenario, base_quote_per_s: float, mode: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return `(new trades, one-row LABELS frame)` with the scenario injected into `trades`."""
    assert_injection_allowed(mode)
    rng = np.random.default_rng(sc.seed)

    real = conform(trades, TRADES).sort(["ts_us", "trade_id"])
    if real.height == 0:
        raise ValueError("cannot inject into an empty trade stream")
    symbols = real["symbol"].unique().to_list()
    if len(symbols) != 1:
        raise ValueError(f"inject needs exactly one symbol, got {symbols}")
    symbol = symbols[0]

    ts = real["ts_us"].to_numpy()
    price = real["price"].to_numpy()
    t0 = int(sc.t_start_us)
    dur_us = int(round(sc.duration_s * 1e6))
    t_peak = t0 + dur_us
    tick = _price_tick(ts, price, t0)
    med_qty, med_quote = _typical_size(ts, real["qty"].to_numpy(), real["quote_qty"].to_numpy(), t0)
    expected_quote = sc.vol_mult * base_quote_per_s * sc.duration_s

    def real_price_at(t: np.ndarray) -> np.ndarray:
        """Unmodified real price of the latest real trade at or before `t` (first trade if none)."""
        return price[np.maximum(np.searchsorted(ts, t, side="right") - 1, 0)]

    if sc.family == "wash_pingpong":
        sim = _pingpong(rng, sc, t0, dur_us, expected_quote, med_qty, ts, price, real_price_at)
        out_real = real
    else:
        sim = _pump(rng, sc, t0, dur_us, expected_quote, med_quote, tick, real_price_at)
        out_real = _reprice_real(real, ts, price, sc, tick)

    # ids below every id already present, so chained injections never collide
    first_id = min(0, int(real["trade_id"].min())) - 1
    sim = _finish_sim(sim, symbol, first_id)
    out = pl.concat([out_real, sim]).sort(["ts_us", "trade_id"])
    return out, _label(sc, symbol, t0, t_peak)


# ------------------------------------------------------------------ helpers
def _price_tick(ts: np.ndarray, price: np.ndarray, t0: int) -> float:
    """Smallest positive gap between distinct prices seen in the 24 h before `t0`."""
    lo = np.searchsorted(ts, t0 - DAY_US, side="left")
    hi = np.searchsorted(ts, t0, side="left")
    distinct = np.unique(price[lo:hi])
    if len(distinct) < 2:
        return FALLBACK_TICK
    return float(np.diff(distinct).min())


def _typical_size(
    ts: np.ndarray, qty: np.ndarray, quote: np.ndarray, t0: int
) -> tuple[float, float]:
    """Median trade qty and quote over the 24 h before `t0` (all trades if that is empty)."""
    lo = np.searchsorted(ts, t0 - DAY_US, side="left")
    hi = np.searchsorted(ts, t0, side="left")
    if hi <= lo:
        lo, hi = 0, len(ts)
    return float(np.median(qty[lo:hi])), float(np.median(quote[lo:hi]))


def _multiplier(ts_us: np.ndarray, sc: Scenario) -> np.ndarray:
    """Price multiplier `m(t)` for trades at `ts_us >= t_start`."""
    t = (ts_us - sc.t_start_us) / 1e6
    dur = sc.duration_s
    ramp = 1.0 + sc.impact * np.clip(t / dur, 0.0, 1.0)
    if sc.dump_tau_s > 0:
        post = 1.0 + sc.impact * np.exp(-np.clip(t - dur, 0.0, None) / sc.dump_tau_s)
    else:
        post = np.ones_like(t)
    return np.where(t < dur, ramp, post)


def _snap(p: np.ndarray, tick: float) -> np.ndarray:
    return np.maximum(np.rint(p / tick) * tick, tick)


def _reprice_real(
    real: pl.DataFrame, ts: np.ndarray, price: np.ndarray, sc: Scenario, tick: float
) -> pl.DataFrame:
    """Apply `m(t)` to real trades with `ts >= t_start`; keep `quote_qty = price * qty`."""
    if sc.impact == 0:
        return real
    m = np.ones(len(ts))
    after = ts >= sc.t_start_us
    m[after] = _multiplier(ts[after], sc)
    changed = after & (m != 1.0)
    if not changed.any():
        return real
    new_price = price.copy()
    new_price[changed] = _snap(price[changed] * m[changed], tick)
    qty = real["qty"].to_numpy()
    new_quote = real["quote_qty"].to_numpy().copy()
    new_quote[changed] = new_price[changed] * qty[changed]
    return real.with_columns(pl.Series("price", new_price), pl.Series("quote_qty", new_quote))


def _pump(
    rng: np.random.Generator,
    sc: Scenario,
    t0: int,
    dur_us: int,
    expected_quote: float,
    med_quote: float,
    tick: float,
    real_price_at,
) -> dict[str, np.ndarray]:
    """Synthetic taker-buy fills in `[t0, t0 + dur)`, in clusters of `rush_size` per millisecond."""
    rush = 1 if sc.family == "pump_stealth" else max(1, int(sc.rush_size))
    mean_quote = max(med_quote, expected_quote / MAX_SYNTH_TRADES)
    ms_lo = -(-t0 // 1000)
    ms_hi = (t0 + dur_us) // 1000 - 1  # the whole millisecond lies inside the window
    n_ms = ms_hi - ms_lo + 1
    if n_ms < 1:
        raise ValueError("pump duration is shorter than one millisecond")
    n_clusters = min(max(1, round(expected_quote / (mean_quote * rush))), n_ms)
    ms = np.sort(rng.choice(n_ms, size=n_clusters, replace=False)) + ms_lo
    # `rush` distinct microseconds inside each millisecond (stars-and-bars, sorted)
    offs = np.sort(rng.integers(0, 1000 - rush + 1, size=(n_clusters, rush)), axis=1)
    offs = offs + np.arange(rush)
    ts = (ms[:, None] * 1000 + offs).ravel().astype(np.int64)
    quote = rng.lognormal(math.log(mean_quote) - QUOTE_SIGMA**2 / 2, QUOTE_SIGMA, len(ts))
    quote *= expected_quote / quote.sum()  # exact injected volume, lognormal shape kept
    px = _snap(real_price_at(ts) * _multiplier(ts, sc), tick)
    qty = quote / px
    return {"ts_us": ts, "price": px, "qty": qty, "side": np.full(len(ts), "buy", dtype=object)}


def _next_round(q: float) -> float:
    """Next value above `q` on the {1, 2, 5} x 10^k ladder."""
    k = math.floor(math.log10(q))
    for mant in (*ROUND_MANTISSAS, 10.0):
        cand = mant * 10.0**k
        if cand > q * (1 + 1e-9):
            return cand
    raise AssertionError("unreachable")  # pragma: no cover


def _nearest_round(q: float) -> float:
    """Value on the {1, 2, 5} x 10^k ladder nearest to `q` in log space."""
    k = math.floor(math.log10(q))
    cands = [m * 10.0**j for j in (k - 1, k, k + 1) for m in ROUND_MANTISSAS]
    return min(cands, key=lambda c: abs(math.log(c / q)))


def _pair_count(expected_quote: float, qty: float, price: float) -> int:
    """Pairs needed to reach `expected_quote` (round up: never inject less than expected)."""
    return max(1, math.ceil(expected_quote / (2.0 * qty * price) - 1e-9))


def _pingpong(
    rng: np.random.Generator,
    sc: Scenario,
    t0: int,
    dur_us: int,
    expected_quote: float,
    med_qty: float,
    ts: np.ndarray,
    price: np.ndarray,
    real_price_at,
) -> dict[str, np.ndarray]:
    """Buy-then-sell pairs (1-20 ms apart), equal round qty, current real price."""
    if dur_us < PAIR_SLOT_US:
        raise ValueError("pingpong duration is too short for a single pair")
    if med_qty <= 0:
        raise ValueError("typical trade qty must be positive to size a pingpong pair")
    pre = np.searchsorted(ts, t0, side="left") - 1
    pre_price = float(price[pre] if pre >= 0 else price[0])
    capacity = dur_us // PAIR_SLOT_US
    qty = _nearest_round(med_qty)
    n_pairs = _pair_count(expected_quote, qty, pre_price)
    while n_pairs > capacity:  # more pairs than fit: use a bigger round qty instead
        qty = _next_round(qty)
        n_pairs = _pair_count(expected_quote, qty, pre_price)
    slot = dur_us // n_pairs
    start = t0 + np.arange(n_pairs, dtype=np.int64) * slot
    start = start + rng.integers(0, slot - PAIR_MAX_GAP_US, size=n_pairs)
    gap = rng.integers(PAIR_MIN_GAP_US, PAIR_MAX_GAP_US + 1, size=n_pairs)
    t = np.empty(2 * n_pairs, dtype=np.int64)
    t[0::2], t[1::2] = start, start + gap
    px = np.repeat(real_price_at(start), 2)  # both legs at the same real price
    side = np.empty(2 * n_pairs, dtype=object)
    side[0::2], side[1::2] = "buy", "sell"
    return {"ts_us": t, "price": px, "qty": np.full(2 * n_pairs, qty), "side": side}


def _finish_sim(sim: dict[str, np.ndarray], symbol: str, first_id: int) -> pl.DataFrame:
    """Time-order the synthetic fills, give them ids `first_id, first_id - 1, ...` (TRADES)."""
    order = np.argsort(sim["ts_us"], kind="stable")
    ts = sim["ts_us"][order]
    price = sim["price"][order]
    qty = sim["qty"][order]
    n = len(ts)
    return pl.DataFrame(
        {
            "ts_us": ts,
            "recv_us": ts,
            "symbol": [symbol] * n,
            "price": price,
            "qty": qty,
            "quote_qty": price * qty,
            "taker_side": sim["side"][order].tolist(),
            "trade_id": first_id - np.arange(n, dtype=np.int64),
            "n_fills": np.ones(n, dtype=np.int32),
            "source": ["sim"] * n,
        },
        schema=TRADES,
    )


def _label(sc: Scenario, symbol: str, t0: int, t_peak: int) -> pl.DataFrame:
    key = f"{symbol}|{t0}|{sc.family}|{sc.seed}"
    return pl.DataFrame(
        {
            "label_id": [hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]],
            "source": ["sim"],
            "symbol": [symbol],
            "t_start_us": [t0],
            "t_end_us": [t_peak],
            "technique_id": [sc.technique_id],
            "family": [sc.family],
            "scenario_seed": [sc.seed],
            "provenance": ["SEMI-SYN"],
        },
        schema=LABELS,
    )
