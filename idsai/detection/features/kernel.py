"""Shared NumPy feature kernel: the only feature math, used by batch and stream drivers.

Conventions (global constraints):
- Bars are 1 s, indexed by bar start; bar index `b` covers `[t0 + b s, t0 + (b+1) s)`.
- A row at `t_end` uses bars `[i_end - W, i_end)` with `i_end = (t_end - t0) / 1 s`; nothing at or
  after `t_end` is read. `close_t = close[i_end - 1]`, `close_{t-W} = close[i_end - W - 1]`.
- Rows are emitted at every `t_end` that is a multiple of `stride_s` with
  `t0 + max(W) <= t_end <= last bar start + 1 s`.
- Baseline samples / refreshes are anchored to the UTC epoch (`t_end % period == 0`), so the
  schedule does not depend on where the data starts.

Reproducibility: there is exactly one code path. `Kernel.extend` (batch, any chunking) and
`Kernel.push_bar` (stream, one bar) both append bars with a sequential `np.cumsum` seeded with
the last prefix value, then call the same `Kernel._emit` per due row in time order. Feeding the
same bars in any chunking therefore gives bit-identical rows. Window sums are differences of
prefix sums anchored at the first bar `t0`; a driver must start at the same first bar as the
batch run to reproduce it bit-for-bit (the baseline schedule itself is start-independent).
"""

import math
from dataclasses import dataclass, fields

import numpy as np
import polars as pl

from idsai.config import FeatureCfg
from idsai.detection.features.registry import (
    REGISTRY,
    WINDOW_BASES,
    feature_columns,
    raw_columns,
    spec_for,
)

S_US = 1_000_000
CUSUM_WINDOW_S = 10
CUSUM_Z_CLIP = 10.0
CUSUM_K = 0.5
CUSUM_MAX = 100.0
KK_WINDOW_S = 1800
KK_LAGS = 24
Z_CLIP = 50.0
IQR_TO_SIGMA = 1.349
FLOOR_LOG = 0.25
FLOOR_RATIO = 0.05
FLOOR_RET_MIN = 1e-4
IMB_EPS = 1e-12
QUANTILES = np.array([0.25, 0.5, 0.75])

# raw_vector packs features in this base order; registry order must match
_RAW_ORDER = (
    "n_trades",
    "quote_vol",
    "buy_quote",
    "rush",
    "pingpong",
    "round_frac",
    "buy_imb",
    "ret",
    "jump",
)
assert WINDOW_BASES == _RAW_ORDER, "registry window-feature order changed; update raw_vector"


# ---------------------------------------------------------------- primitives
def prefix(x: np.ndarray, start=None) -> np.ndarray:
    """Prefix sums along axis 0 with a leading `start` row (default 0): `P[i] = start + sum x[:i]`.

    Integer input -> int64 (exact); float input -> float64, added sequentially in index order.
    """
    x = np.asarray(x)
    dt = np.int64 if np.issubdtype(x.dtype, np.integer) else np.float64
    x = x.astype(dt, copy=False)
    head = np.zeros((1, *x.shape[1:]), dtype=dt)
    if start is not None:
        head[0] = start
    return np.cumsum(np.concatenate([head, x]), axis=0, dtype=dt)


def window_sum(P: np.ndarray, i, w):
    """Sum of x over `[i - w, i)` from its prefix sums."""
    return P[i] - P[i - w]


def transform(name: str, x, quote_unit: float):
    """Transform a raw feature value (or array) before robust z-scoring, per its spec."""
    kind = spec_for(name).transform
    x = np.asarray(x, dtype=np.float64)
    if kind == "log1p":
        return np.log1p(x)
    if kind == "log1p_quote":
        return np.log1p(x / quote_unit)
    return x


def cusum_step(s_prev: float, z: float) -> float:
    """One-sided CUSUM: `min(max(0, s + clip(z, -10, 10) - 0.5), 100)`; NaN z counts as 0."""
    z = float(z)
    if math.isnan(z):
        z = 0.0
    z = min(max(z, -CUSUM_Z_CLIP), CUSUM_Z_CLIP)
    return min(max(0.0, float(s_prev) + z - CUSUM_K), CUSUM_MAX)


def raw_vector(
    pc: np.ndarray,
    pf: np.ndarray,
    close: np.ndarray,
    high: np.ndarray,
    i_end: int,
    windows: np.ndarray,
) -> np.ndarray:
    """Raw window features for the row ending at bar index `i_end` (bars `[i_end - W, i_end)`).

    `pc` = int64 prefix sums of (n_trades, rush_orders, pingpong, n_round), `pf` = float64 prefix
    sums of (buy_quote, sell_quote, quote_vol), both with a leading zero row. Returns a vector in
    `registry.raw_columns` order. Windows longer than `i_end` are NaN. Reads only indices < i_end
    of the bar arrays (and `[:i_end + 1]` of the prefix arrays).
    """
    w = windows
    ok = w <= i_end
    lo = np.where(ok, i_end - w, 0)
    C = (pc[i_end] - pc[lo]).astype(np.float64)
    F = pf[i_end] - pf[lo]
    n, rush, pp, nr = C[:, 0], C[:, 1], C[:, 2], C[:, 3]
    bq, sq, qv = F[:, 0], F[:, 1], F[:, 2]
    tot = bq + sq
    small = tot < IMB_EPS
    imb = np.where(small, 0.0, (bq - sq) / np.where(small, 1.0, tot))
    rf = np.where(n > 0, nr / np.maximum(n, 1.0), 0.0)

    c_end = close[i_end - 1] if i_end >= 1 else np.nan
    if not c_end > 0:  # NaN (no trade yet) or a non-positive price
        c_end = np.nan
    pi = i_end - 1 - w
    c_prev = np.where(pi >= 0, close[np.maximum(pi, 0)], np.nan)
    c_prev = np.where(c_prev > 0, c_prev, np.nan)  # NaN (and non-positive) -> NaN
    hs = high[max(i_end - int(w.max()), 0) : i_end][::-1]
    if len(hs):
        cm = np.maximum.accumulate(hs)  # cm[k] = max(high[i_end-1-k : i_end]); NaN propagates
        hmax = np.where(ok, cm[np.clip(w, 1, len(hs)) - 1], np.nan)
    else:
        hmax = np.full(len(w), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.log(c_end / c_prev)
        jump = hmax / c_prev - 1.0
    raw = np.concatenate([n, qv, bq, rush, pp, rf, imb, ret, jump])
    if not ok.all():
        raw[np.tile(~ok, len(_RAW_ORDER))] = np.nan
    return raw


def kk_row(
    hist_qv: np.ndarray, hist_close: np.ndarray, r: int, step: int, qv_now: float, close_now: float
) -> tuple[float, float]:
    """Kamps-Kleinberg ratios for row `r` against rows `r - k*step`, k = 1..24 (emitted rows only).

    `kk_vol_ratio = quote_vol_1800 / mean(...)`, `kk_price_ratio = close / mean(close ...)`.
    NaN if any point is missing or NaN, or if a mean is not positive.
    """
    if r - KK_LAGS * step < 0:
        return math.nan, math.nan
    idx = r - step * np.arange(1, KK_LAGS + 1)
    m_qv = float(hist_qv[idx].mean())
    m_c = float(hist_close[idx].mean())
    vol = qv_now / m_qv if m_qv > 0 and not math.isnan(qv_now) else math.nan
    price = close_now / m_c if m_c > 0 and not math.isnan(close_now) else math.nan
    return vol, price


# ---------------------------------------------------------------- storage
class _Buf:
    """Append-only growable array (1-D, or 2-D with fixed width)."""

    def __init__(self, dtype, width: int | None = None, cap: int = 1024):
        self.data = np.empty((cap,) if width is None else (cap, width), dtype=dtype)
        self.n = 0

    def extend(self, x: np.ndarray) -> None:
        need = self.n + len(x)
        if need > len(self.data):
            new = np.empty((max(need, 2 * len(self.data)), *self.data.shape[1:]), self.data.dtype)
            new[: self.n] = self.data[: self.n]
            self.data = new
        self.data[self.n : need] = x
        self.n = need

    def view(self) -> np.ndarray:
        return self.data[: self.n]

    def drop_front(self, k: int) -> None:
        if k:
            self.data[: self.n - k] = self.data[k : self.n]
            self.n -= k


@dataclass(frozen=True)
class BarArrays:
    """Column arrays of contiguous 1 s bars (the subset of `bars.BARS` the kernel needs)."""

    t_us: np.ndarray
    n_trades: np.ndarray
    buy_quote: np.ndarray
    sell_quote: np.ndarray
    quote_vol: np.ndarray
    high: np.ndarray
    close: np.ndarray
    rush_orders: np.ndarray
    pingpong: np.ndarray
    n_round: np.ndarray

    _INT = ("t_us", "n_trades", "rush_orders", "pingpong", "n_round")

    @classmethod
    def from_frame(cls, bars: pl.DataFrame) -> "BarArrays":
        cols = {}
        for f in fields(cls):
            s = bars[f.name]
            if f.name in cls._INT:
                if s.null_count():
                    raise ValueError(f"bars column {f.name!r} has nulls")
                cols[f.name] = s.cast(pl.Int64).to_numpy().astype(np.int64, copy=False)
            else:
                cols[f.name] = s.cast(pl.Float64).to_numpy().astype(np.float64, copy=False)
        return cls(**cols)

    def __len__(self) -> int:
        return len(self.t_us)

    def slice(self, i: int, j: int) -> "BarArrays":
        return BarArrays(**{f.name: getattr(self, f.name)[i:j] for f in fields(self)})


# ---------------------------------------------------------------- baseline
class Baseline:
    """Robust per-feature baseline over the prior `baseline_s` of samples (median / IQR)."""

    def __init__(self, cfg: FeatureCfg, floors: np.ndarray):
        self.cfg = cfg
        self._floors = floors  # NaN where the floor is the tick-based return floor
        self._t = _Buf(np.int64)
        self._x = _Buf(np.float64, width=len(floors))
        self._start = 0
        self.med: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.valid: np.ndarray | None = None
        self.tick_rel = math.nan
        self.refreshed_at: int | None = None

    @property
    def n_samples(self) -> int:
        return self._t.n - self._start

    def add(self, t_us: int, xt: np.ndarray) -> None:
        """Store a transformed raw row sampled at `t_us` (NaNs kept)."""
        if self._t.n and t_us <= self._t.data[self._t.n - 1]:
            raise ValueError("baseline samples must be added in increasing time order")
        self._t.extend(np.array([t_us], dtype=np.int64))
        self._x.extend(xt[None, :])

    def refresh(self, t_us: int, rel_recent: np.ndarray) -> None:
        """Recompute stats from samples in `[t_us - baseline_s, t_us)` and the tick size from
        `rel_recent` = `|dclose| / close` of the 1 s bars in the same range."""
        times = self._t.view()
        lo = int(np.searchsorted(times, t_us - self.cfg.baseline_s * S_US, side="left"))
        hi = int(np.searchsorted(times, t_us, side="left"))
        self._start = max(self._start, lo)
        if self._start > 4096 and self._start * 2 > self._t.n:
            self._t.drop_front(self._start)
            self._x.drop_front(self._start)
            hi -= self._start
            self._start = 0
        X = self._x.view()[self._start : hi]
        nf = X.shape[1]
        cnt = np.count_nonzero(~np.isnan(X), axis=0)
        valid = (cnt >= self.cfg.min_baseline_samples) & (cnt > 0)
        q = np.full((3, nf), np.nan)
        dense = valid & (cnt == len(X))
        if dense.any():
            q[:, dense] = np.quantile(X[:, dense], QUANTILES, axis=0, method="linear")
        for j in np.flatnonzero(valid & ~dense):
            q[:, j] = np.nanquantile(X[:, j], QUANTILES, method="linear")
        pos = rel_recent[rel_recent > 0]
        self.tick_rel = float(np.median(pos)) if len(pos) else FLOOR_RET_MIN
        floors = np.where(np.isnan(self._floors), max(FLOOR_RET_MIN, self.tick_rel), self._floors)
        self.med = q[1]
        self.scale = np.maximum((q[2] - q[0]) / IQR_TO_SIGMA, floors)
        self.valid = valid
        self.refreshed_at = t_us

    def z(self, xt: np.ndarray) -> np.ndarray:
        """`clip((x - med) / scale, -50, 50)`; NaN for invalid features, NaN x, or no refresh."""
        if self.med is None:
            return np.full(len(xt), np.nan)
        with np.errstate(invalid="ignore"):
            z = np.clip((xt - self.med) / self.scale, -Z_CLIP, Z_CLIP)
        z[~self.valid] = np.nan
        return z


# ---------------------------------------------------------------- kernel
class Kernel:
    """Streaming-capable feature state for one symbol.

    Feed contiguous 1 s bars with `push_bar` (one) or `extend` (many); every due row is computed
    once, in time order, by `_emit`. Row vectors follow `self.columns`
    (= `registry.feature_columns(cfg)`).
    """

    def __init__(self, cfg: FeatureCfg, quote_unit: float):
        if CUSUM_WINDOW_S not in cfg.windows_s or KK_WINDOW_S not in cfg.windows_s:
            raise ValueError(f"windows_s must contain {CUSUM_WINDOW_S} and {KK_WINDOW_S}")
        if not quote_unit > 0:
            raise ValueError("quote_unit must be > 0")
        self.cfg = cfg
        self.quote_unit = float(quote_unit)
        self.columns = feature_columns(cfg)
        self.raw_cols = raw_columns(cfg)
        self._w = np.array(cfg.windows_s, dtype=np.int64)
        self._max_w = int(self._w.max())
        self._stride_us = cfg.stride_s * S_US
        self._sample_us = cfg.baseline_sample_s * S_US
        self._refresh_us = cfg.baseline_refresh_s * S_US
        kinds = [REGISTRY[c.rsplit("_", 1)[0]] for c in self.raw_cols]
        self._log_idx = np.array([i for i, s in enumerate(kinds) if s.transform == "log1p"])
        self._q_idx = np.array([i for i, s in enumerate(kinds) if s.transform == "log1p_quote"])
        floors = np.array(
            [{"log": FLOOR_LOG, "ratio": FLOOR_RATIO, "ret": np.nan}[s.floor] for s in kinds]
        )
        self.baseline = Baseline(cfg, floors)
        self._i_ret = self.raw_cols.index(f"ret_{CUSUM_WINDOW_S}")
        self._i_imb = self.raw_cols.index(f"buy_imb_{CUSUM_WINDOW_S}")
        self._i_qv = self.raw_cols.index(f"quote_vol_{KK_WINDOW_S}")
        self._kk_step = KK_WINDOW_S // cfg.stride_s

        self.t0_us: int | None = None
        self._pc = _Buf(np.int64, width=4)
        self._pf = _Buf(np.float64, width=3)
        self._pc.extend(np.zeros((1, 4), dtype=np.int64))
        self._pf.extend(np.zeros((1, 3)))
        self._close = _Buf(np.float64)
        self._high = _Buf(np.float64)
        self._rel = _Buf(np.float64)
        self._hist_qv = _Buf(np.float64)
        self._hist_close = _Buf(np.float64)
        self._s_ret = 0.0
        self._s_imb = 0.0

    @property
    def n_bars(self) -> int:
        return self._close.n

    def row_dict(self, values: np.ndarray) -> dict[str, float]:
        return dict(zip(self.columns, values.tolist(), strict=True))

    def push_bar(
        self,
        t_us: int,
        n_trades: int,
        buy_quote: float,
        sell_quote: float,
        quote_vol: float,
        high: float,
        close: float,
        rush_orders: int,
        pingpong: int,
        n_round: int,
    ) -> tuple[int, np.ndarray] | None:
        """Append one bar; return `(t_end_us, row)` if a row ends at `t_us + 1 s`, else None."""
        a = BarArrays(
            t_us=np.array([t_us], dtype=np.int64),
            n_trades=np.array([n_trades], dtype=np.int64),
            buy_quote=np.array([buy_quote], dtype=np.float64),
            sell_quote=np.array([sell_quote], dtype=np.float64),
            quote_vol=np.array([quote_vol], dtype=np.float64),
            high=np.array([high], dtype=np.float64),
            close=np.array([close], dtype=np.float64),
            rush_orders=np.array([rush_orders], dtype=np.int64),
            pingpong=np.array([pingpong], dtype=np.int64),
            n_round=np.array([n_round], dtype=np.int64),
        )
        t_end, rows = self.extend(a)
        return (int(t_end[0]), rows[0]) if len(t_end) else None

    def extend(self, bars: BarArrays) -> tuple[np.ndarray, np.ndarray]:
        """Append contiguous bars; return `(t_end_us[int64], rows[n, len(columns)])` emitted."""
        m = len(bars)
        if m == 0:
            return np.empty(0, dtype=np.int64), np.empty((0, len(self.columns)))
        t = bars.t_us.astype(np.int64, copy=False)
        if self.t0_us is None:
            if t[0] % S_US:
                raise ValueError("bar t_us must be a multiple of 1 s")
            self.t0_us = int(t[0])
        n0 = self.n_bars
        expect = self.t0_us + (n0 + np.arange(m, dtype=np.int64)) * S_US
        if not np.array_equal(t, expect):
            raise ValueError("bars must be contiguous 1 s bars in time order")

        counts = np.column_stack([bars.n_trades, bars.rush_orders, bars.pingpong, bars.n_round])
        flows = np.column_stack([bars.buy_quote, bars.sell_quote, bars.quote_vol])
        self._pc.extend(prefix(counts, start=self._pc.data[self._pc.n - 1])[1:])
        self._pf.extend(prefix(flows, start=self._pf.data[self._pf.n - 1])[1:])
        close = bars.close.astype(np.float64, copy=False)
        prev = np.concatenate([[self._close.data[n0 - 1] if n0 else np.nan], close[:-1]])
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(close - prev) / prev
        self._close.extend(close)
        self._high.extend(bars.high.astype(np.float64, copy=False))
        self._rel.extend(rel)

        t_end = t + S_US
        due = (t_end % self._stride_us == 0) & (t_end >= self.t0_us + self._max_w * S_US)
        i_ends = n0 + 1 + np.flatnonzero(due)
        out = np.empty((len(i_ends), len(self.columns)))
        for r, i_end in enumerate(i_ends.tolist()):
            out[r] = self._emit(i_end)
        return t_end[due], out

    def _emit(self, i_end: int) -> np.ndarray:
        """Row for `t_end = t0 + i_end s`; mutates baseline / CUSUM / KK history (call once, in
        time order). Reads only bars with index < i_end."""
        t_end = self.t0_us + i_end * S_US
        # causal views: bars [0, i_end) and prefix rows [0, i_end]; any look-ahead -> IndexError
        close = self._close.data[:i_end]
        raw = raw_vector(
            self._pc.data[: i_end + 1],
            self._pf.data[: i_end + 1],
            close,
            self._high.data[:i_end],
            i_end,
            self._w,
        )
        xt = raw.copy()
        xt[self._log_idx] = np.log1p(raw[self._log_idx])
        xt[self._q_idx] = np.log1p(raw[self._q_idx] / self.quote_unit)

        bl = self.baseline
        if t_end % self._refresh_us == 0:
            lo = max(i_end - self.cfg.baseline_s, 0)
            bl.refresh(t_end, self._rel.data[lo:i_end])
        z = bl.z(xt)
        if t_end % self._sample_us == 0:
            bl.add(t_end, xt)

        self._s_ret = cusum_step(self._s_ret, z[self._i_ret])
        self._s_imb = cusum_step(self._s_imb, z[self._i_imb])

        qv_now = float(raw[self._i_qv])
        c_now = float(close[i_end - 1])
        kk_v, kk_p = kk_row(
            self._hist_qv.view(),
            self._hist_close.view(),
            self._hist_qv.n,
            self._kk_step,
            qv_now,
            c_now,
        )
        self._hist_qv.extend(np.array([qv_now]))
        self._hist_close.extend(np.array([c_now]))
        return np.concatenate([raw, z, [self._s_ret, self._s_imb, kk_v, kk_p]])


def raw_row(bar_arrays: BarArrays, i_end: int, cfg: FeatureCfg, quote_unit: float) -> dict:
    """Raw (untransformed) window features of the row ending at bar index `i_end` as a dict.

    Convenience for inspection/tests (builds prefix sums over all bars each call). `quote_unit`
    only matters for `transform`; raw values are in quote currency.
    """
    del quote_unit
    a = bar_arrays
    pc = prefix(np.column_stack([a.n_trades, a.rush_orders, a.pingpong, a.n_round]))
    pf = prefix(np.column_stack([a.buy_quote, a.sell_quote, a.quote_vol]))
    w = np.array(cfg.windows_s, dtype=np.int64)
    v = raw_vector(pc, pf, a.close, a.high, i_end, w)
    return dict(zip(raw_columns(cfg), v.tolist(), strict=True))
