"""scripts/measure_pumps.py: an injected pump reads back, only train labels count, read-only."""

import hashlib
import importlib.util
import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from idsai.config import load_settings
from idsai.contracts.frames import LABELS, TRADES
from idsai.data.store import Store
from idsai.detection.simulator import Scenario, base_quote_rate, inject

S = 1_000_000
DAY_US = 86_400 * S
SYM = "PMPBTC"
PUMP_DAY = date(2018, 3, 10)  # train period
TEST_DAY = date(2020, 9, 10)  # test period
TRUTH = {"duration_s": 120.0, "impact": 0.2, "vol_mult": 15.0, "rush_size": 5, "dump_tau_s": 300.0}

_path = Path(__file__).resolve().parents[1] / "scripts" / "measure_pumps.py"
_spec = importlib.util.spec_from_file_location("measure_pumps", _path)
mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mp)


def day_us(day: date) -> int:
    return (day - date(1970, 1, 1)).days * DAY_US


def background(first_day: date, n_days: int, seed: int, symbol: str = SYM) -> pl.DataFrame:
    """About one trade per second, price 1e-3 jittering by one 1e-7 tick, random taker side."""
    rng = np.random.default_rng(seed)
    t0 = day_us(first_day)
    n = n_days * 86_400
    ts = np.sort(t0 + rng.integers(0, n_days * DAY_US, n))
    price = np.round(1e-3 + 1e-7 * rng.integers(-1, 2, n), 7)
    qty = np.round(rng.lognormal(3.0, 0.8, n), 2) + 0.01
    side = np.where(rng.random(n) < 0.5, "buy", "sell")
    return pl.DataFrame(
        {
            "ts_us": ts,
            "recv_us": ts,
            "symbol": [symbol] * n,
            "price": price,
            "qty": qty,
            "quote_qty": price * qty,
            "taker_side": side.tolist(),
            "trade_id": np.arange(1, n + 1, dtype=np.int64),
            "n_fills": np.ones(n, dtype=np.int32),
            "source": ["archive"] * n,
        },
        schema=TRADES,
    )


def label(symbol: str, t0: int) -> pl.DataFrame:
    return pl.DataFrame(
        [(f"{symbol}-{t0}", "sapienza", symbol, t0, t0, "MM-T001", "telegram_pump", None, "REAL")],
        schema=LABELS,
        orient="row",
    )


def pumped(first_day: date, t0: int, seed: int) -> pl.DataFrame:
    """Two days of background with a pump_burst of known TRUTH injected at `t0` (second day)."""
    real = background(first_day, 2, seed)
    prev = real.filter(pl.col("ts_us") < day_us(first_day) + DAY_US)
    sc = Scenario(
        family="pump_burst",
        technique_id="MM-T001",
        sub_technique="MM-T001.001",
        t_start_us=t0,
        duration_s=TRUTH["duration_s"],
        impact=TRUTH["impact"],
        vol_mult=TRUTH["vol_mult"],
        rush_size=TRUTH["rush_size"],
        dump_tau_s=TRUTH["dump_tau_s"],
        seed=11,
        region="train",
    )
    out, _ = inject(real, sc, base_quote_rate(prev), "replay")
    return out


@pytest.fixture
def world(tmp_path):
    """A store with: a train pump (measurable), a test-period pump, a train label without D-1,
    and a train label for a symbol with no data at all."""
    data = tmp_path / "data"
    db = data / "app.duckdb"
    t_train = day_us(PUMP_DAY) + 12 * 3600 * S
    t_test = day_us(TEST_DAY) + 12 * 3600 * S
    t_nobase = day_us(date(2018, 5, 2)) + 10 * 3600 * S
    with Store(data, db) as st:
        st.write_trades(pumped(date(2018, 3, 9), t_train, seed=1))
        st.write_trades(pumped(date(2020, 9, 9), t_test, seed=2))
        st.write_trades(background(date(2018, 5, 2), 1, seed=3))  # no 2018-05-01
        st.write_labels(
            pl.concat(
                [
                    label(SYM, t_train),
                    label(SYM, t_test),
                    label(SYM, t_nobase),
                    label("NODATABTC", t_train),
                ]
            )
        )
    cfg_path = tmp_path / "cfg" / "test.yaml"
    cfg_path.parent.mkdir()
    cfg_path.write_text(
        yaml.safe_dump({"paths": {"data_dir": data.as_posix(), "app_db": db.as_posix()}}),
        encoding="utf-8",
    )
    return {"data": data, "db": db, "cfg": cfg_path, "t_train": t_train}


def _tree_shas(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_injected_pump_reads_back(world):
    cfg = load_settings(world["cfg"])
    res = mp.measure_label(world["data"], SYM, world["t_train"])
    assert res["status"] == "measured"
    assert res["impact"] == pytest.approx(TRUTH["impact"], rel=0.02)
    assert res["duration_s"] == pytest.approx(TRUTH["duration_s"], rel=0.05)
    assert res["vol_mult"] == pytest.approx(TRUTH["vol_mult"], rel=0.05)
    assert res["rush_size"] == TRUTH["rush_size"]
    assert res["dump_tau_s"] == pytest.approx(TRUTH["dump_tau_s"], rel=0.05)
    assert abs(res["t_start_est_us"] - world["t_train"]) <= 5 * S
    assert cfg.splits.train[0] <= PUMP_DAY <= cfg.splits.train[1]


def test_only_train_labels_are_measured(world, tmp_path, capsys):
    out = tmp_path / "out" / "m.json"
    assert mp.main(["--config", str(world["cfg"]), "--out", str(out)]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["label_source"] == "db"
    assert res["n_labels"] == 4
    assert res["n_train_labels"] == 3
    assert res["n_measured"] == 1
    assert res["excluded"]["not_train"] == 1
    assert res["not_train_by_period"] == {"test": 1}
    assert res["excluded"]["no_baseline"] == 1
    assert res["excluded"]["missing_day"] == 1
    assert [r["symbol"] for r in res["labels"] if r["status"] == "measured"] == [SYM]
    block = res["suggested_pump_burst_train"]
    assert set(block) == set(mp.PARAMS)
    assert all(isinstance(v, int) for v in block["rush_size"])
    assert block["impact"][0] <= block["impact"][1]
    printed = capsys.readouterr().out
    assert "not_train 1" in printed and "pump_burst:" in printed
    assert res["flags"] == {"n_duration_censored": 0, "n_dump_censored": 0, "n_overlaps_label": 0}
    assert {"duration_s_incl_censored", "vol_quote_base_s"} <= set(res["quantiles"])


def test_read_only_and_deterministic(world, tmp_path):
    before = _tree_shas(world["data"])
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    mp.main(["--config", str(world["cfg"]), "--out", str(a)])
    mp.main(["--config", str(world["cfg"]), "--out", str(b)])
    assert _tree_shas(world["data"]) == before
    assert a.read_bytes() == b.read_bytes()


def test_labels_fall_back_to_files_without_db(world, tmp_path):
    cfg = load_settings(world["cfg"])
    labels, src = mp.load_labels(cfg, "auto", tmp_path / "missing.duckdb")
    assert src == "files" and labels.height > 0
    with pytest.raises(FileNotFoundError):
        mp.load_labels(cfg, "db", tmp_path / "missing.duckdb")


def test_suggest_floors_and_kept():
    qs = {p: mp.quantiles([]) for p in mp.PARAMS}
    qs["rush_size"] = mp.quantiles([1.0, 2.0, 2.6, 7.4])
    qs["vol_mult"] = mp.quantiles([-1.0, 0.5, 3.0])
    block, kept = mp.suggest_train(qs, {"dump_tau_s": [120, 600]})
    assert block["dump_tau_s"] == [120, 600] and "dump_tau_s" in kept
    lo, hi = block["rush_size"]
    assert isinstance(lo, int) and isinstance(hi, int) and 1 <= lo <= hi
    assert set(block["vol_mult"]) == {"log"} and block["vol_mult"]["log"][0] >= 1.0


# ---- review fixes (Ruling 47): each of these fails on 29d4dca ----
T_PUMP = day_us(PUMP_DAY) + 12 * 3600 * S


def _measure(tmp_path, trades: pl.DataFrame, t0: int = T_PUMP) -> dict:
    data = tmp_path / "d"
    with Store(data, tmp_path / "x.duckdb") as st:
        st.write_trades(trades)
    return mp.measure_label(data, SYM, t0)


def _extra(trades: pl.DataFrame, ts: np.ndarray, quote: np.ndarray, side: str) -> pl.DataFrame:
    """`trades` plus fills at `ts`, each priced like the trade just before it."""
    ts = np.sort(ts.astype(np.int64))
    base_ts = trades["ts_us"].to_numpy()
    px = trades["price"].to_numpy()[np.maximum(np.searchsorted(base_ts, ts, "right") - 1, 0)]
    n = len(ts)
    first = int(trades["trade_id"].max()) + 1
    add = pl.DataFrame(
        {
            "ts_us": ts,
            "recv_us": ts,
            "symbol": [SYM] * n,
            "price": px,
            "qty": quote / px,
            "quote_qty": quote,
            "taker_side": [side] * n,
            "trade_id": np.arange(first, first + n, dtype=np.int64),
            "n_fills": np.ones(n, dtype=np.int32),
            "source": ["archive"] * n,
        },
        schema=TRADES,
    )
    return pl.concat([trades, add]).sort(["ts_us", "trade_id"])


def _ramp_ts(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dur = int(TRUTH["duration_s"] * S)
    return T_PUMP + 20 * S + rng.choice(dur - 40 * S, size=n, replace=False)


def test_rush_size_is_quote_weighted(tmp_path):
    # many tiny retail single buys outnumber the injected 5-fill clusters but carry little quote
    tr = pumped(date(2018, 3, 9), T_PUMP, seed=1)
    med = float(tr["quote_qty"].median())
    tr = _extra(tr, _ramp_ts(4000, 1), np.full(4000, med * 1e-3), "buy")
    res = _measure(tmp_path, tr)
    assert res["status"] == "measured"
    assert res["rush_size"] == TRUTH["rush_size"]


def test_weighted_median():
    assert mp.weighted_median(np.array([1.0, 5.0, 9.0]), np.array([1.0, 10.0, 1.0])) == 5.0
    assert mp.weighted_median(np.array([1.0, 5.0]), np.array([3.0, 1.0])) == 1.0


def test_vol_mult_counts_taker_buys_only(tmp_path):
    # heavy taker selling during the ramp is not what inject adds
    tr = pumped(date(2018, 3, 9), T_PUMP, seed=1)
    prev = tr.filter(pl.col("ts_us") < day_us(PUMP_DAY))
    sell_quote = 10.0 * base_quote_rate(prev) * TRUTH["duration_s"]
    tr = _extra(tr, _ramp_ts(500, 2), np.full(500, sell_quote / 500), "sell")
    res = _measure(tmp_path, tr)
    assert res["status"] == "measured"
    assert res["vol_mult"] == pytest.approx(TRUTH["vol_mult"], rel=0.05)


def test_small_noise_high_is_no_rise(tmp_path):
    # one buy 2 % above the rest, 100 s after the label: above the old 1 % gate
    tr = background(date(2018, 3, 9), 2, seed=4)
    tr = _extra(tr, np.array([T_PUMP + 100 * S]), np.array([1e-3]), "buy")
    spike = pl.col("trade_id") == tr["trade_id"].max()
    tr = tr.with_columns(pl.when(spike).then(1.02e-3).otherwise(pl.col("price")).alias("price"))
    assert _measure(tmp_path, tr)["status"] == "no_rise"


def test_late_peak_is_excluded(tmp_path):
    tr = pumped(date(2018, 3, 9), T_PUMP + 1000 * S, seed=1)
    assert _measure(tmp_path, tr)["status"] == "late_peak"


def _window(prices: list[tuple[float, float, str, float]], t0: int) -> pl.DataFrame:
    """Frame of (offset_s from t0, price, side, quote)."""
    n = len(prices)
    ts = np.array([t0 + int(o * S) for o, _, _, _ in prices], dtype=np.int64)
    px = np.array([p for _, p, _, _ in prices])
    quote = np.array([q for *_, q in prices])
    return pl.DataFrame(
        {
            "ts_us": ts,
            "recv_us": ts,
            "symbol": [SYM] * n,
            "price": px,
            "qty": quote / px,
            "quote_qty": quote,
            "taker_side": [s for _, _, s, _ in prices],
            "trade_id": np.arange(1, n + 1, dtype=np.int64),
            "n_fills": np.ones(n, dtype=np.int32),
            "source": ["archive"] * n,
        },
        schema=TRADES,
    )


def test_rise_without_buying_is_no_volume():
    t0 = T_PUMP
    w = _window([(-299, 1.0, "buy", 1.0), (10, 1.1, "sell", 1.0), (20, 1.0, "buy", 1.0)], t0)
    res = mp.measure_window(w, t0, 1.0, 1.0, 1.0)
    assert res["status"] == "no_volume" and res["vol_mult"] <= 0


def test_duration_censored_at_window_start():
    t0 = T_PUMP
    rows = [(o, 1.1, "buy", 1.0) for o in range(-299, 30, 5)] + [(30, 1.2, "buy", 1000.0)]
    res = mp.measure_window(_window(rows, t0), t0, 1.0, 1.0, 0.0)
    assert res["status"] == "measured" and res["duration_censored"] is True
    ok = mp.measure_window(_window([(-299, 1.0, "buy", 1.0), *rows[-2:]], t0), t0, 1.0, 1.0, 0.0)
    assert ok["duration_censored"] is False


def test_overlap_flags():
    t = T_PUMP
    labs = pl.concat([label(SYM, t), label(SYM, t + 49 * 60 * S), label(SYM, t + 3 * DAY_US)])
    labs = pl.concat([labs, label("OTHERBTC", t)])
    flags = mp.overlap_flags(labs)
    assert flags == {
        f"{SYM}-{t}": True,
        f"{SYM}-{t + 49 * 60 * S}": True,
        f"{SYM}-{t + 3 * DAY_US}": False,
        f"OTHERBTC-{t}": False,
    }
