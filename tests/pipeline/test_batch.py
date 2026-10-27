import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from idsai.config import load_settings
from idsai.contracts.detector import Detector, Evidence
from idsai.contracts.frames import LABELS, TRADES
from idsai.data.store import DAY_US, Store
from idsai.detection.features.windows import windows_schema
from idsai.detection.registry import build_detectors
from idsai.detection.simulator import Scenario
from idsai.pipeline import batch
from idsai.pipeline.batch import DayResult, load_day_windows, score_windows

S = 1_000_000
H = 3600 * S
SYM = "TESTBTC"
D = date(2025, 3, 2)
D_US = (D - date(1970, 1, 1)).days * DAY_US
N_ROWS_DAY = 86_400 // 10


def _day_us(day: date) -> int:
    return (day - date(1970, 1, 1)).days * DAY_US


def make_trades(first_day: date, n_days: int, symbol: str = SYM, seed: int = 5) -> pl.DataFrame:
    """About 1 trade/s of a BTC-quoted pair: price ~ 0.0005 on a coarse tick, random sizes."""
    rng = np.random.default_rng(seed)
    t0 = _day_us(first_day)
    dur = n_days * DAY_US
    n = rng.poisson(1.0 * dur / S)
    ts = np.sort(t0 + rng.integers(0, dur, n))
    price = np.round(5e-4 + 1e-7 * np.cumsum(rng.choice([-1, 0, 0, 1], n)), 7)
    price = np.maximum(price, 1e-7)
    qty = np.round(rng.lognormal(3.0, 1.0, n), 2) + 0.01
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


@pytest.fixture(scope="module")
def trades_3d() -> pl.DataFrame:
    return make_trades(D - timedelta(days=1), 3)


def make_store(tmp_path: Path, trades: pl.DataFrame) -> Store:
    s = Store(tmp_path / "data", tmp_path / "data" / "app.duckdb")
    s.write_trades(trades)
    return s


@pytest.fixture
def cfg(tmp_path):
    return load_settings(overrides={"paths": {"data_dir": str(tmp_path / "data")}})


@pytest.fixture
def store(tmp_path, trades_3d):
    with make_store(tmp_path, trades_3d) as s:
        yield s


def cache_files(cfg) -> list[Path]:
    return sorted((cfg.paths.data_dir / "features").rglob("*.parquet"))


def burst() -> Scenario:
    return Scenario(
        family="pump_burst",
        technique_id="MM-T001",
        sub_technique="MM-T001.001",
        t_start_us=D_US + 12 * H,
        duration_s=60.0,
        impact=0.2,
        vol_mult=20.0,
        rush_size=4,
        dump_tau_s=300.0,
        seed=3,
        region="train",
    )


# ---------------------------------------------------------------- windows + cache
def test_full_day_has_8640_rows_in_range(store, cfg):
    r = load_day_windows(store, SYM, D, cfg)
    assert isinstance(r, DayResult) and r.skipped is None
    assert (r.symbol, r.day) == (SYM, D)
    assert r.windows.height == N_ROWS_DAY == 8640
    assert r.windows.schema == windows_schema(cfg.features)
    t = r.windows["t_end_us"]
    assert t.min() == D_US and t.max() == D_US + DAY_US - 10 * S
    assert r.labels.schema == LABELS and r.labels.height == 0


def test_skip_missing_day(store, cfg):
    r = load_day_windows(store, SYM, D + timedelta(days=5), cfg)
    assert r.skipped == "missing_day"
    assert r.windows.height == 0 and r.windows.schema == windows_schema(cfg.features)
    assert r.labels.schema == LABELS and r.labels.height == 0


def test_skip_missing_baseline(tmp_path, cfg):
    with make_store(tmp_path, make_trades(D, 1)) as s:
        r = load_day_windows(s, SYM, D, cfg)
    assert r.skipped == "no_baseline"
    assert r.windows.height == 0 and r.windows.schema == windows_schema(cfg.features)


def test_skip_missing_day_reported_before_baseline(store, cfg):
    # both D and D-1 missing -> the day itself is reported first
    assert load_day_windows(store, "OTHERBTC", D, cfg).skipped == "missing_day"


def test_skip_no_trades_with_scenario(tmp_path, cfg):
    empty_prev = pl.DataFrame(schema=TRADES)
    with Store(tmp_path / "data", tmp_path / "data" / "app.duckdb") as s:
        s.write_trades(make_trades(D, 1))
        # write_trades ignores empty frames, so create the empty D-1 partition by hand
        part = s._part_path(SYM, D - timedelta(days=1))
        part.parent.mkdir(parents=True, exist_ok=True)
        empty_prev.write_parquet(part)
        assert s.has_day(SYM, D - timedelta(days=1))
        r = load_day_windows(s, SYM, D, cfg, scenario=burst())
    assert r.skipped == "no_trades"
    assert r.windows.height == 0 and r.labels.height == 0
    assert r.windows.schema == windows_schema(cfg.features)


def test_second_clean_call_hits_cache_and_is_identical(store, cfg, monkeypatch):
    first = load_day_windows(store, SYM, D, cfg)
    assert len(cache_files(cfg)) == 1

    def boom(*a, **k):
        raise AssertionError("compute_windows must not run on a cache hit")

    monkeypatch.setattr(batch, "compute_windows", boom)
    second = load_day_windows(store, SYM, D, cfg)
    assert second.windows.equals(first.windows)  # bit-identical, NaN == NaN
    assert second.windows.schema == windows_schema(cfg.features)
    # values are float32-representable
    col = first.windows["z_quote_vol_60"].drop_nans().to_numpy()
    assert (col.astype(np.float32).astype(np.float64) == col).all()


def test_cache_path_layout(store, cfg):
    load_day_windows(store, SYM, D, cfg)
    (path,) = cache_files(cfg)
    sha0, sha1 = store.partition_sha(SYM, D - timedelta(days=1)), store.partition_sha(SYM, D)
    assert path.parent.name == SYM
    assert path.parent.parent.name.startswith(f"{batch.FEATURE_SET_V}-")
    assert path.parent.parent.parent == cfg.paths.data_dir / "features"
    assert path.name == f"{D.isoformat()}-{sha0}{sha1}.parquet"
    assert not list(path.parent.glob("*.tmp"))
    schema = pl.read_parquet_schema(path)
    assert schema["symbol"] == pl.String and schema["t_end_us"] == pl.Int64
    assert schema["z_quote_vol_60"] == pl.Float32


def test_corrupt_cache_is_recomputed(store, cfg):
    first = load_day_windows(store, SYM, D, cfg)
    (path,) = cache_files(cfg)
    path.write_bytes(b"not a parquet file")
    again = load_day_windows(store, SYM, D, cfg)
    assert again.windows.equals(first.windows)
    assert pl.read_parquet(path).height == N_ROWS_DAY  # overwritten with a good file
    # wrong columns also count as corrupt
    pl.DataFrame({"x": [1]}).write_parquet(path)
    assert load_day_windows(store, SYM, D, cfg).windows.equals(first.windows)
    assert pl.read_parquet(path).height == N_ROWS_DAY


def test_scenario_never_touches_cache(store, cfg, monkeypatch):
    r = load_day_windows(store, SYM, D, cfg, scenario=burst())
    assert r.skipped is None and cache_files(cfg) == []
    # even with a cache present, a scenario call must recompute and leave the cache alone
    load_day_windows(store, SYM, D, cfg)
    (path,) = cache_files(cfg)
    mtime = os.stat(path).st_mtime_ns
    called = []
    real = batch.compute_windows

    def spy(*a, **k):
        called.append(1)
        return real(*a, **k)

    monkeypatch.setattr(batch, "compute_windows", spy)
    load_day_windows(store, SYM, D, cfg, scenario=burst())
    assert called and os.stat(path).st_mtime_ns == mtime and len(cache_files(cfg)) == 1


def test_injected_burst_fires_rule_in_label_window(store, cfg):
    sc = burst()
    r = load_day_windows(store, SYM, D, cfg, scenario=sc)
    assert r.labels.schema == LABELS and r.labels.height == 1
    lab = r.labels.row(0, named=True)
    assert lab["family"] == "pump_burst" and lab["scenario_seed"] == sc.seed
    (rule,) = [d for d in build_detectors(["rules"], cfg) if d.name == "rule.100101"]
    scored = score_windows(r.windows, [rule])
    inside = scored.filter(
        (pl.col("t_end_us") > lab["t_start_us"]) & (pl.col("t_end_us") <= lab["t_end_us"] + 60 * S)
    )
    assert inside["score__rule.100101"].max() > 0
    clean = score_windows(load_day_windows(store, SYM, D, cfg).windows, [rule])
    assert clean["score__rule.100101"].max() < 0


def test_clean_run_returns_overlapping_store_labels(store, cfg):
    def label(i, t0, t1, symbol=SYM):
        return pl.DataFrame(
            {
                "label_id": [i],
                "source": ["hu"],
                "symbol": [symbol],
                "t_start_us": [t0],
                "t_end_us": [t1],
                "technique_id": ["MM-T001"],
                "family": ["pump_burst"],
                "scenario_seed": [None],
                "provenance": ["REAL"],
            },
            schema=LABELS,
        )

    store.write_labels(label("in", D_US + H, D_US + 2 * H))
    store.write_labels(label("edge", D_US - H, D_US + 1))  # overlaps the first microsecond
    store.write_labels(label("before", D_US - 2 * H, D_US - H))
    store.write_labels(label("other", D_US + H, D_US + 2 * H, symbol="XBTC"))
    r = load_day_windows(store, SYM, D, cfg)
    assert sorted(r.labels["label_id"].to_list()) == ["edge", "in"]
    assert r.labels.schema == LABELS


def test_include_next_minutes_adds_rows(store, cfg):
    r = load_day_windows(store, SYM, D, cfg, include_next_minutes=10)
    assert r.windows.height == N_ROWS_DAY + 60
    assert r.windows["t_end_us"].max() == D_US + DAY_US + 600 * S - 10 * S
    base = load_day_windows(store, SYM, D, cfg)
    head = r.windows.filter(pl.col("t_end_us") < D_US + DAY_US)
    assert head.equals(base.windows)
    paths = cache_files(cfg)
    assert len(paths) == 2 and any("-n10-" in p.name for p in paths)
    sha = [store.partition_sha(SYM, D + timedelta(days=k)) for k in (-1, 0, 1)]
    (n10,) = [p for p in paths if "-n10-" in p.name]
    assert n10.name == f"{D.isoformat()}-n10-{''.join(sha)}.parquet"


def test_include_next_minutes_without_next_day_uses_none(tmp_path, cfg):
    with make_store(tmp_path, make_trades(D - timedelta(days=1), 2)) as s:
        r = load_day_windows(s, SYM, D, cfg, include_next_minutes=10)
    assert r.skipped is None and r.windows.height == N_ROWS_DAY + 60
    (p,) = cache_files(cfg)
    assert p.name.endswith("none.parquet")


# ---------------------------------------------------------------- score_windows
class Const(Detector):
    layer = "stat"
    technique_ids = ["ANY"]
    requires: list[str] = []

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def score(self, windows):
        return np.full(windows.height, self.value)

    def evidence(self, row):
        return Evidence()


def test_score_windows_adds_float64_columns():
    w = pl.DataFrame({"symbol": ["A", "A"], "t_end_us": [1, 2], "f": [0.5, 1.5]})
    out = score_windows(w, [Const("a", 1), Const("b", 2.5)])
    assert out.columns == ["symbol", "t_end_us", "f", "score__a", "score__b"]
    assert out.schema["score__a"] == pl.Float64
    assert out["score__b"].to_list() == [2.5, 2.5]
    assert w.columns == ["symbol", "t_end_us", "f"]  # input untouched


def test_score_windows_empty_frame_keeps_score_columns():
    w = pl.DataFrame(schema={"symbol": pl.String, "t_end_us": pl.Int64, "f": pl.Float64})
    out = score_windows(w, [Const("a", 1)])
    assert out.height == 0 and out.schema["score__a"] == pl.Float64


def test_score_windows_duplicate_names_raise():
    w = pl.DataFrame({"t_end_us": [1]})
    with pytest.raises(ValueError, match="duplicate"):
        score_windows(w, [Const("a", 1), Const("a", 2)])


def test_score_windows_length_mismatch_raises():
    class Bad(Const):
        def score(self, windows):
            return np.zeros(windows.height + 1)

    with pytest.raises(ValueError, match="rows"):
        score_windows(pl.DataFrame({"t_end_us": [1]}), [Bad("bad", 0)])
