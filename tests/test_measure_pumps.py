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
    assert block["vol_mult"][0] >= 1.0
