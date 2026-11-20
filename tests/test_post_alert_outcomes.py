"""scripts/post_alert_outcomes.py: lead sign, p(t) edges, missing trades, determinism, skips."""

import importlib.util
import json
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from idsai.contracts.frames import TRADES

S = 1_000_000
DAY_US = 86_400 * S
DAY = date(2020, 9, 10)
T0 = (DAY - date(1970, 1, 1)).days * DAY_US + 12 * 3600 * S  # noon of DAY

_path = Path(__file__).resolve().parents[1] / "scripts" / "post_alert_outcomes.py"
_spec = importlib.util.spec_from_file_location("post_alert_outcomes", _path)
pao = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pao)


def price_fn(t: np.ndarray) -> np.ndarray:
    """1.0 before T0, linear to 1.2 at T0+60 s, 1.1 up to T0+600 s, 0.9 after."""
    d = (t - T0) / S
    return np.where(d < 0, 1.0, np.where(d <= 60, 1.0 + 0.2 * d / 60, np.where(d <= 600, 1.1, 0.9)))


def write_trades(data_dir: Path, symbol: str, days: list[date]) -> None:
    """A trade every 10 s on each day, plus 1 s taker buys of qty 100 during the ramp."""
    for day in days:
        start = (day - date(1970, 1, 1)).days * DAY_US
        ts = start + np.arange(0, 86_400, 10, dtype=np.int64) * S
        buys = T0 + np.arange(0, 61, dtype=np.int64) * S
        buys = buys[(buys >= start) & (buys < start + DAY_US)]
        ts_all = np.concatenate([ts, buys])
        qty = np.concatenate([np.ones(ts.size), np.full(buys.size, 100.0)])
        side = ["buy" if i % 2 else "sell" for i in range(ts.size)] + ["buy"] * buys.size
        order = np.argsort(ts_all, kind="stable")
        ts_all, qty = ts_all[order], qty[order]
        side = [side[i] for i in order]
        price = price_fn(ts_all)
        n = ts_all.size
        df = pl.DataFrame(
            {
                "ts_us": ts_all,
                "recv_us": ts_all,
                "symbol": [symbol] * n,
                "price": price,
                "qty": qty,
                "quote_qty": price * qty,
                "taker_side": side,
                "trade_id": np.arange(1, n + 1, dtype=np.int64),
                "n_fills": np.ones(n, dtype=np.int32),
                "source": ["archive"] * n,
            },
            schema=TRADES,
        )
        p = pao.mp.part_path(data_dir, symbol, day)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(p)


def ev(symbol: str, detected: bool, ttd: float | None) -> dict:
    return {
        "symbol": symbol,
        "day": DAY.isoformat(),
        "family": "telegram_pump",
        "detected_any": detected,
        "detected_correct": detected,
        "ttd_s": ttd,
    }


def make_db(path: Path, records: list[dict], labels: list[tuple]) -> None:
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE runs(run_id VARCHAR, experiment VARCHAR, variant VARCHAR, seed BIGINT, "
        "created_us BIGINT, metrics_json VARCHAR, record_json VARCHAR)"
    )
    con.execute("CREATE TABLE labels(label_id VARCHAR, symbol VARCHAR, t_start_us BIGINT)")
    for i, r in enumerate(records):
        con.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
            [f"r{i}", r["experiment"], r["variant"], r["seed"], i, "{}", json.dumps(r)],
        )
    con.executemany("INSERT INTO labels VALUES (?, ?, ?)", labels)
    con.close()


def rec(seed: int, per_event: dict, config: str = "new", budget: float = 2.0, metrics=None):
    return {
        "experiment": "real_t001",
        "variant": "full",
        "budget": budget,
        "seed": seed,
        "config_hash": config,
        "metrics": {"recall_any": 1.0} if metrics is None else metrics,
        "per_event": per_event,
    }


def setup(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    write_trades(data_dir, "AAABTC", [DAY - timedelta(days=1), DAY, DAY + timedelta(days=1)])
    write_trades(data_dir, "BBBBTC", [DAY - timedelta(days=1), DAY])  # no next day: 24h missing
    per0 = {
        "A": ev("AAABTC", True, 30.0),
        "B": ev("BBBBTC", True, 30.0),
        "C": ev("CCC", False, None),
    }
    per1 = {
        "A": ev("AAABTC", True, 120.0),
        "B": ev("BBBBTC", False, None),
        "C": ev("CCC", False, None),
    }
    records = [
        rec(0, {"A": ev("AAABTC", True, 9999.0)}, config="old"),
        rec(0, per0),
        rec(1, per1),
        rec(2, per0) | {"metrics": None},  # failed run: skipped
        rec(3, per0, budget=10.0),  # other budget: ignored
    ]
    db = tmp_path / "research.duckdb"
    make_db(db, records, [("A", "AAABTC", T0), ("B", "BBBBTC", T0), ("C", "CCC", T0)])
    return data_dir, db


def run_main(tmp_path: Path, data_dir: Path, db: Path, name: str) -> dict:
    out = tmp_path / name
    argv = ["--research-db", str(db), "--data-dir", str(data_dir), "--out", str(out)]
    assert pao.main([*argv, "--n-boot", "200", "--app-db", str(tmp_path / "none.duckdb")]) == 0
    return json.loads(out.read_text(encoding="utf-8"))


def test_price_at_edges():
    ts = np.array([10, 20, 30], dtype=np.int64) * S
    price = np.array([1.0, 2.0, 3.0])
    assert pao.price_at(ts, price, 20 * S) == 2.0  # trade exactly at t counts
    assert pao.price_at(ts, price, 20 * S - 1) == 1.0
    assert pao.price_at(ts, price, 5 * S) is None  # before the first trade
    assert pao.price_at(ts, price, 30 * S + 3600 * S) == 3.0  # exactly one hour old
    assert pao.price_at(ts, price, 30 * S + 3600 * S + 1) is None  # older than one hour
    assert pao.extremes(ts, price, 10 * S, 30 * S) == (2.0, 3.0)  # (lo, hi]
    assert pao.extremes(ts, price, 30 * S, 40 * S) is None


def test_outcomes_lead_sign_missing_and_skips(tmp_path):
    data_dir, db = setup(tmp_path)
    r = run_main(tmp_path, data_dir, db, "out.json")
    assert r["seeds"] == [0, 1] and r["config_hash"] == "new"
    assert r["label_source"] == "research_db"
    assert r["n_labels"] == 3 and r["n_detected"] == 2 and r["n_detected_pairs"] == 3
    assert r["n_measured"] == 2 and r["excluded"]["status"] == {"measured": 2}
    pairs = {(p["label_id"], p["seed"]): p for p in r["pairs"]}
    assert set(pairs) == {("A", 0), ("A", 1), ("B", 0)}  # undetected pairs skipped

    a0, a1 = pairs[("A", 0)], pairs[("A", 1)]
    assert a0["alert_us"] == T0 + 30 * S
    assert a0["lead_to_peak_s"] == 30.0  # alert before the peak at T0+60 s
    assert a1["lead_to_peak_s"] == -60.0  # alert after the peak
    assert abs(a0["p_alert"] - 1.1) < 1e-12 and abs(a0["p_peak"] - 1.2) < 1e-12
    assert abs(a0["ret_peak"] - (1.2 / 1.1 - 1)) < 1e-12
    assert abs(a0["ret_15m"] - (0.9 / 1.1 - 1)) < 1e-12
    assert abs(a0["max_up_1h"] - (1.2 / 1.1 - 1)) < 1e-12
    assert abs(a0["drawdown_1h"] - (0.9 / 1.1 - 1)) < 1e-12
    assert a0["ret_24h"] is not None
    assert pairs[("B", 0)]["ret_24h"] is None  # no trade in the hour before alert + 24 h
    assert r["missing_pairs"]["ret_24h"] == 1
    assert pairs[("B", 0)]["drawdown_24h"] is None and pairs[("B", 0)]["max_up_24h"] is None

    f = r["frac_alert_before_peak"]
    assert f["n_pairs"] == 3 and abs(f["pair_mean"] - 2 / 3) < 1e-12
    assert f["n_labels"] == 2 and abs(f["label_mean"] - 0.75) < 1e-12  # A: 0.5, B: 1.0
    assert f["ci95"][0] <= f["label_mean"] <= f["ci95"][1]
    assert r["per_seed_median_lead_to_peak_s"] == {"0": 30.0, "1": -60.0}
    assert (tmp_path / "out.md").is_file()


def test_deterministic(tmp_path):
    data_dir, db = setup(tmp_path)
    run_main(tmp_path, data_dir, db, "a.json")
    run_main(tmp_path, data_dir, db, "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()
    assert (tmp_path / "a.md").read_bytes() == (tmp_path / "b.md").read_bytes()


def test_onset_near_label_view(tmp_path):
    data_dir, db = setup(tmp_path)
    r = run_main(tmp_path, data_dir, db, "out.json")
    o = r["onset_near_label"]
    assert o["onset_min_s"] == -120.0 and o["n_pairs_kept"] == 3 and o["n_pairs_dropped"] == 0

    out = tmp_path / "cut.json"
    argv = ["--research-db", str(db), "--data-dir", str(data_dir), "--out", str(out)]
    assert pao.main([*argv, "--n-boot", "200", "--onset-min-s", "60"]) == 0
    o = json.loads(out.read_text(encoding="utf-8"))["onset_near_label"]
    assert o["n_pairs_kept"] == 1 and o["n_pairs_dropped"] == 2  # only (A, seed 1), ttd 120
    assert o["frac_alert_before_peak"]["pair_mean"] == 0.0
    assert o["quantiles"]["lead_to_peak_s"]["q50"] == -60.0
    assert "onset_near_label" in (tmp_path / "cut.md").read_text(encoding="utf-8")


def test_onset_near_label_drops_early_onsets():
    pairs = [
        {"label_id": "X", "seed": 0, "ttd_s": -500.0, "lead_to_peak_s": 560.0, "ret_1h": -0.1},
        {"label_id": "X", "seed": 1, "ttd_s": -120.0, "lead_to_peak_s": 180.0, "ret_1h": 0.1},
        {"label_id": "Y", "seed": 0, "ttd_s": 30.0, "lead_to_peak_s": -10.0, "ret_1h": None},
    ]
    o = pao.onset_near_label(pairs, -120.0, 100)
    assert o["n_pairs_kept"] == 2 and o["n_pairs_dropped"] == 1  # cutoff is inclusive
    assert o["frac_alert_before_peak"]["pair_mean"] == 0.5
    assert o["frac_ret_1h_negative"]["n_pairs"] == 1  # None ret_1h skipped
    assert o["frac_ret_1h_negative"]["pair_mean"] == 0.0
