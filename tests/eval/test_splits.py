from datetime import date, timedelta

import polars as pl
import pytest

from idsai.config import load_settings
from idsai.contracts.frames import LABELS
from idsai.data.store import DAY_US, Store
from idsai.eval.splits import (
    Split,
    SymDay,
    drift_ranges,
    make_split,
    realized_vol_1m,
    split_def,
)
from tests.synth import S, day_us, make_trades

D0 = date(2025, 1, 1)
SYMS = ("AAABTC", "BBBBTC")
VOLATILE = ("AAABTC", date(2025, 1, 20))
SPLITS = {
    "train": ["2025-01-02", "2025-01-08"],
    "cal": ["2025-01-09", "2025-01-12"],
    "accept": ["2025-01-13", "2025-01-16"],
    "test": ["2025-01-17", "2025-01-24"],
    "normal_days_per_symbol": 2,
    "min_gap_days": 2,
    "drift_v1": None,  # the repo yaml's drift ranges lie outside this train period
    "drift_v2": None,
}
PERIOD_OF_LIST = {
    "train_normal": "train",
    "train_events": "train",
    "cal_normal": "cal",
    "accept_events": "accept",
    "accept_normal": "accept",
    "test_events": "test",
    "test_normal": "test",
    "test_hard_negative": "test",
}


def at(day: date, hh: int, mm: int = 0) -> int:
    return day_us(day) + (hh * 3600 + mm * 60) * S


def label_frame(rows) -> pl.DataFrame:
    """rows: (symbol, t_us)"""
    return pl.DataFrame(
        [(f"{s}-{t}", "hu", s, t, t, "MM-T001", "telegram_pump", None, "REAL") for s, t in rows],
        schema=LABELS,
        orient="row",
    )


LABELS_ROWS = [
    ("AAABTC", at(D0, 12)),  # first store day: no D-1, so not an event; still blocks normals
    ("AAABTC", at(date(2025, 1, 4), 23, 55)),  # train event near midnight -> +10 min
    ("AAABTC", at(date(2025, 1, 14), 23, 49)),  # accept event, 11 min before midnight
    ("BBBBTC", at(date(2025, 1, 18), 9)),  # test event
    ("BBBBTC", at(date(2025, 1, 18), 15)),  # same symbol-day: one SymDay
    ("CCCBTC", at(date(2025, 1, 10), 9)),  # symbol without data: no event, no normals
    ("AAABTC", at(date(2026, 1, 1), 9)),  # outside every period
]


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("split")
    with Store(tmp / "data", tmp / "data" / "app.duckdb") as s:
        for k, sym in enumerate(SYMS):
            s.write_trades(make_trades(D0, 24, symbol=sym, seed=k, rate=0.003))
        vol_day = make_trades(VOLATILE[1], 1, symbol=VOLATILE[0], seed=9, rate=0.003, step=2e-5)
        s.write_trades(vol_day)  # replaces that one partition with a far more volatile day
        yield s


def cfg_with(**splits):
    return load_settings(overrides={"splits": {**SPLITS, **splits}, "eval": {"hard_negative_k": 1}})


@pytest.fixture(scope="module")
def cfg():
    return cfg_with()


@pytest.fixture(scope="module")
def split(store, cfg) -> Split:
    return make_split(label_frame(LABELS_ROWS), store, cfg)


def test_lists_sorted_and_inside_their_period(split, cfg):
    for name, days in split.lists().items():
        assert days == sorted(days), name
        lo, hi = getattr(cfg.splits, PERIOD_OF_LIST[name])
        assert all(lo <= s.day <= hi for s in days), name
    periods = [
        [s.day for n, v in split.lists().items() if PERIOD_OF_LIST[n] == p for s in v]
        for p in ("train", "cal", "accept", "test")
    ]
    for a, b in zip(periods, periods[1:], strict=False):
        assert max(a) < min(b)


def test_event_days(split):
    assert split.train_events == [SymDay("AAABTC", date(2025, 1, 4), 10)]
    assert split.accept_events == [SymDay("AAABTC", date(2025, 1, 14), 0)]
    assert split.test_events == [SymDay("BBBBTC", date(2025, 1, 18), 0)]


def test_midnight_window_boundary(store, cfg):
    just_in = at(date(2025, 1, 4), 23, 50)  # exactly 10 min before midnight
    out = make_split(label_frame([("AAABTC", just_in)]), store, cfg)
    assert out.train_events == [SymDay("AAABTC", date(2025, 1, 4), 10)]


def test_no_normal_day_within_gap_of_a_label(split):
    label_days = {}
    for sym, t in LABELS_ROWS:
        label_days.setdefault(sym, []).append(date(1970, 1, 1) + timedelta(days=t // DAY_US))
    normals = [
        *split.train_normal,
        *split.cal_normal,
        *split.accept_normal,
        *split.test_normal,
        *split.test_hard_negative,
    ]
    assert normals
    for s in normals:
        assert all(abs((s.day - d).days) >= 2 for d in label_days[s.symbol]), s
        assert s.include_next_minutes == 0
    assert {s.symbol for s in normals} == set(SYMS)  # CCCBTC has no data
    # D0 has no D-1 partition and its label blocks D0+1 for AAABTC
    assert all(s.day > date(2025, 1, 2) for s in split.train_normal if s.symbol == "AAABTC")


def test_normal_days_per_symbol_and_period(split):
    for name in ("train_normal", "cal_normal", "accept_normal"):
        for sym in SYMS:
            n = sum(s.symbol == sym for s in getattr(split, name))
            assert n == 2 or (name, sym) == ("accept_normal", "AAABTC"), (name, sym)
    # AAABTC's only accept candidate is 01-16 (01-13..01-15 are within 2 days of 01-14)
    assert [s.day for s in split.accept_normal if s.symbol == "AAABTC"] == [date(2025, 1, 16)]
    test = split.test_normal + split.test_hard_negative
    assert len(test) == 4


def test_hard_negatives_are_the_most_volatile_and_disjoint(store, cfg):
    every = cfg_with(normal_days_per_symbol=50)  # take every candidate day
    sp = make_split(label_frame(LABELS_ROWS), store, every)
    assert sp.test_hard_negative == [SymDay(*VOLATILE)]
    assert not set(sp.test_hard_negative) & set(sp.test_normal)
    assert len(sp.test_normal) >= 8


def test_same_seed_same_split_and_seed_matters(store, cfg):
    labs = label_frame(LABELS_ROWS)
    a, b = make_split(labs, store, cfg), make_split(labs, store, cfg, seed=0)
    assert a == b and a.sha12() == b.sha12()
    others = {make_split(labs, store, cfg, seed=k).sha12() for k in range(1, 6)}
    assert others != {a.sha12()}


def test_split_def_records_counts_and_sha(split, cfg):
    d = split_def(split, cfg)
    assert d["split_seed"] == 0 and d["sha12"] == split.sha12() and len(d["sha12"]) == 12
    assert d["counts"]["train_events"] == 1 and d["splits"]["min_gap_days"] == 2


def test_realized_vol_needs_30_minutes():
    tr = make_trades(D0, 1, rate=0.003)
    assert realized_vol_1m(tr) > 0
    assert realized_vol_1m(tr.head(20)) == 0.0
    assert realized_vol_1m(tr.head(0)) == 0.0


def test_drift_ranges_default_and_validation():
    cfg = load_settings()
    assert drift_ranges(cfg.splits) == (
        (date(2018, 1, 1), date(2018, 12, 31)),
        (date(2018, 7, 1), date(2019, 6, 30)),
    )
    sp = cfg_with().splits  # no drift ranges set: two thirds of the 7-day train period
    assert sp.drift_v1 is None
    assert drift_ranges(sp) == (
        (date(2025, 1, 2), date(2025, 1, 6)),
        (date(2025, 1, 4), date(2025, 1, 8)),
    )
    with pytest.raises(ValueError, match="inside train"):
        cfg_with(drift_v1=["2025-01-01", "2025-01-05"])
