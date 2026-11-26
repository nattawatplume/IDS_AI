"""scripts/diagnose_fa.py pure helpers (rank AUC, Gini, shares) on synthetic arrays."""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_scripts))  # diagnose_fa imports diagnose_market by name
_spec = importlib.util.spec_from_file_location("diagnose_fa", _scripts / "diagnose_fa.py")
fa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fa)


def test_rank_auc_extremes_ties_and_nan():
    assert fa.rank_auc([3, 4, 5], [0, 1, 2]) == 1.0
    assert fa.rank_auc([0, 1, 2], [3, 4, 5]) == 0.0
    assert fa.rank_auc([1, 1], [1, 1, 1]) == 0.5
    # 2 > 1, 2 = 2, 1 < 2, 1 = 1 ... pairwise: (1,1)=.5 (1,2)=0 (2,1)=1 (2,2)=.5 -> 2/4
    assert fa.rank_auc([1, 2], [1, 2]) == 0.5
    assert fa.rank_auc([2, np.nan], [1, np.inf]) == 1.0  # non-finite values dropped
    assert math.isnan(fa.rank_auc([], [1.0]))


def test_rank_auc_matches_pairwise_count():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 5, 40).astype(float)
    b = rng.integers(0, 6, 30).astype(float)
    pair = ((a[:, None] > b[None, :]) + 0.5 * (a[:, None] == b[None, :])).mean()
    assert fa.rank_auc(a, b) == pytest.approx(pair)


def test_gini_and_top_share():
    assert fa.gini([5, 5, 5, 5]) == pytest.approx(0.0)
    assert fa.gini([0, 0, 0, 8]) == pytest.approx(0.75)  # (n - 1) / n when one holds all
    assert math.isnan(fa.gini([0, 0]))
    assert math.isnan(fa.gini([]))
    assert fa.top_share([1, 1, 2, 6], 1) == pytest.approx(0.6)
    assert fa.top_share([1, 1, 2, 6], 5) == pytest.approx(1.0)
    assert math.isnan(fa.top_share([0, 0], 1))


def test_shares_med_iqr_pct_rank():
    s = fa.shares(["a", "b", "a", None], ["a", "b", "c"])
    assert s["a"] == {"n": 2, "share": pytest.approx(2 / 3)}
    assert s["c"]["n"] == 0
    assert math.isnan(fa.shares([None], ["a"])["a"]["share"])
    m = fa.med_iqr([1, 2, 3, 4, 5, None, float("nan")])
    assert (m["n"], m["median"], m["q25"], m["q75"]) == (5, 3.0, 2.0, 4.0)
    assert fa.med_iqr([None])["median"] is None
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    assert fa.pct_rank(2.0, ref) == pytest.approx(0.375)
    assert fa.pct_rank(float("nan"), ref) is None


def test_most_volatile_ties_by_key_order():
    vols = {("B", 1): 2.0, ("A", 2): 2.0, ("A", 1): 1.0, ("C", 1): 3.0}
    assert fa.most_volatile(vols, 2) == {("C", 1), ("A", 2)}
    assert fa.most_volatile(vols, 0) == set()
