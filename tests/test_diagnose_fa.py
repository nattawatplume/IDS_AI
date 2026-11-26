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


# ------------------------------------------------------------------ Rows / profile on a stub stack
class _Det:
    def __init__(self, name, layer, sid=None):
        self.name, self.layer, self.signature_id = name, layer, sid


_DETS = [
    _Det("rule.100102", "rule", 100102),
    _Det("rule.100104", "rule", 100104),
    _Det("stat.robust_z", "stat"),
    _Det("stat.cusum", "stat"),
    _Det("ai.iforest", "ai"),
]


def _stub(p_rows, shift=1.0):
    """A Stack stand-in: one symbol, rows 10 s apart; F = max of the (shifted) -log10 p."""
    from types import SimpleNamespace

    import polars as pl

    p = np.asarray(p_rows, dtype=np.float64)
    nlp = -np.log10(p)
    nlp[:, 4] = np.maximum(0.0, nlp[:, 4] - shift)
    n = p.shape[0]
    cols = {"symbol": ["AAA"] * n, "t_end_us": [(i + 1) * 10 * fa.S_US for i in range(n)]}
    for j, d in enumerate(_DETS):
        cols[f"p__{d.name}"] = p[:, j]
    cols["F"] = nlp.max(axis=1)
    for c in fa.KEEP:
        cols[c] = np.arange(n, dtype=np.float64)
    st = SimpleNamespace(
        fused=pl.DataFrame(cols),
        chief=SimpleNamespace(fused_detectors=_DETS),
        ai=SimpleNamespace(mode="shift", shift=shift),
    )
    return fa.Rows(st)


def test_layer_of():
    assert [fa.layer_of(d) for d in _DETS] == ["rules", "rules", "robust_z", "cusum", "iforest"]


def test_rows_layers_shift_rule_pick_and_profile():
    one = 1.0
    rows = _stub(
        [
            # row 0: rule 100104 and robust_z tie at p = 1e-4 -> two layers over tau = 3
            [one, 1e-4, 1e-4, one, one],
            # row 1: iforest has the smallest raw p (1e-3.5) but after the shift (2.5) robust_z
            # (p = 1e-3, F = 3) drives F
            [one, one, 1e-3, one, 10**-3.5],
            # row 2: no rule fired, cusum drives F
            [one, one, one, 1e-5, one],
            # row 3: every p = 1 -> F = 0, no driver
            [one, one, one, one, one],
            # row 4: rule 100102 drives F
            [1e-6, 1e-2, one, one, one],
        ]
    )
    assert list(rows.rule) == ["100104", None, None, None, "100102"]
    assert rows.nlp[1, fa.LAYERS.index("iforest")] == pytest.approx(2.5)  # per layer
    assert rows.F[3] == 0.0
    recs = [
        {
            "peak": i,
            "onset_row": i,
            "onset_us": int(rows.t[i]),
            "length_min": 0.0,
            "vol_rank": 0.5,
            "direction": "up",
        }
        for i in range(5)
    ]
    prof = fa.profile(rows, recs, tau=3.0)
    assert prof["n"] == 5
    # min p: row0 rules (tie -> detector order: rules before robust_z), row1 iforest,
    # row2 cusum, row3 None, row4 rules
    assert prof["layer_min_p"]["rules"]["n"] == 2
    assert prof["layer_min_p"]["iforest"]["n"] == 1
    assert prof["layer_min_p"]["cusum"]["n"] == 1
    # F: row0 rules (tie -> first), row1 robust_z (iforest loses after the shift), row2 cusum
    assert prof["layer_F"]["rules"]["n"] == 2
    assert prof["layer_F"]["robust_z"]["n"] == 1
    assert prof["layer_F"]["iforest"]["n"] == 0
    assert prof["layer_F"]["rules"]["share"] == pytest.approx(0.5)  # F = 0 row left out
    assert prof["layers_over_tau"] == {"0": 1, "1": 3, "2": 1}
    assert prof["multi_layer"]["n"] == 1
    assert prof["rule_when_rules_drive_F"]["100104"]["n"] == 1
    assert prof["rule_when_rules_drive_F"]["100102"]["n"] == 1


def test_rows_peak_and_at():
    rows = _stub([[1.0, 1.0, 1e-3, 1.0, 1.0], [1.0, 1.0, 1e-5, 1.0, 1.0], [1.0] * 5])
    t = rows.t
    assert rows.peak("AAA", int(t[0]), int(t[2])) == 1
    assert rows.peak("AAA", int(t[2]) + 1, int(t[2]) + 5) is None
    assert rows.peak("ZZZ", 0, int(t[2])) is None
    assert rows.at("AAA", int(t[1])) == 1
    assert rows.at("AAA", int(t[1]) - 5) is None
    assert rows.at("AAA", int(t[1]) - 5, ahead_us=5) == 1


def test_unique_rows_and_overlaps():
    assert fa.unique_rows([3, None, 3, 1, 1]) == ([3, 1], 2)
    w = {"A": [(10, 20)]}
    assert fa.overlaps(w, "A", 20, 30) and not fa.overlaps(w, "A", 21, 30)
    assert not fa.overlaps(w, "B", 0, 100)


def test_most_volatile_matches_eval_rule(monkeypatch):
    from datetime import date
    from types import SimpleNamespace

    import polars as pl

    from idsai.eval import experiments as ex
    from idsai.eval.splits import SymDay

    days = [SymDay(s, date(2018, 1, d)) for s in ("A", "B") for d in (1, 2, 3)]
    vol = {sd: v for sd, v in zip(days, [0.3, 0.1, 0.3, 0.2, 0.5, 0.1], strict=True)}
    by_start = {ex.day_start_us(sd.day): sd for sd in days}

    def read_trades(sym, t0, t1):
        return pl.DataFrame({"sym": [sym], "t0": [t0]})

    monkeypatch.setattr(
        ex, "realized_vol_1m", lambda tr: vol[SymDay(tr["sym"][0], by_start[tr["t0"][0]].day)]
    )
    ctx = SimpleNamespace(store=SimpleNamespace(read_trades=read_trades))
    for k in (0, 1, 2, 3, 6):
        want = set(ex._most_volatile(ctx, days, k))
        got = fa.most_volatile({(sd.symbol, sd.day): vol[sd] for sd in days}, k)
        assert {SymDay(s, d) for s, d in got} == want
