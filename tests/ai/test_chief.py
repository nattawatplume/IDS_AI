import math
import pickle
from typing import Any

import numpy as np
import polars as pl
import pytest

from idsai.ai.chief import CalibrationError, Chief, ChiefStream
from idsai.config import ChiefCfg
from idsai.contracts.alert import Alert
from idsai.contracts.detector import Detector, Evidence
from idsai.detection.rules import Condition, Signature, SignatureDetector
from idsai.detection.stats import CusumDetector

NAN = float("nan")
S_US = 1_000_000
DAY_US = 86_400 * S_US
T0_US = 1_700_006_400 * S_US  # a UTC midnight
CFG = ChiefCfg()
VERSIONS = {"ai.ae": "v1"}


class Stub(Detector):
    """Detector whose scores are supplied directly as `score__<name>` columns."""

    def __init__(self, name: str, layer: str, requires: tuple[str, ...] = ()):
        self.name = name
        self.layer = layer
        self.technique_ids = ["ANY"]
        self.requires = list(requires)

    def score(self, windows: pl.DataFrame) -> np.ndarray:
        return windows[f"score__{self.name}"].to_numpy().astype(np.float64)

    def evidence(self, row: dict[str, Any]) -> Evidence:
        return Evidence.from_items((f, float(row[f]), abs(float(row[f]))) for f in self.requires)


def rule(sid: int, technique: str, feature: str = "a") -> SignatureDetector:
    sig = Signature(
        sid=sid,
        technique=technique,
        layer="rule",
        rev=3,
        name="t",
        conditions=[Condition(feature=feature, op=">=", thr=1.0)],
    )
    return SignatureDetector(sig)


def chief(*dets: Detector, cfg: ChiefCfg = CFG) -> Chief:
    return Chief(list(dets), cfg, config_hash="cfg123", model_versions=VERSIONS)


def frame(symbols, t_s, **scores) -> pl.DataFrame:
    """Rows at `T0 + t_s` seconds; keyword `x_y` maps to column `score__x.y`."""
    n = len(t_s)
    syms = [symbols] * n if isinstance(symbols, str) else list(symbols)
    cols: dict[str, Any] = {
        "symbol": pl.Series("symbol", syms, dtype=pl.String),
        "t_end_us": pl.Series("t_end_us", [T0_US + int(t) * S_US for t in t_s], dtype=pl.Int64),
    }
    for k, v in scores.items():
        name = k.replace("__", "#").replace("_", ".").replace("#", "_")
        cols[f"score__{name}"] = pl.Series(f"score__{name}", v, dtype=pl.Float64)
    return pl.DataFrame(cols)


def cal_with(name: str, values, extra: dict[str, list[float]] | None = None) -> pl.DataFrame:
    n = len(values)
    data = {
        "symbol": ["CALUSDT"] * n,
        "t_end_us": [T0_US + 10 * i * S_US for i in range(n)],
        f"score__{name}": [float(v) for v in values],
    }
    for k, v in (extra or {}).items():
        data[f"score__{k}"] = [float(x) for x in v]
    return pl.DataFrame(data)


# ---------------------------------------------------------------- p-values


def test_pvalue_formula():
    ch = chief(CusumDetector())
    ch.calibrate(cal_with("stat.cusum", [1, 2, 3, 4]))
    assert ch.pvalues("stat.cusum", np.array([5.0, 2.0, np.nan])).tolist() == [0.2, 0.8, 1.0]
    assert ch.n_cal == 4
    assert ch.cal_symbol_days == 1


def test_calibration_drops_non_finite_scores():
    ch = chief(Stub("stat.a", "stat"))
    ch.calibrate(cal_with("stat.a", [1, NAN, 3, math.inf]))
    assert ch.n_cal == 2
    assert ch.pvalues("stat.a", np.array([2.0])).tolist() == [2 / 3]


def test_unfired_rule_gets_p1():
    r = rule(1, "MM-T001.001")
    ch = chief(r)
    ch.calibrate(cal_with(r.name, [-5, -4, -3, -2]))
    # -0.1 is above every calibration score but the rule does not fire (score < 0)
    assert ch.pvalues(r.name, np.array([-0.1, NAN, 0.5])).tolist() == [1.0, 1.0, 0.2]


def test_rule_calibrates_on_all_finite_scores_fired_or_not():
    r = rule(1, "MM-T001.001")
    ch = chief(r)
    ch.calibrate(cal_with(r.name, [-5, -4, 0.5, 2.0]))
    assert ch.n_cal == 4
    assert ch.pvalues(r.name, np.array([1.0])).tolist() == [0.4]  # (1 + 1) / 5


def test_calibrate_missing_column_raises_keyerror():
    ch = chief(Stub("stat.a", "stat"), Stub("ai.b", "ai"))
    with pytest.raises(KeyError, match="score__ai.b"):
        ch.calibrate(cal_with("stat.a", [1, 2, 3]))


def test_pvalues_and_fuse_before_calibrate_raise():
    ch = chief(Stub("stat.a", "stat"))
    with pytest.raises(CalibrationError):
        ch.pvalues("stat.a", np.array([1.0]))
    with pytest.raises(CalibrationError):
        ch.fuse(frame("A", [0], stat_a=[1.0]))


def test_n_cal_is_min_over_fused_detectors():
    ch = chief(Stub("stat.a", "stat"), Stub("ai.b", "ai"), Stub("baseline.c", "baseline"))
    ch.calibrate(
        cal_with("stat.a", [1, 2, 3, 4], extra={"ai.b": [1, 2, NAN, NAN], "baseline.c": [NAN] * 4})
    )
    assert ch.n_cal == 2  # baseline (0 finite) is carried but not fused


# ---------------------------------------------------------------- fusion


def test_fuse_columns_and_F_definition():
    ch = chief(Stub("stat.a", "stat"), Stub("ai.b", "ai"))
    ch.calibrate(cal_with("stat.a", range(1, 100), extra={"ai.b": range(1, 100)}))
    sc = frame("A", [0, 10, 20], stat_a=[150.0, 0.0, 95.0], ai_b=[0.0, 150.0, 98.0])
    out = ch.fuse(sc)
    added = ["p__stat.a", "p__ai.b", "F", "best_detector", "agreement", "technique"]
    assert out.columns == sc.columns + added
    pa, pb = out["p__stat.a"].to_numpy(), out["p__ai.b"].to_numpy()
    np.testing.assert_allclose(out["F"].to_numpy(), np.maximum(-np.log10(pa), -np.log10(pb)))
    assert out["best_detector"].to_list() == ["stat.a", "ai.b", "ai.b"]
    assert out["F"][0] == pytest.approx(2.0)  # p = 1 / 100


def test_all_p_one_gives_F_zero_and_first_fused_detector():
    ch = chief(Stub("baseline.z", "baseline"), Stub("stat.a", "stat"), Stub("ai.b", "ai"))
    ch.calibrate(cal_with("stat.a", [1, 2], extra={"ai.b": [1, 2], "baseline.z": [1, 2]}))
    out = ch.fuse(frame("A", [0], stat_a=[NAN], ai_b=[-1.0], baseline_z=[99.0]))
    assert out["F"][0] == 0.0 and math.copysign(1.0, out["F"][0]) == 1.0
    assert out["best_detector"][0] == "stat.a"
    assert out["technique"][0] == "MM-T000"


def test_technique_from_fired_rule_else_cusum_else_T000():
    r1, r2 = rule(101, "MM-T001.001"), rule(501, "MM-T005.001")
    ch = chief(r1, r2, CusumDetector())
    n = 2000
    ch.calibrate(
        cal_with(
            r1.name,
            [-1.0] * (n - 10) + [1.0] * 10,  # r1 fires on 10 calibration rows
            extra={r2.name: [-1.0] * n, "stat.cusum": list(range(n))},
        )
    )
    sc = frame(
        "A",
        [0, 1000, 2000, 3000, 4000],
        **{
            "rule_101": [0.5, 0.5, -1.0, -1.0, NAN],
            "rule_501": [0.5, -1.0, -1.0, -1.0, -1.0],
            "stat_cusum": [5000.0, 5000.0, 5000.0, 1990.0, 1990.0],
        },
    )
    out = ch.fuse(sc)
    # row 0: both rules fire, r2 is rarer (p = 1/2001 vs 11/2001) -> its parent technique
    # row 1: only r1 fires -> MM-T001 (from the rule even though cusum is more extreme)
    # row 2: no rule fires, cusum -log10 p = log10(2001) >= 3 -> MM-T001
    # rows 3-4: cusum p = 11/2001 (-log10 p ~ 2.26 < 3) -> unclassified
    assert out["technique"].to_list() == ["MM-T005", "MM-T001", "MM-T001", "MM-T000", "MM-T000"]
    alerts = ch.episodes(out, tau=1.0)
    by_t = {a.ts_event_us: a for a in alerts}
    assert len(alerts) == 5
    assert (by_t[T0_US].signature_id, by_t[T0_US].rev) == (501, 3)
    assert by_t[T0_US + 1000 * S_US].signature_id == 101
    assert by_t[T0_US + 2000 * S_US].signature_id is None
    assert by_t[T0_US + 2000 * S_US].rev is None


def test_agreement_ignores_baseline_layer_and_caps_at_3():
    dets = [
        rule(101, "MM-T001.001"),
        Stub("stat.a", "stat"),
        Stub("stat.b", "stat"),
        Stub("ai.c", "ai"),
        Stub("baseline.d", "baseline"),
    ]
    ch = chief(*dets)
    vals = list(range(-500, 500))  # 1000 values; 499 rule scores >= 0 would fire
    ch.calibrate(
        cal_with(
            "rule.101",
            [-1.0] * 1000,
            extra={n: vals for n in ["stat.a", "stat.b", "ai.c", "baseline.d"]},
        )
    )
    sc = frame(
        "A",
        [0, 10, 20, 30],
        rule_101=[2.0, -1.0, -1.0, 2.0],
        stat_a=[999.0, -999.0, 999.0, -999.0],
        stat_b=[999.0, -999.0, 999.0, -999.0],
        ai_c=[999.0, -999.0, -999.0, -999.0],
        baseline_d=[999.0, 999.0, -999.0, -999.0],
    )
    out = ch.fuse(sc)
    assert out["agreement"].to_list() == [3, 0, 1, 1]
    assert out["F"][1] == 0.0  # only the baseline is extreme: it never enters F
    assert out["p__baseline.d"][1] == pytest.approx(1 / 1001)


def test_baselines_excluded_from_F_when_mixed():
    ch = chief(Stub("stat.a", "stat"), Stub("baseline.iforest", "baseline"))
    ch.calibrate(cal_with("stat.a", range(100), extra={"baseline.iforest": range(100)}))
    out = ch.fuse(frame("A", [0, 10], stat_a=[50.0, 200.0], baseline_iforest=[200.0, 50.0]))
    assert out["F"].to_list() == pytest.approx([-math.log10(51 / 101), math.log10(101)])
    assert set(out["best_detector"].to_list()) == {"stat.a"}
    assert "p__baseline.iforest" in out.columns
    (alert,) = ch.episodes(out, tau=1.0)
    assert [d.name for d in alert.detectors] == ["stat.a"]


def test_all_baselines_chief_fuses_over_baselines():
    ch = chief(Stub("baseline.iforest", "baseline"), Stub("baseline.pca", "baseline"))
    ch.calibrate(cal_with("baseline.iforest", range(100), extra={"baseline.pca": range(100)}))
    out = ch.fuse(frame("A", [0, 10], baseline_iforest=[50.0, 200.0], baseline_pca=[200.0, 1.0]))
    assert out["F"].to_list() == pytest.approx([math.log10(101)] * 2)
    assert out["best_detector"].to_list() == ["baseline.pca", "baseline.iforest"]
    assert out["agreement"].to_list() == [0, 0]
    assert out["technique"].to_list() == ["MM-T000", "MM-T000"]
    alerts = ch.episodes(out, tau=1.0)
    assert len(alerts) == 1
    assert {d.layer for d in alerts[0].detectors} == {"baseline"}
    assert alerts[0].severity == 1


# ---------------------------------------------------------------- episodes


def _one_stat_chief() -> Chief:
    ch = chief(Stub("stat.a", "stat", requires=("fa",)), rule(501, "MM-T005.001"), CusumDetector())
    ch.calibrate(
        cal_with(
            "stat.a", range(1000), extra={"rule.501": [-1.0] * 1000, "stat.cusum": range(1000)}
        )
    )
    ch.tau = 1.0
    return ch


def test_episode_merge_split_per_symbol():
    ch = _one_stat_chief()
    hi, lo = 5000.0, -1.0

    def eps(symbols, t_s, a, r=None, c=None):
        n = len(t_s)
        sc = frame(symbols, t_s, stat_a=a, rule_501=r or [-1.0] * n, stat_cusum=c or [lo] * n)
        return ch.episodes(ch.fuse(sc))

    assert len(eps("A", [0, 100, 200], [hi, lo, hi])) == 1
    assert len(eps("A", [0, 300], [hi, hi])) == 1  # gap == suppress_for_s merges
    assert len(eps("A", [0, 200, 400], [hi, lo, hi])) == 2
    # same times on two symbols never merge
    two = eps(["A", "B"], [0, 100], [hi, hi])
    assert sorted(a.symbol for a in two) == ["A", "B"]
    # technique flips inside one episode (rule at t=0, cusum at t=200) -> still one episode
    (flip,) = eps("A", [0, 200], [lo, lo], r=[2.0, -1.0], c=[lo, hi])
    assert flip.window == (T0_US, T0_US + 200 * S_US)
    # peak is the max-F row (rule p = 1/1001 vs cusum p = 1/1001 tie -> earliest)
    assert flip.technique_id == "MM-T005"


def test_episode_fields_from_peak_row():
    ch = _one_stat_chief()
    sc = frame(
        "A",
        [0, 100, 250],
        stat_a=[995.0, 5000.0, 998.0],
        rule_501=[-1.0] * 3,
        stat_cusum=[NAN, NAN, NAN],
    )
    sc = sc.with_columns(pl.Series("fa", [0.1, 7.0, 0.3]))
    (a,) = ch.episodes(ch.fuse(sc))
    assert a.alert_id == f"replay-A-{T0_US}"
    assert a.ts_event_us == T0_US
    assert a.window == (T0_US, T0_US + 250 * S_US)
    assert a.status == "closed"
    assert a.provenance == "replay"
    assert a.fused_p == pytest.approx(1 / 1001)
    assert a.dedup_key == f"A|{T0_US // 300_000_000}"
    assert a.suppress_for_s == 300
    assert a.config_hash == "cfg123" and a.model_versions == VERSIONS
    assert a.evidence == [("fa", 7.0, 7.0)]
    refs = {d.name: d for d in a.detectors}
    assert set(refs) == {"stat.a", "rule.501", "stat.cusum"}
    assert refs["stat.a"].score == 5000.0
    assert refs["stat.cusum"].p == 1.0 and refs["stat.cusum"].score is None
    # best detector (stat.a) requires a column that is absent -> empty evidence
    (b,) = ch.episodes(ch.fuse(sc.drop("fa")), provenance="sim")
    assert b.evidence == [] and b.provenance == "sim"


def test_alert_json_round_trip_with_nan_score_and_inf_feature():
    ch = chief(Stub("stat.a", "stat", requires=("fa", "fb")), CusumDetector())
    ch.calibrate(cal_with("stat.a", range(1000), extra={"stat.cusum": range(1000)}))
    sc = frame("A", [0], stat_a=[5000.0], stat_cusum=[NAN])
    sc = sc.with_columns(pl.Series("fa", [math.inf]), pl.Series("fb", [2.0]))
    (a,) = ch.episodes(ch.fuse(sc), tau=1.0)
    assert {d.name: d.score for d in a.detectors} == {"stat.a": 5000.0, "stat.cusum": None}
    assert a.evidence == [("fb", 2.0, 2.0)]  # the inf item is dropped
    assert Alert.model_validate_json(a.model_dump_json()) == a


def test_severity_from_agreement_only():
    ch = chief(rule(101, "MM-T001.001"), Stub("stat.a", "stat"), Stub("ai.c", "ai"))
    vals = list(range(1000))
    ch.calibrate(cal_with("rule.101", [-1.0] * 1000, extra={"stat.a": vals, "ai.c": vals}))
    hi, mid, lo = 5000.0, 995.0, -1.0  # mid: p = 5/1001 < 0.01 but much smaller F than hi
    sc = frame(
        "A",
        [0, 1000, 2000, 3000],
        rule_101=[-1.0, -1.0, 2.0, -1.0],
        stat_a=[hi, mid, mid, mid],
        ai_c=[lo, mid, mid, lo],
    )
    alerts = sorted(ch.episodes(ch.fuse(sc), tau=1.0), key=lambda a: a.ts_event_us)
    assert [a.agreement for a in alerts] == [1, 2, 3, 1]
    assert [a.severity for a in alerts] == [1, 2, 3, 1]


def test_episodes_before_threshold_raises():
    ch = chief(Stub("stat.a", "stat"))
    ch.calibrate(cal_with("stat.a", range(10)))
    with pytest.raises(CalibrationError):
        ch.episodes(ch.fuse(frame("A", [0], stat_a=[100.0])))


def test_all_F_zero_gives_no_episodes():
    r = rule(101, "MM-T001.001")
    ch = chief(r)
    ch.calibrate(cal_with(r.name, [-1.0] * 2000))
    cal_fused = ch.fuse(cal_with(r.name, [-1.0] * 2000))
    assert cal_fused["F"].max() == 0.0
    tau = ch.fit_threshold(cal_fused, 2.0)
    assert tau > 0 and ch.tau == tau
    assert ch.episodes(cal_fused) == []


# ---------------------------------------------------------------- threshold


def iid(seed: int, n_sym: int = 50, n_days: int = 10, stride_s: int = 60) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    per_sym = n_days * 86_400 // stride_s
    t = T0_US + np.arange(per_sym, dtype=np.int64) * stride_s * S_US
    n = n_sym * per_sym
    return pl.DataFrame(
        {
            "symbol": np.repeat([f"S{i:02d}USDT" for i in range(n_sym)], per_sym),
            "t_end_us": np.tile(t, n_sym),
            "score__stat.a": rng.standard_normal(n),
            "score__stat.b": rng.standard_normal(n),
            "score__ai.c": rng.standard_normal(n),
        }
    )


def test_threshold_respects_budget_out_of_sample():
    ch = chief(Stub("stat.a", "stat"), Stub("stat.b", "stat"), Stub("ai.c", "ai"))
    cal = iid(seed=0)
    ch.calibrate(cal)
    assert ch.cal_symbol_days == 500
    cal_fused = ch.fuse(cal)
    tau = ch.fit_threshold(cal_fused, 2.0)
    assert len(ch.episodes(cal_fused, tau)) / 500 <= 2.0
    rate = len(ch.episodes(ch.fuse(iid(seed=1)), tau)) / 500
    print(f"tau={tau:.4f} out-of-sample episodes per symbol-day={rate:.3f}")
    assert 2.0 * 0.5 <= rate <= 2.0 * 1.5


def test_threshold_is_smallest_feasible_candidate():
    ch = chief(Stub("stat.a", "stat"))
    ch.calibrate(cal_with("stat.a", range(1000)))
    # 4 symbols x 1 day, one row per symbol at increasing extremeness
    sc = frame(["A", "B", "C", "D"], [0, 0, 0, 0], stat_a=[999.5, 998.5, 997.5, 996.5])
    fused = ch.fuse(sc)
    F = sorted(fused["F"].to_list(), reverse=True)
    # tau = F[k] gives k + 1 single-row episodes over 4 symbol-days
    with pytest.warns(RuntimeWarning):  # 4 single-row episodes is the peak rate: saturated
        assert ch.fit_threshold(fused, 2.0) == F[3]
    assert ch.fit_threshold(fused, 0.5) == F[1]
    assert ch.fit_threshold(fused, 0.25) == F[0]
    assert ch.fit_threshold(fused, 0.75) == F[2]  # gallop overshoots; last candidate probed
    assert ch.tau_saturated is False


def test_threshold_ignores_low_tau_where_episodes_merge():
    """Episode rate rises then falls as tau drops (exceedances merge); take the top crossing."""
    ch = chief(Stub("stat.a", "stat"))
    rng = np.random.default_rng(5)
    n = 2000  # one symbol, 10 s stride, one UTC day
    sc = frame("A", list(range(0, 10 * n, 10)), stat_a=rng.standard_normal(n))
    ch.calibrate(sc)
    fused = ch.fuse(sc)
    budget = 20.0
    tau = ch.fit_threshold(fused, budget)

    def rate(x):
        return len(ch.episodes(fused, x))

    cand = sorted(set(fused["F"].to_list()) - {0.0}, reverse=True)
    assert rate(cand[-1]) <= budget  # the merged tail would also "fit": a literal minimum
    i = cand.index(tau)
    assert all(rate(c) <= budget for c in cand[: i + 1])
    assert rate(cand[i + 1]) > budget
    assert (fused["F"] >= tau).sum() < 100


def test_threshold_budget_above_peak_rate_saturates_with_warning():
    ch = chief(Stub("stat.a", "stat"))
    rng = np.random.default_rng(5)
    n = 2000  # one symbol-day; the peak episode rate is far below 1000
    sc = frame("A", list(range(0, 10 * n, 10)), stat_a=rng.standard_normal(n))
    ch.calibrate(sc)
    fused = ch.fuse(sc)
    assert ch.tau_saturated is False
    with pytest.warns(RuntimeWarning, match="tau_saturated"):
        tau = ch.fit_threshold(fused, 1000.0)
    assert ch.tau_saturated is True
    assert tau == min(f for f in fused["F"].to_list() if f > 0)
    assert ch.state_dict()["tau_saturated"] is True
    ch.fit_threshold(fused, 20.0)  # a reachable budget clears the flag
    assert ch.tau_saturated is False
    ch.calibrate(sc)
    assert ch.tau_saturated is False


def test_threshold_guard_raises_on_tiny_calibration():
    ch = chief(Stub("stat.a", "stat"), Stub("ai.b", "ai"))
    ch.calibrate(cal_with("stat.a", [1, 2, 3, 4], extra={"ai.b": [1, 2, 3, 4]}))
    big = cal_with("stat.a", range(100), extra={"ai.b": range(100)})  # 100 rows, 1 symbol-day
    with pytest.raises(CalibrationError, match="budget") as e:
        ch.fit_threshold(ch.fuse(big), 2.0)  # 2/100 < 2/5
    assert "สอบเทียบ" in str(e.value)


# ---------------------------------------------------------------- state


def test_state_dict_round_trip_gives_identical_fuse():
    dets = [rule(101, "MM-T001.001"), Stub("stat.a", "stat"), Stub("baseline.d", "baseline")]
    ch = chief(*dets)
    rng = np.random.default_rng(3)
    extra = {"stat.a": rng.normal(size=500), "baseline.d": rng.normal(size=500)}
    cal = cal_with("rule.101", rng.normal(size=500) - 1, extra=extra)
    ch.calibrate(cal)
    with pytest.warns(RuntimeWarning):  # 500 rows in one symbol-day: budget above peak
        ch.fit_threshold(ch.fuse(cal), 50.0)
    state = pickle.loads(pickle.dumps(ch.state_dict()))
    ch2 = chief(*dets)
    ch2.load_state_dict(state)
    assert (ch2.tau, ch2.n_cal, ch2.cal_symbol_days) == (ch.tau, ch.n_cal, ch.cal_symbol_days)
    assert ch2.tau_saturated is ch.tau_saturated is True
    assert ch2.fuse(cal).equals(ch.fuse(cal))
    with pytest.raises(ValueError, match="detector"):
        chief(*dets[:2]).load_state_dict(state)


def test_state_load_rejects_config_or_model_mismatch():
    ch = chief(Stub("stat.a", "stat"))
    ch.calibrate(cal_with("stat.a", range(10)))
    state = ch.state_dict()
    with pytest.raises(ValueError, match="config_hash"):
        Chief(ch.detectors, CFG, config_hash="other", model_versions=VERSIONS).load_state_dict(
            state
        )
    with pytest.raises(ValueError, match="model_versions"):
        Chief(ch.detectors, CFG, "cfg123", {"ai.ae": "v2"}).load_state_dict(state)
    with pytest.raises(ValueError, match="model_versions"):
        Chief(ch.detectors, CFG, "cfg123", {}).load_state_dict(state)


# ---------------------------------------------------------------- stream


def test_stream_close_and_open_on_one_row():
    ch = _one_stat_chief()
    st = ChiefStream(ch, provenance="replay")
    rows = frame(
        "A",
        [0, 100, 250, 600],
        stat_a=[995.0, 5000.0, -1.0, 995.0],
        rule_501=[-1.0] * 4,
        stat_cusum=[-1.0] * 4,
    ).to_dicts()
    out = [[(a.status, a.alert_id) for a in st.step(r)] for r in rows]
    t = [T0_US + s * S_US for s in (0, 100, 250, 600)]
    assert out[0] == [("open", f"replay-A-{t[0]}")]
    assert out[1] == [("updated", f"replay-A-{t[0]}")]
    assert out[2] == []
    assert out[3] == [("closed", f"replay-A-{t[0]}"), ("open", f"replay-A-{t[3]}")]
    assert [(a.status, a.alert_id) for a in st.flush()] == [("closed", f"replay-A-{t[3]}")]
    assert st.flush() == []


def test_stream_requires_threshold():
    ch = chief(Stub("stat.a", "stat"))
    ch.calibrate(cal_with("stat.a", range(10)))
    with pytest.raises(CalibrationError):
        ChiefStream(ch, provenance="live")


def test_stream_matches_batch_episode_ids():
    dets = [
        rule(101, "MM-T001.001", feature="fa"),
        CusumDetector(),
        Stub("ai.c", "ai", requires=("fa", "fb")),
        Stub("baseline.d", "baseline"),
    ]
    ch = chief(*dets)
    rng = np.random.default_rng(7)
    n_sym, per_sym, stride = 3, 2000, 10
    syms = np.tile([f"S{i}USDT" for i in range(n_sym)], per_sym)  # interleaved rows
    t = np.repeat(T0_US + np.arange(per_sym, dtype=np.int64) * stride * S_US, n_sym)
    n = syms.size
    rule_s = np.where(rng.random(n) < 0.01, rng.random(n), -rng.random(n))
    rule_s[rng.random(n) < 0.02] = NAN
    sc = pl.DataFrame(
        {
            "symbol": syms,
            "t_end_us": t,
            "score__rule.101": rule_s,
            "score__stat.cusum": np.where(rng.random(n) < 0.02, NAN, rng.standard_normal(n)),
            "score__ai.c": rng.standard_normal(n),
            "score__baseline.d": rng.standard_normal(n),
            "fa": rng.standard_normal(n),
            "fb": rng.standard_normal(n),
            "cusum_ret": rng.standard_normal(n),
            "cusum_imb": rng.standard_normal(n),
        }
    )
    ch.calibrate(sc)
    fused = ch.fuse(sc)
    ch.fit_threshold(fused, 15.0)  # generous budget -> many multi-row episodes
    batch = ch.episodes(fused)
    assert len(batch) > 30
    assert any(a.window[0] != a.window[1] for a in batch)

    st = ChiefStream(ch, provenance="replay")
    emitted = [a for r in sc.to_dicts() for a in st.step(r)] + st.flush()
    closed = [a for a in emitted if a.status == "closed"]
    assert {a.status for a in emitted} == {"open", "updated", "closed"}
    assert sorted(a.model_dump_json() for a in closed) == sorted(a.model_dump_json() for a in batch)
    # every open/updated alert id is a final episode id
    assert {a.alert_id for a in emitted} == {a.alert_id for a in batch}
