"""Template explainer: Thai text from an Alert, number validator, edge cases."""

import dataclasses
import math

import pytest

from idsai.contracts.alert import Alert, DetectorRef
from idsai.detection.features.registry import REGISTRY
from idsai.detection.taxonomy import load_taxonomy, technique_ids
from idsai.product.explainer import (
    BANNED_TH,
    Explanation,
    allowed_numbers,
    explain,
    numbers_in,
    validate_numbers,
)


def make_alert(**kw) -> Alert:
    base = dict(
        alert_id="a1",
        ts_event_us=1_761_573_612_000_000,  # 2025-10-27 14:00:12 UTC
        symbol="XYZBTC",
        technique_id="MM-T001",
        severity=2,
        status="open",
        fused_p=0.0004,
        detectors=[
            DetectorRef(name="rule.x", layer="rule", p=0.01, score=1.0),
            DetectorRef(name="stat.z", layer="stat", p=0.02, score=None),
        ],
        agreement=3,
        evidence=[
            ("z_quote_vol_60", 7.1, 7.1),
            ("ret_60", 0.042, 0.5),
            ("n_trades_60", 120.0, 0.2),
        ],
        window=(1, 2),
        dedup_key="k",
        suppress_for_s=60,
        config_hash="h",
        model_versions={},
        provenance="sim",
    )
    base.update(kw)
    return Alert(**base)


def test_skeleton_phrases():
    e = explain(make_alert())
    assert isinstance(e, Explanation)
    t = e.text_th
    assert t.startswith("พบความผิดปกติทางสถิติ (MM-T001 ")
    assert "ที่ XYZBTC เวลา 14:00:12 UTC: " in t
    assert "; 3 จาก 3 ชั้นตรวจจับเห็นพ้อง; ความหายากเทียบตลาดปกติ p ≈ 4.0e-04" in t
    assert "robust z = 7.1" in t and "4.20%" in t
    assert len(e.facts) == 3 and all(isinstance(f, tuple) and len(f) == 2 for f in e.facts)


def test_explanation_is_frozen():
    e = explain(make_alert())
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.text_th = "x"  # type: ignore[misc]


def test_mm_t000_wording_and_yaml_agree():
    e = explain(make_alert(technique_id="MM-T000"))
    assert "(MM-T000 ความผิดปกติที่ยังไม่จัดประเภท)" in e.text_th
    assert load_taxonomy()["MM-T000"]["name_th"] == "ความผิดปกติที่ยังไม่จัดประเภท"


def test_sub_technique_and_unknown_technique():
    tax = load_taxonomy()
    sub = explain(make_alert(technique_id="MM-T001.001")).text_th
    assert "(MM-T001.001 " + tax["MM-T001"]["sub_techniques"][0]["name_th"] + ")" in sub
    unk = explain(make_alert(technique_id="MM-T999")).text_th
    assert "(MM-T999 ความผิดปกติที่ยังไม่จัดประเภท)" in unk


def test_top3_by_contribution_stable():
    ev = [
        ("ret_60", 0.01, 1.0),
        ("buy_imb_60", 0.5, 1.0),
        ("z_rush_60", 3.0, 1.0),
        ("z_n_trades_60", 9.0, 0.1),
    ]
    e = explain(make_alert(evidence=ev))
    assert len(e.facts) == 3
    assert "9.0" not in e.text_th
    assert e.facts[0][0].startswith(REGISTRY["ret"].desc_th)


def test_empty_evidence_no_numbers_fabricated():
    a = make_alert(evidence=[])
    e = explain(a)
    assert e.facts == []
    assert "ความผิดปกติทางสถิติ" in e.text_th
    assert validate_numbers(e.text_th, allowed_numbers(a))


def test_unknown_feature_generic_phrase():
    a = make_alert(evidence=[("ae_recon", 3.25, 1.0), ("cusum_ret", 5.5, 0.5)])
    e = explain(a)
    assert "ae_recon" in e.text_th and "3.25" in e.text_th
    assert validate_numbers(e.text_th, allowed_numbers(a))


def test_detector_score_none_ok():
    a = make_alert(detectors=[DetectorRef(name="s", layer="stat", p=0.5, score=None)])
    assert explain(a).text_th


def test_nonfinite_evidence_does_not_crash():
    a = make_alert(
        evidence=[("z_quote_vol_60", float("nan"), float("nan")), ("ret_60", float("inf"), 1.0)]
    )
    e = explain(a)
    assert "nan" not in e.text_th.lower() and "inf" not in e.text_th.lower()


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"fused_p": 0.0},
        {"fused_p": 1.0},
        {"fused_p": 3.7e-12},
        {"agreement": 0},
        {"evidence": [("ret_60", -0.0421, 1.0), ("z_jump_300", -4.25, 0.9)]},
        {"evidence": [("buy_imb_60", -0.14, 1.0), ("round_frac_60", 0.0001, 0.5)]},
        {
            "evidence": [
                ("quote_vol_60", 1234567.891, 1.0),
                ("kk_vol_ratio", 8.34, 0.5),
                ("cusum_imb", 12.3, 0.2),
            ]
        },
        {
            "evidence": [
                ("pingpong_60", 14.0, 1.0),
                ("rush_60", 3.0, 0.5),
                ("buy_quote_60", 55.55, 0.1),
            ]
        },
        {"technique_id": "MM-T005.001"},
        {"symbol": "1000PEPEUSDT"},
    ],
)
def test_validator_accepts_own_text(kw):
    a = make_alert(**kw)
    assert validate_numbers(explain(a).text_th, allowed_numbers(a))


def test_validator_rejects_altered_number():
    a = make_alert()
    t = explain(a).text_th
    assert not validate_numbers(t.replace("7.1", "9.9"), allowed_numbers(a))
    assert not validate_numbers(t + " เพิ่มอีก 42", allowed_numbers(a))


def test_displayed_rounding_value_in_allowed():
    a = make_alert(evidence=[("buy_imb_60", 0.14, 1.0)], fused_p=0.0004)
    assert 0.14 in allowed_numbers(a)
    assert validate_numbers(explain(a).text_th, allowed_numbers(a))


@pytest.mark.parametrize(
    "text,expected",
    [
        ("พบ 3 รายการ", [3.0]),
        ("p ≈ 4.0e-04", [4.0e-4]),
        ("p 1E+3", [1000.0]),
        ("ผลตอบแทน -4.20%", [-4.2]),
        ("ราคา 4.20%", [4.2]),
        ("ค่า 1,234.5", [1234.5]),
        ("เวลา 14:00:12", [14.0, 0.0, 12.0]),
        ("เลข ๑๒.๕", [12.5]),
        ("MM-T001 MM-T001.002 1000PEPEUSDT z_60s", []),
        ("x-5 และ 7-8", [5.0, 7.0, 8.0]),
        ("ไม่มีตัวเลข", []),
    ],
)
def test_numbers_in(text, expected):
    got = numbers_in(text)
    assert len(got) == len(expected)
    assert all(math.isclose(g, e, rel_tol=1e-12) for g, e in zip(got, expected, strict=True))


def test_validate_numbers_semantics():
    assert validate_numbers("ไม่มีตัวเลข", [])
    assert validate_numbers("ค่า 7.1", [7.12])  # within 1%
    assert not validate_numbers("ค่า 7.1", [7.3])
    assert not validate_numbers("ค่า -7.1", [7.1])  # sign matters
    assert validate_numbers("ค่า 0", [0.0])
    assert not validate_numbers("ค่า 0", [1e-9])  # abs_tol=0
    assert not validate_numbers("ค่า 5", [float("nan"), float("inf")])
    assert not validate_numbers("ค่า ๕", [6.0])  # Thai digits parsed


def test_no_banned_words_everywhere():
    tax = load_taxonomy()
    for tid in [*sorted(technique_ids(tax)), "MM-T999"]:
        text = explain(make_alert(technique_id=tid)).text_th
        assert not [w for w in BANNED_TH if w in text]
        assert "ความผิดปกติทางสถิติ" in text
    for name in REGISTRY:
        col = name if name.startswith(("cusum", "kk_")) else f"{name}_60"
        text = explain(make_alert(evidence=[(col, 1.0, 1.0), (f"z_{col}", 2.0, 0.5)])).text_th
        assert not [w for w in BANNED_TH if w in text]
    for spec in REGISTRY.values():
        assert not [w for w in BANNED_TH if w in spec.desc_th]
    for entry in tax.values():
        assert not [w for w in BANNED_TH if w in entry["name_th"]]


def test_banned_list_and_no_probability_wording():
    assert BANNED_TH == ["โกง", "มิจฉาชีพ", "ผู้ปั่น", "คนร้าย", "ฉ้อโกง"]
    assert "ความน่าจะเป็น" not in explain(make_alert()).text_th
