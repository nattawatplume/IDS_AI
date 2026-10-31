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


def test_full_text_equality():
    e = explain(make_alert())
    assert e.text_th == (
        "พบความผิดปกติทางสถิติ (MM-T001 " + load_taxonomy()["MM-T001"]["name_th"] + ") "
        "ที่ XYZBTC เวลา 14:00:12 UTC: "
        + REGISTRY["quote_vol"].desc_th
        + " (ช่วง 60 วินาที) robust z = 7.1, "
        + REGISTRY["ret"].desc_th
        + " (ช่วง 60 วินาที) 4.20%, "
        + REGISTRY["n_trades"].desc_th
        + " (ช่วง 60 วินาที) 120; "
        "3 จาก 3 ชั้นตรวจจับเห็นพ้อง; ความหายากเทียบตลาดปกติ p ≈ 4.0e-04"
    )
    assert e.facts[0] == (REGISTRY["quote_vol"].desc_th + " (ช่วง 60 วินาที)", "robust z = 7.1")


def test_top3_selection_and_order_unsorted_and_negative():
    ev = [
        ("z_rush_60", 1.0, -1.0),
        ("ret_60", 0.01, 5.0),
        ("buy_imb_60", 0.5, -0.5),
        ("z_n_trades_60", 9.0, 2.0),
    ]
    e = explain(make_alert(evidence=ev))
    assert [v for _, v in e.facts] == ["1.00%", "robust z = 9.0", "0.50"]
    assert "robust z = 1.0" not in e.text_th


def test_top3_ties_keep_input_order():
    ev = [
        ("z_rush_60", 1.0, 1.0),
        ("z_jump_60", 2.0, 1.0),
        ("z_ret_60", 3.0, 1.0),
        ("z_pingpong_60", 4.0, 1.0),
    ]
    assert [v for _, v in explain(make_alert(evidence=ev)).facts] == [
        "robust z = 1.0",
        "robust z = 2.0",
        "robust z = 3.0",
    ]


def test_nan_contribution_ranks_last():
    ev = [
        ("z_rush_60", 1.0, float("nan")),
        ("z_jump_60", 2.0, -9.0),
        ("z_ret_60", 3.0, 0.0),
        ("z_pingpong_60", 4.0, 1.0),
    ]
    assert [v for _, v in explain(make_alert(evidence=ev)).facts] == [
        "robust z = 4.0",
        "robust z = 3.0",
        "robust z = 2.0",
    ]


def test_rounded_display_value_is_allowed():
    a = make_alert(evidence=[("z_quote_vol_60", 0.14, 1.0)])
    assert "robust z = 0.1" in explain(a).text_th
    assert validate_numbers("robust z = 0.1", allowed_numbers(a))
    assert not validate_numbers("robust z = 0.2", allowed_numbers(a))


def test_displayed_fused_p_is_allowed():
    a = make_alert(fused_p=3.66e-5)
    assert "p ≈ 3.7e-05" in explain(a).text_th
    assert 3.7e-5 in allowed_numbers(a)
    assert validate_numbers(explain(a).text_th, allowed_numbers(a))


@pytest.mark.parametrize(
    "ev,shown",
    [
        (("buy_imb_60", 0.14159, 1.0), "0.14"),  # ratio, 2 decimals
        (("quote_vol_60", 1234.56, 1.0), "1234.6"),  # quote, 1 decimal
        (("n_trades_60", 119.6, 1.0), "120"),  # trades, 0 decimals
        (("jump_300", 0.0123456, 1.0), "1.23%"),  # relative -> percent
        (("ret_60", -0.0421, 1.0), "-4.21%"),
        (("cusum_ret", 5.678, 1.0), "5.68"),  # sigma
    ],
)
def test_value_formats(ev, shown):
    e = explain(make_alert(evidence=[ev]))
    assert e.facts[0][1] == shown


def test_window_label_exact():
    e = explain(make_alert(evidence=[("rush_300", 4.0, 1.0)]))
    assert e.facts[0][0] == REGISTRY["rush"].desc_th + " (ช่วง 300 วินาที)"
    e2 = explain(make_alert(evidence=[("cusum_ret", 4.0, 1.0)]))
    assert e2.facts[0][0] == REGISTRY["cusum_ret"].desc_th  # state feature: no window


def test_no_negative_zero():
    a = make_alert(
        evidence=[
            ("ret_60", -0.000001, 3.0),
            ("z_rush_60", -0.04, 2.0),
            ("buy_imb_60", -0.001, 1.0),
        ]
    )
    e = explain(a)
    assert [v for _, v in e.facts] == ["0.00%", "robust z = 0.0", "0.00"]
    assert "-0.0" not in e.text_th and "-0 " not in e.text_th


@pytest.mark.parametrize(
    "text,expected",
    [
        ("พบ 3 รายการ", [3.0]),
        ("3 จาก 3 ชั้น", [3.0, 3.0]),
        ("p ≈ 4.0e-04", [4.0e-4]),
        ("p 1E+3", [1000.0]),
        ("p9e-9", [9e-9]),
        ("ผลตอบแทน -4.20%", [-4.2]),
        ("ราคา 4.20%", [4.2]),
        ("ค่า 1,234.5", [1234.5]),
        ("1,234USDT", [1234.0]),
        ("เวลา 14:00:12", [14.0, 0.0, 12.0]),
        ("เลข ๑๒.๕", [12.5]),
        ("MM-T001 MM-T001.002 (MM-T000)", []),
        ("MM-T001 7", [7.0]),
        ("1000PEPEUSDT", [1000.0]),
        ("5000USDT", [5000.0]),
        ("robustz9.9", [9.9]),
        ("z9.9", [9.9]),
        ("4.2USDT", [4.2]),
        ("10.5BTC", [10.5]),
        ("z_60s", [60.0]),
        (".5", [0.5]),
        ("-.5", [-0.5]),
        ("0.5.1", [0.5, 0.1]),
        ("1_000", [1.0, 0.0]),
        ("0x10", [0.0, 10.0]),
        ("12,3456", [12.0, 3456.0]),
        ("x-5 และ 7-8", [5.0, 7.0, 8.0]),
        ("a -5", [-5.0]),
        ("(-5)", [-5.0]),
        ("+5", [5.0]),
        ("−5", [-5.0]),
        ("–5", [-5.0]),
        ("—5", [-5.0]),
        ("－5", [-5.0]),
        ("﹣5", [-5.0]),
        ("x−5", [5.0]),
        ("ไม่มีตัวเลข", []),
    ],
)
def test_numbers_in(text, expected):
    got = numbers_in(text)
    assert len(got) == len(expected), got
    assert all(
        math.isclose(g, e, rel_tol=1e-12, abs_tol=0) for g, e in zip(got, expected, strict=True)
    )


def test_numbers_in_overflow_underflow_are_nan():
    for t in ("1e999", "1e-999", "-1e999", "0.5e-9999"):
        got = numbers_in(t)
        assert len(got) == 1 and math.isnan(got[0]), t
    assert numbers_in("0e-999") == [0.0]
    assert numbers_in("0") == [0.0]


def test_validate_rejects_nonfinite_numbers_in_text():
    allowed = [0.0, 1.0, 1e300]
    for t in ("1e999", "1e-999", "ค่า 0 และ 1e-999"):
        assert not validate_numbers(t, allowed)
    assert validate_numbers("ค่า 0", allowed)


def test_falsification_percent_and_sign():
    a = make_alert(evidence=[("ret_60", 0.042, 1.0)])
    assert "4.20%" in explain(a).text_th
    assert validate_numbers("4.20%", allowed_numbers(a))
    assert not validate_numbers("0.042%", allowed_numbers(a))
    assert not validate_numbers("–4.2", allowed_numbers(a))
    assert not validate_numbers("-4.2", allowed_numbers(a))
    assert validate_numbers("4.2", allowed_numbers(a))


def test_glued_numbers_cannot_hide():
    a = make_alert()
    assert not validate_numbers("9.9USDT", allowed_numbers(a))
    assert not validate_numbers("robustz9.9", allowed_numbers(a))


@pytest.mark.parametrize(
    "kw",
    [
        {"symbol": "1000PEPEUSDT"},
        {"symbol": "1000X"},
        {"symbol": "BTC-5"},
        {"evidence": [("foo_60", 2.5, 1.0)]},
        {"evidence": [("ae_2", -3.0, 1.0), ("x9", 1.0, 0.5)]},
        {"technique_id": "MM-T999"},
        {"evidence": [("ret_60", -0.000001, 1.0), ("z_rush_60", -0.04, 0.5)]},
    ],
)
def test_explain_text_always_validates(kw):
    a = make_alert(**kw)
    assert validate_numbers(explain(a).text_th, allowed_numbers(a))


# ---------------------------------------------------------------- fix round 2


def _has(allowed, x):
    return any(math.isclose(a, x, rel_tol=1e-9, abs_tol=0) for a in allowed)


@pytest.mark.parametrize("text", ["４e-999", "٤e-999", "๔e-999", "４e999"])
def test_unicode_digits_get_overflow_checks(text):
    got = numbers_in(text)
    assert len(got) == 1 and math.isnan(got[0])
    assert not validate_numbers(text, [0.0, 4.0, 1e300])


def test_unicode_digits_normalised():
    assert numbers_in("４") == [4.0]
    assert numbers_in("٤.٥") == [4.5]
    assert numbers_in("๑๒.๕") == [12.5]


@pytest.mark.parametrize(
    "text", ["XMM-T42.5", "MM-T99999.5%", "MM-T1", "MM-T001.5", "MM-T0011", "xMM-T001"]
)
def test_technique_exemption_is_narrow(text):
    assert numbers_in(text), text
    assert not validate_numbers(text, [3.0, 14.0, 60.0])


@pytest.mark.parametrize("text", ["MM-T001", "MM-T001.002", "(MM-T000 x)", "ที่ MM-T005.001 ได้"])
def test_technique_exemption_exact(text):
    assert numbers_in(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "- 4.2%",
        "\u2212\xa05",
        "-\u200b5",
        "\u02d75",
        "\u27965",
        "\u207b5",
        "\u208b5",
        "\u2010 5",
        "\u20115",
        "\u20125",
        "\u2013\u00a05",
        "\ufe635",
        "\uff0d5",
        "a - 5",
        "1 \u2212 5",
    ],
)
def test_signs_fail_closed(text):
    got = numbers_in(text)
    assert got and got[-1] < 0, got
    assert not validate_numbers(text, [4.2, 5.0])


@pytest.mark.parametrize("text", ["x-5", "x\u20105", "7-8", "7\u2212 8"])
def test_separator_after_letter_or_digit(text):
    assert all(x > 0 for x in numbers_in(text))


def test_every_desc_th_explained_text_validates():
    for name in REGISTRY:
        col = name if name.startswith(("cusum", "kk_")) else f"{name}_60"
        for value in (1.0, -0.5, 0.0):
            a = make_alert(evidence=[(col, value, 1.0), (f"z_{col}", -value, 0.5)])
            t = explain(a).text_th
            assert validate_numbers(t, allowed_numbers(a)), (name, value, t)
    assert "(\u22121)" in REGISTRY["jump"].desc_th


def test_percent_overflow_renders_generic():
    a = make_alert(evidence=[("ret_60", 1e307, 1.0)])
    t = explain(a).text_th
    assert "inf" not in t.lower()
    assert explain(a).facts[0][1] == "1e+307"
    assert validate_numbers(t, allowed_numbers(a))


@pytest.mark.parametrize(
    "us,hms",
    [
        (10**20, None),
        (-1, "23:59:59"),
        (-(10**6) * 86400, "00:00:00"),
        (0, "00:00:00"),
        (86399 * 10**6 + 999_999, "23:59:59"),
    ],
)
def test_time_integer_arithmetic(us, hms):
    a = make_alert(ts_event_us=us)
    t = explain(a).text_th
    sod = (us // 10**6) % 86400
    expect = f"{sod // 3600:02d}:{sod // 60 % 60:02d}:{sod % 60:02d}"
    if hms:
        assert expect == hms
    assert f"เวลา {expect} UTC" in t
    assert validate_numbers(t, allowed_numbers(a))


def test_displayed_percent_needed():
    a = make_alert(evidence=[("ret_60", 0.000056, 1.0)])
    assert explain(a).facts[0][1] == "0.01%"
    assert validate_numbers("0.01%", allowed_numbers(a))  # displayed
    assert validate_numbers("0.0056", allowed_numbers(a))  # value * 100
    assert not validate_numbers("0.000056", allowed_numbers(a))  # raw fraction


def test_displayed_unit_and_raw_paths():
    a = make_alert(evidence=[("buy_imb_60", 0.006, 1.0)])
    assert explain(a).facts[0][1] == "0.01"
    assert validate_numbers("0.01", allowed_numbers(a))  # displayed
    assert validate_numbers("0.006", allowed_numbers(a))  # raw
    z = make_alert(evidence=[("z_rush_60", 0.14, 1.0)])
    assert _has(allowed_numbers(z), 0.14) and _has(allowed_numbers(z), 0.1)
    p = make_alert(fused_p=1.049e-4)
    assert "1.0e-04" in explain(p).text_th
    assert validate_numbers("1.0e-04", allowed_numbers(p))  # displayed
    assert validate_numbers("1.049e-4", allowed_numbers(p))  # raw
    assert not validate_numbers("1.2e-4", allowed_numbers(p))


# ---------------------------------------------------------------- technique from another row

OTHER_ROW_TH = "; การจัดประเภทมาจากแถวอื่นในเหตุการณ์เดียวกัน ตัวเลขข้างต้นมาจากจุดที่ผิดปกติที่สุด"


def test_technique_from_another_row_appends_the_exact_clause():
    plain = explain(make_alert()).text_th
    a = make_alert(peak_technique_id="MM-T000")
    t = explain(a).text_th
    assert t == plain + OTHER_ROW_TH
    assert numbers_in(OTHER_ROW_TH) == []
    assert validate_numbers(t, allowed_numbers(a))


@pytest.mark.parametrize("peak", [None, "MM-T001"])
def test_same_or_unset_peak_technique_leaves_the_text_unchanged(peak):
    assert explain(make_alert(peak_technique_id=peak)).text_th == explain(make_alert()).text_th
