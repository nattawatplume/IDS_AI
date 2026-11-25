"""Phase 2: coin history (2.1) and the check-before-buying page (2.2): rule, API and wording."""

import json

import pytest
from starlette.testclient import TestClient

from idsai.product import check as ck
from idsai.product.api import create_app
from idsai.product.explainer import BANNED_TH
from tests.product.test_api import D_US, SYM, env, make_alert, point  # noqa: F401
from tests.product.test_api_auth import aenv, auth_client  # noqa: F401

S = 1_000_000
MIN = 60 * S
DAY = 86_400 * S
NOW = D_US + 40 * DAY
TAU = 4.0
FORBIDDEN = (
    "ความน่าจะเป็น",
    "probability",
    "accuracy",
    "แม่นยำ",
    "ซื้อได้",
    "ห้ามซื้อ",
    *BANNED_TH,
)
NO_LIQ = {"qvol_5m": None, "qvol_median_24h": None, "ratio": None}
CHECK_KEYS = {
    "symbol",
    "now_us",
    "score",
    "alerts_24h",
    "liquidity",
    "history",
    "label",
    "level",
    "reasons_th",
    "disclaimer_th",
}


def score(f):
    return {"F_now": f, "tau": TAU, "t_end_us": NOW}


def liq(ratio):
    return {"qvol_5m": 1.0, "qvol_median_24h": 1.0, "ratio": ratio}


# ---- the label rule: every branch ----


@pytest.mark.parametrize(
    ("sc", "alerts", "ratio", "want"),
    [
        (None, [], None, ck.LABEL_WARMUP),
        ({"F_now": None, "tau": TAU, "t_end_us": NOW}, [], None, ck.LABEL_WARMUP),
        (score(TAU), [], None, ck.LABEL_HIGH),  # F at tau
        (score(9.0), [], 1.0, ck.LABEL_HIGH),
        (score(0.1), [make_alert(ts=NOW - 59 * MIN, status="open")], 1.0, ck.LABEL_HIGH),
        (score(0.1), [make_alert(ts=NOW - 10 * MIN, status="updated")], 1.0, ck.LABEL_HIGH),
        # a closed alert, or an open one older than 60 min, is only เฝ้าระวัง
        (score(0.1), [make_alert(ts=NOW - 10 * MIN, status="closed")], 1.0, ck.LABEL_WATCH),
        (
            score(0.1),
            [make_alert(ts=NOW - 62 * MIN, status="open", window=(NOW - 62 * MIN, NOW - 61 * MIN))],
            1.0,
            ck.LABEL_WATCH,
        ),
        # an episode that started 3 h ago but last exceeded 5 min ago is still live
        (
            score(0.1),
            [
                make_alert(
                    ts=NOW - 3 * 60 * MIN, status="updated", window=(NOW - 180 * MIN, NOW - MIN * 5)
                )
            ],
            1.0,
            ck.LABEL_HIGH,
        ),
        (score(0.5 * TAU), [], 1.0, ck.LABEL_WATCH),  # F at half of tau
        (score(0.5 * TAU - 1e-9), [], 1.0, ck.LABEL_NORMAL),
        (score(0.1), [], 0.29, ck.LABEL_WATCH),  # thin
        (score(0.1), [], ck.THIN_RATIO, ck.LABEL_NORMAL),  # the edges themselves are normal
        (score(0.1), [], ck.SURGE_RATIO, ck.LABEL_NORMAL),
        (score(0.1), [], 5.01, ck.LABEL_WATCH),  # surge
        (score(0.1), [], None, ck.LABEL_NORMAL),  # liquidity unknown: no liquidity reason
        (score(0.1), [], 1.0, ck.LABEL_NORMAL),
    ],
)
def test_label_rule_branches(sc, alerts, ratio, want):
    label, reasons = ck.label_for(sc, alerts, NO_LIQ if ratio is None else liq(ratio), NOW)
    assert label == want
    assert reasons and all(isinstance(r, str) and r for r in reasons)
    if want == ck.LABEL_WARMUP:
        assert reasons == [ck.WARMUP_REASON_TH]


def test_label_reasons_name_each_cause():
    _, r = ck.label_for(score(2.5), [make_alert(ts=NOW - 2 * 3600 * S)], liq(0.1), NOW)
    assert len(r) == 3 and "24 ชม." in r[0] and "(F)" in r[1] and "สภาพคล่องบาง" in r[2]
    _, r = ck.label_for(score(0.1), [], liq(8.0), NOW)
    assert len(r) == 1 and "สูงผิดปกติ" in r[0]


def test_labels_are_only_the_allowed_ones():
    assert set(ck.LEVELS) == {"ปกติ", "เฝ้าระวัง", "ผิดปกติมาก", "ยังไม่มีข้อมูลพอ"}


# ---- liquidity ----


def test_liquidity_needs_60_points():
    assert ck.liquidity([1.0] * 359, 10) == NO_LIQ  # 59 whole minutes at a 10 s stride
    assert ck.liquidity([], 10) == NO_LIQ
    assert ck.liquidity([1.0] * 360, 10)["ratio"] == pytest.approx(1.0)
    assert ck.liquidity([1.0] * 60, 60)["ratio"] == pytest.approx(1.0)  # 60 s stride: 60 points
    assert ck.liquidity([1.0] * 59, 60) == NO_LIQ


def test_liquidity_per_minute_mean_median_and_ratio():
    # 10 s stride: 6 points a minute. 2 h at 1 per point, then the last 5 min at 3 per point.
    vols = [1.0] * 720 + [3.0] * 30
    out = ck.liquidity(vols, 10)
    assert out["qvol_5m"] == pytest.approx(18.0)  # 3 per 10 s = 18 per minute
    assert out["qvol_median_24h"] == pytest.approx(6.0)
    assert out["ratio"] == pytest.approx(3.0)
    zero = ck.liquidity([0.0] * 400 + [1.0] * 30, 10)  # mostly no trading, then a little
    assert zero["qvol_median_24h"] == 0.0 and zero["ratio"] is None
    assert zero["qvol_5m"] == pytest.approx(6.0)


def test_zero_median_is_a_thin_market():
    zero = {"qvol_5m": 6.0, "qvol_median_24h": 0.0, "ratio": None}
    label, reasons = ck.label_for(score(0.1), [], zero, NOW)
    assert label == ck.LABEL_WATCH and reasons == [ck.THIN_ZERO_TH]
    assert "สภาพคล่องบาง" in ck.THIN_ZERO_TH


# ---- history ----


def test_history_counts_windows_phases_and_never_sim(env):  # noqa: F811
    st = env.store
    st.upsert_alert(make_alert("p", NOW - 2 * DAY, phase="pump", technique_id="MM-T001.001"))
    st.upsert_alert(make_alert("s", NOW - 10 * DAY, prov="live", phase="sell"))
    st.upsert_alert(make_alert("u", NOW - 29 * DAY))
    st.upsert_alert(make_alert("x", NOW - 31 * DAY))
    st.upsert_alert(make_alert("sim", NOW - DAY, prov="sim"))
    for status in ("open", "updated", "closed"):  # one episode, upserted three times
        st.upsert_alert(make_alert("ep", NOW - 3 * DAY, status=status))
    h = ck.coin_history(st, SYM, NOW)
    assert h == {
        "symbol": SYM,
        "now_us": NOW,
        "d7": 2,
        "d30": 4,
        "by_phase_d30": {"pump": 1, "sell": 1, "unknown": 2},
    }


def test_now_is_the_pipeline_clock_else_the_wall_clock(env):  # noqa: F811
    env.pipe.latest_us = NOW
    assert ck.now_us_of(env.pipe) == NOW
    env.pipe.latest_us = None
    assert abs(ck.now_us_of(env.pipe) - ck.now_us_of(None)) < 60 * S


# ---- API ----


@pytest.fixture
def client(env):  # noqa: F811
    with TestClient(create_app(env.state)) as c:
        yield c


def test_history_endpoint_shape_and_replay_clock(client, env):  # noqa: F811
    env.pipe.latest_us = NOW
    env.store.upsert_alert(make_alert("a", NOW - DAY))
    env.store.upsert_alert(make_alert("b", NOW - 8 * DAY, phase="sell"))
    r = client.get(f"/api/symbols/{SYM}/history")
    assert r.status_code == 200
    assert r.json() == {
        "symbol": SYM,
        "now_us": NOW,
        "d7": 1,
        "d30": 2,
        "by_phase_d30": {"pump": 0, "sell": 1, "unknown": 1},
    }
    other = client.get("/api/symbols/OTHERBTC/history").json()
    assert other["d7"] == 0 and other["d30"] == 0


@pytest.mark.parametrize("what", ["history", "check"])
def test_unknown_symbol_is_a_thai_404(client, what):
    r = client.get(f"/api/symbols/NOPE/{what}")
    assert r.status_code == 404 and r.json() == {"detail": "ไม่รู้จัก symbol นี้"}


def test_check_during_warmup(client):
    b = client.get(f"/api/symbols/{SYM}/check").json()
    assert set(b) == CHECK_KEYS
    assert b["score"] is None and b["label"] == ck.LABEL_WARMUP and b["level"] == "warmup"
    assert b["reasons_th"] == [ck.WARMUP_REASON_TH]
    assert b["liquidity"] == NO_LIQ
    assert b["alerts_24h"] == {"count": 0, "latest": None}
    assert b["disclaimer_th"] == "ข้อมูลนี้ไม่ใช่คำแนะนำการลงทุน"


def test_check_shape_with_score_alert_and_liquidity(client, env):  # noqa: F811
    pts = [point(i, F=0.1) for i in range(360)]
    pts[-1]["F"] = 3.6  # tau is 3.5 in FakePipe
    env.pipe.points[SYM] = pts
    env.pipe.vols[SYM] = [1.0] * 360
    env.pipe.latest_us = pts[-1]["t_end_us"]
    now = env.pipe.latest_us
    env.store.upsert_alert(make_alert("old", now - 2 * DAY))
    env.store.upsert_alert(make_alert("sim", now - 5 * MIN, prov="sim"))  # never counted
    env.store.upsert_alert(make_alert("new", now - 2 * 3600 * S, status="closed", phase="pump"))
    b = json.loads(client.get(f"/api/symbols/{SYM}/check").text)
    assert set(b) == CHECK_KEYS
    assert b["score"] == {"F_now": 3.6, "tau": 3.5, "t_end_us": now}
    assert b["alerts_24h"] == {
        "count": 1,
        "latest": {
            "alert_id": "new",
            "ts_event_us": now - 2 * 3600 * S,
            "technique_id": "MM-T001",
            "phase": "pump",
            "status": "closed",
        },
    }
    assert b["liquidity"] == {"qvol_5m": 6.0, "qvol_median_24h": 6.0, "ratio": 1.0}
    assert b["history"]["d7"] == 2 and b["history"]["d30"] == 2
    assert b["label"] == "ผิดปกติมาก" and b["level"] == "high"
    text = json.dumps(b, ensure_ascii=False)
    for w in FORBIDDEN:
        assert w not in text, w


def test_check_without_pipeline_is_warmup(env):  # noqa: F811
    env.state.pipeline = None
    with TestClient(create_app(env.state)) as c:
        b = c.get(f"/api/symbols/{SYM}/check").json()
    assert b["label"] == ck.LABEL_WARMUP and b["liquidity"] == NO_LIQ


def test_series_output_is_unchanged_by_the_check(client, env):  # noqa: F811
    env.pipe.points[SYM] = [point(i) for i in range(3)]
    before = client.get(f"/api/series?symbol={SYM}").content
    client.get(f"/api/symbols/{SYM}/check")
    assert client.get(f"/api/series?symbol={SYM}").content == before
    keys = {"t_end_us", "close", "F", "lane_rule", "lane_stat", "lane_ai", "tau"}
    assert set(json.loads(before)) == keys


@pytest.mark.parametrize("what", ["history", "check"])
def test_history_and_check_need_sign_in(auth_client, what):  # noqa: F811
    assert auth_client().get(f"/api/symbols/{SYM}/{what}").status_code == 401
    assert auth_client("viewer").get(f"/api/symbols/{SYM}/{what}").status_code == 200


# ---- web: what app.js reads exists in what the server sends; the page's wording ----


def test_check_fields_read_by_app_js_exist(client, env):  # noqa: F811
    from tests.product.test_web_static import reads, text

    env.pipe.points[SYM] = [point(i) for i in range(70)]
    env.pipe.latest_us = env.pipe.points[SYM][-1]["t_end_us"]
    env.store.upsert_alert(make_alert("n", env.pipe.latest_us - MIN))
    b = client.get(f"/api/symbols/{SYM}/check").json()
    assert reads("chk") and reads("chk") <= set(b)
    assert reads("score") <= set(b["score"])
    assert reads("day") <= set(b["alerts_24h"])
    assert reads("latest") <= set(b["alerts_24h"]["latest"])
    assert reads("liq") <= set(b["liquidity"])
    hist = client.get(f"/api/symbols/{SYM}/history").json()
    assert hist == b["history"]
    assert reads("hist") and reads("hist") <= set(hist)
    assert reads("phases") <= set(hist["by_phase_d30"])
    js, html = text("app.js"), text("index.html")
    assert "ข้อมูลนี้ไม่ใช่คำแนะนำการลงทุน" in html
    assert "/api/symbols/${encodeURIComponent(symbol)}/${what}" in js
    for level in ("normal", "watch", "high", "warmup"):
        assert f'"{level}"' in js
    page = html[html.index('id="page-check"') : html.index("<!-- Simulate")]
    for word in ("ซื้อได้", "ห้ามซื้อ", "สัญญาณซื้อ ", "ความน่าจะเป็น"):
        assert word not in page and word not in js, word


def test_check_page_debounces_alert_refresh_and_resets_on_change_or_failure():
    from tests.product.test_web_static import text

    js = text("app.js")
    body = js[js.index("function coinAlertsChanged(") :]
    body = body[: body.index("\n}\n")]
    assert "setTimeout(" in body and "COIN_REFRESH_MS" in body
    assert "loadSymbolHistory()" not in body.split("setTimeout(")[0]  # never per message
    assert "const COIN_REFRESH_MS = 1000;" in js
    load = js[js.index("async function loadCheck(") :]
    load = load[: load.index("\n}\n")]
    assert 'resetCheck("–")' in load and "e.status === 404" in load and "e.message" in load
    change = js[js.index('$("check-symbol").addEventListener("change"') :]
    assert change.index("resetCheck(") < change.index("loadCheck()")
    reset = js[js.index("function resetCheck(") :]
    reset = reset[: reset.index("\n}\n")]
    assert 'dataset.level = "warmup"' in reset
    for tile in ("check-score", "check-alerts", "check-liq", "check-hist"):
        assert f'"{tile}"' in reset and f'"{tile}-sub"' in reset
