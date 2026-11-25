"""The web alert history line (1.3) and the track record (2.3): outcomes, API and the page."""

import pytest
from starlette.testclient import TestClient

from idsai.product.api import create_app
from idsai.product.explainer import HISTORY_TH, history_line
from idsai.product.notify import telegram as tg
from tests.product.test_api import D_US, SYM, env, make_alert  # noqa: F401
from tests.product.test_api_auth import aenv, auth_client  # noqa: F401

S = 1_000_000
MIN = 60 * S
HOUR = 60 * MIN


@pytest.fixture
def client(env):  # noqa: F811
    with TestClient(create_app(env.state)) as c:
        yield c


# ---- 1.3: the history line on the web alert detail ----


@pytest.mark.parametrize(
    ("phase", "want"), [("pump", HISTORY_TH), (None, HISTORY_TH), ("sell", None)]
)
def test_alert_detail_history_line_by_phase(client, env, phase, want):  # noqa: F811
    env.store.upsert_alert(make_alert("h1", phase=phase))
    body = client.get("/api/alerts/h1").json()
    assert body["history_th"] == want
    assert history_line(make_alert("x", phase=phase)) == want


def test_telegram_text_keeps_the_same_history_line():
    assert tg.HISTORY_TH is HISTORY_TH
    assert HISTORY_TH == (
        "ข้อมูลในอดีต (pump จริงที่ระบบจับได้ คู่ BTC 2018–2021): 1 ชม. หลังเตือน "
        "ราคาต่ำกว่าตอนเตือน 89% ของครั้ง แต่ในชั่วโมงนั้นราคายังขึ้นต่อได้ (มัธยฐาน +15%)"
    )
    ts = 1_700_000_000 * S
    pump = tg.message_text(make_alert("p", ts=ts, phase="pump"))
    assert pump.splitlines()[3] == HISTORY_TH
    assert HISTORY_TH not in tg.message_text(make_alert("s", ts=ts, phase="sell"))


def test_web_alert_detail_shows_the_history_line_muted_with_its_source():
    from tests.product.test_web_static import text

    js = text("app.js")
    note = js[js.index("function historyNote(a)") :]
    note = note[: note.index("\n}\n")]
    assert "a.history_th" in note and '"small muted history-note"' in note
    assert 'const HISTORY_SOURCE = "ที่มา: ผลเฟส 0 ใน docs/ROADMAP.md";' in js
    show = js[js.index("async function showAlert(id)") :]
    assert "historyNote(a)" in show[: show.index("\n}\n")]
