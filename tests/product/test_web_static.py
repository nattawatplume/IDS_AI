"""Static dashboard: served, vendored chart pinned, no network, safe DOM, Thai wording."""

import dataclasses
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from starlette.testclient import TestClient

from idsai.ai.auditor import Auditor, Check
from idsai.config import load_settings
from idsai.contracts.alert import Alert, DetectorRef
from idsai.contracts.frames import LABELS
from idsai.data.store import Store
from idsai.detection.features.bars import BARS
from idsai.detection.simulator.scenarios import Scenario
from idsai.eval import report
from idsai.product.api import AppState, create_app
from idsai.product.api.app import WEB_DIR
from idsai.product.explainer import BANNED_TH

VENDOR_JS = "vendor/lightweight-charts.standalone.production.js"
PAGES = ("index.html", "app.js", "styles.css")
ROUTES = ("#/home", "#/alerts", "#/simulate", "#/results", "#/activity")
NAV_TH = ("หน้าหลัก", "การแจ้งเตือน", "ทดลองจำลอง", "ผลวัดระบบ")
# the only absolute links the page may carry: attribution and About anchors, never a script source
ALLOWED_URLS = {
    "https://www.tradingview.com/",
    "https://github.com/tradingview/lightweight-charts",
    "https://www.apache.org/licenses/LICENSE-2.0",
    "https://data.binance.vision/",
    "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "http://www.w3.org/2000/svg",  # the SVG namespace name, never fetched
}
FORBIDDEN_WORDS = ("ความน่าจะเป็น", "probability", "accuracy", "แม่นยำ", *BANNED_TH)


def text(name: str) -> str:
    return (WEB_DIR / name).read_text(encoding="utf-8")


@pytest.fixture
def client(tmp_path):
    cfg = load_settings(
        overrides={"paths": {"data_dir": str(tmp_path / "d"), "app_db": str(tmp_path / "a.duckdb")}}
    )
    store = Store(cfg.paths.data_dir, cfg.paths.app_db)
    state = AppState(cfg, store, "replay", None, None, symbols=["TESTBTC"])
    with TestClient(create_app(state)) as c:
        yield c
    store.close()


def test_root_serves_the_dashboard_with_attribution(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    for needle in ("Market IDS", "Binance Vision", "tradingview", "CC BY-NC-SA 4.0"):
        assert needle in r.text


def test_dashboard_files_revalidate_on_every_load(client):
    for path in ("/", "/app.js", "/styles.css"):
        r = client.get(path)
        assert r.status_code == 200 and r.headers["cache-control"] == "no-cache", path
        assert r.headers.get("etag"), path  # revalidation can still answer 304
        again = client.get(path, headers={"if-none-match": r.headers["etag"]})
        assert again.status_code == 304, path


@pytest.mark.parametrize(
    ("path", "ctype"),
    [("app.js", "javascript"), ("styles.css", "text/css"), (VENDOR_JS, "javascript")],
)
def test_assets_are_served_with_their_type(client, path, ctype):
    r = client.get("/" + path)
    assert r.status_code == 200 and ctype in r.headers["content-type"]


def test_vendor_is_pinned_by_version_and_sha256(client):
    version = text("vendor/VERSION")
    assert "lightweight-charts" in version and "5.2.1" in version
    sha = re.search(r"sha256[:\s]+([0-9a-f]{64})", version)
    assert sha, "vendor/VERSION must record the sha256"
    body = client.get("/" + VENDOR_JS).content
    assert hashlib.sha256(body).hexdigest() == sha.group(1)
    assert hashlib.sha256((WEB_DIR / VENDOR_JS).read_bytes()).hexdigest() == sha.group(1)
    assert body.startswith(b"/*!\n * @license\n * TradingView Lightweight Charts")
    notice = text("vendor/NOTICE-lightweight-charts")
    assert "Apache License" in notice and "TradingView" in notice


def test_page_loads_nothing_from_the_network():
    for name in PAGES:
        urls = set(re.findall(r"https?://[^\s\"'<>)`]+", text(name)))
        assert urls <= ALLOWED_URLS, (name, urls - ALLOWED_URLS)
    html = text("index.html")
    # scripts and stylesheets are relative; absolute URLs only ever appear in anchors
    for tag in re.findall(r"<(?:script|link|img|iframe)[^>]*>", html):
        assert "://" not in tag and not re.search(r"(?:src|href)=\"//", tag), tag
    js = text("app.js")
    for url in ALLOWED_URLS:
        js = js.replace(url, "")
    assert "://" not in js  # fetch and WebSocket URLs are relative / built from location
    assert "@import" not in text("styles.css") and "url(" not in text("styles.css")


def test_app_js_never_parses_html():
    js = text("app.js")
    for api in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert api not in js, api


def test_ui_copy_wording():
    for name in PAGES:
        low = text(name).lower()
        for word in FORBIDDEN_WORDS:
            assert word.lower() not in low, (name, word)
    copy = text("index.html") + text("app.js")
    assert "ความผิดปกติทางสถิติ" in copy
    assert "ค่า p รวม" in copy and "ความหายากเทียบตลาดปกติ" in copy
    assert "prevalence" in copy  # precision is never shown without it
    for badge in ("REAL", "SEMI-SYN", "SYN"):
        assert badge in copy


def test_every_route_is_present():
    both = text("index.html") + text("app.js")
    for route in ROUTES:
        assert route in both, route
    assert "#/alerts/" in text("app.js")  # alert detail: #/alerts/{id}


def test_app_js_is_valid_javascript():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    r = subprocess.run([node, "--check", str(WEB_DIR / "app.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ---- contract: what app.js reads exists in what the server really sends ----


def reads(var: str) -> set[str]:
    """Every `var.<name>` that app.js reads (not `x.var.<name>`)."""
    return set(re.findall(rf"(?<![\w.]){re.escape(var)}\.([A-Za-z_]\w*)", text("app.js")))


def test_alert_fields_read_by_app_js_exist():
    alert = set(Alert.model_fields) | {"explanation_th"}  # list item (app.alert_item)
    detail = {"explanation", "feedback"}  # GET /api/alerts/{id}
    dom = {"dataset", "setAttribute", "removeAttribute", "time"}  # `a` is also a tab link / marker
    assert reads("a") - alert - detail - dom == set()
    detected = {"alert_id", "eventS", "wallS"}  # run.detected
    other = {"msg", "getTime", "toISOString"}  # a 422 detail item, a Date
    assert reads("d") <= set(DetectorRef.model_fields) | detected | other
    feedback = {"alert_id", "verdict", "analyst", "ts_us"}  # Store.list_feedback columns
    assert reads("f") <= feedback | {"label", "value"}  # explanation facts


def test_stream_and_read_fields_read_by_app_js_exist(client):
    series = set(client.get("/api/series", params={"symbol": "TESTBTC"}).json())
    assert {"t_end_us", "close", "F", "tau", "lane_rule", "lane_stat", "lane_ai"} <= series
    for lane in re.findall(r'\["(lane_\w+)",', text("app.js")):
        assert lane in series, lane
    assert reads("s") <= series | {"id", "name_th", "definition_th"} | {"mean", "ci"}
    score = (series - {"tau"}) | {"symbol"}  # a WS score = {symbol, **series point}
    assert reads("p") <= score | {"time", "toExponential", "toFixed"}  # also a price point / p
    assert reads("bar") <= set(BARS) | {"symbol"}
    assert reads("h") <= set(client.get("/api/health").json())
    assert reads("warm") <= set(Check.model_fields)
    assert "hours_remaining" in Auditor(load_settings()).warmup(3).value
    assert "hours_remaining" in text("app.js")
    assert reads("label") <= set(LABELS)
    assert reads("sc") <= {f.name for f in dataclasses.fields(Scenario)}
    entry = client.get("/api/taxonomy").json()["MM-T001"]
    assert reads("t") <= set(entry)
    assert reads("sub") <= set(entry["sub_techniques"][0])
    assert client.get("/api/symbols").json() == ["TESTBTC"]  # a plain list of names


def test_report_fields_read_by_app_js_exist(tmp_path):
    cfg = load_settings()
    with Store(tmp_path / "d", str(tmp_path / "r.duckdb")) as store:
        top = report.build_report(store, tmp_path / "r.md", tmp_path / "r.json", cfg=cfg)
    assert reads("report") <= set(top)
    assert reads("cell") <= set(report._cell([{"provenance": "REAL", "seed": 0}], 10))
    assert reads("m") <= set(report.MEAN_METRICS)
    assert reads("ci") <= set(report.CI_METRICS)
    r3 = report._rule_of_three(np.array([0]), np.array([3]), report.UNIT_EVENT)
    assert reads("r3") <= set(r3) and "rule_of_three" in text("app.js")
    assert reads("block") <= set(report._retrain_shell("v1", "v2", None)) | {"ai_only"}
    assert reads("paired") <= set(report._paired([], 10, 0.02))
    # a paired row as report._paired builds it; `row` is also an alerts table row
    row = {"baseline", "n_seeds", "n_events", "diff_mean", "ci", "significant", "reason_th"}
    assert reads("row") <= row | {"provenance", "addEventListener"}


def test_ws_types_match_the_server():
    js = text("app.js")
    server = (WEB_DIR.parent / "api" / "app.py").read_text(encoding="utf-8")
    for kind in ("bar", "score", "alert", "health"):
        assert f'type === "{kind}"' in js and f'"{kind}"' in server, kind
    for event in ("alerts_reset", "resync"):
        assert f'"{event}"' in js, event


def test_every_element_app_js_targets_exists():
    html, js = text("index.html"), text("app.js")
    ids = set(re.findall(r'\bid="([\w-]+)"', html))
    targets = set(re.findall(r'\$\("([\w-]+)"\)', js)) | set(re.findall(r'"(page-[\w-]+)"', js))
    assert targets and targets - ids == set()
    for name in ("home", "alerts", "simulate", "results"):
        assert f'data-route="{name}"' in html and f'"page-{name}"' in js
    for page in ("home", "alerts", "alert", "simulate", "results", "activity", "login"):
        assert f'id="page-{page}"' in html, page


def test_live_mode_gates_replay_and_red_team():
    js = text("app.js")
    body = js[js.index("function setMode(") : js.index("function eventNowUs(")]
    assert 'const live = mode === "live";' in body
    assert '$("replay-bar").hidden = live;' in body
    assert "applyRoleGates();" in body  # the simulate gate runs on every mode change
    reason = js[js.index("function simReason(") :]
    reason = reason[: reason.index("\n}\n")]
    assert "if (!redTeamAllowed(mode))" in reason  # the mode is checked first: fail closed
    start = js[js.index("async function startSimulation(") :]
    assert start.index("if (!redTeamAllowed(state.mode)) return;") < start.index("postJson(")
    assert 'const RED_TEAM_MODES = ["replay", "sim"];' in js


def _simulate_page(html: str) -> str:
    return re.search(r'<section id="page-simulate".*?</section>', html, re.S).group(0)


def test_red_team_form_starts_disabled_until_health_confirms_the_mode():
    page = _simulate_page(text("index.html"))
    controls = re.findall(r"<(?:select|input|button)\b[^>]*>", page)
    assert len(controls) == 6  # three cards, region, offset, symbol
    assert sum('class="card sim-card"' in tag for tag in controls) == 3
    for tag in controls:
        assert re.search(r"\sdisabled[\s>]", tag), tag  # fail closed before any health
    reason = re.search(r'<p id="sim-reason"[^>]*>([^<]*)</p>', page)
    assert reason and "hidden" not in reason.group(0)
    assert reason.group(1) == "รอยืนยันจากเซิร์ฟเวอร์ว่าเป็นโหมดย้อนดู"
    assert 'SIM_WAIT_TH = "รอยืนยันจากเซิร์ฟเวอร์ว่าเป็นโหมดย้อนดู"' in text("app.js")


def test_top_bar_clock_moves_the_red_team_card():
    js = text("app.js")
    body = js[js.index("function renderHealth(") : js.index("function redTeamAllowed(")]
    assert "tickInjections();" in body


# ---- time-to-detect logic, run in node ----

TTD_JS = r"""
const assert = require("assert");
const m = require(process.argv[2]);
const S = 1e6;
const alert = (id, w0, w1, symbol = "X") => ({ alert_id: id, symbol, window: [w0 * S, w1 * S] });
// cursor at 1000 s, scenario 1030-1150 s: matching window [910, 1450]
const old = alert("old", 900, 990); // ended before the click, overlaps the widened window
const run = (known) => ({
  symbol: "X", t_start_us: 1030 * S, t_end_us: 1150 * S, cursor_us: 1000 * S, known,
});

let r = run(m.snapshotAlerts([old]));
assert.strictEqual(m.firstDetection(r, [old]), null, "a pre-existing alert is not a detection");
assert.strictEqual(m.firstDetection(r, [old, alert("new", 1100, 1200)]).alert_id, "new");
assert.strictEqual(m.firstDetection(r, [alert("new", 1100, 1200, "Y")]), null); // other symbol
assert.strictEqual(m.firstDetection(r, [alert("late", 1500, 1600)]), null); // outside the window

// an alert open at the click counts only once it grows past the cursor
r = run(m.snapshotAlerts([alert("open", 950, 995)]));
assert.strictEqual(m.firstDetection(r, [alert("open", 950, 998)]), null);
assert.strictEqual(m.firstDetection(r, [alert("open", 950, 1080)]).alert_id, "open");

// alerts_reset: a deleted detection re-arms the run, and a deleted id that comes back is new
r = run(m.snapshotAlerts([old]));
r.detected = { alert_id: "new" };
assert.strictEqual(m.resetRuns([r], ["new", "old"]), true);
assert.strictEqual(r.detected, null);
assert.strictEqual(m.firstDetection(r, [alert("old", 1040, 1100)]).alert_id, "old");
assert.strictEqual(m.resetRuns([r], ["unrelated"]), false);

// not detected: once the clock passes scenario end + 300 s + 30 s grace with nothing qualifying
r = run(m.snapshotAlerts([]));
assert.strictEqual(m.injectionStatus(r, 1010 * S), "pending");
assert.strictEqual(m.injectionStatus(r, 1200 * S), "waiting");
assert.strictEqual(m.injectionStatus(r, (1150 + 300 + 29) * S), "waiting");
assert.strictEqual(m.injectionStatus(r, (1150 + 300 + 31) * S), "missed");
assert.strictEqual(m.injectionStatus(r, null), "waiting"); // no clock yet: no verdict
// a late qualifying alert still turns it into a detection; a seek back re-arms the wait
const late = m.firstDetection(r, [alert("late", 1440, 1460)]);
assert.strictEqual(late.alert_id, "late");
r.detected = { alert_id: late.alert_id };
assert.strictEqual(m.injectionStatus(r, 5000 * S), "detected");
r.detected = null;
assert.strictEqual(m.injectionStatus(r, 1100 * S), "waiting");

// red team is allowed only in a confirmed replay / sim mode
for (const mode of [null, undefined, "", "live", "LIVE", "other"]) {
  assert.strictEqual(m.redTeamAllowed(mode), false, String(mode));
}
assert.strictEqual(m.redTeamAllowed("replay") && m.redTeamAllowed("sim"), true);
console.log("ok");
"""


def test_time_to_detect_counts_only_new_alerts(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = tmp_path / "ttd.js"
    script.write_text(TTD_JS, encoding="utf-8")
    js = str(WEB_DIR / "app.js")
    r = subprocess.run([node, str(script), js], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


HTML_TAGS = set(
    "a button dd details div dl dt em h1 h2 h3 h4 input label li ol option p section select "
    "small span summary"
    " strong table tbody td th thead tr ul".split()
)


def test_el_builds_only_known_tags():
    tags = set(re.findall(r'\bel\("([^"]+)"', text("app.js")))
    assert tags and tags <= HTML_TAGS, sorted(tags - HTML_TAGS)


def _nav_block(html: str) -> str:
    m = re.search(r'<nav class="nav"[^>]*>(.*?)</nav>', html, re.S)
    assert m, "no main nav"
    return m.group(1)


def test_shell_markup():
    html = text("index.html")
    links = re.findall(r"<a\b([^>]*)>(.*?)</a>", _nav_block(html), re.S)
    labels = [" ".join(re.sub(r"<[^>]+>", " ", body).split()) for _, body in links]
    assert tuple(labels) == NAV_TH, labels
    for attrs, _ in links:
        assert "data-route=" in attrs
    assert 'id="theme-toggle"' not in html and "data-theme-choice" not in html
    assert 'id="help-btn"' in html
    js = text("app.js")
    login = js[js.index("function bindLogin(") :]
    assert '"Escape"' in login[: login.index("\n}\n")]  # the user menu closes on Escape
    for i in ("menu-activity", "menu-tour", "btn-logout"):
        assert f'id="{i}"' in html, i
    js = text("app.js")
    for gone in ("THEMES", "applyTheme", "chooseTheme", "bindTheme"):
        assert gone not in js, gone
    assert 'ROLE_TH = { viewer: "ผู้ชม", analyst: "ผู้วิเคราะห์", admin: "ผู้ดูแล" }' in js


def test_glass_tokens():
    css = text("styles.css")
    for decl in ("--bg: #0b0820", "--ok: #3ee08f", "--warn: #ffb547", "--bad: #ff5d73"):
        assert decl in css, decl
    assert "backdrop-filter" in css and "@supports not (backdrop-filter" in css
    assert "prefers-reduced-motion: reduce" in css
    assert "background-clip: text" not in css and "-webkit-background-clip" not in css
    assert 'data-theme="light"' not in css and "prefers-color-scheme" not in css
    # the accent gradient: only on the primary button and the active nav pill
    accent = list(re.finditer(r"([^{}]+)\{[^}]*linear-gradient\([^)]*--violet", css))
    assert accent
    for m in accent:
        sel = " ".join(m.group(1).split())
        assert "primary" in sel or 'aria-current="page"' in sel, sel
    assert "#8b5cf6" in css and "#3b82f6" in css
    assert "@keyframes drift" in css and ".bg" in css
    weights = set(re.findall(r"font-weight:\s*(\d+)", css))
    assert weights <= {"400", "600"}, weights


def test_background_blobs_stay_subtle():
    css = text("styles.css")
    base = re.search(r"(?m)^\.blob\s*\{([^}]*)\}", css).group(1)
    assert "opacity" not in base  # each blob carries its own, so the sum below is the whole story
    total = 0.0
    for i in (1, 2, 3):
        body = re.search(rf"(?m)^\.blob-{i}\s*\{{([^}}]*)\}}", css).group(1)
        total += float(re.search(r"opacity:\s*([\d.]+)", body).group(1))
    assert total <= 0.35 + 1e-9, total
    assert len(re.findall(r'class="blob blob-\d"', text("index.html"))) == 3


def test_top_bar_has_a_mid_breakpoint():
    css = text("styles.css")
    m = re.search(r"@media \(max-width: 1200px\) \{(.*?)\n\}", css, re.S)
    assert m, "no tablet breakpoint for the top bar"
    for sel in (".brand-name", "#user-name", ".nav a"):
        assert sel in m.group(1), sel
    assert "viewport-fit=cover" in text("index.html")


# emoji and pictographs; dingbats such as check marks and play/pause glyphs stay allowed
EMOJI = re.compile("[\U0001f300-\U0001faff☀-⛿]")
THAI = re.compile("[฀-๿]")
JS_STRING = re.compile(r'"(?:[^"\\\n]|\\.)*"' + r"|'(?:[^'\\\n]|\\.)*'" + r"|`(?:[^`\\]|\\.)*`")


def test_no_emoji_or_bang_in_copy():
    html_text = re.sub(r"<!--.*?-->|<[^>]+>", "\n", text("index.html"), flags=re.S)
    js_strings = JS_STRING.findall(text("app.js"))
    for chunk in [html_text, *js_strings]:
        assert not EMOJI.search(chunk), chunk[:80]
    for chunk in [*html_text.split("\n"), *js_strings]:
        if THAI.search(chunk):
            assert "!" not in chunk, chunk[:80]


def test_auditor_and_budget_live_in_the_home_cards_not_the_header():
    html, js = text("index.html"), text("app.js")
    header = re.search(r"<header\b.*?</header>", html, re.S).group(0)
    for gone in ('id="auditor"', "auditor-text", "budget-meter", "budget-text", "งบเตือน"):
        assert gone not in header, gone
    alerts = re.search(r'<div id="card-alerts".*?<!-- /card-alerts -->', html, re.S).group(0)
    assert 'id="budget-line"' in alerts and 'id="alerts-count"' in alerts
    budget = js[js.index("function renderBudget(") :]
    assert "ใช้งบแจ้งเตือนไป" in budget[: budget.index("\n}\n")]
    health = re.search(r'<details id="card-health".*?</details>', html, re.S).group(0)
    assert 'id="health-checks"' in health and "<summary" in health
    # every check is named in Thai: no raw auditor names on screen
    for name in ("alert_rate", "calibration", "latency", "trade_gaps", "staleness", "warmup"):
        assert re.search(rf'\b{name}: "[^"]*[\u0e00-\u0e7f]', js), name


MOTION_JS = r"""
const assert = require("assert");
globalThis.matchMedia = () => ({ matches: %s });
const m = require(process.argv[2]);
%s
console.log("ok");
"""

PRESENTER_BODY = r"""
const k = m.presenterKey;
assert.strictEqual(k({ key: " ", target: { tagName: "INPUT" } }), null);
assert.strictEqual(k({ key: " ", target: { tagName: "SELECT" } }), null);
assert.strictEqual(k({ key: " ", target: { tagName: "TEXTAREA" } }), null);
assert.strictEqual(k({ key: "1", target: { tagName: "INPUT" } }), null);
assert.strictEqual(k({ key: "?", target: { tagName: "TEXTAREA" } }), null);
// a focused button, menu or link keeps Space for itself, but the page and help keys still work
for (const tagName of ["BUTTON", "SUMMARY", "A"]) {
  assert.strictEqual(k({ key: " ", target: { tagName } }), null, tagName);
  assert.strictEqual(k({ code: "Space", target: { tagName } }), null, tagName);
  assert.strictEqual(k({ key: "2", target: { tagName } }), "#/alerts", tagName);
  assert.strictEqual(k({ code: "Digit4", target: { tagName } }), "#/results", tagName);
  assert.strictEqual(k({ key: "?", target: { tagName } }), "help", tagName);
}
assert.strictEqual(k({ key: "2", target: { tagName: "SELECT" } }), null);
assert.strictEqual(k({ key: "?", target: { tagName: "INPUT" } }), null);
assert.strictEqual(k({ key: "3", target: { tagName: "DIV", isContentEditable: true } }), null);
assert.strictEqual(k({ key: " ", target: { tagName: "DIV", isContentEditable: true } }), null);
assert.strictEqual(k({ key: " ", target: { tagName: "DIV" } }), "play");
assert.strictEqual(k({ key: " " }), "play");
assert.strictEqual(k({ key: "?" }), "help");
assert.strictEqual(k({ key: "1" }), "#/home");
assert.strictEqual(k({ key: "2", target: { tagName: "BODY" } }), "#/alerts");
assert.strictEqual(k({ key: "3", target: { tagName: "BODY" } }), "#/simulate");
assert.strictEqual(k({ code: "Digit4" }), "#/results");
assert.strictEqual(k({ key: "5", target: { tagName: "BODY" } }), null);
assert.strictEqual(k({ key: "t" }), null);
assert.strictEqual(k({ key: "T" }), null);
assert.strictEqual(k({ key: "t", ctrlKey: true, target: { tagName: "BODY" } }), null);
assert.strictEqual(k({ key: "1", metaKey: true, target: { tagName: "BODY" } }), null);
assert.strictEqual(k({ key: "1", altKey: true, target: { tagName: "BODY" } }), null);
// a Thai layout changes ev.key: the physical ev.code still drives the keys
assert.strictEqual(k({ key: "ะ", code: "KeyT", target: { tagName: "BODY" } }), null);
assert.strictEqual(k({ key: "ๅ", code: "Digit1", target: { tagName: "BODY" } }), "#/home");
assert.strictEqual(k({ key: "ๅ", code: "Numpad3", target: { tagName: "BODY" } }), "#/simulate");
assert.strictEqual(k({ key: "ๅ", code: "Digit5", target: { tagName: "BODY" } }), null);
assert.strictEqual(k({ key: "ฦ", code: "Slash", shiftKey: true }), "help");
assert.strictEqual(k({ key: "Spacebar", code: "Space", target: { tagName: "BODY" } }), "play");
assert.deepStrictEqual(m.PAGE_ORDER, ["#/home", "#/alerts", "#/simulate", "#/results"]);
assert.strictEqual(m.DEFAULT_ROUTE, "#/home");
"""

ROUTE_BODY = r"""
const r = m.routeFor;
assert.deepStrictEqual(r("#/monitor"), { route: "#/home", help: false });
assert.deepStrictEqual(r("#/redteam"), { route: "#/simulate", help: false });
assert.deepStrictEqual(r("#/eval"), { route: "#/results", help: false });
assert.deepStrictEqual(r("#/about"), { route: "#/home", help: true });
assert.deepStrictEqual(r("#/taxonomy"), { route: "#/home", help: true });
assert.deepStrictEqual(r("#/alerts/abc"), { route: "#/alerts/abc", help: false });
assert.deepStrictEqual(r("#/alerts"), { route: "#/alerts", help: false });
assert.deepStrictEqual(r("#/home"), { route: "#/home", help: false });
assert.deepStrictEqual(r("#/simulate"), { route: "#/simulate", help: false });
assert.deepStrictEqual(r("#/results"), { route: "#/results", help: false });
assert.deepStrictEqual(r("#/activity"), { route: "#/activity", help: false });
const login = "#/login?next=%23%2Falerts";
assert.deepStrictEqual(r(login), { route: login, help: false });
assert.deepStrictEqual(r("#/login"), { route: "#/login", help: false });
assert.deepStrictEqual(r(""), { route: "#/home", help: false });
assert.deepStrictEqual(r(undefined), { route: "#/home", help: false });
assert.deepStrictEqual(r("#/nope"), { route: "#/home", help: false });
assert.deepStrictEqual(r("#/monitorx"), { route: "#/home", help: false });
assert.deepStrictEqual(r("#/alerts/"), { route: "#/alerts", help: false });
"""

TWEEN_BODY = r"""
const node = { textContent: "0" };
m.tweenNumber(node, 3.87, 2);
assert.strictEqual(node.textContent, "3.87");
assert.strictEqual(m.reducedMotion(), true);
"""


def _run_motion(tmp_path, reduced, body):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = tmp_path / "motion.js"
    script.write_text(MOTION_JS % (reduced, body), encoding="utf-8")
    r = subprocess.run(
        [node, str(script), str(WEB_DIR / "app.js")], capture_output=True, text=True, cwd=tmp_path
    )
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


def test_presenter_keys_skip_fields(tmp_path):
    _run_motion(tmp_path, "false", PRESENTER_BODY)


def test_route_for_redirects(tmp_path):
    _run_motion(tmp_path, "false", ROUTE_BODY)


def test_tween_snaps_under_reduced_motion(tmp_path):
    _run_motion(tmp_path, "true", TWEEN_BODY)


def test_reduced_motion_disables_animation():
    css = text("styles.css")
    block = css.split("prefers-reduced-motion: reduce")[1]
    assert "animation" in block and "transition" in block
    for name in ("pulse", "glow", "pop", "rise", "grow"):
        assert f"@keyframes {name}" in css, name


PLAY_BODY = r"""
assert.strictEqual(m.playToggleAction("live", false), null);
assert.strictEqual(m.playToggleAction("live", true), null);
assert.strictEqual(m.playToggleAction(null, true), null);
assert.strictEqual(m.playToggleAction("replay", true), "pause");
assert.strictEqual(m.playToggleAction("replay", false), "play");
// state follows what control() reports, so Space always does the opposite
assert.strictEqual(m.getReplaying(), true);
m.noteReplayAction("pause");
assert.strictEqual(m.playToggleAction("replay", m.getReplaying()), "play");
m.noteReplayAction("play"); // e.g. the Play button
assert.strictEqual(m.playToggleAction("replay", m.getReplaying()), "pause");
m.noteReplayAction("speed");
assert.strictEqual(m.getReplaying(), true);
"""


def test_space_toggle_follows_replay_state_and_skips_live(tmp_path):
    _run_motion(tmp_path, "false", PLAY_BODY)
    js = text("app.js")
    assert "noteReplayAction(action)" in js and "presenterPlaying" not in js


# ---- monitor: kpi strip, alert rail, synced crosshair ----


def _home(html: str) -> str:
    return re.search(r'<section id="page-home".*?</section>', html, re.S).group(0)


def test_home_markup():
    html = text("index.html")
    home = _home(html)
    for i in (
        "hero-orb", "hero-title", "hero-line", "card-alerts", "card-symbol", "card-health",
        "chart-card", "replay-bar", "rail", "alert-rail", "layers-toggle", "chart-info-body",
    ):  # fmt: skip
        assert f'id="{i}"' in home, i
    assert re.search(r'<details\b[^>]*\bid="chart-details"', home), "chart details is a <details>"
    details = re.search(r'<details\b[^>]*\bid="chart-details".*?</details>', home, re.S).group(0)
    assert 'id="layers-toggle"' in details and 'id="chart-info-body"' in details
    assert "ดูรายละเอียด" in details
    bar = re.search(r'<div id="replay-bar".*?<!-- /replay-bar -->', home, re.S).group(0)
    chips = re.findall(r'<button\b[^>]*\bclass="[^"]*\bchip\b[^"]*"[^>]*>', bar)
    assert len(chips) == 4, chips
    adv = re.search(r'<details\b[^>]*>(?:(?!</details>).)*\bid="advanced-seek"', bar, re.S)
    assert adv, "the seek field sits behind ขั้นสูง in a <details>"
    assert "ขั้นสูง" in bar and "ไปที่เวลา" in bar
    # the old KPI strip and info button are gone
    for gone in ("kpi-", 'class="kpis"', 'id="chart-info"', 'id="replay-controls"'):
        assert gone not in html, gone
    assert "ค่า p รวม" in home and "ความหายากเทียบตลาดปกติ" in home
    # the home page keeps one primary action: the play button
    assert len(re.findall(r'class="[^"]*\bprimary\b', home)) == 1


def test_chart_colours_and_titles():
    css, js = text("styles.css"), text("app.js")
    for decl in ("--line-price: #a5b4fc", "--line-f: #8b5cf6", "--tau: #ffb547"):
        assert decl in css, decl
    init = js[js.index("function initChart(") : js.index("function addLanes(")]
    assert "ระดับความผิดปกติ" in init and "glow" in init
    tau = js[js.index("function setTau(") : js.index("async function loadSeries(")]
    assert 'title: "เส้นเตือน"' in tau and "Dashed" in tau


def test_glass_cards_lift_on_hover_but_not_under_reduced_motion():
    css = text("styles.css")
    hover = re.search(r"(?m)^\.card:hover\s*\{([^}]*)\}", css)
    assert hover and "translateY(-2px)" in hover.group(1) and "--glass-border-hi" in hover.group(1)
    reduced = css.split("prefers-reduced-motion: reduce")[1]
    assert re.search(r"\.card:hover[^{]*\{[^}]*transform:\s*none", reduced)


def test_hero_and_rail_render_only_what_changed():
    js = text("app.js")
    hero = js[js.index("function renderHero(") :]
    hero = hero[: hero.index("\n}\n")]
    assert "heroStatus(" in hero and "flash(" in hero and "state.hero" in hero
    for caller in ("function renderHealth(", "function onScore(", "function alertsChanged("):
        body = js[js.index(caller) :]
        assert "renderHero()" in body[: body.index("\n}\n")], caller
    rail = js[js.index("function renderRail(") :]
    rail = rail[: rail.index("\n}\n")]
    assert ".slice(0, 5)" in rail and "el(" in rail


def test_ws_close_turns_live_dot_off():
    js = text("app.js")
    assert "function setLive(" in js
    close = js.split('addEventListener("close"')[1][:400]
    assert "setLive(false)" in close
    body = js.split("function setLive(")[1].split("\n}\n")[0]
    assert 'classList.toggle("off", !on)' in body
    css = text("styles.css")
    off = css.index(".dot.off")
    assert "var(--dim)" in css[off : css.index("}", off)]
    assert off > css.index(".dot.ok") and off > css.index(".dot.fail")  # source order: off wins


def test_alerts_kpi_counts_the_selected_symbol_only():
    js = text("app.js")
    kpi = js.split("function renderAlertsKpi(")[1].split("\n}\n")[0]
    assert "a.symbol === state.symbol" in kpi and "DAY_US" in kpi
    assert '"–"' in kpi
    assert 'id="alerts-count">–<' in text("index.html")
    change = js.split('$("symbol-select").addEventListener("change"')[1][:400]
    assert "renderBudget()" in change  # renderBudget recounts through renderAlertsKpi


def test_resting_styles_are_visible_when_animation_does_not_run():
    css = text("styles.css").replace("\r\n", "\n")
    for sel in (r"\.rise", r"\.rail-row\.enter", r"\.forest \.ci-line", r"\.pop"):
        rules = re.findall(r"(?m)^" + sel + r"\s*\{([^}]*)\}", css)
        assert rules, sel
        for body in rules:
            assert not re.search(r"opacity\s*:\s*0\b", body), sel
            assert "scaleX(0)" not in body, sel
            assert not re.search(r"animation[^;]*\bboth\b", body), sel


def test_low_severity_is_grey_and_forest_points_avoid_the_dot_rule():
    css = text("styles.css")
    assert '.alert-card[data-sev="low"] { border-left-color: var(--dim)' in css
    assert ".rail-row.sev-1 { border-left-color: var(--dim)" in css
    js = text("app.js")
    assert "class: `pt ${tone}`" in js and "class: `dot ${tone}`" not in js


def test_red_team_miss_is_grey_and_the_banner_class_changes_only_with_state():
    css, js = text("styles.css"), text("app.js")
    hit = re.search(r'\.sim-run\[data-state="detected"\] \.step-result\s*\{([^}]*)\}', css)
    assert hit and "var(--green)" in hit.group(1)
    miss = re.search(r'\.sim-run\[data-state="missed"\] \.step-result\s*\{([^}]*)\}', css)
    assert miss and "var(--dim)" in miss.group(1) and "--red" not in miss.group(1)
    # the old shine sweep is gone, and nothing on the page loops
    assert "sweep" not in css and "sweep" not in js and "rt-banner" not in js
    sim = css[css.index("/* ---- simulate") : css.index("/* ---- /simulate")]
    assert "infinite" not in sim and "linear-gradient" not in sim
    tick = js.split("function tickInjections()")[1].split("\n}\n")[0]
    before, after = tick.split("run.shown !== status")
    assert "data-state" in after.split("\n    }\n")[0]
    assert "data-state" not in before


# ---- alerts list and detail ----


def test_alert_cards_carry_severity_and_detail_has_two_columns():
    js = text("app.js")
    cards = js.split("function alertCard(a)")[1].split("\n}\n")[0]
    assert "data-sev" in cards
    assert "detail-main" in js and "detail-side" in js
    css = text("styles.css")
    assert "[data-sev=" in css and ".detail-side" in css


SIM_STATUS_BODY = r"""
const t = m.simStatusText;
assert.strictEqual(t("pending", {}), "เริ่มแล้ว");
assert.strictEqual(t("waiting", {}), "กำลังรอระบบ");
const hit = { detected: { alert_id: "a", eventS: 41.6, wallS: 2.4 } };
assert.strictEqual(t("detected", hit), "จับได้ใน 42 วินาที (เวลาจริง 2 วินาที)");
const early = { detected: { alert_id: "a", eventS: -3, wallS: 0.2 } };
assert.strictEqual(t("detected", early), "จับได้ใน 0 วินาที (เวลาจริง 0 วินาที)");
assert.strictEqual(t("missed", {}), "ไม่ทันจับ");
assert.strictEqual(t("restarted", {}), "พลาด (replay เริ่มใหม่)");
assert.strictEqual(t("missed", { restarted: true }), "พลาด (replay เริ่มใหม่)");

const req = m.simulateRequest("pump_burst", null, "STEEMBTC");
assert.deepStrictEqual(req,
  { family: "pump_burst", region: "test", offset_s: 60, symbol: "STEEMBTC" });
assert.deepStrictEqual(m.simulateRequest("pump_burst", { offset_s: 120 }, "STEEMBTC"),
  { family: "pump_burst", region: "test", offset_s: 120, symbol: "STEEMBTC" });
assert.deepStrictEqual(
  m.simulateRequest("wash_pingpong", { region: "train", offset_s: 30, symbol: "ETHBTC" }, "X"),
  { family: "wash_pingpong", region: "train", offset_s: 30, symbol: "ETHBTC" });
// an empty field is no override
assert.deepStrictEqual(m.simulateRequest("pump_stealth", { region: "", symbol: "" }, "STEEMBTC"),
  { family: "pump_stealth", region: "test", offset_s: 60, symbol: "STEEMBTC" });
assert.deepStrictEqual(m.SIM_REASON, { live: "ทดลองได้เฉพาะโหมดย้อนดู",
  viewer: "ต้องเป็นผู้วิเคราะห์ขึ้นไป", ended: "การย้อนดูจบแล้ว รอรอบใหม่" });
assert.deepStrictEqual(m.SIM_CARDS, [
  { family: "pump_burst", title: "ปั่นเร็ว", line: "ราคาและปริมาณพุ่งขึ้นในไม่กี่นาที", icon: "steep" },
  { family: "pump_stealth", title: "ปั่นแบบเนียน", line: "ค่อยๆ ดันราคาขึ้นช้าๆ", icon: "ramp" },
  { family: "wash_pingpong", title: "ซื้อขายวนหลอก", line: "ซื้อขายกันเองไปมาให้ดูคึกคัก", icon: "loop" },
]);
"""


def test_simulate_request_and_status_text(tmp_path):
    _run_motion(tmp_path, "false", SIM_STATUS_BODY)


def test_simulate_markup():
    html, js, css = text("index.html"), text("app.js"), text("styles.css")
    page = _simulate_page(html)
    cards = re.findall(r'<button\b[^>]*class="card sim-card"[^>]*>.*?</button>', page, re.S)
    assert len(cards) == 3
    want = [
        ("pump_burst", "ปั่นเร็ว", "ราคาและปริมาณพุ่งขึ้นในไม่กี่นาที"),
        ("pump_stealth", "ปั่นแบบเนียน", "ค่อยๆ ดันราคาขึ้นช้าๆ"),
        ("wash_pingpong", "ซื้อขายวนหลอก", "ซื้อขายกันเองไปมาให้ดูคึกคัก"),
    ]
    for card, (family, title, line) in zip(cards, want, strict=True):
        assert f'id="sim-{family}"' in card and f'data-family="{family}"' in card
        assert 'type="button"' in card
        icon = re.search(r"<svg\b[^>]*>", card)
        assert icon and 'aria-hidden="true"' in icon.group(0)
        assert 'stroke-width="1.5"' in icon.group(0)
        assert 16 <= int(re.search(r'width="(\d+)"', icon.group(0)).group(1)) <= 24
        assert title in card and line in card
        assert f'family: "{family}", title: "{title}", line: "{line}"' in js
    # the advanced fields are folded away
    adv = re.search(r"<details\b[^>]*>.*?</details>", page, re.S)
    assert adv, "no advanced <details>"
    adv = adv.group(0)
    assert "open" not in re.search(r"<details\b[^>]*>", adv).group(0)
    assert re.search(r"<summary[^>]*>([^<]*)</summary>", adv).group(1) == "ตั้งค่าขั้นสูง"
    for i in ("rt-region", "rt-offset", "rt-symbol"):
        assert f'id="{i}"' in adv, i
    assert '<option value="test">ช่วงค่าทดสอบ</option>' in adv
    assert '<option value="train">ช่วงค่าฝึก</option>' in adv
    intro = re.search(r'<p class="sim-intro[^"]*">([^<]*)</p>', page).group(1)
    assert intro == (
        "ใส่เหตุการณ์ปั่นจำลองลงในข้อมูลย้อนหลัง แล้วดูว่าระบบจับได้เร็วแค่ไหน (ไม่แตะข้อมูลจริง)"
    )
    # plain view: no English labels or codes on the page
    plain = re.sub(r"<!--.*?-->|<svg\b.*?</svg>|<[^>]+>", " ", page, flags=re.S)
    assert not re.search(r"[A-Za-z]", plain), plain
    # the cards are glass cards, never the accent gradient
    sim = css[css.index("/* ---- simulate") : css.index("/* ---- /simulate")]
    assert ".sim-card" in sim and "--violet), var(--blue)" not in sim
    # the server's "ended" refusal is the text the page looks for
    server = (WEB_DIR.parent / "api" / "app.py").read_text(encoding="utf-8")
    ended = re.search(r'REPLAY_ENDED_TH = "([^"]+)"', server).group(1)
    assert f'REPLAY_ENDED_TH = "{ended}"' in js


VERDICT_BODY = r"""
const cell = (mean, ci, n) => ({
  mean: { recall_any: mean, fa_per_symbol_day: 1.65, fa_per_symbol_day_hard: 11.2 },
  ci: { recall_any: ci }, n_events: n });
const report = (full, rows) => ({
  runs: 5, experiments: { real_t001: { full: { "2": full } } },
  paired_bootstrap: { budget: 2, metric: "recall_any", rows } });
const T = (id) => m.EVAL_TARGETS.find((t) => t.id === id);
const miss = m.targetVerdict(report(cell(0.739, [0.68, 0.801], 200), []), T("real_recall"));
assert.strictEqual(miss.state, "miss");
assert.strictEqual(miss.value, 0.739);
assert.strictEqual(miss.n, 200);
assert.deepStrictEqual(miss.ci, [0.68, 0.801]);
assert.strictEqual(
  m.targetVerdict(report(cell(0.85, null, 200), []), T("real_recall")).state, "pass");
assert.strictEqual(m.targetVerdict(report(cell(0.85, null, 200), []), T("real_fa")).state, "pass");
assert.strictEqual(
  m.targetVerdict(report(cell(0.85, null, 200), []), T("real_fa_hard")).state, "miss");
// variant absent / experiment absent / budget absent: never pass
assert.strictEqual(
  m.targetVerdict({ runs: 5, experiments: { real_t001: {} } }, T("real_recall")).state,
  "unmeasured");
assert.strictEqual(
  m.targetVerdict({ runs: 5, experiments: {} }, T("semisyn_pump_burst")).state, "unmeasured");
const noBudget = { runs: 5, experiments: { real_t001: { full: { "5": cell(0.9, null, 9) } } } };
assert.strictEqual(m.targetVerdict(noBudget, T("real_recall")).state, "unmeasured");
assert.strictEqual(
  m.targetVerdict(report({ mean: {}, n_events: 3 }, []), T("real_recall")).state, "unmeasured");
// paired: the CI lower bound must clear 0; a baseline wholly better is a miss
const rows = [
  { baseline: "baseline.random", diff_mean: 0.639, ci: [0.57, 0.7], n_events: 200 },
  { baseline: "baseline.iforest", diff_mean: 0.2, ci: [0.1, 0.3], n_events: 200 },
  { baseline: "baseline.kk", diff_mean: -0.17, ci: [-0.222, -0.121], n_events: 200 },
  { baseline: "stat.robust_z", diff_mean: 0.005, ci: [-0.03, 0.04], n_events: 200 },
];
const r = report(cell(0.74, null, 200), rows);
assert.strictEqual(m.targetVerdict(r, T("ae_vs_baseline.iforest")).state, "pass");
// random / pca are reference cards: the diff and CI show, but never a pass
const ref = m.targetVerdict(r, T("ae_vs_baseline.random"));
assert.strictEqual(ref.state, "reference");
assert.strictEqual(ref.value, 0.639);
assert.deepStrictEqual(ref.ci, [0.57, 0.7]);
assert.strictEqual(m.targetVerdict(r, T("ae_vs_baseline.pca")).state, "unmeasured");
assert.strictEqual(
  m.targetVerdict({ runs: 5, experiments: {} }, T("ae_vs_baseline.random")).state, "unmeasured");
const ids = m.EVAL_TARGETS.map((t) => t.id);
assert.ok(!ids.includes("ae_vs_baseline.random") || T("ae_vs_baseline.random").reference === true);
assert.ok(T("ae_vs_baseline.pca").reference === true);
for (const b of ["baseline.iforest", "stat.robust_z", "baseline.kk"]) {
  assert.ok(!T(`ae_vs_${b}`).reference, b);
}
// latency has no metric in report.json: always unmeasured, never a pass
assert.strictEqual(m.targetVerdict(r, T("latency_p99")).state, "unmeasured");
assert.strictEqual(
  m.targetVerdict(report(cell(0.9, null, 9), []), T("latency_p99")).state, "unmeasured");
// recall_correct rides along on the recall cards
const withCorrect = {
  mean: { recall_any: 1, recall_correct: 0.001 },
  ci: { recall_any: [1, 1], recall_correct: [0, 0.003] }, n_events: 401 };
const wash = m.targetVerdict(
  { runs: 5, experiments: { semisyn: { wash_pingpong: { "2": withCorrect } } } },
  T("semisyn_wash_pingpong"));
assert.strictEqual(wash.state, "pass");
assert.strictEqual(wash.correct.value, 0.001);
assert.strictEqual(m.targetVerdict(r, T("ae_vs_baseline.kk")).state, "miss");
assert.strictEqual(m.targetVerdict(r, T("ae_vs_stat.robust_z")).state, "miss");
assert.strictEqual(
  m.targetVerdict({ runs: 5, experiments: {} }, T("ae_vs_baseline.kk")).state, "unmeasured");
assert.strictEqual(m.targetVerdict(null, T("real_recall")).state, "unmeasured");
assert.strictEqual(m.evalEmpty({ runs: 0 }), true);
assert.strictEqual(m.evalEmpty(null), true);
assert.strictEqual(m.evalEmpty({ runs: 3, experiments: {} }), false);
// the chosen budget: the cell at that budget, and the false-alarm cap follows it
const at = (recall, fa) => ({
  mean: { recall_any: recall, fa_per_symbol_day: fa, fa_per_symbol_day_hard: 40 },
  ci: {}, n_events: 200, ci_unit: "event" });
const multi = { runs: 5,
  experiments: { real_t001: { full: {
    "2": cell(0.739, null, 200), "10": at(0.808, 6.1), "0.5": at(0.6, 0.9) } } },
  paired_bootstrap: { budget: 2, metric: "recall_any", rows } };
assert.strictEqual(m.targetVerdict(multi, T("real_recall"), 10).state, "pass");
assert.strictEqual(m.targetVerdict(multi, T("real_recall"), 10).value, 0.808);
assert.strictEqual(m.targetVerdict(multi, T("real_recall"), 10).unit, "event");
// no budget: the target's own
assert.strictEqual(m.targetVerdict(multi, T("real_recall")).value, 0.739);
assert.strictEqual(m.targetVerdict(multi, T("real_fa"), 10).state, "pass"); // 6.1 <= 10
assert.strictEqual(m.targetVerdict(multi, T("real_fa"), 0.5).state, "miss"); // 0.9 > 0.5
assert.strictEqual(m.targetVerdict(multi, T("real_recall"), 0.1).state, "unmeasured");
assert.strictEqual(m.targetVerdict(multi, T("ae_vs_baseline.iforest"), 2).state, "pass");
assert.strictEqual(m.targetVerdict(multi, T("ae_vs_baseline.iforest"), 10).state, "unmeasured");
"""

RESULT_SENTENCE_BODY = r"""
const T = (id) => m.EVAL_TARGETS.find((t) => t.id === id);
const s = m.resultSentence;
assert.strictEqual(s(T("real_recall"), 0.739, { k: 148, n: 200, budget: 2 }),
  "จับเหตุการณ์ที่มีบันทึกได้เฉลี่ย 148 จาก 200 ครั้ง (เมื่อเตือนไม่เกิน 2 ครั้งต่อเหรียญต่อวัน)");
assert.strictEqual(s(T("semisyn_pump_burst"), 0.2955, { budget: 0.5 }),
  "จับเหตุการณ์จำลองได้ 30% (เมื่อเตือนไม่เกิน 0.5 ครั้งต่อเหรียญต่อวัน)");
assert.strictEqual(s(T("real_recall"), 0.739, {}),
  "จับเหตุการณ์ที่มีบันทึกได้ 74% (เมื่อเตือนไม่เกิน 2 ครั้งต่อเหรียญต่อวัน)");
assert.strictEqual(s(T("real_fa"), 1.6498753, {}), "เตือนผิดเฉลี่ย 1.65 ครั้งต่อเหรียญต่อวัน");
assert.strictEqual(s(T("real_fa_hard"), 11.169999, {}), "เตือนผิดในวันที่ไม่มีเหตุการณ์ 11.17 ครั้ง");
assert.strictEqual(s(T("real_fa_hard"), 0, {}), "เตือนผิดในวันที่ไม่มีเหตุการณ์ 0 ครั้ง");
const pair = T("ae_vs_baseline.iforest");
assert.strictEqual(s(pair, 0.2, { ci: [0.1, 0.3] }), "เทียบกับวิธีพื้นฐาน: ดีกว่า");
assert.strictEqual(s(pair, 0.05, { ci: [-0.1, 0.2] }), "เทียบกับวิธีพื้นฐาน: ไม่ต่างอย่างมีนัยสำคัญ");
assert.strictEqual(s(pair, -0.17, { ci: [-0.222, -0.121] }), "เทียบกับวิธีพื้นฐาน: แย่กว่า");
assert.strictEqual(s(T("ae_vs_baseline.pca"), 0.075, { ci: [0.035, 0.122] }),
  "เทียบกับวิธีพื้นฐาน: ดีกว่า");
// gated pairs follow the report's own significant flag and reason
assert.strictEqual(s(pair, 0.01, { ci: [0.005, 0.02], significant: false }),
  "เทียบกับวิธีพื้นฐาน: ไม่ต่างอย่างมีนัยสำคัญ"); // CI above 0 but the gate (effect size) says no
assert.strictEqual(s(pair, 0.2, { ci: [0.1, 0.3], significant: true }), "เทียบกับวิธีพื้นฐาน: ดีกว่า");
assert.strictEqual(s(pair, -0.17, { ci: [-0.222, -0.121], significant: false }),
  "เทียบกับวิธีพื้นฐาน: แย่กว่า");
const few = "ยังตัดสินไม่ได้ มีเหตุการณ์เพียง 3 ครั้ง";
assert.strictEqual(s(pair, 0.3, { ci: [0.1, 0.5], significant: false, reason: few }),
  "เทียบกับวิธีพื้นฐาน: " + few);
const gate = { runs: 5, experiments: {}, paired_bootstrap: { budget: 2, rows: [{
  baseline: "baseline.iforest", diff_mean: 0.01, ci: [0.005, 0.02], n_events: 200,
  significant: false, reason_th: null }] } };
assert.strictEqual(m.targetVerdict(gate, pair).significant, false);
assert.strictEqual(m.targetVerdict(gate, pair).state, "miss");
assert.strictEqual(s(T("latency_p99"), null, {}), "ความเร็วในการเตือน: ยังไม่วัด");
assert.deepStrictEqual(m.RESULT_BADGE, { pass: "ผ่านเป้า", fail: "ยังไม่ถึงเป้า", none: "ยังไม่วัด" });
// nothing measured is never a number in the sentence
for (const t of m.EVAL_TARGETS) {
  const said = s(t, null, { budget: 2 });
  assert.ok(!/\d+(\.\d+)?%|\d+ จาก/.test(said.replace(/ไม่เกิน [\d.]+ ครั้ง/, "")), t.id + " " + said);
  assert.ok(!/[A-Za-z]/.test(said), said);
  assert.ok(!/[A-Za-z]/.test(t.title || ""), t.id);
}
"""


def test_result_sentences(tmp_path):
    _run_motion(tmp_path, "false", RESULT_SENTENCE_BODY)


def test_target_verdicts(tmp_path):
    _run_motion(tmp_path, "false", VERDICT_BODY)


def test_target_verdicts_against_the_real_report(tmp_path):
    rpt = Path(__file__).resolve().parents[2] / "data" / "report.json"
    if not rpt.is_file():
        pytest.skip("data/report.json is not present")
    body = (
        f"const rep = JSON.parse(require('fs').readFileSync({str(rpt)!r}, 'utf8'));\n"
        + r"""
const got = {};
for (const t of m.EVAL_TARGETS) {
  const v = m.targetVerdict(rep, t);
  got[t.id] = [v.state, v.value === null ? null : Math.round(v.value * 1000) / 1000];
}
const want = {
  real_recall: ["miss", 0.739], real_fa: ["pass", 1.65], real_fa_hard: ["miss", 11.17],
  semisyn_pump_burst: ["miss", 0.295], semisyn_pump_stealth: ["miss", 0.351],
  semisyn_wash_pingpong: ["pass", 1],
  "ae_vs_baseline.iforest": ["miss"], "ae_vs_stat.robust_z": ["miss"],
  "ae_vs_baseline.kk": ["miss"],
  "ae_vs_baseline.random": ["reference"], "ae_vs_baseline.pca": ["reference"],
  latency_p99: ["unmeasured", null],
};
for (const [id, [state, value]] of Object.entries(want)) {
  assert.strictEqual(got[id][0], state, id + " " + JSON.stringify(got[id]));
  if (value !== undefined) assert.strictEqual(got[id][1], value, id);
}
"""
    )
    _run_motion(tmp_path, "false", body)


def test_ablation_notes_cell_wraps():
    css = text("styles.css")
    assert re.search(r"td\.notes[^}]*max-width", css)
    js = text("app.js")
    assert 'class: "small muted notes"' in js and "หมายเหตุวิธีวัด" in js
    assert 'el("details"' in js and "uv run idsai report" in js


def test_long_explanations_live_on_about():
    html = text("index.html")
    mon = re.search(r'<section id="page-home".*?</section>', html, re.S).group(0)
    for m in re.finditer(r'<p\b([^>]*)>(.*?)</p>', mon, re.S):
        if 'id="chart-info-body"' in m.group(1):
            continue
        assert len(" ".join(m.group(2).split())) <= 120, m.group(2)[:60]
    about = re.search(r'<aside id="help-drawer".*?</aside>', html, re.S).group(0)
    assert "τ saturated" in about or "τ-saturated" in about
    assert "งบการแจ้งเตือนสูงกว่าอัตราสูงสุดที่ข้อมูลสอบเทียบทำได้" in about
    assert "autocorrelation" in about
    assert "อัตราการแจ้งเตือนจริงอาจสูงกว่างบที่ตั้งไว้" in about


# ---- sign-in, roles, prefs, activity ----


def test_login_and_activity_routes_present():
    html, js = text("index.html"), text("app.js")
    for route in ("#/login", "#/activity"):
        assert route in html + js, route
    ids = set(re.findall(r'\bid="([\w-]+)"', html))
    for i in (
        "page-login", "login-form", "login-user", "login-pass", "login-remember", "login-error",
        "user-chip", "user-name", "user-role", "btn-logout",
        "page-activity", "audit-body", "audit-user", "audit-action", "audit-more", "audit-empty",
    ):
        assert i in ids, i
    assert 'data-route="activity"' in html
    for page in ("page-login", "page-activity"):
        assert f'"{page}"' in js, page
    assert 'type="password"' in html and 'autocomplete="current-password"' in html
    assert "จำฉันไว้" in html and "ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง" not in html  # the server words the error


def test_login_page_has_no_signup():
    low = (text("index.html") + text("app.js")).lower()
    for word in ("สมัคร", "sign up", "signup", "sign-up", "register", "ลืมรหัส"):
        assert word not in low, word


def test_activity_renders_audit_fields_as_text_only():
    js = text("app.js")
    body = js[js.index("function renderAudit(") : js.index("async function loadAudit(")]
    assert '"href"' not in body and "href:" not in body and "innerHTML" not in body
    assert 'el("td"' in body
    assert 'data-route="activity"' in text("index.html")


def test_mutating_buttons_check_can():
    js = text("app.js")
    control = js[js.index("async function control(") : js.index("function bindReplay(")]
    assert (
        'can("analyst")' in control
        and control.index('can("analyst")') < control.index("postJson(")
    )
    fb = js[js.index("function feedbackCard(") : js.index("async function drawDetailChart(")]
    assert 'can("analyst")' in fb and fb.index('can("analyst")') < fb.index("postJson(")
    rt = js[js.index("async function startSimulation(") : js.index("function addInjection(")]
    assert 'can("analyst")' in rt and rt.index('can("analyst")') < rt.index("postJson(")
    gates = js[js.index("function applyRoleGates(") :]
    assert "ต้องเป็น analyst" in js and "title" in gates[:1500]


def test_401_goes_to_login_with_next():
    js = text("app.js")
    body = js[js.index("async function api(") : js.index("function postJson(")]
    assert "401" in body and "sessionLost(" in body
    assert "/api/auth/login" in body  # a failed sign-in must not bounce back to itself


AUTH_BODY = r"""
const s = m.safeNext;
assert.strictEqual(s("#/eval"), "#/eval");
assert.strictEqual(s("#/alerts/abc%20d"), "#/alerts/abc%20d");
for (const bad of ["https://x", "//x", "", null, undefined, "eval", "javascript:alert(1)",
  "#/login?next=%23%2Fx", "#/\nx", 42]) {
  assert.strictEqual(s(bad), "#/home", String(bad));
}
"""

CAN_BODY = r"""
m.setMe(null);
assert.strictEqual(m.can("viewer"), false);
for (const [role, want] of [
  ["viewer", [true, false, false]],
  ["analyst", [true, true, false]],
  ["admin", [true, true, true]],
]) {
  m.setMe({ role });
  const got = ["viewer", "analyst", "admin"].map((n) => m.can(n));
  assert.deepStrictEqual(got, want, role);
}
m.setMe({ role: "admin" });
assert.strictEqual(m.can("root"), false);
m.setMe({ role: "root" });
assert.strictEqual(m.can("viewer"), false);
"""

PREFS_HEAD = r"""
const assert = require("assert");
globalThis.matchMedia = () => ({ matches: false });
const m = require(process.argv[2]);
"""
PREFS_BODY = r"""
const store = {};
globalThis.window = { localStorage: {
  getItem: (k) => (k in store ? store[k] : null),
  setItem: (k, v) => { store[k] = String(v); } } };
const calls = [];
globalThis.fetch = async (path, opts) => {
  calls.push([path, opts]); return { ok: true, status: 200, json: async () => ({}) }; };
(async () => {
  m.setMe({ username: "local", role: "admin", auth_enabled: false, prefs: {} });
  m.savePrefs({ symbol: "ETHUSDT" });
  assert.strictEqual(store["idsai.symbol"], "ETHUSDT");
  await new Promise((r) => setTimeout(r, 700));
  assert.strictEqual(calls.length, 0, "auth off never calls the server");
  m.setMe({ username: "ann", role: "viewer", auth_enabled: true, prefs: {} });
  m.savePrefs({ theme: "light" });
  m.savePrefs({ eval_budget: 2 });
  assert.strictEqual(calls.length, 0, "debounced");
  await new Promise((r) => setTimeout(r, 700));
  assert.strictEqual(calls.length, 1);
  assert.strictEqual(calls[0][0], "/api/me/prefs");
  assert.strictEqual(calls[0][1].method, "PUT");
  assert.deepStrictEqual(JSON.parse(calls[0][1].body), { theme: "light", eval_budget: 2 });
  console.log("ok");
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_next_param_only_accepts_hash_routes(tmp_path):
    _run_motion(tmp_path, "false", AUTH_BODY)


def test_can_matches_role_order(tmp_path):
    _run_motion(tmp_path, "false", CAN_BODY)


def test_prefs_save_is_debounced_and_local_when_auth_is_off(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = tmp_path / "prefs.js"
    script.write_text(
        PREFS_HEAD + PREFS_BODY,
        encoding="utf-8",
    )
    r = subprocess.run(
        [node, str(script), str(WEB_DIR / "app.js")], capture_output=True, text=True, cwd=tmp_path
    )
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


def test_auth_off_keeps_todays_flow():
    js = text("app.js")
    assert "auth_enabled" in js
    html = text("index.html")
    chip = re.search(r'<details id="user-chip"[^>]*>', html)
    assert chip and "hidden" in chip.group(0)  # only shown once /api/me says auth is on
    login = re.search(r'<section id="page-login"[^>]*>', html)
    assert login and "hidden" in login.group(0)



SESSION_LOST_HEAD = r"""
const assert = require("assert");
globalThis.matchMedia = () => ({ matches: false });
const m = require(process.argv[2]);
globalThis.document = { getElementById: () => null, body: { classList: { toggle() {} } } };
globalThis.location = { hash: "#/eval" };
const reply = (status) => async () => ({
  ok: status < 400, status, json: async () => ({ detail: "กรุณาเข้าสู่ระบบ" }) });
"""

SESSION_LOST_BODY = r"""
(async () => {
  // a 401 mid-session forgets the user and lands on the sign-in route with next kept
  m.setMe({ username: "ann", role: "viewer", auth_enabled: true, prefs: {} });
  globalThis.fetch = reply(401);
  await assert.rejects(() => m.api("/api/experiments"), (e) => e.status === 401);
  assert.strictEqual(m.getMe(), null, "stale user must be cleared or route() bounces back");
  assert.strictEqual(location.hash, "#/login?next=%23%2Feval");
  // already on the sign-in route: a further 401 does not rewrite it (no next chain)
  await assert.rejects(() => m.api("/api/alerts"));
  assert.strictEqual(location.hash, "#/login?next=%23%2Feval");
  // a failed sign-in is not a lost session
  m.setMe({ role: "viewer", auth_enabled: true });
  location.hash = "#/login";
  await assert.rejects(() => m.api("/api/auth/login", { method: "POST" }));
  assert.notStrictEqual(m.getMe(), null);
  // WS close 4401: no reconnect, user cleared, sign-in route;
  // other codes keep the socket for the retry loop
  location.hash = "#/alerts";
  const ws = {};
  m.setWs(ws);
  m.onWsClose(ws, 4401);
  assert.strictEqual(m.getWs(), null);
  assert.strictEqual(m.getMe(), null);
  assert.strictEqual(location.hash, "#/login?next=%23%2Falerts");
  console.log("ok");
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_lost_session_routes_to_login_and_ws_4401_stops_reconnecting(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = tmp_path / "lost.js"
    script.write_text(SESSION_LOST_HEAD + SESSION_LOST_BODY, encoding="utf-8")
    r = subprocess.run(
        [node, str(script), str(WEB_DIR / "app.js")], capture_output=True, text=True, cwd=tmp_path
    )
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr
    js = text("app.js")
    close = js[js.index("function onWsClose(") : js.index("function scheduleReconnect(")]
    assert "4401" in close and close.index("4401") < close.index("scheduleReconnect()")
    enter = js[js.index("async function enterApp(") : js.index("async function boot(")]
    assert "if (!state.ws) connect();" in enter  # signing in again reopens the socket


def test_login_form_posts_so_a_credential_never_reaches_the_url():
    form = re.search(r'<form id="login-form"[^>]*>', text("index.html")).group(0)
    assert 'method="post"' in form


def test_feedback_form_uses_the_session_user_when_auth_is_on():
    js = text("app.js")
    body = js[js.index("function feedbackCard"):]
    body = body[: body.index("\n}\n")]
    assert "state.me.username" in body  # the analyst is the signed-in user, not a typed name
    assert "authOn()" in body


RESTART_BODY = r"""
const S = 1e6;
const pending = { symbol: "X", t_start_us: 1030 * S, t_end_us: 1150 * S, known: new Map(),
  detected: null };
const hit = { symbol: "X", t_start_us: 1030 * S, t_end_us: 1150 * S, known: new Map(),
  detected: { alert_id: "a1", eventS: 5, wallS: 1 } };
assert.strictEqual(m.injectionStatus(pending, 1010 * S), "pending");
m.restartRuns([pending, hit]);
assert.strictEqual(pending.restarted, true);
assert.ok(!hit.restarted);
assert.strictEqual(m.injectionStatus(pending, 1010 * S), "missed");
assert.strictEqual(m.injectionStatus(pending, null), "missed");
assert.strictEqual(m.simStatusText("missed", pending), "พลาด (replay เริ่มใหม่)");
assert.ok(m.simStatusText("missed", pending).includes("เริ่มใหม่"));
assert.strictEqual(m.injectionStatus(hit, 1010 * S), "detected");
// the restart's seek deletes the round's alerts: a closed run keeps its verdict
assert.strictEqual(m.resetRuns([pending, hit], ["a1"]), false);
assert.strictEqual(m.injectionStatus(hit, 1010 * S), "detected");
"""


def test_replay_restarted_marks_runs_missed(tmp_path):
    _run_motion(tmp_path, "false", RESTART_BODY)


def test_on_alert_handles_replay_restarted():
    js = text("app.js")
    body = js.split("function onAlert(data)")[1].split("\n}\n")[0]
    assert '"replay_restarted"' in body and "restartRuns(" in body
    check = js.split("function checkInjections()")[1].split("\n}\n")[0]
    assert "closed" in check


FRAMED_BODY = r"""
const url = "https://x.hf.space/#/login?next=%23%2Falerts";
const box = () => ({ hidden: true });
const link = () => ({ href: "#" });
// opened directly: no hint, the link is left alone
const top = { location: { href: url } };
top.top = top;
let b = box(), a = link();
assert.strictEqual(m.applyFramedHint(top, b, a), false);
assert.strictEqual(b.hidden, true);
assert.strictEqual(a.href, "#");
// inside another site's frame: the hint shows and links to this very page
const framed = { top: {}, location: { href: url } };
b = box(); a = link();
assert.strictEqual(m.applyFramedHint(framed, b, a), true);
assert.strictEqual(b.hidden, false);
assert.strictEqual(a.href, url);
// a top that throws on access still counts as a frame
const odd = { location: { href: url } };
Object.defineProperty(odd, "top", { get() { throw new Error("blocked"); } });
assert.strictEqual(m.isFramed(odd), true);
"""


def test_framed_login_shows_a_direct_link(tmp_path):
    _run_motion(tmp_path, "false", FRAMED_BODY)


def test_framed_login_hint_markup():
    html, js = text("index.html"), text("app.js")
    m = re.search(r'<p id="login-framed"[^>]*hidden>(.*?)</p>', html, re.S)
    assert m, "no hidden framed hint on the login page"
    assert 'id="login-framed-link"' in m.group(1) and 'target="_top"' in m.group(1)
    assert "โดยตรง" in m.group(1)
    route_fn = js[js.index("function route()") : js.index("function isFramed(")]
    assert 'applyFramedHint(window, $("login-framed"), $("login-framed-link"))' in route_fn


# ---- sign-in page: split layout, a loader until the session is known ----

BOOT_VIEW_BODY = r"""
const v = m.bootView;
const ann = { username: "ann", role: "viewer", auth_enabled: true };
assert.strictEqual(v(null, null), "loading");
assert.strictEqual(v(undefined, undefined), "loading");
assert.strictEqual(v(ann, null), "loading");
assert.strictEqual(v(null, true), "login");
assert.strictEqual(v(ann, true), "app");
assert.strictEqual(v(null, false), "app");
assert.strictEqual(v({ username: "local", role: "admin", auth_enabled: false }, false), "app");
"""


def test_boot_view(tmp_path):
    _run_motion(tmp_path, "false", BOOT_VIEW_BODY)


def test_boot_shows_only_loader_until_me_resolves():
    html = text("index.html")
    pages = re.findall(r'<section\b[^>]*\bclass="[^"]*\bpage\b[^"]*"[^>]*>', html)
    assert len(pages) >= 7, pages
    for tag in pages:
        assert re.search(r"\shidden\b", tag), tag
    for tag in ("header", "footer"):  # nothing but the background and the loader at first
        found = re.search(rf"<{tag}\b[^>]*>", html).group(0)
        assert re.search(r"\shidden\b", found), found
    loader = re.search(r'<[a-z]+\b[^>]*\bid="boot-loader"[^>]*>', html)
    assert loader, "no boot loader"
    assert not re.search(r"\shidden\b", loader.group(0)), loader.group(0)
    css = text("styles.css")
    for body in re.findall(r"(?m)^[^{}\n]*(?:#boot-loader|\.boot-loader)\s*\{([^}]*)\}", css):
        assert "display: none" not in body and "visibility: hidden" not in body
    js = text("app.js")
    boot = js[js.index("async function boot(") :]
    assert "bootView(" in boot[: boot.index("\n}\n")]


def test_login_markup():
    html = text("index.html")
    m = re.search(r'<section id="page-login"[^>]*>(.*?)</section>', html, re.S)
    assert m, "no sign-in page"
    page = m.group(1)
    assert 'class="login-hero"' in page and "login-card" in page
    assert page.index('class="login-hero"') < page.index('id="login-form"')
    hero = page[page.index('class="login-hero"') : page.index('id="login-form"')]
    tagline = re.search(r'<p class="login-tagline">(.*?)</p>', hero, re.S)
    assert tagline and " ".join(re.sub(r"<[^>]+>", "", tagline.group(1)).split()) == (
        "กล้องวงจรปิดของตลาดคริปโต เตือนเมื่อการซื้อขายผิดปกติทางสถิติ"
    )
    line = re.search(r'<svg class="login-line"[^>]*>', hero)
    assert line and 'aria-hidden="true"' in line.group(0)
    assert "<ul" not in hero and "<ol" not in hero and "<img" not in hero
    button = re.search(r'<button id="login-submit"[^>]*>(.*?)</button>', page, re.S)
    assert button and button.group(1).strip() == "เข้าสู่ระบบ"
    assert 'type="submit"' in button.group(0) and "primary" in button.group(0)
    remember = re.search(r'<input id="login-remember"[^>]*>\s*<span>(.*?)</span>', page, re.S)
    assert remember and remember.group(1).strip() == "จำฉันไว้"
    for i in ("login-user", "login-pass", "login-error", "login-framed", "login-framed-link"):
        assert f'id="{i}"' in page, i
    for word in ("ยินดีต้อนรับ", "ขับเคลื่อนด้วย", "อัจฉริยะ", "ล้ำสมัย", "ทรงพลัง"):
        assert word not in html + text("app.js"), word


def test_login_line_animates_but_rests_drawn():
    css = text("styles.css").replace("\r\n", "\n")
    rules = re.findall(r"(?m)^\.login-line[^{\n]*\{([^}]*)\}", css)
    joined = " ".join(rules)
    assert "stroke-dasharray" in joined and "animation" in joined
    for body in rules:  # with the animation off (reduced motion) the line is still drawn
        assert not re.search(r"opacity\s*:\s*0\b", body)
        assert not re.search(r"animation[^;]*\bboth\b", body)
        assert not re.search(r"stroke-dashoffset\s*:\s*[1-9]", body)
    assert re.search(r"@keyframes [\w-]+\s*\{[^@]*stroke-dashoffset", css)
    narrow = re.search(r"@media \(max-width: 720px\) \{(.*?)\n\}", css, re.S).group(1)
    assert re.search(r"\.login-line\s*\{[^}]*display:\s*none", narrow)
    assert re.search(r"#page-login[^{]*\{[^}]*grid-template-columns:\s*minmax\(0, 1fr\)", narrow)


BOOT_RETRY_BODY = r"""
const t = m.bootRetryText;
const NO_ANSWER = "ยังติดต่อเซิร์ฟเวอร์ไม่ได้ กำลังลองใหม่";
assert.strictEqual(t(null), NO_ANSWER);
assert.strictEqual(t({}), NO_ANSWER);
assert.strictEqual(t({ status: 502 }), "เซิร์ฟเวอร์ตอบกลับผิดพลาด (รหัส 502) กำลังลองใหม่");
assert.strictEqual(t({ status: 403 }), "เซิร์ฟเวอร์ตอบกลับผิดพลาด (รหัส 403) กำลังลองใหม่");
assert.strictEqual(t({ answered: true }), "เซิร์ฟเวอร์ตอบกลับผิดพลาด กำลังลองใหม่");
const d = m.bootRetryDelay;
const want = [3000, 6000, 12000, 24000, 30000, 30000, 30000];
assert.deepStrictEqual([0, 1, 2, 3, 4, 5, 50].map(d), want);
assert.strictEqual(d(-1), 3000);
assert.strictEqual(d(undefined), 3000);
assert.strictEqual(d(1e6), 30000);
"""


def test_boot_retry_text_and_backoff(tmp_path):
    _run_motion(tmp_path, "false", BOOT_RETRY_BODY)


LOAD_ME_BODY = r"""
console.warn = () => {};
const answer = (status, body) => async () => ({ ok: status < 400, status, json: async () => body });
(async () => {
  // a 200 without an account is unknown, never "sign-in is off"
  for (const body of [null, "x", [], 7]) {
    globalThis.fetch = answer(200, body);
    assert.strictEqual(await m.loadMe(), null, JSON.stringify(body));
    assert.strictEqual(m.getMe(), null);
    assert.deepStrictEqual(m.getBootError(), { answered: true });
    assert.strictEqual(m.bootView(m.getMe(), null), "loading");
  }
  globalThis.fetch = answer(502, { detail: "bad gateway" });
  assert.strictEqual(await m.loadMe(), null);
  assert.deepStrictEqual(m.getBootError(), { status: 502 });
  globalThis.fetch = async () => { throw new TypeError("Failed to fetch"); };
  assert.strictEqual(await m.loadMe(), null);
  assert.deepStrictEqual(m.getBootError(), {});
  globalThis.fetch = answer(401, { detail: "กรุณาเข้าสู่ระบบ" });
  assert.strictEqual(await m.loadMe(), true);
  assert.strictEqual(m.getBootError(), null);
  console.log("ok");
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_load_me_tells_unknown_from_auth_off(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = tmp_path / "loadme.js"
    script.write_text(SESSION_LOST_HEAD + LOAD_ME_BODY, encoding="utf-8")
    r = subprocess.run(
        [node, str(script), str(WEB_DIR / "app.js")], capture_output=True, text=True, cwd=tmp_path
    )
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr
    js = text("app.js")
    load = js[js.index("async function loadMe(") : js.index("function showView(")]
    assert "console.warn" in load and "e.status" in load
    boot = js[js.index("async function boot(") :]
    boot = boot[: boot.index("\n}\n")]
    assert "bootRetryDelay(" in boot and "bootRetryText(" in boot and "3000)" not in boot


def test_help_route_waits_for_the_app_view():
    js = text("app.js")
    route_fn = js[js.index("function route()") : js.index("function isFramed(")]
    head = route_fn[: route_fn.index("const r = parseRoute();")]
    assert "openHelp()" in head and 'state.view === "app"' in head and "pendingHelp = true" in head
    enter = js[js.index("async function enterApp(") : js.index("async function boot(")]
    assert "state.pendingHelp" in enter and "openHelp()" in enter


# ---- home: hero status, speed chips, first sentence, viewer gates ----

HERO_BODY = r"""
const S = 1e6;
const now = 10000 * S;
const h = (o) => m.heroStatus({ nowUs: now, alerts: [], f: null, tau: null, auditor: "ok",
  symbol: "STEEMBTC", ...o });
assert.strictEqual(m.HERO_WATCH_RATIO, 0.8);
assert.strictEqual(m.HERO_RECENT_US, 30 * 60 * S);
let r = h({});
assert.deepStrictEqual(r, { level: "normal", title: "ตลาดปกติ",
  line: "ไม่มีแจ้งเตือนใน 30 นาทีล่าสุด", alertId: null });
assert.strictEqual(h({ f: 0.79, tau: 1 }).level, "normal");
r = h({ f: 0.8, tau: 1 });
assert.strictEqual(r.level, "watch");
assert.strictEqual(r.title, "กำลังเฝ้าระวัง");
assert.strictEqual(r.line, "ระดับความผิดปกติใกล้เส้นเตือน");
r = h({ auditor: "warn" });
assert.strictEqual(r.level, "watch");
assert.strictEqual(r.line, "ระบบตรวจตัวเองพบจุดที่ควรดู");
assert.strictEqual(r.alertId, null);
// 16:42 UTC on some day, exactly 30 min before now
const at = 3 * 86400 * S + (16 * 3600 + 42 * 60) * S;
const edge = { alert_id: "a-edge", symbol: "STEEMBTC", ts_event_us: at };
r = m.heroStatus({ nowUs: at + 30 * 60 * S, alerts: [edge], f: null, tau: null, auditor: "ok",
  symbol: "STEEMBTC" });
assert.deepStrictEqual(r, { level: "alert", title: "พบความผิดปกติทางสถิติ",
  line: "ระบบพบรูปแบบการซื้อขายที่ต่างจากปกติของ STEEMBTC เมื่อ 23:42 น.", alertId: "a-edge" });
// an alert after the clock (e.g. just before a seek's reset lands) is not "recent"
r = m.heroStatus({ nowUs: at - 1, alerts: [edge], f: null, tau: null, auditor: "ok", symbol: "X" });
assert.strictEqual(r.level, "normal");
// F at or past the warning line says so
assert.strictEqual(h({ f: 1, tau: 1 }).line, "ระดับความผิดปกติข้ามเส้นเตือน");
assert.strictEqual(h({ f: 0.9, tau: 1 }).line, "ระดับความผิดปกติใกล้เส้นเตือน");
r = m.heroStatus({ nowUs: at + 30 * 60 * S + 1, alerts: [edge], f: 0.1, tau: 1, auditor: "ok",
  symbol: "STEEMBTC" });
assert.strictEqual(r.level, "normal");
// both hold: alert wins, and the newest recent alert is the one named
const older = { alert_id: "a-old", symbol: "STEEMBTC", ts_event_us: now - 20 * 60 * S };
const newer = { alert_id: "a-new", symbol: "STEEMBTC", ts_event_us: now - 5 * 60 * S };
r = h({ alerts: new Map([["a-old", older], ["a-new", newer]]).values(), f: 5, tau: 1,
  auditor: "warn" });
assert.strictEqual(r.level, "alert");
assert.strictEqual(r.alertId, "a-new");
// nulls never throw
for (const o of [{ nowUs: null }, { nowUs: undefined, f: undefined, tau: undefined },
  { alerts: null }, { alerts: undefined, auditor: undefined }, { nowUs: null, alerts: [newer] }]) {
  assert.strictEqual(typeof m.heroStatus({ ...o }).level, "string");
  assert.strictEqual(typeof h(o).level, "string");
}
assert.strictEqual(m.heroStatus({ nowUs: null, alerts: [newer], f: 2, tau: 1 }).level, "watch");
assert.deepStrictEqual(m.SPEED_CHIPS, [{ label: "1×", v: 1 }, { label: "10×", v: 10 },
  { label: "60×", v: 60 }, { label: "เร็วสุด", v: 0 }]);
"""

FIRST_SENTENCE_BODY = r"""
const f = m.firstSentence;
assert.strictEqual(f(""), "");
assert.strictEqual(f(null), "");
assert.strictEqual(f(undefined), "");
assert.strictEqual(f("ราคาขึ้นเร็ว. ปริมาณสูง."), "ราคาขึ้นเร็ว");
assert.strictEqual(f("ราคาขึ้นเร็ว"), "ราคาขึ้นเร็ว");
// sentences joined with "; ", and a decimal point is never a sentence end
const ex = "ราคาขึ้นเร็วกว่าปกติ 3.5 เท่า; ปริมาณซื้อขายสูง; ค่า 1.2 ต่ำ";
assert.strictEqual(f(ex), "ราคาขึ้นเร็วกว่าปกติ 3.5 เท่า");
assert.strictEqual(f("  a   b.  c"), "a b");
const long = "ก".repeat(200);
assert.strictEqual(f(long).length, 120);
assert.ok(f(long).endsWith("…"));
assert.strictEqual(f("ข".repeat(120)), "ข".repeat(120));
"""

GATES_HEAD = r"""
const assert = require("assert");
globalThis.matchMedia = () => ({ matches: false });
const m = require(process.argv[2]);
const nodes = {};
const chips = [0, 1, 2, 3].map(() => ({ disabled: false, title: "",
  removeAttribute(k) { this[k] = ""; } }));
const node = (id) => (nodes[id] ||= { id, disabled: false, hidden: false, textContent: "",
  title: "", removeAttribute(k) { this[k] = ""; }, setAttribute(k, v) { this[k] = v; },
  querySelectorAll: () => (id === "replay-bar" ? chips : []) });
globalThis.document = { getElementById: node, querySelectorAll: () => [],
  body: { classList: { toggle() {} } } };
"""

GATES_BODY = r"""
m.setMe({ username: "v", role: "viewer", auth_enabled: true });
m.applyRoleGates();
for (const id of ["btn-play", "seek-input", "btn-seek"]) {
  assert.strictEqual(nodes[id].disabled, true, id);
}
for (const c of chips) assert.strictEqual(c.disabled, true);
assert.strictEqual(nodes["replay-reason"].hidden, false);
assert.strictEqual(nodes["replay-reason"].textContent, "เฉพาะผู้วิเคราะห์ขึ้นไปที่ควบคุมได้");
m.setMe({ username: "a", role: "analyst", auth_enabled: true });
m.applyRoleGates();
for (const id of ["btn-play", "seek-input", "btn-seek"]) {
  assert.strictEqual(nodes[id].disabled, false, id);
}
for (const c of chips) assert.strictEqual(c.disabled, false);
assert.strictEqual(nodes["replay-reason"].hidden, true);
assert.ok(nodes["btn-play"].title.includes("(กด Space)"), nodes["btn-play"].title);
console.log("ok");
"""


def test_hero_status_rules(tmp_path):
    _run_motion(tmp_path, "false", HERO_BODY)


def test_first_sentence(tmp_path):
    _run_motion(tmp_path, "false", FIRST_SENTENCE_BODY)


def test_hero_and_replay_gates_viewer(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = tmp_path / "gates.js"
    script.write_text(GATES_HEAD + GATES_BODY, encoding="utf-8")
    r = subprocess.run(
        [node, str(script), str(WEB_DIR / "app.js")], capture_output=True, text=True, cwd=tmp_path
    )
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


THAI_TIME_BODY = r"""
const S = 1e6;
const t = Date.UTC(2021, 0, 18, 16, 30, 5) * 1000;
assert.strictEqual(m.fmtThaiTime(t), "23:30 น.");
assert.strictEqual(m.fmtThaiTime(t, true), "23:30:05 น.");
// 20:00 UTC is already the next Thai day
assert.strictEqual(m.fmtThaiTime(Date.UTC(2021, 0, 18, 20, 0) * 1000), "03:00 น.");
assert.strictEqual(m.fmtThaiTime(null), "–");
assert.strictEqual(m.fmtThaiTime(undefined, true), "–");
"""


def test_thai_time(tmp_path):
    _run_motion(tmp_path, "false", THAI_TIME_BODY)


RAIL_BODY = r"""
const tax = JSON.parse(require("fs").readFileSync(process.argv[3], "utf-8"));
const ids = [];
for (const [id, t] of Object.entries(tax)) {
  ids.push(id);
  for (const sub of t.sub_techniques || []) ids.push(sub.id);
}
ids.push("MM-T999", "", null);
for (const tax2 of [tax, null]) {
  for (const id of ids) {
    const text = m.railHeadline({ technique_id: id, symbol: "STEEMBTC",
      ts_event_us: Date.UTC(2021, 0, 18, 16, 30) * 1000 }, tax2);
    for (const bad of ["MM-T", "MT-T", "UTC"]) assert.ok(!text.includes(bad), text);
    assert.ok(!/[A-Za-z]/.test(text.replace("STEEMBTC", "")), text);
    assert.ok(text.includes("STEEMBTC") && text.endsWith("23:30 น."), text);
  }
}
const pump = m.railHeadline({ technique_id: "MM-T001", symbol: "X", ts_event_us: 0 }, tax);
assert.ok(pump.includes("รูปแบบคล้าย"), pump);
"""


def test_rail_headline_is_plain_thai(tmp_path, client):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    tax = tmp_path / "tax.json"
    tax.write_text(json.dumps(client.get("/api/taxonomy").json()), encoding="utf-8")
    script = tmp_path / "rail.js"
    script.write_text(MOTION_JS % ("false", RAIL_BODY), encoding="utf-8")
    r = subprocess.run(
        [node, str(script), str(WEB_DIR / "app.js"), str(tax)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr
    js = text("app.js")
    rail = js[js.index("function renderRail(") :]
    rail = rail[: rail.index("\n}\n")]
    assert "railHeadline(" in rail and "explanation_th" not in rail and "technique_id" not in rail


def test_home_markers_carry_no_text():
    js = text("app.js")
    body = js[js.index("function renderMarkers(") :]
    body = body[: body.index("\n}\n")]
    assert "text:" not in body and "technique_id" not in body


# ---- alerts: cards, filters, plain detail, verdict buttons, toasts ----

# a small fake DOM: enough for el(), clear() and the render functions; app.js is required before
# `document` exists, so it exports its helpers instead of booting
DOM_HEAD = r"""
const assert = require("assert");
globalThis.matchMedia = () => ({ matches: %s });
const m = require(process.argv[2]);
class FakeNode {
  constructor(tag) {
    this.tagName = String(tag).toUpperCase();
    this.children = [];
    this.attrs = {};
    this.style = {};
    this.dataset = {};
    this.parentNode = null;
    this.className = "";
    this.hidden = false;
    this.disabled = false;
    this.title = "";
    this._text = "";
  }
  get firstChild() { return this.children[0] || null; }
  get isConnected() { return true; }
  appendChild(c) {
    if (c.parentNode) c.parentNode.removeChild(c);
    c.parentNode = this;
    this.children.push(c);
    return c;
  }
  removeChild(c) {
    const i = this.children.indexOf(c);
    if (i >= 0) this.children.splice(i, 1);
    c.parentNode = null;
    return c;
  }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
  replaceWith(n) {
    const p = this.parentNode;
    if (!p) return;
    p.children[p.children.indexOf(this)] = n;
    n.parentNode = p;
    this.parentNode = null;
  }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  hasAttribute(k) { return k in this.attrs; }
  removeAttribute(k) { delete this.attrs[k]; }
  addEventListener(type, f) { ((this.listeners ||= {})[type] ||= []).push(f); }
  click() { for (const f of (this.listeners && this.listeners.click) || []) f({}); }
  get classList() {
    const n = this;
    const list = () => n.className.split(/\s+/).filter(Boolean);
    return {
      contains: (c) => list().includes(c),
      add: (c) => { if (!list().includes(c)) n.className = [...list(), c].join(" "); },
      remove: (c) => { n.className = list().filter((x) => x !== c).join(" "); },
      toggle: (c, on) => {
        const want = on === undefined ? !list().includes(c) : !!on;
        const rest = list().filter((x) => x !== c);
        n.className = (want ? [...rest, c] : rest).join(" ");
        return want;
      },
    };
  }
  get textContent() {
    return this.tagName === "#TEXT" ? this._text : this.children.map((c) => c.textContent).join("");
  }
  set textContent(v) {
    this.children = [];
    if (this.tagName === "#TEXT") this._text = String(v);
    else if (String(v) !== "") this.appendChild(textNode(v));
  }
  querySelectorAll() { return []; }
  querySelector() { return null; }
  focus() {}
}
function textNode(v) {
  const t = new FakeNode("#text");
  t._text = String(v);
  return t;
}
const nodes = {};
globalThis.Node = FakeNode;
globalThis.getComputedStyle = () => ({ getPropertyValue: () => "" });
globalThis.window = { localStorage: { getItem: () => null, setItem() {} } };
globalThis.location = { hash: "#/alerts" };
globalThis.document = {
  createElement: (t) => new FakeNode(t),
  createTextNode: textNode,
  createElementNS: (ns, t) => new FakeNode(t),
  getElementById: (id) => (nodes[id] ||= Object.assign(new FakeNode("div"), { id })),
  querySelectorAll: () => [],
  documentElement: new FakeNode("html"),
  body: new FakeNode("body"),
};
const timers = [];
globalThis.setTimeout = (f, ms) => { timers.push({ f, ms }); return timers.length; };
globalThis.clearTimeout = () => {};
const walk = (n, pred, out = []) => {
  if (pred(n)) out.push(n);
  for (const c of n.children) walk(c, pred, out);
  return out;
};
const tick = () => new Promise((r) => setImmediate(r));
"""

FILTER_BODY = r"""
const S = 1e6;
const v = (verdict, ts) => ({ alert_id: "x", verdict, analyst: "ann", ts_us: ts });
const list = [
  { alert_id: "a", ts_event_us: 1 * S, feedback: [] },
  { alert_id: "b", ts_event_us: 4 * S, feedback: [v("tp", 1)] },
  { alert_id: "c", ts_event_us: 3 * S, feedback: [v("tp", 1), v("fp", 2)] },
  { alert_id: "d", ts_event_us: 2 * S, feedback: [v("fp", 1), v("unsure", 2)] },
  { alert_id: "e", ts_event_us: 5 * S },
];
const ids = (k) => m.filterAlerts(list, k).map((a) => a.alert_id);
assert.deepStrictEqual(ids("all"), ["e", "b", "c", "d", "a"]);
assert.deepStrictEqual(ids("open"), ["e", "a"]);
assert.deepStrictEqual(ids("tp"), ["b"]);
assert.deepStrictEqual(ids("fp"), ["c"]);  // the latest verdict counts, not the first
assert.deepStrictEqual(ids("nope"), ["e", "b", "c", "d", "a"]);
assert.deepStrictEqual(list.map((a) => a.alert_id), ["a", "b", "c", "d", "e"]);  // not mutated
// "latest" is by time, whatever order the rows came in
const late = [{ alert_id: "f", ts_event_us: 1, feedback: [v("fp", 9), v("tp", 3)] }];
assert.deepStrictEqual(m.filterAlerts(late, "fp").map((a) => a.alert_id), ["f"]);
assert.deepStrictEqual(m.ALERT_FILTERS, [
  { key: "all", label: "ทั้งหมด" }, { key: "open", label: "ยังไม่ตรวจ" },
  { key: "tp", label: "ผิดปกติจริง" }, { key: "fp", label: "เตือนผิด" }]);
assert.deepStrictEqual(m.VERDICT_TH, { tp: "ผิดปกติจริง", fp: "เตือนผิด", unsure: "ไม่แน่ใจ" });
console.log("ok");
"""


def _run_dom(tmp_path, reduced, body, *args):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = tmp_path / "dom.js"
    script.write_text(DOM_HEAD % reduced + body, encoding="utf-8")
    r = subprocess.run(
        [node, str(script), str(WEB_DIR / "app.js"), *map(str, args)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=tmp_path,
    )
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr + r.stdout


def test_filter_alerts(tmp_path):
    _run_dom(tmp_path, "false", FILTER_BODY)


TOAST_BODY = r"""
(async () => {
  const S = 1e6;
  const at = Date.UTC(2021, 0, 18, 16, 30) * 1000;
  const mk = (id) => ({ alert_id: id, symbol: "STEEMBTC", ts_event_us: at, technique_id: "MM-T001",
    severity: 2, status: "open", window: [at - 60 * S, at] });
  const root = document.getElementById("toast-root");
  m.onAlert(mk("a1"));
  assert.strictEqual(root.children.length, 1);
  const t = root.children[0];
  const said = t.textContent;
  assert.ok(said.includes("STEEMBTC") && said.includes("23:30 น."), said);
  assert.ok(!/MM-T|UTC/.test(t.textContent), t.textContent);
  assert.strictEqual(walk(t, (n) => n.getAttribute("href") === "#/alerts/a1").length, 1);
  assert.ok(!t.classList.contains("slide"), "no slide under reduced motion");
  // an update of an alert already known is not new
  m.onAlert({ ...mk("a1"), status: "updated" });
  assert.strictEqual(root.children.length, 1);
  // it goes after 5 s
  const five = timers.filter((x) => x.ms === 5000);
  assert.strictEqual(five.length, 1);
  five[0].f();
  assert.strictEqual(root.children.length, 0);
  // at most three at once: the oldest makes room
  for (const id of ["b1", "b2", "b3", "b4"]) m.onAlert(mk(id));
  assert.strictEqual(root.children.length, 3);
  const hrefs = root.children.map((c) => walk(c, (n) => n.tagName === "A")[0].getAttribute("href"));
  assert.deepStrictEqual(hrefs, ["#/alerts/b2", "#/alerts/b3", "#/alerts/b4"]);
  // a resync re-reads everything: alerts it brings are not news
  globalThis.fetch = async (path) => ({ ok: true, status: 200, json: async () =>
    (path.startsWith("/api/alerts") ? [mk("c1"), mk("c2"), mk("b1")]
      : path.startsWith("/api/feedback") ? [] : {}) });
  m.onAlert({ event: "resync" });
  for (let i = 0; i < 5; i += 1) await tick();
  assert.ok(m.getAlertIds().includes("c1"));
  assert.strictEqual(root.children.length, 3);
  m.onAlert({ event: "alerts_reset", deleted: ["c1"] });
  for (let i = 0; i < 5; i += 1) await tick();
  assert.ok(!m.getAlertIds().includes("c1"));
  assert.strictEqual(root.children.length, 3);
  // the cards carry the Thai time, never a technique id or UTC
  const list = document.getElementById("alerts-list");
  assert.strictEqual(document.getElementById("alerts-empty").hidden, true);
  assert.ok(!/MM-T|UTC/.test(list.textContent), list.textContent);
  assert.ok(list.textContent.includes("23:30 น."), list.textContent);
  m.onAlert({ event: "alerts_reset", deleted: m.getAlertIds() });
  for (let i = 0; i < 5; i += 1) await tick();
  assert.strictEqual(document.getElementById("alerts-empty").hidden, false);
  console.log("ok");
})().catch((e) => { console.error(e); process.exit(1); });
"""

SLIDE_BODY = r"""
const root = document.getElementById("toast-root");
m.showToast("ข้อความ", null);
assert.strictEqual(root.children.length, 1);
assert.ok(root.children[0].classList.contains("slide"));
assert.strictEqual(walk(root, (n) => n.tagName === "A").length, 0);  // no link without an href
console.log("ok");
"""


def test_new_alert_toasts_once_and_never_on_resync(tmp_path):
    _run_dom(tmp_path, "true", TOAST_BODY)
    _run_dom(tmp_path, "false", SLIDE_BODY)


def test_toast_markup_and_motion():
    html, css = text("index.html"), text("styles.css")
    root = re.search(r'<div id="toast-root"[^>]*>', html)
    assert root and 'aria-live="polite"' in root.group(0)
    assert re.search(r"\.toast\.slide\s*\{[^}]*animation", css)
    reduced = css.split("prefers-reduced-motion: reduce")[1]
    assert "animation: none" in reduced
    banner = re.search(r'<div id="ws-banner"[^>]*>', html).group(0)
    assert "ws-toast" in banner and "hidden" in banner
    js = text("app.js")
    ws = js[js.index("function scheduleReconnect(") :]
    assert 'wsBanner("การเชื่อมต่อหลุด กำลังเชื่อมใหม่…")' in ws[: ws.index("\n}\n")]


def test_alerts_page_markup():
    html = text("index.html")
    page = re.search(r'<section id="page-alerts".*?</section>', html, re.S).group(0)
    for i in ("alerts-filters", "alerts-list", "alerts-empty", "alerts-error"):
        assert f'id="{i}"' in page, i
    empty = re.search(r'<div id="alerts-empty"[^>]*>.*?</div>', page, re.S).group(0)
    assert re.sub(r"<[^>]+>", "", empty).strip() == "ยังไม่มีการแจ้งเตือน ระบบกำลังเฝ้าดูอยู่"
    assert "<table" not in page and "UTC" not in page
    css = text("styles.css")
    assert '.alert-card[data-sev="low"] { border-left-color: var(--dim)' in css
    assert '.alert-card[data-sev="mid"] { border-left-color: var(--amber)' in css
    assert '.alert-card[data-sev="high"] { border-left-color: var(--red)' in css
    # filter chips are flat pills, not glass
    chip = re.search(r"(?m)^\.filter-chip\s*\{([^}]*)\}", css).group(1)
    assert "var(--r-pill)" in chip and "backdrop-filter" not in chip


DETAIL_BODY = r"""
(async () => {
  const fs = require("fs");
  const detail = JSON.parse(fs.readFileSync(process.argv[3], "utf-8"));
  const tax = JSON.parse(fs.readFileSync(process.argv[4], "utf-8"));
  globalThis.fetch = async (path) => ({ ok: true, status: 200, json: async () =>
    (path.startsWith("/api/taxonomy") ? tax : detail) });
  const box = document.getElementById("alert-detail");
  for (const role of ["analyst", "viewer"]) {
    m.setMe({ username: "u", role, auth_enabled: true, prefs: {} });
    m.setRoute("alert");
    await m.showAlert(detail.alert_id);
    const techs = walk(box, (n) => n.tagName === "DETAILS" && n.classList.contains("tech"));
    assert.strictEqual(techs.length, 1);
    const tech = techs[0];
    assert.ok(!tech.hasAttribute("open"), "details start closed");
    assert.strictEqual(tech.children[0].tagName, "SUMMARY");
    assert.strictEqual(tech.children[0].textContent, "ดูรายละเอียด");
    const inside = tech.textContent;
    for (const want of ["4.0e-4", "rule.x", "MM-T001", "replay", "UTC"]) {
      assert.ok(inside.includes(want), want);
    }
    assert.ok(walk(tech, (n) => n.textContent === "rule").length >= 1, "the layer is shown");
    // outside the details: plain Thai, no ids, p-values, layers, provenance or UTC
    tech.parentNode.removeChild(tech);
    const plain = box.textContent;
    for (const bad of ["MM-T", "UTC", "4.0e-4", "rule.x", "replay", "p ≈", "robust", "z_"]) {
      assert.ok(!plain.includes(bad), `${bad} in: ${plain}`);
    }
    assert.ok(!/[A-Za-z]/.test(plain.replaceAll("TESTBTC", "")), plain);
    assert.ok(plain.includes("23:30:05 น."), plain);
    const btns = walk(box, (n) => n.tagName === "BUTTON");
    assert.deepStrictEqual(btns.map((b) => b.textContent), ["ผิดปกติจริง", "เตือนผิด", "ไม่แน่ใจ"]);
    for (const b of btns) assert.strictEqual(b.hasAttribute("disabled"), role === "viewer", role);
  }
  console.log("ok");
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_alert_detail_markup(tmp_path, client):
    ts = 1610987405 * 1_000_000  # 2021-01-18 16:30:05 UTC, 23:30:05 Thai time
    alert = Alert(
        alert_id="a1",
        ts_event_us=ts,
        symbol="TESTBTC",
        technique_id="MM-T001",
        severity=2,
        status="open",
        fused_p=0.0004,
        detectors=[DetectorRef(name="rule.x", layer="rule", p=0.01, score=1.0)],
        agreement=2,
        evidence=[("z_quote_vol_60", 7.1, 7.1), ("ret_60", 0.042, 0.5)],
        window=(ts - 60_000_000, ts),
        dedup_key="k",
        suppress_for_s=60,
        config_hash="h",
        model_versions={},
        provenance="replay",
    )
    client.app.state.idsai.store.upsert_alert(alert)
    detail = tmp_path / "detail.json"
    detail.write_text(json.dumps(client.get("/api/alerts/a1").json()), encoding="utf-8")
    tax = tmp_path / "tax.json"
    tax.write_text(json.dumps(client.get("/api/taxonomy").json()), encoding="utf-8")
    _run_dom(tmp_path, "true", DETAIL_BODY, detail, tax)


RESET_KEEPS_VERDICT_BODY = r"""
(async () => {
  const at = Date.UTC(2021, 0, 18, 16, 30) * 1000;
  const mk = () => ({ alert_id: "r1", symbol: "STEEMBTC", ts_event_us: at, technique_id: "MM-T001",
    severity: 2, status: "open", window: [at - 60e6, at] });
  const fb = [{ alert_id: "r1", verdict: "tp", analyst: "ann", ts_us: 5 }];
  globalThis.fetch = async (path) => ({ ok: true, status: 200, json: async () =>
    (path.startsWith("/api/alerts") ? [mk()] : path.startsWith("/api/feedback") ? fb : {}) });
  m.onAlert({ event: "resync" });
  for (let i = 0; i < 5; i += 1) await tick();
  const list = document.getElementById("alerts-list");
  const chip = () =>
    walk(list, (n) => n.classList.contains("verdict-chip")).map((n) => n.textContent);
  assert.deepStrictEqual(chip(), ["ผิดปกติจริง"]);
  // a seek back deletes the alert on the server, but not its verdicts; the replay brings it back
  m.onAlert({ event: "alerts_reset", deleted: ["r1"] });
  for (let i = 0; i < 5; i += 1) await tick();
  assert.deepStrictEqual(m.getAlertIds(), []);
  m.onAlert(mk());
  assert.deepStrictEqual(chip(), ["ผิดปกติจริง"]);
  const all = m.getAlertsWithFeedback();
  assert.deepStrictEqual(m.filterAlerts(all, "tp").map((a) => a.alert_id), ["r1"]);
  assert.deepStrictEqual(m.filterAlerts(all, "open").map((a) => a.alert_id), []);
  console.log("ok");
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_alerts_reset_keeps_verdicts_for_a_replayed_alert(tmp_path):
    _run_dom(tmp_path, "true", RESET_KEEPS_VERDICT_BODY)


def test_ws_toast_dot_does_not_loop():
    css = text("styles.css")
    dot = re.search(r"\.ws-toast::before\s*\{([^}]*)\}", css).group(1)
    assert "animation" not in dot
    for rule in re.findall(r"\.ws-toast[^{]*\{([^}]*)\}", css):
        assert "infinite" not in rule


SAVE_BODY = r"""
(async () => {
  const fs = require("fs");
  const detail = JSON.parse(fs.readFileSync(process.argv[3], "utf-8"));
  let posts = 0;
  let release;
  let postFails = false;
  let getFails = false;
  globalThis.fetch = async (path, opts) => {
    if (opts && opts.method === "POST") {
      posts += 1;
      await new Promise((r) => { release = r; });
      return postFails
        ? { ok: false, status: 500, json: async () => ({ detail: "พัง" }) }
        : { ok: true, status: 201, json: async () => ({}) };
    }
    if (path.startsWith("/api/taxonomy")) return { ok: true, status: 200, json: async () => ({}) };
    if (getFails) return { ok: false, status: 500, json: async () => ({ detail: "อ่านไม่ได้" }) };
    return { ok: true, status: 200, json: async () => detail };
  };
  m.setMe({ username: "ann", role: "analyst", auth_enabled: true, prefs: {} });
  m.setRoute("alert");
  const box = document.getElementById("alert-detail");
  await m.showAlert(detail.alert_id);
  const btns = walk(box, (n) => n.tagName === "BUTTON");
  const status = walk(box, (n) => n.classList.contains("verdict-status"))[0];
  // the POST succeeds, the follow-up read fails: still "saved", never "บันทึกไม่สำเร็จ"
  getFails = true;
  btns[0].click();
  await tick();
  assert.strictEqual(posts, 1);
  for (const b of btns) assert.strictEqual(b.disabled, true, "buttons wait for the save");
  btns[1].click();  // a second click while saving sends nothing
  await tick();
  assert.strictEqual(posts, 1);
  release();
  for (let i = 0; i < 10; i += 1) await tick();
  assert.ok(status.textContent.startsWith("บันทึกแล้ว: ผิดปกติจริง"), status.textContent);
  assert.ok(!status.textContent.includes("ไม่สำเร็จ"), status.textContent);
  for (const b of btns) assert.strictEqual(b.disabled, false);
  assert.strictEqual(btns[0].getAttribute("aria-pressed"), "true");
  // the POST fails: say so, and the buttons come back
  postFails = true;
  btns[1].click();
  await tick();
  release();
  for (let i = 0; i < 10; i += 1) await tick();
  assert.ok(status.textContent.startsWith("บันทึกไม่สำเร็จ"), status.textContent);
  for (const b of btns) assert.strictEqual(b.disabled, false);
  assert.strictEqual(posts, 2);
  console.log("ok");
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_verdict_save_reports_the_post_and_blocks_double_posts(tmp_path, client):
    ts = 1610987405 * 1_000_000
    alert = Alert(
        alert_id="a1",
        ts_event_us=ts,
        symbol="TESTBTC",
        technique_id="MM-T001",
        severity=2,
        status="open",
        fused_p=0.0004,
        detectors=[DetectorRef(name="rule.x", layer="rule", p=0.01, score=1.0)],
        agreement=2,
        evidence=[("z_quote_vol_60", 7.1, 7.1)],
        window=(ts - 60_000_000, ts),
        dedup_key="k",
        suppress_for_s=60,
        config_hash="h",
        model_versions={},
        provenance="replay",
    )
    client.app.state.idsai.store.upsert_alert(alert)
    detail = tmp_path / "detail.json"
    detail.write_text(json.dumps(client.get("/api/alerts/a1").json()), encoding="utf-8")
    _run_dom(tmp_path, "true", SAVE_BODY, detail)


# ---- simulate page ----

SIM_GATES_BODY = r"""
(async () => {
  const cards = m.SIM_CARDS.map((c) => document.getElementById(`sim-${c.family}`));
  const reason = document.getElementById("sim-reason");
  const posts = [];
  const S = 1e6;
  globalThis.fetch = async (path, opts) => {
    if (opts && opts.method === "POST") posts.push({ path, body: JSON.parse(opts.body) });
    return { ok: true, status: 202, json: async () => ({
      scenario: { technique_id: "MM-T001", sub_technique: "burst", duration_s: 120 },
      label: { symbol: "STEEMBTC", t_start_us: 1060 * S, t_end_us: 1180 * S } }) };
  };
  m.bindSimulate();
  // before any health: closed, even for an analyst
  m.setMe({ username: "a", role: "analyst", auth_enabled: true, prefs: {} });
  m.applyRoleGates();
  for (const c of cards) assert.strictEqual(c.disabled, true, "fail closed");
  assert.strictEqual(reason.hidden, false);
  cards[0].click();
  await tick();
  assert.strictEqual(posts.length, 0);
  // simulate cards disabled for viewer
  m.setMe({ username: "v", role: "viewer", auth_enabled: true, prefs: {} });
  m.setMode("replay");
  for (const c of cards) assert.strictEqual(c.disabled, true, "viewer");
  assert.strictEqual(reason.hidden, false);
  assert.strictEqual(reason.textContent, m.SIM_REASON.viewer);
  // an analyst in replay: open, and one click is one POST with the defaults
  m.setMe({ username: "a", role: "analyst", auth_enabled: true, prefs: {} });
  m.applyRoleGates();
  for (const c of cards) assert.strictEqual(c.disabled, false, "analyst");
  assert.strictEqual(reason.hidden, true);
  document.getElementById("rt-region").value = "test";
  document.getElementById("rt-offset").value = "60";
  document.getElementById("rt-symbol").value = "STEEMBTC";
  cards[0].click();
  for (let i = 0; i < 5; i += 1) await tick();
  assert.strictEqual(posts.length, 1);
  assert.strictEqual(posts[0].path, "/api/redteam/inject");
  assert.deepStrictEqual(posts[0].body, m.simulateRequest("pump_burst", null, "STEEMBTC"));
  // the run is a timeline card with three steps
  const runs = document.getElementById("rt-runs");
  assert.strictEqual(runs.children.length, 1);
  const steps = walk(runs, (n) => n.classList.contains("step"));
  assert.strictEqual(steps.length, 3);
  assert.ok(steps[0].textContent.includes("เริ่มแล้ว"), steps[0].textContent);
  assert.ok(steps[1].textContent.includes("กำลังรอระบบ"), steps[1].textContent);
  assert.ok(runs.children[0].textContent.includes("ปั่นเร็ว"));
  const tech = walk(runs, (n) => n.tagName === "DETAILS");
  assert.strictEqual(tech.length, 1);
  assert.ok(tech[0].textContent.includes("UTC"));
  tech[0].parentNode.removeChild(tech[0]);
  assert.ok(!/[A-Za-z]/.test(runs.textContent.replaceAll("STEEMBTC", "")), runs.textContent);
  // simulate cards disabled in live
  m.setMode("live");
  for (const c of cards) assert.strictEqual(c.disabled, true, "live");
  assert.strictEqual(reason.textContent, m.SIM_REASON.live);
  console.log("ok");
})().catch((e) => { console.error(e); process.exit(1); });
"""


SIM_ENDED_BODY = r"""
(async () => {
  const cards = m.SIM_CARDS.map((c) => document.getElementById(`sim-${c.family}`));
  const reason = document.getElementById("sim-reason");
  globalThis.fetch = async () => ({ ok: false, status: 409,
    json: async () => ({ detail: "การเล่นย้อนหลังนี้จบแล้ว ต้องเริ่มใหม่" }) });
  m.bindSimulate();
  m.setMe({ username: "a", role: "analyst", auth_enabled: true, prefs: {} });
  m.setMode("replay");
  document.getElementById("rt-offset").value = "60";
  cards[1].click();
  for (let i = 0; i < 5; i += 1) await tick();
  for (const c of cards) assert.strictEqual(c.disabled, true, "ended");
  assert.strictEqual(reason.textContent, m.SIM_REASON.ended);
  // a new replay round opens the cards again
  m.onAlert({ event: "replay_restarted" });
  for (const c of cards) assert.strictEqual(c.disabled, false, "new round");
  assert.strictEqual(reason.hidden, true);
  console.log("ok");
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_simulate_cards_disabled_for_viewer_and_in_live(tmp_path):
    _run_dom(tmp_path, "true", SIM_GATES_BODY)


def test_simulate_cards_close_once_the_replay_ended(tmp_path):
    _run_dom(tmp_path, "true", SIM_ENDED_BODY)


def _css_blocks(css: str):
    """(prelude, depth, body) for every brace block, after a balanced-brace parse."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    stack, out, depth, last = [], [], 0, 0
    for i, ch in enumerate(css):
        if ch == "{":
            prelude = css[last:i].split(";")[-1].split("}")[-1].strip()
            stack.append((prelude, depth, i))
            depth += 1
            last = i + 1
        elif ch == "}":
            assert stack, f"unbalanced closing brace at {i}"
            prelude, d, start = stack.pop()
            out.append((prelude, d, css[start + 1 : i], [p for p, _, _ in stack]))
            depth -= 1
            last = i + 1
    assert not stack, f"unclosed block: {stack[-1][0]}"
    return out


def test_styles_css_blocks_are_balanced_and_motion_is_top_level():
    blocks = _css_blocks(text("styles.css"))
    keyframes = [b for b in blocks if b[0].startswith("@keyframes")]
    assert keyframes
    for prelude, depth, _, parents in keyframes:
        assert depth == 0 and not parents, (prelude, parents)  # never inside an @media
    reduced = [b for b in blocks if "prefers-reduced-motion: reduce" in b[0]]
    assert reduced and all(depth == 0 for _, depth, _, _ in reduced)


SIM_ERROR_BODY = r"""
const err = (status, message) => Object.assign(new Error(message), { status });
const t = m.simErrorText;
assert.strictEqual(t(err(409, "การเล่นย้อนหลังนี้จบแล้ว ต้องเริ่มใหม่")), m.SIM_REASON.ended);
assert.strictEqual(t(err(422, "สถานการณ์จำลองจะจบหลังข้อมูลสุดท้าย ลดค่า offset_s")),
  "เหตุการณ์จะเลยข้อมูลสุดท้าย ลองลดเวลาเริ่มในตั้งค่าขั้นสูง");
assert.strictEqual(t(err(422, "ต้องระบุ symbol ที่รู้จัก")), "ไม่รู้จักเหรียญนี้");
assert.strictEqual(t(err(403, "Red-Team ปิดอยู่ในโหมด LIVE เพราะระบบไม่แก้ไขข้อมูลตลาดจริง")),
  m.SIM_REASON.live);
assert.strictEqual(t(err(403, "สิทธิ์ไม่พอ (ต้องเป็น analyst)")), m.SIM_REASON.viewer);
for (const e of [err(422, "Input should be less than or equal to 3600"), err(500, "boom"),
  err(409, "ไม่มีตัวเล่นย้อนหลังสำหรับฉีดข้อมูลจำลอง"), new Error("Failed to fetch"), null]) {
  const out = t(e);
  assert.strictEqual(out, "เริ่มการทดลองไม่สำเร็จ ลองใหม่อีกครั้ง");
  assert.ok(!/[A-Za-z]/.test(out));
}
"""


def test_simulate_errors_are_plain_thai(tmp_path):
    _run_motion(tmp_path, "false", SIM_ERROR_BODY)


SIM_REOPEN_BODY = r"""
(async () => {
  const cards = m.SIM_CARDS.map((c) => document.getElementById(`sim-${c.family}`));
  const status = document.getElementById("rt-status");
  let mode = "ended";
  globalThis.fetch = async (path) => {
    if (path === "/api/replay/control") return { ok: true, status: 200, json: async () => ({}) };
    if (mode === "ended") return { ok: false, status: 409,
      json: async () => ({ detail: "การเล่นย้อนหลังนี้จบแล้ว ต้องเริ่มใหม่" }) };
    return { ok: false, status: 422,
      json: async () => ({ detail: [{ msg: "Input should be valid" }] }) };
  };
  const warns = [];
  console.warn = (...a) => warns.push(a.join(" "));
  m.bindSimulate();
  m.setMe({ username: "a", role: "analyst", auth_enabled: true, prefs: {} });
  m.setMode("replay");
  const end = async () => {
    cards[0].click();
    for (let i = 0; i < 5; i += 1) await tick();
    for (const c of cards) assert.strictEqual(c.disabled, true, "ended");
  };
  // a successful seek or play reopens the cards; a pause does not
  for (const action of ["seek", "play"]) {
    await end();
    assert.strictEqual(await m.control("pause"), true);
    for (const c of cards) assert.strictEqual(c.disabled, true, "pause keeps it closed");
    assert.strictEqual(await m.control(action, action === "seek" ? 1 : undefined), true);
    for (const c of cards) assert.strictEqual(c.disabled, false, action);
  }
  // a mode change is a new round too
  await end();
  m.setMode("live");
  m.setMode("replay");
  for (const c of cards) assert.strictEqual(c.disabled, false, "mode");
  // any other refusal: plain Thai on screen, the raw text only in the console
  mode = "other";
  cards[0].click();
  for (let i = 0; i < 5; i += 1) await tick();
  assert.strictEqual(status.textContent, "เริ่มการทดลองไม่สำเร็จ ลองใหม่อีกครั้ง");
  assert.ok(warns.some((w) => w.includes("Input should be valid")), warns.join("|"));
  console.log("ok");
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_simulate_cards_reopen_after_seek_play_or_mode_change(tmp_path):
    _run_dom(tmp_path, "true", SIM_REOPEN_BODY)


# ---- results: plain cards, budget chips, committee details folded ----

RESULTS_BODY = r"""
(async () => {
  const cell = (recall, fa, hard, unit, prov) => ({
    provenance: prov, n_runs: 5, n_seeds: 5, n_events: 200, ci_unit: unit,
    mean: { recall_any: recall, recall_correct: 0.4, fa_per_symbol_day: fa,
      fa_per_symbol_day_hard: hard, precision: 0.11, prevalence: 0.0017, ttd_median_s: 5.8 },
    ci: { recall_any: [recall - 0.05, recall + 0.05], recall_correct: [0.3, 0.5] },
    rule_of_three: { recall_any: null, recall_correct: null } });
  const every = (c) => ({ "0.1": c, "0.5": c, "2": c, "10": c });
  const row = (baseline, diff, ci) => ({ baseline, diff_mean: diff, ci, n_events: 200, n_seeds: 5,
    provenance: "REAL", significant: ci[0] > 0, reason_th: null });
  const rep = {
    runs: 500, git_sha: ["abc1234"], config_hash: ["53e3b662ab96"], signatures_sha: ["sig99"],
    notes: ["หมายเหตุหนึ่ง"],
    experiments: {
      real_t001: { full: { "0.1": cell(0.54, 0.1, 1, "event", "REAL"),
        "0.5": cell(0.638, 0.5, 3, "event", "REAL"),
        "2": cell(0.739, 1.6498, 11.17, "event", "REAL"),
        "10": cell(0.808, 6.1, 40, "event", "REAL") } },
      semisyn: { pump_burst: every(cell(0.2955, 0.3, 1, "symbol-day", "SEMI-SYN")),
        pump_stealth: every(cell(0.351, 0.3, 1, "symbol-day", "SEMI-SYN")),
        wash_pingpong: every(cell(1, 0.3, 1, "symbol-day", "SEMI-SYN")) } },
    paired_bootstrap: { budget: 2, metric: "recall_any", notes: [], rows: [
      row("baseline.random", 0.639, [0.57, 0.7]), row("baseline.kk", -0.17, [-0.222, -0.121]),
      row("baseline.iforest", -0.158, [-0.208, -0.111]), row("baseline.pca", 0.075, [0.035, 0.122]),
      row("stat.robust_z", 0.005, [-0.033, 0.043])] },
  };
  globalThis.fetch = async () => ({ ok: true, status: 200, json: async () => rep });
  const saved = [];
  window.localStorage.setItem = (k, v) => saved.push([k, v]);
  await m.showEval();
  const box = document.getElementById("eval-body");
  const isCommittee = (n) => n.tagName === "DETAILS" && n.children[0]
    && n.children[0].tagName === "SUMMARY"
    && n.children[0].textContent === "รายละเอียดสำหรับกรรมการ";
  const committee = walk(box, isCommittee);
  assert.strictEqual(committee.length, 1);
  const det = committee[0];
  assert.ok(!det.hasAttribute("open"), "closed by default");
  // the visible text outside a folded subtree
  const textOutside = (skip) => {
    const go = (n) => (n === skip ? ""
      : n.tagName === "#TEXT" ? n._text : n.children.map(go).join(" "));
    return go;
  };
  const outside = textOutside(det);
  const plain = outside(box);
  assert.ok(!/[A-Za-z]/.test(plain), plain);
  for (const no of ["τ", "53e3b662ab96", "_"]) assert.ok(!plain.includes(no), no);
  // the forest plot, metric ids, config hash, run counts and provenance badges: inside only
  const inside = det.textContent;
  for (const s of ["recall_any", "53e3b662ab96", "500 run", "real_t001", "baseline.iforest",
    "prevalence",     "REAL", "SEMI-SYN", "หมายเหตุวิธีวัด"]) {
    assert.ok(inside.includes(s), s);
  }
  const forest = (n) => n.tagName === "SVG" && /\bforest\b/.test(n.attrs.class || "");
  assert.strictEqual(walk(det, forest).length, 1);
  assert.strictEqual(walk(box, forest).length, 1);
  const prov = (n) => /\bbadge\b/.test(n.className)
    && ["REAL", "SEMI-SYN", "SYN"].includes(n.textContent);
  assert.ok(walk(det, prov).length > 0);
  assert.strictEqual(walk(box, prov).length, walk(det, prov).length);

  // plain cards: one per target, the recall card wide, the rest compact
  const cards = () => walk(box, (n) => n.classList.contains("result-card"));
  const card = (id) => cards().find((c) => c.getAttribute("data-target") === id);
  assert.strictEqual(cards().length, m.EVAL_TARGETS.length);
  const wide = cards().filter((c) => c.classList.contains("result-wide"));
  assert.deepStrictEqual(wide.map((c) => c.getAttribute("data-target")), ["real_recall"]);
  const badge = (id) => walk(card(id), (n) => n.classList.contains("result-badge"))[0];
  const has = (id, s) => assert.ok(card(id).textContent.includes(s),
    id + ": " + card(id).textContent);
  has("real_recall", "จับเหตุการณ์ที่มีบันทึกได้เฉลี่ย 148 จาก 200 ครั้ง (เมื่อเตือนไม่เกิน 2 ครั้งต่อเหรียญต่อวัน)");
  assert.strictEqual(badge("real_recall").getAttribute("data-state"), "fail");
  assert.strictEqual(badge("real_recall").textContent, "✗ ยังไม่ถึงเป้า");
  has("real_fa", "เตือนผิดเฉลี่ย 1.65 ครั้งต่อเหรียญต่อวัน");
  assert.strictEqual(badge("real_fa").textContent, "✓ ผ่านเป้า");
  has("real_fa_hard", "เตือนผิดในวันที่ไม่มีเหตุการณ์ 11.17 ครั้ง");
  has("semisyn_pump_burst", "จับเหตุการณ์จำลองได้ 30% (เมื่อเตือนไม่เกิน 2 ครั้งต่อเหรียญต่อวัน)");
  has("ae_vs_baseline.iforest", "เทียบกับวิธีพื้นฐาน: แย่กว่า");
  has("ae_vs_stat.robust_z", "เทียบกับวิธีพื้นฐาน: ไม่ต่างอย่างมีนัยสำคัญ");
  for (const id of ["ae_vs_baseline.random", "ae_vs_baseline.pca"]) {
    has(id, "วิธีเปรียบเทียบ");
    has(id, "เทียบกับวิธีพื้นฐาน: ดีกว่า");
    assert.strictEqual(badge(id), undefined, id + " has no pass / fail");
  }
  has("latency_p99", "ความเร็วในการเตือน: ยังไม่วัด");
  assert.strictEqual(badge("latency_p99").textContent, "— ยังไม่วัด");
  assert.strictEqual(badge("latency_p99").getAttribute("data-state"), "none");

  // budget chips: flat, the saved / default one pressed; a click redraws and is remembered
  const chips = () => walk(box, (n) => n.classList.contains("budget-chip"));
  assert.deepStrictEqual(chips().map((c) => c.textContent), ["0.1", "0.5", "2", "10"]);
  const rowText = chips()[0].parentNode.parentNode.textContent;
  assert.ok(rowText.startsWith("งบเตือน") && rowText.endsWith("ครั้งต่อวัน"), rowText);
  const plain10 = textOutside(det);
  const pressed = () => chips().map((c) => c.getAttribute("aria-pressed"));
  assert.deepStrictEqual(pressed(), ["false", "false", "true", "false"]);
  chips()[3].click();
  assert.deepStrictEqual(pressed(), ["false", "false", "false", "true"]);
  has("real_recall", "(เมื่อเตือนไม่เกิน 10 ครั้งต่อเหรียญต่อวัน)");
  // pass / fail is judged at the project budget only: elsewhere the badge is neutral text
  assert.strictEqual(walk(box, (n) => n.classList.contains("result-badge")
    && ["pass", "fail"].includes(n.getAttribute("data-state"))).length, 0);
  assert.ok(!/[✓✗]/.test(plain10(box)), "no pass / fail glyph at budget 10");
  assert.strictEqual(badge("real_recall").textContent, "ไม่ได้ตั้งเป้าที่งบนี้");
  assert.strictEqual(badge("real_fa").textContent, "ไม่ได้ตั้งเป้าที่งบนี้");
  assert.strictEqual(card("real_recall").getAttribute("data-state"), "neutral");
  assert.strictEqual(badge("latency_p99").textContent, "— ยังไม่วัด");
  has("ae_vs_baseline.iforest", "ยังไม่วัด");
  // the committee targets table is always judged at the project budget, whatever the chip says
  const heads = walk(box, (n) => n.tagName === "H3" && n.textContent.startsWith("เป้าหมายตามแผน"));
  assert.deepStrictEqual(heads.map((h) => h.textContent), ["เป้าหมายตามแผน (งบ 2 ครั้ง/สัญลักษณ์-วัน)"]);
  assert.ok(det.textContent.includes("ผ่านเป้า") && det.textContent.includes("ไม่ถึงเป้า"));
  const row10 = walk(det, (n) => n.tagName === "TR" && n.textContent.includes("recall_any"));
  assert.ok(row10.some((r) => r.textContent.includes("0.739")), "targets table still at budget 2");
  assert.deepStrictEqual(saved, [["idsai.eval_budget", "10"]]);
  assert.strictEqual(walk(box, isCommittee).length, 1);
  // the committee details survive a budget change
  assert.strictEqual(walk(box, isCommittee)[0], det);
  assert.ok(!/[A-Za-z]/.test(outside(box)), outside(box));

  // no report yet: one plain line, the command folded into the committee details
  globalThis.fetch = async () => ({ ok: true, status: 200, json: async () => ({ runs: 0 }) });
  await m.showEval();
  const emptyDet = walk(box, isCommittee);
  assert.strictEqual(emptyDet.length, 1);
  assert.ok(emptyDet[0].textContent.includes("uv run idsai report"));
  const emptyPlain = textOutside(emptyDet[0]);
  assert.ok(!/[A-Za-z]/.test(emptyPlain(box)), emptyPlain(box));
  console.log("ok");
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_results_markup(tmp_path):
    _run_dom(tmp_path, "true", RESULTS_BODY)
    html, css = text("index.html"), text("styles.css")
    page = re.search(r'<section id="page-results".*?</section>', html, re.S).group(0)
    plain = re.sub(r"<!--.*?-->|<[^>]+>", " ", page, flags=re.S)
    assert not re.search(r"[A-Za-z]", plain), plain
    res = css[css.index("/* ---- results") : css.index("/* ---- /results")]
    # chips are flat; the cards are glass; no gradient and nothing loops on this page
    assert ".budget-chip" in res and "linear-gradient" not in res and "infinite" not in res
    assert re.search(r"\.result-wide\s*\{[^}]*flex:\s*2", res)


# ---- help drawer, first-time tour, tour_done ----

TOUR_BODY = r"""
assert.strictEqual(m.tourShouldShow({}, true, null), true);
assert.strictEqual(m.tourShouldShow({ tour_done: true }, true, null), false);
assert.strictEqual(m.tourShouldShow({}, false, "1"), false);
assert.strictEqual(m.tourShouldShow({}, false, null), true);
// auth off reads only this browser
assert.strictEqual(m.tourShouldShow({ tour_done: true }, false, null), true);
assert.deepStrictEqual(m.TOUR_STEPS.map((s) => s.target),
  ["#hero-title", "#chart-card", '[data-route="alerts"]', '[data-route="simulate"]']);
assert.strictEqual(m.TOUR_STEPS[1].text,
  "เส้นม่วงคือระดับความผิดปกติ ถ้าข้ามเส้นเตือนสีส้ม ระบบจะแจ้งเตือน");
const view = { w: 375, h: 812 };
const mid = m.placeBubble(null, view);
assert.strictEqual(mid.centred, true);
assert.ok(mid.left >= 0 && mid.top >= 0);
const low = m.placeBubble({ top: 760, left: 100, width: 100, height: 40 }, view);
assert.strictEqual(low.centred, false);
assert.ok(low.top + 190 <= 760 && low.top >= 0, "above the target, inside the screen");
const high = m.placeBubble({ top: 20, left: 300, width: 70, height: 30 }, view);
assert.ok(high.top >= 50, "below the target");
const edges = [
  { top: 0, left: 0, width: 10, height: 10 },
  { top: 800, left: 370, width: 40, height: 40 },
  { top: 400, left: 350, width: 120, height: 40 }];
for (const r of edges) {
  const p = m.placeBubble(r, view);
  const inside = p.left >= 0 && p.left + 320 <= 375 && p.top >= 0 && p.top + 190 <= 812;
  assert.ok(inside, JSON.stringify(p));
}
"""


def test_tour_pref_rule_and_bubble_placement(tmp_path):
    _run_motion(tmp_path, "false", TOUR_BODY)


TOUR_MISSING_BODY = r"""
document.querySelector = () => null; // a narrow screen: no target is on the page
m.setRoute("home");
m.startTour();
const bubble = document.getElementById("tour-bubble");
assert.strictEqual(document.getElementById("tour").hidden, false);
assert.ok(bubble.classList.contains("centred"), "a missing target centres the bubble");
m.endTour();
assert.strictEqual(document.getElementById("tour").hidden, true);
assert.strictEqual(globalThis.window.localStorage.getItem("idsai.tour_done"), "1");
console.log("ok");
"""


def test_tour_step_with_missing_target_centres(tmp_path):
    body = (
        'const keep = {}; globalThis.window = { localStorage: { getItem: (k) => keep[k] ?? null,'
        " setItem: (k, v) => { keep[k] = String(v); } } };\n"
        'm.setMe({ username: "local", role: "admin", auth_enabled: false, prefs: {} });\n'
        + TOUR_MISSING_BODY
    )
    _run_dom(tmp_path, "false", body)


HELP_TAX_BODY = r"""
const tax = {
  "MM-T001": { name_th: "การปั่นราคาแบบพุ่งเร็ว (Pump)", definition_th: "ราคาและปริมาณพุ่งในเวลาสั้น. ต่อ",
    observability_th: "ดูได้", reference: "Kamps 2018",
    sub_techniques: [{ id: "MM-T001.1", name_th: "ย่อย", definition_th: "x" }] },
  "MM-T000": { name_th: "ไม่จัดประเภท", definition_th: "d" },
};
const box = document.getElementById("help-taxonomy");
m.renderHelpTaxonomy(box, tax);
const walk2 = (n, f) => { f(n); n.children.forEach((c) => walk2(c, f)); };
let outside = "";
let inside = "";
walk2(box, (n) => { if (n.tagName === "DETAILS") inside += n.textContent; });
const strip = (n) => {
  if (n.tagName === "DETAILS") return "";
  return n.tagName === "#TEXT" ? n._text : n.children.map(strip).join("");
};
outside = strip(box);
assert.ok(!outside.includes("MT-") && !outside.includes("MM-"), outside);
assert.ok(outside.includes("การปั่นราคาแบบพุ่งเร็ว"), outside);
assert.ok(!/[A-Za-z]/.test(outside), outside); // no English in the plain view
assert.ok(inside.includes("Kamps 2018"), inside);
const summaries = walk(box, (n) => n.tagName === "SUMMARY").map((n) => n.textContent);
assert.deepStrictEqual(summaries, ["แหล่งอ้างอิง"]);
m.renderHelpTaxonomy(box, null);
assert.ok(box.textContent.length > 0);
console.log("ok");
"""


def test_help_drawer_taxonomy_has_no_ids(tmp_path):
    _run_dom(tmp_path, "false", HELP_TAX_BODY)


def test_help_and_tour_markup():
    html = text("index.html")
    drawer = re.search(r'<aside id="help-drawer".*?</aside>', html, re.S).group(0)
    for heading in ("ระบบนี้ทำอะไร", "รูปแบบที่ระบบมองหา", "คำศัพท์"):
        assert re.search(rf"<h3[^>]*>{heading}</h3>", drawer), heading
    assert "ระบบนี้ไม่ทำอะไร" in drawer
    for term in ("ระดับความผิดปกติ", "เส้นเตือน", "ย้อนดู", "ทดลองจำลอง", "ผิดปกติจริง"):
        assert term in drawer, term
    assert "เริ่มทัวร์" in drawer and 'id="help-taxonomy"' in drawer
    tour = re.search(r'<div id="tour".*?\n  </div>\n', html, re.S).group(0)
    for label in ("ถัดไป", "ข้าม", "เสร็จ"):
        assert f">{label}</button>" in tour, label
    js = text("app.js")
    assert js.count('"Escape"') >= 1 and "keydown" in js
    esc = js[js.index("function onOverlayKey(") :]
    esc = esc[: esc.index("\n}\n")]
    assert "Escape" in esc and "closeHelp()" in esc and "endTour()" in esc and "Tab" in esc
    # the leftovers are gone: About and Taxonomy only live in the drawer now
    assert 'id="page-about"' not in html and 'id="page-taxonomy"' not in html
    assert "function showTaxonomy" not in js and 'href: "#/taxonomy"' not in js
    css = text("styles.css")
    assert "#help-drawer" in css and ".tour-spot" in css and "rgba(5, 4, 20, 0.6)" in css
