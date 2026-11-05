"""Static dashboard: served, vendored chart pinned, no network, safe DOM, Thai wording."""

import dataclasses
import hashlib
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
    assert '$("replay-controls").hidden = live;' in body
    assert "const allowed = redTeamAllowed(mode);" in body
    assert "refused.hidden = allowed;" in body and "REFUSED_TH" in body
    assert 'querySelectorAll("select, input, button")' in body
    assert "node.disabled = !allowed;" in body
    submit = js[js.index('$("redteam-form").addEventListener') :]
    assert submit.index("if (!redTeamAllowed(state.mode)) return;") < submit.index("postJson(")
    assert 'const RED_TEAM_MODES = ["replay", "sim"];' in js


def test_red_team_form_starts_disabled_until_health_confirms_the_mode():
    html = text("index.html")
    start = html.index('<form id="redteam-form"')
    form = html[start : html.index("</form>", start)]
    controls = re.findall(r"<(?:select|input|button)\b[^>]*>", form)
    assert len(controls) == 5
    for tag in controls:
        assert re.search(r"\sdisabled[\s>]", tag), tag  # fail closed before any health
    banner = re.search(r'<div id="redteam-refused"[^>]*>([^<]*)</div>', html)
    assert banner and "hidden" not in banner.group(0) and "LIVE" in banner.group(1)


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
    for sel in (".brand-name", "#auditor-text", "#user-name", ".nav a"):
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


def test_auditor_popup_is_anchored_inside_the_viewport_when_narrow():
    css = text("styles.css")
    m = re.search(r"@media \(max-width: 720px\) \{(.*?)\n\}", css, re.S)
    assert m, "no narrow media block"
    block = m.group(1)
    assert re.search(r"\.auditor\s*\{\s*position:\s*static", block)
    assert re.search(
        r"\.auditor-checks\s*\{[^}]*left:\s*16px[^}]*right:\s*16px[^}]*width:\s*auto", block
    )


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
    for name in ("pulse", "glow", "pop", "rise", "sweep", "grow"):
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


def test_monitor_has_kpi_strip_rail_and_info():
    html = text("index.html")
    for i in [
        "kpi-symbol", "kpi-f", "kpi-tau", "kpi-alerts", "kpi-budget", "kpi-auditor",
        "alert-rail", "layers-toggle", "chart-info", "chart-info-body",
    ]:  # fmt: skip
        assert f'id="{i}"' in html, i
    assert "ค่า p รวม" in html and "ความหายากเทียบตลาดปกติ" in html


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
    assert 'id="kpi-alerts">–<' in text("index.html")
    change = js.split('$("symbol-select").addEventListener("change"')[1][:400]
    assert "renderAlertsKpi()" in change


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
    assert 'tr[data-sev="low"] td:first-child { border-left-color: var(--dim)' in css
    assert ".rail-row.sev-1 { border-left-color: var(--dim)" in css
    js = text("app.js")
    assert "class: `pt ${tone}`" in js and "class: `dot ${tone}`" not in js


def test_red_team_miss_is_grey_and_the_banner_class_changes_only_with_state():
    css = text("styles.css")
    assert re.search(r"\.rt-banner\.hit\s*\{[^}]*var\(--amber\)", css)
    miss = re.search(r"\.rt-banner\.miss\s*\{([^}]*)\}", css).group(1)
    assert "var(--dim)" in miss and "--red" not in miss
    assert not re.search(r"\.rt-ring-arc\.missed\s*\{[^}]*--red", css)
    tick = text("app.js").split("function tickInjections()")[1].split("\n}\n")[0]
    block = tick.split("run.bannerState !== status")[1].split("} else {")[0]
    assert "run.banner.className" in block and "sweep" in block
    assert "run.banner.className" not in tick.split("run.bannerState !== status")[0]


# ---- alerts list and detail ----


def test_alert_rows_carry_severity_and_detail_has_two_columns():
    js = text("app.js")
    table = js.split("function renderAlertsTable()")[1].split("\n}\n")[0]
    assert "data-sev" in table
    assert "detail-main" in js and "detail-side" in js
    css = text("styles.css")
    assert "[data-sev=" in css and ".detail-side" in css


BANNER_BODY = r"""
assert.strictEqual(m.bannerText("detected", 42), "ตรวจพบใน 42 วินาที");
assert.strictEqual(m.bannerText("missed", 300), "ไม่พบภายใน 300 วินาทีหลังจบเหตุการณ์");
assert.strictEqual(m.bannerText("detected", 41.6), "ตรวจพบใน 42 วินาที");
assert.strictEqual(m.bannerText("waiting", 5), "");
assert.strictEqual(m.bannerText("detected", -3), "ตรวจพบใน 0 วินาที");
"""


def test_banner_text_for_hit_and_miss(tmp_path):
    _run_motion(tmp_path, "false", BANNER_BODY)


def test_red_team_ring_and_one_row_form():
    html = text("index.html")
    js = text("app.js")
    assert 'id="rt-ring"' in js or 'id: "rt-ring"' in js or "rt-ring" in js
    assert "rt-banner" in js and "stroke-dashoffset" in js
    assert 'class="form-grid rt-row"' in html or "rt-row" in html


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
"""


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
    about = re.search(r'<section id="page-about".*?</section>', html, re.S).group(0)
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
    rt = js[js.index('$("redteam-form").addEventListener') : js.index("function addInjection(")]
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
assert.strictEqual(m.bannerText("missed", 300, pending), "พลาด (replay เริ่มใหม่)");
assert.ok(m.bannerText("missed", 300, pending).includes("เริ่มใหม่"));
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
    assert len(pages) >= 8, pages
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
