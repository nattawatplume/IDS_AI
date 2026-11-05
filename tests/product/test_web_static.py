"""Static dashboard: served, vendored chart pinned, no network, safe DOM, Thai wording."""

import dataclasses
import hashlib
import re
import shutil
import subprocess

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
ROUTES = ("#/monitor", "#/alerts", "#/redteam", "#/eval", "#/taxonomy", "#/about")
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
    for name in ("monitor", "alerts", "redteam", "eval", "taxonomy", "about"):
        assert f'data-route="{name}"' in html and f'"page-{name}"' in js


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
    form = html[html.index('<form id="redteam-form"') : html.index("</form>")]
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
    "a button dd details div dl dt em h1 h2 h3 h4 input label li ol option p section select small span summary"
    " strong table tbody td th thead tr ul".split()
)


def test_el_builds_only_known_tags():
    tags = set(re.findall(r'\bel\("([^"]+)"', text("app.js")))
    assert tags and tags <= HTML_TAGS, sorted(tags - HTML_TAGS)


def test_both_themes_define_every_token():
    css = text("styles.css")
    tokens = ["--bg", "--panel", "--line", "--ink", "--dim", "--amber", "--red", "--green"]
    blocks = [css.split(':root[data-theme="dark"]')[1], css.split(':root[data-theme="light"]')[1]]
    for t in tokens:
        assert re.search(rf"{t}\s*:", css.split("@media")[0]), t  # bare :root
        for b in blocks:
            assert re.search(rf"{t}\s*:", b.split("}")[0]), t


def test_theme_toggle_exists_and_is_wired():
    assert 'id="theme-toggle"' in text("index.html")
    js = text("app.js")
    assert "function applyTheme(" in js and 'storageSet("theme"' in js


def test_no_capsule_tabs_or_shadows():
    css = text("styles.css")
    assert "box-shadow" not in css
    assert not re.search(r"border-radius:\s*(?:[5-9]|\d{2,})px", css)


def test_auditor_popup_is_anchored_inside_the_viewport_when_narrow():
    css = text("styles.css")
    m = re.search(r"@media \(max-width: 760px\) \{(.*?)\n\}", css, re.S)
    assert m, "no narrow media block"
    block = m.group(1)
    assert re.search(r"\.auditor\s*\{\s*position:\s*static", block)
    assert re.search(r"\.auditor-checks\s*\{[^}]*left:\s*16px[^}]*right:\s*16px[^}]*width:\s*auto", block)


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
assert.strictEqual(k({ key: " ", target: { tagName: "DIV" } }), "toggle-play");
assert.strictEqual(k({ key: "4", target: { tagName: "BODY" } }), "page:4");
assert.strictEqual(k({ key: "7", target: { tagName: "BODY" } }), null);
assert.strictEqual(k({ key: "t", target: { tagName: "BODY" } }), "theme");
assert.strictEqual(k({ key: "t", ctrlKey: true, target: { tagName: "BODY" } }), null);
assert.strictEqual(k({ key: "1", metaKey: true, target: { tagName: "BODY" } }), null);
assert.strictEqual(k({ key: "1", altKey: true, target: { tagName: "BODY" } }), null);
assert.strictEqual(m.PAGE_ORDER.length, 6);
assert.strictEqual(m.PAGE_ORDER[0], "#/monitor");
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
assert.strictEqual(m.bannerText("missed", 300), "ไม่พบภายใน 300 วินาที");
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
const cell = (mean, ci, n) => ({ mean: { recall_any: mean, fa_per_symbol_day: 1.65, fa_per_symbol_day_hard: 11.2 },
  ci: { recall_any: ci }, n_events: n });
const report = (full, rows) => ({ runs: 5, experiments: { real_t001: { full: { "2": full } } },
  paired_bootstrap: { budget: 2, metric: "recall_any", rows } });
const T = (id) => m.EVAL_TARGETS.find((t) => t.id === id);
const miss = m.targetVerdict(report(cell(0.739, [0.68, 0.801], 200), []), T("real_recall"));
assert.strictEqual(miss.state, "miss");
assert.strictEqual(miss.value, 0.739);
assert.strictEqual(miss.n, 200);
assert.deepStrictEqual(miss.ci, [0.68, 0.801]);
assert.strictEqual(m.targetVerdict(report(cell(0.85, null, 200), []), T("real_recall")).state, "pass");
assert.strictEqual(m.targetVerdict(report(cell(0.85, null, 200), []), T("real_fa")).state, "pass");
assert.strictEqual(m.targetVerdict(report(cell(0.85, null, 200), []), T("real_fa_hard")).state, "miss");
// variant absent / experiment absent / budget absent: never pass
assert.strictEqual(m.targetVerdict({ runs: 5, experiments: { real_t001: {} } }, T("real_recall")).state, "unmeasured");
assert.strictEqual(m.targetVerdict({ runs: 5, experiments: {} }, T("semisyn_pump_burst")).state, "unmeasured");
assert.strictEqual(m.targetVerdict({ runs: 5, experiments: { real_t001: { full: { "5": cell(0.9, null, 9) } } } }, T("real_recall")).state, "unmeasured");
assert.strictEqual(m.targetVerdict(report({ mean: {}, n_events: 3 }, []), T("real_recall")).state, "unmeasured");
// paired: the CI lower bound must clear 0; a baseline wholly better is a miss
const rows = [
  { baseline: "baseline.random", diff_mean: 0.639, ci: [0.57, 0.7], n_events: 200 },
  { baseline: "baseline.kk", diff_mean: -0.17, ci: [-0.222, -0.121], n_events: 200 },
  { baseline: "stat.robust_z", diff_mean: 0.005, ci: [-0.03, 0.04], n_events: 200 },
];
const r = report(cell(0.74, null, 200), rows);
assert.strictEqual(m.targetVerdict(r, T("ae_vs_baseline.random")).state, "pass");
assert.strictEqual(m.targetVerdict(r, T("ae_vs_baseline.kk")).state, "miss");
assert.strictEqual(m.targetVerdict(r, T("ae_vs_stat.robust_z")).state, "miss");
assert.strictEqual(m.targetVerdict(r, T("ae_vs_baseline.pca")).state, "unmeasured");
assert.strictEqual(m.targetVerdict({ runs: 5, experiments: {} }, T("ae_vs_baseline.kk")).state, "unmeasured");
assert.strictEqual(m.targetVerdict(null, T("real_recall")).state, "unmeasured");
assert.strictEqual(m.evalEmpty({ runs: 0 }), true);
assert.strictEqual(m.evalEmpty(null), true);
assert.strictEqual(m.evalEmpty({ runs: 3, experiments: {} }), false);
"""


def test_target_verdicts(tmp_path):
    _run_motion(tmp_path, "false", VERDICT_BODY)


def test_ablation_notes_cell_wraps():
    css = text("styles.css")
    assert re.search(r"td\.notes[^}]*max-width", css)
    js = text("app.js")
    assert 'class: "small muted notes"' in js and "หมายเหตุวิธีวัด" in js
    assert 'el("details"' in js and "uv run idsai report" in js


def test_long_explanations_live_on_about():
    html = text("index.html")
    mon = re.search(r'<section id="page-monitor".*?</section>', html, re.S).group(0)
    for m in re.finditer(r'<p\b([^>]*)>(.*?)</p>', mon, re.S):
        if 'id="chart-info-body"' in m.group(1):
            continue
        assert len(" ".join(m.group(2).split())) <= 120, m.group(2)[:60]
    about = re.search(r'<section id="page-about".*?</section>', html, re.S).group(0)
    assert "τ saturated" in about or "τ-saturated" in about
    assert "งบการแจ้งเตือนสูงกว่าอัตราสูงสุดที่ข้อมูลสอบเทียบทำได้" in about
    assert "autocorrelation" in about
    assert "อัตราการแจ้งเตือนจริงอาจสูงกว่างบที่ตั้งไว้" in about
