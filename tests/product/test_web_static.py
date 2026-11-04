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
    assert "refused.hidden = !live;" in body and "REFUSED_TH" in body
    assert 'querySelectorAll("select, input, button")' in body and "node.disabled = live;" in body
    submit = js[js.index('$("redteam-form").addEventListener') :]
    assert submit.index('if (state.mode === "live") return;') < submit.index("postJson(")


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
    "a button dd div dl dt em h1 h2 h3 h4 input label li ol option p section select small span"
    " strong table tbody td th thead tr ul".split()
)


def test_el_builds_only_known_tags():
    tags = set(re.findall(r'\bel\("([^"]+)"', text("app.js")))
    assert tags and tags <= HTML_TAGS, sorted(tags - HTML_TAGS)
