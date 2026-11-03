"""Static dashboard: served, vendored chart pinned, no network, safe DOM, Thai wording."""

import hashlib
import re
import shutil
import subprocess

import pytest
from starlette.testclient import TestClient

from idsai.config import load_settings
from idsai.data.store import Store
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
