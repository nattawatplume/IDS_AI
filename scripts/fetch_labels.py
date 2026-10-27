"""Download the Hu et al. Telegram pump-and-dump log to `data/labels/hu_pd_logs.tsv`.

The upstream repository has no license, so the file is fetched on demand and never committed
(`data/` is gitignored). Only raw.githubusercontent.com and api.github.com are contacted.

    uv run python scripts/fetch_labels.py
"""

import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx

REPO = "Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency"
LISTING_URL = f"https://api.github.com/repos/{REPO}/contents/Data/Telegram/Labeled"
FILE_NAME = "PD_logs_cleaned.txt"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/master/Data/Telegram/Labeled/{FILE_NAME}"
EXPECTED_HEADER = ["channel_id", "session_id", "coin", "exchange", "pair", "timestamp"]
ALLOWED_HOSTS = {"raw.githubusercontent.com", "api.github.com"}
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "labels" / "hu_pd_logs.tsv"


def _check_host(url: str) -> str:
    host = urlparse(url).hostname
    if urlparse(url).scheme != "https" or host not in ALLOWED_HOSTS:
        raise ValueError(f"refusing to contact {url!r}: host not in {sorted(ALLOWED_HOSTS)}")
    return url


def _get(client: httpx.Client, url: str) -> httpx.Response:
    resp = client.get(_check_host(url), follow_redirects=False, timeout=60.0)
    resp.raise_for_status()
    return resp


def verify_listing(entries: list[dict]) -> str:
    """Return the download URL of FILE_NAME from a GitHub contents listing, or raise."""
    for e in entries:
        if e.get("name") == FILE_NAME and e.get("type") == "file":
            return _check_host(e.get("download_url") or RAW_URL)
    raise RuntimeError(f"{FILE_NAME} not found in {LISTING_URL}: upstream layout changed")


def check_header(text: str) -> None:
    first = text.splitlines()[0] if text else ""
    got = first.rstrip("\r").split("\t")
    if got != EXPECTED_HEADER:
        raise RuntimeError(f"unexpected header {got!r}, expected {EXPECTED_HEADER!r}")


def fetch(out: Path = DEFAULT_OUT, client: httpx.Client | None = None) -> Path:
    own = client is None
    client = client or httpx.Client()
    try:
        url = verify_listing(_get(client, LISTING_URL).json())
        text = _get(client, url).content.decode("utf-8")
    finally:
        if own:
            client.close()
    check_header(text)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(text.encode("utf-8"))
    return out


def main() -> int:
    path = fetch()
    n = max(len(path.read_text(encoding="utf-8").splitlines()) - 1, 0)
    print(f"wrote {path} ({n} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
