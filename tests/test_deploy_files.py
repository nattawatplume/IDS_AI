"""Deploy files for the Hugging Face Space image (no Docker needed to check them)."""

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_check():
    spec = importlib.util.spec_from_file_location("check_no_hu", ROOT / "deploy" / "check_no_hu.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_db(data_dir: Path, source: str) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(data_dir / "app.duckdb"))
    con.execute("CREATE TABLE labels(label_id VARCHAR, source VARCHAR)")
    con.execute("INSERT INTO labels VALUES ('a', ?)", [source])
    con.close()


def test_check_no_hu_clean(tmp_path):
    _make_db(tmp_path, "sapienza")
    assert _load_check().main(tmp_path) == 0


def test_check_no_hu_file(tmp_path):
    _make_db(tmp_path, "sapienza")
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "hu_pd_logs.tsv").write_text("x")
    assert _load_check().main(tmp_path) == 1


def test_check_no_hu_row(tmp_path):
    _make_db(tmp_path, "hu")
    assert _load_check().main(tmp_path) == 1


def test_dockerignore_excludes():
    lines = {ln.strip() for ln in (ROOT / ".dockerignore").read_text().splitlines()}
    for want in ("data/", "models/", "*.duckdb*", ".git/", ".venv/", ".claude/", ".superpowers/"):
        assert want in lines


def test_start_script():
    raw = (ROOT / "deploy" / "start.sh").read_bytes()
    assert b"\r" not in raw
    text = raw.decode("utf-8")
    for want in (
        "idsai user seed-env",
        "--port 7860",
        "--auth",
        "--secure-cookies",
        "--loop",
        '--public-host "$SPACE_HOST"',
    ):
        assert want in text
    sh = shutil.which("sh")
    if sh is None:
        pytest.skip("no POSIX sh available")
    env = {k: v for k, v in os.environ.items() if k != "SPACE_HOST"}
    r = subprocess.run(
        [sh, str(ROOT / "deploy" / "start.sh")],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert r.returncode == 1
    assert "SPACE_HOST" in r.stderr
    assert any("\u0e00" <= c <= "\u0e7f" for c in r.stderr)


def test_dockerfile():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    for want in (
        "EXPOSE 7860",
        "USER user",
        "useradd -m -u 1000 user",
        "uv sync --frozen --no-dev",
        "idsai demo-pack",
        "deploy/check_no_hu.py",
        "deploy/start.sh",
    ):
        assert want in text
