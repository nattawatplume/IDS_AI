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


def _workflow(name: str) -> tuple[dict, str]:
    import yaml

    text = (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
    return yaml.safe_load(text), text


def test_deploy_space_workflow():
    wf, text = _workflow("deploy-space.yml")
    triggers = wf[True]  # PyYAML reads the key `on` as True
    assert triggers["push"]["branches"] == ["main"]
    assert "workflow_dispatch" in triggers
    steps = wf["jobs"]["deploy"]["steps"]
    runs = "\n".join(s.get("run", "") for s in steps)
    for want in ("data/", "models/", "*.duckdb", "hu_pd_logs.tsv"):
        assert want in runs
    assert "git ls-files" in runs
    assert "sdk: docker" in runs
    assert "app_port: 7860" in runs
    assert "pinned: false" in runs
    assert "huggingface.co/spaces/${HF_SPACE}" in runs
    assert "HF_SPACE: ${{ vars.HF_SPACE }}" in text
    assert "git push --force" in runs
    assert "set -x" not in text
    for s in steps:
        lines = s.get("run", "").splitlines()
        assert not any(ln.strip().startswith("echo") and "HF_TOKEN" in ln for ln in lines)
    for ln in text.splitlines():
        if "HF_TOKEN" in ln and "secrets." not in ln:
            assert "${HF_TOKEN}" in ln or ln.strip().startswith("HF_TOKEN:")
    assert text.count("secrets.HF_TOKEN") == 1


def test_keepalive_workflow():
    wf, text = _workflow("keepalive.yml")
    triggers = wf[True]
    assert triggers["schedule"] == [{"cron": "0 */6 * * *"}]
    assert "workflow_dispatch" in triggers
    assert "curl -fsS --retry 5" in text
    assert "https://${{ vars.SPACE_URL }}/api/health" in text


def test_readme_hosting_section():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    head = "## เปิดบนเว็บตลอด 24 ชม. (Hugging Face Spaces)"
    assert head in text
    assert text.index("## ระบบล็อกอิน") < text.index(head) < text.index("## รันชุดทดสอบ")
    section = text.split(head, 1)[1].split("\n## ", 1)[0]
    for want in (
        "IDSAI_USERS",
        "HF_TOKEN",
        "HF_SPACE",
        "SPACE_URL",
        "Deploy to Hugging Face Space",
        "Run workflow",
        "hf.space",
        "SameSite",
        "รีสตาร์ต",
        "60 วัน",
    ):
        assert want in section
    for n in range(1, 8):
        assert f"\n{n}. " in section
    assert "ยังไม่มีหน้า Dashboard" not in text
    assert "API ไม่มีระบบล็อกอิน" not in text
