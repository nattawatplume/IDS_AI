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


def test_torch_is_training_only():
    import tomllib

    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert not any(d.lower().startswith("torch") for d in cfg["project"]["dependencies"])
    assert any(d.lower().startswith("torch") for d in cfg["dependency-groups"]["train"])
    assert "train" in cfg["tool"]["uv"]["default-groups"]
    assert "dev" in cfg["tool"]["uv"]["default-groups"]


def test_start_script_render():
    raw = (ROOT / "deploy" / "start.sh").read_bytes()
    assert b"\r" not in raw
    text = raw.decode("utf-8")
    for want in (
        "idsai user seed-env",
        "RENDER_EXTERNAL_HOSTNAME",
        "${PORT:-7860}",
        "${IDSAI_SPEED:-20}",
        '--speed "$SPEED"',
        '--port "$PORT"',
        "--auth",
        "--secure-cookies",
        "--loop",
        '--public-host "$PUBLIC_HOST"',
    ):
        assert want in text
    sh = shutil.which("sh")
    if sh is None:
        pytest.skip("no POSIX sh available")
    env = {
        k: v for k, v in os.environ.items() if k not in ("SPACE_HOST", "RENDER_EXTERNAL_HOSTNAME")
    }
    r = subprocess.run(
        [sh, str(ROOT / "deploy" / "start.sh")],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert r.returncode == 1
    assert "RENDER_EXTERNAL_HOSTNAME" in r.stderr
    assert any("฀" <= c <= "๿" for c in r.stderr)


def test_dockerfile_two_stages():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    lines = text.splitlines()
    froms = [i for i, ln in enumerate(lines) if ln.startswith("FROM ")]
    assert len(froms) == 2
    assert " AS build" in lines[froms[0]]
    build, runtime = "\n".join(lines[: froms[1]]), "\n".join(lines[froms[1] :])
    assert "deploy/check_no_hu.py" in build
    assert "idsai demo-pack" in build
    assert "--no-group train" not in build
    assert "--no-group train" in runtime
    assert "COPY --from=build --chown=user /home/user/app/data" in runtime
    assert "torch" not in runtime
    for stage in (build, runtime):
        assert "useradd -m -u 1000 user" in stage
        assert "mkdir -p /home/user/app && chown user:user /home/user/app" in stage
        assert stage.index("chown user:user /home/user/app") < stage.index("USER user")
        assert "uv sync --frozen --no-dev" in stage
    assert "EXPOSE 7860" in runtime
    assert "deploy/start.sh" in runtime


def _workflow(name: str) -> tuple[dict, str]:
    import yaml

    text = (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
    return yaml.safe_load(text), text


def test_deploy_space_workflow():
    wf, text = _workflow("deploy-space.yml")
    triggers = wf[True]  # PyYAML reads the key `on` as True
    assert triggers["push"]["branches"] == ["main", "feat/mvp"]
    assert wf["permissions"] == {"contents": "read"}
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
            assert "HF_TOKEN}" in ln or "$HF_TOKEN" in ln or ln.strip().startswith("HF_TOKEN:")
    assert text.count("secrets.HF_TOKEN") == 1


def test_keepalive_workflow():
    wf, text = _workflow("keepalive.yml")
    triggers = wf[True]
    assert triggers["schedule"] == [{"cron": "0 */6 * * *"}]
    assert "workflow_dispatch" in triggers
    assert wf["permissions"] == {"contents": "read"}
    assert "SPACE_URL: ${{ vars.SPACE_URL }}" in text
    cmd = 'curl -fsS --retry 5 --retry-delay 30 --max-time 120 "https://$SPACE_URL/api/health"'
    assert cmd in text


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
        "feat/mvp",
        "hf.space",
        "SameSite",
        "รีสตาร์ต",
        "60 วัน",
    ):
        assert want in section
    # GitHub runs `schedule` only from the default branch: the wake-up waits for the merge
    assert "ตัวปลุกทุก 6 ชม. ก็เหมือนกัน จะเริ่มทำงานหลังรวม PR เข้า `main`" in section
    assert "48 ชม." in section
    for n in range(1, 8):
        assert f"\n{n}. " in section
    assert "ยังไม่มีหน้า Dashboard" not in text
    assert "API ไม่มีระบบล็อกอิน" not in text
