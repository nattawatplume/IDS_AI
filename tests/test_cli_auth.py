"""The `user` CLI and the serve guard for public binds."""

import io

import pytest

from idsai import cli
from idsai.cli import CliError, check_serve_auth
from idsai.config import load_settings
from idsai.data.store import Store
from tests.test_cli import write_cfg

PW = "correct horse battery"


@pytest.fixture
def cfg_path(tmp_path):
    return write_cfg(tmp_path)


def run(cfg_path, *argv, stdin: str | None = None, monkeypatch=None):
    if stdin is not None:
        monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    return cli.main(["--config", str(cfg_path), *argv])


def open_store(cfg_path):
    s = load_settings(cfg_path)
    return Store(s.paths.data_dir, s.paths.app_db)


def test_user_add_list_role_disable_enable(cfg_path, monkeypatch, capsys):
    assert run(cfg_path, "user", "add", "alice", "--role", "viewer", "--display-name", "Alice",
               "--password-stdin", stdin=PW + "\n", monkeypatch=monkeypatch) == 0
    assert run(cfg_path, "user", "role", "alice", "analyst") == 0
    assert run(cfg_path, "user", "disable", "alice") == 0
    capsys.readouterr()
    assert run(cfg_path, "user", "list") == 0
    out = capsys.readouterr().out
    assert "alice" in out and "analyst" in out and "disabled" in out
    assert "scrypt$" not in out
    assert run(cfg_path, "user", "enable", "alice") == 0
    assert run(cfg_path, "user", "list") == 0
    out = capsys.readouterr().out
    assert "alice" in out and "disabled" not in out and "scrypt$" not in out
    assert run(cfg_path, "user", "passwd", "alice", "--password-stdin",
               stdin="another long password\n", monkeypatch=monkeypatch) == 0


def test_user_add_rejects_bad_username(cfg_path, monkeypatch, capsys):
    code = run(cfg_path, "user", "add", " Alice ", "--password-stdin",
               stdin=PW + "\n", monkeypatch=monkeypatch)
    assert code == 1
    assert "ชื่อผู้ใช้" in capsys.readouterr().err
    with open_store(cfg_path) as st:
        assert st.list_users() == []


def test_user_add_rejects_short_password_and_duplicate(cfg_path, monkeypatch, capsys):
    assert run(cfg_path, "user", "add", "bob", "--password-stdin",
               stdin="short\n", monkeypatch=monkeypatch) == 1
    assert run(cfg_path, "user", "add", "bob", "--password-stdin",
               stdin=PW + "\n", monkeypatch=monkeypatch) == 0
    assert run(cfg_path, "user", "add", "bob", "--password-stdin",
               stdin=PW + "\n", monkeypatch=monkeypatch) == 1


def test_unknown_user_gives_an_error(cfg_path, capsys):
    for argv in (("role", "ghost", "admin"), ("disable", "ghost"), ("enable", "ghost")):
        assert run(cfg_path, "user", *argv) == 1
        assert "ghost" in capsys.readouterr().err


def test_getpass_mismatch_is_an_error(cfg_path, monkeypatch, capsys):
    answers = iter([PW, PW + "x"])
    monkeypatch.setattr("getpass.getpass", lambda *_a, **_k: next(answers))
    assert run(cfg_path, "user", "add", "carol") == 1
    with open_store(cfg_path) as st:
        assert st.list_users() == []


def test_password_never_accepted_as_argument():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["user", "add", "alice", "--password", "x" * 12])
    with pytest.raises(SystemExit):
        parser.parse_args(["user", "passwd", "alice", "--password", "x" * 12])


def test_user_commands_write_audit_as_cli(cfg_path, monkeypatch):
    run(cfg_path, "user", "add", "dave", "--password-stdin",
        stdin=PW + "\n", monkeypatch=monkeypatch)
    run(cfg_path, "user", "role", "dave", "admin")
    run(cfg_path, "user", "disable", "dave")
    run(cfg_path, "user", "enable", "dave")
    run(cfg_path, "user", "passwd", "dave", "--password-stdin",
        stdin="a brand new password\n", monkeypatch=monkeypatch)
    with open_store(cfg_path) as st:
        rows = sorted(st.list_audit(), key=lambda r: r["ts_us"])
    assert [r["action"] for r in rows] == [
        "user_add", "user_role", "user_disable", "user_enable", "user_passwd"
    ]
    assert {r["username"] for r in rows} == {"cli"} and {r["target"] for r in rows} == {"dave"}


def test_serve_refuses_public_bind_without_auth(cfg_path, capsys):
    assert run(cfg_path, "serve", "--host", "0.0.0.0", "--no-auth") == 1
    assert "ล็อกอิน" in capsys.readouterr().err


def test_serve_refuses_auth_without_admin(cfg_path, capsys):
    assert run(cfg_path, "serve", "--auth") == 1
    assert "uv run idsai user add <ชื่อ> --role admin" in capsys.readouterr().err


def test_serve_loopback_without_auth_still_allowed(tmp_path):
    cfg = load_settings(write_cfg(tmp_path))
    with Store(cfg.paths.data_dir, cfg.paths.app_db) as st:
        check_serve_auth("127.0.0.1", False, st)  # no exception
        with pytest.raises(CliError):
            check_serve_auth("0.0.0.0", False, st)
        with pytest.raises(CliError):
            check_serve_auth("127.0.0.1", True, st)  # auth on, no admin
        st.add_user("root", "", "admin", "x", 1)
        check_serve_auth("0.0.0.0", True, st)


def test_session_purge_runs_hourly_and_outlives_the_pipeline(monkeypatch):
    import asyncio

    monkeypatch.setattr(cli, "SESSION_PURGE_S", 0.01)
    calls: list[int] = []
    store = type("S", (), {"purge_sessions": lambda self, now: calls.append(now) or 0})()
    ran = []

    async def pipeline() -> None:
        ran.append(1)  # ends at once, as a finished replay does

    async def main() -> None:
        task = asyncio.create_task(cli.with_session_purge(store, pipeline)())
        await asyncio.sleep(0.1)
        assert not task.done()  # still purging after the pipeline ended
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(main())
    assert ran == [1] and len(calls) >= 2


def test_auth_flag_overrides_config():
    parser = cli.build_parser()
    on = load_settings(overrides={"auth": {"enabled": True}})
    off = load_settings()
    assert cli.auth_enabled(parser.parse_args(["serve"]), on) is True
    assert cli.auth_enabled(parser.parse_args(["serve", "--no-auth"]), on) is False
    assert cli.auth_enabled(parser.parse_args(["serve"]), off) is False
    assert cli.auth_enabled(parser.parse_args(["serve", "--auth"]), off) is True


def test_user_list_shows_created_and_last_seen(cfg_path, monkeypatch, capsys):
    run(cfg_path, "user", "add", "erin", "--password-stdin", stdin=PW + "\n", monkeypatch=monkeypatch)
    capsys.readouterr()
    assert run(cfg_path, "user", "list") == 0
    out = capsys.readouterr().out
    assert "สร้างเมื่อ" in out and "เข้าล่าสุด -" in out and "UTC" in out
    with open_store(cfg_path) as st:
        uid = st.get_user_by_name("erin")["user_id"]
        st.create_session(uid, "tok", 1_800_000_000 * 10**6, 2_000_000_000 * 10**6, False)
    assert run(cfg_path, "user", "list") == 0
    out = capsys.readouterr().out
    assert "เข้าล่าสุด -" not in out and "2027-01-15" in out
    assert "scrypt$" not in out and "pw_hash" not in out


@pytest.mark.parametrize("name", ["cli", "local"])
def test_user_add_rejects_reserved_names(cfg_path, monkeypatch, capsys, name):
    assert run(cfg_path, "user", "add", name, "--password-stdin",
               stdin=PW + "\n", monkeypatch=monkeypatch) == 1
    assert "สงวน" in capsys.readouterr().err
    with open_store(cfg_path) as st:
        assert st.list_users() == []


def test_user_command_while_another_process_holds_the_db(cfg_path, tmp_path, capsys):
    import subprocess
    import sys

    s = load_settings(cfg_path)
    with open_store(cfg_path):
        pass  # create the file and its tables first
    code = (
        "import duckdb,sys,time;c=duckdb.connect(sys.argv[1]);print('ready',flush=True);"
        "time.sleep(60)"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", code, str(s.paths.app_db)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout.readline().strip() == "ready"
        assert run(cfg_path, "user", "list") == 1
        err = capsys.readouterr().err
        assert "หยุด serve" in err and "Traceback" not in err
    finally:
        holder.kill()
        holder.wait()
