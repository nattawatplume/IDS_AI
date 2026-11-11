# Self Sign-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let visitors sign up as `viewer` from the login page, add an admin "ผู้ใช้" page, and keep accounts in Neon Postgres so they survive Render restarts.

**Architecture:**
- An `AccountBackend` (Postgres, or in-memory for tests) is the source of truth for `users` and `user_prefs`.
- `Store` loads it into DuckDB at boot and writes through it on every account write. Reads stay local.
- New API routes for sign-up and user admin, plus a sign-up form and a users page in the vanilla JS dashboard.

**Tech Stack:** Python 3.13, FastAPI, DuckDB, psycopg 3 (`psycopg[binary]`), vanilla JS, pytest.

**Spec:** `docs/superpowers/specs/2026-11-11-self-signup-design.md`

## Global Constraints

- With `DATABASE_URL` unset, behaviour is exactly as today. Sign-up is off unless `--allow-signup` is passed.
- There is no Postgres call on the read path (`get_user_by_name`, `get_session`, `list_users`, `has_enabled_admin`).
- Sign-up role is always `viewer`.
- Rate limits: per IP 10 sign-ups/hour, site-wide 60/hour, in memory; a 409 name-taken does not consume quota.
- Name rules come from `check_username` (`^[a-z0-9_.-]{3,32}$`; `cli` and `local` are reserved). Password rules come from `check_password` (at least 10 chars) and the password must differ from the username. `display_name` is trimmed to 1–40 chars, with no control chars, and an empty one becomes the username.
- Thai messages, verbatim from the spec:
  - `ชื่อนี้มีคนใช้แล้ว` (409)
  - `สมัครถี่เกินไป ลองใหม่ภายหลัง` (429)
  - `บันทึกบัญชีไม่สำเร็จ ลองใหม่อีกครั้ง` (503)
  - `ต้องมีผู้ดูแลอย่างน้อย 1 คน` (409)
  - `เชื่อมต่อฐานข้อมูลบัญชีไม่ได้ (DATABASE_URL)` (boot exit 1)
- Never log or print `DATABASE_URL`, passwords or hashes.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Never `git add -f`; never commit `.claude/`, `data/` or `models/`. No long `--basetemp`. Tests never touch the real `data/app.duckdb`. Stop any local `idsai serve` before running tests.
- Gates before each commit: `uv run ruff check .` and `uv run pytest -q -m "not network and not slow"`.

## Review Focus

1. **Two requests take the same new name at once.**
   - Expected: exactly one 200 and one 409, with no 500 and no half-written row.
   - Test: Task 3, `test_signup_same_name_twice_is_409`. The unique constraint in the backend wins, and `add_user` holds the store lock across the backend call and the DuckDB insert.
2. **The Postgres write fails after the backend's insert succeeded but before DuckDB's.**
   - Expected: impossible, because DuckDB's insert runs only after the backend returns, and the lock serializes them. A DuckDB failure after a backend success then leaves Postgres ahead, and the next boot reloads from Postgres.
   - Test: Task 1, `test_backend_error_leaves_duckdb_unchanged`.
3. **An admin demotes themself while they are the only admin.**
   - Expected: 409 `ต้องมีผู้ดูแลอย่างน้อย 1 คน`.
   - Test: Task 4, `test_last_admin_cannot_be_demoted_or_disabled`.
4. **A sign-up name that differs only in case, for example `Somchai`.**
   - Expected: 422 because the regex is lowercase-only, never a second account.
   - Test: Task 3, `test_signup_rejects_bad_input` includes `"Somchai"`.
5. **A user disabled in the admin page.**
   - Expected: their open sessions stop working at once, because `get_session` filters `NOT u.disabled`.
   - Test: Task 4, `test_disable_kills_open_session`.

---

### Task 1: Account backend and Store write-through

**Files:**
- Create: `idsai/data/accounts.py`.
- Modify: `idsai/data/store.py`, specifically the user and pref write methods (around lines 474–592).
- Test: `tests/data/test_accounts.py`.

**Interfaces:**
- Produces, in `idsai/data/accounts.py`:
  - `class AccountStoreError(RuntimeError)`.
  - `class AccountBackend(Protocol)` with:
    - `ensure_schema() -> None`;
    - `load_all() -> tuple[list[dict], list[dict]]`. The user dicts have keys `user_id, username, display_name, role, pw_hash, created_us, disabled`; the pref dicts have keys `user_id, json`;
    - `insert_user(row: dict) -> None`, which raises `ValueError` if the name is taken and `AccountStoreError` on any other failure;
    - `update_user(username: str, **fields) -> None`, where `fields` is a subset of `role`, `pw_hash`, `disabled`;
    - `set_prefs(user_id: str, json_text: str) -> None`.
  - `class MemoryAccounts(fail: bool = False)`, which implements it with dicts and has public attributes `users`, `prefs` and `fail`. When `fail` is True, every call raises `AccountStoreError`.
  - `class PostgresAccounts(url: str)` (see Step 3).
  - `build_backend(url: str | None) -> AccountBackend | None`, which returns None for None or `""`.
- Produces, in `Store`: `attach_accounts(backend: AccountBackend) -> None`.

- [ ] **Step 1: Write the failing tests** in `tests/data/test_accounts.py`. Use a tmp-path `Store`.
  - `test_attach_loads_rows_and_replaces_local`:
    - add a local user `old` without a backend;
    - make a `MemoryAccounts` holding user `neo` (admin) and a pref row for it;
    - call `attach_accounts`;
    - assert that `get_user_by_name("old") is None`, that `get_user_by_name("neo")["role"] == "admin"` and that `get_prefs(neo_id) == {...}`.
  - `test_writes_go_to_backend_and_duckdb`: after attaching, call `add_user`, `set_role`, `set_disabled`, `set_password` and `set_prefs`. Assert each change is visible both in `backend.users` / `backend.prefs` and through the Store getters.
  - `test_backend_error_leaves_duckdb_unchanged`:
    - set `backend.fail = True`;
    - assert that `add_user` raises `AccountStoreError` and that the user is absent locally;
    - assert that `set_role` raises and the role is unchanged.
  - `test_taken_name_raises_value_error_from_backend`: put the name in the backend only, bypassing DuckDB. `add_user` raises `ValueError`.
  - `test_no_backend_is_unchanged`: the existing store auth tests already cover this. Add one assertion that `Store._accounts is None` by default.
- [ ] **Step 2: Run** `uv run pytest -q tests/data/test_accounts.py`. Expected: FAIL (import error).
- [ ] **Step 3: Implement.**
  - **`accounts.py`:**
    - Write `MemoryAccounts`.
    - `PostgresAccounts`:
      - imports `psycopg` lazily, inside methods;
      - opens one connection with `connect_timeout=10` and `autocommit=True`;
      - on `psycopg.OperationalError`, closes the connection, reconnects once and retries;
      - maps `psycopg.errors.UniqueViolation` to `ValueError` and any other `psycopg.Error` to `AccountStoreError`, with a message that never includes the URL.
    - The DDL uses the same column names as DuckDB: `user_id TEXT PRIMARY KEY, username TEXT UNIQUE NOT NULL, display_name TEXT, role TEXT, pw_hash TEXT, created_us BIGINT, disabled BOOLEAN`, and `user_prefs(user_id TEXT PRIMARY KEY, json TEXT)`.
  - **`store.py`:**
    - Add `self._accounts = None`.
    - `attach_accounts` runs `ensure_schema`, then `load_all`, then, under `self._lock`, a DuckDB `BEGIN` / `DELETE FROM users` / `DELETE FROM user_prefs` / inserts / `COMMIT`. Then it sets `self._accounts`.
    - In each write method, inside the existing lock and before the DuckDB statement, call the backend when `self._accounts` is not None. `add_user` builds the full row, including the new `user_id` and `disabled=False`, and passes it to `insert_user`. `set_password` passes `pw_hash`.
- [ ] **Step 4: Run** `uv run pytest -q tests/data` and `uv run ruff check .`. Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat(data): account backend with write-through mirror into DuckDB`.

### Task 2: Wire DATABASE_URL into serve and the user commands; psycopg dependency

**Files:**
- Modify: `pyproject.toml` and `uv.lock` (`psycopg[binary]` in `[project] dependencies`).
- Modify: `idsai/cli.py`, specifically `_app_store` near line 104, `cmd_serve` near line 1137 and the `user` command block near lines 1040–1110.
- Test: `tests/test_cli_accounts.py`, and `tests/data/test_accounts_pg.py` (marked network).

**Interfaces:**
- Consumes: `build_backend`, `Store.attach_accounts` and `AccountStoreError` from Task 1.
- Produces, in `idsai/cli.py`:
  - `ATTACH_TRIES = 3` and `ATTACH_WAIT_S = 5.0` as module constants;
  - `attach_accounts_from_env(store: Store) -> None`. It reads `os.environ.get("DATABASE_URL")`; if set, it builds the backend and attaches it, retrying `ATTACH_TRIES` times with `time.sleep(ATTACH_WAIT_S)` between tries. If every try fails, it raises `CliError("เชื่อมต่อฐานข้อมูลบัญชีไม่ได้ (DATABASE_URL)")`.
  - The function also accepts an optional `backend_factory` parameter, defaulting to `build_backend`, which tests use.

- [ ] **Step 1: Write the failing tests** in `tests/test_cli_accounts.py`:
  - `test_no_database_url_attaches_nothing`, with `monkeypatch.delenv`.
  - `test_attach_retries_then_cli_error`:
    - set the env var to `postgresql://u:secret@127.0.0.1:1/x`;
    - set `ATTACH_WAIT_S` to 0;
    - use a factory whose backend's `ensure_schema` always raises `AccountStoreError`;
    - assert a `CliError` with the Thai text, that the factory was called 3 times, and that `"secret"` does not appear in `str(err)`.
  - `test_seed_env_writes_through_and_leaves_signups`:
    - attach a `MemoryAccounts` that already holds `walkin` (viewer);
    - run the `seed-env` code path with `IDSAI_USERS="boss:admin:<10+ chars>"`;
    - assert that `boss` is in `backend.users` and that `walkin` is unchanged.
  - `test_serve_and_user_commands_attach`: monkeypatch `attach_accounts_from_env` to record calls. Assert that both `cmd_serve` (stop before `run_server`, as the existing serve tests do) and `cmd_user` call it.
- [ ] **Step 2: Write the network test** `tests/data/test_accounts_pg.py`, marked `@pytest.mark.network`. It skips unless `IDSAI_TEST_PG_URL` is set.
  - It creates a schema named `idsai_test_<uuid>`, using `search_path` through the connection's `options`.
  - It runs `ensure_schema`, `insert_user`, a duplicate (expecting `ValueError`), `update_user`, `set_prefs` and `load_all` round-trips.
  - It drops the schema in a finally block.
- [ ] **Step 3: Run** `uv run pytest -q tests/test_cli_accounts.py`. Expected: FAIL.
- [ ] **Step 4: Implement.**
  - Add `psycopg[binary]` and run `uv lock`. Check that `git diff uv.lock` only adds psycopg and its wheels and moves no other version; report it if anything else moved.
  - Write `attach_accounts_from_env`.
  - Call it right after `_app_store(...)` opens in `cmd_serve` (before `check_serve_auth`) and in the `user` command (before any action).
- [ ] **Step 5: Run** the full gate. Expected: PASS.
- [ ] **Step 6: Commit** with the message `feat(cli): load accounts from DATABASE_URL (Neon) at boot; seed-env writes through`.

### Task 3: Sign-up API

**Files:**
- Modify: `idsai/product/auth.py` (add `SignupLimiter`), `idsai/product/api/state.py` (new fields), `idsai/product/api/app.py` (routes) and `idsai/cli.py` (the `--allow-signup` flag and the `env_usernames` field).
- Test: `tests/product/test_api_signup.py`, reusing the `aenv` / `auth_client` patterns from `tests/product/test_api_auth.py`.

**Interfaces:**
- Consumes: `AccountStoreError` (Task 1) and `parse_seed_users` (existing).
- Produces:
  - **`auth.SignupLimiter(per_ip: int = 3, total: int = 30, window_s: float = 3600.0)`**:
    - `allow(ip: str, now_s: float) -> bool` records an attempt only when it allows it;
    - it keeps sliding windows (deques of timestamps).
  - **`AppState` new fields:**
    - `allow_signup: bool = False`;
    - `signup_limiter: SignupLimiter = field(default_factory=SignupLimiter)`;
    - `env_usernames: frozenset[str] = frozenset()`.
  - **Routes:**
    - `GET /api/auth/config` is public and returns `{"signup": state.auth_enabled and state.allow_signup}`. `/api/me` gains the same `signup` key.
    - `POST /api/auth/signup` depends on `same_origin`, takes the body `SignupIn(username, display_name="", password)` and returns the same body and cookie as login.
  - **`serve --allow-signup`** is a `store_true` flag. `cmd_serve` sets `state.allow_signup` from it, and sets `state.env_usernames` from the names in `parse_seed_users(os.environ["IDSAI_USERS"])` when that variable is set.

- [ ] **Step 1: Write the failing tests:**
  - `test_signup_limiter_windows`: per IP, the 4th call within 3600 s is False and a call after the window passes is True. Site-wide, the 31st call from distinct IPs is False.
  - `test_config_reports_signup`: the result is False by default and True with `allow_signup=True`.
  - `test_signup_off_is_404`.
  - `test_signup_success_signs_in_as_viewer`:
    - the response is 200 and sets the session cookie;
    - `GET /api/me` shows role `viewer`;
    - `display_name` defaults to the username;
    - the audit has a `signup` row.
  - `test_signup_same_name_twice_is_409`: the second call returns 409 with `ชื่อนี้มีคนใช้แล้ว`.
  - `test_signup_rejects_bad_input`: each of these gets a 422:
    - `"Somchai"`, `"ab"`, `"local"`;
    - a password of 9 chars;
    - a password equal to the username;
    - a display name of 41 chars;
    - a display name containing `"\x07"`.
  - `test_signup_rate_limited_per_ip_and_site`: use a small limiter (per_ip=1, total=2) injected into the state. The 2nd call from the same IP returns 429 with the Thai text, and calls from other IPs hit the site cap. For IP selection, with `cookie_secure` set, the IP comes from the `X-Forwarded-For` header's left-most entry.
  - `test_signup_backend_failure_is_503`: attach `MemoryAccounts(fail=True)`; the call returns 503 with the Thai text and creates no session.
  - `test_signup_refuses_cross_origin`, mirroring the login test.
- [ ] **Step 2: Run** `uv run pytest -q tests/product/test_api_signup.py`. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - **Check order:** auth enabled → signup allowed → body checks (422) → limiter (429) → scrypt via `asyncio.to_thread` inside `verify_slots` → `store.add_user`. `ValueError` becomes 409 and `AccountStoreError` becomes 503.
  - **Session:** create the audit row and the session with the same code path as login. Factor login's session-and-cookie tail into a local helper `_start_session(user, remember)` used by both routes.
- [ ] **Step 4: Run** the full gate. Expected: PASS, and the existing login tests stay unchanged.
- [ ] **Step 5: Commit** with the message `feat(api): self sign-up as viewer with per-IP and site-wide limits`.

### Task 4: User admin API

**Files:**
- Modify: `idsai/product/api/app.py`.
- Test: `tests/product/test_api_users.py`.

**Interfaces:**
- Consumes: `AppState.env_usernames` (Task 3) and the `Admin` dependency (existing).
- Produces:
  - `GET /api/users` returns a list of `{username, display_name, role, disabled, created_us, last_seen_us, from_env}` with no `user_id` and no `pw_hash`.
  - `POST /api/users/{username}/role` takes the body `{role}`, and `POST /api/users/{username}/disabled` takes the body `{disabled: bool}`. Both depend on `same_origin` and return the updated row in the same shape.

- [ ] **Step 1: Write the failing tests:**
  - `test_users_admin_only`: a viewer and an analyst get 403, and an admin gets 200 with no `pw_hash` key.
  - `test_from_env_flag`: set `env_usernames={"admin"}`. Only the env user is flagged.
  - `test_change_role_and_audit`: promote a viewer to analyst, then `GET /api/users` shows it, and the audit has `role_change`. A bad role gets 422 and an unknown user gets 404.
  - `test_last_admin_cannot_be_demoted_or_disabled`: both calls return 409 with `ต้องมีผู้ดูแลอย่างน้อย 1 คน`.
  - `test_admin_cannot_disable_self`: returns 409.
  - `test_disable_kills_open_session`:
    - the viewer's client is signed in;
    - the admin disables the viewer;
    - the viewer's next `GET /api/me` returns 401;
    - the audit has `user_disable`;
    - re-enabling writes `user_enable`.
  - `test_admin_write_backend_failure_is_503`.
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement the guards.** Count enabled admins from `list_users()`. The change is refused when it would leave zero enabled admins.
- [ ] **Step 4: Run** the full gate. Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat(api): admin user list, role change, disable/enable with last-admin guard`.

### Task 5: Web UI (sign-up form and users page)

**Files:**
- Modify: `idsai/product/web/index.html`, `idsai/product/web/app.js` and `idsai/product/web/styles.css`.
- Test: `tests/product/test_web_static.py`.

**Interfaces:**
- Consumes: `/api/auth/config`, `/api/auth/signup` (Task 3) and `/api/users*` (Task 4).

- [ ] **Step 1: Write the failing static tests:**
  - `test_signup_form_present`:
    - `index.html` has a `form#signup-form` containing inputs `#signup-user`, `#signup-name`, `#signup-pass` and `#signup-pass2`;
    - it has the link texts `สมัครสมาชิก` and `มีบัญชีแล้ว? เข้าสู่ระบบ`;
    - the sign-up form starts `hidden`.
  - `test_signup_link_gated_by_config`: `app.js` fetches `/api/auth/config` and shows `#signup-open` only when `signup` is true.
  - `test_users_page_admin_only`:
    - there is a nav item for page `users` with label `ผู้ใช้`;
    - `app.js` hides it unless the role is `admin`;
    - the page calls `/api/users`.
  - `test_signup_client_checks`: `app.js` contains the pattern `[a-z0-9_.-]{3,32}`, the min length 10 and a password-match check, with Thai hints.
  - The existing copy-guard and brace-balance tests still pass.
- [ ] **Step 2: Run** `uv run pytest -q tests/product/test_web_static.py`. Expected: FAIL.
- [ ] **Step 3: Implement the sign-up form.**
  - The card swap is in place: one of `#login-form` or `#signup-form` is visible at a time, and focus moves to the first field.
  - On success, run the same post-login boot as the login form.
- [ ] **Step 4: Implement the users page.**
  - It shows a table with a role `<select>`, a status toggle, last seen in Thai time "น." and the tag `จาก IDSAI_USERS`.
  - Each change sends a POST and shows a toast; on an error it reverts and shows the server's `detail`.
  - Use the glass styles from dashboard v3 and add no library.
- [ ] **Step 5: Check by hand.**
  - Run `uv run idsai serve --mode replay --auth --allow-signup` locally with a temp DB.
  - Sign up, sign in as an admin and change a role.
  - Stop the server.
- [ ] **Step 6: Run** the full gate. Expected: PASS.
- [ ] **Step 7: Commit** with the message `feat(web): sign-up form on the login card and an admin users page`.

### Task 6: Deploy files and README

**Files:**
- Modify: `render.yaml`, `deploy/start.sh` and `README.md` (the hosting section).
- Test: `tests/test_deploy_files.py`.

- [ ] **Step 1: Write the failing tests:**
  - `render.yaml` has an env var `DATABASE_URL` with `sync: false` and no value.
  - The serve line in `start.sh` contains `--allow-signup`.
  - The README hosting section mentions `neon.tech`, `DATABASE_URL` and `Oregon`, and says that new accounts are `viewer`.
  - The existing deploy tests still pass.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - Add a README step right after the `IDSAI_USERS` step, written as plain Thai click-by-click instructions:
    1. sign up at neon.tech with GitHub, which needs no card;
    2. create a project in AWS US West (Oregon);
    3. copy the connection string;
    4. in Render → Environment, add `DATABASE_URL` and save.
  - Add these notes:
    - sign-up is open to anyone with the link;
    - accounts listed in `IDSAI_USERS` are reset at every boot;
    - a restart signs everyone out but keeps the accounts.
- [ ] **Step 4: Run** the full gate. Expected: PASS.
- [ ] **Step 5: Commit** with the message `docs(deploy): DATABASE_URL (Neon) and --allow-signup on Render`.
