# Self sign-up, a user admin page, and accounts that survive restarts

Date: 2026-11-11. Status: the owner approved approach A ("อนุมัติ เขียน spec").

This builds on `2026-11-10-render-hosting-design.md`. Auth, sessions, roles, `seed-env` and the hosting stay as they are; this spec only adds to them.

## 1. Goal and success

Visitors to the hosted dashboard can make their own account from the login page, Facebook-style, and are signed in at once as `viewer`. The owner (admin) changes roles or disables accounts from a new admin page. Accounts outlive Render restarts and deploys.

**Owner decisions:**
- Accounts live in a free external Postgres (Neon). Losing them on every restart was rejected.
- A new account can sign in at once, read-only (`viewer`). There is no approval queue.
- Approach A: Postgres is the source of truth, and a local DuckDB copy serves every read.

**Success means:**
- A visitor can sign up, is signed in, and sees the dashboard as `viewer`.
- After a Render restart or deploy, that account still exists and can sign in. Sessions do not survive, so users sign in again.
- An admin can see every user, change roles and disable or enable accounts in the web UI. The last enabled admin cannot be demoted or disabled.
- With `DATABASE_URL` unset (local runs, CI), behaviour is exactly as today and sign-up is off unless `--allow-signup` is passed.
- Request latency does not change: no Postgres call on the read path.

**Out of scope:**
- Google or Facebook OAuth.
- Email verification.
- Password reset by email.
- Users changing their own password or display name.
- Deleting accounts; disabling is enough.

## 2. Design

### 2.1 Account mirror (`idsai/data/accounts.py`, new)

Introduce a small protocol `AccountBackend`:

| Method | Does |
|---|---|
| `ensure_schema()` | `CREATE TABLE IF NOT EXISTS` for `users` and `user_prefs`, with the same columns as DuckDB |
| `load_all() -> tuple[list[dict], list[dict]]` | Returns all user rows and all pref rows |
| `insert_user(row: dict)` | Insert; raises `ValueError` if the username is taken (unique constraint) |
| `update_user(username: str, **fields)` | Sets any of `role`, `pw_hash`, `disabled` |
| `set_prefs(user_id: str, json_text: str)` | Upsert |

There are two implementations:
- **`PostgresAccounts(url)`** uses `psycopg` (v3) with one connection opened lazily. On `OperationalError` it reconnects once and retries the statement. Neon suspends idle compute, so the first call after a while may take about a second. It sets a 10 s connect timeout.
- **`MemoryAccounts`** is for tests and keeps rows in dicts.

`build_backend(url: str | None)` returns `None` when the URL is empty.

### 2.2 Store write-through (`idsai/data/store.py`)

- `Store.attach_accounts(backend)`:
  1. calls `ensure_schema()` and `load_all()`;
  2. in one DuckDB transaction under the lock, deletes all rows of `users` and `user_prefs` and inserts the loaded rows;
  3. keeps the backend.
- When a backend is attached, the write methods (`add_user`, `set_role`, `set_disabled`, `set_password`, `set_prefs`) call the backend first and then DuckDB, both inside `self._lock`.
  - If the backend raises, DuckDB is not touched and the error propagates. The two copies therefore never diverge, because the process is the only writer: Render free runs one instance.
- Sessions, the audit log, alerts and the rest stay in DuckDB only.
- Reads (`get_user_by_name`, `get_session` with its join, `list_users`, `has_enabled_admin`) are unchanged.

### 2.3 Wiring the backend

- `idsai serve` and `idsai user …` (including `seed-env`) read `DATABASE_URL` from the environment. If it is set, they call `store.attach_accounts(PostgresAccounts(url))` right after opening the store.
- **At boot,** attaching tries 3 times, 5 s apart. If all fail, the process exits 1 with the Thai message `เชื่อมต่อฐานข้อมูลบัญชีไม่ได้ (DATABASE_URL)`, and Render restarts it. The URL is never printed.
- **`seed-env` is unchanged in meaning.** It creates or updates the accounts listed in `IDSAI_USERS`, and those writes now also go to Postgres. Accounts not in `IDSAI_USERS` (sign-ups) are never touched.
- **Env-listed accounts are reset at every boot.** If the admin changes an env-listed user's role in the UI, the next boot puts it back. The README says so.

### 2.4 Sign-up API (`idsai/product/api/app.py`)

**Flag:**
- `serve --allow-signup`, off by default.
- `deploy/start.sh` passes it.
- `/api/me` and a new public `GET /api/auth/config` return `{"signup": bool}` so the login page knows whether to show the link.

**`POST /api/auth/signup`:**
- Depends on `same_origin`.
- Returns 404 when sign-up is off and 409 when auth is off.
- Body: `{username, display_name, password}`.

**Checks:**
1. Run `check_username` (the same regex, 3–32 chars of `a-z0-9_.-`, lowercase). The names `cli` and `local` stay reserved. A bad name gets a 422 with Thai text.
2. Run `check_password` (at least 10 chars). The password must also differ from the username. Failure gets a 422.
3. `display_name` is trimmed to 1–40 chars, and control characters are rejected (422). If it is empty, the username is used.
4. **Rate limits**, in memory and reset on restart, both checked before hashing:
   - per client IP, at most 10 sign-ups per hour (raised from 3 after the final review: a class on one Wi-Fi shares one IP);
   - site-wide, at most 60 per hour.
   - A 409 (name taken) does not consume quota: the slot is refunded.
   - Over a limit, the response is 429 `สมัครถี่เกินไป ลองใหม่ภายหลัง`.
   - **Client IP** is the left-most `X-Forwarded-For` entry when `--secure-cookies` is on (hosted behind Render's proxy), else `request.client.host`. The per-IP limit can be dodged by spoofing the header; the site-wide cap is the hard bound.
5. Hash with scrypt off the event loop, using the same `verify_slots` semaphore as login.
6. Call `store.add_user(username, display_name, "viewer", hash, now)`.
   - A taken name gets 409 `ชื่อนี้มีคนใช้แล้ว`.
   - A backend failure gets 503 `บันทึกบัญชีไม่สำเร็จ ลองใหม่อีกครั้ง`.
7. Write an audit row `signup`, create a session (the remember flag is false) and set the cookie exactly as login does. The response has the same body shape as login.

### 2.5 User admin API

All of these depend on `Admin`; the POSTs also depend on `same_origin`.

**Endpoints:**
- `GET /api/users` returns `list_users()` minus `pw_hash`. It adds `from_env: bool`, true when the name is in the `IDSAI_USERS` names parsed at boot.
- `POST /api/users/{username}/role` with `{role}`, where `role` must be in `ROLES`.
- `POST /api/users/{username}/disabled` with `{disabled: bool}`.

**Guards:**
- An unknown user gets 404.
- Demoting or disabling the last enabled admin gets 409 `ต้องมีผู้ดูแลอย่างน้อย 1 คน`.
- An admin cannot disable themself (409).
- A backend failure gets 503.

**Audit actions:** `role_change` and `user_disable` / `user_enable`, each with the target username.

### 2.6 Web UI (`index.html`, `app.js`, `styles.css`)

**Login card:**
- Under the login button, add the line `ยังไม่มีบัญชี? สมัครสมาชิก`, shown only when `/api/auth/config` says sign-up is on.
- Clicking it swaps the card to a sign-up form in place, with no page change:
  - fields ชื่อผู้ใช้, ชื่อที่แสดง, รหัสผ่าน, ยืนยันรหัสผ่าน;
  - a button สมัครสมาชิก;
  - a link `มีบัญชีแล้ว? เข้าสู่ระบบ` back to login.
- Client-side checks mirror the server: the name pattern, at least 10 chars, the two passwords match. The hint text is Thai.
- On success, the app loads just as after login.

**New admin-only page "ผู้ใช้" in the nav:**
- It is hidden for other roles, and the server enforces the same rule.
- It shows a table: ชื่อผู้ใช้, ชื่อที่แสดง, สิทธิ์ (a select), สถานะ (a toggle), ใช้งานล่าสุด, and a small `จาก IDSAI_USERS` tag where `from_env` is true.
- Changing a value calls the API and shows a toast. On error, it reverts and shows the server's Thai message.

The glass styling, wording and copy guards follow dashboard v3. There is no new library.

### 2.7 Config, deploy, docs

- `pyproject.toml` gains `psycopg[binary]` in the main dependencies, because the runtime image needs it. Regenerate `uv.lock`.
- `render.yaml` gains env var `DATABASE_URL` with `sync: false`.
- `deploy/start.sh` adds `--allow-signup` to the serve line.
- **README hosting gets a new step after `IDSAI_USERS`:**
  1. sign up at neon.tech with GitHub, which needs no card;
  2. create a project in region AWS US West (Oregon), the same region as Render's default;
  3. copy the connection string;
  4. in Render → Environment, add `DATABASE_URL` and save.

  Also note that sign-up is open to anyone with the link and that new accounts are `viewer`.

## 3. Error handling summary

| Case | Result |
|---|---|
| Postgres down at boot | 3 tries, then exit 1 with a Thai message; Render restarts the service |
| Postgres down at write | 503 with Thai text; DuckDB unchanged |
| Name taken | 409 |
| Bad name, bad password or bad display name | 422 |
| Too many sign-ups | 429 |
| Sign-up off | 404 |
| Last admin protected | 409 |

In every case, secrets are never logged: not the URL, not passwords, not hashes.

## 4. Testing

**Unit tests with `MemoryAccounts`:**
- attach loads rows;
- each write goes to the backend and DuckDB;
- a backend error leaves DuckDB unchanged;
- `seed-env` writes through and leaves sign-ups alone.

**API tests:**
- sign-up success sets the cookie, makes the role `viewer` and writes an audit row;
- the 404 when off, the 409 for a taken name, the 422 cases, the 429 for per-IP and site-wide limits, and the 503 on a backend failure;
- the admin endpoints, with role enforcement, the last-admin and self-disable guards, and the `from_env` flag;
- `/api/auth/config`.

**Other tests:**
- **Boot:** with `DATABASE_URL` set to an unreachable address, serve exits 1 with the Thai message and does not print the URL. Retries are shortened in tests.
- **Real Postgres:** a test marked `network` runs only when `IDSAI_TEST_PG_URL` is set. It runs ensure_schema, insert, update and load_all against a real server, inside a temp schema that is dropped afterwards.
- **Web:** static tests check that the sign-up form, the toggle link, the users page and its admin-only nav gating exist. The existing copy-guard tests pass.
- **Deploy files:** `render.yaml` has `DATABASE_URL` with `sync: false`, `start.sh` has `--allow-signup`, and the README mentions neon.tech and `DATABASE_URL`.

**Gates:** `uv run ruff check .` and `uv run pytest -q -m "not network and not slow"` pass.
