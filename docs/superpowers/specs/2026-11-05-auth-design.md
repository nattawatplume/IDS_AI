# Sign-in, roles and per-user memory

Date: 2026-11-05 · Status: approved in chat, spec awaiting review
Depends on: `2026-11-05-dashboard-redesign-design.md` (the sign-in page and the Activity page use its look). Comes before hosting.

## 1. Why

Today the dashboard is meant for one person on 127.0.0.1 with no sign-in. To host it for teachers and the committee, every request must belong to a known user. Only some users may change state: TP/FP verdicts, replay control and Red-Team injects. The system should also remember each user (login, settings, actions).

Success means:
- Accounts are created by an admin only.
- There are 3 roles, enforced on every HTTP route and on the WebSocket.
- "Remember me" lasts 30 days.
- Settings follow the user to any browser.
- An admin can see who did what and when.
- A local run on 127.0.0.1 still works exactly as today, with auth off by default.
- Any other bind refuses to start with auth off.

## 2. Scope

**In scope:**
- A new `idsai/product/auth.py` with hashing, sessions, roles, rate limit and audit helpers.
- Store tables and methods in `idsai/data/store.py`.
- Routes and dependencies in `idsai/product/api/app.py`.
- `AuthCfg` in `idsai/config.py` and `configs/default.yaml`.
- The `idsai user …` CLI and the serve guard in `idsai/cli.py`.
- A sign-in page, a user menu and an Activity page in `idsai/product/web/`.
- Tests.

**Out of scope:**
- Web sign-up, email, password reset by email, OAuth or SSO, 2FA. These are not needed for a class demo.
- Hosting is the next task.

**Constraints:**
- No new dependencies. Hashing uses `hashlib.scrypt`, tokens `secrets.token_urlsafe(32)` and comparison `hmac.compare_digest`.
- No personal data beyond a username and a display name.
- Passwords and tokens are never logged or returned by any API.

## 3. Roles

| action | viewer | analyst | admin |
|---|---|---|---|
| every GET page and `/api/*` read, WS stream | ✓ | ✓ | ✓ |
| `POST /api/alerts/{id}/feedback` | | ✓ | ✓ |
| `POST /api/replay/control` | | ✓ | ✓ |
| `POST /api/redteam/inject` | | ✓ | ✓ |
| `GET /api/audit` (Activity page) | | | ✓ |

- Roles are ordered, viewer < analyst < admin. A route declares its minimum role.
- A request with no valid session gets 401 (Thai message "กรุณาเข้าสู่ระบบ"). A valid session with too low a role gets 403.
- With auth off, every request runs as the built-in user `local`, role admin. That is today's behaviour.

## 4. Storage (`app.duckdb`, created if missing)

```
users(user_id VARCHAR PK, username VARCHAR UNIQUE, display_name VARCHAR, role VARCHAR,
      pw_hash VARCHAR, created_us BIGINT, disabled BOOLEAN)
sessions(token_hash VARCHAR PK, user_id VARCHAR, created_us BIGINT, expires_us BIGINT,
         remember BOOLEAN, last_seen_us BIGINT)
user_prefs(user_id VARCHAR PK, json VARCHAR)
audit_log(ts_us BIGINT, user_id VARCHAR, username VARCHAR, action VARCHAR,
          target VARCHAR, detail VARCHAR)
```

- **`pw_hash`:** `scrypt$n=16384,r=8,p=1$<salt b64>$<hash b64>`, with a 16-byte salt per user.
- **Usernames:** `^[a-z0-9_.-]{3,32}$`. Display name: at most 64 characters.
- **Passwords:** at least 10 characters.
- **`sessions.token_hash`:** the SHA-256 of the cookie token. The raw token exists only in the cookie.
- **`user_prefs.json`:** whitelisted keys only:
  - `theme` (`dark` / `light` / `auto`)
  - `symbol` (one of the served symbols)
  - `eval_budget` (one of the report budgets)
  - Unknown keys get 422.
- **`audit_log` actions:**
  - `login`, `login_failed`, `logout`;
  - `feedback`, `replay_control`, `redteam_inject`;
  - `user_add`, `user_role`, `user_disable`, `user_passwd`.
  - `detail` is small JSON, for example `{"verdict":"tp"}` or `{"family":"pump_burst","region":"test"}`, and never holds a password.
- **Feedback:** `feedback.analyst` is filled from the session username. The client's `analyst` field is ignored when auth is on.

## 5. Sessions and "remember me"

- **Sign in:** `POST /api/auth/login {username, password, remember}`.
  - On success: a new token, then `Set-Cookie: idsai_session=<token>`.
  - Cookie flags: `HttpOnly`, `SameSite=Strict`, `Path=/`, plus `Secure` when `auth.cookie_secure` is true (set this when hosted behind HTTPS).
  - Lifetime with `remember`: `Max-Age` 30 days, and the server sets `expires_us` to now + 30 d.
  - Lifetime without `remember`: no `Max-Age`, so a browser-session cookie, with server `expires_us` = now + 12 h.
  - The response is `{username, display_name, role, prefs}`.
- **Failure:** one generic message, "ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง", for an unknown user, a wrong password or a disabled user.
- **Rate limit:** 5 failures per username within 10 minutes lock that username for 60 s. The lock is in memory and resets on restart, which is acceptable here. A locked attempt still gets the generic message and logs `login_failed`.
- **Every request:**
  - Look up `sha256(cookie)`, then reject it if it is expired or the user is disabled.
  - Update `last_seen_us` at most once a minute.
- **WebSocket:** check the cookie at handshake. Close with code 4401 when there is none, and send nothing.
- **Sign out:** `POST /api/auth/logout` deletes the session row and clears the cookie.
- **Admin changes:**
  - Disabling a user or changing their password deletes all their sessions.
  - Changing a role takes effect on the user's next request, because the role is read from `users` every time.
- Expired sessions are deleted at startup and once an hour.
- The existing `same_origin` dependency also guards the new POST and PUT routes. Together with SameSite=Strict, that covers CSRF.

## 6. API additions

| route | role | body / result |
|---|---|---|
| `POST /api/auth/login` | none | see §5 |
| `POST /api/auth/logout` | any signed-in | 204 |
| `GET /api/me` | any signed-in, or `local` when auth is off | `{username, display_name, role, auth_enabled, prefs}` |
| `PUT /api/me/prefs` | any signed-in | whitelisted keys, returns the stored prefs |
| `GET /api/audit?limit=200&before_us=` | admin | newest first, at most 500 |

`/api/health` stays public, so the sign-in page and a host health check can read it.

## 7. Configuration and the serve guard

```yaml
auth:
  enabled: false          # local default
  cookie_secure: false    # true behind HTTPS
  remember_days: 30
  session_hours: 12
```

- `idsai serve` with a bind outside `LOOPBACK_BINDS` and `auth.enabled: false` exits with a Thai error telling you to turn auth on.
- `--auth` and `--no-auth` flags override the config for one run. `--no-auth` is still refused on a non-loopback bind.
- With auth on and no enabled admin user, serve exits and prints the command to create one.

## 8. CLI

```
idsai user add <username> --role viewer|analyst|admin [--display-name ...]
idsai user list
idsai user passwd <username>
idsai user role <username> <role>
idsai user disable <username>
idsai user enable <username>
```

- Passwords are read with `getpass`, prompted twice. They are never taken as an argument.
- For scripted setup, `--password-stdin` reads a single line.
- Each command writes an audit row as user `cli`.
- `user list` shows the role, disabled flag, created time and last seen, never hashes.

## 9. Web

- **Sign-in page:**
  - Shown when `/api/me` returns 401.
  - In the trading-desk look: the `MARKET·IDS` mark, a username field, a password field, a "จดจำฉัน" checkbox and a button.
  - Shows the generic error, and a lock message after too many tries.
  - No sign-up link.
- **Top bar:** a user chip with the display name and a role tag. Its menu offers the theme choice and "ออกจากระบบ".
- **Role gating in the UI:** buttons a role cannot use are disabled, with a tooltip "ต้องเป็น analyst". The server enforces the same rules either way.
- **Prefs:**
  - Loaded from `/api/me` at start.
  - Each change is saved with `PUT /api/me/prefs`, debounced by 500 ms.
  - With auth off, prefs use `localStorage` as today.
- **Activity page** (`#/activity`, admin only): a table of time, user, action, target and detail, filterable by user and action, with a "load older" button.
- **On a 401 mid-session:** go to the sign-in page, then return to the same hash.

## 10. Testing

- **auth.py:**
  - a hash round trip, with the wrong password failing;
  - hash format and per-user salt;
  - constant-time compare used;
  - token hashing;
  - expiry for remember and non-remember;
  - the rate-limit lock and its release.
- **store:**
  - the tables are created;
  - a unique username;
  - prefs whitelist;
  - audit ordering and paging;
  - disable and passwd delete sessions.
- **API:**
  - login sets the cookie with the right flags, and `Secure` only when configured;
  - the generic error;
  - 401 vs 403 on every mutating route, for each role;
  - feedback analyst taken from the session;
  - prefs round trip;
  - audit rows written for each action;
  - audit admin-only;
  - logout;
  - WS rejected without a cookie and accepted with one.
- **Auth off:** the whole existing API test suite passes unchanged, and `/api/me` returns `local`.
- **CLI:**
  - add, list, passwd, role, disable and enable;
  - password never in argv;
  - the serve guard on a non-loopback bind;
  - no admin, so serve exits.
- **Web static:** the sign-in page elements exist, the Activity route exists, and no network loads.
- **Browser check:**
  - sign in with and without remember;
  - a viewer cannot inject;
  - an analyst can;
  - the admin sees Activity with those rows;
  - prefs follow the user into a fresh browser profile.

## 11. Order of work

1. `auth.py`, store tables and their tests.
2. Config, the serve guard and the `user` CLI.
3. API dependencies, new routes and audit hooks.
4. Web: sign-in page, user chip, role gating, prefs sync and Activity page. This comes after the redesign has landed.
5. Full suite, then the browser check.
