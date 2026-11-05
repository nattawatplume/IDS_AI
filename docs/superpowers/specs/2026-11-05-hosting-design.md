# Hosting on Hugging Face Spaces

Date: 2026-11-05 · Status: approved in chat, spec awaiting review
Depends on: `2026-11-05-auth-design.md` (sign-in must be on for any public bind).

## 1. Why

Today the dashboard runs only while the owner's laptop runs `idsai serve`. Teachers and the committee need a link they can open at any time, from any device, without the laptop on.

Success means:
- A public `https://<user>-<space>.hf.space` URL serves the dashboard 24/7 in replay mode, with sign-in required.
- Nothing in the image or the Space repo comes from `data/`, `models/` or `*.duckdb`. No Hu et al. label is in the image.
- Accounts survive a restart. They are rebuilt from a Space secret at every boot.
- The replay never runs out. When the day ends, it starts again.
- A push to `main`, or a manual run, deploys the Space. No laptop step is needed after the one-time setup.

## 2. Platform choice

**Hugging Face Spaces, Docker SDK, free CPU tier (2 vCPU, 16 GB RAM).**
- It is free, needs no card and gives HTTPS out of the box.
- Its RAM fits torch.
- Oracle Always Free needs a card and server upkeep. Render and Railway free tiers sleep after about 15 minutes or have too little RAM.

Known limits, and how this design handles them:

| Limit | Handling |
|---|---|
| The Space sleeps after 48 h without traffic | A GitHub Actions cron pings `/api/health` every 6 h. |
| The disk is wiped on every restart or rebuild | The demo pack is built into the image. Accounts are re-seeded from the `IDSAI_USERS` secret at boot. Alerts, feedback, the audit log and sessions reset on restart; this is accepted for a demo. |
| Binance's live API blocks US IPs | Hosted mode is replay only. `data.binance.vision` (the archive) is used only at image build. |
| GitHub disables scheduled workflows after 60 days with no repo activity | Documented. Re-enabling the workflow is one click. |

## 3. Scope

**In scope:**
- `Dockerfile`, `.dockerignore` and `deploy/start.sh`.
- New CLI pieces in `idsai/cli.py`:
  - `idsai user seed-env`;
  - the serve flags `--public-host`, `--secure-cookies` and `--loop`.
- Replay looping in `idsai/data/replay.py` and the serve wiring.
- The `--public-host` allowlist and same-origin change in `idsai/product/api/app.py`.
- Hardening carried from the auth work (section 4.6).
- Two workflows: `.github/workflows/deploy-space.yml` and `.github/workflows/keepalive.yml`.
- A README "Hosting" section with click-by-click setup steps.
- Tests.

**Out of scope:**
- Live mode on the host.
- Persistent storage (a paid feature).
- Admin HTTP routes for accounts, since `seed-env` covers them.
- A custom domain.
- Per-user replays. One shared replay serves every viewer, and an analyst's play, pause or seek affects everyone; this is documented.

## 4. Design

### 4.1 Image (`Dockerfile`, `.dockerignore`)

- **Base and tools:** start from `python:3.13-slim` and copy `uv` from its official image. Then run `uv sync --frozen --no-dev`. Torch comes from the CPU index already set in `pyproject.toml`.
- **User:** run as a non-root user with uid 1000, as HF requires. The working directory is `/home/user/app`.
- **`.dockerignore`:** excludes `data/`, `models/`, `*.duckdb*`, `.venv/`, `.git/`, `.superpowers/`, `.claude/`, `runs/`, caches and `tests/`.
- **Build-time data:** run `idsai labels` and then `idsai demo-pack`.
  - `labels` uses Sapienza only, because the Hu file is absent.
  - `demo-pack` downloads about 23 symbol-days from `data.binance.vision` and trains the demo bundle.
- **Build-time guard:** the build fails if `data/labels/hu_pd_logs.tsv` exists, or if any label row in `data/app.duckdb` has source `hu`. The guard is a small `idsai`-free Python check in `deploy/check_no_hu.py`.
- **Port:** `EXPOSE 7860`, and the start command is `deploy/start.sh`.

### 4.2 Start script (`deploy/start.sh`)

```sh
set -eu
idsai user seed-env
exec idsai serve --mode replay --loop --host 0.0.0.0 --port 7860 \
  --auth --secure-cookies --public-host "$SPACE_HOST"
```

HF sets `SPACE_HOST` (for example `plume-market-ids.hf.space`). If `SPACE_HOST` is unset, the script exits with a Thai error.

### 4.3 `idsai user seed-env`

- **Input:** it reads the env var `IDSAI_USERS`, one account per line, in the form `name:role:password`. The line is split on the first two `:` only, so a password may contain `:`. Blank lines and lines starting with `#` are skipped.
- **Per account:**
  - if the account is missing, it is created;
  - if it exists, its role and password are set and it is enabled.
- **Validation:**
  - Names, roles and passwords use the existing checks (`check_username`, `check_password`, ROLES, reserved `cli`/`local`).
  - Any bad line makes the command exit 1 before writing anything. The message names the line number, never the password.
  - At least one `admin` line is required, or the command exits 1.
  - If `IDSAI_USERS` is unset or empty, the command exits 1.
- **Audit:** each create or update writes one audit row with actor `cli`.
- **Output:** it prints `seeded N accounts (created A, updated U)` and never prints a password.

### 4.4 Serve flags

- **`--public-host NAME`** (repeatable):
  - each NAME is added to the Host-header allowlist and to the same-origin check, so the WebSocket and POSTs from `https://NAME` are accepted;
  - NAME must be a bare hostname with no scheme, port or path, otherwise it is a usage error (exit 1).
- **`--secure-cookies`** sets `cfg.auth.cookie_secure = True` for this run. HF serves over HTTPS. Only the browser sees the `Secure` attribute, because uvicorn sits behind HF's TLS proxy.
- **`--loop`** applies to replay only. In live mode it is a usage error.

### 4.5 Replay loop

When `--loop` is set and the replay delivers its last event, serve starts the replay again:
1. It begins from the same start time, using the original trades. Red-Team injections are dropped, because they were made on the in-memory copy.
2. It deletes replay alerts at or after the start, the same call serve makes at startup, so the list does not fill with duplicates.
3. It broadcasts the same event the dashboard already handles for a seek, so charts and the rail reset.

Red-Team runs still pending at the restart end as "missed (replay restarted)" and are not left hanging.

Without `--loop`, behaviour is unchanged: the replay ends and stays ended.

### 4.6 Hardening carried from the auth work

1. **LoginLimiter pruning.** Entries whose newest failure is older than the window, and whose lock has expired, are removed on each check. Memory stays bounded under username rotation.
2. **Login waiter cap.** When more than 16 logins are already waiting on the scrypt semaphore, a new login gets 429 "ระบบไม่ว่าง ลองใหม่อีกครั้ง". scrypt does not run for it, and no failure is counted.
3. **Locked-attempt audit coalescing.** While a username is locked, only the first rejected attempt writes an audit row. Later ones during the same lock do not.
4. **WebSocket session re-check.** An open WebSocket re-checks its session every 60 s and closes with 4401 if the session is expired, deleted or its user is disabled. The dashboard already handles 4401.

### 4.7 Deploy workflow (`.github/workflows/deploy-space.yml`)

- **Triggers:** a push to `main`, and `workflow_dispatch` so it can run on any branch, including `feat/mvp` before the PR merges.
- **Needs:**
  - the repo secret `HF_TOKEN`, a write token;
  - the repo variable `HF_SPACE`, for example `nattawatplume/market-ids`.
- **Steps:**
  1. Check out the repo.
  2. Fail if the tree contains `data/`, `models/`, `*.duckdb` or `hu_pd_logs.tsv`.
  3. Prepend the Space front matter to `README.md`, in the workflow's copy only: `title`, `sdk: docker`, `app_port: 7860`, `pinned: false`.
  4. Commit to a temporary orphan branch and force-push it to `https://huggingface.co/spaces/$HF_SPACE` `main`.
- **Token handling:** the token is passed through the git credential helper, never echoed.

### 4.8 Keep-awake workflow (`.github/workflows/keepalive.yml`)

- It runs on the cron `0 */6 * * *` and on `workflow_dispatch`.
- It runs `curl -fsS --retry 5 --retry-delay 30 --max-time 120 https://$SPACE_URL/api/health`, where `SPACE_URL` is a repo variable such as `nattawatplume-market-ids.hf.space`.
- A failure only marks the run red. Nothing else happens.

### 4.9 Notes for users (README)

- Open the direct `https://<user>-<space>.hf.space` URL, not the huggingface.co Space page. That page shows the app in an iframe, where the `SameSite=Strict` cookie is not sent, so sign-in would not stick.
- Adding or removing an account means editing the `IDSAI_USERS` secret. Changing a secret restarts the Space.
- After any restart, alerts, feedback, sessions and the activity log start empty.

## 5. One-time setup (owner, by hand)

1. Create a Hugging Face account and verify the email.
2. Create a new Space with SDK Docker, the template Blank, hardware CPU basic (free) and visibility Public. Copy its name, for example `nattawatplume/market-ids`.
3. In the Space settings, open "Variables and secrets". Add the secret `IDSAI_USERS` with one line per account, at least one admin.
4. In HF settings, open Access Tokens. Create a token with Write access.
5. In the GitHub repo settings, open Secrets and variables, then Actions:
   - add the secret `HF_TOKEN`;
   - add the variable `HF_SPACE` (`user/space`);
   - add the variable `SPACE_URL` (`user-space.hf.space`).
6. In GitHub Actions, open "Deploy to Hugging Face Space" and choose Run workflow on `feat/mvp`, or merge PR #11.
7. Wait for the Space build (about 10–15 minutes the first time), then open the URL and sign in.

## 6. Error handling

| Case | Behaviour |
|---|---|
| `IDSAI_USERS` is missing, bad or has no admin | `seed-env` exits 1 and the container stops. The HF build log shows the Thai reason with the line number only. |
| `SPACE_HOST` is unset | `start.sh` exits 1 with a Thai message. |
| The archive download fails during build | The build fails, and HF shows the log. A rebuild retries. |
| The replay ends while `--loop` is set | It restarts (4.5). |
| A login storm | Waiters beyond 16 get 429. Limiter memory is pruned. |
| A session is revoked while a WebSocket is open | The socket closes with 4401 within 60 s. |

## 7. Testing

- **Unit tests (pytest, no network)** cover:
  - `seed-env`: create, update, reset password, re-enable, all-or-nothing on a bad line, no admin, unset env, a password containing `:`, and no password in the output;
  - `--public-host`: the allowlist, same-origin acceptance and rejection of a bad name;
  - `--secure-cookies` setting the `Secure` attribute;
  - `--loop` as a usage error in live mode;
  - the replay loop: a restart after the last event, injections dropped, alerts after the start deleted, the reset event broadcast and pending Red-Team runs closed;
  - limiter pruning, the waiter cap returning 429 without running scrypt, audit coalescing while locked, and the WebSocket re-check closing with 4401.
- **The Hu guard script** gets a test with a fake Hu label row.
- **Workflows:** `actionlint`, if it is available locally. Otherwise the YAML is reviewed by eye, and the first run is the test.
- **Docker:** `docker build` and a local `docker run`, if Docker is installed. Otherwise the first HF build is the test, and the owner is told so.
- **Gates:** `uv run ruff check .` and the CI pytest command must pass before any push.
