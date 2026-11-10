# Hosting on Render (free) instead of Hugging Face

Date: 2026-11-10 · Status: owner approved ("do the best and build it")
Replaces the platform choice in `2026-11-05-hosting-design.md`. The app-side pieces from that spec stay: `user seed-env`, `--public-host`, `--secure-cookies`, `--loop` and the hardening. Hugging Face Spaces now requires a paid PRO plan for Docker Spaces, and the owner will not pay.

## 1. Why and what success means

The goal is a public https link to the dashboard that stays up 24/7 for free, with no credit card. Render's free web service fits:
- free, with no card;
- 750 instance hours a month, which is enough for one service all month;
- Docker builds;
- https on `<name>.onrender.com`;
- an automatic deploy on every push.

Its limits are 512 MB of RAM and 0.1 CPU. A free service also spins down after 15 minutes without traffic.

Measured locally on 2026-11-10 (the `serve --mode replay` demo):

| Measure | Result |
|---|---|
| RAM with torch | ~406 MB |
| RAM of torch alone | ~150 MB |
| RAM without torch (estimated) | ~255 MB |
| CPU at 60× | 0.20 core, too much for 0.1 |
| CPU at 20× | 0.06 core, fits |

Success means:
- The running web app never imports torch.
- The hosted replay runs at 20× by default, and the start script accepts `IDSAI_SPEED` to change it.
- `git push` to the deployed branch redeploys Render.
- A GitHub cron pings `/api/health` every 10 minutes so the service does not spin down.
- Detection results are bit-identical to today. Scoring is already a pure-NumPy forward pass on exported weights; this spec only removes the torch import from the scoring path.

## 2. Design

### 2.1 Torch only for training (`idsai/ai/autoencoder.py`)

- Remove the module-level `import torch` and `from torch import nn`. Import them inside the functions that train: `_torch_determinism`, `_build_net` and `fit`.
- `_PLAN` becomes a static constant, `[("0", True), ("2", False), ("3", True), ("5", False)]`, so loading and scoring need no torch.
- A test that has torch builds the real net and asserts that `_layer_plan()` equals `_PLAN`, so the constant cannot drift. Keep `_layer_plan()` as a function that needs torch, used only by that test.
- **Guarantee:** importing `idsai.ai.autoencoder`, loading a saved bundle and scoring all work in a process where torch cannot be imported. A subprocess test proves it: it sets `sys.modules["torch"] = None` before any import, loads a tiny state dict and scores. The test also checks the serve import path, `idsai.cli` and `idsai.product.api`.

### 2.2 torch as a training-only dependency (`pyproject.toml`, `uv.lock`)

- Move `torch` from `[project] dependencies` to a dependency group `train`. Keep the CPU index source.
- Set `[tool.uv] default-groups = ["dev", "train"]`. Local runs and CI (`uv sync --frozen`) still get torch, and the suite is unchanged.
- Regenerate `uv.lock` with `uv lock`.

### 2.3 Two-stage Docker image (`Dockerfile`)

- **Stage `build`:**
  - Install with `uv sync --frozen --no-dev`, which includes the train group.
  - Run `idsai labels && idsai demo-pack && python deploy/check_no_hu.py data`.
- **Stage `runtime`:**
  - Same base and same uid 1000 setup.
  - Install with `uv sync --frozen --no-dev --no-group train`.
  - Copy the app files, then `COPY --from=build` the built `data/` directory (the demo pack, the app DB and the archive) with the right owner.
- A static test asserts:
  - the runtime stage's sync excludes the `train` group;
  - `data/` is copied from the build stage;
  - `check_no_hu.py` runs in the build stage.

### 2.4 Start script (`deploy/start.sh`)

- Host name: `PUBLIC_HOST="${RENDER_EXTERNAL_HOSTNAME:-${SPACE_HOST:-}}"`. If it is empty, exit 1 with a Thai error.
- Port: `PORT="${PORT:-7860}"`. Render sets `PORT`.
- Speed: `SPEED="${IDSAI_SPEED:-20}"`.
- Then run:
  ```
  exec idsai serve --mode replay --loop --speed "$SPEED" --host 0.0.0.0 --port "$PORT" --auth --secure-cookies --public-host "$PUBLIC_HOST"
  ```
- Seeding is unchanged: `idsai user seed-env` runs first.

### 2.5 Render config and workflows

- **New `render.yaml` (Blueprint):** one service with
  - `type: web`, `runtime: docker`, `plan: free`, `name: market-ids`;
  - `healthCheckPath: /api/health` and `autoDeploy: true`;
  - env var `IDSAI_USERS` with `sync: false`, so the owner types it in the dashboard and it is never in git.
- **Delete `.github/workflows/deploy-space.yml`.** Render pulls from GitHub itself, so no token is needed.
- **Update `.github/workflows/keepalive.yml`:**
  - cron `*/10 * * * *` plus `workflow_dispatch`;
  - repo variable `APP_URL`, for example `market-ids.onrender.com`, passed through `env:`;
  - the same curl as before: `curl -fsS --retry 5 --retry-delay 30 --max-time 120 "https://$APP_URL/api/health"`;
  - a Thai error if `APP_URL` is empty, and `permissions: contents: read`.

### 2.6 README hosting section, rewritten in plain Thai, click by click

1. Sign up at render.com with GitHub. No card.
2. Click New → Blueprint, pick the `IDS_AI` repo, and Render reads `render.yaml`. The fallback is New → Web Service → the repo → Docker → Free.
3. Choose the branch: `feat/mvp`, or `main` after the merge.
4. Set the env var `IDSAI_USERS`, one `name:role:password` per line, with at least one admin. The rules are unchanged.
5. Create the service. The first build takes about 15 minutes. Open `https://<name>.onrender.com`.
6. In GitHub, go to Settings → Secrets and variables → Actions → Variables and add `APP_URL` (`<name>.onrender.com`, without https).
7. The keep-awake cron runs only from `main`, so merge the PR. Until then the service sleeps after 15 minutes and takes about a minute to wake.

Keep these notes:
- a restart resets alerts, feedback, sessions and the activity log;
- there is one shared replay for everyone;
- GitHub disables cron after 60 days without activity.

Add: free instances have 0.1 CPU, so the hosted replay runs at 20×, and the dashboard pauses for about 2–3 minutes while each loop round warms up.

## 3. Out of scope

- Persistent storage.
- A paid plan.
- Changing detection or model training.
- Local `idsai serve` behaviour, which stays the same apart from the lazy import.

## 4. Testing

- **Lazy torch:** the subprocess test with torch blocked imports the serve path, loads a state dict and scores.
- **Plan constant:** a test, with torch, asserts `_PLAN == _layer_plan()`.
- **Results unchanged:** the existing autoencoder determinism and score tests pass unchanged.
- **Deploy files:** static tests cover `render.yaml`, the two Docker stages, the `start.sh` defaults (`PORT`, `RENDER_EXTERNAL_HOSTNAME` fallback, speed 20, Thai error), `keepalive.yml` (10-minute cron, `APP_URL`), the README steps, and that `deploy-space.yml` is gone.
- **Gates:** `uv run ruff check .` and `uv run pytest -q -m "not network and not slow"` pass.
- Docker is not installed locally, so the first Render build is the real image test.
