# Render Hosting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Run the hosted dashboard on Render's free plan: no torch at runtime, a two-stage Docker image, a Render blueprint, a 10-minute keep-awake ping and a rewritten README section.

**Architecture:**
- Scoring is already pure NumPy. Torch moves behind lazy imports and into a `train` dependency group.
- The image trains in a build stage and ships a runtime stage that has no torch.

**Tech Stack:** Python 3.13, uv dependency groups, Docker multi-stage, Render blueprint, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-10-render-hosting-design.md`

## Global Constraints

- Detection results stay bit-identical, and the existing autoencoder tests pass unchanged.
- Local and CI installs (`uv sync --frozen`) still include torch through `default-groups = ["dev", "train"]`.
- Thai user-facing messages. Never print secrets. Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never `git add -f`; never commit `.claude/`, `data/` or `models/`. No long `--basetemp`. Stop any local `idsai serve` before running tests.
- Gates before each commit: `uv run ruff check .` and `uv run pytest -q -m "not network and not slow"`.

## Review Focus

1. Some other module on the serve path imports torch (check `idsai/ai/*`, `idsai/eval/*` reachable from `idsai.cli`). Task 1's blocked-torch subprocess test must import `idsai.cli` and `idsai.product.api`.
2. `uv.lock` regenerated with different versions: only the group membership of torch may change (Task 2 checks `git diff uv.lock` touches only group metadata).
3. `start.sh` run under `sh` (dash) with `PORT` unset falls back to 7860. With both host variables unset it exits 1 with a Thai message (Task 2 test).

---

### Task 1: Lazy torch in the autoencoder

**Files:**
- Modify: `idsai/ai/autoencoder.py`.
- Test: `tests/ai/test_autoencoder.py`. Extend it, or add a new test file next to it.

**Interfaces:**
- Produces: `_PLAN: list[tuple[str, bool]] = [("0", True), ("2", False), ("3", True), ("5", False)]` as a module constant.
- `_layer_plan()` remains and imports torch lazily.

- [ ] **Step 1: Write the failing tests.**
  - `test_plan_constant_matches_torch_net`: `_PLAN == _layer_plan()`.
  - `test_serve_path_works_without_torch`: a subprocess runs `python -c` with code that:
    - sets `sys.modules["torch"] = None` first;
    - imports `idsai.cli`, `idsai.product.api` and `idsai.ai.autoencoder`;
    - builds an `AutoencoderDetector` via `load_state_dict` on a small synthetic state (features `["a","b"]`, random weights in torch layout for `_build_net(2, 4, 2)` shapes written by hand, center and scale);
    - scores a 3-row frame and prints `ok`.
  - Assert the return code is 0 and the output contains `ok`.
- [ ] **Step 2: Run the tests and confirm they FAIL.** The subprocess fails because torch is imported at module load.
- [ ] **Step 3: Make torch lazy.**
  - Remove the top-level torch imports.
  - Import `torch` and `torch.nn` inside `_torch_determinism`, `_build_net`, `_layer_plan` and `fit`.
  - Set `_PLAN` to the constant.
  - Update the module docstring line about torch.
- [ ] **Step 4: Run the tests again.** Run `uv run pytest -q tests/ai tests/test_cli.py tests/product -m "not network and not slow"` and confirm PASS, then run ruff.
- [ ] **Step 5: Commit** with the message `refactor(ai): import torch only for training; scoring and serve need no torch`.

### Task 2: train dependency group, two-stage Docker image, start.sh for Render

**Files:**
- Modify: `pyproject.toml`, `uv.lock`, `Dockerfile` and `deploy/start.sh`.
- Test: `tests/test_deploy_files.py`.

**Interfaces:**
- Consumes: Task 1, so the runtime needs no torch.

- [ ] **Step 1: Write the failing tests.**
  - `test_torch_is_training_only`: parse `pyproject.toml` with `tomllib` and check that:
    - torch is not in `project.dependencies`;
    - torch is in `dependency-groups.train`;
    - `tool.uv.default-groups` contains `"train"`.
  - `test_dockerfile_two_stages`: the Dockerfile has `AS build` and a second `FROM`. The second stage's sync line has `--no-group train`. There is a `COPY --from=build` of `data`. `check_no_hu.py` runs before the second `FROM`. `EXPOSE` and uid 1000 still hold.
  - `test_start_script_render`: the script contains `RENDER_EXTERNAL_HOSTNAME`, `${PORT:-7860}` and `${IDSAI_SPEED:-20}`, and the serve line uses `--speed "$SPEED"`. Running it with `sh` and both host variables unset exits 1 with Thai text (skip if no `sh`).
  - Update the existing Dockerfile and start-script tests for the new lines.
- [ ] **Step 2: Run the tests and confirm they FAIL.**
- [ ] **Step 3: Implement.**
  - Edit the `pyproject.toml` group and run `uv lock`. Check that `git diff uv.lock` changes only group membership metadata; if versions moved, report it.
  - Write the Dockerfile per spec §2.3. The build stage and the runtime stage each create `user` uid 1000 and `/home/user/app` owned by `user` before `USER user`. Keep the `.dockerignore`.
  - Write `start.sh` per spec §2.4.
- [ ] **Step 4: Run the full gate and confirm PASS.**
- [ ] **Step 5: Commit** with the message `build: torch in a train group; two-stage image without torch; start.sh reads Render env`.

### Task 3: Render blueprint, keep-awake, README; drop the HF deploy workflow

**Files:**
- Create: `render.yaml`.
- Delete: `.github/workflows/deploy-space.yml`.
- Modify: `.github/workflows/keepalive.yml` and the hosting section of `README.md`.
- Test: `tests/test_deploy_files.py`.

- [ ] **Step 1: Write the failing tests.**
  - `test_render_blueprint`: `yaml.safe_load(render.yaml)` has a service with `type: web`, `runtime: docker`, `plan: free`, `healthCheckPath: /api/health` and an env var `IDSAI_USERS` with `sync: false`. No secret value appears anywhere in the file.
  - `test_no_hf_deploy_workflow`: `.github/workflows/deploy-space.yml` does not exist.
  - Keepalive: the cron is `*/10 * * * *`; it uses `APP_URL` from `vars.` through `env:`; the curl line is exactly `curl -fsS --retry 5 --retry-delay 30 --max-time 120 "https://$APP_URL/api/health"`; it has `permissions: contents: read`; it fails with a Thai message when `APP_URL` is empty.
  - README: the hosting section mentions `render.com`, `IDSAI_USERS`, `APP_URL`, `onrender.com`, the 20× speed, the 2–3 minute warm-up, the merge-to-`main` note for the cron, and the existing notes on restart, the shared replay and the 60-day limit. It no longer tells the user to create a Hugging Face Space.
  - Remove or adapt the old HF-specific workflow and README tests.
- [ ] **Step 2: Run the tests and confirm they FAIL.**
- [ ] **Step 3: Implement** per spec §2.5–2.6, in plain Thai with click-by-click steps.
- [ ] **Step 4: Run the full gate and confirm PASS.**
- [ ] **Step 5: Commit** with the message `ci: Render blueprint and 10-min keep-awake; README hosting on Render; drop HF deploy`.
