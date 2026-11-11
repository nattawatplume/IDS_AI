# Command-Center Motion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the dashboard a command-center / CCTV look with rich motion: a living background, a CCTV camera with a red beam on login, a radar and alert pulses, a chart scan line and a crossing shockwave, and page transitions. Everything has a reduced-motion and effects-off path.

**Architecture:**
- A new `motion.js` owns all animation and exposes one global `Motion` API.
- A new `motion.css` holds the tokens and keyframes.
- `app.js` only calls `Motion.*` through a no-op guard, so the app works unchanged if motion fails to load.

**Tech Stack:** vanilla JS (no build), CSS, inline SVG, canvas 2D, Lightweight Charts (already vendored), pytest static tests.

**Spec:** `docs/superpowers/specs/2026-11-11-command-center-motion-design.md`

## Global Constraints

- **No new library and no build step.** Files live in `idsai/product/web/`. `motion.css` is linked after `styles.css`, and `motion.js` is loaded after `app.js`.
- **Tokens, exact values:**
  - `--fx-calm: #22d3ee`
  - `--fx-watch: #f59e0b`
  - `--fx-alert: #ef4444`
  - `--fx-grid: rgba(56, 189, 248, .08)`
  - the base background gradient runs from `#070b1a` to `#0b1230`; the purple accent stays on buttons.
- **Motion is decoration only:**
  - only `transform`, `opacity` and canvas; no layout shift of readable text;
  - it never changes a number or a word;
  - every fx element has `aria-hidden="true"`.
- **Off switches:** `prefers-reduced-motion: reduce`, or `fx_off`, means canvases are cleared and hidden and CSS animations are `none` (`.fx-off` on `<html>`). State colours still show the mood. Motion also stays off while the tour runs or the help drawer is open.
- **Flash limits:** no flashing faster than 3 per second. The edge flash fires at most once per burst, and bursts are rate-limited to 1 per 2 s.
- **Canvas performance:**
  - one shared `requestAnimationFrame` loop at most 30 fps;
  - device-pixel ratio capped at 2;
  - the loop pauses on `visibilitychange` when hidden;
  - half the dots below 640 px width;
  - no layout reads inside the loop.
- **Size budget:** `motion.js` at most about 25 KB and `motion.css` at most about 12 KB, both unminified.
- **No `innerHTML` in `motion.js`.** Every `Motion` method wraps its body in try/catch.
- **New Thai copy, verbatim:** `ตรวจพบ`, `ยืนยันตัวตนแล้ว`, `ลดเอฟเฟกต์`. The existing copy guards and brace-balance tests must pass.
- **Repo rules:**
  - commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`;
  - never `git add -f`; never commit `.claude/`, `.superpowers/` or `data/`; never push;
  - gates are `uv run ruff check .` and `uv run pytest -q -m "not network and not slow"`;
  - any manual browser check uses a temporary store harness, never the real `data/app.duckdb`, and is stopped afterwards.

## Review Focus

1. **`motion.js` fails to load or throws (404, a syntax error, an old browser).**
   - Expected: the dashboard works exactly as before.
   - Test: Task 1, `test_app_calls_motion_only_through_guard`. Every `Motion.` call in `app.js` goes through `fx(` and none is bare.
2. **A user with OS reduced motion turned on.**
   - Expected: nothing moves and the page still shows the mood colour.
   - Test: Task 1, `test_motion_css_off_paths`. The reduced-motion block and the `.fx-off` rule set `animation: none` and hide `#fx-bg` and `#fx-chart`.
3. **An alert storm: many alerts in one second during a fast replay.**
   - Expected: one edge flash per 2 s, not a strobe.
   - Test: Task 3, `test_alert_burst_rate_limited`. `motion.js` contains the 2000 ms guard constant `BURST_GAP_MS = 2000`.
4. **The anomaly value is null or NaN, or the crossing point is off-screen.**
   - Expected: no shockwave and no error.
   - Tests: Task 4, `test_crossed_up_contract` (the source text of `crossedUp` checks `Number.isFinite`) and the guard in `chartCross` for a null coordinate.
5. **The chart re-created after a symbol switch.**
   - Expected: the overlay re-attaches to the new chart with no duplicate loops.
   - Test: Task 4, `test_chart_attach_replaces_previous`. `chartAttach` detaches the earlier overlay, which needs a `detach` code path.

---

### Task 1: Motion foundation (wiring, tokens, background, off switches)

**Files:**
- Create: `idsai/product/web/motion.js` and `idsai/product/web/motion.css`.
- Modify: `idsai/product/web/index.html` (head link, `<canvas id="fx-bg">` as the first child of body, `<div id="fx-edge">`, the script tag, and a `ลดเอฟเฟกต์` switch in the `#user-chip` `.user-menu`).
- Modify: `idsai/product/web/app.js` (the guard, init, mood calls and the switch).
- Modify: `idsai/product/web/styles.css` (the base background only).
- Test: `tests/product/test_web_static.py`.

**Interfaces:**
- Produces, in `motion.js`: `globalThis.Motion` with `init({ off }: {off: boolean})`, `setOff(off: boolean)`, `mood(m: "calm" | "watch" | "alert")`, `loginEnter()`, `loginLeave(done: () => void)`, `alertBurst({ severity }: {severity: string})`, `chartAttach(api, series, { limitLine })`, `chartCross({ time, value })` and `pageEnter(node: Element)`. Methods not built yet are no-op stubs that later tasks fill in.
- Produces an internal shared loop in `motion.js`: `addLayer(draw: (ctx, t, dt) => void)` and `removeLayer(fn)`. Later tasks register their canvas drawing with it.
- Produces, in `app.js`: `function fx(name, ...args)`. It calls `globalThis.Motion?.[name]?.(...args)` inside try/catch and returns undefined.
- Produces the pref `fx_off: boolean`, saved with `savePrefs` and mirrored to `localStorage["idsai.fx_off"]` (with try/catch).
- Mood mapping in `renderHero`: hero state "calm" is `calm`, any watch or pending state is `watch`, and an active alert state is `alert`. Reuse the existing hero states and pick the closest mapping.

- [ ] **Step 1: Write the failing static tests:**
  - `test_motion_files_linked`: `motion.css` follows `styles.css` in `index.html`, `motion.js` follows `app.js`, and both files exist.
  - `test_fx_elements_hidden_from_screen_readers`: `#fx-bg` and `#fx-edge` have `aria-hidden="true"`.
  - `test_motion_api_defined`: `motion.js` defines all 9 API names from Interfaces, and contains no `innerHTML`.
  - `test_app_calls_motion_only_through_guard`: `app.js` has `function fx(`, and has no `Motion.` text except inside `fx`.
  - `test_motion_css_tokens_and_off_paths`: the four `--fx-` tokens are present with the exact values, there is a `prefers-reduced-motion` block and a `.fx-off` rule with `animation: none`, and `#fx-bg` is hidden under both.
  - `test_fx_switch_copy`: `ลดเอฟเฟกต์` appears in `index.html`.
  - Extend the brace-balance test to cover `motion.js` and `motion.css`.
- [ ] **Step 2: Run** `uv run pytest -q tests/product/test_web_static.py`. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - **Background canvas layer:**
    - a perspective grid drifting slowly in `--fx-grid`;
    - 30–60 rising, fading dots (half below 640 px);
    - a horizontal scan line every 8–12 s;
    - colours follow the current mood token.
  - **Shared loop:** one rAF at most 30 fps, DPR capped at 2, paused on `document.hidden`, and the canvas resized on a debounced `resize`.
  - **Off state:** `init` reads `off || reducedMotion`. In the off state it clears and hides the canvas and sets `.fx-off` on `document.documentElement`. `setOff` toggles the same live.
  - **Wiring in `app.js`:**
    - call `fx("init", { off })` at boot, with `off` from `localStorage` or prefs;
    - wire the switch to `fx("setOff", v)` and the pref;
    - call `fx("mood", …)` in `renderHero`;
    - call `fx("setOff", true)` while the tour runs or the help drawer is open, and restore the user's value afterwards.
- [ ] **Step 4: Run** the focused tests (PASS), then the full gate (PASS).
- [ ] **Step 5: Commit** with the message `feat(web): motion foundation: control-room background, mood colours, effects-off switch`.

### Task 2: Login CCTV scene

**Files:**
- Modify: `index.html` (the camera SVG inside `.login-hero`, beside the existing sample-chart SVG), `motion.js` and `motion.css`.
- Modify: `app.js` (call `fx("loginEnter")` when the login page shows, and `fx("loginLeave", done)` on successful sign-in inside `completeSignIn`, with `done` continuing the existing boot).
- Test: `tests/product/test_web_static.py`.

**Interfaces:**
- Consumes: `Motion` and `fx` from Task 1.
- Produces: `loginEnter()` and `loginLeave(done)`.
  - `loginLeave` always calls `done` within 900 ms, or at once when off, even if an animation fails.

- [ ] **Step 1: Write the failing tests:**
  - `test_login_cctv_markup`:
    - `.login-hero` contains `svg#cctv` with `aria-hidden="true"`;
    - it has a beam element `#cctv-beam`, a REC light `#cctv-rec`, and a hidden tag element whose text is `ตรวจพบ`.
  - `test_login_leave_always_continues`: in `motion.js`, `loginLeave` contains a `setTimeout` fallback that calls `done`, and `app.js` passes the boot continuation into `fx("loginLeave"`.
  - `test_verified_copy`: `ยืนยันตัวตนแล้ว` is present in `index.html` or `motion.js`.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - **Camera:** an inline SVG wall camera. It pans ±18° on a 6–8 s ease-in-out CSS loop. `REC` blinks at 1 Hz.
  - **Beam:** a red semi-transparent cone (`--fx-alert` with a soft gradient edge) attached to the lens, which rotates with the camera.
  - **Beam hit:** `motion.js` checks the beam angle against the spike's angle each frame without layout reads (the angles are computed once on enter and on resize). On a hit, the spike path gets a glow class, a pulse ring spreads from the spike, and the `ตรวจพบ` tag fades in for 1.5 s. There is at most one hit per pan cycle.
  - **Login card:** it slides and fades in. Inputs glow on focus, and the submit button shows a sheen.
  - **`loginLeave`:** a 600–800 ms "ยืนยันตัวตนแล้ว ✓" overlay, then a zoom and fade into the dashboard, then `done()`.
  - **Off state:** the camera is static, the beam is shown dim and still, and `loginLeave` calls `done` at once.
- [ ] **Step 4: Run** the tests and the full gate. Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat(web): login CCTV camera with red beam sweep and verified transition`.

### Task 3: Hero radar, alert pulses and burst, card glow

**Files:**
- Modify: `motion.js`, `motion.css`, `index.html` (the hero orb gets a radar child and a small camera icon in `#hero`) and `app.js` (`onAlert` calls `fx("alertBurst", { severity })`).
- Test: `tests/product/test_web_static.py`.

**Interfaces:**
- Consumes: `Motion.mood` (Task 1) and `fx`.
- Produces: `alertBurst({ severity })`, guarded by the constant `BURST_GAP_MS = 2000`.

- [ ] **Step 1: Write the failing tests:**
  - `test_alert_burst_rate_limited`: `motion.js` contains `BURST_GAP_MS = 2000`, and `app.js` `onAlert` calls `fx("alertBurst"`.
  - `test_hero_radar_markup`: `#hero-orb` contains a radar element with `aria-hidden="true"`, and `#hero` contains the camera icon with `aria-hidden="true"`.
  - `test_edge_flash_once`: the `#fx-edge` keyframes in `motion.css` run `animation-iteration-count: 1` (or `1` in the shorthand).
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - **Calm radar:** a conic-gradient sweep inside the orb, about 4 s per turn.
  - **Alert mood:** 2–3 concentric red rings expanding and fading with about 1.2 s stagger, replacing the sweep. The hero camera icon turns toward the chart.
  - **`alertBurst`:** if at least `BURST_GAP_MS` have passed since the last burst:
    - flash the `#fx-edge` red vignette once;
    - add a red light sweep class to the newest alert card for 1.2 s;
    - set the mood to `alert` until the next `renderHero` decides otherwise.
  - **KPI cards:** a thin glowing border light travels around each card about every 6 s, out of phase between cards. Hover lifts a card 2 px and brightens its border.
- [ ] **Step 4: Run** the tests and the full gate. Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat(web): hero radar and alert pulse rings, rate-limited alert burst, card glow`.

### Task 4: Chart overlay, scan line and crossing shockwave

**Files:**
- Modify: `index.html` (`<canvas id="fx-chart" aria-hidden="true">` absolutely positioned inside the box of `#chart`), `motion.js`, `motion.css`.
- Modify: `app.js`:
  - after `chart.api = LWC.createChart(...)` near line 1121, call `fx("chartAttach", chart.api, <anomaly series>, { limitLine: chart.tauLine })`;
  - add `function crossedUp(prev, cur, limit)`;
  - where new anomaly points are appended, call `fx("chartCross", { time, value })` only when `crossedUp(prevValue, value, tau)` is true.
- Test: `tests/product/test_web_static.py`.

**Interfaces:**
- Consumes: the `addLayer` / `removeLayer` loop (Task 1).
- Produces:
  - `crossedUp(prev: number, cur: number, limit: number) -> boolean`, true only when all three are finite and `prev <= limit < cur`;
  - `chartAttach(api, series, { limitLine })`, which replaces any previous overlay;
  - `chartCross({ time, value })`, which does nothing when `timeToCoordinate` or `priceToCoordinate` returns null.

- [ ] **Step 1: Write the failing tests:**
  - `test_crossed_up_contract`: `app.js` defines `function crossedUp(prev, cur, limit)`, its body contains `Number.isFinite` and `<= limit` and `< cur` (or an equivalent written as `limit < cur`), and it is the only guard before `fx("chartCross"`.
  - `test_chart_attach_replaces_previous`: `motion.js` `chartAttach` calls the previous overlay's detach or `removeLayer` before adding a new one.
  - `test_chart_overlay_canvas`: `#fx-chart` exists with `aria-hidden="true"` and is hidden under `.fx-off` and reduced motion in `motion.css`.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - **Overlay layer:**
    - a vertical scan line at the latest bar's x;
    - a pulsing ring on the latest anomaly point;
    - a soft gradient fill under the price line (prefer setting Lightweight Charts area or line options on the existing series where supported, and draw on the overlay otherwise).
  - **Coordinates:** read through `api.timeScale().timeToCoordinate` and `series.priceToCoordinate`, re-read only on `subscribeVisibleTimeRangeChange` and on new bars. Size the overlay with a `ResizeObserver` on `#chart`.
  - **`chartCross`:** two rings expanding to about 60 px over 900 ms with a short flash at the point. The limit line glows red and shakes about 2 px for 400 ms (a CSS class on the overlay, or a temporary priceLine colour change, restored afterwards). Also call `alertBurst`, which is rate-limited.
- [ ] **Step 4: Run** the tests and the full gate. Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat(web): chart scan line, live-point pulse and shockwave when the score crosses the limit`.

### Task 5: Page transitions and final polish

**Files:**
- Modify: `motion.js`, `motion.css` and `app.js` (`route()` calls `fx("pageEnter", node)` after showing a page, but not on the first paint).
- Test: `tests/product/test_web_static.py`.

**Interfaces:**
- Consumes: everything above.
- Produces: `pageEnter(node)`.

- [ ] **Step 1: Write the failing tests:**
  - `test_page_enter_wired`: `route()` calls `fx("pageEnter"` and skips it on the first paint, through a flag or check visible in the source.
  - `test_motion_size_budget`: `motion.js` is at most 25 * 1024 bytes and `motion.css` is at most 12 * 1024 bytes.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement.** A 300 ms cyan scan line wipes the page from top to bottom, then direct child cards rise in with a 40 ms stagger, using transform and opacity only. Skip this when off.
- [ ] **Step 4: Check by hand in a browser,** at desktop and 375 px width, with a temporary store harness:
  - the login camera, beam hit and sign-in transition;
  - the calm background;
  - a forced crossing shockwave (inject a red-team scenario or nudge `tau`);
  - an alert burst;
  - the page wipe;
  - OS reduced motion and `ลดเอฟเฟกต์` giving a static page;
  - a hidden tab pausing the loop (check `document.hidden` handling through the console).

  Stop the harness afterwards.
- [ ] **Step 5: Run** the full gate. Expected: PASS.
- [ ] **Step 6: Commit** with the message `feat(web): scan-in page transitions; motion size budget`.
