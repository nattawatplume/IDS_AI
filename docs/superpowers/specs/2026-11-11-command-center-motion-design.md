# Command-center look and motion for the dashboard

Date: 2026-11-11. Status: the owner approved approach A ("อนุมัติ เขียน spec").

This builds on `2026-11-06-dashboard-glass-design.md` (v3 "Glass"). The pages, data, wording rules and copy guards stay as they are. This spec changes the look and adds motion.

## 1. Goal and success

The owner finds the login page and the main dashboard plain and lifeless. The background is empty, and nothing happens when the anomaly line crosses the alert limit. The goal is a professional "command center / CCTV" feel with a lot of motion.

**Owner decisions:**
- The style is a command center: a radar sweep, a glowing grid, pulse rings on anomalies, and a scan line on the chart.
- Add a CCTV camera that pans and projects a red beam (the owner's own idea).
- When calm, the screen moves all the time but gently. On an alert, motion bursts: big, red and obvious, so an alert stands out against the calm state.
- Approach A: hand-written CSS, SVG and canvas. No new library, no build step.

**Success means:**
- The login page has a panning CCTV camera with a red beam cone sweeping across a sample chart.
  - When the beam hits the spike, the spike glows, a pulse ring spreads and a `ตรวจพบ` tag appears.
  - Signing in plays a short "verified" transition into the dashboard.
- Every page has a living control-room background: a slowly flowing grid, drifting data dots and an occasional horizontal scan line. It is cyan when calm and shifts to amber or red during an alert state.
- When the anomaly score crosses the alert limit on the main chart, the crossing point gets a visible shockwave, the limit line flashes red and the screen edge flashes red once.
- New alert cards slide in with a red light sweep, and the hero status switches from a calm radar sweep to red pulse rings.
- Motion never blocks reading or clicking, never hides data and never changes any number or word.
- With `prefers-reduced-motion: reduce`, or with the new in-app "ลดเอฟเฟกต์" switch on, every animation stops and the page looks clean and static. All information is still shown.
- A mid-range laptop keeps the page smooth: no visible jank while scrolling, and the canvas pauses when the tab is hidden.

**Out of scope:** sound, 3D or WebGL, new pages, and changes to data, API or wording rules.

## 2. Design

### 2.1 Files and boundaries

- **New `idsai/product/web/motion.js`** is loaded after `app.js`. It owns every animation: the background canvas, the CCTV camera, the radar, the pulses, the shockwave, the chart overlay and the page transition. It exposes one global `Motion` with a small API (§2.2). It reads nothing from the server and holds no app state.
- **New `idsai/product/web/motion.css`** holds the motion styles, keyframes and the new colour tokens. It is linked after `styles.css`.
- **`app.js`** only calls `Motion.*` at the points listed in §2.2, through a guard `const M = globalThis.Motion` that does nothing if it is missing. If `motion.js` fails to load, the app works exactly as today.
- **`index.html`** gains:
  - the background `<canvas id="fx-bg">`;
  - the CCTV camera SVG markup in the login hero;
  - the chart overlay `<canvas id="fx-chart">` inside the chart box;
  - the screen-edge flash element `<div id="fx-edge">`;
  - the two new tags `<link>` and `<script>`.

  All of them carry `aria-hidden="true"`.
- `reducedMotion()` stays in `app.js` and `motion.js` uses the same media query. The "ลดเอฟเฟกต์" switch is stored in the existing user prefs as `fx_off` and also in `localStorage` (try/catch), so it applies before sign-in.

### 2.2 `Motion` API (called by `app.js`)

| Call | When `app.js` calls it | Effect |
|---|---|---|
| `Motion.init({ off })` | Once at boot | Starts the background, or keeps it still if `off` or reduced motion is set |
| `Motion.setOff(bool)` | "ลดเอฟเฟกต์" toggled | Stops or starts everything at once |
| `Motion.mood("calm" \| "watch" \| "alert")` | Each time the hero status is rendered | Background, grid and radar colour: cyan, amber or red |
| `Motion.loginEnter()` / `Motion.loginLeave(done)` | Login page shown / sign-in succeeded | The CCTV scene starts; the "verified" transition runs, then `done()` |
| `Motion.alertBurst({ severity })` | A new alert arrives live (`onAlert`) | Edge flash once, red sweep on the newest card, and the hero switches to pulse rings |
| `Motion.chartAttach(api, series, { limitLine })` | After the main chart is created | Creates the overlay: the scan line at the latest point, the pulsing latest dot and the glow under the price line |
| `Motion.chartCross({ time, value })` | The newest anomaly value is above the limit and the previous one was not | Shockwave at that point on the overlay, plus a limit-line flash and shake |
| `Motion.pageEnter(node)` | After `route()` shows a page | A scan-in wipe, then the cards rise in one after another |

Detecting a crossing is `app.js`'s job, because it already knows the limit (`tauLine`) and the series. `motion.js` only draws. Coordinates come from Lightweight Charts' `timeScale().timeToCoordinate` and `series.priceToCoordinate`. If either returns null (off-screen), there is no shockwave and only the edge flash plays.

### 2.3 Look

- **Colour tokens** (in `motion.css`, added to the existing v3 tokens):
  - `--fx-calm: #22d3ee` (cyan);
  - `--fx-watch: #f59e0b` (amber);
  - `--fx-alert: #ef4444` (red);
  - `--fx-grid: rgba(56, 189, 248, .08)`.

  The base background moves from purple-violet to a deep navy, `#070b1a` to `#0b1230`. The existing purple accent stays on buttons so the brand still matches.
- **CCTV camera:**
  - an inline SVG of a wall-mounted dome or box camera with a blinking `REC` light;
  - it pans about ±18° on a 6–8 s ease-in-out loop;
  - the beam is a semi-transparent red cone (`--fx-alert`, with a soft gradient edge) attached to the lens, which rotates with the camera.

  On the dashboard, a small camera icon sits in the hero card and turns toward the chart when the mood is `alert`.
- **Radar** (hero, calm): a conic-gradient sweep inside the existing orb, about 4 s per turn. In `alert` it becomes 2–3 concentric red rings expanding and fading, staggered by about 1.2 s.
- **Background canvas:**
  - a perspective grid drifting slowly;
  - 30–60 small dots rising and fading;
  - a horizontal scan line every 8–12 s.

  It is drawn at most 30 fps. The device-pixel ratio is capped at 2. The loop pauses on `visibilitychange`, and on mobile widths below 640 px it draws half as many dots.
- **Cards:**
  - a thin glowing border light travels around the KPI cards about every 6 s, out of phase between cards;
  - numbers count with the existing `tweenNumber`;
  - hover lifts a card 2 px and brightens its border.
- **Chart overlay:**
  - a vertical scan line follows the latest bar;
  - the latest anomaly point has a pulsing ring;
  - the price area gets a soft gradient fill, using Lightweight Charts `AreaSeries` options when the series supports it and the overlay otherwise.
  - The shockwave on a crossing is two rings expanding to about 60 px over 900 ms, with a short flash. The limit line glows red and shakes about 2 px for 400 ms.
- **Login form:** the card slides and fades in; inputs get a glowing border on focus; the submit button has a light sheen passing over it.
- **Page transition:** a 300 ms cyan scan line wipes from top to bottom, then the cards rise in with a 40 ms stagger. It is skipped on the first paint.

### 2.4 Behaviour rules

- Motion is decoration only. It never moves layout, so it uses only `transform`, `opacity` and the canvas, and never shifts text the user is reading.
- No flashing faster than 3 times per second, a photosensitivity safety rule. The edge flash fires at most once per alert burst, and bursts are rate-limited to one per 2 s.
- When reduced motion is on or `fx_off` is set:
  - canvases are cleared and hidden;
  - CSS animations are set to `none` (in the existing `@media (prefers-reduced-motion)` block, plus a `.fx-off` class on `<html>`);
  - state colours still show the mood statically.
- Off by default during the guided tour steps and while the help drawer is open, so nothing distracts. Colours stay.
- All new copy is Thai and short: `ตรวจพบ`, `ยืนยันตัวตนแล้ว`, `ลดเอฟเฟกต์`. The existing copy-guard rules still apply.

### 2.5 Performance budget

- One `requestAnimationFrame` loop for all canvases, owned by `motion.js`. There are no per-element timers apart from CSS animations.
- `motion.js` should be at most about 25 KB unminified, and `motion.css` at most about 12 KB.
- No layout reads inside the animation loop. Chart coordinates are re-read only on the chart's own range-change events and on new bars.

## 3. Error handling

- If `Motion` is missing or throws, `app.js` catches the error at each call (the same guard) and carries on. Each `Motion` method wraps its own body in try/catch.
- A missing canvas context (old browser) leaves motion off; the page works.

## 4. Testing

**Static tests in `tests/product/test_web_static.py`:**
- `index.html` links `motion.css` and `motion.js` after `styles.css` and `app.js`.
- `#fx-bg`, `#fx-chart` and `#fx-edge` exist with `aria-hidden="true"`.
- The CCTV SVG exists in the login hero.
- `motion.js` defines every API method in §2.2.
- `app.js` calls `Motion.mood`, `alertBurst`, `chartCross`, `chartAttach`, `pageEnter`, `loginEnter` and `loginLeave` only through the guard.
- `motion.css` contains the four `--fx-*` tokens and a reduced-motion block, plus a `.fx-off` rule that sets animations to `none`.
- There is no `innerHTML` in `motion.js`.
- The brace balance is correct in both new files, and the existing copy guards pass. The new Thai strings are present.

**Logic tests:** the crossing check is a small pure function `crossedUp(prev, cur, limit)` in `app.js`. It returns true only when `prev <= limit < cur`, and false for null or NaN. A test pins its source text: it must exist and be the only caller of `Motion.chartCross`. Its behaviour is covered by the manual check below, since the repo has no JS test runner.

**Manual check in the browser (at desktop and phone width):**
- the login camera and beam hit;
- the sign-in transition;
- the calm background;
- a forced crossing shockwave;
- an alert burst;
- the page wipe;
- reduced motion and "ลดเอฟเฟกต์" give a static page;
- the hidden tab pauses the loop.

**Gates:** `uv run ruff check .` and `uv run pytest -q -m "not network and not slow"` pass.
