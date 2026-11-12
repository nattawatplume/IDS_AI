# HUD Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Restyle the whole dashboard and the login page into the approved flat-neon HUD look from the mockup.

**Architecture:**
- HUD tokens and the `.hud` panel system live in `styles.css`.
- Motion (eye logo, plexus, tilt, glint, wires, scramble, ticker) lives in `motion.js` and `motion.css` behind the existing `fx()` guard.
- Markup changes are in `index.html`, and the ticker data wiring is in `app.js`.

**Tech Stack:** vanilla JS, no build step, CSS, inline SVG, canvas 2D, Lightweight Charts (vendored), pytest static tests.

**Spec:** `docs/superpowers/specs/2026-11-11-hud-redesign-design.md`

**Visual reference:** `docs/superpowers/mockups/2026-11-11-hud-home.html`. Match its look and motion. Never copy its fake data or the wrong "IFOREST" label.

## Global Constraints

- **Palette, exact values:**
  - `--hud-bg: #05070d`, `--hud-panel: #0b1220`, `--hud-line: #00e5ff`, `--hud-violet: #7c3aed`, `--hud-alert: #ff2d55`, `--hud-ok: #22ff88`, `--hud-text: #e6f1ff`, `--hud-muted: #8aa0b8`.
  - The mood tokens map so that calm is `--hud-line`, watch is `#f59e0b` and alert is `--hud-alert`.
- **No gradients:** no `-gradient(` in `styles.css` or `motion.css`, except one allow-listed single-colour radar wedge. No gradient text or buttons, and no aurora or blobs.
- **Fonts:** local only, with no external font requests. Set `--mono: ui-monospace, "Cascadia Mono", Consolas, "JetBrains Mono", monospace` and keep `--font` for Thai.
- **Motion rules from the motion spec still hold:**
  - decoration only, using transform, opacity and canvas;
  - `aria-hidden` on all decoration;
  - reduced motion and `ลดเอฟเฟกต์` give a static page;
  - no flash faster than 3 per second, and bursts and shockwaves at most once per 2 s;
  - one rAF loop, DPR at most 2, paused when the tab is hidden;
  - no `innerHTML` in `motion.js`.
- **Ticker data:** it shows only real app values and `–` for anything missing. It never shows invented numbers.
- **Wording:** all existing Thai copy and the copy-guard tests stay. The new technical tags (`CAM-01`, `SECTOR`, `LIVE`, `REC`, `AUTH · SECURE CHANNEL`) are mono, and decorative ones are `aria-hidden`.
- **Size budgets:** at most 40 KB for `motion.js` and 24 KB for `motion.css`. Update the existing size tests.
- **Repo rules:**
  - commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`;
  - never `git add -f`; never commit `.claude/`, `.superpowers/` or `data/`; never push;
  - gates are `uv run ruff check .` and the full `uv run pytest -q -m "not network and not slow"` (about 1391 tests; report the exact count);
  - manual checks use a throwaway harness or the local replay, never left running.

## Review Focus

1. **A panel on a page other than Home is missed and still looks like the old rounded glass card.** Task 1 test `test_every_page_uses_hud` checks every page container in `index.html` and every JS-built card class.
2. **Contrast and readability on black.** Muted text must stay readable and alert red must not be used for normal text. Task 1 test: no `--hud-alert` in base text rules.
3. **Before data arrives, the ticker shows `–` and never `NaN` or `undefined`.** Task 2 test `test_ticker_reads_state`.
4. **3D tilt on touch devices, or while a user selects text.** Expected: no tilt on `pointer: coarse` and no tilt while dragging. Task 3 test pins `(pointer: fine)` gating.
5. **Login on a phone at 375 px.** The form must be fully visible and usable without scrolling past the decoration. This is checked in Task 5 by hand.

---

### Task 1: HUD tokens and panel system on every page

**Files:**
- Modify: `idsai/product/web/styles.css` (tokens, base, `.hud`, buttons, chips, nav) and `idsai/product/web/motion.css` (mood token mapping).
- Modify: `idsai/product/web/index.html` (add `.hud` and `.hud-head` to all page containers).
- Modify: `idsai/product/web/app.js` (JS-built cards get `.hud`).
- Test: `tests/product/test_web_static.py`.

**Interfaces:**
- Produces:
  - CSS classes `.hud` (chamfer `clip-path`, 1 px `--hud-line` border, four bracket pseudo-elements or spans) and `.hud-head`, which holds `.hud-tag` (mono label), `.hud-rec` (dot) and an optional `.hud-clock`;
  - the HUD tokens above.

  Later tasks rely on these exact class names.

- [ ] **Step 1: Write the failing tests:**
  - `test_hud_tokens`: the 8 tokens are present with exact values.
  - `test_no_gradients`: there is no `-gradient(` in `styles.css` or `motion.css` except a line marked `/* allow: radar wedge */`.
  - `test_every_page_uses_hud`: every `section.page` in `index.html` has its main containers with class `hud`, and the JS card builders (`alertCard`, the results and simulate cards, the users table wrapper) add `hud`.
  - `test_mono_stack`: `--mono` matches the exact stack.
  - `test_no_external_fonts`: there is no `fonts.googleapis` in `index.html` or the CSS.
  - `test_alert_colour_not_base_text`.
  - Update any existing tests that pinned the old tokens or the purple glass colours.
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - Replace the old purple glass tokens and remove every gradient fill. Keep the layout working on every page.
  - Draw the corner brackets with pseudo-elements or spans, sliding outward on hover (transform only).
  - Buttons are flat or outline neon with a hover glow. Chips use the same style.
  - The nav active item gets an underline with bracket ends and a glow; the sliding motion comes in Task 2.
- [ ] **Step 4: Run** the full gate. Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat(web): HUD tokens and chamfered neon panels on every page; no gradients`.

### Task 2: Header (ticker, CCTV-eye logo, sliding nav)

**Files:**
- Modify: `index.html` (header: ticker strip and logo SVG), `motion.js`, `motion.css`.
- Modify: `app.js` (`function tickerItems()` returns `[{ label, value }]` from state, refreshed when prices or health change).
- Test: `tests/product/test_web_static.py`.

**Interfaces:**
- Consumes: the Task 1 classes.
- Produces:
  - `tickerItems() -> Array<{label: string, value: string}>`. Missing values become `"–"`.
  - `Motion.ticker(items)` renders and loops the marquee with text nodes only.
  - `Motion.mood("alert")` also turns the logo eye red and makes it stare.

- [ ] **Step 1: Write the failing tests:**
  - `test_logo_eye_svg`: `svg#logo-eye` in the header has `aria-hidden="true"`, an iris element and a scan-ring element.
  - `test_ticker_reads_state`: `app.js` has `function tickerItems(` that reads state, the ticker markup exists, and there are no literal price-looking numbers in the ticker markup.
  - `test_nav_slider`: the nav has an indicator element moved with transform.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement,** following the mockup:
  - the iris drifts and blinks;
  - the scan ring rotates;
  - `MARKET IDS` is typed out once with a cursor;
  - in alert mood the eye is red and staring;
  - the ticker is a seamless marquee, static when off;
  - the nav indicator slides to the active item on route change.
- [ ] **Step 4: Run** the full gate. Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat(web): CCTV-eye logo, live data ticker, sliding nav indicator`.

### Task 3: Home layout, panel tilt and glint, wires, number scramble, hero glitch

**Files:**
- Modify: `index.html` (Home grid structure), `styles.css` (Home grid), `motion.js`, `motion.css`.
- Modify: `app.js` (number updates go through `fx("scramble", node, text)` with a plain-text fallback).
- Test: `tests/product/test_web_static.py`.

**Interfaces:**
- Produces:
  - `Motion.scramble(node: Element, text: string)`, which ends exactly on `text` (or sets it at once when off);
  - `Motion.tilt` setup on `.hud` elements, gated by `matchMedia("(pointer: fine)")`;
  - the wires layer.

- [ ] **Step 1: Write the failing tests:**
  - `test_home_grid_asymmetric`: the Home grid has the chart in the centre column (check the grid-template-areas names or the column order in CSS).
  - `test_scramble_api`: `motion.js` defines `scramble`, `app.js` calls `fx("scramble"` for the KPI numbers, and the fallback sets `textContent`.
  - `test_tilt_pointer_fine`: the `(pointer: fine)` gating is present.
  - `test_wires_hidden_decoration`: the wires element has `aria-hidden="true"` and is hidden when off.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement,** following the mockup:
  - asymmetric layout: 3, 2 or 1 columns;
  - tilt at most 6° (2.5° on the chart), reset on leave, no tilt while dragging;
  - a glint every few seconds, out of phase between panels;
  - wires with travelling pulses, recomputed on resize, never in the draw loop;
  - the scramble takes about 400 ms;
  - the hero glitch fires once when the mood changes to alert.
- [ ] **Step 4: Run** the full gate. Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat(web): asymmetric HUD home, panel tilt and glint, data wires, number scramble, alert glitch`.

### Task 4: Plexus background and neon chart

**Files:**
- Modify: `motion.js` (replace the grid layer with plexus and add the CRT overlay), `motion.css`, `app.js` (chart series colours from the HUD tokens).
- Test: `tests/product/test_web_static.py`.

**Interfaces:**
- Consumes: the existing shared loop and chart overlay.
- Produces: the plexus layer, 40–70 dots (half below 640 px), mouse bend within 120 px.

- [ ] **Step 1: Write the failing tests:**
  - `test_plexus_layer`: `motion.js` contains the plexus constants (`PLEX_MIN = 40`, `PLEX_MAX = 70`, `PLEX_BEND = 120`) and no longer draws the old perspective grid.
  - `test_chart_hud_colours`: the chart series options in `app.js` use the HUD tokens through `cssVar`: line for price, violet for score, alert above the limit.
  - `test_chart_draw_in`: the overlay has a reveal on first load that is skipped when off.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement,** following the mockup. Keep the shockwave and scan line. The above-limit segments are drawn red on the overlay or through series colour changes, whichever is cleaner with Lightweight Charts.
- [ ] **Step 4: Run** the full gate. Expected: PASS.
- [ ] **Step 5: Commit** with the message `feat(web): neon plexus background and HUD chart colours with draw-in`.

### Task 5: HUD login page and final browser check

**Files:**
- Modify: `index.html` (the login page markup inside the HUD panel), `styles.css`, `motion.css`, `motion.js` (typed tagline, flat red beam).
- Test: `tests/product/test_web_static.py`.

- [ ] **Step 1: Write the failing tests:**
  - `test_login_hud`: `#page-login` uses `.hud` with a header tag `AUTH · SECURE CHANNEL`; the login and sign-up forms keep all their existing ids; the beam has no gradient; the eye logo is present on login.
  - `test_login_typed_tagline`: the tagline text is unchanged, and the typing is done by `motion.js` with an off fallback that shows the full text.
- [ ] **Step 2: Run** them. Expected: FAIL.
- [ ] **Step 3: Implement,** following the spec:
  - the camera scene restyled to flat neon;
  - the typed tagline;
  - HUD inputs and buttons;
  - errors in `--hud-alert`;
  - the "ยืนยันตัวตนแล้ว" transition kept;
  - at 375 px the form comes first and the decoration is reduced.
- [ ] **Step 4: Check by hand** at 1280 px and 375 px, in local replay or a throwaway harness:
  - login and sign-up;
  - Home in calm and alert (inject a red-team scenario);
  - every other page;
  - off mode and reduced motion;
  - the ticker before and after data arrives.

  Stop the server afterwards.
- [ ] **Step 5: Run** the full gate. Expected: PASS.
- [ ] **Step 6: Commit** with the message `feat(web): HUD login page with neon CCTV scene and typed tagline`.
