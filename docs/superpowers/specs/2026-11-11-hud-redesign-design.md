# HUD redesign: flat neon panels, CCTV-eye logo, plexus background, HUD login

Date: 2026-11-11. Status: owner approved the mockup ("ชอบๆ ทำจริงได้เลย ขอหน้า login เท่ๆงี้ด้วยนะ").

This builds on `2026-11-11-command-center-motion-design.md`. `motion.js` and `motion.css`, the `fx()` guard, the off switches and the flash limits all stay. This spec replaces the look.

**Visual reference (binding):** `docs/superpowers/mockups/2026-11-11-hud-home.html`. It is a static mockup on fake data. Copy its look and motion, not its fake numbers or fake labels. Its "IFOREST" label is wrong: the model is an autoencoder with L1/L2/L3 layers and the Chief.

## 1. Owner feedback this answers

- The old design was plain rounded boxes in a grid, with no visual flair.
- Purple-to-blue gradients made it look AI-made, so there are no gradient fills anywhere.
- The top-left logo was dull.
- The owner wants it modern, professional and full of visible motion, with a login page in the same style.

## 2. Design

### 2.1 Palette (flat neon, no gradients)

| Token | Value | Use |
|---|---|---|
| `--hud-bg` | `#05070d` | Page |
| `--hud-panel` | `#0b1220` | Panel fill (solid, may be slightly translucent) |
| `--hud-line` | `#00e5ff` | Primary accent and lines |
| `--hud-violet` | `#7c3aed` | Secondary series only |
| `--hud-alert` | `#ff2d55` | Alert states only |
| `--hud-ok` | `#22ff88` | Normal and OK status only |
| `--hud-text` | `#e6f1ff` | Main text |
| `--hud-muted` | `#8aa0b8` | Muted text |

- **No gradients:** no `linear-gradient`, `radial-gradient` or `conic-gradient` fills, no gradient text, no gradient buttons, no aurora or blob backgrounds. Two exceptions:
  - single-colour glow through `box-shadow` or `drop-shadow`;
  - a radar sweep, which may use one conic wedge of a single colour.
- **The `--fx-*` mood tokens** map to the HUD colours: calm uses `--hud-line`, watch uses `#f59e0b`, alert uses `--hud-alert`.
- **Fonts:** use local font stacks only, with no external font requests. Thai text uses the existing `--font`. Numbers and labels use `--mono: ui-monospace, "Cascadia Mono", Consolas, "JetBrains Mono", monospace`.

### 2.2 HUD panel system (all pages)

- **One panel class `.hud`,** applied to every card-like container on every page:
  - chamfered corners through `clip-path`;
  - a 1 px `--hud-line` border;
  - four corner brackets that slide outward on hover.
- **Header strip:** every panel gets `<div class="hud-head">`. It contains:
  - a mono label like `CAM-01 · SECTOR A · LIVE`, with each panel's own label and number;
  - a REC dot blinking at no more than 1 Hz;
  - a small live clock, but only on panels that have live data.
- **Panel motion:**
  - a 3D tilt that follows the mouse, at most 6° on small panels and 2.5° on the chart;
  - a light glint along the border every few seconds.
- **Connector wires:** thin lines join the Home panels, with small pulses travelling along them. They are decoration only and use a single SVG or canvas.
- **Buttons:** outline or flat solid. They glow on hover and show a ripple on click. Chips such as the speed buttons get the same treatment.
- **Nav:** the active item is marked by a sliding underline with bracket ends and a glow, not a pill.

### 2.3 Header

- **Ticker strip** at the very top, scrolling as a marquee. It shows only real values the app already has:
  - each known symbol with its last price, where available;
  - the feed and mode (`REPLAY` or `LIVE`);
  - health;
  - today's alerts.

  Missing values are shown as `–`. It never shows invented numbers.
- **Logo, an animated CCTV eye** in inline SVG:
  - the iris drifts left and right and blinks;
  - a thin scan ring rotates around it;
  - `MARKET IDS` types itself out with a blinking cursor on first load;
  - in alert mood the eye turns red and stares.

### 2.4 Home layout

- **The chart is the big centre screen.** On the left are the status hero (with the radar and a score meter) and the stat panels. On the right are the recent alerts and replay controls.
- **Layout by width:** 3 columns on desktop, 2 on tablet, 1 below 640 px.
- **Number scramble:** big numbers flicker through random glyphs, then settle on the real value when they change. Under reduced motion or `fx_off` they snap straight to the value.
- **Alert glitch:** in alert mood the hero title glitches once to red and shows the existing alert wording.

### 2.5 Background and chart

- **Background canvas:** the grid layer from the earlier spec is replaced by a cyan plexus.
  - Dots drift, lines join near neighbours and fade with distance, and lines bend toward the mouse within 120 px.
  - There are 40–70 dots, or half that below 640 px, and a faint CRT scanline overlay.
  - The existing shared loop, DPR cap and pause-when-hidden rules apply.
- **Chart colours:** price is `--hud-line`, the anomaly score is `--hud-violet`, and a score above the limit is `--hud-alert`. The limit line is dashed, and the lines glow.
- **Chart load:** on first load the lines draw themselves in through the overlay reveal. The scan line, pulse and shockwave from Task 4 of the earlier spec stay.

### 2.6 Login page (HUD)

- **Layout:** same palette and panel system, with the plexus background.
- **Left side:**
  - the CCTV camera scene from the earlier spec, restyled to flat neon with no gradient beam fill; the beam is one translucent red colour;
  - the animated eye logo;
  - the tagline typed out terminal-style.
- **Right side:** the login and sign-up forms inside one HUD panel with a header like `AUTH · SECURE CHANNEL`.
  - Inputs are underlined or outlined in `--hud-line`.
  - The submit button is flat neon.
  - Errors show in `--hud-alert`.
- **Sign-in:** the "ยืนยันตัวตนแล้ว" transition stays.

## 3. Rules kept

- **Motion stays decoration only:** it uses transform, opacity and canvas, does not change words or numbers, and every decoration is `aria-hidden`.
- **Off and flash rules:** reduced motion and `ลดเอฟเฟกต์` give a static page. No flash faster than 3 per second, and bursts and shockwaves stay rate-limited to once per 2 s.
- **Wording:** all existing Thai wording and copy guards stay. The new labels are mono technical tags (`CAM-01`, `SECTOR`, `LIVE`, `REC`, `AUTH · SECURE CHANNEL`) and are `aria-hidden` where they are decorative.
- **No new library.** The size budgets become 40 KB for `motion.js` and 24 KB for `motion.css`, both unminified.

## 4. Testing

- **Static tests:**
  - no `-gradient(` in `styles.css` or `motion.css`, apart from one allow-listed radar wedge;
  - the HUD tokens are present;
  - `.hud` is applied to the main containers on every page;
  - the logo SVG exists and is `aria-hidden`;
  - the ticker exists and reads from state, not hard-coded numbers;
  - the plexus layer exists;
  - the login page uses `.hud`;
  - the existing tests still pass, updated where a markup change is intended.
- **Manual browser check** at 1280 px and 375 px: Home, every other page, login and sign-up, alert mood, and off mode.
- **Gates:** `uv run ruff check .` and `uv run pytest -q -m "not network and not slow"`.
