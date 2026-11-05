# Dashboard redesign: "Trading desk" look with motion

Date: 2026-11-05 · Status: approved in chat (look A), spec awaiting review

## 1. Why

The Task 19 dashboard works, but its main audience is teachers and a committee watching on a projector. To them it reads as a template:
- capsule tabs and rounded cards everywhere;
- long paragraphs of light grey text under the chart and on Eval;
- no clear order of importance;
- many-colour chart lines.

The goal is a calm, professional surveillance-desk look where the main numbers are seen first, plus motion that helps the eye follow what changed.

Success means:
- On each page, a viewer finds the main number in under 3 seconds, on a projector.
- No paragraph of explanation shows by default. Explanations open on demand (ⓘ) or live on About.
- The look reads as a trading or SOC desk, not a generic app.
- Nothing about detection, the API or any reported number changes.

## 2. Scope

**In scope:**
- `idsai/product/web/index.html`, `app.js` and `styles.css`.
- `tests/product/test_web_static.py`: update the copy and structure tests, and add new ones.

**Out of scope:**
- API, WebSocket types, detection and eval code: none of it changes.
- Login and user accounts: a separate spec, `2026-11-05-auth-design.md`, still to be written.
- Hosting: a separate task after auth.

**Constraints kept from Task 19:**
- No build step and no network loads; the existing test guards this.
- System fonts only, light and dark themes, and no horizontal scroll at 360 px.
- `el()` builds only known tags; never parse HTML strings.
- The vendored lightweight-charts 5.2.1 stays as is.

## 3. Visual system

**Tokens on `:root`:**
- Dark is the default identity. Light is a full equal palette for bright rooms.
- Themes follow `prefers-color-scheme`, plus a manual toggle in the top bar. The choice is kept in `localStorage` and also works without it.

| token | dark | light | role |
|---|---|---|---|
| `--bg` | `#0d1014` | `#eef1f5` | page |
| `--panel` | `#13171d` | `#ffffff` | raised areas |
| `--line` | `#232a33` | `#d5dbe3` | 1 px hairlines |
| `--ink` | `#e6e9ee` | `#141a22` | text |
| `--dim` | `#7c8796` | `#5b6675` | labels |
| `--amber` | `#f2a93b` | `#b86e00` | the one accent: alerts, active tab, F line |
| `--red` | `#ff5d52` | `#c2271d` | τ line, over budget, high severity |
| `--green` | `#3fcf8e` | `#1d8a57` | healthy |

- Chart lines: price uses `--ink` at 80 %; L1, L2 and L3 use grey steps of `--dim`. Only F, τ and the markers carry colour.
- **Type:**
  - Text uses the system stack (Segoe UI, Leelawadee UI, Noto Sans Thai and so on).
  - Every number uses the mono stack with `tabular-nums`.
  - Scale: 11 / 13 / 15 / 20 / 28 px.
  - Section labels: 11 px, uppercase, letter-spacing 0.08 em, `--dim`.
- **Frames:**
  - Use hairline dividers instead of cards. Radius is 4 px at most, with no shadows.
  - Tabs are text with a 2 px amber underline, not capsules.
  - Badges (REPLAY, LIVE, REAL, SEMI-SYN) are square-ish outline tags.

## 4. Motion

All motion is CSS transitions or keyframes, or `requestAnimationFrame` for number tweens. Everything is disabled under `prefers-reduced-motion: reduce`, where values just snap.

| where | what | timing |
|---|---|---|
| top bar | live dot pulses while events arrive; stops when paused or stale | 1.6 s loop |
| top bar | budget meter width eases; amber at 80 % or more, red above 100 % | 400 ms |
| tabs | amber underline slides to the active tab; page content fades and rises 4 px | 150 ms |
| KPI strip | numbers tween to the new value; a rising alert count flashes amber once | 300 ms / 600 ms |
| alert rail | a new alert slides in at the top; its amber left edge glows, then settles | 250 ms + 1.5 s |
| chart | new marker pops (scale 0.6 → 1); alert window shading fades in | 200 ms |
| chart | crosshair synced across price, F and layer lanes, with values in a hover readout | live |
| Red-Team | elapsed ring runs while waiting; detection gives an amber sweep banner "ตรวจพบใน N วินาที"; no detection gives a plain grey "ไม่พบภายใน N วินาที" | ring 1 s steps |
| Eval | target cards rise in one after another; bars grow from 0 on first open of the page | 60 ms stagger / 500 ms |

**Presenter keys** (ignored while typing in a field):
- Space plays or pauses replay.
- 1–6 switch pages.
- `t` toggles the theme.

## 5. Pages

- **Top bar:**
  - Left: `MARKET·IDS`, then the mode tag.
  - Middle: the tab row.
  - Right: event clock with the live dot, Auditor dot, budget meter and theme toggle.
  - Warm-up text moves into the Auditor tooltip.
- **Monitor:**
  - KPI strip: symbol, F now / τ, alerts in 24 h against the budget, Auditor.
  - Below it, a two-column body:
    - Chart on the left: price, then F with τ, then a collapsed "L1 · L2 · L3" lane that expands on click.
    - Recent alerts rail on the right, last 8, each linking to its detail.
  - Replay controls form one compact row above the chart.
  - The chart legend paragraph becomes an ⓘ popover.
- **Alerts:**
  - Dense table with a severity stripe on each row's left edge, and a row hover highlight.
  - The detail page uses two columns: chart and evidence on the left; the Thai explanation, detector p-values, technique card and TP/FP/unsure buttons on the right.
- **Red-Team:**
  - The form is one row: family, region, seconds back, symbol, then Inject.
  - The timer ring and result banner are the page's focal point.
  - The LIVE-mode notice stays.
- **Eval:**
  - The page opens with target cards, one per Task 20 target, each marked pass, miss or not measured. Each shows its value, CI and n read from `report.json`, never hard-coded.
  - Next comes the AE − baseline forest plot from the paired data, then the retrain chart, then the full tables under a budget selector.
  - Long notes collapse into a "หมายเหตุวิธีวัด" disclosure.
  - Fix the tall rows in the ablation table: the notes cell gets a max width and wraps.
- **Taxonomy / About:** re-typeset only, same content. About gains the explanations removed from the other pages.

## 6. Data and errors

- The existing loaders stay unchanged; target cards derive from fields already in `report.json`.
- The thresholds the pass/miss rules compare against, such as recall 0.8 at budget 2, sit in one constant at the top of `app.js`, with a comment pointing to the plan.
- If `report.json` has no runs, Eval shows one line saying to run `uv run idsai report`.
- If the WebSocket drops, the live dot turns grey and the clock dims. The existing reconnect logic stays.

## 7. Testing

- **`test_web_static.py`:** keep every existing test, adjusting only those that assert old copy or old class names. Add tests that check:
  - the reduced-motion rule exists;
  - the theme toggle exists;
  - presenter keys skip inputs;
  - the Eval target constants are present;
  - no new network loads.
- **Browser check** with the preview tools on `idsai serve --mode replay`, for each page:
  - dark and light;
  - 1280 px and 375 px wide;
  - no console errors;
  - inject a burst, then confirm the alert rail animates and Red-Team shows its result;
  - Eval numbers still match `docs/REPORT.md`.
- Full pytest suite stays green.

## 8. Order of work

1. Tokens, type and frame in `styles.css`, plus the top-bar markup.
2. Motion layer: CSS keyframes, the tween helper and the reduced-motion guard.
3. Monitor.
4. Alerts and alert detail.
5. Red-Team.
6. Eval.
7. Taxonomy and About.
8. Tests and browser verification.
