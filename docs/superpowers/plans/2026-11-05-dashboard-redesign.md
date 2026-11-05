# Dashboard Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-skin the SOC dashboard in the "Trading desk" look with purposeful motion, without changing the API, detection or any reported number.

**Architecture:** The change is confined to the three static files in `idsai/product/web/`. Design tokens live on `:root` in `styles.css`. A small motion layer (CSS keyframes plus two JS helpers) is shared by all pages. Each page's render function in `app.js` is reshaped in place. Pure logic, such as Eval target verdicts and key handling, is written as top-level functions so the existing node-based tests can load `app.js` and call them.

**Tech Stack:** Plain HTML/CSS/JS, no build step; vendored lightweight-charts 5.2.1; pytest with `tests/product/test_web_static.py` (static checks plus `node` when installed).

**Spec:** `docs/superpowers/specs/2026-11-05-dashboard-redesign-design.md`

## Global Constraints

- Change only `idsai/product/web/{index.html,app.js,styles.css}` and `tests/product/test_web_static.py`. No API, WS, detection or eval change.
- No build step and no network loads. `test_page_loads_nothing_from_the_network` must stay green.
- System fonts only:
  - text: `"Segoe UI", "Leelawadee UI", "Noto Sans Thai", -apple-system, Roboto, sans-serif`
  - numbers: `ui-monospace, "SF Mono", Consolas, monospace` with `font-variant-numeric: tabular-nums`
- `el()` builds only tags in `HTML_TAGS`, or the test's list extended deliberately. No `innerHTML`, `insertAdjacentHTML` or `eval(`.
- Tokens, with dark first and light second:
  - `--bg #0d1014/#eef1f5`
  - `--panel #13171d/#ffffff`
  - `--line #232a33/#d5dbe3`
  - `--ink #e6e9ee/#141a22`
  - `--dim #7c8796/#5b6675`
  - `--amber #f2a93b/#b86e00`
  - `--red #ff5d52/#c2271d`
  - `--green #3fcf8e/#1d8a57`
- Radius at most 4 px, no box-shadow, 1 px hairlines, and tabs as text with a 2 px amber underline.
- Type scale 11 / 13 / 15 / 20 / 28 px. Section labels are 11 px, uppercase, letter-spacing 0.08 em.
- All motion is disabled under `@media (prefers-reduced-motion: reduce)`.
- Keep these words: "ความผิดปกติทางสถิติ", "ค่า p รวม", "ความหายากเทียบตลาดปกติ", "prevalence", and the badges REAL / SEMI-SYN / SYN. Keep every hash route.
- Works at 360 px wide with no horizontal page scroll. Light and dark both follow `prefers-color-scheme`, plus the manual toggle.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Light theme on a projector.** Amber on `#eef1f5` must stay readable, so light amber is `#b86e00`, not the dark value. A test asserts both palettes define every token (Task 1).
2. **Reduced motion.** With it on, numbers must still update; they snap instead of tweening. A node test calls `tweenNumber` with reduced motion on and checks the text is final at once (Task 2).
3. **Typing in a field.** Pressing Space or 1–6 in the Red-Team seconds field must not play replay or switch page. A node test checks `presenterKey` ignores INPUT, SELECT and TEXTAREA targets (Task 2).
4. **Empty or partial `report.json`.**
   - With `{"runs": 0}`, Eval shows one line, not broken cards.
   - A target whose metric is missing shows "ยังไม่ได้วัด", never "ผ่าน".
   - A node test covers both (Task 6).
5. **WebSocket drop.** The live dot must stop pulsing and turn grey, so a frozen screen never looks live. A static test checks `setLive(false)` is called in the WS close handler (Task 3).

---

### Task 1: Visual system, top bar and theme toggle

**Files:**
- Modify: `idsai/product/web/styles.css` (rewrite the tokens and base rules)
- Modify: `idsai/product/web/index.html:11-50` (top bar and tabs)
- Modify: `idsai/product/web/app.js` (theme functions, `chartColors()` reads the new tokens)
- Test: `tests/product/test_web_static.py`

**Interfaces:**
- Produces:
  - `applyTheme(choice: "auto"|"dark"|"light") -> void` sets `document.documentElement.dataset.theme` (removes the attribute for `auto`), calls `restyleCharts()`, and stores the choice under the key `"theme"` via `storageSet`.
  - `currentTheme() -> string`
  - Element ids: `#theme-toggle`, `#live-dot`, `#clock`, `#auditor-dot`, `#budget-meter`.
  - CSS tokens as listed in Global Constraints. Lines: `--line-price` (= ink at 80 %), `--line-f` (= amber), `--line-l1/2/3` (dim steps), `--tau` (= red).

- [ ] **Step 1: Write failing tests**

```python
def test_both_themes_define_every_token():
    css = text("styles.css")
    tokens = ["--bg", "--panel", "--line", "--ink", "--dim", "--amber", "--red", "--green"]
    blocks = [css.split(":root[data-theme=\"dark\"]")[1], css.split(":root[data-theme=\"light\"]")[1]]
    for t in tokens:
        assert re.search(rf"{t}\s*:", css.split("@media")[0]), t  # bare :root
        for b in blocks:
            assert re.search(rf"{t}\s*:", b.split("}")[0]), t

def test_theme_toggle_exists_and_is_wired():
    assert 'id="theme-toggle"' in text("index.html")
    js = text("app.js")
    assert "function applyTheme(" in js and 'storageSet("theme"' in js

def test_no_capsule_tabs_or_shadows():
    css = text("styles.css")
    assert "box-shadow" not in css
    assert not re.search(r"border-radius:\s*(?:[5-9]|\d{2,})px", css)
```

- [ ] **Step 2: Run** `uv run pytest tests/product/test_web_static.py -k "theme or capsule" -v`. Expected: 3 FAIL.
- [ ] **Step 3: Implement.** Do the following:
  - Rewrite `styles.css` tokens:
    - bare `:root` holds the dark palette;
    - `@media (prefers-color-scheme: light) { :root:not([data-theme="dark"]) {…} }` holds the light one;
    - also add the explicit `:root[data-theme="dark"]` and `:root[data-theme="light"]` blocks.
  - Add base rules for type, the hairline frame, tag badges and tab underline. Keep every class that `app.js` still uses.
  - Top-bar markup:
    - brand `MARKET·IDS`, then the mode tag, then the tabs;
    - right group: clock with `#live-dot`, `#auditor-dot`, budget meter, and `#theme-toggle` (a `button` cycling auto → dark → light).
  - On load, read the stored theme with `storageGet("theme")`.
- [ ] **Step 4: Run** `uv run pytest tests/product/test_web_static.py -v`. Expected: all PASS. Adjust only old-class assertions that the restyle legitimately renamed.
- [ ] **Step 5: Commit** `feat(web): trading-desk tokens, top bar and theme toggle`

### Task 2: Motion layer and presenter keys

**Files:**
- Modify: `idsai/product/web/styles.css` (keyframes `pulse`, `glow`, `pop`, `rise`, `sweep`, `grow`; the reduced-motion block)
- Modify: `idsai/product/web/app.js` (helpers plus key handler)
- Test: `tests/product/test_web_static.py`

**Interfaces:**
- Produces:
  - `reducedMotion() -> boolean` reads `matchMedia("(prefers-reduced-motion: reduce)")`.
  - `tweenNumber(node, to: number, digits: number, ms = 300) -> void` writes the final `fmtNum(to, digits)` immediately when `reducedMotion()` is true; otherwise it uses rAF from the node's current numeric text.
  - `flash(node, cls = "flash") -> void` re-triggers a one-shot CSS class.
  - `presenterKey(ev) -> string|null` returns `"toggle-play"`, `"page:<n>"`, `"theme"` or `null`. It returns null when `ev.target.tagName` is INPUT, SELECT or TEXTAREA, or when ctrl, meta or alt is pressed.
  - `PAGE_ORDER = ["#/monitor","#/alerts","#/redteam","#/eval","#/taxonomy","#/about"]`

- [ ] **Step 1: Write failing tests.** Two node tests in the style of `test_time_to_detect_counts_only_new_alerts`. Each script loads `app.js` with stubbed `document` and `matchMedia`, the same way `TTD_JS` does.

```python
def test_presenter_keys_skip_fields(tmp_path):
    # script asserts:
    #   presenterKey({key:" ", target:{tagName:"INPUT"}}) === null
    #   presenterKey({key:" ", target:{tagName:"DIV"}}) === "toggle-play"
    #   presenterKey({key:"4", target:{tagName:"BODY"}}) === "page:4"
    #   presenterKey({key:"t", ctrlKey:true, target:{tagName:"BODY"}}) === null
    ...

def test_tween_snaps_under_reduced_motion(tmp_path):
    # matchMedia stub returns matches:true; node = {textContent:"0"}
    # tweenNumber(node, 3.87, 2) -> node.textContent === "3.87" synchronously
    ...

def test_reduced_motion_disables_animation():
    css = text("styles.css")
    block = css.split("prefers-reduced-motion: reduce")[1]
    assert "animation" in block and "transition" in block
```

- [ ] **Step 2: Run** `uv run pytest tests/product/test_web_static.py -k "presenter or tween or reduced" -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the helpers and keyframes, and bind `keydown` on `document` to dispatch `presenterKey` results:
  - toggle-play → `control("play"|"pause")`, which is a no-op in LIVE;
  - page → `location.hash = PAGE_ORDER[n-1]`;
  - theme → cycle `applyTheme`.
- [ ] **Step 4: Run** the whole file. Expected: PASS.
- [ ] **Step 5: Commit** `feat(web): motion layer and presenter keys`

### Task 3: Monitor page

**Files:**
- Modify: `idsai/product/web/index.html:51-90`
- Modify: `idsai/product/web/app.js`: `renderHealth`, `renderBudget`, `initChart`, `renderMarkers`, `onAlert`, `onScore`, WS handlers
- Test: `tests/product/test_web_static.py`

**Interfaces:**
- Consumes: `tweenNumber` and `flash` (Task 2), and the tokens (Task 1).
- Produces:
  - KPI ids: `#kpi-symbol`, `#kpi-f`, `#kpi-tau`, `#kpi-alerts`, `#kpi-budget`, `#kpi-auditor`.
  - Rail: `#alert-rail`, holding at most 8 newest `a` rows that link to `alertHref(id)`. A new row gets the class `enter`.
  - Layer lane: `#layers-toggle` (a button) expands or collapses the L1/L2/L3 panes.
  - Info: `#chart-info` (a button) toggles `#chart-info-body`, which holds the old legend paragraph text.
  - `setLive(on: boolean) -> void` toggles the `live` class on `#live-dot` and `dim` on `#clock`.

- [ ] **Step 1: Write failing tests**

```python
def test_monitor_has_kpi_strip_rail_and_info():
    html = text("index.html")
    for i in ["kpi-symbol","kpi-f","kpi-tau","kpi-alerts","kpi-budget","kpi-auditor",
              "alert-rail","layers-toggle","chart-info","chart-info-body"]:
        assert f'id="{i}"' in html, i

def test_ws_close_turns_live_dot_off():
    js = text("app.js")
    close = js.split("onclose")[1][:400]
    assert "setLive(false)" in close
```

- [ ] **Step 2: Run** `-k "monitor or live_dot"`. Expected: FAIL.
- [ ] **Step 3: Implement** the following:
  - KPI updates go through `tweenNumber`. When the alert count rises, `flash` `#kpi-alerts`.
  - The rail rebuilds from `sortedAlerts()` on every `onAlert`.
  - The budget meter width uses a CSS transition. Classes: `warn` at 0.8 of the budget or more, `over` above 1.0.
  - Chart colours: price `--line-price`, F `--line-f`, τ `--tau`, layers in dim steps.
  - Crosshair sync: subscribe `subscribeCrosshairMove` on each chart and call `setCrosshairPosition` on the others. A readout line `#readout` shows time, price and F.
  - Markers use `--amber`, or `--red` for high severity.
  - The `#chart-info-body` text keeps the legend sentence, including "ค่า p รวม / ความหายากเทียบตลาดปกติ".
  - `setLive(true)` on each message, `setLive(false)` in `onclose` and on pause.
- [ ] **Step 4: Run** the whole file. Expected: PASS, including `test_every_element_app_js_targets_exists`.
- [ ] **Step 5: Commit** `feat(web): monitor with kpi strip, alert rail and synced crosshair`

### Task 4: Alerts list and alert detail

**Files:**
- Modify: `idsai/product/web/index.html:91-111`
- Modify: `idsai/product/web/app.js`: `renderAlertsTable`, `showAlert`, `techniqueCard`, `feedbackCard`, `drawDetailChart`
- Test: `tests/product/test_web_static.py`

**Interfaces:**
- Produces:
  - Rows carry `data-sev="low|mid|high"`, which drives a left stripe through CSS.
  - The detail layout is `.detail` with `.detail-main` (chart and evidence) and `.detail-side` (explanation, p-values, technique, feedback).

- [ ] **Step 1: Write the failing test** `test_alert_rows_carry_severity_and_detail_has_two_columns`. Assert `data-sev` is set in `renderAlertsTable`, and that `"detail-main"` and `"detail-side"` both appear in `app.js`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Rows get a hover state. Under 760 px the detail stacks into one column. The feedback buttons and their behaviour stay unchanged.
- [ ] **Step 4: Run** the whole file. Expected: PASS.
- [ ] **Step 5: Commit** `feat(web): dense alerts table and two-column alert detail`

### Task 5: Red-Team timer ring and result banner

**Files:**
- Modify: `idsai/product/web/index.html:112-149`
- Modify: `idsai/product/web/app.js`: `renderInjections`, `tickInjections`
- Test: `tests/product/test_web_static.py`

**Interfaces:**
- Consumes: `injectionStatus(run, nowUs)` and `firstDetection`, both unchanged.
- Produces:
  - `#rt-ring` is an SVG circle whose `stroke-dashoffset` follows elapsed / window.
  - `#rt-banner` gets the class `hit` with the text `ตรวจพบใน N วินาที`, or `miss` with `ไม่พบภายใน N วินาที`.

- [ ] **Step 1: Write the failing test.** Add a node test, `test_banner_text_for_hit_and_miss`: give it a run with a detection after 42 s and a run that timed out after 300 s, and assert the two texts exactly.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
  - The form becomes one row (family, region, seconds back, symbol, Inject).
  - Expose `bannerText(status, seconds) -> string` for the test.
  - `svg()` already exists for the ring.
  - The LIVE notice and fail-closed behaviour stay.
- [ ] **Step 4: Run** the whole file. Expected: PASS, including `test_live_mode_gates_replay_and_red_team` and `test_red_team_form_starts_disabled_until_health_confirms_the_mode`.
- [ ] **Step 5: Commit** `feat(web): red-team timer ring and result banner`

### Task 6: Eval: target cards, forest plot and collapsed notes

**Files:**
- Modify: `idsai/product/web/app.js`: `showEval`, `experimentTable`, `pairedCard`, plus new functions
- Modify: `idsai/product/web/styles.css`
- Test: `tests/product/test_web_static.py`

**Interfaces:**
- Consumes: `report.json` fields already read by `showEval`. `test_report_fields_read_by_app_js_exist` guards any new field read.
- Produces:
  - `EVAL_TARGETS`: a top-level const array of `{id, label, experiment, variant, metric, budget, cmp, threshold}`. Comment: `// targets: docs/superpowers/plans/2026-10-27-market-ids-mvp.md Task 20 Step 6`. Entries:
    - recall_any ≥ 0.8 for real_t001/full;
    - recall_any ≥ 0.8 for semisyn pump_burst, pump_stealth and wash_pingpong;
    - FA ≤ budget for real_t001/full;
    - FA hard ≤ 0 for real_t001/full;
    - AE vs each baseline: the paired diff CI lower bound must be > 0.
    - All at budget 2.
  - `targetVerdict(report, target) -> {state: "pass"|"miss"|"unmeasured", value, ci, n}`
  - `forestPlot(paired) -> SVGElement`

- [ ] **Step 1: Write failing tests**

```python
def test_target_verdicts(tmp_path):
    # node script builds a minimal report with real_t001/full at budget 2:
    #   recall_any mean 0.739 ci [0.68,0.801]          -> "miss"
    #   recall_any 0.85                                -> "pass"
    #   variant absent                                 -> "unmeasured"
    #   report {"runs":0}: showEval-independent helper evalEmpty(report) === true
    ...

def test_ablation_notes_cell_wraps():
    css = text("styles.css")
    assert re.search(r"td\.notes[^}]*max-width", css)
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Do the following:
  - Page order:
    1. target cards: pass / miss / ยังไม่ได้วัด, with value, CI and n, rising in one after another;
    2. the forest plot (AE − baseline);
    3. the retrain chart;
    4. the budget selector and full tables.
  - Notes go into `details`/`summary` titled "หมายเหตุวิธีวัด". Add `details` and `summary` to `el()`'s allowed tags and to `HTML_TAGS` in the test.
  - The notes cell gets the class `notes`.
  - Bars and the plot grow on first open. Under reduced motion they show at once.
  - Missing data never renders as pass.
- [ ] **Step 4: Run** the whole file. Expected: PASS.
- [ ] **Step 5: Commit** `feat(web): eval target cards, forest plot, collapsible notes`

### Task 7: Taxonomy and About, then copy cleanup

**Files:**
- Modify: `idsai/product/web/index.html:156-200`
- Modify: `idsai/product/web/app.js` (`loadTaxonomy` markup only)
- Test: `tests/product/test_web_static.py`

- [ ] **Step 1: Write the failing test** `test_long_explanations_live_on_about`. Assert the Monitor section has no `<p>` longer than 120 characters outside `#chart-info-body`, and that About contains the τ-saturated and autocorrelation sentences.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Move the removed explanations into About sections and re-typeset the Taxonomy cards as hairline rows.
- [ ] **Step 4: Run** `uv run pytest tests/product -v`. Expected: PASS, including `test_ui_copy_wording`.
- [ ] **Step 5: Commit** `feat(web): re-typeset taxonomy and about; explanations move to about`

### Task 8: Verification

- [ ] **Step 1:** Run `uv run pytest -q`. Expected: the full suite passes; the last count was 1091 passed.
- [ ] **Step 2:** Browser check with `.claude/launch.json` `idsai-replay` and the preview tools:
  - each page at 1280×800 and 375×812, in dark and light;
  - no console errors;
  - play at the fastest speed and see the KPIs tween and the rail animate;
  - inject `pump_burst/test`, then see the ring and then the banner;
  - Eval numbers equal `docs/REPORT.md` (0.739 [0.680, 0.801], −0.170 [−0.222, −0.121]);
  - with OS reduced motion on, nothing animates.
- [ ] **Step 3:** Save screenshots to `data/review_page/` (gitignored). Note the results in `.superpowers/sdd/2026-10-27-market-ids-mvp/progress.md`.
- [ ] **Step 4: Commit** any fixes found: `fix(web): …`
