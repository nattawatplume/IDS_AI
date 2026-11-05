# Dashboard v3 (Glass, 4 pages, easy mode) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-skin and restructure the vanilla-JS dashboard into a dark purple–blue glass look with a Facebook-style login, 4 Thai-named pages, a traffic-light status, one-click simulations, plain-language results, a help drawer and a first-time tour. Everything that works today must keep working.

**Architecture:**
- Keep `idsai/product/web/{index.html,app.js,styles.css}` and every behaviour already in `app.js`: auth, WebSocket, replay control and loop, Red-Team timing, verdicts and eval data.
- Add small pure functions (exported through the existing `module.exports` block at the end of `app.js`) for each new decision, so the node harness can test them. The DOM code calls those functions.
- `styles.css` is rewritten around one token set.
- The only server change is one prefs key.

**Tech Stack:** HTML, CSS and vanilla JS (no build step), vendored lightweight-charts 5.2.1, FastAPI, and pytest with the node harness in `tests/product/test_web_static.py`.

**Spec:** `docs/superpowers/specs/2026-11-06-dashboard-glass-design.md`. Section 4.12, "Not AI-made", overrides any conflicting line.

## Global Constraints

- **Palette:**

  | Token | Value |
  |---|---|
  | background | `#0b0820` |
  | blobs | `#7c4dff`, `#2f6bff`, `#00c2ff`, total opacity ≤ 35 % |
  | glass fill | `rgba(255,255,255,0.06)` |
  | blur | `blur(16px) saturate(140%)` |
  | glass border | `rgba(255,255,255,0.12)` |
  | fallback fill | `rgba(20,18,48,0.85)` |
  | text / dim text | `#eef0ff` / `#a5a8c8` |
  | normal / warning / alert | `#3ee08f` / `#ffb547` / `#ff5d73` |
  | accent gradient | `#8b5cf6` → `#3b82f6` |

  The accent gradient is used **only** on the primary button and the active nav pill.

- **Spacing and shape:**
  - spacing scale 4/8/12/16/24/32 px;
  - radius 16 px for cards, 10 px for buttons, 999 px for pills;
  - text weights 400 and 600;
  - type sizes 28 px for the page title, 32–40 px for the hero sentence, 28 px for card values, 15 px for body;
  - numbers use `tabular-nums`.
- **No-AI-look rules:**
  - no emoji;
  - no gradient text;
  - no sparkle decorations;
  - no exclamation marks;
  - no "ยินดีต้อนรับ";
  - no marketing adjectives (ล้ำสมัย, อัจฉริยะ, ทรงพลัง, "ขับเคลื่อนด้วย AI").
- **Copy rules:**
  - Outside `ดูรายละเอียด` / `ขั้นสูง` / `รายละเอียดสำหรับกรรมการ` sections, visible text contains none of `MT-T`, `recall_any`, `config_hash`, `pump_burst`, `pump_stealth`, `wash_pingpong`, `region`, `τ`, `L1/L2/L3`.
  - The words "ความผิดปกติทางสถิติ" are used, and no person is accused.
  - A p-value is never called a probability, and there is no accuracy metric.
- **Motion:** every animation is off under `@media (prefers-reduced-motion: reduce)`, and JS checks `reducedMotion()`. Only the background drift and the status orb loop.
- **Responsive:** at ≤ 720 px the layout is a single column with a bottom tab bar, and nothing scrolls sideways at 375 px.
- **Unchanged rules:**
  - Red-Team stays fail-closed until `/api/health` confirms replay or sim, and is impossible in live mode.
  - Role gates are unchanged.
  - No external network.
  - No `innerHTML` with data; use `el()`.
- **Routes:**
  - `#/home` (default), `#/alerts`, `#/alerts/<id>`, `#/simulate`, `#/results`, `#/activity` (admin).
  - Redirects: `#/monitor`→`#/home`, `#/redteam`→`#/simulate`, `#/eval`→`#/results`, and `#/taxonomy` and `#/about` → `#/home` with the help drawer open.
- **Keys:** `1`–`4` switch the pages in nav order, Space toggles play, and `?` opens help. They match `ev.code` too (Digit1–4, Space, Slash with shift). `t` is removed.
- **Gates:** `uv run ruff check .` and `uv run pytest -q -m "not network and not slow"` pass before every commit.
- **Commits:** end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never `git add -f`; never commit `.claude/`, `data/` or `models/`. No long `--basetemp`, and no local `idsai serve` may be left running.

## Review Focus

1. **A session that expires on the login→home transition** must land back on the login, without a blank glass screen (Task 2, test `test_boot_shows_only_loader_until_me_resolves`).
2. **Viewers** must see the replay bar and the simulate cards read-only, each with the Thai reason, not hidden and not clickable (Tasks 3 and 5, tests `heroAndReplayGates viewer` and `simulate cards disabled for viewer`).
3. **No alerts at all:** a fresh replay start must show "ตลาดปกติ", not a crash on an empty list or a null F/τ (Task 3, `heroStatus` cases with `f=null` and `tau=null`).
4. **Tour targets** missing on narrow screens (bottom bar) must not throw. The step falls back to a centred bubble (Task 7, test `tour step with missing target centres`).
5. **Old bookmarked hashes** (`#/monitor`, `#/eval`, `#/about`) still land somewhere sensible (Task 1, test `routeFor redirects`).

---

### Task 1: Glass shell (tokens, background, top bar, 4-page nav, routes, keys)

**Files:**
- Modify:
  - `idsai/product/web/styles.css`: rewrite the tokens and shell styles, and remove the light theme;
  - `idsai/product/web/index.html`: header, nav and the user menu;
  - `idsai/product/web/app.js`:
    - remove `THEMES` / `applyTheme` / `chooseTheme` / `bindTheme` and the theme pref usage;
    - change `PAGE_ORDER` and `DEFAULT_ROUTE`;
    - routing;
    - `presenterKey` / `onPresenterKey`.
- Test: `tests/product/test_web_static.py`. Replace the theme tests (`test_both_themes_define_every_token`, `test_theme_toggle_exists_and_is_wired`, `test_no_capsule_tabs_or_shadows`) with the new ones below.

**Interfaces:**
- Produces:
  - `PAGE_ORDER = ["#/home", "#/alerts", "#/simulate", "#/results"]` and `DEFAULT_ROUTE = "#/home"`;
  - `routeFor(hash: string) -> { route: string, help: boolean }`, which applies the redirects in Global Constraints. An unknown hash gives `{route: "#/home", help: false}`, and `#/alerts/<id>` passes through;
  - `presenterKey(ev) -> "play" | "help" | "#/home" | "#/alerts" | "#/simulate" | "#/results" | null`;
  - `ROLE_TH = { viewer: "ผู้ชม", analyst: "ผู้วิเคราะห์", admin: "ผู้ดูแล" }`;
  - the help button `#help-btn`, the user-menu items `#menu-activity` (admin only), `#menu-tour` and `#btn-logout`;
  - the page sections `#page-home`, `#page-alerts`, `#page-alert`, `#page-simulate`, `#page-results`, `#page-activity` and `#page-login`. The old section ids are renamed, and later tasks fill them.

- [ ] **Step 1: Write the failing tests.**
  - The node test `routeFor redirects`:
    - `routeFor("#/monitor")` gives `{route:"#/home",help:false}`;
    - `#/redteam` gives `#/simulate`, and `#/eval` gives `#/results`;
    - `#/about` and `#/taxonomy` give `{route:"#/home",help:true}`;
    - `#/alerts/abc` passes through;
    - `""` and `#/nope` give `#/home`.
  - The node test `presenterKey`:
    - `{key:"1"}` gives `#/home` and `{code:"Digit4"}` gives `#/results`;
    - `{key:" "}` gives `play` and `{key:"?"}` gives `help`;
    - `{key:"t"}` gives null;
    - a key event from an input or textarea target gives null.
  - `test_shell_markup`:
    - 4 nav links whose text is exactly หน้าหลัก, การแจ้งเตือน, ทดลองจำลอง, ผลวัดระบบ;
    - no `#theme-toggle` and no `data-theme-choice`;
    - `#help-btn` exists.
  - `test_glass_tokens`: `styles.css` defines `--bg: #0b0820`, `--ok: #3ee08f`, `--warn: #ffb547`, `--bad: #ff5d73`, `backdrop-filter`, an `@supports not (backdrop-filter` fallback, and a `prefers-reduced-motion: reduce` block. It has no `background-clip: text`.
  - `test_no_emoji_or_bang_in_copy`: `index.html` text and `app.js` string literals contain no character in the emoji ranges U+1F300–U+1FAFF or U+2600–U+27BF, and no `!` inside Thai strings.
- [ ] **Step 2: Run the tests.** Run `uv run pytest tests/product/test_web_static.py -q`. The new tests should FAIL.
- [ ] **Step 3: Implement the shell.**
  - The background is a fixed `.bg` layer holding 3 blurred blobs with a 36 s drift keyframe.
  - The header is a glass bar: logo, nav pills, mode chip, `?`, user chip.
  - At ≤ 720 px the nav becomes a fixed bottom bar with simple inline SVG icons.
  - Route handling calls `routeFor`. When `help` is true it opens the drawer through a `openHelp()` stub that Task 7 fills; in this task it is a no-op.
  - Update `test_every_route_is_present` and `test_every_element_app_js_targets_exists` for the new ids.
- [ ] **Step 4: Run the tests again.** Run `uv run pytest tests/product -q` and ruff. Everything should PASS.
- [ ] **Step 5: Commit.** Use the message `feat(web): glass shell, 4-page Thai nav, route redirects, presenter keys`.

### Task 2: Login page (split layout, no dashboard flash)

**Files:**
- Modify:
  - `index.html`: `#page-login`;
  - `styles.css`;
  - `app.js`: `boot`, `enterApp`, `bindLogin`, `showLoginError`.
- Test: `tests/product/test_web_static.py`.

**Interfaces:**
- Consumes: the Task 1 shell.
- Produces:
  - `#page-login` contains `.login-hero` (left: logo, tagline, `<svg class="login-line">`) and `.login-card` (right: `#login-form` with the existing field ids, `#login-remember` and the submit button);
  - `bootView(me: object|null, authEnabled: boolean|null) -> "loading" | "login" | "app"`. A null `authEnabled` means not known yet, which gives `loading`.

- [ ] **Step 1: Write the failing tests.**
  - The node test `bootView`:
    - `(null, null)` gives `loading`;
    - `(null, true)` gives `login`;
    - `({...}, true)` and `(null, false)` give `app`.
  - `test_boot_shows_only_loader_until_me_resolves`: in `index.html` every `.page` and the header start `hidden`, and a `#boot-loader` element exists and is visible by default.
  - `test_login_markup`:
    - the tagline text is exactly "กล้องวงจรปิดของตลาดคริปโต เตือนเมื่อการซื้อขายผิดปกติทางสถิติ";
    - the button text is "เข้าสู่ระบบ";
    - the checkbox label is "จำฉันไว้";
    - there is no `<ul>` inside `.login-hero`.
  - The existing login tests (framed hint, `safeNext`, error messages) still pass unchanged.
- [ ] **Step 2: Run the tests.** Run `uv run pytest tests/product/test_web_static.py -q`. They should FAIL.
- [ ] **Step 3: Implement the login page.**
  - The `.login-line` path is drawn with `stroke-dasharray` and animated by a dash-offset loop. It is static under reduced motion.
  - At ≤ 720 px the hero sits above the card, with the line hidden.
- [ ] **Step 4: Run the tests again.** Run `uv run pytest tests/product -q` and ruff. Everything should PASS.
- [ ] **Step 5: Commit.** Use the message `feat(web): split login page, loader until the session is known`.

### Task 3: Home page (hero status, small cards, chart, replay bar, rail)

**Files:**
- Modify:
  - `index.html`: `#page-home`;
  - `styles.css`;
  - `app.js`: `renderHealth`, `renderAlertsKpi`, `renderBudget`, the chart init and restyle, `bindReplay`, `renderRail`, and the readout.
- Test: `tests/product/test_web_static.py`.

**Interfaces:**
- Consumes: `state.alerts`, the latest F and τ series values (`lastValue`), the auditor health, `eventNowUs()`, `can("analyst")` and `setMode`.
- Produces:
  - `heroStatus({ nowUs, alerts, f, tau, auditor, symbol }) -> { level: "normal"|"watch"|"alert", title: string, line: string, alertId: string|null }`, with these rules:
    - **alert:** an alert whose `ts_event_us >= nowUs - 30*60*S_US`. The title is "พบความผิดปกติทางสถิติ" and the line reads "ระบบพบรูปแบบการซื้อขายที่ต่างจากปกติของ {symbol} เมื่อ {HH:MM} น.", with `alertId` set to the newest such alert.
    - **watch:** otherwise, `f >= 0.8*tau` (when both are numbers) or `auditor === "warn"`. The title is "กำลังเฝ้าระวัง" and the line is "ระดับความผิดปกติใกล้เส้นเตือน" or "ระบบตรวจตัวเองพบจุดที่ควรดู".
    - **normal:** otherwise. The title is "ตลาดปกติ" and the line is "ไม่มีแจ้งเตือนใน 30 นาทีล่าสุด".
    - A null `nowUs`, `f` or `tau` never throws.
  - `HERO_WATCH_RATIO = 0.8` and `HERO_RECENT_US = 30*60*S_US`.
  - `SPEED_CHIPS = [{label:"1×",v:1},{label:"10×",v:10},{label:"60×",v:60},{label:"เร็วสุด",v:0}]`.
  - `firstSentence(text: string) -> string`, which cuts at the first `" "` after a Thai full stop or at the first `"."`, capped at 120 characters with `…`.

- [ ] **Step 1: Write the failing tests.** These are node tests.
  - `heroStatus`:
    - no alerts with `f=null` and `tau=null` gives normal;
    - `f=0.79,tau=1` gives normal, and `f=0.8,tau=1` gives watch;
    - `auditor:"warn"` gives watch;
    - an alert at exactly `now-30min` gives alert, with that `alertId`;
    - an alert at `now-30min-1µs` with `f=0.1` gives normal;
    - when both conditions hold, alert wins.
  - `heroAndReplayGates viewer`: with `can` false for analyst, the replay buttons are `disabled` and the reason text "เฉพาะผู้วิเคราะห์ขึ้นไปที่ควบคุมได้" is present. Use the existing `applyRoleGates` harness pattern.
  - `firstSentence` cases.
  - The markup test `test_home_markup`:
    - `#hero-orb`, `#hero-title`, `#hero-line`, the 3 small cards (`#card-alerts`, `#card-symbol`, `#card-health`), `#chart-details` as a `<details>`, `#replay-bar` containing 4 `.chip` speed buttons, `#advanced-seek` inside a `<details>`, and `#rail`;
    - the L1/L2/L3 lanes toggle lives inside `#chart-details`.
- [ ] **Step 2: Run the tests.** Run `uv run pytest tests/product/test_web_static.py -q`. They should FAIL.
- [ ] **Step 3: Implement the Home page.**
  - `renderHero()` runs on every health, score and alert update, and only touches the DOM when the level or text changes. The orb class changes through `flash`.
  - The chart colours come from the CSS vars: price `#a5b4fc`, F series `#8b5cf6` with a glow, the τ line dashed `#ffb547` titled "เส้นเตือน", and the F pane titled "ระดับความผิดปกติ".
  - The rail shows up to 5 mini-cards built with `el()`.
  - The old KPI row and the top-bar Auditor and budget elements are removed. Their data moves into the small cards: the budget line "ใช้งบแจ้งเตือนไป x/y" goes in `#card-alerts`, and the auditor checks go in an expandable list in `#card-health`.
- [ ] **Step 4: Run the tests again.** Run `uv run pytest tests/product -q` and ruff. Everything should PASS.
- [ ] **Step 5: Commit.** Use the message `feat(web): home page with traffic-light status, small cards, glass chart, replay chips`.

### Task 4: Alerts list, detail and the new-alert toast

**Files:**
- Modify:
  - `index.html`: `#page-alerts`, `#page-alert`, `#toast-root`;
  - `styles.css`;
  - `app.js`: `renderAlertsTable` → `renderAlertCards`, `showAlert`, `feedbackCard`, `techniqueCard`, `onAlert`.
- Test: `tests/product/test_web_static.py`.

**Interfaces:**
- Consumes: `VERDICT_TH`, `SEVERITY_TH`, `firstSentence` (Task 3) and the existing feedback POST.
- Produces:
  - `ALERT_FILTERS = [{key:"all",label:"ทั้งหมด"},{key:"open",label:"ยังไม่ตรวจ"},{key:"tp",label:"ผิดปกติจริง"},{key:"fp",label:"เตือนผิด"}]`;
  - `filterAlerts(alerts: Alert[], key: string) -> Alert[]`. `open` means no feedback entry; `tp` and `fp` match the latest verdict; the result is sorted newest first;
  - `VERDICT_TH` values become `{tp:"ผิดปกติจริง", fp:"เตือนผิด", unsure:"ไม่แน่ใจ"}`, with no "(TP)";
  - `showToast(text: string, href: string|null)`. It shows for 5 s, keeps at most 3 at once, and does not slide under reduced motion.

- [ ] **Step 1: Write the failing tests.**
  - The node test `filterAlerts`: 4 alerts with mixed feedback give the right subsets and order. An alert whose latest verdict is `unsure` counts in neither `tp` nor `fp`, and not in `open` either.
  - A node test where a new alert through `onAlert` calls `showToast` once, and a resync or `alerts_reset` calls it 0 times.
  - `test_alert_detail_markup`:
    - the detail uses 3 buttons with the exact texts ผิดปกติจริง / เตือนผิด / ไม่แน่ใจ;
    - the p-values, the layers, the technique and the provenance are rendered inside a `<details class="tech">` whose summary is "ดูรายละเอียด".
  - The empty state text is exactly "ยังไม่มีการแจ้งเตือน ระบบกำลังเฝ้าดูอยู่".
- [ ] **Step 2: Run the tests.** Run `uv run pytest tests/product/test_web_static.py -q`. They should FAIL.
- [ ] **Step 3: Implement the alerts pages.**
  - The cards have a left severity bar coloured by severity: low is the dim colour, medium is warning, high is alert.
  - The filter chips are flat pills.
  - The detail chart is restyled with the Task 3 colours.
  - `wsBanner` becomes a sticky glass toast with the text "การเชื่อมต่อหลุด กำลังเชื่อมใหม่…" and the same show/hide logic.
  - Inline API-error messages render inside the card that failed.
- [ ] **Step 4: Run the tests again.** Run `uv run pytest tests/product -q` and ruff. Everything should PASS.
- [ ] **Step 5: Commit.** Use the message `feat(web): alert cards with filters, plain detail, verdict buttons, toasts`.

### Task 5: Simulate page (one-click cards, run timeline, advanced settings)

**Files:**
- Modify:
  - `index.html`: `#page-simulate`;
  - `styles.css`;
  - `app.js`: `bindRedTeam`, `addInjection`, the run rendering, `bannerText`, `setMode`.
- Test: `tests/product/test_web_static.py`.

**Interfaces:**
- Consumes:
  - `redTeamAllowed(mode)`, `can("analyst")`;
  - the existing inject POST body `{family, region, offset_s, symbol}`;
  - `injectionStatus`, `firstDetection`, `restartRuns` and the `run.closed` / `run.restarted` flags, all unchanged.
- Produces:
  - `SIM_CARDS = [{family:"pump_burst",title:"ปั่นเร็ว",line:"ราคาและปริมาณพุ่งขึ้นในไม่กี่นาที",icon:"steep"},{family:"pump_stealth",title:"ปั่นแบบเนียน",line:"ค่อยๆ ดันราคาขึ้นช้าๆ",icon:"ramp"},{family:"wash_pingpong",title:"ซื้อขายวนหลอก",line:"ซื้อขายกันเองไปมาให้ดูคึกคัก",icon:"loop"}]`;
  - `simulateRequest(family: string, adv: {region?:string, offset_s?:number, symbol?:string}|null, currentSymbol: string) -> {family, region, offset_s, symbol}`. The defaults are `region:"test"`, `offset_s:60` and `symbol: currentSymbol`; advanced values override them;
  - `simStatusText(status: string, run) -> string`:
    - pending gives "เริ่มแล้ว";
    - waiting gives "กำลังรอระบบ";
    - detected gives "จับได้ใน {eventS} วินาที (เวลาจริง {wallS} วินาที)";
    - missed gives "ไม่ทันจับ";
    - restarted gives "พลาด (replay เริ่มใหม่)";
  - `SIM_REASON = { live: "ทดลองได้เฉพาะโหมดย้อนดู", viewer: "ต้องเป็นผู้วิเคราะห์ขึ้นไป", ended: "การย้อนดูจบแล้ว รอรอบใหม่" }`.

- [ ] **Step 1: Write the failing tests.** These are node tests.
  - `simulateRequest("pump_burst", null, "STEEMBTC")` deep-equals `{family:"pump_burst",region:"test",offset_s:60,symbol:"STEEMBTC"}`, and an override `{offset_s:120}` changes only that field.
  - `simStatusText` covers all 5 cases.
  - `simulate cards disabled for viewer` and `simulate cards disabled in live`: the 3 cards are `disabled`, with the matching `SIM_REASON` text shown.
  - The existing fail-closed tests are re-pointed to the new card buttons, and stay disabled before health arrives.
  - `test_simulate_markup`:
    - 3 `.sim-card` buttons, each holding an inline `<svg>`;
    - the advanced fields sit inside `<details>` whose summary is "ตั้งค่าขั้นสูง";
    - the region options are labelled "ช่วงค่าทดสอบ" and "ช่วงค่าฝึก";
    - the intro text is exactly "ใส่เหตุการณ์ปั่นจำลองลงในข้อมูลย้อนหลัง แล้วดูว่าระบบจับได้เร็วแค่ไหน (ไม่แตะข้อมูลจริง)".
- [ ] **Step 2: Run the tests.** Run `uv run pytest tests/product/test_web_static.py -q`. They should FAIL.
- [ ] **Step 3: Implement the Simulate page.**
  - A card click sends one POST with `simulateRequest(...)`.
  - The run list renders timeline cards with 3 dots: started, waiting, result.
  - The time-to-detect tests (TTD_JS) keep passing unchanged.
- [ ] **Step 4: Run the tests again.** Run `uv run pytest tests/product -q` and ruff. Everything should PASS.
- [ ] **Step 5: Commit.** Use the message `feat(web): one-click simulate cards, run timeline, advanced settings folded`.

### Task 6: Results page (plain cards, committee details)

**Files:**
- Modify:
  - `index.html`: `#page-results`;
  - `styles.css`;
  - `app.js`: the eval rendering, `EVAL_TARGETS` labels, `targetVerdict`, `forestPlot`, `evalEmpty`.
- Test: `tests/product/test_web_static.py`.

**Interfaces:**
- Consumes: `/api/experiments` data and the existing `targetVerdict(report, target, budget)`.
- Produces:
  - `resultSentence(target, value, extra) -> string`. It gives the Thai sentence per spec 4.8:
    - recall: "จับเหตุการณ์ที่มีบันทึกได้ {k} จาก {n} ครั้ง (เมื่อเตือนไม่เกิน {b} ครั้งต่อเหรียญต่อวัน)", falling back to "{pct}%" when k and n are absent;
    - FA: "เตือนผิดเฉลี่ย {x} ครั้งต่อเหรียญต่อวัน";
    - hard FA: "เตือนผิดในวันที่ไม่มีเหตุการณ์ {x} ครั้ง";
    - baseline delta: "เทียบกับวิธีพื้นฐาน: ดีกว่า / ไม่ต่างอย่างมีนัยสำคัญ / แย่กว่า", based on whether the CI excludes 0;
    - latency: "ความเร็วในการเตือน: ยังไม่วัด".
  - `RESULT_BADGE = { pass: "ผ่านเป้า", fail: "ยังไม่ถึงเป้า", none: "ยังไม่วัด" }`.
  - The reference cards (random / PCA) are labelled "วิธีเปรียบเทียบ".
  - The budget chips show "งบเตือน 0.1 / 0.5 / 2 / 10 ครั้งต่อวัน", and the `eval_budget` pref is kept.

- [ ] **Step 1: Write the failing tests.**
  - The node test `resultSentence`: one case per kind with exact strings, plus a delta CI of `[-0.1, 0.2]` giving "ไม่ต่างอย่างมีนัยสำคัญ".
  - `test_results_markup`: the forest plot, metric ids, config hash, run counts and provenance badges render only inside `<details>` whose summary is "รายละเอียดสำหรับกรรมการ".
  - The existing eval tests are updated to the new structure, with the same data checks.
- [ ] **Step 2: Run the tests.** Run `uv run pytest tests/product/test_web_static.py -q`. They should FAIL.
- [ ] **Step 3: Implement the Results page.** Card sizes follow the content: the primary recall card is wide, and the others are compact.
- [ ] **Step 4: Run the tests again.** Run `uv run pytest tests/product -q` and ruff. Everything should PASS.
- [ ] **Step 5: Commit.** Use the message `feat(web): plain-language results with committee details folded`.

### Task 7: Help drawer, first-time tour, and the `tour_done` pref

**Files:**
- Modify:
  - `idsai/product/api/app.py`: `PrefsIn` gains `tour_done: StrictBool | None = None`;
  - `index.html`: `#help-drawer`, `#tour`;
  - `styles.css`;
  - `app.js`: `openHelp` (filling the Task 1 stub), `loadTaxonomy` rendering into the drawer, the tour, and `#menu-tour`.
- Test:
  - `tests/product/test_api_auth.py`;
  - `tests/product/test_web_static.py`.

**Interfaces:**
- Consumes: `savePrefs`, `loadedPrefs`, `authOn`, `storageGet` / `storageSet`, `/api/taxonomy` and `reducedMotion`.
- Produces:
  - `TOUR_STEPS`, 4 steps with these targets and texts:
    1. `#hero-title`: "ดูตรงนี้ก่อน บอกว่าตอนนี้ตลาดปกติหรือไม่";
    2. `#chart-card`: "เส้นม่วงคือระดับความผิดปกติ ถ้าข้ามเส้นเตือนสีส้ม ระบบจะแจ้งเตือน";
    3. `[data-route="alerts"]`: "รายการแจ้งเตือนทั้งหมด กดเข้าไปเพื่ออ่านคำอธิบายและบันทึกผลตรวจ";
    4. `[data-route="simulate"]`: "ลองใส่การปั่นจำลองแล้วดูว่าระบบจับได้เร็วแค่ไหน".
  - `tourShouldShow(prefs: object, authOn: boolean, stored: string|null) -> boolean`. With auth on it is `!prefs.tour_done`; with auth off it is `stored !== "1"`.
  - `markTourDone()` saves the pref `{tour_done: true}`, or sets localStorage `idsai.tour_done = "1"` when auth is off.
  - `placeBubble(rect: {top,left,width,height}|null, viewport: {w,h}) -> {top,left,centred:boolean}`. A null rect gives a centred bubble, and the bubble always stays fully inside the viewport.
  - The drawer has 3 sections:
    1. "ระบบนี้ทำอะไร", about 5 sentences. It keeps "ระบบนี้ไม่ทำอะไร": no trading, no investment advice, no accusations;
    2. "รูปแบบที่ระบบมองหา": cards showing the Thai name and one line, with no `MT-` ids. Citations sit inside `<details>` "แหล่งอ้างอิง";
    3. "คำศัพท์": entries for ระดับความผิดปกติ, เส้นเตือน, ย้อนดู, ทดลองจำลอง, ผิดปกติจริง / เตือนผิด, plus a "เริ่มทัวร์" button.

- [ ] **Step 1: Write the failing tests.**
  - The API test:
    - `PUT /api/me/prefs {"tour_done": true}` gives 200 and `GET /api/me` prefs contain `tour_done: true`;
    - `{"tour_done": "yes"}` gives 422.
  - The node test `tourShouldShow`:
    - `({}, true, null)` is true;
    - `({tour_done:true}, true, null)` is false;
    - `({}, false, "1")` is false;
    - `({}, false, null)` is true.
  - The node test `tour step with missing target centres`: `placeBubble(null, {w:375,h:812}).centred === true`. A rect at the bottom edge places the bubble above it, inside the viewport.
  - The node test `help drawer taxonomy has no ids`: rendering a sample taxonomy with ids `MT-T001` puts no "MT-" in the drawer text outside `<details>`.
  - `test_help_and_tour_markup`:
    - the drawer has the 3 section headings;
    - the tour has the buttons ถัดไป, ข้าม and เสร็จ;
    - the drawer and the tour handle Escape (`keydown` with `Escape` in app.js).
- [ ] **Step 2: Run the tests.** Run `uv run pytest tests/product -q`. They should FAIL.
- [ ] **Step 3: Implement the drawer, the tour and the pref.**
  - The tour starts after `enterApp` lands on Home when `tourShouldShow` is true.
  - The spotlight is a fixed overlay with a transparent cut-out made with `box-shadow: 0 0 0 9999px rgba(5,4,20,.6)` on a highlight box.
  - Focus moves into the bubble and is trapped there.
- [ ] **Step 4: Run the tests again.** Run `uv run pytest tests/product -q` and ruff. Everything should PASS.
- [ ] **Step 5: Commit.** Use the message `feat(web): help drawer, first-time tour, tour_done pref`.

### Task 8: Copy guard and final polish

**Files:**
- Modify: `tests/product/test_web_static.py`. Small copy fixes go in `index.html` and `app.js` if the guard finds any.
- Test: the same file.

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write the failing test** `test_default_view_has_no_code_words`.
  - Parse `index.html` with `html.parser`, dropping the text inside `<details>` elements, `<script>` and `<style>`.
  - Also collect the Thai UI string constants in app.js: `*_TH` maps, `SIM_CARDS`, `TOUR_STEPS`, `RESULT_BADGE`, `ALERT_FILTERS`, `SPEED_CHIPS` and `ROLE_TH`.
  - Assert that none of `MT-T`, `recall_any`, `config_hash`, `pump_burst`, `pump_stealth`, `wash_pingpong`, `region`, `τ`, `L1/L2/L3`, `ยินดีต้อนรับ`, `อัจฉริยะ`, `ล้ำสมัย`, `ทรงพลัง` appears in that text. `family` values in `SIM_CARDS` are data, not copy: check only its `title` and `line` fields.
- [ ] **Step 2: Run the test.** Run it and fix any copy it flags in `index.html` and `app.js`.
- [ ] **Step 3: Run the full gate.** Run `uv run pytest -q -m "not network and not slow"` and `uv run ruff check .`. Everything should PASS.
- [ ] **Step 4: Commit.** Use the message `test(web): guard the default view against code words and hype copy`.
