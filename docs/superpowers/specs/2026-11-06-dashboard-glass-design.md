# Dashboard v3: Glass look, 4 pages, made easy for non-experts

Date: 2026-11-06 · Status: spec approved by owner (with 4.12 added)
Supersedes the look and page layout of `2026-11-05-dashboard-redesign-design.md`. Behaviour from that spec and from `2026-11-05-auth-design.md` and `2026-11-05-hosting-design.md` stays unless this spec changes it.

## 1. Why

The owner finds the current dashboard plain, hard to understand and cluttered with code-like text. Examples:
- Eval shows config hashes, `recall_any`, and variant and experiment ids.
- Taxonomy shows `MT-T000` ids and long citations.
- Red-Team asks for `pump_burst` and `test/train`.

The main users are teachers and committee members who open the hosted URL. They are not experts.

Success means:
1. Opening the site shows a polished login page first, like Facebook's split layout, whenever sign-in is on.
2. Within about 10 seconds of signing in, a first-time user can tell whether the market looks normal and what to click next.
3. No page shows ids, hashes, variable names or English jargon unless the user opens "ดูรายละเอียด" (details).
4. The look is a dark purple–blue "glass" style with tasteful motion, and all motion is off under `prefers-reduced-motion`.
5. Everything that works today still works:
   - sign-in, roles, per-user prefs, activity log;
   - live WebSocket updates;
   - replay control and the replay loop;
   - Red-Team runs and their detected/missed timing;
   - alert verdicts;
   - eval results.

## 2. Scope

**In scope:**
- A rework of `idsai/product/web/index.html`, `styles.css` and `app.js`. This means new page structure, new styles, new copy, a help drawer and a first-time tour.
- One server change: `PrefsIn` in `idsai/product/api/app.py` gains `tour_done: bool | None`.
- Tests:
  - `tests/product/test_web_static.py` and its node harness;
  - `tests/product/test_api_auth.py` for the prefs key.
- Removing or replacing assertions on things this spec removes, such as the theme toggle and the old tab names.

**Out of scope:**
- Any change to detection, the API's data or other routes.
- A JS framework or build step. The dashboard stays vanilla HTML/JS/CSS with the vendored lightweight-charts.
- A light theme. It is removed, and the `theme` pref is ignored but still accepted so old saved prefs do not error.
- New data or metrics.

## 3. Rules that still bind

- Say "ความผิดปกติทางสถิติ". Never accuse a person or say "การปั่นราคา" as a finding; a simulated scenario may be called "การปั่นจำลอง".
- Never call a p-value a probability. There is no accuracy metric.
- Red-Team is impossible in live mode and stays fail-closed until `/api/health` confirms replay or sim.
- Role gates are unchanged: viewers cannot control the replay, give verdicts or run simulations. They see why in Thai.
- Static files keep their no-cache headers.
- The page needs no external network: no CDN fonts or scripts. The fonts are the system Thai and Latin fonts already used.

## 4. Design

### 4.1 Look ("Glass")

**Background:**
- Near-black navy `#0b0820`, under a fixed layer of 3 large blurred radial blobs (purple `#7c4dff`, blue `#2f6bff`, cyan `#00c2ff`) at about 25–35 % opacity.
- The blobs drift slowly on a 30–40 s loop.

**Surfaces:**
- Glass cards: `rgba(255,255,255,0.06)` fill, `backdrop-filter: blur(16px) saturate(140%)`, a 1 px `rgba(255,255,255,0.12)` border, 16 px radius and a soft shadow.
- Where `backdrop-filter` is unsupported, the fill falls back to `rgba(20,18,48,0.85)`.

**Text and status colours:**
- Text is `#eef0ff`, and dim text is `#a5a8c8`.
- Status colours glow through a matching `box-shadow`:
  - normal is green `#3ee08f`;
  - warning is amber `#ffb547`;
  - alert is coral `#ff5d73`.
- The accent gradient runs from `#8b5cf6` to `#3b82f6` and is used on primary buttons, the active nav pill and focus rings.

**Type:** the system UI font stack with a Thai fallback. Numbers use `font-variant-numeric: tabular-nums`. The hierarchy is:
- page title 28 px;
- the hero status sentence 32–40 px;
- card values 28 px;
- body text 15 px.

**Motion** (all disabled under `prefers-reduced-motion: reduce`):
- background drift;
- page enter: fade plus an 8 px rise, 220 ms;
- card hover: lift 2 px with a brighter border;
- numbers count up, reusing `tweenNumber`;
- the status orb pulses (green slow, amber or coral faster);
- a new alert pops a toast in from the top right for 5 s;
- the chart lines glow;
- buttons have a pressed state.

**Responsive:**
- Mobile is ≤ 720 px wide. The layout becomes a single column, the nav moves to a bottom tab bar with 4 icons and labels, and the login stacks vertically.
- No horizontal scroll at 375 px.

### 4.2 Login (first screen when sign-in is on)

- The page is split, Facebook-style:
  - **Left:** the Market IDS logo mark, a one-line Thai tagline ("กล้องวงจรปิดของตลาดคริปโต เตือนเมื่อการซื้อขายผิดปกติทางสถิติ") and a decorative glowing line chart that animates. This is a static SVG or canvas with no data.
  - **Right:** a glass card with the username, the password, "จำฉันไว้" and one large gradient button "เข้าสู่ระบบ". The error text appears inline.
- When auth is on and there is no session, `/` goes straight to the login, with no flash of the dashboard. While the session is checked, the app shows only the background and a small loader.
- When auth is off (a local run), the login never shows.
- These behaviours stay: the framed-page hint, `next=` with `safeNext`, the lock, 429 and wrong-password messages.

### 4.3 Shell

**Top bar (glass):**
- the logo;
- 4 nav pills;
- a compact replay chip ("ย้อนดู" in replay, "สด" in live, using the existing mode badge);
- a "?" help button;
- the user menu.

**User menu:**
- name and role badge (role names in Thai: ผู้ชม / ผู้วิเคราะห์ / ผู้ดูแล);
- "กิจกรรม" (admin only), which opens the Activity page;
- "ดูทัวร์อีกครั้ง";
- "ออกจากระบบ".

The theme toggle is removed. The existing top-bar Auditor and budget details move to the Home page's small cards.

### 4.4 Pages and routes

| Route | Thai name | Replaces |
|---|---|---|
| `#/home` (default) | หน้าหลัก | Monitor |
| `#/alerts`, `#/alerts/<id>` | การแจ้งเตือน | Alerts and alert detail |
| `#/simulate` | ทดลองจำลอง | Red-Team |
| `#/results` | ผลวัดระบบ | Eval |
| `#/activity` | กิจกรรม (admin, from the user menu) | Activity |

Old hashes redirect: `#/monitor` → `#/home`, `#/redteam` → `#/simulate`, `#/eval` → `#/results`. `#/taxonomy` and `#/about` open the help drawer on Home. Presenter keys: `1`–`4` switch the 4 pages, Space toggles play, and `?` opens help. `t` is removed.

### 4.5 หน้าหลัก (Home)

1. **Hero status card** (full width):
   - A pulsing orb and one big sentence, derived from current state:
     - **normal (green):** "ตลาดปกติ", when there is no open alert in the last 30 min of event time and the latest F < τ;
     - **watch (amber):** "กำลังเฝ้าระวัง", when the latest F ≥ 0.8 τ or the auditor reports `warn`;
     - **alert (coral):** "พบความผิดปกติทางสถิติ", when an alert opened in the last 30 min of event time.
   - Under it, one plain line, e.g. "ระบบพบรูปแบบการซื้อขายที่ต่างจากปกติของ STEEMBTC เมื่อ 16:42 น." The line links to that alert.
2. **Three small cards:**
   - "แจ้งเตือนวันนี้" (24 h count, count-up);
   - "เหรียญที่ดูอยู่" (the symbol select lives here when more than one symbol exists);
   - "สุขภาพระบบ" (ปกติ / เฝ้าระวัง / มีปัญหา from the auditor; clicking it expands the checks in Thai).
   - The old budget meter moves into the alerts card as a small "ใช้งบแจ้งเตือนไป x/y" line.
3. **Chart card:**
   - The price line, plus a second pane "ระดับความผิดปกติ" (the existing F series) with a dashed "เส้นเตือน" (τ). Alert markers are shown.
   - A crosshair readout in plain words.
   - The L1/L2/L3 lanes and the info text stay, behind a "ดูรายละเอียด" toggle on the card, closed by default.
4. **Replay bar** (replay mode only, analyst or above):
   - a big ▶/❚❚ button and speed chips `1×` `10×` `60×` `เร็วสุด`;
   - a progress line showing the replay clock;
   - "ไปที่เวลา" behind "ขั้นสูง".
   - A viewer sees the bar read-only with "เฉพาะผู้วิเคราะห์ขึ้นไปที่ควบคุมได้".
5. **"แจ้งเตือนล่าสุด" rail:** up to 5 alert mini-cards. Each shows a severity dot, the time and the plain Thai headline from `explanation_th`'s first sentence, and links to the detail.

### 4.6 การแจ้งเตือน (Alerts)

**List:**
- Glass cards, newest first, one per alert. Each card shows:
  - a severity colour bar and its Thai label (ต่ำ / กลาง / สูง);
  - the time and the symbol;
  - one plain sentence;
  - the verdict chip if one was given.
- Filter chips: ทั้งหมด / ยังไม่ตรวจ / ผิดปกติจริง / เตือนผิด.
- The empty state says "ยังไม่มีการแจ้งเตือน ระบบกำลังเฝ้าดูอยู่" with a soft illustration (CSS only).

**Detail:**
- At the top, the full Thai explanation and the event chart. The existing detail chart is restyled.
- In the middle, three large buttons: "ผิดปกติจริง", "เตือนผิด" and "ไม่แน่ใจ". They are analyst-gated, and the existing feedback list sits below them.
- "ดูรายละเอียด" (closed) holds the technical evidence: p-values, F, the layers, the technique card, and the provenance badges.

### 4.7 ทดลองจำลอง (Simulate)

**Intro:** one plain line, "ใส่เหตุการณ์ปั่นจำลองลงในข้อมูลย้อนหลัง แล้วดูว่าระบบจับได้เร็วแค่ไหน (ไม่แตะข้อมูลจริง)".

**Three big cards.** One click starts a run with the defaults: `region=test`, `offset_s=60`, and the current symbol.

| Card | family | Title | One line |
|---|---|---|---|
| steep-line SVG | `pump_burst` | ปั่นเร็ว | ราคาและปริมาณพุ่งขึ้นในไม่กี่นาที |
| gentle-ramp SVG | `pump_stealth` | ปั่นแบบเนียน | ค่อยๆ ดันราคาขึ้นช้าๆ |
| loop-arrows SVG | `wash_pingpong` | ซื้อขายวนหลอก | ซื้อขายกันเองไปมาให้ดูคึกคัก |

**Runs:** each run shows as a timeline card, "เริ่มแล้ว → กำลังรอระบบ → จับได้ใน X วินาที ✓ / ไม่ทันจับ / พลาด (replay เริ่มใหม่)". It uses the existing `injectionStatus` logic and shows the event-time and wall-clock seconds in plain words.

**"ตั้งค่าขั้นสูง" (closed)** holds the existing region, offset and symbol fields, with Thai labels:
- region `test` = "ช่วงค่าทดสอบ";
- region `train` = "ช่วงค่าฝึก".

**Disabled and refused states** use the same gates as today, with Thai reasons:
- in live mode: "ทดลองได้เฉพาะโหมดย้อนดู";
- for a viewer: "ต้องเป็นผู้วิเคราะห์ขึ้นไป";
- once the replay has ended.

### 4.8 ผลวัดระบบ (Results)

**Plain summary cards** from the same `EVAL_TARGETS` data. Each card has a Thai sentence plus a ✓ ผ่านเป้า / ✗ ยังไม่ถึงเป้า / — ยังไม่วัด badge, for example:
- "จับเหตุการณ์ที่มีบันทึกได้ X จาก Y ครั้ง (เมื่อเตือนไม่เกิน 2 ครั้งต่อเหรียญต่อวัน)" for recall_any;
- "เตือนผิดเฉลี่ย X ครั้งต่อเหรียญต่อวัน";
- "เตือนผิดในวันที่ไม่มีเหตุการณ์ X ครั้ง";
- "เทียบกับวิธีพื้นฐาน: ดีกว่า / ไม่ต่างอย่างมีนัยสำคัญ / แย่กว่า";
- "ความเร็วในการเตือน: ยังไม่วัด".

**Rules:**
- The reference cards (random / PCA) read as "วิธีเปรียบเทียบ".
- The budget selector stays as chips "งบเตือน 0.1 / 0.5 / 2 / 10 ครั้งต่อวัน".

**"รายละเอียดสำหรับกรรมการ" (closed)** holds the forest plot, metric ids, config hash, run counts and the provenance badges, all unchanged in content.

### 4.9 Help drawer ("?")

A right-side glass drawer with three sections:
1. **"ระบบนี้ทำอะไร":** the existing About text, shortened to about 5 plain sentences, keeping what the system does not do.
2. **"รูปแบบที่ระบบมองหา":** the taxonomy as simple cards, each with the Thai name and a one-line description only. No `MT-` ids. Citations are inside a closed "แหล่งอ้างอิง" per card.
3. **"คำศัพท์":** 5–6 short definitions (ระดับความผิดปกติ, เส้นเตือน, ย้อนดู, ทดลองจำลอง, ผิดปกติจริง/เตือนผิด), plus a "เริ่มทัวร์" button.

### 4.10 First-time tour

- It has 4 steps, each a spotlight with a glass bubble:
  1. the hero status;
  2. the chart, explaining "เส้นเตือน";
  3. the alerts nav;
  4. the simulate nav.
- Buttons: "ถัดไป", "ข้าม" and "เสร็จ".
- It shows once per user, after the first successful sign-in landing on Home. The record is the pref `tour_done: true`, or `localStorage` when auth is off. It can be replayed from the help drawer and the user menu.
- On a phone it still works, with bubbles placed above or below the target.
- Escape closes it. Focus is trapped inside the bubble. Under reduced motion it has no animation.

### 4.11 Accessibility and copy

- Every control has a Thai label. Focus rings are visible (the accent gradient outline), and contrast is at least 4.5:1 for body text on glass.
- The status is never shown by colour alone: the orb always has the sentence beside it.
- Copy rule: no English or code identifiers outside "ดูรายละเอียด" sections. The brand "Market IDS" and coin symbols are allowed. Numbers have Thai units.

### 4.12 Not "AI-made" (owner's request, 2026-11-06)

The owner wants it to look easy and good, but not generated. These rules override anything above that conflicts with them.

**Icons:**
- No emoji anywhere. The simulate cards use small single-colour inline SVG line icons (16–24 px, 1.5 px stroke) drawn for this project: a steep line, a gentle ramp, and two looping arrows.

**Restraint:**
- The accent gradient appears only on the primary button and the active nav pill.
- No gradient text, no sparkle or star decorations, no rainbow glow.
- Background blobs stay subtle: total opacity ≤ 35 %, and they never sit behind body text at full strength.
- Glass is for cards only, not for every small chip. Chips are flat, semi-transparent fills.

**Copy:**
- Short, specific and factual, written like a person on the team.
- No "ยินดีต้อนรับสู่…!", no exclamation marks, no "ขับเคลื่อนด้วย AI", no marketing adjectives (ล้ำสมัย, อัจฉริยะ, ทรงพลัง).
- Headline sentences state facts from the data, e.g. "ตลาดปกติ · ไม่มีแจ้งเตือนใน 30 นาทีล่าสุด".

**Layout honesty:**
- Real data first, with no placeholder or decorative cards.
- Card sizes follow content, not a uniform 3-up grid everywhere.
- One clear primary action per screen.

**Consistency:**
- One spacing scale (4/8/12/16/24/32 px) and one radius family (16 px cards, 10 px buttons, 999 px pills).
- Two text weights (400/600).
- Numbers right-aligned in tables.

**Motion with purpose:** animate only things that change, such as status, new alerts, numbers and page entry. Nothing loops except the background drift and the status orb.

**Login page:** the left side shows the logo, the one-line tagline and the animated line, with no feature bullet list and no stock-style illustration.

## 5. Error handling

| Case | Behaviour |
|---|---|
| The session expires mid-use (401, or WS 4401) | Existing `sessionLost` flow, back to the login with `next=`. |
| The WebSocket drops | The existing banner, restyled as a glass toast "การเชื่อมต่อหลุด กำลังเชื่อมใหม่…". |
| An API call fails | An inline Thai message in the card that needed it; the rest of the page stays usable. |
| No alerts, no results, no taxonomy | Friendly empty states, never a blank card. |
| `backdrop-filter` unsupported | Opaque fallback colour (4.1). |

## 6. Testing

**Node harness** (`tests/product/test_web_static.py`). Tests for:
- the hero status derivation: normal, watch and alert, at the boundaries 0.8 τ, τ and 30 min;
- old route redirects, and presenter keys `1`–`4` and `?`;
- simulate one-click: the defaults sent are `region=test`, `offset_s=60` and the current symbol;
- the tour: shows once, `tour_done` is saved, it is skipped when done, and it can be replayed;
- the copy guard: the visible text of each page's default (closed-details) state contains none of `MT-T`, `recall_any`, `config_hash`, `pump_burst`, `region`, `F /`, `τ` outside details;
- role gates still hold, and Red-Team stays fail-closed.

**Static markup tests:** 4 nav items with Thai names, no theme toggle, the login split layout present, and a reduced-motion media query present.

**API test:** `PUT /api/me/prefs {"tour_done": true}` round-trips, and a non-bool value gives 422.

**Manual check in the browser pane**, at 1280×800 and 375×812:
- login;
- Home in normal and alert states;
- an alert detail and a verdict;
- a simulate run detected;
- Results;
- the help drawer and the tour;
- reduced motion.

**Gates:** `uv run ruff check .` and `uv run pytest -q -m "not network and not slow"` pass.
