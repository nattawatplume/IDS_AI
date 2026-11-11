/* Market IDS dashboard: one plain script, no build step, same-origin fetch and WebSocket only.
 *
 * Every server string reaches the DOM through textContent (`el`); nothing is parsed as HTML.
 * API times are µs UTC; the chart uses whole seconds (UTCTimestamp) and the plain view shows Thai time (UTC+7); UTC only appears in the folded details.
 * The chart library throws on an update older than the last point, so a bar or score whose time
 * is not after the last plotted one is dropped (one can arrive after a seek's alerts_reset).
 */
"use strict";

const LWC = typeof window === "undefined" ? undefined : window.LightweightCharts;
const S_US = 1e6;
const DAY_US = 86400 * S_US;
const AGREEMENT_P = 0.01;
const MATCH_BEFORE_US = 120 * S_US; // eval match rule: [t_start - 120 s, t_end + 300 s]
const MATCH_AFTER_US = 300 * S_US;
const MISSED_GRACE_US = 30 * S_US; // pipeline lag allowed past the matching window
const MARKER_SLACK_S = 10; // an alert further than this outside the plotted range gets no marker
const FUSION_LAYERS = ["rule", "stat", "ai"];
const LANES = [
  ["lane_rule", "L1 กฎ (rule)", "--line-l1"],
  ["lane_stat", "L2 สถิติ (stat)", "--line-l2"],
  ["lane_ai", "L3 AI", "--line-l3"],
];
const SEVERITY_TH = { 1: "ต่ำ", 2: "กลาง", 3: "สูง" };
const STATUS_TH = { open: "เปิด", updated: "อัปเดต", closed: "ปิดแล้ว" };
const CHECK_TH = { ok: "ปกติ", warn: "เฝ้าระวัง", fail: "มีปัญหา" };
// the auditor's check names, as a teacher would read them
const CHECK_NAME_TH = {
  alert_rate: "จำนวนการแจ้งเตือนเทียบงบ",
  calibration: "การเทียบมาตรฐานของคะแนน",
  calibration_resolution: "ความละเอียดของการเทียบมาตรฐาน",
  latency: "ความเร็วในการคำนวณ",
  trade_gaps: "ช่วงที่ข้อมูลการซื้อขายขาดหาย",
  exclusion: "ช่วงที่ตัดออกจากการตรวจ",
  staleness: "ความสดของข้อมูล",
  warmup: "การเก็บข้อมูลตั้งต้น",
};
// the home hero: an alert in the last 30 min of event time, else F near τ or an auditor warning
const HERO_WATCH_RATIO = 0.8;
const HERO_RECENT_US = 30 * 60 * S_US;
const SPEED_CHIPS = [
  { label: "1×", v: 1 },
  { label: "10×", v: 10 },
  { label: "60×", v: 60 },
  { label: "เร็วสุด", v: 0 },
];
const VIEWER_REPLAY_TH = "เฉพาะผู้วิเคราะห์ขึ้นไปที่ควบคุมได้";
const VIEWER_VERDICT_TH = "เฉพาะผู้วิเคราะห์ขึ้นไปที่ให้ผลตรวจได้";
const HEADLINE_MAX = 120;
const VERDICT_TH = { tp: "ผิดปกติจริง", fp: "เตือนผิด", unsure: "ไม่แน่ใจ" };
const VERDICTS = ["tp", "fp", "unsure"];
const ALERT_FILTERS = [
  { key: "all", label: "ทั้งหมด" },
  { key: "open", label: "ยังไม่ตรวจ" },
  { key: "tp", label: "ผิดปกติจริง" },
  { key: "fp", label: "เตือนผิด" },
];
const ALERT_PAGE = 50; // cards drawn at a time; "แสดงเพิ่ม" adds the next page
const TOAST_MS = 5000;
const TOAST_MAX = 3;
const RED_TEAM_MODES = ["replay", "sim"];
// the simulate page: one card per scenario family; a click runs it with SIM_DEFAULTS
const SIM_CARDS = [
  { family: "pump_burst", title: "ปั่นเร็ว", line: "ราคาและปริมาณพุ่งขึ้นในไม่กี่นาที", icon: "steep" },
  { family: "pump_stealth", title: "ปั่นแบบเนียน", line: "ค่อยๆ ดันราคาขึ้นช้าๆ", icon: "ramp" },
  { family: "wash_pingpong", title: "ซื้อขายวนหลอก", line: "ซื้อขายกันเองไปมาให้ดูคึกคัก", icon: "loop" },
];
const SIM_DEFAULTS = { region: "test", offset_s: 60 };
const SIM_REASON = {
  live: "ทดลองได้เฉพาะโหมดย้อนดู",
  viewer: "ต้องเป็นผู้วิเคราะห์ขึ้นไป",
  ended: "การย้อนดูจบแล้ว รอรอบใหม่",
};
const SIM_WAIT_TH = "รอยืนยันจากเซิร์ฟเวอร์ว่าเป็นโหมดย้อนดู";
const SIM_OFFSET_TH = "เวลาเริ่มต้องอยู่ระหว่าง 1 ถึง 3600 วินาที";
const SIM_PAST_END_TH = "เหตุการณ์จะเลยข้อมูลสุดท้าย ลองลดเวลาเริ่มในตั้งค่าขั้นสูง";
const SIM_SYMBOL_TH = "ไม่รู้จักเหรียญนี้";
const SIM_FAILED_TH = "เริ่มการทดลองไม่สำเร็จ ลองใหม่อีกครั้ง";
const REPLAY_ENDED_TH = "การเล่นย้อนหลังนี้จบแล้ว ต้องเริ่มใหม่"; // the server's 409 text (api/app.py)
const PROVENANCES = ["REAL", "SEMI-SYN", "SYN"];

// Eval targets: docs/superpowers/plans/2026-10-27-market-ids-mvp.md Task 20 Step 6 (spec section 6).
// "pair" cards need the AE-minus-baseline CI lower bound above 0; "reference" pairs show the
// diff and CI only (the MVP spec asks the AE to beat iforest / robust_z / kk, not random / pca).
// `label` (metric ids) is for the committee table; `title` is the plain Thai card title.
// The false-alarm target is "≤ the budget": its cap follows the chosen budget (`capIsBudget`).
const SEMISYN_TH = { pump_burst: "ปั่นเร็ว", pump_stealth: "ปั่นแบบเนียน", wash_pingpong: "ซื้อขายวนหลอก" };
const BASELINE_TH = {
  "baseline.iforest": "วิธีแยกจุดผิดปกติด้วยต้นไม้สุ่ม",
  "stat.robust_z": "วิธีวัดระยะห่างจากค่าปกติ",
  "baseline.kk": "กฎตรวจการปั่นจากงานวิจัยเดิม",
  "baseline.random": "การเตือนแบบสุ่ม",
  "baseline.pca": "วิธีลดมิติข้อมูล",
};
const EVAL_TARGETS = [
  { id: "real_recall", label: "recall_any ≥ 0.8 · REAL (real_t001)", title: "จับเหตุการณ์ในข้อมูลจริง", experiment: "real_t001", variant: "full", metric: "recall_any", budget: 2, cmp: ">=", threshold: 0.8, correct: true },
  { id: "real_fa", label: "เตือนผิด ≤ งบ · REAL", title: "การเตือนผิดโดยเฉลี่ย", experiment: "real_t001", variant: "full", metric: "fa_per_symbol_day", budget: 2, cmp: "<=", threshold: 2, capIsBudget: true },
  { id: "real_fa_hard", label: "เตือนผิด (hard) ≤ 0 · REAL", title: "การเตือนผิดในวันปกติ", experiment: "real_t001", variant: "full", metric: "fa_per_symbol_day_hard", budget: 2, cmp: "<=", threshold: 0 },
  { id: "latency_p99", label: "compute latency p99 < 2 s", title: "ความเร็วในการเตือน", cmp: "none", threshold: 2 },
  ...["pump_burst", "pump_stealth", "wash_pingpong"].map((v) => ({
    id: `semisyn_${v}`, label: `recall_any ≥ 0.8 · SEMI-SYN ${v}`, title: SEMISYN_TH[v], experiment: "semisyn", variant: v, metric: "recall_any", budget: 2, cmp: ">=", threshold: 0.8, correct: true,
  })),
  ...["baseline.iforest", "stat.robust_z", "baseline.kk"].map((b) => ({
    id: `ae_vs_${b}`, label: `AE เหนือ ${b}`, title: BASELINE_TH[b], experiment: null, variant: b, metric: "recall_any", budget: 2, cmp: "pair", threshold: 0,
  })),
  ...["baseline.random", "baseline.pca"].map((b) => ({
    id: `ae_vs_${b}`, label: `AE เทียบ ${b}`, title: BASELINE_TH[b], experiment: null, variant: b, metric: "recall_any", budget: 2, cmp: "pair", threshold: 0, reference: true,
  })),
];

const state = {
  mode: null,
  replaying: true, // a replay starts running; control() keeps this in step
  health: null,
  symbols: [],
  symbol: null,
  alerts: new Map(), // alert_id -> item
  feedback: new Map(), // alert_id -> verdict rows, oldest first (GET /api/feedback)
  alertFilter: "all", // the chosen ALERT_FILTERS key
  alertsShown: ALERT_PAGE,
  taxonomy: null,
  lastBarUs: null,
  ws: null,
  wsRetry: 0,
  wsEverOpen: false,
  route: null,
  detailChart: null,
  injections: [],
  simBusy: false, // an inject POST is on its way: the cards wait for it
  simEnded: false, // the server said this replay round has ended; a replay_restarted reopens
  railIds: null, // alert ids already on the rail; null until the first render
  alertCount: 0,
  alertCountKnown: false, // the count has been shown once: only a later rise flashes the card
  alertsFailed: false, // the first /api/alerts failed: the hero says the status could not be read
  alertsLoaded: false, // the first /api/alerts answered: before it, a count is not yet the real one
  hero: null, // the last heroStatus shown: the DOM is touched only when it changes
  speed: null, // the speed chip the server last accepted
  layersOpen: false,
  me: null, // GET /api/me: { username, display_name, role, auth_enabled, prefs }
  started: false, // the data boot (symbols, series, alerts, websocket) has run
  view: "loading", // bootView(): only the loader shows until /api/me answers
  bootError: null, // why the last /api/me gave no usable answer: { status } or { answered } or {}
  pendingHelp: false, // a help route asked before the app was entered: opened by enterApp()
  evalBudget: null, // saved Eval budget (string, as the option value)
  users: [], // the admin Users page rows
  audit: [], // Activity rows already loaded, newest first
  auditDone: false,
};

// ------------------------------------------------------------------ roles + sign-in state

const ROLE_ORDER = ["viewer", "analyst", "admin"];
const AUDIT_ACTIONS = [
  "login", "login_failed", "logout", "feedback", "replay_control", "redteam_inject",
  "user_add", "user_role", "user_disable", "user_enable", "user_passwd",
  "signup", "role_change", "user_seed",
];
// parent techniques whose taxonomy name_th holds Latin words ("wash trading"): the plain view names them here
const TECHNIQUE_TH = { "MM-T005": "การซื้อขายวนหลอก" };
const NEED_ANALYST_TH = "ต้องเป็นผู้วิเคราะห์ขึ้นไป";
const DEFAULT_ROUTE = "#/home";
const ROLE_TH = { viewer: "ผู้ชม", analyst: "ผู้วิเคราะห์", admin: "ผู้ดูแล" };
const MODE_TH = { replay: "ย้อนดู", live: "สด", sim: "จำลอง" };

function setMe(me) {
  state.me = me && typeof me === "object" ? me : null;
}

// mirrors auth.role_at_least: an unknown role or an unknown need is never allowed
function can(need) {
  const role = state.me && state.me.role;
  const have = ROLE_ORDER.indexOf(role);
  const want = ROLE_ORDER.indexOf(need);
  return have >= 0 && want >= 0 && have >= want;
}

// only an in-page route may be the sign-in return target: never a URL, never the sign-in page itself
function safeNext(raw) {
  if (typeof raw !== "string" || !raw.startsWith("#/") || raw.startsWith("#/login")) return DEFAULT_ROUTE;
  for (let i = 0; i < raw.length; i += 1) {
    if (raw.charCodeAt(i) < 32) return DEFAULT_ROUTE;
  }
  return raw;
}

// the session is gone (expired, signed out elsewhere): forget the user, or route() would send
// the sign-in page straight back to `next`, and every page that fetches would 401 in a loop
function sessionLost() {
  closeTour(false); // the tour and the drawer belong to the app, not the sign-in page
  closeHelp();
  setMe(null);
  const chip = typeof document !== "undefined" ? $("user-chip") : null;
  if (chip) chip.hidden = true;
  goLogin();
}

function goLogin() {
  const here = location.hash || "";
  if (here.startsWith("#/login")) return;
  const next = safeNext(here);
  location.hash = next === DEFAULT_ROUTE && !here.startsWith(DEFAULT_ROUTE) ? "#/login" : `#/login?next=${encodeURIComponent(next)}`;
}

// what the first paint may show: nothing but the loader until we know whether sign-in is on,
// then the sign-in page (auth on, no session) or the app; so the dashboard never flashes
function bootView(me, authEnabled) {
  if (authEnabled !== true && authEnabled !== false) return "loading";
  if (me || !authEnabled) return "app";
  return "login";
}

// the loader's text while /api/me gives no usable answer: no answer at all (network) differs from
// a server that answered with an error, so a broken deploy does not read as a dropped connection
function bootRetryText(err) {
  const status = err && err.status;
  if (isNum(status)) return `เซิร์ฟเวอร์ตอบกลับผิดพลาด (รหัส ${status}) กำลังลองใหม่`;
  if (err && err.answered) return "เซิร์ฟเวอร์ตอบกลับผิดพลาด กำลังลองใหม่";
  return "ยังติดต่อเซิร์ฟเวอร์ไม่ได้ กำลังลองใหม่";
}

// ms before retry number `attempt` (0-based): 3 s, 6 s, 12 s, 24 s, then 30 s
function bootRetryDelay(attempt) {
  const n = Number.isInteger(attempt) && attempt > 0 ? attempt : 0;
  return Math.min(3000 * 2 ** Math.min(n, 4), 30000);
}

// ------------------------------------------------------------------ helpers

function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "onclick") node.addEventListener("click", v);
    else node.setAttribute(k, v === true ? "" : String(v));
  }
  append(node, children);
  return node;
}

function append(node, children) {
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

function $(id) {
  return document.getElementById(id);
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function isNum(x) {
  return typeof x === "number" && Number.isFinite(x);
}

function fmtUs(us) {
  if (!isNum(us)) return "–";
  const d = new Date(Math.floor(us / 1000));
  if (Number.isNaN(d.getTime())) return "–";
  return d.toISOString().replace("T", " ").slice(0, 19) + " UTC";
}

function fmtSec(t) {
  return fmtUs(t * S_US);
}

// Thai clock time (Asia/Bangkok, UTC+7, no daylight saving) for plain view: "23:30 น."; "UTC" stays in details
const TH_OFFSET_US = 7 * 3600 * S_US;

function thaiClock(us, withSeconds) {
  if (!isNum(us)) return null;
  const d = new Date(Math.floor((us + TH_OFFSET_US) / 1000));
  if (Number.isNaN(d.getTime())) return null;
  return d.toISOString().slice(11, withSeconds ? 19 : 16);
}

function fmtThaiTime(us, withSeconds = false) {
  const t = thaiClock(us, withSeconds);
  return t === null ? "–" : `${t} น.`;
}

function fmtP(p) {
  if (!isNum(p)) return "–";
  return p < 0.001 ? p.toExponential(1) : p.toFixed(3);
}

function fmtNum(x, digits = 3) {
  if (!isNum(x)) return "–";
  if (digits > 0 && x !== 0 && Math.abs(x) < 10 ** -digits) return x.toExponential(1);
  return x.toFixed(digits);
}

function nlog10(p) {
  return isNum(p) && p > 0 ? -Math.log10(p) : null;
}

function toSec(us) {
  return Math.floor(us / S_US);
}

async function api(path, options) {
  const res = await fetch(path, options);
  let body = null;
  try {
    body = await res.json();
  } catch (e) {
    body = null;
  }
  if (res.status === 401 && path !== "/api/auth/login") sessionLost(); // a failed sign-in must not bounce to itself
  if (!res.ok) {
    const detail = body && body.detail;
    const msg =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((d) => d.msg).join("; ")
          : `คำขอไม่สำเร็จ (HTTP ${res.status})`;
    const err = new Error(msg);
    err.status = res.status;
    throw err;
  }
  return body;
}

function postJson(path, data) {
  return api(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(data),
  });
}

function safely(fn) {
  return (...args) => {
    try {
      return fn(...args);
    } catch (e) {
      console.warn("dashboard:", e);
      return undefined;
    }
  };
}

function storageGet(key) {
  try {
    return window.localStorage.getItem(key);
  } catch (e) {
    return null;
  }
}

function storageSet(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch (e) {
    /* private mode: the value just is not remembered */
  }
}

// ------------------------------------------------------------------ prefs

// the server still accepts a saved `theme`; the page has one look and ignores it
const PREF_KEYS = { symbol: "idsai.symbol", eval_budget: "idsai.eval_budget" };
let prefsPending = {};
let prefsTimer = null;

function authOn() {
  return !!(state.me && state.me.auth_enabled);
}

// the saved choices: the account's when auth is on, this browser's otherwise
function loadedPrefs() {
  if (authOn()) return state.me.prefs || {};
  const out = {};
  const symbol = storageGet(PREF_KEYS.symbol);
  if (symbol) out.symbol = symbol;
  const raw = storageGet(PREF_KEYS.eval_budget);
  if (raw && Number.isFinite(Number(raw))) out.eval_budget = Number(raw);
  return out;
}

function flushPrefs() {
  const body = prefsPending;
  prefsPending = {};
  prefsTimer = null;
  if (!Object.keys(body).length) return;
  api("/api/me/prefs", {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  })
    .then(() => {
      if (state.me) state.me.prefs = { ...state.me.prefs, ...body };
    })
    .catch((e) => console.warn("dashboard: prefs not saved:", e.message));
}

// auth off: this browser's localStorage, at once. auth on: the account, one PUT 500 ms after the last change
function savePrefs(patch) {
  if (!authOn()) {
    for (const [k, v] of Object.entries(patch || {})) {
      if (PREF_KEYS[k]) storageSet(PREF_KEYS[k], String(v));
    }
    return;
  }
  prefsPending = { ...prefsPending, ...patch };
  if (prefsTimer !== null) clearTimeout(prefsTimer);
  prefsTimer = setTimeout(flushPrefs, 500);
}

function provenanceBadges(text) {
  const parts = String(text || "")
    .split("/")
    .filter(Boolean);
  return parts.map((p) => el("span", { class: `badge ${PROVENANCES.includes(p) ? p : ""}` }, p));
}

function parentTechnique(id) {
  return String(id || "").split(".")[0];
}

// ------------------------------------------------------------------ top bar

// ------------------------------------------------------------------ help drawer + tour

const TOUR_KEY = "idsai.tour_done";
const TOUR_STEPS = [
  { target: "#hero-title", text: "ดูตรงนี้ก่อน บอกว่าตอนนี้ตลาดปกติหรือไม่" },
  { target: "#chart-card", text: "เส้นม่วงคือระดับความผิดปกติ ถ้าข้ามเส้นเตือนสีส้ม ระบบจะแจ้งเตือน" },
  { target: '[data-route="alerts"]', text: "รายการแจ้งเตือนทั้งหมด กดเข้าไปเพื่ออ่านคำอธิบายและบันทึกผลตรวจ" },
  { target: '[data-route="simulate"]', text: "ลองใส่การปั่นจำลองแล้วดูว่าระบบจับได้เร็วแค่ไหน" },
];
const BUBBLE_W = 320;
const BUBBLE_H = 190; // the guess used until the real height can be measured
const BUBBLE_GAP = 12;
const BUBBLE_MARGIN = 16;
const SPOT_PAD = 6;
const FOCUSABLE = 'button:not([disabled]), a[href], summary, input, select, [tabindex]:not([tabindex="-1"])';

const helpState = { open: false, returnTo: null };
const tourState = { open: false, i: 0, returnTo: null };

// the tour shows once per user: the account's pref when auth is on, this browser's otherwise
function tourShouldShow(prefs, auth, stored) {
  return auth ? !(prefs && prefs.tour_done) : stored !== "1";
}

function markTourDone() {
  if (authOn()) {
    state.me.prefs = { ...(state.me.prefs || {}), tour_done: true }; // no second tour this session
    savePrefs({ tour_done: true });
  } else {
    storageSet(TOUR_KEY, "1");
  }
}

// where the bubble goes: below the target, above it when there is no room, centred without a target;
// always fully inside the viewport. `size` is the measured bubble, when known
function placeBubble(rect, viewport, size) {
  const vw = viewport.w;
  const vh = viewport.h;
  const bw = Math.min((size && size.w) || BUBBLE_W, vw - 2 * BUBBLE_MARGIN);
  const bh = (size && size.h) || BUBBLE_H;
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(v, hi));
  const left = (x) => Math.round(clamp(x, BUBBLE_MARGIN, vw - bw - BUBBLE_MARGIN));
  const top = (y) => Math.round(clamp(y, BUBBLE_MARGIN, vh - bh - BUBBLE_MARGIN));
  if (!rect) return { top: top((vh - bh) / 2), left: left((vw - bw) / 2), centred: true };
  const below = rect.top + rect.height + BUBBLE_GAP;
  const above = rect.top - BUBBLE_GAP - bh;
  let y = below;
  if (below + bh > vh - BUBBLE_MARGIN) y = above >= BUBBLE_MARGIN ? above : vh - bh - BUBBLE_MARGIN;
  return { top: top(y), left: left(rect.left + rect.width / 2 - bw / 2), centred: false };
}

function focusables(root) {
  return Array.from(root.querySelectorAll(FOCUSABLE)).filter(
    (n) => !n.hidden && (typeof n.getClientRects !== "function" || n.getClientRects().length > 0),
  );
}

// Tab stays inside `root`
function trapTab(ev, root) {
  const items = focusables(root);
  if (!items.length) {
    ev.preventDefault();
    root.focus();
    return;
  }
  const first = items[0];
  const last = items[items.length - 1];
  const active = document.activeElement;
  if (!root.contains(active) || active === root || (ev.shiftKey && active === first)) {
    ev.preventDefault();
    (ev.shiftKey ? last : first).focus();
  } else if (!ev.shiftKey && active === last) {
    ev.preventDefault();
    first.focus();
  }
}

// false when the node or an ancestor is hidden, or it sits inside a closed <details> (not as its summary)
function isShown(node) {
  for (let n = node, child = null; n; child = n, n = n.parentNode) {
    if (n.hidden === true) return false;
    if (n.tagName === "DETAILS" && !n.open && !(child && child.tagName === "SUMMARY")) return false;
  }
  return true;
}

// back to where the user was; when that is hidden now (the tour button inside a closed menu),
// the user chip's summary, else the help button
function restoreFocus(node) {
  let to = node && typeof node.focus === "function" && node.isConnected !== false && isShown(node) ? node : null;
  if (!to) {
    const chip = $("user-chip");
    const summary = chip && !chip.hidden && typeof chip.querySelector === "function" ? chip.querySelector("summary") : null;
    to = summary || $("help-btn");
  }
  if (to && typeof to.focus === "function") to.focus();
}

// Escape closes the tour or the drawer, Tab stays inside it; true while one is open (no other key acts)
function onOverlayKey(ev) {
  if (!tourState.open && !helpState.open) return false;
  if (ev.key === "Escape") {
    ev.preventDefault();
    if (tourState.open) endTour();
    else closeHelp();
  } else if (ev.key === "Tab") {
    trapTab(ev, tourState.open ? $("tour-bubble") : $("help-drawer"));
  }
  return true;
}

// the taxonomy as plain cards: the Thai name and one line; the citation sits in a closed "แหล่งอ้างอิง"
function renderHelpTaxonomy(box, tax) {
  clear(box);
  for (const [id, t] of Object.entries(tax || {})) {
    const name = thaiTechniqueName(tax, id);
    if (!name) continue;
    box.appendChild(
      el(
        "div",
        { class: "help-card" },
        el("h4", {}, name),
        el("p", { class: "small" }, firstSentence(t.definition_th)),
        t.reference ? el("details", { class: "help-more" }, el("summary", {}, "แหล่งอ้างอิง"), el("p", { class: "small muted" }, t.reference)) : null,
      ),
    );
  }
  if (!box.firstChild) box.appendChild(el("p", { class: "empty" }, "ยังโหลดรายการรูปแบบไม่ได้"));
}

// `where`: the id of a drawer section to scroll to
function openHelp(where) {
  const drawer = $("help-drawer");
  if (!drawer || tourState.open) return;
  if (!helpState.open) {
    helpState.returnTo = document.activeElement || null;
    helpState.open = true;
    fxApply();
    $("help-backdrop").hidden = false;
    drawer.hidden = false;
    flash(drawer, "enter");
    loadTaxonomy().then((tax) => renderHelpTaxonomy($("help-taxonomy"), tax));
  }
  const section = typeof where === "string" ? $(where) : null;
  if (section && typeof section.scrollIntoView === "function") section.scrollIntoView({ block: "start" });
  $("help-close").focus();
}

function closeHelp() {
  if (!helpState.open) return;
  helpState.open = false;
  fxApply();
  $("help-drawer").hidden = true;
  $("help-backdrop").hidden = true;
  restoreFocus(helpState.returnTo);
  helpState.returnTo = null;
}

function showTourStep(i) {
  const step = TOUR_STEPS[i];
  if (!step) return;
  tourState.i = i;
  let target = null;
  try {
    target = document.querySelector(step.target);
  } catch (e) {
    /* a missing or odd target means a centred bubble */
  }
  const vw = (typeof window !== "undefined" && window.innerWidth) || 360;
  const vh = (typeof window !== "undefined" && window.innerHeight) || 640;
  let rect = null;
  if (target && typeof target.getBoundingClientRect === "function") {
    if (typeof target.scrollIntoView === "function") target.scrollIntoView({ block: "nearest" });
    const b = target.getBoundingClientRect();
    if (b.width > 0 && b.height > 0) {
      const top = Math.max(0, b.top - SPOT_PAD);
      const left = Math.max(0, b.left - SPOT_PAD);
      rect = {
        top,
        left,
        width: Math.min(vw, b.right + SPOT_PAD) - left,
        height: Math.min(vh, b.bottom + SPOT_PAD) - top,
      };
    }
  }
  const spot = $("tour-spot");
  spot.classList.toggle("none", !rect);
  for (const k of ["top", "left", "width", "height"]) spot.style[k] = rect ? `${Math.round(rect[k])}px` : "";
  $("tour-text").textContent = step.text;
  $("tour-count").textContent = `ขั้นที่ ${i + 1} จาก ${TOUR_STEPS.length}`;
  const last = i === TOUR_STEPS.length - 1;
  $("tour-next").hidden = last;
  $("tour-done").hidden = !last;
  const bubble = $("tour-bubble");
  const measured = bubble.offsetHeight > 0 ? { w: bubble.offsetWidth, h: bubble.offsetHeight } : null;
  const at = placeBubble(rect, { w: vw, h: vh }, measured);
  bubble.classList.toggle("centred", at.centred);
  bubble.style.top = `${at.top}px`;
  bubble.style.left = `${at.left}px`;
  (last ? $("tour-done") : $("tour-next")).focus();
}

function startTour() {
  if (tourState.open) return;
  if (helpState.open) closeHelp();
  tourState.open = true;
  fxApply();
  tourState.returnTo = document.activeElement || null;
  $("tour").hidden = false;
  showTourStep(0);
  if (state.route !== "home") {
    location.hash = DEFAULT_ROUTE; // the first two targets live on Home: look again once it shows
    setTimeout(() => {
      if (tourState.open && tourState.i === 0) showTourStep(0);
    }, 200);
  }
}

// any way out counts as seen: finished, skipped or Escape; a lost session is not "seen"
function closeTour(seen) {
  if (!tourState.open) return;
  tourState.open = false;
  fxApply();
  $("tour").hidden = true;
  if (seen) markTourDone();
  restoreFocus(tourState.returnTo);
  tourState.returnTo = null;
}

function endTour() {
  closeTour(true);
}

function maybeStartTour() {
  if (state.route === "home" && !helpState.open && tourShouldShow(loadedPrefs(), authOn(), storageGet(TOUR_KEY))) {
    startTour();
  }
}

function bindHelp() {
  $("help-btn").addEventListener("click", () => openHelp());
  $("help-close").addEventListener("click", closeHelp);
  $("help-backdrop").addEventListener("click", closeHelp);
  $("help-tour").addEventListener("click", startTour);
  $("tour-next").addEventListener("click", () => showTourStep(tourState.i + 1));
  $("tour-skip").addEventListener("click", endTour);
  $("tour-done").addEventListener("click", endTour);
  $("footer-about").addEventListener("click", (ev) => {
    ev.preventDefault();
    if (state.view === "app") openHelp(); // the link is only shown in the app; nothing is queued from elsewhere
  });
  window.addEventListener("resize", () => {
    if (tourState.open) showTourStep(tourState.i);
  });
}

// ------------------------------------------------------------------ motion + presenter keys

const PAGE_ORDER = ["#/home", "#/alerts", "#/simulate", "#/results"];

function reducedMotion() {
  const mm = typeof globalThis.matchMedia === "function" ? globalThis.matchMedia : null;
  return !!(mm && mm.call(globalThis, "(prefers-reduced-motion: reduce)").matches);
}

// ------------------------------------------------------------------ command-center motion (motion.js)

const FX_KEY = "idsai.fx_off";

// the only door to motion.js: it does nothing when that file is missing, and a throw never reaches the page
function fx(name, ...args) {
  try {
    globalThis.Motion?.[name]?.(...args);
  } catch (e) {
    /* decoration only: the page works without it */
  }
  return undefined;
}

// the "ลดเอฟเฟกต์" choice: the account's pref when auth is on and it has one, this browser's mirror otherwise
function fxUserOff() {
  const pref = authOn() ? loadedPrefs().fx_off : undefined;
  return typeof pref === "boolean" ? pref : storageGet(FX_KEY) === "1";
}

// the guided tour and the help drawer keep the screen still; the user's choice comes back afterwards
function fxOff() {
  return fxUserOff() || tourState.open || helpState.open;
}

function fxApply() {
  fx("setOff", fxOff());
  try {
    const sw = $("menu-fx");
    if (sw) sw.setAttribute("aria-pressed", String(fxUserOff()));
  } catch (e) {
    /* the switch is only a view of the choice */
  }
}

function setFxOff(value) {
  const v = !!value;
  storageSet(FX_KEY, v ? "1" : "0");
  if (authOn()) {
    state.me.prefs = { ...(state.me.prefs || {}), fx_off: v };
    savePrefs({ fx_off: v });
  }
  fxApply();
}

// hero level -> background mood: calm, watch (also while the state is still unknown), alert
function heroMood(level) {
  return level === "alert" ? "alert" : level === "normal" ? "calm" : "watch";
}

// count a number up or down to `to`; snaps when motion is reduced
function tweenNumber(node, to, digits, ms = 300) {
  if (!node) return;
  if (!isNum(to)) {
    node.textContent = fmtNum(to, digits);
    return;
  }
  const raf = typeof globalThis.requestAnimationFrame === "function" ? globalThis.requestAnimationFrame : null;
  if (reducedMotion() || !raf) {
    node.textContent = fmtNum(to, digits);
    return;
  }
  const from = Number.parseFloat(node.textContent);
  if (!Number.isFinite(from) || from === to) {
    node.textContent = fmtNum(to, digits);
    return;
  }
  if (node._tween) node._tween.cancelled = true;
  const job = { cancelled: false };
  node._tween = job;
  const t0 = performance.now();
  const step = (now) => {
    if (job.cancelled) return;
    const k = Math.min(1, (now - t0) / ms);
    const eased = 1 - (1 - k) * (1 - k) * (1 - k);
    node.textContent = fmtNum(k >= 1 ? to : from + (to - from) * eased, digits);
    if (k < 1) raf(step);
    else node._tween = null;
  };
  raf(step);
}

// re-trigger a one-shot CSS animation class
function flash(node, cls = "flash") {
  if (!node || reducedMotion()) return;
  node.classList.remove(cls);
  void node.offsetWidth; // restart the animation
  node.classList.add(cls);
}

// a field owns every key typed into it
const KEY_OWNERS = ["INPUT", "SELECT", "TEXTAREA"];
// a button, menu or link owns only Space (it presses, opens or follows); 1-4 and ? still work
const SPACE_OWNERS = ["BUTTON", "SUMMARY", "A"];

// "play", "help", one of PAGE_ORDER, or null; never while a field has focus
function presenterKey(ev) {
  if (!ev || ev.ctrlKey || ev.metaKey || ev.altKey) return null;
  const target = ev.target;
  if (target && (KEY_OWNERS.includes(target.tagName) || target.isContentEditable)) return null;
  // ev.code is the physical key: it still works when a Thai layout changes ev.key
  const key = ev.key || "";
  const code = ev.code || "";
  if (key === " " || code === "Space") return target && SPACE_OWNERS.includes(target.tagName) ? null : "play";
  if (key === "?" || (code === "Slash" && ev.shiftKey)) return "help";
  const digit = /^[1-4]$/.test(key) ? key : (/^(?:Digit|Numpad)([1-4])$/.exec(code) || [])[1];
  return digit ? PAGE_ORDER[Number(digit) - 1] : null;
}

// what Space should do now: the opposite of the replay state, nothing in LIVE or unknown mode
function playToggleAction(mode, playing) {
  if (!mode || mode === "live") return null;
  return playing ? "pause" : "play";
}

// one source of truth for the replay state, fed by control()'s successful answers
function noteReplayAction(action) {
  if (action === "play") state.replaying = true;
  else if (action === "pause") state.replaying = false;
}

function onPresenterKey(ev) {
  if (onOverlayKey(ev)) return; // the drawer or the tour owns the keys while it is open
  if (state.view !== "app" || state.route === "login") return; // nothing to present before sign-in
  const act = presenterKey(ev);
  if (!act) return;
  ev.preventDefault();
  if (act === "play") {
    const next = playToggleAction(state.mode, state.replaying);
    if (next) control(next);
  } else if (act === "help") {
    openHelp();
  } else {
    location.hash = act;
  }
}

// the top-bar pulse: on while data flows, off on pause or a dropped connection
function setLive(on) {
  const dot = $("live-dot");
  const clock = $("clock");
  if (dot) {
    dot.classList.toggle("live", !!on);
    dot.classList.toggle("off", !on); // grey whatever the mode lamp said
  }
  if (clock) clock.classList.toggle("dim", !on);
}

function renderHealth(h) {
  state.health = h;
  setMode(h.mode);
  if (isNum(h.event_us)) $("clock").textContent = fmtThaiTime(h.event_us, true);
  else if (isNum(state.lastBarUs)) $("clock").textContent = fmtThaiTime(state.lastBarUs, true);
  renderReplayClock(eventNowUs());
  renderHealthCard(h);
  renderBudget();
  renderHero();
  tickInjections(); // the Red-Team card moves with the clock above it
}

// "สุขภาพระบบ": the auditor's overall word, and its checks in Thai behind the card's summary
function renderHealthCard(h) {
  const checks = Array.isArray(h.checks) ? h.checks : [];
  $("health-dot").className = `dot ${h.overall || ""}`;
  $("health-text").textContent = CHECK_TH[h.overall] || "–";
  const warm = checks.find((c) => c.name === "warmup");
  let sub = "กดเพื่อดูรายการตรวจ";
  if (warm && warm.status !== "ok") {
    const hours = warm.value && warm.value.hours_remaining;
    sub = isNum(hours)
      ? `กำลังเก็บข้อมูลตั้งต้น (เหลือประมาณ ${hours.toFixed(1)} ชม.)`
      : "กำลังเก็บข้อมูลตั้งต้น";
  }
  $("health-sub").textContent = sub;
  const list = clear($("health-checks"));
  for (const c of checks) {
    list.appendChild(
      el(
        "li",
        {},
        el(
          "div",
          { class: "check-head" },
          el("span", { class: `dot ${c.status}` }),
          el("span", {}, CHECK_NAME_TH[c.name] || "การตรวจอื่น"),
          el("span", { class: `check-state ${c.status}` }, CHECK_TH[c.status] || "–"),
        ),
        c.detail_th ? el("div", { class: "muted small" }, c.detail_th) : null,
      ),
    );
  }
  if (isNum(h.late_trades) && h.late_trades > 0) {
    list.appendChild(el("li", { class: "muted small" }, `การซื้อขายที่มาช้า ${h.late_trades} รายการ`));
  }
}

function redTeamAllowed(mode) {
  // fail closed: only a mode the server confirmed as replay / sim enables injection
  return RED_TEAM_MODES.includes(mode);
}

function setMode(mode) {
  if (!mode || mode === state.mode) return;
  state.mode = mode;
  state.replaying = true; // a new mode starts with a running replay
  state.simEnded = false; // a new mode is a new replay round
  const lamp = $("live-dot");
  lamp.classList.remove("ok", "fail");
  lamp.classList.add(mode === "live" ? "fail" : "ok");
  const badge = $("mode-badge");
  badge.textContent = MODE_TH[mode] || String(mode);
  badge.className = `badge badge-mode ${mode}`;
  const live = mode === "live";
  $("replay-bar").hidden = live;
  renderPlayButton();
  applyRoleGates(); // the simulate cards open only in a confirmed replay / sim mode
}

// why the simulate cards are closed, in Thai; "" when they are open
function simReason(mode, analyst, ended) {
  if (!redTeamAllowed(mode)) return mode === "live" ? SIM_REASON.live : SIM_WAIT_TH;
  if (!analyst) return SIM_REASON.viewer;
  if (ended) return SIM_REASON.ended;
  return "";
}

function applySimGates() {
  const why = simReason(state.mode, can("analyst"), state.simEnded);
  for (const c of SIM_CARDS) $(`sim-${c.family}`).disabled = why !== "" || state.simBusy;
  for (const node of [$("rt-region"), $("rt-offset"), $("rt-symbol")]) node.disabled = why !== "";
  const box = $("sim-reason");
  box.hidden = why === "";
  box.textContent = why;
}

// UI convenience only: the server enforces the role on every route
function applyRoleGates() {
  const ok = can("analyst");
  const gate = (node) => {
    if (ok) node.removeAttribute("title");
    else node.title = NEED_ANALYST_TH;
  };
  const replay = [...["btn-play", "seek-input", "btn-seek"].map($), ...$("replay-bar").querySelectorAll(".chip")];
  for (const node of replay) {
    node.disabled = !ok;
    gate(node);
  }
  // a viewer still sees the bar, read-only, and is told why in Thai
  const reason = $("replay-reason");
  reason.hidden = ok;
  reason.textContent = ok ? "" : VIEWER_REPLAY_TH;
  applySimGates();
  for (const node of document.querySelectorAll(".feedback button, .feedback input")) {
    node.disabled = !ok;
    gate(node);
  }
  $("menu-activity").hidden = !can("admin");
  $("nav-users").hidden = !can("admin");
  renderPlayButton(); // the gate above cleared every title; the play button gets its "(กดแป้นเว้นวรรค)" back
}

function eventNowUs() {
  const h = state.health;
  if (h && isNum(h.event_us)) return h.event_us;
  return state.lastBarUs;
}

// "ใช้งบแจ้งเตือนไป x/y": the selected coin's alerts of the last 24 h against the per-coin daily budget
function renderBudget() {
  const h = state.health;
  const budget = h && h.budget_per_symbol_day;
  const n = renderAlertsKpi();
  const fill = $("budget-fill");
  const line = $("budget-line");
  if (!isNum(budget) || !isNum(n)) {
    line.textContent = "ใช้งบแจ้งเตือนไป –";
    line.removeAttribute("title");
    fill.style.width = "0";
    return;
  }
  const frac = budget > 0 ? n / budget : 0;
  fill.style.width = `${Math.min(frac, 1) * 100}%`;
  fill.className = `meter-fill${frac > 1 ? " over" : frac >= 0.8 ? " warn" : ""}`;
  line.textContent = `ใช้งบแจ้งเตือนไป ${n}/${budget}`;
  line.title = `งบคือจำนวนแจ้งเตือนที่ตั้งไว้ต่อเหรียญต่อวัน (${budget} ครั้ง)`;
}

// the card counts the selected symbol's alerts of the last 24 h (the unit of the budget); null without a clock
function renderAlertsKpi(now = eventNowUs()) {
  const box = $("alerts-count");
  if (!state.symbol || !isNum(now) || !state.alertsLoaded) {
    box.textContent = "–";
    state.alertCount = 0;
    state.alertCountKnown = false;
    return null;
  }
  let n = 0;
  for (const a of state.alerts.values()) {
    if (a.symbol === state.symbol && a.ts_event_us > now - DAY_US && a.ts_event_us <= now) n += 1;
  }
  const rose = state.alertCountKnown && n > state.alertCount;
  state.alertCount = n;
  if (state.alertsLoaded) state.alertCountKnown = true;
  tweenNumber(box, n, 0);
  if (rose) flash($("card-alerts"));
  return n;
}

// ------------------------------------------------------------------ home: hero status

// the one sentence at the top of Home; pure, so the rules are tested in node
function heroStatus({ nowUs, alerts, f, tau, auditor, symbol } = {}) {
  let recent = null;
  if (isNum(nowUs)) {
    for (const a of alerts || []) {
      if (!a || !isNum(a.ts_event_us) || a.ts_event_us < nowUs - HERO_RECENT_US || a.ts_event_us > nowUs) continue;
      if (!recent || a.ts_event_us > recent.ts_event_us) recent = a;
    }
  }
  if (recent) {
    const where = recent.symbol || symbol || "";
    return {
      level: "alert",
      title: "พบความผิดปกติทางสถิติ",
      line: `ระบบพบรูปแบบการซื้อขายที่ต่างจากปกติของ ${where} เมื่อ ${fmtThaiTime(recent.ts_event_us)}`,
      alertId: typeof recent.alert_id === "string" ? recent.alert_id : null,
    };
  }
  const near = isNum(f) && isNum(tau) && f >= HERO_WATCH_RATIO * tau;
  if (near || auditor === "warn") {
    const over = near && f >= tau;
    return {
      level: "watch",
      title: "กำลังเฝ้าระวัง",
      line: over ? "ระดับความผิดปกติข้ามเส้นเตือน" : near ? "ระดับความผิดปกติใกล้เส้นเตือน" : "ระบบตรวจตัวเองพบจุดที่ควรดู",
      alertId: null,
    };
  }
  return { level: "normal", title: "ตลาดปกติ", line: "ไม่มีแจ้งเตือนใน 30 นาทีล่าสุด", alertId: null };
}

// before the first alerts answer, "no alert" is not yet known: never claim a calm market
function heroPending(failed) {
  return {
    level: "wait",
    title: failed ? "ยังอ่านสถานะไม่ได้" : "กำลังอ่านสถานะ",
    line: failed ? "อ่านรายการแจ้งเตือนไม่ได้ จึงยังบอกสถานะตลาดไม่ได้ ระบบจะลองใหม่เมื่อเชื่อมต่อได้" : "",
    alertId: null,
  };
}

// runs on every health, score and alert update; writes the DOM only when the sentence changes
function renderHero() {
  if (!state.health && !isNum(state.lastBarUs) && !state.alertsFailed) return; // no clock yet: keep "กำลังอ่านสถานะ"
  const next = !state.alertsLoaded ? heroPending(state.alertsFailed) : heroStatus({
    nowUs: eventNowUs(),
    alerts: state.alerts.values(),
    f: chart.now.f,
    tau: chart.tau,
    auditor: state.health && state.health.overall,
    symbol: state.symbol,
  });
  const prev = state.hero;
  if (prev && prev.level === next.level && prev.title === next.title && prev.line === next.line && prev.alertId === next.alertId) {
    return;
  }
  state.hero = next;
  fx("mood", heroMood(next.level));
  $("hero").dataset.level = next.level;
  $("hero-title").textContent = next.title;
  const line = clear($("hero-line"));
  line.appendChild(next.alertId ? el("a", { href: alertHref(next.alertId) }, next.line) : document.createTextNode(next.line));
  if (!prev || prev.level !== next.level) {
    const orb = $("hero-orb");
    orb.className = `orb ${next.level}`;
    flash(orb, "orb-change");
  }
}

// the replay bar's clock: the event time, and how far through its Thai day it is
function renderReplayClock(now) {
  const clock = $("replay-clock");
  const fill = $("replay-fill");
  if (!isNum(now)) {
    clock.textContent = "–";
    fill.style.width = "0";
    return;
  }
  clock.textContent = `เวลาในข้อมูล ${fmtThaiTime(now, true)}`;
  clock.title = fmtUs(now);
  const th = now + TH_OFFSET_US;
  const day = (((th % DAY_US) + DAY_US) % DAY_US) / DAY_US;
  fill.style.width = `${(day * 100).toFixed(2)}%`;
}

// the first sentence of an explanation, for a one-line headline: it ends at a "." or ";" that is
// followed by a space or the end (so 3.5 and 1.2e-5 stay whole), and is cut to 120 characters
function firstSentence(text) {
  const flat = String(text ?? "").replace(/\s+/g, " ").trim();
  const stop = /[.;](?= |$)/.exec(flat);
  const head = (stop ? flat.slice(0, stop.index) : flat).trim();
  return head.length > HEADLINE_MAX ? `${head.slice(0, HEADLINE_MAX - 1)}…` : head;
}


// ------------------------------------------------------------------ monitor chart

const chart = {
  api: null,
  price: null,
  f: null,
  fGlow: null, // a wide faint copy under the F line: its glow
  lanes: {},
  tau: null,
  tauLine: null,
  markers: null,
  last: {}, // series key -> last plotted time (s)
  priceTimes: [], // sorted times of the price series, for snapping markers
  now: { price: null, f: null }, // latest plotted values, shown while the pointer is away
  hover: false, // the pointer is on the chart: the readout follows it, not the stream
};

function chartColors() {
  return {
    bg: cssVar("--panel"),
    text: cssVar("--dim"),
    grid: cssVar("--line"),
    price: cssVar("--line-price"),
    f: cssVar("--line-f"),
    fGlow: cssVar("--line-f-glow"),
    tau: cssVar("--tau"),
    shade: cssVar("--shade"),
  };
}

function baseChartOptions() {
  const c = chartColors();
  return {
    autoSize: true,
    layout: {
      background: { type: "solid", color: c.bg },
      textColor: c.text,
      fontFamily: cssVar("--font"),
      attributionLogo: true,
      panes: { separatorColor: c.grid },
    },
    grid: { vertLines: { color: c.grid }, horzLines: { color: c.grid } },
    rightPriceScale: { borderColor: c.grid },
    timeScale: {
      borderColor: c.grid,
      timeVisible: true,
      secondsVisible: true,
      // ticks in Thai time too: a day mark shows the date (dd/mm), the rest the clock
      tickMarkFormatter: (t, kind) => {
        if (!isNum(t)) return "";
        if (kind <= 2) return new Date((t * S_US + TH_OFFSET_US) / 1000).toISOString().slice(5, 10).split("-").reverse().join("/");
        return thaiClock(t * S_US, kind === 4);
      },
    },
    localization: { timeFormatter: (t) => (isNum(t) ? fmtThaiTime(t * S_US, true) : "") },
  };
}

function lineOptions(color, title, extra) {
  return {
    color,
    title,
    lineWidth: 1.5,
    priceLineVisible: false,
    lastValueVisible: true,
    ...extra,
  };
}

function initChart() {
  if (!LWC || chart.api) return;
  const c = chartColors();
  chart.api = LWC.createChart($("chart"), baseChartOptions());
  chart.price = chart.api.addSeries(
    LWC.LineSeries,
    lineOptions(c.price, "ราคา", { priceFormat: { type: "price", precision: 8, minMove: 1e-8 } }),
    0,
  );
  // the glow goes first so the F line draws on top of it; it has no label and no crosshair dot
  chart.fGlow = chart.api.addSeries(
    LWC.LineSeries,
    lineOptions(c.fGlow, "", { lineWidth: 6, lastValueVisible: false, crosshairMarkerVisible: false }),
    1,
  );
  chart.f = chart.api.addSeries(
    LWC.LineSeries,
    lineOptions(c.f, "ระดับความผิดปกติ", {
      lineWidth: 2,
      // keep the τ line in view even while F stays far below it
      autoscaleInfoProvider: (original) => {
        const r = original();
        if (!r || !r.priceRange || !isNum(chart.tau)) return r;
        r.priceRange.maxValue = Math.max(r.priceRange.maxValue, chart.tau * 1.05);
        return r;
      },
    }),
    1,
  );
  if (state.layersOpen) addLanes();
  chart.markers = LWC.createSeriesMarkers(chart.price, []);
  const panes = chart.api.panes();
  if (panes[0]) panes[0].setStretchFactor(3);
  if (panes[1]) panes[1].setStretchFactor(1.4);
  // one chart, one crosshair: it already spans the price, F and layer panes; the readout follows it
  chart.api.subscribeCrosshairMove(onCrosshair);
  renderReadout(null, chart.now.price, chart.now.f);
}

function addLanes() {
  LANES.forEach(([key, title, color], i) => {
    chart.lanes[key] = chart.api.addSeries(LWC.LineSeries, lineOptions(cssVar(color), title), 2 + i);
  });
}

function removeLanes() {
  for (const [key] of LANES) {
    if (chart.lanes[key]) chart.api.removeSeries(chart.lanes[key]);
    chart.lanes[key] = null;
    delete chart.last[key];
  }
}

// the L1/L2/L3 panes are folded away by default; opening them re-reads the series
function setLayersOpen(open) {
  state.layersOpen = open;
  const btn = $("layers-toggle");
  btn.setAttribute("aria-expanded", String(open));
  btn.textContent = open ? "ซ่อน L1/L2/L3" : "แสดง L1/L2/L3";
  $("chart").classList.toggle("lanes", open); // three more panes need more height
  if (!chart.api) return;
  if (open) {
    addLanes();
    loadSeries();
  } else {
    removeLanes();
  }
}

function bindMonitor() {
  $("layers-toggle").addEventListener("click", () => setLayersOpen(!state.layersOpen));
}

// the crosshair in plain words: when, the price, and how the level stands against the warning line
function readoutText(time, price, f, tau) {
  const when = isNum(time) ? `เวลา ${fmtThaiTime(time * S_US, true)}` : "ล่าสุด";
  const parts = [when, `ราคา ${isNum(price) ? fmtNum(price, 8) : "–"}`];
  let level = `ระดับความผิดปกติ ${isNum(f) ? fmtNum(f, 2) : "–"}`;
  if (isNum(f) && isNum(tau)) level += f >= tau ? ` (ถึงเส้นเตือน ${fmtNum(tau, 2)})` : ` (เส้นเตือน ${fmtNum(tau, 2)})`;
  parts.push(level);
  return parts.join(" · ");
}

function renderReadout(time, price, f) {
  $("readout").textContent = readoutText(time, price, f, chart.tau);
}

function renderSymbolPrice() {
  const p = chart.now.price;
  $("symbol-price").textContent = `ราคาล่าสุด ${isNum(p) ? fmtNum(p, 8) : "–"}`;
}

function onCrosshair(param) {
  if (!param || !isNum(param.time) || !param.seriesData) {
    chart.hover = false;
    renderReadout(null, chart.now.price, chart.now.f); // pointer left: show the latest
    return;
  }
  chart.hover = true;
  const at = (series) => {
    const pt = series && param.seriesData.get(series);
    return pt && isNum(pt.value) ? pt.value : null;
  };
  renderReadout(param.time, at(chart.price), at(chart.f));
}

function seriesData(times, values) {
  const out = [];
  let last = -Infinity;
  for (let i = 0; i < times.length; i += 1) {
    const t = times[i];
    if (!isNum(t) || t <= last) continue; // strictly increasing, as the chart requires
    last = t;
    out.push(isNum(values[i]) ? { time: t, value: values[i] } : { time: t });
  }
  return out;
}

function lastValue(data) {
  for (let i = data.length - 1; i >= 0; i -= 1) if ("value" in data[i]) return data[i].value;
  return null;
}

function lastTime(data) {
  return data.length ? data[data.length - 1].time : -Infinity;
}

function setTau(tau) {
  if (!chart.f) return;
  chart.tau = isNum(tau) ? tau : null;
  if (chart.tauLine) {
    chart.f.removePriceLine(chart.tauLine);
    chart.tauLine = null;
  }
  if (isNum(tau)) {
    chart.tauLine = chart.f.createPriceLine({
      price: tau,
      color: cssVar("--tau"),
      lineWidth: 1,
      lineStyle: LWC.LineStyle ? LWC.LineStyle.Dashed : 2,
      axisLabelVisible: true,
      title: "เส้นเตือน",
    });
  }
}

async function loadSeries() {
  if (!state.symbol || !chart.api) return;
  const symbol = state.symbol;
  let s;
  try {
    s = await api(`/api/series?symbol=${encodeURIComponent(symbol)}`);
  } catch (e) {
    console.warn("series:", e.message);
    return;
  }
  if (symbol !== state.symbol) return; // the picker moved on meanwhile
  const times = (s.t_end_us || []).map((t) => (isNum(t) ? toSec(t) : null));
  const price = seriesData(times, s.close || []);
  chart.price.setData(price);
  chart.priceTimes = price.filter((p) => "value" in p).map((p) => p.time);
  chart.last = { price: lastTime(price) };
  const f = seriesData(times, s.F || []);
  chart.f.setData(f);
  chart.fGlow.setData(f);
  chart.last.f = lastTime(f);
  chart.now = { price: lastValue(price), f: lastValue(f) };
  for (const [key] of LANES) {
    if (!chart.lanes[key]) continue;
    const d = seriesData(times, s[key] || []);
    chart.lanes[key].setData(d);
    chart.last[key] = lastTime(d);
  }
  setTau(s.tau);
  renderReadout(null, chart.now.price, chart.now.f);
  renderSymbolPrice();
  $("chart-empty").hidden = price.length > 0;
  renderMarkers();
  renderHero();
}

function nearestTime(t) {
  const arr = chart.priceTimes;
  if (!arr.length) return null;
  let lo = 0;
  let hi = arr.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < t) lo = mid + 1;
    else hi = mid;
  }
  if (lo > 0 && Math.abs(arr[lo - 1] - t) <= Math.abs(arr[lo] - t)) lo -= 1;
  return arr[lo];
}

function sevColor(sev) {
  return sev >= 3 ? cssVar("--red") : cssVar("--amber");
}

function renderMarkers() {
  if (!chart.markers) return;
  const seen = new Set();
  const markers = [];
  const items = [...state.alerts.values()]
    .filter((a) => a.symbol === state.symbol)
    .sort((a, b) => a.ts_event_us - b.ts_event_us);
  const arr = chart.priceTimes;
  for (const a of items) {
    const ts = toSec(a.ts_event_us);
    if (!arr.length || ts < arr[0] - MARKER_SLACK_S || ts > arr[arr.length - 1] + MARKER_SLACK_S) {
      continue; // outside the plotted range: never pinned onto its edge
    }
    const t = nearestTime(ts);
    if (t === null || seen.has(t)) continue;
    seen.add(t);
    markers.push({
      time: t,
      position: "aboveBar",
      shape: "arrowDown",
      color: sevColor(a.severity), // colour and shape only: no technique id on the Home chart
    });
  }
  markers.sort((a, b) => a.time - b.time);
  chart.markers.setMarkers(markers);
}

function plot(key, series, t, value) {
  if (!series || !isNum(value) || !isNum(t)) return false;
  const last = chart.last[key] ?? -Infinity;
  if (t <= last) return false; // stale (e.g. after a seek): dropped on purpose
  series.update({ time: t, value });
  chart.last[key] = t;
  if (key === "price" || key === "f") chart.now[key] = value;
  return true;
}

function onBar(bar) {
  if (!bar || !isNum(bar.t_us)) return;
  const closeUs = bar.t_us + S_US;
  if (!isNum(state.lastBarUs) || closeUs > state.lastBarUs) state.lastBarUs = closeUs;
  if (state.mode === "live") $("clock").textContent = fmtThaiTime(state.lastBarUs, true);
  if (bar.symbol !== state.symbol || !chart.api) return;
  const t = toSec(closeUs); // a bar closes at t_us + 1 s, where its score row ends
  if (plot("price", chart.price, t, bar.close)) {
    chart.priceTimes.push(t);
    $("chart-empty").hidden = true;
    renderSymbolPrice();
  }
}

function onScore(p) {
  if (!p || p.symbol !== state.symbol || !chart.api || !isNum(p.t_end_us)) return;
  const t = toSec(p.t_end_us);
  if (plot("f", chart.f, t, p.F)) {
    chart.fGlow.update({ time: t, value: p.F });
    if (!chart.hover) renderReadout(null, chart.now.price, chart.now.f);
  }
  for (const [key] of LANES) plot(key, chart.lanes[key], t, p[key]);
  renderHero();
}

// ------------------------------------------------------------------ replay controls

async function control(action, value) {
  const status = $("replay-status");
  if (!can("analyst")) {
    status.textContent = NEED_ANALYST_TH;
    return false;
  }
  status.textContent = "…";
  try {
    const body = value === undefined ? { action } : { action, value };
    const r = await postJson("/api/replay/control", body);
    noteReplayAction(action);
    if (action === "play" || action === "seek") {
      state.simEnded = false; // the server took it: the cursor can move again, let the cards try
      applySimGates();
    }
    if (action === "play" || action === "pause") setLive(state.replaying);
    renderPlayButton();
    // the button and the chips already show play and speed; only a jump says where it landed
    status.textContent = action === "seek" ? `ไปที่ ${fmtThaiTime(r.cursor_us, true)} แล้ว` : "";
    return true;
  } catch (e) {
    console.warn("replay control:", e && e.message);
    status.textContent = e && e.status === 409 && e.message === REPLAY_ENDED_TH ? REPLAY_ENDED_TH : plainErrorText(e);
    return false;
  }
}

// one big button: ❚❚ while the replay runs, ▶ while it is paused
function renderPlayButton() {
  const btn = $("btn-play");
  const playing = state.replaying;
  btn.textContent = playing ? "❚❚" : "▶";
  const label = playing ? "หยุดชั่วคราว" : "เล่นต่อ";
  btn.setAttribute("aria-label", label);
  if (can("analyst")) btn.title = `${label} (กดแป้นเว้นวรรค)`;
}

function renderSpeedChips() {
  for (const chip of $("replay-bar").querySelectorAll(".chip")) {
    chip.setAttribute("aria-pressed", String(Number(chip.dataset.speed) === state.speed));
  }
}

// the "go to time" box takes Thai time (UTC+7, the time shown everywhere): the server wants UTC microseconds
function seekInputToUs(raw) {
  const text = String(raw ?? "");
  if (!text) return null;
  const ms = Date.parse(text.length === 16 ? `${text}:00Z` : `${text}Z`); // read as if UTC, then take the 7 h back
  return Number.isNaN(ms) ? null : Math.round(ms) * 1000 - TH_OFFSET_US;
}

function bindReplay() {
  $("btn-play").addEventListener("click", () => {
    const next = playToggleAction(state.mode, state.replaying);
    if (next) control(next);
  });
  const chips = $("replay-bar").querySelectorAll(".chip");
  chips.forEach((chip, i) => {
    const c = SPEED_CHIPS[i];
    chip.textContent = c.label;
    chip.dataset.speed = String(c.v);
    chip.addEventListener("click", async () => {
      if (await control("speed", c.v)) {
        state.speed = c.v;
        renderSpeedChips();
      }
    });
  });
  $("btn-seek").addEventListener("click", () => {
    const raw = $("seek-input").value;
    const us = seekInputToUs(raw);
    if (us === null) {
      $("replay-status").textContent = "ใส่วันและเวลา (เวลาไทย) ก่อน";
      return;
    }
    control("seek", us);
  });
}

// ------------------------------------------------------------------ alerts

async function loadAlerts() {
  const err = $("alerts-error");
  let items;
  let rows;
  try {
    [items, rows] = await Promise.all([
      api("/api/alerts?limit=1000"),
      api("/api/feedback").catch((e) => {
        console.warn("feedback:", e.message);
        return null; // the cards still show; only the verdict chips and filters lack data
      }),
    ]);
  } catch (e) {
    console.warn("alerts:", e.message);
    err.textContent = `อ่านรายการแจ้งเตือนไม่ได้ (${plainErrorText(e)}) จะลองใหม่เมื่อเชื่อมต่อได้`;
    err.hidden = false;
    state.alertsFailed = true;
    renderAlertsKpi();
    renderHero();
    return;
  }
  state.alerts = new Map(items.map((a) => [a.alert_id, a]));
  if (rows) state.feedback = groupFeedback(rows);
  state.alertsLoaded = true;
  state.alertsFailed = false;
  err.textContent = rows ? "" : "อ่านผลการตรวจไม่ได้ ตัวกรองตามผลตรวจอาจยังไม่ครบ";
  err.hidden = !!rows;
  alertsChanged();
}

function groupFeedback(rows) {
  const out = new Map();
  for (const f of rows || []) {
    if (!f || typeof f.alert_id !== "string") continue;
    if (!out.has(f.alert_id)) out.set(f.alert_id, []);
    out.get(f.alert_id).push(f);
  }
  return out;
}

function alertsChanged() {
  renderAlertCards();
  renderRail();
  renderMarkers();
  renderBudget();
  renderHero();
  checkInjections();
}

function onAlert(data) {
  if (!data || typeof data !== "object") return;
  if (data.event === "resync") {
    loadAlerts(); // a re-read, not news: no toast
    return;
  }
  if (data.event === "replay_restarted") {
    restartRuns(state.injections);
    renderInjections();
    state.simEnded = false; // a new round: the cards open again
    applySimGates();
    return;
  }
  if (data.event === "alerts_reset") {
    // verdicts stay: the server keeps them, and a replayed alert comes back under the same id
    for (const id of data.deleted || []) state.alerts.delete(id);
    if (resetRuns(state.injections, data.deleted || [])) renderInjections();
    alertsChanged();
    loadSeries();
    return;
  }
  if (typeof data.alert_id !== "string") return;
  const fresh = !state.alerts.has(data.alert_id); // an update of a known alert is not a new one
  state.alerts.set(data.alert_id, data);
  alertsChanged();
  if (fresh) showToast(`แจ้งเตือนใหม่ · ${railHeadline(data, state.taxonomy)}`, alertHref(data.alert_id));
}

// a toast in the top right for TOAST_MS; at most TOAST_MAX at once, the oldest makes room
function showToast(text, href) {
  const root = $("toast-root");
  if (!root) return;
  while (root.children.length >= TOAST_MAX) root.removeChild(root.firstChild);
  const toast = el("div", { class: reducedMotion() ? "toast" : "toast slide" });
  const close = () => toast.remove();
  append(toast, [
    href ? el("a", { class: "toast-text", href, onclick: close }, text) : el("span", { class: "toast-text" }, text),
    el("button", { type: "button", class: "toast-close", "aria-label": "ปิด", onclick: close }, "×"),
  ]);
  root.appendChild(toast);
  setTimeout(close, TOAST_MS);
}

function sortedAlerts() {
  return [...state.alerts.values()].sort((a, b) => b.ts_event_us - a.ts_event_us);
}

function alertHref(id) {
  return `#/alerts/${encodeURIComponent(id)}`;
}

function sevKey(severity) {
  return severity >= 3 ? "high" : severity === 2 ? "mid" : "low";
}

// the verdict of the newest row (by time; a tie goes to the later row), or null
function latestVerdict(rows) {
  let best = null;
  for (const f of rows || []) {
    if (f && (!best || !(f.ts_us < best.ts_us))) best = f;
  }
  return best ? best.verdict : null;
}

// alerts carrying their `feedback` rows, kept by ALERT_FILTERS key, newest first; never mutates
function filterAlerts(alerts, key) {
  return [...(alerts || [])]
    .filter((a) => {
      if (key === "open") return !(a.feedback && a.feedback.length);
      if (key === "tp" || key === "fp") return latestVerdict(a.feedback) === key;
      return true;
    })
    .sort((x, y) => y.ts_event_us - x.ts_event_us);
}

function withFeedback(a) {
  return { ...a, feedback: state.feedback.get(a.alert_id) || [] };
}

function verdictChip(verdict) {
  return verdict ? el("span", { class: `verdict-chip v-${verdict}` }, VERDICT_TH[verdict] || verdict) : null;
}

function renderAlertFilters(all) {
  const box = clear($("alerts-filters"));
  for (const flt of ALERT_FILTERS) {
    const n = filterAlerts(all, flt.key).length;
    box.appendChild(
      el(
        "button",
        {
          type: "button",
          class: "filter-chip",
          "aria-pressed": state.alertFilter === flt.key ? "true" : "false",
          onclick: () => {
            state.alertFilter = flt.key;
            state.alertsShown = ALERT_PAGE;
            renderAlertCards();
          },
        },
        flt.label,
        el("span", { class: "filter-count" }, n),
      ),
    );
  }
}

function alertCard(a) {
  return el(
    "a",
    { class: "card alert-card", href: alertHref(a.alert_id), "data-sev": sevKey(a.severity) },
    el(
      "span",
      { class: "alert-card-main" },
      el("span", { class: "alert-card-text" }, patternText(a, state.taxonomy)),
      el(
        "span",
        { class: "alert-card-meta small muted" },
        el("span", { class: `sev sev-${a.severity}` }, `ระดับ${SEVERITY_TH[a.severity] || "–"}`),
        el("span", {}, a.symbol),
      ),
    ),
    el(
      "span",
      { class: "alert-card-side" },
      verdictChip(latestVerdict(a.feedback)),
      el("span", { class: "alert-card-time" }, fmtThaiTime(a.ts_event_us)),
    ),
  );
}

// the alerts page: filter chips with counts, then one glass card per alert, newest first
function renderAlertCards() {
  const list = clear($("alerts-list"));
  const all = [...state.alerts.values()].map(withFeedback);
  $("alerts-empty").hidden = all.length > 0;
  $("alerts-filters").hidden = all.length === 0;
  renderAlertFilters(all);
  const items = filterAlerts(all, state.alertFilter);
  for (const a of items.slice(0, state.alertsShown)) list.appendChild(alertCard(a));
  $("alerts-none").hidden = !(all.length && !items.length);
  const more = $("alerts-more");
  const left = items.length - state.alertsShown;
  more.hidden = left <= 0;
  more.textContent = `แสดงเพิ่ม (เหลืออีก ${left} รายการ)`;
}

// a technique's Thai name for plain view: the parent's name_th without its English "(...)" gloss;
// null when what is left still has Latin letters, or for MM-T000 (not yet classified)
function thaiTechniqueName(tax, techniqueId) {
  const parent = parentTechnique(techniqueId);
  const t = tax && tax[parent];
  if (!t || typeof t.name_th !== "string" || /-T000$/.test(parent)) return null;
  if (TECHNIQUE_TH[parent]) return TECHNIQUE_TH[parent]; // a name whose taxonomy text keeps Latin words
  const name = t.name_th.replace(/\s*\([^)]*\)/g, "").trim();
  return name && !/[A-Za-z]/.test(name) ? name : null;
}

// what was seen, in Thai and built from fields: the pattern's name, or the plain fallback
function patternText(a, tax) {
  const name = thaiTechniqueName(tax, a && a.technique_id);
  return name ? `รูปแบบคล้าย${name}` : "พบความผิดปกติทางสถิติ";
}

// the rail's headline, built from fields: the pattern in Thai, the coin and the Thai time
function railHeadline(a, tax) {
  return `${patternText(a, tax)} ที่ ${(a && a.symbol) || "–"} เวลา ${fmtThaiTime(a && a.ts_event_us)}`;
}

// the newest five alerts beside the chart as mini-cards; a card that was not there before rises in
function renderRail() {
  const box = clear($("alert-rail"));
  const items = sortedAlerts().slice(0, 5);
  $("rail-empty").hidden = items.length > 0;
  const seen = state.railIds;
  for (const a of items) {
    const fresh = seen !== null && !seen.has(a.alert_id);
    box.appendChild(
      el(
        "a",
        { class: `rail-row sev-${a.severity}${fresh ? " enter" : ""}`, href: alertHref(a.alert_id) },
        el(
          "span",
          { class: "rail-meta small" },
          el("span", { class: `rail-dot sev-${a.severity}`, "aria-hidden": "true" }),
          el("span", { class: "rail-sev muted" }, `ระดับ${SEVERITY_TH[a.severity] || "–"}`),
        ),
        el("span", { class: "rail-headline" }, railHeadline(a, state.taxonomy)),
      ),
    );
  }
  state.railIds = new Set(items.map((a) => a.alert_id));
}

async function loadTaxonomy() {
  if (state.taxonomy) return state.taxonomy;
  try {
    state.taxonomy = await api("/api/taxonomy");
  } catch (e) {
    console.warn("taxonomy:", e.message);
    state.taxonomy = null;
  }
  return state.taxonomy;
}

function destroyDetailChart() {
  if (state.detailChart) {
    state.detailChart.remove();
    state.detailChart = null;
  }
}

// one explanation fact for plain view, or null when it would need English: "robust z = 7.1"
// reads as how many usual swings above (or below) normal; anything else with Latin letters stays
// for "ดูรายละเอียด"
function plainFact(f) {
  if (!f || typeof f.label !== "string" || typeof f.value !== "string") return null;
  let value = f.value;
  const z = /^robust z = (-?[\d.]+)$/.exec(value);
  if (z) {
    const x = Number(z[1]);
    value = `${x >= 0 ? "สูง" : "ต่ำ"}กว่าปกติ ${Math.abs(x)} เท่าของช่วงแกว่งปกติ`;
  }
  const text = `${f.label}: ${value}`;
  return /[A-Za-z]/.test(text) ? null : text;
}

function kv(pairs) {
  return el(
    "dl",
    { class: "kv" },
    pairs.flatMap(([k, v, cls]) => [el("dt", {}, k), el("dd", cls ? { class: cls } : {}, v)]),
  );
}

async function showAlert(id) {
  const box = clear($("alert-detail"));
  destroyDetailChart();
  box.appendChild(el("p", { class: "muted" }, "กำลังโหลด…"));
  let a;
  try {
    a = await api(`/api/alerts/${encodeURIComponent(id)}`);
  } catch (e) {
    clear(box).appendChild(
      el("div", { class: "card" }, el("p", { class: "inline-error" }, `เปิดการแจ้งเตือนนี้ไม่ได้ (${e.message})`)),
    );
    return;
  }
  const tax = await loadTaxonomy();
  if (state.route !== "alert") return;
  clear(box);
  if (Array.isArray(a.feedback)) state.feedback.set(a.alert_id, a.feedback);

  // the head: what was seen, the level, the coin, the Thai time and the verdict so far
  const chipSlot = el("span", { class: "verdict-slot" }, verdictChip(latestVerdict(a.feedback)));
  box.appendChild(
    el(
      "div",
      { class: "detail-head" },
      el("h1", {}, patternText(a, tax)),
      el(
        "p",
        { class: "detail-meta" },
        el("span", { class: `sev sev-${a.severity}` }, `ระดับ${SEVERITY_TH[a.severity] || "–"}`),
        el("span", {}, a.symbol),
        el("span", { class: "muted" }, fmtThaiTime(a.ts_event_us, true)),
        chipSlot,
      ),
    ),
  );

  // the top: the plain explanation beside the price around the alert
  const facts = ((a.explanation && a.explanation.facts) || []).map(plainFact).filter(Boolean);
  const mini = el("div", { class: "chart-mini" });
  box.appendChild(
    el(
      "div",
      { class: "card detail-top" },
      el(
        "div",
        { class: "detail" },
        el(
          "div",
          { class: "detail-side" },
          el("p", { class: "explain" }, `ระบบพบรูปแบบการซื้อขายที่ต่างจากปกติของ ${a.symbol} เมื่อ ${fmtThaiTime(a.ts_event_us, true)}`),
          facts.length
            ? [el("h3", {}, "สิ่งที่ต่างจากปกติ"), el("ul", { class: "facts" }, facts.map((t) => el("li", {}, t)))]
            : null,
          el("p", { class: "small muted" }, "เป็นความผิดปกติทางสถิติเมื่อเทียบกับตลาดปกติ ให้คนตรวจต่อ ไม่ได้บอกว่าใครทำอะไร"),
        ),
        el("div", { class: "detail-main" }, el("h3", {}, "ราคาช่วงนั้น แถบสีคือช่วงที่ระบบเตือน"), mini),
      ),
    ),
  );

  box.appendChild(
    feedbackCard(a, (rows) => {
      state.feedback.set(a.alert_id, rows);
      clear(chipSlot).appendChild(verdictChip(latestVerdict(rows)) || document.createTextNode(""));
      renderAlertCards();
    }),
  );
  box.appendChild(techDetails(a, tax));
  drawDetailChart(mini, a);
}

// "ดูรายละเอียด": the full explanation, the numbers, the detectors with their layers and p-values,
// the evidence, the technique card and the provenance; closed by default
function techDetails(a, tax) {
  const explanation = (a.explanation && a.explanation.text_th) || a.explanation_th || "";
  const facts = (a.explanation && a.explanation.facts) || [];
  const F = nlog10(a.fused_p);
  const [w0, w1] = Array.isArray(a.window) ? a.window : [null, null];
  return el(
    "details",
    { class: "card tech" },
    el("summary", {}, "ดูรายละเอียด"),
    el(
      "div",
      { class: "tech-grid" },
      el(
        "section",
        { class: "tech-sec" },
        el("h3", {}, "คำอธิบายเต็ม"),
        el("p", { class: "small" }, explanation),
        facts.length ? el("ul", { class: "facts small" }, facts.map((f) => el("li", {}, `${f.label}: ${f.value}`))) : null,
        el("h3", {}, "สรุปตัวเลข"),
        kv([
          ["ค่า p รวม / ความหายากเทียบตลาดปกติ", `${fmtP(a.fused_p)} (F = −log10 p = ${fmtNum(F, 2)})`, "mono"],
          ["ชั้นที่เห็นพ้อง", `${a.agreement}/3 ชั้น มี p < ${AGREEMENT_P}`],
          ["เวลาเกิด", fmtUs(a.ts_event_us), "mono"],
          ["ช่วงเวลา", `${fmtUs(w0)} – ${fmtUs(w1)}`, "mono"],
          ["สถานะ", STATUS_TH[a.status] || a.status],
          ["เทคนิค", a.technique_id, "mono"],
          ["เทคนิคของจุดสูงสุด", a.peak_technique_id || "–", "mono"],
          ["ที่มาของข้อมูล", el("span", { class: "badge" }, a.provenance)],
          ["รหัส", a.alert_id, "mono"],
        ]),
      ),
      el(
        "section",
        { class: "tech-sec" },
        el("h3", {}, "ตัวตรวจจับและค่า p"),
        el(
          "div",
          { class: "table-wrap" },
          el(
            "table",
            { class: "table" },
            el(
              "thead",
              {},
              el("tr", {}, el("th", {}, "ตัวตรวจจับ"), el("th", {}, "ชั้น"), el("th", { class: "num" }, "ค่า p"), el("th", {}, "เห็นพ้อง")),
            ),
            el(
              "tbody",
              {},
              (a.detectors || []).map((d) =>
                el(
                  "tr",
                  {},
                  el("td", { class: "mono" }, d.name),
                  el("td", {}, d.layer),
                  el("td", { class: "num" }, fmtP(d.p)),
                  el("td", {}, FUSION_LAYERS.includes(d.layer) && d.p < AGREEMENT_P ? "✓" : ""),
                ),
              ),
            ),
          ),
        ),
        el("h3", {}, "หลักฐานรายคุณลักษณะ"),
        (a.evidence || []).length
          ? el(
              "div",
              { class: "table-wrap" },
              el(
                "table",
                { class: "table" },
                el("thead", {}, el("tr", {}, el("th", {}, "คุณลักษณะ"), el("th", { class: "num" }, "ค่า"), el("th", { class: "num" }, "น้ำหนัก"))),
                el(
                  "tbody",
                  {},
                  a.evidence.map(([name, value, contrib]) =>
                    el(
                      "tr",
                      {},
                      el("td", { class: "mono" }, name),
                      el("td", { class: "num" }, fmtNum(value, 4)),
                      el("td", { class: "num" }, fmtNum(contrib, 2)),
                    ),
                  ),
                ),
              ),
            )
          : el("p", { class: "muted small" }, "ไม่มีหลักฐานรายคุณลักษณะแนบมา"),
        techniqueCard(tax, a.technique_id),
      ),
    ),
  );
}

function techniqueCard(tax, techniqueId) {
  const card = el("div", { class: "tech-technique" }, el("h3", {}, "การ์ดเทคนิค"));
  const parent = parentTechnique(techniqueId);
  const t = tax && tax[parent];
  if (!t) {
    card.appendChild(el("p", { class: "muted small" }, `ไม่พบ ${techniqueId} ใน taxonomy`));
    return card;
  }
  const sub = (t.sub_techniques || []).find((s) => s.id === techniqueId);
  append(card, [
    el("p", {}, el("strong", { class: "mono" }, parent), ` ${t.name_th || ""}`),
    el("p", { class: "small" }, t.definition_th || ""),
    sub ? el("p", { class: "small" }, el("strong", { class: "mono" }, sub.id), ` ${sub.name_th || ""}: ${sub.definition_th || ""}`) : null,
    el("p", { class: "small muted" }, t.observability_th || ""),
    el("p", { class: "small" }, el("button", { type: "button", class: "link-btn", onclick: () => openHelp("help-patterns") }, "ดูรูปแบบทั้งหมดที่ระบบมองหา")),
  ]);
  return card;
}

// the verdict: three large buttons (analyst and up), the status in this card, and who said what
function feedbackCard(a, onSaved) {
  const card = el(
    "div",
    { class: "card verdict-card" },
    el("h2", { class: "verdict-title" }, "ผลตรวจของคุณ"),
    el("p", { class: "small muted" }, "ดูกราฟและคำอธิบายแล้ว รายการนี้ผิดปกติจริงหรือไม่"),
  );
  const analyst = el("input", {
    type: "text",
    maxlength: "64",
    placeholder: "ชื่อผู้วิเคราะห์",
    value: storageGet("idsai.analyst") || "",
    "aria-label": "ชื่อผู้วิเคราะห์",
  });
  const status = el("p", { class: "small verdict-status", role: "status" });
  const list = el("ul", { class: "verdict-history small" });
  const buttons = {};
  const renderList = (items) => {
    clear(list);
    const rows = items || [];
    for (const f of [...rows].reverse()) {
      list.appendChild(el("li", {}, `${fmtThaiTime(f.ts_us)} · ${f.analyst}: ${VERDICT_TH[f.verdict] || f.verdict}`));
    }
    if (!rows.length) list.appendChild(el("li", { class: "muted" }, "ยังไม่มีใครตรวจรายการนี้"));
    const latest = latestVerdict(rows);
    for (const v of VERDICTS) buttons[v].setAttribute("aria-pressed", latest === v ? "true" : "false");
  };
  let saving = false;
  const setBusy = (on) => {
    saving = on;
    for (const v of VERDICTS) buttons[v].disabled = on; // one save at a time: no double post
  };
  const send = async (verdict) => {
    if (saving) return;
    if (!can("analyst")) {
      status.textContent = VIEWER_VERDICT_TH;
      return;
    }
    // signed in: the server records the session user, so no name is typed or remembered here
    const name = authOn() ? state.me.username : analyst.value.trim();
    if (!name) {
      status.textContent = "ใส่ชื่อผู้วิเคราะห์ก่อน";
      return;
    }
    if (!authOn()) storageSet("idsai.analyst", name);
    status.classList.remove("inline-error");
    status.textContent = "กำลังบันทึก…";
    setBusy(true);
    try {
      await postJson(`/api/alerts/${encodeURIComponent(a.alert_id)}/feedback`, { verdict, analyst: name });
    } catch (e) {
      status.classList.add("inline-error");
      console.warn("feedback:", e && e.message);
      status.textContent = `บันทึกไม่สำเร็จ (${plainErrorText(e)})`;
      setBusy(false);
      return;
    }
    // saved: the follow-up read only refreshes the history; its failure is not a failed save
    status.textContent = `บันทึกแล้ว: ${VERDICT_TH[verdict]}`;
    let rows = [...(state.feedback.get(a.alert_id) || a.feedback || []), { alert_id: a.alert_id, verdict, analyst: name, ts_us: Date.now() * 1000 }];
    try {
      const fresh = await api(`/api/alerts/${encodeURIComponent(a.alert_id)}`);
      if (Array.isArray(fresh.feedback)) rows = fresh.feedback;
    } catch (e) {
      status.textContent = `บันทึกแล้ว: ${VERDICT_TH[verdict]} (อ่านประวัติล่าสุดไม่ได้)`;
    }
    renderList(rows);
    if (onSaved) onSaved(rows);
    setBusy(false);
  };
  const allowed = can("analyst");
  const gate = allowed ? {} : { disabled: true, title: VIEWER_VERDICT_TH };
  if (!allowed) {
    analyst.disabled = true;
    analyst.title = VIEWER_VERDICT_TH;
  }
  for (const v of VERDICTS) {
    buttons[v] = el("button", { type: "button", class: `verdict-btn v-${v}`, ...gate, onclick: () => send(v) }, VERDICT_TH[v]);
  }
  append(card, [
    authOn() ? null : el("label", { class: "field verdict-name" }, el("span", {}, "ผู้วิเคราะห์"), analyst),
    el("div", { class: "verdict-buttons" }, VERDICTS.map((v) => buttons[v])),
    allowed ? null : el("p", { class: "small muted" }, VIEWER_VERDICT_TH),
    status,
    el("h3", { class: "verdict-history-title" }, "ประวัติการตรวจ"),
    list,
  ]);
  renderList(a.feedback);
  return card;
}

async function drawDetailChart(node, a) {
  if (!LWC) return;
  const [w0, w1] = a.window;
  const pad = 30 * 60 * S_US;
  let s;
  try {
    const q = `symbol=${encodeURIComponent(a.symbol)}&start_us=${w0 - pad}&end_us=${w1 + pad}`;
    s = await api(`/api/series?${q}`);
  } catch (e) {
    s = null;
  }
  if (state.route !== "alert" || !node.isConnected) return;
  const times = ((s && s.t_end_us) || []).map((t) => (isNum(t) ? toSec(t) : null));
  const price = seriesData(times, (s && s.close) || []);
  if (!price.length) {
    node.replaceWith(
      el(
        "p",
        { class: "muted small chart-none" },
        s ? "ไม่มีข้อมูลราคาของช่วงนี้ในหน่วยความจำ (ระบบเก็บย้อนหลัง 24 ชั่วโมงของการเล่นครั้งนี้)" : "อ่านข้อมูลราคาไม่ได้ ลองเปิดหน้านี้ใหม่อีกครั้ง",
      ),
    );
    return;
  }
  const c = LWC.createChart(node, baseChartOptions());
  const line = c.addSeries(
    LWC.LineSeries,
    lineOptions(cssVar("--line-price"), "ราคา", { priceFormat: { type: "price", precision: 8, minMove: 1e-8 } }),
  );
  const shade = c.addSeries(LWC.HistogramSeries, {
    priceScaleId: "shade",
    color: cssVar("--shade"),
    priceLineVisible: false,
    lastValueVisible: false,
    base: 0,
  });
  c.priceScale("shade").applyOptions({ scaleMargins: { top: 0, bottom: 0 }, visible: false });
  const s0 = toSec(w0);
  const s1 = toSec(w1);
  const shadeData = () =>
    price.map((p) => (p.time >= s0 && p.time <= s1 ? { time: p.time, value: 1, color: cssVar("--shade") } : { time: p.time }));
  line.setData(price);
  shade.setData(shadeData());
  const t = price.reduce((best, p) => (Math.abs(p.time - toSec(a.ts_event_us)) < Math.abs(best - toSec(a.ts_event_us)) ? p.time : best), price[0].time);
  // colour and shape only: the technique id stays in "ดูรายละเอียด"
  LWC.createSeriesMarkers(line, [{ time: t, position: "aboveBar", shape: "arrowDown", color: sevColor(a.severity) }]);
  c.timeScale().fitContent();
  state.detailChart = {
    remove: () => c.remove(),
  };
}

// ------------------------------------------------------------------ red team

function simTitle(family) {
  const card = SIM_CARDS.find((c) => c.family === family);
  return card ? card.title : "การปั่นจำลอง";
}

// the POST body for one card click: the defaults, with any filled-in advanced field on top
function simulateRequest(family, adv, currentSymbol) {
  const req = { family, region: SIM_DEFAULTS.region, offset_s: SIM_DEFAULTS.offset_s, symbol: currentSymbol || null };
  for (const k of ["region", "offset_s", "symbol"]) {
    const v = adv ? adv[k] : undefined;
    if (v !== undefined && v !== null && v !== "") req[k] = v;
  }
  return req;
}

// a refused inject, in plain Thai: the server's own text can carry field names in English
function simErrorText(err) {
  const status = err && err.status;
  const msg = String((err && err.message) || "");
  if (status === 409 && msg === REPLAY_ENDED_TH) return SIM_REASON.ended;
  if (status === 403) return msg.includes("สิทธิ์") ? SIM_REASON.viewer : SIM_REASON.live;
  if (status === 422 && /offset|ข้อมูลสุดท้าย/.test(msg)) return SIM_PAST_END_TH;
  if (status === 422 && /symbol/.test(msg)) return SIM_SYMBOL_TH;
  return SIM_FAILED_TH;
}

// any other refused or failed request, in plain Thai: the server's own text can be English or carry field names
const NET_FAILED_TH = "ติดต่อเซิร์ฟเวอร์ไม่ได้";
const ACT_FAILED_TH = "ทำรายการไม่สำเร็จ ลองใหม่อีกครั้ง";
const ACT_FORBIDDEN_TH = "สิทธิ์ไม่พอสำหรับการทำรายการนี้";
const ACT_CONFLICT_TH = "ตอนนี้ทำรายการนี้ไม่ได้ เพราะสถานะเปลี่ยนไปแล้ว";
const ACT_INVALID_TH = "ข้อมูลที่ส่งไม่ถูกต้อง";
function plainErrorText(err) {
  const status = err && err.status;
  if (!isNum(status)) return NET_FAILED_TH; // fetch itself failed: no HTTP answer at all
  if (status === 403) return ACT_FORBIDDEN_TH;
  if (status === 409) return ACT_CONFLICT_TH;
  if (status === 422) return ACT_INVALID_TH;
  return ACT_FAILED_TH; // 401 is handled by sessionLost; 5xx and the rest
}

// the advanced fields; null when the start offset is out of range
function readAdvanced() {
  const raw = String($("rt-offset").value ?? "").trim();
  let offset;
  if (raw !== "") {
    const n = Number(raw);
    if (!Number.isFinite(n) || n <= 0 || n > 3600) return null;
    offset = Math.round(n);
  }
  return { region: $("rt-region").value, offset_s: offset, symbol: $("rt-symbol").value };
}

function bindSimulate() {
  for (const c of SIM_CARDS) {
    const card = $(`sim-${c.family}`);
    card.addEventListener("click", () => startSimulation(c.family, card));
  }
}

async function startSimulation(family, card) {
  if (!redTeamAllowed(state.mode)) return;
  const status = $("rt-status");
  if (!can("analyst")) {
    status.textContent = SIM_REASON.viewer;
    return;
  }
  if (state.simBusy || state.simEnded) return;
  const adv = readAdvanced();
  if (adv === null) {
    status.textContent = SIM_OFFSET_TH;
    $("sim-advanced").open = true;
    return;
  }
  const body = simulateRequest(family, adv, state.symbol);
  const title = simTitle(family);
  status.textContent = `กำลังเริ่ม ${title}`;
  state.simBusy = true;
  card.setAttribute("aria-busy", "true");
  applySimGates();
  const known = snapshotAlerts(state.alerts.values()); // before the click: never a detection
  try {
    const r = await postJson("/api/redteam/inject", body);
    status.textContent = `เริ่ม ${title} แล้ว ดูผลด้านล่าง`;
    addInjection(r, body, known);
  } catch (e) {
    if (e.status === 409 && e.message === REPLAY_ENDED_TH) {
      state.simEnded = true; // said once, on the reason line
      status.textContent = "";
    } else {
      const text = simErrorText(e);
      if (text === SIM_FAILED_TH) console.warn("simulate:", e && e.message);
      status.textContent = text;
    }
  } finally {
    state.simBusy = false;
    card.removeAttribute("aria-busy");
    applySimGates();
  }
}

function addInjection(r, body, known) {
  const label = r.label || {};
  const run = {
    // the server starts the scenario at its cursor + offset_s, so this is the cursor at the click
    cursor_us: isNum(label.t_start_us) ? label.t_start_us - body.offset_s * S_US : null,
    known,
    family: body.family,
    region: body.region,
    symbol: label.symbol || body.symbol,
    scenario: r.scenario || {},
    t_start_us: label.t_start_us,
    t_end_us: label.t_end_us,
    wallStart: Date.now(),
    detected: null,
    card: null,
  };
  state.injections.unshift(run);
  renderInjections();
  checkInjections();
}

// ---- time-to-detect (pure: exercised by tests/product/test_web_static.py through node) ----

function snapshotAlerts(alerts) {
  // alert_id -> window end, of every alert on screen when Red-Team was clicked
  const known = new Map();
  for (const a of alerts) known.set(a.alert_id, a.window[1]);
  return known;
}

function matchesInjection(a, run) {
  if (a.symbol !== run.symbol || !isNum(run.t_start_us) || !isNum(run.t_end_us)) return false;
  const [w0, w1] = a.window;
  return w1 >= run.t_start_us - MATCH_BEFORE_US && w0 <= run.t_end_us + MATCH_AFTER_US;
}

function eligibleForRun(a, run) {
  // new since the click, or one that existed and has grown past both its old end and the cursor
  const known = run.known || new Map();
  if (!known.has(a.alert_id)) return true;
  const floor = Math.max(known.get(a.alert_id), isNum(run.cursor_us) ? run.cursor_us : -Infinity);
  return a.window[1] > floor;
}

function firstDetection(run, alerts) {
  let best = null;
  for (const a of alerts) {
    if (!eligibleForRun(a, run) || !matchesInjection(a, run)) continue;
    if (best === null || a.window[0] < best.window[0]) best = a;
  }
  return best;
}

function resetRuns(runs, deleted) {
  // after a seek: a deleted alert no longer counts (the timer re-arms, the injected trades stay)
  // and a deleted id that comes back is a new alert
  const gone = new Set(deleted);
  let changed = false;
  for (const run of runs) {
    if (run.closed) continue; // a run of an earlier replay round keeps its verdict
    for (const id of gone) if (run.known) run.known.delete(id);
    if (run.detected && gone.has(run.detected.alert_id)) {
      run.detected = null;
      changed = true;
    }
  }
  return changed;
}

function restartRuns(runs) {
  // the replay day started again from the original trades: every run so far is over; one not
  // detected yet never will be (its injected trades are gone) and counts as missed
  for (const run of runs) {
    run.closed = true;
    if (!run.detected) run.restarted = true;
  }
}

function injectionStatus(run, nowUs) {
  // "detected" wins whenever it comes; "missed" once the eval matching window (scenario end +
  // 300 s) plus a grace has passed without a qualifying alert; it follows the clock, so a seek
  // back makes the run "waiting" again
  if (run.detected) return "detected";
  if (run.restarted) return "missed";
  if (!isNum(nowUs) || !isNum(run.t_start_us)) return "waiting";
  if (isNum(run.t_end_us) && nowUs > run.t_end_us + MATCH_AFTER_US + MISSED_GRACE_US) return "missed";
  return nowUs < run.t_start_us ? "pending" : "waiting";
}

function checkInjections() {
  let changed = false;
  for (const run of state.injections) {
    if (run.detected || run.closed) continue; // a closed run's trades are gone
    const a = firstDetection(run, state.alerts.values());
    if (a) {
      run.detected = {
        alert_id: a.alert_id,
        eventS: (a.window[0] - run.t_start_us) / S_US,
        wallS: (Date.now() - run.wallStart) / 1000,
      };
      changed = true;
    }
  }
  if (changed) renderInjections();
}

function renderInjections() {
  const box = clear($("rt-runs"));
  $("rt-empty").hidden = state.injections.length > 0;
  for (const run of state.injections) box.appendChild(runCard(run));
  tickInjections();
}

// one run: started -> waiting -> result, then the technical facts folded away
function runCard(run) {
  const sc = run.scenario || {};
  const step = (cls, label, ...mark) => {
    const text = el("span", { class: "step-label" }, label);
    const sub = el("span", { class: "step-sub small" });
    const node = el(
      "li",
      { class: `step ${cls}` },
      el("span", { class: "step-dot", "aria-hidden": "true" }, ...mark),
      el("span", { class: "step-text" }, text, sub),
    );
    return { node, text, sub };
  };
  const steps = [
    step("step-start", simStatusText("pending", run)),
    step("step-wait", simStatusText("waiting", run)),
    step(
      "step-result",
      "ผลลัพธ์",
      svg(
        "svg",
        { viewBox: "0 0 12 12", width: 10, height: 10, fill: "none", stroke: "currentColor", "stroke-width": 1.5, "stroke-linecap": "round", "stroke-linejoin": "round", class: "step-check" },
        svg("path", { d: "M2.5 6.2l2.3 2.3 4.7-5" }),
      ),
    ),
  ];
  const list = el("ol", { class: "sim-steps" }, ...steps.map((st) => st.node));
  const note = el("p", { class: "sim-run-note small" });
  const tech = el(
    "details",
    { class: "sim-tech small" },
    el("summary", {}, "ดูรายละเอียด"),
    el(
      "dl",
      { class: "kv small" },
      el("dt", {}, "ช่วงเหตุการณ์ (UTC)"),
      el("dd", { class: "mono" }, `${fmtUs(run.t_start_us)} – ${fmtUs(run.t_end_us)}`),
      el("dt", {}, "เทคนิค"),
      el("dd", { class: "mono" }, `${run.family} · ${sc.technique_id || "–"} / ${sc.sub_technique || "–"}`),
      el("dt", {}, "พารามิเตอร์"),
      el(
        "dd",
        { class: "mono" },
        `region ${run.region} · ${fmtNum(sc.duration_s, 0)} s · impact ${fmtNum(sc.impact, 3)} · vol ×${fmtNum(sc.vol_mult, 1)}`,
      ),
      el("dt", {}, "ที่มาข้อมูล"),
      el("dd", {}, el("span", { class: "badge SEMI-SYN" }, "SEMI-SYN")),
    ),
  );
  const card = el(
    "div",
    { class: "card sim-run" },
    el(
      "div",
      { class: "sim-run-head" },
      el("h3", {}, `${simTitle(run.family)} · ${run.symbol || "–"}`),
      el("span", { class: "small muted" }, `เหตุการณ์เริ่ม ${fmtThaiTime(run.t_start_us)}`),
    ),
    list,
    note,
    tech,
  );
  run.card = card;
  run.steps = steps;
  run.note = note;
  run.shown = null; // the status drawn; data-state changes only when it does
  return card;
}

// one line per run state: the step labels and the verdict
function simStatusText(status, run) {
  if (status === "restarted" || (status === "missed" && run && run.restarted)) return "พลาด (replay เริ่มใหม่)";
  if (status === "pending") return "เริ่มแล้ว";
  if (status === "waiting") return "กำลังรอระบบ";
  if (status === "missed") return "ไม่ทันจับ";
  if (status === "detected" && run && run.detected) {
    const ev = Math.max(0, Math.round(Number(run.detected.eventS) || 0));
    const wall = Math.max(0, Math.round(Number(run.detected.wallS) || 0));
    return `จับได้ใน ${ev} วินาที (เวลาจริง ${wall} วินาที)`;
  }
  return "";
}

function setText(node, value) {
  if (node.textContent !== value) node.textContent = value;
}

function tickInjections() {
  const now = eventNowUs();
  for (const run of state.injections) {
    if (!run.card) continue;
    const status = injectionStatus(run, now);
    const verdict = status === "detected" || status === "missed";
    if (run.shown !== status) {
      run.shown = status;
      run.card.setAttribute("data-state", status);
      setText(run.steps[2].text, verdict ? simStatusText(status, run) : "ผลลัพธ์");
      clear(run.note);
      if (status === "detected") {
        run.note.appendChild(el("a", { href: alertHref(run.detected.alert_id) }, "ดูการแจ้งเตือนที่จับได้"));
      }
      run.note.hidden = status !== "detected";
    }
    // the small lines under the steps follow the clock
    const toStart = isNum(now) && isNum(run.t_start_us) ? Math.ceil((run.t_start_us - now) / S_US) : null;
    setText(run.steps[0].sub, status === "pending" && toStart !== null ? `เหตุการณ์จะเริ่มในอีก ${toStart} วินาที` : "");
    const waited = Math.max(0, Math.round((Date.now() - run.wallStart) / 1000));
    setText(run.steps[1].sub, status === "pending" || status === "waiting" ? `รอมาแล้ว ${waited} วินาที` : "");
    let why = "";
    if (status === "missed") {
      why = run.restarted
        ? "ข้อมูลจำลองหายไปเมื่อการย้อนดูเริ่มรอบใหม่"
        : "ไม่มีการแจ้งเตือนภายใน 5 นาทีหลังเหตุการณ์จบ";
    }
    setText(run.steps[2].sub, why);
  }
}

// ------------------------------------------------------------------ results (eval)

function evalEmpty(report) {
  return !report || !report.runs || !report.experiments;
}

// pass / miss / unmeasured: absent data is never a pass. `budget` (alerts per symbol-day) picks the
// report cell; it defaults to the target's own, and a false-alarm cap of "≤ budget" follows it.
function targetVerdict(report, target, budget) {
  const none = { state: "unmeasured", value: null, ci: null, n: null };
  if (evalEmpty(report) || target.cmp === "none") return none; // the report has no latency metric
  const b = budget === null || budget === undefined || !Number.isFinite(Number(budget)) ? target.budget : Number(budget);
  if (target.cmp === "pair") {
    const paired = report.paired_bootstrap;
    if (!paired || Number(paired.budget) !== b) return none;
    const found = (paired.rows || []).find((r) => r.baseline === target.variant);
    if (!found || !isNum(found.diff_mean) || !Array.isArray(found.ci) || !isNum(found.ci[0])) return none;
    // the report's own gate (CI, minimum effect, minimum events) decides; older reports fall back to the CI
    const gated = typeof found.significant === "boolean" ? found.significant : found.ci[0] > target.threshold;
    const st = target.reference ? "reference" : gated ? "pass" : "miss";
    return { state: st, value: found.diff_mean, ci: found.ci, n: found.n_events, significant: gated, reason: found.reason_th || null };
  }
  const variants = report.experiments[target.experiment] || {};
  const cell = (variants[target.variant] || {})[String(b)];
  const value = cell && cell.mean ? cell.mean[target.metric] : null;
  if (!isNum(value)) return none;
  const ci = cell.ci && Array.isArray(cell.ci[target.metric]) ? cell.ci[target.metric] : null;
  const cap = target.capIsBudget ? b : target.threshold;
  const ok = target.cmp === ">=" ? value >= cap : value <= cap;
  const out = { state: ok ? "pass" : "miss", value, ci, n: cell.n_events, unit: cell.ci_unit || null, seeds: cell.n_seeds };
  if (target.correct && isNum(cell.mean.recall_correct)) {
    const cc = cell.ci && cell.ci.recall_correct;
    out.correct = { value: cell.mean.recall_correct, ci: Array.isArray(cc) ? cc : null };
  }
  return out;
}

const TARGET_STATE_TH = { pass: "ผ่านเป้า", miss: "ไม่ถึงเป้า", unmeasured: "ยังไม่ได้วัด", reference: "อ้างอิง" };
// the plain badge: a glyph and a word, never colour alone
const RESULT_BADGE = { pass: "ผ่านเป้า", fail: "ยังไม่ถึงเป้า", none: "ยังไม่วัด" };
const NO_TARGET_TH = "ไม่ได้ตั้งเป้าที่งบนี้"; // the badge off the project budget: numbers only, no verdict
const RESULT_GLYPH = { pass: "✓", fail: "✗", none: "—" };
const RESULT_STATE = { pass: "pass", miss: "fail", unmeasured: "none" }; // a reference card has no badge
const PER_DAY_TH = "ครั้งต่อเหรียญต่อวัน";
const MAIN_BUDGET = "2"; // the budget the project's targets are set at

// 1.6498 -> "1.65", 11.17 -> "11.17", 2 -> "2"
function fmtPlain(x) {
  return String(Number(x.toFixed(2)));
}

// one plain Thai sentence per card (spec 4.8); `extra` = { budget, k, n, ci }
function resultSentence(target, value, extra) {
  const x = extra || {};
  const b = isNum(x.budget) ? x.budget : target.budget;
  if (target.cmp === "none") return "ความเร็วในการเตือน: ยังไม่วัด";
  if (target.cmp === "pair") {
    const ci = x.ci;
    if (!Array.isArray(ci) || !isNum(ci[0]) || !isNum(ci[1])) return "เทียบกับวิธีพื้นฐาน: ยังไม่วัดที่งบเตือนนี้";
    if (x.reason) return `เทียบกับวิธีพื้นฐาน: ${x.reason}`;
    // the report's own gate says "better"; otherwise the 95% CI of the difference: wholly below 0 is worse
    const better = typeof x.significant === "boolean" ? x.significant : ci[0] > 0;
    return `เทียบกับวิธีพื้นฐาน: ${better ? "ระบบนี้ดีกว่าวิธีนี้" : ci[1] < 0 ? "ระบบนี้แย่กว่าวิธีนี้" : "ไม่ต่างกันอย่างมีนัยสำคัญ"}`;
  }
  if (target.metric === "recall_any") {
    const at = `(เมื่อเตือนไม่เกิน ${fmtPlain(b)} ${PER_DAY_TH})`;
    const what = target.experiment === "semisyn" ? "จับเหตุการณ์จำลองได้" : "จับเหตุการณ์ที่มีบันทึกได้";
    if (!isNum(value)) return `${what}: ยังไม่มีผล ${at}`;
    // a count is a mean over the test rounds
    const got = isNum(x.k) && isNum(x.n) ? `เฉลี่ย ${x.k} จาก ${x.n} ครั้ง` : ` ${Math.round(value * 100)}%`;
    return `${what}${got} ${at}`;
  }
  if (target.metric === "fa_per_symbol_day") {
    return isNum(value) ? `เตือนผิดเฉลี่ย ${fmtPlain(value)} ${PER_DAY_TH}` : "เตือนผิดเฉลี่ย: ยังไม่มีผล";
  }
  if (target.metric === "fa_per_symbol_day_hard") {
    return isNum(value) ? `เตือนผิดในวันที่ไม่มีเหตุการณ์ ${fmtPlain(value)} ครั้ง` : "เตือนผิดในวันที่ไม่มีเหตุการณ์: ยังไม่มีผล";
  }
  return "ยังไม่มีผล";
}

function resultGoal(target, budget) {
  if (target.reference) return "ใช้เป็นจุดเทียบ ไม่มีเป้า";
  if (target.cmp === "none") return `เป้า: ไม่เกิน ${target.threshold} วินาที`;
  if (target.cmp === "pair") return "เป้า: ดีกว่าอย่างมีนัยสำคัญ";
  if (target.metric === "recall_any") return `เป้า: อย่างน้อย ${Math.round(target.threshold * 100)}%`;
  if (target.capIsBudget) return `เป้า: ไม่เกิน ${fmtPlain(budget)} ${PER_DAY_TH}`;
  return `เป้า: ${fmtPlain(target.threshold)} ครั้ง`;
}

function resultBadge(st) {
  return el(
    "span",
    { class: "result-badge", "data-state": st },
    el("span", { "aria-hidden": "true" }, RESULT_GLYPH[st]),
    ` ${RESULT_BADGE[st]}`,
  );
}

// the last big number shown per card, so a budget change counts from it
const resultShown = {};

function resultCard(target, verdict, budget) {
  const wide = target.id === "real_recall";
  let st = RESULT_STATE[verdict.state] || null;
  // pass / fail is judged at the project budget only; elsewhere the numbers stand without a verdict
  const judged = String(budget) === MAIN_BUDGET;
  if (!judged && (st === "pass" || st === "fail")) st = "neutral";
  const extra = { budget, ci: verdict.ci, significant: verdict.significant, reason: verdict.reason };
  // a count of events only where the unit is the event (a symbol-day ratio is no count)
  if (verdict.unit === "event" && isNum(verdict.value) && isNum(verdict.n) && verdict.n > 0) {
    extra.k = Math.round(verdict.value * verdict.n);
    extra.n = verdict.n;
  }
  const card = el(
    "div",
    {
      class: `card result-card${wide ? " result-wide" : ""}${target.reference ? " result-ref-card" : ""}`,
      "data-target": target.id,
      "data-state": st || "reference",
    },
    el(
      "div",
      { class: "result-head" },
      el("h3", { class: "result-title" }, target.title),
      st === "neutral"
        ? el("span", { class: "result-badge", "data-state": "neutral" }, NO_TARGET_TH)
        : st
          ? resultBadge(st)
          : el("span", { class: "result-ref" }, "วิธีเปรียบเทียบ"),
    ),
  );
  if (wide && isNum(verdict.value)) {
    const pct = Math.round(verdict.value * 100);
    const num = el("span", {}, String(isNum(resultShown[target.id]) ? resultShown[target.id] : pct));
    card.appendChild(el("div", { class: "result-value" }, num, el("span", { class: "result-value-unit" }, "%")));
    tweenNumber(num, pct, 0);
    resultShown[target.id] = pct;
    card.appendChild(
      el(
        "div",
        { class: "result-bar", "aria-hidden": "true" },
        el("span", { class: "result-fill", style: `width: ${Math.max(0, Math.min(100, pct))}%` }),
        st === "neutral" ? null : el("span", { class: "result-mark", style: `left: ${Math.round(target.threshold * 100)}%` }),
      ),
    );
  }
  card.appendChild(el("p", { class: "result-sentence" }, resultSentence(target, verdict.value, extra)));
  const seeds = wide && isNum(verdict.seeds) && verdict.seeds > 1 ? `ค่าเฉลี่ยจากการทดสอบ ${verdict.seeds} รอบ` : "";
  const goal = st === "neutral" ? "" : resultGoal(target, budget);
  const goalLine = [goal, seeds].filter(Boolean).join(" · ");
  if (goalLine) card.appendChild(el("p", { class: "result-goal small muted" }, goalLine));
  return card;
}

const RESULT_GROUPS = [
  {
    title: "ข้อมูลการซื้อขายจริงย้อนหลัง",
    lead: "เทียบกับเหตุการณ์ที่มีบันทึกไว้แล้ว",
    pick: (tg) => tg.experiment === "real_t001" || tg.cmp === "none",
  },
  {
    title: "เหตุการณ์ปั่นจำลองที่ใส่ลงในข้อมูลจริง",
    lead: "ใส่เหตุการณ์ที่รู้คำตอบอยู่แล้วลงไป แล้วดูว่าระบบจับได้เท่าไร",
    pick: (tg) => tg.experiment === "semisyn",
  },
  {
    title: "เทียบกับวิธีอื่น",
    lead: "ตัวตรวจที่เรียนรู้รูปแบบการซื้อขายปกติ เทียบกับวิธีอื่นที่งบเตือนเดียวกัน",
    pick: (tg) => tg.cmp === "pair",
  },
];

function renderResultCards(box, report, budget) {
  clear(box);
  for (const g of RESULT_GROUPS) {
    box.appendChild(
      el(
        "section",
        { class: "result-group" },
        el("h2", {}, g.title),
        el("p", { class: "small muted result-lead" }, g.lead),
        el(
          "div",
          { class: "result-cards" },
          EVAL_TARGETS.filter(g.pick).map((t) => resultCard(t, targetVerdict(report, t, budget), budget)),
        ),
      ),
    );
  }
}

function committeeBox(...children) {
  return el(
    "details",
    { class: "card committee" },
    el("summary", {}, "รายละเอียดสำหรับกรรมการ"),
    el("div", { class: "committee-body" }, children),
  );
}

// the committee's view of the targets: ids, value, CI, n and the goal as the MVP spec states it
function targetRow(target, verdict) {
  const paired = target.cmp === "pair";
  const goal =
    target.cmp === "none"
      ? `< ${target.threshold} s`
      : target.reference
        ? "ไม่มีเป้า (อ้างอิง)"
        : paired
          ? "CI ขอบล่าง > 0"
          : `${target.cmp === ">=" ? "≥" : "≤"} ${target.capIsBudget ? "งบ" : target.threshold}`;
  const none = verdict.state === "unmeasured";
  return el(
    "tr",
    {},
    el("td", { class: "mono" }, target.label),
    el("td", {}, TARGET_STATE_TH[verdict.state]),
    el("td", { class: "num mono" }, none ? "–" : paired ? `${fmtSigned(verdict.value)} [${fmtSigned(verdict.ci[0])}, ${fmtSigned(verdict.ci[1])}]` : ciText(verdict.value, verdict.ci)),
    el("td", { class: "num" }, verdict.n ?? "–"),
    el("td", { class: "mono" }, goal),
    el("td", { class: "small mono muted" }, verdict.correct ? `recall_correct ${ciText(verdict.correct.value, verdict.correct.ci)}` : ""),
  );
}

function targetTable(report, budget) {
  return el(
    "div",
    { class: "committee-sec" },
    el("h3", {}, `เป้าหมายตามแผน (งบ ${budget} ครั้ง/สัญลักษณ์-วัน)`),
    el(
      "div",
      { class: "table-wrap" },
      el(
        "table",
        { class: "table" },
        el("thead", {}, el("tr", {}, ["เป้า", "ผล", "ค่า [95% CI]", "n", "เกณฑ์", "หมายเหตุ"].map((h) => el("th", {}, h)))),
        el("tbody", {}, EVAL_TARGETS.map((t) => targetRow(t, targetVerdict(report, t, budget)))),
      ),
    ),
  );
}

function fmtSigned(x) {
  return isNum(x) ? `${x >= 0 ? "+" : ""}${x.toFixed(3)}` : "–";
}

async function showEval() {
  const box = clear($("eval-body"));
  box.appendChild(el("p", { class: "muted" }, "กำลังโหลด…"));
  let report;
  try {
    report = await api("/api/experiments");
  } catch (e) {
    console.warn("experiments:", e && e.message);
    clear(box).appendChild(el("p", { class: "empty" }, `อ่านผลวัดระบบไม่ได้ (${plainErrorText(e)})`));
    return;
  }
  clear(box);
  if (evalEmpty(report)) {
    box.appendChild(
      el("div", { class: "card empty-state" }, el("p", { class: "empty-text" }, "ยังไม่มีผลวัดระบบ เมื่อสร้างรายงานผลแล้วให้เปิดหน้านี้ใหม่")),
    );
    box.appendChild(committeeBox(el("p", { class: "small" }, "ยังไม่มีรายงานผลการประเมิน · รัน uv run idsai report แล้วเปิดหน้านี้ใหม่")));
    return;
  }

  const found = new Set();
  for (const variants of Object.values(report.experiments)) {
    for (const cells of Object.values(variants)) Object.keys(cells).forEach((b) => found.add(b));
  }
  const budgets = [...found].sort((a, b) => Number(a) - Number(b));
  const saved = state.evalBudget;
  let budget = budgets.includes(saved) ? saved : budgets.includes(MAIN_BUDGET) ? MAIN_BUDGET : budgets[0] || MAIN_BUDGET;

  // the chips: flat pills, one pressed; the plain cards and the committee tables follow them
  const chips = el("div", { class: "budget-chips", role: "group", "aria-label": "งบเตือน ครั้งต่อวัน" });
  const note = el("p", { class: "small muted budget-note" });
  const plain = el("div", { class: "results-plain" });
  const targets = el("div"); // the committee's targets: always at the project budget, not the chip
  targets.appendChild(targetTable(report, Number(MAIN_BUDGET)));
  const tables = el("div");
  const draw = () => {
    for (const c of chips.children) c.setAttribute("aria-pressed", String(c.getAttribute("data-budget") === budget));
    note.textContent = budget === MAIN_BUDGET ? "" : `เป้าหมายของโครงการตั้งไว้ที่งบเตือน ${MAIN_BUDGET} ครั้งต่อวัน`;
    note.hidden = budget === MAIN_BUDGET;
    renderResultCards(plain, report, Number(budget));
    clear(tables);
    for (const [exp, variants] of Object.entries(report.experiments)) tables.appendChild(experimentTable(exp, variants, budget));
  };
  for (const b of budgets) {
    chips.appendChild(
      el(
        "button",
        {
          type: "button",
          class: "budget-chip",
          "data-budget": b,
          "aria-pressed": "false",
          onclick: () => {
            if (budget === b) return;
            budget = b;
            state.evalBudget = b;
            savePrefs({ eval_budget: Number(b) });
            draw();
          },
        },
        b,
      ),
    );
  }
  box.appendChild(
    el(
      "div",
      { class: "budget-row" },
      el("span", { class: "budget-label" }, "งบเตือน"),
      chips,
      el("span", { class: "budget-unit" }, "ครั้งต่อวัน"),
    ),
  );
  box.appendChild(note);
  box.appendChild(plain);

  const notes = [...(report.notes || []), ...((report.paired_bootstrap || {}).notes || [])];
  const retrain = report.retrain_decision;
  box.appendChild(
    committeeBox(
      el(
        "p",
        { class: "small muted" },
        `ทั้งหมด ${report.runs} run · git ${(report.git_sha || []).join(", ") || "–"} · config ${(report.config_hash || []).join(", ") || "–"} · signatures ${(report.signatures_sha || []).join(", ") || "–"}`,
      ),
      el(
        "p",
        { class: "small muted" },
        "ตัวเลขหลักคือ recall ต่อเหตุการณ์ (พร้อม 95% CI) และจำนวนการเตือนผิดต่อสัญลักษณ์-วัน precision แสดงคู่กับ prevalence เสมอ ทุกแถวมีป้ายที่มา",
      ),
      targets,
      report.paired_bootstrap ? pairedCard(report.paired_bootstrap) : null,
      retrain ? retrainCard("Champion / challenger (full stack)", retrain) : null,
      retrain && retrain.ai_only ? retrainCard("AE alone (diagnostic)", retrain.ai_only) : null,
      tables,
      notes.length
        ? el("details", { class: "notes-box" }, el("summary", {}, "หมายเหตุวิธีวัด"), el("ul", { class: "small" }, notes.map((n) => el("li", {}, n))))
        : null,
    ),
  );
  draw();
}

function ciText(mean, ci) {
  if (!isNum(mean)) return "–";
  const base = mean.toFixed(3);
  return Array.isArray(ci) && isNum(ci[0]) && isNum(ci[1]) ? `${base} [${ci[0].toFixed(3)}, ${ci[1].toFixed(3)}]` : base;
}

function ruleOfThreeText(metric, r3) {
  // recall 0 or 1 has a one-point bootstrap CI: the report's 3/n bound says how far it may be off
  if (!r3 || !isNum(r3.bound)) return null;
  const unit = r3.unit === "event" ? "เหตุการณ์" : r3.unit;
  const reach = r3.side === "upper" ? "สูงถึง" : "ต่ำถึง";
  return `${metric} ${r3.hits} จาก ${r3.n} ${unit}: ช่วงจริงอาจ${reach}ราว ${Math.round(r3.bound * 100)}% (rule of three)`;
}

function experimentTable(exp, variants, budget) {
  const rows = [];
  for (const [variant, cells] of Object.entries(variants)) {
    const cell = cells[budget];
    if (!cell) continue;
    const m = cell.mean || {};
    const ci = cell.ci || {};
    const r3 = cell.rule_of_three || {};
    const flags = [
      ...["recall_any", "recall_correct"].map((k) => ruleOfThreeText(k, r3[k])),
      cell.tau_saturated_any ? "τ saturated" : null,
      cell.deterministic ? "deterministic" : null,
      cell.note,
      ...(cell.errors || []),
    ].filter(Boolean);
    rows.push(
      el(
        "tr",
        {},
        el("td", { class: "mono" }, variant),
        el("td", {}, provenanceBadges(cell.provenance)),
        el("td", { class: "num" }, `${cell.n_runs} (${cell.n_seeds} seed)`),
        el("td", { class: "num mono" }, ciText(m.recall_any, ci.recall_any)),
        el("td", { class: "num mono" }, ciText(m.recall_correct, ci.recall_correct)),
        el("td", { class: "num mono" }, fmtNum(m.fa_per_symbol_day, 2)),
        el("td", { class: "num mono" }, fmtNum(m.fa_per_symbol_day_hard, 2)),
        el("td", { class: "num mono" }, `${fmtNum(m.precision, 3)} (prevalence ${fmtNum(m.prevalence, 4)})`),
        el("td", { class: "num mono" }, fmtNum(m.ttd_median_s, 0)),
        el("td", { class: "num" }, cell.n_events),
        el("td", { class: "small muted notes" }, flags.join(" · ")),
      ),
    );
  }
  const card = el("div", { class: "committee-sec" }, el("h3", {}, exp));
  if (!rows.length) {
    card.appendChild(el("p", { class: "muted small" }, "ไม่มีผลที่งบนี้"));
    return card;
  }
  card.appendChild(
    el(
      "div",
      { class: "table-wrap" },
      el(
        "table",
        { class: "table" },
        el(
          "thead",
          {},
          el(
            "tr",
            {},
            ["variant", "ที่มา", "run", "recall (any) [95% CI]", "recall (correct)", "เตือนผิด/สัญลักษณ์-วัน", "เตือนผิด (hard)", "precision (prevalence)", "TTD มัธยฐาน (s)", "เหตุการณ์", "หมายเหตุ"].map(
              (h) => el("th", {}, h),
            ),
          ),
        ),
        el("tbody", {}, rows),
      ),
    ),
  );
  return card;
}

function pairedCard(paired) {
  const card = el("div", { class: "committee-sec" }, el("h3", {}, `เทียบแบบจับคู่ (paired bootstrap) ที่งบ ${paired.budget} · ${paired.metric}`));
  if ((paired.rows || []).length) card.appendChild(forestPlot(paired));
  const rows = (paired.rows || []).map((row) =>
    el(
      "tr",
      {},
      el("td", { class: "mono" }, `ai.autoencoder_trade vs ${row.baseline || "–"}`),
      el("td", {}, provenanceBadges(row.provenance)),
      el("td", { class: "num mono" }, ciText(row.diff_mean, row.ci)),
      el("td", { class: "num" }, row.n_events),
      el("td", {}, row.significant ? "ผ่านเกณฑ์" : "ไม่ผ่านเกณฑ์"),
      el("td", { class: "small muted" }, row.reason_th || ""),
    ),
  );
  if (rows.length) {
    card.appendChild(
      el(
        "div",
        { class: "table-wrap" },
        el(
          "table",
          { class: "table" },
          el("thead", {}, el("tr", {}, ["คู่", "ที่มา", "ผลต่าง [95% CI]", "เหตุการณ์", "ผล", "หมายเหตุ"].map((h) => el("th", {}, h)))),
          el("tbody", {}, rows),
        ),
      ),
    );
  } else {
    card.appendChild(el("p", { class: "muted small" }, "ยังไม่มีผล"));
  }
  return card;
}

const SVG_NS = "http://www.w3.org/2000/svg";

function svg(tag, attrs, ...children) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, String(v));
  for (const c of children) node.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}

// AE minus each baseline: one CI per row; right of 0 = AE better
function forestPlot(paired) {
  const rows = (paired.rows || []).filter((r) => isNum(r.diff_mean) && Array.isArray(r.ci) && isNum(r.ci[0]) && isNum(r.ci[1]));
  const W = 520;
  const rowH = 28;
  const left = 120;
  const top = 8;
  const H = top + rowH * rows.length + 28;
  const lim = Math.max(0.1, ...rows.map((r) => Math.max(Math.abs(r.ci[0]), Math.abs(r.ci[1]))));
  const x = (v) => left + ((v + lim) / (2 * lim)) * (W - left - 12);
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "forest", role: "img", "aria-label": "ผลต่าง recall ของ AE เทียบแต่ละ baseline พร้อม 95% CI" });
  root.appendChild(svg("line", { x1: x(0), x2: x(0), y1: top, y2: H - 22, class: "zero" }));
  [-lim, 0, lim].forEach((v) => root.appendChild(svg("text", { x: x(v), y: H - 6, "text-anchor": "middle", class: "tick" }, v.toFixed(2))));
  rows.forEach((r, i) => {
    const cy = top + rowH * i + rowH / 2;
    const tone = r.ci[0] > 0 ? "good" : r.ci[1] < 0 ? "bad" : "flat";
    root.appendChild(svg("text", { x: left - 8, y: cy + 4, "text-anchor": "end" }, r.baseline));
    root.appendChild(svg("line", { x1: x(r.ci[0]), x2: x(r.ci[1]), y1: cy, y2: cy, class: `ci-line ${tone}`, style: `transform-origin: ${x(r.diff_mean)}px ${cy}px` }));
    root.appendChild(svg("circle", { cx: x(r.diff_mean), cy, r: 3.5, class: `pt ${tone}` }));
  });
  return root;
}

function whiskerChart(series, order) {
  const W = 520;
  const H = 220;
  const left = 44;
  const bottom = 30;
  const top = 12;
  const plotH = H - top - bottom;
  const y = (v) => top + plotH * (1 - Math.max(0, Math.min(1, v)));
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, class: "whiskers", role: "img", "aria-label": "recall ก่อนและหลัง retrain พร้อม 95% CI" });
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    root.appendChild(svg("line", { x1: left, x2: W - 8, y1: y(v), y2: y(v), class: "axis" }));
    root.appendChild(svg("text", { x: left - 6, y: y(v) + 4, "text-anchor": "end" }, v.toFixed(2)));
  }
  const slot = (W - left - 8) / order.length;
  order.forEach((name, i) => {
    const s = series[name] || {};
    const cx = left + slot * (i + 0.5);
    if (isNum(s.mean)) {
      const bw = Math.min(70, slot * 0.4);
      root.appendChild(svg("rect", { x: cx - bw / 2, y: y(s.mean), width: bw, height: y(0) - y(s.mean), class: "bar", rx: 4 }));
      root.appendChild(svg("text", { x: cx + bw / 2 + 6, y: y(s.mean) + 4 }, s.mean.toFixed(3)));
    }
    if (Array.isArray(s.ci) && isNum(s.ci[0]) && isNum(s.ci[1])) {
      const [lo, hi] = s.ci;
      root.appendChild(svg("line", { x1: cx, x2: cx, y1: y(lo), y2: y(hi), class: "ci" }));
      root.appendChild(svg("line", { x1: cx - 10, x2: cx + 10, y1: y(lo), y2: y(lo), class: "ci" }));
      root.appendChild(svg("line", { x1: cx - 10, x2: cx + 10, y1: y(hi), y2: y(hi), class: "ci" }));
    }
    root.appendChild(svg("text", { x: cx, y: H - 8, "text-anchor": "middle" }, name));
  });
  return root;
}

function retrainCard(title, block) {
  const card = el(
    "div",
    { class: "committee-sec" },
    el("div", { class: "detail-head" }, el("h3", {}, title), provenanceBadges(block.provenance)),
    el(
      "p",
      { class: "small muted" },
      `${block.champion} → ${block.challenger} · ${block.metric} · งบ ${block.budget} · split ${block.split} · ${block.n_seeds} seed · ${block.n_events} เหตุการณ์`,
    ),
  );
  if (block.series) card.appendChild(whiskerChart(block.series, [block.champion, block.challenger]));
  const verdict =
    block.accept === true ? "ยอมรับ challenger" : block.accept === false ? "คง champion ไว้" : "ยังตัดสินไม่ได้";
  card.appendChild(
    el(
      "p",
      { class: `decision ${block.accept === true ? "accept" : block.accept === false ? "reject" : ""}` },
      `Decision: ${verdict}`,
    ),
  );
  if (isNum(block.diff_mean)) card.appendChild(el("p", { class: "small mono" }, `ผลต่าง ${ciText(block.diff_mean, block.ci)}`));
  if (block.reason_th) card.appendChild(el("p", { class: "small" }, block.reason_th));
  if ((block.notes || []).length) card.appendChild(el("ul", { class: "small muted" }, block.notes.map((n) => el("li", {}, n))));
  return card;
}

// ------------------------------------------------------------------ router

const PAGES = {
  login: "page-login",
  activity: "page-activity",
  users: "page-users",
  home: "page-home",
  alerts: "page-alerts",
  alert: "page-alert",
  simulate: "page-simulate",
  results: "page-results",
};

// old links keep working: the 6-tab names map onto the 4 pages; About and Taxonomy live in help
const ROUTE_REDIRECTS = { "#/monitor": "#/home", "#/redteam": "#/simulate", "#/eval": "#/results" };
const HELP_ROUTES = ["#/about", "#/taxonomy"];
const KNOWN_ROUTES = ["#/home", "#/alerts", "#/simulate", "#/results", "#/activity", "#/users"];

// where a hash should land: { route, help }; anything unknown goes home
function routeFor(hash) {
  const raw = typeof hash === "string" ? hash : "";
  if (ROUTE_REDIRECTS[raw]) return { route: ROUTE_REDIRECTS[raw], help: false };
  if (HELP_ROUTES.includes(raw)) return { route: DEFAULT_ROUTE, help: true };
  if (KNOWN_ROUTES.includes(raw)) return { route: raw, help: false };
  if (raw === "#/alerts/") return { route: "#/alerts", help: false };
  if (/^#\/alerts\/./.test(raw)) return { route: raw, help: false };
  if (raw === "#/login" || raw.startsWith("#/login?")) return { route: raw, help: false };
  return { route: DEFAULT_ROUTE, help: false };
}

function parseRoute() {
  const hash = routeFor(location.hash).route;
  const m = /^#\/alerts\/(.+)$/.exec(hash);
  if (m) {
    let id = m[1];
    try {
      id = decodeURIComponent(id);
    } catch (e) {
      /* keep it raw */
    }
    return { name: "alert", id };
  }
  const [path, query] = hash.replace(/^#\//, "").split("?");
  if (path === "login") return { name: "login", next: safeNext(new URLSearchParams(query || "").get("next")) };
  return { name: PAGES[path] ? path : "home" };
}

function route() {
  const hash = location.hash || "";
  const target = routeFor(hash);
  if (target.help) {
    if (state.view === "app") openHelp();
    else state.pendingHelp = true; // no help over the loader or the sign-in page
  }
  if (target.route !== hash) {
    location.replace(target.route); // the hashchange that follows renders the page
    return;
  }
  if (state.view === "loading") return; // boot() routes once /api/me has answered
  const r = parseRoute();
  if (r.name !== "login" && !state.me) {
    goLogin(); // no session yet: a typed or bookmarked page waits behind the sign-in
    return;
  }
  if ((r.name === "activity" || r.name === "users") && !can("admin")) {
    // a convenience redirect: GET /api/audit answers 403 for anyone else anyway
    location.hash = DEFAULT_ROUTE;
    return;
  }
  if (r.name === "login" && state.me) {
    location.hash = r.next; // already signed in (or auth is off): nothing to do here
    return;
  }
  const cameFrom = state.route;
  state.route = r.name;
  document.body.classList.toggle("login-mode", r.name === "login");
  $("topbar").hidden = r.name === "login"; // the sign-in page carries its own logo
  const chip = $("user-chip");
  if (chip) chip.open = false;
  if (r.name !== "alert") destroyDetailChart();
  for (const [name, id] of Object.entries(PAGES)) $(id).hidden = name !== r.name;
  const tab = r.name === "alert" ? "alerts" : r.name;
  for (const a of document.querySelectorAll(".nav a")) {
    if (a.dataset.route === tab) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  }
  if (r.name === "home") {
    initChart();
  } else if (r.name === "alert") {
    showAlert(r.id);
  } else if (r.name === "results") {
    showEval();
  } else if (r.name === "activity") {
    showActivity();
  } else if (r.name === "users") {
    showUsers();
  } else if (r.name === "login") {
    applyFramedHint(window, $("login-framed"), $("login-framed-link"));
    if (cameFrom !== "login") showAuthForm("login"); // a lost session lands on sign-in, not on a half-filled sign-up
    else if (!$("login-form").hidden) $("login-user").focus();
  }
}

// ------------------------------------------------------------------ sign-in, user chip

let loginFails = 0;

// inside someone else's frame (e.g. a reverse proxy or an embedding page) the browser drops the
// SameSite=Strict cookie, so sign-in cannot stick: say so and link to the site itself
function isFramed(win) {
  try {
    return win.top !== win;
  } catch (e) {
    return true; // a cross-origin top that cannot even be compared is still a frame
  }
}

function applyFramedHint(win, box, link) {
  const framed = isFramed(win);
  box.hidden = !framed;
  if (framed) link.href = win.location.href;
  return framed;
}

function showLoginError(text) {
  const box = $("login-error");
  box.hidden = !text;
  box.textContent = text || "";
}

// with auth off the one local user is shown too, so the menu (activity, tour) stays reachable
function renderUserChip() {
  const chip = $("user-chip");
  const me = state.me;
  chip.hidden = !me;
  if (!me) return;
  const on = authOn();
  $("user-name").textContent = on ? me.display_name || me.username : "ใช้งานในเครื่อง";
  $("user-role").textContent = ROLE_TH[me.role] || "";
  $("btn-logout").hidden = !on;
}

// the sign-in answer or /api/me: apply the account, its saved choices, and the role gates
function adoptMe(me) {
  setMe(me);
  const prefs = loadedPrefs();
  if (isNum(prefs.eval_budget)) state.evalBudget = String(prefs.eval_budget);
  renderUserChip();
  applyRoleGates();
  fxApply(); // the account's choice may differ from this browser's
}

// whether sign-in is on: true or false once /api/me answers, null when it could not be reached
async function loadMe() {
  state.bootError = null;
  try {
    const me = await api("/api/me");
    if (!me || typeof me !== "object" || Array.isArray(me)) {
      // a 200 without an account is not "sign-in is off": keep asking
      setMe(null);
      state.bootError = { answered: true };
      console.warn("dashboard: /api/me answered without an account", me);
      return null;
    }
    adoptMe(me);
    return !!me.auth_enabled;
  } catch (e) {
    setMe(null);
    if (e.status === 401) return true; // only sign-in answers 401
    state.bootError = isNum(e.status) ? { status: e.status } : {};
    console.warn("dashboard: /api/me failed:", e.status || "no answer", e.message);
    return null;
  }
}

// leave the loader for the sign-in page or the app
function showView(view) {
  state.view = view;
  $("boot-loader").hidden = view !== "loading";
  $("site-footer").hidden = view !== "app"; // the About link opens the drawer, which only exists in the app
}

// what follows a sign-in or a sign-up: take the account, go where the user was headed, load the app
async function completeSignIn(me) {
  const next = parseRoute().next || DEFAULT_ROUTE;
  adoptMe({ ...me, auth_enabled: true });
  location.hash = next;
  await enterApp();
}

// ------------------------------------------------------------------ sign-up

const SIGNUP_NAME_RE = /^[a-z0-9_.-]{3,32}$/;
const SIGNUP_MIN_PASS = 10;

// the first thing wrong with the form as typed, in plain Thai, or "" when it can go to the server
function signupProblem(username, password, again) {
  if (!SIGNUP_NAME_RE.test(username)) return "ชื่อผู้ใช้ใช้ได้แค่ a-z ตัวเลข _ . - และต้องยาว 3 ถึง 32 ตัว";
  if (password.length < SIGNUP_MIN_PASS) return `รหัสผ่านสั้นไป ต้องมีอย่างน้อย ${SIGNUP_MIN_PASS} ตัว`;
  if (password === username) return "รหัสผ่านต้องไม่เหมือนชื่อผู้ใช้";
  if (password !== again) return "รหัสผ่านสองช่องไม่ตรงกัน";
  return "";
}

function showSignupError(text) {
  const box = $("signup-error");
  box.hidden = !text;
  box.textContent = text || "";
}

// one card at a time, in the same place; the cursor goes to the first field
function showAuthForm(which) {
  const signup = which === "signup";
  $("login-form").hidden = signup;
  $("signup-form").hidden = !signup;
  showLoginError("");
  showSignupError("");
  $(signup ? "signup-user" : "login-user").focus();
}

// the link to sign-up shows only when the server says sign-up is open; any failure leaves it hidden
async function loadSignupConfig() {
  let open = false;
  try {
    const cfg = await api("/api/auth/config");
    open = !!cfg && cfg.signup === true;
  } catch (e) {
    open = false;
  }
  $("signup-line").hidden = !open;
  $("login-note").hidden = open;
  if (!open && !$("signup-form").hidden) showAuthForm("login");
}

function bindSignup() {
  $("signup-open").addEventListener("click", () => showAuthForm("signup"));
  $("signup-back").addEventListener("click", () => showAuthForm("login"));
  $("signup-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const btn = $("signup-submit");
    const username = $("signup-user").value.trim().toLowerCase();
    const password = $("signup-pass").value;
    const problem = signupProblem(username, password, $("signup-pass2").value);
    if (problem) {
      showSignupError(problem);
      return;
    }
    btn.disabled = true;
    showSignupError("");
    try {
      const me = await postJson("/api/auth/signup", {
        username,
        display_name: $("signup-name").value.trim(),
        password,
      });
      $("signup-pass").value = "";
      $("signup-pass2").value = "";
      await completeSignIn(me);
    } catch (e) {
      $("signup-pass").value = "";
      $("signup-pass2").value = "";
      showSignupError(e.message);
    } finally {
      btn.disabled = false;
    }
  });
}

function bindLogin() {
  $("login-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const btn = $("login-submit");
    const username = $("login-user").value.trim();
    const password = $("login-pass").value;
    if (!username || !password) return;
    btn.disabled = true;
    showLoginError("");
    try {
      const me = await postJson("/api/auth/login", {
        username,
        password,
        remember: $("login-remember").checked,
      });
      loginFails = 0;
      $("login-pass").value = "";
      await completeSignIn(me);
    } catch (e) {
      loginFails += 1;
      $("login-pass").value = "";
      showLoginError(
        loginFails >= 5 ? `${e.message} · ลองผิดหลายครั้ง ระบบอาจล็อกชั่วคราว รอ 1 นาทีแล้วลองใหม่` : e.message,
      );
    } finally {
      btn.disabled = false;
    }
  });
  bindSignup();
  $("btn-logout").addEventListener("click", async () => {
    try {
      await api("/api/auth/logout", { method: "POST" });
    } catch (e) {
      /* the session is gone either way */
    }
    location.hash = "#/login";
    location.reload(); // drop every bit of the last user's data from memory
  });
  document.addEventListener("click", (ev) => {
    const chip = $("user-chip");
    if (chip.open && !chip.contains(ev.target)) chip.open = false;
  });
  document.addEventListener("keydown", (ev) => {
    const chip = $("user-chip");
    if (ev.key !== "Escape" || !chip.open) return;
    chip.open = false;
    chip.querySelector("summary").focus();
  });
}

// ------------------------------------------------------------------ activity (admin)

const AUDIT_PAGE = 100;

function auditFilters() {
  return { user: $("audit-user").value.trim().toLowerCase(), action: $("audit-action").value };
}

// the server may send the detail as JSON text or already parsed: either way it is shown as plain text
function auditDetail(d) {
  if (d === null || d === undefined || d === "") return "–";
  if (typeof d === "string") return d;
  try {
    return JSON.stringify(d);
  } catch (e) {
    return "–";
  }
}

// every field is untrusted (a failed sign-in's username is whatever was typed): text nodes only, no links
function renderAudit() {
  const want = auditFilters();
  const rows = state.audit.filter(
    (r) => (!want.user || String(r.username || "").toLowerCase().includes(want.user)) && (!want.action || r.action === want.action),
  );
  const body = clear($("audit-body"));
  for (const r of rows) {
    body.appendChild(
      el(
        "tr",
        {},
        el("td", { class: "mono when" }, fmtUs(r.ts_us)),
        el("td", {}, r.username === null || r.username === undefined || r.username === "" ? "–" : r.username),
        el("td", { class: "mono" }, r.action),
        el("td", {}, r.target || "–"),
        el("td", { class: "mono detail" }, auditDetail(r.detail)),
      ),
    );
  }
  $("audit-empty").hidden = rows.length > 0;
  $("audit-more").hidden = state.auditDone;
  $("audit-status").textContent = `${rows.length} จาก ${state.audit.length} รายการที่โหลดแล้ว`;
}

async function loadAudit(older) {
  const q = new URLSearchParams({ limit: String(AUDIT_PAGE) });
  const last = state.audit[state.audit.length - 1];
  if (older && last) q.set("before_us", String(last.ts_us));
  const btn = $("audit-more");
  btn.disabled = true;
  try {
    const rows = await api(`/api/audit?${q}`);
    const list = Array.isArray(rows) ? rows : [];
    state.audit = older ? state.audit.concat(list) : list;
    state.auditDone = list.length < AUDIT_PAGE;
  } catch (e) {
    $("audit-status").textContent = e.message;
  } finally {
    btn.disabled = false;
  }
  renderAudit();
}

function showActivity() {
  const sel = $("audit-action");
  if (!sel.options.length) {
    sel.appendChild(el("option", { value: "" }, "ทั้งหมด"));
    for (const a of AUDIT_ACTIONS) sel.appendChild(el("option", { value: a }, a));
  }
  loadAudit(false);
}

function bindActivity() {
  $("audit-user").addEventListener("input", renderAudit);
  $("audit-action").addEventListener("change", renderAudit);
  $("audit-more").addEventListener("click", () => loadAudit(true));
}

// ------------------------------------------------------------------ users (admin)

// "14:05 น." for today, "3 พ.ย. 14:05 น." for an earlier day (Thai time)
const TH_MONTHS = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."];

function fmtThaiDayTime(us, nowUs) {
  if (!isNum(us) || us <= 0) return "ยังไม่เคยเข้า";
  const day = (t) => Math.floor((t + TH_OFFSET_US) / (86400 * S_US));
  const time = fmtThaiTime(us);
  if (isNum(nowUs) && day(us) === day(nowUs)) return time;
  const when = new Date(Math.floor((us + TH_OFFSET_US) / 1000));
  return `${when.getUTCDate()} ${TH_MONTHS[when.getUTCMonth()]} ${time}`;
}

function usersStatus(text) {
  $("users-status").textContent = text || "";
}

// one control changed: send it, toast the result; on a refusal put the control back and say why
async function changeUser(u, control, path, body, undo, done) {
  control.disabled = true;
  try {
    const saved = await postJson(`/api/users/${encodeURIComponent(u.username)}/${path}`, body);
    Object.assign(u, saved);
    showToast(done(saved));
    renderUsers();
  } catch (e) {
    undo();
    showToast(e.message);
    control.disabled = false;
  }
}

function userRow(u, nowUs) {
  const mine = !!state.me && u.username === state.me.username;
  const name = u.display_name && u.display_name !== u.username ? u.display_name : null;
  const role = el("select", { "aria-label": `สิทธิ์ของ ${u.username}`, disabled: mine });
  for (const r of ROLE_ORDER) role.appendChild(el("option", { value: r }, ROLE_TH[r]));
  role.value = u.role;
  role.addEventListener("change", () => {
    const before = u.role;
    changeUser(u, role, "role", { role: role.value }, () => (role.value = before), (upd) => `${upd.username} เป็น${ROLE_TH[upd.role]}แล้ว`);
  });
  const on = !u.disabled;
  const sw = el(
    "button",
    {
      type: "button",
      class: "switch",
      role: "switch",
      "aria-checked": String(on),
      "aria-label": `เปิดใช้บัญชี ${u.username}`,
      disabled: mine,
      title: mine ? "ปิดบัญชีตัวเองไม่ได้" : null,
    },
    el("span", { class: "switch-knob", "aria-hidden": "true" }),
  );
  sw.addEventListener("click", () => {
    sw.setAttribute("aria-checked", String(!on));
    changeUser(u, sw, "disabled", { disabled: on }, () => sw.setAttribute("aria-checked", String(on)), (upd) =>
      upd.disabled ? `ปิดบัญชี ${upd.username} แล้ว` : `เปิดบัญชี ${upd.username} แล้ว`,
    );
  });
  return el(
    "tr",
    { class: u.disabled ? "gone" : null },
    el(
      "td",
      {},
      el("span", { class: "user-name" }, u.username),
      name ? el("span", { class: "muted small user-sub" }, name) : null,
      mine ? el("span", { class: "tag" }, "คุณ") : null,
      u.from_env ? el("span", { class: "tag", title: "บัญชีนี้มาจากค่าตั้งของเซิร์ฟเวอร์" }, "จาก IDSAI_USERS") : null,
    ),
    el("td", {}, role),
    el("td", {}, el("span", { class: "switch-wrap" }, sw, el("span", { class: "small" }, on ? "ใช้งานอยู่" : "ปิดอยู่"))),
    el("td", { class: "when" }, fmtThaiDayTime(u.last_seen_us, nowUs)),
  );
}

function renderUsers() {
  const nowUs = Date.now() * 1000;
  const body = clear($("users-body"));
  for (const u of state.users) body.appendChild(userRow(u, nowUs));
  usersStatus(state.users.length ? `${state.users.length} บัญชี` : "ยังไม่มีบัญชี");
}

async function showUsers() {
  usersStatus("กำลังโหลด");
  try {
    const rows = await api("/api/users");
    state.users = Array.isArray(rows) ? rows : [];
    renderUsers();
  } catch (e) {
    usersStatus(e.message);
  }
}

// ------------------------------------------------------------------ websocket

function wsBanner(text) {
  const b = $("ws-banner");
  b.hidden = !text;
  b.textContent = text || "";
}

function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  let ws;
  try {
    ws = new WebSocket(`${proto}//${location.host}/ws/stream`);
  } catch (e) {
    scheduleReconnect();
    return;
  }
  state.ws = ws;
  ws.addEventListener("open", () => {
    const again = state.wsEverOpen;
    state.wsEverOpen = true;
    state.wsRetry = 0;
    wsBanner("");
    if (again) {
      // whatever happened while we were away: re-read it
      loadAlerts();
      loadSeries();
    }
  });
  ws.addEventListener("message", safely((ev) => dispatch(JSON.parse(ev.data))));
  ws.addEventListener("close", (ev) => onWsClose(ws, ev && ev.code));
  ws.addEventListener("error", () => {
    /* close follows */
  });
}

function onWsClose(ws, code) {
  setLive(false);
  if (state.ws !== ws) return;
  if (code === 4401) {
    // no valid session: retrying cannot help. Sign in, then enterApp connects again
    state.ws = null;
    state.wsRetry = 0;
    sessionLost();
    return;
  }
  scheduleReconnect();
}

function scheduleReconnect() {
  const delay = Math.min(1000 * 2 ** state.wsRetry, 15000);
  state.wsRetry += 1;
  wsBanner("การเชื่อมต่อหลุด กำลังเชื่อมใหม่…");
  setTimeout(connect, delay);
}

function dispatch(msg) {
  if (!msg || typeof msg !== "object") return;
  const { type, data } = msg;
  setLive(state.replaying || state.mode === "live"); // paused replay: off, whatever the server still sends
  if (type === "bar") onBar(data);
  else if (type === "score") onScore(data);
  else if (type === "alert") onAlert(data);
  else if (type === "health" && data && typeof data === "object") renderHealth(data);
}

// ------------------------------------------------------------------ boot

function fillSymbols() {
  for (const id of ["symbol-select", "rt-symbol"]) {
    const sel = clear($(id));
    for (const s of state.symbols) sel.appendChild(el("option", { value: s }, s));
    sel.value = state.symbol || "";
  }
  $("symbol-name").textContent = state.symbol || "–";
  $("symbol-pick").hidden = state.symbols.length < 2; // one coin: nothing to choose
}

// the part of boot that needs a signed-in user: symbols, health, series, alerts, websocket
async function enterApp() {
  showView("app");
  if (state.pendingHelp) {
    state.pendingHelp = false;
    openHelp();
  }
  if (state.started) {
    // signed in again mid-session: the same page, fresh data
    applyRoleGates();
    route();
    loadAlerts();
    loadSeries();
    if (!state.ws) connect(); // the 4401 close left no socket
    return;
  }
  state.started = true;
  route();
  try {
    state.symbols = await api("/api/symbols");
  } catch (e) {
    state.symbols = [];
  }
  const savedSymbol = loadedPrefs().symbol;
  state.symbol = state.symbols.includes(savedSymbol) ? savedSymbol : state.symbols[0] || null;
  fillSymbols();
  try {
    renderHealth(await api("/api/health"));
  } catch (e) {
    $("health-text").textContent = "อ่านสถานะไม่ได้";
  }
  initChart();
  loadTaxonomy().then(() => renderRail()); // the rail names patterns in Thai once the taxonomy is in
  await Promise.all([loadSeries(), loadAlerts()]);
  connect();
  setInterval(safely(tickInjections), 1000);
  maybeStartTour();
}

async function boot() {
  window.addEventListener("hashchange", safely(route));
  bindHelp();
  $("menu-tour").addEventListener("click", () => {
    $("user-chip").open = false;
    startTour();
  });
  bindLogin();
  const fxSwitch = $("menu-fx");
  if (fxSwitch) fxSwitch.addEventListener("click", () => setFxOff(fxSwitch.getAttribute("aria-pressed") !== "true"));
  bindActivity();
  document.addEventListener("keydown", safely(onPresenterKey));
  bindReplay();
  bindMonitor();
  bindSimulate();
  $("alerts-more").addEventListener("click", () => {
    state.alertsShown += ALERT_PAGE;
    renderAlertCards();
  });
  $("symbol-select").addEventListener("change", (ev) => {
    state.symbol = ev.target.value;
    $("rt-symbol").value = state.symbol;
    $("symbol-name").textContent = state.symbol || "–";
    savePrefs({ symbol: state.symbol });
    state.alertCount = 0;
    state.alertCountKnown = false; // a new coin's count is not news
    renderBudget(); // the count and the budget line are per coin
    loadSeries();
  });
  if (!LWC) console.warn("dashboard: the chart library did not load");

  loadSignupConfig();
  let authEnabled = await loadMe();
  fx("init", { off: fxOff() }); // motion.js loads right after this file, so it is ready once the first answer is in
  fxApply();
  for (let attempt = 0; bootView(state.me, authEnabled) === "loading"; attempt += 1) {
    // no usable answer: keep only the loader up, say why, and ask again with a growing wait
    $("boot-text").textContent = bootRetryText(state.bootError);
    await new Promise((resolve) => setTimeout(resolve, bootRetryDelay(attempt)));
    authEnabled = await loadMe();
  }
  if (bootView(state.me, authEnabled) === "login") {
    showView("login");
    route(); // the 401 already sent the page to the sign-in route
    return;
  }
  await enterApp();
}

if (typeof document !== "undefined") {
  boot().catch((e) => console.warn("dashboard boot:", e));
} else if (typeof module !== "undefined") {
  // node (tests only): the pure helpers
  module.exports = {
    snapshotAlerts,
    matchesInjection,
    eligibleForRun,
    firstDetection,
    resetRuns,
    restartRuns,
    injectionStatus,
    simStatusText,
    simulateRequest,
    simReason,
    simErrorText,
    control,
    SIM_CARDS,
    SIM_REASON,
    setMode,
    bindSimulate,
    redTeamAllowed,
    reducedMotion,
    EVAL_TARGETS,
    targetVerdict,
    resultSentence,
    RESULT_BADGE,
    showEval,
    evalEmpty,
    forestPlot,
    tweenNumber,
    flash,
    presenterKey,
    bootView,
    bootRetryText,
    bootRetryDelay,
    loadMe,
    getBootError: () => state.bootError,
    playToggleAction,
    noteReplayAction,
    getReplaying: () => state.replaying,
    PAGE_ORDER,
    DEFAULT_ROUTE,
    heroStatus,
    heroPending,
    renderHero,
    renderAlertsKpi,
    loadAlerts,
    plainErrorText,
    getState: () => state,
    HERO_WATCH_RATIO,
    HERO_RECENT_US,
    SPEED_CHIPS,
    firstSentence,
    applyRoleGates,
    ALERT_FILTERS,
    thaiTechniqueName,
    trapTab,
    seekInputToUs,
    showView,
    VERDICT_TH,
    filterAlerts,
    onAlert,
    showToast,
    showAlert,
    TOUR_STEPS,
    tourShouldShow,
    markTourDone,
    placeBubble,
    renderHelpTaxonomy,
    startTour,
    endTour,
    fx,
    fxUserOff,
    setFxOff,
    heroMood,
    // every Thai UI string table: the copy guard in the tests reads these
    copyTables: () => ({
      SEVERITY_TH,
      STATUS_TH,
      CHECK_TH,
      CHECK_NAME_TH,
      VIEWER_REPLAY_TH,
      VIEWER_VERDICT_TH,
      VERDICT_TH,
      SIM_WAIT_TH,
      SIM_OFFSET_TH,
      SIM_PAST_END_TH,
      SIM_SYMBOL_TH,
      SIM_FAILED_TH,
      REPLAY_ENDED_TH,
      SEMISYN_TH,
      BASELINE_TH,
      NEED_ANALYST_TH,
      TECHNIQUE_TH,
      NET_FAILED_TH,
      ACT_FAILED_TH,
      ACT_FORBIDDEN_TH,
      ACT_CONFLICT_TH,
      ACT_INVALID_TH,
      ROLE_TH,
      MODE_TH,
      TARGET_STATE_TH,
      NO_TARGET_TH,
      PER_DAY_TH,
      SIM_REASON,
      RESULT_BADGE,
      SIM_CARDS: SIM_CARDS.map((c) => ({ title: c.title, line: c.line })),
      TOUR_STEPS,
      ALERT_FILTERS,
      SPEED_CHIPS,
    }),
    sessionLost,
    bindHelp,
    helpIsOpen: () => helpState.open,
    setView: (v) => {
      state.view = v;
    },
    openHelp,
    closeHelp,
    setRoute: (r) => {
      state.route = r;
    },
    getAlertIds: () => [...state.alerts.keys()],
    getAlertsWithFeedback: () => [...state.alerts.values()].map(withFeedback),
    fmtThaiTime,
    railHeadline,
    ROLE_TH,
    routeFor,
    safeNext,
    isFramed,
    applyFramedHint,
    can,
    setMe,
    savePrefs,
    api,
    onWsClose,
    getMe: () => state.me,
    setWs: (w) => {
      state.ws = w;
    },
    getWs: () => state.ws,
  };
}
