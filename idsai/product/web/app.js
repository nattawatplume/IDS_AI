/* Market IDS dashboard: one plain script, no build step, same-origin fetch and WebSocket only.
 *
 * Every server string reaches the DOM through textContent (`el`); nothing is parsed as HTML.
 * API times are µs UTC; the chart uses whole seconds (UTCTimestamp) and every shown time is UTC.
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
const CHECK_TH = { ok: "ปกติ", warn: "เฝ้าระวัง", fail: "ผิดปกติ" };
const VERDICT_TH = { tp: "ผิดปกติจริง (TP)", fp: "แจ้งเตือนผิด (FP)", unsure: "ไม่แน่ใจ" };
const REFUSED_TH =
  "Red-Team ปิดอยู่ในโหมด LIVE เพราะระบบไม่แก้ไขข้อมูลตลาดจริง (ใช้ได้เฉพาะโหมด replay / sim)";
const UNKNOWN_MODE_TH =
  "Red-Team ยังปิดอยู่จนกว่าจะยืนยันจากเซิร์ฟเวอร์ได้ว่าไม่ใช่โหมด LIVE (ใช้ได้เฉพาะโหมด replay / sim)";
const RED_TEAM_MODES = ["replay", "sim"];
const PROVENANCES = ["REAL", "SEMI-SYN", "SYN"];

// Eval targets: docs/superpowers/plans/2026-10-27-market-ids-mvp.md Task 20 Step 6 (spec section 6).
// "pair" cards need the AE-minus-baseline CI lower bound above 0; "reference" pairs show the
// diff and CI only (the MVP spec asks the AE to beat iforest / robust_z / kk, not random / pca).
const EVAL_TARGETS = [
  { id: "real_recall", label: "recall_any ≥ 0.8 · REAL (real_t001)", experiment: "real_t001", variant: "full", metric: "recall_any", budget: 2, cmp: ">=", threshold: 0.8, correct: true },
  ...["pump_burst", "pump_stealth", "wash_pingpong"].map((v) => ({
    id: `semisyn_${v}`, label: `recall_any ≥ 0.8 · SEMI-SYN ${v}`, experiment: "semisyn", variant: v, metric: "recall_any", budget: 2, cmp: ">=", threshold: 0.8, correct: true,
  })),
  { id: "real_fa", label: "เตือนผิด ≤ งบ · REAL", experiment: "real_t001", variant: "full", metric: "fa_per_symbol_day", budget: 2, cmp: "<=", threshold: 2 },
  { id: "real_fa_hard", label: "เตือนผิด (hard) ≤ 0 · REAL", experiment: "real_t001", variant: "full", metric: "fa_per_symbol_day_hard", budget: 2, cmp: "<=", threshold: 0 },
  ...["baseline.iforest", "stat.robust_z", "baseline.kk"].map((b) => ({
    id: `ae_vs_${b}`, label: `AE เหนือ ${b}`, experiment: null, variant: b, metric: "recall_any", budget: 2, cmp: "pair", threshold: 0,
  })),
  ...["baseline.random", "baseline.pca"].map((b) => ({
    id: `ae_vs_${b}`, label: `AE เทียบ ${b}`, experiment: null, variant: b, metric: "recall_any", budget: 2, cmp: "pair", threshold: 0, reference: true,
  })),
  { id: "latency_p99", label: "compute latency p99 < 2 s", cmp: "none", threshold: 2 },
];

const state = {
  mode: null,
  replaying: true, // a replay starts running; control() keeps this in step
  health: null,
  symbols: [],
  symbol: null,
  alerts: new Map(), // alert_id -> item
  taxonomy: null,
  lastBarUs: null,
  ws: null,
  wsRetry: 0,
  wsEverOpen: false,
  route: null,
  detailChart: null,
  injections: [],
  railIds: null, // alert ids already on the rail; null until the first render
  alertCount: 0,
  layersOpen: false,
  me: null, // GET /api/me: { username, display_name, role, auth_enabled, prefs }
  started: false, // the data boot (symbols, series, alerts, websocket) has run
  evalBudget: null, // saved Eval budget (string, as the option value)
  audit: [], // Activity rows already loaded, newest first
  auditDone: false,
};

// ------------------------------------------------------------------ roles + sign-in state

const ROLE_ORDER = ["viewer", "analyst", "admin"];
const AUDIT_ACTIONS = [
  "login", "login_failed", "logout", "feedback", "replay_control", "redteam_inject",
  "user_add", "user_role", "user_disable", "user_enable", "user_passwd",
];
const NEED_ANALYST_TH = "ต้องเป็น analyst";
const DEFAULT_ROUTE = "#/monitor";

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

function goLogin() {
  const here = location.hash || "";
  if (here.startsWith("#/login")) return;
  const next = safeNext(here);
  location.hash = next === DEFAULT_ROUTE && !here.startsWith(DEFAULT_ROUTE) ? "#/login" : `#/login?next=${encodeURIComponent(next)}`;
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
  if (res.status === 401 && path !== "/api/auth/login") goLogin(); // a failed sign-in must not bounce to itself
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

const PREF_KEYS = { theme: "theme", symbol: "idsai.symbol", eval_budget: "idsai.eval_budget" };
let prefsPending = {};
let prefsTimer = null;

function authOn() {
  return !!(state.me && state.me.auth_enabled);
}

// the saved choices: the account's when auth is on, this browser's otherwise
function loadedPrefs() {
  if (authOn()) return state.me.prefs || {};
  const out = {};
  const theme = storageGet(PREF_KEYS.theme);
  if (theme) out.theme = theme;
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

const THEMES = ["auto", "dark", "light"];

function currentTheme() {
  const t = document.documentElement.dataset.theme;
  return THEMES.includes(t) ? t : "auto";
}

function applyTheme(choice) {
  const c = THEMES.includes(choice) ? choice : "auto";
  if (c === "auto") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = c;
  const btn = $("theme-toggle");
  if (btn) {
    btn.textContent = c.toUpperCase();
    btn.title = `ธีม: ${c}`;
  }
  for (const b of document.querySelectorAll("[data-theme-choice]")) {
    b.setAttribute("aria-pressed", String(b.dataset.themeChoice === c));
  }
  storageSet("theme", c);
  restyleCharts();
}

// a theme change the user made: apply it and remember it with the account
function chooseTheme(choice) {
  applyTheme(choice);
  savePrefs({ theme: currentTheme() });
}

function bindTheme() {
  const saved = storageGet("theme");
  applyTheme(THEMES.includes(saved) ? saved : "auto");
  $("theme-toggle").addEventListener("click", () => {
    chooseTheme(THEMES[(THEMES.indexOf(currentTheme()) + 1) % THEMES.length]);
  });
  for (const b of document.querySelectorAll("[data-theme-choice]")) {
    b.addEventListener("click", () => chooseTheme(b.dataset.themeChoice));
  }
}

// ------------------------------------------------------------------ motion + presenter keys

const PAGE_ORDER = ["#/monitor", "#/alerts", "#/redteam", "#/eval", "#/taxonomy", "#/about"];

function reducedMotion() {
  const mm = typeof globalThis.matchMedia === "function" ? globalThis.matchMedia : null;
  return !!(mm && mm.call(globalThis, "(prefers-reduced-motion: reduce)").matches);
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

function presenterKey(ev) {
  if (!ev || ev.ctrlKey || ev.metaKey || ev.altKey) return null;
  const tag = ev.target && ev.target.tagName;
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return null;
  // ev.code is the physical key: it still works when a Thai layout changes ev.key
  const code = ev.code || "";
  if (ev.key === " " || code === "Space") return "toggle-play";
  if (ev.key === "t" || ev.key === "T" || code === "KeyT") return "theme";
  if (/^[1-6]$/.test(ev.key || "")) return `page:${ev.key}`;
  const digit = /^(?:Digit|Numpad)([1-6])$/.exec(code);
  if (digit) return `page:${digit[1]}`;
  return null;
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
  const act = presenterKey(ev);
  if (!act) return;
  ev.preventDefault();
  if (act === "toggle-play") {
    const next = playToggleAction(state.mode, state.replaying);
    if (next) control(next);
  } else if (act === "theme") {
    chooseTheme(THEMES[(THEMES.indexOf(currentTheme()) + 1) % THEMES.length]);
  } else {
    location.hash = PAGE_ORDER[Number(act.slice(5)) - 1];
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
  if (isNum(h.event_us)) $("clock").textContent = fmtUs(h.event_us);
  else if (isNum(state.lastBarUs)) $("clock").textContent = fmtUs(state.lastBarUs);

  const checks = Array.isArray(h.checks) ? h.checks : [];
  const dot = $("auditor-dot");
  dot.className = `dot ${h.overall || ""}`;
  const warm = checks.find((c) => c.name === "warmup");
  let text = `${CHECK_TH[h.overall] || h.overall || "–"}`;
  if (warm && warm.status !== "ok") {
    const hours = warm.value && warm.value.hours_remaining;
    text = isNum(hours)
      ? `กำลังอุ่นเครื่อง (เหลือประมาณ ${hours.toFixed(1)} ชม.)`
      : "กำลังอุ่นเครื่อง";
  }
  $("auditor-text").textContent = text;
  const kpiAud = $("kpi-auditor");
  kpiAud.textContent = text;
  kpiAud.className = `kpi-value ${h.overall || ""}`;
  $("auditor").title = text; // the warm-up wording lives in the tooltip, the bar stays short
  const list = clear($("auditor-checks"));
  for (const c of checks) {
    list.appendChild(
      el(
        "li",
        {},
        el("span", { class: `dot ${c.status}`, style: "display:inline-block;margin-right:6px" }),
        el("strong", {}, c.name),
        ` · ${CHECK_TH[c.status] || c.status}`,
        el("div", { class: "muted" }, c.detail_th || ""),
      ),
    );
  }
  if (isNum(h.late_trades) && h.late_trades > 0) {
    list.appendChild(el("li", { class: "muted" }, `การซื้อขายที่มาช้า: ${h.late_trades}`));
  }
  renderBudget();
  tickInjections(); // the Red-Team card moves with the clock above it
}

function redTeamAllowed(mode) {
  // fail closed: only a mode the server confirmed as replay / sim enables injection
  return RED_TEAM_MODES.includes(mode);
}

function setMode(mode) {
  if (!mode || mode === state.mode) return;
  state.mode = mode;
  state.replaying = true; // a new mode starts with a running replay
  const lamp = $("live-dot");
  lamp.classList.remove("ok", "fail");
  lamp.classList.add(mode === "live" ? "fail" : "ok");
  const badge = $("mode-badge");
  badge.textContent = String(mode).toUpperCase();
  badge.className = `badge badge-mode ${mode}`;
  const live = mode === "live";
  $("replay-controls").hidden = live;
  const allowed = redTeamAllowed(mode);
  const refused = $("redteam-refused");
  refused.hidden = allowed;
  refused.textContent = allowed ? "" : live ? REFUSED_TH : UNKNOWN_MODE_TH;
  for (const node of $("redteam-form").querySelectorAll("select, input, button")) {
    node.disabled = !allowed;
  }
  applyRoleGates();
}

// UI convenience only: the server enforces the role on every route
function applyRoleGates() {
  const ok = can("analyst");
  const gate = (node) => {
    if (ok) node.removeAttribute("title");
    else node.title = NEED_ANALYST_TH;
  };
  for (const id of ["btn-play", "btn-pause", "speed-select", "seek-input", "btn-seek"]) {
    const node = $(id);
    node.disabled = !ok;
    gate(node);
  }
  const allowed = redTeamAllowed(state.mode) && ok;
  for (const node of $("redteam-form").querySelectorAll("select, input, button")) {
    node.disabled = !allowed;
    gate(node);
  }
  for (const node of document.querySelectorAll(".feedback button, .feedback input")) {
    node.disabled = !ok;
    gate(node);
  }
  $("tab-activity").hidden = !can("admin");
}

function eventNowUs() {
  const h = state.health;
  if (h && isNum(h.event_us)) return h.event_us;
  return state.lastBarUs;
}

function renderBudget() {
  const h = state.health;
  const budget = h && h.budget_per_symbol_day;
  const now = eventNowUs();
  const fill = $("budget-fill");
  if (!isNum(budget) || !isNum(now)) {
    $("budget-text").textContent = "–";
    $("kpi-budget").textContent = "งบ –";
    fill.style.width = "0";
    renderAlertsKpi(now);
    return;
  }
  const nSym = Math.max(state.symbols.length, 1);
  let n = 0;
  for (const a of state.alerts.values()) {
    if (a.ts_event_us > now - DAY_US && a.ts_event_us <= now) n += 1;
  }
  const rate = n / nSym;
  const frac = budget > 0 ? rate / budget : 0;
  fill.style.width = `${Math.min(frac, 1) * 100}%`;
  fill.className = `meter-fill${frac > 1 ? " over" : frac >= 0.8 ? " warn" : ""}`;
  $("budget-text").textContent = `${rate.toFixed(1)} / ${budget} ครั้งต่อสัญลักษณ์-วัน`;
  $("kpi-budget").textContent = `งบ ${budget}`;
  renderAlertsKpi(now);
}

// the KPI counts the selected symbol's alerts of the last 24 h (the unit of the budget above)
function renderAlertsKpi(now = eventNowUs()) {
  const box = $("kpi-alerts");
  if (!state.symbol || !isNum(now)) {
    box.textContent = "–";
    state.alertCount = 0;
    return;
  }
  let n = 0;
  for (const a of state.alerts.values()) {
    if (a.symbol === state.symbol && a.ts_event_us > now - DAY_US && a.ts_event_us <= now) n += 1;
  }
  const rose = n > state.alertCount;
  state.alertCount = n;
  tweenNumber(box, n, 0);
  if (rose) flash(box.parentNode);
}

// ------------------------------------------------------------------ monitor chart

const chart = {
  api: null,
  price: null,
  f: null,
  lanes: {},
  tau: null,
  tauLine: null,
  markers: null,
  last: {}, // series key -> last plotted time (s)
  priceTimes: [], // sorted times of the price series, for snapping markers
  now: { price: null, f: null }, // latest plotted values, shown while the pointer is away
};

function chartColors() {
  return {
    bg: cssVar("--panel"),
    text: cssVar("--dim"),
    grid: cssVar("--line"),
    price: cssVar("--line-price"),
    f: cssVar("--line-f"),
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
    timeScale: { borderColor: c.grid, timeVisible: true, secondsVisible: true },
    localization: { timeFormatter: (t) => (isNum(t) ? fmtSec(t) : "") },
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
  chart.f = chart.api.addSeries(
    LWC.LineSeries,
    lineOptions(c.f, "F (−log10 p)", {
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
  if (!chart.api) return;
  if (open) {
    addLanes();
    loadSeries();
  } else {
    removeLanes();
  }
}

function setInfoOpen(open) {
  $("chart-info").setAttribute("aria-expanded", String(open));
  $("chart-info-body").hidden = !open;
}

function bindMonitor() {
  $("layers-toggle").addEventListener("click", () => setLayersOpen(!state.layersOpen));
  $("chart-info").addEventListener("click", () => setInfoOpen($("chart-info-body").hidden));
}

function renderReadout(time, price, f) {
  const parts = [];
  if (isNum(time)) parts.push(fmtSec(time));
  parts.push(`ราคา ${isNum(price) ? fmtNum(price, 8) : "–"}`);
  parts.push(`F ${isNum(f) ? fmtNum(f, 2) : "–"}`);
  $("readout").textContent = parts.join(" · ");
}

function onCrosshair(param) {
  if (!param || !isNum(param.time) || !param.seriesData) {
    renderReadout(null, chart.now.price, chart.now.f); // pointer left: show the latest
    return;
  }
  const at = (series) => {
    const pt = series && param.seriesData.get(series);
    return pt && isNum(pt.value) ? pt.value : null;
  };
  renderReadout(param.time, at(chart.price), at(chart.f));
}

function restyleCharts() {
  if (chart.api) {
    chart.api.applyOptions(baseChartOptions());
    chart.price.applyOptions({ color: cssVar("--line-price") });
    chart.f.applyOptions({ color: cssVar("--line-f") });
    LANES.forEach(([key, , color]) => {
      if (chart.lanes[key]) chart.lanes[key].applyOptions({ color: cssVar(color) });
    });
    if (chart.tauLine) chart.tauLine.applyOptions({ color: cssVar("--tau") });
  }
  if (state.detailChart) state.detailChart.restyle();
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

function showTau(tau) {
  $("kpi-tau").textContent = fmtNum(tau, 2);
}

function showF(f) {
  tweenNumber($("kpi-f"), f, 2);
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
      title: "τ",
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
  chart.last.f = lastTime(f);
  chart.now = { price: lastValue(price), f: lastValue(f) };
  renderReadout(null, chart.now.price, chart.now.f);
  for (const [key] of LANES) {
    if (!chart.lanes[key]) continue;
    const d = seriesData(times, s[key] || []);
    chart.lanes[key].setData(d);
    chart.last[key] = lastTime(d);
  }
  setTau(s.tau);
  showTau(s.tau);
  showF(chart.now.f);
  $("chart-empty").hidden = price.length > 0;
  renderMarkers();
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
      color: sevColor(a.severity),
      text: a.technique_id,
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
  if (state.mode === "live") $("clock").textContent = fmtUs(state.lastBarUs);
  if (bar.symbol !== state.symbol || !chart.api) return;
  const t = toSec(closeUs); // a bar closes at t_us + 1 s, where its score row ends
  if (plot("price", chart.price, t, bar.close)) {
    chart.priceTimes.push(t);
    $("chart-empty").hidden = true;
  }
}

function onScore(p) {
  if (!p || p.symbol !== state.symbol || !chart.api || !isNum(p.t_end_us)) return;
  const t = toSec(p.t_end_us);
  if (plot("f", chart.f, t, p.F)) showF(p.F);
  for (const [key] of LANES) plot(key, chart.lanes[key], t, p[key]);
}

// ------------------------------------------------------------------ replay controls

async function control(action, value) {
  const status = $("replay-status");
  if (!can("analyst")) {
    status.textContent = NEED_ANALYST_TH;
    return;
  }
  status.textContent = "…";
  try {
    const body = value === undefined ? { action } : { action, value };
    const r = await postJson("/api/replay/control", body);
    noteReplayAction(action);
    if (action === "play" || action === "pause") setLive(state.replaying);
    status.textContent = `ตำแหน่ง ${fmtUs(r.cursor_us)}`;
  } catch (e) {
    status.textContent = e.message;
  }
}

function bindReplay() {
  $("btn-play").addEventListener("click", () => control("play"));
  $("btn-pause").addEventListener("click", () => control("pause"));
  $("speed-select").addEventListener("change", (ev) => {
    const v = Number(ev.target.value);
    if (Number.isFinite(v)) control("speed", v);
  });
  $("btn-seek").addEventListener("click", () => {
    const raw = $("seek-input").value;
    const ms = Date.parse(raw.length === 16 ? `${raw}:00Z` : `${raw}Z`); // the input is UTC
    if (!raw || Number.isNaN(ms)) {
      $("replay-status").textContent = "ใส่วันและเวลา (UTC) ก่อน";
      return;
    }
    control("seek", Math.round(ms) * 1000);
  });
}

// ------------------------------------------------------------------ alerts

async function loadAlerts() {
  try {
    const items = await api("/api/alerts?limit=1000");
    state.alerts = new Map(items.map((a) => [a.alert_id, a]));
  } catch (e) {
    console.warn("alerts:", e.message);
    return;
  }
  alertsChanged();
}

function alertsChanged() {
  renderAlertsTable();
  renderRail();
  renderMarkers();
  renderBudget();
  checkInjections();
}

function onAlert(data) {
  if (!data || typeof data !== "object") return;
  if (data.event === "resync") {
    loadAlerts();
    return;
  }
  if (data.event === "alerts_reset") {
    for (const id of data.deleted || []) state.alerts.delete(id);
    if (resetRuns(state.injections, data.deleted || [])) renderInjections();
    alertsChanged();
    loadSeries();
    return;
  }
  if (typeof data.alert_id !== "string") return;
  state.alerts.set(data.alert_id, data);
  alertsChanged();
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

function renderAlertsTable() {
  const body = clear($("alerts-body"));
  const items = sortedAlerts();
  $("alerts-empty").hidden = items.length > 0;
  $("alerts-table").hidden = items.length === 0;
  for (const a of items) {
    const row = el(
      "tr",
      { class: "clickable", tabindex: "0", "data-sev": sevKey(a.severity) },
      el("td", {}, el("span", { class: `sev sev-${a.severity}` }, SEVERITY_TH[a.severity] || a.severity)),
      el("td", { class: "mono" }, a.technique_id),
      el("td", {}, a.symbol),
      el("td", { class: "mono" }, fmtUs(a.ts_event_us)),
      el("td", { class: "num mono" }, fmtP(a.fused_p)),
      el("td", { class: "num" }, `${a.agreement}/3`),
      el("td", {}, STATUS_TH[a.status] || a.status),
    );
    const go = () => {
      location.hash = alertHref(a.alert_id);
    };
    row.addEventListener("click", go);
    row.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") go();
    });
    body.appendChild(row);
  }
}

// newest eight alerts beside the chart; a row that was not there before slides in
function renderRail() {
  const box = clear($("alert-rail"));
  const items = sortedAlerts().slice(0, 8);
  $("rail-empty").hidden = items.length > 0;
  const seen = state.railIds;
  for (const a of items) {
    const fresh = seen !== null && !seen.has(a.alert_id);
    box.appendChild(
      el(
        "a",
        { class: `rail-row sev-${a.severity}${fresh ? " enter" : ""}`, href: alertHref(a.alert_id) },
        el("strong", { class: "mono" }, `${a.technique_id} · ${a.symbol}`),
        el("span", { class: "muted small mono" }, `${fmtUs(a.ts_event_us).slice(11)} · p ${fmtP(a.fused_p)} · ${a.agreement}/3 ชั้น`),
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

async function showAlert(id) {
  const box = clear($("alert-detail"));
  destroyDetailChart();
  box.appendChild(el("p", { class: "muted" }, "กำลังโหลด…"));
  let a;
  try {
    a = await api(`/api/alerts/${encodeURIComponent(id)}`);
  } catch (e) {
    clear(box).appendChild(el("p", { class: "empty" }, e.message));
    return;
  }
  const tax = await loadTaxonomy();
  if (state.route !== "alert") return;
  clear(box);
  const F = nlog10(a.fused_p);

  box.appendChild(
    el(
      "div",
      { class: "detail-head" },
      el("h1", {}, "พบความผิดปกติทางสถิติ"),
      el("span", { class: `sev sev-${a.severity}` }, `ระดับ${SEVERITY_TH[a.severity] || a.severity}`),
      el("span", { class: "badge" }, a.provenance),
    ),
  );
  box.appendChild(
    el(
      "p",
      { class: "muted" },
      `${a.symbol} · ${a.technique_id} · ${fmtUs(a.ts_event_us)} · สถานะ ${STATUS_TH[a.status] || a.status}`,
    ),
  );

  const detail = el("div", { class: "detail" });
  const main = el("div", { class: "detail-main" });
  const side = el("div", { class: "detail-side" });
  detail.appendChild(main);
  detail.appendChild(side);
  box.appendChild(detail);

  const explanation = (a.explanation && a.explanation.text_th) || a.explanation_th || "";
  const facts = (a.explanation && a.explanation.facts) || [];
  side.appendChild(
    el(
      "div",
      { class: "card" },
      el("p", { class: "explain" }, explanation),
      facts.length
        ? el(
            "ul",
            { class: "facts" },
            facts.map((f) => el("li", {}, `${f.label}: ${f.value}`)),
          )
        : null,
    ),
  );

  const chartCard = el(
    "div",
    { class: "card chart-card" },
    el("h3", {}, "ช่วงเวลาของการแจ้งเตือน (แถบสี)"),
  );
  const mini = el("div", { class: "chart-mini" });
  chartCard.appendChild(mini);
  main.appendChild(chartCard);

  side.appendChild(
    el(
      "div",
      { class: "card" },
      el("h3", {}, "สรุปตัวเลข"),
      el(
        "dl",
        { class: "kv" },
        el("dt", {}, "ค่า p รวม / ความหายากเทียบตลาดปกติ"),
        el("dd", { class: "mono" }, `${fmtP(a.fused_p)} (F = −log10 p = ${fmtNum(F, 2)})`),
        el("dt", {}, "ชั้นที่เห็นพ้อง"),
        el("dd", {}, `${a.agreement}/3 ชั้น มี p < ${AGREEMENT_P}`),
        el("dt", {}, "ช่วงเวลา"),
        el("dd", { class: "mono" }, `${fmtUs(a.window[0])} – ${fmtUs(a.window[1])}`),
        el("dt", {}, "เทคนิคของจุดสูงสุด"),
        el("dd", { class: "mono" }, a.peak_technique_id || "–"),
        el("dt", {}, "รหัส"),
        el("dd", { class: "mono" }, a.alert_id),
      ),
    ),
  );

  side.appendChild(techniqueCard(tax, a.technique_id));

  main.appendChild(
    el(
      "div",
      { class: "card" },
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
                    el("td", { class: "num mono" }, fmtNum(value, 4)),
                    el("td", { class: "num mono" }, fmtNum(contrib, 2)),
                  ),
                ),
              ),
            ),
          )
        : el("p", { class: "muted" }, "ไม่มีหลักฐานรายคุณลักษณะแนบมา"),
    ),
  );

  side.appendChild(
    el(
      "div",
      { class: "card" },
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
                el("td", { class: "num mono" }, fmtP(d.p)),
                el("td", {}, FUSION_LAYERS.includes(d.layer) && d.p < AGREEMENT_P ? "✓" : ""),
              ),
            ),
          ),
        ),
      ),
    ),
  );

  side.appendChild(feedbackCard(a));
  drawDetailChart(mini, a);
}

function techniqueCard(tax, techniqueId) {
  const card = el("div", { class: "card" }, el("h3", {}, "การ์ดเทคนิค"));
  const parent = parentTechnique(techniqueId);
  const t = tax && tax[parent];
  if (!t) {
    card.appendChild(el("p", { class: "muted" }, `ไม่พบ ${techniqueId} ใน taxonomy`));
    return card;
  }
  const sub = (t.sub_techniques || []).find((s) => s.id === techniqueId);
  append(card, [
    el("p", {}, el("strong", { class: "mono" }, parent), ` ${t.name_th || ""}`),
    el("p", { class: "small" }, t.definition_th || ""),
    sub ? el("p", { class: "small" }, el("strong", { class: "mono" }, sub.id), ` ${sub.name_th || ""}: ${sub.definition_th || ""}`) : null,
    el("p", { class: "small muted" }, t.observability_th || ""),
    el("p", {}, el("a", { href: "#/taxonomy" }, "ดู taxonomy ทั้งหมด")),
  ]);
  return card;
}

function feedbackCard(a) {
  const card = el("div", { class: "card" }, el("h3", {}, "ผลการตรวจของผู้วิเคราะห์"));
  const analyst = el("input", {
    type: "text",
    maxlength: "64",
    placeholder: "ชื่อผู้วิเคราะห์",
    value: storageGet("idsai.analyst") || (authOn() ? state.me.display_name || "" : ""),
    "aria-label": "ชื่อผู้วิเคราะห์",
  });
  const status = el("span", { class: "small" });
  const list = el("ul", { class: "small" });
  const renderList = (items) => {
    clear(list);
    for (const f of items || []) {
      list.appendChild(el("li", {}, `${fmtUs(f.ts_us)} · ${f.analyst}: ${VERDICT_TH[f.verdict] || f.verdict}`));
    }
  };
  renderList(a.feedback);
  const send = async (verdict) => {
    if (!can("analyst")) {
      status.textContent = NEED_ANALYST_TH;
      return;
    }
    const name = analyst.value.trim();
    if (!name) {
      status.textContent = "ใส่ชื่อผู้วิเคราะห์ก่อน";
      return;
    }
    storageSet("idsai.analyst", name);
    status.textContent = "กำลังบันทึก…";
    try {
      await postJson(`/api/alerts/${encodeURIComponent(a.alert_id)}/feedback`, { verdict, analyst: name });
      status.textContent = `บันทึกแล้ว: ${VERDICT_TH[verdict]}`;
      const fresh = await api(`/api/alerts/${encodeURIComponent(a.alert_id)}`);
      renderList(fresh.feedback);
    } catch (e) {
      status.textContent = e.message;
    }
  };
  const gate = can("analyst") ? {} : { disabled: true, title: NEED_ANALYST_TH };
  if (!can("analyst")) {
    analyst.disabled = true;
    analyst.title = NEED_ANALYST_TH;
  }
  append(card, [
    el(
      "div",
      { class: "feedback" },
      el("label", { class: "field" }, el("span", {}, "ผู้วิเคราะห์"), analyst),
      ["tp", "fp", "unsure"].map((v) => el("button", { type: "button", ...gate, onclick: () => send(v) }, VERDICT_TH[v])),
    ),
    status,
    list,
  ]);
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
    node.replaceWith(el("p", { class: "muted small" }, "ไม่มีข้อมูลกราฟของช่วงนี้ในหน่วยความจำของระบบ (เก็บย้อนหลัง 24 ชั่วโมงของการเล่นครั้งนี้)"));
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
  LWC.createSeriesMarkers(line, [
    { time: t, position: "aboveBar", shape: "arrowDown", color: sevColor(a.severity), text: a.technique_id },
  ]);
  c.timeScale().fitContent();
  state.detailChart = {
    remove: () => c.remove(),
    restyle: () => {
      c.applyOptions(baseChartOptions());
      line.applyOptions({ color: cssVar("--line-price") });
      shade.setData(shadeData());
    },
  };
}

// ------------------------------------------------------------------ red team

function bindRedTeam() {
  $("redteam-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    if (!redTeamAllowed(state.mode)) return;
    const status = $("rt-status");
    if (!can("analyst")) {
      status.textContent = NEED_ANALYST_TH;
      return;
    }
    const offset = Number($("rt-offset").value);
    if (!Number.isFinite(offset) || offset <= 0 || offset > 3600) {
      status.textContent = "offset ต้องอยู่ระหว่าง 1 ถึง 3600 วินาที";
      return;
    }
    const body = {
      family: $("rt-family").value,
      region: $("rt-region").value,
      offset_s: Math.round(offset),
      symbol: $("rt-symbol").value || null,
    };
    status.textContent = "กำลังฉีด…";
    $("rt-submit").disabled = true;
    const known = snapshotAlerts(state.alerts.values()); // before the click: never a detection
    try {
      const r = await postJson("/api/redteam/inject", body);
      status.textContent = "ฉีดแล้ว กำลังจับเวลา";
      addInjection(r, body, known);
    } catch (e) {
      status.textContent = e.message;
    } finally {
      $("rt-submit").disabled = !(redTeamAllowed(state.mode) && can("analyst"));
    }
  });
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
    node: null,
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
    for (const id of gone) if (run.known) run.known.delete(id);
    if (run.detected && gone.has(run.detected.alert_id)) {
      run.detected = null;
      changed = true;
    }
  }
  return changed;
}

function injectionStatus(run, nowUs) {
  // "detected" wins whenever it comes; "missed" once the eval matching window (scenario end +
  // 300 s) plus a grace has passed without a qualifying alert; it follows the clock, so a seek
  // back makes the run "waiting" again
  if (run.detected) return "detected";
  if (!isNum(nowUs) || !isNum(run.t_start_us)) return "waiting";
  if (isNum(run.t_end_us) && nowUs > run.t_end_us + MATCH_AFTER_US + MISSED_GRACE_US) return "missed";
  return nowUs < run.t_start_us ? "pending" : "waiting";
}

function checkInjections() {
  let changed = false;
  for (const run of state.injections) {
    if (run.detected) continue;
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
  for (const run of state.injections) {
    const sc = run.scenario;
    const ring = svg("circle", {
      id: "rt-ring",
      class: "rt-ring-arc",
      cx: 26,
      cy: 26,
      r: RING_R,
      fill: "none",
      "stroke-width": 4,
      "stroke-dasharray": RING_C.toFixed(2),
      "stroke-dashoffset": RING_C.toFixed(2),
      transform: "rotate(-90 26 26)",
    });
    const ringSvg = svg(
      "svg",
      { viewBox: "0 0 52 52", class: "rt-ring", role: "img", "aria-label": "เวลาที่ผ่านไปเทียบกับช่วงตรวจจับ" },
      svg("circle", { cx: 26, cy: 26, r: RING_R, fill: "none", "stroke-width": 4, class: "rt-ring-track" }),
      ring,
    );
    const banner = el("div", { class: "rt-banner", id: "rt-banner" });
    banner.hidden = true;
    const detail = el("div", { class: "small timer-detail" });
    const timer = el("div", { class: "timer rt-timer" }, ringSvg, el("div", { class: "rt-timer-text" }, banner, detail));
    const card = el(
      "div",
      { class: "card" },
      el(
        "div",
        { class: "detail-head" },
        el("h3", {}, `${run.family} · ${run.symbol}`),
        el("span", { class: "badge SEMI-SYN" }, "SEMI-SYN"),
      ),
      el(
        "dl",
        { class: "kv small" },
        el("dt", {}, "ช่วงเหตุการณ์จำลอง"),
        el("dd", { class: "mono" }, `${fmtUs(run.t_start_us)} – ${fmtUs(run.t_end_us)}`),
        el("dt", {}, "เทคนิค"),
        el("dd", { class: "mono" }, `${sc.technique_id || "–"} / ${sc.sub_technique || "–"}`),
        el("dt", {}, "พารามิเตอร์"),
        el(
          "dd",
          { class: "mono" },
          `region ${run.region} · ${fmtNum(sc.duration_s, 0)} s · impact ${fmtNum(sc.impact, 3)} · vol ×${fmtNum(sc.vol_mult, 1)}`,
        ),
      ),
      timer,
    );
    run.node = timer;
    run.ring = ring;
    run.banner = banner;
    run.detail = detail;
    run.bannerState = null;
    box.appendChild(card);
  }
  tickInjections();
}

const RING_R = 22;
const RING_C = 2 * Math.PI * RING_R;

function bannerText(status, seconds) {
  // the verdict line on a Red-Team card; empty while the run is still undecided
  const n = Math.max(0, Math.round(Number(seconds) || 0));
  if (status === "detected") return `ตรวจพบใน ${n} วินาที`;
  if (status === "missed") return `ไม่พบภายใน ${n} วินาทีหลังจบเหตุการณ์`;
  return "";
}

function tickInjections() {
  const now = eventNowUs();
  for (const run of state.injections) {
    if (!run.node || !run.ring) continue;
    const status = injectionStatus(run, now);
    const windowS = Math.max(1, (run.t_end_us - run.t_start_us) / S_US + MATCH_AFTER_US / S_US);
    const ev = isNum(now) && isNum(run.t_start_us) ? (now - run.t_start_us) / S_US : null;
    let frac = ev === null ? 0 : Math.max(0, Math.min(1, ev / windowS));
    if (status === "detected") frac = Math.max(0, Math.min(1, run.detected.eventS / windowS));
    else if (status === "missed") frac = 1;
    run.ring.setAttribute("stroke-dasharray", RING_C.toFixed(2));
    run.ring.setAttribute("stroke-dashoffset", (RING_C * (1 - frac)).toFixed(2));
    run.ring.setAttribute("class", `rt-ring-arc ${status}`);

    const seconds = status === "detected" ? run.detected.eventS : MATCH_AFTER_US / S_US;
    const verdict = status === "detected" || status === "missed";
    run.banner.hidden = !verdict;
    if (verdict) {
      run.banner.textContent = bannerText(status, seconds);
      if (run.bannerState !== status) { // class only on a state change: a tick never restarts the sweep
        run.bannerState = status;
        const hit = status === "detected";
        run.banner.className = `rt-banner ${hit ? "hit" : "miss"}`;
        if (hit && !reducedMotion()) run.banner.classList.add("sweep");
      }
    } else {
      run.bannerState = null;
      run.banner.className = "rt-banner";
    }

    clear(run.detail);
    if (status === "missed") {
      run.detail.textContent =
        "ไม่พบความผิดปกติทางสถิติภายในช่วงที่นับตามเกณฑ์ประเมิน (ถึงสิ้นสุดเหตุการณ์ + 300 s) · นับเป็นการตรวจไม่พบ";
    } else if (status === "detected") {
      const d = run.detected;
      append(run.detail, [
        `ตรวจพบความผิดปกติทางสถิติ: ${d.eventS >= 0 ? "+" : ""}${d.eventS.toFixed(0)} s (เวลาเหตุการณ์) · ${d.wallS.toFixed(1)} s (เวลาจริง) `,
        el("a", { href: alertHref(d.alert_id), class: "small" }, "ดูการแจ้งเตือน"),
      ]);
    } else {
      const wall = (Date.now() - run.wallStart) / 1000;
      const evText =
        ev === null ? "–" : ev < 0 ? `อีก ${(-ev).toFixed(0)} s จะเริ่ม` : `+${ev.toFixed(0)} s`;
      run.detail.textContent = `รอการตรวจจับ… ${evText} (เวลาเหตุการณ์) · ${wall.toFixed(0)} s (เวลาจริง)`;
    }
  }
}

// ------------------------------------------------------------------ eval

function evalEmpty(report) {
  return !report || !report.runs || !report.experiments;
}

// pass / miss / unmeasured: absent data is never a pass
function targetVerdict(report, target) {
  const none = { state: "unmeasured", value: null, ci: null, n: null };
  if (evalEmpty(report) || target.cmp === "none") return none; // the report has no latency metric
  if (target.cmp === "pair") {
    const paired = report.paired_bootstrap;
    if (!paired || Number(paired.budget) !== Number(target.budget)) return none;
    const found = (paired.rows || []).find((r) => r.baseline === target.variant);
    if (!found || !isNum(found.diff_mean) || !Array.isArray(found.ci) || !isNum(found.ci[0])) return none;
    const st = target.reference ? "reference" : found.ci[0] > target.threshold ? "pass" : "miss";
    return { state: st, value: found.diff_mean, ci: found.ci, n: found.n_events };
  }
  const variants = report.experiments[target.experiment] || {};
  const cell = (variants[target.variant] || {})[String(target.budget)];
  const value = cell && cell.mean ? cell.mean[target.metric] : null;
  if (!isNum(value)) return none;
  const ci = cell.ci && Array.isArray(cell.ci[target.metric]) ? cell.ci[target.metric] : null;
  const ok = target.cmp === ">=" ? value >= target.threshold : value <= target.threshold;
  const out = { state: ok ? "pass" : "miss", value, ci, n: cell.n_events };
  if (target.correct && isNum(cell.mean.recall_correct)) {
    const cc = cell.ci && cell.ci.recall_correct;
    out.correct = { value: cell.mean.recall_correct, ci: Array.isArray(cc) ? cc : null };
  }
  return out;
}

const TARGET_STATE_TH = { pass: "ผ่านเป้า", miss: "ไม่ถึงเป้า", unmeasured: "ยังไม่ได้วัด", reference: "อ้างอิง" };

function fmtSigned(x) {
  return isNum(x) ? `${x >= 0 ? "+" : ""}${x.toFixed(3)}` : "–";
}

function targetCard(target, verdict, index) {
  const paired = target.cmp === "pair";
  const goal =
    target.cmp === "none"
      ? `< ${target.threshold} s`
      : target.reference
        ? "ไม่มีเป้า (อ้างอิง)"
        : paired
          ? "CI ขอบล่าง > 0"
          : `${target.cmp === ">=" ? "≥" : "≤"} ${target.threshold}`;
  const none = verdict.state === "unmeasured";
  return el(
    "div",
    { class: `target ${verdict.state} rise`, style: `animation-delay: ${index * 50}ms` },
    el("span", { class: "label" }, target.label),
    el("strong", { class: `verdict ${verdict.state}` }, TARGET_STATE_TH[verdict.state]),
    el("span", { class: "mono big" }, none ? "–" : paired ? fmtSigned(verdict.value) : fmtNum(verdict.value, 3)),
    el(
      "span",
      { class: "small mono muted" },
      none ? `เป้า ${goal}` : `${verdict.ci ? ciText(verdict.value, verdict.ci) : "ไม่มี CI"} · n ${verdict.n ?? "–"} · เป้า ${goal}`,
    ),
    verdict.correct
      ? el("span", { class: "small mono muted" }, `recall_correct ${ciText(verdict.correct.value, verdict.correct.ci)}`)
      : null,
  );
}

async function showEval() {
  const box = clear($("eval-body"));
  box.appendChild(el("p", { class: "muted" }, "กำลังโหลด…"));
  let report;
  try {
    report = await api("/api/experiments");
  } catch (e) {
    clear(box).appendChild(el("p", { class: "empty" }, e.message));
    return;
  }
  clear(box);
  if (evalEmpty(report)) {
    box.appendChild(el("p", { class: "empty" }, "ยังไม่มีรายงานผลการประเมิน · รัน uv run idsai report แล้วเปิดหน้านี้ใหม่"));
    return;
  }
  box.appendChild(
    el(
      "p",
      { class: "small muted" },
      `ทั้งหมด ${report.runs} run · git ${(report.git_sha || []).join(", ") || "–"} · config ${(report.config_hash || []).join(", ") || "–"} · signatures ${(report.signatures_sha || []).join(", ") || "–"}`,
    ),
  );
  box.appendChild(
    el(
      "p",
      { class: "small muted" },
      "ตัวเลขหลักคือ recall ต่อเหตุการณ์ (พร้อม 95% CI) และจำนวนการเตือนผิดต่อสัญลักษณ์-วัน precision แสดงคู่กับ prevalence เสมอ ทุกแถวมีป้ายที่มา",
    ),
  );

  box.appendChild(el("h2", { class: "section-label" }, "เป้าหมาย (งบ 2 ครั้ง/สัญลักษณ์-วัน)"));
  box.appendChild(el("div", { class: "targets" }, EVAL_TARGETS.map((t, i) => targetCard(t, targetVerdict(report, t), i))));

  if (report.paired_bootstrap) box.appendChild(pairedCard(report.paired_bootstrap));
  if (report.retrain_decision) {
    box.appendChild(retrainCard("Champion / challenger (full stack)", report.retrain_decision));
    if (report.retrain_decision.ai_only) box.appendChild(retrainCard("AE alone (diagnostic)", report.retrain_decision.ai_only));
  }

  const budgets = new Set();
  for (const variants of Object.values(report.experiments)) {
    for (const cells of Object.values(variants)) Object.keys(cells).forEach((b) => budgets.add(b));
  }
  const sorted = [...budgets].sort((a, b) => Number(a) - Number(b));
  const select = el(
    "select",
    { "aria-label": "งบการแจ้งเตือน" },
    sorted.map((b) => el("option", { value: b }, `${b} ครั้ง/สัญลักษณ์-วัน`)),
  );
  const saved = state.evalBudget;
  select.value = sorted.includes(saved) ? saved : sorted.includes("2") ? "2" : sorted[0] || "";
  const tables = el("div");
  box.appendChild(el("div", { class: "toolbar" }, el("label", { class: "field" }, el("span", {}, "งบการแจ้งเตือน"), select)));
  box.appendChild(tables);
  const draw = () => {
    clear(tables);
    for (const [exp, variants] of Object.entries(report.experiments)) {
      tables.appendChild(experimentTable(exp, variants, select.value));
    }
  };
  select.addEventListener("change", () => {
    draw();
    const b = Number(select.value);
    if (Number.isFinite(b)) {
      state.evalBudget = String(b);
      savePrefs({ eval_budget: b });
    }
  });
  draw();

  const notes = [...(report.notes || []), ...((report.paired_bootstrap || {}).notes || [])];
  if (notes.length) {
    box.appendChild(
      el("details", { class: "card notes-box" }, el("summary", {}, "หมายเหตุวิธีวัด"), el("ul", { class: "small" }, notes.map((n) => el("li", {}, n)))),
    );
  }
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
  const card = el("div", { class: "card" }, el("h3", {}, exp));
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
  const card = el("div", { class: "card" }, el("h3", {}, `เทียบแบบจับคู่ (paired bootstrap) ที่งบ ${paired.budget} · ${paired.metric}`));
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
    { class: "card" },
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

// ------------------------------------------------------------------ taxonomy

async function showTaxonomy() {
  const box = $("taxonomy-body");
  const tax = await loadTaxonomy();
  clear(box);
  if (!tax) {
    box.appendChild(el("p", { class: "empty" }, "โหลด taxonomy ไม่สำเร็จ"));
    return;
  }
  for (const [id, t] of Object.entries(tax)) {
    box.appendChild(
      el(
        "div",
        { class: "tax-row" },
        el("h3", {}, el("span", { class: "mono" }, id), ` ${t.name_th || ""}`),
        el("p", { class: "small" }, t.definition_th || ""),
        el("p", { class: "small muted" }, t.observability_th || ""),
        (t.sub_techniques || []).length
          ? el(
              "ul",
              { class: "small tax-subs" },
              t.sub_techniques.map((s) => el("li", {}, el("span", { class: "mono" }, s.id), ` ${s.name_th || ""}: ${s.definition_th || ""}`)),
            )
          : null,
        t.reference ? el("p", { class: "small muted" }, `อ้างอิง: ${t.reference}`) : null,
      ),
    );
  }
}

// ------------------------------------------------------------------ router

const PAGES = {
  login: "page-login",
  activity: "page-activity",
  monitor: "page-monitor",
  alerts: "page-alerts",
  alert: "page-alert",
  redteam: "page-redteam",
  eval: "page-eval",
  taxonomy: "page-taxonomy",
  about: "page-about",
};

function parseRoute() {
  const hash = location.hash || "#/monitor";
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
  return { name: PAGES[path] ? path : "monitor" };
}

function route() {
  const r = parseRoute();
  if (r.name === "activity" && !can("admin")) {
    // a convenience redirect: GET /api/audit answers 403 for anyone else anyway
    location.hash = DEFAULT_ROUTE;
    return;
  }
  if (r.name === "login" && state.me) {
    location.hash = r.next; // already signed in (or auth is off): nothing to do here
    return;
  }
  state.route = r.name;
  document.body.classList.toggle("login-mode", r.name === "login");
  if (r.name !== "alert") destroyDetailChart();
  for (const [name, id] of Object.entries(PAGES)) $(id).hidden = name !== r.name;
  const tab = r.name === "alert" ? "alerts" : r.name;
  for (const a of document.querySelectorAll(".tabs a")) {
    if (a.dataset.route === tab) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  }
  if (r.name === "monitor") {
    initChart();
  } else if (r.name === "alert") {
    showAlert(r.id);
  } else if (r.name === "eval") {
    showEval();
  } else if (r.name === "taxonomy") {
    showTaxonomy();
  } else if (r.name === "activity") {
    showActivity();
  } else if (r.name === "login") {
    $("login-user").focus();
  }
}

// ------------------------------------------------------------------ sign-in, user chip

let loginFails = 0;

function showLoginError(text) {
  const box = $("login-error");
  box.hidden = !text;
  box.textContent = text || "";
}

function renderUserChip() {
  const chip = $("user-chip");
  const on = authOn();
  chip.hidden = !on;
  $("theme-toggle").hidden = on; // with an account the theme lives in the user menu
  if (!on) return;
  $("user-name").textContent = state.me.display_name || state.me.username;
  $("user-role").textContent = String(state.me.role || "").toUpperCase();
}

// the sign-in answer or /api/me: apply the account, its saved choices, and the role gates
function adoptMe(me) {
  setMe(me);
  const prefs = loadedPrefs();
  if (prefs.theme && THEMES.includes(prefs.theme)) applyTheme(prefs.theme);
  if (isNum(prefs.eval_budget)) state.evalBudget = String(prefs.eval_budget);
  renderUserChip();
  applyRoleGates();
}

async function loadMe() {
  try {
    adoptMe(await api("/api/me"));
    return true;
  } catch (e) {
    setMe(null);
    return false;
  }
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
      const next = parseRoute().next || DEFAULT_ROUTE;
      adoptMe({ ...me, auth_enabled: true });
      location.hash = next;
      await enterApp();
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
  ws.addEventListener("close", () => {
    setLive(false);
    if (state.ws === ws) scheduleReconnect();
  });
  ws.addEventListener("error", () => {
    /* close follows */
  });
}

function scheduleReconnect() {
  const delay = Math.min(1000 * 2 ** state.wsRetry, 15000);
  state.wsRetry += 1;
  wsBanner(`การเชื่อมต่อสดหลุด กำลังเชื่อมต่อใหม่ในอีก ${Math.round(delay / 1000)} วินาที (ข้อมูลบนจออาจไม่เป็นปัจจุบัน)`);
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
  $("kpi-symbol").textContent = state.symbol || "–";
}

// the part of boot that needs a signed-in user: symbols, health, series, alerts, websocket
async function enterApp() {
  if (state.started) {
    // signed in again mid-session: the same page, fresh data
    applyRoleGates();
    route();
    loadAlerts();
    loadSeries();
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
    $("auditor-text").textContent = "อ่านสถานะไม่ได้";
  }
  initChart();
  await Promise.all([loadSeries(), loadAlerts()]);
  connect();
  setInterval(safely(tickInjections), 1000);
}

async function boot() {
  window.addEventListener("hashchange", safely(route));
  bindTheme();
  bindLogin();
  bindActivity();
  document.addEventListener("keydown", safely(onPresenterKey));
  bindReplay();
  bindMonitor();
  bindRedTeam();
  $("symbol-select").addEventListener("change", (ev) => {
    state.symbol = ev.target.value;
    $("rt-symbol").value = state.symbol;
    $("kpi-symbol").textContent = state.symbol || "–";
    savePrefs({ symbol: state.symbol });
    state.alertCount = 0;
    renderAlertsKpi();
    loadSeries();
  });
  const media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
  if (media && media.addEventListener) media.addEventListener("change", safely(restyleCharts));
  if (!LWC) console.warn("dashboard: the chart library did not load");

  if (!(await loadMe())) {
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
    injectionStatus,
    bannerText,
    redTeamAllowed,
    reducedMotion,
    EVAL_TARGETS,
    targetVerdict,
    evalEmpty,
    forestPlot,
    tweenNumber,
    flash,
    presenterKey,
    playToggleAction,
    noteReplayAction,
    getReplaying: () => state.replaying,
    PAGE_ORDER,
    safeNext,
    can,
    setMe,
    savePrefs,
  };
}
