/* Market IDS: HUD motion on top of motion.js. Decoration only: it extends globalThis.Motion with
   scramble(node, text) and sets up the Home panels' mouse tilt, the border glint, the circuit wires and
   the one-shot title glitch. It reads nothing from the server and never changes a number or a word,
   except that scramble ends on exactly the text it was given. It shares motion.js's loop (addLayer) and
   its off state (the fx-off class on <html>, which also covers reduced motion). Every exported method
   wraps its body in try/catch. It never builds markup from strings. Without motion.js it does nothing,
   and app.js falls back to plain textContent. */
(function () {
  "use strict";

  var G = globalThis;
  var DOC = G.document;
  var M = G.Motion;
  if (!DOC || !M || typeof M.addLayer !== "function") return; // node (tests) or motion.js missing

  var SVGNS = "http://www.w3.org/2000/svg";
  var SCRAMBLE_MS = 400;
  var SCRAMBLE_STEP_MS = 40; // glyphs change at ~25 Hz, not every frame
  var SCRAMBLE_GAP_MS = 3000; // a node that just changed (scramble or snap) snaps instead: a number that changes every second must not flicker
  var GLYPHS = "0123456789#%&@*+";
  var TILT_SMALL = 6; // degrees
  var TILT_CHART = 2.5;
  var TILT_BIG_W = 700; // a panel this wide or this tall tilts like the chart
  var TILT_BIG_H = 520;
  var TILT_SEL = "main .hud:not(.hud-sm):not(.toast)";
  var GLITCH_GAP_MS = 2000; // the same floor as the alert burst
  var GLITCH_MS = 800;
  var WIRE_PAIRS = [
    ["hero", "chart-card", 0, 0.16],
    ["card-alerts", "chart-card", 1, 0.4],
    ["card-symbol", "chart-card", 2, 0.63],
    ["card-health", "chart-card", 3, 0.86],
    ["chart-card", "rail", 0, 0.26],
    ["chart-card", "replay-bar", 1, 0.78],
  ];
  var PLEX_MIN = 40; // dots on a wide screen, scaled by area
  var PLEX_MAX = 70;
  var PLEX_BEND = 120; // px: lines bend toward the pointer inside this radius
  var PLEX_LINK = 130; // px: neighbours closer than this are joined
  var PLEX_SMALL_W = 640; // below this width half the dots are drawn
  var PLEX_ALPHA = 0.21; // the mockup's .38 line alpha at its .55 canvas opacity
  var REVEAL_MS = 900;
  var WIRE_IDS = ["hero", "card-alerts", "card-symbol", "card-health", "chart-card", "rail", "replay-bar"];

  function reduced() {
    try {
      return !!(typeof G.matchMedia === "function" && G.matchMedia("(prefers-reduced-motion: reduce)").matches);
    } catch (e) {
      return false;
    }
  }

  // motion.js sets fx-off on <html> for the user's choice, the tour and reduced motion alike
  function isOff() {
    var root = DOC.documentElement;
    return reduced() || !!(root && root.classList.contains("fx-off"));
  }

  function now() {
    return G.performance && typeof G.performance.now === "function" ? G.performance.now() : Date.now();
  }

  // ---------------------------------------------------------------- number scramble

  var jobs = [];
  var layerOn = false;

  function settle(job) {
    var i = jobs.indexOf(job);
    if (i >= 0) jobs.splice(i, 1);
    if (job.node) {
      // someone else wrote the node meanwhile (not our last frame): leave their text alone
      var mine = job.last === null ? job.start : job.last; // what the node holds if nobody else wrote it
      if (!job.abandoned && job.node.textContent === mine) job.node.textContent = job.text;
      job.node._hudScramble = null;
      job.node.removeAttribute("aria-label");
    }
    if (!jobs.length && layerOn) {
      layerOn = false;
      M.removeLayer(drawScramble);
    }
  }

  function abandon(job) {
    job.abandoned = true; // settle clears the job but leaves the outside text alone
    settle(job);
  }

  function settleAll() {
    var all = jobs.slice();
    for (var i = 0; i < all.length; i += 1) settle(all[i]);
  }

  // only digits flicker, so a Thai word or a unit next to the number never changes
  function frame(job, p) {
    var text = job.text;
    var n = text.length;
    var out = "";
    for (var i = 0; i < n; i += 1) {
      var ch = text.charAt(i);
      if (ch < "0" || ch > "9" || p > ((i + 1) / (n + 1)) * 0.85 + 0.1) out += ch;
      else out += GLYPHS.charAt(Math.floor(Math.random() * GLYPHS.length));
    }
    return out;
  }

  function drawScramble(ctx, t) {
    var all = jobs.slice();
    for (var i = 0; i < all.length; i += 1) {
      var job = all[i];
      if (!job.t0) job.t0 = t;
      var p = (t - job.t0) / SCRAMBLE_MS;
      if (p >= 1) {
        settle(job);
        continue;
      }
      if (t - job.upd < SCRAMBLE_STEP_MS) continue;
      if (job.last !== null && job.node.textContent !== job.last) {
        abandon(job); // the text was changed from outside: stop writing over it
        continue;
      }
      job.upd = t;
      job.last = frame(job, p);
      job.node.textContent = job.last;
    }
  }

  function scramble(node, text) {
    var to = String(text);
    if (!node) return;
    if (node._hudScramble) settle(node._hudScramble);
    var t = now();
    var last = node._hudScrambleAt || -1e9;
    if (node.textContent === to) return;
    if (isOff() || DOC.hidden || t - last < SCRAMBLE_GAP_MS || !/[0-9]/.test(to)) {
      node.textContent = to;
      node._hudScrambleAt = t; // a snap counts too, so a 1 Hz value never starts flickering
      return;
    }
    node._hudScrambleAt = t;
    var job = { node: node, text: to, t0: 0, upd: -1e9, last: null, start: node.textContent, abandoned: false };
    node._hudScramble = job;
    node.setAttribute("aria-label", to); // assistive tech reads the final value, never the scrambled glyphs
    jobs.push(job);
    if (!layerOn) {
      layerOn = true;
      M.addLayer(drawScramble);
    }
    // the loop pauses in a hidden tab and stops when effects go off: the number must still land
    G.setTimeout(function () {
      if (node._hudScramble === job) settle(job);
    }, SCRAMBLE_MS + 250);
  }

  // ---------------------------------------------------------------- panel tilt

  var fine = null;
  var tiltNode = null;
  var tiltBox = null;
  var tiltMax = TILT_SMALL;

  function pointerFine() {
    try {
      if (!fine && typeof G.matchMedia === "function") fine = G.matchMedia("(pointer: fine)");
      return !!(fine && fine.matches);
    } catch (e) {
      return false;
    }
  }

  function tiltReset() {
    if (tiltNode) tiltNode.style.removeProperty("transform");
    tiltNode = null;
    tiltBox = null;
  }

  function selecting() {
    try {
      var sel = G.getSelection && G.getSelection();
      return !!(sel && !sel.isCollapsed && String(sel).length);
    } catch (e) {
      return false;
    }
  }

  function onPointerMove(e) {
    try {
      if (!pointerFine() || isOff() || (e.pointerType && e.pointerType !== "mouse" && e.pointerType !== "pen")) {
        if (tiltNode) tiltReset();
        return;
      }
      // a held button (a drag, a text selection, the chart pan) and a selection keep the panel still
      if (e.buttons || selecting()) {
        if (tiltNode) tiltReset();
        return;
      }
      var target = e.target && e.target.closest ? e.target.closest(TILT_SEL) : null;
      if (target !== tiltNode) {
        tiltReset();
        if (!target) return;
        tiltNode = target;
        tiltBox = target.getBoundingClientRect(); // read once, before it tilts: the rect then holds no tilt
        var big = tiltBox.width > TILT_BIG_W || tiltBox.height > TILT_BIG_H || target.id === "chart-card";
        tiltMax = big ? TILT_CHART : TILT_SMALL;
      }
      var b = tiltBox;
      if (!b || !b.width || !b.height) return;
      var x = (e.clientX - b.left) / b.width - 0.5;
      var y = (e.clientY - b.top) / b.height - 0.5;
      x = Math.max(-0.5, Math.min(0.5, x));
      y = Math.max(-0.5, Math.min(0.5, y));
      tiltNode.style.transform =
        "perspective(1100px) rotateX(" + (-y * tiltMax * 2).toFixed(2) + "deg) rotateY(" + (x * tiltMax * 2).toFixed(2) + "deg)";
    } catch (err) {
      /* decoration only */
    }
  }

  function tiltInit() {
    DOC.addEventListener("pointermove", onPointerMove, { passive: true });
    DOC.addEventListener("pointerdown", tiltReset, { passive: true });
    DOC.addEventListener("dragstart", tiltReset, { passive: true });
    DOC.addEventListener("selectstart", tiltReset, { passive: true });
    DOC.documentElement.addEventListener("mouseleave", tiltReset, { passive: true });
    G.addEventListener("blur", tiltReset);
    G.addEventListener("scroll", function () {
      tiltBox = null; // the panel moved under the pointer: measure again on the next move
      if (tiltNode) tiltReset();
    }, { passive: true });
    G.addEventListener("resize", tiltReset);
  }

  // ---------------------------------------------------------------- border glint

  // a thin bar turns behind the panel's fill and shows along the edge; each panel is a little out of phase
  function glintInit() {
    var list = DOC.querySelectorAll(".page > .hud, #page-home .hud");
    for (var i = 0; i < list.length; i += 1) {
      var p = list[i];
      var host = p.tagName === "DETAILS" ? p.querySelector("summary") || p : p; // a closed <details> draws only its summary
      if (host.querySelector(":scope > .fx-glint")) continue;
      var g = DOC.createElement("i");
      g.className = "fx-glint";
      g.setAttribute("aria-hidden", "true");
      g.style.setProperty("--gd", (1.4 + i * 1.05).toFixed(2) + "s");
      host.insertBefore(g, host.firstChild);
    }
  }

  // ---------------------------------------------------------------- circuit wires

  var wires = null;
  var wireHost = null;
  var wireRaf = 0; // the pending layout timer

  function svgEl(tag, attrs) {
    var e = DOC.createElementNS(SVGNS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }

  function box(n) {
    var l = n.offsetLeft;
    var t = n.offsetTop;
    var w = n.offsetWidth;
    var h = n.offsetHeight;
    return { l: l, t: t, w: w, h: h, r: l + w, b: t + h };
  }

  // one wire, as a list of steps from A to B: horizontal in the gutter when the panels sit side by side,
  // straight down when one is above the other. Returns null when they overlap.
  function route(A, B, aChart, bChart, lane, f) {
    var yA = aChart ? A.t + A.h * f : A.t + Math.min(46, A.h / 2);
    var yB = bChart ? B.t + B.h * f : B.t + Math.min(46, B.h / 2);
    var x1;
    var y1;
    var x2;
    var y2;
    var gx;
    if (A.r <= B.l) {
      x1 = A.r;
      x2 = B.l;
      y1 = yA;
      y2 = yB;
      gx = x1 + 7 + lane * 5;
      return { d: "M" + x1 + " " + y1 + "H" + gx + "V" + y2 + "H" + x2, len: Math.abs(gx - x1) + Math.abs(y2 - y1) + Math.abs(x2 - gx), a: [x1, y1], b: [x2, y2] };
    }
    if (B.r <= A.l) {
      x1 = A.l;
      x2 = B.r;
      y1 = yA;
      y2 = yB;
      gx = x1 - 7 - lane * 5;
      return { d: "M" + x1 + " " + y1 + "H" + gx + "V" + y2 + "H" + x2, len: Math.abs(gx - x1) + Math.abs(y2 - y1) + Math.abs(x2 - gx), a: [x1, y1], b: [x2, y2] };
    }
    if (A.b <= B.t) {
      x1 = Math.max(A.l, B.l) + 30 + lane * 10;
      x2 = x1;
      y1 = A.b;
      y2 = B.t;
      return { d: "M" + x1 + " " + y1 + "V" + y2, len: y2 - y1, a: [x1, y1], b: [x2, y2] };
    }
    return null;
  }

  function layoutWires() {
    wireRaf = 0;
    if (!wires || !wireHost) return;
    while (wires.firstChild) wires.removeChild(wires.firstChild);
    if (isOff()) return; // nothing is drawn while effects are off
    if (!wireHost.offsetWidth) return; // another page is showing
    var panels = {};
    var list = [];
    var i;
    for (i = 0; i < WIRE_IDS.length; i += 1) {
      var n = DOC.getElementById(WIRE_IDS[i]);
      if (n && n.offsetWidth && n.offsetHeight) {
        panels[WIRE_IDS[i]] = n;
        list.push(n);
      }
    }
    var pairs = [];
    var stacked = list.length > 1;
    for (i = 1; i < list.length; i += 1) if (Math.abs(list[i].offsetLeft - list[0].offsetLeft) > 2) stacked = false;
    if (stacked) {
      list.sort(function (a, b) {
        return a.offsetTop - b.offsetTop;
      });
      for (i = 1; i < list.length; i += 1) pairs.push([list[i - 1].id, list[i].id, i % 3, 0.5]);
    } else {
      for (i = 0; i < WIRE_PAIRS.length; i += 1) pairs.push(WIRE_PAIRS[i]);
    }
    var k = 0;
    for (i = 0; i < pairs.length; i += 1) {
      var P = pairs[i];
      if (!panels[P[0]] || !panels[P[1]]) continue;
      var r = route(box(panels[P[0]]), box(panels[P[1]]), P[0] === "chart-card", P[1] === "chart-card", P[2], P[3]);
      if (!r || r.len < 4) continue;
      wires.appendChild(svgEl("path", { class: "hud-wire", d: r.d }));
      wires.appendChild(svgEl("rect", { class: "hud-wire-node", x: r.a[0] - 2, y: r.a[1] - 2, width: 4, height: 4 }));
      wires.appendChild(svgEl("rect", { class: "hud-wire-node", x: r.b[0] - 2, y: r.b[1] - 2, width: 4, height: 4 }));
      var pulse = svgEl("path", { class: "hud-wire-pulse", d: r.d });
      pulse.style.setProperty("--len", String(Math.round(r.len)));
      pulse.style.setProperty("--dur", (r.len / 50 + 1.2).toFixed(2) + "s"); // travel at ~90 px/s, then a pause
      pulse.style.setProperty("--dl", (k * 0.9).toFixed(2) + "s");
      wires.appendChild(pulse);
      k += 1;
    }
  }

  function scheduleWires() {
    if (wireRaf) return;
    wireRaf = G.setTimeout(layoutWires, 60); // coalesces a burst of size changes; a timer also runs in a hidden tab
  }

  function wiresInit() {
    wireHost = DOC.getElementById("page-home");
    wires = DOC.getElementById("home-wires");
    if (!wireHost || !wires) return;
    if (typeof G.ResizeObserver === "function") {
      var ro = new G.ResizeObserver(scheduleWires); // a layout change, never a frame
      ro.observe(wireHost);
      for (var i = 0; i < WIRE_IDS.length; i += 1) {
        var n = DOC.getElementById(WIRE_IDS[i]);
        if (n) ro.observe(n);
      }
    } else {
      G.addEventListener("resize", scheduleWires);
    }
    scheduleWires();
  }

  // ---------------------------------------------------------------- hero glitch + off state

  var lastMood = "calm";
  var glitchAt = -1e9;
  var glitchTimer = 0;

  function glitchEnd() {
    var h = DOC.getElementById("hero-title");
    if (h) h.classList.remove("hud-glitch");
  }

  function glitch() {
    var h = DOC.getElementById("hero-title");
    var t = now();
    if (!h || isOff() || t - glitchAt < GLITCH_GAP_MS) return;
    glitchAt = t;
    h.classList.add("hud-glitch");
    G.clearTimeout(glitchTimer);
    glitchTimer = G.setTimeout(glitchEnd, GLITCH_MS);
  }

  function onRoot() {
    try {
      var root = DOC.documentElement;
      if (isOff()) {
        settleAll();
        typeEnd();
        if (revealAnim) revealAnim.cancel(); // effects off: the chart shows at once
        tiltReset();
        glitchEnd();
      } else if (wires && !wires.firstChild) scheduleWires(); // effects came back: draw the wires again
      clockSync();
      var mood = root.getAttribute("data-mood") || "calm";
      if (mood !== lastMood) {
        var wasAlert = lastMood === "alert";
        lastMood = mood;
        if (mood === "alert" && !wasAlert) glitch(); // once per move into alert, not on every pass that repeats it
      }
    } catch (e) {
      /* decoration only */
    }
  }

  // ---------------------------------------------------------------- plexus background (#fx-bg)

  var plex = { pts: [], w: 0, h: 0, mx: -9999, my: -9999, on: false, ctx: null, n: 0 };

  function plexCount(w, h) {
    var n = Math.max(PLEX_MIN, Math.min(PLEX_MAX, Math.round((w * h) / 20000)));
    return w < PLEX_SMALL_W ? Math.round(n / 2) : n;
  }

  // flat, single colour (the eased mood colour); no gradients, no layout reads
  function drawPlexus(ctx, t, dt) {
    var bg = M.bg();
    var w = bg.w;
    var h = bg.h;
    var pts = plex.pts;
    if (w !== plex.w || h !== plex.h) {
      plex.w = w;
      plex.h = h;
      var n = plexCount(w, h);
      while (pts.length < n) {
        pts.push({ x: Math.random() * w, y: Math.random() * h, vx: (Math.random() - 0.5) * 24, vy: (Math.random() - 0.5) * 24 });
      }
      pts.length = n;
    }
    ctx.clearRect(0, 0, w, h);
    var i;
    for (i = 0; i < pts.length; i += 1) {
      var p = pts[i];
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      if (p.x < 0 || p.x > w) {
        p.vx = -p.vx;
        p.x = Math.max(0, Math.min(w, p.x));
      }
      if (p.y < 0 || p.y > h) {
        p.vy = -p.vy;
        p.y = Math.max(0, Math.min(h, p.y));
      }
    }
    var col = Math.round(bg.rgb[0]) + "," + Math.round(bg.rgb[1]) + "," + Math.round(bg.rgb[2]);
    var near = plex.mx > -9000;
    var d2max = PLEX_LINK * PLEX_LINK;
    ctx.strokeStyle = "rgb(" + col + ")";
    ctx.lineWidth = 0.7;
    for (i = 0; i < pts.length; i += 1) {
      var a = pts[i];
      for (var j = i + 1; j < pts.length; j += 1) {
        var b = pts[j];
        var dx = a.x - b.x;
        var dy = a.y - b.y;
        var d2 = dx * dx + dy * dy;
        if (d2 > d2max) continue;
        var mx = (a.x + b.x) / 2;
        var my = (a.y + b.y) / 2;
        var alpha = (1 - Math.sqrt(d2) / PLEX_LINK) * PLEX_ALPHA * 1.8;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        var bent = false;
        if (near) {
          var qx = plex.mx - mx;
          var qy = plex.my - my;
          var md = Math.sqrt(qx * qx + qy * qy);
          if (md < PLEX_BEND) {
            var k = (1 - md / PLEX_BEND) * 0.6;
            alpha = Math.min(0.4, alpha + 0.12);
            ctx.quadraticCurveTo(mx + qx * k, my + qy * k, b.x, b.y);
            bent = true;
          }
        }
        if (!bent) ctx.lineTo(b.x, b.y);
        ctx.globalAlpha = alpha;
        ctx.stroke();
      }
    }
    ctx.fillStyle = "rgb(" + col + ")";
    for (i = 0; i < pts.length; i += 1) {
      var q = pts[i];
      var ex = plex.mx - q.x;
      var ey = plex.my - q.y;
      ctx.globalAlpha = near && ex * ex + ey * ey < PLEX_BEND * PLEX_BEND ? 0.9 : 0.4;
      ctx.fillRect(q.x - 1, q.y - 1, 2, 2);
    }
    ctx.globalAlpha = 1;
  }

  function plexInit() {
    var canvas = DOC.getElementById("fx-bg");
    plex.ctx = canvas && typeof canvas.getContext === "function" ? canvas.getContext("2d") : null;
    if (!plex.ctx) return;
    M.addLayer(drawPlexus, { ctx: plex.ctx });
    G.addEventListener(
      "pointermove",
      function (e) {
        if (!pointerFine() || (e.pointerType && e.pointerType !== "mouse" && e.pointerType !== "pen")) return;
        plex.mx = e.clientX;
        plex.my = e.clientY;
      },
      { passive: true },
    );
    // the pointer left the window: pointerleave never fires on document, mouseleave on <html> does
    function plexLeave() {
      plex.mx = plex.my = -9999;
    }
    DOC.documentElement.addEventListener("mouseleave", plexLeave, { passive: true });
    G.addEventListener("blur", plexLeave);
  }

  // ---------------------------------------------------------------- chart draw-in (first full load only)

  var revealed = false;
  var revealAnim = null;

  // the lines draw themselves left to right: the chart is clipped, and the clip recedes
  function chartReveal() {
    var host = DOC.getElementById("chart");
    if (revealed || !host || !host.offsetWidth) return; // not on screen yet: the first visible load plays it
    revealed = true;
    if (isOff() || typeof host.animate !== "function") return;
    var anim = host.animate(
      [{ clipPath: "inset(0 100% 0 0)" }, { clipPath: "inset(0 0 0 0)" }],
      { duration: REVEAL_MS, easing: "cubic-bezier(.3,.7,.2,1)" },
    );
    anim.onfinish = anim.oncancel = function () {
      revealAnim = null;
    };
    revealAnim = anim;
  }

  // ---------------------------------------------------------------- login tagline, typed

  var TYPE_STEP_MS = 38;
  var typeTimer = 0;
  var typeFull = null; // the spans' real text, kept while typing

  function typeEnd() {
    var tag = DOC.getElementById("login-tagline");
    if (typeTimer) {
      G.clearInterval(typeTimer);
      typeTimer = 0;
    }
    if (!tag || !typeFull) return;
    var spans = tag.querySelectorAll(".tg");
    for (var i = 0; i < spans.length && i < typeFull.length; i++) spans[i].textContent = typeFull[i];
    tag.classList.remove("is-typing");
    tag.style.minHeight = "";
    typeFull = null;
  }

  // the words never change: the same text appears one letter at a time, and anything that stops it shows it all
  function typeTagline() {
    var tag = DOC.getElementById("login-tagline");
    var page = DOC.getElementById("page-login");
    typeEnd();
    if (!tag || !page || page.hidden || isOff()) return;
    var spans = tag.querySelectorAll(".tg"); // the sr-only copy holds the full text for assistive tech; these are aria-hidden
    if (!spans.length) return;
    typeFull = [];
    for (var i = 0; i < spans.length; i++) typeFull.push(spans[i].textContent);
    tag.style.minHeight = tag.offsetHeight + "px"; // the line count does not jump while it types
    tag.classList.add("is-typing");
    for (var j = 0; j < spans.length; j++) spans[j].textContent = "";
    var si = 0;
    var ci = 0;
    typeTimer = G.setInterval(function () {
      try {
        if (isOff() || page.hidden) return typeEnd();
        ci++;
        spans[si].textContent = typeFull[si].slice(0, ci);
        if (ci >= typeFull[si].length) {
          si++;
          ci = 0;
          if (si >= spans.length) typeEnd();
        }
      } catch (e) {
        typeEnd();
      }
    }, TYPE_STEP_MS);
  }

  function typeInit() {
    var page = DOC.getElementById("page-login");
    if (!page || typeof G.MutationObserver !== "function") return;
    new G.MutationObserver(function () {
      if (page.hidden) typeEnd();
      else if (!typeTimer && !typeFull) typeTagline();
    }).observe(page, { attributes: true, attributeFilter: ["hidden"] });
    if (!page.hidden) typeTagline();
  }

  // ---------------------------------------------------------------- live clock (.hud-clock)

  var clockFn = null; // the app's provider: returns the text to show (its own event/replay time)
  var clockTimer = 0;

  // one shared timer, once a second, writing the same text to every clock; a hidden tab or effects off do nothing
  function clockTick(force) {
    if (!clockFn || (!force && (DOC.hidden || isOff()))) return;
    var text = String(clockFn());
    var list = DOC.querySelectorAll(".hud-clock");
    for (var i = 0; i < list.length; i += 1) if (list[i].textContent !== text) list[i].textContent = text;
  }

  function clockSync() {
    if (clockFn && !isOff()) {
      if (!clockTimer) clockTimer = G.setInterval(guardTick, 1000);
      guardTick();
    } else if (clockTimer) {
      G.clearInterval(clockTimer); // off: the clock stays as it was, static
      clockTimer = 0;
    }
  }

  function guardTick() {
    try {
      clockTick(false);
    } catch (e) {
      /* decoration only */
    }
  }

  // ---------------------------------------------------------------- click ripple (buttons and chips)

  var RIPPLE_MS = 650;

  function ripple(e) {
    if (isOff()) return;
    var t = e.target && e.target.closest ? e.target.closest("button, .chip") : null;
    if (!t || t.disabled || t.getAttribute("aria-disabled") === "true") return;
    var r = t.getBoundingClientRect();
    if (!r.width || !r.height) return;
    var old = t.querySelector(":scope > .fx-ripple");
    if (old) old.parentNode.removeChild(old);
    var x = e.detail === 0 ? r.width / 2 : e.clientX - r.left; // a keyboard press has no pointer: from the centre
    var y = e.detail === 0 ? r.height / 2 : e.clientY - r.top;
    var far = Math.sqrt(Math.max(x, r.width - x) * Math.max(x, r.width - x) + Math.max(y, r.height - y) * Math.max(y, r.height - y));
    var wrap = DOC.createElement("span");
    wrap.className = "fx-ripple";
    wrap.setAttribute("aria-hidden", "true");
    var dot = DOC.createElement("i");
    dot.style.left = x + "px";
    dot.style.top = y + "px";
    dot.style.setProperty("--rs", ((far * 2) / 10 + 1).toFixed(1));
    wrap.appendChild(dot);
    var moved = false;
    if (G.getComputedStyle(t).position === "static") {
      t.style.position = "relative"; // the layer is positioned against the button
      moved = true;
    }
    t.appendChild(wrap);
    var done = false;
    function end() {
      if (done) return;
      done = true;
      if (wrap.parentNode) wrap.parentNode.removeChild(wrap);
      if (moved && !t.querySelector(":scope > .fx-ripple")) t.style.removeProperty("position");
    }
    dot.addEventListener("animationend", end);
    G.setTimeout(end, RIPPLE_MS + 150);
  }

  function rippleInit() {
    DOC.addEventListener(
      "click",
      function (e) {
        try {
          ripple(e);
        } catch (err) {
          /* decoration only */
        }
      },
      { passive: true },
    );
  }

  // ---------------------------------------------------------------- public

  M.chartReveal = function () {
    try {
      chartReveal();
    } catch (e) {
      /* decoration only */
    }
  };

  // the app hands over one function that returns the clock text; hud.js ticks it once a second
  M.clockSource = function (fn) {
    try {
      clockFn = typeof fn === "function" ? fn : null;
      clockTick(true); // show it now, even while effects are off: the clock is then static
      clockSync();
    } catch (e) {
      /* decoration only */
    }
  };

  M.scramble = function (node, text) {
    try {
      scramble(node, text);
    } catch (e) {
      try {
        if (node) node.textContent = String(text);
      } catch (e2) {
        /* decoration only */
      }
    }
  };

  try {
    lastMood = DOC.documentElement.getAttribute("data-mood") || "calm";
    tiltInit();
    glintInit();
    wiresInit();
    plexInit();
    typeInit();
    rippleInit();
    if (typeof G.MutationObserver === "function") {
      new G.MutationObserver(onRoot).observe(DOC.documentElement, { attributes: true, attributeFilter: ["class", "data-mood"] });
    }
  } catch (e) {
    /* decoration only */
  }
})();
