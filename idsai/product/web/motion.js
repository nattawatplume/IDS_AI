/* Market IDS: command-center motion. Decoration only: canvas, transform and opacity. It reads nothing
   from the server, holds no app state, and never changes a number or a word. app.js talks to it through
   one guard (fx), so a missing or failing file leaves the page working as before.
   Every public method wraps its body in try/catch. It never builds markup from strings. */
(function () {
  "use strict";

  var G = globalThis;
  var DOC = G.document;
  if (!DOC) return; // node (tests): nothing to draw on

  var FRAME_MS = 1000 / 30; // one shared loop, at most 30 fps
  var MOODS = { calm: [34, 211, 238], watch: [245, 158, 11], alert: [239, 68, 68] }; // --fx-calm / --fx-watch / --fx-alert
  var GRID_ALPHA = 0.08; // --fx-grid
  var SMALL_W = 640; // below this width half the dots are drawn

  var S = {
    off: true, // the effective state: the request or reduced motion
    want: false, // what the app last asked for
    mood: "calm",
    rgb: MOODS.calm.slice(),
    layers: [],
    login: null,
    raf: 0,
    last: 0,
    canvas: null,
    ctx: null,
    w: 0,
    h: 0,
    dots: [],
    scanAt: 0,
    scanFrom: -1,
    resizeTimer: 0,
    started: false,
  };

  function reduced() {
    try {
      return !!(typeof G.matchMedia === "function" && G.matchMedia("(prefers-reduced-motion: reduce)").matches);
    } catch (e) {
      return false;
    }
  }

  // ---------------------------------------------------------------- shared loop

  // draw(ctx, t, dt): t in ms (the rAF clock), dt in seconds. Each layer clears and draws its own canvas.
  // opts.ctx: the context the layer draws on (so it can be cleared when effects go off)
  function addLayer(draw, opts) {
    if (typeof draw !== "function") return draw;
    S.layers.push({ draw: draw, ctx: (opts && opts.ctx) || S.ctx });
    start();
    return draw;
  }

  function removeLayer(draw) {
    for (var i = S.layers.length - 1; i >= 0; i -= 1) {
      if (S.layers[i].draw === draw) S.layers.splice(i, 1);
    }
  }

  function tick(now) {
    S.raf = 0;
    if (S.off || DOC.hidden || !S.layers.length) return;
    S.raf = G.requestAnimationFrame(tick);
    if (now - S.last < FRAME_MS - 1) return;
    var dt = Math.min((now - S.last) / 1000, 0.1);
    S.last = now;
    var layers = S.layers.slice();
    for (var i = 0; i < layers.length; i += 1) {
      try {
        layers[i].draw(layers[i].ctx, now, dt);
      } catch (e) {
        removeLayer(layers[i].draw); // a broken layer must not take the others down
      }
    }
  }

  function start() {
    if (S.raf || S.off || DOC.hidden || !S.layers.length || typeof G.requestAnimationFrame !== "function") return;
    S.last = 0;
    S.raf = G.requestAnimationFrame(tick);
  }

  function stop() {
    if (S.raf && typeof G.cancelAnimationFrame === "function") G.cancelAnimationFrame(S.raf);
    S.raf = 0;
  }

  function clearAll() {
    for (var i = 0; i < S.layers.length; i += 1) {
      var ctx = S.layers[i].ctx;
      if (!ctx || !ctx.canvas) continue;
      ctx.save();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      ctx.restore();
    }
  }

  // ---------------------------------------------------------------- on / off

  function apply(off) {
    S.off = !!off;
    var root = DOC.documentElement;
    if (root) root.classList.toggle("fx-off", S.off);
    if (S.canvas) S.canvas.hidden = S.off || !S.ctx;
    if (S.off) {
      stop();
      clearAll();
    } else {
      size();
      start();
    }
  }

  function effective() {
    return S.want || reduced() || !S.ctx;
  }

  // ---------------------------------------------------------------- background canvas

  function dotCount() {
    var n = Math.max(30, Math.min(60, Math.round((S.w * S.h) / 32000)));
    return S.w < SMALL_W ? Math.round(n / 2) : n;
  }

  function seedDots() {
    var n = dotCount();
    while (S.dots.length > n) S.dots.pop();
    while (S.dots.length < n) {
      S.dots.push({ x: Math.random() * S.w, y: Math.random() * S.h, v: 8 + Math.random() * 22, r: 0.8 + Math.random() * 1.4, a: 0.25 + Math.random() * 0.45 });
    }
  }

  // sizes happen on a debounced resize, never inside the loop
  function size() {
    var c = S.canvas;
    if (!c || !S.ctx) return;
    var dpr = Math.min(G.devicePixelRatio || 1, 2);
    S.w = G.innerWidth || 1;
    S.h = G.innerHeight || 1;
    c.width = Math.round(S.w * dpr);
    c.height = Math.round(S.h * dpr);
    S.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    seedDots();
  }

  function onResize() {
    if (S.resizeTimer) G.clearTimeout(S.resizeTimer);
    S.resizeTimer = G.setTimeout(function () {
      S.resizeTimer = 0;
      if (!S.off) size();
    }, 150);
  }

  function rgba(a) {
    return "rgba(" + Math.round(S.rgb[0]) + "," + Math.round(S.rgb[1]) + "," + Math.round(S.rgb[2]) + "," + a + ")";
  }

  function drawBackground(ctx, t, dt) {
    var w = S.w;
    var h = S.h;
    // the colour eases toward the mood
    var goal = MOODS[S.mood];
    var k = 1 - Math.exp(-dt * 3);
    for (var c = 0; c < 3; c += 1) S.rgb[c] += (goal[c] - S.rgb[c]) * k;

    ctx.clearRect(0, 0, w, h);

    // perspective grid: rows slide toward the viewer, columns meet at the horizon
    var hy = h * 0.42;
    var rows = 11;
    var phase = (t / 7000) % 1;
    ctx.lineWidth = 1;
    ctx.strokeStyle = rgba(GRID_ALPHA);
    for (var i = 0; i < rows; i += 1) {
      var z = (i + phase) / rows;
      var y = hy + (h - hy) * z * z;
      ctx.globalAlpha = Math.min(1, 0.25 + z * 1.2);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    var fade = ctx.createLinearGradient(0, hy, 0, h);
    fade.addColorStop(0, rgba(0));
    fade.addColorStop(1, rgba(GRID_ALPHA * 1.5));
    ctx.strokeStyle = fade;
    var cols = 16;
    var span = Math.max(w * 0.18, 90);
    var drift = Math.sin(t / 9000) * span * 0.25;
    ctx.beginPath();
    for (var j = 0; j <= cols; j += 1) {
      ctx.moveTo(w / 2, hy);
      ctx.lineTo(w / 2 + (j - cols / 2) * span + drift, h);
    }
    ctx.stroke();
    ctx.strokeStyle = rgba(GRID_ALPHA * 2);
    ctx.beginPath();
    ctx.moveTo(0, hy);
    ctx.lineTo(w, hy);
    ctx.stroke();

    // rising data dots, faded at both ends of their path
    ctx.fillStyle = rgba(1);
    for (var d = 0; d < S.dots.length; d += 1) {
      var p = S.dots[d];
      p.y -= p.v * dt;
      if (p.y < -6) {
        p.y = h + 6;
        p.x = Math.random() * w;
      }
      ctx.globalAlpha = p.a * Math.max(0, Math.sin(Math.PI * (1 - p.y / h)));
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, 6.2832);
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    // one horizontal scan line every 8 to 12 seconds
    if (!S.scanAt) S.scanAt = t + 4000 + Math.random() * 4000;
    if (S.scanFrom < 0 && t >= S.scanAt) S.scanFrom = t;
    if (S.scanFrom >= 0) {
      var prog = (t - S.scanFrom) / 2800;
      if (prog >= 1) {
        S.scanFrom = -1;
        S.scanAt = t + 8000 + Math.random() * 4000;
      } else {
        var sy = prog * h;
        var band = ctx.createLinearGradient(0, sy - 70, 0, sy);
        band.addColorStop(0, rgba(0));
        band.addColorStop(1, rgba(0.12));
        ctx.fillStyle = band;
        ctx.fillRect(0, sy - 70, w, 70);
        ctx.fillStyle = rgba(0.3);
        ctx.fillRect(0, sy, w, 1);
      }
    }
  }

  // ---------------------------------------------------------------- login: the CCTV scene

  var PAN_MS = 3600; // one sweep; there and back is two (the CSS animation cctv-pan, +-18 degrees, alternate)
  var PAN_DEG = 18;
  var HIT_DEG = 2; // the beam is on the spike when it is this close, in degrees
  var LEAVE_MS = 620; // the dashboard starts loading here; the fallback below never lets it wait longer

  function num(el, name) {
    return Number(el.getAttribute(name)) || 0;
  }

  // the angles are fixed by the artwork, so they are read once on enter and again on resize, never per frame
  function aim(L) {
    var wrap = L.root.querySelector("#cctv-beam-wrap");
    var target = L.root.querySelector("#cctv-target");
    if (!wrap || !target) return;
    var x = num(wrap, "data-x");
    var y = num(wrap, "data-y");
    L.base = num(wrap, "data-a");
    L.goal = (Math.atan2(num(target, "cy") - y, num(target, "cx") - x) * 180) / Math.PI;
  }

  function onLoginResize() {
    if (S.login) aim(S.login);
  }

  // the pan angle in degrees, from the CSS animation's own clock (no layout read); null before it runs
  function panAngle(L, t) {
    var a = L.anim;
    if (!a && t >= L.retry) {
      L.retry = t + 600;
      try {
        var list = typeof L.pan.getAnimations === "function" ? L.pan.getAnimations() : [];
        a = L.anim = list.length ? list[0] : null;
      } catch (e) {
        a = L.anim = null;
      }
    }
    var ms = a && a.currentTime != null ? Number(a.currentTime) : NaN;
    if (!(ms >= 0)) {
      L.anim = null; // cancelled (effects went off and back on): look it up again
      return null;
    }
    var n = Math.floor(ms / PAN_MS);
    var p = (ms - n * PAN_MS) / PAN_MS;
    if (n % 2 === 1) p = 1 - p;
    L.cycle = Math.floor(n / 2);
    return -PAN_DEG + 2 * PAN_DEG * ((1 - Math.cos(Math.PI * p)) / 2);
  }

  function pulse(el, frames, opts) {
    if (el && typeof el.animate === "function") el.animate(frames, opts);
  }

  // the beam crosses the spike: it glows, a ring spreads from it and the tag shows for 1.5 s
  function beamHit(L) {
    if (L.spike) L.spike.classList.add("is-hit");
    if (L.hitTimer) G.clearTimeout(L.hitTimer);
    L.hitTimer = G.setTimeout(function () {
      L.hitTimer = 0;
      if (L.spike) L.spike.classList.remove("is-hit");
    }, 1500);
    for (var i = 0; i < L.rings.length; i += 1) {
      pulse(L.rings[i], [{ transform: "scale(.35)", opacity: 0.95 }, { transform: "scale(3.4)", opacity: 0 }], { duration: 950, delay: i * 200, easing: "ease-out" });
    }
    pulse(L.tag, [{ opacity: 0 }, { opacity: 1, offset: 0.14 }, { opacity: 1, offset: 0.86 }, { opacity: 0 }], { duration: 1500 });
  }

  function loginDraw(ctx, t) {
    var L = S.login;
    if (!L || L.leaving || !L.pan || L.page.hidden) return;
    var deg = panAngle(L, t);
    if (deg === null) return;
    if (Math.abs(L.base + deg - L.goal) < HIT_DEG && L.cycle !== L.hitCycle) {
      L.hitCycle = L.cycle; // at most one hit per pan cycle
      beamHit(L);
    }
  }

  function loginEnter() {
    var root = DOC.getElementById("cctv");
    if (!root) return;
    var L = S.login;
    if (!L || L.root !== root) {
      L = S.login = {
        root: root,
        page: DOC.getElementById("page-login") || root,
        pan: root.querySelector(".cctv-pan"),
        spike: DOC.getElementById("cctv-spike"),
        tag: root.querySelector("#cctv-tag"),
        rings: root.querySelectorAll(".cctv-ring"),
        base: 0,
        goal: 0,
        anim: undefined,
        retry: 0,
        cycle: 0,
        hitCycle: -1,
        hitTimer: 0,
        leaving: false,
        on: false,
      };
      G.addEventListener("resize", onLoginResize);
    }
    L.leaving = false;
    L.hitCycle = -1;
    L.anim = undefined;
    aim(L);
    if (!L.on) {
      L.on = true;
      addLayer(loginDraw); // it runs once the loop does (after init, and not while effects are off)
    }
  }

  // the "ยืนยันตัวตนแล้ว" overlay, built from nodes (never from markup strings)
  function verifyOverlay() {
    var ov = DOC.createElement("div");
    ov.id = "fx-verify";
    ov.setAttribute("aria-hidden", "true");
    var box = DOC.createElement("div");
    box.className = "fx-verify-box";
    var mark = DOC.createElement("span");
    mark.className = "fx-verify-mark";
    mark.textContent = "✓";
    var label = DOC.createElement("span");
    label.className = "fx-verify-text";
    label.textContent = "ยืนยันตัวตนแล้ว";
    box.appendChild(mark);
    box.appendChild(label);
    ov.appendChild(box);
    DOC.body.appendChild(ov);
    return { ov: ov, mark: mark, label: label };
  }

  // 600 ms of "ยืนยันตัวตนแล้ว ✓", then a zoom and fade into the dashboard. done() always runs: at once when
  // effects are off, and from a timer even if something below throws.
  function loginLeave(done) {
    var called = false;
    function fin() {
      if (called) return;
      called = true;
      try {
        if (typeof done === "function") done();
      } catch (e) {
        /* the caller's problem, never ours */
      }
    }
    try {
      var L = S.login;
      if (S.off || (L && L.leaving) || !DOC.body || typeof DOC.createElement !== "function") {
        fin();
        return;
      }
      G.setTimeout(fin, LEAVE_MS + 80); // the guarantee: well inside 900 ms whatever happens next
      if (L) {
        L.leaving = true;
        if (L.on) removeLayer(loginDraw);
        L.on = false;
      }
      var v = verifyOverlay();
      if (typeof v.ov.animate !== "function") {
        v.ov.remove();
        fin();
        return;
      }
      pulse(v.ov, [{ opacity: 0 }, { opacity: 1 }], { duration: 200, fill: "forwards", easing: "ease-out" });
      pulse(v.mark, [{ transform: "scale(.5)", opacity: 0 }, { transform: "scale(1.12)", opacity: 1, offset: 0.6 }, { transform: "scale(1)", opacity: 1 }], { duration: 420, delay: 80, easing: "cubic-bezier(.2,.8,.2,1)", fill: "backwards" });
      pulse(v.label, [{ transform: "translateY(8px)", opacity: 0 }, { transform: "none", opacity: 1 }], { duration: 320, delay: 180, easing: "ease-out", fill: "backwards" });
      G.setTimeout(function () {
        fin(); // the dashboard builds underneath while the overlay zooms away
        pulse(v.ov, [{ opacity: 1, transform: "scale(1)" }, { opacity: 0, transform: "scale(1.2)" }], { duration: 420, easing: "cubic-bezier(.4,0,.2,1)", fill: "forwards" });
        G.setTimeout(function () {
          if (v.ov.parentNode) v.ov.parentNode.removeChild(v.ov);
        }, 480);
      }, LEAVE_MS);
    } catch (e) {
      fin();
    }
  }

  // ---------------------------------------------------------------- alert burst

  var BURST_GAP_MS = 2000; // bursts are rate-limited; with the one-shot flash this keeps well under 3 flashes a second
  var SWEEP_MS = 1200;

  // restart a one-shot CSS animation class (a single layout read per burst, never in the loop)
  function replay(node, cls) {
    node.classList.remove(cls);
    void node.offsetWidth;
    node.classList.add(cls);
  }

  function alertBurst(info) {
    void info;
    var now = G.performance && typeof G.performance.now === "function" ? G.performance.now() : Date.now();
    if (S.burstAt !== undefined && now - S.burstAt < BURST_GAP_MS) return;
    S.burstAt = now;
    Motion.mood("alert"); // the next hero pass decides again
    if (S.off) return; // the colour shows, nothing moves
    var edge = DOC.getElementById("fx-edge");
    if (edge) {
      replay(edge, "is-flash"); // one iteration; it clears itself
      if (!edge.fxBound) {
        edge.fxBound = true;
        edge.addEventListener("animationend", function () {
          edge.classList.remove("is-flash");
        });
      }
    }
    // the newest card: the first of the alerts page and the first row of the home rail
    var cards = DOC.querySelectorAll("#alerts-list > .alert-card:first-child, #alert-rail > .rail-row:first-child");
    for (var i = 0; i < cards.length; i += 1) replay(cards[i], "fx-sweep");
    if (S.sweepTimer) G.clearTimeout(S.sweepTimer);
    S.sweepTimer = G.setTimeout(function () {
      S.sweepTimer = 0;
      var on = DOC.querySelectorAll(".fx-sweep");
      for (var j = 0; j < on.length; j += 1) on[j].classList.remove("fx-sweep");
    }, SWEEP_MS);
  }

  // ---------------------------------------------------------------- public API

  var Motion = {
    // opts.off: the user's "ลดเอฟเฟกต์" choice. Reduced motion also keeps everything still.
    init: function (opts) {
      try {
        S.want = !!(opts && opts.off);
        if (!S.started) {
          S.started = true;
          S.canvas = DOC.getElementById("fx-bg");
          S.ctx = S.canvas && typeof S.canvas.getContext === "function" ? S.canvas.getContext("2d") : null;
          if (S.ctx) addLayer(drawBackground, { ctx: S.ctx });
          G.addEventListener("resize", onResize);
          DOC.addEventListener("visibilitychange", function () {
            if (DOC.hidden) stop();
            else start();
          });
          var mq = typeof G.matchMedia === "function" ? G.matchMedia("(prefers-reduced-motion: reduce)") : null;
          if (mq && typeof mq.addEventListener === "function") {
            mq.addEventListener("change", function () {
              apply(effective());
            });
          }
        }
        apply(effective());
      } catch (e) {
        /* motion stays off; the page works */
      }
    },

    setOff: function (off) {
      try {
        S.want = !!off;
        apply(effective());
      } catch (e) {
        /* decoration only */
      }
    },

    // "calm" | "watch" | "alert": the colour of the background; the root attribute also tints the static page
    mood: function (m) {
      try {
        S.mood = MOODS[m] ? m : "calm";
        if (DOC.documentElement) DOC.documentElement.setAttribute("data-mood", S.mood);
      } catch (e) {
        /* decoration only */
      }
    },

    // the login page's camera and beam; safe before init (the layer waits for the loop) and safe to repeat
    loginEnter: function () {
      try {
        loginEnter();
      } catch (e) {
        /* decoration only */
      }
    },

    // done() runs within 900 ms, and at once when effects are off
    loginLeave: function (done) {
      try {
        loginLeave(done);
      } catch (e) {
        try {
          if (typeof done === "function") done();
        } catch (e2) {
          /* decoration only */
        }
      }
    },

    // a live alert: the screen edge flashes once, the newest card gets a red sweep, the mood goes red until
    // the next hero pass decides. At most one burst per BURST_GAP_MS; info.severity is not needed to draw it.
    alertBurst: function (info) {
      try {
        alertBurst(info);
      } catch (e) {
        /* decoration only */
      }
    },

    // later tasks fill these in; until then they do nothing
    chartAttach: function (api, series, opts) {
      try {
        void api;
        void series;
        void opts;
      } catch (e) {
        /* decoration only */
      }
    },

    chartCross: function (point) {
      try {
        void point;
      } catch (e) {
        /* decoration only */
      }
    },

    pageEnter: function (node) {
      try {
        void node;
      } catch (e) {
        /* decoration only */
      }
    },

    // the shared loop, for the scenes added on top of this
    addLayer: addLayer,
    removeLayer: removeLayer,
  };

  G.Motion = Motion;
})();
