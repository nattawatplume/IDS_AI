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

    // later tasks fill these in; until then they do nothing
    loginEnter: function () {
      try {
        /* the CCTV scene */
      } catch (e) {
        /* decoration only */
      }
    },

    loginLeave: function (done) {
      try {
        if (typeof done === "function") done();
      } catch (e) {
        /* decoration only */
      }
    },

    alertBurst: function (info) {
      try {
        void info;
      } catch (e) {
        /* decoration only */
      }
    },

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
