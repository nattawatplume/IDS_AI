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
  var MOODS = { calm: [0, 229, 255], watch: [245, 158, 11], alert: [255, 45, 85] }; // --fx-calm / --fx-watch / --fx-alert

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

  // a one-shot cut off mid-flight never fires animationend; remove what it left behind
  function dropTransient() {
    var edge = DOC.getElementById("fx-edge");
    if (edge) edge.classList.remove("is-flash");
    var k;
    var sw = DOC.querySelectorAll(".fx-sweep");
    for (k = 0; k < sw.length; k += 1) sw[k].remove();
    var ri = DOC.querySelectorAll(".fx-rise");
    for (k = 0; k < ri.length; k += 1) {
      ri[k].classList.remove("fx-rise");
      ri[k].style.removeProperty("--fx-i");
    }
    var wp = DOC.querySelectorAll(".fx-wipe");
    for (k = 0; k < wp.length; k += 1) if (wp[k].parentNode) wp[k].parentNode.removeChild(wp[k]);
  }

  function apply(off) {
    S.off = !!off;
    var root = DOC.documentElement;
    if (root) root.classList.toggle("fx-off", S.off);
    if (S.canvas) S.canvas.hidden = S.off || !S.ctx;
    if (S.off) {
      stop();
      clearAll();
      dropTransient();
      headerOff();
    } else {
      size();
      start();
      navPlace();
    }
  }

  function effective() {
    return S.want || reduced() || !S.ctx;
  }

  // ---------------------------------------------------------------- background canvas

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
  }

  function onResize() {
    if (S.resizeTimer) G.clearTimeout(S.resizeTimer);
    S.resizeTimer = G.setTimeout(function () {
      S.resizeTimer = 0;
      if (!S.off) size();
      navPlace();
    }, 150);
  }

  // the mood colour eases here, for every layer; the plexus (hud.js) draws the canvas itself
  function drawBackground(ctx, t, dt) {
    var goal = MOODS[S.mood];
    var k = 1 - Math.exp(-dt * 3);
    for (var c = 0; c < 3; c += 1) S.rgb[c] += (goal[c] - S.rgb[c]) * k;
  }

  // ---------------------------------------------------------------- header: eye, typed name, ticker, nav underline

  var H = { iris: null, lid: null, name: null, bar: null, text: "", typing: 0, timer: 0, x: 0, tx: 0, next: 900, blink: -999, nextBlink: 3200, ready: false };
  var NAV = { a: null, placed: false };
  var TK = { sig: "", vals: [], items: null, one: 0, reps: 0 };
  var HF = { raf: 0, ro: null };
  var TYPE_AT = 300; // ms before the first letter

  function headerInit() {
    if (H.ready) return;
    H.ready = true;
    H.iris = DOC.getElementById("logo-iris");
    H.lid = DOC.getElementById("logo-lid");
    H.name = DOC.getElementById("brand-name");
    H.bar = DOC.getElementById("topbar");
    H.text = H.name ? H.name.textContent : "";
    if (H.iris) addLayer(eyeDraw);
    watchHeader();
  }

  // the nav shifts when a link appears (the admin's Users) or the strip is resized: re-place the underline and
  // refill the marquee, once per frame at most, from the observer and never from the draw loop
  function watchHeader() {
    if (typeof G.ResizeObserver !== "function") return;
    var run = function () {
      HF.raf = 0;
      try {
        navPlace();
        tickerFit();
      } catch (e) {
        /* decoration only */
      }
    };
    var RO = G.ResizeObserver;
    HF.ro = new RO(function () {
      if (!HF.raf) HF.raf = G.requestAnimationFrame(run);
    });
    var sels = [".nav", ".brand", ".tk-view"];
    for (var i = 0; i < sels.length; i += 1) {
      var n = DOC.querySelector(sels[i]);
      if (n) HF.ro.observe(n);
    }
  }

  // the name types itself out once, the first time the bar is on screen; the plain text stays in the markup
  function typeStart() {
    if (!H.name || H.typing || !H.text) return;
    H.typing = 1;
    H.name.textContent = "";
    var n = 0;
    var step = function () {
      H.timer = 0;
      n += 1;
      H.name.textContent = H.text.slice(0, n);
      if (n < H.text.length) H.timer = G.setTimeout(step, 65 + Math.random() * 70);
      else H.typing = 2;
    };
    H.timer = G.setTimeout(step, TYPE_AT);
  }

  // the iris drifts left and right and blinks; in alert mood it centres and stares (the red and the size are CSS)
  function eyeDraw(ctx, t, dt) {
    if (!H.iris || (H.bar && H.bar.hidden)) return;
    if (!H.typing) typeStart();
    var alert = S.mood === "alert";
    if (alert) H.tx = 0;
    else if (t > H.next) {
      var o = [-7, -4.5, 0, 3, 6.5];
      H.tx = o[Math.floor(Math.random() * o.length)];
      H.next = t + 900 + Math.random() * 1900;
    }
    H.x += (H.tx - H.x) * Math.min(1, dt * 7);
    H.iris.setAttribute("transform", "translate(" + (22 + H.x).toFixed(2) + " 22)");
    var sy = 1;
    if (!alert) {
      if (t > H.nextBlink) {
        H.blink = t;
        H.nextBlink = t + 3500 + Math.random() * 3500;
      }
      var b = t - H.blink;
      if (b >= 0 && b < 190) sy = 1 - Math.sin((b / 190) * Math.PI) * 0.9;
    }
    if (H.lid) H.lid.setAttribute("transform", "translate(0 22) scale(1 " + sy.toFixed(3) + ") translate(0 -22)");
  }

  // effects off: the eye rests, the name is whole, the static nav underline takes over
  function headerOff() {
    if (H.timer) G.clearTimeout(H.timer);
    H.timer = 0;
    if (H.typing === 1) {
      if (H.name) H.name.textContent = H.text;
      H.typing = 2;
    }
    H.x = 0;
    H.tx = 0;
    if (H.iris) H.iris.setAttribute("transform", "translate(22 22)");
    if (H.lid) H.lid.removeAttribute("transform");
    NAV.placed = false;
    navPlace();
  }

  // text nodes only: each item is "LABEL" + <b>value</b>; the list is repeated to fill the strip
  function tickerGroup(items, reps) {
    var g = DOC.createElement("div");
    g.className = "tk-group";
    for (var r = 0; r < reps; r += 1) {
      for (var i = 0; i < items.length; i += 1) {
        var sp = DOC.createElement("span");
        var b = DOC.createElement("b");
        b.textContent = items[i].value;
        sp.appendChild(DOC.createTextNode(items[i].label));
        sp.appendChild(b);
        if (r) sp.className = "tk-rep";
        g.appendChild(sp);
        (TK.vals[i] = TK.vals[i] || []).push(b);
      }
    }
    return g;
  }

  function ticker(items) {
    var track = DOC.getElementById("tk-track");
    if (!track || !Array.isArray(items) || !items.length) return;
    TK.items = items;
    var sig = items
      .map(function (it) {
        return it.label;
      })
      .join("|");
    var i;
    if (sig === TK.sig) {
      // same shape: only the values change, in place, so the marquee does not jump
      for (i = 0; i < items.length; i += 1) {
        for (var k = 0; k < TK.vals[i].length; k += 1) {
          if (TK.vals[i][k].textContent !== items[i].value) TK.vals[i][k].textContent = items[i].value;
        }
      }
      return;
    }
    TK.sig = sig;
    TK.vals = [];
    while (track.firstChild) track.removeChild(track.firstChild);
    track.hidden = false;
    if (track.parentNode) track.parentNode.classList.add("is-live");
    var probe = tickerGroup(items, 1);
    track.appendChild(probe);
    var one = probe.offsetWidth;
    var view = track.parentNode ? track.parentNode.clientWidth : 0;
    var reps = one > 0 && view > 0 ? Math.min(6, Math.max(1, Math.ceil(view / one))) : 3;
    TK.vals = [];
    TK.one = one;
    TK.reps = reps;
    track.removeChild(probe);
    track.appendChild(tickerGroup(items, reps));
    track.appendChild(tickerGroup(items, reps));
    track.style.setProperty("--tk-dur", Math.max(20, Math.round(((one || 600) * reps) / 45)) + "s");
  }

  // the strip got wider than the repeats (or was hidden when first built): build the repeats again
  function tickerFit() {
    var track = DOC.getElementById("tk-track");
    var view = track && track.parentNode ? track.parentNode.clientWidth : 0;
    if (!TK.items || !view) return;
    if (!TK.one || view > TK.one * TK.reps) {
      TK.sig = "";
      ticker(TK.items);
    }
  }

  // the active nav link: the underline slides to it with a transform; off or hidden, the link's own underline shows
  function navPlace() {
    var ind = DOC.querySelector(".nav-ind");
    var nav = ind && ind.parentNode;
    if (!nav) return;
    var a = NAV.a;
    if (S.off || !a || !a.offsetWidth) {
      nav.classList.remove("has-ind");
      return;
    }
    var first = !NAV.placed;
    if (first) ind.style.transition = "none";
    ind.style.width = Math.max(0, a.offsetWidth - 16) + "px";
    ind.style.transform = "translate3d(" + (a.offsetLeft + 8) + "px,0,0)";
    nav.classList.add("has-ind");
    if (first) {
      void ind.offsetWidth;
      ind.style.transition = "";
      NAV.placed = true;
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
          var Lg = S.login;
          if (Lg && Lg.leaving) {
            Lg.leaving = false; // a failed sign-in stays on the login page: the beam returns
            if (!Lg.on && !Lg.page.hidden) {
              Lg.on = true;
              addLayer(loginDraw);
            }
          }
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

  // the sweep is its own aria-hidden element inside the card (never the card's ::after, which a .hud
  // uses for its fill), and it is removed when the one pass is over
  function addSweep(card) {
    var old = card.querySelector(".fx-sweep");
    if (old) old.remove();
    var sw = DOC.createElement("span");
    sw.className = "fx-sweep";
    sw.setAttribute("aria-hidden", "true");
    card.appendChild(sw);
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
        var clear = function () {
          edge.classList.remove("is-flash");
        };
        edge.addEventListener("animationend", clear);
        edge.addEventListener("animationcancel", clear);
      }
    }
    // the newest card: the first of the alerts page and the first row of the home rail
    var cards = DOC.querySelectorAll("#alerts-list > .alert-card:first-child, #alert-rail > .rail-row:first-child");
    for (var i = 0; i < cards.length; i += 1) addSweep(cards[i]);
    if (S.sweepTimer) G.clearTimeout(S.sweepTimer);
    S.sweepTimer = G.setTimeout(function () {
      S.sweepTimer = 0;
      var on = DOC.querySelectorAll(".fx-sweep");
      for (var j = 0; j < on.length; j += 1) on[j].remove();
    }, SWEEP_MS);
  }

  // ---------------------------------------------------------------- main chart overlay

  var SHOCK_MS = 900; // two rings expand to 60 px
  var RING_GAP_MS = 140;
  var LIMIT_MS = 400; // limit line glows red, shakes 2 px
  var READ_MS = 250; // re-read the latest point this often
  var FILL_MAX = 300;

  function chartDetach() {
    var O = S.chartOv;
    S.chartOv = null;
    if (!O) return;
    removeLayer(O.draw);
    if (O.ro) O.ro.disconnect();
    try {
      O.api.timeScale().unsubscribeVisibleTimeRangeChange(O.onRange);
    } catch (e) {
      /* the chart is gone */
    }
    O.canvas.width = O.canvas.width; // clear
  }

  // pane offsets and plot width: layout reads, on resize and range change only
  function chartLayout(O) {
    try {
      var y0 = O.host.getBoundingClientRect().top;
      var panes = O.api.panes();
      O.top = [];
      O.ph = [];
      for (var i = 0; i < panes.length; i += 1) {
        O.top.push(panes[i].getHTMLElement().getBoundingClientRect().top - y0);
        O.ph.push(panes[i].getHeight());
      }
      O.pw = O.api.timeScale().width();
    } catch (e) {
      /* keep what was read before */
    }
    O.dirty = true;
  }

  function limitLineOf(O) {
    return O.getLimit ? O.getLimit() : O.limitLine;
  }

  function limitPrice(O) {
    try {
      var p = limitLineOf(O).options().price;
      return typeof p === "number" ? p : null;
    } catch (e) {
      return null;
    }
  }

  function chartRead(O) {
    var ts = O.api.timeScale();
    if (O.api.panes().length !== O.ph.length) G.setTimeout(chartLayout, 150, O); // panes added or removed
    var pt = O.latest ? O.latest() : null;
    O.px = null;
    O.py = null;
    if (pt && Number.isFinite(pt.time) && Number.isFinite(pt.value)) {
      var x = ts.timeToCoordinate(pt.time);
      var y = O.series.priceToCoordinate(pt.value);
      if (x !== null && y !== null) {
        O.px = x;
        O.py = y + (O.top[1] || 0);
      }
    }
    var lp = limitPrice(O);
    var ly = lp === null ? null : O.series.priceToCoordinate(lp);
    O.ty = ly === null ? null : ly + (O.top[1] || 0);
    var stamp = pt ? pt.time : null;
    if (O.refill || stamp !== O.lastTime) {
      O.refill = false;
      O.lastTime = stamp;
      O.fill = [];
      if (O.priceSeries) {
        var d = O.priceSeries.data();
        var from = Math.max(0, d.length - 600);
        var step = Math.max(1, Math.ceil((d.length - from) / FILL_MAX));
        for (var i = from; i < d.length; i += step) {
          if (typeof d[i].value !== "number") continue;
          var fx0 = ts.timeToCoordinate(d[i].time);
          var fy0 = O.priceSeries.priceToCoordinate(d[i].value);
          if (fx0 !== null && fy0 !== null) O.fill.push(fx0, fy0 + (O.top[0] || 0));
        }
      }
    }
    O.dirty = false;
  }

  function ringAt(ctx, x, y, r, a, c, w) {
    if (a <= 0.01) return;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.lineWidth = w;
    ctx.strokeStyle = "rgba(" + c + "," + a + ")";
    ctx.stroke();
  }

  function drawFill(ctx, O, c) {
    var f = O.fill;
    if (f.length < 4 || !O.ph[0]) return;
    var top = O.top[0] || 0;
    var bottom = top + O.ph[0];
    ctx.beginPath();
    ctx.moveTo(f[0], f[1]);
    for (var i = 2; i < f.length; i += 2) ctx.lineTo(f[i], f[i + 1]);
    ctx.lineTo(f[f.length - 2], bottom);
    ctx.lineTo(f[0], bottom);
    ctx.closePath();
    ctx.fillStyle = "rgba(" + c + ",.07)"; // flat and faint
    ctx.fill();
  }

  function drawScan(ctx, O, c, t) {
    var x = Math.round(O.px);
    ctx.fillStyle = "rgba(" + c + ",.35)";
    ctx.fillRect(x - 0.5, 0, 1, O.h);
    var sy = ((t % 3200) / 3200) * O.h; // bright spot travelling down
    ctx.fillStyle = "rgba(" + c + ",.85)";
    ctx.fillRect(x - 1, sy - 12, 2, 24);
  }

  function drawCross(ctx, O, t, plotW) {
    var X = O.cross;
    var e = Math.max(0, t - X.t0);
    var red = "255,45,85";
    if (e > SHOCK_MS + RING_GAP_MS) {
      O.cross = null;
      return;
    }
    if (e < 180) {
      var fa = 1 - e / 180; // short flash
      ctx.beginPath();
      ctx.arc(X.x, X.y, 14, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,255,255," + 0.8 * fa + ")";
      ctx.fill();
    }
    for (var k = 0; k < 2; k += 1) {
      var p = (e - k * RING_GAP_MS) / SHOCK_MS;
      if (p < 0 || p > 1) continue;
      var ease = 1 - Math.pow(1 - p, 3);
      ringAt(ctx, X.x, X.y, 8 + 52 * ease, 0.9 * (1 - p), red, 3 - 1.8 * p);
    }
    if (e < LIMIT_MS && O.ty !== null) {
      var lk = 1 - e / LIMIT_MS;
      var yy = O.ty + Math.sin(e / 28) * 2 * lk;
      ctx.save();
      ctx.shadowColor = "rgba(" + red + ",.9)";
      ctx.shadowBlur = 14;
      ctx.beginPath();
      ctx.moveTo(0, yy);
      ctx.lineTo(plotW, yy);
      ctx.lineWidth = 2;
      ctx.strokeStyle = "rgba(" + red + "," + (0.35 + 0.65 * lk) + ")";
      ctx.stroke();
      ctx.restore();
    }
  }

  function chartDraw(ctx, t) {
    var O = S.chartOv;
    if (!O || !O.w || !O.h) return;
    ctx.clearRect(0, 0, O.w, O.h);
    if (O.dirty || t - O.read > READ_MS) {
      O.read = t;
      chartRead(O);
    }
    var c = S.rgb.join(",");
    var plotW = O.pw || O.w;
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, plotW, O.h);
    ctx.clip();
    drawFill(ctx, O, c);
    if (O.px !== null) drawScan(ctx, O, c, t);
    if (O.px !== null && O.py !== null) {
      var ph = (t % 1600) / 1600; // pulsing ring on the latest point
      ringAt(ctx, O.px, O.py, 4 + ph * 12, 0.8 * (1 - ph), c, 1.5);
      ctx.beginPath();
      ctx.arc(O.px, O.py, 3, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(" + c + ",.95)";
      ctx.fill();
    }
    ctx.restore();
    if (O.cross) drawCross(ctx, O, t, plotW); // not clipped: the rings may cross the axis
  }

  // replaces any overlay made before, then draws on #fx-chart
  function chartAttach(api, series, opts) {
    chartDetach();
    var canvas = DOC.getElementById("fx-chart");
    var host = DOC.getElementById("chart");
    var ctx = canvas && canvas.getContext && canvas.getContext("2d");
    if (!ctx || !host || !api || !series) return;
    opts = opts || {};
    var O = {
      api: api,
      series: series,
      priceSeries: opts.priceSeries,
      limitLine: opts.limitLine,
      getLimit: opts.getLimit,
      latest: opts.latest,
      canvas: canvas,
      host: host,
      ctx: ctx,
      draw: chartDraw,
      w: 0,
      h: 0,
      top: [],
      ph: [],
      px: null,
      py: null,
      ty: null,
      fill: [],
      read: 0,
      dirty: true,
      refill: true,
    };
    O.onRange = function () {
      O.refill = true;
      chartLayout(O);
    };
    if (typeof G.ResizeObserver === "function") {
      O.ro = new G.ResizeObserver(function (entries) {
        var r = entries[0] && entries[0].contentRect;
        if (!r) return;
        if (r.width <= 0 || r.height <= 0) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          O.w = 0; // hidden page: nothing to draw until it has a size again
          O.h = 0;
          return;
        }
        var dpr = Math.min(G.devicePixelRatio || 1, 2);
        O.w = Math.round(r.width);
        O.h = Math.round(r.height);
        canvas.width = Math.round(r.width * dpr);
        canvas.height = Math.round(r.height * dpr);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        O.refill = true;
        chartLayout(O);
      });
      O.ro.observe(host);
    }
    api.timeScale().subscribeVisibleTimeRangeChange(O.onRange);
    S.chartOv = O;
    addLayer(chartDraw, { ctx: ctx });
  }

  function chartCross(point) {
    var O = S.chartOv;
    Motion.alertBurst({ severity: "high" }); // rate-limited
    if (S.off || !O || !point) return;
    var nowMs = G.performance.now();
    if (S.crossAt !== undefined && nowMs - S.crossAt < BURST_GAP_MS) return; // the shockwave obeys the same gap
    S.crossAt = nowMs;
    var x = O.api.timeScale().timeToCoordinate(point.time);
    var y = O.series.priceToCoordinate(point.value);
    if (x === null || y === null) return; // off-screen: only the burst
    O.cross = { x: x, y: y + (O.top[1] || 0), t0: G.performance.now() };
  }

  // ---------------------------------------------------------------- public API

  function pageEnter(node) {
    if (S.off || !node || !DOC.body || typeof DOC.createElement !== "function") return;
    var line = DOC.createElement("div");
    line.className = "fx-wipe";
    line.setAttribute("aria-hidden", "true");
    DOC.body.appendChild(line);
    G.setTimeout(function () {
      if (line.parentNode) line.parentNode.removeChild(line);
    }, 420);
    var kids = node.children;
    var rise = [];
    for (var i = 0; i < kids.length && rise.length < 8; i += 1) {
      if (!kids[i].hidden) rise.push(kids[i]);
    }
    rise.forEach(function (kid, n) {
      kid.classList.remove("fx-rise");
      void kid.offsetWidth; // restart the animation when the same page is entered again
      kid.style.setProperty("--fx-i", String(n));
      kid.classList.add("fx-rise");
    });
    G.setTimeout(function () {
      rise.forEach(function (kid) {
        kid.classList.remove("fx-rise");
        kid.style.removeProperty("--fx-i");
      });
    }, 1000);
  }

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
          headerInit();
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

    // the main chart: scan line, live-point ring, price glow
    chartAttach: function (api, series, opts) {
      try {
        chartAttach(api, series, opts);
      } catch (e) {
        /* decoration only */
      }
    },

    // the score crossed the limit going up: rings, red limit line, edge flash
    chartCross: function (point) {
      try {
        chartCross(point);
      } catch (e) {
        /* decoration only */
      }
    },

    // a cyan scan line wipes down the page, then the cards rise in one after another
    pageEnter: function (node) {
      try {
        pageEnter(node);
      } catch (e) {
        /* decoration only */
      }
    },

    // the data strip: items is [{label, value}]; text nodes only; the same labels update the values in place
    ticker: function (items) {
      try {
        ticker(items);
      } catch (e) {
        /* decoration only */
      }
    },

    // the nav link that is current now: the underline slides to it
    navTo: function (a) {
      try {
        NAV.a = a || null;
        navPlace();
      } catch (e) {
        /* decoration only */
      }
    },

    // the shared loop, for the scenes added on top of this
    addLayer: addLayer,
    removeLayer: removeLayer,

    // the background canvas state for the plexus (hud.js): size in CSS px and the eased mood colour
    bg: function () {
      return S;
    },
  };

  G.Motion = Motion;
})();
