/*
 * Hero: "Same problem, two worlds".
 * Left canvas: the world the model believes (Figure 1's three drift steps, then the
 * ball runs in that world). Right canvas: the world the problem specifies.
 */
(function (root) {
  'use strict';
  const { physics: P, scene: S } = root.AWM;
  const ST = S.STYLES;

  const BOUNDS = { xmin: -3.4, xmax: 2.05, ymin: -2.32, ymax: 2.7 };
  const TS = 1.0; // simulation seconds per real second

  const trueSim = P.simulate({ k: 1, drop: 0, bounds: BOUNDS });
  const beliefSim = P.simulate({ k: 0, drop: 1, bounds: BOUNDS });

  const T = { step1: 1.3, step1End: 2.3, step2: 2.9, step3: 4.5, step3End: 5.7, release: 6.1 };
  T.motionEnd = T.release + Math.max(trueSim.events.tEnd, beliefSim.events.tEnd) / TS;
  T.tagline = T.motionEnd + 0.2;
  T.total = T.tagline + 3.6;

  const ease = (x) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));
  const ramp = (t, a, b) => ease((t - a) / (b - a));

  // Ball centres for every sample, computed once.
  function centers(sim) {
    return sim.samples.map((s) => S.ballCenter(sim.geo, s, s.phase === 'flight' && sim.events.tO !== null ? s.t - sim.events.tO : null));
  }
  const trueCenters = centers(trueSim);
  const beliefCenters = centers(beliefSim);
  const flightStart = trueSim.events.tO === null ? trueCenters.length : Math.ceil(trueSim.events.tO / trueSim.dtOut);

  function nearTrack(geo) {
    return (x, y) => {
      const onArc = x >= geo.C.x - 0.1 && y <= 0.15 && Math.abs(Math.hypot(x - geo.C.x, y - geo.C.y) - geo.R) < 0.24;
      const nearO = Math.hypot(x - geo.O.x, y - geo.O.y) < 0.3;
      return onArc || nearO;
    };
  }

  function drawCommonLabels(v, geo, st, alpha = 1) {
    S.label(v, 'A', { x: geo.A.x, y: geo.A.y }, st.label, { dy: 18, alpha });
    S.label(v, 'B', geo.B, st.label, { dy: 18, alpha });
    S.label(v, 'O', geo.O, st.label, { dx: -14, dy: -12, alpha });
  }

  function idxAt(sim, tau) {
    return Math.max(0, Math.min(sim.samples.length - 1, Math.floor(tau / sim.dtOut)));
  }

  // "v_O^2 < 0" with proper sub/superscripts.
  function drawVO2(v, at, color, alpha) {
    const ctx = v.ctx;
    const x = v.X(at.x);
    const y = v.Y(at.y);
    ctx.save();
    ctx.globalAlpha *= alpha;
    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 8;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.font = "italic 600 18px 'Times New Roman', serif";
    ctx.fillText('v', x, y);
    const wv = ctx.measureText('v').width;
    ctx.font = "600 11.5px 'Times New Roman', serif";
    ctx.fillText('2', x + wv + 1, y - 8);
    ctx.font = "italic 600 11.5px 'Times New Roman', serif";
    ctx.fillText('O', x + wv + 1, y + 4);
    ctx.font = "600 16px 'Inter', system-ui, sans-serif";
    ctx.fillText('< 0', x + wv + 14, y);
    ctx.restore();
  }

  /* ---------------------------- belief world ---------------------------- */

  function renderBelief(v, t) {
    v.clear();
    const drop = ramp(t, T.step1, T.step1End);
    const geo = drop >= 1 ? beliefSim.geo : P.geometry({ drop });
    const dissolve = ramp(t, T.step3, T.step3End);
    const tau = (t - T.release) * TS;
    const st = ST.belief;

    S.shadeField(v, geo.fieldX, st, { alpha: 1 - dissolve });
    S.drawField(v, geo.fieldX, t, st, { dissolve, exclude: nearTrack(geo) });
    if (dissolve < 1) {
      S.label(v, 'E', { x: 1.45, y: -1.55 }, st.label, { alpha: 1 - dissolve, font: S.MATH_FONT.replace('16px', '19px') });
    }

    // The geometry the problem states, kept as a faint reference.
    if (t >= T.step1) {
      const a = 0.5 * ramp(t, T.step1, T.step1End);
      const trueGeo = trueSim.geo;
      S.drawCurve(v, [{ x: trueGeo.A.x - S.WALL_GAP, y: -1 }, trueGeo.B], 'rgba(200,210,230,0.9)', { width: 1, dash: [3, 5], alpha: a });
      S.label(v, 'stated: AB horizontal', { x: -2.1, y: -0.82 }, 'rgba(200,210,230,0.75)', {
        font: "500 11px 'JetBrains Mono', ui-monospace, monospace", alpha: a,
      });
    }

    // Glitch while the geometry is being corrupted.
    const glitch = Math.max(0, Math.sin(Math.PI * Math.min(1, Math.max(0, (t - T.step1) / (T.step1End - T.step1 + 0.25)))));
    if (glitch > 0.02) {
      const j = (S.hash(Math.floor(t * 30), 3) - 0.5) * 2;
      S.drawTrack(v, geo, st, { color: `rgba(${S.RGB.bad},0.65)`, offsetX: -glitch * (4 + 4 * j), alpha: glitch });
      S.drawTrack(v, geo, st, { color: 'rgba(170,255,255,0.6)', offsetX: glitch * (4 - 3 * j), alpha: glitch });
    }
    const flicker = 0.9 + 0.1 * Math.sin(t * 23) * Math.sin(t * 7.3);
    S.drawTrack(v, geo, st, { alpha: flicker });
    S.drawPoint(v, geo.B, st);
    S.drawPoint(v, geo.O, st);
    drawCommonLabels(v, geo, st);

    // Red dimension mark for the wrong height, as in Figure 1.
    if (t >= T.step1) {
      const a = ramp(t, T.step1 + 0.4, T.step1End + 0.2);
      const xd = geo.B.x + 0.24;
      const red = `rgba(${S.RGB.bad},0.95)`;
      S.drawCurve(v, [geo.A, { x: xd, y: geo.A.y }], red, { width: 1, dash: [3, 4], alpha: a * 0.6 });
      S.drawArrow(v, { x: xd, y: (geo.A.y + geo.B.y) / 2 }, { x: xd, y: geo.B.y }, red, { width: 1.3, head: 7, alpha: a });
      S.drawArrow(v, { x: xd, y: (geo.A.y + geo.B.y) / 2 }, { x: xd, y: geo.A.y }, red, { width: 1.3, head: 7, alpha: a });
      S.label(v, 'R', { x: xd, y: (geo.A.y + geo.B.y) / 2 }, red, { dx: 12, alpha: a });
    }

    // Step 2: O becomes unreachable.
    if (t >= T.step2) {
      const a = ramp(t, T.step2, T.step2 + 0.4) * (t > T.release ? 0.7 : 1);
      const pulse = 0.5 + 0.5 * Math.sin((t - T.step2) * 6);
      const ctx = v.ctx;
      ctx.save();
      ctx.globalAlpha = a;
      ctx.strokeStyle = `rgba(${S.RGB.bad},${0.55 + 0.45 * pulse})`;
      ctx.shadowColor = `rgba(${S.RGB.bad},0.9)`;
      ctx.shadowBlur = 12;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(v.X(0), v.Y(0), v.L(0.15 + 0.04 * pulse), 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
      drawVO2(v, { x: -0.86, y: 0.3 }, `rgba(${S.RGB.bad},1)`, a);
    }

    // Ball and spring.
    let sample;
    let center;
    if (tau <= 0) {
      sample = { x: geo.A.x, y: geo.A.y, phase: 'spring', s: 0 };
      center = S.ballCenter(geo, sample);
    } else {
      sample = P.sampleAt(beliefSim, tau);
      center = S.ballCenter(geo, sample);
      const i1 = idxAt(beliefSim, tau);
      const i0 = Math.max(0, i1 - Math.round(1.6 / beliefSim.dtOut));
      S.drawTrail(v, beliefCenters.slice(i0, i1 + 1).concat([center]), S.RGB.belief, { width: 2.2, alpha: 0.8 });
    }
    S.drawSpring(v, geo, sample.s, beliefSim.s0, st);
    S.drawBall(v, center, st);

    const tStall = beliefSim.events.tStall;
    if (tStall !== null && tau >= tStall && tau < tStall + 1.8) {
      const a = 1 - ramp(tau, tStall + 1.0, tStall + 1.8);
      const p = beliefCenters[idxAt(beliefSim, tStall)];
      S.label(v, 'v = 0', { x: p.x, y: p.y }, st.label, { dx: 34, dy: -14, alpha: a, font: "italic 600 14px 'Times New Roman', serif" });
    }
  }

  /* ----------------------------- true world ----------------------------- */

  function renderTrue(v, t) {
    v.clear();
    const geo = trueSim.geo;
    const st = ST.true;
    const tau = (t - T.release) * TS;

    S.shadeField(v, geo.fieldX, st);
    S.drawField(v, geo.fieldX, t, st, { exclude: nearTrack(geo) });
    S.label(v, 'E', { x: 1.45, y: -1.55 }, `rgba(${S.RGB.field},1)`, { font: S.MATH_FONT.replace('16px', '19px') });
    S.label(v, 'θ = 45°', { x: 1.45, y: -1.95 }, `rgba(${S.RGB.field},0.95)`, { font: "italic 500 14px 'Times New Roman', serif" });

    S.drawTrack(v, geo, st);
    S.drawPoint(v, geo.B, st);
    S.drawPoint(v, geo.O, st);
    drawCommonLabels(v, geo, st);

    let sample;
    let center;
    if (tau <= 0) {
      sample = { x: geo.A.x, y: geo.A.y, phase: 'spring', s: 0 };
      center = S.ballCenter(geo, sample);
    } else {
      sample = P.sampleAt(trueSim, tau);
      const tO = trueSim.events.tO;
      center = S.ballCenter(geo, sample, sample.phase === 'flight' ? tau - tO : null);
      const i1 = idxAt(trueSim, tau);
      // Persistent parabola once the ball has left O.
      if (i1 > flightStart) {
        S.drawCurve(v, trueCenters.slice(flightStart, i1 + 1), `rgba(${S.RGB.amber},0.55)`, { width: 1.6, glow: 10, glowColor: `rgba(${S.RGB.amber},0.6)` });
        const a = ramp(tau, tO + 0.8, tO + 1.4);
        S.label(v, 'y² = 6Rx', { x: 1.15, y: 1.65 }, `rgba(${S.RGB.amber},1)`, { align: 'left', alpha: a, font: "italic 600 16px 'Times New Roman', serif" });
      }
      const i0 = Math.max(0, i1 - Math.round(1.3 / trueSim.dtOut));
      S.drawTrail(v, trueCenters.slice(i0, i1 + 1).concat([center]), S.RGB.amber, { width: 2.6 });
    }
    S.drawSpring(v, geo, sample.phase === 'flight' || sample.phase === 'arc' ? null : sample.s, trueSim.s0, st);
    const inFrame = center.y < BOUNDS.ymax + 0.2 && center.x < BOUNDS.xmax + 0.2;
    if (inFrame) S.drawBall(v, center, st);
  }

  /* ------------------------------- player ------------------------------- */

  function init() {
    const stage = document.getElementById('heroStage');
    if (!stage) return;
    const beliefView = S.createView(document.getElementById('beliefCanvas'), BOUNDS);
    const trueView = S.createView(document.getElementById('trueCanvas'), BOUNDS);
    const chips = [...stage.querySelectorAll('[data-step]')];
    const notes = [...stage.querySelectorAll('[data-note]')];
    const beliefStatus = stage.querySelector('[data-status="belief"]');
    const tagline = stage.querySelector('.tagline');
    const playBtn = document.getElementById('heroPlay');
    const restartBtn = document.getElementById('heroRestart');
    const bar = document.getElementById('heroBar');
    const fill = bar.querySelector('.timeline__fill');

    const reduced = root.matchMedia && root.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const params = new URLSearchParams(root.location.search);
    let t = 0;
    let playing = !reduced;
    let visible = true;
    let last = null;

    if (reduced) t = T.tagline - 0.05;
    if (params.has('t')) {
      t = Math.min(T.total, Math.max(0, parseFloat(params.get('t')) || 0));
      playing = false;
    }

    // Timeline markers at the story beats.
    const beats = [
      { at: T.step1, text: '1' },
      { at: T.step2, text: '2' },
      { at: T.step3, text: '3' },
      { at: T.release, text: '▶' },
    ];
    beats.forEach((b) => {
      const m = document.createElement('button');
      m.className = 'timeline__mark';
      m.type = 'button';
      m.style.left = `${(b.at / T.total) * 100}%`;
      m.textContent = b.text;
      m.setAttribute('aria-label', b.text === '▶' ? 'Jump to release' : `Jump to step ${b.text}`);
      m.addEventListener('click', (e) => {
        e.stopPropagation();
        t = b.at + 0.01;
        draw();
      });
      bar.appendChild(m);
    });
    bar.addEventListener('click', (e) => {
      const r = bar.getBoundingClientRect();
      t = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * T.total;
      draw();
    });

    const tauToReal = (tau) => T.release + tau / TS;
    const trueO = trueSim.events.tO === null ? Infinity : tauToReal(trueSim.events.tO);
    const beliefStall = beliefSim.events.tStall === null ? Infinity : tauToReal(beliefSim.events.tStall);
    const stepStarts = [T.step1, T.step2, T.step3, T.release];

    function updateCaptions() {
      chips.forEach((c, i) => {
        c.classList.toggle('is-on', t >= stepStarts[i]);
        c.classList.toggle('is-active', t >= stepStarts[i] && t < stepStarts[i + 1]);
      });
      notes.forEach((n) => {
        const k = +n.dataset.note;
        const on = k === 0 || (k === 1 && t >= trueO) || (k === 2 && t >= trueO + 0.9);
        n.classList.toggle('is-on', on);
      });
      beliefStatus.classList.toggle('is-on', t >= beliefStall);
      const tag = t >= T.tagline && t < T.total - 0.45;
      tagline.classList.toggle('is-on', tag);
      stage.classList.toggle('is-tagline', tag);
      fill.style.width = `${(t / T.total) * 100}%`;
      playBtn.setAttribute('aria-label', playing ? 'Pause' : 'Play');
      playBtn.classList.toggle('is-playing', playing);
    }

    function draw() {
      renderBelief(beliefView, t);
      renderTrue(trueView, t);
      updateCaptions();
    }

    function frame(now) {
      if (playing && visible) {
        if (last !== null) t += Math.min(0.05, (now - last) / 1000);
        last = now;
        if (t >= T.total) t = 0;
        draw();
      } else {
        last = null;
      }
      root.requestAnimationFrame(frame);
    }

    playBtn.addEventListener('click', () => {
      playing = !playing;
      if (playing && t >= T.total - 0.05) t = 0;
      updateCaptions();
    });
    restartBtn.addEventListener('click', () => {
      t = 0;
      playing = true;
      draw();
    });

    if ('IntersectionObserver' in root) {
      new IntersectionObserver((entries) => {
        visible = entries[0].isIntersecting;
      }, { threshold: 0.05 }).observe(stage);
    }

    let resizeTimer = null;
    root.addEventListener('resize', () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        beliefView.resize();
        trueView.resize();
        draw();
      }, 120);
    });

    draw();
    root.requestAnimationFrame(frame);
    root.AWM.hero = { seek: (x) => { t = x; draw(); }, T };
  }

  root.AWM.initHero = init;
})(typeof self !== 'undefined' ? self : this);
