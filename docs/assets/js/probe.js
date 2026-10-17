/*
 * Method scene M4: drop a rollout's world into the executable world the checker
 * builds, and probe it (invariants, trace consistency, response under perturbation).
 * Rollouts are the two cases of Appendix F; tests follow the Appendix M checker.
 */
(function (root) {
  'use strict';
  const { physics: P, scene: S } = root.AWM;
  const ST = S.STYLES;

  const BOUNDS = { xmin: -3.75, xmax: 2.5, ymin: -1.9, ymax: 3.35 };
  const ARROW_SCALE = 0.55; // world units per sqrt(gR)
  const CURVE_T = 2.6;

  const LINES = {
    1: [
      '[representation]  type: timeline',
      '    S1: A → B    S2: B → O    S3: after O',
      '[sub2]  W_g + W_E = ½ m v_O² − ½ m v_B²',
      '        W_g = −m g R ;  W_E = 2 m g R',
      '        ⇒ v_O = √(3gR)',
      '[sub3]  a_x = g ;  a_y = 0',
      '        x = ½ g t² ;  y = v_O t   ⇒ y² = 6Rx',
    ],
    2: [
      '[representation]  type: interaction_graph',
      '    nodes: ball · spring · gravity · field · rail',
      '[sub2]  F_Ey = m g cancels gravity',
      '        normal force does no work',
      '        ⇒ kinetic energy unchanged: v_O = v_B = √(gR)',
      '[sub3]  a_x = g ;  a_y = 0',
      '        x = ½ g t² ;  y = v_O t   ⇒ y² = 2Rx',
    ],
  };
  const TRANSLATE_LINES = [
    '# translator f (Qwen2.5-Coder-1.5B) → executable program ŵ',
    '# ŵ emits  E_p · v_O · a_x · a_y · (x, y)',
  ];

  // Run timeline in milliseconds.
  const TL = { type: 0, typeEnd: 1900, translate: 1900, emit: 2400, fly: 2500, flyEnd: 3900, inv: 4000, invDone: 4800, trace: 5000, traceDone: 5600, pert: 5800, pertMorphEnd: 6700, pertDone: 6900, rs: 7200, rsEnd: 7900, insight: 7900, done: 8000 };

  const ease = (x) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));
  const ramp = (t, a, b) => ease((t - a) / (b - a));
  const fmt = (x, d = 2) => (Math.abs(x) < 0.005 ? 0 : x).toFixed(d);

  function escapeHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function highlight(line) {
    let h = escapeHtml(line);
    if (h.startsWith('#')) return `<span class="t-com">${h}</span>`;
    h = h.replace(/^(\s*)(\[[a-z0-9_]+\])/, '$1<span class="t-tag">$2</span>');
    h = h.replace(/(⇒|→)/g, '<span class="t-arrow">$1</span>');
    h = h.replace(/(type:|nodes:)/g, '<span class="t-key">$1</span>');
    return h;
  }

  function trajectory(out, tMax = CURVE_T) {
    const pts = [];
    for (let t = 0; t <= tMax + 1e-9; t += 0.02) {
      const p = P.trajectoryPoint(out, t);
      if (p.x > BOUNDS.xmax + 0.3 || p.y > BOUNDS.ymax + 0.3 || p.y < BOUNDS.ymin - 0.3) {
        pts.push(p);
        break;
      }
      pts.push(p);
    }
    return pts;
  }

  function drawGrid(v) {
    const { ctx, bounds } = v;
    ctx.save();
    ctx.lineWidth = 1;
    for (let x = Math.ceil(bounds.xmin * 2) / 2; x <= bounds.xmax; x += 0.5) {
      ctx.strokeStyle = Math.abs(x) < 1e-9 ? 'rgba(71,85,105,0.35)' : 'rgba(71,85,105,0.08)';
      ctx.beginPath();
      ctx.moveTo(v.X(x), 0);
      ctx.lineTo(v.X(x), v.h);
      ctx.stroke();
    }
    for (let y = Math.ceil(bounds.ymin * 2) / 2; y <= bounds.ymax; y += 0.5) {
      ctx.strokeStyle = Math.abs(y) < 1e-9 ? 'rgba(71,85,105,0.35)' : 'rgba(71,85,105,0.08)';
      ctx.beginPath();
      ctx.moveTo(0, v.Y(y));
      ctx.lineTo(v.w, v.Y(y));
      ctx.stroke();
    }
    ctx.restore();
    S.label(v, 'x', { x: bounds.xmax - 0.12, y: 0 }, 'rgba(71,85,105,0.8)', { dy: -10, font: "italic 13px 'Times New Roman', serif" });
    S.label(v, 'y', { x: 0, y: bounds.ymax - 0.12 }, 'rgba(71,85,105,0.8)', { dx: 10, font: "italic 13px 'Times New Roman', serif" });
  }

  function init() {
    const section = document.getElementById('probe');
    if (!section) return;
    const view = S.createView(document.getElementById('probeCanvas'), BOUNDS);
    const $ = (sel) => section.querySelector(sel);
    const els = {
      cases: [...section.querySelectorAll('[data-case]')],
      term: $('#probeTerm'),
      emit: $('#probeEmit'),
      cards: { inv: $('[data-probe="inv"]'), trace: $('[data-probe="trace"]'), pert: $('[data-probe="pert"]') },
      beam: $('#balanceBeam'),
      panL: $('#balancePanL'),
      panR: $('#balancePanR'),
      panLVal: $('#balanceL'),
      panRVal: $('#balanceR'),
      checks: $('#invChecks'),
      traceRead: $('#traceRead'),
      params: [...section.querySelectorAll('[data-param]')],
      range: $('#pertRange'),
      rangeLabel: $('#pertLabel'),
      chart: $('#pertChart'),
      pertRead: $('#pertRead'),
      pertNote: $('#pertNote'),
      rsRows: { inv: $('[data-rs="inv"]'), trace: $('[data-rs="trace"]'), pert: $('[data-rs="pert"]') },
      rsValue: $('#rsValue'),
      insight: $('#probeInsight'),
      rerun: $('#probeRun'),
    };

    const reduced = root.matchMedia && root.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const params = new URLSearchParams(root.location.search);
    const state = {
      caseId: params.get('probe') === '2' ? 2 : 1,
      param: 'R',
      pertR: 1.1,
      fieldK: 1,
      start: null,
      elapsed: 0,
      started: false,
      result: null,
      visible: false,
      lastInput: -1e9,
      dispR: 1,
      dispK: 1,
      lastFrame: null,
    };

    const program = () => P.programs[state.caseId];

    function computeResult() {
      state.result = P.runChecker(program(), { pertR: state.pertR });
    }

    /* ----------------------------- DOM updates ----------------------------- */

    function setHTML(el, html) {
      if (el.innerHTML !== html) el.innerHTML = html;
    }

    function setStatus(card, status) {
      if (card.dataset.status === status) return;
      card.dataset.status = status;
      const badge = card.querySelector('.pcard__status');
      badge.textContent = { idle: 'waiting', running: 'probing…', pass: 'pass', fail: 'fail', explore: 'explore' }[status];
    }

    function renderTerminal(e) {
      const lines = LINES[state.caseId];
      const per = (TL.typeEnd - TL.type) / lines.length;
      let html = '';
      for (let i = 0; i < lines.length; i++) {
        const local = (e - TL.type - i * per) / per;
        if (local <= 0) break;
        const n = Math.floor(Math.min(1, local) * lines[i].length);
        html += `<div class="t-line">${highlight(lines[i].slice(0, n))}${local < 1 ? '<span class="t-caret"></span>' : ''}</div>`;
      }
      if (e >= TL.translate) {
        TRANSLATE_LINES.forEach((l, i) => {
          if (e >= TL.translate + i * 220) html += `<div class="t-line">${highlight(l)}</div>`;
        });
      }
      setHTML(els.term, html);
    }

    function renderEmit(e) {
      const r = state.result;
      const on = e >= TL.emit;
      els.emit.classList.toggle('is-on', on);
      setHTML(els.emit, `
        <span><i>E<sub>p</sub></i> = ${on ? fmt(r.out.Ep) : '—'} <small>mgR</small></span>
        <span class="${state.caseId === 2 && on ? 'is-off' : ''}"><i>v<sub>O</sub></i> = ${on ? fmt(r.out.vO) : '—'} <small>√(gR)</small></span>
        <span><i>a<sub>x</sub></i> = ${on ? fmt(r.out.ax) : '—'} <small>g</small></span>
        <span><i>a<sub>y</sub></i> = ${on ? fmt(r.out.ay) : '—'} <small>g</small></span>`.replace(/\n\s*/g, ''));
    }

    function renderInvariants(e) {
      const r = state.result;
      const card = els.cards.inv;
      if (e < TL.inv) setStatus(card, 'idle');
      else if (e < TL.invDone) setStatus(card, 'running');
      else setStatus(card, r.invPass ? 'pass' : 'fail');

      const work = r.invariants.find((c) => c.key === 'work');
      const p = ramp(e, TL.inv, TL.invDone);
      const diff = work.expected - work.model;
      const angle = -Math.max(-14, Math.min(14, diff * 14)) * p;
      const L = 70; // half beam length in svg units
      const rad = (angle * Math.PI) / 180;
      els.beam.setAttribute('transform', `rotate(${angle.toFixed(2)} 110 34)`);
      els.panL.setAttribute('transform', `translate(${(-L * Math.cos(rad) + L).toFixed(2)} ${(-L * Math.sin(rad)).toFixed(2)})`);
      els.panR.setAttribute('transform', `translate(${(L * Math.cos(rad) - L).toFixed(2)} ${(L * Math.sin(rad)).toFixed(2)})`);
      els.panLVal.textContent = e >= TL.inv ? `${fmt(work.expected)} mgR` : '—';
      els.panRVal.textContent = e >= TL.inv ? `${fmt(work.model * p + work.expected * (1 - p))} mgR` : '—';

      const rows = r.invariants.map((c, i) => {
        const show = e >= TL.inv + 150 * (i + 1);
        const status = !show ? '' : e < TL.invDone ? '…' : c.pass ? '✓' : '✗';
        const cls = !show ? 'is-wait' : e < TL.invDone ? 'is-run' : c.pass ? 'is-pass' : 'is-fail';
        return `<li class="${cls}"><span>${c.label}</span><span class="num">${show ? `${fmt(c.model)} vs ${fmt(c.expected)} ${c.unit}` : ''}</span><b>${status}</b></li>`;
      }).join('');
      setHTML(els.checks, rows);
    }

    function renderTrace(e) {
      const r = state.result;
      const card = els.cards.trace;
      if (e < TL.trace) setStatus(card, 'idle');
      else if (e < TL.traceDone) setStatus(card, 'running');
      else setStatus(card, r.trace.pass ? 'pass' : 'fail');
      const tr = r.trace;
      setHTML(els.traceRead, e >= TL.trace
        ? `At its emitted point <i>x</i> = ${fmt(tr.point.x)}, its own <i>a<sub>x</sub></i>, <i>a<sub>y</sub></i>, <i>v<sub>O</sub></i> give <i>y</i> = ${fmt(tr.yFromTrace)}; it emitted <i>y</i> = ${fmt(tr.point.y)}.`
        : 'Recompute the trajectory from the rollout’s own quantities and compare.');
    }

    // Response curves: v_O of the true world and of the rollout's program across the perturbed input.
    let chartSig = '';
    function drawResponse(f, showMarker) {
      const scored = state.param === 'R';
      const dom = scored ? [0.7, 1.4] : [0.5, 1.5];
      const sig = `${state.caseId}|${state.param}|${f.toFixed(3)}|${showMarker}|${els.chart.clientWidth}`;
      if (sig === chartSig) return;
      chartSig = sig;
      const C = root.AWM.charts;
      const box = els.chart;
      box.innerHTML = '';
      const w = Math.max(box.clientWidth, 260);
      const h = 176;
      const m = { l: 40, r: 84, t: 12, b: 32 };
      const svg = C.el('svg', { width: w, height: h, viewBox: `0 0 ${w} ${h}`, class: 'viz' }, box);
      const X = (x) => m.l + ((x - dom[0]) / (dom[1] - dom[0])) * (w - m.l - m.r);
      const Y = (v) => m.t + (1 - (v - 0.5) / 2.0) * (h - m.t - m.b);
      const G = (x) => (scored ? P.givens({ R: x }) : P.givens({ k: x }));
      [1, 1.5, 2].forEach((v) => {
        C.el('line', { x1: m.l, x2: w - m.r, y1: Y(v), y2: Y(v), class: 'viz-grid' }, svg);
        C.el('text', { x: m.l - 7, y: Y(v), class: 'viz-axis', 'text-anchor': 'end', 'dominant-baseline': 'middle' }, svg, v.toFixed(1));
      });
      const ticks = scored ? [0.7, 0.85, 1, 1.15, 1.3] : [0.5, 0.75, 1, 1.25, 1.5];
      ticks.forEach((x) => C.el('text', { x: X(x), y: h - m.b + 15, class: 'viz-axis', 'text-anchor': 'middle' }, svg, `×${x}`));
      C.el('text', { x: (m.l + w - m.r) / 2, y: h - 3, class: 'viz-axis viz-axis--title', 'text-anchor': 'middle' }, svg, scored ? 'radius R, relative to the problem' : 'field strength qE, relative to the problem');
      C.el('text', { x: 2, y: 9, class: 'viz-axis viz-axis--title' }, svg, 'v_O  [√(gR)]');
      C.el('line', { x1: X(1), x2: X(1), y1: m.t, y2: h - m.b, class: 'viz-ref' }, svg);
      const curve = (fn) => {
        let d = '';
        for (let i = 0; i <= 60; i++) {
          const x = dom[0] + ((dom[1] - dom[0]) * i) / 60;
          d += `${i ? 'L' : 'M'}${X(x).toFixed(1)},${Y(fn(x)).toFixed(1)}`;
        }
        return d;
      };
      const tFn = (x) => P.trueWorld(G(x)).vO;
      const rFn = (x) => program().run(G(x)).vO;
      C.el('path', { d: curve(tFn), fill: 'none', stroke: `rgb(${S.RGB.amber})`, 'stroke-width': 2.4 }, svg);
      C.el('path', { d: curve(rFn), fill: 'none', stroke: `rgb(${S.RGB.belief})`, 'stroke-width': 2.2, 'stroke-dasharray': '6 4' }, svg);
      const yt = Y(tFn(dom[1]));
      let yr = Y(rFn(dom[1]));
      if (Math.abs(yr - yt) < 14) yr = yt + (yr >= yt ? 14 : -14);
      C.el('text', { x: w - m.r + 8, y: yt, class: 'viz-label', fill: `rgb(${S.RGB.amber})`, 'dominant-baseline': 'middle' }, svg, 'true world');
      C.el('text', { x: w - m.r + 8, y: yr, class: 'viz-label', fill: `rgb(${S.RGB.belief})`, 'dominant-baseline': 'middle' }, svg, 'rollout');
      if (showMarker) {
        C.el('line', { x1: X(f), x2: X(f), y1: m.t, y2: h - m.b, class: 'viz-cross' }, svg);
        [[tFn, S.RGB.amber], [rFn, S.RGB.belief]].forEach(([fn, rgb]) => {
          C.el('circle', { cx: X(f), cy: Y(fn(f)), r: 4.5, fill: `rgb(${rgb})`, stroke: 'var(--viz-surface)', 'stroke-width': 2 }, svg);
          C.el('circle', { cx: X(1), cy: Y(fn(1)), r: 3, fill: 'none', stroke: `rgb(${rgb})`, 'stroke-width': 1.5 }, svg);
        });
      }
    }

    function renderPert(e) {
      const card = els.cards.pert;
      const scored = state.param === 'R';
      const res = scored ? P.perturbR(program(), state.pertR) : P.perturbField(program(), state.fieldK);
      if (e < TL.pert) setStatus(card, 'idle');
      else if (e < TL.pertDone) setStatus(card, 'running');
      else setStatus(card, res.pass ? 'pass' : 'fail');
      card.classList.toggle('is-explore', !scored);
      if (e >= TL.pertDone) card.querySelector('.pcard__status').textContent = `${res.pass ? 'pass' : 'fail'}${scored ? '' : ' · unscored'}`;

      const live = e >= TL.pertDone;
      els.range.disabled = !live;
      els.params.forEach((b) => (b.disabled = !live));
      const target = scored ? state.pertR : state.fieldK;
      const p = scored ? ramp(e, TL.pert, TL.pertMorphEnd) : 1;
      drawResponse(1 + (target - 1) * p, e >= TL.pert);
      setHTML(els.rangeLabel, scored
        ? `<i>R</i> → ${state.pertR.toFixed(2)}<i>R</i>`
        : `<i>qE</i> → ${state.fieldK.toFixed(2)}·<i>qE</i>`);
      let why = '';
      if (state.caseId === 1) why = 'The curves coincide: the rollout responds exactly like the true world.';
      else if (scored) why = 'Same shape, lower curve. The probe compares how <i>v<sub>O</sub></i> responds, not its value, so Case 2 passes at every <i>R</i>. Catching the wrong value is the invariant probe’s job.';
      else why = 'Flat line: Case 2’s program never uses the field, while the true world speeds up with <i>qE</i>. This probe exposes the error.';
      setHTML(els.pertRead, e < TL.pert
        ? 'Change one input, re-run the same program, and compare with how the true world responds.'
        : `Ratio test: true world ×${res.expected.toFixed(3)}, rollout ×${res.observed.toFixed(3)}. ${why}`);
      els.pertNote.hidden = scored;
    }

    function renderRS(e) {
      const r = state.result;
      const order = ['inv', 'trace', 'pert'];
      order.forEach((k, i) => {
        const row = els.rsRows[k];
        const p = ramp(e, TL.rs + i * 150, TL.rs + i * 150 + 400);
        row.querySelector('b').style.width = `${r.scores[k] * p * 100}%`;
        row.querySelector('em').textContent = e >= TL.rs + i * 150 + 400 ? r.scores[k].toFixed(0) : '–';
        row.classList.toggle('is-zero', e >= TL.rs && r.scores[k] === 0);
      });
      const p = ramp(e, TL.rs, TL.rsEnd);
      els.rsValue.textContent = e >= TL.rs ? (r.RS * p).toFixed(2) : '—';
      els.rsValue.parentElement.dataset.grade = e >= TL.rsEnd ? (r.RS >= 0.99 ? 'pass' : 'fail') : '';
    }

    function renderInsight(e) {
      const on = e >= TL.insight;
      let html = '';
      if (state.caseId === 1) {
        html = '<b>Every probe passes.</b> Under every test, the rollout behaves like the true world, so <i>R</i><sub>S</sub> = 1.';
      } else if (state.param === 'qE') {
        html = '<b>Turn the field knob.</b> The true world speeds the ball up or slows it down, but Case 2’s world does not react: it never used the electric work. A world that can be probed in more ways exposes more errors.';
      } else {
        html = '<b>One probe is not enough.</b> Case 2 is self-consistent and even scales correctly with <i>R</i>, so the trace and perturbation probes pass. Only the work–energy invariant catches the missing horizontal electric work (Δ<i>KE</i> = 0 ≠ <i>mgR</i>).';
      }
      els.insight.classList.toggle('is-on', on);
      setHTML(els.insight, html);
    }

    function renderPanel(e) {
      renderTerminal(e);
      renderEmit(e);
      renderInvariants(e);
      renderTrace(e);
      renderPert(e);
      renderRS(e);
      renderInsight(e);
    }

    /* ------------------------------- canvas ------------------------------- */

    // Target world shown on the canvas: nominal, except while perturbing.
    function worldTarget(e, now) {
      if (e < TL.pert) return { R: 1, k: 1 };
      if (e < TL.rs) return { R: state.param === 'R' ? state.pertR : 1, k: 1 };
      if (state.param === 'qE') return { R: 1, k: state.fieldK };
      return { R: now - state.lastInput < 1600 ? state.pertR : 1, k: 1 };
    }

    // Launch-speed readout in the top-left corner of the canvas.
    function drawReadout(v, vTrue, vModel, alpha) {
      const ctx = v.ctx;
      const x = 14, y = 14, w = 196, h = 70;
      ctx.save();
      ctx.globalAlpha *= alpha;
      ctx.fillStyle = 'rgba(255,255,255,0.92)';
      ctx.strokeStyle = 'rgba(71,85,105,0.3)';
      ctx.lineWidth = 1;
      S.roundRect(ctx, x, y, w, h, 9);
      ctx.fill();
      ctx.stroke();
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'left';
      ctx.font = "500 11px 'JetBrains Mono', ui-monospace, monospace";
      ctx.fillStyle = 'rgba(51,65,85,0.9)';
      ctx.fillText('launch speed at O  [√(gR)]', x + 11, y + 16);
      ctx.font = "600 12.5px 'JetBrains Mono', ui-monospace, monospace";
      ctx.fillStyle = `rgba(${S.RGB.amber},1)`;
      ctx.fillText('true world', x + 11, y + 37);
      ctx.fillText(fmt(vTrue), x + w - 50, y + 37);
      ctx.fillStyle = `rgba(${S.RGB.belief},1)`;
      ctx.fillText('rollout', x + 11, y + 56);
      ctx.fillText(fmt(vModel), x + w - 50, y + 56);
      ctx.restore();
    }

    function renderCanvas(e, time) {
      const v = view;
      v.clear();
      drawGrid(v);
      const R = state.dispR;
      const k = state.dispK;
      const geo = P.geometry({ R });
      const G = P.givens({ R, k });
      const tw = P.trueWorld(G);
      const cand = program().run(G);
      const st = ST.blueprint;
      const arcExclude = (x, y) => (x >= geo.C.x - 0.1 && y <= 0.15 && Math.abs(Math.hypot(x - geo.C.x, y - geo.C.y) - geo.R) < 0.24) || Math.hypot(x, y) < 0.35;

      S.shadeField(v, geo.fieldX, st);
      S.drawField(v, geo.fieldX, time, st, { strength: k, exclude: arcExclude, spacing: 0.7 });
      S.drawTrack(v, geo, st);
      S.drawSpring(v, geo, null, 0.45, st);
      S.drawPoint(v, geo.B, st);
      S.drawPoint(v, geo.O, st);
      S.label(v, 'A', geo.A, st.label, { dy: 18 });
      S.label(v, 'B', geo.B, st.label, { dy: 18 });
      S.label(v, 'O', geo.O, st.label, { dx: -13, dy: 13 });

      // Law templates bound to this problem.
      const tagA = ramp(e, 0, 900);
      const arcMid = P.arcPoint(geo, Math.PI / 4);
      const trueAt = P.trajectoryPoint(tw, 0.8);
      S.tag(v, 'energy conservation', { x: -2.75, y: 0.05 }, { x: geo.A.x - 0.15, y: geo.A.y + 0.12 }, { alpha: tagA });
      S.tag(v, 'work–energy theorem', { x: -1.75, y: 1.05 }, arcMid, { alpha: tagA });
      S.tag(v, 'Newton II · F = qE', { x: 1.5, y: -0.45 }, { x: 0.85, y: -1.0 }, { alpha: tagA });
      S.tag(v, 'kinematics', { x: 1.75, y: 0.55 }, trueAt, { alpha: tagA });

      // Responses: the true world (amber) and the rollout's program (cyan).
      const fly = e >= TL.done ? 1 : ramp(e, TL.fly, TL.flyEnd);
      if (e >= TL.fly) {
        const trueCurve = trajectory(tw);
        const candCurve = trajectory(cand);
        const cut = (pts) => pts.slice(0, Math.max(2, Math.ceil(pts.length * fly)));
        const tc = cut(trueCurve);
        const cc = cut(candCurve);
        S.drawCurve(v, tc, `rgba(${S.RGB.amber},0.9)`, { width: 2, glow: 8, glowColor: `rgba(${S.RGB.amber},0.6)` });
        S.drawCurve(v, cc, `rgba(${S.RGB.belief},0.95)`, { width: 2, dash: [7, 5], glow: 12, glowColor: `rgba(${S.RGB.belief},0.8)` });
        if (fly < 1) {
          S.drawBall(v, tc[tc.length - 1], ST.true, { r: 0.075 });
          S.drawBall(v, cc[cc.length - 1], ST.belief, { r: 0.075 });
        }
      }

      // Launch velocities at O.
      if (e >= TL.emit) {
        const a = ramp(e, TL.emit, TL.emit + 400);
        S.drawArrow(v, { x: 0.07, y: 0 }, { x: 0.07, y: tw.vO * ARROW_SCALE }, `rgba(${S.RGB.amber},0.95)`, { width: 2.2, alpha: a, glow: 6 });
        S.drawArrow(v, { x: -0.07, y: 0 }, { x: -0.07, y: cand.vO * ARROW_SCALE }, `rgba(${S.RGB.belief},0.95)`, { width: 2.2, alpha: a, glow: 8 });
        drawReadout(v, tw.vO, cand.vO, a);
      }

      // Invariant probe: the work done along the arc.
      if (e >= TL.inv) {
        const a = ramp(e, TL.inv, TL.inv + 400) * (e >= TL.trace ? 0.55 : 1);
        const ctx = v.ctx;
        ctx.save();
        ctx.globalAlpha = a;
        ctx.strokeStyle = state.result.invPass ? `rgba(${S.RGB.good},0.9)` : `rgba(${S.RGB.bad},0.9)`;
        ctx.shadowColor = ctx.strokeStyle;
        ctx.shadowBlur = 14 * S.GLOW;
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.arc(v.X(geo.C.x), v.Y(geo.C.y), v.L(geo.R), Math.PI / 2, 0, true);
        ctx.stroke();
        ctx.restore();
        const mono = "500 11.5px 'JetBrains Mono', ui-monospace, monospace";
        const work = state.result.invariants.find((c) => c.key === 'work');
        S.label(v, `W_g + W_E = ${fmt(work.expected)} mgR`, { x: -0.25, y: -1.5 }, `rgba(${S.RGB.amber},1)`, { font: mono, alpha: a });
        S.label(v, `ΔKE (rollout) = ${fmt(work.model)} mgR`, { x: -0.25, y: -1.72 }, `rgba(${S.RGB.belief},1)`, { font: mono, alpha: a });
      }

      // Trace probe: points recomputed from the rollout's own quantities.
      if (e >= TL.trace) {
        const a = ramp(e, TL.trace, TL.traceDone);
        const ctx = v.ctx;
        ctx.save();
        ctx.globalAlpha = a;
        for (let i = 1; i <= 8; i++) {
          const p = P.trajectoryPoint(cand, i * 0.25);
          if (p.x > BOUNDS.xmax || p.y > BOUNDS.ymax) break;
          ctx.fillStyle = 'rgba(30,41,59,0.85)';
          ctx.beginPath();
          ctx.arc(v.X(p.x), v.Y(p.y), 2.4, 0, Math.PI * 2);
          ctx.fill();
        }
        const pe = P.trajectoryPoint(cand, 1.0);
        ctx.strokeStyle = 'rgba(30,41,59,0.85)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(v.X(pe.x), v.Y(pe.y), 7, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }
    }

    /* ------------------------------ run loop ------------------------------ */

    function startRun() {
      computeResult();
      state.start = performance.now();
      state.elapsed = 0;
      state.started = true;
      if (reduced) state.elapsed = TL.done;
    }

    function finishRun() {
      state.elapsed = TL.done;
      state.start = performance.now() - TL.done;
    }

    function frame(now) {
      if (state.visible) {
        if (state.started && state.elapsed < TL.done) state.elapsed = Math.min(TL.done, now - state.start);
        const e = state.started ? state.elapsed : -1;
        const dt = state.lastFrame === null ? 1 : Math.min(0.1, (now - state.lastFrame) / 1000);
        state.lastFrame = now;
        const target = worldTarget(e, now);
        const f = reduced ? 1 : 1 - Math.exp(-dt / 0.16);
        state.dispR += (target.R - state.dispR) * f;
        state.dispK += (target.k - state.dispK) * f;
        renderCanvas(e, now / 1000);
        renderPanel(e);
      }
      root.requestAnimationFrame(frame);
    }

    els.cases.forEach((b) => {
      b.addEventListener('click', () => {
        state.caseId = +b.dataset.case;
        els.cases.forEach((x) => x.setAttribute('aria-selected', String(x === b)));
        state.param = 'R';
        els.params.forEach((x) => x.setAttribute('aria-checked', String(x.dataset.param === 'R')));
        setRange();
        startRun();
      });
    });

    function setRange() {
      if (state.param === 'R') {
        els.range.min = '0.7';
        els.range.max = '1.4';
        els.range.value = String(state.pertR);
      } else {
        els.range.min = '0.5';
        els.range.max = '1.5';
        els.range.value = String(state.fieldK);
      }
    }

    els.params.forEach((b) => {
      b.addEventListener('click', () => {
        state.param = b.dataset.param;
        els.params.forEach((x) => x.setAttribute('aria-checked', String(x === b)));
        if (state.param === 'qE' && Math.abs(state.fieldK - 1) < 1e-6) state.fieldK = 1.3;
        setRange();
      });
    });

    els.range.addEventListener('input', () => {
      const x = parseFloat(els.range.value);
      state.lastInput = performance.now();
      if (state.param === 'R') {
        state.pertR = x;
        computeResult();
      } else {
        state.fieldK = x;
      }
    });

    els.rerun.addEventListener('click', startRun);

    els.cases.forEach((x) => x.setAttribute('aria-selected', String(+x.dataset.case === state.caseId)));
    computeResult();
    setRange();

    if (params.has('probeDone')) {
      startRun();
      finishRun();
    }

    if ('IntersectionObserver' in root) {
      new IntersectionObserver((entries) => {
        state.visible = entries[0].isIntersecting;
        if (state.visible && !state.started && entries[0].intersectionRatio > 0.3) startRun();
      }, { threshold: [0, 0.3, 0.6] }).observe(section.querySelector('.probe'));
    } else {
      state.visible = true;
      startRun();
    }

    let resizeTimer = null;
    root.addEventListener('resize', () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => view.resize(), 120);
    });

    root.requestAnimationFrame(frame);
  }

  root.AWM.initProbe = init;
})(typeof self !== 'undefined' ? self : this);
