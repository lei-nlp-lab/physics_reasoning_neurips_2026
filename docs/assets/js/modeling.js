/*
 * Stage I of the method.
 * M1: the [modeling] block, with each field linked to what it pins down in the believed world.
 * M2: PMF, comparing extracted motifs of the believed world with the true one.
 */
(function (root) {
  'use strict';
  const { physics: P, scene: S } = root.AWM;
  const ST = S.STYLES;
  const BOUNDS = { xmin: -3.4, xmax: 2.05, ymin: -2.32, ymax: 2.7 };
  const MONO = "500 11.5px 'JetBrains Mono', ui-monospace, monospace";
  const SERIF = "italic 600 15px 'Times New Roman', serif";
  const RED = `rgba(${S.RGB.bad},1)`;
  const CYAN = `rgba(${S.RGB.belief},1)`;

  /* ------------------------------- M1 data ------------------------------- */
  const FIELDS = ['objects', 'constraints', 'representation', 'mapping', 'plan'];
  const BLOCKS = {
    faithful: {
      objects: ['ball: mass m, charge q > 0', 'spring: light, compressed at A', 'track: AB smooth, horizontal · BO smooth quarter arc, radius R', 'field: uniform, 45° to +x, for x ≥ −R · |F_E| = √2 mg'],
      constraints: ['AB horizontal ⇒ h_A = h_B', 'smooth rail ⇒ the normal force does no work', 'v_B = √(gR) · leaves O vertically upward'],
      representation: ['timeline · S1: A → B · S2: B → O · S3: after O'],
      mapping: ['[energy conservation] E_p = ½ m v_B²', '[work–energy] W_g + W_E = ½ m v_O² − ½ m v_B²', '[Newton II] m a_x = qE cos 45°, m a_y = qE sin 45° − mg', '[kinematics] x = ½ a_x t², y = v_O t + ½ a_y t²'],
      plan: ['Q1: E_p on S1 · Q2: v_O on S2, using v_B · Q3: trajectory on S3, using v_O'],
    },
    drifted: {
      objects: ['ball: mass m, charge q > 0', 'spring: light, compressed at A', 'track: AB smooth · BO smooth quarter arc, radius R', 'field: uniform, 45° to +x · |F_E| = √2 mg'],
      constraints: ['!assumes h_B − h_A = R', 'smooth rail ⇒ the normal force does no work', 'v_B = √(gR)', '!then sets E = 0 to remove v_O² < 0'],
      representation: ['!timeline · S1: A → B, rising R · S2: B → O · S3: after O'],
      mapping: ['![energy] (3/2) mgR + ▢ = ½ m v_O² + mgR + mgR', '!⇒ v_O² < 0 ⇒ contradiction ⇒ E ← 0'],
      plan: ['!Q2: v_O by energy conservation, with the field ignored'],
    },
  };

  function renderBlock(el, belief, active) {
    const b = BLOCKS[belief];
    el.innerHTML = `<div class="block__bar"><span>[modeling]</span><em>${belief === 'faithful' ? 'faithful belief' : 'drifted belief · Figure 1'}</em></div>` + FIELDS.map((f) => {
      const lines = b[f].map((l) => {
        const bad = l.startsWith('!');
        return `<li class="${bad ? 'is-bad' : ''}">${bad ? l.slice(1) : l}${bad ? '<b>✗</b>' : ''}</li>`;
      }).join('');
      return `<div class="block__row${active === f ? ' is-active' : ''}" data-field="${f}" tabindex="0"><span class="block__key">[${f}]</span><ul>${lines}</ul></div>`;
    }).join('');
  }

  /* ------------------------------- M1 scene ------------------------------ */
  function drawM1(v, belief, field, time) {
    v.clear();
    const drifted = belief === 'drifted';
    const geo = P.geometry({ drop: drifted ? 1 : 0 });
    const st = ST.belief;
    if (!drifted) {
      S.shadeField(v, geo.fieldX, st);
      S.drawField(v, geo.fieldX, time, st, { exclude: (x, y) => (x >= geo.C.x - 0.1 && y <= 0.15 && Math.abs(Math.hypot(x - geo.C.x, y - geo.C.y) - geo.R) < 0.24) || Math.hypot(x, y) < 0.3 });
    }
    S.drawTrack(v, geo, st);
    S.drawPoint(v, geo.B, st);
    S.drawPoint(v, geo.O, st);
    S.label(v, 'A', geo.A, st.label, { dy: 18 });
    S.label(v, 'B', geo.B, st.label, { dy: 18 });
    S.label(v, 'O', geo.O, st.label, { dx: -13, dy: -12 });
    S.drawSpring(v, geo, 0, 0.45, st);
    const ball = S.ballCenter(geo, { x: geo.A.x, y: geo.A.y, phase: 'spring', s: 0 });
    S.drawBall(v, ball, st);

    // The path the model believes the ball takes.
    let path = [];
    if (!drifted) {
      for (let t = 0; t <= 1.95; t += 0.03) path.push({ x: 0.5 * t * t, y: Math.sqrt(3) * t });
    } else {
      for (let phi = 0; phi <= Math.PI / 3; phi += 0.03) path.push(P.arcPoint(geo, phi));
    }
    S.drawCurve(v, path, `rgba(${S.RGB.belief},0.75)`, { width: 1.6, dash: [5, 5], glow: 8, glowColor: `rgba(${S.RGB.belief},0.7)` });
    if (drifted) {
      const ctx = v.ctx;
      ctx.save();
      ctx.strokeStyle = RED;
      ctx.shadowColor = RED;
      ctx.shadowBlur = 10 * S.GLOW;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(v.X(0), v.Y(0), v.L(0.16), 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }

    const A = (f) => (field === f ? 1 : 0);
    // objects
    if (A('objects')) {
      S.tag(v, 'ball · m, q > 0', { x: -2.75, y: drifted ? -1.35 : -0.25 }, ball, { color: CYAN });
      S.tag(v, 'spring', { x: -3.0, y: drifted ? -0.85 : 0.3 }, { x: geo.A.x - 0.3, y: geo.A.y + 0.12 }, { color: CYAN });
      S.tag(v, 'track AB · arc BO (R)', { x: -1.75, y: drifted ? 0.2 : 0.55 }, P.arcPoint(geo, 0.6), { color: CYAN });
      S.tag(v, drifted ? 'field: deleted (E = 0)' : 'field E · 45°', { x: 1.25, y: -1.5 }, drifted ? null : { x: 1.0, y: -0.9 }, { color: drifted ? RED : CYAN });
    }
    // constraints
    if (A('constraints')) {
      if (!drifted) {
        S.tag(v, 'AB horizontal: h_A = h_B', { x: -1.85, y: -0.55 }, { x: -1.85, y: -1 }, { color: CYAN });
        S.tag(v, 'smooth: N does no work', { x: 0.95, y: -0.4 }, P.arcPoint(geo, 1.0), { color: CYAN });
        S.tag(v, 'v_B = √(gR)', { x: -0.85, y: -1.62 }, geo.B, { color: CYAN });
        S.drawArrow(v, { x: 0, y: 0.05 }, { x: 0, y: 0.75 }, CYAN, { width: 2, glow: 8 });
        S.tag(v, 'leaves O vertically', { x: 0.95, y: 0.55 }, { x: 0.02, y: 0.45 }, { color: CYAN });
      } else {
        const xd = geo.B.x + 0.24;
        S.drawArrow(v, { x: xd, y: -1.5 }, { x: xd, y: -1 }, RED, { width: 1.3, head: 7 });
        S.drawArrow(v, { x: xd, y: -1.5 }, { x: xd, y: -2 }, RED, { width: 1.3, head: 7 });
        S.tag(v, 'h_B − h_A = R  ✗', { x: 0.55, y: -1.5 }, { x: xd, y: -1.5 }, { color: RED });
        S.tag(v, 'E = 0  ✗', { x: 1.25, y: -0.4 }, null, { color: RED });
        S.tag(v, 'v_B = √(gR)', { x: -1.95, y: -0.7 }, geo.B, { color: CYAN });
      }
    }
    // representation: timeline bands
    if (A('representation')) {
      const band = (pts, label, at, color = `rgba(${S.RGB.belief},0.28)`) => {
        S.drawCurve(v, pts, color, { width: 12 });
        S.tag(v, label, at, null, { color: CYAN });
      };
      const wall = { x: geo.A.x, y: geo.A.y };
      band([wall, geo.B], 'S1  A → B', { x: (geo.A.x + geo.B.x) / 2, y: (geo.A.y + geo.B.y) / 2 + 0.45 });
      const arc = [];
      for (let phi = 0; phi <= (drifted ? Math.PI / 3 : Math.PI / 2) + 1e-9; phi += 0.05) arc.push(P.arcPoint(geo, phi));
      band(arc, 'S2  B → O', { x: 0.45, y: -1.05 });
      if (!drifted) {
        band(path, 'S3  after O', { x: 1.45, y: 1.55 });
      } else {
        S.tag(v, 'S3  never reached', { x: 1.1, y: 0.6 }, null, { color: RED });
      }
    }
    // mapping: equations attached to stages
    if (A('mapping')) {
      if (!drifted) {
        S.tag(v, 'E_p = ½ m v_B²', { x: -2.35, y: -0.35 }, { x: -2.2, y: -0.92 }, { color: CYAN });
        S.tag(v, 'W_g + W_E = ½mv_O² − ½mv_B²', { x: -0.55, y: 0.85 }, P.arcPoint(geo, 0.9), { color: CYAN });
        S.tag(v, 'm a_x = qE cos45°, a_y = 0', { x: 1.05, y: 2.3 }, path[Math.floor(path.length * 0.55)], { color: CYAN });
      } else {
        S.tag(v, '(3/2)mgR + ▢ = ½mv_O² + 2mgR', { x: -0.75, y: 0.8 }, P.arcPoint(geo, 0.8), { color: RED });
        S.tag(v, '⇒ v_O² < 0  ⇒  E ← 0', { x: 0.9, y: 0.3 }, { x: 0.12, y: 0.08 }, { color: RED });
      }
    }
    // plan: sub-questions placed on the world
    if (A('plan')) {
      const badge = (text, at, color) => S.tag(v, text, at, null, { color, font: "600 12px 'JetBrains Mono', ui-monospace, monospace" });
      badge('Q1 · E_p', { x: geo.A.x + 0.35, y: geo.A.y + 0.55 }, CYAN);
      badge(drifted ? 'Q2 · v_O = ?' : 'Q2 · v_O', { x: -0.55, y: 0.35 }, drifted ? RED : CYAN);
      if (!drifted) badge('Q3 · trajectory', { x: 1.25, y: 1.15 }, CYAN);
    }
    if (!field) S.label(v, 'hover a field of the block →', { x: -1.3, y: 2.35 }, 'rgba(71,85,105,0.8)', { font: MONO });
    void SERIF;
  }

  function initM1() {
    const sec = document.getElementById('m1');
    if (!sec) return;
    const blockEl = document.getElementById('m1Block');
    const view = S.createView(document.getElementById('m1Canvas'), BOUNDS);
    const state = { belief: 'faithful', field: 'objects', pinned: false, visible: false, auto: true, lastSwitch: 0 };
    const tabs = [...sec.querySelectorAll('[data-belief]')];
    const reduced = root.matchMedia && root.matchMedia('(prefers-reduced-motion: reduce)').matches;

    function setField(f, byUser) {
      state.field = f;
      if (byUser) state.auto = false;
      renderBlock(blockEl, state.belief, state.field);
    }
    blockEl.addEventListener('mouseover', (e) => {
      const row = e.target.closest('[data-field]');
      if (row && row.dataset.field !== state.field) setField(row.dataset.field, true);
    });
    blockEl.addEventListener('focusin', (e) => {
      const row = e.target.closest('[data-field]');
      if (row) setField(row.dataset.field, true);
    });
    tabs.forEach((b) => b.addEventListener('click', () => {
      state.belief = b.dataset.belief;
      tabs.forEach((x) => x.setAttribute('aria-selected', String(x === b)));
      renderBlock(blockEl, state.belief, state.field);
    }));
    renderBlock(blockEl, state.belief, state.field);

    function frame(now) {
      if (state.visible) {
        if (state.auto && !reduced && now - state.lastSwitch > 2600) {
          state.lastSwitch = now;
          const i = FIELDS.indexOf(state.field);
          setField(FIELDS[(i + 1) % FIELDS.length], false);
        }
        drawM1(view, state.belief, state.field, now / 1000);
      }
      root.requestAnimationFrame(frame);
    }
    if ('IntersectionObserver' in root) {
      new IntersectionObserver((en) => {
        state.visible = en[0].isIntersecting;
        if (state.visible) state.lastSwitch = performance.now();
      }, { threshold: 0.2 }).observe(sec);
    } else state.visible = true;
    root.addEventListener('resize', () => view.resize());
    root.requestAnimationFrame(frame);
  }

  /* ------------------------------- M2 data ------------------------------- */
  const REF = {
    qel: [
      ['e_p', 'energy_conservation', 'release_ab'],
      ['v_b', 'energy_conservation', 'release_ab'],
      ['w_g', 'work_energy_theorem', 'arc_bo'],
      ['w_e', 'work_energy_theorem', 'arc_bo'],
      ['v_o', 'work_energy_theorem', 'arc_bo'],
      ['a_x', 'newton_second_law', 'after_o'],
      ['a_y', 'newton_second_law', 'after_o'],
      ['y', 'kinematics', 'after_o'],
    ],
    vcl: [
      ['energy_conservation@release_ab', 'v_b', 'work_energy_theorem@arc_bo'],
      ['work_energy_theorem@arc_bo', 'v_o', 'kinematics@after_o'],
      ['newton_second_law@after_o', 'a_x', 'kinematics@after_o'],
      ['newton_second_law@after_o', 'a_y', 'kinematics@after_o'],
    ],
    aeb: [
      ['horizontal_ab', 'energy_conservation', 'release_ab'],
      ['smooth_track', 'work_energy_theorem', 'arc_bo'],
      ['uniform_field', 'work_energy_theorem', 'arc_bo'],
      ['uniform_field', 'newton_second_law', 'after_o'],
    ],
  };
  const DRIFTED = {
    qel: [
      ['e_p', 'energy_conservation', 'release_ab'],
      ['v_b', 'energy_conservation', 'release_ab'],
      ['h_a', 'energy_conservation', 'release_ab'],
      ['v_o', 'energy_conservation', 'arc_bo'],
      ['a_y', 'newton_second_law', 'after_o'],
      ['y', 'kinematics', 'after_o'],
    ],
    vcl: [
      ['energy_conservation@release_ab', 'v_b', 'energy_conservation@arc_bo'],
      ['energy_conservation@arc_bo', 'v_o', 'kinematics@after_o'],
      ['newton_second_law@after_o', 'a_y', 'kinematics@after_o'],
    ],
    aeb: [
      ['raised_a', 'energy_conservation', 'release_ab'],
      ['smooth_track', 'energy_conservation', 'arc_bo'],
      ['no_field', 'energy_conservation', 'arc_bo'],
    ],
  };
  const clone = (m) => ({ qel: m.qel.map((t) => [...t]), vcl: m.vcl.map((t) => [...t]), aeb: m.aeb.map((t) => [...t]) });

  // Error types of Table 13, applied to the faithful belief.
  const ERRORS = {
    theorem: {
      name: 'Wrong theorem selected',
      expect: ['decreases', 'may decrease', 'unchanged'],
      apply(m) {
        m.qel = m.qel.map((t) => (t[1] === 'kinematics' ? [t[0], 'energy_conservation', t[2]] : t));
        m.vcl = m.vcl.map((t) => (t[2] === 'kinematics@after_o' ? [t[0], t[1], 'energy_conservation@after_o'] : t));
      },
    },
    variable: {
      name: 'Correct theorem but missing variable',
      expect: ['decreases', 'may decrease', 'unchanged'],
      apply(m) { m.qel = m.qel.filter((t) => t[0] !== 'w_e'); },
    },
    coupling: {
      name: 'Broken interaction between equations',
      expect: ['unchanged', 'decreases', 'unchanged'],
      apply(m) { m.vcl = m.vcl.filter((t) => t[1] !== 'v_o'); },
    },
    assumption: {
      name: 'Missed premise or assumption',
      expect: ['unchanged', 'unchanged', 'decreases'],
      apply(m) { m.aeb = m.aeb.filter((t) => !(t[0] === 'uniform_field' && t[2] === 'arc_bo')); },
    },
    stage: {
      name: 'Missed object or physical stage',
      expect: ['decreases', 'decreases', 'may decrease'],
      apply(m) {
        m.qel = m.qel.filter((t) => t[2] !== 'after_o');
        m.vcl = m.vcl.filter((t) => !t[0].endsWith('after_o') && !t[2].endsWith('after_o'));
        m.aeb = m.aeb.filter((t) => t[2] !== 'after_o');
      },
    },
  };

  const key = (t) => t.join('|');
  function score(ref, bel) {
    const refKeys = new Set(ref.map(key));
    const belKeys = new Set(bel.map(key));
    const matched = [...belKeys].filter((k) => refKeys.has(k)).length;
    const p = belKeys.size ? matched / belKeys.size : 0;
    const r = refKeys.size ? matched / refKeys.size : 0;
    return { p, r, f1: p + r ? (2 * p * r) / (p + r) : 0, refKeys, belKeys };
  }

  const fmtTuple = (type, t) => (type === 'vcl' ? `${t[0]} —${t[1]}→ ${t[2]}` : `(${t.join(', ')})`);
  const TYPES = [
    { id: 'qel', name: 'QEL', full: 'quantity · law · context', what: 'Is the right law applied to the right quantity?' },
    { id: 'vcl', name: 'VCL', full: 'variable coupling link', what: 'Are equations chained through the right shared variable?' },
    { id: 'aeb', name: 'AEB', full: 'assumption · law · context', what: 'Is every law used under its required assumptions?' },
  ];

  function initM2() {
    const sec = document.getElementById('m2');
    if (!sec) return;
    const colsEl = document.getElementById('pmfCols');
    const expectEl = document.getElementById('pmfExpect');
    const rmEl = document.getElementById('pmfRM');
    const presetBtns = [...sec.querySelectorAll('[data-pmf]')];
    const errBtns = [...sec.querySelectorAll('[data-err]')];
    const state = { preset: 'faithful', errors: new Set() };

    function belief() {
      if (state.preset === 'drifted') return clone(DRIFTED);
      const m = clone(REF);
      ['stage', 'theorem', 'variable', 'coupling', 'assumption'].forEach((e) => { if (state.errors.has(e)) ERRORS[e].apply(m); });
      return m;
    }

    function render() {
      const bel = belief();
      const scores = TYPES.map((T) => score(REF[T.id], bel[T.id]));
      const RM = scores.reduce((a, s) => a + s.f1, 0) / 3;
      colsEl.innerHTML = TYPES.map((T, i) => {
        const s = scores[i];
        const ref = REF[T.id].map((t) => {
          const ok = s.belKeys.has(key(t));
          return `<li class="${ok ? 'is-ok' : 'is-miss'}"><b>${ok ? '✓' : '✗'}</b><span>${fmtTuple(T.id, t)}</span></li>`;
        }).join('');
        const extra = bel[T.id].filter((t) => !s.refKeys.has(key(t))).map((t) => `<li class="is-extra"><b>+</b><span>${fmtTuple(T.id, t)}</span></li>`).join('');
        return `<div class="pmfcol">
          <div class="pmfcol__head"><div><strong>${T.name}</strong><span>${T.full}</span></div><em>F1 ${s.f1.toFixed(2)}</em></div>
          <p class="pmfcol__what">${T.what}</p>
          <div class="pmfcol__bar"><i style="width:${(s.f1 * 100).toFixed(1)}%"></i></div>
          <p class="pmfcol__pr">precision ${s.p.toFixed(2)} · recall ${s.r.toFixed(2)}</p>
          <ul class="motifs">${ref}${extra}</ul>
        </div>`;
      }).join('');
      rmEl.textContent = RM.toFixed(2);
      rmEl.parentElement.dataset.grade = RM > 0.95 ? 'pass' : RM > 0.6 ? 'mid' : 'fail';

      let html = '';
      if (state.preset === 'drifted') {
        html = '<b>Figure 1’s drift.</b> A wrong geometry and a deleted field: every motif type drops, and AEB collapses because every assumption the model wrote down is wrong.';
      } else if (!state.errors.size) {
        html = '<b>Faithful belief.</b> Every reference motif is matched: <i>R</i><sub>M</sub> = 1. <span class="legendinline"><b class="ok">✓</b> matched · <b class="miss">✗</b> missing from the belief · <b class="extra">+</b> extra tuple not in the true world</span>';
      } else {
        html = [...state.errors].map((e) => {
          const E = ERRORS[e];
          return `<b>${E.name}</b> (Table 13): QEL ${E.expect[0]} · VCL ${E.expect[1]} · AEB ${E.expect[2]}.`;
        }).join('<br>');
      }
      expectEl.innerHTML = html;
      errBtns.forEach((b) => {
        b.setAttribute('aria-pressed', String(state.errors.has(b.dataset.err)));
        b.disabled = state.preset === 'drifted';
      });
    }

    presetBtns.forEach((b) => b.addEventListener('click', () => {
      state.preset = b.dataset.pmf;
      if (state.preset === 'drifted') state.errors.clear();
      presetBtns.forEach((x) => x.setAttribute('aria-selected', String(x === b)));
      render();
    }));
    errBtns.forEach((b) => b.addEventListener('click', () => {
      const e = b.dataset.err;
      if (state.errors.has(e)) state.errors.delete(e);
      else state.errors.add(e);
      render();
    }));
    render();
  }

  root.AWM.initModeling = () => {
    initM1();
    initM2();
  };
})(typeof self !== 'undefined' ? self : this);
