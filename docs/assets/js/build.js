/* M3: synthesizing the checker, i.e. building the true world of one problem in code. */
(function (root) {
  'use strict';

  const ENTS = [
    ['ball', 'm, q > 0'],
    ['spring', 'light · stores E_p at A'],
    ['track AB', 'horizontal · smooth'],
    ['arc BO', 'radius R · smooth'],
    ['field', 'E at 45° · x ≥ −R'],
    ['givens', 'v_B = √(gR) · |qE| = √2 mg'],
    ['asked', 'E_p · v_O · trajectory'],
  ];

  // Tiles are examples of laws named in the paper; the bound ones carry the entity they attach to.
  const LIB = [
    ['energy conservation', 'spring · S1'],
    ['work–energy theorem', 'arc BO · S2'],
    ["Newton's second law", 'field · S3'],
    ['force decomposition', 'qE cos 45°, qE sin 45°'],
    ['kinematics', 'flight · S3'],
    ['momentum conservation'],
    ['charge conservation'],
    ["Coulomb's law"],
    ["Ohm's law"],
    ["Kirchhoff's laws"],
    ["Gauss's law"],
    ['continuity equation'],
    ['first law of thermodynamics'],
    ['ideal gas law'],
    ['adiabatic relation'],
    ['series RLC impedance'],
  ];

  const CODE = [
    ['class PhysicsChecker:', ''],
    ['  def __init__(self, sol):', '# variable contract'],
    ['    self.E_p = sol.get("E_p")  # None if missing', ''],
    ['    self.v_O = sol.get("v_O") ...', ''],
    ['  def test_invariant_spring_energy', '# E_p = ½mgR'],
    ['  def test_invariant_arc_work_energy', '# ΔKE = W_g + W_E'],
    ['  def test_invariant_horizontal_acc', '# a_x = g'],
    ['  def test_invariant_vertical_acc', '# a_y = 0'],
    ['  def test_trace_consistency', '# y(x) from own a, v_O'],
    ['  def test_response_under_perturbations', '# 1.1R ⇒ v_O × √1.1'],
  ];
  const TESTS = CODE.map((c, i) => (c[0].includes('def test_') ? i : -1)).filter((i) => i >= 0);

  const TL = { ents: 0, entsEnd: 1700, bind: 1800, bindEnd: 3900, emit: 4000, emitEnd: 6600, valid: 6700, validEnd: 8300, done: 8500 };

  function esc(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function init() {
    const box = document.getElementById('build');
    if (!box) return;
    const steps = [...box.querySelectorAll('[data-bstep]')];
    const entsEl = document.getElementById('buildEnts');
    const libEl = document.getElementById('buildLib');
    const codeEl = document.getElementById('buildCode');
    const reduced = root.matchMedia && root.matchMedia('(prefers-reduced-motion: reduce)').matches;

    entsEl.innerHTML = ENTS.map(([k, v]) => `<li><b>${k}</b><span>${v}</span></li>`).join('');
    libEl.innerHTML = LIB.map(([name, bind]) => `<span class="libtile${bind ? ' is-bindable' : ''}">${name}${bind ? `<em>→ ${bind}</em>` : ''}</span>`).join('') + '<span class="libtile libtile--more">+131 more</span>';
    const entEls = [...entsEl.children];
    const tileEls = [...libEl.querySelectorAll('.is-bindable')];

    let start = null;
    let running = false;
    let last = '';

    function render(e) {
      const stage = e >= TL.valid ? 3 : e >= TL.emit ? 2 : e >= TL.bind ? 1 : 0;
      steps.forEach((s, i) => {
        s.classList.toggle('is-active', i === stage && e < TL.done);
        s.classList.toggle('is-done', i < stage || e >= TL.done);
      });
      entEls.forEach((li, i) => li.classList.toggle('is-on', e >= TL.ents + i * 220));
      tileEls.forEach((t, i) => t.classList.toggle('is-bound', e >= TL.bind + 250 + i * 380));
      libEl.classList.toggle('is-binding', e >= TL.bind);

      // code: lines appear during "emit"; test lines get a pass mark during "valid"
      const per = (TL.emitEnd - TL.emit) / CODE.length;
      let html = '';
      CODE.forEach(([code, com], i) => {
        if (e < TL.emit + i * per) return;
        const ti = TESTS.indexOf(i);
        const ok = ti >= 0 && e >= TL.valid + 150 + ti * 230;
        html += `<span class="cl${ok ? ' is-ok' : ''}"><b class="cl__gut">${ok ? '✓' : ''}</b>${esc(code)}${com ? ` <i>${esc(com)}</i>` : ''}</span>`;
      });
      if (e >= TL.validEnd) html += '<span class="cl cl--final"><b class="cl__gut"></b>→ all tests pass · validated in a sandbox · frozen for training</span>';
      if (html !== last) {
        codeEl.innerHTML = html;
        last = html;
      }
    }

    function frame(now) {
      if (running) {
        const e = now - start;
        render(Math.min(e, TL.done));
        if (e >= TL.done) running = false;
      }
      root.requestAnimationFrame(frame);
    }

    function run() {
      if (reduced) {
        render(TL.done);
        return;
      }
      start = performance.now();
      running = true;
    }

    document.getElementById('buildRun').addEventListener('click', run);
    render(-1);
    let started = false;
    if ('IntersectionObserver' in root) {
      new IntersectionObserver((en) => {
        if (en[0].isIntersecting && !started) {
          started = true;
          run();
        }
      }, { threshold: 0.35 }).observe(box);
    } else run();
    root.requestAnimationFrame(frame);
  }

  root.AWM.initBuild = init;
})(typeof self !== 'undefined' ? self : this);
