/* M5: how GRPO turns the joint reward into a gradient. M6: scaffolding during training vs inference. */
(function (root) {
  'use strict';

  // Eight illustrative rollouts on one hard problem: every final answer is wrong.
  const ROLLOUTS = [
    { RM: 0.82, RS: 0.67, RA: 0, note: 'faithful world, one execution slip' },
    { RM: 0.29, RS: 0.0, RA: 0, note: 'drifted world, program does not run' },
    { RM: 0.64, RS: 0.33, RA: 0, note: 'minor modeling gap' },
    { RM: 0.71, RS: 0.67, RA: 0, note: 'faithful world, wrong final simplification' },
    { RM: 0.43, RS: 0.33, RA: 0, note: 'missed an assumption' },
    { RM: 0.29, RS: 0.0, RA: 0, note: 'drifted world' },
    { RM: 0.57, RS: 0.67, RA: 0, note: 'consistent derivation, wrong law choice' },
    { RM: 0.36, RS: 0.33, RA: 0, note: 'broken coupling between stages' },
  ];

  function initGrpo() {
    const box = document.getElementById('m5');
    if (!box) return;
    const table = document.getElementById('grpoTable');
    const verdict = document.getElementById('grpoVerdict');
    const btns = [...box.querySelectorAll('[data-reward]')];
    let mode = 'joint';

    function render() {
      const R = ROLLOUTS.map((r) => (mode === 'joint' ? 0.4 * r.RM + 0.4 * r.RS + 0.2 * r.RA : r.RA));
      const mean = R.reduce((a, b) => a + b, 0) / R.length;
      const std = Math.sqrt(R.reduce((a, b) => a + (b - mean) ** 2, 0) / R.length);
      const adv = R.map((x) => (std > 1e-9 ? (x - mean) / std : 0));
      const maxAdv = 2;
      let html = `<div class="gr gr--head"><span>rollout</span><span>R<sub>M</sub></span><span>R<sub>S</sub></span><span>R<sub>A</sub></span><span>reward R</span><span>advantage</span></div>`;
      ROLLOUTS.forEach((r, i) => {
        const a = Math.max(-maxAdv, Math.min(maxAdv, adv[i]));
        const off = mode === 'outcome';
        html += `<div class="gr${off ? ' is-off' : ''}" title="${r.note}">
          <span class="gr__id">#${i + 1}</span>
          <span class="gr__mini"><i style="width:${r.RM * 100}%" class="c-m"></i><em>${r.RM.toFixed(2)}</em></span>
          <span class="gr__mini"><i style="width:${r.RS * 100}%" class="c-s"></i><em>${r.RS.toFixed(2)}</em></span>
          <span class="gr__ra">✗</span>
          <span class="gr__mini gr__mini--r"><i style="width:${R[i] * 100}%" class="c-r"></i><em>${R[i].toFixed(2)}</em></span>
          <span class="gr__adv"><i class="${a >= 0 ? 'is-up' : 'is-down'}" style="width:${(Math.abs(a) / maxAdv) * 50}%;${a >= 0 ? 'left:50%' : `left:${50 - (Math.abs(a) / maxAdv) * 50}%`}"></i><em>${a >= 0 ? '+' : '−'}${Math.abs(a).toFixed(2)}</em></span>
        </div>`;
      });
      table.innerHTML = html;
      table.classList.toggle('is-outcome', mode === 'outcome');
      verdict.innerHTML = mode === 'outcome'
        ? '<b>No gradient.</b> Every reward is 0, so every advantage is 0. The group is silently dropped, together with the hard problem it came from.'
        : '<b>A usable gradient.</b> No answer is right, yet the rewards differ: rollouts that built a more faithful world, or stayed consistent with physics, are pushed up; drifted ones are pushed down.';
      verdict.classList.toggle('is-bad', mode === 'outcome');
    }

    btns.forEach((b) => b.addEventListener('click', () => {
      mode = b.dataset.reward;
      btns.forEach((x) => x.setAttribute('aria-selected', String(x === b)));
      render();
    }));
    render();
  }

  function initScaffold() {
    const box = document.getElementById('scaffold');
    if (!box) return;
    const fig = box.querySelector('.scaffold__fig');
    const cap = document.getElementById('scaffoldCap');
    const btns = [...box.querySelectorAll('[data-phase]')];
    fig.innerHTML = `
      <svg viewBox="0 0 760 240" role="img" aria-label="During training, the extractor, translator and checkers surround the policy; at inference only the policy remains.">
        <g class="sc-aux">
          <path class="sc-link" d="M215 62 C 290 62, 300 100, 318 108"/>
          <path class="sc-link" d="M215 178 C 290 178, 300 140, 318 132"/>
          <path class="sc-link" d="M545 62 C 470 62, 460 100, 442 108"/>
          <g class="sc-box sc-box--m" transform="translate(30 34)"><rect width="185" height="56" rx="12"/><text x="92" y="24">motif extractor</text><text x="92" y="42" class="sc-sub">gpt-4o-mini · PMF</text></g>
          <g class="sc-box sc-box--s" transform="translate(30 150)"><rect width="185" height="56" rx="12"/><text x="92" y="24">translator</text><text x="92" y="42" class="sc-sub">Qwen2.5-Coder-1.5B</text></g>
          <g class="sc-box sc-box--s" transform="translate(545 34)"><rect width="185" height="56" rx="12"/><text x="92" y="24">checkers</text><text x="92" y="42" class="sc-sub">one per training problem</text></g>
        </g>
        <path class="sc-out" d="M442 132 C 470 140, 480 178, 545 178"/>
        <g class="sc-box sc-box--p" transform="translate(318 84)"><rect width="124" height="72" rx="16"/><text x="62" y="32">policy</text><text x="62" y="51" class="sc-sub">with its AWM</text></g>
        <g class="sc-box sc-box--a" transform="translate(545 150)"><rect width="185" height="56" rx="12"/><text x="92" y="24">answer</text><text x="92" y="42" class="sc-sub">same cost as CoT</text></g>
      </svg>`;
    const caps = {
      train: 'During training, the policy is surrounded by the PMF extractor, the translator and one checker per problem. They only compute rewards.',
      infer: 'At inference the scaffolding is gone. The policy answers alone: no extractor, translator or checker is called.',
    };
    function set(phase) {
      box.classList.toggle('is-infer', phase === 'infer');
      cap.textContent = caps[phase];
      btns.forEach((x) => x.setAttribute('aria-selected', String(x.dataset.phase === phase)));
    }
    btns.forEach((b) => b.addEventListener('click', () => set(b.dataset.phase)));
    set('train');
  }

  root.AWM.initLoop = () => {
    initGrpo();
    initScaffold();
  };
})(typeof self !== 'undefined' ? self : this);
