/* Results: Table 1, SubPass@k and out-of-distribution transfer. */
(function (root) {
  'use strict';
  const { data: D, charts: C } = root.AWM;
  const COL = D.COLORS;
  const colorOf = (m) => COL[m] || COL.baseline;

  function tabs(container, items, initial, onPick) {
    container.innerHTML = '';
    items.forEach((it) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.setAttribute('role', 'tab');
      b.textContent = it;
      b.setAttribute('aria-selected', String(it === initial));
      b.addEventListener('click', () => {
        [...container.children].forEach((x) => x.setAttribute('aria-selected', String(x === b)));
        onPick(it);
      });
      container.appendChild(b);
    });
    onPick(initial);
  }

  function legend(container, entries) {
    container.innerHTML = entries.map(([name, color, dash]) => `<span><i class="lgd${dash ? ' lgd--dash' : ''}" style="--c:${color}"></i>${name}</span>`).join('');
  }

  /* ------------------------------- Table 1 ------------------------------- */
  function table1() {
    const tabsEl = document.getElementById('t1Tabs');
    if (!tabsEl) return;
    const table = document.getElementById('t1Table');
    const bars = document.getElementById('t1Bars');
    bars.innerHTML = '<div class="t1bars__one"></div><div class="t1bars__one"></div>';
    const [barA, barB] = bars.children;
    const cols = [...D.PR_COLS, ...D.PH_COLS];

    function render(bb) {
      const rows = D.table1[bb];
      const best = cols.map((_, j) => Math.max(...D.METHODS.map((m) => rows[m][j])));
      const bestBase = cols.map((_, j) => Math.max(...D.METHODS.filter((m) => m !== 'RLAWM').map((m) => rows[m][j])));
      let html = '<thead><tr><th rowspan="2" class="rt-method">Method</th><th colspan="5" class="rt-group">PhysReason</th><th colspan="7" class="rt-group">PHYSICS</th></tr><tr>';
      cols.forEach((c, j) => { html += `<th class="${j === 4 || j === 11 ? 'rt-overall' : ''}${j === 5 ? ' rt-split' : ''}">${c.replace('&', '&amp;')}</th>`; });
      html += '</tr></thead><tbody>';
      D.METHODS.forEach((m) => {
        const ours = m === 'RLAWM';
        html += `<tr class="${ours ? 'is-ours' : ''}"><th class="rt-method">${ours ? '<span class="rt-dot"></span>' : ''}${m}</th>`;
        rows[m].forEach((v, j) => {
          const isBest = Math.abs(v - best[j]) < 1e-9;
          html += `<td class="${isBest ? 'is-best' : ''}${j === 4 || j === 11 ? ' rt-overall' : ''}${j === 5 ? ' rt-split' : ''}">${v.toFixed(1)}</td>`;
        });
        html += '</tr>';
      });
      html += '<tr class="rt-delta"><th class="rt-method">Δ vs best baseline</th>';
      rows.RLAWM.forEach((v, j) => {
        const d = v - bestBase[j];
        html += `<td class="${d >= 0 ? 'is-up' : 'is-down'}${j === 4 || j === 11 ? ' rt-overall' : ''}${j === 5 ? ' rt-split' : ''}">${d >= 0 ? '+' : '−'}${Math.abs(d).toFixed(1)}</td>`;
      });
      html += '</tr></tbody>';
      table.innerHTML = html;

      const mk = (idx) => D.METHODS.map((m) => ({ label: m, value: rows[m][idx], color: m === 'RLAWM' ? COL.RLAWM : COL.baseline, strong: m === 'RLAWM' }));
      C.hbars(barA, mk(4), { max: 60, title: 'PhysReason · Overall', labelW: 150 });
      C.hbars(barB, mk(11), { max: 36, title: 'PHYSICS · Overall', labelW: 150 });
    }
    tabs(tabsEl, Object.keys(D.table1), 'Qwen3-VL-8B', render);
  }

  /* ------------------------------ SubPass@k ------------------------------ */
  function passk() {
    const tabsEl = document.getElementById('pkTabs');
    if (!tabsEl) return;
    const chart = document.getElementById('pkChart');
    const order = ['RLAWM', 'Reference-Likert', 'RLVR', 'Vanilla'];
    legend(document.getElementById('pkLegend'), order.map((m) => [m === 'Vanilla' ? 'Vanilla (base model)' : m, colorOf(m), m === 'Vanilla']));
    document.getElementById('pkLegend').insertAdjacentHTML('beforeend', '<em class="chart__note">Values recovered from Figure 3; the k = 1 points match Table 1.</em>');
    tabs(tabsEl, ['Qwen3-VL-8B', 'Qwen3-VL-4B'], 'Qwen3-VL-8B', (bb) => {
      C.lines(chart, {
        xs: D.passk.ks,
        xLabel: 'number of samples k',
        yLabel: 'SubPass@k (%)',
        series: order.map((m) => ({ name: m, color: colorOf(m), dash: m === 'Vanilla', strong: m === 'RLAWM', values: D.passk[bb][m] })),
      });
    });
  }

  /* --------------------------------- OOD --------------------------------- */
  function ood() {
    const box = document.getElementById('oodCharts');
    if (!box) return;
    const order = ['RLVR', 'Reference-Likert', 'RLAWM'];
    box.innerHTML = D.ood.map(() => '<div class="ood__one"></div>').join('');
    D.ood.forEach((o, i) => {
      const max = Math.max(...Object.values(o.values)) * 1.25;
      C.hbars(box.children[i], order.map((m) => ({ label: m, value: o.values[m], color: colorOf(m), strong: m === 'RLAWM' })), { max, title: `${o.name} · ${o.backbone}`, labelW: 128, barH: 16, gap: 8 });
    });
    legend(document.getElementById('oodLegend'), order.map((m) => [m, colorOf(m)]));
    document.getElementById('oodLegend').insertAdjacentHTML('beforeend', '<em class="chart__note">Each panel has its own scale.</em>');
  }

  function init() {
    table1();
    passk();
    ood();
  }

  root.AWM.initResults = init;
})(typeof self !== 'undefined' ? self : this);
