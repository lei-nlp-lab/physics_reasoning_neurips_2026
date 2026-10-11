/*
 * Small SVG chart toolkit: horizontal bars and a line chart with crosshair.
 * Charts re-render on resize and share one tooltip.
 */
(function (root) {
  'use strict';
  const AWM = (root.AWM = root.AWM || {});
  const NS = 'http://www.w3.org/2000/svg';

  function el(tag, attrs = {}, parent = null, text = null) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (text !== null) n.textContent = text;
    if (parent) parent.appendChild(n);
    return n;
  }

  /* ------------------------------- tooltip ------------------------------- */
  let tipEl = null;
  function tip(html, x, y) {
    tipEl = tipEl || document.getElementById('tip');
    if (!tipEl) return;
    tipEl.innerHTML = html;
    tipEl.hidden = false;
    const r = tipEl.getBoundingClientRect();
    let left = x + 14;
    let top = y - r.height - 12;
    if (left + r.width > root.innerWidth - 8) left = x - r.width - 14;
    if (top < 8) top = y + 16;
    tipEl.style.left = `${left}px`;
    tipEl.style.top = `${top}px`;
  }
  function untip() {
    tipEl = tipEl || document.getElementById('tip');
    if (tipEl) tipEl.hidden = true;
  }

  /* ------------------------------ responsive ----------------------------- */
  const renderers = new Map();
  const ro = 'ResizeObserver' in root
    ? new ResizeObserver((entries) => entries.forEach((e) => {
      const f = renderers.get(e.target);
      if (f) f();
    }))
    : null;
  function mount(container, render) {
    renderers.set(container, render);
    if (ro) ro.observe(container);
    render();
  }

  function svgFor(container, height) {
    container.innerHTML = '';
    const w = Math.max(container.clientWidth, 240);
    const svg = el('svg', { width: w, height, viewBox: `0 0 ${w} ${height}`, class: 'viz' }, container);
    return { svg, w };
  }

  const fmt = (v, d = 1) => Number(v).toFixed(d);
  const swatch = (c, dash) => `<i class="tip__sw${dash ? ' tip__sw--dash' : ''}" style="--c:${c}"></i>`;

  /* ---------------------------- horizontal bars --------------------------- */
  // rows: [{ label, value, color, strong }]
  function hbars(container, rows, opts = {}) {
    const { max = Math.max(...rows.map((r) => r.value)) * 1.08, unit = '', decimals = 1, barH = 18, gap = 9, labelW = 170, title = null } = opts;
    mount(container, () => {
      const top = title ? 22 : 4;
      const h = top + rows.length * (barH + gap) + 4;
      const { svg, w } = svgFor(container, h);
      if (title) el('text', { x: 0, y: 13, class: 'viz-title' }, svg, title);
      const lw = Math.min(labelW, w * 0.42);
      const x0 = lw;
      const span = w - x0 - 52;
      rows.forEach((r, i) => {
        const y = top + i * (barH + gap);
        const len = Math.max(2, (r.value / max) * span);
        el('text', { x: x0 - 10, y: y + barH / 2, class: `viz-label${r.strong ? ' is-strong' : ''}`, 'text-anchor': 'end', 'dominant-baseline': 'middle' }, svg, r.label);
        el('path', { d: `M${x0},${y} h${len - 4} a4,4 0 0 1 4,4 v${barH - 8} a4,4 0 0 1 -4,4 h${-(len - 4)} z`, fill: r.color, class: 'viz-bar' }, svg);
        el('text', { x: x0 + len + 6, y: y + barH / 2, class: `viz-value${r.strong ? ' is-strong' : ''}`, 'dominant-baseline': 'middle' }, svg, fmt(r.value, decimals) + unit);
        const hit = el('rect', { x: 0, y: y - gap / 2, width: w, height: barH + gap, fill: 'transparent', class: 'viz-hit' }, svg);
        hit.addEventListener('mousemove', (e) => tip(`${swatch(r.color)}<b>${r.label}</b><br>${fmt(r.value, decimals)}${unit}${r.note ? `<br><span>${r.note}</span>` : ''}`, e.clientX, e.clientY));
        hit.addEventListener('mouseleave', untip);
      });
    });
  }

  /* ------------------------------ line chart ------------------------------ */
  // cfg: { xs, series: [{ name, color, dash, values }], yLabel, xLabel }
  function lines(container, cfg) {
    mount(container, () => {
      const h = Math.min(330, Math.max(240, container.clientWidth * 0.62));
      const { svg, w } = svgFor(container, h);
      const m = { l: 40, r: 124, t: 12, b: 38 };
      const all = cfg.series.flatMap((s) => s.values);
      const lo = Math.floor(Math.min(...all) / 10) * 10;
      const hi = Math.ceil(Math.max(...all) / 10) * 10;
      const lx = (k) => Math.log10(k) / Math.log10(cfg.xs[cfg.xs.length - 1]);
      const X = (k) => m.l + lx(k) * (w - m.l - m.r);
      const Y = (v) => m.t + (1 - (v - lo) / (hi - lo)) * (h - m.t - m.b);

      for (let v = lo; v <= hi + 1e-9; v += 10) {
        el('line', { x1: m.l, x2: w - m.r, y1: Y(v), y2: Y(v), class: 'viz-grid' }, svg);
        el('text', { x: m.l - 8, y: Y(v), class: 'viz-axis', 'text-anchor': 'end', 'dominant-baseline': 'middle' }, svg, v);
      }
      cfg.xs.forEach((k) => {
        el('text', { x: X(k), y: h - m.b + 18, class: 'viz-axis', 'text-anchor': 'middle' }, svg, k);
      });
      el('text', { x: (m.l + w - m.r) / 2, y: h - 4, class: 'viz-axis viz-axis--title', 'text-anchor': 'middle' }, svg, cfg.xLabel || '');
      if (cfg.yLabel) el('text', { x: 0, y: 10, class: 'viz-axis viz-axis--title' }, svg, cfg.yLabel);

      // series: draw emphasised one last
      const order = [...cfg.series].sort((a, b) => (a.strong ? 1 : 0) - (b.strong ? 1 : 0));
      order.forEach((s) => {
        const d = s.values.map((v, i) => `${i ? 'L' : 'M'}${X(cfg.xs[i])},${Y(v)}`).join('');
        el('path', { d, fill: 'none', stroke: s.color, 'stroke-width': s.strong ? 2.6 : 2, 'stroke-dasharray': s.dash ? '5 4' : 'none', 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, svg);
        s.values.forEach((v, i) => el('circle', { cx: X(cfg.xs[i]), cy: Y(v), r: 4, fill: s.color, stroke: 'var(--viz-surface)', 'stroke-width': 2 }, svg));
      });

      // direct labels at the right end, spread to avoid collisions
      const ends = cfg.series.map((s) => ({ s, y: Y(s.values[s.values.length - 1]) })).sort((a, b) => a.y - b.y);
      for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 15) ends[i].y = ends[i - 1].y + 15;
      ends.forEach(({ s, y }) => {
        el('text', { x: w - m.r + 10, y, class: `viz-label${s.strong ? ' is-strong' : ''}`, 'dominant-baseline': 'middle' }, svg, `${s.name} ${fmt(s.values[s.values.length - 1])}`);
      });

      // crosshair + tooltip
      const cross = el('line', { y1: m.t, y2: h - m.b, class: 'viz-cross', visibility: 'hidden' }, svg);
      const hit = el('rect', { x: m.l - 10, y: m.t, width: w - m.l - m.r + 20, height: h - m.t - m.b, fill: 'transparent' }, svg);
      hit.addEventListener('mousemove', (e) => {
        const r = svg.getBoundingClientRect();
        const px = e.clientX - r.left;
        let best = 0;
        cfg.xs.forEach((k, i) => { if (Math.abs(X(k) - px) < Math.abs(X(cfg.xs[best]) - px)) best = i; });
        cross.setAttribute('x1', X(cfg.xs[best]));
        cross.setAttribute('x2', X(cfg.xs[best]));
        cross.setAttribute('visibility', 'visible');
        const rows = [...cfg.series].sort((a, b) => b.values[best] - a.values[best])
          .map((s) => `${swatch(s.color, s.dash)}${s.name}<b>${fmt(s.values[best])}</b>`).join('<br>');
        tip(`<span>k = ${cfg.xs[best]}</span><br>${rows}`, e.clientX, e.clientY);
      });
      hit.addEventListener('mouseleave', () => {
        cross.setAttribute('visibility', 'hidden');
        untip();
      });
    });
  }

  AWM.charts = { el, tip, untip, hbars, lines, mount };
})(typeof self !== 'undefined' ? self : this);
