/* Page-level behaviour: math, count-ups, reveal on scroll, BibTeX copy, figure lightbox. */
(function (root) {
  'use strict';
  const doc = root.document;
  const reduced = root.matchMedia && root.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function renderMath(el = doc.body) {
    if (typeof root.renderMathInElement !== 'function') return;
    root.renderMathInElement(el, {
      delimiters: [{ left: '\\(', right: '\\)', display: false }, { left: '\\[', right: '\\]', display: true }],
      throwOnError: false,
    });
  }

  function countUps() {
    const els = [...doc.querySelectorAll('[data-count]')];
    const run = (el) => {
      const target = parseFloat(el.dataset.count);
      const dec = +(el.dataset.decimals || 0);
      const pre = el.dataset.prefix || '';
      const suf = el.dataset.suffix || '';
      if (reduced) {
        el.textContent = pre + target.toFixed(dec) + suf;
        return;
      }
      const t0 = performance.now();
      const dur = 1300;
      const step = (now) => {
        const p = Math.min(1, (now - t0) / dur);
        const v = target * (1 - Math.pow(1 - p, 3));
        el.textContent = pre + v.toFixed(dec) + suf;
        if (p < 1) root.requestAnimationFrame(step);
      };
      root.requestAnimationFrame(step);
    };
    if (!('IntersectionObserver' in root)) return els.forEach(run);
    const io = new IntersectionObserver((entries) => {
      entries.forEach((en) => {
        if (en.isIntersecting) {
          run(en.target);
          io.unobserve(en.target);
        }
      });
    }, { threshold: 0.6 });
    els.forEach((el) => io.observe(el));
  }

  function reveals() {
    const els = [...doc.querySelectorAll('.reveal')];
    if (reduced || !('IntersectionObserver' in root)) return els.forEach((el) => el.classList.add('is-in'));
    const io = new IntersectionObserver((entries) => {
      entries.forEach((en) => {
        if (en.isIntersecting) {
          en.target.classList.add('is-in');
          io.unobserve(en.target);
        }
      });
    }, { threshold: 0.15 });
    els.forEach((el) => io.observe(el));
  }

  function bibtex() {
    const btn = doc.getElementById('copyBib');
    const pre = doc.getElementById('bibtex-text');
    if (!btn || !pre) return;
    btn.addEventListener('click', async () => {
      try {
        await root.navigator.clipboard.writeText(pre.textContent.trim());
        btn.textContent = 'Copied';
      } catch (e) {
        const range = doc.createRange();
        range.selectNodeContents(pre);
        const sel = root.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        btn.textContent = 'Selected — press Ctrl+C';
      }
      setTimeout(() => (btn.textContent = 'Copy'), 1800);
    });
  }

  function lightbox() {
    const box = doc.getElementById('lightbox');
    if (!box) return;
    const img = box.querySelector('img');
    const cap = box.querySelector('figcaption');
    doc.querySelectorAll('[data-lightbox]').forEach((a) => {
      a.addEventListener('click', (e) => {
        e.preventDefault();
        img.src = a.getAttribute('href');
        img.alt = a.dataset.alt || '';
        cap.textContent = a.dataset.caption || '';
        box.hidden = false;
        doc.body.style.overflow = 'hidden';
      });
    });
    const close = () => {
      box.hidden = true;
      doc.body.style.overflow = '';
    };
    box.addEventListener('click', close);
    doc.addEventListener('keydown', (e) => e.key === 'Escape' && !box.hidden && close());
  }

  function nav() {
    const bar = doc.querySelector('.nav');
    if (!bar) return;
    const onScroll = () => bar.classList.toggle('is-scrolled', root.scrollY > 24);
    root.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  function caseTabs() {
    const tabs = [...doc.querySelectorAll('[data-casetab]')];
    const panels = [...doc.querySelectorAll('[data-casepanel]')];
    tabs.forEach((t) => t.addEventListener('click', () => {
      tabs.forEach((x) => x.setAttribute('aria-selected', String(x === t)));
      panels.forEach((p) => { p.hidden = p.dataset.casepanel !== t.dataset.casetab; });
    }));
  }

  // Highlight the nav link of the section in view.
  function navSpy() {
    const links = [...doc.querySelectorAll('.nav__links a')];
    const map = new Map(links.map((a) => [a.getAttribute('href').slice(1), a]));
    if (!('IntersectionObserver' in root)) return;
    const io = new IntersectionObserver((entries) => {
      entries.forEach((en) => {
        if (en.isIntersecting) links.forEach((a) => a.classList.toggle('is-current', a === map.get(en.target.id)));
      });
    }, { rootMargin: '-45% 0px -50% 0px' });
    map.forEach((_, id) => { const s = doc.getElementById(id); if (s) io.observe(s); });
  }

  function start() {
    const A = root.AWM;
    [A.initResults, A.initModeling, A.initBuild, A.initLoop].forEach((f) => { if (f) f(); });
    renderMath();
    countUps();
    reveals();
    bibtex();
    lightbox();
    nav();
    navSpy();
    caseTabs();
    if (A.initHero) A.initHero();
    if (A.initProbe) A.initProbe();
  }

  if (doc.readyState === 'complete') start();
  else root.addEventListener('load', start);
})(window);
