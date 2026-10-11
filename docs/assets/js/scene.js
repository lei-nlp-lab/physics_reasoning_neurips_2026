/*
 * Canvas drawing helpers shared by every scene: world-to-screen views and the
 * track, spring, ball, field and trail of the arc-and-field problem.
 */
(function (root) {
  'use strict';
  const AWM = (root.AWM = root.AWM || {});

  const RGB = {
    belief: '79,216,255',
    amber: '255,165,61',
    field: '111,141,255',
    bad: '255,77,109',
    good: '62,229,154',
    white: '231,237,247',
  };

  const STYLES = {
    true: {
      track: 'rgba(231,237,247,0.92)', trackWidth: 2.4, dash: [], glow: 0,
      ball: 'amber', field: RGB.field, fieldAlpha: 0.5, label: 'rgba(214,224,240,0.9)', spring: 'rgba(214,224,240,0.85)',
    },
    belief: {
      track: `rgba(${RGB.belief},0.95)`, trackWidth: 2, dash: [7, 6], glow: 14, glowColor: `rgba(${RGB.belief},0.75)`,
      ball: 'belief', field: RGB.belief, fieldAlpha: 0.5, label: `rgba(${RGB.belief},0.95)`, spring: `rgba(${RGB.belief},0.9)`,
    },
    blueprint: {
      track: 'rgba(196,214,245,0.85)', trackWidth: 1.6, dash: [], glow: 0,
      ball: 'amber', field: RGB.field, fieldAlpha: 0.42, label: 'rgba(160,182,220,0.95)', spring: 'rgba(196,214,245,0.75)',
    },
  };

  const BALL_R = 0.09;
  const WALL_GAP = 0.5;   // wall sits this far behind A along the track
  const MATH_FONT = "italic 600 16px 'STIX Two Text', 'Times New Roman', Times, serif";

  function createView(canvas, bounds, pad = 14) {
    const ctx = canvas.getContext('2d');
    const view = { canvas, ctx, bounds, pad, w: 1, h: 1, s: 1, ox: 0, oy: 0 };
    view.resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(root.devicePixelRatio || 1, 2);
      view.w = Math.max(rect.width, 1);
      view.h = Math.max(rect.height, 1);
      canvas.width = Math.round(view.w * dpr);
      canvas.height = Math.round(view.h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const b = view.bounds;
      const sx = (view.w - 2 * pad) / (b.xmax - b.xmin);
      const sy = (view.h - 2 * pad) / (b.ymax - b.ymin);
      view.s = Math.min(sx, sy);
      view.ox = (view.w - view.s * (b.xmax - b.xmin)) / 2;
      view.oy = (view.h - view.s * (b.ymax - b.ymin)) / 2;
    };
    view.X = (x) => view.ox + (x - view.bounds.xmin) * view.s;
    view.Y = (y) => view.oy + (view.bounds.ymax - y) * view.s;
    view.L = (d) => d * view.s;
    view.clear = () => ctx.clearRect(0, 0, view.w, view.h);
    view.resize();
    return view;
  }

  function withGlow(ctx, st, fn) {
    ctx.save();
    if (st.glow) {
      ctx.shadowColor = st.glowColor;
      ctx.shadowBlur = st.glow;
    }
    fn();
    ctx.restore();
  }

  // Seeded pseudo-random in [0, 1) from integer inputs.
  function hash(a, b, c = 0) {
    let h = (a * 374761393 + b * 668265263 + c * 2147483647) | 0;
    h = (h ^ (h >>> 13)) * 1274126177;
    h ^= h >>> 16;
    return ((h >>> 0) % 100000) / 100000;
  }

  // Track: wall, A->B segment, quarter arc B->O.
  function drawTrack(view, geo, st, { alpha = 1, offsetX = 0, color = null, showWall = true } = {}) {
    const { ctx } = view;
    const wall = { x: geo.A.x - WALL_GAP * geo.dir.x, y: geo.A.y - WALL_GAP * geo.dir.y };
    const n = { x: -geo.dir.y, y: geo.dir.x };
    ctx.save();
    ctx.globalAlpha *= alpha;
    ctx.translate(offsetX, 0);
    withGlow(ctx, st, () => {
      ctx.strokeStyle = color || st.track;
      ctx.lineWidth = st.trackWidth;
      ctx.lineCap = 'round';
      ctx.setLineDash(st.dash);
      ctx.beginPath();
      ctx.moveTo(view.X(wall.x), view.Y(wall.y));
      ctx.lineTo(view.X(geo.B.x), view.Y(geo.B.y));
      ctx.arc(view.X(geo.C.x), view.Y(geo.C.y), view.L(geo.R), Math.PI / 2, 0, true);
      ctx.stroke();
      if (showWall) {
        ctx.setLineDash([]);
        ctx.beginPath();
        const top = { x: wall.x + 0.42 * n.x, y: wall.y + 0.42 * n.y };
        ctx.moveTo(view.X(wall.x - 0.02 * n.x), view.Y(wall.y - 0.02 * n.y));
        ctx.lineTo(view.X(top.x), view.Y(top.y));
        ctx.stroke();
        // hatch behind the wall
        ctx.lineWidth = 1;
        for (let i = 0; i < 5; i++) {
          const f = (i + 0.5) / 5;
          const p = { x: wall.x + f * 0.42 * n.x, y: wall.y + f * 0.42 * n.y };
          ctx.beginPath();
          ctx.moveTo(view.X(p.x), view.Y(p.y));
          ctx.lineTo(view.X(p.x - 0.09 * geo.dir.x - 0.07 * n.x), view.Y(p.y - 0.09 * geo.dir.y - 0.07 * n.y));
          ctx.stroke();
        }
      }
    });
    ctx.restore();
  }

  function drawPoint(view, p, st, { r = 3, alpha = 1 } = {}) {
    const { ctx } = view;
    ctx.save();
    ctx.globalAlpha *= alpha;
    withGlow(ctx, st, () => {
      ctx.fillStyle = st.track;
      ctx.beginPath();
      ctx.arc(view.X(p.x), view.Y(p.y), r, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.restore();
  }

  // Spring from the wall to the ball (or relaxed once the ball has left).
  function drawSpring(view, geo, sBall, s0, st, { alpha = 1 } = {}) {
    const { ctx } = view;
    const n = { x: -geo.dir.y, y: geo.dir.x };
    const lift = 0.1; // spring axis height above the track surface
    const start = -WALL_GAP;
    const end = Math.min(sBall === null ? s0 : sBall, s0) - BALL_R;
    const coils = 9;
    const amp = 0.065;
    const pt = (s, off) => ({
      x: geo.A.x + s * geo.dir.x + (lift + off) * n.x,
      y: geo.A.y + s * geo.dir.y + (lift + off) * n.y,
    });
    ctx.save();
    ctx.globalAlpha *= alpha;
    withGlow(ctx, { ...st, glow: st.glow ? 8 : 0 }, () => {
      ctx.strokeStyle = st.spring;
      ctx.lineWidth = 1.4;
      ctx.setLineDash([]);
      ctx.beginPath();
      const lead = 0.06;
      let p = pt(start, 0);
      ctx.moveTo(view.X(p.x), view.Y(p.y));
      p = pt(start + lead, 0);
      ctx.lineTo(view.X(p.x), view.Y(p.y));
      const span = end - start - 2 * lead;
      for (let i = 1; i <= coils * 2; i++) {
        const s = start + lead + (span * i) / (coils * 2 + 1);
        p = pt(s, i % 2 ? amp : -amp);
        ctx.lineTo(view.X(p.x), view.Y(p.y));
      }
      p = pt(end - lead, 0);
      ctx.lineTo(view.X(p.x), view.Y(p.y));
      p = pt(end, 0);
      ctx.lineTo(view.X(p.x), view.Y(p.y));
      ctx.stroke();
    });
    ctx.restore();
  }

  function drawBall(view, p, st, { r = BALL_R, alpha = 1 } = {}) {
    const { ctx } = view;
    const x = view.X(p.x);
    const y = view.Y(p.y);
    const rr = view.L(r);
    ctx.save();
    ctx.globalAlpha *= alpha;
    if (st.ball === 'amber') {
      ctx.shadowColor = `rgba(${RGB.amber},0.75)`;
      ctx.shadowBlur = 18;
      const grad = ctx.createRadialGradient(x - rr * 0.35, y - rr * 0.35, rr * 0.1, x, y, rr);
      grad.addColorStop(0, '#ffe2b0');
      grad.addColorStop(0.45, '#ffa53d');
      grad.addColorStop(1, '#d9700f');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(x, y, rr, 0, Math.PI * 2);
      ctx.fill();
    } else {
      ctx.shadowColor = `rgba(${RGB.belief},0.9)`;
      ctx.shadowBlur = 16;
      const grad = ctx.createRadialGradient(x, y, rr * 0.1, x, y, rr);
      grad.addColorStop(0, `rgba(${RGB.belief},0.85)`);
      grad.addColorStop(1, `rgba(${RGB.belief},0.12)`);
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(x, y, rr, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = `rgba(${RGB.belief},0.95)`;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
    ctx.restore();
  }

  function shadeField(view, x0, st, { alpha = 1 } = {}) {
    const { ctx, bounds } = view;
    const X0 = view.X(x0);
    ctx.save();
    ctx.globalAlpha *= alpha;
    const grad = ctx.createLinearGradient(X0, 0, view.X(bounds.xmax), 0);
    grad.addColorStop(0, `rgba(${st.field},0.05)`);
    grad.addColorStop(1, `rgba(${st.field},0.012)`);
    ctx.fillStyle = grad;
    ctx.fillRect(X0, 0, view.w - X0, view.h);
    ctx.strokeStyle = `rgba(${st.field},0.16)`;
    ctx.setLineDash([3, 6]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(X0, 0);
    ctx.lineTo(X0, view.h);
    ctx.stroke();
    ctx.restore();
  }

  // Lattice of 45-degree arrows flowing with time. `dissolve` breaks them into particles.
  function drawField(view, x0, time, st, { strength = 1, dissolve = 0, spacing = 0.74, alpha = 1, exclude = null } = {}) {
    if (dissolve >= 1 || alpha <= 0) return;
    const { ctx, bounds } = view;
    const len = 0.34 * Math.max(strength, 0.15);
    const d = Math.SQRT1_2;
    const flow = ((time * 0.22) % spacing + spacing) % spacing;
    ctx.save();
    ctx.beginPath();
    ctx.rect(view.X(x0), 0, view.w - view.X(x0), view.h);
    ctx.clip();
    ctx.lineWidth = 1.5;
    ctx.lineCap = 'round';
    const i0 = Math.floor((x0 - bounds.ymax - 2) / spacing);
    const i1 = Math.ceil((bounds.xmax - bounds.ymin + 2) / spacing);
    for (let i = i0; i <= i1; i++) {
      for (let j = -12; j <= 12; j++) {
        // lattice rotated 45 deg so arrows line up along the flow
        const u = i * spacing + flow;           // along the field direction
        const v = j * spacing * 0.92;            // across
        const cx = u * d - v * d + x0 + 0.25;
        const cy = u * d + v * d + bounds.ymin - 0.4;
        if (cx < x0 - 0.1 || cx > bounds.xmax + 0.4 || cy < bounds.ymin - 0.4 || cy > bounds.ymax + 0.4) continue;
        if (exclude && exclude(cx, cy)) continue;
        const r1 = hash(i, j, 1);
        const fade = Math.min(1, Math.min(u - flow + spacing * 2, 99) / spacing) ;
        const a = st.fieldAlpha * alpha * (1 - dissolve) * (0.55 + 0.45 * r1) * Math.max(fade, 0.35);
        if (dissolve > 0) {
          for (let q = 0; q < 4; q++) {
            const ang = hash(i, j, q + 7) * Math.PI * 2;
            const dist = dissolve * (0.25 + 0.5 * hash(i, j, q + 13));
            const px = cx + Math.cos(ang) * dist + (q / 3 - 0.5) * len * d;
            const py = cy + Math.sin(ang) * dist + dissolve * 0.35 + (q / 3 - 0.5) * len * d;
            ctx.fillStyle = `rgba(${st.field},${(1 - dissolve) * 0.9 * alpha})`;
            ctx.fillRect(view.X(px) - 1, view.Y(py) - 1, 2, 2);
          }
        }
        if (a <= 0.01) continue;
        const x1 = cx - (len / 2) * d;
        const y1 = cy - (len / 2) * d;
        const x2 = cx + (len / 2) * d;
        const y2 = cy + (len / 2) * d;
        ctx.strokeStyle = `rgba(${st.field},${a})`;
        ctx.beginPath();
        ctx.moveTo(view.X(x1), view.Y(y1));
        ctx.lineTo(view.X(x2), view.Y(y2));
        const head = 0.075;
        ctx.moveTo(view.X(x2), view.Y(y2));
        ctx.lineTo(view.X(x2 - head * Math.cos(Math.PI / 4 - 0.5)), view.Y(y2 - head * Math.sin(Math.PI / 4 - 0.5)));
        ctx.moveTo(view.X(x2), view.Y(y2));
        ctx.lineTo(view.X(x2 - head * Math.cos(Math.PI / 4 + 0.5)), view.Y(y2 - head * Math.sin(Math.PI / 4 + 0.5)));
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  // Polyline whose opacity grows toward the head: one soft glow pass, then a fading core.
  function drawTrail(view, pts, rgb, { width = 2.4, alpha = 1, glow = 10, fadeTail = true } = {}) {
    if (pts.length < 2) return;
    const { ctx } = view;
    const n = pts.length;
    const step = Math.max(1, Math.floor(n / 160));
    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    if (glow) {
      const from = Math.floor(n * 0.45);
      ctx.strokeStyle = `rgba(${rgb},${0.16 * alpha})`;
      ctx.lineWidth = width * 4;
      ctx.beginPath();
      ctx.moveTo(view.X(pts[from].x), view.Y(pts[from].y));
      for (let i = from + step; i < n; i += step) ctx.lineTo(view.X(pts[i].x), view.Y(pts[i].y));
      ctx.lineTo(view.X(pts[n - 1].x), view.Y(pts[n - 1].y));
      ctx.stroke();
    }
    for (let i = step; i < n; i += step) {
      const f = fadeTail ? i / (n - 1) : 1;
      ctx.strokeStyle = `rgba(${rgb},${alpha * (0.08 + 0.92 * f * f)})`;
      ctx.lineWidth = width * (0.5 + 0.5 * f);
      ctx.beginPath();
      ctx.moveTo(view.X(pts[i - step].x), view.Y(pts[i - step].y));
      ctx.lineTo(view.X(pts[Math.min(i, n - 1)].x), view.Y(pts[Math.min(i, n - 1)].y));
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawCurve(view, pts, color, { width = 1.6, dash = [], alpha = 1, glow = 0, glowColor = null } = {}) {
    if (pts.length < 2) return;
    const { ctx } = view;
    ctx.save();
    ctx.globalAlpha *= alpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.lineJoin = 'round';
    if (glow) {
      ctx.shadowColor = glowColor || color;
      ctx.shadowBlur = glow;
    }
    ctx.beginPath();
    ctx.moveTo(view.X(pts[0].x), view.Y(pts[0].y));
    for (let i = 1; i < pts.length; i++) ctx.lineTo(view.X(pts[i].x), view.Y(pts[i].y));
    ctx.stroke();
    ctx.restore();
  }

  function drawArrow(view, from, to, color, { width = 2, head = 9, alpha = 1, glow = 0 } = {}) {
    const { ctx } = view;
    const x1 = view.X(from.x), y1 = view.Y(from.y), x2 = view.X(to.x), y2 = view.Y(to.y);
    const ang = Math.atan2(y2 - y1, x2 - x1);
    ctx.save();
    ctx.globalAlpha *= alpha;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = width;
    ctx.lineCap = 'round';
    if (glow) {
      ctx.shadowColor = color;
      ctx.shadowBlur = glow;
    }
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2 - Math.cos(ang) * head * 0.6, y2 - Math.sin(ang) * head * 0.6);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - head * Math.cos(ang - 0.42), y2 - head * Math.sin(ang - 0.42));
    ctx.lineTo(x2 - head * Math.cos(ang + 0.42), y2 - head * Math.sin(ang + 0.42));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  function label(view, text, p, color, { dx = 0, dy = 0, align = 'center', baseline = 'middle', font = MATH_FONT, alpha = 1 } = {}) {
    const { ctx } = view;
    ctx.save();
    ctx.globalAlpha *= alpha;
    ctx.fillStyle = color;
    ctx.font = font;
    ctx.textAlign = align;
    ctx.textBaseline = baseline;
    ctx.fillText(text, view.X(p.x) + dx, view.Y(p.y) + dy);
    ctx.restore();
  }

  // Small rounded tag with a leader line to an anchor point.
  function tag(view, text, at, anchor, { color = 'rgba(160,182,220,0.95)', bg = 'rgba(10,16,30,0.85)', alpha = 1, font = "500 11.5px 'JetBrains Mono', ui-monospace, monospace" } = {}) {
    const { ctx } = view;
    ctx.save();
    ctx.globalAlpha *= alpha;
    ctx.font = font;
    const padX = 7;
    const w = ctx.measureText(text).width + padX * 2;
    const h = 20;
    const x = view.X(at.x) - w / 2;
    const y = view.Y(at.y) - h / 2;
    if (anchor) {
      ctx.strokeStyle = color;
      ctx.globalAlpha *= 0.55;
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.moveTo(view.X(at.x), view.Y(at.y));
      ctx.lineTo(view.X(anchor.x), view.Y(anchor.y));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha /= 0.55;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(view.X(anchor.x), view.Y(anchor.y), 2.2, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = bg;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    roundRect(ctx, x, y, w, h, 6);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, x + w / 2, y + h / 2 + 0.5);
    ctx.restore();
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  // Ball centre for a simulation sample: lifted off the track by its radius.
  function ballCenter(geo, sample, tSinceO = null) {
    const r = BALL_R;
    if (sample.phase === 'arc') {
      const dx = geo.C.x - sample.x, dy = geo.C.y - sample.y;
      const d = Math.hypot(dx, dy) || 1;
      return { x: sample.x + (dx / d) * r, y: sample.y + (dy / d) * r };
    }
    if (sample.phase === 'flight' || (sample.phase === 'end' && sample.s === null)) {
      const f = tSinceO === null ? 1 : Math.min(1, tSinceO / 0.18);
      return { x: sample.x - r * (1 - f), y: sample.y };
    }
    const n = { x: -geo.dir.y, y: geo.dir.x };
    return { x: sample.x + n.x * r, y: sample.y + n.y * r };
  }

  AWM.scene = {
    RGB, STYLES, BALL_R, WALL_GAP, MATH_FONT,
    createView, drawTrack, drawPoint, drawSpring, drawBall, shadeField, drawField,
    drawTrail, drawCurve, drawArrow, label, tag, ballCenter, hash, roundRect,
  };
})(typeof self !== 'undefined' ? self : this);
