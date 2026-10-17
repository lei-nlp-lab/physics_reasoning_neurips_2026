/*
 * Physics of the arc-and-field problem used throughout the page
 * (paper Figure 1 and Appendix F). World units: g = 1, nominal R = 1.
 *
 * Geometry: B = (-R, -R), O = (0, 0), arc centre C = (-R, 0).
 * The quarter arc rises from B (horizontal tangent) to O (vertical tangent).
 * A uniform field fills x >= -R. Its strength is k * sqrt(2) mg at 45 deg,
 * so inside the field the effective gravity is (k g, (k - 1) g).
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.AWM = Object.assign(root.AWM || {}, { physics: api });
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const g = 1;
  const THETA = Math.PI / 4;

  // `drop` lowers A below B: the model's wrong geometry h_B - h_A = drop.
  function geometry({ R = 1, L = 1.6, drop = 0 } = {}) {
    const B = { x: -R, y: -R };
    const A = { x: -R - L, y: -R - drop };
    const dx = B.x - A.x;
    const dy = B.y - A.y;
    const rampLen = Math.hypot(dx, dy);
    return {
      R, L, drop, A, B,
      O: { x: 0, y: 0 },
      C: { x: -R, y: 0 },
      rampLen,
      dir: { x: dx / rampLen, y: dy / rampLen },
      fieldX: B.x,
    };
  }

  function arcPoint(geo, phi) {
    return { x: geo.C.x + geo.R * Math.sin(phi), y: geo.C.y - geo.R * Math.cos(phi) };
  }

  /*
   * Integrate one launch: spring -> track A->B -> arc -> flight (or slide back).
   * Returns samples every dtOut seconds plus key events.
   */
  function simulate(opts = {}) {
    const {
      R = 1, L = 1.6, drop = 0, k = 1,
      vB = Math.sqrt(g * R), s0 = 0.45,
      tMax = 9, dtOut = 1 / 120,
      bounds = { xmin: -4, xmax: 2.2, ymin: -3, ymax: 3.4 },
    } = opts;
    const geo = geometry({ R, L, drop });
    const ge = { x: k * g, y: (k - 1) * g };   // effective gravity in the field
    const along = -g * geo.dir.y;               // gravity along A->B (no field there)
    const h = 1 / 4000;

    // Speed when the ball leaves the spring, chosen so that |v| = vB at B.
    const vLaunch = Math.sqrt(Math.max(vB * vB + 2 * g * (geo.B.y - (geo.A.y + s0 * geo.dir.y)), 1e-9));
    const omega = vLaunch / s0;
    const tSpring = Math.PI / (2 * omega);

    let phase = 'spring';
    let s = 0, sd = 0, phi = 0, phid = 0;
    let fx = 0, fy = 0, fvx = 0, fvy = 0;
    const events = { tSpring, tB: null, tO: null, tStall: null, tBack: null, tEnd: null, stallPhi: null };
    let vO = null;

    const samples = [];
    let t = 0;
    let nextOut = 0;

    const position = () => {
      if (phase === 'arc') return arcPoint(geo, phi);
      if (phase === 'flight') return { x: fx, y: fy };
      return { x: geo.A.x + s * geo.dir.x, y: geo.A.y + s * geo.dir.y };
    };
    const speed = () => {
      if (phase === 'arc') return R * Math.abs(phid);
      if (phase === 'flight') return Math.hypot(fvx, fvy);
      return Math.abs(sd);
    };
    const arcAcc = (p) => (ge.x * Math.cos(p) + ge.y * Math.sin(p)) / R;

    while (t <= tMax) {
      if (t >= nextOut - 1e-12) {
        const p = position();
        samples.push({ t: nextOut, x: p.x, y: p.y, phase, s: phase === 'arc' || phase === 'flight' ? null : s, speed: speed() });
        nextOut += dtOut;
      }
      if (phase === 'end') break;

      switch (phase) {
        case 'spring':
          if (t + h >= tSpring) {
            phase = 'track';
            s = s0;
            sd = vLaunch;
          } else {
            s = s0 * (1 - Math.cos(omega * (t + h)));
            sd = s0 * omega * Math.sin(omega * (t + h));
          }
          break;
        case 'track':
          sd += along * h;
          s += sd * h;
          if (s >= geo.rampLen) {
            phase = 'arc';
            phi = 0;
            phid = sd / R;
            events.tB = t + h;
          }
          break;
        case 'arc': {
          // RK4 on (phi, phid)
          const k1p = phid, k1v = arcAcc(phi);
          const k2p = phid + 0.5 * h * k1v, k2v = arcAcc(phi + 0.5 * h * k1p);
          const k3p = phid + 0.5 * h * k2v, k3v = arcAcc(phi + 0.5 * h * k2p);
          const k4p = phid + h * k3v, k4v = arcAcc(phi + h * k3p);
          const prevPhid = phid;
          phi += (h / 6) * (k1p + 2 * k2p + 2 * k3p + k4p);
          phid += (h / 6) * (k1v + 2 * k2v + 2 * k3v + k4v);
          if (prevPhid > 0 && phid <= 0 && events.tStall === null) {
            events.tStall = t + h;
            events.stallPhi = phi;
          }
          if (phi >= Math.PI / 2) {
            phase = 'flight';
            fx = geo.O.x;
            fy = geo.O.y;
            fvx = 0;
            fvy = R * phid;
            vO = fvy;
            events.tO = t + h;
          } else if (phi <= 0 && phid < 0) {
            phase = 'back';
            s = geo.rampLen;
            sd = R * phid;
            events.tBack = t + h;
          }
          break;
        }
        case 'back':
          sd += along * h;
          s += sd * h;
          if (s <= s0) {
            s = s0;
            phase = 'end';
            events.tEnd = t + h;
          }
          break;
        case 'flight':
          fvx += ge.x * h;
          fvy += ge.y * h;
          fx += fvx * h;
          fy += fvy * h;
          if (fx > bounds.xmax + 0.3 || fy > bounds.ymax + 0.3 || fy < bounds.ymin - 0.3) {
            phase = 'end';
            events.tEnd = t + h;
          }
          break;
        default:
          break;
      }
      t += h;
    }
    if (events.tEnd === null) events.tEnd = t;
    return { geo, samples, events, vO, dtOut, vLaunch, s0 };
  }

  function sampleAt(sim, t) {
    const { samples, dtOut } = sim;
    if (t <= 0) return samples[0];
    const i = t / dtOut;
    const i0 = Math.floor(i);
    if (i0 >= samples.length - 1) return samples[samples.length - 1];
    const a = samples[i0];
    const b = samples[i0 + 1];
    const f = i - i0;
    return {
      t,
      x: a.x + (b.x - a.x) * f,
      y: a.y + (b.y - a.y) * f,
      phase: a.phase,
      s: a.s === null || b.s === null ? a.s : a.s + (b.s - a.s) * f,
      speed: a.speed + (b.speed - a.speed) * f,
    };
  }

  /* ---------------- Checker world (paper Appendix F and M) ---------------- */

  // Givens of the problem. When R is perturbed the defining constraint
  // v_B = sqrt(gR) is preserved, exactly as in the Appendix M checker.
  function givens({ R = 1, k = 1 } = {}) {
    return { m: 1, g, R, vB: Math.sqrt(g * R), qE: k * Math.SQRT2 * g, theta: THETA };
  }

  // What the true world does with these givens.
  function trueWorld(G) {
    const arcWork = -G.m * G.g * G.R + G.qE * (Math.cos(G.theta) + Math.sin(G.theta)) * G.R;
    const vO2 = G.vB * G.vB + (2 * arcWork) / G.m;
    return {
      arcWork,
      reachesO: vO2 > 0,
      vO: Math.sqrt(Math.max(vO2, 0)),
      ax: (G.qE * Math.cos(G.theta)) / G.m,
      ay: (G.qE * Math.sin(G.theta) - G.m * G.g) / G.m,
    };
  }

  // The two rollouts of Appendix F, written as the programs the translator emits.
  const programs = {
    1: {
      id: 1,
      name: 'Case 1 · timeline',
      run(G) {
        const Ep = 0.5 * G.m * G.vB * G.vB;
        const Wg = -G.m * G.g * G.R;
        const WE = G.qE * Math.cos(G.theta) * G.R + G.qE * Math.sin(G.theta) * G.R;
        const vO = Math.sqrt(Math.max(G.vB * G.vB + (2 * (Wg + WE)) / G.m, 0));
        return { Ep, vO, ax: (G.qE * Math.cos(G.theta)) / G.m, ay: (G.qE * Math.sin(G.theta) - G.m * G.g) / G.m };
      },
    },
    2: {
      id: 2,
      name: 'Case 2 · interaction graph',
      run(G) {
        const Ep = 0.5 * G.m * G.vB * G.vB;
        const vO = G.vB; // "vertical forces cancel ... kinetic energy unchanged"
        return { Ep, vO, ax: (G.qE * Math.cos(G.theta)) / G.m, ay: (G.qE * Math.sin(G.theta) - G.m * G.g) / G.m };
      },
    },
  };

  // Emitted trajectory point at time t after O.
  function trajectoryPoint(out, t) {
    return { x: 0.5 * out.ax * t * t, y: out.vO * t + 0.5 * out.ay * t * t };
  }

  const close = (a, b, tol = 1e-2) => Math.abs(a - b) <= tol;

  /*
   * The three tests of the illustrative checker in Appendix M.
   * Each category has one test, so R_S = mean of three pass/fail results.
   */
  function runChecker(program, { pertR = 1.1 } = {}) {
    const G = givens();
    const out = program.run(G);
    const world = trueWorld(G);
    const scale = G.m * G.g * G.R;
    const dKE = 0.5 * G.m * (out.vO * out.vO - G.vB * G.vB);

    const invariants = [
      { key: 'spring', label: 'Spring energy', model: out.Ep / scale, expected: 0.5, unit: 'mgR' },
      { key: 'work', label: 'Arc work–energy', model: dKE / scale, expected: world.arcWork / scale, unit: 'mgR' },
      { key: 'ax', label: 'Horizontal acceleration', model: out.ax / G.g, expected: world.ax / G.g, unit: 'g' },
      { key: 'ay', label: 'Vertical acceleration', model: out.ay / G.g, expected: world.ay / G.g, unit: 'g' },
    ].map((c) => ({ ...c, pass: close(c.model, c.expected) }));
    const invPass = invariants.every((c) => c.pass);

    // Trace: recompute y from the candidate's own x, a_x, a_y, v_O.
    const p = trajectoryPoint(out, 1.0);
    const tFromX = Math.sqrt((2 * p.x) / out.ax);
    const yFromTrace = out.vO * tFromX + 0.5 * out.ay * tFromX * tFromX;
    const traceResidual = Math.abs(p.y - yFromTrace) / Math.max(G.R, Math.abs(p.y), Math.abs(yFromTrace));
    const tracePass = traceResidual <= 1e-2;

    const pert = perturbR(program, pertR);

    const scores = { inv: invPass ? 1 : 0, trace: tracePass ? 1 : 0, pert: pert.pass ? 1 : 0 };
    const RS = (scores.inv + scores.trace + scores.pert) / 3;
    return { givens: G, out, world, dKE, invariants, invPass, trace: { point: p, yFromTrace, residual: traceResidual, pass: tracePass }, pert, scores, RS };
  }

  // Scored perturbation (Appendix M): re-run the same program at R and fR.
  function perturbR(program, f) {
    const v0 = program.run(givens({ R: 1 })).vO;
    const v1 = program.run(givens({ R: f })).vO;
    const observed = v1 / v0;
    const expected = Math.sqrt(f);
    const directional = f > 1 ? v1 > v0 : f < 1 ? v1 < v0 : true;
    return { param: 'R', factor: f, observed, expected, pass: directional && close(observed, expected, 1e-2 * expected) };
  }

  // Exploratory perturbation of the field strength (not part of the scored checker).
  function perturbField(program, k) {
    const v0 = program.run(givens({ k: 1 })).vO;
    const v1 = program.run(givens({ k })).vO;
    const t0 = trueWorld(givens({ k: 1 })).vO;
    const t1 = trueWorld(givens({ k })).vO;
    const observed = v1 / v0;
    const expected = t1 / t0;
    return { param: 'qE', factor: k, observed, expected, pass: close(observed, expected, 1e-2 * expected) };
  }

  return { g, THETA, geometry, arcPoint, simulate, sampleAt, givens, trueWorld, programs, trajectoryPoint, runChecker, perturbR, perturbField };
});
