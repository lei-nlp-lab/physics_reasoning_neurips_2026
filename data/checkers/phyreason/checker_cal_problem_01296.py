import math
import sympy as sp
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Problem constants
        self.L = 0.5           # rod length (m)
        self.H = 0.2           # base height (m)
        self.g = 10.0          # gravity (m/s^2)
        self.v0_up = math.sqrt(15)   # initial upward velocity (m/s)
        self.v_down = math.sqrt(5)   # downward velocity at base (m/s)

        # Safe extraction of results using getattr with fallback to results dict
        # Map expected keys to safe access
        self.results = {}
        expected_keys = [
            'x', 't_2', 'a_down', 't_total', 'v_base', 'fraca_upa_down',
            't', 'x_ring', 'a_2', 'F_f', 'x_base', 't_1', 'v_ring', 'v_1', 'a_up'
        ]
        for key in expected_keys:
            # Try direct attribute, then results dict, then set to 0.0
            val = getattr(solution_instance, key, None)
            if val is None:
                val = getattr(solution_instance, 'results', {}).get(key, 0.0)
            self.results[key] = float(val) if isinstance(val, (int, float)) else val

    def test_invariants(self) -> Tuple[bool, List[str]]:
        logs = []
        success = True

        # 1. Check kinematic invariants for up/down motions
        # Upward: v0^2 = 2*a_up*L
        a_up_expected = self.v0_up**2 / (2 * self.L)  # Should be 15
        if not math.isclose(a_up_expected, self.results.get('a_up', 0), rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: a_up invariant. Expected {a_up_expected:.3f}, got {self.results.get('a_up', 0):.3f}")
            success = False
        else:
            logs.append("PASS: a_up matches kinematic equation v0² = 2a_up L")

        # Downward: v_down^2 = 2*a_down*L
        a_down_expected = self.v_down**2 / (2 * self.L)  # Should be 5
        if not math.isclose(a_down_expected, self.results.get('a_down', 0), rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: a_down invariant. Expected {a_down_expected:.3f}, got {self.results.get('a_down', 0):.3f}")
            success = False
        else:
            logs.append("PASS: a_down matches kinematic equation v² = 2a_down L")

        # 2. Check Newton's second law consistency (up + down)
        # Up: mg + F_f = m*a_up  → F_f = m*(a_up - g)
        # Down: mg - F_f = m*a_down → F_f = m*(g - a_down)
        # Equate: a_up - g = g - a_down → a_up + a_down = 2g
        a_up_val = self.results.get('a_up', 0)
        a_down_val = self.results.get('a_down', 0)
        if not math.isclose(a_up_val + a_down_val, 2*self.g, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Newton's law invariant a_up + a_down = 2g. {a_up_val}+{a_down_val} != {2*self.g}")
            success = False
        else:
            logs.append("PASS: Newton's law invariant a_up + a_down = 2g holds")

        # 3. Check ratio a_up/a_down = 3
        if a_down_val != 0:
            ratio = a_up_val / a_down_val
            if not math.isclose(ratio, 3.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Ratio a_up/a_down ≈ {ratio:.3f}, expected 3")
                success = False
            else:
                logs.append(f"PASS: Ratio a_up/a_down = {ratio:.3f} ≈ 3")

        # 4. Check energy/work-friction relation (optional but insightful)
        # Work done by friction = change in mechanical energy
        # Up: W_f = ΔE = (0 - 0.5*m*v0²) - m*g*L = -0.5*m*v0² - m*g*L
        # Down: W_f = ΔE = (0.5*m*v_down² - 0) - m*g*L = 0.5*m*v_down² - m*g*L
        # Since friction does negative work going up and negative going down (opposes motion),
        # we can check consistency: |W_f_up| should equal |W_f_down|? Not exactly because
        # displacements are same but friction constant? Actually, friction is constant (kinetic).
        # So W_f = -F_f * L (up) and W_f = -F_f * L (down) → same magnitude.
        # Therefore: 0.5*m*v0² + m*g*L = m*g*L - 0.5*m*v_down² → v0² + v_down² = 0? No.
        # Let's derive: Up: -F_f*L = -0.5*m*v0² - m*g*L → F_f = m*(g + v0²/(2L)) = m*(g + a_up)
        # Down: -F_f*L = 0.5*m*v_down² - m*g*L → F_f = m*(g - v_down²/(2L)) = m*(g - a_down)
        # Which is exactly the Newton's law we already checked. So skip.

        # 5. Check a_2 = a_down (from step 6)
        a_2_val = self.results.get('a_2', 0)
        if not math.isclose(a_2_val, a_down_val, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: a_2 should equal a_down. a_2={a_2_val}, a_down={a_down_val}")
            success = False
        else:
            logs.append(f"PASS: a_2 = a_down = {a_2_val:.3f}")

        return success, logs

    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        logs = []
        success = True

        # Extract values
        t_1 = self.results.get('t_1', 0)
        v_1 = self.results.get('v_1', 0)
        a_2 = self.results.get('a_2', 0)
        t_2 = self.results.get('t_2', 0)
        t_total = self.results.get('t_total', 0)
        v_base = self.results.get('v_base', 0)
        v_ring = self.results.get('v_ring', 0)
        t = self.results.get('t', 0)
        x_ring = self.results.get('x_ring', 0)
        x_base = self.results.get('x_base', 0)
        x = self.results.get('x', 0)

        # 1. Check free fall of base: H = 0.5*g*t1^2
        t1_expected = math.sqrt(2 * self.H / self.g)  # 0.2
        if not math.isclose(t_1, t1_expected, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: t_1 from free fall. Expected {t1_expected:.3f}, got {t_1:.3f}")
            success = False
        else:
            logs.append(f"PASS: t_1 = sqrt(2H/g) = {t_1:.3f} s")

        # 2. v1 = g*t1
        v1_expected = self.g * t_1
        if not math.isclose(v_1, v1_expected, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: v_1 = g*t1. Expected {v1_expected:.3f}, got {v_1:.3f}")
            success = False
        else:
            logs.append(f"PASS: v_1 = g*t1 = {v_1:.3f} m/s")

        # 3. t_2 from quadratic: L = v1*t2 + 0.5*a2*t2^2
        # Solve 0.5*a2*t2^2 + v1*t2 - L = 0
        disc = v_1**2 + 2 * a_2 * self.L
        if disc < 0:
            logs.append("FAIL: t_2 quadratic discriminant negative")
            success = False
        else:
            t2_pos = (-v_1 + math.sqrt(disc)) / a_2
            if not math.isclose(t_2, t2_pos, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: t_2 from quadratic. Expected {t2_pos:.3f}, got {t_2:.3f}")
                success = False
            else:
                logs.append(f"PASS: t_2 from quadratic = {t_2:.3f} s")

        # 4. t_total = t1 + t2
        if not math.isclose(t_total, t_1 + t_2, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: t_total = t1 + t2. {t_total} != {t_1}+{t_2}")
            success = False
        else:
            logs.append(f"PASS: t_total = t1 + t2 = {t_total:.3f} s")

        # 5. v_base = (3/4)*g*t1? Wait, step 9 says v_base = (3/4)*g*t1 = 1.5
        v_base_expected = 0.75 * self.g * t_1
        if not math.isclose(v_base, v_base_expected, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: v_base expected {v_base_expected:.3f}, got {v_base:.3f}")
            success = False
        else:
            logs.append(f"PASS: v_base = (3/4)*g*t1 = {v_base:.3f} m/s")

        # 6. v_ring = 2 m/s (given in step 10). Could also check consistency with collision?
        # We'll just check it's positive.
        if v_ring <= 0:
            logs.append(f"FAIL: v_ring should be positive, got {v_ring}")
            success = False
        else:
            logs.append(f"PASS: v_ring = {v_ring:.3f} m/s")

        # 7. t from condition v_ring - a_up*t = v_base - g*t
        a_up_val = self.results.get('a_up', 0)
        t_expected = (v_ring - v_base) / (a_up_val - self.g) if (a_up_val - self.g) != 0 else 0
        if not math.isclose(t, t_expected, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: t from velocity equality. Expected {t_expected:.3f}, got {t:.3f}")
            success = False
        else:
            logs.append(f"PASS: t = (v_ring - v_base)/(a_up - g) = {t:.3f} s")

        # 8. x_ring = v_ring*t - 0.5*a_up*t^2
        x_ring_expected = v_ring * t - 0.5 * a_up_val * t**2
        if not math.isclose(x_ring, x_ring_expected, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: x_ring kinematic. Expected {x_ring_expected:.3f}, got {x_ring:.3f}")
            success = False
        else:
            logs.append(f"PASS: x_ring = v_ring*t - 0.5*a_up*t² = {x_ring:.3f} m")

        # 9. x_base = v_base*t - 0.5*g*t^2
        x_base_expected = v_base * t - 0.5 * self.g * t**2
        if not math.isclose(x_base, x_base_expected, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: x_base kinematic. Expected {x_base_expected:.3f}, got {x_base:.3f}")
            success = False
        else:
            logs.append(f"PASS: x_base = v_base*t - 0.5*g*t² = {x_base:.3f} m")

        # 10. x = x_ring - x_base
        x_expected = x_ring - x_base
        if not math.isclose(x, x_expected, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: x = x_ring - x_base. Expected {x_expected:.3f}, got {x:.3f}")
            success = False
        else:
            logs.append(f"PASS: x = x_ring - x_base = {x:.3f} m")

        # 11. Final target answers
        ratio = self.results.get('fraca_upa_down', 0)
        if not math.isclose(ratio, 3.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Target ratio 3, got {ratio:.3f}")
            success = False
        if not math.isclose(t_total, 0.4, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Target t_total 0.4s, got {t_total:.3f}")
            success = False
        if not (math.isclose(t, 0.1, rel_tol=1e-2, abs_tol=1e-2) and
                math.isclose(x, 0.025, rel_tol=1e-2, abs_tol=1e-2)):
            logs.append(f"FAIL: Target (t, x) = (0.1s, 0.025m), got ({t:.3f}s, {x:.3f}m)")
            success = False

        return success, logs

    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        logs = []
        success = True

        # Perturb the initial upward velocity slightly and recompute a_up
        # Since v0_up is given as sqrt(15), we perturb by 1%
        v0_up_original = self.v0_up
        perturb_factor = 1.01
        v0_up_perturbed = v0_up_original * perturb_factor

        # Recompute a_up from kinematics: a_up = v0^2 / (2L)
        a_up_original = v0_up_original**2 / (2 * self.L)
        a_up_perturbed = v0_up_perturbed**2 / (2 * self.L)

        # Check monotonicity: if v0 increases, a_up should increase
        if a_up_perturbed <= a_up_original:
            logs.append(f"FAIL: Monotonicity w.r.t v0_up: a_up did not increase when v0 increased")
            success = False
        else:
            logs.append(f"PASS: a_up increases with v0_up (perturb +1%: {a_up_original:.3f} -> {a_up_perturbed:.3f})")

        # Restore original (not strictly needed but for clarity)
        v0_up_perturbed = v0_up_original

        # Perturb the rod length L and check the ratio a_up/a_down
        # Since a_up and a_down are proportional to 1/L, ratio should remain constant? Wait:
        # a_up = v0_up^2/(2L), a_down = v_down^2/(2L), so ratio = (v0_up^2)/(v_down^2) = 15/5 = 3, independent of L.
        # So ratio should not change. Let's verify.
        L_original = self.L
        L_perturbed = L_original * 1.01  # +1%
        ratio_original = a_up_original / (self.v_down**2 / (2 * L_original))
        ratio_perturbed = (v0_up_original**2 / (2 * L_perturbed)) / (self.v_down**2 / (2 * L_perturbed))
        if not math.isclose(ratio_original, ratio_perturbed, rel_tol=1e-10):
            logs.append(f"FAIL: Ratio a_up/a_down should be independent of L. Changed from {ratio_original} to {ratio_perturbed}")
            success = False
        else:
            logs.append(f"PASS: Ratio a_up/a_down constant (3) when L perturbed")

        # Perturb the gravitational acceleration g and check Newton's invariant
        g_original = self.g
        g_perturbed = g_original * 0.99  # -1%
        # a_up + a_down = 2g should hold with new g? Not necessarily, because a_up and a_down depend on friction.
        # Actually, from Newton's laws: a_up = g + F_f/m, a_down = g - F_f/m, so sum = 2g always.
        # Since we don't have F_f/m, we can't recompute a_up and a_down directly.
        # So skip this perturbation.

        return success, logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        all_logs = []
        overall_success = True

        # Test invariants
        success, logs = self.test_invariants()
        all_logs.append("=== Invariants Test ===")
        all_logs.extend(logs)
        if not success:
            overall_success = False

        # Test trace consistency
        success, logs = self.test_trace_consistency()
        all_logs.append("\n=== Trace Consistency Test ===")
        all_logs.extend(logs)
        if not success:
            overall_success = False

        # Test monotonicity
        success, logs = self.test_monotonicity()
        all_logs.append("\n=== Monotonicity Test ===")
        all_logs.extend(logs)
        if not success:
            overall_success = False

        return overall_success, all_logs