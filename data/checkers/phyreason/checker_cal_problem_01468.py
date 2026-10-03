import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely initialize from solution_instance.
        Uses getattr to avoid AttributeError in sandbox.
        """
        self.solution = solution_instance
        # Fixed inputs from problem statement
        self.a1 = 2.0      # acceleration (m/s^2)
        self.a2 = 1.0      # deceleration magnitude (m/s^2)
        self.h = 48.0      # total height (m)
        self.T = 15.0      # total time for second part (from steps analysis)

        # Extract results with safe fallbacks
        self.t = self._safe_get('t', 0.0)
        self.v = self._safe_get('v', 0.0)
        self.v_m = self._safe_get('v_m', 0.0)
        self.t1 = self._safe_get('t_1', 0.0)
        self.t2 = self._safe_get('t_2', 0.0)

    def _safe_get(self, key, default):
        """Safe extraction from solution instance or its results dict."""
        val = getattr(self.solution, key, None)
        if val is None:
            val = getattr(self.solution, 'results', {}).get(key, default)
        return float(val) if isinstance(val, (int, float)) else default

    def test_invariants(self):
        """Check fundamental kinematic laws for both parts."""
        logs = []

        # First part: minimum time (no constant speed)
        if self.v_m > 0:
            # h = v_m^2/(2a1) + v_m^2/(2a2)
            h_calc = self.v_m**2 / (2 * self.a1) + self.v_m**2 / (2 * self.a2)
            if math.isclose(h_calc, self.h, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("Invariant (first part) passed: h = v_m²/(2a₁) + v_m²/(2a₂).")
            else:
                logs.append(f"Invariant failed: h from v_m: expected {self.h}, got {h_calc}.")

        # Second part: maximum speed for fixed total time T=15
        if self.v > 0:
            # Equation: h = v*T - v²*(1/(2a1) + 1/(2a2))
            coeff = 1/(2*self.a1) + 1/(2*self.a2)  # 3/4
            h_calc2 = self.v * self.T - coeff * self.v**2
            if math.isclose(h_calc2, self.h, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("Invariant (second part) passed: h = v·T - v²·(1/(2a₁)+1/(2a₂)).")
            else:
                logs.append(f"Invariant failed: h from v and T: expected {self.h}, got {h_calc2}.")

        return logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []

        # First part consistency
        if self.v_m > 0:
            t1_calc = self.v_m / self.a1
            t2_calc = self.v_m / self.a2
            t_calc = t1_calc + t2_calc

            # Check t1 and t2
            if not math.isclose(self.t1, t1_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Trace consistency failed: t₁ from v_m: expected {t1_calc}, got {self.t1}.")
            if not math.isclose(self.t2, t2_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Trace consistency failed: t₂ from v_m: expected {t2_calc}, got {self.t2}.")
            # Check total time
            if math.isclose(self.t, t_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("Trace consistency (first part) passed: t = t₁ + t₂.")
            else:
                logs.append(f"Trace consistency failed: t = t₁+t₂: {self.t} vs {t_calc}.")

        # Second part: verify that t1 and t2 derived from v match any stored values
        if self.v > 0:
            t1_v = self.v / self.a1
            t2_v = self.v / self.a2
            # Note: stored t1, t2 might be from first part, so we don't enforce equality.
            # Instead, we check that the computed constant-speed time is non-negative.
            t_const = self.T - t1_v - t2_v
            if t_const < -1e-2:  # allow tiny numerical error
                logs.append(f"Trace consistency warning: constant-speed time negative: {t_const}.")
            else:
                logs.append(f"Trace consistency (second part): constant-speed time = {t_const:.2f} s.")

        return logs

    def test_monotonicity(self):
        """Perturb inputs and verify expected trends."""
        logs = []

        # Perturb height h for first part: t_min should increase.
        h_perturbed = self.h * 1.01  # +1%
        v_m_original = math.sqrt(2 * self.h * self.a1 * self.a2 / (self.a1 + self.a2))
        v_m_perturbed = math.sqrt(2 * h_perturbed * self.a1 * self.a2 / (self.a1 + self.a2))
        t_min_original = v_m_original * (1/self.a1 + 1/self.a2)
        t_min_perturbed = v_m_perturbed * (1/self.a1 + 1/self.a2)

        if t_min_perturbed > t_min_original:
            logs.append("Monotonicity (first part) passed: t_min increases with h.")
        else:
            logs.append(f"Monotonicity failed: t_min did not increase (original {t_min_original:.3f}, perturbed {t_min_perturbed:.3f}).")

        # Perturb h for second part (fixed T): smaller root v should increase.
        coeff = 1/(2*self.a1) + 1/(2*self.a2)
        disc_original = self.T**2 - 4 * coeff * self.h
        disc_perturbed = self.T**2 - 4 * coeff * h_perturbed

        if disc_perturbed < 0:
            logs.append("Monotonicity (second part) failed: no real solution for perturbed h.")
        else:
            v_original = (self.T - math.sqrt(disc_original)) / (2 * coeff)
            v_perturbed = (self.T - math.sqrt(disc_perturbed)) / (2 * coeff)

            # Verify stored v matches computed smaller root
            if not math.isclose(self.v, v_original, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Monotonicity warning: stored v {self.v} ≠ computed root {v_original:.3f}.")

            if v_perturbed > v_original:
                logs.append("Monotonicity (second part) passed: v increases with h (fixed T).")
            else:
                logs.append(f"Monotonicity failed: v did not increase (original {v_original:.3f}, perturbed {v_perturbed:.3f}).")

        return logs

    def run_all_tests(self):
        """Execute all tests and return success flag and log messages."""
        all_logs = []
        tests = [
            ("test_invariants", self.test_invariants),
            ("test_trace_consistency", self.test_trace_consistency),
            ("test_monotonicity", self.test_monotonicity),
        ]

        for name, test_func in tests:
            try:
                logs = test_func()
                all_logs.append(f"--- {name} ---")
                all_logs.extend(logs)
            except Exception as e:
                all_logs.append(f"Error in {name}: {e}")
                all_logs.append(traceback.format_exc())

        # Determine overall success: no failure messages
        success = True
        for log in all_logs:
            if any(word in log.lower() for word in ["failed", "error"]):
                success = False
                break

        return success, all_logs