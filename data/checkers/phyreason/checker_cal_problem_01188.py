import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Constants
        self.g = 10.0
        self.theta = 37.0  # degrees, but sin/cos values given
        self.sin_theta = 0.6
        self.cos_theta = 0.8
        self.m_A = 10.0
        self.m_B = 14.0
        self.mu1 = 0.5
        self.mu2 = 0.4

        # Safe extraction of results
        self.T = self._safe_get(solution_instance, 'T')
        self.F = self._safe_get(solution_instance, 'F')
        self.f = self._safe_get(solution_instance, 'f')  # f_1 from step 4
        self.f_1 = self._safe_get(solution_instance, 'f_1', default=self.f)
        self.f_2 = self._safe_get(solution_instance, 'f_2')
        self.N_1 = self._safe_get(solution_instance, 'N_1')
        self.N_2 = self._safe_get(solution_instance, 'N_2')

        # Store originals for perturbation tests
        self.original_m_A = self.m_A
        self.original_m_B = self.m_B

    def _safe_get(self, obj, key, default=None):
        """Safely retrieve attribute from solution_instance or its results dict."""
        val = getattr(obj, key, None)
        if val is None:
            results = getattr(obj, 'results', {})
            val = results.get(key, default)
        return val

    def test_invariants(self):
        """Check fundamental physics laws (equilibrium, friction)."""
        logs = []
        success = True

        # 1. f_1 = T * cosθ
        if self.T is not None and self.f_1 is not None:
            expected = self.T * self.cos_theta
            if not math.isclose(self.f_1, expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: f_1 = {self.f_1} ≠ T*cosθ = {expected}")
                success = False
            else:
                logs.append(f"OK: f_1 = T*cosθ = {self.f_1}")

        # 2. N_1 = T * sinθ + m_A*g
        if self.T is not None and self.N_1 is not None:
            expected = self.T * self.sin_theta + self.m_A * self.g
            if not math.isclose(self.N_1, expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: N_1 = {self.N_1} ≠ T*sinθ + m_A*g = {expected}")
                success = False
            else:
                logs.append(f"OK: N_1 = T*sinθ + m_A*g = {self.N_1}")

        # 3. f_1 = μ1 * N_1
        if self.f_1 is not None and self.N_1 is not None:
            expected = self.mu1 * self.N_1
            if not math.isclose(self.f_1, expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: f_1 = {self.f_1} ≠ μ1*N_1 = {expected}")
                success = False
            else:
                logs.append(f"OK: f_1 = μ1*N_1 = {self.f_1}")

        # 4. N_2 = N_1 + m_B*g
        if self.N_1 is not None and self.N_2 is not None:
            expected = self.N_1 + self.m_B * self.g
            if not math.isclose(self.N_2, expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: N_2 = {self.N_2} ≠ N_1 + m_B*g = {expected}")
                success = False
            else:
                logs.append(f"OK: N_2 = N_1 + m_B*g = {self.N_2}")

        # 5. f_2 = μ2 * N_2
        if self.f_2 is not None and self.N_2 is not None:
            expected = self.mu2 * self.N_2
            if not math.isclose(self.f_2, expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: f_2 = {self.f_2} ≠ μ2*N_2 = {expected}")
                success = False
            else:
                logs.append(f"OK: f_2 = μ2*N_2 = {self.f_2}")

        # 6. F = f_1 + f_2
        if self.F is not None and self.f_1 is not None and self.f_2 is not None:
            expected = self.f_1 + self.f_2
            if not math.isclose(self.F, expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: F = {self.F} ≠ f_1 + f_2 = {expected}")
                success = False
            else:
                logs.append(f"OK: F = f_1 + f_2 = {self.F}")

        return success, logs

    def test_trace_consistency(self):
        """Cross-check intermediate and final results against expected values."""
        logs = []
        success = True
        target_T = 100.0
        target_F = 200.0

        # Tension
        if self.T is not None:
            if not math.isclose(self.T, target_T, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: T = {self.T} ≠ target {target_T}")
                success = False
            else:
                logs.append(f"OK: T = {target_T}")

        # External force
        if self.F is not None:
            if not math.isclose(self.F, target_F, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: F = {self.F} ≠ target {target_F}")
                success = False
            else:
                logs.append(f"OK: F = {target_F}")

        # Intermediate values (from steps)
        if self.f_1 is not None and not math.isclose(self.f_1, 80.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: f_1 = {self.f_1} ≠ expected 80")
            success = False
        else:
            logs.append("OK: f_1 = 80")

        if self.N_1 is not None and not math.isclose(self.N_1, 160.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: N_1 = {self.N_1} ≠ expected 160")
            success = False
        else:
            logs.append("OK: N_1 = 160")

        if self.N_2 is not None and not math.isclose(self.N_2, 300.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: N_2 = {self.N_2} ≠ expected 300")
            success = False
        else:
            logs.append("OK: N_2 = 300")

        if self.f_2 is not None and not math.isclose(self.f_2, 120.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: f_2 = {self.f_2} ≠ expected 120")
            success = False
        else:
            logs.append("OK: f_2 = 120")

        return success, logs

    def test_monotonicity(self):
        """Perturb masses and verify expected trends in F."""
        logs = []
        success = True

        # Helper to compute F given masses
        def compute_F(mA, mB):
            T = (self.mu1 * mA * self.g) / (self.cos_theta - self.mu1 * self.sin_theta)
            N1 = T * self.sin_theta + mA * self.g
            f1 = self.mu1 * N1
            N2 = N1 + mB * self.g
            f2 = self.mu2 * N2
            return f1 + f2

        F_orig = compute_F(self.m_A, self.m_B)

        # Perturb m_A +1 kg
        F_new_A = compute_F(self.m_A + 1.0, self.m_B)
        if not F_new_A > F_orig:
            logs.append(f"FAIL: F does not increase when m_A increases. F_orig={F_orig}, F_new={F_new_A}")
            success = False
        else:
            logs.append(f"OK: F increases from {F_orig:.1f} to {F_new_A:.1f} when m_A increases by 1 kg")

        # Perturb m_B +1 kg
        F_new_B = compute_F(self.m_A, self.m_B + 1.0)
        if not F_new_B > F_orig:
            logs.append(f"FAIL: F does not increase when m_B increases. F_orig={F_orig}, F_new={F_new_B}")
            success = False
        else:
            logs.append(f"OK: F increases from {F_orig:.1f} to {F_new_B:.1f} when m_B increases by 1 kg")

        return success, logs

    def run_all_tests(self):
        """Execute all tests and return aggregate result."""
        all_logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for name, test_func in tests:
            all_logs.append(f"\n=== {name} Test ===")
            try:
                success, logs = test_func()
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR during {name}: {traceback.format_exc()}")
                overall_success = False

        return overall_success, all_logs