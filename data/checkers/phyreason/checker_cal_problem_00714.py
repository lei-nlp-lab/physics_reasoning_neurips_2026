import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs (with fallback to problem defaults)
        self.k = self._safe_get(solution_instance, 'k', 100.0)
        self.L = self._safe_get(solution_instance, 'L', 2.0)
        self.m = self._safe_get(solution_instance, 'm', 0.1)
        self.I = self._safe_get(solution_instance, 'I', 3.0)
        self.x = self._safe_get(solution_instance, 'x', 0.02)   # 2 cm in meters
        self.g = self._safe_get(solution_instance, 'g', 10.0)

        # Safe extraction of results
        self.F_a = self._safe_get_result(solution_instance, 'F_a')
        self.B = self._safe_get_result(solution_instance, 'B')

        # Store for monotonicity restoration
        self._original_m = self.m
        self._original_I = self.I

    def _safe_get(self, obj, attr, default):
        """Safely get attribute from instance or its inputs dict."""
        try:
            val = getattr(obj, attr, None)
            if val is None:
                inputs = getattr(obj, 'inputs', {})
                val = inputs.get(attr, default)
            return float(val)
        except Exception:
            return default

    def _safe_get_result(self, obj, key):
        """Safely get result from instance or its results dict."""
        try:
            val = getattr(obj, key, None)
            if val is None:
                results = getattr(obj, 'results', {})
                val = results.get(key, None)
            if val is not None:
                # If it's a string like "3N", extract number
                if isinstance(val, str):
                    import re
                    match = re.search(r'[-+]?\d*\.?\d+', val)
                    if match:
                        val = float(match.group())
                return float(val)
        except Exception:
            pass
        return None

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        logs = []
        success = True

        # 1. Vertical force equilibrium: 2*k*x = m*g + F_a
        if self.F_a is not None:
            equilibrium = 2 * self.k * self.x - self.m * self.g - self.F_a
            if self._is_close(equilibrium, 0.0):
                logs.append("PASS: Vertical force equilibrium holds.")
            else:
                logs.append(f"FAIL: Vertical equilibrium deviation = {equilibrium:.3f}")
                success = False
        else:
            logs.append("SKIP: F_a not available for equilibrium check.")
            success = False

        # 2. Ampere's force law: F_a = I * L * B
        if self.F_a is not None and self.B is not None:
            ampere_law = self.F_a - self.I * self.L * self.B
            if self._is_close(ampere_law, 0.0):
                logs.append("PASS: Ampere's force law F_a = I*L*B holds.")
            else:
                logs.append(f"FAIL: Ampere's law deviation = {ampere_law:.3f}")
                success = False
        else:
            logs.append("SKIP: F_a or B missing for Ampere's law check.")
            success = False

        # 3. Direction consistency (F_a should be positive for downward)
        if self.F_a is not None:
            if self.F_a > 0:
                logs.append("PASS: F_a positive (consistent with downward direction).")
            else:
                logs.append(f"WARN: F_a is {self.F_a}, expected positive for downward force.")
        else:
            logs.append("SKIP: F_a missing for direction check.")

        return success, logs

    def test_trace_consistency(self):
        logs = []
        success = True

        # Compare with expected intermediate values from steps analysis
        expected_F_a = 3.0
        expected_B = 0.5

        if self.F_a is not None:
            if self._is_close(self.F_a, expected_F_a):
                logs.append(f"PASS: F_a = {self.F_a:.3f} matches expected ~3 N.")
            else:
                logs.append(f"FAIL: F_a = {self.F_a:.3f}, expected ~{expected_F_a} N.")
                success = False
        else:
            logs.append("SKIP: F_a missing for trace check.")
            success = False

        if self.B is not None:
            if self._is_close(self.B, expected_B):
                logs.append(f"PASS: B = {self.B:.3f} matches expected ~0.5 T.")
            else:
                logs.append(f"FAIL: B = {self.B:.3f}, expected ~{expected_B} T.")
                success = False
        else:
            logs.append("SKIP: B missing for trace check.")
            success = False

        return success, logs

    def test_monotonicity(self):
        logs = []
        success = True

        # Perturb mass m upward by 1%
        perturb_factor = 1.01
        m_perturbed = self.m * perturb_factor

        # Compute B_perturbed using the derived formula B = (2*k*x - m*g) / (I*L)
        if all(v is not None for v in [self.k, self.x, self.I, self.L]):
            B_perturbed = (2 * self.k * self.x - m_perturbed * self.g) / (self.I * self.L)

            if self.B is not None:
                # Expected: B should decrease when m increases
                if B_perturbed < self.B:
                    logs.append(f"PASS: Increasing mass reduces B ({B_perturbed:.3f} < {self.B:.3f}).")
                else:
                    logs.append(f"FAIL: Monotonicity violated: B_perturbed = {B_perturbed:.3f}, original B = {self.B:.3f}.")
                    success = False
            else:
                logs.append("SKIP: Original B missing for monotonicity check.")
                success = False
        else:
            logs.append("SKIP: Missing parameters for monotonicity computation.")
            success = False

        # Restore original m (just in case)
        self.m = self._original_m

        return success, logs

    def run_all_tests(self):
        all_logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False

        return overall_success, all_logs