import math
import sympy as sp
from sympy import symbols, sin, cos, tan, sqrt, Eq, solve, simplify

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        # Input parameters
        self.m = self._safe_get(solution_instance, 'm')
        self.mu = self._safe_get(solution_instance, 'mu')
        self.theta = self._safe_get(solution_instance, 'theta')
        self.g = self._safe_get(solution_instance, 'g', 9.8)

        # Results (with fallbacks)
        self.F_min = self._safe_get(solution_instance, 'F_min')
        self.F_max = self._safe_get(solution_instance, 'F_max')
        self.a_min = self._safe_get(solution_instance, 'a_min')
        self.a_max = self._safe_get(solution_instance, 'a_max')
        self.F = self._safe_get(solution_instance, 'F')  # optional, if provided

        # Internal state for logs
        self.logs = []

    def _safe_get(self, obj, key, default=None):
        """
        Safely retrieve an attribute or dictionary value with fallback.
        """
        val = getattr(obj, key, None)
        if val is None:
            # Try results dictionary if it exists
            results = getattr(obj, 'results', {})
            val = results.get(key, default)
        return val

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """
        Fuzzy numerical matching with tolerance.
        """
        # Handle None and sympy expressions
        if a is None or b is None:
            return False
        try:
            a_val = float(a) if not isinstance(a, (int, float)) else a
            b_val = float(b) if not isinstance(b, (int, float)) else b
        except (TypeError, ValueError):
            # Symbolic comparison
            try:
                diff = simplify(a - b)
                return diff == 0 or abs(float(diff.evalf())) <= abs_tol
            except:
                return False
        return math.isclose(a_val, b_val, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """
        Verify Newton's Second Law and static friction conditions at boundaries.
        """
        logs = []
        success = True

        # Check 1: System acceleration formula from step_1: a = (F - 3μ m g) / (3m)
        # For F_min and F_max, compute a_sys and compare with provided a_min and a_max
        if self.F_min is not None:
            a_sys_min = (self.F_min - 3 * self.mu * self.m * self.g) / (3 * self.m)
            if self.a_min is not None:
                if not self._is_close(a_sys_min, self.a_min):
                    logs.append(f"FAIL: a_sys from F_min {a_sys_min} != stored a_min {self.a_min}")
                    success = False
                else:
                    logs.append("PASS: a_sys from F_min matches stored a_min")
            # Also compute theoretical a_min_boundary from step_3
            denom = cos(self.theta) + self.mu * sin(self.theta)
            if abs(denom) > 1e-12:
                a_min_boundary = self.g * (sin(self.theta) - self.mu * cos(self.theta)) / denom
                if not self._is_close(a_sys_min, a_min_boundary):
                    logs.append(f"FAIL: a_sys from F_min {a_sys_min} != theoretical a_min_boundary {a_min_boundary}")
                    success = False
                else:
                    logs.append("PASS: a_sys from F_min matches theoretical a_min_boundary")
            else:
                logs.append("WARN: denominator near zero in a_min_boundary, skipping check")

        if self.F_max is not None:
            a_sys_max = (self.F_max - 3 * self.mu * self.m * self.g) / (3 * self.m)
            if self.a_max is not None:
                if not self._is_close(a_sys_max, self.a_max):
                    logs.append(f"FAIL: a_sys from F_max {a_sys_max} != stored a_max {self.a_max}")
                    success = False
                else:
                    logs.append("PASS: a_sys from F_max matches stored a_max")
            # Theoretical a_max_boundary from step_6
            denom = cos(self.theta) - self.mu * sin(self.theta)
            if abs(denom) > 1e-12:
                a_max_boundary = self.g * (sin(self.theta) + self.mu * cos(self.theta)) / denom
                if not self._is_close(a_sys_max, a_max_boundary):
                    logs.append(f"FAIL: a_sys from F_max {a_sys_max} != theoretical a_max_boundary {a_max_boundary}")
                    success = False
                else:
                    logs.append("PASS: a_sys from F_max matches theoretical a_max_boundary")
            else:
                logs.append("WARN: denominator near zero in a_max_boundary, skipping check")

        # Check 2: Static friction condition for top wedge at boundaries
        # For F_min (impending slip down): equations from step_2
        #   N cosθ + f sinθ = m g
        #   N sinθ - f cosθ = m a
        # with f = μ N
        if self.F_min is not None and self.a_min is not None:
            N, f = symbols('N f')
            a_val = a_sys_min if self.a_min is None else self.a_min
            eq1 = Eq(N * cos(self.theta) + f * sin(self.theta), self.m * self.g)
            eq2 = Eq(N * sin(self.theta) - f * cos(self.theta), self.m * a_val)
            sol = solve((eq1, eq2), (N, f), dict=True)
            if sol:
                N_val = sol[0][N]
                f_val = sol[0][f]
                ratio = f_val / N_val
                if not self._is_close(ratio, self.mu):
                    logs.append(f"FAIL: f/N = {ratio} != μ = {self.mu} at F_min")
                    success = False
                else:
                    logs.append("PASS: Static friction condition holds at F_min")

        # For F_max (impending slip up): equations from step_5
        #   N cosθ - f sinθ = m g
        #   N sinθ + f cosθ = m a
        # with f = μ N
        if self.F_max is not None and self.a_max is not None:
            N, f = symbols('N f')
            a_val = a_sys_max if self.a_max is None else self.a_max
            eq1 = Eq(N * cos(self.theta) - f * sin(self.theta), self.m * self.g)
            eq2 = Eq(N * sin(self.theta) + f * cos(self.theta), self.m * a_val)
            sol = solve((eq1, eq2), (N, f), dict=True)
            if sol:
                N_val = sol[0][N]
                f_val = sol[0][f]
                ratio = f_val / N_val
                if not self._is_close(ratio, self.mu):
                    logs.append(f"FAIL: f/N = {ratio} != μ = {self.mu} at F_max")
                    success = False
                else:
                    logs.append("PASS: Static friction condition holds at F_max")

        self.logs.extend(logs)
        return success

    def test_trace_consistency(self):
        """
        Cross-check intermediate formulas with final results.
        """
        logs = []
        success = True

        # Compute F_min and F_max from the formulas in steps 4 and 7
        # Step 4: F_min = 3 m g (μ + (sinθ - μ cosθ)/(cosθ + μ sinθ))
        denom_min = cos(self.theta) + self.mu * sin(self.theta)
        if abs(denom_min) > 1e-12:
            F_min_formula = 3 * self.m * self.g * (self.mu + (sin(self.theta) - self.mu * cos(self.theta)) / denom_min)
            if self.F_min is not None and not self._is_close(self.F_min, F_min_formula):
                logs.append(f"FAIL: stored F_min {self.F_min} != formula F_min {F_min_formula}")
                success = False
            else:
                logs.append("PASS: F_min matches formula from step_4")
        else:
            logs.append("WARN: denominator near zero for F_min formula, skipping")

        # Step 7: F_max = 3 m g (μ + (sinθ + μ cosθ)/(cosθ - μ sinθ))
        denom_max = cos(self.theta) - self.mu * sin(self.theta)
        if abs(denom_max) > 1e-12:
            F_max_formula = 3 * self.m * self.g * (self.mu + (sin(self.theta) + self.mu * cos(self.theta)) / denom_max)
            if self.F_max is not None and not self._is_close(self.F_max, F_max_formula):
                logs.append(f"FAIL: stored F_max {self.F_max} != formula F_max {F_max_formula}")
                success = False
            else:
                logs.append("PASS: F_max matches formula from step_7")
        else:
            logs.append("WARN: denominator near zero for F_max formula, skipping")

        # Check a_min and a_max from steps 3 and 6
        if self.a_min is not None:
            a_min_formula = self.g * (sin(self.theta) - self.mu * cos(self.theta)) / denom_min
            if not self._is_close(self.a_min, a_min_formula):
                logs.append(f"FAIL: stored a_min {self.a_min} != formula a_min {a_min_formula}")
                success = False
            else:
                logs.append("PASS: a_min matches formula from step_3")

        if self.a_max is not None:
            a_max_formula = self.g * (sin(self.theta) + self.mu * cos(self.theta)) / denom_max
            if not self._is_close(self.a_max, a_max_formula):
                logs.append(f"FAIL: stored a_max {self.a_max} != formula a_max {a_max_formula}")
                success = False
            else:
                logs.append("PASS: a_max matches formula from step_6")

        self.logs.extend(logs)
        return success

    def test_monotonicity(self):
        """
        Perturb mass m and check that F_min and F_max scale linearly.
        """
        logs = []
        success = True

        if self.F_min is None or self.F_max is None:
            logs.append("SKIP: F_min or F_max not available for monotonicity test")
            self.logs.extend(logs)
            return False

        # Perturb m by 1%
        m_perturbed = self.m * 1.01
        # Recompute F_min and F_max using the formulas (same as in trace_consistency)
        denom_min = cos(self.theta) + self.mu * sin(self.theta)
        denom_max = cos(self.theta) - self.mu * sin(self.theta)
        if abs(denom_min) > 1e-12 and abs(denom_max) > 1e-12:
            F_min_perturbed = 3 * m_perturbed * self.g * (self.mu + (sin(self.theta) - self.mu * cos(self.theta)) / denom_min)
            F_max_perturbed = 3 * m_perturbed * self.g * (self.mu + (sin(self.theta) + self.mu * cos(self.theta)) / denom_max)

            # Expected scaling factor is 1.01 (linear in m)
            expected_factor = 1.01
            actual_factor_min = F_min_perturbed / self.F_min
            actual_factor_max = F_max_perturbed / self.F_max

            if not self._is_close(actual_factor_min, expected_factor):
                logs.append(f"FAIL: F_min scaling factor {actual_factor_min} != {expected_factor}")
                success = False
            else:
                logs.append("PASS: F_min scales linearly with m")

            if not self._is_close(actual_factor_max, expected_factor):
                logs.append(f"FAIL: F_max scaling factor {actual_factor_max} != {expected_factor}")
                success = False
            else:
                logs.append("PASS: F_max scales linearly with m")
        else:
            logs.append("WARN: denominator near zero, skipping monotonicity test")

        self.logs.extend(logs)
        return success

    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        self.logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]

        for test_name, test_func in tests:
            self.logs.append(f"\n--- Running {test_name} Test ---")
            try:
                success = test_func()
                overall_success = overall_success and success
                self.logs.append(f"{test_name} Test: {'PASS' if success else 'FAIL'}")
            except Exception as e:
                self.logs.append(f"{test_name} Test raised exception: {e}")
                overall_success = False

        return overall_success, self.logs