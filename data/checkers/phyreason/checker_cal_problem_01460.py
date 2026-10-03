import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the PhysicsChecker with safe attribute access.
        """
        self.solution = solution_instance

        # Extract inputs with safe defaults
        self.m1 = self._safe_get('m1', 1.0)
        self.m2 = self._safe_get('m2', 1.0)
        self.theta = self._safe_get('theta', 0.0)
        self.g0 = self._safe_get('g0', 1.62)
        self.a = self._safe_get('a', 1.0)

        # Extract results with safe defaults
        self.F = self._safe_get('F', 0.0)
        self.T_x = self._safe_get('Tₓ', self._safe_get('T_x', 0.0))
        self.T_y = self._safe_get('T_y', 0.0)
        self.T = self._safe_get('T', 0.0)

    def _safe_get(self, key, default):
        """
        Safely retrieve an attribute from the solution instance.
        First tries direct attribute, then results dictionary.
        """
        if hasattr(self.solution, key):
            return getattr(self.solution, key)
        elif hasattr(self.solution, 'results') and isinstance(self.solution.results, dict):
            return self.solution.results.get(key, default)
        else:
            return default

    def test_invariants(self):
        """
        Test fundamental physics invariants:
        1. Newton's First Law: Net force on stationary combined vehicle is zero.
        2. Newton's Second Law: Thrust components satisfy F_net = m*a.
        Returns a list of log messages.
        """
        logs = []
        try:
            # 1. Check Newton's First Law for the combined vehicle (stationary)
            expected_F = (self.m1 + self.m2) * self.g0
            if not math.isclose(self.F, expected_F, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Force F = {self.F} does not match expected weight (m1+m2)*g0 = {expected_F}")
            else:
                logs.append("PASS: Newton's First Law (equilibrium) satisfied for combined vehicle.")

            # 2. Check Newton's Second Law for the ascent vehicle (thrust components)
            # Expected T_x from step 2
            expected_T_x = self.m1 * self.g0 * math.sin(self.theta)
            if not math.isclose(self.T_x, expected_T_x, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: T_x = {self.T_x} does not match m1*g0*sinθ = {expected_T_x}")

            # Expected T_y from step 3 (with a=0 for stationary? But note: during liftoff, a>0)
            # Actually, during liftoff, the vehicle is accelerating, so T_y = m1*(g0*cosθ + a)
            expected_T_y = self.m1 * (self.g0 * math.cos(self.theta) + self.a)
            if not math.isclose(self.T_y, expected_T_y, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: T_y = {self.T_y} does not match m1*(g0*cosθ + a) = {expected_T_y}")

            if math.isclose(self.T_x, expected_T_x, rel_tol=1e-2, abs_tol=1e-2) and \
               math.isclose(self.T_y, expected_T_y, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("PASS: Newton's Second Law satisfied for thrust components.")

            # 3. Check magnitude T from components
            expected_T_from_components = math.sqrt(self.T_x**2 + self.T_y**2)
            if not math.isclose(self.T, expected_T_from_components, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: T = {self.T} does not match sqrt(T_x² + T_y²) = {expected_T_from_components}")
            else:
                logs.append("PASS: Thrust magnitude consistent with components.")

        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
        return logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results (T_x, T_y, T) with the final expression for T.
        Returns a list of log messages.
        """
        logs = []
        try:
            # Final expression for T from step 5
            expected_T_expr = self.m1 * math.sqrt(
                self.g0**2 + self.a**2 + 2 * self.g0 * self.a * math.cos(self.theta)
            )
            if not math.isclose(self.T, expected_T_expr, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(
                    f"FAIL: T = {self.T} does not match m1*sqrt(g0² + a² + 2*g0*a*cosθ) = {expected_T_expr}"
                )
            else:
                logs.append("PASS: Final thrust T matches the derived expression.")

            # Check that T_x and T_y are consistent with the expression
            # T_x^2 + T_y^2 = m1^2 * (g0^2 + a^2 + 2*g0*a*cosθ)
            lhs = self.T_x**2 + self.T_y**2
            rhs = self.m1**2 * (self.g0**2 + self.a**2 + 2 * self.g0 * self.a * math.cos(self.theta))
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(
                    f"FAIL: T_x² + T_y² = {lhs} does not match m1²*(g0² + a² + 2*g0*a*cosθ) = {rhs}"
                )
            else:
                logs.append("PASS: Intermediate components T_x, T_y are consistent with the final T.")

        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
        return logs

    def test_monotonicity(self):
        """
        Perturb input parameters and check expected trends.
        We perturb m1 and a, then restore original values.
        Returns a list of log messages.
        """
        logs = []
        try:
            original_m1 = self.m1
            original_a = self.a

            # 1. Perturb m1 by +1%
            delta_m1 = 0.01 * original_m1
            self.m1 = original_m1 + delta_m1

            # Recompute T using the formula (we cannot call the solution again)
            T_perturbed_m1 = self.m1 * math.sqrt(
                self.g0**2 + self.a**2 + 2 * self.g0 * self.a * math.cos(self.theta)
            )
            T_original = self.T  # original T from results
            expected_T_m1 = T_original * (1 + delta_m1 / original_m1)  # linear scaling

            if not math.isclose(T_perturbed_m1, expected_T_m1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(
                    f"FAIL: Perturbed T (m1+1%) = {T_perturbed_m1} does not match expected linear scaling = {expected_T_m1}"
                )
            else:
                logs.append("PASS: Thrust T scales linearly with m1 (monotonicity).")

            # Restore m1
            self.m1 = original_m1

            # 2. Perturb a by +1%
            delta_a = 0.01 * original_a
            self.a = original_a + delta_a

            # Recompute T
            T_perturbed_a = self.m1 * math.sqrt(
                self.g0**2 + self.a**2 + 2 * self.g0 * self.a * math.cos(self.theta)
            )

            # Since T is increasing in a (for a>0 and cosθ>0), check T_perturbed_a > T_original
            if T_perturbed_a <= T_original:
                logs.append(
                    f"FAIL: Perturbed T (a+1%) = {T_perturbed_a} is not greater than original T = {T_original}"
                )
            else:
                logs.append("PASS: Thrust T increases with acceleration a (monotonicity).")

            # Restore a
            self.a = original_a

        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            # Restore original values in case of error
            self.m1 = original_m1 if 'original_m1' in locals() else self.m1
            self.a = original_a if 'original_a' in locals() else self.a
        return logs

    def run_all_tests(self):
        """
        Run all physics audits and return a success flag and log list.
        """
        all_logs = []
        success = True

        try:
            # Run each test
            tests = [
                ("Invariant Tests", self.test_invariants),
                ("Trace Consistency Tests", self.test_trace_consistency),
                ("Monotonicity Tests", self.test_monotonicity),
            ]

            for test_name, test_func in tests:
                all_logs.append(f"--- {test_name} ---")
                logs = test_func()
                all_logs.extend(logs)
                # Check for any FAIL or ERROR messages
                for log in logs:
                    if log.startswith("FAIL") or log.startswith("ERROR"):
                        success = False

        except Exception as e:
            all_logs.append(f"CRITICAL ERROR in run_all_tests: {traceback.format_exc()}")
            success = False

        return success, all_logs