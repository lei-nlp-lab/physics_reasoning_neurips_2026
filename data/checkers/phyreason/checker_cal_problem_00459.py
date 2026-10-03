import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        # Constants
        self.c = 3e8  # m/s (used in steps)
        self.v = 0.8 * self.c  # given in problem

        # Given problem inputs
        self.t0 = 60.0  # round-trip time in spacecraft frame (given)

        # Extract results using safe attribute access
        self.x = self._safe_get(solution_instance, 'x')
        self.x2 = self._safe_get(solution_instance, 'x_2')
        self.t = self._safe_get(solution_instance, 't')
        self.x1 = self._safe_get(solution_instance, 'x_1')
        self.x1_plus_x2 = self._safe_get(solution_instance, 'x_1+x_2')
        self.x2_minus_x1 = self._safe_get(solution_instance, 'x_2-x_1')
        self.t1 = self._safe_get(solution_instance, 't_1')

        # Derived quantities (if not directly provided)
        if self.t1 is None:
            self.t1 = self.t0 / 2.0  # from time symmetry
        if self.x1 is None and self.t is not None:
            self.x1 = (self.c * self.t - self.v * self.t) / 2.0
        if self.x1_plus_x2 is None and self.t is not None:
            self.x1_plus_x2 = self.c * self.t
        if self.x2_minus_x1 is None and self.t is not None:
            self.x2_minus_x1 = self.v * self.t

    def _safe_get(self, solution_instance, key):
        """Safely get attribute from solution instance or its results dict."""
        val = getattr(solution_instance, key, None)
        if val is None:
            val = getattr(solution_instance, 'results', {}).get(key, None)
        return val

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy numerical comparison."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """Check fundamental physical invariants."""
        logs = []
        success = True

        # 1. Time symmetry in spacecraft frame: t1 = t0/2
        expected_t1 = self.t0 / 2.0
        if not self._is_close(self.t1, expected_t1):
            logs.append(f"Time symmetry failed: t1={self.t1}, expected {expected_t1}")
            success = False
        else:
            logs.append("Time symmetry OK.")

        # 2. Distance in spacecraft frame: x = c * t1
        expected_x = self.c * self.t1
        if not self._is_close(self.x, expected_x):
            logs.append(f"Spacecraft frame distance failed: x={self.x}, expected {expected_x}")
            success = False
        else:
            logs.append("Spacecraft frame distance OK.")

        # 3. Time dilation: t = t0 / sqrt(1 - v^2/c^2)
        gamma = 1.0 / math.sqrt(1.0 - (self.v/self.c)**2)
        expected_t = self.t0 * gamma
        if not self._is_close(self.t, expected_t):
            logs.append(f"Time dilation failed: t={self.t}, expected {expected_t}")
            success = False
        else:
            logs.append("Time dilation OK.")

        # 4. Earth frame simultaneous equations:
        #    x1 + x2 = c * t
        expected_sum = self.c * self.t
        if not self._is_close(self.x1_plus_x2, expected_sum):
            logs.append(f"Sum distance failed: x1+x2={self.x1_plus_x2}, expected {expected_sum}")
            success = False
        else:
            logs.append("Sum distance OK.")

        #    x2 - x1 = v * t
        expected_diff = self.v * self.t
        if not self._is_close(self.x2_minus_x1, expected_diff):
            logs.append(f"Difference distance failed: x2-x1={self.x2_minus_x1}, expected {expected_diff}")
            success = False
        else:
            logs.append("Difference distance OK.")

        # 5. Solve for x2: x2 = (c*t + v*t)/2
        expected_x2 = (self.c * self.t + self.v * self.t) / 2.0
        if not self._is_close(self.x2, expected_x2):
            logs.append(f"Earth frame reception distance failed: x2={self.x2}, expected {expected_x2}")
            success = False
        else:
            logs.append("Earth frame reception distance OK.")

        # 6. Solve for x1: x1 = (c*t - v*t)/2
        expected_x1 = (self.c * self.t - self.v * self.t) / 2.0
        if not self._is_close(self.x1, expected_x1):
            logs.append(f"Earth frame emission distance failed: x1={self.x1}, expected {expected_x1}")
            success = False
        else:
            logs.append("Earth frame emission distance OK.")

        return success, logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True

        # Step 1: t1 should be 30 s
        if not self._is_close(self.t1, 30.0):
            logs.append(f"Step 1 inconsistency: t1={self.t1}, expected 30 s")
            success = False

        # Step 2: x should be c * t1 = 9e9 m
        expected_x_step2 = self.c * 30.0
        if not self._is_close(self.x, expected_x_step2):
            logs.append(f"Step 2 inconsistency: x={self.x}, expected {expected_x_step2}")
            success = False

        # Step 6: t should be 100 s (from time dilation)
        expected_t_step6 = 100.0
        if not self._is_close(self.t, expected_t_step6):
            logs.append(f"Step 6 inconsistency: t={self.t}, expected {expected_t_step6}")
            success = False

        # Step 9: x2 should be (c*t + v*t)/2 = 2.7e10 m
        expected_x2_step9 = (self.c * 100.0 + self.v * 100.0) / 2.0
        if not self._is_close(self.x2, expected_x2_step9):
            logs.append(f"Step 9 inconsistency: x2={self.x2}, expected {expected_x2_step9}")
            success = False

        if success:
            logs.append("All trace steps consistent.")

        return success, logs

    def test_monotonicity(self):
        """
        Perturb the round-trip time t0 and check that distances change monotonically.
        """
        logs = []
        success = True

        original_t0 = self.t0
        perturbed_t0 = original_t0 * 1.1  # increase by 10%

        # Recalculate using formulas
        perturbed_t1 = perturbed_t0 / 2.0
        perturbed_x = self.c * perturbed_t1
        perturbed_gamma = 1.0 / math.sqrt(1.0 - (self.v/self.c)**2)
        perturbed_t = perturbed_t0 * perturbed_gamma
        perturbed_x2 = (self.c * perturbed_t + self.v * perturbed_t) / 2.0

        # Check monotonicity: increasing t0 should increase x and x2
        if not (perturbed_x > self.x and perturbed_x2 > self.x2):
            logs.append(f"Monotonicity failed: perturbed x={perturbed_x}, original x={self.x}; "
                        f"perturbed x2={perturbed_x2}, original x2={self.x2}")
            success = False
        else:
            logs.append(f"Monotonicity OK: increasing t0 increases both distances.")

        # Restore original t0 (though not strictly necessary as we work on a copy)
        self.t0 = original_t0

        return success, logs

    def run_all_tests(self):
        """Execute all tests and return aggregated results."""
        all_logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]

        for name, test_func in tests:
            success, logs = test_func()
            all_logs.append(f"=== {name} ===")
            all_logs.extend(logs)
            if not success:
                overall_success = False

        return overall_success, all_logs