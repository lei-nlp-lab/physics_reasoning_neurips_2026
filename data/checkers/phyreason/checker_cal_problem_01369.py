import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the checker with the solution instance.
        Use safe attribute access to avoid AttributeError in sandbox.
        """
        # Fixed inputs from the problem statement
        self.d = 60.0
        self.v1 = 6.0
        self.v2 = 3.0

        # Extract results from solution_instance with safe fallbacks
        self.t_min = getattr(solution_instance, 't_min', solution_instance.results.get('t_min', 0.0))
        self.x = getattr(solution_instance, 'x', solution_instance.results.get('x', 0.0))
        self.theta = getattr(solution_instance, 'theta', solution_instance.results.get('theta', 0.0))

        # Store the solution instance for potential future use (if needed)
        self.solution = solution_instance

    def test_invariants(self):
        """
        Verify that the results satisfy fundamental physics laws.
        """
        logs = []
        success = True

        # 1. Check t_min = d / v2 (minimum time to cross by pointing directly across)
        expected_t_min = self.d / self.v2
        if not math.isclose(self.t_min, expected_t_min, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Invariant violated: t_min = {self.t_min} but expected d/v2 = {expected_t_min}")
            success = False
        else:
            logs.append(f"t_min = d/v2 check passed: {self.t_min} ≈ {expected_t_min}")

        # 2. Check cos(theta) = v2 / v1 (condition for shortest crossing distance)
        cos_theta = math.cos(math.radians(self.theta))
        expected_cos = self.v2 / self.v1
        if not math.isclose(cos_theta, expected_cos, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Invariant violated: cos(theta) = {cos_theta} but expected v2/v1 = {expected_cos}")
            success = False
        else:
            logs.append(f"cos(theta) = v2/v1 check passed: {cos_theta} ≈ {expected_cos}")

        # 3. Check x = d / cos(theta) (shortest crossing distance)
        expected_x = self.d / cos_theta
        if not math.isclose(self.x, expected_x, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Invariant violated: x = {self.x} but expected d/cos(theta) = {expected_x}")
            success = False
        else:
            logs.append(f"x = d/cos(theta) check passed: {self.x} ≈ {expected_x}")

        return success, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with the final answer.
        """
        logs = []
        success = True

        # Compute intermediate quantities from the results
        sin_theta = math.sin(math.radians(self.theta))
        # Time to cross for the shortest distance path
        t_x = self.d / (self.v2 * sin_theta)
        # Ground speed magnitude from parallelogram rule
        v_ground = math.sqrt(self.v1**2 + self.v2**2 - 2 * self.v1 * self.v2 * math.cos(math.radians(self.theta)))

        # Check that x = v_ground * t_x
        expected_x = v_ground * t_x
        if not math.isclose(self.x, expected_x, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Trace inconsistency: x = {self.x} but v_ground * t_x = {expected_x}")
            success = False
        else:
            logs.append(f"x = v_ground * t_x check passed: {self.x} ≈ {expected_x}")

        # Check that t_x > t_min (since shortest distance takes longer than minimum time)
        if not (t_x > self.t_min and not math.isclose(t_x, self.t_min, rel_tol=1e-2, abs_tol=1e-2)):
            logs.append(f"Trace inconsistency: t_x = {t_x} should be greater than t_min = {self.t_min}")
            success = False
        else:
            logs.append(f"t_x > t_min check passed: {t_x} > {self.t_min}")

        return success, logs

    def test_monotonicity(self):
        """
        Perturb an input (v2) and verify that outputs change in the expected direction.
        """
        logs = []
        success = True

        # Store original values
        original_v2 = self.v2
        original_t_min = self.t_min
        original_x = self.x

        # Perturb v2 by +10%
        perturbed_v2 = original_v2 * 1.1

        # Recalculate expected results using physics formulas
        perturbed_t_min = self.d / perturbed_v2
        perturbed_cos_theta = perturbed_v2 / self.v1
        # Ensure cos does not exceed 1 (though with v2=3.3, v1=6, it's 0.55)
        if perturbed_cos_theta > 1.0:
            perturbed_cos_theta = 1.0
        perturbed_x = self.d / perturbed_cos_theta  # = self.d * self.v1 / perturbed_v2

        # Check monotonic trends: as v2 increases, t_min decreases and x decreases
        if not (perturbed_t_min < original_t_min and not math.isclose(perturbed_t_min, original_t_min, rel_tol=1e-2)):
            logs.append(f"Monotonicity violation: t_min should decrease when v2 increases. "
                        f"Original t_min = {original_t_min}, perturbed = {perturbed_t_min}")
            success = False
        else:
            logs.append(f"t_min monotonicity passed: {perturbed_t_min} < {original_t_min}")

        if not (perturbed_x < original_x and not math.isclose(perturbed_x, original_x, rel_tol=1e-2)):
            logs.append(f"Monotonicity violation: x should decrease when v2 increases. "
                        f"Original x = {original_x}, perturbed = {perturbed_x}")
            success = False
        else:
            logs.append(f"x monotonicity passed: {perturbed_x} < {original_x}")

        # Restore original v2 (not strictly necessary as we only used a local copy)
        # No changes were made to the instance.

        return success, logs

    def run_all_tests(self):
        """
        Run all tests and return overall success and logs.
        """
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
                all_logs.append(f"=== {test_name} ===")
                all_logs.append(f"Error during test: {e}")
                all_logs.append(traceback.format_exc())
                overall_success = False

        return overall_success, all_logs