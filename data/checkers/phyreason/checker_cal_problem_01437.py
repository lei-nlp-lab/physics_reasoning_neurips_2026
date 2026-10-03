import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Never assume direct attribute access; use getattr and .get().
        """
        # Inputs (with defaults)
        self.m = getattr(solution_instance, 'm', 1.0)  # mass in kg
        self.g = getattr(solution_instance, 'g', 10.0) # gravity in m/s^2
        self.t1 = getattr(solution_instance, 't1', 0.5) # time when force is removed

        # Results dictionary (if available)
        results = getattr(solution_instance, 'results', {})

        # Extract results with safe fallbacks
        self.a1 = results.get('a_1', results.get('a1', 0.0))
        self.a2 = results.get('a_2', results.get('a2', 0.0))
        self.F = results.get('F', 0.0)
        self.s1 = results.get('s_1', results.get('s1', 0.0))
        self.s2 = results.get('s_2', results.get('s2', 0.0))
        self.s = results.get('s', 0.0)
        self.L = results.get('L', 0.0)

        self.logs = []

    def _log(self, msg, success):
        """Helper to log test results."""
        self.logs.append((msg, success))

    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy numerical matching."""
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """
        Verify physical invariants using purely numeric checks.
        """
        success = True

        # 1. Newton's Second Law: F = m*(a1 + a2) for no-friction case.
        expected_F = self.m * (self.a1 + self.a2)
        if not self._isclose(self.F, expected_F):
            self._log(f"Newton's Second Law fails: F = {self.F}, expected m*(a1+a2) = {expected_F}", False)
            success = False
        else:
            self._log(f"Newton's Second Law holds: F = {self.F} ≈ {expected_F}", True)

        # 2. Additional check: a2 should be positive (magnitude) and less than g (since it's g*sinθ)
        if self.a2 <= 0 or self.a2 >= self.g:
            self._log(f"Plausibility check: a2 = {self.a2} should be between 0 and g={self.g}", False)
            success = False
        else:
            self._log(f"Plausibility: a2 = {self.a2} is between 0 and g={self.g}", True)

        return success

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with the final answer.
        From the steps analysis:
        - s = s1 - s2
        - L = s1 + s2
        """
        success = True

        # Check displacement s = s1 - s2
        expected_s = self.s1 - self.s2
        if not self._isclose(self.s, expected_s):
            self._log(f"Displacement inconsistency: s = {self.s}, expected s1 - s2 = {expected_s}", False)
            success = False
        else:
            self._log(f"Displacement consistent: s = {self.s} ≈ s1 - s2 = {expected_s}", True)

        # Check total distance L = s1 + s2
        expected_L = self.s1 + self.s2
        if not self._isclose(self.L, expected_L):
            self._log(f"Distance inconsistency: L = {self.L}, expected s1 + s2 = {expected_L}", False)
            success = False
        else:
            self._log(f"Distance consistent: L = {self.L} ≈ s1 + s2 = {expected_L}", True)

        return success

    def test_monotonicity(self):
        """
        Perturb mass and check that force scales linearly if accelerations are constant.
        """
        success = True
        try:
            # Perturb mass by +10%
            m_perturbed = self.m * 1.10
            # Assuming same accelerations (from v-t graph, independent of mass)
            F_perturbed = m_perturbed * (self.a1 + self.a2)
            if not (F_perturbed > self.F):
                self._log(f"Monotonicity check failed: increasing mass did not increase F as expected. F_original={self.F}, F_perturbed={F_perturbed}", False)
                success = False
            else:
                self._log(f"Monotonicity holds: F increases with mass (F_original={self.F}, F_perturbed={F_perturbed})", True)
        except Exception as e:
            self._log(f"Monotonicity test error: {traceback.format_exc()}", False)
            success = False
        return success

    def run_all_tests(self):
        """
        Run all tests and return (success_bool, logs_list).
        """
        self.logs = []
        overall_success = True

        tests = [
            ("Trace Consistency", self.test_trace_consistency),
            ("Invariants", self.test_invariants),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"{test_name} crashed: {e}", False)
                overall_success = False

        return overall_success, self.logs