import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from the solution instance."""
        self.solution = solution_instance
        self.extract_inputs()
        self.extract_results()
        self.logs = []

    def extract_inputs(self):
        """Extract input parameters with safe fallbacks."""
        self.m1 = getattr(self.solution, 'm1', getattr(self.solution, 'results', {}).get('m1', 0.4))
        self.m2 = getattr(self.solution, 'm2', getattr(self.solution, 'results', {}).get('m2', 0.1))
        self.mu = getattr(self.solution, 'mu', getattr(self.solution, 'results', {}).get('mu', 0.2))
        self.d = getattr(self.solution, 'd', getattr(self.solution, 'results', {}).get('d', 0.4))
        self.g = getattr(self.solution, 'g', getattr(self.solution, 'results', {}).get('g', 10.0))

    def extract_results(self):
        """Extract result quantities with safe fallbacks."""
        self.F_f1 = getattr(self.solution, 'F_f1', getattr(self.solution, 'results', {}).get('F_f1', None))
        self.t = getattr(self.solution, 't', getattr(self.solution, 'results', {}).get('t', None))
        self.a = getattr(self.solution, 'a', getattr(self.solution, 'results', {}).get('a', None))
        self.F_f2 = getattr(self.solution, 'F_f2', getattr(self.solution, 'results', {}).get('F_f2', None))
        self.a_2 = getattr(self.solution, 'a_2', getattr(self.solution, 'results', {}).get('a_2', None))
        self.a_1 = getattr(self.solution, 'a_1', getattr(self.solution, 'results', {}).get('a_1', None))

    def isclose(self, a, b):
        """Robust fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

    def test_invariants(self):
        """Verify fundamental physics laws."""
        logs = []
        try:
            # 1. Check kinetic friction forces
            F_f1_expected_kinetic = self.mu * self.m1 * self.g
            F_f2_expected = self.mu * (self.m1 + self.m2) * self.g

            if self.F_f2 is not None and not self.isclose(self.F_f2, F_f2_expected):
                logs.append(f"F_f2 mismatch: {self.F_f2} vs expected {F_f2_expected}")

            # 2. Determine regime and check consistency
            if self.a_1 is not None:
                if self.isclose(self.a_1, self.mu * self.g):  # Sliding regime
                    # In sliding regime, F_f1 should be static friction from Q1, not kinetic
                    # So we don't check F_f1 against kinetic value here
                    pass
                else:  # No-sliding regime
                    if self.a is not None and self.isclose(self.a_1, self.a):
                        if self.F_f1 is not None and not self.isclose(self.F_f1, self.m1 * self.a):
                            logs.append(f"No-sliding: F_f1 = {self.F_f1}, expected m1*a = {self.m1 * self.a}")
                    else:
                        logs.append(f"Regime ambiguous: a_1={self.a_1}, a={self.a}")

        except Exception as e:
            logs.append(f"test_invariants error: {e}\n{traceback.format_exc()}")
        return logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # Check if time t is consistent with relative motion
            if self.t is not None and self.a_2 is not None and self.a_1 is not None:
                if self.a_2 > self.a_1:  # Relative motion expected
                    rel_displacement = 0.5 * (self.a_2 - self.a_1) * self.t**2
                    if not self.isclose(rel_displacement, self.d):
                        logs.append(f"Relative motion inconsistency: displacement {rel_displacement} != d={self.d}")
                else:
                    logs.append("Warning: a_2 <= a_1 implies no relative motion, but t is provided.")

            # Check if static friction force is within limits
            if self.F_f1 is not None:
                max_static = self.mu * self.m1 * self.g
                if self.F_f1 > max_static * 1.01:  # Allow 1% tolerance
                    logs.append(f"F_f1 exceeds max static friction: {self.F_f1} > {max_static}")

        except Exception as e:
            logs.append(f"test_trace_consistency error: {e}\n{traceback.format_exc()}")
        return logs

    def test_monotonicity(self):
        """Perturb an input and verify expected trends."""
        logs = []
        original_mu = self.mu
        try:
            # Increase mu by 1%
            self.mu = original_mu * 1.01

            # Recalculate expected friction forces
            F_f1_new = self.mu * self.m1 * self.g
            F_f2_new = self.mu * (self.m1 + self.m2) * self.g

            # Check monotonic increase - only check F_f2 since F_f1 is static from Q1
            if self.F_f2 is not None and F_f2_new < self.F_f2:
                logs.append(f"Monotonicity violation: F_f2 does not increase with mu")

            # Restore original mu
            self.mu = original_mu

        except Exception as e:
            logs.append(f"test_monotonicity error: {e}\n{traceback.format_exc()}")
            self.mu = original_mu
        return logs

    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        self.logs = []
        self.logs.extend(self.test_invariants())
        self.logs.extend(self.test_trace_consistency())
        self.logs.extend(self.test_monotonicity())
        success = len(self.logs) == 0
        return success, self.logs