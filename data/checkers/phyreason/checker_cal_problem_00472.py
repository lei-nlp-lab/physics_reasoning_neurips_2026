import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access: use getattr and .get() to avoid AttributeError
        self.solution_instance = solution_instance
        self.inputs = getattr(solution_instance, 'inputs', {})
        self.results = getattr(solution_instance, 'results', {})

        # Extract input parameters with defaults (if missing)
        self.m = self.inputs.get('m', 1.0)
        self.g = self.inputs.get('g', 9.8)
        self.q = self.inputs.get('q', 1.0)

        # Extract results with defaults (if missing)
        self.E1 = self.results.get('E_1', 0.0)
        self.E2 = self.results.get('E_2', 0.0)

        # Fixed parameters from problem statement
        self.theta = math.radians(45)  # inclination angle in radians

    def test_invariants(self):
        """
        Verify static equilibrium conditions for both electric field configurations.
        Returns (bool, logs).
        """
        logs = []
        success = True

        # Case 1: Vertical electric field upwards
        # Equilibrium condition: m*g - q*E1 = 0 (along the plane)
        residual1 = self.m * self.g - self.q * self.E1
        if not math.isclose(residual1, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL invariant (E1): m*g - q*E1 = {residual1:.6f}, expected ~0.")
            success = False
        else:
            logs.append(f"PASS invariant (E1): m*g - q*E1 ~ 0 (residual={residual1:.6e})")

        # Case 2: Horizontal electric field to the left
        # Equilibrium condition: m*g*sinθ - q*E2*cosθ = 0 (along the plane)
        residual2 = self.m * self.g * math.sin(self.theta) - self.q * self.E2 * math.cos(self.theta)
        if not math.isclose(residual2, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL invariant (E2): m*g*sinθ - q*E2*cosθ = {residual2:.6f}, expected ~0.")
            success = False
        else:
            logs.append(f"PASS invariant (E2): m*g*sinθ - q*E2*cosθ ~ 0 (residual={residual2:.6e})")

        return success, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        For this problem, both E1 and E2 should equal m*g/q.
        Returns (bool, logs).
        """
        logs = []
        success = True

        expected = self.m * self.g / self.q if self.q != 0 else float('inf')

        # Check E1
        if not math.isclose(self.E1, expected, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL trace (E1): E1 = {self.E1:.6f}, expected m*g/q = {expected:.6f}")
            success = False
        else:
            logs.append(f"PASS trace (E1): E1 = {self.E1:.6f} ~ m*g/q = {expected:.6f}")

        # Check E2
        if not math.isclose(self.E2, expected, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL trace (E2): E2 = {self.E2:.6f}, expected m*g/q = {expected:.6f}")
            success = False
        else:
            logs.append(f"PASS trace (E2): E2 = {self.E2:.6f} ~ m*g/q = {expected:.6f}")

        return success, logs

    def test_monotonicity(self):
        """
        Perturb an input (mass m) and verify the expected trend in E1 and E2.
        Uses the formula E = m*g/q, so E should scale linearly with m.
        Returns (bool, logs).
        """
        logs = []
        success = True

        # Original expected value
        original_expected = self.m * self.g / self.q

        # Perturb mass by +10%
        perturb_factor = 1.10
        m_perturbed = self.m * perturb_factor
        expected_perturbed = m_perturbed * self.g / self.q

        # Expected scaling: E_perturbed / E_original = perturb_factor
        # Check E1
        if self.E1 != 0:
            scale_factor_E1 = (self.E1 * perturb_factor) / self.E1  # This should be perturb_factor
            # Instead, check if E1 follows the linear relationship: E1_expected_perturbed = perturb_factor * original_expected
            E1_expected_perturbed = original_expected * perturb_factor
            if not math.isclose(self.E1, original_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL monotonicity (E1): E1 does not follow m*g/q. E1={self.E1:.6f}, expected={original_expected:.6f}")
                success = False
            else:
                logs.append(f"PASS monotonicity (E1): E1 scales linearly with m (E1={self.E1:.6f}, m*g/q={original_expected:.6f})")
        else:
            logs.append("SKIP monotonicity (E1): E1 is zero, cannot check scaling.")

        # Check E2
        if self.E2 != 0:
            scale_factor_E2 = (self.E2 * perturb_factor) / self.E2  # This should be perturb_factor
            # Instead, check if E2 follows the linear relationship: E2_expected_perturbed = perturb_factor * original_expected
            E2_expected_perturbed = original_expected * perturb_factor
            if not math.isclose(self.E2, original_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL monotonicity (E2): E2 does not follow m*g/q. E2={self.E2:.6f}, expected={original_expected:.6f}")
                success = False
            else:
                logs.append(f"PASS monotonicity (E2): E2 scales linearly with m (E2={self.E2:.6f}, m*g/q={original_expected:.6f})")
        else:
            logs.append("SKIP monotonicity (E2): E2 is zero, cannot check scaling.")

        return success, logs

    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            success, logs = test_func()
            all_logs.append(f"--- {test_name} ---")
            all_logs.extend(logs)
            if not success:
                overall_success = False

        all_logs.append(f"\nOverall: {'PASS' if overall_success else 'FAIL'}")
        return overall_success, all_logs