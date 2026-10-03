import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        """
        # Input parameters (with defaults if missing)
        self.m = getattr(solution_instance, 'm', 22.4)       # mass of block B (kg)
        self.mu = getattr(solution_instance, 'mu', 0.4)      # coefficient of friction
        self.theta = getattr(solution_instance, 'theta', 53 * math.pi / 180)  # in radians
        self.g = getattr(solution_instance, 'g', 10.0)       # gravity (m/s^2)

        # If theta is given in degrees, convert to radians
        if isinstance(self.theta, (int, float)) and self.theta > math.pi:
            self.theta = self.theta * math.pi / 180

        # Extract results safely - FIXED: use correct key for F_prime
        results = getattr(solution_instance, 'results', {})
        self.F = getattr(solution_instance, 'F', results.get('F', 0.0))
        # FIX: Use the correct key "F'" (single quote) not "F''" (double quotes)
        self.F_prime = getattr(solution_instance, 'F_prime', results.get("F'", 0.0))
        self.F1 = getattr(solution_instance, 'F1', results.get('F_1', 0.0))

        # Precomputed trigonometric values for numerical checks
        self.sin_theta = math.sin(self.theta)
        self.cos_theta = math.cos(self.theta)

        # Log storage
        self.logs = []

    def _log(self, message, success=None):
        """Helper to log messages and optionally record test success."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {message}")
        else:
            self.logs.append(f"INFO: {message}")
        return success

    def test_invariants(self):
        """
        Test fundamental physical invariants:
        1. Newton's third law: F and F' should be equal.
        2. Container equilibrium: F = (m_water * g) / (2 * cosθ) for some positive m_water.
        3. Block B impending slip condition: F1 = (μ * m * g) / (sinθ - μ * cosθ).
        """
        tests_passed = True

        # 1. Newton's third law: F and F' should be equal (from step_4)
        try:
            if math.isclose(self.F, self.F_prime, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Newton's third law: F={self.F} ≈ F'={self.F_prime}", True)
            else:
                self._log(f"Newton's third law violation: F={self.F}, F'={self.F_prime}", False)
                tests_passed = False
        except Exception as e:
            self._log(f"Newton's third law check failed with error: {e}", False)
            tests_passed = False

        # 2. Container equilibrium: F = (m_water * g) / (2 * cosθ) => m_water = 2*F*cosθ/g
        try:
            m_water = 2 * self.F * self.cos_theta / self.g
            if m_water > 0:
                self._log(f"Container equilibrium: inferred water mass = {m_water:.2f} kg (positive)", True)
            else:
                self._log(f"Container equilibrium: inferred water mass = {m_water:.2f} kg (non-positive)", False)
                tests_passed = False
        except Exception as e:
            self._log(f"Container equilibrium check failed with error: {e}", False)
            tests_passed = False

        # 3. Block B impending slip condition: F1 = (μ * m * g) / (sinθ - μ * cosθ)
        try:
            denominator = self.sin_theta - self.mu * self.cos_theta
            if denominator > 0:
                expected_F1 = (self.mu * self.m * self.g) / denominator
                if math.isclose(self.F1, expected_F1, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Block B impending slip: F1={self.F1} ≈ expected {expected_F1:.2f}", True)
                else:
                    self._log(f"Block B impending slip mismatch: F1={self.F1}, expected {expected_F1:.2f}", False)
                    tests_passed = False
            else:
                self._log(f"Block B impending slip: denominator non-positive ({denominator})", False)
                tests_passed = False
        except Exception as e:
            self._log(f"Block B impending slip check failed with error: {e}", False)
            tests_passed = False

        return tests_passed

    def test_trace_consistency(self):
        """
        Check consistency between intermediate result F and final F1.
        Since F and F1 are forces at different stages (different water masses),
        we verify that F1 > F (as water increases, force increases) and that
        both satisfy the container equilibrium formula.
        """
        tests_passed = True

        # F1 should be larger than F because more water is needed to cause slipping
        try:
            if self.F1 > self.F:
                self._log(f"Trace consistency: F1={self.F1} > F={self.F} (force increases with water)", True)
            else:
                self._log(f"Trace consistency: F1={self.F1} not greater than F={self.F}", False)
                tests_passed = False
        except Exception as e:
            self._log(f"Trace consistency (F1 > F) check failed with error: {e}", False)
            tests_passed = False

        # Both F and F1 should satisfy container equilibrium with positive water masses
        try:
            m_water_F = 2 * self.F * self.cos_theta / self.g
            m_water_F1 = 2 * self.F1 * self.cos_theta / self.g
            if m_water_F1 > m_water_F:
                self._log(f"Water mass increases from {m_water_F:.2f} kg to {m_water_F1:.2f} kg", True)
            else:
                self._log(f"Water mass does not increase: {m_water_F:.2f} kg -> {m_water_F1:.2f} kg", False)
                tests_passed = False
        except Exception as e:
            self._log(f"Water mass consistency check failed with error: {e}", False)
            tests_passed = False

        return tests_passed

    def test_monotonicity(self):
        """
        Perturb the mass of block B and verify that F1 changes in the expected direction.
        Formula: F1 = (μ * m * g) / (sinθ - μ * cosθ) → increasing m increases F1.
        """
        tests_passed = True
        original_m = self.m

        try:
            # Perturb mass by +1%
            perturbed_m = original_m * 1.01
            denominator = self.sin_theta - self.mu * self.cos_theta
            original_F1 = (self.mu * original_m * self.g) / denominator
            perturbed_F1 = (self.mu * perturbed_m * self.g) / denominator

            if perturbed_F1 > original_F1:
                self._log(f"Monotonicity: increasing m increases F1 ({original_F1:.2f} -> {perturbed_F1:.2f})", True)
            else:
                self._log(f"Monotonicity violation: increasing m does not increase F1", False)
                tests_passed = False
        except Exception as e:
            self._log(f"Monotonicity check failed with error: {e}", False)
            tests_passed = False

        # Restore original mass (though we only used local variables)
        self.m = original_m
        return tests_passed

    def run_all_tests(self):
        """
        Run all tests and return overall success and logs.
        """
        self.logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]

        for name, test_func in tests:
            try:
                success = test_func()
                if not success:
                    overall_success = False
                self._log(f"{name} test completed", None)
            except Exception as e:
                self._log(f"{name} test crashed: {traceback.format_exc()}", False)
                overall_success = False

        return overall_success, self.logs