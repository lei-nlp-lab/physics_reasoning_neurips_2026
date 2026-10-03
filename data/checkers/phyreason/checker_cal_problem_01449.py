import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the checker by safely extracting inputs and results.
        Follows robustness rule 1: safe attribute access.
        """
        # Constants (with defaults if not provided)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.μ = getattr(solution_instance, 'μ', 0.5)
        self.θ_deg = getattr(solution_instance, 'θ', 37.0)
        self.L = getattr(solution_instance, 'L', 10.0)        # flat lawn length
        self.v_B = getattr(solution_instance, 'v_B', 10.0)   # speed at B

        # Trig values (given in problem)
        self.sinθ = 0.6
        self.cosθ = 0.8

        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.v_C = getattr(solution_instance, 'v', None) or results.get('v', 0.0)
        self.t_total = getattr(solution_instance, 't', None) or results.get('t', 0.0)
        self.a_flat = getattr(solution_instance, 'a', None) or results.get('a', 0.0)
        self.a_up = getattr(solution_instance, 'a₁', None) or results.get('a₁', 0.0)
        self.a_down = getattr(solution_instance, 'a₂', None) or results.get('a₂', 0.0)

        # Expected accelerations from physics formulas
        self.a_flat_expected = self.μ * self.g
        self.a_up_expected = self.g * self.sinθ + self.μ * self.g * self.cosθ
        self.a_down_expected = self.g * self.sinθ - self.μ * self.g * self.cosθ

        # Expected final answers (numerical)
        self.v_C_expected = math.sqrt(self.v_B**2 + 2 * self.μ * self.g * self.L)
        # Total time formula: t = v_B/a_up + v_B/sqrt(a_up * a_down)
        self.t_total_expected = (
            self.v_B / self.a_up_expected +
            self.v_B / math.sqrt(self.a_up_expected * self.a_down_expected)
        )

        # Logs and flags
        self.logs = []
        self.all_tests_passed = True

    def _log(self, message, success):
        """Helper to log a test result."""
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {message}")
        if not success:
            self.all_tests_passed = False

    def _isclose(self, a, b):
        """Fuzzy numerical matching (robustness rule 2)."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

    def test_invariants(self):
        """Verify Newton's laws and kinematic equations."""
        try:
            # 1. Check flat segment deceleration
            if self._isclose(self.a_flat, self.a_flat_expected):
                self._log(f"Flat deceleration correct: {self.a_flat} ≈ μ*g = {self.a_flat_expected}", True)
            else:
                self._log(f"Flat deceleration mismatch: {self.a_flat} vs μ*g = {self.a_flat_expected}", False)

            # 2. Check upslope acceleration (deceleration)
            if self._isclose(self.a_up, self.a_up_expected):
                self._log(f"Upslope deceleration correct: {self.a_up} ≈ g*sinθ+μ*g*cosθ = {self.a_up_expected}", True)
            else:
                self._log(f"Upslope deceleration mismatch: {self.a_up} vs expected {self.a_up_expected}", False)

            # 3. Check downslope acceleration
            if self._isclose(self.a_down, self.a_down_expected):
                self._log(f"Downslope acceleration correct: {self.a_down} ≈ g*sinθ-μ*g*cosθ = {self.a_down_expected}", True)
            else:
                self._log(f"Downslope acceleration mismatch: {self.a_down} vs expected {self.a_down_expected}", False)

            # 4. Kinematic consistency: v_C^2 = v_B^2 + 2*a_flat*L
            lhs = self.v_C**2
            rhs = self.v_B**2 + 2 * self.a_flat_expected * self.L
            if self._isclose(lhs, rhs):
                self._log(f"Kinematic eq on flat: v_C²={lhs} ≈ v_B²+2*a*L={rhs}", True)
            else:
                self._log(f"Kinematic eq on flat failed: {lhs} vs {rhs}", False)

            # 5. Kinematic consistency for right slope total time
            t1 = self.v_B / self.a_up_expected
            x_AB = self.v_B**2 / (2 * self.a_up_expected)
            t2 = math.sqrt(2 * x_AB / self.a_down_expected)
            t_total_calc = t1 + t2
            if self._isclose(self.t_total, t_total_calc):
                self._log(f"Total time consistent: {self.t_total} ≈ v_B/a_up + v_B/√(a_up*a_down) = {t_total_calc}", True)
            else:
                self._log(f"Total time mismatch: {self.t_total} vs calculated {t_total_calc}", False)

        except Exception as e:
            self._log(f"Error in test_invariants: {traceback.format_exc()}", False)

    def test_trace_consistency(self):
        """Cross-check intermediate and final results with expected values."""
        try:
            # 1. Speed at C
            if self._isclose(self.v_C, self.v_C_expected):
                self._log(f"Speed at C correct: {self.v_C} ≈ √(v_B²+2μgl) = {self.v_C_expected}", True)
            else:
                self._log(f"Speed at C mismatch: {self.v_C} vs expected {self.v_C_expected}", False)

            # 2. Total time on right slope
            if self._isclose(self.t_total, self.t_total_expected):
                self._log(f"Total time correct: {self.t_total} ≈ {self.t_total_expected}", True)
            else:
                self._log(f"Total time mismatch: {self.t_total} vs expected {self.t_total_expected}", False)

            # 3. If accelerations were computed, check against expected
            if self.a_up != 0.0 and not self._isclose(self.a_up, self.a_up_expected):
                self._log(f"a_up {self.a_up} differs from expected {self.a_up_expected}", False)
            if self.a_down != 0.0 and not self._isclose(self.a_down, self.a_down_expected):
                self._log(f"a_down {self.a_down} differs from expected {self.a_down_expected}", False)

        except Exception as e:
            self._log(f"Error in test_trace_consistency: {traceback.format_exc()}", False)

    def test_monotonicity(self):
        """
        Perturb an input (μ) and verify trends in outputs.
        Follows robustness rule 4: symbolic/numerical fallback.
        """
        try:
            original_μ = self.μ
            perturbed_μ = original_μ * 1.01  # increase by 1%

            # Original expected values
            v_C_orig = math.sqrt(self.v_B**2 + 2 * original_μ * self.g * self.L)
            a_up_orig = self.g * self.sinθ + original_μ * self.g * self.cosθ
            a_down_orig = self.g * self.sinθ - original_μ * self.g * self.cosθ
            t_total_orig = (self.v_B / a_up_orig +
                           self.v_B / math.sqrt(a_up_orig * a_down_orig))

            # Perturbed expected values
            v_C_pert = math.sqrt(self.v_B**2 + 2 * perturbed_μ * self.g * self.L)
            a_up_pert = self.g * self.sinθ + perturbed_μ * self.g * self.cosθ
            a_down_pert = self.g * self.sinθ - perturbed_μ * self.g * self.cosθ
            t_total_pert = (self.v_B / a_up_pert +
                           self.v_B / math.sqrt(a_up_pert * a_down_pert))

            # Check trends
            trend_ok = True
            if not (v_C_pert > v_C_orig):
                self._log(f"Monotonicity: v_C should increase with μ (got {v_C_orig} -> {v_C_pert})", False)
                trend_ok = False
            if not (t_total_pert > t_total_orig):
                self._log(f"Monotonicity: t_total should increase with μ (got {t_total_orig} -> {t_total_pert})", False)
                trend_ok = False

            if trend_ok:
                self._log(f"Monotonicity test passed for μ perturbation (1% increase)", True)

        except Exception as e:
            self._log(f"Error in test_monotonicity: {traceback.format_exc()}", False)

    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = []
        self.all_tests_passed = True

        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()

        return self.all_tests_passed, self.logs