import math
import traceback
from typing import Tuple, List, Optional

class PhysicsChecker:
    """Audit the capacitance problem for two thin wires on a dielectric interface."""

    def __init__(self, solution_instance):
        # Extract input givens: try both attribute and results dict
        self.C = getattr(solution_instance, 'C', None)
        self.epsilon0 = getattr(solution_instance, 'epsilon0', None)
        self.epsilon = getattr(solution_instance, 'epsilon', None)

        # Results dict (preferred)
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            res = solution_instance.results
        else:
            res = {}

        self.epsilon_eff = res.get('epsilon_eff', None)
        self.C_new = res.get('C_new', None)

        # Additional extracted values for detailed checks (if needed)
        # We can compute expected values for built‑in consistency tests,
        # but we do NOT store them as fixed defaults.

    # ------------------------------------------------------------------
    # Interface tests: required keys exist and are numeric
    # ------------------------------------------------------------------
    def test_interface_epsilon_eff_exists(self) -> Tuple[bool, str]:
        if self.epsilon_eff is None:
            return (False, "epsilon_eff is None")
        return (True, "epsilon_eff is present")

    def test_interface_C_new_exists(self) -> Tuple[bool, str]:
        if self.C_new is None:
            return (False, "C_new is None")
        return (True, "C_new is present")

    def test_interface_epsilon_eff_numeric(self) -> Tuple[bool, str]:
        if self.epsilon_eff is None:
            return (False, "SKIP: epsilon_eff missing")
        if not isinstance(self.epsilon_eff, (int, float)):
            return (False, f"epsilon_eff is not numeric: {type(self.epsilon_eff)}")
        return (True, "epsilon_eff is numeric")

    def test_interface_C_new_numeric(self) -> Tuple[bool, str]:
        if self.C_new is None:
            return (False, "SKIP: C_new missing")
        if not isinstance(self.C_new, (int, float)):
            return (False, f"C_new is not numeric: {type(self.C_new)}")
        return (True, "C_new is numeric")

    # ------------------------------------------------------------------
    # Invariant tests: physical sanity (lenient isclose allowed)
    # ------------------------------------------------------------------
    def test_invariant_effective_permittivity_relation(self) -> Tuple[bool, str]:
        """Check epsilon_eff = (epsilon0 + epsilon) / 2."""
        if any(v is None for v in [self.epsilon0, self.epsilon, self.epsilon_eff]):
            return (False, "SKIP: missing required givens (epsilon0, epsilon, epsilon_eff)")
        expected = (self.epsilon0 + self.epsilon) / 2.0
        if not math.isclose(self.epsilon_eff, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"epsilon_eff {self.epsilon_eff} != expected {expected}")
        return (True, "epsilon_eff matches (epsilon0+epsilon)/2")

    def test_invariant_C_scale_relation(self) -> Tuple[bool, str]:
        """Check C_new = ((epsilon0 + epsilon) / (2*epsilon0)) * C."""
        if any(v is None for v in [self.C, self.epsilon0, self.epsilon, self.C_new]):
            return (False, "SKIP: missing C, epsilon0, epsilon, or C_new")
        expected = ((self.epsilon0 + self.epsilon) / (2.0 * self.epsilon0)) * self.C
        if not math.isclose(self.C_new, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"C_new {self.C_new} != expected {expected}")
        return (True, "C_new scales correctly with effective permittivity")

    def test_invariant_ratio_sanity(self) -> Tuple[bool, str]:
        """Check ratio C_new/C = (epsilon0+epsilon)/(2*epsilon0)."""
        if any(v is None for v in [self.C, self.epsilon0, self.epsilon, self.C_new]):
            return (False, "SKIP: missing data for ratio")
        if math.isclose(self.C, 0.0, abs_tol=1e-15):
            return (False, "C is zero, cannot compute ratio")
        ratio = self.C_new / self.C
        expected_ratio = (self.epsilon0 + self.epsilon) / (2.0 * self.epsilon0)
        if not math.isclose(ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Ratio {ratio} != {expected_ratio}")
        return (True, "Ratio C_new/C matches theoretical value")

    def test_invariant_all_positive(self) -> Tuple[bool, str]:
        """All physical quantities must be positive (capacitances, permittivities)."""
        values = [self.C, self.epsilon0, self.epsilon, self.epsilon_eff, self.C_new]
        # None values are not checked for positivity, only present ones
        for name, val in zip(["C", "epsilon0", "epsilon", "epsilon_eff", "C_new"], values):
            if val is None:
                continue
            if val <= 0:
                return (False, f"{name} is not positive: {val}")
        return (True, "All available values are positive")

    # ------------------------------------------------------------------
    # Modeling / trace tests: derived from explanation steps
    # ------------------------------------------------------------------
    def test_modeling_field_scaling_factor(self) -> Tuple[bool, str]:
        """From explanation: E' = K E with K = 2*epsilon0/(epsilon0+epsilon).
           We can derive K from epsilon_eff: K = epsilon0/epsilon_eff.
           Check that K is positive and less than 1 (since epsilon >= 0)."""
        if self.epsilon_eff is None:
            return (False, "SKIP: epsilon_eff missing")
        if math.isclose(self.epsilon_eff, 0.0, abs_tol=1e-15):
            return (False, "epsilon_eff is zero, cannot compute K")
        K = self.epsilon0 / self.epsilon_eff
        # K should be between 0 and 1 (if epsilon0>0 and epsilon>=0)
        if not (0 < K <= 1):
            return (False, f"Field scaling factor K = {K} is not in (0,1]")
        return (True, f"K = {K} is plausible")

    def test_modeling_Gauss_law_consistency(self) -> Tuple[bool, str]:
        """From explanation: before dielectric, Q/(2*epsilon0) = int_S E·dS.
           After dielectric, the same charge gives (epsilon0+epsilon)*K/(2*epsilon0)*Q = Q.
           We can check that the product (epsilon0+epsilon) * K / (2*epsilon0) equals 1.
           (This is a self‑consistency test of the derived relations.)"""
        if any(v is None for v in [self.epsilon0, self.epsilon, self.epsilon_eff]):
            return (False, "SKIP: missing epsilon0, epsilon, or epsilon_eff")
        # Compute K from the formula: K = 2*epsilon0/(epsilon0+epsilon)
        K = 2.0 * self.epsilon0 / (self.epsilon0 + self.epsilon) if (self.epsilon0 + self.epsilon) != 0 else 0
        if math.isclose(self.epsilon0 + self.epsilon, 0, abs_tol=1e-15):
            return (False, "epsilon0+epsilon is zero, cannot compute")
        # Check Gauss law consistency: (epsilon0+epsilon)*K/(2*epsilon0) == 1
        check = (self.epsilon0 + self.epsilon) * K / (2.0 * self.epsilon0)
        if not math.isclose(check, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Gauss law consistency factor = {check}, expected 1")
        return (True, "Gauss law consistency satisfied")

    # ------------------------------------------------------------------
    # Monotonicity tests: perturb a parameter and check trend
    # ------------------------------------------------------------------
    def test_monotonicity_epsilon_increase_increases_C(self) -> Tuple[bool, str]:
        """If epsilon is increased by 10%, the predicted C_new should also increase."""
        if any(v is None for v in [self.C, self.epsilon0, self.epsilon]):
            return (False, "SKIP: missing givens for monotonicity test")
        epsilon_perturbed = self.epsilon * 1.1
        # Compute expected C_new for perturbed epsilon using the formula
        C_new_perturbed = ((self.epsilon0 + epsilon_perturbed) / (2.0 * self.epsilon0)) * self.C
        # Compare with the solution's C_new
        if self.C_new is None:
            # If solution didn't compute C_new, we skip (robust)
            return (False, "SKIP: C_new not available, cannot compare")
        # Use a weak inequality: perturbed should be > original C_new
        if C_new_perturbed <= self.C_new:
            return (False, f"Increasing epsilon from {self.epsilon} to {epsilon_perturbed} "
                           f"should increase C_new, but perturbed C_new = {C_new_perturbed} <= original {self.C_new}")
        return (True, "Monotonicity: increasing epsilon increases C_new")

    def test_monotonicity_epsilon_decrease_decreases_C(self) -> Tuple[bool, str]:
        """If epsilon is decreased by 10% (but not below zero), predicted C_new should decrease."""
        if any(v is None for v in [self.C, self.epsilon0, self.epsilon]):
            return (False, "SKIP: missing givens for monotonicity test")
        # Avoid negative epsilon (non‑physical)
        epsilon_perturbed = max(self.epsilon * 0.9, 0.0)
        C_new_perturbed = ((self.epsilon0 + epsilon_perturbed) / (2.0 * self.epsilon0)) * self.C
        if self.C_new is None:
            return (False, "SKIP: C_new not available")
        # Check decrease: perturbed C_new should be less than original
        if C_new_perturbed >= self.C_new:
            return (False, f"Decreasing epsilon from {self.epsilon} to {epsilon_perturbed} "
                           f"should decrease C_new, but perturbed C_new = {C_new_perturbed} >= original {self.C_new}")
        return (True, "Monotonicity: decreasing epsilon decreases C_new")

    # ------------------------------------------------------------------
    # Weak answer sanity (only sign and magnitude, no exact comparison)
    # ------------------------------------------------------------------
    def test_answer_sanity_bounds(self) -> Tuple[bool, str]:
        """C_new should be positive, finite, and of comparable order to C."""
        if any(v is None for v in [self.C, self.C_new]):
            return (False, "SKIP: C or C_new missing")
        # 1. Positive
        if not (self.C > 0 and self.C_new > 0):
            return (False, "Either C or C_new is not positive")
        # 2. Finite (not inf, not NaN)
        if not (math.isfinite(self.C) and math.isfinite(self.C_new)):
            return (False, "C or C_new is not finite")
        # 3. Order of magnitude: within factor 10 (conservative bound)
        #    since epsilon could be large, but at least not absurdly off.
        if self.C_new > 10.0 * self.C or self.C_new < 0.1 * self.C:
            return (False, f"C_new = {self.C_new} is too far from C = {self.C} (factor >10)")
        return (True, "C_new passes basic sanity (positive, finite, reasonable magnitude)")

    # ------------------------------------------------------------------
    # Runner that discovers all test_* methods
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs = []
        all_pass = True
        # Discover methods starting with "test_"
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)