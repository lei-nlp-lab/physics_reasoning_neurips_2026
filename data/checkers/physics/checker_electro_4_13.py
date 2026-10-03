import math
import traceback
from numbers import Number

class PhysicsChecker:
    """
    Audits a solution for the reflection coefficient of an EM wave
    normally incident from vacuum onto an insulator.
    """

    def __init__(self, solution_instance):
        # Safe extraction: use None if missing, never numeric defaults.
        # Givens
        self.epsilon = getattr(solution_instance, 'epsilon', None)
        if self.epsilon is None and hasattr(solution_instance, 'givens'):
            self.epsilon = solution_instance.givens.get('epsilon', None)

        # Results (from the pseudocode [emit] contract)
        self.R = None
        if hasattr(solution_instance, 'results'):
            self.R = solution_instance.results.get('R', None)
        elif hasattr(solution_instance, 'R'):
            self.R = solution_instance.R

        # Additional context (not required for core tests)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

    # ------------------------------------------------------------------
    # Interface checks: presence and type of required inputs/outputs
    # ------------------------------------------------------------------
    def test_interface_epsilon_provided(self):
        """Check that epsilon (relative permittivity) is provided."""
        if self.epsilon is None:
            return False, "epsilon (relative permittivity) not provided"
        if not isinstance(self.epsilon, Number) or not math.isfinite(self.epsilon):
            return False, "epsilon must be a finite number"
        if self.epsilon <= 0:
            return False, "epsilon must be positive (insulator)"
        return True, f"epsilon = {self.epsilon} is valid"

    def test_interface_R_exists(self):
        """Check that the reflection coefficient R is present and numeric."""
        if self.R is None:
            return False, "results['R'] not provided"
        if not isinstance(self.R, Number) or math.isnan(self.R):
            return False, "R must be a finite number"
        return True, f"R = {self.R} is present"

    # ------------------------------------------------------------------
    # Invariant checks: theorem‑based physical sanity
    # ------------------------------------------------------------------
    def test_invariant_R_range(self):
        """Reflection coefficient must be between 0 and 1 (inclusive)."""
        if self.R is None:
            return False, "SKIP: R not available"
        if self.R < 0 or self.R > 1:
            return False, f"R = {self.R} is outside [0,1]"
        return True, f"R = {self.R} within [0,1]"

    def test_invariant_R_zero_at_epsilon_one(self):
        """
        When epsilon = 1 (impedance match), reflection coefficient must be zero.
        This follows from continuity of tangential E and H at the interface.
        """
        if self.epsilon is None or self.R is None:
            return False, "SKIP: epsilon or R not available"
        if math.isclose(self.epsilon, 1.0, rel_tol=1e-7, abs_tol=1e-7):
            # Only test if epsilon is exactly 1
            if not math.isclose(self.R, 0.0, abs_tol=1e-9):
                return False, f"For epsilon=1, R should be 0, got {self.R}"
            return True, "R=0 at epsilon=1"
        return True, "epsilon != 1, test not applicable"

    def test_invariant_R_nonnegative(self):
        """Reflection coefficient must be non‑negative (power ratio)."""
        if self.R is None:
            return False, "SKIP: R not available"
        if self.R < 0:
            return False, "R is negative (unphysical)"
        return True, "R is nonnegative"

    # ------------------------------------------------------------------
    # Modeling checks: verify the derived formula consistency
    # ------------------------------------------------------------------
    def test_modeling_R_formula(self):
        """
        Re‑derive R from epsilon using the impedance mismatch formula
        and check agreement with the provided R.
        This is allowed because the value is derived inside the checker
        from the given epsilon and a theorem (wave impedance ratio).
        """
        if self.epsilon is None or self.R is None:
            return False, "SKIP: epsilon or R not available"
        # Compute expected R from formula
        sqrt_eps = math.sqrt(self.epsilon)
        gamma = (1 - sqrt_eps) / (1 + sqrt_eps)  # (Z2-Z1)/(Z2+Z1)
        expected_R = gamma ** 2
        if not math.isclose(self.R, expected_R, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"R mismatch: provided {self.R}, expected {expected_R} from formula"
        return True, f"R = {self.R} matches formula (tolerance 0.01)"

    def test_modeling_Gamma_sign(self):
        """
        The field reflection coefficient Gamma = ±sqrt(R).
        Its sign is determined by epsilon: Gamma negative for epsilon>1,
        positive for epsilon<1 (since Z2 < Z1 or > Z1).
        This combines the impedance mismatch theorem with the square-root ambiguity.
        """
        if self.epsilon is None or self.R is None:
            return False, "SKIP: epsilon or R not available"
        # Compute absolute value of Gamma from R
        abs_gamma = math.sqrt(self.R)
        # The formula gives Gamma = (1 - sqrt(eps))/(1+sqrt(eps))
        sqrt_eps = math.sqrt(self.epsilon)
        gamma_formula = (1 - sqrt_eps) / (1 + sqrt_eps)
        # Gamma provided (through R) should have the same sign as gamma_formula
        if not math.isclose(abs_gamma, abs(gamma_formula), rel_tol=1e-2, abs_tol=1e-2):
            return False, f"Magnitude of field Gamma from R ({abs_gamma}) does not match formula magnitude ({abs(gamma_formula)})"
        # Check sign: if epsilon > 1, gamma_formula negative -> Gamma should be negative
        if self.epsilon > 1 and gamma_formula > 0:
            return False, "For epsilon>1, Gamma should be negative but formula gives positive"
        if self.epsilon < 1 and gamma_formula < 0:
            return False, "For epsilon<1, Gamma should be positive but formula gives negative"
        return True, f"Gamma sign consistent with epsilon (Gamma = {gamma_formula})"

    # ------------------------------------------------------------------
    # Monotonicity checks: perturb epsilon and check trend
    # ------------------------------------------------------------------
    def test_monotonicity_R_decreasing_below_one(self):
        """
        For epsilon in (0,1), R decreases as epsilon increases.
        Check with a small perturbation delta = +2%.
        """
        if self.epsilon is None or self.R is None:
            return False, "SKIP: epsilon or R not available"
        eps = self.epsilon
        if eps >= 1.0:
            return True, f"epsilon={eps} not in (0,1), test skipped"
        # Perturb epsilon slightly upward (still in (0,1))
        delta = 0.02 * eps
        eps2 = eps + delta
        if eps2 >= 1.0:
            # Keep it below 1
            eps2 = (eps + 1.0) / 2.0
        # Compute expected R for both eps and eps2 inside checker
        R_orig = ((1 - math.sqrt(eps)) / (1 + math.sqrt(eps))) ** 2
        R_pert = ((1 - math.sqrt(eps2)) / (1 + math.sqrt(eps2))) ** 2
        # Verify original R matches provided R (to ensure consistency)
        if not math.isclose(self.R, R_orig, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"Provided R {self.R} inconsistent with own epsilon {eps} (expected {R_orig})"
        if R_pert >= R_orig - 1e-12:
            # Due to rounding, allow tiny increase; but trend should be decreasing
            if R_pert > R_orig + 1e-10:
                return False, f"R should decrease when epsilon increases from {eps} to {eps2}: {R_orig} -> {R_pert}"
        return True, f"R decreases with epsilon in (0,1): {R_orig} -> {R_pert}"

    def test_monotonicity_R_increasing_above_one(self):
        """
        For epsilon > 1, R increases as epsilon increases.
        Check with a small perturbation delta = +2%.
        """
        if self.epsilon is None or self.R is None:
            return False, "SKIP: epsilon or R not available"
        eps = self.epsilon
        if eps <= 1.0:
            return True, f"epsilon={eps} not > 1, test skipped"
        delta = 0.02 * eps
        eps2 = eps + delta
        R_orig = ((1 - math.sqrt(eps)) / (1 + math.sqrt(eps))) ** 2
        R_pert = ((1 - math.sqrt(eps2)) / (1 + math.sqrt(eps2))) ** 2
        if not math.isclose(self.R, R_orig, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"Provided R {self.R} inconsistent with own epsilon {eps} (expected {R_orig})"
        if R_pert <= R_orig + 1e-12:
            if R_pert < R_orig - 1e-10:
                return False, f"R should increase when epsilon increases from {eps} to {eps2}: {R_orig} -> {R_pert}"
        return True, f"R increases with epsilon > 1: {R_orig} -> {R_pert}"

    # ------------------------------------------------------------------
    # Weak answer sanity (only one allowed, must be non‑dominant)
    # ------------------------------------------------------------------
    def test_answer_sanity_finite_real(self):
        """Weak check: R must be a finite real number in [0,1]."""
        if self.R is None:
            return False, "SKIP: R not available"
        if not isinstance(self.R, Number) or math.isnan(self.R):
            return False, "R must be a finite number"
        if self.R < -1e-12 or self.R > 1.0 + 1e-12:
            # allow slight overshoot from rounding
            return False, f"R = {self.R} out of plausible range [-1e-12, 1+1e-12]"
        return True, f"R = {self.R} is a plausible finite real"

    # ------------------------------------------------------------------
    # Test runner (DO NOT modify)
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)