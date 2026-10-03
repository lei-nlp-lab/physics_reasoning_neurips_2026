import math
import traceback
from typing import Any, Dict, List, Tuple

class PhysicsChecker:
    """
    PhysicsChecker for a mass sliding off a rotating disk and becoming a projectile.
    Expected result: horizontal distance x = sqrt(2*mu*R*h)
    """

    def __init__(self, solution_instance: Any):
        # Extract results – only key expected is 'x'
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}
        self.x = self.results.get('x', None)

        # Extract givens – required: g, mu, R, h
        self.givens = getattr(solution_instance, 'givens', None)
        if self.givens is None:
            self.givens = {}
        self.g = self.givens.get('g', None)
        self.mu = self.givens.get('mu', None)
        self.R = self.givens.get('R', None)
        self.h = self.givens.get('h', None)

        # Additional attributes (not used as numeric defaults)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_result_x_exists(self) -> Tuple[bool, str]:
        """Check that the result key 'x' is present and not None."""
        if self.x is None:
            return False, "SKIP: result 'x' is missing or None"
        return True, "Result 'x' is present"

    def test_interface_givens_complete(self) -> Tuple[bool, str]:
        """Verify that all required givens are provided and not None."""
        missing = []
        for name, val in [('g', self.g), ('mu', self.mu), ('R', self.R), ('h', self.h)]:
            if val is None:
                missing.append(name)
        if missing:
            return False, f"SKIP: missing givens: {', '.join(missing)}"
        return True, "All required givens (g, mu, R, h) are present"

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity checks)
    # ------------------------------------------------------------------
    def test_invariant_centripetal_equals_friction(self) -> Tuple[bool, str]:
        """
        From sliding condition: v^2 = mu * g * R.
        Use v = x / t, where t = sqrt(2h/g), and check that (x/t)^2 ≈ mu*g*R.
        """
        if any(v is None for v in (self.x, self.g, self.mu, self.R, self.h)):
            return False, "SKIP: missing data for centripetal/friction invariant"
        t = math.sqrt(2.0 * self.h / self.g)
        v = self.x / t
        v_sq_expected = self.mu * self.g * self.R
        v_sq_actual = v * v
        if math.isclose(v_sq_actual, v_sq_expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"v^2 ≈ mu*g*R: {v_sq_actual:.4f} ≈ {v_sq_expected:.4f}"
        else:
            return False, f"v^2 = {v_sq_actual:.4f}, expected {v_sq_expected:.4f}"

    def test_invariant_time_of_flight(self) -> Tuple[bool, str]:
        """
        Check that x / v ≈ t_expected, where v = sqrt(mu*g*R) and t_expected = sqrt(2h/g).
        This is a consistency relation between the two main physical steps.
        """
        if any(v is None for v in (self.x, self.g, self.mu, self.R, self.h)):
            return False, "SKIP: missing data for time-of-flight invariant"
        v_from_friction = math.sqrt(self.mu * self.g * self.R)
        t_deduced = self.x / v_from_friction
        t_expected = math.sqrt(2.0 * self.h / self.g)
        if math.isclose(t_deduced, t_expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"t_deduced ({t_deduced:.4f}) ≈ t_expected ({t_expected:.4f})"
        else:
            return False, f"t_deduced={t_deduced:.4f}, expected t={t_expected:.4f}"

    # ------------------------------------------------------------------
    # Modeling / trace quality tests (based on explanation steps or formula re-derivation)
    # ------------------------------------------------------------------
    def test_modeling_explanation_contains_friction_condition(self) -> Tuple[bool, str]:
        """
        Check that the explanation_steps contain the critical step:
        centripetal force = maximum static friction.
        """
        if self.explanation_steps is None:
            return False, "SKIP: no explanation_steps available"
        text = str(self.explanation_steps).lower()
        keywords = ["mv^2/r = μmg", "mv^2 / r = mu mg",
                     "centripetal force equals maximum static friction",
                     "friction provides centripetal force"]
        if any(k in text for k in keywords):
            return True, "Explanation includes the friction/centripetal condition"
        else:
            return False, "Explanation does not mention the centripetal-friction condition"

    def test_modeling_formula_consistency(self) -> Tuple[bool, str]:
        """
        Re-derive x from givens using the theoretical formula x = sqrt(2*mu*R*h)
        and compare to the emitted x. This is a numerical modeling check.
        """
        if any(v is None for v in (self.x, self.g, self.mu, self.R, self.h)):
            return False, "SKIP: missing data for formula consistency"
        # Derive expected x from the formula (inside checker, not hardcoded from answers)
        x_expected = math.sqrt(2.0 * self.mu * self.R * self.h)
        if math.isclose(self.x, x_expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"x ({self.x:.4f}) ≈ sqrt(2*mu*R*h) = {x_expected:.4f}"
        else:
            return False, f"x = {self.x:.4f}, but derived value = {x_expected:.4f}"

    # ------------------------------------------------------------------
    # Monotonicity test (trend check via input perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_friction_increases_distance(self) -> Tuple[bool, str]:
        """
        Increase mu by 10% and recompute x (using the analytic formula).
        The perturbed x must be larger than the baseline x.
        """
        if any(v is None for v in (self.x, self.mu, self.R, self.h)):
            return False, "SKIP: missing data for monotonicity test"
        if self.mu <= 0:
            return False, "SKIP: mu <= 0, monotonicity not meaningful"
        # baseline x computed from givens (we use the theoretical formula)
        x_baseline = math.sqrt(2.0 * self.mu * self.R * self.h)
        # perturb mu by +10%
        mu_perturbed = self.mu * 1.1
        x_perturbed = math.sqrt(2.0 * mu_perturbed * self.R * self.h)
        if x_perturbed > x_baseline:
            return True, f"10% increase in mu increases x: {x_perturbed:.4f} > {x_baseline:.4f}"
        else:
            return False, f"Monotonicity failed: x_perturbed ({x_perturbed:.4f}) ≤ baseline ({x_baseline:.4f})"

    # ------------------------------------------------------------------
    # Weak answer sanity (only positivity and finiteness)
    # ------------------------------------------------------------------
    def test_answer_sanity_positive_and_finite(self) -> Tuple[bool, str]:
        """Check that the emitted x is positive and finite."""
        if self.x is None:
            return False, "SKIP: no x value"
        if not math.isfinite(self.x):
            return False, "x is not finite (inf or NaN)"
        if self.x <= 0:
            return False, f"x is not positive (got {self.x})"
        return True, f"x = {self.x} is positive and finite"

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_ methods, collecting results."""
        logs: List[str] = []
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

    # Convenience alias
    def check(self) -> Tuple[bool, List[str]]:
        return self.run_all_tests()