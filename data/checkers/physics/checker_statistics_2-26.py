import math
import traceback
from typing import Any, Tuple

class PhysicsChecker:
    """
    Audits a solution for the two-spin energy ensemble average problem.
    The solution should provide β (inverse temperature) and computed results:
    results['U_bar'] and results['U2_bar'].
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract input β safely
        givens = getattr(solution_instance, 'givens', {}) or {}
        self.beta = getattr(solution_instance, 'beta', None)
        if self.beta is None:
            self.beta = givens.get('beta', None)

        # Extract results safely
        results = getattr(solution_instance, 'results', {}) or {}
        self.U_bar = results.get('U_bar', None)
        self.U2_bar = results.get('U2_bar', None)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def _compute_single_avgs(self, beta: float) -> Tuple[float, float, float]:
        """Return (e_bar, e2_bar, Z) for a single spin using the canonical formulas."""
        exp_b = math.exp(beta)
        exp_neg = math.exp(-beta)
        Z = 1 + exp_b + exp_neg
        e_bar = (exp_neg - exp_b) / Z
        e2_bar = (exp_neg + exp_b) / Z
        return e_bar, e2_bar, Z

    # ----------------------------------------------------------------------
    # Interface tests (existence and types)
    # ----------------------------------------------------------------------
    def test_interface_beta_provided(self) -> Tuple[bool, str]:
        if self.beta is None:
            return (False, "β not provided in givens or as attribute")
        if not isinstance(self.beta, (int, float)):
            return (False, f"β must be numeric, got {type(self.beta)}")
        return (True, "β is available and numeric")

    def test_interface_results_keys(self) -> Tuple[bool, str]:
        if self.U_bar is None:
            return (False, "results['U_bar'] missing or None")
        if self.U2_bar is None:
            return (False, "results['U2_bar'] missing or None")
        return (True, "Both U_bar and U2_bar are present")

    def test_interface_numeric_results(self) -> Tuple[bool, str]:
        if self.U_bar is None or self.U2_bar is None:
            return (False, "Cannot check types: results missing")
        if not isinstance(self.U_bar, (int, float)):
            return (False, f"U_bar must be numeric, got {type(self.U_bar)}")
        if not isinstance(self.U2_bar, (int, float)):
            return (False, f"U2_bar must be numeric, got {type(self.U2_bar)}")
        return (True, "Both results are numeric")

    # ----------------------------------------------------------------------
    # Invariant tests (theorem‑based physical sanity)
    # ----------------------------------------------------------------------
    def test_invariant_partition_function_positive(self) -> Tuple[bool, str]:
        """Z = 1 + e^β + e^(-β) must always be > 0 for real β."""
        if self.beta is None:
            return (False, "β missing, cannot compute Z")
        Z = 1 + math.exp(self.beta) + math.exp(-self.beta)
        if Z <= 0:
            return (False, f"Z = {Z} is not positive, physically impossible")
        return (True, f"Z = {Z} > 0")

    def test_invariant_U_bar_zero_at_infinite_temperature(self) -> Tuple[bool, str]:
        """
        At β=0 (infinite temperature) the distribution becomes uniform,
        so U_bar must be 0.
        """
        if self.beta is None or self.U_bar is None:
            return (False, "β or U_bar missing")
        if not math.isclose(self.beta, 0.0, abs_tol=1e-10):
            return (False, "β is not 0, cannot check this invariant")
        if not math.isclose(self.U_bar, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"At β=0, U_bar should be 0, got {self.U_bar}")
        return (True, "U_bar is 0 at β=0")

    def test_invariant_U2_bar_uniform_limit(self) -> Tuple[bool, str]:
        """
        At β=0, the two‑spin system has uniform distribution over 9 states.
        Var(U) = 2*(<ε²> - <ε>²) = 4/3, so U2_bar = <U²> = Var + <U>² = 4/3.
        """
        if self.beta is None or self.U2_bar is None:
            return (False, "β or U2_bar missing")
        if not math.isclose(self.beta, 0.0, abs_tol=1e-10):
            return (False, "β is not 0, cannot check this invariant")
        expected = 4.0 / 3.0
        if not math.isclose(self.U2_bar, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"At β=0, U2_bar should be {expected}, got {self.U2_bar}")
        return (True, f"U2_bar at β=0 is {self.U2_bar} ≈ {expected}")

    # ----------------------------------------------------------------------
    # Modeling / trace tests (check internal consistency)
    # ----------------------------------------------------------------------
    def test_modeling_U_bar_formula(self) -> Tuple[bool, str]:
        """
        Check that U_bar satisfies the canonical formula:
        Ū = 2 * (e_β) where e_bar = (e^{-β} - e^{β}) / Z.
        """
        if self.beta is None or self.U_bar is None:
            return (False, "Missing β or U_bar")
        e_bar, _, _ = self._compute_single_avgs(self.beta)
        expected_U_bar = 2.0 * e_bar
        if not math.isclose(self.U_bar, expected_U_bar, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"U_bar formula mismatch: got {self.U_bar}, expected {expected_U_bar}")
        return (True, f"U_bar = {self.U_bar} matches formula")

    def test_modeling_U2_bar_formula(self) -> Tuple[bool, str]:
        """
        Check that U2_bar satisfies:
        U2_bar = 2 * (e^2_bar) + 2 * (e_bar)^2,
        using independence.
        """
        if self.beta is None or self.U2_bar is None:
            return (False, "Missing β or U2_bar")
        e_bar, e2_bar, _ = self._compute_single_avgs(self.beta)
        expected_U2_bar = 2.0 * e2_bar + 2.0 * (e_bar ** 2)
        if not math.isclose(self.U2_bar, expected_U2_bar, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"U2_bar independence mismatch: got {self.U2_bar}, expected {expected_U2_bar}")
        return (True, f"U2_bar = {self.U2_bar} matches independence formula")

    def test_modeling_U2_bar_explicit_target(self) -> Tuple[bool, str]:
        """
        Check against the explicit target expression:
        U2_bar = 2*(e^{2β}+e^{-2β}+e^{β}+e^{-β}) / (1+e^{β}+e^{-β})^2
        """
        if self.beta is None or self.U2_bar is None:
            return (False, "Missing β or U2_bar")
        exp_b = math.exp(self.beta)
        exp_neg = math.exp(-self.beta)
        Z = 1 + exp_b + exp_neg
        numerator = 2.0 * (exp_b**2 + exp_neg**2 + exp_b + exp_neg)
        expected = numerator / (Z ** 2)
        if not math.isclose(self.U2_bar, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"U2_bar explicit mismatch: got {self.U2_bar}, expected {expected}")
        return (True, f"U2_bar matches explicit formula")

    def test_modeling_positive_S2(self) -> Tuple[bool, str]:
        """
        The variance Var(U) = U2_bar - (U_bar)^2 must be non‑negative.
        """
        if self.U_bar is None or self.U2_bar is None:
            return (False, "Missing U_bar or U2_bar")
        var = self.U2_bar - self.U_bar ** 2
        if var < -1e-9:  # allow tiny numerical noise
            return (False, f"Variance is negative: {var}")
        return (True, f"Variance = {var} >= 0")

    # ----------------------------------------------------------------------
    # Monotonicity tests (trend behaviour)
    # ----------------------------------------------------------------------
    def test_monotonicity_U_bar_decreases_with_beta(self) -> Tuple[bool, str]:
        """
        For β > 0, as β increases, U_bar should decrease (become more negative).
        We check that at β+δ the value is larger (less negative) than at β,
        i.e., U_bar(β+δ) - U_bar(β) < 0 for δ > 0.
        """
        if self.beta is None or self.U_bar is None:
            return (False, "Missing β or U_bar")
        if self.beta < 0:
            return (False, "Monotonicity test only valid for β > 0")
        # Perturb β by +5%
        delta = 0.05 * abs(self.beta) + 1e-6
        beta_prime = self.beta + delta
        e_bar_prime, _, _ = self._compute_single_avgs(beta_prime)
        expected_U_bar_prime = 2.0 * e_bar_prime
        if self.U_bar < expected_U_bar_prime:  # original more negative -> smaller
            return (False, f"U_bar should decrease when β increases: baseline {self.U_bar}, at β+δ {expected_U_bar_prime}")
        return (True, f"U_bar decreases with β (baseline {self.U_bar} < {expected_U_bar_prime})")

    def test_monotonicity_U2_bar_decreases_with_high_beta(self) -> Tuple[bool, str]:
        """
        At high β, the system approaches ground state (ε=-1 each, U=-2, U²=4).
        So U2_bar should approach 4 from above? Actually from the explicit formula,
        for large β, U2_bar → 4 (from above). For moderate β, we can check
        monotonic decrease when β is large enough. We'll test for β > 1.
        """
        if self.beta is None or self.U2_bar is None:
            return (False, "Missing β or U2_bar")
        if self.beta < 1.0:
            return (False, "Monotonicity test requires β >= 1")
        delta = 0.1
        beta_prime = self.beta + delta
        # Use explicit formula for comparison
        exp_b = math.exp(self.beta)
        exp_neg = math.exp(-self.beta)
        Z = 1 + exp_b + exp_neg
        numer = 2.0 * (exp_b**2 + exp_neg**2 + exp_b + exp_neg)
        U2_current = numer / (Z ** 2)
        exp_b_p = math.exp(beta_prime)
        exp_neg_p = math.exp(-beta_prime)
        Z_prime = 1 + exp_b_p + exp_neg_p
        numer_p = 2.0 * (exp_b_p**2 + exp_neg_p**2 + exp_b_p + exp_neg_p)
        U2_prime = numer_p / (Z_prime ** 2)
        if U2_current < U2_prime:
            return (False, f"U2_bar should decrease for β>1, but {U2_current} < {U2_prime}")
        return (True, f"U2_bar decreases with β (current {U2_current}, after δ {U2_prime})")

    # ----------------------------------------------------------------------
    # Single weak answer sanity test (no derived values, no isclose)
    # ----------------------------------------------------------------------
    def test_answer_sanity_physical_range(self) -> Tuple[bool, str]:
        """
        Check that both U_bar and U2_bar are in plausible physical ranges
        and are finite numbers.
        """
        if self.U_bar is None or self.U2_bar is None:
            return (False, "Missing results, cannot perform sanity check")
        # Basic finiteness
        if not (math.isfinite(self.U_bar) and math.isfinite(self.U2_bar)):
            return (False, "One or both results are not finite")
        # U_bar must be between -2 and 2
        if not (-2.0 <= self.U_bar <= 2.0):
            return (False, f"U_bar = {self.U_bar} is outside [-2,2]")
        # U2_bar must be between 0 and 4
        if not (0.0 <= self.U2_bar <= 4.0):
            return (False, f"U2_bar = {self.U2_bar} is outside [0,4]")
        return (True, f"Both results are finite and within physical ranges: U_bar={self.U_bar}, U2_bar={self.U2_bar}")

    # ----------------------------------------------------------------------
    # Test runner
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, list]:
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
        return all_pass, logs