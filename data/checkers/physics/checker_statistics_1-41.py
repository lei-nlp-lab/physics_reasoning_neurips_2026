import math
import traceback
from typing import Any, Dict, List, Tuple

class PhysicsChecker:
    """
    Auditor for heat pump equilibrium temperature problem.
    Checks physical consistency, modeling quality, and answer sanity.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract givens safely
        self.T0 = getattr(solution_instance, 'T0', None)
        self.W = getattr(solution_instance, 'W', None)
        self.alpha = getattr(solution_instance, 'alpha', None)

        # Extract results (expected key from pseudocode emit)
        self.results: Dict[str, Any] = {}
        if hasattr(solution_instance, 'results') and solution_instance.results is not None:
            self.results = solution_instance.results
        self.T_eq = self.results.get('T_eq', None)

        # Explanation steps (optional, for context)
        self.steps = getattr(solution_instance, 'explanation_steps', None)

    # ----------------------------------------------------------------------
    # Interface Tests
    # ----------------------------------------------------------------------
    def test_interface_T_eq_exists(self) -> Tuple[bool, str]:
        """Check that the result key 'T_eq' is present and not None."""
        if self.T_eq is None:
            return (False, "T_eq is missing or None in results")
        return (True, "T_eq field present")

    def test_interface_givens_not_none(self) -> Tuple[bool, str]:
        """Check that all required givens are provided."""
        missing = []
        if self.T0 is None:
            missing.append("T0")
        if self.W is None:
            missing.append("W")
        if self.alpha is None:
            missing.append("alpha")
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All givens present")

    # ----------------------------------------------------------------------
    # Invariant Tests (theorem-based, use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_heat_balance(self) -> Tuple[bool, str]:
        """
        Equilibrium condition: W * T_eq / (T_eq - T0) == alpha * (T_eq - T0).
        Derived from Carnot COP and heat balance.
        """
        if any(x is None for x in [self.T_eq, self.T0, self.W, self.alpha]):
            return (False, "SKIP: missing data")
        if self.T_eq <= self.T0:
            return (False, "T_eq <= T0, cannot evaluate heat balance")
        try:
            lhs = self.W * self.T_eq / (self.T_eq - self.T0)
            rhs = self.alpha * (self.T_eq - self.T0)
            if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "Heat balance holds within tolerance")
            else:
                return (False, f"LHS {lhs:.4f} != RHS {rhs:.4f}")
        except (ZeroDivisionError, OverflowError, ValueError) as e:
            return (False, f"Numerical error in heat balance: {e}")

    def test_invariant_T_eq_greater_than_T0(self) -> Tuple[bool, str]:
        """Physical constraint: T_eq > T0 (building warmer than outside)."""
        if self.T_eq is None or self.T0 is None:
            return (False, "SKIP: missing T_eq or T0")
        if self.T_eq > self.T0:
            return (True, "T_eq > T0")
        else:
            return (False, f"T_eq ({self.T_eq}) <= T0 ({self.T0})")

    def test_invariant_T_eq_positive(self) -> Tuple[bool, str]:
        """Weak sanity: T_eq is a positive temperature."""
        if self.T_eq is None:
            return (False, "SKIP: T_eq missing")
        if self.T_eq > 0:
            return (True, "T_eq positive")
        else:
            return (False, f"T_eq = {self.T_eq} is not positive")

    # ----------------------------------------------------------------------
    # Modeling / Trace Tests (at least two required)
    # ----------------------------------------------------------------------
    def test_modeling_formula_consistency(self) -> Tuple[bool, str]:
        """
        Re-derive T_eq from givens using the closed-form solution and compare.
        This is a re-derivation inside the checker from physical relationships.
        """
        if any(x is None for x in [self.T0, self.W, self.alpha, self.T_eq]):
            return (False, "SKIP: missing data for formula check")
        try:
            term = self.W / (2.0 * self.alpha)
            discriminant = term**2 + (self.W * self.T0) / self.alpha
            if discriminant < 0:
                return (False, "Discriminant negative in formula (unphysical givens)")
            expected_T_eq = self.T0 + term + math.sqrt(discriminant)
            if math.isclose(self.T_eq, expected_T_eq, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "Solution matches theoretical formula")
            else:
                return (False, f"T_eq {self.T_eq:.4f} != expected {expected_T_eq:.4f}")
        except (ZeroDivisionError, ValueError, OverflowError) as e:
            return (False, f"Error computing formula: {e}")

    def test_modeling_COP_consistency(self) -> Tuple[bool, str]:
        """
        Check consistency: Carnot COP = T_eq/(T_eq - T0) should equal
        the ratio Q_loss / W = alpha*(T_eq - T0)/W.
        Both are derived from the equilibrium condition.
        """
        if any(x is None for x in [self.T_eq, self.T0, self.W, self.alpha]):
            return (False, "SKIP: missing data for COP check")
        if self.T_eq <= self.T0:
            return (False, "T_eq <= T0, cannot compute COP")
        try:
            carnot_COP = self.T_eq / (self.T_eq - self.T0)
            actual_COP = self.alpha * (self.T_eq - self.T0) / self.W
            if math.isclose(carnot_COP, actual_COP, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "COP consistency holds (Carnot = Q_loss/W)")
            else:
                return (False, f"Carnot COP {carnot_COP:.4f} != actual COP {actual_COP:.4f}")
        except (ZeroDivisionError, ValueError, OverflowError) as e:
            return (False, f"Numerical error in COP check: {e}")

    def test_modeling_quadratic_solution(self) -> Tuple[bool, str]:
        """
        Check that T_eq satisfies the quadratic equation derived from
        the heat balance: alpha*T^2 - (2*alpha*T0 + W)*T + alpha*T0^2 = 0.
        """
        if any(x is None for x in [self.T_eq, self.T0, self.alpha, self.W]):
            return (False, "SKIP: missing data for quadratic check")
        try:
            quadratic_value = self.alpha * self.T_eq**2 - (2*self.alpha*self.T0 + self.W)*self.T_eq + self.alpha*self.T0**2
            if math.isclose(quadratic_value, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "T_eq satisfies the quadratic equation")
            else:
                return (False, f"Quadratic evaluates to {quadratic_value:.4f}, not zero")
        except (OverflowError, ValueError) as e:
            return (False, f"Error in quadratic check: {e}")

    # ----------------------------------------------------------------------
    # Monotonicity Tests (perturb one input and check trend)
    # ----------------------------------------------------------------------
    def test_monotonicity_W_increases_T_eq(self) -> Tuple[bool, str]:
        """
        Increasing W should increase T_eq (all else constant).
        Perturb W by +5% and compute new T_eq from the theoretical formula.
        """
        if any(x is None for x in [self.T0, self.W, self.alpha, self.T_eq]):
            return (False, "SKIP: missing data")
        try:
            baseline_T_eq = self._compute_T_eq(self.T0, self.W, self.alpha)
            perturbed_W = self.W * 1.05
            new_T_eq = self._compute_T_eq(self.T0, perturbed_W, self.alpha)
            if new_T_eq > baseline_T_eq:
                return (True, f"Increasing W from {self.W} to {perturbed_W} raises T_eq ({new_T_eq:.4f} > {baseline_T_eq:.4f})")
            else:
                return (False, f"Monotonicity failed: W increased but T_eq did not increase")
        except Exception as e:
            return (False, f"Monotonicity test crashed: {e}")

    def test_monotonicity_alpha_decreases_T_eq(self) -> Tuple[bool, str]:
        """
        Increasing α should decrease T_eq (all else constant).
        Perturb α by +5% and compute new T_eq.
        """
        if any(x is None for x in [self.T0, self.W, self.alpha, self.T_eq]):
            return (False, "SKIP: missing data")
        try:
            baseline_T_eq = self._compute_T_eq(self.T0, self.W, self.alpha)
            perturbed_alpha = self.alpha * 1.05
            new_T_eq = self._compute_T_eq(self.T0, self.W, perturbed_alpha)
            if new_T_eq < baseline_T_eq:
                return (True, f"Increasing α from {self.alpha} to {perturbed_alpha} lowers T_eq ({new_T_eq:.4f} < {baseline_T_eq:.4f})")
            else:
                return (False, f"Monotonicity failed: α increased but T_eq did not decrease")
        except Exception as e:
            return (False, f"Monotonicity test crashed: {e}")

    # ----------------------------------------------------------------------
    # Weak Answer Sanity (only one method, no isclose, no re‑derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_finite_and_sign(self) -> Tuple[bool, str]:
        """
        Weak sanity: T_eq must be finite, non-negative (or >0).
        No comparison with expected values.
        """
        if self.T_eq is None:
            return (False, "SKIP: T_eq missing")
        if not math.isfinite(self.T_eq):
            return (False, f"T_eq is not finite: {self.T_eq}")
        if self.T_eq <= 0:
            return (False, f"T_eq is not positive: {self.T_eq}")
        return (True, "T_eq is finite and positive")

    # ----------------------------------------------------------------------
    # Helper method (used in monotonicity tests)
    # ----------------------------------------------------------------------
    @staticmethod
    def _compute_T_eq(T0: float, W: float, alpha: float) -> float:
        """
        Compute equilibrium temperature from the derived closed‑form expression.
        """
        term = W / (2.0 * alpha)
        discriminant = term**2 + (W * T0) / alpha
        if discriminant < 0:
            raise ValueError("Negative discriminant in T_eq calculation")
        return T0 + term + math.sqrt(discriminant)

    # ----------------------------------------------------------------------
    # Test Runner
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and execute all test methods."""
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)