import math
import traceback
from typing import Any, Tuple, List

class PhysicsChecker:
    """
    Auditor class for checking the physics solution of a cyclist maximum speed problem.
    """

    def __init__(self, solution_instance: Any):
        """
        Initialize checker with data from solution_instance.
        Uses safe defaults (None) for missing attributes.
        """
        # Results dictionary
        results = getattr(solution_instance, 'results', {}) or {}
        self.V = results.get('V', None)

        # Givens (expected keys: 'a', 'P_prop')
        givens = getattr(solution_instance, 'givens', {}) or {}
        self.a = givens.get('a', None)
        self.P_prop = givens.get('P_prop', None)

        # Optional explanation steps and pseudocode (for trace consistency)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

    def _validate_numeric(self, value: Any, name: str) -> Tuple[bool, str]:
        """Check if value is a finite number (not None, not inf, not nan)."""
        if value is None:
            return False, f"VALIDATION: {name} is missing (None)"
        try:
            v = float(value)
            if math.isnan(v) or math.isinf(v):
                return False, f"VALIDATION: {name} is NaN or inf"
            return True, f"VALIDATION: {name} = {v}"
        except (TypeError, ValueError):
            return False, f"VALIDATION: {name} is not numeric: {value}"

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------

    def test_interface_result_V_exists(self) -> Tuple[bool, str]:
        """Check that the result key 'V' exists."""
        passed, msg = True, "Result key 'V' present in results."
        if self.V is None:
            passed = False
            msg = "Result key 'V' is missing (None)."
        return (passed, msg)

    def test_interface_result_V_numeric(self) -> Tuple[bool, str]:
        """Check that V is a valid finite number."""
        passed, msg = self._validate_numeric(self.V, "V")
        return passed, msg

    def test_interface_givens_available(self) -> Tuple[bool, str]:
        """Check that required givens are present (a, P_prop)."""
        if self.a is None or self.P_prop is None:
            return (False, "Required givens missing: a or P_prop")
        return (True, "All required givens present.")

    # ----------------------------------------------------------------------
    # Invariant / physics theorem tests (use math.isclose)
    # ----------------------------------------------------------------------

    def test_invariant_power_balance(self) -> Tuple[bool, str]:
        """Check that propulsive power ≈ resistive power: P_prop = a * V^2."""
        if self.a is None or self.P_prop is None:
            return (False, "SKIP: missing givens for power balance")
        if self.V is None:
            return (False, "SKIP: V not available")

        P_resist = self.a * (self.V ** 2)
        if math.isclose(self.P_prop, P_resist, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Power balance holds: P_prop={self.P_prop}, a*V^2={P_resist:.4f}")
        else:
            return (False, f"Power balance fails: P_prop={self.P_prop}, a*V^2={P_resist:.4f}")

    def test_invariant_force_balance(self) -> Tuple[bool, str]:
        """Check that propulsive force ≈ resistive force at steady state: F_prop = a*V."""
        if self.a is None or self.P_prop is None or self.V is None:
            return (False, "SKIP: missing data for force balance")
        if self.V == 0:
            return (False, "V is zero, force balance indeterminate")
        # Propulsive force = P_prop / V (since power = force * velocity)
        F_prop = self.P_prop / self.V
        F_resist = self.a * self.V
        if math.isclose(F_prop, F_resist, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Force balance: F_prop={F_prop:.4f} N, F_resist={F_resist:.4f} N")
        else:
            return (False, f"Force imbalance: F_prop={F_prop:.4f}, F_resist={F_resist:.4f}")

    def test_invariant_derived_velocity_formula(self) -> Tuple[bool, str]:
        """Check that the computed V ≈ sqrt(P_prop / a) (theoretical derivation)."""
        if self.a is None or self.P_prop is None or self.V is None:
            return (False, "SKIP: missing givens or V")
        if self.a <= 0:
            return (False, "Non-positive a, cannot compute sqrt")
        expected_V = math.sqrt(self.P_prop / self.a)
        if math.isclose(self.V, expected_V, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"V = {self.V:.4f} matches sqrt(P_prop/a) = {expected_V:.4f}")
        else:
            return (False, f"V = {self.V:.4f} does not match sqrt(P_prop/a) = {expected_V:.4f}")

    # ----------------------------------------------------------------------
    # Modeling / trace consistency tests (not hardcoding exact numbers)
    # ----------------------------------------------------------------------

    def test_modeling_resistive_power_positive(self) -> Tuple[bool, str]:
        """Check that resistive power (a*V^2) is positive."""
        if self.a is None or self.V is None:
            return (False, "SKIP: missing data")
        P_resist = self.a * (self.V ** 2)
        if P_resist > 0:
            return (True, f"Resistive power positive: {P_resist:.4f} W")
        else:
            return (False, f"Resistive power non-positive: {P_resist}")

    def test_modeling_velocity_positive(self) -> Tuple[bool, str]:
        """Check that V is positive (physically required)."""
        if self.V is None:
            return (False, "SKIP: V missing")
        if self.V > 0:
            return (True, f"V = {self.V} > 0")
        else:
            return (False, f"V = {self.V} is not positive")

    def test_modeling_velocity_finite(self) -> Tuple[bool, str]:
        """Check that V is finite (not inf or NaN)."""
        if self.V is None:
            return (False, "SKIP: V missing")
        try:
            v = float(self.V)
            if math.isfinite(v):
                return (True, "V is finite")
            else:
                return (False, "V is not finite")
        except:
            return (False, "V cannot be converted to float")

    # ----------------------------------------------------------------------
    # Monotonicity tests (perturb input, compare baseline V to new V)
    # ----------------------------------------------------------------------

    def test_monotonicity_velocity_increases_with_power(self) -> Tuple[bool, str]:
        """
        Check that a small increase in propulsive power leads to an increase in velocity.
        Use the solution's V as baseline (if available), then compute the new V
        under a +1% power using the same power balance model.
        """
        if self.a is None or self.P_prop is None:
            return (False, "SKIP: missing givens")
        if self.a <= 0:
            return (False, "Non-positive a, cannot compute")

        # Baseline: use solution's V if available, else compute theoretical
        if self.V is not None and self.V > 0:
            V_base = float(self.V)
        else:
            V_base = math.sqrt(self.P_prop / self.a)

        # Perturb power by +1%
        delta = 0.01
        P_new = self.P_prop * (1 + delta)
        V_new = math.sqrt(P_new / self.a)

        if V_new > V_base:
            return (True, f"Monotonicity: +1% power gives V_new={V_new:.4f} > V_base={V_base:.4f}")
        else:
            return (False, f"Monotonicity violation: V_new={V_new:.4f} <= V_base={V_base:.4f}")

    def test_monotonicity_velocity_decreases_with_resistance(self) -> Tuple[bool, str]:
        """
        Check that a small increase in resistance coefficient a leads to a decrease in velocity.
        Use the solution's V as baseline (if available), then compute the new V
        under a +5% resistance using the same power balance model.
        """
        if self.a is None or self.P_prop is None:
            return (False, "SKIP: missing givens")
        if self.a <= 0:
            return (False, "Non-positive a, cannot compute")

        # Baseline: use solution's V if available, else compute theoretical
        if self.V is not None and self.V > 0:
            V_base = float(self.V)
        else:
            V_base = math.sqrt(self.P_prop / self.a)

        # Perturb a by +5%
        delta = 0.05
        a_new = self.a * (1 + delta)
        V_new = math.sqrt(self.P_prop / a_new)

        if V_new < V_base:
            return (True, f"Monotonicity: +5% a gives V_new={V_new:.4f} < V_base={V_base:.4f}")
        else:
            return (False, f"Monotonicity violation: V_new={V_new:.4f} >= V_base={V_base:.4f}")

    # ----------------------------------------------------------------------
    # Weak answer sanity (maximum one such test; no isclose, no re-derivation)
    # ----------------------------------------------------------------------

    def test_answer_sanity_positive_finite(self) -> Tuple[bool, str]:
        """Weak sanity: V is positive and finite."""
        if self.V is None:
            return (False, "V missing")
        try:
            v = float(self.V)
            if math.isfinite(v) and v > 0:
                return (True, "V positive and finite")
            else:
                return (False, "V not positive or not finite")
        except:
            return (False, "V not convertible to float")

    # ----------------------------------------------------------------------
    # Main runner
    # ----------------------------------------------------------------------

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover and execute all test_* methods, collecting results.
        Returns (success, logs).
        """
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
                passed, msg = False, f"{name} CRASHED: {e}"

            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)