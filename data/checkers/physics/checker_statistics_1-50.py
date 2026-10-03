import math
import sympy as sp
from typing import Any, Dict, List, Tuple, Optional, Callable


class PhysicsChecker:
    """
    Physics checker for entropy change in reversible isothermal and free expansion of ideal gas.
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract all relevant results and givens from the solution instance.
        Use None defaults for missing data.  NEVER use numeric fallbacks in __init__.
        """
        # Results from [emit] contract
        results = getattr(solution_instance, 'results', {}) or {}
        self.delta_S_gas_rev = results.get('delta_S_gas_rev', None)
        self.delta_S_univ_rev = results.get('delta_S_univ_rev', None)
        self.delta_S_gas_free = results.get('delta_S_gas_free', None)
        self.delta_S_univ_free = results.get('delta_S_univ_free', None)

        # Optional givens – use None if not present (no numeric defaults)
        givens = getattr(solution_instance, 'givens', {}) or {}
        self.n = givens.get('n', None)          # do NOT default to 1
        self.V1 = givens.get('V1', None)
        self.V2 = givens.get('V2', None)
        self.T = givens.get('T', None)

        # Explanation steps (list of dicts) – used for modeling checks
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', []) or []

        # Universal gas constant – define a symbol (sympy has no built-in R)
        self.R = sp.Symbol('R', real=True, positive=True)

        # We will NOT precompute expected dS here because n may be None.
        # Instead compute on the fly in tests.

    def _is_sympy(self, val: Any) -> bool:
        """Check if value is a sympy expression."""
        return isinstance(val, sp.Basic)

    def _to_num(self, val: Any) -> Optional[float]:
        """Try to convert value to float; handle sympy expressions."""
        if val is None:
            return None
        try:
            if self._is_sympy(val):
                # Replace symbolic R with numeric value (8.314462618 J/(mol·K))
                if val.has(self.R):
                    val_sub = val.subs(self.R, 8.314462618)
                else:
                    val_sub = val
                return float(val_sub.evalf())
            else:
                return float(val)
        except (TypeError, ValueError, AttributeError):
            return None

    def _safe_close(self, a: Any, b: Any, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> Tuple[bool, str]:
        """
        Compare two values numerically, handling sympy and None.
        Uses math.isclose (rel_tol=1e-2, abs_tol=1e-2) by default – appropriate for invariant/modeling tests.
        """
        if a is None or b is None:
            return False, "One or both values are None"
        if self._is_sympy(a) or self._is_sympy(b):
            # Symbolic comparison: simplify difference
            diff = sp.simplify(a - b)
            if diff == 0:
                return True, "Symbolic exact equality"
            else:
                # Fallback to numeric
                num_a = self._to_num(a)
                num_b = self._to_num(b)
                if num_a is None or num_b is None:
                    return False, f"Cannot evaluate numerically: {a}, {b}"
                return (math.isclose(num_a, num_b, rel_tol=rel_tol, abs_tol=abs_tol),
                        f"Numerical: {num_a} vs {num_b}")
        else:
            # Both numeric
            num_a = float(a)
            num_b = float(b)
            return (math.isclose(num_a, num_b, rel_tol=rel_tol, abs_tol=abs_tol),
                    f"Numeric: {num_a} vs {num_b}")

    # ---------- Helper: compute expected dS from n and R ----------
    def _expected_dS(self) -> Any:
        """Return expected symbolic entropy change: n*R*ln(2). If n missing assume 1 mole."""
        n_val = self.n if self.n is not None else 1
        # If n is symbolic, use it; else treat as int
        if self._is_sympy(n_val):
            return n_val * self.R * sp.log(2)
        else:
            return sp.nsimplify(n_val) * self.R * sp.log(2)

    # ------------------ Interface Checks ------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """All four result keys must be present and not None."""
        keys = ['delta_S_gas_rev', 'delta_S_univ_rev', 'delta_S_gas_free', 'delta_S_univ_free']
        for key in keys:
            if getattr(self, key, None) is None:
                return False, f"Missing key: {key}"
        return True, "All required keys present"

    def test_interface_key_types_extractable(self) -> Tuple[bool, str]:
        """Check that values can be converted to numeric (or are sympy)."""
        for key in ['delta_S_gas_rev', 'delta_S_univ_rev', 'delta_S_gas_free', 'delta_S_univ_free']:
            val = getattr(self, key)
            if val is None:
                continue
            if not (self._is_sympy(val) or isinstance(val, (int, float))):
                return False, f"{key} is not numeric or sympy: {type(val)}"
        return True, "All values are numeric or sympy"

    # ------------------ Invariant Checks (use math.isclose) ------------------
    def test_invariant_gas_same_state_function(self) -> Tuple[bool, str]:
        """ΔS_gas for reversible and free expansion must be equal (state function)."""
        return self._safe_close(self.delta_S_gas_rev, self.delta_S_gas_free,
                                rel_tol=1e-2, abs_tol=1e-2)

    def test_invariant_univ_rev_zero(self) -> Tuple[bool, str]:
        """ΔS_univ for reversible process must be zero."""
        return self._safe_close(self.delta_S_univ_rev, 0, rel_tol=1e-2, abs_tol=1e-2)

    def test_invariant_univ_free_equals_gas_free(self) -> Tuple[bool, str]:
        """ΔS_univ for free expansion equals ΔS_gas (surroundings unchanged)."""
        return self._safe_close(self.delta_S_univ_free, self.delta_S_gas_free,
                                rel_tol=1e-2, abs_tol=1e-2)

    def test_invariant_rev_gas_expected_form(self) -> Tuple[bool, str]:
        """ΔS_gas_rev should equal nR ln(2) (symbolically or numerically)."""
        expected = self._expected_dS()
        return self._safe_close(self.delta_S_gas_rev, expected,
                                rel_tol=1e-2, abs_tol=1e-2)

    def test_invariant_free_gas_expected_form(self) -> Tuple[bool, str]:
        """ΔS_gas_free should equal nR ln(2)."""
        expected = self._expected_dS()
        return self._safe_close(self.delta_S_gas_free, expected,
                                rel_tol=1e-2, abs_tol=1e-2)

    def test_invariant_rev_univ_expected_form(self) -> Tuple[bool, str]:
        """ΔS_univ_rev should be zero."""
        return self._safe_close(self.delta_S_univ_rev, 0, rel_tol=1e-8, abs_tol=1e-8)

    # ------------------ Modeling Checks (based on explanation steps) ------------------
    def test_modeling_heat_work_relation_rev(self) -> Tuple[bool, str]:
        """
        If Q_rev or W_rev are present in explanation steps, check ΔS_gas_rev = Q_rev/T.
        For isothermal expansion, Q_rev = W_rev.
        """
        # Look for Q_rev or W_rev in explanation steps
        Q_rev = None
        W_rev = None
        for step in self.explanation_steps:
            if isinstance(step, dict):
                if 'Q_rev' in step:
                    Q_rev = step['Q_rev']
                if 'W_rev' in step:
                    W_rev = step['W_rev']
        if Q_rev is not None and self.T is not None:
            expected_dS = Q_rev / self.T
            return self._safe_close(self.delta_S_gas_rev, expected_dS,
                                    rel_tol=1e-2, abs_tol=1e-2)
        elif W_rev is not None and self.T is not None:
            expected_dS = W_rev / self.T
            return self._safe_close(self.delta_S_gas_rev, expected_dS,
                                    rel_tol=1e-2, abs_tol=1e-2)
        else:
            # Not enough info, skip
            return True, "SKIP: Q_rev or W_rev/T not available"

    def test_modeling_free_expansion_adiabatic(self) -> Tuple[bool, str]:
        """
        For free expansion, Q=0 and W=0, so ΔS_surr=0.
        Check if the explanation mentions adiabatic or Q=0.
        """
        found_adiabatic = False
        for step in self.explanation_steps:
            text = str(step).lower()
            if 'adiabatic' in text or 'q=0' in text:
                found_adiabatic = True
                break
        if found_adiabatic:
            return True, "Explanation includes adiabatic condition for free expansion"
        else:
            # Not required; just note it's missing
            return True, "SKIP: No explicit adiabatic reference in explanation"

    # ------------------ Monotonicity Checks ------------------
    def test_monotonicity_volume_ratio(self) -> Tuple[bool, str]:
        """
        If V1 and V2 are available, perturb V2 by +10% and check that ΔS_gas increases.
        Uses explicit numeric recomputation.
        """
        if self.V1 is None or self.V2 is None:
            return True, "SKIP: V1 or V2 not provided"

        v1 = self._to_num(self.V1)
        v2 = self._to_num(self.V2)
        if v1 is None or v2 is None:
            return True, "SKIP: Cannot convert V1/V2 to numeric"

        # Baseline ΔS_gas (reversible or free, both same)
        baseline = self._to_num(self.delta_S_gas_rev)
        if baseline is None:
            return True, "SKIP: Cannot evaluate baseline ΔS_gas_rev"

        # Use n (default to 1 if not given)
        n_num = self._to_num(self.n) if self.n is not None else 1
        if n_num is None:
            n_num = 1
        R_num = 8.314462618
        new_v2 = v2 * 1.1
        new_dS = n_num * R_num * math.log(new_v2 / v1)

        if new_dS <= baseline:
            return False, f"Increasing V2 from {v2} to {new_v2} should increase ΔS (baseline={baseline}, new={new_dS})"
        return True, f"Monotonicity holds: ΔS increases when V2 increases"

    # ------------------ Weak Final Answer Sanity (at most one) ------------------
    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """
        Weak sanity: all four results should be finite, non-NaN, and have expected signs.
        delta_S_univ_rev is zero, others positive.
        Does NOT use math.isclose (uses simple inequality tolerance for zero check).
        """
        for key, expected_sign in [
            ('delta_S_gas_rev', 'positive'),
            ('delta_S_univ_rev', 'zero'),
            ('delta_S_gas_free', 'positive'),
            ('delta_S_univ_free', 'positive')
        ]:
            val = getattr(self, key)
            if val is None:
                return False, f"{key} is None"
            if self._is_sympy(val):
                num = self._to_num(val)
                if num is not None:
                    if math.isnan(num) or math.isinf(num):
                        return False, f"{key} is NaN or Inf"
                    if expected_sign == 'positive' and num <= 0:
                        return False, f"{key} should be positive but got {num}"
                    if expected_sign == 'zero' and abs(num) > 1e-6:
                        return False, f"{key} should be zero but got {num}"
                # If cannot evaluate, skip judgment (weak sanity)
            else:
                try:
                    num = float(val)
                except (TypeError, ValueError):
                    return False, f"{key} cannot be converted to float"
                if math.isnan(num) or math.isinf(num):
                    return False, f"{key} is NaN or Inf"
                if expected_sign == 'positive' and num <= 0:
                    return False, f"{key} should be positive but got {num}"
                if expected_sign == 'zero' and abs(num) > 1e-6:
                    return False, f"{key} should be zero but got {num}"
        return True, "All results pass weak sanity checks (finite, expected sign)"

    # ------------------ Test Runner ------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all test_ methods, run them, collect logs.
        Returns (all_pass, logs).
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
        return all_pass, logs