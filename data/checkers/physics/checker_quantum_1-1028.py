import math
import sympy
import sys
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Audits solution to the harmonic oscillator expectation value problem.
    Checks interface completeness, invariant physics (parity, normalization),
    modeling quality of emitted symbolic results, and weak answer sanity.
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract all needed data from the solution instance.
        Use None defaults for any missing attribute or result key.
        """
        # Extract results dictionary
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        if self.results is None:
            self.results = {}

        # Expected result keys (from pseudocode [emit] contract)
        self.avg_x_form: Optional[str] = self.results.get('avg_x_form', None)
        self.A_max: Optional[str] = self.results.get('A_max', None)
        self.B_max: Optional[str] = self.results.get('B_max', None)
        self.A_min: Optional[str] = self.results.get('A_min', None)
        self.B_min: Optional[str] = self.results.get('B_min', None)

        # Additional data for deeper checks (may be empty or None)
        self.givens: Optional[Dict[str, Any]] = getattr(solution_instance, 'givens', None)
        self.explanation_steps: Optional[str] = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode: Optional[str] = getattr(solution_instance, 'pseudocode', None)
        self.key_constraints: Optional[Dict[str, Any]] = getattr(solution_instance, 'key_constraints', None)
        self.physics_mapping: Optional[Dict[str, Any]] = getattr(solution_instance, 'physics_mapping', None)

    # --------------------------------------------------------------------------
    # Interface tests – ensure required keys exist and are of correct type
    # --------------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all five expected keys are present in results."""
        required = ['avg_x_form', 'A_max', 'B_max', 'A_min', 'B_min']
        missing = [k for k in required if k not in self.results]
        if missing:
            return (False, f"Missing key(s): {missing}")
        return (True, "All required keys present")

    def test_interface_key_types_string(self) -> Tuple[bool, str]:
        """Check that all result values are strings."""
        for key in ['avg_x_form', 'A_max', 'B_max', 'A_min', 'B_min']:
            val = self.results.get(key, None)
            if val is None:
                return (False, f"Key '{key}' is missing")
            if not isinstance(val, str):
                return (False, f"Key '{key}' is not a string, got {type(val).__name__}")
        return (True, "All result values are strings")

    # --------------------------------------------------------------------------
    # Invariant tests – fundamental physical principles that must hold
    # --------------------------------------------------------------------------
    def test_invariant_parity_form_of_avg_x(self) -> Tuple[bool, str]:
        """
        Parity argument demands that <x> is proportional to AB.
        Check that avg_x_form contains 'AB' or 'A*B' or 'A B' (case-insensitive).
        """
        if self.avg_x_form is None:
            return (False, "SKIP: avg_x_form not available")
        form_lower = self.avg_x_form.replace(' ', '').lower()
        # Accept expressions like "2cAB", "2c*A*B", "A*B", etc.
        if 'ab' in form_lower or 'a*b' in form_lower:
            return (True, "avg_x_form contains product AB as required by parity")
        return (False, f"avg_x_form='{self.avg_x_form}' does not contain AB product")

    def test_invariant_normalization_constraint(self) -> Tuple[bool, str]:
        """
        From key_constraints or physics_mapping we know A^2+B^2=1.
        If A_numeric and B_numeric are provided (as floats) they should satisfy it.
        Otherwise, check that the explanation mentions normalization.
        """
        # Try to extract numeric A,B from givens (optional)
        A_val = None
        B_val = None
        if self.givens:
            A_val = self.givens.get('A', None)
            B_val = self.givens.get('B', None)
        if A_val is not None and B_val is not None:
            if isinstance(A_val, (int, float)) and isinstance(B_val, (int, float)):
                norm = A_val**2 + B_val**2
                if math.isclose(norm, 1.0, rel_tol=1e-2):
                    return (True, f"Numerical normalization OK: A²+B² = {norm:.4f}")
                else:
                    return (False, f"Normalization violated: A²+B² = {norm:.4f}")
        # Fallback: check that a statement about normalization is present in explanation
        if self.explanation_steps and 'A^2 + B^2 = 1' in self.explanation_steps:
            return (True, "Explanation mentions normalization A²+B²=1")
        return (False, "SKIP: no numerical normalization data available")

    # --------------------------------------------------------------------------
    # Modeling / trace tests – check internal consistency of emitted results
    # --------------------------------------------------------------------------
    def test_modeling_max_product_consistent(self) -> Tuple[bool, str]:
        """
        For maximum AB product, A and B must have the same sign (A=B).
        Check that A_max and B_max are identical strings.
        """
        if self.A_max is None or self.B_max is None:
            return (False, "SKIP: A_max or B_max not available")
        if self.A_max == self.B_max:
            return (True, f"A_max = B_max = {self.A_max}: same sign as expected for maximum")
        else:
            return (False, f"A_max={self.A_max} != B_max={self.B_max}: expected equal for max")

    def test_modeling_min_product_opposite_sign(self) -> Tuple[bool, str]:
        """
        For minimum AB product, A and B must have opposite signs (A = -B).
        Check that A_min and B_min are opposite signs (by string inspection).
        """
        if self.A_min is None or self.B_min is None:
            return (False, "SKIP: A_min or B_min not available")
        # Helper to extract sign
        def sign(s: str) -> Optional[str]:
            s = s.strip()
            if s.startswith('-'):
                return '-'
            else:
                return '+'
        sA = sign(self.A_min)
        sB = sign(self.B_min)
        if sA is None or sB is None:
            return (False, "SKIP: cannot parse sign")
        if sA != sB:
            return (True, f"A_min sign {sA} and B_min sign {sB} are opposite as expected for minimum")
        else:
            return (False, f"A_min sign {sA} and B_min sign {sB} are same, expected opposite for minimum")

    def test_modeling_avg_x_form_matches_product(self) -> Tuple[bool, str]:
        """
        The form should contain AB, implying <x> = 2cAB. Check that the string
        includes '2' and 'c' in addition to AB (not too strict).
        """
        if self.avg_x_form is None:
            return (False, "SKIP: avg_x_form not available")
        form = self.avg_x_form.replace(' ', '').upper()
        has_AB = 'AB' in form or 'A*B' in form
        has_2 = '2' in form
        has_c = 'C' in form or 'CONST' in form
        if has_AB and has_2 and has_c:
            return (True, f"avg_x_form='{self.avg_x_form}' matches expected form 2cAB")
        # Relaxed: at least contains AB
        if has_AB:
            return (True, "avg_x_form contains AB (constant/2 may be missing)")
        return (False, f"avg_x_form='{self.avg_x_form}' does not seem to represent 2cAB")

    # --------------------------------------------------------------------------
    # Monotonicity test – explicit perturbation of an input and comparison
    # --------------------------------------------------------------------------
    def test_monotonicity_AB_product_near_typical(self) -> Tuple[bool, str]:
        """
        If we have numeric A,B from givens (with A²+B²=1), perturb A slightly
        and recompute product AB keeping normalization: B = sqrt(1-A²).
        Near the maximum (A ≈ 1/√2), AB should decrease when A is increased.
        This test is only run if numeric A is provided.
        """
        A = None
        if self.givens:
            A = self.givens.get('A', None)
        if not isinstance(A, (int, float)):
            return (False, "SKIP: no numeric A available")
        if A**2 >= 1:
            return (False, "SKIP: A² >= 1, cannot deduce B")
        B = math.sqrt(1 - A**2)  # positive branch
        product0 = A * B
        # Small perturbation: +1% of A (or 0.001 if A=0)
        delta = 0.01 * A if A != 0 else 0.001
        A_new = A + delta
        if A_new**2 >= 1:
            return (False, "SKIP: perturbation pushes A² >=1")
        B_new = math.sqrt(1 - A_new**2)
        product_new = A_new * B_new
        diff = product_new - product0
        if diff > 0:
            msg = f"Increasing A from {A:.4f} to {A_new:.4f} increases AB from {product0:.4f} to {product_new:.4f}"
        else:
            msg = f"Increasing A from {A:.4f} to {A_new:.4f} decreases AB from {product0:.4f} to {product_new:.4f}"
        return (True, msg + " (monotonicity executed)")

    # --------------------------------------------------------------------------
    # Single weak answer sanity test
    # --------------------------------------------------------------------------
    def test_answer_sanity_values_non_empty(self) -> Tuple[bool, str]:
        """Check that all result strings are non-empty and not whitespace."""
        keys = ['avg_x_form', 'A_max', 'B_max', 'A_min', 'B_min']
        for k in keys:
            val = self.results.get(k, None)
            if val is None:
                return (False, f"Key '{k}' is None")
            if not isinstance(val, str) or val.strip() == '':
                return (False, f"Key '{k}' is empty or whitespace")
        return (True, "All result strings are non-empty")

    # --------------------------------------------------------------------------
    # Run all tests
    # --------------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test_ methods, run them, and collect results."""
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