import math
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Audits a solution to the asymmetric top energy level problem.
    """
    def __init__(self, solution_instance: Any) -> None:
        # Extract givens safely
        givens = getattr(solution_instance, 'givens', {})
        if not isinstance(givens, dict):
            givens = {}
        self.I   = givens.get('I', None)
        self.I3  = givens.get('I3', None)
        self.hbar = givens.get('hbar', None)
        self.A   = givens.get('A', None)

        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}
        self.E_JK    = results.get('E_JK', None)
        self.E0      = results.get('E0', None)
        self.E1      = results.get('E1', None)
        self.E_plus  = results.get('E_plus', None)
        self.E_minus = results.get('E_minus', None)

        # Explanation steps
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', [])

        # Working flag if givens are complete for numerical tests
        self._num_ready = (self.I is not None and self.hbar is not None and
                           self.I3 is not None and self.A is not None)

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_required_results_exist(self) -> Tuple[bool, str]:
        """Check that all expected result keys are present (not None)."""
        if self.E_JK is None:
            return False, "E_JK is missing"
        if self.E0 is None:
            return False, "E0 is missing"
        if self.E1 is None:
            return False, "E1 is missing"
        if self.E_plus is None:
            return False, "E_plus is missing"
        if self.E_minus is None:
            return False, "E_minus is missing"
        return True, "all five required result keys present"

    def test_interface_givens_exist(self) -> Tuple[bool, str]:
        """Check that all given parameters are numeric."""
        if self.I is None:
            return False, "I is missing"
        if self.I3 is None:
            return False, "I3 is missing"
        if self.hbar is None:
            return False, "hbar is missing"
        if self.A is None:
            return False, "A is missing"
        for name, val in [("I",self.I),("I3",self.I3),("hbar",self.hbar),("A",self.A)]:
            if not isinstance(val, (int, float)):
                return False, f"{name} is not numeric, got {type(val)}"
        return True, "all four givens present and numeric"

    def test_interface_explanation_steps_present(self) -> Tuple[bool, str]:
        """Check that explanation_steps list is not empty."""
        if not self.explanation_steps:
            return False, "explanation_steps is empty or missing"
        return True, f"explanation_steps has {len(self.explanation_steps)} step(s)"

    # ----------------------------------------------------------------------
    # Invariant / theorem-based sanity tests
    # ----------------------------------------------------------------------
    def test_invariant_E0_zero(self) -> Tuple[bool, str]:
        """E0 should be zero (J=0 state)."""
        if self.E0 is None:
            return False, "SKIP: E0 not available"
        if math.isclose(self.E0, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"E0 = {self.E0} is zero (within tolerance)"
        else:
            return False, f"E0 = {self.E0} is not zero"

    def test_invariant_E1_positive_and_value(self) -> Tuple[bool, str]:
        """E1 should be positive and equal to ħ²/I."""
        if not self._num_ready or self.E1 is None:
            return False, "SKIP: missing givens or E1"
        expected = self.hbar**2 / self.I
        if math.isclose(self.E1, expected, rel_tol=1e-2):
            return True, f"E1 = {self.E1}, expected ≈ {expected}"
        else:
            return False, f"E1 = {self.E1}, expected ≈ {expected}"

    def test_invariant_energy_splitting_symmetric(self) -> Tuple[bool, str]:
        """
        For a symmetric top (A=0), the J=1, K=±1 states are degenerate.
        Here we have nonzero A, so the splitting should be Aħ²/(2I²).
        Value re‑derived inside checker.
        """
        if not self._num_ready or self.E_plus is None or self.E_minus is None:
            return False, "SKIP: missing givens or split energies"
        split = self.E_plus - self.E_minus
        expected_split = self.A * self.hbar**2 / (2 * self.I**2)
        if math.isclose(abs(split), abs(expected_split), rel_tol=1e-2):
            return True, f"splitting = {split}, expected ≈ {expected_split}"
        else:
            return False, f"splitting = {split}, expected ≈ {expected_split}"

    def test_invariant_average_energy(self) -> Tuple[bool, str]:
        """
        The average of E_plus and E_minus should equal the unperturbed
        energy for J=1, |K|=1: ħ²/(2I) + ħ²/(2I3).
        """
        if not self._num_ready or self.E_plus is None or self.E_minus is None:
            return False, "SKIP: missing givens or split energies"
        avg = (self.E_plus + self.E_minus) / 2.0
        expected = self.hbar**2/(2*self.I) + self.hbar**2/(2*self.I3)
        if math.isclose(avg, expected, rel_tol=1e-2):
            return True, f"average = {avg}, expected ≈ {expected}"
        else:
            return False, f"average = {avg}, expected ≈ {expected}"

    # ----------------------------------------------------------------------
    # Modeling / trace tests
    # ----------------------------------------------------------------------
    def test_modeling_general_formula_structure(self) -> Tuple[bool, str]:
        """
        Check that the general symmetric‑top formula (E_JK) contains the
        correct functional dependence on J and K.
        (Assumes it is a sympy expression or a string.)
        """
        if self.E_JK is None:
            return False, "SKIP: E_JK not available"
        formula = self.E_JK
        if hasattr(formula, 'free_symbols'):
            symbols = {str(s) for s in formula.free_symbols}
            if 'J' in symbols and ('K' in symbols or 'm' in symbols):
                return True, "formula contains J and K/m"
            else:
                return False, f"formula symbols {symbols} lack J and K/m"
        elif isinstance(formula, str):
            has_J = 'J' in formula and 'J+1' in formula
            has_K = 'K' in formula or 'm' in formula
            if has_J and has_K:
                return True, "string formula contains J(J+1) and K term"
            else:
                return False, "string formula missing J or K dependence"
        else:
            return False, f"unexpected type for E_JK: {type(formula)}"

    def test_modeling_perturbation_matrix_offdiagonal(self) -> Tuple[bool, str]:
        """
        Check that the explanation steps indicate that the off‑diagonal
        matrix element <1,1|H'|1,-1> is nonzero.
        """
        if not self.explanation_steps:
            return False, "SKIP: no explanation steps"
        steps_text = " ".join(str(s) for s in self.explanation_steps).lower()
        keywords = ["1, -1", "1,1", "off-diagonal", "matrix element"]
        found = sum(kw in steps_text for kw in keywords)
        if found >= 2:
            return True, "explanation steps mention coupling of |1,1> and |1,-1>"
        else:
            return False, "explanation steps do not clearly describe the off‑diagonal coupling"

    # ----------------------------------------------------------------------
    # Monotonicity tests
    # ----------------------------------------------------------------------
    def test_monotonicity_splitting_increases_with_A(self) -> Tuple[bool, str]:
        """
        Verify that the splitting E_plus - E_minus increases with A.
        Perturb A by +10% and compare the expected new splitting to the baseline.
        """
        if not self._num_ready or self.E_plus is None or self.E_minus is None:
            return False, "SKIP: missing givens or split energies"
        # Baseline splitting
        S0 = self.E_plus - self.E_minus
        # Increase A by 10% and compute expected new splitting using derived formula
        A_new = self.A * 1.1
        S_new_expected = A_new * self.hbar**2 / (2 * self.I**2)
        if self.A >= 0:
            if S_new_expected >= S0:
                return True, f"splitting increases with A (baseline {S0}, new {S_new_expected})"
            else:
                return False, "expected splitting to increase, but computed decrease"
        else:
            if abs(S_new_expected) >= abs(S0):
                return True, f"splitting magnitude increases with |A| (baseline {abs(S0)}, new {abs(S_new_expected)})"
            else:
                return False, "expected splitting magnitude to increase, but computed decrease"

    # ----------------------------------------------------------------------
    # Answer sanity test (single weak method)
    # ----------------------------------------------------------------------
    def test_answer_sanity(self) -> Tuple[bool, str]:
        """
        Weak sanity checks on all three main answer quantities:
        - E0 is finite and non‑negative (should be zero).
        - E1 is finite and positive.
        - Ordering of E_plus and E_minus consistent with sign of A.
        No math.isclose, no re‑derivation.
        """
        issues = []
        # E0
        if self.E0 is None:
            issues.append("E0 missing")
        elif not math.isfinite(self.E0):
            issues.append("E0 not finite")
        elif self.E0 < 0:
            issues.append("E0 negative")

        # E1
        if self.E1 is None:
            issues.append("E1 missing")
        elif not math.isfinite(self.E1):
            issues.append("E1 not finite")
        elif self.E1 <= 0:
            issues.append("E1 non-positive")

        # Ordering
        if self.E_plus is None or self.E_minus is None:
            issues.append("split energies missing")
        else:
            if not (math.isfinite(self.E_plus) and math.isfinite(self.E_minus)):
                issues.append("split energies not finite")
            else:
                if self.A is not None:
                    if self.A > 0 and self.E_plus < self.E_minus:
                        issues.append("E_plus < E_minus but A > 0")
                    elif self.A < 0 and self.E_plus > self.E_minus:
                        issues.append("E_plus > E_minus but A < 0")

        if issues:
            return False, "; ".join(issues)
        return True, "all weak sanity checks passed"

    # ----------------------------------------------------------------------
    # Runner
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover and run all test_* methods, returning (overall_pass, logs).
        Each test runs in a try/except to prevent crashes.
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)