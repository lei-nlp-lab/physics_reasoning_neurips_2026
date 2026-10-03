import math
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Audits a solution for the physics library postcard readability problem.
    Checks diffraction limit, bit pitch calculation, and resolution constraints.
    """

    def __init__(self, solution_instance: Any):
        """
        Extract all relevant data from the solution.
        Use None defaults for missing values.
        """
        # Givens (expected in solution_instance.givens or similar)
        givens = getattr(solution_instance, 'givens', {}) or {}
        self.I = givens.get('I', None)          # total bits in library
        self.A = givens.get('A', None)          # postcard area [m^2]
        self.d = givens.get('d', None)          # microscope resolution [m]

        # Results (expected in solution_instance.results)
        results = getattr(solution_instance, 'results', {}) or {}
        self.p = results.get('p', None)         # required bit pitch [m]
        self.p_min = results.get('p_min', None) # minimum resolvable pitch [m]

        # Store raw instance for potential fallback access
        self.solution = solution_instance

    # ------------------------------------------------------------------
    # Interface checks: required keys exist and are valid numbers
    # ------------------------------------------------------------------
    def test_interface_results_keys(self) -> Tuple[bool, str]:
        """Check that results contain 'p' and 'p_min' and are numeric."""
        if self.p is None:
            return False, "Missing result key 'p'"
        if self.p_min is None:
            return False, "Missing result key 'p_min'"
        if not isinstance(self.p, (int, float)):
            return False, f"'p' is not a number, got {type(self.p)}"
        if not isinstance(self.p_min, (int, float)):
            return False, f"'p_min' is not a number, got {type(self.p_min)}"
        return True, "All required result keys present and numeric"

    def test_interface_givens_available(self) -> Tuple[bool, str]:
        """Check that critical givens are provided (not None)."""
        missing = []
        if self.I is None:
            missing.append('I')
        if self.A is None:
            missing.append('A')
        if self.d is None:
            missing.append('d')
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return True, "All critical givens available"

    # ------------------------------------------------------------------
    # Invariant tests: physical laws (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_bit_pitch_relation(self) -> Tuple[bool, str]:
        """Check p^2 * I ≈ A (derived from p = sqrt(A/I))."""
        if any(x is None for x in [self.p, self.I, self.A]):
            return (False, "SKIP: missing data for pitch relation")
        lhs = self.p**2 * self.I
        if not math.isclose(lhs, self.A, rel_tol=1e-2, abs_tol=1e-2):
            return (False,
                    f"p^2 * I = {lhs:.3e}, expected A = {self.A:.3e}")
        return True, "Bit pitch relation holds within tolerance"

    def test_invariant_readability_condition(self) -> Tuple[bool, str]:
        """Require p >= p_min (bits are resolvable)."""
        if any(x is None for x in [self.p, self.p_min]):
            return (False, "SKIP: missing pitch data")
        # Allow 1% rounding slack: p must be at least 99% of p_min
        if self.p < self.p_min * (1 - 1e-2):
            return (False,
                    f"p = {self.p:.3e} < p_min = {self.p_min:.3e} (not readable)")
        return True, "Readability condition satisfied (p >= p_min)"

    # ------------------------------------------------------------------
    # Modeling / trace quality checks
    # ------------------------------------------------------------------
    def test_modeling_pitch_calculation(self) -> Tuple[bool, str]:
        """Check that p is computed as sqrt(A/I) (correct formula)."""
        if any(x is None for x in [self.p, self.I, self.A]):
            return (False, "SKIP: missing data for pitch calculation")
        expected = math.sqrt(self.A / self.I)
        if not math.isclose(self.p, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False,
                    f"p = {self.p:.3e}, expected sqrt(A/I) = {expected:.3e}")
        return True, "p computed correctly as sqrt(A/I)"

    def test_modeling_resolution_equality(self) -> Tuple[bool, str]:
        """Check that p_min is set equal to the given resolution d."""
        if any(x is None for x in [self.p_min, self.d]):
            return (False, "SKIP: missing data for resolution equality")
        if not math.isclose(self.p_min, self.d, rel_tol=1e-2, abs_tol=1e-2):
            return (False,
                    f"p_min = {self.p_min:.3e}, expected d = {self.d:.3e}")
        return True, "p_min correctly set equal to resolution d"

    def test_trace_readability_judgment(self) -> Tuple[bool, str]:
        """Verify the final judgment (readable) is consistent with computed values."""
        if any(x is None for x in [self.p, self.p_min]):
            return (False, "SKIP: missing pitch data")
        # According to the pseudocode, readability requires p >= p_min.
        if self.p < self.p_min:
            return (False,
                    f"Judgment inconsistent: p = {self.p:.3e} < p_min = {self.p_min:.3e}")
        return True, "Readability judgment consistent with computed pitches"

    # ------------------------------------------------------------------
    # Monotonicity test: perturb I and check p decreases
    # ------------------------------------------------------------------
    def test_monotonicity_pitch_with_bits(self) -> Tuple[bool, str]:
        """If I increases by 1%, p should decrease (for fixed A)."""
        if any(x is None for x in [self.p, self.I, self.A]):
            return (False, "SKIP: missing data for monotonicity test")
        # Compute baseline p from given data
        baseline = math.sqrt(self.A / self.I)
        # Perturb I upward by 1%
        I_pert = self.I * 1.01
        p_pert = math.sqrt(self.A / I_pert)
        if p_pert >= baseline:
            return (False,
                    f"p did not decrease: baseline {baseline:.3e}, perturbed {p_pert:.3e}")
        return True, "Monotonicity holds: p decreases when I increases"

    # ------------------------------------------------------------------
    # Weak final answer sanity (no isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_finite(self) -> Tuple[bool, str]:
        """Ensure p and p_min are finite positive numbers."""
        if self.p is None or self.p_min is None:
            return (False, "SKIP: missing answer values")
        if not (math.isfinite(self.p) and self.p > 0):
            return (False, f"p is not finite positive: {self.p}")
        if not (math.isfinite(self.p_min) and self.p_min > 0):
            return (False, f"p_min is not finite positive: {self.p_min}")
        return True, "Answer values are finite and positive"

    # ------------------------------------------------------------------
    # Test runner (mandatory)
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test_ methods and run them safely."""
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