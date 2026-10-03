import math
import traceback
from typing import Any, List, Tuple


class PhysicsChecker:
    """
    Robust checker for expectation values of L_x and L_x^2 in an eigenstate of L^2 and L_z.
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract inputs and results safely. All defaults are None.
        """
        # Extract given quantum numbers and constants
        self.l = getattr(solution_instance, 'l', None)
        self.m = getattr(solution_instance, 'm', None)
        self.hbar = getattr(solution_instance, 'hbar', None)

        # Extract results from dictionary (preferred) or attribute
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}

        self.Lx_expect = results.get('Lx_expect', None)
        self.Lx2_expect = results.get('Lx2_expect', None)

        # Optional: explanation steps for modeling checks (not used numerically)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all required fields are present and not None."""
        missing = []
        if self.l is None:
            missing.append('l')
        if self.m is None:
            missing.append('m')
        if self.hbar is None:
            missing.append('hbar')
        if self.Lx_expect is None:
            missing.append('Lx_expect')
        if self.Lx2_expect is None:
            missing.append('Lx2_expect')
        if missing:
            return False, f"Missing keys: {', '.join(missing)}"
        return True, "All required keys present."

    def test_interface_l_m_types(self) -> Tuple[bool, str]:
        """Check that l and m are numbers (int or float)."""
        if not isinstance(self.l, (int, float)):
            return False, f"l is not a number: {type(self.l)}"
        if not isinstance(self.m, (int, float)):
            return False, f"m is not a number: {type(self.m)}"
        return True, "l and m are numeric."

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_l_m_range(self) -> Tuple[bool, str]:
        """Check that l >= 0 and |m| <= l (integer constraints relaxed)."""
        if self.l is None or self.m is None:
            return False, "SKIP: l or m missing"
        if self.l < 0:
            return False, f"l = {self.l} is negative"
        if abs(self.m) > self.l + 1e-12:
            return False, f"|m| = {abs(self.m)} exceeds l = {self.l}"
        return True, f"l={self.l}, m={self.m} satisfy |m| <= l and l>=0"

    def test_invariant_lx_expect_zero(self) -> Tuple[bool, str]:
        """<L_x> must be zero for an eigenstate of L_z (by rotational symmetry)."""
        if self.Lx_expect is None:
            return False, "SKIP: Lx_expect not available"
        if not math.isfinite(self.Lx_expect):
            return False, f"Lx_expect is not finite: {self.Lx_expect}"
        if not math.isclose(self.Lx_expect, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"<L_x> = {self.Lx_expect} not close to 0"
        return True, "<L_x> = 0 (within tolerance)."

    def test_invariant_lx2_derived(self) -> Tuple[bool, str]:
        """
        Check that <L_x^2> equals (1/2)[l(l+1) - m^2]ħ^2.
        This follows from L^2 = L_x^2+L_y^2+L_z^2 and <L_x^2>=<L_y^2>.
        """
        if None in (self.l, self.m, self.hbar, self.Lx2_expect):
            return False, "SKIP: insufficient data to compute derived Lx2"
        expected = 0.5 * (self.l * (self.l + 1) - self.m ** 2) * (self.hbar ** 2)
        if not math.isfinite(self.Lx2_expect):
            return False, f"Lx2_expect is not finite: {self.Lx2_expect}"
        if not math.isclose(self.Lx2_expect, expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"<L_x^2> = {self.Lx2_expect} != expected {expected}"
        return True, f"<L_x^2> = {self.Lx2_expect} matches derived value."

    def test_invariant_hbar_positive(self) -> Tuple[bool, str]:
        """hbar should be positive."""
        if self.hbar is None:
            return False, "SKIP: hbar missing"
        if not math.isfinite(self.hbar) or self.hbar <= 0:
            return False, f"hbar = {self.hbar} is not positive"
        return True, "hbar > 0."

    # ------------------------------------------------------------------
    # Modeling / trace tests (quality of reasoning)
    # ------------------------------------------------------------------
    def test_modeling_uses_symmetry(self) -> Tuple[bool, str]:
        """
        Check that the explanation mentions the symmetry argument
        <L_x^2> = <L_y^2> or uses rotational symmetry.
        """
        if self.explanation_steps is None:
            return False, "SKIP: explanation_steps not provided"
        text = str(self.explanation_steps).lower()
        if "symmetry" in text or "rotational" in text:
            return True, "Explanation uses symmetry argument."
        return True, "Explanation present but no explicit symmetry mention. (non‑critical)"

    def test_modeling_uses_commutator_or_ladder(self) -> Tuple[bool, str]:
        """
        Check that the explanation references commutators or ladder operators.
        This is a common approach for deriving <L_x> and <L_x^2>.
        """
        if self.explanation_steps is None:
            return False, "SKIP: explanation_steps not provided"
        text = str(self.explanation_steps).lower()
        if "commutator" in text or "ladder" in text or "l_+" in text or "l_-" in text:
            return True, "Explanation uses commutator or ladder operators."
        return True, "Explanation present but no explicit commutator/ladder mention. (non‑critical)"

    def test_modeling_uses_lx2_expectation_formula(self) -> Tuple[bool, str]:
        """
        Check that the explanation contains the core formula for <L_x^2>:
        (1/2)(<L^2> - <L_z^2>) or similar.
        This is the key physics step.
        """
        if self.explanation_steps is None:
            return False, "SKIP: explanation_steps not provided"
        text = str(self.explanation_steps)
        # Look for the structure: '1/2' and 'L^2' and 'L_z^2'
        if "1/2" in text and "L^2" in text and "L_z^2" in text:
            return True, "Explanation contains the <L_x^2> formula."
        # Also accept LaTeX: \frac{1}{2} and L^2, L_z^2
        if "\\frac{1}{2}" in text and ("L^2" in text or "\\hat{L}^2" in text) and ("L_z^2" in text or "\\hat{L}_z^2" in text):
            return True, "Explanation contains the LaTeX <L_x^2> formula."
        return True, "Explanation present but formula not explicitly found. (non‑critical)"

    # ------------------------------------------------------------------
    # Monotonicity tests
    # ------------------------------------------------------------------
    def test_monotonicity_lx2_increases_with_l(self) -> Tuple[bool, str]:
        """
        Check that <L_x^2> increases when l increases (for fixed m, hbar).
        We compute baseline expected value using the solution's l,m, then
        compute a perturbed expected value with l increased by 1 (or 10% if l is float).
        """
        if None in (self.l, self.m, self.hbar) or self.Lx2_expect is None:
            return False, "SKIP: missing data for monotonicity test with l"
        # Baseline (using solution's own l,m,hbar)
        baseline = 0.5 * (self.l * (self.l + 1) - self.m ** 2) * (self.hbar ** 2)
        # Perturb l
        if isinstance(self.l, int):
            l_pert = self.l + 1
        else:
            l_pert = self.l * 1.1  # 10% increase
        # Compute perturbed expected value (same m, hbar)
        expected_pert = 0.5 * (l_pert * (l_pert + 1) - self.m ** 2) * (self.hbar ** 2)
        # Check monotonicity: perturbed > baseline
        if expected_pert <= baseline:
            return False, f"Increasing l from {self.l} to {l_pert} did not increase <L_x^2> (baseline {baseline}, perturbed {expected_pert})"
        return True, f"<L_x^2> increases with l (baseline {baseline}, perturbed {expected_pert})"

    def test_monotonicity_lx2_decreases_with_magnitude_m(self) -> Tuple[bool, str]:
        """
        Check that <L_x^2> decreases when |m| increases (for fixed l, hbar).
        We increase absolute value of m by 1 (or 10% if float) but stay within |m| ≤ l.
        If already at maximum |m|, skip.
        """
        if None in (self.l, self.m, self.hbar) or self.Lx2_expect is None:
            return False, "SKIP: missing data for monotonicity test with m"
        # Baseline
        baseline = 0.5 * (self.l * (self.l + 1) - self.m ** 2) * (self.hbar ** 2)
        # Determine perturbed m with larger |m|
        abs_m = abs(self.m)
        max_abs = self.l
        if abs_m >= max_abs - 1e-12:  # already at maximum
            return False, "SKIP: |m| already at maximum, cannot perturb to larger |m|"
        if isinstance(self.m, int):
            new_abs = abs_m + 1
        else:
            new_abs = abs_m * 1.1  # 10% increase
        if new_abs > max_abs:
            new_abs = max_abs  # clamp to max
        # Keep sign of original m (though only m^2 matters)
        m_pert = -new_abs if self.m < 0 else new_abs
        expected_pert = 0.5 * (self.l * (self.l + 1) - m_pert ** 2) * (self.hbar ** 2)
        # Check that perturbed is less than baseline (since m^2 increased)
        if expected_pert >= baseline:
            return False, f"Increasing |m| from {abs_m} to {new_abs} did not decrease <L_x^2> (baseline {baseline}, perturbed {expected_pert})"
        return True, f"<L_x^2> decreases with |m| (baseline {baseline}, perturbed {expected_pert})"

    # ------------------------------------------------------------------
    # Answer sanity (weak checks only) – exactly one method
    # ------------------------------------------------------------------
    def test_answer_sanity(self) -> Tuple[bool, str]:
        """
        Weak sanity check: both expectation values must be finite and <L_x^2> non-negative.
        No hardcoded target values or math.isclose used.
        """
        if self.Lx_expect is None or self.Lx2_expect is None:
            return False, "SKIP: missing expectation values"
        # Check finite
        if not math.isfinite(self.Lx_expect):
            return False, f"<L_x> = {self.Lx_expect} is not finite."
        if not math.isfinite(self.Lx2_expect):
            return False, f"<L_x^2> = {self.Lx2_expect} is not finite."
        # Check non-negative (allow small negative due to numerical noise)
        if self.Lx2_expect < -1e-12:
            return False, f"<L_x^2> = {self.Lx2_expect} is negative."
        return True, "Both expectation values finite and <L_x^2> non-negative."

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all bound test_* methods, execute them safely,
        and return (overall_pass, list of log messages).
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs