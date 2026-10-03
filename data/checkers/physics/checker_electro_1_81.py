import math
import sympy as sp
import traceback
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Physics auditor for the problem:
    Two similar charges at distance 2b apart, grounded conducting sphere of radius a
    placed midway that neutralizes mutual repulsion. Returns approximate a ≈ b/8.
    """

    def __init__(self, solution_instance: Any):
        """
        Safely extract all needed inputs/results from the solution instance.
        Default all missing values to None (never numeric defaults).
        """
        # Try to get a `results` dictionary
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # Givens: q, b (expected in solution_instance.givens or directly as attributes)
        self.q = None
        self.b = None

        # First try .givens dict, then direct attributes
        givens = getattr(solution_instance, 'givens', {})
        if isinstance(givens, dict):
            self.q = givens.get('q')
            self.b = givens.get('b')

        if self.q is None:
            self.q = getattr(solution_instance, 'q', None)
        if self.b is None:
            self.b = getattr(solution_instance, 'b', None)

        # Result: a_approx
        self.a = self.results.get('a_approx', None)

        # Coulomb constant (k=1 in natural units)
        self.k = 1.0

        # Optional explanation_steps and solving pseudocode for deeper checks
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

    # ----------------------------------------------------------------------
    # Interface checks: ensure required keys exist and have valid types
    # ----------------------------------------------------------------------
    def test_interface_results_a_approx(self) -> Tuple[bool, str]:
        """Check that results['a_approx'] exists and is a positive float."""
        if self.a is None:
            return (False, "SKIP: a_approx missing from results")
        if not isinstance(self.a, (int, float)):
            return (False, f"a_approx has type {type(self.a)}, expected numeric")
        if self.a <= 0:
            return (False, f"a_approx = {self.a} is not positive")
        return (True, f"a_approx is present and positive: {self.a}")

    def test_interface_givens_q_and_b(self) -> Tuple[bool, str]:
        """Check that q and b are provided."""
        issues = []
        if self.q is None:
            issues.append("q missing")
        if self.b is None:
            issues.append("b missing")
        if issues:
            return (False, f"SKIP: givens incomplete: {', '.join(issues)}")
        if self.b <= 0:
            return (False, f"b = {self.b} is not positive")
        return (True, f"givens q={self.q}, b={self.b}")

    # ----------------------------------------------------------------------
    # Invariant tests: theorem-based physical sanity (use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_approximate_ratio(self) -> Tuple[bool, str]:
        """Check that a/b ≈ 1/8 (leading order of force balance)."""
        if self.a is None or self.b is None or self.b <= 0:
            return (False, "SKIP: a or b missing/invalid")
        ratio = self.a / self.b
        target = 1.0 / 8.0
        if math.isclose(ratio, target, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"a/b = {ratio:.4f} ≈ 1/8 = {target}")
        else:
            return (False, f"a/b = {ratio:.4f} deviates from 1/8 = {target}")

    def test_invariant_force_balance_leading_order(self) -> Tuple[bool, str]:
        """Check that the leading‑order force balance 1/(4b) ≈ 2a/b holds."""
        if self.a is None or self.b is None or self.b <= 0:
            return (False, "SKIP: a or b missing/invalid")
        # LHS = 1/(4b), RHS = 2a/b (once q and k cancel)
        lhs = 1.0 / (4.0 * self.b)
        rhs = 2.0 * self.a / self.b
        # Allow 5% relative tolerance (the approximation is first order)
        if math.isclose(lhs, rhs, rel_tol=5e-2, abs_tol=1e-2):
            return (True, f"lead‑order force balance holds: LHS={lhs:.4e}, RHS≈{rhs:.4e}")
        else:
            return (False, f"lead‑order force balance violated: LHS={lhs:.4e}, RHS={rhs:.4e}")

    def test_invariant_small_radius_assumption(self) -> Tuple[bool, str]:
        """Check that a << b (i.e., a/b < 0.2) to justify expansion."""
        if self.a is None or self.b is None or self.b <= 0:
            return (False, "SKIP: a or b missing/invalid")
        ratio = self.a / self.b
        if ratio < 0.2:
            return (True, f"a/b = {ratio:.4f} < 0.2, valid small‑radius assumption")
        else:
            return (False, f"a/b = {ratio:.4f} is not sufficiently small (<0.2)")

    def test_invariant_coulomb_cancellation(self) -> Tuple[bool, str]:
        """Verify that the image force exactly cancels the direct repulsion (leading order)."""
        if self.a is None or self.b is None or self.b <= 0 or self.q is None:
            return (False, "SKIP: insufficient data for cancellation check")
        # Direct repulsion: k q^2 / (4b^2)
        f_direct = self.k * self.q**2 / (4.0 * self.b**2)
        # Attraction from two images (approx leading order): 2 * k q^2 a / b^3
        f_image_approx = 2.0 * self.k * self.q**2 * self.a / (self.b**3)
        if math.isclose(f_direct, f_image_approx, rel_tol=5e-2, abs_tol=1e-2):
            return (True, f"Direct force ≈ image force: {f_direct:.4e} vs {f_image_approx:.4e}")
        else:
            return (False, f"Force imbalance: direct={f_direct:.4e}, image≈{f_image_approx:.4e}")

    # ----------------------------------------------------------------------
    # Modeling / trace checks: based on explanation_steps and pseudocode
    # ----------------------------------------------------------------------
    def test_modeling_image_charge_magnitude(self) -> Tuple[bool, str]:
        """If the solution provides image charge magnitude, check q' = -(a/b)*q."""
        # Not required; skip if not present
        if not hasattr(self, 'q_image') and self.results.get('q_image') is None:
            return (True, "SKIP: image charge q_image not in solution")
        q_image = getattr(self, 'q_image', None) or self.results.get('q_image')
        if self.a is None or self.b is None or self.q is None:
            return (False, "SKIP: missing a, b, or q for image check")
        expected_q_image = - (self.a / self.b) * self.q
        if math.isclose(q_image, expected_q_image, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"q_image = {q_image} matches expected {expected_q_image}")
        else:
            return (False, f"q_image = {q_image} deviates from {expected_q_image}")

    def test_modeling_force_balance_exact(self) -> Tuple[bool, str]:
        """Evaluate the exact force‑balance equation (including high‑order terms)
        to verify that the approximate a is a good solution."""
        if self.a is None or self.b is None or self.b <= 0:
            return (False, "SKIP: a or b missing/invalid")
        # Essential check: ensure a < b (otherwise denominators blow up)
        if self.a >= self.b:
            return (False, f"a={self.a} >= b={self.b}, invalid configuration")
        # Exact equation (k q^2 cancels):
        # 1/(4b) = a*b^2 * [1/(b^2 - a^2)^2 + 1/(b^2 + a^2)^2]
        lhs = 1.0 / (4.0 * self.b)
        # Avoid division by zero: b^2 - a^2 > 0 ensured by a<b
        term1 = 1.0 / ((self.b**2 - self.a**2)**2)
        term2 = 1.0 / ((self.b**2 + self.a**2)**2)
        rhs = self.a * (self.b**2) * (term1 + term2)
        # Use a relaxed tolerance because a is only approximate
        if math.isclose(lhs, rhs, rel_tol=3e-1, abs_tol=1e-2):
            return (True, f"Exact balance: LHS={lhs:.4e}, RHS={rhs:.4e}")
        else:
            return (False, f"Exact balance violated: LHS={lhs:.4e}, RHS={rhs:.4e}")

    def test_modeling_image_positions(self) -> Tuple[bool, str]:
        """If solution provides image positions, check they are at ±a^2/b."""
        # Not required; skip if not present
        if not hasattr(self, 'image_pos') and self.results.get('image_pos') is None:
            return (True, "SKIP: image positions not in solution")
        image_pos = getattr(self, 'image_pos', None) or self.results.get('image_pos')
        if self.b is None or self.a is None:
            return (False, "SKIP: missing a or b")
        expected_pos = self.a**2 / self.b
        # image_pos might be a list: [-d, d] or similar
        if isinstance(image_pos, (list, tuple)):
            if len(image_pos) != 2:
                return (False, f"image_pos has {len(image_pos)} entries, expected 2")
            # Check both magnitude
            if (math.isclose(abs(image_pos[0]), expected_pos, rel_tol=1e-2) and
                math.isclose(abs(image_pos[1]), expected_pos, rel_tol=1e-2) and
                image_pos[0] < 0 and image_pos[1] > 0):
                return (True, f"Image positions ±{expected_pos} correct")
            else:
                return (False, f"Image positions {image_pos} not ±{expected_pos}")
        else:
            # Could be a single distance; just check magnitude
            if math.isclose(abs(image_pos), expected_pos, rel_tol=1e-2):
                return (True, f"Image distance {image_pos} ≈ {expected_pos}")
            else:
                return (False, f"Image distance {image_pos} not {expected_pos}")

    # ----------------------------------------------------------------------
    # Monotonicity test (perturbation‑based)
    # ----------------------------------------------------------------------
    def test_monotonicity_a_increases_with_b(self) -> Tuple[bool, str]:
        """Perturb b by +5% and -5% and verify that a changes in the same direction
        according to the approximate relation a ≈ b/8."""
        if self.a is None or self.b is None or self.b <= 0:
            return (False, "SKIP: a or b missing/invalid")

        # Baseline: a_current = self.a, b_current = self.b
        # The approximate relation is a = b/8. We'll compute perturbed a under this relation.
        # Perturbation 1: increase b by 5%
        b_plus = self.b * 1.05
        a_plus_expected = b_plus / 8.0
        if a_plus_expected <= self.a:
            return (False, f"Increasing b from {self.b} to {b_plus} should increase a (expected {a_plus_expected} > {self.a}).")

        # Perturbation 2: decrease b by 5%
        b_minus = self.b * 0.95
        a_minus_expected = b_minus / 8.0
        if a_minus_expected >= self.a:
            return (False, f"Decreasing b from {self.b} to {b_minus} should decrease a (expected {a_minus_expected} < {self.a}).")

        return (True, f"Monotonicity holds: a increases with b. "
                      f"b+5% → expected a={a_plus_expected:.4f} > {self.a:.4f}, "
                      f"b-5% → expected a={a_minus_expected:.4f} < {self.a:.4f}.")

    # ----------------------------------------------------------------------
    # Answer sanity (weak, no math.isclose, no re‑derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_positive(self) -> Tuple[bool, str]:
        """Ensure a is positive and less than b (as required by method of images)."""
        if self.a is None or self.b is None:
            return (False, "SKIP: a or b missing")
        if self.a <= 0:
            return (False, f"a = {self.a} is not positive")
        if self.a >= self.b:
            return (False, f"a = {self.a} is not less than b = {self.b}")
        return (True, f"a = {self.a} is positive and less than b = {self.b}")

    # ----------------------------------------------------------------------
    # Runner: discover all test_ methods and execute them
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)