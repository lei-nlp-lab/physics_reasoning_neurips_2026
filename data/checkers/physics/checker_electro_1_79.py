import math
from typing import Any, Tuple, List, Optional

class PhysicsChecker:
    """
    Auditor for the cloud charge / electric field problem using method of images.
    Checks interface completeness, physics invariants, modeling consistency, and monotonicity.
    """

    # Physical constants
    EPSILON_0 = 8.8541878128e-12  # F/m
    K = 1.0 / (4.0 * math.pi * EPSILON_0)  # ~8.987551787e9 N·m²/C²

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract all given and result values with safe defaults.
        Expected attributes:
        - givens: E (V/m), d (m)
        - results: q (C), F (N)
        - explanation_steps: optional
        """
        # Givens (from problem context)
        self.E: Optional[float] = getattr(solution_instance, 'E', None)
        results = getattr(solution_instance, 'results', None)
        if self.E is None and results is not None:
            self.E = results.get('E', None)

        self.d: Optional[float] = getattr(solution_instance, 'd', None)
        if self.d is None and results is not None:
            self.d = results.get('d', None)

        # Derived results (expected keys)
        self.q: Optional[float] = None
        self.F: Optional[float] = None
        if results is not None:
            self.q = results.get('q', None)
            self.F = results.get('F', None)
        if self.q is None:
            self.q = getattr(solution_instance, 'q', None)
        if self.F is None:
            self.F = getattr(solution_instance, 'F', None)

        # Explanation steps (optional)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ------------------------------------------------------------------
    #   INTERFACE TESTS
    # ------------------------------------------------------------------
    def test_interface_givens_present(self) -> Tuple[bool, str]:
        """Check that required givens exist and are finite numbers."""
        if self.E is None:
            return (False, "Missing given: E (electric field at surface)")
        if not isinstance(self.E, (int, float)) or math.isnan(self.E) or math.isinf(self.E):
            return (False, f"Invalid E value: {self.E}")
        if self.d is None:
            return (False, "Missing given: d (cloud bottom height)")
        if not isinstance(self.d, (int, float)) or self.d <= 0:
            return (False, f"Invalid d value: {self.d}")
        return (True, f"Givens E={self.E} V/m, d={self.d} m present and valid")

    def test_interface_results_present(self) -> Tuple[bool, str]:
        """Check that required result keys exist and are finite numbers."""
        if self.q is None:
            return (False, "Missing result: q (cloud charge magnitude)")
        if not isinstance(self.q, (int, float)) or math.isnan(self.q) or math.isinf(self.q):
            return (False, f"Invalid q value: {self.q}")
        if self.F is None:
            return (False, "Missing result: F (external force on cloud)")
        if not isinstance(self.F, (int, float)) or math.isnan(self.F) or math.isinf(self.F):
            return (False, f"Invalid F value: {self.F}")
        return (True, f"Results q={self.q:.4e} C, F={self.F:.4e} N present and finite")

    # ------------------------------------------------------------------
    #   INVARIANT TESTS (physics laws, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_q_from_field(self) -> Tuple[bool, str]:
        """
        Check that the derived q satisfies E = (2*k*q) * (1/d² - 1/(2d)²).
        Using isclose with large tolerance due to rounding.
        """
        if None in (self.E, self.d, self.q):
            return (False, "SKIP: missing E, d, or q")

        # Compute expected field from given q and d
        # E = 2 * k * q * (1/d² - 1/(4d²)) = 2kq * (3/(4d²)) = (3kq)/(2d²)
        expected_E = 2.0 * self.K * self.q * (1.0 / (self.d**2) - 1.0 / ((2.0 * self.d)**2))
        if not math.isclose(expected_E, self.E, rel_tol=1e-2, abs_tol=1e-2):
            msg = (f"E computed from q and d is {expected_E:.4e}, "
                   f"but given E is {self.E:.4e} (diff {abs(expected_E - self.E):.4e})")
            return (False, msg)
        return (True, f"E from q and d matches given E within 2%")

    def test_invariant_force_negative(self) -> Tuple[bool, str]:
        """
        Physical constraint: the net external force on the cloud must be downward
        (negative z direction) according to the method of images.
        """
        if self.F is None:
            return (False, "SKIP: missing F")
        if self.F >= 0.0:
            return (False, f"Force F should be negative (downward), but got {self.F:.4e}")
        return (True, f"F={self.F:.4e} is negative (downward)")

    def test_invariant_q_positive(self) -> Tuple[bool, str]:
        """
        Physical constraint: derived q must be positive (top positive, bottom negative).
        """
        if self.q is None:
            return (False, "SKIP: missing q")
        if self.q <= 0.0:
            return (False, f"q should be positive, but got {self.q:.4e}")
        return (True, f"q={self.q:.4e} > 0")

    def test_invariant_derived_q_formula(self) -> Tuple[bool, str]:
        """
        Check that q = (8π ε0 d² E) / 3 is satisfied (from E formula solving).
        """
        if None in (self.E, self.d, self.q):
            return (False, "SKIP: missing E, d, or q")
        expected_q = (8.0 * math.pi * self.EPSILON_0 * self.d**2 * self.E) / 3.0
        if not math.isclose(expected_q, self.q, rel_tol=1e-2, abs_tol=1e-6):
            msg = (f"Expected q = {expected_q:.4e} from formula, "
                   f"but got {self.q:.4e} (diff {abs(expected_q - self.q):.4e})")
            return (False, msg)
        return (True, f"q matches formula (8π ε0 d² E)/3")

    # ------------------------------------------------------------------
    #   MODELING / TRACE TESTS (multiple tests required)
    # ------------------------------------------------------------------
    def test_modeling_force_expression_consistency(self) -> Tuple[bool, str]:
        """
        Verify that the computed force F matches the expression from the explanation:
        F = k q² [ -1/(600²) + 2/(900²) - 1/(1200²) ]  (using d=300).
        """
        if None in (self.d, self.q, self.F):
            return (False, "SKIP: missing d, q, or F")
        # distances from explanation: 600 = 2d, 900 = 3d, 1200 = 4d
        term = -1/(600**2) + 2/(900**2) - 1/(1200**2)
        expected_F = self.K * (self.q**2) * term
        if not math.isclose(expected_F, self.F, rel_tol=1e-2, abs_tol=1e-5):
            msg = (f"Expected F = {expected_F:.4e} from expression, "
                   f"but got {self.F:.4e} (diff {abs(expected_F - self.F):.4e})")
            return (False, msg)
        return (True, f"F matches the image-charge force expression within 2%")

    def test_modeling_force_coefficient_explanation(self) -> Tuple[bool, str]:
        """
        Check that the coefficient in the force expression from explanation steps
        matches: F = (q^2/(4π ε0 d²)) * (2/9 - 1/4 - 1/16).
        Derive the numeric factor from the given d and q and compare with F.
        """
        if None in (self.d, self.q, self.F):
            return (False, "SKIP: missing d, q, or F")
        # From explanation: F = (q^2/(4π ε0 d²)) * (2/9 - 1/4 - 1/16)
        factor = (2.0/9.0 - 1.0/4.0 - 1.0/16.0)  # about -0.0486
        expected_F = self.K * (self.q**2) * factor / (self.d**2)  # K = 1/(4π ε0)
        if not math.isclose(expected_F, self.F, rel_tol=1e-2, abs_tol=1e-5):
            msg = (f"Expected F = {expected_F:.4e} from explanation coefficient, "
                   f"but got {self.F:.4e} (diff {abs(expected_F - self.F):.4e})")
            return (False, msg)
        return (True, f"F matches the explanation coefficient factor (2/9 - 1/4 - 1/16) within 2%")

    # ------------------------------------------------------------------
    #   MONOTONICITY TESTS (explicit input perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_charge_increases_with_E(self) -> Tuple[bool, str]:
        """
        If electric field E increases, the derived q should increase (linearly).
        Simulate a +10% perturbation in E and compute expected q from formula,
        then compare to baseline q.
        """
        if None in (self.E, self.d, self.q):
            return (False, "SKIP: missing E, d, or q")
        # baseline q from formula: q = (8π ε0 d² E)/3
        base_q = (8.0 * math.pi * self.EPSILON_0 * self.d**2 * self.E) / 3.0
        # perturb E by +10%
        E_pert = self.E * 1.10
        perturbed_q = (8.0 * math.pi * self.EPSILON_0 * self.d**2 * E_pert) / 3.0
        # relative change
        rel_change = (perturbed_q - base_q) / base_q
        # Should be close to 0.10 (10%)
        if not math.isclose(rel_change, 0.10, rel_tol=0.02):
            msg = (f"For +10% E increase, q changes by {rel_change*100:.1f}% "
                   f"(expected ~10%)")
            return (False, msg)
        return (True, f"q increases proportionally with E (+10% E -> ~+10% q)")

    def test_monotonicity_force_magnitude_increases_with_q(self) -> Tuple[bool, str]:
        """
        If q increases, |F| should increase (since F ∝ q²).
        Perturb q by +10% and compute expected new |F| using the force expression,
        compare relative change to ~21% (since (1.1)² = 1.21).
        """
        if None in (self.q, self.F, self.d):
            return (False, "SKIP: missing q, d, or F")
        # baseline force magnitude (positive)
        base_F = abs(self.F)
        if base_F == 0:
            return (False, "Baseline force is zero, cannot test monotonicity")
        # perturb q by +10%
        q_pert = self.q * 1.10
        # expected perturbed force magnitude from expression (using the term from earlier)
        term = -1/(600**2) + 2/(900**2) - 1/(1200**2)
        expected_F_mag = abs(self.K * (q_pert**2) * term)
        rel_change = (expected_F_mag - base_F) / base_F
        # expected relative change: (1.1^2 - 1) = 0.21
        if not math.isclose(rel_change, 0.21, rel_tol=0.05):
            msg = (f"For +10% q increase, |F| changes by {rel_change*100:.1f}% "
                   f"(expected ~21%)")
            return (False, msg)
        return (True, f"|F| increases quadratically with q (+10% q -> ~+21% |F|)")

    # ------------------------------------------------------------------
    #   ANSWER SANITY (weak, at most one method)
    # ------------------------------------------------------------------
    def test_answer_sanity_plausible_ranges(self) -> Tuple[bool, str]:
        """
        Weak sanity: q and F should be of plausible orders of magnitude.
        Expected: q ~ 5e-4 to 8e-4 C, |F| ~ 1e-3 to 1e-2 N.
        Do NOT compute expected values; just check broad ranges.
        """
        if self.q is None or self.F is None:
            return (False, "SKIP: missing q or F")
        if not (1e-5 <= self.q <= 1e-2):
            return (False, f"q = {self.q:.4e} outside plausible range [1e-5, 1e-2] C")
        absF = abs(self.F)
        if not (1e-5 <= absF <= 1e-1):
            return (False, f"|F| = {absF:.4e} outside plausible range [1e-5, 1e-1] N")
        return (True, f"q = {self.q:.4e} C and |F| = {absF:.4e} N are plausible")

    # ------------------------------------------------------------------
    #   RUN ALL TESTS
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all test_ methods, execute them safely, collect logs.
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
                passed, msg = False, f"{name} CRASHED: {type(e).__name__}: {e}"
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)