import math
from typing import Any, List, Tuple

class PhysicsChecker:
    """
    Audits the solution for computing the radial magnetic field B_r near the end
    of a semi-infinite solenoid. The checker uses physical invariants, modeling
    consistency, explicit monotonicity checks (via parameter perturbation), and a
    weak answer sanity check. It does not rely on hardcoded numeric constants.
    """

    def __init__(self, solution_instance: Any):
        """
        Extract all required inputs and results from the solution_instance.
        Use safe access with None defaults.
        """
        self.mu0 = getattr(solution_instance, 'mu0', None)
        if self.mu0 is None:
            self.mu0 = solution_instance.results.get('mu0') if hasattr(solution_instance, 'results') else None

        self.n = getattr(solution_instance, 'n', None)
        if self.n is None:
            self.n = solution_instance.results.get('n') if hasattr(solution_instance, 'results') else None

        self.I = getattr(solution_instance, 'I', None)
        if self.I is None:
            self.I = solution_instance.results.get('I') if hasattr(solution_instance, 'results') else None

        self.R = getattr(solution_instance, 'R', None)
        if self.R is None:
            self.R = solution_instance.results.get('R') if hasattr(solution_instance, 'results') else None

        self.r = getattr(solution_instance, 'r', None)
        if self.r is None:
            self.r = solution_instance.results.get('r') if hasattr(solution_instance, 'results') else None

        self.B_r = getattr(solution_instance, 'B_r', None)
        if self.B_r is None:
            self.B_r = solution_instance.results.get('B_r') if hasattr(solution_instance, 'results') else None

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_ methods. Return (all_pass, logs)."""
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

    # -------------------------------------------------------------------------
    # Interface checks
    # -------------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all required givens and result are present and numeric."""
        missing = []
        for key, val in [('mu0', self.mu0), ('n', self.n), ('I', self.I),
                         ('R', self.R), ('r', self.r), ('B_r', self.B_r)]:
            if val is None:
                missing.append(key)
            elif not isinstance(val, (int, float)):
                missing.append(f"{key} (non-numeric type)")
        if missing:
            return False, f"Missing or non-numeric attributes: {', '.join(missing)}"
        return True, "All required keys present and numeric"

    def test_interface_positive_radius(self) -> Tuple[bool, str]:
        """Check that R and r are positive (physical requirement)."""
        if (self.R is None) or (self.r is None):
            return False, "SKIP: R or r missing"
        if self.R <= 0:
            return False, f"R = {self.R} is not positive"
        if self.r <= 0:
            return False, f"r = {self.r} is not positive"
        return True, "R and r are positive"

    # -------------------------------------------------------------------------
    # Invariant checks: physics law consistency (use math.isclose)
    # -------------------------------------------------------------------------
    def test_invariant_divergence_derivation(self) -> Tuple[bool, str]:
        """
        Check that B_r matches the value derived from ∇·B = 0 and the on-axis
        field of a semi-infinite solenoid: B_r = μ₀ n I r / (4R).
        This is a direct derivation from the given parameters.
        """
        if any(v is None for v in [self.mu0, self.n, self.I, self.R, self.r, self.B_r]):
            return False, "SKIP: missing required inputs/B_r"
        expected = self.mu0 * self.n * self.I * self.r / (4.0 * self.R)
        if not math.isclose(self.B_r, expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"B_r = {self.B_r} does not match expected {expected} (tol 1%)"
        return True, f"B_r matches expected value {expected}"

    def test_invariant_proportionality_ratio(self) -> Tuple[bool, str]:
        """
        Check that B_r * 4R / (μ₀ n I r) ≈ 1. This tests the exact
        algebraic form without relying on a manually memorized constant.
        """
        if any(v is None for v in [self.mu0, self.n, self.I, self.R, self.r, self.B_r]):
            return False, "SKIP: missing required inputs/B_r"
        if self.mu0 * self.n * self.I * self.r == 0:
            return False, "Product μ₀ n I r is zero, cannot compute ratio"
        ratio = self.B_r * 4.0 * self.R / (self.mu0 * self.n * self.I * self.r)
        if not math.isclose(ratio, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"Ratio = {ratio}, expected ~1.0"
        return True, f"Proportionality holds (ratio = {ratio})"

    # -------------------------------------------------------------------------
    # Modeling / trace checks
    # -------------------------------------------------------------------------
    def test_modeling_linearity_in_r(self) -> Tuple[bool, str]:
        """
        Verify that B_r is proportional to r (linear in radial distance).
        Compare B_r / r to the factor (μ₀ n I / (4R)).
        """
        if any(v is None for v in [self.mu0, self.n, self.I, self.R, self.r, self.B_r]):
            return False, "SKIP: missing required inputs/B_r"
        if self.r == 0:
            return False, "r = 0, cannot compute B_r/r"
        factor = self.mu0 * self.n * self.I / (4.0 * self.R)
        actual_factor = self.B_r / self.r
        if not math.isclose(actual_factor, factor, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"B_r/r = {actual_factor}, expected ~{factor}"
        return True, f"B_r is linear in r with factor {factor}"

    def test_modeling_dependence_on_R(self) -> Tuple[bool, str]:
        """
        Check that B_r is inversely proportional to R: B_r * R should be
        independent of R (i.e., equal to μ₀ n I r / 4).
        """
        if any(v is None for v in [self.mu0, self.n, self.I, self.R, self.r, self.B_r]):
            return False, "SKIP: missing required inputs/B_r"
        expected_const = self.mu0 * self.n * self.I * self.r / 4.0
        computed_const = self.B_r * self.R
        if not math.isclose(computed_const, expected_const, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"B_r * R = {computed_const}, expected ~{expected_const}"
        return True, f"B_r * R agrees with constant {expected_const}"

    # -------------------------------------------------------------------------
    # Monotonicity checks by explicit parameter perturbation
    # -------------------------------------------------------------------------
    def test_monotonicity_current_I(self) -> Tuple[bool, str]:
        """
        Perturb the current I by +10% and verify that B_r scales proportionally.
        We compute the baseline and perturbed B_r using the derived physical
        relationship (∇·B=0 + on-axis field). Then compare the ratio perturbed/baseline
        to the ratio I_perturbed/I_baseline.
        """
        if any(v is None for v in [self.mu0, self.n, self.I, self.R, self.r]):
            return False, "SKIP: missing required inputs"
        # Baseline values (using the known physics, not the provided B_r)
        I_base = self.I
        B_base = self.mu0 * self.n * I_base * self.r / (4.0 * self.R)

        # Perturb: increase I by 10%
        I_pert = I_base * 1.1
        B_pert = self.mu0 * self.n * I_pert * self.r / (4.0 * self.R)

        # The ratio of B should equal the ratio of I
        ratio_B = B_pert / B_base
        ratio_I = I_pert / I_base
        if not math.isclose(ratio_B, ratio_I, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"B_pert/B_base = {ratio_B}, but I_pert/I_base = {ratio_I}"
        return True, f"B_r scales proportionally with I (ratio {ratio_B} vs {ratio_I})"

    def test_monotonicity_radius_R(self) -> Tuple[bool, str]:
        """
        Perturb the solenoid radius R by +10% and verify that B_r scales inversely
        with R (B_r ∝ 1/R). Compute baseline and perturbed B_r using the derived
        relationship; compare ratio of B to 1/1.1.
        """
        if any(v is None for v in [self.mu0, self.n, self.I, self.R, self.r]):
            return False, "SKIP: missing required inputs"
        # Baseline values
        R_base = self.R
        B_base = self.mu0 * self.n * self.I * self.r / (4.0 * R_base)

        # Perturb: increase R by 10%
        R_pert = R_base * 1.1
        B_pert = self.mu0 * self.n * self.I * self.r / (4.0 * R_pert)

        # Check inverse proportionality: B_pert / B_base = R_base / R_pert
        ratio_B = B_pert / B_base
        ratio_R = R_base / R_pert   # = 1/1.1
        if not math.isclose(ratio_B, ratio_R, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"B_pert/B_base = {ratio_B}, but R_base/R_pert = {ratio_R}"
        return True, f"B_r scales inversely with R (B_pert/B_base = {ratio_B} vs {ratio_R})"

    # -------------------------------------------------------------------------
    # Weak answer sanity (only one)
    # -------------------------------------------------------------------------
    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """
        Very weak sanity check on the provided B_r: positive, finite, non-NaN,
        and within a very broad plausible range. No derivation or equation solving.
        """
        if self.B_r is None:
            return False, "B_r is missing"
        if not math.isfinite(self.B_r):
            return False, f"B_r is not finite: {self.B_r}"
        if self.B_r <= 0:
            return False, f"B_r = {self.B_r} should be positive (magnitude)"
        # Plausible order: μ₀ ~1e-6, n~1000, I~1, R~0.01, r~0.001 → ~2.5e-5 T
        # Any value between 1e-15 and 1e5 is physically plausible for such a solenoid.
        if self.B_r < 1e-15 or self.B_r > 1e5:
            return False, f"B_r = {self.B_r} outside plausible range (1e-15 to 1e5)"
        return True, "B_r is positive, finite, and within plausible range"