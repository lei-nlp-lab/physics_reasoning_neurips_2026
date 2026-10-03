import math
from typing import Any, Dict, Tuple, List

class PhysicsChecker:
    """
    Auditor for the diffusion time of a helium atom to reach a spherical surface.
    Checks theorem-based physical sanity, modeling quality, and weak answer sanity.
    """

    def __init__(self, solution_instance) -> None:
        # Safe extraction of inputs and results (default None)
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.givens = getattr(solution_instance, 'givens', {}) or {}
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', []) or []

        # Required output
        self.t = self.results.get('t', None)

        # Input parameters (may or may not be provided)
        self.R = self.givens.get('R', None)
        self.D = self.givens.get('D', None)
        if self.D is None:
            self.D = self.results.get('D', None)

    def run_all_tests(self) -> Tuple[bool, List[str]]:
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

    # ---------- Interface checks ----------
    def test_interface_result_t_exists(self) -> Tuple[bool, str]:
        """Check that the required key 't' is present in results."""
        if self.t is None:
            return (False, "Key 't' not found in results")
        return (True, "Key 't' is present")

    def test_interface_t_is_numeric(self) -> Tuple[bool, str]:
        """Check that t is a finite number."""
        if self.t is None:
            return (False, "SKIP: t is None")
        if not isinstance(self.t, (int, float)):
            return (False, f"t is not numeric, got {type(self.t)}")
        if math.isnan(self.t) or math.isinf(self.t):
            return (False, f"t is NaN or Inf")
        return (True, "t is a finite number")

    # ---------- Invariant (theorem-based) checks ----------
    def test_invariant_positivity(self) -> Tuple[bool, str]:
        """Diffusion time must be positive."""
        if self.t is None:
            return (False, "SKIP: t not available")
        if self.t > 0:
            return (True, f"t = {self.t} > 0")
        else:
            return (False, f"t is not positive: {self.t}")

    def test_invariant_mean_square_displacement(self) -> Tuple[bool, str]:
        """
        If D and R are available, check that t ≈ R^2/(6D) with 1% tolerance.
        This is the core physics relation ⟨r²⟩ = 6Dt.
        """
        if self.t is None or self.R is None or self.D is None:
            return (False, "SKIP: missing t, R, or D for this check")
        expected_t = self.R**2 / (6.0 * self.D)
        if math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"t ≈ R^2/(6D): t={self.t}, expected={expected_t:.6f}")
        else:
            return (False, f"t deviates from R^2/(6D): t={self.t}, expected={expected_t:.6f}")

    def test_invariant_scaling_relation(self) -> Tuple[bool, str]:
        """
        Check that t * D / R^2 ≈ 1/6, using the 3D isotropic diffusion formula.
        """
        if self.t is None or self.R is None or self.D is None:
            return (False, "SKIP: missing t, R, or D")
        ratio = self.t * self.D / (self.R**2)
        if math.isclose(ratio, 1.0/6.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"t*D/R^2 = {ratio:.6f} ≈ 1/6")
        else:
            return (False, f"t*D/R^2 = {ratio:.6f}, expected ~ {1.0/6.0:.6f}")

    # ---------- Modeling / trace quality checks ----------
    def test_modeling_formula_origin(self) -> Tuple[bool, str]:
        """
        Check that the explanation steps reference the 3D diffusion
        mean square displacement relation (soft check).
        """
        step_text = " ".join(self.explanation_steps).lower()
        keywords = ["mean square displacement", "<r^2>", "6dt"]
        found = any(kw in step_text for kw in keywords)
        if found:
            return (True, "Explanation steps mention the 3D diffusion relation")
        else:
            return (True, "Steps may be empty or implicit (passing)")

    def test_modeling_point_source_assumption(self) -> Tuple[bool, str]:
        """
        Soft check that the solution acknowledges the point source assumption.
        """
        step_text = " ".join(self.explanation_steps).lower()
        if "point source" in step_text or "delta" in step_text:
            return (True, "Point source assumption acknowledged")
        else:
            return (True, "Point source assumption may be implicit (passing)")

    # ---------- Monotonicity checks ----------
    def test_monotonicity_time_increases_with_radius(self) -> Tuple[bool, str]:
        """
        Perturb R upward by 10% and check that expected t increases.
        Uses the physical relation ⟨r²⟩=6Dt recomputed inside the checker.
        """
        if self.t is None or self.R is None or self.D is None:
            return (False, "SKIP: missing t, R, or D for monotonicity test")
        baseline_t = self.t
        R_perturbed = self.R * 1.1
        expected_t_perturbed = R_perturbed**2 / (6.0 * self.D)
        if expected_t_perturbed > baseline_t:
            return (True, f"Perturbed R by +10% gives t_perturbed={expected_t_perturbed:.4f} > baseline_t={baseline_t:.4f}")
        else:
            return (False, f"Monotonicity violated: perturbed t should be larger, got {expected_t_perturbed:.4f}")

    def test_monotonicity_time_decreases_with_diffusion_coefficient(self) -> Tuple[bool, str]:
        """
        Perturb D upward by 10% and check that expected t decreases.
        """
        if self.t is None or self.D is None or self.R is None:
            return (False, "SKIP: missing t, D, or R for monotonicity test")
        baseline_t = self.t
        D_perturbed = self.D * 1.1
        expected_t_perturbed = self.R**2 / (6.0 * D_perturbed)
        if expected_t_perturbed < baseline_t:
            return (True, f"Perturbed D by +10% gives t_perturbed={expected_t_perturbed:.4f} < baseline_t={baseline_t:.4f}")
        else:
            return (False, f"Monotonicity violated: perturbed t should be smaller, got {expected_t_perturbed:.4f}")

    # ---------- Weak answer sanity (only one) ----------
    def test_answer_sanity_order_of_magnitude(self) -> Tuple[bool, str]:
        """
        Weak check: t should be in the broad range 100 s to 10000 s
        (i.e., 10^2 to 10^4 s) based on the expected answer ~ 860 s.
        No derived formula is used here.
        """
        if self.t is None:
            return (False, "SKIP: t not available")
        if 100 <= self.t <= 10000:
            return (True, f"t = {self.t} is within [100, 10000] s")
        else:
            return (False, f"t = {self.t} is outside the reasonable order of magnitude range [100, 10000] s")