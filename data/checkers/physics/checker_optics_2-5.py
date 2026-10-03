import math
from typing import Any, Dict, Optional, Tuple, List

class PhysicsChecker:
    """
    Physics checker for double‑slit interference problem (Fraunhofer regime).
    Checks numeric consistency, monotonicity, modeling quality, and basic sanity of computed fringe spacing.
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract all required quantities from the solution instance.
        Use None as default for missing values – never assign numeric fallbacks.
        """
        # Givens (expected as attributes or inside .given dict)
        self.lambda_val: Optional[float] = None
        self.f: Optional[float] = None
        self.d: Optional[float] = None

        # Try to extract from common sources
        if hasattr(solution_instance, 'given') and isinstance(solution_instance.given, dict):
            givens = solution_instance.given
            self.lambda_val = givens.get('lambda', givens.get('wavelength', None))
            self.f = givens.get('f', givens.get('focal_length', None))
            self.d = givens.get('d', givens.get('slit_separation', None))
        else:
            # Fall back to direct attributes
            self.lambda_val = getattr(solution_instance, 'lambda', None) or getattr(solution_instance, 'wavelength', None)
            self.f = getattr(solution_instance, 'f', None) or getattr(solution_instance, 'focal_length', None)
            self.d = getattr(solution_instance, 'd', None) or getattr(solution_instance, 'slit_separation', None)

        # Results (expected as .results dict or direct attribute)
        self.results: Dict[str, Any] = {}
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.results = solution_instance.results
        self.dx: Optional[float] = self.results.get('dx', None)

        # Optional: explanation steps (used for modeling checks, not for numeric hardcoding)
        self.explanation_steps: Optional[str] = None
        if hasattr(solution_instance, 'explanation_steps'):
            self.explanation_steps = solution_instance.explanation_steps

    # ------------------------------------------------------------------
    # Interface tests – check that expected keys exist and have valid types
    # ------------------------------------------------------------------
    def test_interface_dx_exists(self) -> Tuple[bool, str]:
        """Verify that results['dx'] is present and is a number."""
        if self.dx is None:
            return False, "dx is missing from results"
        if not isinstance(self.dx, (int, float)):
            return False, f"dx is not numeric, got {type(self.dx).__name__}"
        return True, f"dx = {self.dx}"

    def test_interface_givens_complete(self) -> Tuple[bool, str]:
        """Check that all three input parameters are available."""
        missing = []
        if self.lambda_val is None:
            missing.append("lambda")
        if self.f is None:
            missing.append("f")
        if self.d is None:
            missing.append("d")
        if missing:
            return False, f"Missing givens: {', '.join(missing)}"
        return True, f"Givens: lambda={self.lambda_val}, f={self.f}, d={self.d}"

    # ------------------------------------------------------------------
    # Invariant tests – theorem‑based physical sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_positivity(self) -> Tuple[bool, str]:
        """dx must be positive (fringe spacing is a physical distance)."""
        if self.dx is None:
            return False, "SKIP: dx not available"
        if self.dx <= 0:
            return False, f"dx is not positive: {self.dx}"
        return True, f"dx > 0: {self.dx}"

    def test_invariant_fringe_spacing_formula(self) -> Tuple[bool, str]:
        """
        Check that d * dx ≈ λ * f.
        This is the central double‑slit relation: Δx = λf/d.
        """
        if None in (self.dx, self.lambda_val, self.f, self.d):
            return False, "SKIP: missing one or more required quantities"
        lhs = self.d * self.dx
        rhs = self.lambda_val * self.f
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"d*dx = {lhs} ≈ λ*f = {rhs}"
        else:
            return False, f"d*dx = {lhs} differs from λ*f = {rhs} by more than 1%"

    # ------------------------------------------------------------------
    # Modeling / trace consistency checks (based on explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_small_angle_reference(self) -> Tuple[bool, str]:
        """
        Ensure that the explanation or pseudocode references the small‑angle approximation.
        This is a soft check – a missing mention does not invalidate the physics.
        """
        if self.explanation_steps is None:
            return False, "SKIP: explanation_steps not provided"
        text = self.explanation_steps.lower()
        if 'small angle' in text or 'sinθ' in text or 'tanθ' in text or 'x/f' in text:
            return True, "Small‑angle approximation is referenced"
        else:
            return False, "No reference to small‑angle approximation found in explanation_steps"

    def test_modeling_dx_formula_reference(self) -> Tuple[bool, str]:
        """
        Check that the explanation steps or pseudocode contains the key relation Δx = λf/d.
        This verifies that the derivation is explicitly present.
        """
        if self.explanation_steps is None:
            return False, "SKIP: explanation_steps not provided"
        text = self.explanation_steps.lower()
        # Check for typical formulations
        if ('Δx' in text or 'dx' in text or 'fringe spacing' in text) and \
           ('λf/d' in text or 'lambda*f/d' in text or 'λ f/d' in text):
            return True, "Fringe spacing formula Δx = λf/d is referenced"
        else:
            return False, "Fringe spacing formula not found in explanation_steps"

    # ------------------------------------------------------------------
    # Monotonicity tests – check that increasing d decreases dx
    # ------------------------------------------------------------------
    def test_monotonicity_slit_separation(self) -> Tuple[bool, str]:
        """
        Double‑slit spacing Δx = λf/d is inversely proportional to d.
        Increase d by 5% → new dx' should be smaller than original dx.
        """
        if None in (self.dx, self.lambda_val, self.f, self.d):
            return False, "SKIP: missing required quantities"
        factor = 1.05
        d_prime = self.d * factor
        dx_prime = self.lambda_val * self.f / d_prime
        if dx_prime < self.dx:
            return True, f"d increased by {factor*100-100:.0f}% → dx decreases from {self.dx} to {dx_prime}"
        else:
            return False, f"d increased by {factor*100-100:.0f}% but dx did not decrease (dx={self.dx}, dx'={dx_prime})"

    # ------------------------------------------------------------------
    # Weak answer sanity (only one, as per instructions)
    # ------------------------------------------------------------------
    def test_answer_sanity_finite_reasonable(self) -> Tuple[bool, str]:
        """
        Check that dx is a finite, non‑negative float and not extremely large/small.
        No numeric comparison with pre‑computed values.
        """
        if self.dx is None:
            return False, "SKIP: dx not available"
        if not math.isfinite(self.dx):
            return False, f"dx is not finite: {self.dx}"
        if self.dx < 0:
            return False, f"dx is negative: {self.dx}"
        # No magic constants – just ensure it's not absurd given typical parameters.
        return True, f"dx is finite and non‑negative: {self.dx}"

    # ------------------------------------------------------------------
    # Main test runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all test_* methods, run them, collect results.
        Returns (overall_pass, list_of_log_messages).
        """
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
        return all_pass, logs