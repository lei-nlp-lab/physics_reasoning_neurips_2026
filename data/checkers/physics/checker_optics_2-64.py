import math
import traceback
from typing import Any, List, Tuple

class PhysicsChecker:
    """
    Physics auditor for single-slit diffraction crater size estimation.
    Checks interface completeness, physical invariants, modeling quality, monotonicity,
    and a weak answer sanity test.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract givens safely; never use numeric defaults.
        self.lambda_ = getattr(solution_instance, 'lambda', None)
        if self.lambda_ is None and hasattr(solution_instance, 'givens'):
            self.lambda_ = solution_instance.givens.get('lambda', None)
        self.D = getattr(solution_instance, 'D', None)
        if self.D is None and hasattr(solution_instance, 'givens'):
            self.D = solution_instance.givens.get('D', None)
        self.L = getattr(solution_instance, 'L', None)
        if self.L is None and hasattr(solution_instance, 'givens'):
            self.L = solution_instance.givens.get('L', None)

        # Extract result s from the emit contract.
        self.s = None
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.s = solution_instance.results.get('s', None)

    # ------------------------------------------------------------------
    # Interface tests: required inputs and output exist and are numeric.
    # ------------------------------------------------------------------
    def test_interface_given_lambda(self) -> Tuple[bool, str]:
        if self.lambda_ is None:
            return (False, "SKIP: lambda not provided")
        if not isinstance(self.lambda_, (int, float)):
            return (False, "lambda must be numeric")
        return (True, f"lambda = {self.lambda_} m")

    def test_interface_given_D(self) -> Tuple[bool, str]:
        if self.D is None:
            return (False, "SKIP: D not provided")
        if not isinstance(self.D, (int, float)):
            return (False, "D must be numeric")
        return (True, f"D = {self.D} m")

    def test_interface_given_L(self) -> Tuple[bool, str]:
        if self.L is None:
            return (False, "SKIP: L not provided")
        if not isinstance(self.L, (int, float)):
            return (False, "L must be numeric")
        return (True, f"L = {self.L} m")

    def test_interface_result_s(self) -> Tuple[bool, str]:
        if self.s is None:
            return (False, "SKIP: result 's' (crater diameter) not found")
        if not isinstance(self.s, (int, float)):
            return (False, "result s must be numeric")
        return (True, f"s = {self.s} m")

    # ---------------------------------------------------------------
    # Physical invariants (theorem-based, use math.isclose)
    # ---------------------------------------------------------------
    def test_invariant_diffraction_angle_consistency(self) -> Tuple[bool, str]:
        """Check that s / L ≈ lambda / D (small angle form)."""
        if any(x is None for x in (self.lambda_, self.D, self.L, self.s)):
            return (False, "SKIP: missing one or more quantities for angle check")
        if self.D == 0 or self.L == 0:
            return (False, "D or L is zero, cannot compute")
        theta_from_givens = self.lambda_ / self.D
        theta_from_result = self.s / self.L
        if math.isclose(theta_from_givens, theta_from_result, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"θ ≈ λ/D = {theta_from_givens:.3e} rad ≈ s/L = {theta_from_result:.3e} rad")
        else:
            return (False, f"θ mismatch: λ/D = {theta_from_givens:.3e}, s/L = {theta_from_result:.3e}")

    def test_invariant_small_angle_approximation_valid(self) -> Tuple[bool, str]:
        """Check that λ/D << 1, justifying small‑angle approximation."""
        if self.lambda_ is None or self.D is None or self.D == 0:
            return (False, "SKIP: cannot compute small angle condition")
        ratio = self.lambda_ / self.D
        if ratio < 1e-3:
            return (True, f"λ/D = {ratio:.2e} << 1, small‑angle approximation is valid")
        else:
            return (False, f"λ/D = {ratio:.2e} not small enough, small‑angle may be inaccurate")

    def test_invariant_diffraction_formula_structure(self) -> Tuple[bool, str]:
        """Check that s * D / (L * λ) is close to 1 (within rounding)."""
        if any(x is None for x in (self.lambda_, self.D, self.L, self.s)):
            return (False, "SKIP: missing data for diffraction formula check")
        if self.D == 0 or self.lambda_ == 0 or self.L == 0:
            return (False, "zero denominator in formula")
        computed = (self.s * self.D) / (self.L * self.lambda_)
        if math.isclose(computed, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"s*D/(L*λ) = {computed:.3f} ≈ 1")
        else:
            return (False, f"s*D/(L*λ) = {computed:.3f} deviates from 1")

    # -----------------------------------------------------------------
    # Modeling / trace consistency tests
    # -----------------------------------------------------------------
    def test_modeling_angular_resolution_computable(self) -> Tuple[bool, str]:
        """Check that the angular resolution can be computed from givens."""
        if self.lambda_ is None or self.D is None or self.D == 0:
            return (False, "SKIP: cannot compute angular resolution")
        theta = self.lambda_ / self.D
        if theta > 0:
            return (True, f"Angular resolution θ ≈ {theta:.3e} rad")
        else:
            return (False, "Angular resolution must be positive")

    def test_modeling_answer_derived_from_formula(self) -> Tuple[bool, str]:
        """Cross‑check that the reported s matches the direct formula computation (weak modeling)."""
        if any(x is None for x in (self.lambda_, self.D, self.L, self.s)):
            return (False, "SKIP: cannot compute expected s")
        expected = self.L * self.lambda_ / self.D
        if math.isclose(self.s, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"s = {self.s:.1f} m matches expected {expected:.1f} m")
        else:
            return (False, f"s = {self.s:.1f} m, expected {expected:.1f} m (from L*λ/D)")

    # -----------------------------------------------------------------
    # Monotonicity tests: explicitly perturb an input and compare output
    # -----------------------------------------------------------------
    def test_monotonicity_linear_size_increases_with_L(self) -> Tuple[bool, str]:
        """Perturb L by +10% and verify that the predicted s increases."""
        if any(x is None for x in (self.lambda_, self.D, self.L, self.s)):
            return (False, "SKIP: cannot perform monotonicity test")
        s_baseline = self.s
        L_perturbed = self.L * 1.10
        s_perturbed = L_perturbed * self.lambda_ / self.D
        if s_perturbed > s_baseline:
            return (True, f"L increased by 10% → s_perturbed = {s_perturbed:.1f} m > s_baseline = {s_baseline:.1f} m")
        else:
            return (False, f"Monotonicity fails: s_perturbed {s_perturbed:.1f} <= s_baseline {s_baseline:.1f}")

    def test_monotonicity_linear_size_decreases_with_D(self) -> Tuple[bool, str]:
        """Perturb D by +10% and verify that the predicted s decreases."""
        if any(x is None for x in (self.lambda_, self.D, self.L, self.s)):
            return (False, "SKIP: cannot perform monotonicity test")
        if self.D == 0:
            return (False, "D is zero")
        s_baseline = self.s
        D_perturbed = self.D * 1.10
        s_perturbed = self.L * self.lambda_ / D_perturbed
        if s_perturbed < s_baseline:
            return (True, f"D increased by 10% → s_perturbed = {s_perturbed:.1f} m < s_baseline = {s_baseline:.1f} m")
        else:
            return (False, f"Monotonicity fails: s_perturbed {s_perturbed:.1f} >= s_baseline {s_baseline:.1f}")

    # -----------------------------------------------------------------
    # Weak answer sanity (must NOT use math.isclose or re‑derive)
    # -----------------------------------------------------------------
    def test_answer_sanity_sign_and_range(self) -> Tuple[bool, str]:
        """Check that s is positive, finite, and within a plausible range (100–1000 m)."""
        if self.s is None:
            return (False, "SKIP: s not available")
        if not (isinstance(self.s, (int, float)) and math.isfinite(self.s)):
            return (False, "s must be a finite number")
        if self.s <= 0:
            return (False, f"s = {self.s} must be positive (crater diameter)")
        if 100.0 <= self.s <= 1000.0:
            return (True, f"s = {self.s:.1f} m is within plausible range [100, 1000] m")
        else:
            return (False, f"s = {self.s:.1f} m is outside plausible range [100, 1000] m")

    # -----------------------------------------------------------------
    # Main runner
    # -----------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs: List[str] = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name, None)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)