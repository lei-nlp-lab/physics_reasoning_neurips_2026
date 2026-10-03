import math
import traceback
from typing import Any, Dict, List, Optional, Tuple, Callable

class PhysicsChecker:
    """
    Robust checker for a camera lens physics problem.
    Checks thin lens equation, magnification, Rayleigh resolution,
    and monotonicity of object distance with focal length.
    """

    def __init__(self, solution_instance: Any):
        # --- Extract givens from solution_instance (try multiple sources) ---
        self.f = self._get_given(solution_instance, 'f', default=None)
        self.h_o = self._get_given(solution_instance, 'h_o', default=None)
        self.h_i = self._get_given(solution_instance, 'h_i', default=None)
        self.lambda_val = self._get_given(solution_instance, 'lambda', default=None)
        self.D = self._get_given(solution_instance, 'D', default=None)

        # --- Extract results (emitted quantities) ---
        results = getattr(solution_instance, 'results', {}) or {}
        self.d_o = results.get('d_o', None)
        self.d_i = results.get('d_i', None)
        self.delta_x = results.get('delta_x', None)

        # --- Compute derived quantities for convenience (None if missing) ---
        self.magnification = None
        if self.h_o is not None and self.h_i is not None and self.h_o != 0:
            self.magnification = self.h_i / self.h_o

    def _get_given(self, sol: Any, name: str, default: Any = None) -> Any:
        """Try to retrieve a given constant from various possible locations."""
        # From direct attribute of solution_instance
        val = getattr(sol, name, None)
        if val is not None:
            return val
        # From a 'givens' dict/object attribute
        givens = getattr(sol, 'givens', None)
        if givens is not None:
            if isinstance(givens, dict):
                val = givens.get(name, None)
            else:
                val = getattr(givens, name, None)
            if val is not None:
                return val
        # From results dict (sometimes givens are stored there)
        results = getattr(sol, 'results', {}) or {}
        val = results.get(name, None)
        if val is not None:
            return val
        return default

    # --------------------------------------------------------------
    # Interface checks: ensure required keys exist and have valid types
    # --------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that the three required result keys are present and not None."""
        if self.d_o is None:
            return False, "Missing required result 'd_o'"
        if self.d_i is None:
            return False, "Missing required result 'd_i'"
        if self.delta_x is None:
            return False, "Missing required result 'delta_x'"
        return True, "All required results present"

    def test_interface_givens_valid(self) -> Tuple[bool, str]:
        """Check that all expected givens are available and have sensible values."""
        missing = []
        for name, val in [('f', self.f), ('h_o', self.h_o), ('h_i', self.h_i),
                          ('lambda', self.lambda_val), ('D', self.D)]:
            if val is None:
                missing.append(name)
        if missing:
            return False, f"Missing givens: {', '.join(missing)}"
        # Also check they are positive
        for name, val in [('f', self.f), ('h_o', self.h_o), ('h_i', self.h_i),
                          ('lambda', self.lambda_val), ('D', self.D)]:
            if val is not None and val <= 0:
                return False, f"Given '{name}' must be positive, got {val}"
        return True, "All givens present and positive"

    # --------------------------------------------------------------
    # Invariant checks: physics laws (use math.isclose with 1% tolerance)
    # --------------------------------------------------------------
    def test_invariant_thin_lens(self) -> Tuple[bool, str]:
        """Check thin lens equation: 1/f = 1/d_o + 1/d_i."""
        if self.f is None or self.d_o is None or self.d_i is None:
            return False, "SKIP: missing f, d_o, or d_i"
        if self.f <= 0 or self.d_o <= 0 or self.d_i <= 0:
            return False, "All distances must be positive"
        lhs = 1.0 / self.f
        rhs = 1.0 / self.d_o + 1.0 / self.d_i
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"1/f={lhs:.5e}, 1/d_o+1/d_i={rhs:.5e} (rel. error {abs(lhs-rhs)/lhs:.2%})"
        else:
            return False, f"Thin lens mismatch: 1/f={lhs:.5e}, sum={rhs:.5e}"

    def test_invariant_magnification(self) -> Tuple[bool, str]:
        """Check |m| = h_i/h_o = d_i/d_o."""
        if self.h_o is None or self.h_i is None or self.d_o is None or self.d_i is None:
            return False, "SKIP: missing h_o, h_i, d_o, or d_i"
        if self.h_o == 0 or self.d_o == 0:
            return False, "division by zero"
        mag_from_heights = self.h_i / self.h_o
        mag_from_distances = self.d_i / self.d_o
        # We expect these to be equal (magnitude, sign not considered)
        if math.isclose(mag_from_heights, mag_from_distances, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"h_i/h_o={mag_from_heights:.5f}, d_i/d_o={mag_from_distances:.5f}"
        else:
            return False, f"Magnification mismatch: heights {mag_from_heights:.5f}, distances {mag_from_distances:.5f}"

    def test_invariant_rayleigh_resolution(self) -> Tuple[bool, str]:
        """Check Rayleigh criterion: delta_x = 1.22 * lambda * d_i / D."""
        if self.delta_x is None or self.lambda_val is None or self.d_i is None or self.D is None:
            return False, "SKIP: missing delta_x, lambda, d_i, or D"
        if self.D <= 0:
            return False, "Aperture D must be positive"
        predicted = 1.22 * self.lambda_val * self.d_i / self.D
        if math.isclose(self.delta_x, predicted, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"delta_x={self.delta_x:.5e}, predicted={predicted:.5e} (rel. error {abs(self.delta_x-predicted)/predicted:.2%})"
        else:
            return False, f"Rayleigh mismatch: got {self.delta_x:.5e}, expected ~{predicted:.5e}"

    def test_invariant_positive_distances(self) -> Tuple[bool, str]:
        """Check all distances are positive (d_o, d_i, f)."""
        issues = []
        for name, val in [('f', self.f), ('d_o', self.d_o), ('d_i', self.d_i)]:
            if val is not None and val <= 0:
                issues.append(f"{name} = {val} (must be positive)")
        if issues:
            return False, "; ".join(issues)
        return True, "All distances positive"

    # --------------------------------------------------------------
    # Monotonicity check (one test)
    # --------------------------------------------------------------
    def test_monotonicity_d_o_increases_with_f(self) -> Tuple[bool, str]:
        """
        If focal length f increases (with fixed magnification m),
        object distance d_o should increase proportionally.
        We test this by perturbing f by +1% and checking d_o also increases.
        """
        if self.f is None or self.d_o is None or self.magnification is None:
            return False, "SKIP: missing f, d_o, or magnification"
        if self.magnification <= 0:
            return False, "Magnification must be positive for real image"
        # Baseline: compute d_o from formula to verify consistency
        # First, check that the formula holds to within tolerance
        expected_d_o = (1 + self.magnification) * self.f / self.magnification
        if not math.isclose(self.d_o, expected_d_o, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"d_o {self.d_o} does not match formula (expected {expected_d_o:.2f})"
        # Now monotonicity: increase f by 1%
        f_perturbed = self.f * 1.01
        d_o_perturbed = (1 + self.magnification) * f_perturbed / self.magnification
        if d_o_perturbed > self.d_o:
            return True, f"d_o increases with f: {self.d_o:.2f} -> {d_o_perturbed:.2f} for +1% f"
        else:
            return False, f"Monotonicity failed: d_o did not increase (baseline {self.d_o:.2f}, perturbed {d_o_perturbed:.2f})"

    # --------------------------------------------------------------
    # Modeling / trace quality tests (at least two)
    # --------------------------------------------------------------
    def test_modeling_angular_resolution_consistency(self) -> Tuple[bool, str]:
        """
        Check that angular resolution from Rayleigh criterion matches delta_x / d_i.
        delta_x / d_i should equal 1.22 * lambda / D.
        """
        if self.delta_x is None or self.d_i is None or self.lambda_val is None or self.D is None:
            return False, "SKIP: missing delta_x, d_i, lambda, or D"
        if self.d_i <= 0 or self.D <= 0:
            return False, "d_i and D must be positive"
        angular_from_res = self.delta_x / self.d_i
        predicted_angular = 1.22 * self.lambda_val / self.D
        if math.isclose(angular_from_res, predicted_angular, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"delta_x/d_i={angular_from_res:.5e}, 1.22*lambda/D={predicted_angular:.5e} (rel. error {abs(angular_from_res-predicted_angular)/predicted_angular:.2%})"
        else:
            return False, f"Angular resolution mismatch: delta_x/d_i={angular_from_res:.5e}, expected {predicted_angular:.5e}"

    def test_modeling_demagnification(self) -> Tuple[bool, str]:
        """
        For a demagnified image (h_i < h_o), image distance d_i must be less than object distance d_o.
        Check that the emitted d_i and d_o satisfy this inequality.
        """
        if self.d_o is None or self.d_i is None:
            return False, "SKIP: missing d_o or d_i"
        if self.h_o is None or self.h_i is None:
            # Fallback: use magnification derived from distances if heights unavailable
            if self.magnification is not None:
                if self.magnification < 1:
                    if self.d_i < self.d_o:
                        return True, f"Demagnification (m={self.magnification:.4f} < 1): d_i={self.d_i:.2f} < d_o={self.d_o:.2f}"
                    else:
                        return False, f"Demagnification expected but d_i >= d_o: d_i={self.d_i:.2f}, d_o={self.d_o:.2f}"
                else:
                    # Not demagnified, skip this check
                    return True, "Image not demagnified (m>=1), skipping inequality check"
            else:
                return False, "SKIP: cannot determine magnification"
        # Compare heights directly
        if self.h_i < self.h_o:
            # Expect d_i < d_o
            if self.d_i < self.d_o:
                return True, f"Demagnification: h_i={self.h_i} < h_o={self.h_o}, d_i={self.d_i:.2f} < d_o={self.d_o:.2f}"
            else:
                return False, f"Demagnification expected but d_i >= d_o: d_i={self.d_i:.2f}, d_o={self.d_o:.2f}"
        else:
            # Image not demagnified, skip
            return True, "Image not demagnified (h_i >= h_o), skipping inequality check"

    def test_modeling_reasonable_distances(self) -> Tuple[bool, str]:
        """Check that d_o and d_i are plausible given the problem dimensions."""
        if self.d_o is None or self.d_i is None:
            return False, "SKIP: missing d_o or d_i"
        # object distance should be > 0 and probably < 10 m (1000 cm)
        if self.d_o <= 0 or self.d_o > 1000:
            return False, f"d_o = {self.d_o} cm seems out of range (>0, ≤1000 cm)"
        # image distance: should be > f (if f known)
        if self.f is not None and self.d_i <= self.f:
            return False, f"d_i = {self.d_i} cm should be > f = {self.f} cm for real image"
        return True, "Distances within reasonable range"

    # --------------------------------------------------------------
    # Weak answer sanity (no isclose, no re-derivation)
    # --------------------------------------------------------------
    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """
        Only check sign, finiteness, and very broad magnitude.
        No isclose, no expected value computation.
        """
        issues = []
        for name, val in [('d_o', self.d_o), ('d_i', self.d_i), ('delta_x', self.delta_x)]:
            if val is None:
                continue
            if val <= 0:
                issues.append(f"{name} must be positive, got {val}")
            if math.isnan(val) or math.isinf(val):
                issues.append(f"{name} is NaN or Inf")
        # delta_x should be very small (less than 0.1 cm typically)
        if self.delta_x is not None and self.delta_x > 1.0:
            issues.append(f"delta_x = {self.delta_x} cm is too large (>1 cm)")
        if issues:
            return False, "; ".join(issues)
        return True, "Answer sanity passed (positive, finite, delta_x < 1 cm)"

    # --------------------------------------------------------------
    # Run all tests
    # --------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test_* methods, execute them with try/except, collect logs."""
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
                passed, msg = False, f"{name} CRASHED: {str(e)[:200]}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)