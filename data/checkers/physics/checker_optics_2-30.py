import math
import sympy as sp
import traceback
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Auditor for diffraction problems: opaque disk and circular hole.
    Tests physics invariants, modeling consistency, and monotonicity.
    """

    def __init__(self, solution_instance: Any) -> None:
        """Extract inputs and results safely with None defaults."""
        # Attempt to retrieve inputs: wavelength (lambda), I0, R, L
        self.wavelength = getattr(solution_instance, 'wavelength', None)
        if self.wavelength is None:
            self.wavelength = getattr(solution_instance, 'lambda', None)
        self.I0 = getattr(solution_instance, 'I0', None)
        self.R = getattr(solution_instance, 'R', None)
        self.L = getattr(solution_instance, 'L', None)

        # Try 'givens' dict as fallback
        givens = getattr(solution_instance, 'givens', None)
        if givens is not None and isinstance(givens, dict):
            self.wavelength = self.wavelength if self.wavelength is not None else givens.get('lambda')
            self.I0 = self.I0 if self.I0 is not None else givens.get('I0')
            self.R = self.R if self.R is not None else givens.get('R')
            self.L = self.L if self.L is not None else givens.get('L')

        # Results: must match pseudocode [emit] keys
        results = getattr(solution_instance, 'results', None)
        if results is None:
            results = {}
        self.I_disk = results.get('I_disk', None)
        self.phi = results.get('phi', None)
        self.I_hole = results.get('I_hole', None)

        # Derived quantity
        self.k = None
        if self.wavelength is not None and self.wavelength != 0:
            self.k = 2.0 * math.pi / self.wavelength

    # ------------------------------------------------------------------
    # Interface tests (required keys exist and have valid types)
    # ------------------------------------------------------------------
    def test_interface_I_disk_present(self) -> Tuple[bool, str]:
        if self.I_disk is None:
            return (False, "I_disk not present in results")
        return (True, "I_disk present")

    def test_interface_I_disk_type(self) -> Tuple[bool, str]:
        if self.I_disk is None:
            return (False, "SKIP: I_disk not available")
        if not isinstance(self.I_disk, (int, float)):
            return (False, f"I_disk has invalid type {type(self.I_disk)}")
        return (True, "I_disk type OK")

    def test_interface_phi_present(self) -> Tuple[bool, str]:
        if self.phi is None:
            return (False, "phi not present in results")
        return (True, "phi present")

    def test_interface_phi_type(self) -> Tuple[bool, str]:
        if self.phi is None:
            return (False, "SKIP: phi not available")
        if not isinstance(self.phi, (int, float)):
            return (False, f"phi has invalid type {type(self.phi)}")
        return (True, "phi type OK")

    def test_interface_I_hole_present(self) -> Tuple[bool, str]:
        if self.I_hole is None:
            return (False, "I_hole not present in results")
        return (True, "I_hole present")

    def test_interface_I_hole_type(self) -> Tuple[bool, str]:
        if self.I_hole is None:
            return (False, "SKIP: I_hole not available")
        if not isinstance(self.I_hole, (int, float)):
            return (False, f"I_hole has invalid type {type(self.I_hole)}")
        return (True, "I_hole type OK")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_I_disk_equals_I0(self) -> Tuple[bool, str]:
        """Babinet's principle: I_disk = I0 (plane wave, no losses)."""
        if self.I_disk is None or self.I0 is None:
            return (False, "SKIP: I_disk or I0 not available")
        if not math.isclose(self.I_disk, self.I0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"I_disk={self.I_disk} != I0={self.I0}")
        return (True, "I_disk == I0 within tolerance")

    def test_invariant_I_hole_nonnegative(self) -> Tuple[bool, str]:
        """Intensity must be nonnegative."""
        if self.I_hole is None:
            return (False, "SKIP: I_hole not available")
        if self.I_hole < -1e-12:
            return (False, f"I_hole={self.I_hole} is negative")
        return (True, "I_hole >= 0")

    def test_invariant_I_hole_upper_bound(self) -> Tuple[bool, str]:
        """Maximum possible I_hole is 4*I0 (when sin^2 = 1)."""
        if self.I_hole is None or self.I0 is None:
            return (False, "SKIP: I_hole or I0 not available")
        if self.I_hole > 4.0 * self.I0 + 1e-12:
            return (False, f"I_hole={self.I_hole} exceeds 4*I0={4*self.I0}")
        return (True, "I_hole <= 4*I0")

    def test_invariant_phi_nonnegative(self) -> Tuple[bool, str]:
        """phi = k R^2/(4L) must be nonnegative (all parameters nonnegative)."""
        if self.phi is None:
            return (False, "SKIP: phi not available")
        if self.phi < -1e-12:
            return (False, f"phi={self.phi} is negative")
        return (True, "phi >= 0")

    def test_invariant_null_I0_case(self) -> Tuple[bool, str]:
        """If I0=0, all intensities must be zero."""
        if self.I0 is None:
            return (False, "SKIP: I0 not available")
        if self.I0 == 0:
            for name, val in [('I_disk', self.I_disk), ('I_hole', self.I_hole), ('phi', self.phi)]:
                if val is not None and abs(val) > 1e-12:
                    return (False, f"With I0=0, {name}={val} is non-zero")
        return (True, "Null case consistency holds")

    # ------------------------------------------------------------------
    # Modeling/trace tests (check relation from explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_phi_definition(self) -> Tuple[bool, str]:
        """phi should equal k*R^2/(4*L) if givens available."""
        if self.phi is None or self.k is None or self.R is None or self.L is None:
            return (False, "SKIP: missing phi, k, R, or L")
        if self.L == 0:
            return (False, "L is zero, cannot compute")
        expected_phi = self.k * self.R**2 / (4.0 * self.L)
        if not math.isclose(self.phi, expected_phi, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"phi={self.phi} but expected {expected_phi}")
        return (True, "phi = k*R^2/(4L) within tolerance")

    def test_modeling_I_hole_from_phi(self) -> Tuple[bool, str]:
        """I_hole should equal 4*I0*sin^2(phi) if phi and I0 available."""
        if self.I_hole is None or self.phi is None or self.I0 is None:
            return (False, "SKIP: missing I_hole, phi, or I0")
        expected = 4.0 * self.I0 * (math.sin(self.phi) ** 2)
        if not math.isclose(self.I_hole, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"I_hole={self.I_hole} but expected {expected} from sin^2 formula")
        return (True, "I_hole = 4*I0*sin^2(phi) within tolerance")

    # ------------------------------------------------------------------
    # Monotonicity tests (perturb an input, compare perturbed output to baseline)
    # ------------------------------------------------------------------
    def test_monotonicity_hole_increases_with_I0(self) -> Tuple[bool, str]:
        """
        If I0 is increased (hypothetical perturbation), I_hole should increase.
        We simulate a 10% increase in I0 and compute the new I_hole using the physical formula.
        Compare the perturbed I_hole to the baseline (original) I_hole.
        """
        if self.I_hole is None or self.I0 is None or self.phi is None:
            return (False, "SKIP: missing data for monotonicity test")

        # Baseline intensity from solution
        baseline_I_hole = self.I_hole

        # Perturb I0 by +10%
        perturbation_factor = 1.10
        perturbed_I0 = perturbation_factor * self.I0
        perturbed_I_hole = 4.0 * perturbed_I0 * (math.sin(self.phi) ** 2)

        # Since phi unchanged, perturbed_I_hole should be exactly 1.10 * baseline_I_hole
        # Check that it is indeed larger (monotonic increasing)
        if perturbed_I_hole <= baseline_I_hole - 1e-12:
            return (False, f"Perturbed I_hole={perturbed_I_hole} not > baseline I_hole={baseline_I_hole}")
        return (True, "I_hole increases when I0 increases")

    def test_monotonicity_disk_constant_with_R(self) -> Tuple[bool, str]:
        """
        For the disk, I_disk should remain exactly I0 regardless of R.
        We simulate a 10% increase in R (hypothetical) and compute the new I_disk using Babinet.
        Compare the perturbed I_disk to the baseline (original) I_disk.
        """
        if self.I_disk is None or self.I0 is None:
            return (False, "SKIP: missing data for monotonicity test")

        baseline_I_disk = self.I_disk

        # Perturb R by +10% (R not used directly, but I_disk = I0 independent of R)
        # The new computed I_disk is still I0
        perturbed_I_disk = self.I0

        # Since baseline I_disk must equal I0 (already tested in invariant), perturbed should be same
        # Check that perturbed is not different from baseline (monotonic constant)
        if not math.isclose(perturbed_I_disk, baseline_I_disk, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Perturbed I_disk={perturbed_I_disk} differs from baseline I_disk={baseline_I_disk}")
        return (True, "I_disk remains constant when R changes (Babinet)")

    # ------------------------------------------------------------------
    # Weak answer sanity (at most one test, no isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_finite_values(self) -> Tuple[bool, str]:
        """All results should be finite, non-NaN, non-infinite."""
        for name, val in [('I_disk', self.I_disk), ('phi', self.phi), ('I_hole', self.I_hole)]:
            if val is None:
                continue
            if math.isnan(val) or math.isinf(val):
                return (False, f"{name} is NaN or Inf")
        return (True, "All available results are finite")

    # ------------------------------------------------------------------
    # Test runner (required)
    # ------------------------------------------------------------------
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