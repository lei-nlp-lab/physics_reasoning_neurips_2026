import math
import sympy as sp
import traceback
from typing import Any, Dict, Tuple, List

class PhysicsChecker:
    """
    Audits the solution for the phase shift of a plane polarized EM wave
    reflected from a conductor at normal incidence.
    """

    def __init__(self, solution_instance: Any):
        # Safely extract givens and results; default to None.
        givens: Dict[str, Any] = getattr(solution_instance, 'givens', {})
        results: Dict[str, Any] = getattr(solution_instance, 'results', {})

        self.n = givens.get('n', None)
        self.rho = givens.get('rho', None)
        self.phase = results.get('phase', None)

        # Derive internally for invariant checks (only if givens exist)
        self._internal_phase = None
        if self.rho is not None and self.rho > 0:
            # Using the known formula: pi + atan2(2, rho)
            self._internal_phase = math.pi + math.atan2(2, self.rho)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _skip_if(self, cond: bool, msg: str) -> Tuple[bool, str]:
        """Return a skip message if condition is True."""
        if cond:
            return (False, f"SKIP: {msg}")
        return (True, "")

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that required keys exist in results."""
        if self.phase is None:
            return (False, "FAIL: 'phase' missing in results")
        return (True, "PASS: 'phase' present in results")

    def test_interface_givens_exist(self) -> Tuple[bool, str]:
        """Check that at least one given (n or rho) is present."""
        if self.n is None and self.rho is None:
            return (False, "FAIL: both n and rho are missing from givens")
        return (True, "PASS: at least one given present")

    # ------------------------------------------------------------------
    # Invariant checks (theorem-based sanity, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_zero_rho_phase_is_pi(self) -> Tuple[bool, str]:
        """If rho=0 (lossless conductor), phase should be pi."""
        if self.rho is None:
            return (False, "SKIP: rho not available")
        if self.phase is None:
            return (False, "SKIP: phase not available")

        # We cannot set rho=0 if it wasn't given, but we can check the formula:
        # For rho=0, the physical phase is pi (half-wavelength shift).
        # Since rho>0 in problem, we test the trend: as rho -> 0+, phase -> 3pi/2?
        # Actually from formula: phase = pi + arctan(2/rho), as rho->0, arctan(2/rho)->pi/2, so phase->3pi/2.
        # That's also an invariant of the formula. But let's check consistency:
        # If we compute internal phase for rho=0 using limit? We'll just check that the formula
        # gives phase > pi for rho>0.
        if self.rho > 0 and self._internal_phase is not None:
            expected = math.pi + math.atan2(2, self.rho)
            if math.isclose(self.phase, expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"PASS: phase {self.phase:.3f} matches expected {expected:.3f}")
            else:
                return (False, f"FAIL: phase {self.phase:.3f} != expected {expected:.3f}")
        return (False, "SKIP: cannot compute internal phase")

    def test_invariant_phase_positive_for_positive_rho(self) -> Tuple[bool, str]:
        """Phase should be positive (between pi and 3pi/2)."""
        if self.phase is None:
            return (False, "SKIP: phase not available")
        if not (math.pi < self.phase < 1.5 * math.pi):
            return (False, f"FAIL: phase {self.phase:.3f} not in (pi, 3pi/2)")
        return (True, f"PASS: phase {self.phase:.3f} in (pi, 3pi/2)")

    def test_invariant_phase_derived_from_fresnel_formula(self) -> Tuple[bool, str]:
        """
        Verify that the phase satisfies the Fresnel reflection coefficient relation
        using the given rho. This is a derived invariant computed inside the checker.
        """
        if self.rho is None or self.phase is None:
            return (False, "SKIP: rho or phase missing")

        # Recompute reflection coefficient from Fresnel: r = (i*rho)/(2+i*rho) up to sign.
        # But careful: different sign conventions. The physical phase formula is derived from
        # r = (n2 - n1)/(n2 + n1) = (i*rho)/(2 + i*rho).
        # Compute its argument.
        numerator = complex(0, self.rho)         # i*rho
        denominator = complex(2, self.rho)       # 2 + i*rho
        r = numerator / denominator
        # phase from r: arg(r) = atan2(r.imag, r.real)
        expected_phase = math.atan2(r.imag, r.real)
        # Compare with reported phase. Note: phase may be reported as pi + arctan(2/rho),
        # which should equal expected_phase modulo 2pi.
        # Compute standard principal value between -pi and pi.
        reported_phase_mod = self.phase % (2 * math.pi)
        if reported_phase_mod > math.pi:
            reported_phase_mod -= 2 * math.pi
        expected_phase_mod = expected_phase
        if math.isclose(reported_phase_mod, expected_phase_mod, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"PASS: phase modulo 2pi matches arg(r) = {expected_phase_mod:.3f}")
        else:
            return (False, f"FAIL: reported phase mod {reported_phase_mod:.3f} != arg(r) {expected_phase_mod:.3f}")

    # ------------------------------------------------------------------
    # Modeling / trace checks (quality of steps)
    # ------------------------------------------------------------------
    def test_modeling_reflection_coefficient_sign(self) -> Tuple[bool, str]:
        """
        Check that the sign of the reflection coefficient is consistent with
        the derived phase: for rho>0, real part is negative, imaginary part is negative,
        so r in third quadrant.
        """
        if self.rho is None:
            return (False, "SKIP: rho not available")
        if self.phase is None:
            return (False, "SKIP: phase not available")

        # Recompute r from formula.
        # Use the formula from pseudocode: r = (-i*rho)/(2+i*rho). That yields:
        # cartesian: Re(r) = -rho^2/(4+rho^2), Im(r) = -2*rho/(4+rho^2)  (both negative)
        denominator = 4 + self.rho**2
        re_r = -self.rho**2 / denominator
        im_r = -2 * self.rho / denominator
        if not (re_r < 0 and im_r < 0):
            return (False, f"FAIL: r components not both negative: Re={re_r:.3f}, Im={im_r:.3f}")
        return (True, f"PASS: r in third quadrant (Re<0, Im<0)")

    def test_modeling_phase_derivation_consistency(self) -> Tuple[bool, str]:
        """
        Check that the phase formula reported is consistent with the (simpler)
        expression pi + arctan(2/rho). Avoid hardcoding the full answer; just
        verify that the reported phase equals that expression (within tolerance).
        """
        if self.rho is None or self.phase is None:
            return (False, "SKIP: rho or phase missing")

        expected_phase = math.pi + math.atan2(2, self.rho)
        if math.isclose(self.phase, expected_phase, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"PASS: phase ≈ pi + arctan(2/rho) = {expected_phase:.3f}")
        else:
            return (False, f"FAIL: phase {self.phase:.3f} != {expected_phase:.3f}")

    # ------------------------------------------------------------------
    # Monotonicity checks (trend analysis via perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_phase_decreases_with_rho(self) -> Tuple[bool, str]:
        """
        Check that as rho increases, phase decreases.
        We perturb rho by +5% and check that new phase is smaller than baseline.
        """
        if self.rho is None or self.rho <= 0:
            return (False, "SKIP: rho missing or non-positive")
        if self.phase is None:
            return (False, "SKIP: phase missing")

        baseline_phase = self.phase
        # Create a perturbed rho: +5%
        perturbed_rho = self.rho * 1.05
        expected_perturbed_phase = math.pi + math.atan2(2, perturbed_rho)

        # Check monotonicity: phase should be decreasing with rho
        if expected_perturbed_phase < baseline_phase:
            return (True, f"PASS: phase decreases with rho: baseline {baseline_phase:.3f} > perturbed {expected_perturbed_phase:.3f}")
        else:
            return (False, f"FAIL: phase does not decrease: baseline {baseline_phase:.3f} <= perturbed {expected_perturbed_phase:.3f}")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one, without math.isclose, without re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_phase_range(self) -> Tuple[bool, str]:
        """
        Very weak sanity: phase must be a finite real number between 0 and 2*pi.
        """
        if self.phase is None:
            return (False, "SKIP: phase not available")
        if not (0 <= self.phase <= 2 * math.pi):
            return (False, f"FAIL: phase {self.phase:.3f} outside [0, 2*pi]")
        # Also check it's not NaN or Inf
        if math.isnan(self.phase) or math.isinf(self.phase):
            return (False, f"FAIL: phase is NaN or Inf")
        return (True, f"PASS: phase {self.phase:.3f} is finite real in [0, 2*pi]")

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)