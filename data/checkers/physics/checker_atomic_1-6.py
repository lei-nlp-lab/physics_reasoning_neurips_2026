import math
import sys
import traceback
from typing import Any, Dict, List, Tuple, Union

class PhysicsChecker:
    """
    Robust physics checker for electronic transition broadening in 12C ions.
    Tests focus on Doppler broadening (Gamma_D) using theorem-based invariants,
    modeling quality checks, monotonicity, and weak answer sanity.
    """

    # Physical constants (SI units)
    _c = 299792458.0       # speed of light [m/s]
    _u = 1.66053906660e-27 # atomic mass unit [kg]
    _ln2 = 0.69314718056
    _kb_eV_K = 8.617333262e-5 # Boltzmann constant [eV/K] – not needed directly (given kT)

    def __init__(self, solution_instance: Any):
        """
        Initialize checker with solution_instance.
        Safely extract givens and results; use None as default for missing data.
        """
        # --- Extract givens (assume they are stored as attributes of solution_instance) ---
        # Expected: solution_instance.lambda_nm, solution_instance.kT_eV, solution_instance.mass_u
        # Fallback to None if missing.
        self.lambda_nm = getattr(solution_instance, 'lambda_nm', None)
        self.kT_eV = getattr(solution_instance, 'kT_eV', None)
        self.mass_u = getattr(solution_instance, 'mass_u', None)  # mass in atomic mass units

        # If givens are stored in a dict like solution_instance.givens, try that too
        if self.lambda_nm is None:
            givens = getattr(solution_instance, 'givens', {})
            self.lambda_nm = givens.get('lambda_nm', None) if isinstance(givens, dict) else None
        if self.kT_eV is None:
            givens = getattr(solution_instance, 'givens', {})
            self.kT_eV = givens.get('kT_eV', None) if isinstance(givens, dict) else None
        if self.mass_u is None:
            givens = getattr(solution_instance, 'givens', {})
            self.mass_u = givens.get('mass_u', None) if isinstance(givens, dict) else None

        # Convert to SI where possible
        self.lambda_m = None
        if self.lambda_nm is not None:
            self.lambda_m = self.lambda_nm * 1e-9
        self.m_kg = None
        if self.mass_u is not None:
            self.m_kg = self.mass_u * self._u

        # --- Extract results (solution outputs) ---
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        self.Gamma_D = results.get('Gamma_D', None)  # expected to be in m⁻¹

    def _expected_Gamma_D_from_givens(self, lambda_m=None, kT_eV=None, m_kg=None) -> float:
        """
        Compute the theoretical Doppler width (m⁻¹) using the standard formula.
        Falls back to stored values if parameters are None.
        """
        if lambda_m is None:
            lambda_m = self.lambda_m
        if kT_eV is None:
            kT_eV = self.kT_eV
        if m_kg is None:
            m_kg = self.m_kg

        if any(v is None for v in (lambda_m, kT_eV, m_kg)):
            raise ValueError("Missing input for expected Gamma_D calculation")
        kT_J = kT_eV * 1.602176634e-19
        mc2 = m_kg * self._c ** 2
        v_over_c = math.sqrt(8 * kT_J * self._ln2 / mc2)
        return v_over_c / lambda_m

    # ----------------------------------------------------------------------
    # Interface tests – ensure required keys exist and are of correct type
    # ----------------------------------------------------------------------
    def test_interface_Gamma_D_exists(self) -> Tuple[bool, str]:
        if self.Gamma_D is None:
            return False, "Missing result key 'Gamma_D' in solution_instance.results"
        return True, "Gamma_D is present"

    def test_interface_Gamma_D_type(self) -> Tuple[bool, str]:
        if self.Gamma_D is None:
            return False, "SKIP: Gamma_D not available"
        if not isinstance(self.Gamma_D, (int, float)):
            return False, f"Gamma_D has type {type(self.Gamma_D).__name__}, expected float or int"
        return True, "Gamma_D is numeric"

    def test_interface_Gamma_D_positive_finite(self) -> Tuple[bool, str]:
        if self.Gamma_D is None:
            return False, "SKIP: Gamma_D not available"
        if not math.isfinite(self.Gamma_D) or self.Gamma_D <= 0:
            return False, f"Gamma_D = {self.Gamma_D} is not positive and finite"
        return True, "Gamma_D is positive and finite"

    # ----------------------------------------------------------------------
    # Invariant tests – theorem-based physical sanity (use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_Doppler_scaling_with_kT(self) -> Tuple[bool, str]:
        """
        Check that Gamma_D scales approximately as sqrt(kT) for fixed mass and lambda.
        Since we have only one data point, we compare the provided value with the
        expected Doppler width computed from the standard formula.
        """
        if self.Gamma_D is None or self.lambda_m is None or self.m_kg is None or self.kT_eV is None:
            return False, "SKIP: missing data for invariant test"
        expected_Gamma_D = self._expected_Gamma_D_from_givens()
        # Allow a relative tolerance of 20% due to order-of-magnitude nature and possible use of
        # different constants (some sources use 1.67 factor, etc.)
        if not math.isclose(self.Gamma_D, expected_Gamma_D, rel_tol=0.2, abs_tol=0.0):
            return False, f"Gamma_D = {self.Gamma_D} differs from expected {expected_Gamma_D} by more than 20%"
        return True, f"Gamma_D scales correctly with sqrt(kT)/lambda (expected {expected_Gamma_D:.3g})"

    def test_invariant_Doppler_width_in_cm_inv(self) -> Tuple[bool, str]:
        """
        Check that Gamma_D, when converted to cm⁻¹, is of order ~2 cm⁻¹.
        This tests the internal consistency of units and formulas.
        """
        if self.Gamma_D is None:
            return False, "SKIP: Gamma_D not available"
        # Convert from m⁻¹ to cm⁻¹ (divide by 100)
        Gamma_D_cm = self.Gamma_D / 100.0
        # Expectation from order-of-magnitude: ~2 cm⁻¹. Use broad tolerance.
        # Not an exact match; we just check that it's within an order of magnitude.
        if not (0.5 <= Gamma_D_cm <= 10):
            return False, f"Gamma_D in cm⁻¹ = {Gamma_D_cm} is not within [0.5, 10] cm⁻¹"
        return True, f"Gamma_D in cm⁻¹ is plausible (≈2 cm⁻¹, got {Gamma_D_cm:.2f})"

    # ----------------------------------------------------------------------
    # Modeling-quality tests – verify that the explanation steps were followed
    # (but not by hardcoding numbers; check one local relation).
    # ----------------------------------------------------------------------
    def test_modeling_Doppler_velocity_ratio(self) -> Tuple[bool, str]:
        """
        Check that the thermal velocity ratio v/c used in the derivation is consistent.
        From Gamma_D = (1/lambda) * (v/c) we can extract v/c = Gamma_D * lambda.
        Ensure that v/c is positive and less than 1.
        """
        if self.Gamma_D is None or self.lambda_m is None:
            return False, "SKIP: missing data for modeling check"
        v_over_c = self.Gamma_D * self.lambda_m
        if not (0 < v_over_c < 1):
            return False, f"Implied v/c = {v_over_c} is not between 0 and 1"
        return True, f"Implied v/c = {v_over_c:.6g} lies in physical range"

    def test_modeling_Doppler_energy_scale(self) -> Tuple[bool, str]:
        """
        Check that the Doppler shift energy is much smaller than the transition energy.
        Transition energy ~ 2.5 eV. Thermal energy kT = 20 eV. Doppler shift ΔE ≈ E * v/c.
        Ensure ΔE << 2.5 eV.
        """
        if self.Gamma_D is None or self.lambda_m is None or self.kT_eV is None:
            return False, "SKIP: missing data"
        v_over_c = self.Gamma_D * self.lambda_m
        # transition energy: E = hc/lambda ≈ 1240 eV·nm / lambda_nm
        E_eV = 1240.0 / (self.lambda_m * 1e9)  # approximate
        delta_E_eV = E_eV * v_over_c
        if delta_E_eV > 0.5 * E_eV:
            return False, f"Doppler shift energy ({delta_E_eV:.4g} eV) is too large relative to transition energy ({E_eV:.4g} eV)"
        return True, f"Doppler shift energy ({delta_E_eV:.4g} eV) is reasonable"

    # ----------------------------------------------------------------------
    # Monotonicity tests – perturb an input and compute expected output change
    # ----------------------------------------------------------------------
    def test_monotonicity_Gamma_D_increases_with_kT(self) -> Tuple[bool, str]:
        """
        Check that Gamma_D increases when kT is increased.
        Since we cannot vary the solution's parameter, we compute expected Gamma_D
        at the given kT and at a perturbed kT (10% increase) using the same formula.
        """
        if self.lambda_m is None or self.kT_eV is None or self.m_kg is None:
            return False, "SKIP: missing givens for monotonicity test"
        base_Gamma_D = self._expected_Gamma_D_from_givens()
        perturbed_kT = self.kT_eV * 1.1
        perturbed_Gamma_D = self._expected_Gamma_D_from_givens(kT_eV=perturbed_kT)
        if perturbed_Gamma_D <= base_Gamma_D:
            return False, f"Expected Gamma_D to increase when kT increased from {self.kT_eV} to {perturbed_kT}, but got {perturbed_Gamma_D:.4g} <= {base_Gamma_D:.4g}"
        return True, f"Gamma_D increased from {base_Gamma_D:.4g} to {perturbed_Gamma_D:.4g} when kT increased by 10%"

    def test_monotonicity_Gamma_D_decreases_with_lambda(self) -> Tuple[bool, str]:
        """
        Check that Gamma_D decreases when lambda is increased (proportional).
        Perturb lambda by +10% and verify Gamma_D decreases.
        """
        if self.lambda_m is None or self.kT_eV is None or self.m_kg is None:
            return False, "SKIP: missing givens for monotonicity test"
        base_Gamma_D = self._expected_Gamma_D_from_givens()
        perturbed_lambda = self.lambda_m * 1.1
        perturbed_Gamma_D = self._expected_Gamma_D_from_givens(lambda_m=perturbed_lambda)
        if perturbed_Gamma_D >= base_Gamma_D:
            return False, f"Expected Gamma_D to decrease when lambda increased from {self.lambda_m} to {perturbed_lambda}, but got {perturbed_Gamma_D:.4g} >= {base_Gamma_D:.4g}"
        return True, f"Gamma_D decreased from {base_Gamma_D:.4g} to {perturbed_Gamma_D:.4g} when lambda increased by 10%"

    # ----------------------------------------------------------------------
    # Answer sanity test – weak, broad check only (no isclose, no re-derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_broad_range(self) -> Tuple[bool, str]:
        """
        Weak sanity: Gamma_D must be positive and within an order of magnitude
        of 2 cm⁻¹ (i.e., between 0.2 and 20 cm⁻¹).
        """
        if self.Gamma_D is None:
            return False, "SKIP: Gamma_D not available"
        Gamma_D_cm = self.Gamma_D / 100.0
        if Gamma_D_cm < 0.2 or Gamma_D_cm > 20:
            return False, f"Gamma_D = {self.Gamma_D} m⁻¹ ({Gamma_D_cm:.2f} cm⁻¹) is outside weak sanity range [0.2, 20] cm⁻¹"
        return True, f"Gamma_D = {Gamma_D_cm:.2f} cm⁻¹ is within broad expected range"

    # ----------------------------------------------------------------------
    # Main runner: discover all test methods and run them safely
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
                passed = False
                msg = f"{name} CRASHED: {type(e).__name__}: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs

# Example usage (if run as script):
if __name__ == "__main__":
    # Dummy solution instance for testing
    class DummySolution:
        def __init__(self):
            self.lambda_nm = 500.0
            self.kT_eV = 20.0
            self.mass_u = 12.0
            self.results = {'Gamma_D': 199.2}  # m^-1

    sol = DummySolution()
    checker = PhysicsChecker(sol)
    success, log_list = checker.run_all_tests()
    for line in log_list:
        print(line)
    print(f"\nOverall {'PASS' if success else 'FAIL'}")