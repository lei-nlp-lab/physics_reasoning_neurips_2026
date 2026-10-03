import math
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

class PhysicsChecker:
    """
    Auditor for solution to three-part physics question:
    (a) critical angle (b) solar flux (c) electron flow rate
    """
    def __init__(self, solution_instance: Any):
        # Safe extraction from solution_instance.results
        self.results: Optional[Dict[str, Any]] = getattr(solution_instance, 'results', None)
        if self.results is None:
            # Fallback if results attribute missing entirely
            self.results = {}

        # Extract each quantity with None default
        self.theta_c_deg: Optional[float] = self.results.get('theta_c_deg', None)
        self.flux: Optional[float] = self.results.get('flux', None)
        self.N: Optional[float] = self.results.get('N', None)

        # Also extract any givens if stored in solution_instance (e.g., 'givens')
        self.givens: Optional[Dict[str, Any]] = getattr(solution_instance, 'givens', None)
        if self.givens is None:
            self.givens = {}

        self.n_water: Optional[float] = self.givens.get('n_water', None)
        self.n_air: Optional[float] = self.givens.get('n_air', None)
        self.solar_constant: Optional[float] = self.givens.get('solar_constant', None)
        self.P_bulb: Optional[float] = self.givens.get('P_bulb', None)
        self.V_bulb: Optional[float] = self.givens.get('V_bulb', None)
        self.e: Optional[float] = self.givens.get('e', None)

        # Explanation steps (optional)
        self.explanation_steps: Optional[List[str]] = getattr(solution_instance, 'explanation_steps', None)

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_ methods, return (all_pass, logs)."""
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs

    # ----------------------------------------------------------------
    # INTERFACE TESTS
    # ----------------------------------------------------------------
    def test_interface_theta_c_deg_exists(self) -> Tuple[bool, str]:
        """Check that result key 'theta_c_deg' exists and is a number."""
        if self.theta_c_deg is None:
            return False, "Missing 'theta_c_deg' result."
        if not isinstance(self.theta_c_deg, (int, float)):
            return False, f"theta_c_deg should be numeric, got {type(self.theta_c_deg)}."
        return True, f"theta_c_deg = {self.theta_c_deg}"

    def test_interface_flux_exists(self) -> Tuple[bool, str]:
        """Check that result key 'flux' exists and is a number."""
        if self.flux is None:
            return False, "Missing 'flux' result."
        if not isinstance(self.flux, (int, float)):
            return False, f"flux should be numeric, got {type(self.flux)}."
        return True, f"flux = {self.flux}"

    def test_interface_N_exists(self) -> Tuple[bool, str]:
        """Check that result key 'N' exists and is a number."""
        if self.N is None:
            return False, "Missing 'N' result."
        if not isinstance(self.N, (int, float)):
            return False, f"N should be numeric, got {type(self.N)}."
        return True, f"N = {self.N}"

    # ----------------------------------------------------------------
    # INVARIANT TESTS (theorem-based, use math.isclose)
    # ----------------------------------------------------------------
    def test_invariant_critical_angle_snells(self) -> Tuple[bool, str]:
        """
        Theorem: n_water * sin(theta_c) = n_air * sin(90°)
        If n_water and n_air are present, check consistency with result.
        """
        if self.theta_c_deg is None:
            return False, "SKIP: theta_c_deg missing."
        n_w = self.n_water
        n_a = self.n_air
        if n_w is None or n_a is None:
            return False, "SKIP: n_water or n_air not provided as givens."
        theta_c_rad = math.radians(self.theta_c_deg)
        lhs = n_w * math.sin(theta_c_rad)
        rhs = n_a * math.sin(math.pi/2)  # sin(90°) = 1
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"Snell's law satisfied: n_w*sin(θ_c)={lhs:.4f}, n_a*sin(90°)={rhs:.4f}"
        else:
            return False, f"Snell's law violated: n_w*sin(θ_c)={lhs:.4f}, expected {rhs:.4f}"

    def test_invariant_flux_unit_conversion(self) -> Tuple[bool, str]:
        """
        Theorem: 1 W/m² = 1e-4 W/cm², so flux in W/cm² ≈ solar_constant / 10000.
        If solar_constant given, check consistency.
        """
        if self.flux is None:
            return False, "SKIP: flux missing."
        sc = self.solar_constant
        if sc is None:
            return False, "SKIP: solar_constant not provided as given."
        expected_flux = sc / 10000.0
        # Allow 2% relative error due to rounding
        if math.isclose(self.flux, expected_flux, rel_tol=2e-2, abs_tol=1e-2):
            return True, f"Flux {self.flux} W/cm² matches {sc}/10000 = {expected_flux} W/cm²."
        else:
            return False, f"Flux {self.flux} W/cm² does not match expected {expected_flux} W/cm² (from {sc} W/m²)."

    def test_invariant_electron_flux_formula(self) -> Tuple[bool, str]:
        """
        Theorem: N = I / e, where I = P / V.
        If P_bulb and V_bulb are given, check N against derived I/e.
        """
        if self.N is None:
            return False, "SKIP: N missing."
        P = self.P_bulb
        V = self.V_bulb
        e_charge = self.e
        if P is None or V is None or e_charge is None:
            return False, "SKIP: missing bulb parameters (P, V, e)."
        I = P / V
        expected_N = I / e_charge
        if math.isclose(self.N, expected_N, rel_tol=1e-1, abs_tol=1e18):  # order-of-magnitude tolerance
            return True, f"N = {self.N:.2e} agrees with I/e = {expected_N:.2e}."
        else:
            return False, f"N = {self.N:.2e} does not match I/e = {expected_N:.2e}."

    # ----------------------------------------------------------------
    # MODELING / TRACE TESTS (based on explanation steps)
    # ----------------------------------------------------------------
    def test_modeling_uses_correct_indices(self) -> Tuple[bool, str]:
        """Check that explanation mentions n_water ≈ 4/3 or 1.33, n_air ≈ 1."""
        if not self.explanation_steps:
            return False, "SKIP: no explanation steps available."
        # Search through all steps for key indices
        combined = " ".join(self.explanation_steps).lower()
        has_n_water = "1.33" in combined or "4/3" in combined
        has_n_air = "1.0" in combined or "1 " in combined  # rough
        if has_n_water and has_n_air:
            return True, "Explanation references n_water ≈ 1.33 and n_air ≈ 1."
        else:
            return False, "Explanation missing reference to appropriate refractive indices."

    def test_modeling_flux_conversion_method(self) -> Tuple[bool, str]:
        """Check that explanation converts m² to cm² correctly (factor 10^4)."""
        if not self.explanation_steps:
            return False, "SKIP: no explanation steps."
        combined = " ".join(self.explanation_steps).lower()
        if "10000" in combined or "10^4" in combined or "10,000" in combined:
            return True, "Explanation uses correct conversion factor 1 m² = 10^4 cm²."
        else:
            return False, "Explanation does not mention the area conversion factor."

    def test_modeling_electron_count_from_current(self) -> Tuple[bool, str]:
        """Check that explanation derives N = I/e with I from P/V."""
        if not self.explanation_steps:
            return False, "SKIP: no explanation steps."
        combined = " ".join(self.explanation_steps).lower()
        has_power_voltage = "w" in combined and "v" in combined
        has_electron_charge = "1.6e-19" in combined or "e" in combined
        if has_power_voltage and has_electron_charge:
            return True, "Explanation uses power/voltage for current and electron charge."
        else:
            return False, "Explanation missing steps linking power, voltage, current, and charge."

    # ----------------------------------------------------------------
    # MONOTONICITY TEST
    # ----------------------------------------------------------------
    def test_monotonicity_critical_angle_decreases_with_n_water(self) -> Tuple[bool, str]:
        """
        Property: Increasing n_water decreases θ_c (since θ_c = arcsin(1/n_water)).
        If both n_water and theta_c_deg are present, imagine a small increase in n_water.
        """
        if self.theta_c_deg is None or self.n_water is None:
            return False, "SKIP: missing theta_c_deg or n_water."
        # Baseline n_water and theta
        n0 = self.n_water
        theta0_rad = math.radians(self.theta_c_deg)
        # Perturb: increase n_water by 5%
        n1 = n0 * 1.05
        theta1_rad = math.asin(1.0 / n1)  # assume n_air=1
        if theta1_rad < theta0_rad:
            return True, f"n_water increased from {n0:.2f} to {n1:.2f}: θ_c decreased (monotonicity holds)."
        else:
            return False, "Monotonicity violated: increasing n_water should decrease θ_c."

    # ----------------------------------------------------------------
    # WEAK SANITY TEST (must not re-derive final answer, no math.isclose)
    # ----------------------------------------------------------------
    def test_answer_sanity_plausibility(self) -> Tuple[bool, str]:
        """Weak final-check: all results positive and within plausible orders."""
        checks_ok = True
        msgs = []
        if self.theta_c_deg is not None:
            if not (0 < self.theta_c_deg < 90):
                checks_ok = False
                msgs.append(f"theta_c_deg={self.theta_c_deg} not in (0,90).")
            elif 40 < self.theta_c_deg < 60:  # plausible range around 50°
                msgs.append("theta_c_deg plausible ~50°.")
            else:
                msgs.append(f"theta_c_deg={self.theta_c_deg} not near 50° but acceptable.")
        if self.flux is not None:
            if not (self.flux > 0):
                checks_ok = False
                msgs.append(f"flux={self.flux} not positive.")
            elif 1e-2 < self.flux < 1.0:  # plausible order of magnitude ~0.1 W/cm²
                msgs.append(f"flux={self.flux} plausible (~0.1 W/cm²).")
            else:
                msgs.append(f"flux={self.flux} not typical but remains positive.")
        if self.N is not None:
            if not (self.N > 0):
                checks_ok = False
                msgs.append(f"N={self.N} not positive.")
            elif 1e17 < self.N < 1e20:  # plausible around 10^19
                msgs.append(f"N={self.N:.2e} plausible (~10¹⁹).")
            else:
                msgs.append(f"N={self.N:.2e} not typical but positive.")
        if not checks_ok:
            return False, "; ".join(msgs)
        return True, "All results positive and within plausible orders: " + "; ".join(msgs)