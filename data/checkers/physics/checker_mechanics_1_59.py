import math
import sympy as sp
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Robust physics checker for the moon escape velocity and thermal velocity problem.
    Tests include interface validation, theorem-based invariants, modeling quality,
    monotonicity, and a weak answer sanity check.
    """

    def __init__(self, solution_instance: Any):
        # ------------------- Private fields for givens (safe extraction) -------------------
        # Earth circumference (m)
        self.C = getattr(solution_instance, 'C', None) or solution_instance.results.get('C', None) if hasattr(solution_instance, 'results') else None
        # Surface gravity of Earth (m/s^2)
        self.g = getattr(solution_instance, 'g', None) or solution_instance.results.get('g', None) if hasattr(solution_instance, 'results') else None
        # Ratio of moon/earth diameters
        self.ratio_diameter = getattr(solution_instance, 'ratio_diameter', None) or solution_instance.results.get('ratio_diameter', None) if hasattr(solution_instance, 'results') else None
        # Ratio of moon/earth masses
        self.ratio_mass = getattr(solution_instance, 'ratio_mass', None) or solution_instance.results.get('ratio_mass', None) if hasattr(solution_instance, 'results') else None
        # Moon temperature in Celsius
        self.T_C = getattr(solution_instance, 'T_C', None) or solution_instance.results.get('T_C', None) if hasattr(solution_instance, 'results') else None
        # O2 molecular mass in amu
        self.m_O2_amu = getattr(solution_instance, 'm_O2_amu', None) or solution_instance.results.get('m_O2_amu', None) if hasattr(solution_instance, 'results') else None
        # Boltzmann constant (J/K)
        self.k = getattr(solution_instance, 'k', None) or solution_instance.results.get('k', None) if hasattr(solution_instance, 'results') else None
        # Atomic mass unit (kg)
        self.amu = getattr(solution_instance, 'amu', None) or solution_instance.results.get('amu', None) if hasattr(solution_instance, 'results') else None

        # ------------------- Results from solution -------------------
        self.v_esc = None
        self.v_th = None
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.v_esc = solution_instance.results.get('v_esc', None)
            self.v_th = solution_instance.results.get('v_th', None)
        else:
            # Fallback to direct attributes
            self.v_esc = getattr(solution_instance, 'v_esc', None)
            self.v_th = getattr(solution_instance, 'v_th', None)

        # ------------------- Derived constants for internal use (computed from safe givens) -------------------
        # We compute these only if the required givens are present, otherwise None.
        self._R_e = None
        if self.C is not None:
            self._R_e = self.C / (2 * math.pi)    # Earth radius in meters

        self._v_esc_expected = None
        if self.g is not None and self._R_e is not None and self.ratio_diameter is not None and self.ratio_mass is not None:
            # v_esc = sqrt(2 * ratio_mass / ratio_diameter * g * R_e)
            self._v_esc_expected = math.sqrt(2.0 * self.ratio_mass / self.ratio_diameter * self.g * self._R_e)

        self._T_K = None
        if self.T_C is not None:
            self._T_K = self.T_C + 273.15

        self._m_O2_kg = None
        if self.m_O2_amu is not None and self.amu is not None:
            self._m_O2_kg = self.m_O2_amu * self.amu

        self._v_th_expected = None
        if self.k is not None and self._T_K is not None and self._m_O2_kg is not None:
            self._v_th_expected = math.sqrt(3.0 * self.k * self._T_K / self._m_O2_kg)

    # ------------------------------------------------------------------
    # Interface tests: check that expected keys exist and are numeric
    # ------------------------------------------------------------------
    def test_interface_results_keys(self) -> Tuple[bool, str]:
        """Check that results contain 'v_esc' and 'v_th' keys."""
        if self.v_esc is None:
            return (False, "FAIL: 'v_esc' missing or None in results")
        if self.v_th is None:
            return (False, "FAIL: 'v_th' missing or None in results")
        return (True, "PASS: Both required result keys present")

    def test_interface_v_esc_type(self) -> Tuple[bool, str]:
        """Check that v_esc is a numeric type (int or float)."""
        if self.v_esc is None:
            return (False, "SKIP: v_esc not available")
        if not isinstance(self.v_esc, (int, float)):
            return (False, f"FAIL: v_esc type is {type(self.v_esc).__name__}, expected int or float")
        return (True, "PASS: v_esc is numeric")

    def test_interface_v_th_type(self) -> Tuple[bool, str]:
        """Check that v_th is a numeric type."""
        if self.v_th is None:
            return (False, "SKIP: v_th not available")
        if not isinstance(self.v_th, (int, float)):
            return (False, f"FAIL: v_th type is {type(self.v_th).__name__}, expected int or float")
        return (True, "PASS: v_th is numeric")

    # ------------------------------------------------------------------
    # Invariant tests: theorem-based physical relationships (lenient isclose)
    # ------------------------------------------------------------------
    def test_invariant_earth_radius(self) -> Tuple[bool, str]:
        """Check that Earth radius derived from circumference is ~6.366e6 m (1% tolerance)."""
        if self._R_e is None:
            return (False, "SKIP: cannot compute Earth radius - C missing")
        # Expected: C/(2π) = 4e7/(2π) ≈ 6.36619772368e6
        expected_R_e = 4e7 / (2 * math.pi)
        if math.isclose(self._R_e, expected_R_e, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "PASS: Earth radius consistent with circumference")
        else:
            return (False, f"FAIL: Earth radius = {self._R_e:.4e}, expected ~{expected_R_e:.4e}")

    def test_invariant_escape_velocity_formula(self) -> Tuple[bool, str]:
        """Check that v_esc matches sqrt(2 * ratio_mass/ratio_diameter * g * R_e) within 2%."""
        if self.v_esc is None or self._v_esc_expected is None:
            return (False, "SKIP: insufficient data to compute expected escape velocity")
        if math.isclose(self.v_esc, self._v_esc_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "PASS: v_esc satisfies the derived formula")
        else:
            return (False, f"FAIL: v_esc = {self.v_esc:.4e}, expected = {self._v_esc_expected:.4e} (tol 2%)")

    def test_invariant_thermal_velocity_formula(self) -> Tuple[bool, str]:
        """Check that v_th matches sqrt(3kT/m_O2) within 2%."""
        if self.v_th is None or self._v_th_expected is None:
            return (False, "SKIP: insufficient data to compute expected thermal velocity")
        if math.isclose(self.v_th, self._v_th_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "PASS: v_th satisfies the formula")
        else:
            return (False, f"FAIL: v_th = {self.v_th:.4e}, expected = {self._v_th_expected:.4e} (tol 2%)")

    # ------------------------------------------------------------------
    # Modeling / trace tests: qualitative relationships and step consistency
    # ------------------------------------------------------------------
    def test_modeling_escape_faster_than_thermal(self) -> Tuple[bool, str]:
        """Escape velocity must be larger than thermal velocity (ballpark factor ~4.4)."""
        if self.v_esc is None or self.v_th is None:
            return (False, "SKIP: missing velocity values")
        if self.v_esc > self.v_th:
            return (True, "PASS: v_esc > v_th (escape dominates)")
        else:
            return (False, f"FAIL: v_esc = {self.v_esc:.4e}, v_th = {self.v_th:.4e}, expected v_esc > v_th")

    def test_modeling_oxygen_speed_insufficient(self) -> Tuple[bool, str]:
        """Thermal speed of O2 should be significantly lower than escape speed."""
        if self.v_esc is None or self.v_th is None:
            return (False, "SKIP: missing velocities")
        # Factor should be around 4.4 if correct; we just check at least factor 2.
        if self.v_esc >= 2.0 * self.v_th:
            return (True, "PASS: v_esc is at least twice v_th")
        else:
            return (False, f"FAIL: v_esc/v_th = {self.v_esc/self.v_th:.3f}, expected at least 2")

    # ------------------------------------------------------------------
    # Monotonicity tests: perturb input and check output trend
    # ------------------------------------------------------------------
    def test_monotonicity_escape_with_gravity(self) -> Tuple[bool, str]:
        """If g increases by 1%, v_esc should also increase (approximately)."""
        if self.g is None or self._R_e is None or self.ratio_diameter is None or self.ratio_mass is None:
            return (False, "SKIP: missing givens to evaluate monotonicity")
        base = math.sqrt(2.0 * self.ratio_mass / self.ratio_diameter * self.g * self._R_e)
        g_perturbed = self.g * 1.01
        perturbed = math.sqrt(2.0 * self.ratio_mass / self.ratio_diameter * g_perturbed * self._R_e)
        if perturbed > base:
            return (True, f"PASS: v_esc increases from {base:.4e} to {perturbed:.4e} when g increases 1%")
        else:
            return (False, f"FAIL: v_esc did not increase: base={base:.4e}, perturbed={perturbed:.4e}")

    def test_monotonicity_thermal_with_temperature(self) -> Tuple[bool, str]:
        """If temperature increases by 5%, v_th should increase."""
        if self.k is None or self._T_K is None or self._m_O2_kg is None:
            return (False, "SKIP: missing data for thermal monotonicity")
        base = math.sqrt(3.0 * self.k * self._T_K / self._m_O2_kg)
        T_perturbed = self._T_K * 1.05
        perturbed = math.sqrt(3.0 * self.k * T_perturbed / self._m_O2_kg)
        if perturbed > base:
            return (True, f"PASS: v_th increases from {base:.4e} to {perturbed:.4e} when T increases 5%")
        else:
            return (False, f"FAIL: v_th did not increase: base={base:.4e}, perturbed={perturbed:.4e}")

    # ------------------------------------------------------------------
    # Weak answer sanity (final answer check – no isclose, no derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_sign_finite(self) -> Tuple[bool, str]:
        """Both velocities must be positive and finite."""
        if self.v_esc is None or self.v_th is None:
            return (False, "SKIP: velocities not available")
        if not (self.v_esc > 0 and math.isfinite(self.v_esc)):
            return (False, f"FAIL: v_esc not positive/finite: {self.v_esc}")
        if not (self.v_th > 0 and math.isfinite(self.v_th)):
            return (False, f"FAIL: v_th not positive/finite: {self.v_th}")
        # Also check order of magnitude (ballpark: escape ~2e3, thermal ~5e2)
        if self.v_esc < 500 or self.v_esc > 5000:
            return (False, f"FAIL: v_esc magnitude unreasonable: {self.v_esc}")
        if self.v_th < 100 or self.v_th > 1500:
            return (False, f"FAIL: v_th magnitude unreasonable: {self.v_th}")
        return (True, "PASS: velocities are positive, finite, and in plausible range")

    # ------------------------------------------------------------------
    # Main runner: discover and execute all test methods
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test_* methods, run them in try/except, return overall pass/fail and logs."""
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

# Example usage (if needed for testing)
if __name__ == "__main__":
    # Dummy solution instance for demonstration
    class DummySolution:
        def __init__(self):
            self.C = 4e7
            self.g = 9.8
            self.ratio_diameter = 0.27
            self.ratio_mass = 0.0123
            self.T_C = 100
            self.m_O2_amu = 32
            self.k = 1.38e-23
            self.amu = 1.66e-27
            # Results
            self.results = {
                'v_esc': 2.38e3,
                'v_th': 538.0
            }

    checker = PhysicsChecker(DummySolution())
    ok, msgs = checker.run_all_tests()
    for m in msgs:
        print(m)
    print(f"\nOverall: {'PASS' if ok else 'FAIL'}")