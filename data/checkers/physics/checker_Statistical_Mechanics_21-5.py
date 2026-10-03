import math
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Auditor for qualitative boiling problem:
    (a) open bottle in boiling water -> no boiling
    (b) corked and isolated -> boiling ceases due to pressure buildup
    (c) cold water restarts boiling by condensing vapor
    """
    def __init__(self, solution_instance: Any):
        # Extract results dict safely
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # Expected keys from pseudocode emit contract (qualitative)
        self.T_bottle = self.results.get('T_bottle', None)           # '100 degC'
        self.boiling_occurs = self.results.get('boiling_occurs', None)  # False
        self.P_total_final = self.results.get('P_total_final', None)    # '> 1 atm'
        self.T_final = self.results.get('T_final', None)                # '< 100 degC'
        self.method = self.results.get('method', None)                  # 'cold water'
        self.reason = self.results.get('reason', None)                  # string

        # Optional: explanation_steps from solution (list of strings or dicts)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        if not isinstance(self.explanation_steps, (list, tuple)):
            self.explanation_steps = None

    # ------------------------------------------------------------------
    # Interface tests – check that required keys exist and have valid types
    # ------------------------------------------------------------------
    def test_interface_T_bottle_present(self) -> Tuple[bool, str]:
        if self.T_bottle is None:
            return (False, "MISSING: results['T_bottle']")
        if not isinstance(self.T_bottle, str):
            return (False, f"TYPE: T_bottle should be str, got {type(self.T_bottle).__name__}")
        return (True, f"T_bottle = '{self.T_bottle}'")

    def test_interface_boiling_occurs_present(self) -> Tuple[bool, str]:
        if self.boiling_occurs is None:
            return (False, "MISSING: results['boiling_occurs']")
        if not isinstance(self.boiling_occurs, bool):
            return (False, f"TYPE: boiling_occurs should be bool, got {type(self.boiling_occurs).__name__}")
        return (True, f"boiling_occurs = {self.boiling_occurs}")

    def test_interface_P_total_final_present(self) -> Tuple[bool, str]:
        if self.P_total_final is None:
            return (False, "MISSING: results['P_total_final']")
        if not isinstance(self.P_total_final, str):
            return (False, f"TYPE: P_total_final should be str, got {type(self.P_total_final).__name__}")
        return (True, f"P_total_final = '{self.P_total_final}'")

    def test_interface_T_final_present(self) -> Tuple[bool, str]:
        if self.T_final is None:
            return (False, "MISSING: results['T_final']")
        if not isinstance(self.T_final, str):
            return (False, f"TYPE: T_final should be str, got {type(self.T_final).__name__}")
        return (True, f"T_final = '{self.T_final}'")

    def test_interface_method_present(self) -> Tuple[bool, str]:
        if self.method is None:
            return (False, "MISSING: results['method']")
        if not isinstance(self.method, str):
            return (False, f"TYPE: method should be str, got {type(self.method).__name__}")
        return (True, f"method = '{self.method}'")

    def test_interface_reason_present(self) -> Tuple[bool, str]:
        if self.reason is None:
            return (False, "MISSING: results['reason']")
        if not isinstance(self.reason, str):
            return (False, f"TYPE: reason should be str, got {type(self.reason).__name__}")
        return (True, f"reason present")

    def test_interface_explanation_steps(self) -> Tuple[bool, str]:
        if self.explanation_steps is None:
            return (False, "MISSING: explanation_steps (list or tuple expected)")
        if not isinstance(self.explanation_steps, (list, tuple)):
            return (False, f"TYPE: explanation_steps should be list/tuple, got {type(self.explanation_steps).__name__}")
        if len(self.explanation_steps) == 0:
            return (False, "explanation_steps is empty")
        return (True, f"explanation_steps has {len(self.explanation_steps)} entries")

    # ------------------------------------------------------------------
    # Invariant tests – theorem-based physical sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_boiling_condition_open(self) -> Tuple[bool, str]:
        """
        Part (a): open bottle, so total pressure = 1 atm.
        Boiling requires P_vap = P_total. At 100°C, P_vap = 1 atm.
        Since heat flow stops, boiling does not occur. This test checks that
        if boiling_occurs is False, the temperature is '100 degC'.
        """
        if self.T_bottle is None or self.boiling_occurs is None:
            return (False, "SKIP: missing T_bottle or boiling_occurs")
        if not self.boiling_occurs:
            expected_temp = '100 degC'
            if self.T_bottle == expected_temp:
                return (True, "boiling_occurs=False and T_bottle='100 degC' consistent")
            else:
                return (False, f"Expected T_bottle='{expected_temp}' when boiling_occurs=False, got '{self.T_bottle}'")
        else:
            # boiling_occurs=True could be for part (b)/(c) where brief boiling may occur
            return (True, "boiling_occurs=True (not part (a)); no further invariant")

    def test_invariant_pressure_increase_raises_boiling_point(self) -> Tuple[bool, str]:
        """
        Part (b): after corking, total pressure > 1 atm => boiling temperature > 100°C.
        Check that final temperature is '< 100 degC' (because it's lower than the new boiling point)
        and that final pressure is '> 1 atm'.
        """
        if self.P_total_final is None or self.T_final is None:
            return (False, "SKIP: missing P_total_final or T_final")
        if '>' in self.P_total_final and '<' in self.T_final:
            return (True, "P_total_final > 1 atm and T_final < 100°C consistent with elevated boiling point")
        else:
            return (False, f"Pressure/temperature relation not as expected: P_total_final='{self.P_total_final}', T_final='{self.T_final}'")

    def test_invariant_cold_water_reduces_pressure(self) -> Tuple[bool, str]:
        """
        Part (c): pouring cold water condenses vapor, reduces pressure -> lowers boiling point.
        Check that method = 'cold water' and reason mentions condensation.
        """
        if self.method is None or self.reason is None:
            return (False, "SKIP: missing method or reason")
        expected_method = 'cold water'
        if self.method.lower() != expected_method:
            return (False, f"method should be '{expected_method}', got '{self.method}'")
        low_reason = self.reason.lower()
        keywords = ['condense', 'pressure', 'reduce', 'lower']
        if any(k in low_reason for k in keywords):
            return (True, "method='cold water' and reason mentions condensation/pressure reduction")
        else:
            return (False, f"reason does not mention condensation or pressure: '{self.reason}'")

    # ------------------------------------------------------------------
    # Modeling quality tests – based on explanation_steps and reasoning
    # ------------------------------------------------------------------
    def test_modeling_part_a_thermal_equilibrium(self) -> Tuple[bool, str]:
        """
        Check that explanation_steps for part (a) mentions thermal equilibrium or no heat flow.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps")
        combined = ' '.join(self.explanation_steps).lower()
        keywords = ['thermal equilibrium', 'heat flow', 'temperature equal', 'no heat']
        if any(k in combined for k in keywords):
            return (True, "Part (a) reasoning includes thermal equilibrium concept")
        else:
            return (False, "Explanation does not mention thermal equilibrium or heat flow cessation")

    def test_modeling_part_b_pressure_and_latent_heat(self) -> Tuple[bool, str]:
        """
        Check that explanation_steps for part (b) mentions pressure increase, boiling point rise, latent heat.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps")
        combined = ' '.join(self.explanation_steps).lower()
        keywords = ['pressure', 'boiling point', 'latent heat', 'cool']
        matches = [k for k in keywords if k in combined]
        if len(matches) >= 2:
            return (True, f"Part (b) reasoning mentions: {matches}")
        else:
            return (False, f"Explanation missing key concepts; found only {len(matches)} of required keywords")

    def test_modeling_part_c_cold_water_reasoning(self) -> Tuple[bool, str]:
        """
        Check that explanation_steps for part (c) mentions condensation and pressure drop.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps")
        combined = ' '.join(self.explanation_steps).lower()
        keywords = ['cold water', 'condense', 'pressure decrease', 'vapor condense']
        if any(k in combined for k in keywords):
            return (True, "Part (c) reasoning includes cold water and condensation effect")
        else:
            return (False, "Explanation does not mention condensation or cold water effect")

    # ------------------------------------------------------------------
    # Monotonicity tests – explicitly perturb an input and compare output
    # ------------------------------------------------------------------
    def test_monotonicity_clausius_clapeyron(self) -> Tuple[bool, str]:
        """
        Clausius-Clapeyron: saturation pressure increases monotonically with temperature.
        Perturb temperature by +1 K and verify P increases.
        """
        try:
            L = 2.26e6   # J/kg, latent heat of vaporization
            R = 461      # J/(kg·K), gas constant for water vapor
            T1 = 373.15  # K (100°C)
            P1 = 101325  # Pa (1 atm)
            T2 = T1 + 1.0  # perturb by +1 K
            # Approximate integrated form: ln(P2/P1) = (L/R)*(1/T1 - 1/T2)
            P2 = P1 * math.exp((L / R) * (1.0 / T1 - 1.0 / T2))
            if P2 > P1:
                return (True, "Perturbation: T +1 K -> P increases (monotonic)")
            else:
                return (False, "Computed P2 <= P1; Clausius-Clapeyron monotonicity violated")
        except Exception as e:
            return (False, f"Calculation error: {e}")

    def test_monotonicity_total_pressure_vs_boiling_point(self) -> Tuple[bool, str]:
        """
        Explicit perturbation: increase total pressure by 10% and compute new boiling point.
        Verify that boiling point increases.
        """
        try:
            L = 2.26e6   # J/kg
            R = 461      # J/(kg·K)
            T_ref = 373.15  # K (100°C)
            P_ref = 101325  # Pa
            # Perturb: increase pressure by 10%
            P_new = P_ref * 1.1
            # Solve Clausius-Clapeyron for new boiling point T_new:
            # ln(P_new/P_ref) = (L/R)*(1/T_ref - 1/T_new)
            # => 1/T_new = 1/T_ref - (R/L)*ln(P_new/P_ref)
            inv_T_new = 1.0 / T_ref - (R / L) * math.log(P_new / P_ref)
            T_new = 1.0 / inv_T_new
            if T_new > T_ref:
                return (True, f"Perturbation: P +10% -> T_new = {T_new:.2f} K > {T_ref:.2f} K (monotonic)")
            else:
                return (False, f"T_new <= T_ref; monotonicity violated")
        except Exception as e:
            return (False, f"Calculation error: {e}")

    # ------------------------------------------------------------------
    # Weak answer sanity test – only one, no math.isclose, no re‑derivation
    # ------------------------------------------------------------------
    def test_answer_sanity_part_a_sign(self) -> Tuple[bool, str]:
        """
        Weak sanity: part (a) answer should be 'No' or equivalent that water will not boil.
        boiling_occurs should be False.
        """
        if self.boiling_occurs is None:
            return (False, "SKIP: boiling_occurs not available")
        if self.boiling_occurs is False:
            return (True, "boiling_occurs is False (correct for part a)")
        else:
            return (False, "boiling_occurs is True, but expected False for part (a)")

    # ------------------------------------------------------------------
    # Master runner - discovers all test_ methods and runs them
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
                passed, msg = method()  # type: Tuple[bool, str]
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)