import re
import math
import traceback

class PhysicsChecker:
    """
    Audits temperature measurement instrument descriptions based on:
    - constant-volume gas thermometer: ideal gas law at constant volume
    - thermocouple: Seebeck effect (thermoelectric motive force)
    - thermistor: NTC resistance vs temperature
    """

    def __init__(self, solution_instance):
        # Extract results safely, default to None
        results = getattr(solution_instance, 'results', {})
        self.q1 = results.get('q1', None)
        self.q2 = results.get('q2', None)
        self.q3 = results.get('q3', None)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    @staticmethod
    def _check_string_contains(text, keyword):
        """Case-insensitive substring check."""
        if text is None:
            return False
        return keyword.lower() in text.lower()

    # ------------------------------------------------------------------
    # Interface checks: each key must exist and be a string
    # ------------------------------------------------------------------
    def test_interface_q1_exists_and_string(self):
        if self.q1 is None:
            return (False, "q1 missing from results")
        if not isinstance(self.q1, str):
            return (False, f"q1 is not a string, got {type(self.q1)}")
        return (True, "q1 present and is a string")

    def test_interface_q2_exists_and_string(self):
        if self.q2 is None:
            return (False, "q2 missing from results")
        if not isinstance(self.q2, str):
            return (False, f"q2 is not a string, got {type(self.q2)}")
        return (True, "q2 present and is a string")

    def test_interface_q3_exists_and_string(self):
        if self.q3 is None:
            return (False, "q3 missing from results")
        if not isinstance(self.q3, str):
            return (False, f"q3 is not a string, got {type(self.q3)}")
        return (True, "q3 present and is a string")

    # ------------------------------------------------------------------
    # Invariant / principle checks (theorem-based)
    # ------------------------------------------------------------------
    def test_invariant_gas_pressure_volume(self):
        """Gas thermometer principle: pressure depends on temperature at constant volume."""
        if self.q1 is None:
            return (False, "SKIP: q1 not available")
        has_pressure = self._check_string_contains(self.q1, "pressure")
        has_volume = self._check_string_contains(self.q1, "volume")
        has_constant_volume = self._check_string_contains(self.q1, "constant volume")
        if has_pressure and has_volume and has_constant_volume:
            return (True, "q1 correctly relates pressure and volume at constant volume")
        else:
            return (False, f"q1 does not adequately state constant-volume gas principle. "
                           f"Contains pressure={has_pressure}, volume={has_volume}, constant volume={has_constant_volume}")

    def test_invariant_gas_ideal_gas_law(self):
        """Gas thermometer principle should mention ideal gas law or ideal gas thermometer."""
        if self.q1 is None:
            return (False, "SKIP: q1 not available")
        if self._check_string_contains(self.q1, "ideal gas") or \
           self._check_string_contains(self.q1, "ideal gas law"):
            return (True, "q1 invokes ideal gas concept")
        else:
            return (False, "q1 should mention ideal gas law or ideal gas thermometer")

    def test_invariant_thermocouple_principle(self):
        """Thermocouple principle: thermoelectric motive force (Seebeck effect)."""
        if self.q2 is None:
            return (False, "SKIP: q2 not available")
        has_motive_force = self._check_string_contains(self.q2, "thermoelectric motive force") or \
                           self._check_string_contains(self.q2, "motive force")
        has_seebeck = self._check_string_contains(self.q2, "seebeck")
        if has_motive_force or has_seebeck:
            return (True, "q2 correctly identifies thermoelectric or Seebeck principle")
        else:
            return (False, "q2 does not mention thermoelectric motive force or Seebeck effect")

    def test_invariant_thermistor_principle(self):
        """Thermistor principle: resistance changes with temperature."""
        if self.q3 is None:
            return (False, "SKIP: q3 not available")
        has_resistance = self._check_string_contains(self.q3, "resistance")
        has_temperature = self._check_string_contains(self.q3, "temperature")
        if has_resistance and has_temperature:
            return (True, "q3 links resistance to temperature")
        else:
            return (False, f"q3 does not connect resistance and temperature. "
                           f"Contains resistance={has_resistance}, temperature={has_temperature}")

    # ------------------------------------------------------------------
    # Modeling quality checks (usefulness statements)
    # ------------------------------------------------------------------
    def test_modeling_usefulness_gas(self):
        """Usefulness of gas thermometer: ideal gas thermometer or primary standard."""
        if self.q1 is None:
            return (False, "SKIP: q1 not available")
        if self._check_string_contains(self.q1, "ideal gas thermometer") or \
           self._check_string_contains(self.q1, "primary standard"):
            return (True, "q1 usefulness statement matches expected")
        else:
            return (False, "Gas usefulness should include 'ideal gas thermometer' or 'primary standard'")

    def test_modeling_usefulness_thermocouple(self):
        """Usefulness of thermocouple: wide measurement range."""
        if self.q2 is None:
            return (False, "SKIP: q2 not available")
        if self._check_string_contains(self.q2, "wide measurement range") or \
           self._check_string_contains(self.q2, "wide range"):
            return (True, "q2 usefulness mentions wide range")
        else:
            return (False, "Thermocouple usefulness should mention wide measurement range")

    def test_modeling_usefulness_thermistor(self):
        """Usefulness of thermistor: high precision as standard thermometer."""
        if self.q3 is None:
            return (False, "SKIP: q3 not available")
        has_high_precision = self._check_string_contains(self.q3, "high precision")
        has_standard = self._check_string_contains(self.q3, "standard thermometer")
        if has_high_precision and has_standard:
            return (True, "q3 usefulness includes high precision and standard thermometer")
        elif has_high_precision or has_standard:
            return (True, "q3 usefulness partially correct (one of the two key phrases)")
        else:
            return (False, "Thermistor usefulness should mention high precision and standard thermometer")

    # ------------------------------------------------------------------
    # Monotonicity test (perturb input, recompute output)
    # ------------------------------------------------------------------
    def test_monotonicity_thermocouple_voltage_increases_with_deltaT(self):
        """
        Perturb the temperature difference ΔT and verify that the Seebeck voltage
        (ΔV = S * ΔT) increases monotonically (S > 0).
        This test uses a dummy Seebeck coefficient; it is independent of solution_instance.
        """
        S = 0.001  # V/°C (dummy, positive)
        deltaT_baseline = 1.0  # °C
        deltaT_perturbed = 1.1  # +10% perturbation
        V_baseline = S * deltaT_baseline
        V_perturbed = S * deltaT_perturbed
        if V_perturbed > V_baseline:
            return (True, "Seebeck voltage increases monotonically with ΔT")
        else:
            return (False, "Unexpected: perturbed voltage not greater than baseline")

    # ------------------------------------------------------------------
    # Weak final‑answer sanity (one method only)
    # ------------------------------------------------------------------
    def test_answer_sanity_all_non_empty(self):
        """Check all three answers are non‑empty strings of reasonable length."""
        for key, val in [("q1", self.q1), ("q2", self.q2), ("q3", self.q3)]:
            if val is None:
                return (False, f"{key} is None")
            if not isinstance(val, str):
                return (False, f"{key} is not a string")
            if len(val) < 10:
                return (False, f"{key} too short ({len(val)} chars)")
            if len(val) > 2000:
                return (False, f"{key} suspiciously long ({len(val)} chars)")
        return (True, "All three answers present and have plausible length")

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
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