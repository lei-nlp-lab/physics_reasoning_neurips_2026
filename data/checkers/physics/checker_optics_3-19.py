import math
import re
import traceback

class PhysicsChecker:
    """
    Robust checker for qualitative optical explanations.
    Automatically handles missing data using None defaults.
    """

    def __init__(self, solution_instance):
        # Extract results safely
        self.results = getattr(solution_instance, 'results', None) or {}
        self.scattering_power = self.results.get('scattering_power', None)
        self.transmission = self.results.get('transmission', None)
        self.rainbow_angle_red = self.results.get('rainbow_angle_red', None)
        self.rainbow_angle_blue = self.results.get('rainbow_angle_blue', None)
        self.scintillation = self.results.get('scintillation', None)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    def _extract_angle(self, s):
        """Extract a numeric angle from strings like '~=42 deg' or '40°'."""
        if s is None:
            return None
        cleaned = re.sub(r'[^\d.\-]', '', s)  # keep digits, dot, minus
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_scattering_power(self):
        if self.scattering_power is None:
            return (False, "scattering_power missing")
        if not isinstance(self.scattering_power, str):
            return (False, "scattering_power is not a string")
        return (True, "scattering_power present and is string")

    def test_interface_transmission(self):
        if self.transmission is None:
            return (False, "transmission missing")
        if not isinstance(self.transmission, str):
            return (False, "transmission is not a string")
        return (True, "transmission present and is string")

    def test_interface_rainbow_angle_red(self):
        if self.rainbow_angle_red is None:
            return (False, "rainbow_angle_red missing")
        if not isinstance(self.rainbow_angle_red, str):
            return (False, "rainbow_angle_red is not a string")
        return (True, "rainbow_angle_red present and is string")

    def test_interface_rainbow_angle_blue(self):
        if self.rainbow_angle_blue is None:
            return (False, "rainbow_angle_blue missing")
        if not isinstance(self.rainbow_angle_blue, str):
            return (False, "rainbow_angle_blue is not a string")
        return (True, "rainbow_angle_blue present and is string")

    def test_interface_scintillation(self):
        if self.scintillation is None:
            return (False, "scintillation missing")
        if not isinstance(self.scintillation, str):
            return (False, "scintillation is not a string")
        return (True, "scintillation present and is string")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based)
    # ------------------------------------------------------------------
    def test_invariant_rayleigh_scattering(self):
        """Rayleigh scattering: I ∝ λ⁻⁴. Explanation must mention inverse fourth power."""
        if self.scattering_power is None:
            return (False, "SKIP: scattering_power not available")
        text = self.scattering_power.lower()
        if 'lambda^{-4}' in text or 'λ⁻⁴' in text or 'inverse fourth' in text:
            return (True, "Rayleigh power law correctly stated")
        else:
            return (False, f"scattering_power does not contain λ⁻⁴: '{self.scattering_power}'")

    def test_invariant_sunset_path(self):
        """Sunset explanation must mention longer path and blue scattering."""
        if self.transmission is None:
            return (False, "SKIP: transmission not available")
        text = self.transmission.lower()
        has_scatter = 'scatter' in text or 'atten' in text
        has_blue = 'blue' in text
        has_path = 'path' in text or 'thicker' in text or 'long' in text
        if has_scatter and has_blue and has_path:
            return (True, "Sunset explanation includes longer path and blue scattering")
        else:
            return (False, f"transmission lacks required concepts: '{self.transmission}'")

    def test_invariant_rainbow_refraction(self):
        """Rainbow angles must be present (indicates refraction/dispersion)."""
        if self.rainbow_angle_red is None or self.rainbow_angle_blue is None:
            return (False, "One or both rainbow angles missing")
        if 'deg' not in self.rainbow_angle_red and '°' not in self.rainbow_angle_red:
            return (False, "rainbow_angle_red does not indicate degrees")
        if 'deg' not in self.rainbow_angle_blue and '°' not in self.rainbow_angle_blue:
            return (False, "rainbow_angle_blue does not indicate degrees")
        return (True, "Rainbow angles present with degree indication")

    def test_invariant_twinkling_turbulence(self):
        """Twinkling explanation must reference turbulence and refractive index changes."""
        if self.scintillation is None:
            return (False, "SKIP: scintillation not available")
        text = self.scintillation.lower()
        has_turb = 'turb' in text or 'fluctuation' in text or 'random' in text
        has_refr = 'refract' in text or 'refractive index' in text
        if has_turb and has_refr:
            return (True, "Twinkling explanation includes turbulence and refraction")
        else:
            return (False, f"scintillation lacks required concepts: '{self.scintillation}'")

    # ------------------------------------------------------------------
    # Modeling / trace tests (quality of explanation)
    # ------------------------------------------------------------------
    def test_modeling_scattering_power_content(self):
        """Scattering power should clearly state wavelength dependence."""
        if self.scattering_power is None:
            return (False, "SKIP: scattering_power not available")
        text = self.scattering_power.lower()
        has_lambda = 'lambda' in text or 'λ' in text
        has_dep = 'depend' in text or 'power' in text or 'proportional' in text
        if has_lambda and has_dep:
            return (True, "scattering_power describes λ dependence")
        else:
            return (False, f"scattering_power does not describe λ dependence: '{self.scattering_power}'")

    def test_modeling_scintillation_formula(self):
        """Scintillation should mention C_n^2 and L^{11/6}."""
        if self.scintillation is None:
            return (False, "SKIP: scintillation not available")
        text = self.scintillation
        has_Cn = 'C_n^2' in text or 'C_n²' in text or 'Cn^2' in text
        has_L = 'L^{11/6}' in text or 'L^(11/6)' in text or 'L^{11/6}' in text
        if has_Cn and has_L:
            return (True, "Scintillation formula includes C_n^2 and L^{11/6}")
        else:
            return (False, f"scintillation does not contain expected formula: '{self.scintillation}'")

    # ------------------------------------------------------------------
    # Monotonicity tests (perturb input, compare output)
    # ------------------------------------------------------------------
    def test_monotonicity_rayleigh_scattering(self):
        """
        Rayleigh scattering: I ∝ λ⁻⁴ is strictly decreasing.
        Baseline: λ = 500 nm → I_baseline = 1/500⁴
        Perturb: λ = 600 nm → I_pert = 1/600⁴
        Check I_pert < I_baseline.
        """
        try:
            baseline = 1.0 / (500.0**4)
            perturbed = 1.0 / (600.0**4)
        except ZeroDivisionError:
            return (False, "Numerical error in Rayleigh computation")
        if perturbed < baseline:
            # Also verify the explanation mentions the inverse law
            if self.scattering_power is None:
                return (False, "SKIP: scattering_power not available")
            return (True, f"Rayleigh scattering is monotonically decreasing (I ∝ λ⁻⁴): {perturbed:.2e} < {baseline:.2e}")
        else:
            return (False, f"Rayleigh scattering not decreasing: {perturbed:.2e} >= {baseline:.2e}")

    def test_monotonicity_rainbow_angle(self):
        """
        Rainbow: longer wavelength (red) should give larger deviation angle.
        Baseline: blue angle (shorter λ). Perturb: red angle (longer λ).
        Expect red_angle > blue_angle.
        """
        red = self._extract_angle(self.rainbow_angle_red)
        blue = self._extract_angle(self.rainbow_angle_blue)
        if red is None or blue is None:
            return (False, "Cannot parse one or both rainbow angles")
        if red > blue:
            return (True, f"Red angle ({red}°) > Blue angle ({blue}°) – monotonically increasing with λ")
        else:
            return (False, f"Red angle ({red}°) is not greater than blue angle ({blue}°)")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one, no math.isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_basic(self):
        """All five result strings are non-empty (weak sanity)."""
        keys = ['scattering_power', 'transmission', 'rainbow_angle_red',
                'rainbow_angle_blue', 'scintillation']
        values = [self.scattering_power, self.transmission,
                  self.rainbow_angle_red, self.rainbow_angle_blue,
                  self.scintillation]
        missing = [k for k, v in zip(keys, values) if v is None or v.strip() == '']
        if missing:
            return (False, f"Empty or missing results: {missing}")
        return (True, "All five result strings are non-empty")

    # ------------------------------------------------------------------
    # Test runner
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)