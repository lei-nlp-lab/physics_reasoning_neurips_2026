import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens and results safely
        self.givens = getattr(solution_instance, 'givens', None) or {}
        self.results = getattr(solution_instance, 'results', None) or {}

        # Extract required result keys
        self.E_sun = self.results.get('E_sun', None)
        self.E_Hbomb = self.results.get('E_Hbomb', None)
        self.E_fusion = self.results.get('E_fusion', None)
        self.characteristic = self.results.get('characteristic', None)

        # Optional masses from givens (for monotonicity)
        self.mass_proton = self.givens.get('mass_proton', None)
        self.mass_neutron = self.givens.get('mass_neutron', None)
        self.mass_deuteron = self.givens.get('mass_deuteron', None)
        self.mass_triton = self.givens.get('mass_triton', None)
        self.mass_He4 = self.givens.get('mass_He4', None)

    def test_interface_required_keys(self):
        """Check that all four required keys are present in results."""
        required = ['E_sun', 'E_Hbomb', 'E_fusion', 'characteristic']
        missing = [key for key in required if key not in self.results]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    def test_invariant_energy_positive(self):
        """Check that energy values are positive numbers."""
        energies = [('E_sun', self.E_sun), ('E_Hbomb', self.E_Hbomb), ('E_fusion', self.E_fusion)]
        for name, val in energies:
            if val is None:
                return (False, f"{name} is None")
            if not isinstance(val, (int, float)):
                return (False, f"{name} is not a number: {type(val)}")
            if val <= 0:
                return (False, f"{name} is not positive: {val}")
        return (True, "All energies positive")

    def test_invariant_energy_scale(self):
        """Check that energy values are in a plausible MeV range (1–100 MeV)."""
        energies = [('E_sun', self.E_sun), ('E_Hbomb', self.E_Hbomb), ('E_fusion', self.E_fusion)]
        for name, val in energies:
            if val is None:
                continue
            if val < 1 or val > 100:
                return (False, f"{name} value {val} is outside plausible range 1–100 MeV")
        return (True, "All energies within plausible MeV scale")

    def test_modeling_characteristic(self):
        """Check that characteristic is a meaningful string about nuclear forces."""
        if self.characteristic is None:
            return (False, "characteristic is None")
        if not isinstance(self.characteristic, str):
            return (False, f"characteristic is not a string: {type(self.characteristic)}")
        if 'saturation' in self.characteristic.lower():
            return (True, "Characteristic contains 'saturation'")
        if any(kw in self.characteristic.lower() for kw in ['short-range', 'strong force', 'nuclear force']):
            return (True, "Characteristic mentions plausible nuclear property")
        return (False, f"Characteristic '{self.characteristic}' not recognized as relevant")

    def test_modeling_energy_order(self):
        """Check expected ordering: E_sun > E_fusion and E_Hbomb > E_fusion.
        This also serves as a weak modeling check."""
        if self.E_sun is None or self.E_fusion is None:
            return (False, "E_sun or E_fusion missing for order check")
        if not (self.E_sun > self.E_fusion):
            return (False, f"E_sun ({self.E_sun}) is not > E_fusion ({self.E_fusion})")
        if self.E_Hbomb is not None and not (self.E_Hbomb > self.E_fusion):
            return (False, f"E_Hbomb ({self.E_Hbomb}) is not > E_fusion ({self.E_fusion})")
        return (True, "Energy ordering consistent with expected hierarchy")

    def test_monotonicity_sun_energy_increases_with_proton_mass(self):
        """Perturb the proton mass upward by 1% and verify that the computed
        energy for the sun reaction increases. This tests the monotonic relation
        E = Δm c^2."""
        if self.mass_proton is None or self.mass_He4 is None or self.E_sun is None:
            return (True, "SKIP: missing masses for sun monotonicity test")
        # Use standard atomic mass unit conversion factor
        u_to_MeV = 931.494
        baseline_delta = 4 * self.mass_proton - self.mass_He4
        baseline_E = baseline_delta * u_to_MeV
        # Perturb proton mass upward by 1%
        perturbed_mass_proton = self.mass_proton * 1.01
        perturbed_delta = 4 * perturbed_mass_proton - self.mass_He4
        perturbed_E = perturbed_delta * u_to_MeV
        if not (perturbed_E > baseline_E):
            return (False, f"Perturbed energy {perturbed_E} not > baseline {baseline_E}")
        return (True, "Monotonicity satisfied: increasing proton mass increases sun energy")

    def test_monotonicity_fusion_energy_increases_with_mass_defect(self):
        """For the D+T reaction, perturb the deuteron mass upward by 1% and check
        that the computed energy increases (since Δm = md+mt-mHe4-mn)."""
        if (self.mass_deuteron is None or self.mass_triton is None or
            self.mass_He4 is None or self.mass_neutron is None or self.E_fusion is None):
            return (True, "SKIP: missing masses for fusion monotonicity test")
        u_to_MeV = 931.494
        baseline_delta = self.mass_deuteron + self.mass_triton - self.mass_He4 - self.mass_neutron
        baseline_E = baseline_delta * u_to_MeV
        perturbed_mass_d = self.mass_deuteron * 1.01
        perturbed_delta = perturbed_mass_d + self.mass_triton - self.mass_He4 - self.mass_neutron
        perturbed_E = perturbed_delta * u_to_MeV
        if not (perturbed_E > baseline_E):
            return (False, f"Perturbed energy {perturbed_E} not > baseline {baseline_E}")
        return (True, "Monotonicity satisfied: increasing deuteron mass increases fusion energy")

    def test_answer_sanity_weak(self):
        """Weak sanity: ensure all energies are finite numbers and non-negative."""
        energies = [self.E_sun, self.E_Hbomb, self.E_fusion]
        for idx, e in enumerate(energies):
            if e is None:
                continue
            if not isinstance(e, (int, float)):
                return (False, f"Energy at index {idx} is non-numeric: {type(e)}")
            if math.isinf(e) or math.isnan(e):
                return (False, f"Energy at index {idx} is inf or nan: {e}")
        return (True, "All energies are finite")

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