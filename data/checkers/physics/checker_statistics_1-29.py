import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens safely from either .givens dict or direct attribute
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            givens = solution_instance.givens
        else:
            givens = {}

        # Universal constants: use None if not provided
        self.sigma = givens.get('sigma', getattr(solution_instance, 'sigma', None))
        self.c = givens.get('c', getattr(solution_instance, 'c', None))

        # Sun parameters
        self.T_S = givens.get('T_S', getattr(solution_instance, 'T_S', None))
        self.R_S = givens.get('R_S', getattr(solution_instance, 'R_S', None))
        self.d = givens.get('d', getattr(solution_instance, 'd', None))

        # Earth parameters
        self.R_E = givens.get('R_E', getattr(solution_instance, 'R_E', None))

        # Chondrule parameters
        self.R = givens.get('R', getattr(solution_instance, 'R', None))

        # Results
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            results = solution_instance.results
        else:
            results = {}

        self.T_E = results.get('T_E', None)
        self.F_E = results.get('F_E', None)
        self.T_c = results.get('T_c', None)
        self.F_c = results.get('F_c', None)

    # -------------------- Interface Checks --------------------
    def test_interface_required_results_keys(self):
        """Check that all required result keys exist and are numeric."""
        required = [('T_E', self.T_E), ('F_E', self.F_E),
                    ('T_c', self.T_c), ('F_c', self.F_c)]
        missing = [name for name, val in required if val is None]
        if missing:
            return (False, f"Missing results: {', '.join(missing)}")
        non_numeric = [name for name, val in required if not isinstance(val, (int, float))]
        if non_numeric:
            return (False, f"Non-numeric results: {', '.join(non_numeric)}")
        return (True, "All required result keys present and numeric")

    def test_interface_required_givens_present(self):
        """Check that essential givens are present (non-None and positive)."""
        required = [
            ('T_S', self.T_S),
            ('R_S', self.R_S),
            ('R_E', self.R_E),
            ('d', self.d),
            ('R', self.R),
            ('sigma', self.sigma),
            ('c', self.c)
        ]
        missing = [name for name, val in required if val is None]
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        non_positive = [name for name, val in required if val is not None and val <= 0]
        if non_positive:
            return (False, f"Non-positive givens: {', '.join(non_positive)}")
        return (True, "All essential givens present and positive")

    # -------------------- Invariant Checks (theorem-based) --------------------
    def test_invariant_earth_energy_balance(self):
        """Check that Earth’s emitted power equals absorbed solar power."""
        if any(v is None for v in (self.T_S, self.R_S, self.d, self.R_E, self.T_E, self.sigma)):
            return (False, "SKIP: missing required data for energy balance")

        S = self.sigma * (self.T_S ** 4) * (self.R_S / self.d) ** 2
        absorbed = math.pi * (self.R_E ** 2) * S
        emitted = 4 * math.pi * (self.R_E ** 2) * self.sigma * (self.T_E ** 4)

        if not math.isclose(absorbed, emitted, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Earth energy imbalance: absorbed {absorbed:.3e} vs emitted {emitted:.3e}")
        return (True, "Earth energy balance holds within 1% tolerance")

    def test_invariant_earth_radiation_force(self):
        """Check that force on Earth equals absorbed power / c."""
        if any(v is None for v in (self.T_S, self.R_S, self.d, self.R_E, self.F_E, self.c, self.sigma)):
            return (False, "SKIP: missing required data for force check")

        S = self.sigma * (self.T_S ** 4) * (self.R_S / self.d) ** 2
        absorbed = math.pi * (self.R_E ** 2) * S
        expected_force = absorbed / self.c

        if not math.isclose(self.F_E, expected_force, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"F_E {self.F_E:.3e} does not match absorbed power/c {expected_force:.3e}")
        return (True, "Radiation force on Earth consistent with absorbed power / c")

    def test_invariant_chondrule_force_scaling(self):
        """Check that chondrule force scales with area (R^2) relative to Earth."""
        if any(v is None for v in (self.R_E, self.F_E, self.R, self.F_c)):
            return (False, "SKIP: missing R_E, F_E, R, or F_c")

        expected_ratio = (self.R ** 2) / (self.R_E ** 2)
        actual_ratio = self.F_c / self.F_E

        if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Force ratio {actual_ratio:.3e} does not match area ratio {expected_ratio:.3e}")
        return (True, "Chondrule force scales correctly with cross-sectional area")

    def test_invariant_chondrule_temperature_identical(self):
        """Check that chondrule temperature equals Earth temperature."""
        if self.T_c is None or self.T_E is None:
            return (False, "SKIP: T_c or T_E missing")
        if not math.isclose(self.T_c, self.T_E, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Chondrule temperature {self.T_c:.2f} differs from Earth temperature {self.T_E:.2f}")
        return (True, "Chondrule temperature matches Earth temperature")

    # -------------------- Modeling / Trace Checks --------------------
    def test_modeling_solar_flux_self_consistency(self):
        """Check that solar flux S derived from T_S, R_S, d equals 4σT_E^4."""
        if any(v is None for v in (self.T_S, self.R_S, self.d, self.T_E, self.sigma)):
            return (False, "SKIP: missing data for solar flux consistency")

        S_from_sun = self.sigma * (self.T_S ** 4) * (self.R_S / self.d) ** 2
        S_from_earth = 4 * self.sigma * (self.T_E ** 4)

        if not math.isclose(S_from_sun, S_from_earth, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"S from sun {S_from_sun:.3e} vs S from earth {S_from_earth:.3e} not consistent")
        return (True, "Solar flux derived from Sun and from Earth temperature agree")

    def test_modeling_chondrule_force_from_flux(self):
        """Check that chondrule force F_c = π R^2 S / c using re-derived S."""
        if any(v is None for v in (self.T_S, self.R_S, self.d, self.R, self.F_c, self.c, self.sigma)):
            return (False, "SKIP: missing data for chondrule force check")

        S = self.sigma * (self.T_S ** 4) * (self.R_S / self.d) ** 2
        expected_force = math.pi * (self.R ** 2) * S / self.c

        if not math.isclose(self.F_c, expected_force, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"F_c {self.F_c:.3e} does not match πR²S/c {expected_force:.3e}")
        return (True, "Chondrule force consistent with formula πR²S/c")

    # -------------------- Monotonicity Check --------------------
    def test_monotonicity_earth_force_increase_with_size(self):
        """Check that increasing R_E increases F_E (square law) using formula recomputation."""
        if any(v is None for v in (self.T_S, self.R_S, self.d, self.R_E, self.c, self.sigma)):
            return (False, "SKIP: cannot perform monotonicity test due to missing data")

        S = self.sigma * (self.T_S ** 4) * (self.R_S / self.d) ** 2
        baseline_force = math.pi * (self.R_E ** 2) * S / self.c

        # Perturb R_E by +5%
        R_E_new = self.R_E * 1.05
        new_force = math.pi * (R_E_new ** 2) * S / self.c

        expected_fractional_increase = (R_E_new ** 2) / (self.R_E ** 2) - 1  # = 0.1025
        actual_fractional_increase = (new_force - baseline_force) / baseline_force

        if not math.isclose(actual_fractional_increase, expected_fractional_increase, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Force increase {actual_fractional_increase:.4f} not consistent with area increase {expected_fractional_increase:.4f}")
        return (True, "Monotonicity: F_E increases with R_E^2 as expected")

    # -------------------- Weak Answer Sanity --------------------
    def test_answer_sanity_temperature_and_force_ranges(self):
        """Very weak sanity: check that temperatures are plausible and forces are positive and finite."""
        if self.T_E is not None:
            if not (250 < self.T_E < 330):
                return (False, f"T_E = {self.T_E:.1f} K outside plausible range (250-330 K)")
        if self.F_E is not None:
            if not (0 < self.F_E < 1e10):
                return (False, f"F_E = {self.F_E:.3e} N outside plausible range (0, 1e10)")
        if self.T_c is not None:
            if not (250 < self.T_c < 330):
                return (False, f"T_c = {self.T_c:.1f} K outside plausible range (250-330 K)")
        if self.F_c is not None:
            if not (0 < self.F_c < 1e-5):
                return (False, f"F_c = {self.F_c:.3e} N outside plausible range (0, 1e-5)")
        return (True, "All answer-sanity range checks passed")

    # -------------------- Test Runner --------------------
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)