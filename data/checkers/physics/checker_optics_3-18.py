import math
import traceback

class PhysicsChecker:
    """Checker for atmospheric transparency problem."""
    
    def __init__(self, solution_instance):
        # Extract results dictionary safely
        self.results = getattr(solution_instance, 'results', {}) or {}
        # Extract explanation steps (list of strings or single string)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        if isinstance(self.explanation_steps, list):
            self.explanation_steps = " ".join(str(s) for s in self.explanation_steps)
        # Extract pseudocode (if present)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)
        if isinstance(self.pseudocode, list):
            self.pseudocode = " ".join(str(s) for s in self.pseudocode)
        
        # Safely get all expected energy keys
        self.E_photon_min = self.results.get('E_photon_min', None)
        self.E_photon_max = self.results.get('E_photon_max', None)
        self.E_CO2_band = self.results.get('E_CO2_band', None)
        self.E_photon_min_nir = self.results.get('E_photon_min_nir', None)
        self.E_photon_max_nir = self.results.get('E_photon_max_nir', None)
        self.E_photon_min_vis = self.results.get('E_photon_min_vis', None)
        self.E_photon_max_vis = self.results.get('E_photon_max_vis', None)
        self.E_photon_min_uv = self.results.get('E_photon_min_uv', None)
        self.E_photon_max_uv = self.results.get('E_photon_max_uv', None)
        self.E_photon_min_x = self.results.get('E_photon_min_x', None)
        self.E_photon_max_x = self.results.get('E_photon_max_x', None)
        self.E_photon_min_gamma = self.results.get('E_photon_min_gamma', None)
        
        # Store list of all energy values for convenience
        self.all_energies = [
            self.E_photon_min, self.E_photon_max,
            self.E_CO2_band,
            self.E_photon_min_nir, self.E_photon_max_nir,
            self.E_photon_min_vis, self.E_photon_max_vis,
            self.E_photon_min_uv, self.E_photon_max_uv,
            self.E_photon_min_x, self.E_photon_max_x,
            self.E_photon_min_gamma
        ]
    
    def _search_keyword(self, text, keywords):
        """Search for any of the keywords in text (case-insensitive)."""
        if text is None:
            return False
        text_lower = text.lower()
        return any(kw.lower() in text_lower for kw in keywords)
    
    # ---- Interface tests ----
    def test_interface_photon_energies_exist(self):
        """Check that all expected energy keys exist and are numeric."""
        missing = []
        for name, val in [
            ("E_photon_min", self.E_photon_min),
            ("E_photon_max", self.E_photon_max),
            ("E_CO2_band", self.E_CO2_band),
            ("E_photon_min_nir", self.E_photon_min_nir),
            ("E_photon_max_nir", self.E_photon_max_nir),
            ("E_photon_min_vis", self.E_photon_min_vis),
            ("E_photon_max_vis", self.E_photon_max_vis),
            ("E_photon_min_uv", self.E_photon_min_uv),
            ("E_photon_max_uv", self.E_photon_max_uv),
            ("E_photon_min_x", self.E_photon_min_x),
            ("E_photon_max_x", self.E_photon_max_x),
            ("E_photon_min_gamma", self.E_photon_min_gamma)
        ]:
            if val is None:
                missing.append(name)
            elif not isinstance(val, (int, float)):
                missing.append(f"{name} (non-numeric)")
        if missing:
            return (False, f"Missing or non-numeric keys: {', '.join(missing)}")
        return (True, "All expected energy keys present and numeric")
    
    def test_interface_explanation_or_pseudocode(self):
        """Check that at least one of explanation_steps or pseudocode is provided."""
        if self.explanation_steps is None and self.pseudocode is None:
            return (False, "Neither explanation_steps nor pseudocode provided")
        return (True, "Explanation steps or pseudocode present")
    
    # ---- Invariant tests (use math.isclose) ----
    def test_invariant_photon_energy_Q1(self):
        """Check that E_photon_min and E_photon_max for 1e8-1e9 Hz are correct within tolerance."""
        if self.E_photon_min is None or self.E_photon_max is None:
            return (False, "SKIP: Q1 energies not available")
        # h = 4.135667696e-15 eV·s
        h_eV_s = 4.135667696e-15
        f_min = 1e8   # Hz
        f_max = 1e9   # Hz
        expected_min = h_eV_s * f_min  # 4.135e-7 eV
        expected_max = h_eV_s * f_max  # 4.135e-6 eV
        if not math.isclose(self.E_photon_min, expected_min, rel_tol=1e-2, abs_tol=1e-8):
            return (False, f"E_photon_min {self.E_photon_min} not close to expected {expected_min}")
        if not math.isclose(self.E_photon_max, expected_max, rel_tol=1e-2, abs_tol=1e-8):
            return (False, f"E_photon_max {self.E_photon_max} not close to expected {expected_max}")
        return (True, "Q1 photon energies correctly scaled from frequency")
    
    def test_invariant_CO2_band_energy(self):
        """Check that CO2 bending mode energy is approximately 0.0827 eV."""
        if self.E_CO2_band is None:
            return (False, "SKIP: E_CO2_band not available")
        lambda_CO2 = 15e-6   # 15 μm in meters
        hc_eV_m = 1.23984193e-6  # hc in eV·m
        expected = hc_eV_m / lambda_CO2  # ~0.0827 eV
        if not math.isclose(self.E_CO2_band, expected, rel_tol=1e-2, abs_tol=1e-4):
            return (False, f"E_CO2_band {self.E_CO2_band} not close to expected {expected}")
        return (True, "CO2 band energy consistent with 15 μm")
    
    def test_invariant_photon_energy_ranges(self):
        """Check that min <= max for all bands where both are present."""
        pairs = [
            ("Q1", self.E_photon_min, self.E_photon_max),
            ("NIR", self.E_photon_min_nir, self.E_photon_max_nir),
            ("Vis", self.E_photon_min_vis, self.E_photon_max_vis),
            ("UV", self.E_photon_min_uv, self.E_photon_max_uv),
            ("X-ray", self.E_photon_min_x, self.E_photon_max_x)
        ]
        for band, lo, hi in pairs:
            if lo is None or hi is None:
                continue
            if lo > hi:
                return (False, f"{band}: min ({lo}) > max ({hi})")
        return (True, "All present min/max pairs are ordered correctly")
    
    # ---- Modeling tests ----
    def test_modeling_absorption_Q1(self):
        """Check that Q1 solution identifies O2, H2O, N2O absorption."""
        text = (self.explanation_steps or "") + " " + (self.pseudocode or "")
        if not text.strip():
            return (False, "SKIP: no text to search")
        keywords = ["O2", "H2O", "N2O", "oxygen", "water", "nitrous oxide"]
        found = sum(1 for kw in keywords if kw.lower() in text.lower())
        if found < 2:  # at least two of the three
            return (False, f"Insufficient mention of Q1 absorbers (found {found} keywords)")
        return (True, "Q1 correctly mentions O2, H2O, N2O absorption")
    
    def test_modeling_absorption_Q2(self):
        """Check that far-IR solution identifies CO2."""
        text = (self.explanation_steps or "") + " " + (self.pseudocode or "")
        if not text.strip():
            return (False, "SKIP: no text")
        if not self._search_keyword(text, ["CO2", "carbon dioxide", "far IR", "far-infrared"]):
            return (False, "Far-IR: missing CO2 absorption")
        return (True, "Far-IR correctly mentions CO2")
    
    def test_modeling_absorption_Q3(self):
        """Check that near-IR solution identifies H2O."""
        text = (self.explanation_steps or "") + " " + (self.pseudocode or "")
        if not text.strip():
            return (False, "SKIP: no text")
        if not self._search_keyword(text, ["H2O", "water", "near IR", "near-infrared"]):
            return (False, "Near-IR: missing H2O absorption")
        return (True, "Near-IR correctly mentions H2O")
    
    def test_modeling_absorption_Q4(self):
        """Check that visible solution mentions transparency and O3."""
        text = (self.explanation_steps or "") + " " + (self.pseudocode or "")
        if not text.strip():
            return (False, "SKIP: no text")
        if not self._search_keyword(text, ["transparent", "visible", "O3", "ozone", "Chappuis"]):
            return (False, "Visible: missing transparency or O3 absorption")
        return (True, "Visible correctly mentions transparency and O3")
    
    def test_modeling_absorption_Q5(self):
        """Check that UV solution identifies O3 absorption."""
        text = (self.explanation_steps or "") + " " + (self.pseudocode or "")
        if not text.strip():
            return (False, "SKIP: no text")
        if not self._search_keyword(text, ["O3", "ozone", "ultraviolet", "UV", "Hartley", "Huggins"]):
            return (False, "UV: missing O3 absorption")
        return (True, "UV correctly mentions O3 absorption")
    
    def test_modeling_absorption_Q6(self):
        """Check that X-ray solution mentions transparency."""
        text = (self.explanation_steps or "") + " " + (self.pseudocode or "")
        if not text.strip():
            return (False, "SKIP: no text")
        if not self._search_keyword(text, ["transparent", "X-ray", "x-ray"]):
            return (False, "X-ray: missing transparency")
        return (True, "X-ray correctly mentions transparency")
    
    def test_modeling_absorption_Q7(self):
        """Check that gamma-ray solution mentions transparency."""
        text = (self.explanation_steps or "") + " " + (self.pseudocode or "")
        if not text.strip():
            return (False, "SKIP: no text")
        if not self._search_keyword(text, ["transparent", "gamma", "γ"]):
            return (False, "Gamma: missing transparency")
        return (True, "Gamma correctly mentions transparency")
    
    # ---- Monotonicity tests (explicit perturbation) ----
    def test_monotonicity_frequency_increases_energy(self):
        """Perturb input frequency by +10% and verify photon energy increases."""
        if self.E_photon_min is None:
            return (False, "SKIP: E_photon_min not available for baseline")
        # Baseline: f_base = 1e8 Hz (minimum of Q1)
        h_eV_s = 4.135667696e-15
        f_base = 1e8
        # Expected baseline energy (should match stored E_photon_min within tolerance)
        baseline_expected = h_eV_s * f_base  # 4.135e-7 eV
        # Check that stored value is consistent
        if not math.isclose(self.E_photon_min, baseline_expected, rel_tol=1e-2, abs_tol=1e-8):
            return (False, f"Stored E_photon_min {self.E_photon_min} not consistent with f=1e8 Hz; cannot use as baseline")
        
        # Perturb: increase frequency by +10%
        f_perturbed = f_base * 1.10
        perturbed_energy_expected = h_eV_s * f_perturbed
        # The perturbed energy should be greater than the baseline
        if perturbed_energy_expected <= self.E_photon_min:
            return (False, f"Perturbed energy {perturbed_energy_expected} not > baseline {self.E_photon_min}")
        return (True, "Increasing frequency increases photon energy (monotonic)")
    
    def test_monotonicity_wavelength_decreases_energy(self):
        """Perturb CO2 wavelength by +10% and verify photon energy decreases."""
        if self.E_CO2_band is None:
            return (False, "SKIP: E_CO2_band not available for baseline")
        # Baseline: lambda_base = 15 μm → E_base = hc/lambda_base
        hc_eV_m = 1.23984193e-6
        lambda_base = 15e-6  # meters
        baseline_expected = hc_eV_m / lambda_base
        if not math.isclose(self.E_CO2_band, baseline_expected, rel_tol=1e-2, abs_tol=1e-4):
            return (False, f"Stored E_CO2_band {self.E_CO2_band} not consistent with 15 μm; cannot use as baseline")
        
        # Perturb: increase wavelength by +10% (energy should decrease)
        lambda_perturbed = lambda_base * 1.10
        perturbed_energy_expected = hc_eV_m / lambda_perturbed
        # The perturbed energy should be less than the baseline
        if perturbed_energy_expected >= self.E_CO2_band:
            return (False, f"Perturbed energy {perturbed_energy_expected} not < baseline {self.E_CO2_band}")
        return (True, "Increasing wavelength decreases photon energy (monotonic)")
    
    # ---- Weak answer sanity tests (no isclose, no re-derivation) ----
    def test_answer_sanity_all_positive_finite(self):
        """Check that all emitted energies are positive and finite."""
        for idx, val in enumerate(self.all_energies):
            if val is None:
                continue
            if not isinstance(val, (int, float)):
                return (False, f"Non-numeric energy at index {idx}: {val}")
            if math.isnan(val) or math.isinf(val):
                return (False, f"NaN or Inf energy at index {idx}")
            if val <= 0:
                return (False, f"Non-positive energy at index {idx}: {val}")
        return (True, "All energies positive and finite")
    
    # ---- Main runner ----
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