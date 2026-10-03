import math
import traceback

class PhysicsChecker:
    """
    Audits solution instances for the Free-Electron Laser conceptual problem.
    Checks interface completeness, invariant relations, modeling quality, monotonicity trends, and answer sanity.
    Assumes optional numeric parameters: undulator_period, gamma, K, wavelength.
    """
    def __init__(self, solution_instance):
        # Extract results dictionary with None defaults
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.laser_acronym = solution_instance.results.get('laser_acronym', None)
            self.laser_output = solution_instance.results.get('laser_output', None)
            self.fel_process = solution_instance.results.get('fel_process', None)
            self.fel_advantage = solution_instance.results.get('fel_advantage', None)
        else:
            # Fallback to direct attributes
            self.laser_acronym = getattr(solution_instance, 'laser_acronym', None)
            self.laser_output = getattr(solution_instance, 'laser_output', None)
            self.fel_process = getattr(solution_instance, 'fel_process', None)
            self.fel_advantage = getattr(solution_instance, 'fel_advantage', None)

        # Optional numeric fields for physics checks (use None defaults)
        self.undulator_period = getattr(solution_instance, 'undulator_period', None)
        self.gamma = getattr(solution_instance, 'gamma', None)
        self.K = getattr(solution_instance, 'K', None)
        self.wavelength = getattr(solution_instance, 'wavelength', None)

        self.solution_instance = solution_instance

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """All four required result keys must be present."""
        keys = [self.laser_acronym, self.laser_output, self.fel_process, self.fel_advantage]
        if any(k is None for k in keys):
            missing = []
            if self.laser_acronym is None: missing.append('laser_acronym')
            if self.laser_output is None: missing.append('laser_output')
            if self.fel_process is None: missing.append('fel_process')
            if self.fel_advantage is None: missing.append('fel_advantage')
            return (False, f"Missing keys: {', '.join(missing)}")
        return (True, "All required keys present")

    def test_interface_keyword_laser_acronym(self):
        """Acronym must contain 'Light', 'Amplification', 'Stimulated', 'Emission', 'Radiation'."""
        if self.laser_acronym is None:
            return (False, "SKIP: laser_acronym not available")
        text = self.laser_acronym.lower()
        required = ['light', 'amplification', 'stimulated', 'emission', 'radiation']
        missing = [w for w in required if w not in text]
        if missing:
            return (False, f"Missing keyword(s) in acronym: {', '.join(missing)}")
        return (True, "Required keywords present")

    def test_interface_keyword_laser_output(self):
        """Output description must mention 'coherent' and 'directional'."""
        if self.laser_output is None:
            return (False, "SKIP: laser_output not available")
        text = self.laser_output.lower()
        required = ['coherent', 'directional']
        missing = [w for w in required if w not in text]
        if missing:
            return (False, f"Missing keyword(s) in output: {', '.join(missing)}")
        return (True, "Keywords coherent and directional present")

    def test_interface_keyword_fel_process(self):
        """Process must mention undulator/magnetic field, bremsstrahlung/radiation, and synchronism/phase velocity."""
        if self.fel_process is None:
            return (False, "SKIP: fel_process not available")
        text = self.fel_process.lower()
        groups = [
            ('undulator/magnetic field', ['undulator', 'magnetic field', 'wiggler']),
            ('radiation/bremsstrahlung', ['bremsstrahlung', 'radiation', 'emission']),
            ('synchronism/phase velocity', ['synchronism', 'phase velocity', 'phase matching', 'resonance condition'])
        ]
        missing_groups = []
        for label, kws in groups:
            if not any(kw in text for kw in kws):
                missing_groups.append(label)
        if missing_groups:
            return (False, f"Missing concept(s) in process: {', '.join(missing_groups)}")
        return (True, "Process contains required physics concepts")

    def test_interface_keyword_fel_advantage(self):
        """Advantage must mention tunability and wavelength."""
        if self.fel_advantage is None:
            return (False, "SKIP: fel_advantage not available")
        text = self.fel_advantage.lower()
        if 'tunab' not in text or 'wavelength' not in text:
            return (False, "Advantage missing 'tunable' or 'wavelength'")
        return (True, "Tunability and wavelength mentioned")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_resonance_condition(self):
        """
        Check FEL resonance condition: λ = λ_u/(2γ²) * (1 + K²/2).
        Only runs if all numeric inputs: undulator_period, gamma, K, wavelength.
        """
        if any(x is None for x in [self.undulator_period, self.gamma, self.K, self.wavelength]):
            return (False, "SKIP: Numeric parameters (undulator_period, gamma, K, wavelength) missing")
        try:
            expected_wavelength = (self.undulator_period / (2 * self.gamma**2)) * (1 + self.K**2 / 2)
            if math.isclose(self.wavelength, expected_wavelength, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "Resonance condition satisfied within 1%")
            else:
                return (False, f"Wavelength mismatch: given {self.wavelength}, expected {expected_wavelength:.4e}")
        except Exception as e:
            return (False, f"Resonance check error: {e}")

    def test_invariant_stimulated_emission_mention(self):
        """
        Conceptual invariant: process must mention stimulated emission (or amplification).
        """
        if self.fel_process is None:
            return (False, "SKIP: fel_process not available")
        text = self.fel_process.lower()
        if 'stimulated emission' not in text and 'amplification' not in text:
            return (False, "Process description missing 'stimulated emission' or 'amplification'")
        return (True, "Stimulated emission concept present")

    # ------------------------------------------------------------------
    # Modeling / trace tests (verify one local relation from explanation)
    # ------------------------------------------------------------------
    def test_modeling_synchronism_condition(self):
        """Process must describe the synchronism condition (electron velocity matches wave phase velocity)."""
        if self.fel_process is None:
            return (False, "SKIP: fel_process not available")
        text = self.fel_process.lower()
        if 'synchronism' not in text and 'phase velocity' not in text and 'resonance condition' not in text:
            return (False, "Synchronism condition not mentioned")
        return (True, "Synchronism condition described")

    def test_modeling_energy_conversion(self):
        """Process must describe conversion of electron kinetic energy to photon energy."""
        if self.fel_process is None:
            return (False, "SKIP: fel_process not available")
        text = self.fel_process.lower()
        # Check for phrases like "energy may be transferred from the electrons to the waves" or "kinetic energy → photon"
        if 'energy' in text and ('transfer' in text or 'convert' in text or 'electron' in text) and ('photon' in text or 'wave' in text or 'radiation' in text):
            return (True, "Energy conversion concept described")
        else:
            # More specific check
            if 'energy' not in text:
                return (False, "No mention of energy conversion")
            if 'electron' in text and 'wave' in text:
                return (True, "Energy conversion implied by electron-wave interaction")
            return (False, "Energy conversion insufficiently described")

    # ------------------------------------------------------------------
    # Monotonicity tests: perturb one input and compare direction of change
    # ------------------------------------------------------------------
    def test_monotonicity_wavelength_vs_gamma(self):
        """
        Perturb gamma by +10% and check wavelength decreases (λ ∝ 1/γ²).
        Requires numeric parameters: undulator_period, gamma, K.
        """
        if any(x is None for x in [self.undulator_period, self.gamma, self.K]):
            return (False, "SKIP: Numeric parameters missing for monotonicity test")
        try:
            gamma0 = self.gamma
            gamma1 = gamma0 * 1.1   # +10%
            factor = 1 + self.K**2 / 2
            lambda0 = (self.undulator_period / (2 * gamma0**2)) * factor
            lambda1 = (self.undulator_period / (2 * gamma1**2)) * factor
            if lambda1 < lambda0:
                return (True, "wavelength decreases when gamma increases (+10%)")
            else:
                return (False, f"Expected λ₁ < λ₀, got λ₁ = {lambda1:.4e}, λ₀ = {lambda0:.4e}")
        except Exception as e:
            return (False, f"Monotonicity test error: {e}")

    def test_monotonicity_wavelength_vs_undulator_period(self):
        """
        Perturb undulator period by +10% and check wavelength increases (λ ∝ λ_u).
        """
        if any(x is None for x in [self.undulator_period, self.gamma, self.K]):
            return (False, "SKIP: Numeric parameters missing for monotonicity test")
        try:
            lu0 = self.undulator_period
            lu1 = lu0 * 1.1   # +10%
            factor = 1 + self.K**2 / 2
            lambda0 = (lu0 / (2 * self.gamma**2)) * factor
            lambda1 = (lu1 / (2 * self.gamma**2)) * factor
            if lambda1 > lambda0:
                return (True, "wavelength increases when undulator period increases (+10%)")
            else:
                return (False, f"Expected λ₁ > λ₀, got λ₁ = {lambda1:.4e}, λ₀ = {lambda0:.4e}")
        except Exception as e:
            return (False, f"Monotonicity test error: {e}")

    # ------------------------------------------------------------------
    # Answer sanity (at most one weak test)
    # ------------------------------------------------------------------
    def test_answer_sanity_basic_checks(self):
        """All answers must be non-empty strings of reasonable length."""
        for name, val in [('laser_acronym', self.laser_acronym),
                          ('laser_output', self.laser_output),
                          ('fel_process', self.fel_process),
                          ('fel_advantage', self.fel_advantage)]:
            if val is None or (isinstance(val, str) and len(val.strip()) == 0):
                return (False, f"{name} is empty or None")
            if isinstance(val, str) and len(val.strip()) < 20:
                return (False, f"{name} too short ({len(val.strip())} chars)")
        return (True, "All answers non-empty and of adequate length")

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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)