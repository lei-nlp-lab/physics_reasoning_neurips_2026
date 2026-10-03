import math
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for adiabatic compression of a mixture of ideal gases.
    Checks correctness of effective gamma, final volume, temperature, and pressure.
    """
    
    def __init__(self, solution_instance):
        # Extract given quantities, defaulting to None if missing (no numeric fallbacks)
        self.n_He = self._safe_get(solution_instance, 'n_He', None, 'givens')
        self.n_N2 = self._safe_get(solution_instance, 'n_N2', None, 'givens')
        self.gamma_He = self._safe_get(solution_instance, 'gamma_He', None, 'givens')
        self.gamma_N2 = self._safe_get(solution_instance, 'gamma_N2', None, 'givens')
        self.T0 = self._safe_get(solution_instance, 'T0', None, 'givens')
        self.V0 = self._safe_get(solution_instance, 'V0', None, 'givens')
        self.deltaV_fraction = self._safe_get(solution_instance, 'deltaV_fraction', None, 'givens')
        self.R = self._safe_get(solution_instance, 'R', None, 'givens')   # No fallback – use None if missing
        
        # Extract results (from results dict)
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}
        self.gamma_eff = self.results.get('gamma_eff', None)
        self.V_f = self.results.get('V_f', None)
        self.T_f = self.results.get('T_f', None)
        self.P_f = self.results.get('P_f', None)
        
        # Precompute derived quantities only if all required givens are present
        self.n_total = None
        self.Cv1 = None
        self.Cv2 = None
        self.Cp1 = None
        self.Cp2 = None
        self.Cv_mix = None
        self.Cp_mix = None
        self.gamma_eff_computed = None
        self.P0 = None
        
        if all(x is not None for x in [self.n_He, self.n_N2, self.gamma_He, self.gamma_N2, self.R]):
            self.n_total = self.n_He + self.n_N2
            self.Cv1 = self.R / (self.gamma_He - 1)
            self.Cv2 = self.R / (self.gamma_N2 - 1)
            self.Cp1 = self.gamma_He * self.Cv1
            self.Cp2 = self.gamma_N2 * self.Cv2
            self.Cv_mix = self.n_He * self.Cv1 + self.n_N2 * self.Cv2
            self.Cp_mix = self.n_He * self.Cp1 + self.n_N2 * self.Cp2
            if self.Cv_mix != 0:
                self.gamma_eff_computed = self.Cp_mix / self.Cv_mix
        if self.n_total is not None and self.R is not None and self.T0 is not None and self.V0 is not None:
            if self.V0 != 0:
                self.P0 = self.n_total * self.R * self.T0 / self.V0
            else:
                self.P0 = None
    
    def _safe_get(self, obj, attr, default, source_type='attributes'):
        """Safely extract an attribute from object or from a dict."""
        if source_type == 'givens':
            # Try attribute first, then givens dict if it exists
            val = getattr(obj, attr, None)
            if val is None:
                givens = getattr(obj, 'givens', {})
                if isinstance(givens, dict):
                    val = givens.get(attr, None)
            return val
        else:
            return getattr(obj, attr, default)
    
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
        return all_pass, logs
    
    # --- Interface checks ---
    def test_interface_gamma_eff_key(self):
        if self.gamma_eff is None:
            return (False, "Missing required key 'gamma_eff' in results")
        if not isinstance(self.gamma_eff, (int, float)):
            return (False, f"'gamma_eff' is not numeric, got {type(self.gamma_eff)}")
        return (True, "gamma_eff present and numeric")
    
    def test_interface_V_f_key(self):
        if self.V_f is None:
            return (False, "Missing required key 'V_f' in results")
        if not isinstance(self.V_f, (int, float)):
            return (False, f"'V_f' is not numeric, got {type(self.V_f)}")
        return (True, "V_f present and numeric")
    
    def test_interface_T_f_key(self):
        if self.T_f is None:
            return (False, "Missing required key 'T_f' in results")
        if not isinstance(self.T_f, (int, float)):
            return (False, f"'T_f' is not numeric, got {type(self.T_f)}")
        return (True, "T_f present and numeric")
    
    def test_interface_P_f_key(self):
        if self.P_f is None:
            return (False, "Missing required key 'P_f' in results")
        if not isinstance(self.P_f, (int, float)):
            return (False, f"'P_f' is not numeric, got {type(self.P_f)}")
        return (True, "P_f present and numeric")
    
    # --- Invariant checks (use math.isclose) ---
    def test_invariant_gamma_eff_formula(self):
        if self.gamma_eff is None or self.gamma_eff_computed is None:
            return (False, "SKIP: gamma_eff or required givens missing")
        if not math.isclose(self.gamma_eff, self.gamma_eff_computed, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"gamma_eff {self.gamma_eff} differs from computed {self.gamma_eff_computed}")
        return (True, f"gamma_eff consistent (computed={self.gamma_eff_computed})")
    
    def test_invariant_adiabatic_TV(self):
        if self.T0 is None or self.V0 is None or self.V_f is None or self.T_f is None:
            return (False, "SKIP: missing T0, V0, V_f, or T_f")
        gamma = self.gamma_eff if self.gamma_eff is not None else self.gamma_eff_computed
        if gamma is None:
            return (False, "SKIP: gamma_eff not available")
        if self.V_f <= 0 or self.V0 <= 0:
            return (False, "Invalid volume values")
        left = self.T0 * (self.V0 ** (gamma - 1))
        right = self.T_f * (self.V_f ** (gamma - 1))
        if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Adiabatic TV relation not satisfied: T0*V0^(γ-1)={left}, T_f*V_f^(γ-1)={right}")
        return (True, "Adiabatic TV relation satisfied")
    
    def test_invariant_adiabatic_PV(self):
        if self.P0 is None or self.V0 is None or self.V_f is None or self.P_f is None:
            return (False, "SKIP: missing P0, V0, V_f, or P_f")
        gamma = self.gamma_eff if self.gamma_eff is not None else self.gamma_eff_computed
        if gamma is None:
            return (False, "SKIP: gamma_eff not available")
        if self.V_f <= 0 or self.V0 <= 0:
            return (False, "Invalid volume values")
        left = self.P0 * (self.V0 ** gamma)
        right = self.P_f * (self.V_f ** gamma)
        if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Adiabatic PV relation not satisfied: P0*V0^γ={left}, P_f*V_f^γ={right}")
        return (True, "Adiabatic PV relation satisfied")
    
    def test_invariant_initial_pressure_ideal_gas(self):
        if self.P0 is None:
            return (False, "SKIP: cannot compute P0")
        if self.P_f is None:
            return (False, "SKIP: P_f not available")
        if self.P0 <= 0 or self.P_f <= 0:
            return (False, "Pressure values should be positive")
        return (True, f"Initial pressure computed as {self.P0} Pa, final pressure {self.P_f} Pa")
    
    # --- Modeling checks ---
    def test_modeling_gamma_eff_between_extremes(self):
        if self.gamma_eff is None:
            return (False, "SKIP: gamma_eff not available")
        if self.gamma_He is None or self.gamma_N2 is None:
            return (False, "SKIP: individual gammas missing")
        expected_min = min(self.gamma_He, self.gamma_N2)
        expected_max = max(self.gamma_He, self.gamma_N2)
        if not (expected_min < self.gamma_eff < expected_max):
            return (False, f"gamma_eff={self.gamma_eff} not between {expected_min} and {expected_max}")
        return (True, f"gamma_eff lies between individual gammas")
    
    def test_modeling_volume_reduction(self):
        if self.V0 is None or self.V_f is None or self.deltaV_fraction is None:
            return (False, "SKIP: missing volume data")
        expected_V_f = self.V0 * (1 - self.deltaV_fraction)
        if not math.isclose(self.V_f, expected_V_f, rel_tol=1e-2, abs_tol=1e-6):
            return (False, f"V_f={self.V_f} differs from expected {expected_V_f}")
        return (True, "V_f matches 1% reduction")
    
    def test_modeling_temperature_increases(self):
        if self.T0 is None or self.T_f is None:
            return (False, "SKIP: missing temperatures")
        if self.T_f <= self.T0:
            return (False, f"T_f={self.T_f} should be > T0={self.T0}")
        return (True, "Temperature increases after compression")
    
    def test_modeling_pressure_increases(self):
        if self.P0 is None or self.P_f is None:
            return (False, "SKIP: missing pressures")
        if self.P_f <= self.P0:
            return (False, f"P_f={self.P_f} should be > P0={self.P0}")
        return (True, "Pressure increases after compression")
    
    # --- Monotonicity checks (explicit perturbation) ---
    def test_monotonicity_gamma_eff_with_helium_fraction(self):
        if self.n_He is None or self.n_N2 is None or self.gamma_He is None or self.gamma_N2 is None or self.R is None:
            return (False, "SKIP: missing givens for gamma computation")
        gamma_eff_baseline = self.gamma_eff_computed
        if gamma_eff_baseline is None:
            return (False, "SKIP: could not compute baseline gamma_eff")
        perturbed_n_He = self.n_He * 1.1   # +10%
        new_Cv_mix = perturbed_n_He * self.Cv1 + self.n_N2 * self.Cv2
        new_Cp_mix = perturbed_n_He * self.Cp1 + self.n_N2 * self.Cp2
        if new_Cv_mix == 0:
            return (False, "Division by zero in monotonicity computation")
        gamma_eff_perturbed = new_Cp_mix / new_Cv_mix
        if gamma_eff_perturbed <= gamma_eff_baseline:
            return (False, f"Increasing n_He did not increase gamma_eff: baseline {gamma_eff_baseline}, perturbed {gamma_eff_perturbed}")
        return (True, "Monotonicity: gamma_eff increases with helium fraction")
    
    def test_monotonicity_T_f_with_gamma(self):
        if self.T0 is None or self.V0 is None or self.V_f is None or self.gamma_eff is None:
            return (False, "SKIP: missing data for T_f monotonicity")
        gamma_up = self.gamma_eff + 0.01
        T_f_up = self.T0 * (self.V0 / self.V_f) ** (gamma_up - 1)
        if T_f_up <= self.T_f:
            return (False, f"Increasing gamma should increase T_f, but baseline={self.T_f}, perturbed={T_f_up}")
        return (True, "Monotonicity: T_f increases with gamma")
    
    def test_monotonicity_P_f_with_gamma(self):
        if self.P0 is None or self.V0 is None or self.V_f is None or self.gamma_eff is None:
            return (False, "SKIP: missing data for P_f monotonicity")
        gamma_up = self.gamma_eff + 0.01
        P_f_up = self.P0 * (self.V0 / self.V_f) ** gamma_up
        if P_f_up <= self.P_f:
            return (False, f"Increasing gamma should increase P_f, but baseline={self.P_f}, perturbed={P_f_up}")
        return (True, "Monotonicity: P_f increases with gamma")
    
    # --- Weak answer sanity (exactly one method) ---
    def test_answer_sanity_final(self):
        """Single weak sanity check for T_f and P_f (no re-derivation, no math.isclose)."""
        if self.T_f is None and self.P_f is None:
            return (False, "SKIP: both T_f and P_f missing")
        messages = []
        if self.T_f is not None:
            if not math.isfinite(self.T_f):
                messages.append("T_f is not finite")
            elif not (250 <= self.T_f <= 400):
                messages.append(f"T_f={self.T_f} outside plausible range [250,400] K")
            else:
                messages.append("T_f plausible")
        if self.P_f is not None:
            if not math.isfinite(self.P_f):
                messages.append("P_f is not finite")
            elif not (1e4 <= self.P_f <= 1e6):
                messages.append(f"P_f={self.P_f} outside plausible range [1e4,1e6] Pa")
            else:
                messages.append("P_f plausible")
        if any("not" in m for m in messages):
            return (False, "; ".join(messages))
        return (True, "; ".join(messages))