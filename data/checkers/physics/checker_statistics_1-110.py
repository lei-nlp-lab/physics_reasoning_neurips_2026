import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Audits a solution to the water-ice reversible heat engine problem.
    Checks interface completeness, thermodynamic invariants (entropy balance),
    modeling consistency (heat and work equations), monotonicity, and weak answer sanity.
    """
    def __init__(self, solution_instance):
        # Extract result quantities (safe defaults = None)
        self.results = getattr(solution_instance, 'results', {})
        # Expected result keys from the emit contract
        self.T_wf = self.results.get('T_wf', None)
        self.Q_c = self.results.get('Q_c', None)
        self.m_ice = self.results.get('m_ice', None)
        self.Q_h = self.results.get('Q_h', None)
        self.W = self.results.get('W', None)

        # Extract given physical constants (from results, givens dict, or direct attribute)
        self.m_w = (self.results.get('m_w')
                    or getattr(solution_instance, 'm_w', None))
        self.c = (self.results.get('c')
                  or getattr(solution_instance, 'c', None))
        self.T_w0 = (self.results.get('T_w0')
                     or getattr(solution_instance, 'T_w0', None))
        self.T_ice = (self.results.get('T_ice')
                      or getattr(solution_instance, 'T_ice', None))
        self.L_f = (self.results.get('L_f')
                    or getattr(solution_instance, 'L_f', None))

        # Explanation steps might contain intermediate checks; we store it for reference
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Convert temperatures to Kelvin (assuming Celsius input for T_wf, T_w0, T_ice)
        self.T_wf_K = self._to_kelvin(self.T_wf)
        self.T_w0_K = self._to_kelvin(self.T_w0)
        self.T_ice_K = self._to_kelvin(self.T_ice)

    def _to_kelvin(self, temp_celsius):
        """Convert Celsius to Kelvin, handling None and already-Kelvin values."""
        if temp_celsius is None:
            return None
        # If value > 200, assume already Kelvin (e.g., 373)
        if isinstance(temp_celsius, (int, float)):
            if temp_celsius > 200:
                return temp_celsius
            else:
                return temp_celsius + 273.15
        return None

    # ----------------------------------------------------------------------
    # Interface tests: required keys exist and have valid types
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check all five required result keys are present and not None."""
        required = ['T_wf', 'Q_c', 'm_ice', 'Q_h', 'W']
        missing = [k for k in required if self.results.get(k) is None]
        if missing:
            msg = f"Missing required keys: {missing}"
            return (False, msg)
        return (True, "All required keys present")

    # ----------------------------------------------------------------------
    # Invariant tests: physics must satisfy entropy balance
    # ----------------------------------------------------------------------
    def test_invariant_entropy_balance(self):
        """Total entropy change of water + ice must be zero (reversible process)."""
        if None in (self.T_wf_K, self.T_w0_K, self.T_ice_K, self.Q_c, self.m_w, self.c):
            return (False, "SKIP: missing data for entropy balance")

        # ΔS_water = m_w * c * ln(T_wf / T_w0) (note c in cal/g/degC, m_w in g)
        if self.T_wf_K <= 0 or self.T_w0_K <= 0:
            return (False, "Non-positive absolute temperature encountered")
        dS_water = self.m_w * self.c * math.log(self.T_wf_K / self.T_w0_K)
        # ΔS_ice = Q_c / T_ice (T_ice in Kelvin)
        if self.T_ice_K <= 0:
            return (False, "Ice temperature <= 0 (should be 273K)")
        dS_ice = self.Q_c / self.T_ice_K
        total_dS = dS_water + dS_ice

        # Expect total ≈ 0 (with tolerance)
        if math.isclose(total_dS, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Entropy sum {total_dS:.4f} ≈ 0")
        else:
            return (False, f"Entropy sum {total_dS:.4f} not close to 0")

    def test_invariant_final_temperature(self):
        """Final water temperature must equal ice temperature (0°C = 273K)."""
        if self.T_wf is None or self.T_ice is None:
            return (False, "SKIP: missing temperature data")
        # Use Kelvin conversion to avoid unit ambiguity
        if self.T_wf_K is None or self.T_ice_K is None:
            return (False, "Could not convert temperatures to Kelvin")
        if math.isclose(self.T_wf_K, self.T_ice_K, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"T_wf ({self.T_wf}°C) matches T_ice ({self.T_ice}°C)")
        else:
            return (False, f"Temperature mismatch: T_wf={self.T_wf}°C, T_ice={self.T_ice}°C")

    # ----------------------------------------------------------------------
    # Modeling tests: verify the thermodynamic relationships used
    # ----------------------------------------------------------------------
    def test_modeling_water_heat(self):
        """Q_h = m_w * c * (T_w0 - T_wf) (calories, with Celsius temperatures)."""
        if any(v is None for v in (self.Q_h, self.m_w, self.c, self.T_w0, self.T_wf)):
            return (False, "SKIP: missing data for water heat transfer")
        expected_Q_h = self.m_w * self.c * (self.T_w0 - self.T_wf)
        if math.isclose(self.Q_h, expected_Q_h, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Q_h = {self.Q_h} ≈ {expected_Q_h}")
        else:
            return (False, f"Q_h = {self.Q_h}, expected {expected_Q_h}")

    def test_modeling_ice_melt(self):
        """m_ice = Q_c / L_f."""
        if any(v is None for v in (self.m_ice, self.Q_c, self.L_f)):
            return (False, "SKIP: missing data for ice melting")
        if self.Q_c <= 0 or self.L_f <= 0:
            return (False, "Q_c or L_f non-positive")
        expected_m = self.Q_c / self.L_f
        if math.isclose(self.m_ice, expected_m, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"m_ice = {self.m_ice} ≈ {expected_m}")
        else:
            return (False, f"m_ice = {self.m_ice}, expected {expected_m}")

    def test_modeling_work_balance(self):
        """W = Q_h - Q_c (first law for the heat engine)."""
        if any(v is None for v in (self.W, self.Q_h, self.Q_c)):
            return (False, "SKIP: missing data for work balance")
        expected_W = self.Q_h - self.Q_c
        if math.isclose(self.W, expected_W, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"W = {self.W} ≈ {expected_W}")
        else:
            return (False, f"W = {self.W}, expected {expected_W}")

    # ----------------------------------------------------------------------
    # Monotonicity tests: perturb an input and check output trend
    # ----------------------------------------------------------------------
    def test_monotonicity_work_decreases_with_higher_ice_temperature(self):
        """
        If ice temperature is increased (e.g., from 0°C to +5% in Kelvin),
        the work output should decrease because the Carnot efficiency drops.
        This re-derives W using the same thermodynamic relations from the problem.
        """
        # We need baseline ice temperature and water mass, c, initial temp, and work
        required = (self.m_w, self.c, self.T_w0, self.T_ice, self.W, self.L_f)
        if any(v is None for v in required):
            return (False, "SKIP: missing data for monotonicity test")

        # Convert temperatures to Kelvin for thermodynamic calculation
        T_ice_baseline_K = self._to_kelvin(self.T_ice)
        T_w0_K = self._to_kelvin(self.T_w0)
        if T_ice_baseline_K is None or T_w0_K is None:
            return (False, "SKIP: temperature conversion failed")

        # Perturb ice temperature upward by +5% (relative to Kelvin)
        T_ice_perturbed_K = T_ice_baseline_K * 1.05
        # Final water temperature equals perturbed ice temperature
        T_wf_perturbed_K = T_ice_perturbed_K

        # Recompute Q_c using entropy balance: ΔS_water + ΔS_ice = 0
        # ΔS_water = m_w * c * ln(T_wf_perturbed_K / T_w0_K)
        # ΔS_ice = Q_c_perturbed / T_ice_perturbed_K
        # Q_c_perturbed = - m_w * c * T_ice_perturbed_K * ln(T_wf_perturbed_K / T_w0_K)
        dS_water_perturbed = self.m_w * self.c * math.log(T_wf_perturbed_K / T_w0_K)
        Q_c_perturbed = - self.m_w * self.c * T_ice_perturbed_K * math.log(T_wf_perturbed_K / T_w0_K)

        # Recompute Q_h: Q_h = m_w * c * (T_w0 - T_wf) [Celsius difference]
        # Since T_w0 is in °C, T_wf_perturbed in °C = T_wf_perturbed_K - 273.15
        # But easier: use Kelvin temperatures and specific heat in same units.
        # Note: m_w is in grams, c=1 cal/g°C, but using Kelvin difference is same magnitude.
        T_w0_C = self.T_w0 if self.T_w0 < 200 else self.T_w0 - 273.15  # normalise to Celsius
        T_wf_perturbed_C = T_wf_perturbed_K - 273.15
        Q_h_perturbed = self.m_w * self.c * (T_w0_C - T_wf_perturbed_C)

        # Compute perturbed work
        W_perturbed = Q_h_perturbed - Q_c_perturbed

        # Compare: perturbed work should be less than baseline work (because higher T_ice reduces efficiency)
        # Use relative check, allow some tolerance for rounding
        if W_perturbed < self.W:
            return (True, f"Baseline W={self.W}, perturbed W={W_perturbed:.2f} (decrease as expected)")
        else:
            return (False, f"Monotonicity violated: baseline W={self.W}, perturbed W={W_perturbed:.2f} (should be smaller)")

    # ----------------------------------------------------------------------
    # Weak answer sanity tests (no math.isclose, no re-derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_positivity(self):
        """Final temperature near 0°C, melted ice positive, work positive."""
        issues = []

        # T_wf should be around 0°C (or 273K)
        if self.T_wf is not None:
            # accept if small (Celsius) or ~273 (Kelvin)
            if isinstance(self.T_wf, (int, float)):
                if self.T_wf > 200 and not (270 < self.T_wf < 280):
                    issues.append(f"T_wf = {self.T_wf} does not look like 0°C (273K)")
                elif self.T_wf < 200 and not (-10 < self.T_wf < 10):
                    issues.append(f"T_wf = {self.T_wf}°C not close to 0°C")
        else:
            issues.append("T_wf absent")

        # m_ice should be positive and not extremely large (say < 1000 kg)
        if self.m_ice is not None:
            if self.m_ice <= 0:
                issues.append("m_ice is non-positive")
            elif self.m_ice > 1000:
                issues.append("m_ice > 1000 kg, unrealistic")
        else:
            issues.append("m_ice absent")

        # Work should be positive and of reasonable magnitude (10^3 to 10^5 cal)
        if self.W is not None:
            if self.W <= 0:
                issues.append("Work non-positive")
            elif self.W > 1e6:
                issues.append("Work > 1e6 cal, too large")
        else:
            issues.append("Work absent")

        if issues:
            return (False, "; ".join(issues))
        return (True, "Sanity checks passed: all quantities have plausible signs/magnitudes")

    # ----------------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------------
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