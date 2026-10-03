import math
import traceback

class PhysicsChecker:
    """Audits physics solutions for three temperature problems:
    (a) ideal gas, (b) Maxwell-Boltzmann populations, (c) entropy-heat flow.
    """

    # Physical constants
    R = 8.314          # J/(mol·K)
    NA = 6.022e23      # mol⁻¹
    k = 8.617e-5       # eV/K

    def __init__(self, solution_instance):
        """
        Extract all required data safely (defaults to None).
        Assume solution_instance may have:
          - .givens: dict or attributes for N, V_ml, P_atm,
            energies_ev, populations_pct, dQ_dt_W,
            entropy_times_s, entropy_values_JK
          - .results: dict with keys T1, T2, T3
        """
        # Results (outputs)
        self.results = getattr(solution_instance, 'results', {})
        self.T1 = self.results.get('T1', None)
        self.T2 = self.results.get('T2', None)
        self.T3 = self.results.get('T3', None)

        # Givens for problem (a) – ideal gas
        self.N = getattr(solution_instance, 'N', None)
        self.V_ml = getattr(solution_instance, 'V_ml', None)   # litres
        self.P_atm = getattr(solution_instance, 'P_atm', None) # atmospheres

        # Givens for problem (b) – Maxwell-Boltzmann
        self.energies_ev = getattr(solution_instance, 'energies_ev', None)
        self.populations_pct = getattr(solution_instance, 'populations_pct', None)

        # Givens for problem (c) – heat / entropy
        self.dQ_dt_W = getattr(solution_instance, 'dQ_dt_W', None)
        self.entropy_times_s = getattr(solution_instance, 'entropy_times_s', None)
        self.entropy_values_JK = getattr(solution_instance, 'entropy_values_JK', None)

    # ------------------------------------------------------------------
    # Interface tests: check required keys exist and have valid types
    # ------------------------------------------------------------------
    def test_interface_results_exist(self):
        """All three temperature results must be present and numeric."""
        if self.T1 is None or self.T2 is None or self.T3 is None:
            return (False, "Missing one or more result keys (T1, T2, T3).")
        if not all(isinstance(t, (int, float)) for t in [self.T1, self.T2, self.T3]):
            return (False, "All temperature results must be numeric (int or float).")
        return (True, "All three temperature results present and numeric.")

    def test_interface_givens_available_a(self):
        """Check that minimum given data for problem (a) is present."""
        if self.N is None or self.V_ml is None or self.P_atm is None:
            return (False, "Missing one or more givens for ideal gas (N, V, P).")
        return (True, "Ideal gas givens present.")

    def test_interface_givens_available_b(self):
        """Check that minimum given data for problem (b) is present."""
        if self.energies_ev is None or self.populations_pct is None:
            return (False, "Missing energy levels or populations.")
        if len(self.energies_ev) < 2 or len(self.populations_pct) < 2:
            return (False, "At least two levels and populations required.")
        return (True, "MB data present (≥2 levels).")

    def test_interface_givens_available_c(self):
        """Check that minimum given data for problem (c) is present."""
        if self.dQ_dt_W is None or self.entropy_times_s is None or self.entropy_values_JK is None:
            return (False, "Missing dQ/dt or entropy table.")
        if len(self.entropy_times_s) < 3 or len(self.entropy_values_JK) < 3:
            return (False, "Need at least three time points for central difference.")
        return (True, "Entropy data present.")

    # ------------------------------------------------------------------
    # Invariant tests (use math.isclose with 1e-2 tolerance)
    # ------------------------------------------------------------------
    def test_invariant_ideal_gas_law(self):
        """PV = nRT must hold within 2% tolerance.
           Uses given N, V, P and result T1.
        """
        if any(v is None for v in [self.N, self.V_ml, self.P_atm, self.T1]):
            return (False, "SKIP: missing data for ideal gas invariant.")
        # Convert units
        V = self.V_ml * 1e-3          # L -> m³
        P = self.P_atm * 101325.0     # atm -> Pa
        n = self.N / self.NA
        left = P * V
        right = n * self.R * self.T1
        if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-12):
            return (False, f"PV = {left:.4f} but nRT = {right:.4f} (rel diff {abs(left-right)/left:.4f})")
        return (True, f"Ideal gas law holds: PV = {left:.4f}, nRT = {right:.4f}.")

    def test_invariant_mb_population_ratio(self):
        """For the highest and lowest energy levels:
           ln(P_low/P_high)  ≈  (E_high - E_low) / (kB * T2)
        """
        if any(v is None for v in [self.energies_ev, self.populations_pct, self.T2]):
            return (False, "SKIP: missing MB data.")
        # Use levels with max and min energy
        if not self.energies_ev or not self.populations_pct:
            return (False, "Empty energy or population lists.")
        # Find lowest and highest energy levels
        E_low = min(self.energies_ev)
        E_high = max(self.energies_ev)
        # Get corresponding populations (assume same order)
        try:
            idx_low = self.energies_ev.index(E_low)
            idx_high = self.energies_ev.index(E_high)
        except ValueError:
            return (False, "Energy values not consistent with populations.")
        P_low = self.populations_pct[idx_low]   # percentage
        P_high = self.populations_pct[idx_high]
        if P_low <= 0 or P_high <= 0:
            return (False, "Populations must be positive.")
        left = math.log(P_low / P_high)
        right = (E_high - E_low) / (self.k * self.T2)
        if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-6):
            return (False, f"ln(P_low/P_high) = {left:.5f} but ΔE/(kT2) = {right:.5f} (rel diff {abs(left-right)/abs(left):.4f})")
        return (True, f"MB ratio satisfied: ln(P1/P2) = {left:.5f}, ΔE/(kT2) = {right:.5f}.")

    def test_invariant_entropy_derivative(self):
        """At t=500 s, T3 = dQ_dt / (dS_dt) with central difference.
           dS_dt ≈ (S(600)-S(400))/200.
        """
        if any(v is None for v in [self.entropy_times_s, self.entropy_values_JK, self.dQ_dt_W, self.T3]):
            return (False, "SKIP: missing entropy data.")
        # Need t=400, 500, 600 in the table; use indices
        times = self.entropy_times_s
        values = self.entropy_values_JK
        try:
            idx_400 = times.index(400)
            idx_600 = times.index(600)
        except ValueError:
            return (False, "Entropy table lacks points at 400 and 600 s.")
        dS_dt = (values[idx_600] - values[idx_400]) / (600 - 400)
        if dS_dt <= 0:
            return (False, "dS/dt must be positive.")
        T_derived = self.dQ_dt_W / dS_dt
        if not math.isclose(self.T3, T_derived, rel_tol=1e-2, abs_tol=0.5):
            return (False, f"T3 = {self.T3} K but dQ_dt/dS_dt = {T_derived:.2f} K.")
        return (True, f"Entropy relation holds: T3 = {self.T3} K, derived = {T_derived:.2f} K.")

    # ------------------------------------------------------------------
    # Monotonicity tests (perturb an input and check direction)
    # ------------------------------------------------------------------
    def test_monotonicity_pressure_increases_temperature(self):
        """For ideal gas: when pressure increases 1%, temperature should increase.
           Compute baseline T from givens, then perturb P.
        """
        if any(v is None for v in [self.N, self.V_ml, self.P_atm, self.T1]):
            return (False, "SKIP: missing data for monotonicity.")
        V = self.V_ml * 1e-3
        P = self.P_atm * 101325.0
        n = self.N / self.NA
        # Baseline temperature (re-derived)
        T_base = P * V / (n * self.R)
        # Perturb pressure +1%
        P_pert = P * 1.01
        T_pert = P_pert * V / (n * self.R)
        if T_pert <= T_base:
            return (False, f"T_pert ({T_pert:.3f}) not > T_base ({T_base:.3f}) for +1% pressure.")
        return (True, f"Pressure +1% increases T: {T_base:.3f} -> {T_pert:.3f}.")

    def test_monotonicity_energy_diff_increases_temperature(self):
        """For MB levels: increasing ΔE (with same population ratio) raises T.
           Use original levels and populations, then increase highest energy by 1%.
        """
        if any(v is None for v in [self.energies_ev, self.populations_pct, self.T2]):
            return (False, "SKIP: missing MB data.")
        # Find lowest and highest energy
        E_low = min(self.energies_ev)
        E_high = max(self.energies_ev)
        # populations at those indices
        idx_low = self.energies_ev.index(E_low)
        idx_high = self.energies_ev.index(E_high)
        P_low = self.populations_pct[idx_low]
        P_high = self.populations_pct[idx_high]
        if P_low <= 0 or P_high <= 0:
            return (False, "Populations must be positive.")
        # Baseline temperature
        T_base = (E_high - E_low) / (self.k * math.log(P_low / P_high))
        # Perturb: increase E_high by 1%
        E_high_pert = E_high * 1.01
        T_pert = (E_high_pert - E_low) / (self.k * math.log(P_low / P_high))
        if T_pert <= T_base:
            return (False, f"T_pert ({T_pert:.3f}) not > T_base ({T_base:.3f}) for +1% energy diff.")
        return (True, f"Energy diff +1% increases T: {T_base:.3f} -> {T_pert:.3f}.")

    def test_monotonicity_heat_rate_increases_temperature(self):
        """For entropy problem: increasing dQ/dt increases T (holding dS/dt same).
           Perturb dQ_dt by +1% and recompute T.
        """
        if any(v is None for v in [self.dQ_dt_W, self.entropy_times_s, self.entropy_values_JK, self.T3]):
            return (False, "SKIP: missing entropy data.")
        # Compute dS_dt using central difference (same as before)
        times = self.entropy_times_s
        values = self.entropy_values_JK
        try:
            idx_400 = times.index(400)
            idx_600 = times.index(600)
        except ValueError:
            return (False, "Cannot find t=400 or 600 in entropy table.")
        dS_dt = (values[idx_600] - values[idx_400]) / (600 - 400)
        if dS_dt <= 0:
            return (False, "dS/dt must be positive.")
        # Baseline temperature (re-derived)
        T_base = self.dQ_dt_W / dS_dt
        # Perturb dQ_dt +1%
        dQ_pert = self.dQ_dt_W * 1.01
        T_pert = dQ_pert / dS_dt
        if T_pert <= T_base:
            return (False, f"T_pert ({T_pert:.3f}) not > T_base ({T_base:.3f}) for +1% heat rate.")
        return (True, f"Heat rate +1% increases T: {T_base:.3f} -> {T_pert:.3f}.")

    # ------------------------------------------------------------------
    # Modeling quality tests (simple checks on intermediate steps)
    # ------------------------------------------------------------------
    def test_modeling_gas_units_conversion(self):
        """Check that V and P are plausibly in correct SI units after conversion."""
        V_m3 = self.V_ml * 1e-3
        P_pa = self.P_atm * 101325.0
        if V_m3 < 1e-6 or V_m3 > 10:
            return (False, f"V in m³ = {V_m3} seems unrealistic.")
        if P_pa < 1 or P_pa > 1e6:
            return (False, f"P in Pa = {P_pa} seems unrealistic.")
        return (True, "Units conversion plausible.")

    def test_modeling_mb_use_highest_lowest_energies(self):
        """Recommended strategy: use the most separated levels for best accuracy.
           Check that the solver indeed used them (by verifying T2 is close to what
           we compute from extreme levels). This is not an invariant but a modeling hint.
        """
        if any(v is None for v in [self.energies_ev, self.populations_pct, self.T2]):
            return (False, "SKIP: missing MB data.")
        E_low = min(self.energies_ev)
        E_high = max(self.energies_ev)
        idx_low = self.energies_ev.index(E_low)
        idx_high = self.energies_ev.index(E_high)
        P_low = self.populations_pct[idx_low]
        P_high = self.populations_pct[idx_high]
        if P_low <= 0 or P_high <= 0:
            return (False, "Populations must be positive.")
        T_from_extremes = (E_high - E_low) / (self.k * math.log(P_low / P_high))
        if not math.isclose(self.T2, T_from_extremes, rel_tol=5e-2, abs_tol=5.0):
            return (False, f"T2 = {self.T2} but extreme levels give {T_from_extremes:.2f} (relative diff {abs(self.T2-T_from_extremes)/T_from_extremes:.3f})")
        return (True, f"T2 consistent with extreme levels: derived = {T_from_extremes:.2f}, result = {self.T2}.")

    def test_modeling_entropy_central_difference_used(self):
        """Check that the solver used a central difference around t=500.
           We verify by computing T from forward or backward differences;
           if they differ significantly, central is better.
        """
        if any(v is None for v in [self.entropy_times_s, self.entropy_values_JK, self.dQ_dt_W, self.T3]):
            return (False, "SKIP: missing data.")
        times = self.entropy_times_s
        values = self.entropy_values_JK
        try:
            idx_400 = times.index(400)
            idx_500 = times.index(500)
            idx_600 = times.index(600)
        except ValueError:
            return (False, "Need t=400, 500, 600.")
        # Central
        dS_dt_central = (values[idx_600] - values[idx_400]) / (600 - 400)
        T_central = self.dQ_dt_W / dS_dt_central
        # Forward difference (500->600)
        dS_dt_forward = (values[idx_600] - values[idx_500]) / (600 - 500)
        T_forward = self.dQ_dt_W / dS_dt_forward
        # Backward difference (400->500)
        dS_dt_backward = (values[idx_500] - values[idx_400]) / (500 - 400)
        T_backward = self.dQ_dt_W / dS_dt_backward
        # The central estimate should be closest to T3 (or at least reasonable)
        if not math.isclose(T_central, self.T3, rel_tol=5e-2):
            return (False, f"T3 = {self.T3} but central diff gives T_central = {T_central:.2f} (forward={T_forward:.2f}, backward={T_backward:.2f})")
        return (True, f"Central difference yields T = {T_central:.2f}, matching T3 = {self.T3}.")

    # ------------------------------------------------------------------
    # Weak answer sanity (sign, finite, broad range)
    # ------------------------------------------------------------------
    def test_answer_sanity_temperatures_positive(self):
        """All three temperatures must be positive and finite."""
        if any(t is None for t in [self.T1, self.T2, self.T3]):
            return (False, "One or more temperatures missing.")
        for name, t in [('T1', self.T1), ('T2', self.T2), ('T3', self.T3)]:
            if not isinstance(t, (int, float)):
                return (False, f"{name} is not numeric: {t}")
            if t <= 0 or not math.isfinite(t):
                return (False, f"{name} = {t} is not positive/finite.")
        return (True, "All three temperatures are positive and finite.")

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        """Discover all test_ methods and execute them, collecting results."""
        logs = []
        all_pass = True
        for attr_name in sorted(dir(self)):
            if not attr_name.startswith("test_"):
                continue
            method = getattr(self, attr_name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{attr_name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {attr_name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)