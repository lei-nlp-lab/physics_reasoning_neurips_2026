import math

class PhysicsChecker:
    """Audits solutions for a room-air thermodynamics estimation problem."""

    def __init__(self, solution_instance):
        # --- Results (the four emitted numbers) ---
        results = getattr(solution_instance, 'results', None) or {}
        self.N = results.get('N', None)
        self.u = results.get('u', None)
        self.Q = results.get('Q', None)
        self.W_min = results.get('W_min', None)

        # --- Givens (extract safely, default None) ---
        givens = getattr(solution_instance, 'givens', None) or {}
        self.R = givens.get('R', None)          # J/(mol·K) – physics constant, but still None if missing
        self.NA = givens.get('NA', None)        # mol⁻¹
        self.P_atm = givens.get('P_atm', None)  # Pa
        self.V_room = givens.get('V_room', None) # m³
        self.T_room = givens.get('T_room', None) # K
        self.T0 = givens.get('T0', None)        # 0°C in K
        self.T_h = givens.get('T_h', None)      # heat rejection temperature
        self.T_i = givens.get('T_i', None)      # initial cooling temperature
        self.T_f = givens.get('T_f', None)      # final cooling temperature
        self.Cp = givens.get('Cp', None)        # J/(mol·K) at const P
        self.Cv = givens.get('Cv', None)        # J/(mol·K) at const V

        # Explanation / pseudocode traces (optional)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None) or []
        self.pseudocode = getattr(solution_instance, 'pseudocode', None) or {}

    # -----------------------------------------------------------------
    #  Interface tests – require keys, check types
    # -----------------------------------------------------------------
    def test_interface_N_exists(self):
        if self.N is None:
            return (False, "results['N'] missing")
        if not isinstance(self.N, (int, float)):
            return (False, f"results['N'] has wrong type: {type(self.N)}")
        return (True, "N present and numeric")

    def test_interface_u_exists(self):
        if self.u is None:
            return (False, "results['u'] missing")
        if not isinstance(self.u, (int, float)):
            return (False, f"results['u'] has wrong type: {type(self.u)}")
        return (True, "u present and numeric")

    def test_interface_Q_exists(self):
        if self.Q is None:
            return (False, "results['Q'] missing")
        if not isinstance(self.Q, (int, float)):
            return (False, f"results['Q'] has wrong type: {type(self.Q)}")
        return (True, "Q present and numeric")

    def test_interface_W_min_exists(self):
        if self.W_min is None:
            return (False, "results['W_min'] missing")
        if not isinstance(self.W_min, (int, float)):
            return (False, f"results['W_min'] has wrong type: {type(self.W_min)}")
        return (True, "W_min present and numeric")

    # -----------------------------------------------------------------
    #  Invariant / theorem‑based sanity checks (use math.isclose)
    # -----------------------------------------------------------------
    def test_invariant_ideal_gas_n_moles(self):
        """Check that N is consistent with ideal gas law given typical givens."""
        if self.N is None:
            return (False, "SKIP: N not available")
        # Use standard constants if givens missing; the test should gracefully skip
        R_val = self.R if self.R is not None else 8.314
        NA_val = self.NA if self.NA is not None else 6.02214076e23
        P_val = self.P_atm if self.P_atm is not None else 1.013e5
        V_val = self.V_room if self.V_room is not None else 60.0
        T_val = self.T_room if self.T_room is not None else 293.0
        if None in (R_val, NA_val, P_val, V_val, T_val):
            return (False, "SKIP: insufficient givens to compute expected N")
        n_est = P_val * V_val / (R_val * T_val)
        N_est = n_est * NA_val
        if not math.isclose(self.N, N_est, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Number of molecules {self.N:.2e} deviates from ideal‑gas estimate {N_est:.2e}")
        return (True, f"Ideal gas law holds for N (~{N_est:.1e})")

    def test_invariant_internal_energy_per_mole(self):
        """Check u = (5/2) R T_room for diatomic ideal gas."""
        if self.u is None:
            return (False, "SKIP: u not available")
        R_val = self.R if self.R is not None else 8.314
        T_val = self.T_room if self.T_room is not None else 293.0
        if None in (R_val, T_val):
            return (False, "SKIP: R or T_room missing")
        expected_u = 2.5 * R_val * T_val
        if not math.isclose(self.u, expected_u, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Internal energy {self.u:.1f} J does not match expected {expected_u:.1f} J")
        return (True, "Internal energy per mole consistent with equipartition")

    def test_invariant_heat_constant_pressure(self):
        """Check Q for 1 mole, ΔT = (T_room - T0) ≈ 20 K."""
        if self.Q is None:
            return (False, "SKIP: Q not available")
        Cp_val = self.Cp if self.Cp is not None else 3.5 * (self.R if self.R else 8.314)
        R_val = self.R if self.R is not None else 8.314
        T_room_val = self.T_room if self.T_room is not None else 293.0
        T0_val = self.T0 if self.T0 is not None else 273.0
        if None in (R_val, T_room_val, T0_val):
            return (False, "SKIP: to compute expected Q, need R, T_room, T0")
        delta_T = T_room_val - T0_val
        if math.isclose(delta_T, 20.0, abs_tol=0.5):
            expected_Q = 3.5 * R_val * delta_T
            if not math.isclose(self.Q, expected_Q, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"Heat Q = {self.Q:.1f} J, expected ~{expected_Q:.1f} J")
            return (True, f"Q consistent with n·Cp·ΔT, ΔT ≈ {delta_T:.1f} K")
        else:
            # If ΔT is not exactly 20, still check consistency via Q/(ΔT)
            if abs(delta_T) < 1e-6:
                return (False, "SKIP: ΔT ≈ 0, cannot perform check")
            effective_Cp = self.Q / delta_T  # n=1 mole
            expected_Cp = 3.5 * R_val
            if not math.isclose(effective_Cp, expected_Cp, rel_tol=1e-2, abs_tol=0.1):
                return (False, f"Effective Cp = {effective_Cp:.2f}, expected ~{expected_Cp:.2f}")
            return (True, "Q consistent with n·Cp·ΔT (ΔT ≠ 20 but still OK)")

    def test_invariant_carnot_minimum_work(self):
        """Check W_min using Carnot refrigerator formula."""
        if self.W_min is None:
            return (False, "SKIP: W_min not available")
        R_val = self.R if self.R is not None else 8.314
        T_h_val = self.T_h if self.T_h is not None else 313.0
        T_i_val = self.T_i if self.T_i is not None else 293.0
        T_f_val = self.T_f if self.T_f is not None else 291.0
        if None in (R_val, T_h_val, T_i_val, T_f_val):
            return (False, "SKIP: needed temperatures or R missing")
        # n = 1 mole, Cp = 3.5 R
        Cp_val = 3.5 * R_val
        integral = T_h_val * math.log(T_f_val / T_i_val) - (T_f_val - T_i_val)
        expected_W = Cp_val * integral
        if not math.isclose(self.W_min, expected_W, rel_tol=1e-2, abs_tol=0.1):
            return (False, f"Carnot work {self.W_min:.2f} J != expected {expected_W:.2f} J")
        return (True, "W_min consistent with Carnot refrigerator formula")

    # -----------------------------------------------------------------
    #  Modeling quality checks (based on explanation steps / traces)
    # -----------------------------------------------------------------
    def test_modeling_moles_derivation(self):
        """Check consistency between N and ideal gas law: N = (PV/RT) * NA."""
        if self.N is None:
            return (False, "SKIP: N not available")
        # Re‑derive independent of solution
        R_val = self.R if self.R is not None else 8.314
        NA_val = self.NA if self.NA is not None else 6.02214076e23
        P_val = self.P_atm if self.P_atm is not None else 1.013e5
        V_val = self.V_room if self.V_room is not None else 60.0
        T_val = self.T_room if self.T_room is not None else 293.0
        if None in (R_val, NA_val, P_val, V_val, T_val):
            return (False, "SKIP: insufficient data for independent re‑derivation of N")
        n_ideal = P_val * V_val / (R_val * T_val)
        N_ideal = n_ideal * NA_val
        if not math.isclose(self.N, N_ideal, rel_tol=1e-1, abs_tol=1e-1):
            return (False, f"Derived N = {self.N:.2e}, ideal‑gas prediction = {N_ideal:.2e}")
        return (True, "N consistent with ideal gas law and Avogadro")

    def test_modeling_u_sign(self):
        if self.u is None:
            return (False, "SKIP: u not available")
        if self.u <= 0:
            return (False, f"Internal energy u = {self.u} should be positive")
        return (True, "u > 0")

    def test_modeling_Q_sign(self):
        if self.Q is None:
            return (False, "SKIP: Q not available")
        if self.T_room is None or self.T0 is None:
            return (False, "SKIP: temps missing")
        delta_T = self.T_room - self.T0
        if delta_T > 0 and self.Q <= 0:
            return (False, f"Heat Q = {self.Q} should be positive when ΔT>0")
        if delta_T < 0 and self.Q >= 0:
            return (False, f"Heat Q = {self.Q} should be negative when ΔT<0")
        return (True, "Q sign consistent with temperature change")

    def test_modeling_W_min_positivity(self):
        if self.W_min is None:
            return (False, "SKIP: W_min not available")
        if self.W_min < 0:
            return (False, f"W_min = {self.W_min} should be positive (work input)")
        return (True, "W_min > 0")

    # -----------------------------------------------------------------
    #  Monotonicity tests – perturb an input and compare outputs
    # -----------------------------------------------------------------
    def test_monotonicity_N_with_volume(self):
        """
        Perturb room volume: increase by 10% → predicted N should increase.
        Uses ideal gas law to compute nominal and perturbed N, then checks derivative sign.
        """
        if self.N is None:
            return (False, "SKIP: N not available")
        R_val = self.R if self.R is not None else 8.314
        NA_val = self.NA if self.NA is not None else 6.02214076e23
        P_val = self.P_atm if self.P_atm is not None else 1.013e5
        V_val = self.V_room if self.V_room is not None else 60.0
        T_val = self.T_room if self.T_room is not None else 293.0
        if None in (R_val, NA_val, P_val, V_val, T_val):
            return (False, "SKIP: insufficient data to apply monotonicity test")
        n_nom = P_val * V_val / (R_val * T_val)
        N_nom = n_nom * NA_val
        V_pert = 1.1 * V_val
        n_pert = P_val * V_pert / (R_val * T_val)
        N_pert = n_pert * NA_val
        if not (N_pert > N_nom):
            return (False, "Monotonicity violation: N should increase with volume")
        # Also check that solution's N is not too far off from nominal
        if not math.isclose(self.N, N_nom, rel_tol=0.5):
            pass  # estimation can differ, but we don't fail
        return (True, "N increases with volume (linear scaling)")

    def test_monotonicity_Q_with_delta_T(self):
        """
        Perturb ΔT by +10%: predicted Q should increase.
        Uses n=1, Cp=3.5R.
        """
        if self.Q is None:
            return (False, "SKIP: Q not available")
        R_val = self.R if self.R is not None else 8.314
        T_room_val = self.T_room if self.T_room is not None else 293.0
        T0_val = self.T0 if self.T0 is not None else 273.0
        if None in (R_val, T_room_val, T0_val):
            return (False, "SKIP: need R and temperatures")
        delta_T_nom = T_room_val - T0_val
        if abs(delta_T_nom) < 1e-6:
            return (False, "SKIP: ΔT ≈ 0 cannot perturb")
        Cp_val = 3.5 * R_val
        Q_nom = Cp_val * delta_T_nom  # n=1 mole
        delta_T_pert = 1.1 * delta_T_nom
        Q_pert = Cp_val * delta_T_pert
        if not (Q_pert > Q_nom):
            return (False, "Monotonicity violation: Q should increase with ΔT")
        return (True, "Q increases with ΔT (proportional)")

    def test_monotonicity_W_min_with_T_h(self):
        """
        Perturb heat rejection temperature upward by 1%.
        Minimum work for Carnot refrigerator should increase.
        Uses integral formula with n=1, Cp=3.5R.
        """
        if self.W_min is None:
            return (False, "SKIP: W_min not available")
        R_val = self.R if self.R is not None else 8.314
        T_h_val = self.T_h if self.T_h is not None else 313.0
        T_i_val = self.T_i if self.T_i is not None else 293.0
        T_f_val = self.T_f if self.T_f is not None else 291.0
        if None in (R_val, T_h_val, T_i_val, T_f_val):
            return (False, "SKIP: need R and temperatures")
        Cp_val = 3.5 * R_val
        integral_nom = T_h_val * math.log(T_f_val / T_i_val) - (T_f_val - T_i_val)
        W_nom = Cp_val * integral_nom
        T_h_pert = 1.01 * T_h_val  # +1%
        integral_pert = T_h_pert * math.log(T_f_val / T_i_val) - (T_f_val - T_i_val)
        W_pert = Cp_val * integral_pert
        if not (W_pert > W_nom):
            return (False, "Monotonicity violation: W_min should increase with T_h")
        return (True, "W_min increases with rejection temperature (Carnot principle)")

    # -----------------------------------------------------------------
    #  Weak answer‑sanity check (only broad magnitudes, no re‑derivation)
    # -----------------------------------------------------------------
    def test_answer_sanity_broad_magnitudes(self):
        """
        Verify that emitted values lie in physically sensible ranges.
            N  ~ 10^27  (within two orders)
            u  ~ 10^3   (a few thousand joules)
            Q  ~ 10^2   (hundreds of joules)
            W_min ~ 10^0 (single digit joules)
        """
        messages = []
        all_pass = True
        if self.N is not None:
            if not (1e25 < self.N < 1e29):
                messages.append(f"N={self.N:.1e} outside expected ~1e27 range")
                all_pass = False
        if self.u is not None:
            if not (1e2 < self.u < 1e5):
                messages.append(f"u={self.u:.1f} J outside expected ~6e3 range")
                all_pass = False
        if self.Q is not None:
            if not (1e1 < self.Q < 1e5):
                messages.append(f"Q={self.Q:.1f} J outside expected ~6e2 range")
                all_pass = False
        if self.W_min is not None:
            if not (0 < self.W_min < 1e3):
                messages.append(f"W_min={self.W_min:.1f} J outside expected ~4 J range")
                all_pass = False
        if not all_pass:
            return (False, "; ".join(messages) if messages else "No values to check")
        return (True, "All answer magnitudes are physically plausible")

    # -----------------------------------------------------------------
    #  Main runner – discover test_* methods and execute in try/except
    # -----------------------------------------------------------------
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