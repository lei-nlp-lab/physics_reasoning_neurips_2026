import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract inputs from the problem context (constants)
        self.a = 0.50  # m
        self.w = 0.10  # m
        self.v_b = 20.0  # m/s
        self.n_resistors = 20
        self.R_resistor = 500e6  # Ω
        self.I_beam = 25e-6  # A
        self.V_accel = 500e3  # V
        self.b = 0.10  # m
        self.d_i = 0.004  # m (0.40 cm)
        self.d_o = 0.005  # m (0.50 cm)
        self.D_i = 0.20  # m
        self.D_o = 0.50  # m
        self.n_coils = 24
        self.rho = 1.7e-8  # Ω·m
        self.Z = 9  # fluorine nucleus atomic number

        # Physical constants
        self.epsilon0 = 8.854187817e-12  # F/m
        self.k_e = 8.987551787e9  # N·m²/C² (numeric)
        self.q = 1.60217662e-19  # C
        self.m_p = 1.6726219e-27  # kg
        self.c_water = 4186  # J/(kg·K)
        self.rho_water = 1000  # kg/m³

        # Extract results from solution instance
        self.results = getattr(solution_instance, 'results', {})

        # Helper function to get a result safely
        def safe_get(key, default=None):
            return self.results.get(key, default)

        # Extract all expected quantities with correct key names
        self.E = safe_get('E', 0.0)
        self.C = safe_get('C', 0.0)
        self.R_chain = safe_get('R_chain', 0.0)  # total chain resistance
        if self.R_chain == 0.0:
            self.R_chain = safe_get('R', 0.0)  # fallback if stored as 'R'
        self.tau = safe_get('tau', 0.0)  # time constant
        self.I_R = safe_get('I_R', 0.0)
        self.I_total = safe_get('I_total', 0.0)
        self.dA_dt = safe_get('delta_A/delta_t', 0.0)
        self.sigma = safe_get('sigma', 0.0)
        self.B = safe_get('B', 0.0)
        self.L = safe_get('L', 0.0)
        self.A_pipe = safe_get('A', 0.0)
        self.R_spiral = safe_get('R_spiral', 0.0)
        if self.R_spiral == 0.0:
            self.R_spiral = safe_get('R', 0.0)  # fallback if stored as 'R'
        self.P = safe_get('P', 0.0)
        self.Q_flow = safe_get('Q', 0.0)
        self.r_closest = safe_get('r', 0.0)

        # Compute expected values from inputs
        self.E_expected = self.V_accel / self.a
        self.C_expected = 4 * math.pi * self.epsilon0 * self.a
        self.R_chain_expected = self.n_resistors * self.R_resistor
        self.tau_expected = self.R_chain_expected * self.C_expected
        self.I_R_expected = self.V_accel / self.R_chain_expected
        self.I_total_expected = self.I_R_expected + self.I_beam
        self.dA_dt_expected = self.v_b * self.w
        self.sigma_expected = self.I_total_expected / self.dA_dt_expected
        
        # For B: B = (tan(θ/2)/b) * sqrt(2 m_p V_accel / q)
        theta_half_rad = math.radians(5.0 / 2.0)
        self.B_expected = (math.tan(theta_half_rad) / self.b) * math.sqrt(2 * self.m_p * self.V_accel / self.q)
        
        self.L_expected = math.pi * (self.D_o**2 - self.D_i**2) / (4 * self.d_o)
        self.A_pipe_expected = math.pi * (self.d_o**2 - self.d_i**2) / 4
        self.R_spiral_expected = self.rho * self.L_expected / self.A_pipe_expected
        
        # For closest approach with Z=9
        self.r_closest_expected = self.k_e * self.Z * self.q / self.V_accel

        # Store logs
        self.logs = []

    def log(self, message, success=None):
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {message}")
        else:
            self.logs.append(f"INFO: {message}")

    def fuzzy_equal(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        self.log("Running invariant tests...")

        # 1. Electric field
        if self.fuzzy_equal(self.E, self.E_expected):
            self.log(f"Electric field E = {self.E} matches V/a = {self.E_expected}", True)
        else:
            self.log(f"Electric field mismatch: E={self.E}, expected {self.E_expected}", False)

        # 2. Capacitance
        if self.fuzzy_equal(self.C, self.C_expected):
            self.log(f"Capacitance C = {self.C} matches 4πε0a = {self.C_expected}", True)
        else:
            self.log(f"Capacitance mismatch: C={self.C}, expected {self.C_expected}", False)

        # 3. Total resistance of chain
        if self.fuzzy_equal(self.R_chain, self.R_chain_expected):
            self.log(f"Chain resistance R_chain = {self.R_chain} matches 20*500MΩ = {self.R_chain_expected}", True)
        else:
            self.log(f"Chain resistance mismatch: R_chain={self.R_chain}, expected {self.R_chain_expected}", False)

        # 4. Time constant
        if self.fuzzy_equal(self.tau, self.tau_expected):
            self.log(f"Time constant τ = {self.tau} matches R*C = {self.tau_expected}", True)
        else:
            self.log(f"Time constant mismatch: τ={self.tau}, expected {self.tau_expected}", False)

        # 5. Current conservation
        I_total_calc = self.I_R + self.I_beam
        if self.fuzzy_equal(self.I_total, I_total_calc):
            self.log(f"Total current I_total = {self.I_total} matches I_R + I_beam = {I_total_calc}", True)
        else:
            self.log(f"Current conservation mismatch: I_total={self.I_total}, I_R+I_beam={I_total_calc}", False)

        # 6. Area sweep rate
        if self.fuzzy_equal(self.dA_dt, self.dA_dt_expected):
            self.log(f"Area sweep rate ΔA/Δt = {self.dA_dt} matches v_b*w = {self.dA_dt_expected}", True)
        else:
            self.log(f"Area sweep mismatch: ΔA/Δt={self.dA_dt}, expected {self.dA_dt_expected}", False)

        # 7. Surface charge density
        sigma_calc = self.I_total / (self.v_b * self.w) if self.v_b * self.w != 0 else 0
        if self.fuzzy_equal(self.sigma, sigma_calc):
            self.log(f"Surface charge density σ = {self.sigma} matches I_total/(v_b*w) = {sigma_calc}", True)
        else:
            self.log(f"Surface charge density mismatch: σ={self.sigma}, expected {sigma_calc}", False)

        # 8. Spiral length
        if self.fuzzy_equal(self.L, self.L_expected):
            self.log(f"Spiral length L = {self.L} matches π(Do²-Di²)/(4do) = {self.L_expected}", True)
        else:
            self.log(f"Spiral length mismatch: L={self.L}, expected {self.L_expected}", False)

        # 9. Pipe cross-sectional area
        if self.fuzzy_equal(self.A_pipe, self.A_pipe_expected):
            self.log(f"Pipe area A = {self.A_pipe} matches π/4(do²-di²) = {self.A_pipe_expected}", True)
        else:
            self.log(f"Pipe area mismatch: A={self.A_pipe}, expected {self.A_pipe_expected}", False)

        # 10. Spiral resistance
        if self.fuzzy_equal(self.R_spiral, self.R_spiral_expected):
            self.log(f"Spiral resistance R_spiral = {self.R_spiral} matches ρL/A = {self.R_spiral_expected}", True)
        else:
            self.log(f"Spiral resistance mismatch: R_spiral={self.R_spiral}, expected {self.R_spiral_expected}", False)

        # 11. Proton kinetic energy
        v_proton = math.sqrt(2 * self.q * self.V_accel / self.m_p)
        K = 0.5 * self.m_p * v_proton**2
        delta_U = self.q * self.V_accel
        if self.fuzzy_equal(K, delta_U):
            self.log(f"Kinetic energy {K} equals qV = {delta_U}", True)
        else:
            self.log(f"Energy conservation mismatch: K={K}, qV={delta_U}", False)

        # 12. Closest approach
        if self.r_closest > 0:
            if self.fuzzy_equal(self.r_closest, self.r_closest_expected, rel_tol=1e-2, abs_tol=1e-15):
                self.log(f"Closest approach r = {self.r_closest} matches k_e Z q / V = {self.r_closest_expected}", True)
            else:
                self.log(f"Closest approach mismatch: r={self.r_closest}, expected {self.r_closest_expected}", False)

        passes = sum("PASS" in log for log in self.logs[-20:])
        fails = sum("FAIL" in log for log in self.logs[-20:])
        self.log(f"Invariant tests: {passes} passed, {fails} failed.")

    def test_trace_consistency(self):
        self.log("Running trace consistency tests...")

        # Target values from ground truth
        target_values = {
            'E': 1e6,
            'tau': 0.556,
            'sigma': 37.5e-6,
            'B': 0.0894,
            'L': 33,
            'R_spiral': 0.079,
            'Q_flow_L_s': 0.07,
            'r_closest': 2.59e-14
        }

        # E
        if self.fuzzy_equal(self.E, target_values['E']):
            self.log(f"E = {self.E} matches target {target_values['E']} V/m", True)
        else:
            self.log(f"E mismatch: {self.E} vs target {target_values['E']} V/m", False)

        # τ
        if self.fuzzy_equal(self.tau, target_values['tau'], abs_tol=0.001):
            self.log(f"τ = {self.tau} matches target {target_values['tau']} s", True)
        else:
            self.log(f"τ mismatch: {self.tau} vs target {target_values['tau']} s", False)

        # σ
        if self.fuzzy_equal(self.sigma, target_values['sigma'], abs_tol=1e-6):
            self.log(f"σ = {self.sigma} matches target {target_values['sigma']} C/m²", True)
        else:
            self.log(f"σ mismatch: {self.sigma} vs target {target_values['sigma']} C/m²", False)

        # B
        if self.fuzzy_equal(self.B, target_values['B'], abs_tol=0.001):
            self.log(f"B = {self.B} matches target {target_values['B']} T", True)
        else:
            self.log(f"B mismatch: {self.B} vs target {target_values['B']} T", False)

        # L
        if self.fuzzy_equal(self.L, target_values['L'], abs_tol=0.1):
            self.log(f"L = {self.L} matches target {target_values['L']} m", True)
        else:
            self.log(f"L mismatch: {self.L} vs target {target_values['L']} m", False)

        # R_spiral
        if self.fuzzy_equal(self.R_spiral, target_values['R_spiral'], abs_tol=0.001):
            self.log(f"R_spiral = {self.R_spiral} matches target {target_values['R_spiral']} Ω", True)
        else:
            self.log(f"R_spiral mismatch: {self.R_spiral} vs target {target_values['R_spiral']} Ω", False)

        # Q_flow (already in L/s)
        if self.fuzzy_equal(self.Q_flow, target_values['Q_flow_L_s'], abs_tol=0.01):
            self.log(f"Q_flow = {self.Q_flow} L/s matches target {target_values['Q_flow_L_s']} L/s", True)
        else:
            self.log(f"Q_flow mismatch: {self.Q_flow} L/s vs target {target_values['Q_flow_L_s']} L/s", False)

        # r_closest
        if self.fuzzy_equal(self.r_closest, target_values['r_closest'], rel_tol=1e-2):
            self.log(f"r_closest = {self.r_closest} matches target {target_values['r_closest']} m", True)
        else:
            self.log(f"r_closest mismatch: {self.r_closest} vs target {target_values['r_closest']} m", False)

        passes = sum("PASS" in log for log in self.logs[-20:])
        fails = sum("FAIL" in log for log in self.logs[-20:])
        self.log(f"Trace consistency tests: {passes} passed, {fails} failed.")

    def test_monotonicity(self):
        self.log("Running monotonicity test...")

        original_V = self.V_accel
        perturbed_V = original_V * 1.1

        v_original = math.sqrt(2 * self.q * original_V / self.m_p)
        v_perturbed = math.sqrt(2 * self.q * perturbed_V / self.m_p)
        
        K_original = 0.5 * self.m_p * v_original**2
        K_perturbed = 0.5 * self.m_p * v_perturbed**2

        if K_perturbed > K_original:
            self.log(f"Monotonicity holds: K increased from {K_original} to {K_perturbed} when V increased by 10%", True)
        else:
            self.log(f"Monotonicity violated: K did not increase with V", False)

        expected_ratio = perturbed_V / original_V
        actual_ratio = K_perturbed / K_original
        if self.fuzzy_equal(actual_ratio, expected_ratio, rel_tol=1e-2):
            self.log(f"Kinetic energy ratio {actual_ratio} matches voltage ratio {expected_ratio}", True)
        else:
            self.log(f"Kinetic energy ratio mismatch: {actual_ratio} vs {expected_ratio}", False)

    def run_all_tests(self):
        self.logs = []
        try:
            self.test_invariants()
            self.test_trace_consistency()
            self.test_monotonicity()

            fail_count = sum("FAIL" in log for log in self.logs)
            success = (fail_count == 0)

            if success:
                self.log("All tests passed!", True)
            else:
                self.log(f"{fail_count} test(s) failed.", False)

            return success, self.logs

        except Exception as e:
            self.log(f"Exception during testing: {traceback.format_exc()}", False)
            return False, self.logs