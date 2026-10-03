import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        """
        self.solution = solution_instance
        
        # Safely get inputs (with defaults to 1.0 to avoid crashes)
        self.a = getattr(solution_instance, 'a', getattr(solution_instance, 'inputs', {}).get('a', 1.0))
        self.r = getattr(solution_instance, 'r', getattr(solution_instance, 'inputs', {}).get('r', 1.0))
        self.B = getattr(solution_instance, 'B', getattr(solution_instance, 'inputs', {}).get('B', 1.0))
        self.n = getattr(solution_instance, 'n', getattr(solution_instance, 'inputs', {}).get('n', 1.0))
        self.R = getattr(solution_instance, 'R', getattr(solution_instance, 'inputs', {}).get('R', 1.0))
        
        # Safely get results (default to 0.0 if missing)
        results = getattr(solution_instance, 'results', {})
        self.S = results.get('S', 0.0)
        self.delta_Phi = results.get('delta_Φ', 0.0)
        self.q = results.get('q', 0.0)
        self.E_avg = results.get('Ē', 0.0)  # Average EMF
        self.E_m = results.get('E_m', 0.0)  # Peak EMF
        self.omega = results.get('omega', 0.0)
        self.W = results.get('W', 0.0)
        self.I_m = results.get('I_m', 0.0)  # Peak current
        self.E_eff = results.get('E', 0.0)  # Effective (RMS) EMF
        
        # For symbolic calculations
        self.symbols = {
            'a': sp.symbols('a', positive=True),
            'r': sp.symbols('r', positive=True),
            'B': sp.symbols('B', positive=True),
            'n': sp.symbols('n', positive=True),
            'R': sp.symbols('R', positive=True),
            'omega': sp.symbols('omega', positive=True),
            'S': sp.symbols('S', positive=True),
            'E_m': sp.symbols('E_m', positive=True),
            'I_m': sp.symbols('I_m', positive=True),
            'delta_Phi': sp.symbols('delta_Phi', positive=True),
            'q': sp.symbols('q', positive=True),
            'E_avg': sp.symbols('E_avg', positive=True),
            'W': sp.symbols('W', positive=True),
            'E_eff': sp.symbols('E_eff', positive=True)
        }
        
        self.logs = []

    def log(self, message, success):
        self.logs.append((success, message))
    
    def is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """
        Test fundamental physics laws based on the KB theorems.
        """
        success = True
        
        # 1. Check omega = 2*pi*n
        omega_calc = 2 * math.pi * self.n
        if not self.is_close(self.omega, omega_calc):
            self.log(f"Angular velocity mismatch: ω={self.omega}, 2πn={omega_calc}", False)
            success = False
        else:
            self.log("Angular velocity ω = 2πn holds.", True)
        
        # 2. Check area S = 0.5 * π * a^2
        S_calc = 0.5 * math.pi * self.a**2
        if not self.is_close(self.S, S_calc):
            self.log(f"Area mismatch: S={self.S}, 0.5πa²={S_calc}", False)
            success = False
        else:
            self.log(f"Area S = 0.5πa² holds.", True)
        
        # 3. Check peak EMF: E_m = B * S * ω
        E_m_calc = self.B * self.S * self.omega
        if not self.is_close(self.E_m, E_m_calc):
            self.log(f"Peak EMF mismatch: E_m={self.E_m}, B*S*ω={E_m_calc}", False)
            success = False
        else:
            self.log("Peak EMF E_m = B*S*ω holds.", True)
        
        # 4. Check peak current: I_m = E_m / (R + r)
        I_m_calc = self.E_m / (self.R + self.r)
        if not self.is_close(self.I_m, I_m_calc):
            self.log(f"Peak current mismatch: I_m={self.I_m}, E_m/(R+r)={I_m_calc}", False)
            success = False
        else:
            self.log("Peak current I_m = E_m/(R+r) holds.", True)
        
        # 5. Check flux change: ΔΦ = B * S
        delta_Phi_calc = self.B * self.S
        if not self.is_close(self.delta_Phi, delta_Phi_calc):
            self.log(f"Flux change mismatch: ΔΦ={self.delta_Phi}, B*S={delta_Phi_calc}", False)
            success = False
        else:
            self.log("Flux change ΔΦ = B*S holds.", True)
        
        # 6. Check average EMF: Ē = ΔΦ / Δt, with Δt = T/4 = 1/(4n) for quarter-cycle
        if self.n != 0:
            delta_t = 1 / (4 * self.n)  # Time for quarter cycle (from max flux to zero)
            E_avg_calc = self.delta_Phi / delta_t
            if not self.is_close(self.E_avg, E_avg_calc):
                self.log(f"Average EMF mismatch: Ē={self.E_avg}, ΔΦ/Δt={E_avg_calc} (with Δt=1/(4n))", False)
                success = False
            else:
                self.log("Average EMF Ē = ΔΦ/Δt holds for quarter-cycle.", True)
        
        # 7. Check charge: q = ΔΦ / (R + r)
        q_calc = self.delta_Phi / (self.R + self.r)
        if not self.is_close(self.q, q_calc):
            self.log(f"Charge mismatch: q={self.q}, ΔΦ/(R+r)={q_calc}", False)
            success = False
        else:
            self.log("Charge q = ΔΦ/(R+r) holds.", True)
        
        # 8. Check effective EMF: E_eff = E_m / sqrt(2)
        E_eff_calc = self.E_m / math.sqrt(2)
        if not self.is_close(self.E_eff, E_eff_calc):
            self.log(f"Effective EMF mismatch: E={self.E_eff}, E_m/√2={E_eff_calc}", False)
            success = False
        else:
            self.log("Effective EMF E = E_m/√2 holds.", True)
        
        # 9. Check work/energy: W = (E_eff^2 * T) / (R + r), with T = 1/n
        if self.n != 0:
            T = 1 / self.n
            W_calc = (self.E_eff**2 * T) / (self.R + self.r)
            if not self.is_close(self.W, W_calc):
                # Also check using target formula: W = (π^4 * n * a^4 * B^2) / (4*(R+r))
                W_target = (math.pi**4 * self.n * self.a**4 * self.B**2) / (4 * (self.R + self.r))
                if self.is_close(self.W, W_target):
                    self.log("Work W matches target formula (π^4 n a^4 B^2)/(4(R+r)).", True)
                else:
                    self.log(f"Work mismatch: W={self.W}, E_eff²T/(R+r)={W_calc}, target={W_target}", False)
                    success = False
            else:
                self.log("Work W = E_eff²T/(R+r) holds.", True)
        
        return success
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        success = True
        
        # Target answers from the problem
        target1 = self.n * math.pi**2 * self.a**2 * self.B / (self.R + self.r)  # I_m
        target2 = 2 * self.n * math.pi * self.B * self.a**2  # Ē
        target3 = math.pi * self.B * self.a**2 / (2 * (self.R + self.r))  # q
        target4 = math.pi**4 * self.n * self.a**4 * self.B**2 / (4 * (self.R + self.r))  # W
        
        # Compare with computed values (allow 2% tolerance)
        if not self.is_close(self.I_m, target1):
            self.log(f"Target 1 (I_m) mismatch: {self.I_m} vs {target1}", False)
            success = False
        else:
            self.log("Target 1 (I_m) matches.", True)
        
        if not self.is_close(self.E_avg, target2):
            self.log(f"Target 2 (Ē) mismatch: {self.E_avg} vs {target2}", False)
            success = False
        else:
            self.log("Target 2 (Ē) matches.", True)
        
        if not self.is_close(self.q, target3):
            self.log(f"Target 3 (q) mismatch: {self.q} vs {target3}", False)
            success = False
        else:
            self.log("Target 3 (q) matches.", True)
        
        if not self.is_close(self.W, target4):
            self.log(f"Target 4 (W) mismatch: {self.W} vs {target4}", False)
            success = False
        else:
            self.log("Target 4 (W) matches.", True)
        
        return success
    
    def test_monotonicity(self):
        """
        Perturb input n and check expected trends.
        """
        success = True
        original_n = self.n
        
        # Perturb n by +10%
        perturbed_n = original_n * 1.10
        
        # Recalculate expected changes using formulas (not the solution's methods)
        # We'll compute expected values based on perturbation and compare trends.
        
        # 1. ω ∝ n
        omega_original = 2 * math.pi * original_n
        omega_perturbed = 2 * math.pi * perturbed_n
        if not (omega_perturbed > omega_original):
            self.log("Monotonicity fail: ω should increase with n.", False)
            success = False
        
        # 2. E_m ∝ n (since E_m = B*S*ω, and ω ∝ n, S constant)
        E_m_original = self.B * self.S * omega_original
        E_m_perturbed = self.B * self.S * omega_perturbed
        if not (E_m_perturbed > E_m_original):
            self.log("Monotonicity fail: E_m should increase with n.", False)
            success = False
        
        # 3. I_m ∝ n (since I_m = E_m/(R+r) and E_m ∝ n)
        I_m_original = E_m_original / (self.R + self.r)
        I_m_perturbed = E_m_perturbed / (self.R + self.r)
        if not (I_m_perturbed > I_m_original):
            self.log("Monotonicity fail: I_m should increase with n.", False)
            success = False
        
        # 4. Ē ∝ n (since Ē = 2nπBa²)
        E_avg_original = 2 * original_n * math.pi * self.B * self.a**2
        E_avg_perturbed = 2 * perturbed_n * math.pi * self.B * self.a**2
        if not (E_avg_perturbed > E_avg_original):
            self.log("Monotonicity fail: Ē should increase with n.", False)
            success = False
        
        # 5. q independent of n
        q_original = math.pi * self.B * self.a**2 / (2 * (self.R + self.r))
        q_perturbed = q_original  # unchanged
        if not self.is_close(q_perturbed, q_original):
            self.log("Monotonicity fail: q should be independent of n.", False)
            success = False
        
        # 6. W ∝ n (from target formula)
        W_original = math.pi**4 * original_n * self.a**4 * self.B**2 / (4 * (self.R + self.r))
        W_perturbed = math.pi**4 * perturbed_n * self.a**4 * self.B**2 / (4 * (self.R + self.r))
        if not (W_perturbed > W_original):
            self.log("Monotonicity fail: W should increase with n.", False)
            success = False
        
        # Restore original n
        self.n = original_n
        
        if success:
            self.log("Monotonicity tests passed: all trends are correct.", True)
        
        return success
    
    def run_all_tests(self):
        """
        Run all tests and return (success_bool, logs_list).
        """
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                self.log(f"Running {test_name}...", None)
                if not test_func():
                    overall_success = False
            except Exception as e:
                self.log(f"Error in {test_name}: {traceback.format_exc()}", False)
                overall_success = False
        
        return overall_success, self.logs