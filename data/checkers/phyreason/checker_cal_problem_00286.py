import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract input parameters (safe access)
        self.g = 10.0
        self.MA = getattr(solution_instance, 'MA', getattr(solution_instance, 'M_A', math.sqrt(3)))
        self.MC = getattr(solution_instance, 'MC', getattr(solution_instance, 'M_C', 2 * math.sqrt(3)))
        self.MD = getattr(solution_instance, 'MD', getattr(solution_instance, 'M_D', 0.5))
        
        # Extract results with fallbacks (use results dict if available)
        if hasattr(solution_instance, 'results'):
            self.results = solution_instance.results
        else:
            self.results = {}
        
        self.m = getattr(solution_instance, 'm', self.results.get('m', 0.0))
        self.f = getattr(solution_instance, 'f', self.results.get('f', 0.0))
        self.N = getattr(solution_instance, 'N', self.results.get('N', 0.0))
        self.mu = getattr(solution_instance, 'mu', self.results.get('mu', 0.0))
        
        # Convert angles to radians
        self.theta_B = math.radians(60)  # Angle between ribbons at B
        self.theta_C = math.radians(30)  # Angle of ribbon at C
        
        # Store for monotonicity backup
        self._original_MA = self.MA
        
    def test_invariants(self):
        logs = []
        success = True
        
        try:
            # 1. Equilibrium of object B: 2*MA*g*cos(30°) = m*g
            T_A = self.MA * self.g  # Tension from block A
            left_B = 2 * T_A * math.cos(math.radians(30))
            right_B = self.m * self.g
            
            if not math.isclose(left_B, right_B, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: B equilibrium: 2*MA*g*cos30° = {left_B:.2f} ≠ m*g = {right_B:.2f}")
                success = False
            else:
                logs.append(f"PASS: B equilibrium: 2*MA*g*cos30° = m*g ≈ {left_B:.2f}")
            
            # 2. Horizontal equilibrium of object C: MA*g*cos30° = MD*g + f
            left_C = T_A * math.cos(self.theta_C)
            right_C = self.MD * self.g + self.f
            
            if not math.isclose(left_C, right_C, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: C horizontal equilibrium: T*cos30° = {left_C:.2f} ≠ MD*g + f = {right_C:.2f}")
                success = False
            else:
                logs.append(f"PASS: C horizontal equilibrium: T*cos30° = MD*g + f ≈ {left_C:.2f}")
            
            # 3. Vertical equilibrium of object C: N = MC*g - T*sin30°
            calc_N = self.MC * self.g - T_A * math.sin(self.theta_C)
            if not math.isclose(calc_N, self.N, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: C vertical equilibrium: N_calc = {calc_N:.2f} ≠ N_given = {self.N:.2f}")
                success = False
            else:
                logs.append(f"PASS: C vertical equilibrium: N = MC*g - T*sin30° ≈ {calc_N:.2f}")
            
            # 4. Friction definition: f = μ*N
            if self.N != 0:
                calc_mu = self.f / self.N
                target_mu = 2 * math.sqrt(3) / 9  # From expected answer
                # Check against both calculated and target
                if not (math.isclose(calc_mu, self.mu, rel_tol=1e-2, abs_tol=1e-2) and 
                        math.isclose(calc_mu, target_mu, rel_tol=1e-2, abs_tol=1e-2)):
                    logs.append(f"FAIL: Friction def: f/N = {calc_mu:.4f} ≠ μ_given = {self.mu:.4f} or target {target_mu:.4f}")
                    success = False
                else:
                    logs.append(f"PASS: Friction def: f = μ*N with μ ≈ {calc_mu:.4f}")
            
            # 5. Symbolic verification using pure numeric checks (no SymPy)
            # Equation 1: 2*MA*g*cos(30°) = m*g
            eq1_lhs = 2 * self.MA * self.g * math.cos(self.theta_C)
            eq1_rhs = self.m * self.g
            if not math.isclose(eq1_lhs, eq1_rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Numeric eq1: 2*MA*g*cos30° = {eq1_lhs:.2f} ≠ m*g = {eq1_rhs:.2f}")
                success = False
            else:
                logs.append("PASS: Numeric eq1 satisfied")
            
            # Equation 2: MA*g*cos30° = MD*g + μ*(MC*g - MA*g*sin30°)
            eq2_lhs = self.MA * self.g * math.cos(self.theta_C)
            eq2_rhs = self.MD * self.g + self.mu * (self.MC * self.g - self.MA * self.g * math.sin(self.theta_C))
            if not math.isclose(eq2_lhs, eq2_rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Numeric eq2: T*cos30° = {eq2_lhs:.2f} ≠ MD*g + μ*(MC*g - T*sin30°) = {eq2_rhs:.2f}")
                success = False
            else:
                logs.append("PASS: Numeric eq2 satisfied")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        logs = []
        success = True
        
        try:
            # Expected intermediate values from steps analysis
            expected_m = 3.0
            expected_f = 10.0
            expected_N = 15 * math.sqrt(3)  # ≈25.98
            expected_mu = 2 * math.sqrt(3) / 9  # ≈0.3849
            
            # Check against results
            checks = [
                ("m", self.m, expected_m),
                ("f", self.f, expected_f),
                ("N", self.N, expected_N),
                ("μ", self.mu, expected_mu)
            ]
            
            for name, value, expected in checks:
                if not math.isclose(value, expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: {name} = {value:.4f} ≠ expected {expected:.4f}")
                    success = False
                else:
                    logs.append(f"PASS: {name} = {value:.4f} matches expected")
            
            # Check internal consistency: f should equal μ*N
            if self.N != 0:
                calc_f = self.mu * self.N
                if not math.isclose(calc_f, self.f, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: μ*N = {calc_f:.2f} ≠ f = {self.f:.2f}")
                    success = False
                else:
                    logs.append(f"PASS: f = μ*N = {calc_f:.2f}")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        logs = []
        success = True
        
        try:
            # Perturb MA by +10%
            perturb_factor = 1.10
            MA_perturbed = self.MA * perturb_factor
            
            # Recalculate using physics formulas (not solution methods)
            # 1. New mass of B
            T_A_new = MA_perturbed * self.g
            m_new = 2 * MA_perturbed * math.cos(self.theta_C)
            
            # 2. New friction force f
            f_new = T_A_new * math.cos(self.theta_C) - self.MD * self.g
            
            # 3. New normal force N
            N_new = self.MC * self.g - T_A_new * math.sin(self.theta_C)
            
            # 4. New μ (if N ≠ 0)
            if N_new != 0:
                mu_new = f_new / N_new
            else:
                mu_new = float('inf')
            
            # Check trends (all should increase with MA)
            trends = [
                ("m", self.m, m_new, m_new > self.m),
                ("f", self.f, f_new, f_new > self.f),
                ("μ", self.mu, mu_new, mu_new > self.mu if N_new != 0 else True)
            ]
            
            for name, old_val, new_val, condition in trends:
                if condition:
                    logs.append(f"PASS: {name} increases from {old_val:.4f} to {new_val:.4f} when MA increases")
                else:
                    logs.append(f"FAIL: {name} does not increase (old={old_val:.4f}, new={new_val:.4f})")
                    success = False
            
            # N should decrease
            if N_new < self.N:
                logs.append(f"PASS: N decreases from {self.N:.4f} to {N_new:.4f}")
            else:
                logs.append(f"FAIL: N does not decrease (old={self.N:.4f}, new={N_new:.4f})")
                success = False
                
            # Restore original MA
            self.MA = self._original_MA
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            success = False
            # Restore on error
            self.MA = self._original_MA
            
        return success, logs
    
    def run_all_tests(self):
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False
        
        return overall_success, all_logs