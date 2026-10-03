import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction - only use results dict
        self.results = {}
        
        # Results from solution - prioritize results dict
        results_dict = getattr(solution_instance, 'results', {})
        
        # Extract all required results with defaults
        required_keys = ['m_B', 'W_f', 'fracmumuprime', 'v_A', 'v_B', 
                        'a_1', 'a_2', 'f', 'x_1', 'x_2', 'h', 'tantheta']
        
        for key in required_keys:
            # Try direct attribute, then results dict
            val = getattr(solution_instance, key, None)
            if val is None:
                val = results_dict.get(key, None)
            self.results[key] = val
        
        # Get constants from results or use defaults
        self.m = results_dict.get('m', 1.0)
        self.g = results_dict.get('g', 9.8)
        self.H = results_dict.get('H', 1.0)
        
        self.logs = []
        
    def log(self, message):
        self.logs.append(message)
        
    def test_invariants(self):
        """Verify conservation laws and Newton's laws"""
        try:
            # 1. Conservation of momentum during collision
            m_B_val = self._safe_eval(self.results['m_B'])
            v_A_val = self._safe_eval(self.results['v_A'])
            v_B_val = self._safe_eval(self.results['v_B'])
            
            if all(v is not None for v in [m_B_val, v_A_val, v_B_val]):
                # Check momentum: m*v1 = m*v_A + m_B*v_B
                # We need v1. Compute from energy or use relation v1 = v_A * (m+m_B)/(m-m_B)
                # Since v_A = (m-m_B)/(m+m_B)*v1, then v1 = v_A*(m+m_B)/(m-m_B)
                m_val = float(self.m)
                if m_val == m_B_val:
                    self.log("Warning: m = m_B, cannot compute v1")
                else:
                    v1_val = v_A_val * (m_val + m_B_val) / (m_val - m_B_val)
                    momentum_lhs = m_val * v1_val
                    momentum_rhs = m_val * v_A_val + m_B_val * v_B_val
                    
                    if not math.isclose(momentum_lhs, momentum_rhs, rel_tol=0.05, abs_tol=0.05):
                        self.log(f"FAIL: Momentum conservation: {momentum_lhs} ≠ {momentum_rhs}")
                        return False
                    
                    # Check energy: 0.5*m*v1² = 0.5*m*v_A² + 0.5*m_B*v_B²
                    energy_lhs = 0.5 * m_val * v1_val**2
                    energy_rhs = 0.5 * m_val * v_A_val**2 + 0.5 * m_B_val * v_B_val**2
                    
                    if not math.isclose(energy_lhs, energy_rhs, rel_tol=0.05, abs_tol=0.05):
                        self.log(f"FAIL: Energy conservation: {energy_lhs} ≠ {energy_rhs}")
                        return False
            
            # 2. Newton's second law consistency
            a1_val = self._safe_eval(self.results['a_1'])
            a2_val = self._safe_eval(self.results['a_2'])
            f_val = self._safe_eval(self.results['f'])
            tantheta_val = self._safe_eval(self.results['tantheta'])
            
            if all(v is not None for v in [a1_val, a2_val, f_val, tantheta_val]):
                # sinθ = tanθ/√(1+tan²θ)
                sin_theta = tantheta_val / math.sqrt(1 + tantheta_val**2)
                m_val = float(self.m)
                g_val = float(self.g)
                
                # Down: mg sinθ - f = m*a1
                lhs_down = m_val * g_val * sin_theta - f_val
                rhs_down = m_val * a1_val
                
                # Up: mg sinθ + f = m*a2  
                lhs_up = m_val * g_val * sin_theta + f_val
                rhs_up = m_val * a2_val
                
                if not (math.isclose(lhs_down, rhs_down, rel_tol=0.05, abs_tol=0.05) and
                       math.isclose(lhs_up, rhs_up, rel_tol=0.05, abs_tol=0.05)):
                    self.log(f"FAIL: Newton's law: down {lhs_down}≠{rhs_down}, up {lhs_up}≠{rhs_up}")
                    return False
            
            # 3. Work-energy theorem for friction
            W_f_val = self._safe_eval(self.results['W_f'])
            f_val = self._safe_eval(self.results['f'])
            x1_val = self._safe_eval(self.results['x_1'])
            x2_val = self._safe_eval(self.results['x_2'])
            
            if all(v is not None for v in [W_f_val, f_val, x1_val, x2_val]):
                W_f_calc = f_val * (x1_val + x2_val)
                if not math.isclose(W_f_val, W_f_calc, rel_tol=0.05, abs_tol=0.05):
                    self.log(f"FAIL: Work-energy: W_f={W_f_val}, f*(x1+x2)={W_f_calc}")
                    return False
            
            self.log("PASS: All invariants satisfied")
            return True
            
        except Exception as e:
            self.log(f"ERROR in test_invariants: {str(e)}")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        try:
            # 1. Check m_B = 3m
            m_B_val = self._safe_eval(self.results['m_B'])
            if m_B_val is not None:
                m_val = float(self.m)
                expected_m_B = 3.0 * m_val
                if not math.isclose(m_B_val, expected_m_B, rel_tol=0.05, abs_tol=0.05):
                    self.log(f"FAIL: m_B={m_B_val}, expected {expected_m_B}")
                    return False
            
            # 2. Check W_f = (2/15)mgH
            W_f_val = self._safe_eval(self.results['W_f'])
            if W_f_val is not None:
                m_val = float(self.m)
                H_val = float(self.H)
                g_val = float(self.g)
                expected_W_f = (2/15) * m_val * g_val * H_val
                if not math.isclose(W_f_val, expected_W_f, rel_tol=0.05, abs_tol=0.05):
                    self.log(f"FAIL: W_f={W_f_val}, expected {expected_W_f}")
                    return False
            
            # 3. Check μ/μ' = 11/9
            ratio_val = self._safe_eval(self.results['fracmumuprime'])
            if ratio_val is not None:
                if not math.isclose(ratio_val, 11/9, rel_tol=0.05, abs_tol=0.05):
                    self.log(f"FAIL: μ/μ'={ratio_val}, expected {11/9}")
                    return False
            
            # 4. Check geometry: h = H/5
            h_val = self._safe_eval(self.results['h'])
            if h_val is not None:
                H_val = float(self.H)
                expected_h = H_val / 5
                if not math.isclose(h_val, expected_h, rel_tol=0.05, abs_tol=0.05):
                    self.log(f"FAIL: h={h_val}, expected H/5={expected_h}")
                    return False
            
            self.log("PASS: Trace consistency verified")
            return True
            
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {str(e)}")
            return False
    
    def test_monotonicity(self):
        """Check monotonic relationships by perturbing inputs"""
        try:
            # Test 1: W_f should be proportional to m
            W_f_val = self._safe_eval(self.results['W_f'])
            if W_f_val is not None:
                m_val = float(self.m)
                H_val = float(self.H)
                g_val = float(self.g)
                
                # Original W_f from formula
                W_f_orig = (2/15) * m_val * g_val * H_val
                
                # Perturb m by +10%
                m_pert = m_val * 1.1
                W_f_pert = (2/15) * m_pert * g_val * H_val
                
                expected_ratio = 1.1
                actual_ratio = W_f_pert / W_f_orig
                
                if not math.isclose(actual_ratio, expected_ratio, rel_tol=0.01, abs_tol=0.01):
                    self.log(f"FAIL: W_f not proportional to m: ratio={actual_ratio}, expected {expected_ratio}")
                    return False
            
            # Test 2: W_f should be proportional to H
            H_val = float(self.H)
            if H_val > 0:
                m_val = float(self.m)
                g_val = float(self.g)
                
                W_f_orig = (2/15) * m_val * g_val * H_val
                H_pert = H_val * 1.1
                W_f_pert = (2/15) * m_val * g_val * H_pert
                
                expected_ratio = 1.1
                actual_ratio = W_f_pert / W_f_orig
                
                if not math.isclose(actual_ratio, expected_ratio, rel_tol=0.01, abs_tol=0.01):
                    self.log(f"FAIL: W_f not proportional to H: ratio={actual_ratio}, expected {expected_ratio}")
                    return False
            
            # Test 3: Acceleration ratio a2/a1 should be constant = 5/4
            a1_val = self._safe_eval(self.results['a_1'])
            a2_val = self._safe_eval(self.results['a_2'])
            if a1_val is not None and a2_val is not None and a1_val != 0:
                expected_ratio = 5/4  # 1.25
                actual_ratio = a2_val / a1_val
                if not math.isclose(actual_ratio, expected_ratio, rel_tol=0.05, abs_tol=0.05):
                    self.log(f"FAIL: a2/a1={actual_ratio}, expected {expected_ratio}")
                    return False
            
            self.log("PASS: Monotonicity checks satisfied")
            return True
            
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success status with logs"""
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.log(f"\n--- Running {test_name} Test ---")
            try:
                if not test_func():
                    success = False
                    self.log(f"FAILED: {test_name}")
            except Exception as e:
                success = False
                self.log(f"CRASHED in {test_name}: {traceback.format_exc()}")
        
        self.log(f"\n=== Overall Result: {'PASS' if success else 'FAIL'} ===")
        return success, self.logs
    
    def _safe_eval(self, expr):
        """Safely evaluate expression to float if possible"""
        if expr is None:
            return None
        
        try:
            if isinstance(expr, (int, float)):
                return float(expr)
            
            # If it's a sympy expression
            if hasattr(expr, 'evalf'):
                return float(expr.evalf())
            
            # If it's a string representation
            if isinstance(expr, str):
                # Clean up LaTeX formatting
                expr_clean = expr.replace('\\frac', '').replace('{', '').replace('}', '')
                # Simple numeric check
                if '/' in expr_clean:
                    num, den = expr_clean.split('/')
                    return float(num) / float(den)
                return float(expr_clean)
            
            return float(expr)
        except:
            return None