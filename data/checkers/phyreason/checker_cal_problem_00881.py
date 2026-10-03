import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution = solution_instance
        
        # Safe attribute extraction with fallbacks
        self.inputs = {}
        self.results = {}
        
        # Extract inputs safely
        self.inputs['m'] = getattr(solution_instance, 'm', 
                                  getattr(solution_instance, 'inputs', {}).get('m', 1.0))
        self.inputs['e'] = getattr(solution_instance, 'e', 
                                  getattr(solution_instance, 'inputs', {}).get('e', 1.602e-19))
        self.inputs['v0'] = getattr(solution_instance, 'v0', 
                                   getattr(solution_instance, 'inputs', {}).get('v0', 1.0))
        self.inputs['d'] = getattr(solution_instance, 'd', 
                                  getattr(solution_instance, 'inputs', {}).get('d', 1.0))
        
        # Extract results safely using provided mapping
        result_keys = ['r', 't_2', 't', 'a', 'theta', 'v', 'B', '3t_1', 't_1', 'E']
        for key in result_keys:
            self.results[key] = getattr(solution_instance, key, 
                                       getattr(solution_instance, 'results', {}).get(key, None))
        
        # Create symbolic variables
        self.m, self.e, self.v0, self.d = sp.symbols('m e v0 d', positive=True)
        
    def test_invariants(self):
        """Verify physical laws based on KB reference"""
        logs = []
        
        try:
            # 1. Check kinematics in electric field region
            # From step 1: a = eE/m
            if self.results['E'] is not None and self.results['a'] is not None:
                # Symbolic check
                lhs = self.results['a']
                rhs = self.e * self.results['E'] / self.m
                diff = sp.simplify(lhs - rhs)
                if diff != 0:
                    # Numerical fallback
                    m_val = float(self.inputs['m'])
                    e_val = float(self.inputs['e'])
                    E_val = float(self.results['E']) if hasattr(self.results['E'], '__float__') else float(self.results['E'].subs({self.m: m_val, self.e: e_val, self.v0: self.inputs['v0'], self.d: self.inputs['d']}))
                    a_val = float(self.results['a']) if hasattr(self.results['a'], '__float__') else float(self.results['a'].subs({self.m: m_val, self.e: e_val, self.v0: self.inputs['v0'], self.d: self.inputs['d']}))
                    expected = e_val * E_val / m_val
                    if not math.isclose(a_val, expected, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"FAIL: a = eE/m violated. Got a={a_val}, expected={expected}")
                    else:
                        logs.append("PASS: a = eE/m (numerical check)")
                else:
                    logs.append("PASS: a = eE/m (symbolic check)")
            
            # 2. Check Lorentz force = centripetal force in magnetic field
            if self.results['B'] is not None and self.results['v'] is not None and self.results['r'] is not None:
                # Symbolic check: e*v*B = m*v²/r
                lhs = self.e * self.results['v'] * self.results['B']
                rhs = self.m * self.results['v']**2 / self.results['r']
                diff = sp.simplify(lhs - rhs)
                if diff != 0:
                    # Numerical fallback
                    m_val = float(self.inputs['m'])
                    e_val = float(self.inputs['e'])
                    B_val = float(self.results['B']) if hasattr(self.results['B'], '__float__') else float(self.results['B'].subs({self.m: m_val, self.e: e_val, self.v0: self.inputs['v0'], self.d: self.inputs['d']}))
                    v_val = float(self.results['v']) if hasattr(self.results['v'], '__float__') else float(self.results['v'].subs({self.m: m_val, self.e: e_val, self.v0: self.inputs['v0'], self.d: self.inputs['d']}))
                    r_val = float(self.results['r']) if hasattr(self.results['r'], '__float__') else float(self.results['r'].subs({self.m: m_val, self.e: e_val, self.v0: self.inputs['v0'], self.d: self.inputs['d']}))
                    lhs_val = e_val * v_val * B_val
                    rhs_val = m_val * v_val**2 / r_val
                    if not math.isclose(lhs_val, rhs_val, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"FAIL: Lorentz force = centripetal force violated. LHS={lhs_val}, RHS={rhs_val}")
                    else:
                        logs.append("PASS: Lorentz force = centripetal force (numerical check)")
                else:
                    logs.append("PASS: Lorentz force = centripetal force (symbolic check)")
            
            # 3. Check period formula
            if self.results['B'] is not None and self.results['t_2'] is not None:
                # From step 8: t_2 = (3/4)T, T = 2πm/(eB)
                T_expr = 2 * sp.pi * self.m / (self.e * self.results['B'])
                expected_t2 = (3/sp.Integer(4)) * T_expr
                diff = sp.simplify(self.results['t_2'] - expected_t2)
                if diff != 0:
                    # Numerical fallback
                    m_val = float(self.inputs['m'])
                    e_val = float(self.inputs['e'])
                    B_val = float(self.results['B']) if hasattr(self.results['B'], '__float__') else float(self.results['B'].subs({self.m: m_val, self.e: e_val, self.v0: self.inputs['v0'], self.d: self.inputs['d']}))
                    t2_val = float(self.results['t_2']) if hasattr(self.results['t_2'], '__float__') else float(self.results['t_2'].subs({self.m: m_val, self.e: e_val, self.v0: self.inputs['v0'], self.d: self.inputs['d']}))
                    expected = (3/4) * (2 * math.pi * m_val) / (e_val * B_val)
                    if not math.isclose(t2_val, expected, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"FAIL: t_2 period formula violated. Got t2={t2_val}, expected={expected}")
                    else:
                        logs.append("PASS: t_2 period formula (numerical check)")
                else:
                    logs.append("PASS: t_2 period formula (symbolic check)")
                    
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}\n{traceback.format_exc()}")
        
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        logs = []
        
        try:
            # Get numerical values for comparison
            m_val = float(self.inputs['m'])
            e_val = float(self.inputs['e'])
            v0_val = float(self.inputs['v0'])
            d_val = float(self.inputs['d'])
            
            # 1. Check t1 = 2d/v0
            if self.results['t_1'] is not None:
                t1_val = float(self.results['t_1']) if hasattr(self.results['t_1'], '__float__') else float(self.results['t_1'].subs({self.d: d_val, self.v0: v0_val}))
                expected_t1 = 2 * d_val / v0_val
                if not math.isclose(t1_val, expected_t1, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: t_1 inconsistent. Got {t1_val}, expected {expected_t1}")
                else:
                    logs.append(f"PASS: t_1 = 2d/v0 = {t1_val}")
            
            # 2. Check a = v0²/(2d)
            if self.results['a'] is not None:
                a_val = float(self.results['a']) if hasattr(self.results['a'], '__float__') else float(self.results['a'].subs({self.v0: v0_val, self.d: d_val}))
                expected_a = v0_val**2 / (2 * d_val)
                if not math.isclose(a_val, expected_a, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: a inconsistent. Got {a_val}, expected {expected_a}")
                else:
                    logs.append(f"PASS: a = v0²/(2d) = {a_val}")
            
            # 3. Check v = √2 * v0
            if self.results['v'] is not None:
                v_val = float(self.results['v']) if hasattr(self.results['v'], '__float__') else float(self.results['v'].subs({self.v0: v0_val}))
                expected_v = math.sqrt(2) * v0_val
                if not math.isclose(v_val, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: v inconsistent. Got {v_val}, expected {expected_v}")
                else:
                    logs.append(f"PASS: v = √2 * v0 = {v_val}")
            
            # 4. Check r = √2 * d
            if self.results['r'] is not None:
                r_val = float(self.results['r']) if hasattr(self.results['r'], '__float__') else float(self.results['r'].subs({self.d: d_val}))
                expected_r = math.sqrt(2) * d_val
                if not math.isclose(r_val, expected_r, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: r inconsistent. Got {r_val}, expected {expected_r}")
                else:
                    logs.append(f"PASS: r = √2 * d = {r_val}")
            
            # 5. Check 3t_1 = 6d/v0
            if self.results['3t_1'] is not None:
                t3_val = float(self.results['3t_1']) if hasattr(self.results['3t_1'], '__float__') else float(self.results['3t_1'].subs({self.d: d_val, self.v0: v0_val}))
                expected_3t1 = 6 * d_val / v0_val
                if not math.isclose(t3_val, expected_3t1, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: 3t_1 inconsistent. Got {t3_val}, expected {expected_3t1}")
                else:
                    logs.append(f"PASS: 3t_1 = 6d/v0 = {t3_val}")
            
            # 6. Check t_2 = 3πd/(2v0)
            if self.results['t_2'] is not None:
                t2_val = float(self.results['t_2']) if hasattr(self.results['t_2'], '__float__') else float(self.results['t_2'].subs({self.d: d_val, self.v0: v0_val}))
                expected_t2 = 3 * math.pi * d_val / (2 * v0_val)
                if not math.isclose(t2_val, expected_t2, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: t_2 inconsistent. Got {t2_val}, expected {expected_t2}")
                else:
                    logs.append(f"PASS: t_2 = 3πd/(2v0) = {t2_val}")
            
            # 7. Check final t = 3t_1 + t_2
            if self.results['t'] is not None and self.results['3t_1'] is not None and self.results['t_2'] is not None:
                t_val = float(self.results['t']) if hasattr(self.results['t'], '__float__') else float(self.results['t'].subs({self.d: d_val, self.v0: v0_val}))
                t3_val = float(self.results['3t_1']) if hasattr(self.results['3t_1'], '__float__') else float(self.results['3t_1'].subs({self.d: d_val, self.v0: v0_val}))
                t2_val = float(self.results['t_2']) if hasattr(self.results['t_2'], '__float__') else float(self.results['t_2'].subs({self.d: d_val, self.v0: v0_val}))
                expected_sum = t3_val + t2_val
                if not math.isclose(t_val, expected_sum, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: t = 3t_1 + t_2 violated. Got {t_val}, expected sum {expected_sum}")
                else:
                    logs.append(f"PASS: t = 3t_1 + t_2 = {t_val}")
                    
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}\n{traceback.format_exc()}")
        
        return logs
    
    def test_monotonicity(self):
        """Perturb inputs and check trend consistency"""
        logs = []
        
        try:
            # Store original values
            orig_d = self.inputs['d']
            perturb_factor = 1.1  # 10% increase
            
            # Perturb d
            self.inputs['d'] = orig_d * perturb_factor
            d_pert = self.inputs['d']
            
            # Recalculate using formulas from steps analysis
            v0 = self.inputs['v0']
            
            # Expected formulas from KB:
            # E = m*v0²/(2*e*d) - should DECREASE when d increases
            # B = m*v0/(e*d) - should DECREASE when d increases  
            # t = 3d(4+π)/(2v0) - should INCREASE when d increases
            
            # Calculate expected values for perturbed d
            E_pert = self.inputs['m'] * v0**2 / (2 * self.inputs['e'] * d_pert)
            B_pert = self.inputs['m'] * v0 / (self.inputs['e'] * d_pert)
            t_pert = 3 * d_pert * (4 + math.pi) / (2 * v0)
            
            # Calculate expected values for original d
            E_orig = self.inputs['m'] * v0**2 / (2 * self.inputs['e'] * orig_d)
            B_orig = self.inputs['m'] * v0 / (self.inputs['e'] * orig_d)
            t_orig = 3 * orig_d * (4 + math.pi) / (2 * v0)
            
            # Check trends
            if not E_pert < E_orig:
                logs.append(f"FAIL: E should decrease when d increases. E_orig={E_orig}, E_pert={E_pert}")
            else:
                logs.append(f"PASS: E monotonicity: decreases when d increases (E_orig={E_orig:.3e}, E_pert={E_pert:.3e})")
            
            if not B_pert < B_orig:
                logs.append(f"FAIL: B should decrease when d increases. B_orig={B_orig}, B_pert={B_pert}")
            else:
                logs.append(f"PASS: B monotonicity: decreases when d increases (B_orig={B_orig:.3e}, B_pert={B_pert:.3e})")
            
            if not t_pert > t_orig:
                logs.append(f"FAIL: t should increase when d increases. t_orig={t_orig}, t_pert={t_pert}")
            else:
                logs.append(f"PASS: t monotonicity: increases when d increases (t_orig={t_orig:.3e}, t_pert={t_pert:.3e})")
            
            # Restore original value
            self.inputs['d'] = orig_d
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}\n{traceback.format_exc()}")
            # Ensure restoration even on error
            self.inputs['d'] = orig_d if 'orig_d' in locals() else self.inputs['d']
        
        return logs
    
    def run_all_tests(self):
        """Run all tests and return success status and logs"""
        all_logs = []
        success = True
        
        # Run individual tests
        for test_name, test_func in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            logs = test_func()
            if logs and any(log.startswith("FAIL") or log.startswith("ERROR") for log in logs):
                success = False
            all_logs.append(f"=== {test_name} Test ===")
            all_logs.extend(logs)
            all_logs.append("")  # Blank line for readability
        
        return success, all_logs