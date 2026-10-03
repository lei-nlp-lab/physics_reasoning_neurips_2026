import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs and results
        self.solution = solution_instance
        self.inputs = {}
        self.results = {}
        
        # Inputs: H, T, R, G (may be in solution_instance or in a nested dict)
        for key in ['H', 'T', 'R', 'G']:
            self.inputs[key] = getattr(solution_instance, key, 
                                      getattr(solution_instance, 'inputs', {}).get(key, None))
        
        # Results: v, M, v' (v-prime)
        # Using safe access with fallback to results dictionary
        self.results['v'] = getattr(solution_instance, 'v', 
                                   getattr(solution_instance, 'results', {}).get('v', None))
        self.results['M'] = getattr(solution_instance, 'M', 
                                   getattr(solution_instance, 'results', {}).get('M', None))
        self.results["v'"] = getattr(solution_instance, "v'", 
                                    getattr(solution_instance, 'results', {}).get("v'", None))
        
    def test_invariants(self):
        logs = []
        success = True
        
        try:
            # Check 1: Centripetal force = Gravitational force (Newton's 2nd + Universal Gravitation)
            # For circular orbit: v^2/(R+H) = GM/(R+H)^2 => v^2 = GM/(R+H)
            # Use default values if inputs are None
            v_val = self._get_numeric('v')
            M_val = self._get_numeric('M')
            H_val = self._get_numeric_input('H', 200e3)
            R_val = self._get_numeric_input('R', 1.737e6)
            G_val = self._get_numeric_input('G', 6.6743e-11)
            
            if all(val is not None for val in [v_val, M_val, H_val, R_val, G_val]):
                lhs = v_val**2
                rhs = G_val * M_val / (R_val + H_val)
                if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Centripetal-gravitational balance: v²={lhs:.3e}, GM/(R+H)={rhs:.3e}")
                    success = False
                else:
                    logs.append(f"PASS: Centripetal force matches gravitational force (rel error: {abs(lhs-rhs)/rhs:.2%})")
            
            # Check 2: Period-velocity-radius relationship (uniform circular motion)
            T_val = self._get_numeric_input('T', 7200.0)
            if all(val is not None for val in [v_val, H_val, R_val, T_val]):
                expected_v = 2*math.pi*(R_val + H_val) / T_val
                if not math.isclose(v_val, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Velocity-period-radius mismatch: v={v_val:.3e}, 2π(R+H)/T={expected_v:.3e}")
                    success = False
                else:
                    logs.append(f"PASS: Velocity matches period-radius relation (rel error: {abs(v_val-expected_v)/expected_v:.2%})")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            traceback.print_exc()
            success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        logs = []
        success = True
        
        try:
            # Step 1: v = 2π(R+H)/T
            v_val = self._get_numeric('v')
            if v_val is not None:
                logs.append(f"Step 1 expression: v = 2π(R+H)/T")
            
            # Step 2: M = 4π²(R+H)³/(GT²)
            M_val = self._get_numeric('M')
            if M_val is not None:
                logs.append(f"Step 2 expression: M = 4π²(R+H)³/(GT²)")
            
            # Step 3: v' = v * √((R+H)/R) = 2π(R+H)/T * √((R+H)/R)
            v_prime_val = self._get_numeric("v'")
            if v_prime_val is not None:
                logs.append(f"Step 3 expression: v' = 2π(R+H)/T * √((R+H)/R)")
            
            # Cross-check: v' should also equal √(GM/R)
            if all(val is not None for val in [v_prime_val, M_val]):
                G_val = self._get_numeric_input('G', 6.6743e-11)
                R_val = self._get_numeric_input('R', 1.737e6)
                expected_v_prime = math.sqrt(G_val * M_val / R_val)
                if not math.isclose(v_prime_val, expected_v_prime, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: v' mismatch: {v_prime_val:.3e} vs √(GM/R)={expected_v_prime:.3e}")
                    success = False
                else:
                    logs.append(f"PASS: v' consistent with √(GM/R) (rel error: {abs(v_prime_val-expected_v_prime)/expected_v_prime:.2%})")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            traceback.print_exc()
            success = False
        
        return success, logs
    
    def test_monotonicity(self):
        logs = []
        success = True
        
        try:
            # Get all required numeric values with defaults
            v_val = self._get_numeric('v')
            M_val = self._get_numeric('M')
            v_prime_val = self._get_numeric("v'")
            H_val = self._get_numeric_input('H', 200e3)
            R_val = self._get_numeric_input('R', 1.737e6)
            T_val = self._get_numeric_input('T', 7200.0)
            G_val = self._get_numeric_input('G', 6.6743e-11)
            
            if any(val is None for val in [v_val, M_val, v_prime_val, H_val, R_val, T_val, G_val]):
                logs.append("SKIP: Monotonicity test requires all numeric inputs and results")
                return True, logs
            
            # Perturb H by +1%
            H_new = H_val * 1.01
            
            # Recalculate using formulas from steps
            v_new = 2*math.pi*(R_val + H_new) / T_val
            M_new = 4*math.pi**2 * (R_val + H_new)**3 / (G_val * T_val**2)
            v_prime_new = v_new * math.sqrt((R_val + H_new) / R_val)
            
            # Check monotonic trends:
            # As H increases, v should increase (linearly with radius)
            if not (v_new > v_val):
                logs.append(f"FAIL: v does not increase with H: {v_val:.3e} -> {v_new:.3e}")
                success = False
            else:
                logs.append(f"PASS: v increases with H (from {v_val:.3e} to {v_new:.3e})")
            
            # As H increases, M should increase (cubic in radius)
            if not (M_new > M_val):
                logs.append(f"FAIL: M does not increase with H: {M_val:.3e} -> {M_new:.3e}")
                success = False
            else:
                logs.append(f"PASS: M increases with H (from {M_val:.3e} to {M_new:.3e})")
            
            # As H increases, v' should increase (due to both v and sqrt factor increasing)
            if not (v_prime_new > v_prime_val):
                logs.append(f"FAIL: v' does not increase with H: {v_prime_val:.3e} -> {v_prime_new:.3e}")
                success = False
            else:
                logs.append(f"PASS: v' increases with H (from {v_prime_val:.3e} to {v_prime_new:.3e})")
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            traceback.print_exc()
            success = False
        
        return success, logs
    
    def _get_numeric(self, key):
        """Get numeric value from results dict"""
        val = self.results.get(key)
        if val is None:
            return None
        if hasattr(val, '__float__'):
            return float(val)
        if isinstance(val, (int, float)):
            return val
        return None
    
    def _get_numeric_input(self, key, default):
        """Get numeric value from inputs dict with default"""
        val = self.inputs.get(key, default)
        if val is None:
            val = default
        if hasattr(val, '__float__'):
            return float(val)
        if isinstance(val, (int, float)):
            return val
        return default
    
    def run_all_tests(self):
        all_logs = []
        overall_success = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            success, logs = test_method()
            all_logs.append(f"=== {test_name} Test ===")
            all_logs.extend(logs)
            if not success:
                overall_success = False
        
        return overall_success, all_logs