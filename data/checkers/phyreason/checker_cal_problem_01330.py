import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallbacks
        self.solution = solution_instance
        self.logs = []
        
        # Extract input parameters
        self.P_rated = 80e3  # 80 kW to W (from problem statement)
        self.v_max = 20.0    # m/s
        self.m = 2000.0      # kg
        self.a = 2.0         # m/s²
        
        # Extract results safely (using KB mapping)
        self.F_f = self._safe_get('F_f')
        self.F = self._safe_get('F')
        self.t = self._safe_get('t')
        self.v = self._safe_get('v')
        self.F_prime = self._safe_get('F_prime', alt_key="F'")
        self.a_prime = self._safe_get('a_prime', alt_key="a'")
        self.v_1 = self._safe_get('v_1')
        self.P = self._safe_get('P')
        
        # Define sympy symbols for symbolic checks
        self.P_sym = sp.symbols('P')
        self.F_sym = sp.symbols('F')
        self.F_f_sym = sp.symbols('F_f')
        self.m_sym = sp.symbols('m')
        self.a_sym = sp.symbols('a')
        self.v_sym = sp.symbols('v')
        self.v_max_sym = sp.symbols('v_max')
        self.F_prime_sym = sp.symbols('F_prime')
        self.a_prime_sym = sp.symbols('a_prime')
        self.v_1_sym = sp.symbols('v_1')
        self.t_sym = sp.symbols('t')
        
    def _safe_get(self, attr, alt_key=None):
        """Safe extraction of attributes/results with fallback to 0.0"""
        # Try direct attribute
        val = getattr(self.solution, attr, None)
        if val is not None:
            return float(val)
        
        # Try results dictionary
        if hasattr(self.solution, 'results'):
            val = self.solution.results.get(attr, None)
            if val is not None:
                return float(val)
            
            # Try alternative key
            if alt_key:
                val = self.solution.results.get(alt_key, None)
                if val is not None:
                    return float(val)
        
        # Try common variations
        variations = [
            attr.lower(),
            attr.upper(),
            attr.replace('_', ''),
            attr.replace('_', ' ')
        ]
        
        if hasattr(self.solution, 'results'):
            for var in variations:
                val = self.solution.results.get(var, None)
                if val is not None:
                    return float(val)
        
        self.logs.append(f"Warning: Could not find {attr}, using default 0.0")
        return 0.0
    
    def test_invariants(self):
        """Test fundamental physics laws"""
        logs = []
        success = True
        
        try:
            # 1. Equilibrium at maximum speed: F_f = P_rated / v_max
            expected_F_f = self.P_rated / self.v_max
            if not math.isclose(self.F_f, expected_F_f, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Resistance force mismatch. Got {self.F_f}, expected {expected_F_f}")
                success = False
            else:
                logs.append(f"PASS: Resistance force {self.F_f} ≈ {expected_F_f}")
            
            # 2. Newton's 2nd Law during acceleration: F = F_f + m*a
            expected_F = self.F_f + self.m * self.a
            if not math.isclose(self.F, expected_F, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Traction force mismatch. Got {self.F}, expected {expected_F}")
                success = False
            else:
                logs.append(f"PASS: Traction force {self.F} ≈ {expected_F}")
            
            # 3. Power formula: v = P_rated / F
            if self.F != 0:
                expected_v = self.P_rated / self.F
                if not math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Velocity at rated power mismatch. Got {self.v}, expected {expected_v}")
                    success = False
                else:
                    logs.append(f"PASS: Velocity {self.v} ≈ {expected_v}")
            
            # 4. Kinematics: t = v / a
            if self.a != 0:
                expected_t = self.v / self.a
                if not math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Time mismatch. Got {self.t}, expected {expected_t}")
                    success = False
                else:
                    logs.append(f"PASS: Time {self.t} ≈ {expected_t}")
            
            # 5. Instantaneous power at t=3s: P = F * v_1
            expected_P = self.F * self.v_1
            if not math.isclose(self.P, expected_P, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Instantaneous power mismatch. Got {self.P}, expected {expected_P}")
                success = False
            else:
                logs.append(f"PASS: Instantaneous power {self.P} ≈ {expected_P}")
            
            # 6. Traction force at 16 m/s: F' = P_rated / 16
            expected_F_prime = self.P_rated / 16.0
            if not math.isclose(self.F_prime, expected_F_prime, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Traction force at 16 m/s mismatch. Got {self.F_prime}, expected {expected_F_prime}")
                success = False
            else:
                logs.append(f"PASS: Traction force at 16 m/s {self.F_prime} ≈ {expected_F_prime}")
            
            # 7. Newton's 2nd Law at 16 m/s: a' = (F' - F_f) / m
            expected_a_prime = (self.F_prime - self.F_f) / self.m
            if not math.isclose(self.a_prime, expected_a_prime, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Acceleration at 16 m/s mismatch. Got {self.a_prime}, expected {expected_a_prime}")
                success = False
            else:
                logs.append(f"PASS: Acceleration at 16 m/s {self.a_prime} ≈ {expected_a_prime}")
            
            # 8. Kinematics at t=3s: v_1 = a * 3
            expected_v_1 = self.a * 3.0
            if not math.isclose(self.v_1, expected_v_1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Velocity at t=3s mismatch. Got {self.v_1}, expected {expected_v_1}")
                success = False
            else:
                logs.append(f"PASS: Velocity at t=3s {self.v_1} ≈ {expected_v_1}")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Check internal consistency of intermediate results"""
        logs = []
        success = True
        
        try:
            # Check that all steps lead to the target answers
            target_answers = [4000.0, 5.0, 48000.0, 0.5]
            computed_answers = [self.F_f, self.t, self.P, self.a_prime]
            
            for i, (computed, target) in enumerate(zip(computed_answers, target_answers)):
                if not math.isclose(computed, target, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Answer {i+1} mismatch. Got {computed}, expected {target}")
                    success = False
                else:
                    logs.append(f"PASS: Answer {i+1} {computed} ≈ {target}")
                    
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Test that perturbations produce expected trends"""
        logs = []
        success = True
        
        try:
            # Store original values
            original_m = self.m
            
            # Perturbation 1: Increase mass by 10%
            self.m = original_m * 1.1
            
            # Recompute derived quantities using formulas
            F_f_new = self.P_rated / self.v_max  # Unchanged
            F_new = F_f_new + self.m * self.a    # Should increase
            v_new = self.P_rated / F_new if F_new != 0 else 0  # Should decrease
            t_new = v_new / self.a if self.a != 0 else 0  # Should decrease
            v_1_new = self.a * 3.0  # Unchanged (kinematics)
            P_new = F_new * v_1_new  # Should increase (F increased, v_1 unchanged)
            F_prime_new = self.P_rated / 16.0  # Unchanged
            a_prime_new = (F_prime_new - F_f_new) / self.m  # Should decrease
            
            # Check trends
            if not (F_new > self.F):
                logs.append(f"FAIL: Traction force didn't increase with mass increase")
                success = False
            if not (v_new < self.v):
                logs.append(f"FAIL: Velocity at rated power didn't decrease with mass increase")
                success = False
            if not (t_new < self.t):
                logs.append(f"FAIL: Time to reach rated power didn't decrease with mass increase")
                success = False
            if not (P_new > self.P):
                logs.append(f"FAIL: Instantaneous power didn't increase with mass increase")
                success = False
            if not (a_prime_new < self.a_prime):
                logs.append(f"FAIL: Acceleration at 16 m/s didn't decrease with mass increase")
                success = False
                
            if success:
                logs.append("PASS: All monotonicity trends as expected")
            
            # Restore original mass
            self.m = original_m
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            # Restore original mass on error
            if 'original_m' in locals():
                self.m = original_m
        
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return summary"""
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} ===")
            success, logs = test_func()
            all_logs.extend(logs)
            if not success:
                overall_success = False
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs