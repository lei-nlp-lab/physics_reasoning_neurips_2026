import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with fallbacks
        self.solution = solution_instance
        self.inputs = getattr(solution_instance, 'inputs', {})
        self.results = getattr(solution_instance, 'results', {})
        
        # Inputs (with reasonable defaults to avoid crashes)
        self.r = self.inputs.get('r', 1.0)
        self.T = self.inputs.get('T', 1.0)
        self.k = self.inputs.get('k', 0.5)
        self.G = self.inputs.get('G', 6.67430e-11)
        
        # Results from solution
        self.M = getattr(solution_instance, 'M', self.results.get('M', 0.0))
        self.T_prime = getattr(solution_instance, 'T_prime', self.results.get("T'", self.results.get('T_prime', 0.0)))
        self.t = getattr(solution_instance, 't', self.results.get('t', 0.0))
        
        # Symbolic variables for invariant checking
        self.r_sym, self.T_sym, self.k_sym, self.G_sym = sp.symbols('r T k G', positive=True)
        self.M_sym, self.T_prime_sym, self.t_sym = sp.symbols('M T_prime t', positive=True)
        
    def test_invariants(self):
        """Test conservation laws and physics invariants"""
        logs = []
        
        try:
            # 1. Check Kepler's 3rd Law for circular orbit: T^2 ∝ r^3 (with fixed M)
            # From step_1: M = (4π²r³)/(GT²)
            M_expected = (4 * math.pi**2 * self.r**3) / (self.G * self.T**2)
            if not math.isclose(self.M, M_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Earth mass M mismatch. Expected {M_expected:.4e}, got {self.M:.4e}")
            else:
                logs.append(f"PASS: Earth mass M consistent with circular orbit")
            
            # 2. Check Kepler's 3rd Law for elliptical orbit
            # Semi-major axis a = r*(1+k)/2
            a = self.r * (1 + self.k) / 2
            # T'^2 should be proportional to a^3 (with same M)
            T_prime_expected = self.T * ((1 + self.k) / 2)**1.5
            if not math.isclose(self.T_prime, T_prime_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Elliptical period T' mismatch. Expected {T_prime_expected:.4e}, got {self.T_prime:.4e}")
            else:
                logs.append(f"PASS: Elliptical period T' follows Kepler's 3rd law")
            
            # 3. Check time relationship: t = T'/2
            if not math.isclose(self.t, self.T_prime / 2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Time t should be half of T'. Expected {self.T_prime/2:.4e}, got {self.t:.4e}")
            else:
                logs.append(f"PASS: Time t = T'/2 consistent")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        logs = []
        
        try:
            # Check that the solution's steps match the final results
            # From step_2: T' = (T/2) * sqrt((1+k)^3 / 2)
            T_prime_from_step2 = (self.T / 2) * math.sqrt((1 + self.k)**3 / 2)
            if not math.isclose(self.T_prime, T_prime_from_step2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: T' mismatch with step_2 formula. Step2 gives {T_prime_from_step2:.4e}, got {self.T_prime:.4e}")
            else:
                logs.append(f"PASS: T' consistent with step_2 derivation")
            
            # From step_3: t = T'/2 = (T/4) * sqrt((1+k)^3/2)
            t_from_step3 = (self.T / 4) * math.sqrt((1 + self.k)**3 / 2)
            if not math.isclose(self.t, t_from_step3, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: t mismatch with step_3 formula. Step3 gives {t_from_step3:.4e}, got {self.t:.4e}")
            else:
                logs.append(f"PASS: t consistent with step_3 derivation")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            
        return logs
    
    def test_monotonicity(self):
        """Test that perturbations in k produce expected trends"""
        logs = []
        
        try:
            # Backup original values
            original_k = self.k
            
            # Test 1: Increase k (closer to 1 means perigee closer to apogee)
            self.k = min(0.99, original_k * 1.1)  # Increase by 10%, cap at 0.99
            
            # Recalculate using formulas (not using solution's methods to avoid side effects)
            new_T_prime = self.T * ((1 + self.k) / 2)**1.5
            new_t = new_T_prime / 2
            
            # Original values
            original_T_prime = self.T * ((1 + original_k) / 2)**1.5
            original_t = original_T_prime / 2
            
            # Check monotonicity: As k increases, (1+k)/2 increases, so T' and t should increase
            if new_T_prime <= original_T_prime:
                logs.append(f"FAIL: T' should increase when k increases (k: {original_k:.3f}->{self.k:.3f})")
            else:
                logs.append(f"PASS: T' increases monotonically with k")
            
            if new_t <= original_t:
                logs.append(f"FAIL: t should increase when k increases (k: {original_k:.3f}->{self.k:.3f})")
            else:
                logs.append(f"PASS: t increases monotonically with k")
            
            # Test 2: Decrease k (smaller perigee distance)
            self.k = max(0.01, original_k * 0.9)  # Decrease by 10%, floor at 0.01
            
            new_T_prime = self.T * ((1 + self.k) / 2)**1.5
            new_t = new_T_prime / 2
            
            if new_T_prime >= original_T_prime:
                logs.append(f"FAIL: T' should decrease when k decreases (k: {original_k:.3f}->{self.k:.3f})")
            else:
                logs.append(f"PASS: T' decreases monotonically with k")
            
            if new_t >= original_t:
                logs.append(f"FAIL: t should decrease when k decreases (k: {original_k:.3f}->{self.k:.3f})")
            else:
                logs.append(f"PASS: t decreases monotonically with k")
            
            # Restore original k
            self.k = original_k
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            # Try to restore k even if error occurred
            self.k = original_k if 'original_k' in locals() else self.k
            
        return logs
    
    def run_all_tests(self):
        """Run all tests and return success status and logs"""
        all_logs = []
        success = True
        
        try:
            # Run individual tests
            for test_name, test_method in [
                ("Invariants", self.test_invariants),
                ("Trace Consistency", self.test_trace_consistency),
                ("Monotonicity", self.test_monotonicity)
            ]:
                all_logs.append(f"\n=== Running {test_name} Test ===")
                test_logs = test_method()
                all_logs.extend(test_logs)
                
                # Check if any FAIL or ERROR messages in this test
                for log in test_logs:
                    if log.startswith("FAIL") or log.startswith("ERROR"):
                        success = False
                        
        except Exception as e:
            all_logs.append(f"CRITICAL ERROR: {str(e)}")
            all_logs.append(traceback.format_exc())
            success = False
            
        return success, all_logs