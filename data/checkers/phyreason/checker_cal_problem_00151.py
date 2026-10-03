import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction for inputs
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', None))
        self.q = getattr(solution_instance, 'q', solution_instance.results.get('q', None))
        self.v = getattr(solution_instance, 'v', solution_instance.results.get('v', None))
        self.B0 = getattr(solution_instance, 'B0', solution_instance.results.get('B0', None))
        
        # Safe attribute extraction for results
        self.R1 = getattr(solution_instance, 'R1', solution_instance.results.get('R1', None))
        self.R2 = getattr(solution_instance, 'R2', solution_instance.results.get('R2', None))
        self.d = getattr(solution_instance, 'd', solution_instance.results.get('d', None))
        self.N = getattr(solution_instance, 'N', solution_instance.results.get('N', None))
        self.T1 = getattr(solution_instance, 'T1', solution_instance.results.get('T1', None))
        self.T2 = getattr(solution_instance, 'T2', solution_instance.results.get('T2', None))
        self.T = getattr(solution_instance, 'T', solution_instance.results.get('T', None))
        self.t = getattr(solution_instance, 't', solution_instance.results.get('t', None))
        self.delta_t = getattr(solution_instance, 'delta_t', solution_instance.results.get('delta_t', None))
        
        # Particle B results
        self.R1_prime = getattr(solution_instance, 'R1_prime', solution_instance.results.get('R1_prime', None))
        self.R2_prime = getattr(solution_instance, 'R2_prime', solution_instance.results.get('R2_prime', None))
        self.d_prime = getattr(solution_instance, 'd_prime', solution_instance.results.get('d_prime', None))
        self.T1_prime = getattr(solution_instance, 'T1_prime', solution_instance.results.get('T1_prime', None))
        self.T2_prime = getattr(solution_instance, 'T2_prime', solution_instance.results.get('T2_prime', None))
        self.frac_qprime_mprime = getattr(solution_instance, 'frac_qprime_mprime', solution_instance.results.get('frac_qprime_mprime', None))
        
        self.logs = []
        
    def log(self, message):
        self.logs.append(message)
    
    def test_invariants(self):
        """Verify fundamental physics laws"""
        try:
            # Numerical check if values are provided
            if all(val is not None for val in [self.m, self.q, self.v, self.B0, self.R1, self.R2]):
                R1_num = self.m*self.v/(2*self.q*self.B0)
                R2_num = self.m*self.v/(3*self.q*self.B0)
                
                if not math.isclose(self.R1, R1_num, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: R1 mismatch. Expected ~{R1_num}, got {self.R1}")
                    return False
                if not math.isclose(self.R2, R2_num, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: R2 mismatch. Expected ~{R2_num}, got {self.R2}")
                    return False
            
            # Check period formulas
            if all(val is not None for val in [self.m, self.q, self.B0, self.T1, self.T2]):
                T1_num = math.pi*self.m/(self.q*self.B0)
                T2_num = 2*math.pi*self.m/(3*self.q*self.B0)
                
                if not math.isclose(self.T1, T1_num, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: T1 mismatch. Expected ~{T1_num}, got {self.T1}")
                    return False
                if not math.isclose(self.T2, T2_num, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: T2 mismatch. Expected ~{T2_num}, got {self.T2}")
                    return False
            
            # Check total period T = (T1 + T2)/2
            if all(val is not None for val in [self.T1, self.T2, self.T]):
                T_expected = (self.T1 + self.T2) / 2
                if not math.isclose(self.T, T_expected, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: T mismatch. Expected ~{T_expected}, got {self.T}")
                    return False
            
            # Check geometric relationship d = 2R1 - 2R2
            if all(val is not None for val in [self.R1, self.R2, self.d]):
                d_expected = 2*self.R1 - 2*self.R2
                if not math.isclose(self.d, d_expected, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: d mismatch. Expected ~{d_expected}, got {self.d}")
                    return False
            
            # Check N = 2R1/d (should be integer 3)
            if all(val is not None for val in [self.R1, self.d, self.N]):
                N_expected = 2*self.R1/self.d
                if not math.isclose(N_expected, 3.0, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: N calculation. Expected ~3, got {N_expected}")
                    return False
                if not math.isclose(self.N, 3.0, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: N value. Expected 3, got {self.N}")
                    return False
            
            self.log("PASS: All invariants satisfied")
            return True
            
        except Exception as e:
            self.log(f"ERROR in test_invariants: {str(e)}")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        try:
            # Check that t = 3T
            if all(val is not None for val in [self.t, self.T]):
                if not math.isclose(self.t, 3*self.T, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: t should be 3T. t={self.t}, 3T={3*self.T}")
                    return False
            
            # Check Δt = t - T1/2
            if all(val is not None for val in [self.delta_t, self.t, self.T1]):
                delta_t_expected = self.t - self.T1/2
                if not math.isclose(self.delta_t, delta_t_expected, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: Δt mismatch. Expected ~{delta_t_expected}, got {self.delta_t}")
                    return False
            
            # Verify target answers
            # 1. d = mv/(3qB0)
            if self.d is not None and all(val is not None for val in [self.m, self.v, self.q, self.B0]):
                d_expected = self.m*self.v/(3*self.q*self.B0)
                if not math.isclose(self.d, d_expected, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: d doesn't match target formula. Expected ~{d_expected}, got {self.d}")
                    return False
            
            # 2. Δt = 2πm/(qB0)
            if self.delta_t is not None and all(val is not None for val in [self.m, self.q, self.B0]):
                delta_t_expected = 2*math.pi*self.m/(self.q*self.B0)
                if not math.isclose(self.delta_t, delta_t_expected, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: Δt doesn't match target formula. Expected ~{delta_t_expected}, got {self.delta_t}")
                    return False
            
            # 3. q'/m' = 2q/m (minimum value)
            if self.frac_qprime_mprime is not None and all(val is not None for val in [self.q, self.m]):
                expected_ratio = 2*self.q/self.m
                if not math.isclose(self.frac_qprime_mprime, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: q'/m' ratio. Expected ~{expected_ratio}, got {self.frac_qprime_mprime}")
                    return False
            
            self.log("PASS: Trace consistency verified")
            return True
            
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {str(e)}")
            return False
    
    def test_monotonicity(self):
        """Perturb inputs and check expected trends"""
        try:
            if any(val is None for val in [self.m, self.q, self.v, self.B0]):
                self.log("SKIP: Missing input values for monotonicity test")
                return True
            
            # Save original values
            original_m = self.m
            original_q = self.q
            original_R1 = self.R1
            original_T1 = self.T1
            
            # Test 1: Increase mass by 10%
            self.m = original_m * 1.1
            
            # R1, R2, d should increase proportionally with m
            R1_perturbed = self.m*self.v/(2*self.q*self.B0)
            if original_R1 is not None:
                if not R1_perturbed > original_R1:
                    self.log(f"FAIL: R1 should increase with mass. Original: {original_R1}, Perturbed: {R1_perturbed}")
                    self.m = original_m
                    return False
            
            # Test 2: Increase charge by 10%
            self.q = original_q * 1.1
            self.m = original_m  # Restore mass
            
            # R1 should decrease with increased charge
            R1_perturbed = self.m*self.v/(2*self.q*self.B0)
            if original_R1 is not None:
                if not R1_perturbed < original_R1:
                    self.log(f"FAIL: R1 should decrease with increased charge. Original: {original_R1}, Perturbed: {R1_perturbed}")
                    self.q = original_q
                    return False
            
            # Test 3: Period T1 should decrease with increased charge
            T1_perturbed = math.pi*self.m/(self.q*self.B0)
            if original_T1 is not None:
                if not T1_perturbed < original_T1:  # T1 ∝ 1/q, so should decrease
                    self.log(f"FAIL: T1 should decrease with increased charge. Original: {original_T1}, Perturbed: {T1_perturbed}")
                    self.q = original_q
                    return False
            
            # Restore original values
            self.m = original_m
            self.q = original_q
            
            self.log("PASS: Monotonicity trends verified")
            return True
            
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {str(e)}")
            # Try to restore values on error
            if 'original_m' in locals():
                self.m = original_m
            if 'original_q' in locals():
                self.q = original_q
            return False
    
    def run_all_tests(self):
        """Execute all tests and return results"""
        self.logs = []
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        success = True
        for test_name, test_func in tests:
            try:
                if not test_func():
                    success = False
                    self.log(f"{test_name} test failed")
            except Exception as e:
                success = False
                self.log(f"Exception in {test_name}: {str(e)}")
                self.log(traceback.format_exc())
        
        return success, self.logs