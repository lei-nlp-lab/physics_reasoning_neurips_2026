import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallback
        self.results = {}
        inputs_dict = getattr(solution_instance, 'inputs', {})
        
        # Extract inputs safely
        self.U0 = getattr(solution_instance, 'U0', inputs_dict.get('U0', 0.0))
        self.B0 = getattr(solution_instance, 'B0', inputs_dict.get('B0', 0.0))
        self.R = getattr(solution_instance, 'R', inputs_dict.get('R', 0.0))
        self.a = getattr(solution_instance, 'a', inputs_dict.get('a', 0.0))
        self.b = getattr(solution_instance, 'b', inputs_dict.get('b', 0.0))
        self.e = getattr(solution_instance, 'e', inputs_dict.get('e', 1.6e-19))
        self.m = getattr(solution_instance, 'm', inputs_dict.get('m', 9.11e-31))
        self.pressure = getattr(solution_instance, 'p', inputs_dict.get('p', 0.0))
        self.I_plate = getattr(solution_instance, 'I', inputs_dict.get('I', 0.0))
        
        # Extract results safely
        results_dict = getattr(solution_instance, 'results', {})
        self.v0 = results_dict.get('v_0', 0.0)
        self.r = results_dict.get('r', 0.0)
        self.v = results_dict.get('v', 0.0)
        self.N = results_dict.get('N', 0.0)
        self.n = results_dict.get('n', 0.0)
        self.E_k = results_dict.get('E_k', 0.0)
        
        self.logs = []
        
    def log(self, message, success=True):
        """Helper to log messages"""
        self.logs.append(f"{'✓' if success else '✗'} {message}")
        
    def test_invariants(self):
        """Verify fundamental physical laws"""
        try:
            # 1. Check kinetic energy theorem from step1: v0 = sqrt(2eU0/m)
            if self.U0 > 0 and self.e > 0 and self.m > 0:
                expected_v0 = math.sqrt(2 * self.e * self.U0 / self.m)
                if math.isclose(self.v0, expected_v0, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Kinetic Energy Theorem: v0 = √(2eU0/m) holds")
                else:
                    self.log(f"Kinetic Energy Theorem mismatch: got {self.v0:.2e}, expected {expected_v0:.2e}", False)
                    return False
            
            # 2. Check Lorentz force as centripetal force: B0*e*v0 = m*v0²/r with r = R/2
            if self.B0 > 0 and self.R > 0 and self.e > 0 and self.m > 0:
                expected_v0_num = self.e * self.B0 * self.R / (2 * self.m)
                if math.isclose(self.v0, expected_v0_num, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Lorentz force condition: v0 = eB0R/(2m) holds")
                else:
                    self.log(f"Lorentz force mismatch: got {self.v0:.2e}, expected {expected_v0_num:.2e}", False)
                    return False
            
            # 3. Check current definition: I = N*e
            if self.N > 0 and self.e > 0:
                expected_I = self.N * self.e
                if math.isclose(self.I_plate, expected_I, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Current definition: I = N*e holds")
                else:
                    self.log(f"Current definition mismatch: got I={self.I_plate:.2e}, expected {expected_I:.2e}", False)
                    return False
            
            # 4. Check pressure relation from step8: p = (m*v*N)/(a*b*e)
            if self.N > 0 and self.v > 0 and self.a > 0 and self.b > 0 and self.e > 0:
                expected_p = (self.m * self.v * self.N) / (self.a * self.b * self.e)
                if math.isclose(self.pressure, expected_p, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Pressure relation: p = (m*v*N)/(a*b*e) holds")
                else:
                    self.log(f"Pressure mismatch: got {self.pressure:.2e}, expected {expected_p:.2e}", False)
                    return False
            
            # 5. Check kinetic energy definition: E_k = n*(1/2)*m*v²
            if self.n > 0 and self.v > 0 and self.m > 0:
                expected_Ek = self.n * 0.5 * self.m * self.v**2
                if math.isclose(self.E_k, expected_Ek, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Kinetic energy definition: E_k = n*(1/2)*m*v² holds")
                else:
                    self.log(f"Kinetic energy mismatch: got {self.E_k:.2e}, expected {expected_Ek:.2e}", False)
                    return False
            
            return True
            
        except Exception as e:
            self.log(f"Error in test_invariants: {str(e)}", False)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer"""
        try:
            # Check consistency between steps
            # From step2: r = R/2
            if self.R > 0:
                expected_r = self.R / 2
                if math.isclose(self.r, expected_r, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Geometry consistency: r = R/2 holds")
                else:
                    self.log(f"Geometry mismatch: got r={self.r:.2e}, expected {expected_r:.2e}", False)
                    return False
            
            # From step5: N = n*a*b/(2πR)
            if self.n > 0 and self.a > 0 and self.b > 0 and self.R > 0:
                expected_N = self.n * self.a * self.b / (2 * math.pi * self.R)
                if math.isclose(self.N, expected_N, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Linear density consistency: N = n*a*b/(2πR) holds")
                else:
                    self.log(f"Linear density mismatch: got N={self.N:.2e}, expected {expected_N:.2e}", False)
                    return False
            
            # From step7: n = 2πR*I/(e*a*b)
            if self.I_plate != 0 and self.R > 0 and self.a > 0 and self.b > 0 and self.e > 0:
                expected_n = 2 * math.pi * self.R * self.I_plate / (self.e * self.a * self.b)
                if math.isclose(self.n, expected_n, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"n definition consistency: n = 2πRI/(e*a*b) holds")
                else:
                    self.log(f"n definition mismatch: got n={self.n:.2e}, expected {expected_n:.2e}", False)
                    return False
            
            # From step9: v = e*a*b*p/(m*I)
            if self.pressure > 0 and self.I_plate != 0 and self.a > 0 and self.b > 0 and self.e > 0 and self.m > 0:
                expected_v = self.e * self.a * self.b * self.pressure / (self.m * self.I_plate)
                if math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Velocity from pressure: v = e*a*b*p/(m*I) holds")
                else:
                    self.log(f"Velocity mismatch: got v={self.v:.2e}, expected {expected_v:.2e}", False)
                    return False
            
            # Final expression consistency: E_k = π*R*e*a*b*p²/(m*I)
            if self.pressure > 0 and self.I_plate != 0 and self.R > 0 and self.a > 0 and self.b > 0 and self.e > 0 and self.m > 0:
                expected_Ek_final = math.pi * self.R * self.e * self.a * self.b * self.pressure**2 / (self.m * self.I_plate)
                if math.isclose(self.E_k, expected_Ek_final, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Final expression consistency: E_k = π*R*e*a*b*p²/(m*I) holds")
                else:
                    self.log(f"Final expression mismatch: got {self.E_k:.2e}, expected {expected_Ek_final:.2e}", False)
                    return False
            
            return True
            
        except Exception as e:
            self.log(f"Error in test_trace_consistency: {str(e)}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb inputs and check trends"""
        try:
            all_pass = True
            
            # Test 1: Perturb U0 and check v0 (from step1)
            if self.U0 > 0 and self.v0 > 0 and self.e > 0 and self.m > 0:
                original_U0 = self.U0
                perturbed_U0 = original_U0 * 1.05  # 5% increase
                
                original_v0 = self.v0
                expected_v0_perturbed = math.sqrt(2 * self.e * perturbed_U0 / self.m)
                
                # v0 should increase with U0
                if expected_v0_perturbed > original_v0:
                    self.log(f"Monotonicity: v0 increases with U0 (√ relationship)")
                else:
                    self.log(f"Monotonicity violated: v0 should increase with U0", False)
                    all_pass = False
            
            # Test 2: Perturb B0 and check v0 (from step4)
            if self.B0 > 0 and self.R > 0 and self.e > 0 and self.m > 0:
                original_B0 = self.B0
                perturbed_B0 = original_B0 * 1.05  # 5% increase
                
                original_v0 = self.v0
                expected_v0_perturbed = self.e * perturbed_B0 * self.R / (2 * self.m)
                
                # v0 should increase linearly with B0
                if expected_v0_perturbed > original_v0:
                    self.log(f"Monotonicity: v0 increases linearly with B0")
                else:
                    self.log(f"Monotonicity violated: v0 should increase with B0", False)
                    all_pass = False
            
            # Test 3: Perturb pressure p and check E_k
            if self.pressure > 0 and self.I_plate != 0 and self.E_k > 0 and self.R > 0 and self.a > 0 and self.b > 0 and self.e > 0 and self.m > 0:
                original_p = self.pressure
                perturbed_p = original_p * 1.05  # 5% increase
                
                original_Ek = self.E_k
                expected_Ek_perturbed = math.pi * self.R * self.e * self.a * self.b * perturbed_p**2 / (self.m * self.I_plate)
                
                # E_k should increase quadratically with p
                if expected_Ek_perturbed > original_Ek:
                    self.log(f"Monotonicity: E_k increases quadratically with p")
                else:
                    self.log(f"Monotonicity violated: E_k should increase with p", False)
                    all_pass = False
            
            # Test 4: Perturb I and check E_k
            if self.I_plate > 0 and self.E_k > 0 and self.R > 0 and self.a > 0 and self.b > 0 and self.e > 0 and self.m > 0:
                original_I = self.I_plate
                perturbed_I = original_I * 1.05  # 5% increase
                
                original_Ek = self.E_k
                expected_Ek_perturbed = math.pi * self.R * self.e * self.a * self.b * self.pressure**2 / (self.m * perturbed_I)
                
                # E_k should decrease with I
                if expected_Ek_perturbed < original_Ek:
                    self.log(f"Monotonicity: E_k decreases with I (inverse relationship)")
                else:
                    self.log(f"Monotonicity violated: E_k should decrease with I", False)
                    all_pass = False
            
            return all_pass
            
        except Exception as e:
            self.log(f"Error in test_monotonicity: {str(e)}", False)
            return False
    
    def run_all_tests(self):
        """Execute all tests and return results"""
        self.logs = []  # Reset logs
        
        success = True
        test_results = []
        
        # Run each test
        tests = [
            ("Physical Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                test_success = test_func()
                test_results.append((test_name, test_success))
                if not test_success:
                    success = False
            except Exception as e:
                self.log(f"{test_name} crashed: {str(e)}", False)
                test_results.append((test_name, False))
                success = False
        
        # Summary
        self.log(f"\n=== SUMMARY ===")
        for test_name, test_success in test_results:
            status = "PASS" if test_success else "FAIL"
            self.log(f"{test_name}: {status}")
        
        return success, self.logs