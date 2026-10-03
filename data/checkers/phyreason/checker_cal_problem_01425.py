import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Input constants from problem statement
        self.E = 40.0
        self.R0 = 20.0
        self.r = 5.0
        self.I1_given = 0.5
        self.I2_given = 1.5
        self.P_mech_given = 5.0
        
        # Safe extraction from solution instance (using getattr with fallback to results dict)
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract results with safe defaults
        self.P = getattr(solution_instance, 'P', self.results.get('P', 0.0))
        self.t_2 = getattr(solution_instance, 't_2', self.results.get('t_2', 0.0))
        self.R_t2 = getattr(solution_instance, 'R_{t2}', self.results.get('R_t2', 0.0))
        self.R_t1 = getattr(solution_instance, 'R_{t1}', self.results.get('R_t1', 0.0))
        self.I_t = getattr(solution_instance, 'I_t', self.results.get('I_t', 0.0))
        self.t_1 = getattr(solution_instance, 't_1', self.results.get('t_1', 0.0))
        self.U = getattr(solution_instance, 'U', self.results.get('U', 0.0))
        
        # Derived quantities (for internal checking)
        self.I_motor = None  # Current through motor when S is closed
        self.calculated_R_t1 = None
        self.calculated_R_t2 = None
        
    def fuzzy_equal(self, a, b):
        """Robust fuzzy equality check with 2% tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        """Verify fundamental physics laws."""
        logs = []
        success = True
        
        try:
            # 1. Open circuit: Ohm's Law
            # I1 = E / (R0 + R_t1)
            expected_I1 = self.E / (self.R0 + self.R_t1)
            if not self.fuzzy_equal(self.I1_given, expected_I1):
                logs.append(f"FAIL: Open circuit Ohm's Law: I1_given={self.I1_given}, calculated={expected_I1}")
                success = False
            else:
                logs.append("PASS: Open circuit Ohm's Law")
            
            # 2. Thermistor linear model: R_t = 70 - t
            # Check for both open and closed states
            expected_R_t1 = 70 - self.t_1
            expected_R_t2 = 70 - self.t_2
            if not self.fuzzy_equal(self.R_t1, expected_R_t1):
                logs.append(f"FAIL: Thermistor model at t1: R_t1={self.R_t1}, expected={expected_R_t1}")
                success = False
            else:
                logs.append("PASS: Thermistor model at t1")
            
            if not self.fuzzy_equal(self.R_t2, expected_R_t2):
                logs.append(f"FAIL: Thermistor model at t2: R_t2={self.R_t2}, expected={expected_R_t2}")
                success = False
            else:
                logs.append("PASS: Thermistor model at t2")
            
            # 3. Closed circuit: Voltage across parallel branch
            # U = E - I2 * R0
            expected_U = self.E - self.I2_given * self.R0
            if not self.fuzzy_equal(self.U, expected_U):
                logs.append(f"FAIL: Closed circuit voltage: U={self.U}, expected={expected_U}")
                success = False
            else:
                logs.append("PASS: Closed circuit voltage")
            
            # 4. Motor power equation (numerical check)
            # First solve for I_motor from quadratic: P = U*I_m - I_m²*r
            # Rearranged: r*I_m² - U*I_m + P = 0
            a = self.r
            b = -self.U
            c = self.P_mech_given
            discriminant = b**2 - 4*a*c
            
            if discriminant < 0:
                logs.append("FAIL: Motor power quadratic has no real solution")
                success = False
            else:
                I_motor1 = (-b + math.sqrt(discriminant)) / (2*a)
                I_motor2 = (-b - math.sqrt(discriminant)) / (2*a)
                
                # Choose the physically reasonable solution (should be between 0 and I2)
                valid_solutions = [I for I in [I_motor1, I_motor2] if 0 <= I <= self.I2_given]
                
                if not valid_solutions:
                    logs.append("FAIL: No valid motor current found")
                    success = False
                else:
                    self.I_motor = valid_solutions[0]
                    
                    # Check KCL: I_t = I2 - I_motor
                    expected_I_t = self.I2_given - self.I_motor
                    if not self.fuzzy_equal(self.I_t, expected_I_t):
                        logs.append(f"FAIL: KCL at junction: I_t={self.I_t}, expected={expected_I_t}")
                        success = False
                    else:
                        logs.append("PASS: KCL at junction")
                    
                    # Check Ohm's Law for thermistor in closed state: R_t2 = U / I_t
                    expected_R_t2_ohm = self.U / self.I_t if abs(self.I_t) > 1e-10 else float('inf')
                    if not self.fuzzy_equal(self.R_t2, expected_R_t2_ohm):
                        logs.append(f"FAIL: Ohm's Law for thermistor: R_t2={self.R_t2}, expected={expected_R_t2_ohm}")
                        success = False
                    else:
                        logs.append("PASS: Ohm's Law for thermistor")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        try:
            # Target answers from problem
            target_t1 = 10.0
            target_t2 = 50.0
            
            # Check t1
            if not self.fuzzy_equal(self.t_1, target_t1):
                logs.append(f"FAIL: t1={self.t_1}, target={target_t1}")
                success = False
            else:
                logs.append("PASS: t1 matches target")
            
            # Check t2
            if not self.fuzzy_equal(self.t_2, target_t2):
                logs.append(f"FAIL: t2={self.t_2}, target={target_t2}")
                success = False
            else:
                logs.append("PASS: t2 matches target")
            
            # Check that R_t1 and R_t2 are consistent with t values
            expected_R_t1_from_t = 70 - self.t_1
            expected_R_t2_from_t = 70 - self.t_2
            
            if not self.fuzzy_equal(self.R_t1, expected_R_t1_from_t):
                logs.append(f"FAIL: R_t1 inconsistent with t1: {self.R_t1} vs {expected_R_t1_from_t}")
                success = False
            else:
                logs.append("PASS: R_t1 consistent with t1")
            
            if not self.fuzzy_equal(self.R_t2, expected_R_t2_from_t):
                logs.append(f"FAIL: R_t2 inconsistent with t2: {self.R_t2} vs {expected_R_t2_from_t}")
                success = False
            else:
                logs.append("PASS: R_t2 consistent with t2")
            
            # Check that mechanical power matches given value
            if not self.fuzzy_equal(self.P, self.P_mech_given):
                logs.append(f"FAIL: Mechanical power P={self.P}, given={self.P_mech_given}")
                success = False
            else:
                logs.append("PASS: Mechanical power matches given value")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Perturb temperature and check thermistor resistance trend."""
        logs = []
        success = True
        
        try:
            # Save original values
            original_t1 = self.t_1
            original_t2 = self.t_2
            
            # Test 1: Increase t1 by 1°C, R_t1 should decrease (negative temp coefficient)
            perturbed_t1 = original_t1 + 1.0
            perturbed_R_t1 = 70 - perturbed_t1
            
            if perturbed_R_t1 >= self.R_t1:
                logs.append(f"FAIL: Monotonicity for thermistor: t↑ from {original_t1} to {perturbed_t1}, R should ↓ but {self.R_t1} -> {perturbed_R_t1}")
                success = False
            else:
                logs.append("PASS: Thermistor shows negative temperature coefficient (t↑ → R↓)")
            
            # Test 2: Decrease t2 by 1°C, R_t2 should increase
            perturbed_t2 = original_t2 - 1.0
            perturbed_R_t2 = 70 - perturbed_t2
            
            if perturbed_R_t2 <= self.R_t2:
                logs.append(f"FAIL: Monotonicity for thermistor: t↓ from {original_t2} to {perturbed_t2}, R should ↑ but {self.R_t2} -> {perturbed_R_t2}")
                success = False
            else:
                logs.append("PASS: Thermistor monotonicity consistent in both directions")
            
            # Test 3: Check motor power monotonicity with respect to current
            # The mechanical power formula: P = U*I_m - I_m²*r is a downward parabola
            # Maximum occurs at I_m = U/(2r) = 10/(10) = 1A (for our values)
            # At I_m = 1A, we get P_max = 5W
            # If we perturb I_m slightly up or down, P should decrease
            
            if self.I_motor is not None:
                # Perturb I_motor by ±0.1A
                for delta in [0.1, -0.1]:
                    perturbed_I_m = self.I_motor + delta
                    perturbed_P = self.U * perturbed_I_m - perturbed_I_m**2 * self.r
                    
                    if perturbed_P > self.P_mech_given:
                        logs.append(f"FAIL: Motor power should decrease when perturbing I_m from {self.I_motor} to {perturbed_I_m}, but got P={perturbed_P} > {self.P_mech_given}")
                        success = False
                    else:
                        logs.append(f"PASS: Motor power decreases when I_m perturbed by {delta}A (expected for parabolic maximum)")
            
            # Restore original values (important!)
            self.t_1 = original_t1
            self.t_2 = original_t2
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}\n{traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return overall success status and logs."""
        all_logs = []
        overall_success = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"\n=== Running {test_name} Test ===")
            success, logs = test_method()
            
            all_logs.extend(logs)
            if not success:
                overall_success = False
                all_logs.append(f"FAIL: {test_name} test failed")
            else:
                all_logs.append(f"PASS: {test_name} test passed")
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        if overall_success:
            all_logs.append("ALL TESTS PASSED")
        else:
            all_logs.append("SOME TESTS FAILED")
        
        return overall_success, all_logs