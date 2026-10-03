import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safe initialization with fallback values."""
        # Extract inputs from problem (fixed values)
        self.R1 = 2.0  # Ω
        self.U1 = 3.0  # V
        self.R2 = 6.0  # Ω
        self.U2 = 4.5  # V
        
        # Extract results safely using getattr and dictionary fallback
        results = getattr(solution_instance, 'results', {})
        
        # Key mapping as specified
        self.r = getattr(solution_instance, 'r', results.get('r', 0.0))
        self.I2 = getattr(solution_instance, 'I_2', results.get('I_2', 0.0))
        self.I1 = getattr(solution_instance, 'I_1', results.get('I_1', 0.0))
        self.E = getattr(solution_instance, 'E', results.get('E', 0.0))
        
        # Store for later use
        self.solution = solution_instance
        self.logs = []
        
    def log(self, msg, status):
        """Helper for logging."""
        self.logs.append(f"{status}: {msg}")
    
    def test_invariants(self):
        """Test physical laws: Ohm's Law and circuit equations."""
        try:
            # 1. Check Ohm's Law for both measurements
            # I1 should equal U1/R1
            expected_I1 = self.U1 / self.R1
            if not math.isclose(self.I1, expected_I1, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Ohm's Law violation: I1={self.I1} ≠ U1/R1={expected_I1}", "FAIL")
                return False
            else:
                self.log(f"Ohm's Law satisfied for case 1: I1={self.I1} ≈ {expected_I1}", "PASS")
            
            # I2 should equal U2/R2
            expected_I2 = self.U2 / self.R2
            if not math.isclose(self.I2, expected_I2, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Ohm's Law violation: I2={self.I2} ≠ U2/R2={expected_I2}", "FAIL")
                return False
            else:
                self.log(f"Ohm's Law satisfied for case 2: I2={self.I2} ≈ {expected_I2}", "PASS")
            
            # 2. Check closed circuit equations: E = U + I*r for both cases
            # Case 1: E should equal U1 + I1*r
            E_from_case1 = self.U1 + self.I1 * self.r
            if not math.isclose(self.E, E_from_case1, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Circuit equation violation (case 1): E={self.E} ≠ U1+I1*r={E_from_case1}", "FAIL")
                return False
            else:
                self.log(f"Circuit equation satisfied for case 1: E={self.E} ≈ {E_from_case1}", "PASS")
            
            # Case 2: E should equal U2 + I2*r
            E_from_case2 = self.U2 + self.I2 * self.r
            if not math.isclose(self.E, E_from_case2, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Circuit equation violation (case 2): E={self.E} ≠ U2+I2*r={E_from_case2}", "FAIL")
                return False
            else:
                self.log(f"Circuit equation satisfied for case 2: E={self.E} ≈ {E_from_case2}", "PASS")
            
            # 3. Both cases should give the same E (internal consistency)
            if not math.isclose(E_from_case1, E_from_case2, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Inconsistent EMF: E_case1={E_from_case1} ≠ E_case2={E_from_case2}", "FAIL")
                return False
            else:
                self.log(f"EMF consistent between both cases: {E_from_case1} ≈ {E_from_case2}", "PASS")
            
            return True
        except Exception as e:
            self.log(f"Invariant test failed with error: {str(e)}", "ERROR")
            return False
    
    def test_trace_consistency(self):
        """Check that intermediate steps match final results."""
        try:
            # According to steps analysis:
            # Step 2: I1 = U1/R1 = 1.5A
            expected_I1 = 1.5
            if not math.isclose(self.I1, expected_I1, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Step 2 mismatch: I1={self.I1} ≠ expected {expected_I1}A", "FAIL")
                return False
            else:
                self.log(f"Step 2 consistency: I1={self.I1} ≈ {expected_I1}A", "PASS")
            
            # Step 4: I2 = U2/R2 = 0.75A
            expected_I2 = 0.75
            if not math.isclose(self.I2, expected_I2, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Step 4 mismatch: I2={self.I2} ≠ expected {expected_I2}A", "FAIL")
                return False
            else:
                self.log(f"Step 4 consistency: I2={self.I2} ≈ {expected_I2}A", "PASS")
            
            # Step 5: E = 6V (from target answers)
            expected_E = 6.0
            if not math.isclose(self.E, expected_E, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Step 5 mismatch: E={self.E}V ≠ expected {expected_E}V", "FAIL")
                return False
            else:
                self.log(f"Step 5 consistency: E={self.E}V ≈ {expected_E}V", "PASS")
            
            # Step 6: r = 2Ω (from target answers)
            expected_r = 2.0
            if not math.isclose(self.r, expected_r, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Step 6 mismatch: r={self.r}Ω ≠ expected {expected_r}Ω", "FAIL")
                return False
            else:
                self.log(f"Step 6 consistency: r={self.r}Ω ≈ {expected_r}Ω", "PASS")
            
            # Verify the simultaneous equations solution
            # From equations: E = U1 + I1*r and E = U2 + I2*r
            # Subtract: 0 = (U2 - U1) + (I2 - I1)*r
            # So r = (U1 - U2) / (I2 - I1)
            calc_r = (self.U1 - self.U2) / (self.I2 - self.I1)
            if not math.isclose(self.r, calc_r, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Simultaneous eq mismatch: r={self.r} ≠ (U1-U2)/(I2-I1)={calc_r}", "FAIL")
                return False
            else:
                self.log(f"Simultaneous eq solution consistent: r={self.r} ≈ {calc_r}", "PASS")
            
            return True
        except Exception as e:
            self.log(f"Trace consistency test failed with error: {str(e)}", "ERROR")
            return False
    
    def test_monotonicity(self):
        """Test that increasing R increases U (terminal voltage)."""
        try:
            # From circuit theory: U = E * R/(R + r)
            # Derivative dU/dR = E*r/(R+r)^2 > 0 for positive E, r, R
            # So U should increase when R increases
            
            # Check given data: R2 > R1 should imply U2 > U1
            if self.R2 > self.R1 and not self.U2 > self.U1:
                self.log(f"Monotonicity violation: R2={self.R2} > R1={self.R1} but U2={self.U2} ≤ U1={self.U1}", "FAIL")
                return False
            else:
                self.log(f"Monotonicity satisfied for given data: R↑ → U↑", "PASS")
            
            # Perturbation test: Increase R slightly, compute new U
            # Using the derived E and r from solution
            R_test = self.R1 + 0.5  # Small increase
            U_test = self.E * R_test / (R_test + self.r)
            
            # U_test should be greater than U1
            if not U_test > self.U1:
                self.log(f"Perturbation test failed: R={R_test} gives U={U_test} ≤ U1={self.U1}", "FAIL")
                return False
            else:
                self.log(f"Perturbation test passed: R↑ from {self.R1} to {R_test} → U↑ from {self.U1} to {U_test}", "PASS")
            
            # Additional check: Decreasing R should decrease U
            R_test2 = self.R2 - 0.5
            U_test2 = self.E * R_test2 / (R_test2 + self.r)
            if not U_test2 < self.U2:
                self.log(f"Decreasing R test failed: R={R_test2} gives U={U_test2} ≥ U2={self.U2}", "FAIL")
                return False
            else:
                self.log(f"Decreasing R test passed: R↓ from {self.R2} to {R_test2} → U↓ from {self.U2} to {U_test2}", "PASS")
            
            return True
        except Exception as e:
            self.log(f"Monotonicity test failed with error: {str(e)}", "ERROR")
            return False
    
    def run_all_tests(self):
        """Run all tests and return success status with logs."""
        self.logs = []  # Reset logs
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        success = True
        for test_name, test_func in tests:
            self.log(f"Running {test_name}...", "INFO")
            try:
                if not test_func():
                    success = False
                    self.log(f"{test_name} failed", "INFO")
                else:
                    self.log(f"{test_name} passed", "INFO")
            except Exception as e:
                success = False
                self.log(f"{test_name} crashed: {traceback.format_exc()}", "ERROR")
        
        return success, self.logs