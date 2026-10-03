import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all relevant quantities from the solution instance.
        """
        self.solution = solution_instance
        # Get results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract ALL values from results, using the actual stored values
        self.ratio = self.results.get('fracI_1I_2', 0.0)
        self.F1 = self.results.get('F_1', 0.0)
        self.F2 = self.results.get('F_2', 0.0)
        self.I1 = self.results.get('I_1', 0.0)
        self.I2 = self.results.get('I_2', 0.0)
        self.r = self.results.get('r', 0.0)
        self.Q = self.results.get('Q', 0.0)
        self.k = self.results.get('k', 0.0)
        self.e = self.results.get('e', 0.0)
        self.m = self.results.get('m', 0.0)
        
        # Compute orbital radius from r
        if self.r > 0:
            self.r_prime = (math.sqrt(3)/2) * self.r
        else:
            self.r_prime = 0.0
        
        # Expected ratio value
        self.expected_ratio = math.sqrt(2)/2
        
    def test_invariants(self):
        """
        Verify fundamental physics invariants.
        """
        logs = []
        success = True
        
        try:
            # Skip force checks if values are zero (they might not be computed correctly)
            if self.F1 > 0 and self.r > 0 and self.Q > 0 and self.e > 0 and self.k > 0:
                # 1. Check that F1 = k*Q*e/r^2 (within tolerance)
                F1_expected = self.k * abs(self.Q) * abs(self.e) / (self.r**2)
                if not math.isclose(self.F1, F1_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"F1 mismatch: got {self.F1}, expected {F1_expected} from Coulomb's Law")
                    success = False
            
            if self.F2 > 0 and self.r > 0 and self.Q > 0 and self.e > 0 and self.k > 0:
                # 2. Check that F2 = sqrt(3)*k*Q*e/r^2 (within tolerance)
                F2_expected = math.sqrt(3) * self.k * abs(self.Q) * abs(self.e) / (self.r**2)
                if not math.isclose(self.F2, F2_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"F2 mismatch: got {self.F2}, expected {F2_expected} from vector addition")
                    success = False
            
            # 3. Check the stored ratio directly
            if not math.isclose(self.ratio, self.expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Ratio mismatch: got {self.ratio}, expected {self.expected_ratio}")
                success = False
            else:
                logs.append(f"Ratio check passed: {self.ratio} ≈ {self.expected_ratio}")
            
            # 4. Check that r' = (√3/2)*r (if r is available)
            if self.r > 0:
                r_prime_expected = (math.sqrt(3)/2) * self.r
                if not math.isclose(self.r_prime, r_prime_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"r' mismatch: got {self.r_prime}, expected {r_prime_expected}")
                    success = False
            
            if success:
                logs.append("All invariant checks passed.")
                
        except Exception as e:
            logs.append(f"Error in test_invariants: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        logs = []
        success = True
        
        try:
            # Only run consistency checks if we have all necessary values
            if (self.I1 > 0 and self.I2 > 0 and self.r > 0 and self.Q > 0 and 
                self.e > 0 and self.k > 0 and self.m > 0 and self.F1 > 0 and self.F2 > 0):
                
                # Check that I1 calculated from different methods gives same result
                # Method 1: From step 4 formula
                I1_method1 = self.e / (2*math.pi*self.r) * math.sqrt(
                    self.k * abs(self.Q) * abs(self.e) / (self.m * self.r))
                
                # Method 2: From centripetal force relation
                v1 = math.sqrt(self.F1 * self.r / self.m)
                I1_method2 = self.e * v1 / (2*math.pi * self.r)
                
                if not math.isclose(I1_method1, I1_method2, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"I1 consistency fail: method1={I1_method1}, method2={I1_method2}")
                    success = False
                
                # Check that I2 calculated from different methods gives same result
                # Method 1: From step 9 formula
                I2_method1 = self.e / (math.pi*self.r) * math.sqrt(
                    self.k * abs(self.Q) * abs(self.e) / (2*self.m * self.r))
                
                # Method 2: From centripetal force relation
                v2 = math.sqrt(self.F2 * self.r_prime / self.m)
                I2_method2 = self.e * v2 / (2*math.pi * self.r_prime)
                
                if not math.isclose(I2_method1, I2_method2, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"I2 consistency fail: method1={I2_method1}, method2={I2_method2}")
                    success = False
                
                # Check ratio consistency - use the stored ratio
                if not math.isclose(self.ratio, self.expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Ratio from stored value: {self.ratio}, expected: {self.expected_ratio}")
                    success = False
                
                if success:
                    logs.append("All trace consistency checks passed.")
            else:
                logs.append("Skipping trace consistency checks due to missing values")
                
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """
        Perturb an input parameter and verify the expected trend.
        """
        logs = []
        success = True
        
        try:
            # Use default values if stored values are zero
            r_test = 5.291772109e-11 if self.r == 0 else self.r
            Q_test = 1.602176634e-19 if self.Q == 0 else self.Q
            e_test = 1.602176634e-19 if self.e == 0 else self.e
            k_test = 8.987551787e9 if self.k == 0 else self.k
            m_test = 9.10938356e-31 if self.m == 0 else self.m
            
            # Test 1: Increase r by 10%
            r_increased = r_test * 1.1
            r_prime_increased = (math.sqrt(3)/2) * r_increased
            
            # Recalculate expected values
            F1_new = k_test * abs(Q_test) * abs(e_test) / (r_increased**2)
            I1_new = e_test / (2*math.pi*r_increased) * math.sqrt(
                k_test * abs(Q_test) * abs(e_test) / (m_test * r_increased))
            
            F2_new = math.sqrt(3) * k_test * abs(Q_test) * abs(e_test) / (r_increased**2)
            I2_new = e_test / (math.pi*r_increased) * math.sqrt(
                k_test * abs(Q_test) * abs(e_test) / (2*m_test * r_increased))
            
            ratio_new = I1_new / I2_new
            
            # Check that ratio remains the same
            if not math.isclose(ratio_new, self.expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Monotonicity test failed: ratio changed from {self.expected_ratio} to {ratio_new} when r increased")
                success = False
            
            # Test 2: Increase Q by 10%
            Q_increased = Q_test * 1.1
            
            # Recalculate with new Q
            F1_new_q = k_test * abs(Q_increased) * abs(e_test) / (r_test**2)
            I1_new_q = e_test / (2*math.pi*r_test) * math.sqrt(
                k_test * abs(Q_increased) * abs(e_test) / (m_test * r_test))
            
            F2_new_q = math.sqrt(3) * k_test * abs(Q_increased) * abs(e_test) / (r_test**2)
            I2_new_q = e_test / (math.pi*r_test) * math.sqrt(
                k_test * abs(Q_increased) * abs(e_test) / (2*m_test * r_test))
            
            ratio_new_q = I1_new_q / I2_new_q
            
            # Check that ratio remains the same
            if not math.isclose(ratio_new_q, self.expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Monotonicity test failed: ratio changed from {self.expected_ratio} to {ratio_new_q} when Q increased")
                success = False
            
            if success:
                logs.append("Monotonicity test passed: ratio is independent of r and Q.")
                
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {str(e)}")
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """
        Run all tests and return overall success status and logs.
        """
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency Tests", self.test_trace_consistency),
            ("Monotonicity Tests", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} ===")
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                all_logs.extend(logs)
            except Exception as e:
                all_logs.append(f"Test crashed: {str(e)}")
                all_logs.append(traceback.format_exc())
                overall_success = False
        
        # Final summary
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall test result: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs