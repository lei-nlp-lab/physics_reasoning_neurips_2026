import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with safe defaults
        self.N1 = 1140  # Given in problem context
        self.U1 = getattr(solution_instance, 'U1', self.results.get('U1', 0.0))
        self.U2 = getattr(solution_instance, 'U2', self.results.get('U2', 0.0))
        
        # Extract computed results
        self.N2 = getattr(solution_instance, 'N2', self.results.get('N_2', 0.0))
        self.fracI_1I_2 = getattr(solution_instance, 'fracI_1I_2', 
                                 self.results.get('fracI_1I_2', 0.0))
        
        # Logs for reporting
        self.logs = []
    
    def _log(self, message, is_error=False):
        self.logs.append(f"{'ERROR' if is_error else 'INFO'}: {message}")
    
    def test_invariants(self):
        """Verify transformer equations from KB"""
        tests_passed = 0
        total_tests = 0
        
        try:
            # Define symbols
            V_p, V_s, I_p, I_s, N_p, N_s = sp.symbols('V_p V_s I_p I_s N_p N_s')
            
            # Test 1: Voltage-turn ratio (V_s/V_p = N_s/N_p)
            lhs_voltage = self.U2 / self.U1 if self.U1 != 0 else 0
            rhs_turns = self.N2 / self.N1 if self.N1 != 0 else 0
            
            total_tests += 1
            if math.isclose(lhs_voltage, rhs_turns, rel_tol=0.02, abs_tol=0.02):
                tests_passed += 1
                self._log(f"Voltage-turn ratio check passed: U2/U1={lhs_voltage:.3f}, N2/N1={rhs_turns:.3f}")
            else:
                self._log(f"Voltage-turn ratio mismatch: U2/U1={lhs_voltage:.3f}, N2/N1={rhs_turns:.3f}", True)
            
            # Test 2: Power conservation (V_p*I_p = V_s*I_s) -> I_s/I_p = V_p/V_s
            # Since we don't have absolute currents, use the ratio I1/I2
            if self.fracI_1I_2 != 0:
                current_ratio = 1/self.fracI_1I_2 if self.fracI_1I_2 != 0 else 0
                voltage_ratio_inverse = self.U1/self.U2 if self.U2 != 0 else 0
                
                total_tests += 1
                if math.isclose(current_ratio, voltage_ratio_inverse, rel_tol=0.02, abs_tol=0.02):
                    tests_passed += 1
                    self._log(f"Power conservation check passed: I2/I1={current_ratio:.3f}, U1/U2={voltage_ratio_inverse:.3f}")
                else:
                    self._log(f"Power conservation mismatch: I2/I1={current_ratio:.3f}, U1/U2={voltage_ratio_inverse:.3f}", True)
            
            # Test 3: Current-turn relationship (I_s/I_p = N_p/N_s)
            if self.fracI_1I_2 != 0:
                current_ratio_from_turns = self.N1/self.N2 if self.N2 != 0 else 0
                
                total_tests += 1
                if math.isclose(1/self.fracI_1I_2, current_ratio_from_turns, rel_tol=0.02, abs_tol=0.02):
                    tests_passed += 1
                    self._log(f"Current-turn relationship check passed: I2/I1={1/self.fracI_1I_2:.3f}, N1/N2={current_ratio_from_turns:.3f}")
                else:
                    self._log(f"Current-turn relationship mismatch: I2/I1={1/self.fracI_1I_2:.3f}, N1/N2={current_ratio_from_turns:.3f}", True)
        
        except Exception as e:
            self._log(f"Invariant test failed with error: {str(e)}", True)
        
        return tests_passed, total_tests
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer"""
        tests_passed = 0
        total_tests = 0
        
        try:
            # Check Step 1: N2 = (U2/U1) * N1
            expected_N2 = (self.U2 / self.U1) * self.N1 if self.U1 != 0 else 0
            total_tests += 1
            
            if math.isclose(self.N2, expected_N2, rel_tol=0.02, abs_tol=0.02):
                tests_passed += 1
                self._log(f"Step 1 consistency passed: N2={self.N2}, expected={expected_N2:.1f}")
            else:
                self._log(f"Step 1 inconsistency: N2={self.N2}, expected={expected_N2:.1f}", True)
            
            # Check Step 2: I1/I2 = N2/N1 = U2/U1
            expected_ratio1 = self.N2 / self.N1 if self.N1 != 0 else 0
            expected_ratio2 = self.U2 / self.U1 if self.U1 != 0 else 0
            
            total_tests += 2
            if math.isclose(self.fracI_1I_2, expected_ratio1, rel_tol=0.02, abs_tol=0.02):
                tests_passed += 1
                self._log(f"Step 2 consistency (N2/N1) passed: I1/I2={self.fracI_1I_2}, N2/N1={expected_ratio1:.3f}")
            else:
                self._log(f"Step 2 inconsistency (N2/N1): I1/I2={self.fracI_1I_2}, N2/N1={expected_ratio1:.3f}", True)
            
            if math.isclose(self.fracI_1I_2, expected_ratio2, rel_tol=0.02, abs_tol=0.02):
                tests_passed += 1
                self._log(f"Step 2 consistency (U2/U1) passed: I1/I2={self.fracI_1I_2}, U2/U1={expected_ratio2:.3f}")
            else:
                self._log(f"Step 2 inconsistency (U2/U1): I1/I2={self.fracI_1I_2}, U2/U1={expected_ratio2:.3f}", True)
        
        except Exception as e:
            self._log(f"Trace consistency test failed with error: {str(e)}", True)
        
        return tests_passed, total_tests
    
    def test_monotonicity(self):
        """Perturb input U2 and check if N2 and current ratio change correctly"""
        tests_passed = 0
        total_tests = 0
        
        try:
            # Store original values
            original_U2 = self.U2
            
            # Test 1: Increase U2 by 10%
            perturbed_U2 = original_U2 * 1.10
            self.U2 = perturbed_U2
            
            # Recalculate using transformer equations
            new_N2 = (perturbed_U2 / self.U1) * self.N1 if self.U1 != 0 else 0
            new_ratio = perturbed_U2 / self.U1 if self.U1 != 0 else 0
            
            total_tests += 2
            
            # Check direction of change
            if new_N2 > self.N2:
                tests_passed += 1
                self._log(f"Monotonicity test 1 passed: N2 increases with U2 (original={self.N2}, new={new_N2:.1f})")
            else:
                self._log(f"Monotonicity test 1 failed: N2 should increase with U2", True)
            
            if new_ratio > self.fracI_1I_2:
                tests_passed += 1
                self._log(f"Monotonicity test 2 passed: I1/I2 increases with U2 (original={self.fracI_1I_2}, new={new_ratio:.3f})")
            else:
                self._log(f"Monotonicity test 2 failed: I1/I2 should increase with U2", True)
            
            # Test 2: Decrease U2 by 5%
            perturbed_U2 = original_U2 * 0.95
            self.U2 = perturbed_U2
            
            new_N2 = (perturbed_U2 / self.U1) * self.N1 if self.U1 != 0 else 0
            new_ratio = perturbed_U2 / self.U1 if self.U1 != 0 else 0
            
            total_tests += 2
            
            if new_N2 < self.N2:
                tests_passed += 1
                self._log(f"Monotonicity test 3 passed: N2 decreases with U2 (original={self.N2}, new={new_N2:.1f})")
            else:
                self._log(f"Monotonicity test 3 failed: N2 should decrease with U2", True)
            
            if new_ratio < self.fracI_1I_2:
                tests_passed += 1
                self._log(f"Monotonicity test 4 passed: I1/I2 decreases with U2 (original={self.fracI_1I_2}, new={new_ratio:.3f})")
            else:
                self._log(f"Monotonicity test 4 failed: I1/I2 should decrease with U2", True)
            
            # Restore original value
            self.U2 = original_U2
        
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {str(e)}", True)
            # Restore original value in case of error
            self.U2 = original_U2
        
        return tests_passed, total_tests
    
    def run_all_tests(self):
        """Execute all tests and return success status and logs"""
        self.logs = []
        total_passed = 0
        total_tests = 0
        
        # Run individual test suites
        try:
            # Test invariants
            self._log("=== Testing Invariants ===")
            passed, total = self.test_invariants()
            total_passed += passed
            total_tests += total
            
            # Test trace consistency
            self._log("\n=== Testing Trace Consistency ===")
            passed, total = self.test_trace_consistency()
            total_passed += passed
            total_tests += total
            
            # Test monotonicity
            self._log("\n=== Testing Monotonicity ===")
            passed, total = self.test_monotonicity()
            total_passed += passed
            total_tests += total
            
            # Calculate success rate
            success_rate = total_passed / total_tests if total_tests > 0 else 0
            success = success_rate >= 0.8  # 80% threshold for success
            
            self._log(f"\n=== Summary ===")
            self._log(f"Tests passed: {total_passed}/{total_tests}")
            self._log(f"Success rate: {success_rate:.1%}")
            self._log(f"Overall: {'PASS' if success else 'FAIL'}")
            
        except Exception as e:
            self._log(f"Test suite failed with error: {str(e)}\n{traceback.format_exc()}", True)
            success = False
        
        return success, self.logs