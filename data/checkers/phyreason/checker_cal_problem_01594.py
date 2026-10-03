import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of input parameters (hardcoded from problem statement)
        self.R1 = 2.0  # Ω
        self.R2 = 3.0  # Ω
        self.R3 = 4.0  # Ω
        
        # Safe extraction of results with fallbacks
        self.fracU_1U_2 = getattr(solution_instance, 'fracU_1U_2', 
                                 getattr(solution_instance, 'results', {}).get('fracU_1U_2', 0.0))
        self.I_1 = getattr(solution_instance, 'I_1', 
                          getattr(solution_instance, 'results', {}).get('I_1', 0.0))
        self.I_2 = getattr(solution_instance, 'I_2', 
                          getattr(solution_instance, 'results', {}).get('I_2', 0.0))
        self.fracU_1primeU_2prime = getattr(solution_instance, 'fracU_1primeU_2prime', 
                                          getattr(solution_instance, 'results', {}).get('fracU_1primeU_2prime', 0.0))
        
        # Symbolic variables for laws
        self.V, self.I, self.R = sp.symbols('V I R', positive=True)
        self.I0, self.I1 = sp.symbols('I0 I1', positive=True)
        self.P = sp.symbols('P', positive=True)
        
        self.logs = []
        
    def _log(self, message, status="INFO"):
        self.logs.append(f"[{status}] {message}")
        
    def _is_close(self, a, b):
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        """Verify circuit laws and conservation principles"""
        try:
            # 1. Series current invariance (first scenario - R1 and R2 in series)
            if not self._is_close(self.I_1, self.I_2):
                self._log(f"Series current law violated: I_1={self.I_1}, I_2={self.I_2}", "WARNING")
                return False
            self._log(f"Series current law satisfied: I_1={self.I_1:.3f} A, I_2={self.I_2:.3f} A")
            
            # 2. Voltage ratio check for first scenario (R1 and R2 in series)
            expected_ratio1 = self.R1 / self.R2  # Should be 2/3 ≈ 0.6667
            if not self._is_close(self.fracU_1U_2, expected_ratio1):
                self._log(f"Voltage ratio mismatch in first scenario: expected {expected_ratio1:.3f}, got {self.fracU_1U_2:.3f}", "WARNING")
                return False
            self._log(f"First voltage ratio correct: U1/U2 = {self.fracU_1U_2:.3f} ≈ R1/R2 = {expected_ratio1:.3f}")
            
            # 3. Voltage ratio check for second scenario (R2||R3 in series with R1)
            R_parallel = (self.R2 * self.R3) / (self.R2 + self.R3)  # Should be 12/7 ≈ 1.7143
            expected_ratio2 = self.R1 / R_parallel  # Should be 7/6 ≈ 1.1667
            if not self._is_close(self.fracU_1primeU_2prime, expected_ratio2):
                self._log(f"Voltage ratio mismatch in second scenario: expected {expected_ratio2:.3f}, got {self.fracU_1primeU_2prime:.3f}", "WARNING")
                return False
            self._log(f"Second voltage ratio correct: U1'/U2' = {self.fracU_1primeU_2prime:.3f} ≈ R1/R_parallel = {expected_ratio2:.3f}")
            
            # 4. Ohm's Law consistency (numerical fallback if we had voltage)
            # Since we don't have voltage, we check that ratios follow from resistance ratios
            # This is already covered above
            
            return True
        except Exception as e:
            self._log(f"Invariant test failed with error: {str(e)}\n{traceback.format_exc()}", "ERROR")
            return False
    
    def test_trace_consistency(self):
        """Check intermediate calculations match final results"""
        try:
            # From STEPS ANALYSIS, the answers should match exactly
            target1 = 2/3  # First target answer
            target2 = 7/6  # Second target answer
            
            if not self._is_close(self.fracU_1U_2, target1):
                self._log(f"First ratio doesn't match target: expected {target1:.3f}, got {self.fracU_1U_2:.3f}", "WARNING")
                return False
                
            if not self._is_close(self.fracU_1primeU_2prime, target2):
                self._log(f"Second ratio doesn't match target: expected {target2:.3f}, got {self.fracU_1primeU_2prime:.3f}", "WARNING")
                return False
                
            self._log(f"Both ratios match target answers: {target1:.3f} and {target2:.3f}")
            return True
        except Exception as e:
            self._log(f"Trace consistency test failed: {str(e)}", "ERROR")
            return False
    
    def test_monotonicity(self):
        """Perturb resistances and verify expected trends"""
        try:
            original_R1 = self.R1
            
            # Test 1: Increase R1 by 10%, check that both ratios increase
            self.R1 *= 1.1
            
            # Recalculate expected ratios with perturbed R1
            new_ratio1 = self.R1 / self.R2
            R_parallel = (self.R2 * self.R3) / (self.R2 + self.R3)
            new_ratio2 = self.R1 / R_parallel
            
            # Restore original R1
            self.R1 = original_R1
            
            # Check monotonic trends
            if new_ratio1 <= self.fracU_1U_2:
                self._log(f"Monotonicity violation: increasing R1 should increase U1/U2 ratio", "WARNING")
                return False
                
            if new_ratio2 <= self.fracU_1primeU_2prime:
                self._log(f"Monotonicity violation: increasing R1 should increase U1'/U2' ratio", "WARNING")
                return False
                
            self._log("Monotonicity test passed: increasing R1 increases both voltage ratios")
            
            # Test 2: Increase R2 by 10%, check that first ratio decreases
            original_R2 = self.R2
            self.R2 *= 1.1
            new_ratio1 = self.R1 / self.R2
            self.R2 = original_R2
            
            if new_ratio1 >= self.fracU_1U_2:
                self._log(f"Monotonicity violation: increasing R2 should decrease U1/U2 ratio", "WARNING")
                return False
                
            self._log("Monotonicity test passed: increasing R2 decreases U1/U2 ratio")
            
            return True
        except Exception as e:
            self._log(f"Monotonicity test failed: {str(e)}", "ERROR")
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
            self._log(f"Running {test_name}...")
            if not test_func():
                success = False
                self._log(f"{test_name} failed", "ERROR")
            else:
                self._log(f"{test_name} passed", "SUCCESS")
        
        return success, self.logs

# Example usage (commented out for the sandbox):
# if __name__ == "__main__":
#     # Mock solution instance for testing
#     class MockSolution:
#         def __init__(self):
#             self.results = {
#                 'fracU_1U_2': 2/3,
#                 'I_1': 0.5,
#                 'I_2': 0.5,
#                 'fracU_1primeU_2prime': 7/6
#             }
#     
#     solution = MockSolution()
#     checker = PhysicsChecker(solution)
#     success, logs = checker.run_all_tests()
#     print(f"Success: {success}")
#     for log in logs:
#         print(log)