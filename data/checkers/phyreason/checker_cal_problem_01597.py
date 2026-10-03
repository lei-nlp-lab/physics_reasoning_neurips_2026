import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution_instance = solution_instance
        # Extract inputs (these are known from problem statement)
        self.E = 3.0  # V
        self.U1 = 2.4  # V
        self.q = 6.0  # C
        
        # Extract results safely
        self.W = getattr(solution_instance, 'W', getattr(solution_instance, 'results', {}).get('W', 0.0))
        self.W_1 = getattr(solution_instance, 'W_1', getattr(solution_instance, 'results', {}).get('W_1', 0.0))
        self.W_2 = getattr(solution_instance, 'W_2', getattr(solution_instance, 'results', {}).get('W_2', 0.0))
        self.U_2 = getattr(solution_instance, 'U_2', getattr(solution_instance, 'results', {}).get('U_2', 0.0))
        
        # Expected values from steps analysis
        self.expected = {
            'W': 18.0,
            'W_1': 14.4,
            'W_2': 3.6,
            'U_2': 0.6
        }
        
        # Tolerance for fuzzy matching
        self.rel_tol = 0.02  # 2%
        self.abs_tol = 0.02
    
    def _is_close(self, a, b):
        """Fuzzy numerical comparison"""
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
    
    def test_invariants(self):
        """Verify work formulas and energy conservation"""
        logs = []
        
        try:
            # Symbolic definitions
            E, U1, q, W, W_1, W_2, U_2 = sp.symbols('E U1 q W W_1 W_2 U_2')
            
            # 1. Check work formulas
            # W = E * q
            if not self._is_close(self.W, self.E * self.q):
                logs.append(f"FAIL: W = {self.W} ≠ E*q = {self.E * self.q}")
            
            # W_1 = U1 * q
            if not self._is_close(self.W_1, self.U1 * self.q):
                logs.append(f"FAIL: W_1 = {self.W_1} ≠ U1*q = {self.U1 * self.q}")
            
            # W_2 = U_2 * q
            if not self._is_close(self.W_2, self.U_2 * self.q):
                logs.append(f"FAIL: W_2 = {self.W_2} ≠ U_2*q = {self.U_2 * self.q}")
            
            # 2. Check Kirchhoff's Voltage Law (U_2 = E - U1)
            if not self._is_close(self.U_2, self.E - self.U1):
                logs.append(f"FAIL: U_2 = {self.U_2} ≠ E - U1 = {self.E - self.U1}")
            
            # 3. Check energy conservation: W = W_1 + W_2
            if not self._is_close(self.W, self.W_1 + self.W_2):
                logs.append(f"FAIL: W = {self.W} ≠ W_1 + W_2 = {self.W_1 + self.W_2}")
            
            if not logs:
                logs.append("PASS: All invariants satisfied")
                
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        logs = []
        
        try:
            # Check each result against expected values from steps analysis
            for key, expected_val in self.expected.items():
                actual = getattr(self, key)
                if not self._is_close(actual, expected_val):
                    logs.append(f"FAIL: {key} = {actual} ≠ expected {expected_val}")
            
            # Additional consistency: W_2 should equal W - W_1 (from step 5)
            if not self._is_close(self.W_2, self.W - self.W_1):
                logs.append(f"FAIL: W_2 = {self.W_2} ≠ W - W_1 = {self.W - self.W_1}")
            
            # Check U_2 = E - U1 (from step 3)
            if not self._is_close(self.U_2, self.E - self.U1):
                logs.append(f"FAIL: U_2 = {self.U_2} ≠ E - U1 = {self.E - self.U1}")
            
            if not logs:
                logs.append("PASS: All trace consistency checks passed")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def test_monotonicity(self):
        """Perturb charge q and verify linear relationships"""
        logs = []
        
        try:
            # Store original values
            original_q = self.q
            
            # Perturbation: increase charge by 10%
            perturbed_q = original_q * 1.10
            
            # Recalculate using the formulas from steps analysis
            new_W = self.E * perturbed_q
            new_W_1 = self.U1 * perturbed_q
            new_U_2 = self.E - self.U1  # Should remain constant
            new_W_2 = new_U_2 * perturbed_q
            
            # Verify monotonic increase (all should increase with q)
            if not (new_W > self.W):
                logs.append(f"FAIL: W doesn't increase with q: {new_W} ≤ {self.W}")
            
            if not (new_W_1 > self.W_1):
                logs.append(f"FAIL: W_1 doesn't increase with q: {new_W_1} ≤ {self.W_1}")
            
            if not (new_W_2 > self.W_2):
                logs.append(f"FAIL: W_2 doesn't increase with q: {new_W_2} ≤ {self.W_2}")
            
            # Verify linear relationships (ratios should be equal)
            q_ratio = perturbed_q / original_q
            
            # Check W ratio
            W_ratio = new_W / self.W if self.W != 0 else float('inf')
            if not self._is_close(W_ratio, q_ratio):
                logs.append(f"FAIL: W ratio {W_ratio} ≠ q ratio {q_ratio}")
            
            # Check W_1 ratio
            W_1_ratio = new_W_1 / self.W_1 if self.W_1 != 0 else float('inf')
            if not self._is_close(W_1_ratio, q_ratio):
                logs.append(f"FAIL: W_1 ratio {W_1_ratio} ≠ q ratio {q_ratio}")
            
            # Check W_2 ratio
            W_2_ratio = new_W_2 / self.W_2 if self.W_2 != 0 else float('inf')
            if not self._is_close(W_2_ratio, q_ratio):
                logs.append(f"FAIL: W_2 ratio {W_2_ratio} ≠ q ratio {q_ratio}")
            
            # Verify U_2 remains constant
            if not self._is_close(new_U_2, self.U_2):
                logs.append(f"FAIL: U_2 changed with q: {new_U_2} ≠ {self.U_2}")
            
            # Restore original q (though it's just a local copy)
            self.q = original_q
            
            if not logs:
                logs.append("PASS: Monotonicity and linearity checks passed")
                
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            # Restore original q in case of error
            self.q = original_q if 'original_q' in locals() else 6.0
        
        return logs
    
    def run_all_tests(self):
        """Run all tests and return success status and logs"""
        all_logs = []
        success = True
        
        # Run each test
        tests = [
            ("Invariants Test", self.test_invariants),
            ("Trace Consistency Test", self.test_trace_consistency),
            ("Monotonicity Test", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n=== {test_name} ===")
            test_logs = test_func()
            all_logs.extend(test_logs)
            
            # Check if test had any failures (logs containing "FAIL:" or "ERROR:")
            for log in test_logs:
                if log.startswith("FAIL:") or log.startswith("ERROR:"):
                    success = False
                    break
        
        return success, all_logs

# Example usage (commented out for production):
# if __name__ == "__main__":
#     class MockSolution:
#         def __init__(self):
#             self.results = {
#                 'W': 18.0,
#                 'W_1': 14.4,
#                 'W_2': 3.6,
#                 'U_2': 0.6
#             }
#     
#     mock_solution = MockSolution()
#     checker = PhysicsChecker(mock_solution)
#     success, logs = checker.run_all_tests()
#     print(f"Success: {success}")
#     for log in logs:
#         print(log)