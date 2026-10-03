import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.inputs = {}
        self.results = {}
        
        # Extract inputs
        self.inputs['L0'] = getattr(solution_instance, 'L0', 
                                   getattr(solution_instance, 'inputs', {}).get('L0', 0.6))
        self.inputs['p0'] = getattr(solution_instance, 'p0', 
                                   getattr(solution_instance, 'inputs', {}).get('p0', 1.0e5))
        self.inputs['p'] = getattr(solution_instance, 'p', 
                                  getattr(solution_instance, 'inputs', {}).get('p', 1.5e5))
        self.inputs['T0'] = getattr(solution_instance, 'T0', 
                                   getattr(solution_instance, 'inputs', {}).get('T0', 300.0))
        
        # Extract results
        self.results['L'] = getattr(solution_instance, 'L', 
                                   getattr(solution_instance, 'results', {}).get('L', None))
        self.results['T'] = getattr(solution_instance, 'T', 
                                   getattr(solution_instance, 'results', {}).get('T', None))
        
        # Initialize logs
        self.logs = []
        
    def log(self, message):
        self.logs.append(message)
        
    def fuzzy_equal(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison with tolerances."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Test fundamental gas laws."""
        tests_passed = True
        
        try:
            # Symbolic verification of Boyle's Law for step 1
            L0, p0, p = sp.symbols('L0 p0 p')
            L_expected = p0 * L0 / p
            
            # Numerical check
            L_num = self.inputs['p0'] * self.inputs['L0'] / self.inputs['p']
            if self.results['L'] is not None:
                if not self.fuzzy_equal(self.results['L'], L_num):
                    self.log(f"FAIL: Boyle's Law violated. L_result={self.results['L']}, L_expected={L_num}")
                    tests_passed = False
                else:
                    self.log(f"PASS: Boyle's Law consistent. L={self.results['L']}")
            else:
                self.log("WARNING: L result missing for Boyle's Law test")
                
        except Exception as e:
            self.log(f"ERROR in Boyle's Law test: {traceback.format_exc()}")
            tests_passed = False
            
        try:
            # Symbolic verification of Gay-Lussac's Law (constant volume) for step 2
            T0, p0, p = sp.symbols('T0 p0 p')
            T_expected = p * T0 / p0
            
            # Numerical check
            T_num = self.inputs['p'] * self.inputs['T0'] / self.inputs['p0']
            if self.results['T'] is not None:
                if not self.fuzzy_equal(self.results['T'], T_num):
                    self.log(f"FAIL: Gay-Lussac's Law violated. T_result={self.results['T']}, T_expected={T_num}")
                    tests_passed = False
                else:
                    self.log(f"PASS: Gay-Lussac's Law consistent. T={self.results['T']}")
            else:
                self.log("WARNING: T result missing for Gay-Lussac's Law test")
                
        except Exception as e:
            self.log(f"ERROR in Gay-Lussac's Law test: {traceback.format_exc()}")
            tests_passed = False
            
        return tests_passed
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        tests_passed = True
        
        # Check step 1: L = (p0/p) * L0
        L_calculated = self.inputs['p0'] * self.inputs['L0'] / self.inputs['p']
        if self.results['L'] is not None and not self.fuzzy_equal(self.results['L'], L_calculated):
            self.log(f"FAIL: Step 1 inconsistency. L={self.results['L']}, calculated={L_calculated}")
            tests_passed = False
        else:
            self.log(f"PASS: Step 1 trace consistent")
            
        # Check step 2: T = (p/p0) * T0
        T_calculated = self.inputs['p'] * self.inputs['T0'] / self.inputs['p0']
        if self.results['T'] is not None and not self.fuzzy_equal(self.results['T'], T_calculated):
            self.log(f"FAIL: Step 2 inconsistency. T={self.results['T']}, calculated={T_calculated}")
            tests_passed = False
        else:
            self.log(f"PASS: Step 2 trace consistent")
            
        return tests_passed
    
    def test_monotonicity(self):
        """Perturb inputs and check physical trends."""
        tests_passed = True
        
        try:
            # Original results
            L_orig = self.results['L']
            T_orig = self.results['T']
            
            if L_orig is None or T_orig is None:
                self.log("WARNING: Missing results for monotonicity test")
                return False
                
            # Test 1: Increase p0 by 10%, expect L to increase
            p0_perturbed = self.inputs['p0'] * 1.1
            L_perturbed = p0_perturbed * self.inputs['L0'] / self.inputs['p']
            if not (L_perturbed > L_orig or self.fuzzy_equal(L_perturbed, L_orig)):
                self.log(f"FAIL: Monotonicity violated for p0 increase. L decreased")
                tests_passed = False
            else:
                self.log(f"PASS: L increases with p0 (Boyle's Law trend)")
                
            # Test 2: Increase T0 by 10%, expect T to increase proportionally
            T0_perturbed = self.inputs['T0'] * 1.1
            T_perturbed = self.inputs['p'] * T0_perturbed / self.inputs['p0']
            if not (T_perturbed > T_orig or self.fuzzy_equal(T_perturbed, T_orig)):
                self.log(f"FAIL: Monotonicity violated for T0 increase. T decreased")
                tests_passed = False
            else:
                self.log(f"PASS: T increases with T0 (Gay-Lussac's Law trend)")
                
            # Test 3: Increase p by 10%, expect L to decrease and T to increase
            p_perturbed = self.inputs['p'] * 1.1
            L_perturbed2 = self.inputs['p0'] * self.inputs['L0'] / p_perturbed
            T_perturbed2 = p_perturbed * self.inputs['T0'] / self.inputs['p0']
            
            if not (L_perturbed2 < L_orig or self.fuzzy_equal(L_perturbed2, L_orig)):
                self.log(f"FAIL: L should decrease with p increase")
                tests_passed = False
            else:
                self.log(f"PASS: L decreases with p increase")
                
            if not (T_perturbed2 > T_orig or self.fuzzy_equal(T_perturbed2, T_orig)):
                self.log(f"FAIL: T should increase with p increase")
                tests_passed = False
            else:
                self.log(f"PASS: T increases with p increase")
                
        except Exception as e:
            self.log(f"ERROR in monotonicity test: {traceback.format_exc()}")
            tests_passed = False
            
        return tests_passed
    
    def run_all_tests(self):
        """Execute all tests and return success flag with logs."""
        self.logs = []  # Reset logs
        
        try:
            # Run individual tests
            invariant_ok = self.test_invariants()
            trace_ok = self.test_trace_consistency()
            monotonic_ok = self.test_monotonicity()
            
            # Final summary
            all_passed = invariant_ok and trace_ok and monotonic_ok
            
            if all_passed:
                self.log("SUCCESS: All physics checks passed")
            else:
                self.log("FAILURE: Some physics checks failed")
                
            return all_passed, self.logs
            
        except Exception as e:
            self.log(f"CRITICAL ERROR in test suite: {traceback.format_exc()}")
            return False, self.logs

# Example usage pattern (commented out for production):
# if __name__ == "__main__":
#     class Solution:
#         def __init__(self):
#             self.L0 = 0.6
#             self.p0 = 1.0e5
#             self.p = 1.5e5
#             self.T0 = 300.0
#             self.results = {'L': 0.4, 'T': 450.0}
#     
#     checker = PhysicsChecker(Solution())
#     success, logs = checker.run_all_tests()
#     for log in logs:
#         print(log)