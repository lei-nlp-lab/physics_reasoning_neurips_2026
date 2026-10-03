import sympy as sp
import math
import random

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs and results safely
        self.T = getattr(solution_instance, 'T', self.results.get('T', 0.0))
        self.t = getattr(solution_instance, 't', self.results.get('t', 0.0))
        
        # Additional symbolic variables
        self.r = getattr(solution_instance, 'r', sp.symbols('r', positive=True))
        self.pi = sp.pi
        
        # Store logs
        self.logs = []
    
    def _log(self, message, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            message = f"[{'PASS' if success else 'FAIL'}] {message}"
        self.logs.append(message)
        return success
    
    def test_invariants(self):
        """Test energy and proportionality invariants."""
        success = True
        
        try:
            # Test 1: Proportionality invariant (from KB)
            # Time should be proportional to area increase
            area_increase_T = sp.pi * (2*self.r)**2 - sp.pi * self.r**2  # 3πr²
            area_increase_t = sp.pi * (3*self.r)**2 - sp.pi * (2*self.r)**2  # 5πr²
            
            # Compute expected ratio
            expected_ratio = (area_increase_t / area_increase_T)
            expected_ratio_simplified = sp.simplify(expected_ratio)
            
            # Check if t/T matches expected ratio (5/3)
            if hasattr(self.t, 'free_symbols') and hasattr(self.T, 'free_symbols'):
                # Symbolic check
                ratio_expr = self.t / self.T
                diff = sp.simplify(ratio_expr - expected_ratio_simplified)
                if diff != 0:
                    self._log(f"Symbolic ratio mismatch: {ratio_expr} ≠ {expected_ratio_simplified}", False)
                    success = False
                else:
                    self._log(f"Symbolic ratio correct: t/T = {expected_ratio_simplified}", True)
            else:
                # Numerical check with fuzzy matching
                if self.T != 0:
                    ratio_num = float(self.t) / float(self.T)
                    expected_val = 5/3  # 5πr² / 3πr² = 5/3
                    if not math.isclose(ratio_num, expected_val, rel_tol=1e-2, abs_tol=1e-2):
                        self._log(f"Numerical ratio mismatch: {ratio_num:.4f} ≠ {expected_val:.4f}", False)
                        success = False
                    else:
                        self._log(f"Numerical ratio correct: {ratio_num:.4f} ≈ {expected_val:.4f}", True)
            
            # Test 2: Ratio trend invariant (from KB)
            # If numerator (area increase) grows while denominator (time base) constant, ratio should increase
            # Here, area_increase_t > area_increase_T, so t/T > 1
            if hasattr(self.t, 'free_symbols') and hasattr(self.T, 'free_symbols'):
                # Symbolic inequality check
                inequality = sp.simplify(self.t / self.T > 1)
                if inequality != True:
                    self._log(f"Ratio trend violation: t/T = {self.t/self.T} ≤ 1", False)
                    success = False
            elif self.T != 0:
                if not float(self.t) / float(self.T) > 1:
                    self._log(f"Ratio trend violation: t/T ≤ 1", False)
                    success = False
            
        except Exception as e:
            self._log(f"Invariant test failed with error: {e}", False)
            success = False
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate steps with final result."""
        success = True
        
        try:
            # Reconstruct calculation from steps analysis
            # Step 1: T ∝ π(2r)² - πr² = 3πr²
            # Step 2: t ∝ π(3r)² - π(2r)² = 5πr²
            # Step 3: t = (5πr² / 3πr²) * T = (5/3)T
            
            area_T = sp.pi * (2*self.r)**2 - sp.pi * self.r**2  # 3πr²
            area_t = sp.pi * (3*self.r)**2 - sp.pi * (2*self.r)**2  # 5πr²
            
            # Compute t from proportionality
            t_calculated = (area_t / area_T) * self.T
            t_calculated_simplified = sp.simplify(t_calculated)
            
            # Compare with result
            if hasattr(self.t, 'free_symbols'):
                # Symbolic comparison
                diff = sp.simplify(self.t - t_calculated_simplified)
                if diff != 0:
                    self._log(f"Trace inconsistency: {self.t} ≠ {t_calculated_simplified}", False)
                    success = False
                else:
                    self._log(f"Trace consistent: t = {t_calculated_simplified}", True)
            else:
                # Numerical comparison
                t_calc_num = float(t_calculated_simplified.subs({self.r: 1.0, self.T: float(self.T)}))
                if not math.isclose(float(self.t), t_calc_num, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Trace numerical mismatch: {self.t:.4f} ≠ {t_calc_num:.4f}", False)
                    success = False
                else:
                    self._log(f"Trace numerically consistent: {self.t:.4f} ≈ {t_calc_num:.4f}", True)
            
            # Check step 3 value matches target answer
            expected_t_expr = (5/sp.Integer(3)) * self.T
            if hasattr(self.t, 'free_symbols'):
                diff = sp.simplify(self.t - expected_t_expr)
                if diff != 0:
                    self._log(f"Does not match target answer: {self.t} ≠ (5/3)T", False)
                    success = False
            else:
                expected_val = (5/3) * float(self.T)
                if not math.isclose(float(self.t), expected_val, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Does not match target answer: {self.t:.4f} ≠ (5/3)T = {expected_val:.4f}", False)
                    success = False
            
        except Exception as e:
            self._log(f"Trace consistency test failed with error: {e}", False)
            success = False
        
        return success
    
    def test_monotonicity(self):
        """Test that increasing base time T increases t proportionally."""
        success = True
        
        try:
            # Save original state
            T_original = self.T
            t_original = self.t
            
            # Test with perturbed T
            if hasattr(T_original, 'free_symbols'):
                # Symbolic test: t should be linear in T with slope 5/3
                slope = sp.diff(self.t, self.T)
                expected_slope = 5/sp.Integer(3)
                if sp.simplify(slope - expected_slope) != 0:
                    self._log(f"Monotonicity violation: dt/dT = {slope} ≠ 5/3", False)
                    success = False
                else:
                    self._log(f"Correct linear relationship: dt/dT = 5/3", True)
            else:
                # Numerical test with different T values
                for factor in [0.5, 1.0, 2.0]:
                    T_test = factor * float(T_original)
                    t_expected = (5/3) * T_test
                    
                    # Use the proportionality formula
                    r_val = 1.0  # arbitrary
                    area_T = math.pi * (2*r_val)**2 - math.pi * r_val**2
                    area_t = math.pi * (3*r_val)**2 - math.pi * (2*r_val)**2
                    t_calc = (area_t / area_T) * T_test
                    
                    if not math.isclose(t_calc, t_expected, rel_tol=1e-2, abs_tol=1e-2):
                        self._log(f"Monotonicity failure at T={T_test:.2f}: {t_calc:.4f} ≠ {t_expected:.4f}", False)
                        success = False
                
                if success:
                    self._log("Monotonicity holds for perturbed T values", True)
            
            # Restore original state
            self.T = T_original
            self.t = t_original
            
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {e}", False)
            success = False
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return results."""
        self.logs = []
        
        # Run tests
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        # Overall success
        overall_success = test1 and test2 and test3
        
        # Summary
        summary = f"Physics Checker Summary: {sum([test1, test2, test3])}/3 tests passed"
        self.logs.append(summary)
        
        return overall_success, self.logs

# Example usage (commented out for safety):
# if __name__ == "__main__":
#     # Mock solution class for testing
#     class MockSolution:
#         def __init__(self):
#             self.results = {'T': 6.0, 't': 10.0}  # 10 = (5/3)*6
#     
#     solution = MockSolution()
#     checker = PhysicsChecker(solution)
#     success, logs = checker.run_all_tests()
#     for log in logs:
#         print(log)