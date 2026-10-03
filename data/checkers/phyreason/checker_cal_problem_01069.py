import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extracts all required inputs and results from the solution instance.
        Uses getattr and dictionary fallbacks to prevent AttributeErrors.
        """
        self.solution = solution_instance
        
        # Safe extraction of results (with defaults if missing)
        self.results = {}
        result_keys = ['t_2', '|a|', 't', 'a', 't_3', 't_1', 'v_1']
        for key in result_keys:
            self.results[key] = getattr(solution_instance, key, 
                                       getattr(solution_instance, 'results', {}).get(key, None))
        
        # Known constants from problem statement
        self.g = 10.0  # m/s^2
        self.x1 = 125.0  # m - free fall distance
        self.x2 = 150.0  # m - uniform motion distance  
        self.x3 = 270.0  # m - deceleration distance
        self.v2 = 4.0  # m/s - final speed
        
        # Validation flags
        self.tests_passed = True
        self.logs = []
    
    def log(self, message, success):
        """Helper to log test results"""
        self.tests_passed = self.tests_passed and success
        self.logs.append(f"{'✓' if success else '✗'} {message}")
    
    def test_invariants(self):
        """Test physical invariants: kinematic equations must hold"""
        try:
            v1 = self.results.get('v_1')
            a = self.results.get('a')
            t1 = self.results.get('t_1')
            t2 = self.results.get('t_2') 
            t3 = self.results.get('t_3')
            total_t = self.results.get('t')
            abs_a = self.results.get('|a|')
            
            # 1. Free fall: v1^2 = 2*g*x1
            if v1 is not None:
                v1_sq_calc = 2 * self.g * self.x1
                v1_sq_actual = v1**2
                if math.isclose(v1_sq_actual, v1_sq_calc, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Free fall equation valid: v1² = {v1_sq_actual:.2f} ≈ 2*g*x1 = {v1_sq_calc:.2f}", True)
                else:
                    self.log(f"Free fall equation violated: v1² = {v1_sq_actual:.2f}, 2*g*x1 = {v1_sq_calc:.2f}", False)
            
            # 2. Deceleration phase: v2^2 - v1^2 = 2*a*x3
            if v1 is not None and a is not None:
                left_side = self.v2**2 - v1**2
                right_side = 2 * a * self.x3
                if math.isclose(left_side, right_side, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Deceleration equation valid: v2²-v1² = {left_side:.2f} ≈ 2*a*x3 = {right_side:.2f}", True)
                else:
                    self.log(f"Deceleration equation violated: v2²-v1² = {left_side:.2f}, 2*a*x3 = {right_side:.2f}", False)
            
            # 3. Uniform motion: x2 = v1 * t2
            if v1 is not None and t2 is not None:
                x2_calc = v1 * t2
                if math.isclose(x2_calc, self.x2, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Uniform motion valid: v1*t2 = {x2_calc:.2f} ≈ x2 = {self.x2:.2f}", True)
                else:
                    self.log(f"Uniform motion violated: v1*t2 = {x2_calc:.2f}, x2 = {self.x2:.2f}", False)
            
            # 4. Free fall time: t1 = v1/g
            if v1 is not None and t1 is not None:
                t1_calc = v1 / self.g
                if math.isclose(t1, t1_calc, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Free fall time valid: t1 = {t1:.2f} ≈ v1/g = {t1_calc:.2f}", True)
                else:
                    self.log(f"Free fall time violated: t1 = {t1:.2f}, v1/g = {t1_calc:.2f}", False)
            
            # 5. Deceleration time: t3 = (v2 - v1)/a
            if v1 is not None and a is not None and t3 is not None:
                if abs(a) > 1e-9:  # Avoid division by zero
                    t3_calc = (self.v2 - v1) / a
                    if math.isclose(t3, t3_calc, rel_tol=1e-2, abs_tol=1e-2):
                        self.log(f"Deceleration time valid: t3 = {t3:.2f} ≈ (v2-v1)/a = {t3_calc:.2f}", True)
                    else:
                        self.log(f"Deceleration time violated: t3 = {t3:.2f}, (v2-v1)/a = {t3_calc:.2f}", False)
            
            # 6. Total time: t = t1 + t2 + t3
            if t1 is not None and t2 is not None and t3 is not None and total_t is not None:
                t_sum = t1 + t2 + t3
                if math.isclose(total_t, t_sum, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Total time valid: t = {total_t:.2f} ≈ t1+t2+t3 = {t_sum:.2f}", True)
                else:
                    self.log(f"Total time violated: t = {total_t:.2f}, t1+t2+t3 = {t_sum:.2f}", False)
            
            # 7. |a| should equal absolute value of a
            if a is not None and abs_a is not None:
                if math.isclose(abs_a, abs(a), rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Acceleration magnitude valid: |a| = {abs_a:.2f} ≈ |{a:.2f}|", True)
                else:
                    self.log(f"Acceleration magnitude violated: |a| = {abs_a:.2f}, |a| from a = {abs(a):.2f}", False)
        
        except Exception as e:
            self.log(f"Invariant test failed with error: {str(e)}", False)
    
    def test_trace_consistency(self):
        """Test that intermediate results are consistent with each other"""
        try:
            v1 = self.results.get('v_1')
            a = self.results.get('a')
            t1 = self.results.get('t_1')
            t2 = self.results.get('t_2')
            t3 = self.results.get('t_3')
            
            # Check that v1 from step 1 is used consistently in all subsequent steps
            if v1 is not None:
                # Recalculate v1 from free fall
                v1_calc = math.sqrt(2 * self.g * self.x1)
                if math.isclose(v1, v1_calc, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"v1 consistent with free fall: {v1:.2f} ≈ √(2*g*x1) = {v1_calc:.2f}", True)
                else:
                    self.log(f"v1 inconsistent: {v1:.2f} ≠ √(2*g*x1) = {v1_calc:.2f}", False)
            
            # Check that deceleration a is negative (upward)
            if a is not None:
                if a < 0:
                    self.log(f"Acceleration direction correct (negative/upward): a = {a:.2f} m/s²", True)
                else:
                    self.log(f"Acceleration direction suspect (should be negative): a = {a:.2f} m/s²", False)
            
            # Verify time calculations are physically plausible
            times = [t1, t2, t3]
            if all(t is not None for t in times):
                if all(t > 0 for t in times):
                    self.log("All time intervals positive (physically plausible)", True)
                else:
                    self.log(f"Some time intervals non-positive: t1={t1}, t2={t2}, t3={t3}", False)
        
        except Exception as e:
            self.log(f"Trace consistency test failed with error: {str(e)}", False)
    
    def test_monotonicity(self):
        """Test that perturbations in input produce expected trends"""
        try:
            # Store original value
            original_x1 = self.x1
            
            # Test 1: Increase free fall distance should increase v1
            self.x1 = original_x1 * 1.1  # +10%
            v1_perturbed = math.sqrt(2 * self.g * self.x1)
            
            # Restore original
            self.x1 = original_x1
            v1_original = math.sqrt(2 * self.g * self.x1)
            
            if v1_perturbed > v1_original:
                self.log(f"Monotonicity check passed: x1↑ ({original_x1}→{original_x1*1.1}) ⇒ v1↑ ({v1_original:.2f}→{v1_perturbed:.2f})", True)
            else:
                self.log(f"Monotonicity check failed: v1 didn't increase with x1", False)
            
            # Test 2: Increase deceleration magnitude should decrease deceleration time
            a_original = self.results.get('a')
            v1_original = self.results.get('v_1')
            
            if a_original is not None and v1_original is not None:
                # Make deceleration 10% stronger (more negative)
                a_perturbed = a_original * 1.1  # More negative if original is negative
                t3_original = (self.v2 - v1_original) / a_original
                t3_perturbed = (self.v2 - v1_original) / a_perturbed
                
                # For negative a, stronger deceleration (more negative) means smaller |t3|?
                # Actually: t3 = (v2 - v1)/a, if a becomes more negative, t3 becomes less positive
                if a_original < 0:  # Original deceleration is upward
                    if abs(a_perturbed) > abs(a_original):  # Stronger deceleration
                        if abs(t3_perturbed) < abs(t3_original):
                            self.log(f"Stronger deceleration reduces deceleration time: |a|↑ ⇒ t3↓", True)
                        else:
                            self.log(f"Monotonicity suspect: stronger deceleration didn't reduce time", False)
            
        except Exception as e:
            self.log(f"Monotonicity test failed with error: {str(e)}", False)
    
    def run_all_tests(self):
        """Run all tests and return success status and logs"""
        self.tests_passed = True
        self.logs = []
        
        self.logs.append("=== Physics Checker Results ===")
        
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        return self.tests_passed, self.logs

# Optional: Example usage guard
if __name__ == "__main__":
    # Mock solution class for testing
    class MockSolution:
        def __init__(self):
            self.results = {
                'v_1': 50.0,
                'a': -4.6,
                '|a|': 4.6,
                't_1': 5.0,
                't_2': 3.0,
                't_3': 10.0,
                't': 18.0
            }
    
    mock_solution = MockSolution()
    checker = PhysicsChecker(mock_solution)
    success, logs = checker.run_all_tests()
    
    for log in logs:
        print(log)
    print(f"\nOverall: {'All tests passed' if success else 'Some tests failed'}")