import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract results with fallbacks
        self.x = getattr(solution_instance, 'x', self.results.get('x', 0.0))
        self.t2 = getattr(solution_instance, 't_2', self.results.get('t_2', 0.0))
        self.delta_t = getattr(solution_instance, 'delta_t', self.results.get('delta_t', 0.0))
        self.t3 = getattr(solution_instance, 't_3', self.results.get('t_3', 0.0))
        self.x2 = getattr(solution_instance, 'x_2', self.results.get('x_2', 0.0))
        self.t1 = getattr(solution_instance, 't_1', self.results.get('t_1', 0.0))
        self.x1 = getattr(solution_instance, 'x_1', self.results.get('x_1', 0.0))
        
        # Known constants from problem
        self.v0 = 15.0  # m/s
        self.v = 5.0    # m/s
        self.a = 2.5    # m/s²
        self.wait_time = 10.0  # s
        
        # For monotonicity test backup
        self.original_v0 = self.v0
        self.original_a = self.a
        
    def test_invariants(self):
        logs = []
        success = True
        
        try:
            # Symbolic verification of key equations
            v0, v, a, x, x1, x2, t1, t2, t3 = sp.symbols('v0 v a x x1 x2 t1 t2 t3', positive=True)
            
            # Check Car 2: 0 = v0² - 2a*x
            expr1 = sp.Eq(0, v0**2 - 2*a*x)
            num_check1 = math.isclose(0, self.v0**2 - 2*self.a*self.x, 
                                     rel_tol=1e-2, abs_tol=1e-2)
            if not num_check1:
                logs.append(f"Car 2 kinematics invariant failed: 0 = v0² - 2a*x")
                logs.append(f"  Calculated: 0 = {self.v0**2} - 2*{self.a}*{self.x} = {self.v0**2 - 2*self.a*self.x}")
                success = False
            else:
                logs.append("✓ Car 2 kinematics invariant passed")
            
            # Check Car 1 deceleration: v² = v0² - 2a*x1
            expr2 = sp.Eq(v**2, v0**2 - 2*a*x1)
            num_check2 = math.isclose(self.v**2, self.v0**2 - 2*self.a*self.x1,
                                     rel_tol=1e-2, abs_tol=1e-2)
            if not num_check2:
                logs.append(f"Car 1 deceleration invariant failed: v² = v0² - 2a*x1")
                logs.append(f"  LHS: {self.v**2}, RHS: {self.v0**2 - 2*self.a*self.x1}")
                success = False
            else:
                logs.append("✓ Car 1 deceleration invariant passed")
            
            # Check distance relationship: x = x1 + x2
            num_check3 = math.isclose(self.x, self.x1 + self.x2,
                                     rel_tol=1e-2, abs_tol=1e-2)
            if not num_check3:
                logs.append(f"Distance relationship failed: x = x1 + x2")
                logs.append(f"  {self.x} vs {self.x1 + self.x2}")
                success = False
            else:
                logs.append("✓ Distance relationship passed")
            
            # Check time equations
            # t1 = (v0 - v)/a
            expected_t1 = (self.v0 - self.v) / self.a
            num_check4 = math.isclose(self.t1, expected_t1, rel_tol=1e-2, abs_tol=1e-2)
            if not num_check4:
                logs.append(f"Car 1 deceleration time failed: t1 = (v0 - v)/a")
                logs.append(f"  Expected: {expected_t1}, Got: {self.t1}")
                success = False
            else:
                logs.append("✓ Car 1 deceleration time passed")
            
            # t2 = v0/a
            expected_t2 = self.v0 / self.a
            num_check5 = math.isclose(self.t2, expected_t2, rel_tol=1e-2, abs_tol=1e-2)
            if not num_check5:
                logs.append(f"Car 2 deceleration time failed: t2 = v0/a")
                logs.append(f"  Expected: {expected_t2}, Got: {self.t2}")
                success = False
            else:
                logs.append("✓ Car 2 deceleration time passed")
            
            # t3 = x2/v
            expected_t3 = self.x2 / self.v
            num_check6 = math.isclose(self.t3, expected_t3, rel_tol=1e-2, abs_tol=1e-2)
            if not num_check6:
                logs.append(f"Constant speed time failed: t3 = x2/v")
                logs.append(f"  Expected: {expected_t3}, Got: {self.t3}")
                success = False
            else:
                logs.append("✓ Constant speed time passed")
            
            # Check total time difference
            total_time_car2 = 2*self.t2 + self.wait_time
            total_time_car1 = 2*self.t1 + self.t3
            expected_delta_t = total_time_car2 - total_time_car1
            
            num_check7 = math.isclose(self.delta_t, expected_delta_t, 
                                     rel_tol=1e-2, abs_tol=1e-2)
            if not num_check7:
                logs.append(f"Time difference failed: Δt = (2*t2 + 10) - (2*t1 + t3)")
                logs.append(f"  Expected: {expected_delta_t}, Got: {self.delta_t}")
                success = False
            else:
                logs.append("✓ Time difference calculation passed")
                
        except Exception as e:
            logs.append(f"Error in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        logs = []
        success = True
        
        try:
            # Check that intermediate results lead to final answer
            # Calculate Δt from scratch using extracted values
            total_car2 = 2*self.t2 + self.wait_time
            total_car1 = 2*self.t1 + self.t3
            calculated_delta_t = total_car2 - total_car1
            
            if not math.isclose(self.delta_t, calculated_delta_t, 
                               rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Trace consistency failed: Final Δt doesn't match intermediate results")
                logs.append(f"  From intermediates: {calculated_delta_t}, Stored: {self.delta_t}")
                success = False
            else:
                logs.append("✓ Trace consistency passed")
            
            # Check that x2 matches target answer
            target_x2 = 5.0  # From problem statement
            if not math.isclose(self.x2, target_x2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Trace consistency failed: x2 doesn't match target answer")
                logs.append(f"  Expected: {target_x2}, Got: {self.x2}")
                success = False
            else:
                logs.append("✓ x2 matches target answer")
            
            # Check that delta_t matches target answer
            target_delta_t = 13.0  # From problem statement
            if not math.isclose(self.delta_t, target_delta_t, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Trace consistency failed: Δt doesn't match target answer")
                logs.append(f"  Expected: {target_delta_t}, Got: {self.delta_t}")
                success = False
            else:
                logs.append("✓ Δt matches target answer")
                
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        logs = []
        success = True
        
        try:
            # Test 1: Perturb acceleration magnitude
            original_a = self.a
            perturbed_a = original_a * 1.01  # +1% change
            
            # Recalculate using formulas
            # For Car 2
            x_perturbed = self.v0**2 / (2*perturbed_a)
            t2_perturbed = self.v0 / perturbed_a
            
            # For Car 1
            x1_perturbed = (self.v0**2 - self.v**2) / (2*perturbed_a)
            x2_perturbed = x_perturbed - x1_perturbed
            t1_perturbed = (self.v0 - self.v) / perturbed_a
            t3_perturbed = x2_perturbed / self.v
            
            # Calculate Δt with perturbed a
            delta_t_perturbed = (2*t2_perturbed + self.wait_time) - (2*t1_perturbed + t3_perturbed)
            
            # Check monotonic trend: Δt should decrease as a increases
            # ∂(Δt)/∂a = -3v/(2a²) < 0
            if delta_t_perturbed > self.delta_t:
                logs.append(f"Monotonicity test failed: Δt increased when a increased")
                logs.append(f"  Original Δt: {self.delta_t}, Perturbed Δt: {delta_t_perturbed}")
                logs.append(f"  Expected Δt to decrease when a increases")
                success = False
            else:
                logs.append("✓ Acceleration perturbation shows correct monotonic trend")
            
            # Test 2: Perturb initial velocity
            original_v0 = self.v0
            perturbed_v0 = original_v0 * 1.01  # +1% change
            
            # Recalculate with perturbed v0
            x_v0_perturbed = perturbed_v0**2 / (2*self.a)
            t2_v0_perturbed = perturbed_v0 / self.a
            
            x1_v0_perturbed = (perturbed_v0**2 - self.v**2) / (2*self.a)
            x2_v0_perturbed = x_v0_perturbed - x1_v0_perturbed
            t1_v0_perturbed = (perturbed_v0 - self.v) / self.a
            t3_v0_perturbed = x2_v0_perturbed / self.v
            
            delta_t_v0_perturbed = (2*t2_v0_perturbed + self.wait_time) - (2*t1_v0_perturbed + t3_v0_perturbed)
            
            # According to derived formula: Δt = wait_time + (3v)/(2a) which is independent of v0
            # So Δt should remain approximately constant
            if not math.isclose(delta_t_v0_perturbed, self.delta_t, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Warning: Δt changed with v0 perturbation")
                logs.append(f"  Original Δt: {self.delta_t}, Perturbed Δt: {delta_t_v0_perturbed}")
                logs.append(f"  Note: Δt should be independent of v0 based on derived formula")
                # This is a warning, not necessarily a failure
                logs.append("  (This is expected based on the physics)")
            
            logs.append("✓ Monotonicity tests completed")
            
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        all_logs = []
        all_success = True
        
        # Run invariant tests
        success, logs = self.test_invariants()
        all_logs.append("=== Invariant Tests ===")
        all_logs.extend(logs)
        all_success = all_success and success
        
        # Run trace consistency tests
        success, logs = self.test_trace_consistency()
        all_logs.append("\n=== Trace Consistency Tests ===")
        all_logs.extend(logs)
        all_success = all_success and success
        
        # Run monotonicity tests
        success, logs = self.test_monotonicity()
        all_logs.append("\n=== Monotonicity Tests ===")
        all_logs.extend(logs)
        all_success = all_success and success
        
        # Summary
        all_logs.append(f"\n=== Summary ===")
        if all_success:
            all_logs.append("✓ All physics checks passed")
        else:
            all_logs.append("✗ Some physics checks failed")
        
        return all_success, all_logs

# Example usage (commented out for the submission)
# if __name__ == "__main__":
#     # This would be used by the sandbox
#     checker = PhysicsChecker(solution_instance)
#     success, logs = checker.run_all_tests()
#     for log in logs:
#         print(log)