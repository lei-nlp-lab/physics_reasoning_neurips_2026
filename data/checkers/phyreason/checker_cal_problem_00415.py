import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs (with defaults from problem context)
        self.a1 = 0.1  # acceleration phase (given in problem)
        self.t1 = 4.0  # acceleration time
        self.t3 = 2.0  # deceleration time
        self.total_time = 48.0  # total anthem duration
        
        # Calculate t2 (constant velocity phase) from total time
        self.t2 = self.total_time - self.t1 - self.t3
        
        # Extract results safely
        self.v = getattr(solution_instance, 'v', self.results.get('v', None))
        self.a_decel = getattr(solution_instance, 'a', self.results.get('a', None))
        
        # Expected values from physics
        self.expected_v = self.a1 * self.t1  # v = a1 * t1 = 0.4 m/s
        self.expected_a_decel = -self.expected_v / self.t3  # a_decel = (0 - v)/t3 = -0.2 m/s²
        
        # Total displacement calculation for consistency checks
        self.s1 = 0.5 * self.a1 * self.t1**2  # displacement during acceleration
        self.s2 = self.expected_v * self.t2  # displacement during constant velocity
        self.s3 = self.expected_v * self.t3 + 0.5 * self.expected_a_decel * self.t3**2  # during deceleration
        self.expected_total_displacement = self.s1 + self.s2 + self.s3  # should be 18.0 m
        
        # Logs storage
        self.logs = []
        
    def _log(self, message, is_error=False):
        """Helper to log messages"""
        self.logs.append(f"{'ERROR' if is_error else 'INFO'}: {message}")
        
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison with physics tolerance"""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        
    def test_invariants(self):
        """Test conservation laws and basic physics invariants"""
        try:
            # Test 1: Velocity after acceleration phase
            if self.v is None:
                self._log("Velocity 'v' not found in solution", is_error=True)
                return False
                
            if not self._is_close(self.v, self.expected_v):
                self._log(f"Velocity mismatch: expected {self.expected_v}, got {self.v}", is_error=True)
                return False
            self._log(f"Velocity check passed: v = {self.v} m/s ≈ {self.expected_v} m/s")
            
            # Test 2: Deceleration magnitude
            if self.a_decel is None:
                self._log("Deceleration 'a' not found in solution", is_error=True)
                return False
                
            if not self._is_close(self.a_decel, self.expected_a_decel):
                self._log(f"Deceleration mismatch: expected {self.expected_a_decel}, got {self.a_decel}", is_error=True)
                return False
            self._log(f"Deceleration check passed: a = {self.a_decel} m/s² ≈ {self.expected_a_decel} m/s²")
            
            # Test 3: Final velocity should be zero (from problem statement)
            final_v = self.expected_v + self.expected_a_decel * self.t3
            if not self._is_close(final_v, 0.0):
                self._log(f"Final velocity not zero: {final_v}", is_error=True)
                return False
            self._log("Final velocity check passed: v_final = 0 m/s")
            
            # Test 4: Time consistency
            total_calc_time = self.t1 + self.t2 + self.t3
            if not self._is_close(total_calc_time, self.total_time):
                self._log(f"Time sum mismatch: {total_calc_time} ≠ {self.total_time}", is_error=True)
                return False
            self._log(f"Time consistency check passed: t1 + t2 + t3 = {total_calc_time} s")
            
            return True
            
        except Exception as e:
            self._log(f"Invariant test failed with exception: {str(e)}", is_error=True)
            return False
            
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer"""
        try:
            # Use extracted v and a_decel to reconstruct total displacement
            if self.v is None or self.a_decel is None:
                self._log("Missing v or a_decel for consistency check", is_error=True)
                return False
                
            # Reconstruct phases using solution's values
            s1_calc = 0.5 * self.a1 * self.t1**2
            s2_calc = self.v * self.t2
            s3_calc = self.v * self.t3 + 0.5 * self.a_decel * self.t3**2
            total_displacement_calc = s1_calc + s2_calc + s3_calc
            
            # Check against expected total displacement (18.0 m)
            if not self._is_close(total_displacement_calc, self.expected_total_displacement):
                self._log(f"Total displacement inconsistency: calculated {total_displacement_calc}, expected {self.expected_total_displacement}", is_error=True)
                return False
            self._log(f"Trace consistency passed: total displacement = {total_displacement_calc} m ≈ {self.expected_total_displacement} m")
            
            # Verify v and a_decel relationship: a_decel = -v / t3
            expected_from_v = -self.v / self.t3
            if not self._is_close(self.a_decel, expected_from_v):
                self._log(f"v-a_decel relationship violated: a_decel = {self.a_decel}, -v/t3 = {expected_from_v}", is_error=True)
                return False
            self._log("v-a_decel relationship check passed")
            
            return True
            
        except Exception as e:
            self._log(f"Trace consistency test failed with exception: {str(e)}", is_error=True)
            return False
            
    def test_monotonicity(self):
        """Perturb an input and check expected trends"""
        try:
            # Perturb acceleration a1 by +10%
            perturbed_a1 = self.a1 * 1.1  # 0.11 m/s²
            
            # Recalculate with perturbed input
            perturbed_v = perturbed_a1 * self.t1
            perturbed_s1 = 0.5 * perturbed_a1 * self.t1**2
            perturbed_s2 = perturbed_v * self.t2
            perturbed_a_decel = -perturbed_v / self.t3
            perturbed_s3 = perturbed_v * self.t3 + 0.5 * perturbed_a_decel * self.t3**2
            perturbed_total = perturbed_s1 + perturbed_s2 + perturbed_s3
            
            # Original total displacement
            original_total = self.expected_total_displacement
            
            # Check monotonicity: increased acceleration should increase total displacement
            if perturbed_total <= original_total:
                self._log(f"Monotonicity violation: a1↑ from {self.a1} to {perturbed_a1}, but displacement {perturbed_total} ≤ {original_total}", is_error=True)
                return False
            self._log(f"Monotonicity check passed: a1↑ from {self.a1} to {perturbed_a1} → displacement↑ from {original_total} to {perturbed_total}")
            
            # Additional check: velocity should also increase monotonically with a1
            if perturbed_v <= self.expected_v:
                self._log(f"Velocity monotonicity violation: a1↑ but v {perturbed_v} ≤ {self.expected_v}", is_error=True)
                return False
            self._log(f"Velocity monotonicity: a1↑ → v↑ from {self.expected_v} to {perturbed_v}")
            
            # Restore original value implicitly (no mutation needed)
            return True
            
        except Exception as e:
            self._log(f"Monotonicity test failed with exception: {str(e)}", is_error=True)
            return False
            
    def run_all_tests(self):
        """Execute all tests and return success status and logs"""
        self.logs = []  # Reset logs
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"\nRunning {test_name} test...")
            try:
                if not test_func():
                    success = False
                    self._log(f"{test_name} test FAILED", is_error=True)
                else:
                    self._log(f"{test_name} test PASSED")
            except Exception as e:
                success = False
                self._log(f"{test_name} test CRASHED: {traceback.format_exc()}", is_error=True)
        
        return success, self.logs

# Example usage stub (not part of checker itself)
if __name__ == "__main__":
    # Mock solution class for testing
    class MockSolution:
        def __init__(self):
            self.v = 0.4
            self.a = -0.2
            self.results = {'v': 0.4, 'a': -0.2}
    
    mock_solution = MockSolution()
    checker = PhysicsChecker(mock_solution)
    success, logs = checker.run_all_tests()
    
    print(f"Success: {success}")
    for log in logs:
        print(log)