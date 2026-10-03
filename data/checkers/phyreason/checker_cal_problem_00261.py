import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from solution instance."""
        # Extract results with safe fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract key quantities with safe access
        self.d = self.results.get('d', None)
        self.x_A1 = self.results.get('x_A1', None)
        self.t_2 = self.results.get('t_2', None)
        self.t_total = self.results.get('t_total', None)
        self.a = self.results.get('a', None)
        self.x_B = self.results.get('x_textB', None)
        self.x_A = self.results.get('x_textA', None)
        self.t_1 = self.results.get('t_1', None)
        self.x_B1 = self.results.get('x_B1', None)
        
        # Known constants from problem statement
        self.v_A = 4.0  # Car A's constant velocity
        self.v_0 = 10.0  # Car B's initial velocity
        self.x0 = 35.0  # Initial separation
        
        # Store logs for reporting
        self.logs = []

    def _log(self, message, success=True):
        """Helper to log test results."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {message}")
        return success

    def _is_close(self, val1, val2, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison for physics quantities."""
        if val1 is None or val2 is None:
            return False
        return math.isclose(val1, val2, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """Test fundamental physics laws from KB."""
        tests_passed = []
        
        try:
            # 1. Check acceleration magnitude (should be positive 2 m/s²)
            if self.a is not None:
                expected_a_mag = 2.0
                if self._is_close(abs(self.a), expected_a_mag):
                    tests_passed.append(self._log(f"Acceleration magnitude matches: {abs(self.a)} vs {expected_a_mag}"))
                else:
                    tests_passed.append(self._log(f"Acceleration mismatch: {abs(self.a)} vs {expected_a_mag}", False))
            
            # 2. Check velocity equality at t_1
            if self.t_1 is not None and self.a is not None:
                v_B_at_t1 = self.v_0 + self.a * self.t_1  # a is negative
                if self._is_close(v_B_at_t1, self.v_A):
                    tests_passed.append(self._log(f"Velocity equality at t_1 holds: {v_B_at_t1} = {self.v_A}"))
                else:
                    tests_passed.append(self._log(f"Velocity mismatch at t_1: {v_B_at_t1} ≠ {self.v_A}", False))
            
            # 3. Check average velocity theorem for car B during t_1
            if all(v is not None for v in [self.x_B1, self.t_1]):
                v_avg_B = (self.v_0 + self.v_A) / 2
                expected_x_B1 = v_avg_B * self.t_1
                if self._is_close(self.x_B1, expected_x_B1):
                    tests_passed.append(self._log(f"Average velocity theorem for B: x_B1 = {self.x_B1} matches {expected_x_B1}"))
                else:
                    tests_passed.append(self._log(f"Average velocity theorem fails for B: {self.x_B1} ≠ {expected_x_B1}", False))
            
            # 4. Check uniform motion for car A during t_1
            if all(v is not None for v in [self.x_A1, self.t_1]):
                expected_x_A1 = self.v_A * self.t_1
                if self._is_close(self.x_A1, expected_x_A1):
                    tests_passed.append(self._log(f"Uniform motion for A: x_A1 = {self.x_A1} matches {expected_x_A1}"))
                else:
                    tests_passed.append(self._log(f"Uniform motion fails for A: {self.x_A1} ≠ {expected_x_A1}", False))
            
            # 5. Check geometry relationship for maximum distance
            if all(v is not None for v in [self.d, self.x_B1, self.x_A1]):
                expected_d = self.x0 + self.x_B1 - self.x_A1
                if self._is_close(self.d, expected_d):
                    tests_passed.append(self._log(f"Geometry relationship: d = {self.d} matches {expected_d}"))
                else:
                    tests_passed.append(self._log(f"Geometry relationship fails: {self.d} ≠ {expected_d}", False))
            
            # 6. Check stopping distance for B
            if self.x_B is not None:
                t_stop = 5.0  # Given in problem
                expected_x_B = (self.v_0 / 2) * t_stop
                if self._is_close(self.x_B, expected_x_B):
                    tests_passed.append(self._log(f"Stopping distance for B: x_B = {self.x_B} matches {expected_x_B}"))
                else:
                    tests_passed.append(self._log(f"Stopping distance fails: {self.x_B} ≠ {expected_x_B}", False))
            
            # 7. Check distance for A during B's stop
            if self.x_A is not None:
                t_stop = 5.0
                expected_x_A = self.v_A * t_stop
                if self._is_close(self.x_A, expected_x_A):
                    tests_passed.append(self._log(f"Distance for A during B's stop: x_A = {self.x_A} matches {expected_x_A}"))
                else:
                    tests_passed.append(self._log(f"Distance for A fails: {self.x_A} ≠ {expected_x_A}", False))
            
            # 8. Check catch-up time t_2
            if all(v is not None for v in [self.t_2, self.x_B, self.x_A]):
                expected_t_2 = (self.x_B + self.x0 - self.x_A) / self.v_A
                if self._is_close(self.t_2, expected_t_2):
                    tests_passed.append(self._log(f"Catch-up time: t_2 = {self.t_2} matches {expected_t_2}"))
                else:
                    tests_passed.append(self._log(f"Catch-up time fails: {self.t_2} ≠ {expected_t_2}", False))
            
            # 9. Check total time
            if all(v is not None for v in [self.t_total, self.t_2]):
                t_stop = 5.0
                expected_total = t_stop + self.t_2
                if self._is_close(self.t_total, expected_total):
                    tests_passed.append(self._log(f"Total time: t_total = {self.t_total} matches {expected_total}"))
                else:
                    tests_passed.append(self._log(f"Total time fails: {self.t_total} ≠ {expected_total}", False))
            
            return all(tests_passed)
            
        except Exception as e:
            self._log(f"Invariant test failed with error: {str(e)}", False)
            return False

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        tests_passed = []
        
        try:
            # Target answers from problem: 44m and 15s
            target_d = 44.0
            target_t_total = 15.0
            
            # Check against computed values
            if self.d is not None:
                if self._is_close(self.d, target_d):
                    tests_passed.append(self._log(f"Maximum distance d={self.d} matches target {target_d}m"))
                else:
                    tests_passed.append(self._log(f"Maximum distance mismatch: {self.d} ≠ {target_d}", False))
            
            if self.t_total is not None:
                if self._is_close(self.t_total, target_t_total):
                    tests_passed.append(self._log(f"Total time t_total={self.t_total} matches target {target_t_total}s"))
                else:
                    tests_passed.append(self._log(f"Total time mismatch: {self.t_total} ≠ {target_t_total}", False))
            
            # Check that all steps are consistent with each other
            if all(v is not None for v in [self.x_B1, self.x_A1]):
                reconstructed_d = self.x0 + self.x_B1 - self.x_A1
                if self.d is not None and self._is_close(self.d, reconstructed_d):
                    tests_passed.append(self._log(f"Trace consistency: d={self.d} = x0 + x_B1 - x_A1 = {reconstructed_d}"))
                else:
                    tests_passed.append(self._log(f"Trace inconsistency in d calculation", False))
            
            if all(v is not None for v in [self.t_2]):
                t_stop = 5.0
                reconstructed_total = t_stop + self.t_2
                if self.t_total is not None and self._is_close(self.t_total, reconstructed_total):
                    tests_passed.append(self._log(f"Trace consistency: t_total={self.t_total} = t_stop + t_2 = {reconstructed_total}"))
                else:
                    tests_passed.append(self._log(f"Trace inconsistency in t_total calculation", False))
            
            return all(tests_passed)
            
        except Exception as e:
            self._log(f"Trace consistency test failed with error: {str(e)}", False)
            return False

    def test_monotonicity(self):
        """Perturb an input and check expected trends."""
        if self.a is None:
            self._log("Cannot test monotonicity: acceleration not available", False)
            return False
            
        original_a_mag = abs(self.a)
        
        try:
            # Perturb acceleration magnitude by +10%
            perturbed_a_mag = original_a_mag * 1.1
            perturbed_a = -perturbed_a_mag  # Negative for deceleration
            
            # Store original values for restoration
            original_values = {
                'd': self.d,
                't_total': self.t_total,
                't_1': self.t_1,
                'x_B1': self.x_B1,
                'x_A1': self.x_A1,
                'x_B': self.x_B,
                'x_A': self.x_A,
                't_2': self.t_2
            }
            
            # Recalculate with perturbed acceleration
            # 1. New t_1 (time when velocities equal)
            new_t_1 = (self.v_0 - self.v_A) / perturbed_a_mag
            
            # 2. New displacements during t_1
            new_x_B1 = ((self.v_0 + self.v_A) / 2) * new_t_1
            new_x_A1 = self.v_A * new_t_1
            
            # 3. New maximum distance
            new_d = self.x0 + new_x_B1 - new_x_A1
            
            # 4. New stopping time for B (using magnitude)
            new_t_stop = self.v_0 / perturbed_a_mag
            
            # 5. New stopping distances
            new_x_B = (self.v_0 / 2) * new_t_stop
            new_x_A = self.v_A * new_t_stop
            
            # 6. New catch-up time
            new_t_2 = (new_x_B + self.x0 - new_x_A) / self.v_A
            
            # 7. New total time
            new_t_total = new_t_stop + new_t_2
            
            # Check monotonic trends
            tests_passed = []
            
            # With increased acceleration magnitude:
            # - B decelerates faster, so t_1 should decrease
            if new_t_1 < original_values['t_1']:
                tests_passed.append(self._log(f"Monotonicity: t_1 decreases with increased acceleration ({new_t_1:.3f} < {original_values['t_1']})"))
            else:
                tests_passed.append(self._log(f"Monotonicity violation: t_1 should decrease", False))
            
            # - Maximum distance d should decrease (B gets closer to A faster)
            if new_d < original_values['d']:
                tests_passed.append(self._log(f"Monotonicity: d decreases with increased acceleration ({new_d:.3f} < {original_values['d']})"))
            else:
                tests_passed.append(self._log(f"Monotonicity violation: d should decrease", False))
            
            # - Stopping time should decrease
            original_t_stop = self.v_0 / original_a_mag
            if new_t_stop < original_t_stop:
                tests_passed.append(self._log(f"Monotonicity: stopping time decreases with increased acceleration ({new_t_stop:.3f} < {original_t_stop:.3f})"))
            else:
                tests_passed.append(self._log(f"Monotonicity violation: stopping time should decrease", False))
            
            # - Total time should generally decrease
            if new_t_total < original_values['t_total']:
                tests_passed.append(self._log(f"Monotonicity: t_total decreases with increased acceleration ({new_t_total:.3f} < {original_values['t_total']})"))
            else:
                # This could be acceptable if the change is small
                if self._is_close(new_t_total, original_values['t_total']):
                    tests_passed.append(self._log(f"Monotonicity: t_total approximately unchanged"))
                else:
                    tests_passed.append(self._log(f"Monotonicity warning: t_total increased unexpectedly", False))
            
            return all(tests_passed)
            
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {str(e)}", False)
            return False

    def run_all_tests(self):
        """Execute all tests and return success status with logs."""
        self.logs = []
        
        try:
            inv_ok = self.test_invariants()
            trace_ok = self.test_trace_consistency()
            mono_ok = self.test_monotonicity()
            
            success = inv_ok and trace_ok and mono_ok
            
            # Add summary
            if success:
                self._log("All physics checks passed!")
            else:
                self._log("Some physics checks failed", False)
            
            return success, self.logs
            
        except Exception as e:
            error_msg = f"Checker crashed: {str(e)}\n{traceback.format_exc()}"
            self.logs.append(f"CRITICAL: {error_msg}")
            return False, self.logs