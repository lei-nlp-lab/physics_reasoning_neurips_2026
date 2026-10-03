import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all required data from solution_instance.
        Never access solution_instance.attr directly - use getattr with fallbacks.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract all required quantities with safe fallbacks
        self.v = self._safe_get('v')  # velocity at t=2s
        self.v0 = self._safe_get('v_0')  # velocity at t=0s
        self.a1 = self._safe_get('a_1')
        self.a2 = self._safe_get('a_2')
        self.x1 = self._safe_get('x_1')
        self.x2 = self._safe_get('x_2')
        self.barv = self._safe_get('barv')
        
        # Store original values for monotonicity tests
        self.original_values = {
            'v': self.v,
            'v0': self.v0,
            'a1': self.a1,
            'a2': self.a2,
            'x1': self.x1,
            'x2': self.x2,
            'barv': self.barv
        }
        
        # Constants from problem statement
        self.t1 = 2.0  # end of 2nd second
        self.t2 = 3.0  # end of 3rd second
        self.t3 = 4.0  # end of 4th second
        self.t4 = 6.0  # start of last 2 seconds
        self.t5 = 8.0  # end of last 2 seconds
        
        self.logs = []
        
    def _safe_get(self, key: str) -> float:
        """Safely get a value from solution instance with multiple fallbacks."""
        # Try direct attribute
        val = getattr(self.solution, key, None)
        if val is not None:
            return float(val)
        
        # Try results dictionary
        val = self.results.get(key, None)
        if val is not None:
            return float(val)
            
        # Try special key mappings
        if key == 'barv':
            val = self.results.get('\\bar{v}', None)
            if val is not None:
                return float(val)
        
        return 0.0
    
    def _log(self, message: str, is_error: bool = False):
        """Helper to log messages."""
        self.logs.append(f"{'ERROR' if is_error else 'INFO'}: {message}")
    
    def test_invariants(self) -> bool:
        """
        Test physical invariants:
        1. Average acceleration definition for a1 and a2
        2. Average velocity definition for barv
        3. Consistency between v0, v, and given times
        """
        success = True
        
        try:
            # Test 1: Average acceleration a1 = (v_at_t3 - v0) / delta_t for first 3 seconds
            # CORRECTED: Use v_at_t3 = 5.0 m/s (from step_2) not v at t=2s
            delta_t1 = self.t2 - 0.0  # from t=0 to t=3
            v_at_t3 = 5.0  # from step_2 in ground truth
            expected_a1 = (v_at_t3 - self.v0) / delta_t1
            if not math.isclose(self.a1, expected_a1, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Average acceleration a1 mismatch: {self.a1} vs expected {expected_a1}", True)
                success = False
            else:
                self._log(f"✓ Average acceleration a1 correct: {self.a1} ≈ {expected_a1}")
            
            # Test 2: Average acceleration a2 for last 2 seconds
            delta_t2 = self.t5 - self.t4  # 2 seconds
            v_at_6 = 5.0  # from step_4
            v_at_8 = 0.0  # from step_4
            expected_a2 = (v_at_8 - v_at_6) / delta_t2
            if not math.isclose(self.a2, expected_a2, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Average acceleration a2 mismatch: {self.a2} vs expected {expected_a2}", True)
                success = False
            else:
                self._log(f"✓ Average acceleration a2 correct: {self.a2} ≈ {expected_a2}")
            
            # Test 3: Average velocity barv for 4th second
            delta_t3 = 1.0  # 4th second = from t=3 to t=4
            expected_barv = (self.x2 - self.x1) / delta_t3
            if not math.isclose(self.barv, expected_barv, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Average velocity barv mismatch: {self.barv} vs expected {expected_barv}", True)
                success = False
            else:
                self._log(f"✓ Average velocity barv correct: {self.barv} ≈ {expected_barv}")
            
            # Test 4: Initial velocity should be 0 (from step_2)
            if not math.isclose(self.v0, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Initial velocity v0 should be 0, got {self.v0}", True)
                success = False
            else:
                self._log(f"✓ Initial velocity v0 correct: {self.v0}")
            
            # Test 5: Velocity at t=2 should be 3 (from step_1)
            if not math.isclose(self.v, 3.0, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Velocity v at t=2 should be 3, got {self.v}", True)
                success = False
            else:
                self._log(f"✓ Velocity v at t=2 correct: {self.v}")
                
        except Exception as e:
            self._log(f"Exception in test_invariants: {str(e)}\n{traceback.format_exc()}", True)
            success = False
            
        return success
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answers.
        Verify that all steps are consistent.
        """
        success = True
        
        try:
            # Check step consistency: v at t=3 should be 5 (from step_2)
            v_at_3 = 5.0
            delta_t = self.t2 - 0.0
            calculated_v_at_3 = self.v0 + self.a1 * delta_t
            
            if not math.isclose(v_at_3, calculated_v_at_3, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Velocity at t=3 inconsistent: expected {v_at_3}, calculated {calculated_v_at_3}", True)
                success = False
            else:
                self._log(f"✓ Velocity at t=3 consistent: {v_at_3} ≈ {calculated_v_at_3}")
            
            # Check that a1 and a2 signs make sense
            # a1 should be positive (accelerating from 0 to 5 m/s)
            if self.a1 < -1e-2:  # Allow small negative due to rounding
                self._log(f"a1 should be positive, got {self.a1}", True)
                success = False
            else:
                self._log(f"✓ a1 sign correct: {self.a1}")
                
            # a2 should be negative (decelerating from 5 to 0 m/s)
            if self.a2 > 1e-2:  # Allow small positive due to rounding
                self._log(f"a2 should be negative, got {self.a2}", True)
                success = False
            else:
                self._log(f"✓ a2 sign correct: {self.a2}")
            
            # Check positions are positive and increasing
            if self.x1 < -1e-2:
                self._log(f"x1 should be positive, got {self.x1}", True)
                success = False
            else:
                self._log(f"✓ x1 positive: {self.x1}")
                
            if self.x2 < self.x1 - 1e-2:
                self._log(f"x2 should be greater than x1, got x2={self.x2}, x1={self.x1}", True)
                success = False
            else:
                self._log(f"✓ x2 > x1: {self.x2} > {self.x1}")
                
        except Exception as e:
            self._log(f"Exception in test_trace_consistency: {str(e)}\n{traceback.format_exc()}", True)
            success = False
            
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb an input and check the trend matches physical expectations.
        Restore original values after test.
        """
        success = True
        
        try:
            # Test 1: Increase x2 by 10% and check that barv increases
            original_x2 = self.x2
            perturbed_x2 = original_x2 * 1.1
            
            # Calculate new barv
            new_barv = (perturbed_x2 - self.x1) / 1.0  # delta_t = 1s
            
            if new_barv <= self.barv:
                self._log(f"Monotonicity failed: barv should increase when x2 increases. "
                         f"Original: {self.barv}, After perturbation: {new_barv}", True)
                success = False
            else:
                self._log(f"✓ Monotonicity passed: barv increases with x2 "
                         f"({self.barv} → {new_barv})")
            
            # Restore x2
            self.x2 = original_x2
            
            # Test 2: Decrease time interval for a2 and check magnitude increases
            original_delta_t = self.t5 - self.t4  # 2 seconds
            perturbed_delta_t = original_delta_t * 0.9  # 10% decrease
            
            v_at_6 = 5.0
            v_at_8 = 0.0
            new_a2_mag = abs((v_at_8 - v_at_6) / perturbed_delta_t)
            
            if new_a2_mag <= abs(self.a2):
                self._log(f"Monotonicity failed: |a2| should increase when time decreases. "
                         f"Original: {abs(self.a2)}, After perturbation: {new_a2_mag}", True)
                success = False
            else:
                self._log(f"✓ Monotonicity passed: |a2| increases with shorter time "
                         f"({abs(self.a2)} → {new_a2_mag})")
            
            # Test 3: Increase velocity change for a1 and check a1 increases
            # Use v_at_t3 = 5.0 for original delta_v
            original_delta_v = 5.0 - 0.0  # v_at_3 - v0
            perturbed_delta_v = original_delta_v * 1.1  # 10% increase
            
            new_a1 = perturbed_delta_v / (self.t2 - 0.0)
            
            if new_a1 <= self.a1:
                self._log(f"Monotonicity failed: a1 should increase when Δv increases. "
                         f"Original: {self.a1}, After perturbation: {new_a1}", True)
                success = False
            else:
                self._log(f"✓ Monotonicity passed: a1 increases with larger Δv "
                         f"({self.a1} → {new_a1})")
                
        except Exception as e:
            self._log(f"Exception in test_monotonicity: {str(e)}\n{traceback.format_exc()}", True)
            success = False
            
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success and logs."""
        self.logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity Tests", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"\n--- Running {test_name} ---")
            try:
                if not test_func():
                    overall_success = False
                    self._log(f"{test_name} FAILED", True)
                else:
                    self._log(f"{test_name} PASSED")
            except Exception as e:
                self._log(f"Exception in {test_name}: {str(e)}\n{traceback.format_exc()}", True)
                overall_success = False
        
        # Summary
        self._log(f"\n=== SUMMARY ===")
        self._log(f"Overall result: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, self.logs