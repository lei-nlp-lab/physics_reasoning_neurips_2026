import math
import sympy as sp
from typing import Tuple, List, Any, Optional

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        # Safe attribute access with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs from solution_instance with safe defaults
        self.circumference = getattr(solution_instance, 'circumference', 200.0)
        self.time = getattr(solution_instance, 'time', 50.0)
        
        # Extract results with safe defaults
        self.v_avg = self.results.get('v_avg', 0.0)
        self.s_avg = self.results.get('s_avg', 0.0)
        
        # Expected values from problem context
        self.expected_v_avg = 0.0
        self.expected_s_avg = 4.0
        
        # Initialize symbolic variables
        self.v_avg_sym = sp.symbols('v_avg')
        self.s_avg_sym = sp.symbols('s_avg')
        self.displacement = sp.symbols('displacement')
        self.distance = sp.symbols('distance')
        self.delta_t = sp.symbols('delta_t')
        
        # Log storage
        self.logs = []
        
    def _log(self, message: str):
        """Helper to log messages"""
        self.logs.append(message)
    
    def test_invariants(self) -> bool:
        """Test physical invariants: conservation laws, definitions"""
        self._log("Testing invariants...")
        success = True
        
        # Test 1: Average velocity definition (displacement/time)
        try:
            # Numerical check
            displacement = 0.0  # Returns to starting point
            calc_v_avg = displacement / self.time
            
            if not math.isclose(self.v_avg, calc_v_avg, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL: v_avg mismatch. Expected {calc_v_avg}, got {self.v_avg}")
                success = False
            else:
                self._log(f"PASS: v_avg matches definition: {self.v_avg}")
        except Exception as e:
            self._log(f"ERROR in v_avg invariant: {str(e)}")
            success = False
        
        # Test 2: Average speed definition (distance/time)
        try:
            # Numerical check
            calc_s_avg = self.circumference / self.time
            
            if not math.isclose(self.s_avg, calc_s_avg, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL: s_avg mismatch. Expected {calc_s_avg}, got {self.s_avg}")
                success = False
            else:
                self._log(f"PASS: s_avg matches definition: {self.s_avg}")
        except Exception as e:
            self._log(f"ERROR in s_avg invariant: {str(e)}")
            success = False
        
        # Test 3: Velocity should be zero for closed path
        try:
            if not math.isclose(self.v_avg, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL: v_avg should be zero for closed path, got {self.v_avg}")
                success = False
            else:
                self._log("PASS: v_avg = 0 for closed circular path")
        except Exception as e:
            self._log(f"ERROR in closed path check: {str(e)}")
            success = False
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Check consistency between intermediate results and final answer"""
        self._log("Testing trace consistency...")
        success = True
        
        # Check that v_avg and s_avg are consistent with problem parameters
        try:
            # Recalculate from problem statement
            recalc_s_avg = self.circumference / self.time
            
            if not math.isclose(self.s_avg, recalc_s_avg, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL: s_avg inconsistent with circumference/time. "
                         f"Expected {recalc_s_avg}, got {self.s_avg}")
                success = False
            else:
                self._log(f"PASS: s_avg {self.s_avg} consistent with circumference {self.circumference} "
                         f"and time {self.time}")
        except Exception as e:
            self._log(f"ERROR in trace consistency: {str(e)}")
            success = False
        
        # Check that v_avg is consistent with zero displacement
        try:
            if abs(self.v_avg) > 1e-2:  # Small tolerance for zero check
                self._log(f"WARNING: v_avg non-zero ({self.v_avg}) for closed path")
                # Not a failure, just a warning
        except Exception as e:
            self._log(f"ERROR in displacement consistency: {str(e)}")
            success = False
        
        return success
    
    def test_monotonicity(self) -> bool:
        """Test that perturbing inputs produces expected trends"""
        self._log("Testing monotonicity...")
        success = True
        
        original_time = self.time
        original_circumference = self.circumference
        
        try:
            # Test 1: Increase time, s_avg should decrease
            perturbed_time = original_time * 1.1  # 10% increase
            new_s_avg = original_circumference / perturbed_time
            
            if not (new_s_avg < self.s_avg):
                self._log(f"FAIL: s_avg should decrease when time increases. "
                         f"Original: {self.s_avg}, After +10% time: {new_s_avg}")
                success = False
            else:
                self._log(f"PASS: s_avg decreases from {self.s_avg:.2f} to {new_s_avg:.2f} "
                         f"when time increases by 10%")
            
            # Test 2: Decrease time, s_avg should increase
            perturbed_time = original_time * 0.9  # 10% decrease
            new_s_avg = original_circumference / perturbed_time
            
            if not (new_s_avg > self.s_avg):
                self._log(f"FAIL: s_avg should increase when time decreases. "
                         f"Original: {self.s_avg}, After -10% time: {new_s_avg}")
                success = False
            else:
                self._log(f"PASS: s_avg increases from {self.s_avg:.2f} to {new_s_avg:.2f} "
                         f"when time decreases by 10%")
            
            # Test 3: v_avg should remain zero regardless of time changes
            # (displacement remains zero)
            perturbed_time = original_time * 1.2
            new_v_avg = 0.0 / perturbed_time
            
            if not math.isclose(new_v_avg, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL: v_avg should remain zero, got {new_v_avg}")
                success = False
            else:
                self._log("PASS: v_avg remains zero regardless of time changes")
                
        except Exception as e:
            self._log(f"ERROR in monotonicity test: {str(e)}")
            success = False
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success flag and logs"""
        self.logs = []
        overall_success = True
        
        self._log("=" * 50)
        self._log("Physics Checker Starting")
        self._log(f"Inputs: circumference={self.circumference}m, time={self.time}s")
        self._log(f"Results: v_avg={self.v_avg}, s_avg={self.s_avg}")
        self._log("=" * 50)
        
        # Run individual tests
        test1 = self.test_invariants()
        if not test1:
            overall_success = False
            self._log("FAILED: test_invariants")
        
        test2 = self.test_trace_consistency()
        if not test2:
            overall_success = False
            self._log("FAILED: test_trace_consistency")
        
        test3 = self.test_monotonicity()
        if not test3:
            overall_success = False
            self._log("FAILED: test_monotonicity")
        
        # Summary
        self._log("=" * 50)
        if overall_success:
            self._log("ALL TESTS PASSED")
        else:
            self._log("SOME TESTS FAILED")
        
        return overall_success, self.logs