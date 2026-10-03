import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract known constants from problem
        self.a2 = 2.0  # m/s^2 (given)
        self.t1 = 5.0  # s (given)
        self.t2 = 10.0  # s (given)
        self.v_final = 40.0  # m/s (given)
        
        # Extract solution results with safe access
        self.v0 = self.results.get('v_0', 0.0)
        self.a1 = self.results.get('a_1', 0.0)
        
        # Test logs
        self.logs = []
        
    def _log(self, message, success):
        self.logs.append((message, success))
        
    def _isclose(self, a, b):
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        """Test physics invariants using motion equations"""
        try:
            # Test 1: Check final velocity equation v_final = v0 + a2*t2
            expected_v0 = self.v_final - self.a2 * self.t2
            if self._isclose(self.v0, expected_v0):
                self._log(f"Velocity continuity: v0 = {self.v0} ≈ v_final - a2*t2 = {expected_v0}", True)
            else:
                self._log(f"Velocity continuity FAILED: v0 = {self.v0} ≠ {expected_v0}", False)
                return False
            
            # Test 2: Check first phase: v0 = 0 + a1*t1
            expected_a1 = self.v0 / self.t1 if self.t1 != 0 else 0
            if self._isclose(self.a1, expected_a1):
                self._log(f"First phase acceleration: a1 = {self.a1} ≈ v0/t1 = {expected_a1}", True)
            else:
                self._log(f"First phase acceleration FAILED: a1 = {self.a1} ≠ {expected_a1}", False)
                return False
            
            # Test 3: Direct consistency check (no symbolic solving needed)
            # v0 should be positive and less than v_final
            if self.v0 > 0 and self.v0 < self.v_final:
                self._log(f"v0 bounds check: 0 < {self.v0} < {self.v_final}", True)
            else:
                self._log(f"v0 bounds FAILED: {self.v0} not between 0 and {self.v_final}", False)
                return False
            
            # a1 should be positive
            if self.a1 > 0:
                self._log(f"a1 positive check: {self.a1} > 0", True)
            else:
                self._log(f"a1 positive FAILED: {self.a1} ≤ 0", False)
                return False
            
            return True
            
        except Exception as e:
            self._log(f"Invariant test failed with error: {str(e)}", False)
            return False
    
    def test_trace_consistency(self):
        """Check intermediate steps against final answer"""
        try:
            # Step 1 check: v0 = v_final - a2*t2 = 40 - 2*10 = 20
            step1_value = 20.0
            if self._isclose(self.v0, step1_value):
                self._log(f"Step 1 consistency: v0 = {self.v0} ≈ {step1_value}", True)
            else:
                self._log(f"Step 1 FAILED: v0 = {self.v0} ≠ {step1_value}", False)
                return False
            
            # Step 2 check: a1 = v0/t1 = 20/5 = 4
            step2_value = 4.0
            if self._isclose(self.a1, step2_value):
                self._log(f"Step 2 consistency: a1 = {self.a1} ≈ {step2_value}", True)
            else:
                self._log(f"Step 2 FAILED: a1 = {self.a1} ≠ {step2_value}", False)
                return False
            
            # Check total time and final velocity consistency
            v_from_phases = self.a1 * self.t1 + self.a2 * self.t2
            if self._isclose(self.v_final, v_from_phases):
                self._log(f"Total motion consistency: v_final = {self.v_final} ≈ a1*t1 + a2*t2 = {v_from_phases}", True)
            else:
                self._log(f"Total motion FAILED: v_final = {self.v_final} ≠ {v_from_phases}", False)
                return False
            
            return True
            
        except Exception as e:
            self._log(f"Trace consistency test failed with error: {str(e)}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb inputs and check output trends"""
        try:
            # Save original values
            orig_v0, orig_a1 = self.v0, self.a1
            
            # Test 1: Increase a2, v0 should decrease (since v0 = v_final - a2*t2)
            perturb_a2 = self.a2 * 1.1  # +10%
            new_v0 = self.v_final - perturb_a2 * self.t2
            new_a1 = new_v0 / self.t1
            
            if new_v0 < orig_v0 and new_a1 < orig_a1:
                self._log(f"Monotonicity 1: Increasing a2 decreases v0 ({new_v0:.2f} < {orig_v0:.2f}) and a1 ({new_a1:.2f} < {orig_a1:.2f})", True)
            else:
                self._log(f"Monotonicity 1 FAILED: Expected decreasing trend", False)
                return False
            
            # Test 2: Increase t2, v0 should decrease (v0 = v_final - a2*t2)
            perturb_t2 = self.t2 * 1.1  # +10%
            new_v0 = self.v_final - self.a2 * perturb_t2
            new_a1 = new_v0 / self.t1
            
            if new_v0 < orig_v0 and new_a1 < orig_a1:
                self._log(f"Monotonicity 2: Increasing t2 decreases v0 ({new_v0:.2f} < {orig_v0:.2f}) and a1 ({new_a1:.2f} < {orig_a1:.2f})", True)
            else:
                self._log(f"Monotonicity 2 FAILED: Expected decreasing trend", False)
                return False
            
            # Test 3: Increase v_final, v0 should increase
            perturb_vf = self.v_final * 1.1  # +10%
            new_v0 = perturb_vf - self.a2 * self.t2
            new_a1 = new_v0 / self.t1
            
            if new_v0 > orig_v0 and new_a1 > orig_a1:
                self._log(f"Monotonicity 3: Increasing v_final increases v0 ({new_v0:.2f} > {orig_v0:.2f}) and a1 ({new_a1:.2f} > {orig_a1:.2f})", True)
            else:
                self._log(f"Monotonicity 3 FAILED: Expected increasing trend", False)
                return False
            
            # Restore original values
            self.v0, self.a1 = orig_v0, orig_a1
            
            return True
            
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {str(e)}", False)
            return False
    
    def run_all_tests(self):
        """Run all tests and return success status with logs"""
        self.logs = []
        
        try:
            # Run tests
            test1 = self.test_invariants()
            test2 = self.test_trace_consistency()
            test3 = self.test_monotonicity()
            
            success = test1 and test2 and test3
            
            # Summary
            if success:
                self._log("All physics checks passed!", True)
            else:
                self._log("Some physics checks failed", False)
            
            return success, self.logs
            
        except Exception as e:
            error_msg = f"Checker crashed: {str(e)}\n{traceback.format_exc()}"
            self._log(error_msg, False)
            return False, self.logs