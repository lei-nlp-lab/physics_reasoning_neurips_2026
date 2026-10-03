import sympy as sp
import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract all needed values safely
        self.a1 = self._safe_get('a_1')
        self.v2 = self._safe_get('v_2')
        self.a2 = self._safe_get('a_2')
        self.v = self._safe_get('v')
        self.s = self._safe_get('s')
        
        # Problem constants from KB
        self.t1 = 20.0  # time when v=20 m/s in stage 1
        self.t2 = 50.0  # time at bottom of slope
        self.t3 = 25.0  # time to stop on horizontal surface
        self.t4 = 6.0   # time after reaching bottom for speed calculation
        self.v1_measured = 20.0  # speed at t1
        
        # Store logs
        self.logs = []
        
    def _safe_get(self, key):
        """Safe extraction from results with fallback to 0.0"""
        return self.results.get(key, 0.0)
    
    def _is_close(self, a, b, tol=0.02):
        """Fuzzy comparison with 2% tolerance"""
        return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
    
    def _log(self, msg, success=None):
        """Log message with optional success flag"""
        if success is not None:
            status = "PASS" if success else "FAIL"
            msg = f"[{status}] {msg}"
        self.logs.append(msg)
        return success
    
    def test_invariants(self) -> bool:
        """Verify kinematic formulas and conservation principles"""
        try:
            # Test 1: Stage 1 acceleration consistency
            # From v = v0 + a*t with v0=0, v1=20 at t1=20
            expected_a1 = self.v1_measured / self.t1
            if not self._is_close(self.a1, expected_a1):
                return self._log(f"Stage 1 acceleration mismatch: got {self.a1}, expected {expected_a1}", False)
            self._log(f"Stage 1 acceleration correct: {self.a1} ≈ {expected_a1}", True)
            
            # Test 2: Velocity at bottom from stage 1
            # v2 = a1 * t2 (from rest)
            expected_v2 = self.a1 * self.t2
            if not self._is_close(self.v2, expected_v2):
                return self._log(f"Velocity at bottom mismatch: got {self.v2}, expected {expected_v2}", False)
            self._log(f"Velocity at bottom correct: {self.v2} ≈ {expected_v2}", True)
            
            # Test 3: Stage 2 deceleration from v2 to 0 in t3=25s
            # a2 = (0 - v2) / t3
            expected_a2 = -self.v2 / self.t3
            if not self._is_close(self.a2, expected_a2):
                return self._log(f"Stage 2 acceleration mismatch: got {self.a2}, expected {expected_a2}", False)
            self._log(f"Stage 2 acceleration correct: {self.a2} ≈ {expected_a2}", True)
            
            # Test 4: Speed after 6s on horizontal
            # v = v2 + a2 * t4
            expected_v = self.v2 + self.a2 * self.t4
            if not self._is_close(self.v, expected_v):
                return self._log(f"Speed after 6s mismatch: got {self.v}, expected {expected_v}", False)
            self._log(f"Speed after 6s correct: {self.v} ≈ {expected_v}", True)
            
            # Test 5: Distance on horizontal plane using displacement formula
            # s = v2*t3 + 0.5*a2*t3^2
            expected_s = self.v2 * self.t3 + 0.5 * self.a2 * (self.t3 ** 2)
            if not self._is_close(self.s, expected_s):
                return self._log(f"Horizontal distance mismatch: got {self.s}, expected {expected_s}", False)
            self._log(f"Horizontal distance correct: {self.s} ≈ {expected_s}", True)
            
            # Test 6: Symbolic verification using sympy
            t = sp.symbols('t')
            a1_sym = sp.symbols('a1')
            v2_sym = sp.symbols('v2')
            a2_sym = sp.symbols('a2')
            
            # Stage 1: v(t) = a1*t
            v1_expr = a1_sym * t
            # At t1=20, v=20
            eq1 = sp.Eq(v1_expr.subs(t, self.t1), self.v1_measured)
            sol1 = sp.solve(eq1, a1_sym)[0]
            if not self._is_close(float(sol1), self.a1):
                return self._log(f"Symbolic a1 mismatch: {float(sol1)} vs {self.a1}", False)
            
            # Stage 2: v(t) = v2 + a2*t (with t measured from start of stage 2)
            v2_expr = v2_sym + a2_sym * t
            # At t=t3=25, v=0
            eq2 = sp.Eq(v2_expr.subs(t, self.t3), 0)
            sol2 = sp.solve(eq2, a2_sym)[0]
            if not self._is_close(float(sol2.subs(v2_sym, self.v2)), self.a2):
                return self._log("Symbolic a2 equation incorrect", False)
                
            return True
            
        except Exception as e:
            self._log(f"Invariant test failed with error: {str(e)}", False)
            return False
    
    def test_trace_consistency(self) -> bool:
        """Check intermediate results are consistent with final answers"""
        try:
            # The problem has three target answers: a2, v, s
            # Verify our extracted values match these targets
            
            # Target 1: a2 = -2 m/s²
            if not self._is_close(self.a2, -2.0):
                return self._log(f"a2 doesn't match target: got {self.a2}, expected -2", False)
            self._log("a2 matches target (-2 m/s²)", True)
            
            # Target 2: v = 38 m/s
            if not self._is_close(self.v, 38.0):
                return self._log(f"v doesn't match target: got {self.v}, expected 38", False)
            self._log("v matches target (38 m/s)", True)
            
            # Target 3: s = 625 m
            if not self._is_close(self.s, 625.0):
                return self._log(f"s doesn't match target: got {self.s}, expected 625", False)
            self._log("s matches target (625 m)", True)
            
            # Check step-by-step consistency
            # Step 1: a1 = 1 m/s²
            expected_a1 = 1.0
            if not self._is_close(self.a1, expected_a1):
                return self._log(f"Step 1 inconsistency: a1={self.a1}, expected {expected_a1}", False)
            
            # Step 2: v2 = 50 m/s
            expected_v2 = 50.0
            if not self._is_close(self.v2, expected_v2):
                return self._log(f"Step 2 inconsistency: v2={self.v2}, expected {expected_v2}", False)
            
            # Step 4: a2 = -2 m/s² (already checked)
            # Step 5: v = 38 m/s (already checked)
            # Step 6: s = 625 m (already checked)
            
            self._log("All steps consistent with final answers", True)
            return True
            
        except Exception as e:
            self._log(f"Trace consistency test failed: {str(e)}", False)
            return False
    
    def test_monotonicity(self) -> bool:
        """Perturb input and verify expected trends"""
        try:
            # Original values for restoration
            original_a1 = self.a1
            original_v2 = self.v2
            original_a2 = self.a2
            original_v = self.v
            original_s = self.s
            
            # Perturb the measured speed at t1=20s from 20 to 21 m/s
            perturbed_v1 = 21.0
            
            # Recalculate using kinematic formulas
            # New a1 = v1_perturbed / t1
            new_a1 = perturbed_v1 / self.t1
            
            # New v2 = new_a1 * t2
            new_v2 = new_a1 * self.t2
            
            # New a2 = (0 - new_v2) / t3
            new_a2 = -new_v2 / self.t3
            
            # New v after 6s = new_v2 + new_a2 * t4
            new_v = new_v2 + new_a2 * self.t4
            
            # New horizontal distance s = new_v2 * t3 + 0.5 * new_a2 * t3^2
            new_s = new_v2 * self.t3 + 0.5 * new_a2 * (self.t3 ** 2)
            
            # Check monotonic trends:
            # 1. a1 should increase (from 1.0 to 1.05)
            if not new_a1 > original_a1:
                return self._log(f"Monotonicity fail: a1 didn't increase ({original_a1} -> {new_a1})", False)
            
            # 2. v2 should increase (from 50 to 52.5)
            if not new_v2 > original_v2:
                return self._log(f"Monotonicity fail: v2 didn't increase ({original_v2} -> {new_v2})", False)
            
            # 3. a2 should become more negative (from -2 to -2.1)
            if not new_a2 < original_a2:
                return self._log(f"Monotonicity fail: a2 didn't decrease ({original_a2} -> {new_a2})", False)
            
            # 4. v after 6s should increase (from 38 to ~39.9)
            # Note: v = 50 + (-2)*6 = 38 vs 52.5 + (-2.1)*6 = 39.9
            if not new_v > original_v:
                return self._log(f"Monotonicity fail: v didn't increase ({original_v} -> {new_v})", False)
            
            # 5. Horizontal distance should increase (from 625 to ~656.25)
            if not new_s > original_s:
                return self._log(f"Monotonicity fail: s didn't increase ({original_s} -> {new_s})", False)
            
            self._log("All monotonic trends hold as expected", True)
            
            # Restore original values (in case checker is reused)
            self.a1 = original_a1
            self.v2 = original_v2
            self.a2 = original_a2
            self.v = original_v
            self.s = original_s
            
            return True
            
        except Exception as e:
            self._log(f"Monotonicity test failed: {str(e)}", False)
            return False
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status and logs"""
        self.logs = []
        self.logs.append("=== PhysicsChecker Started ===")
        
        try:
            # Run individual tests
            test1 = self.test_invariants()
            test2 = self.test_trace_consistency()
            test3 = self.test_monotonicity()
            
            success = all([test1, test2, test3])
            
            self.logs.append(f"\n=== Summary ===")
            self.logs.append(f"Invariant tests: {'PASS' if test1 else 'FAIL'}")
            self.logs.append(f"Trace consistency: {'PASS' if test2 else 'FAIL'}")
            self.logs.append(f"Monotonicity: {'PASS' if test3 else 'FAIL'}")
            self.logs.append(f"Overall: {'PASS' if success else 'FAIL'}")
            
            return success, self.logs
            
        except Exception as e:
            error_msg = f"PhysicsChecker crashed: {str(e)}\n{traceback.format_exc()}"
            self.logs.append(error_msg)
            return False, self.logs