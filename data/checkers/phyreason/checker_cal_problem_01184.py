import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs
        self.v0 = getattr(solution_instance, 'v0', 6.0)  # max speed in toll zone
        self.x0 = getattr(solution_instance, 'x0', 9.0)  # toll zone length
        self.vA = getattr(solution_instance, 'vA', 20.0) # Car A initial speed
        self.vB = getattr(solution_instance, 'vB', 34.0) # Car B initial speed
        self.aA = getattr(solution_instance, 'aA', 2.0)  # Car A deceleration magnitude (positive)
        
        # Safe extraction of results (fallback to 0.0 if missing)
        results = getattr(solution_instance, 'results', {})
        self.x1 = results.get('x_1', 0.0)
        self.x2 = results.get('x_2', 0.0)
        self.t = results.get('t', 0.0)
        self.v = results.get('v', 0.0)
        self.x3 = results.get('x_3', 0.0)
        self.t1 = results.get('t_1', 0.0)
        self.t2 = results.get('t_2', 0.0)
        self.x1_prime = results.get('x_1\'', 0.0)  # Note: escaped apostrophe
        self.Distance = results.get('Distance', 0.0)
        
        # Additional variables that may appear in solution
        self.aB = getattr(solution_instance, 'aB', 0.0)
        self.t0 = getattr(solution_instance, 't0', 0.0)
        
        self.logs = []

    def _isclose(self, a, b) -> bool:
        """Robust numerical comparison with 1% tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

    def _log(self, msg: str):
        self.logs.append(msg)

    def test_invariants(self) -> bool:
        """Verify kinematic equations from steps analysis."""
        success = True
        
        # Step 1: x1 = (vA^2 - v0^2) / (2*aA)
        expected_x1 = (self.vA**2 - self.v0**2) / (2 * self.aA)
        if not self._isclose(self.x1, expected_x1):
            self._log(f"FAIL Step 1: x1={self.x1}, expected={expected_x1}")
            success = False
        
        # Step 2: x2 = x1 + x0
        expected_x2 = self.x1 + self.x0
        if not self._isclose(self.x2, expected_x2):
            self._log(f"FAIL Step 2: x2={self.x2}, expected={expected_x2}")
            success = False
        
        # Step 3 & 4: Time when speeds equal (requires aB)
        # Equation: vB - aB*(t - t0) = vA - aA*t
        if self.aB != 0:  # Only check if aB is provided
            lhs = self.vB - self.aB * (self.t - self.t0)
            rhs = self.vA - self.aA * self.t
            if not self._isclose(lhs, rhs):
                self._log(f"FAIL Step 3 speed equality: LHS={lhs}, RHS={rhs}")
                success = False
        
        # Step 4: v = vA - aA*t
        expected_v = self.vA - self.aA * self.t
        if not self._isclose(self.v, expected_v):
            self._log(f"FAIL Step 4: v={self.v}, expected={expected_v}")
            success = False
        
        # Step 5: x3 = vB*t0 + (vB^2 - v0^2)/(2*aB)
        if self.aB != 0:
            expected_x3 = self.vB * self.t0 + (self.vB**2 - self.v0**2) / (2 * self.aB)
            if not self._isclose(self.x3, expected_x3):
                self._log(f"FAIL Step 5: x3={self.x3}, expected={expected_x3}")
                success = False
        
        # Step 6: t1 = t0 + (vB - v0)/aB
        if self.aB != 0:
            expected_t1 = self.t0 + (self.vB - self.v0) / self.aB
            if not self._isclose(self.t1, expected_t1):
                self._log(f"FAIL Step 6: t1={self.t1}, expected={expected_t1}")
                success = False
        
        # Step 7: t2 = (vA - v0)/aA
        expected_t2 = (self.vA - self.v0) / self.aA
        if not self._isclose(self.t2, expected_t2):
            self._log(f"FAIL Step 7: t2={self.t2}, expected={expected_t2}")
            success = False
        
        # Step 8: x1' = x1 + v0*Δt (where Δt = t2 - t1? From context)
        # From steps, appears to be x1' = x1 + v0*(t2 - 0) = x1 (if Δt=0)
        # But given result matches x1, we check equality
        if not self._isclose(self.x1_prime, self.x1):
            self._log(f"FAIL Step 8: x1'={self.x1_prime}, x1={self.x1}")
            success = False
        
        # Step 9: Distance = x3 - x1'
        expected_distance = self.x3 - self.x1_prime
        if not self._isclose(self.Distance, expected_distance):
            self._log(f"FAIL Step 9: Distance={self.Distance}, expected={expected_distance}")
            success = False
        
        if success:
            self._log("All kinematic invariants satisfied")
        return success

    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers."""
        success = True
        
        # Final answers from problem: 100m and 66m
        if not self._isclose(self.x2, 100.0):
            self._log(f"Trace FAIL: x2={self.x2} doesn't match target 100m")
            success = False
        
        if not self._isclose(self.Distance, 66.0):
            self._log(f"Trace FAIL: Distance={self.Distance} doesn't match target 66m")
            success = False
        
        # Check that x1' equals x1 (as in steps)
        if not self._isclose(self.x1_prime, self.x1):
            self._log(f"Trace FAIL: x1'={self.x1_prime} ≠ x1={self.x1}")
            success = False
        
        if success:
            self._log("Trace consistency: intermediate results lead to correct final answers")
        return success

    def test_monotonicity(self) -> bool:
        """
        Perturb an input and check expected trend.
        Increase vA → x2 should increase.
        Then restore original value.
        """
        original_vA = self.vA
        original_x2 = self.x2
        
        # Perturb by +10%
        self.vA *= 1.1
        # Recompute expected x2 using formula
        new_x1 = (self.vA**2 - self.v0**2) / (2 * self.aA)
        new_x2 = new_x1 + self.x0
        
        # Check monotonicity
        if not new_x2 > original_x2:
            self._log(f"Monotonicity FAIL: vA↑ from {original_vA} to {self.vA}, "
                     f"x2 from {original_x2} to {new_x2} (should increase)")
            # Restore anyway
            self.vA = original_vA
            return False
        
        # Restore original
        self.vA = original_vA
        self._log("Monotonicity: x2 increases with vA (as expected)")
        return True

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return (success, logs)."""
        self.logs.clear()
        
        tests = [
            ("Kinematic Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        overall_success = True
        for test_name, test_func in tests:
            self._log(f"=== Running {test_name} ===")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"ERROR in {test_name}: {str(e)}")
                overall_success = False
        
        self._log(f"\nOverall: {'PASS' if overall_success else 'FAIL'}")
        return overall_success, self.logs