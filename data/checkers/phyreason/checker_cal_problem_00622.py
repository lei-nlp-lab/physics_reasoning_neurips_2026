import math
import traceback
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        # Safe attribute access using getattr with fallback to results dict
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract all required quantities with safe defaults
        self.v_x = float(getattr(solution_instance, 'v_x', self.results.get('v_x', 0.0)))
        self.v_y = float(getattr(solution_instance, 'v_y', self.results.get('v_y', 0.0)))
        self.v_2 = float(getattr(solution_instance, 'v_2', self.results.get('v_2', 0.0)))
        self.x = float(getattr(solution_instance, 'x', self.results.get('x', 0.0)))
        self.y = float(getattr(solution_instance, 'y', self.results.get('y', 0.0)))
        self.s = float(getattr(solution_instance, 's', self.results.get('s', 0.0)))
        
        # Initialize logs
        self.logs = []
        
    def _log(self, message: str, success: bool = True):
        """Helper to log test results"""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {message}")
        return success
        
    def test_invariants(self) -> bool:
        """Test fundamental physics invariants from KB"""
        success = True
        
        # Test 1: Velocity vector composition (Theorem: Velocity vector composition)
        # v_2 should equal sqrt(v_x^2 + v_y^2)
        expected_v2 = math.sqrt(self.v_x**2 + self.v_y**2)
        if not math.isclose(self.v_2, expected_v2, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(
                f"Velocity magnitude mismatch: v_2={self.v_2}, sqrt(v_x^2+v_y^2)={expected_v2}",
                False
            )
        else:
            self._log(f"Velocity vector composition verified: v_2 ≈ √({self.v_x}^2 + {self.v_y}^2)")
        
        # Test 2: Displacement vector composition (Theorem: Displacement vector composition)
        # s should equal sqrt(x^2 + y^2)
        expected_s = math.sqrt(self.x**2 + self.y**2)
        if not math.isclose(self.s, expected_s, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(
                f"Displacement magnitude mismatch: s={self.s}, sqrt(x^2+y^2)={expected_s}",
                False
            )
        else:
            self._log(f"Displacement vector composition verified: s ≈ √({self.x}^2 + {self.y}^2)")
        
        # Test 3: Check consistency between uniform motion and constant velocity
        # For uniform linear motion in x, v_x should be constant
        # From problem context: at t=2s, x=8m, so v_x should be x/t = 4 m/s
        if not math.isclose(self.v_x, 4.0, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(
                f"Uniform motion check: v_x={self.v_x}, expected ~4 m/s from x=8m at t=2s",
                False
            )
        else:
            self._log("Uniform motion in x-direction verified (constant v_x)")
        
        # Test 4: Check uniformly accelerated motion consistency - CORRECTED
        # For y: v_y=6 m/s at t=2s, y=6m at t=2s
        # For constant acceleration starting from rest: y = 0.5*a*t^2, v_y = a*t
        # Therefore v_y = 2*y/t (not 2*y/4 as in the original checker)
        # With y=6 m and t=2 s: v_y = 2*6/2 = 6 m/s, which matches the given value
        expected_v_y_from_y = 2 * self.y / 2.0  # t=2s, v_y = 2y/t
        if not math.isclose(self.v_y, expected_v_y_from_y, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(
                f"Accelerated motion check: v_y={self.v_y}, expected from y-displacement ~{expected_v_y_from_y}",
                False
            )
        else:
            self._log("Uniformly accelerated motion in y-direction verified")
        
        return success
        
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answers"""
        success = True
        
        # Check that the computed results match the expected values from STEPS ANALYSIS
        expected_values = {
            'v_x': 4.0,
            'v_y': 6.0,
            'v_2': 2*math.sqrt(13),  # ≈ 7.211
            'x': 8.0,
            'y': 6.0,
            's': 10.0
        }
        
        for key, expected in expected_values.items():
            actual = getattr(self, key, None)
            if actual is not None:
                if not math.isclose(actual, expected, rel_tol=1e-2, abs_tol=1e-2):
                    success = self._log(
                        f"Trace mismatch for {key}: {actual} ≠ expected {expected}",
                        False
                    )
                else:
                    self._log(f"Trace consistency for {key}: {actual} ≈ {expected}")
        
        # Additional check: v_2 should be 2√13
        expected_v2_symbolic = 2 * math.sqrt(13)
        if not math.isclose(self.v_2, expected_v2_symbolic, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(
                f"Symbolic check: v_2={self.v_2} ≠ 2√13≈{expected_v2_symbolic}",
                False
            )
        
        return success
        
    def test_monotonicity(self) -> bool:
        """Test that perturbations in inputs produce expected trends"""
        success = True
        
        # Save original values
        orig_v_x = self.v_x
        orig_v_y = self.v_y
        orig_x = self.x
        orig_y = self.y
        
        # Test 1: Increase v_x by 1%, v_2 should increase
        perturbed_v_x = orig_v_x * 1.01
        new_v2 = math.sqrt(perturbed_v_x**2 + orig_v_y**2)
        if not new_v2 > self.v_2:
            success = self._log(
                f"Monotonicity failed: v_2 didn't increase when v_x increased from {orig_v_x} to {perturbed_v_x}",
                False
            )
        else:
            self._log(f"Monotonicity: v_2 increases with v_x (∂v_2/∂v_x > 0)")
        
        # Test 2: Increase v_y by 1%, v_2 should increase
        perturbed_v_y = orig_v_y * 1.01
        new_v2 = math.sqrt(orig_v_x**2 + perturbed_v_y**2)
        if not new_v2 > self.v_2:
            success = self._log(
                f"Monotonicity failed: v_2 didn't increase when v_y increased from {orig_v_y} to {perturbed_v_y}",
                False
            )
        else:
            self._log(f"Monotonicity: v_2 increases with v_y (∂v_2/∂v_y > 0)")
        
        # Test 3: Increase x by 1%, s should increase
        perturbed_x = orig_x * 1.01
        new_s = math.sqrt(perturbed_x**2 + orig_y**2)
        if not new_s > self.s:
            success = self._log(
                f"Monotonicity failed: s didn't increase when x increased from {orig_x} to {perturbed_x}",
                False
            )
        else:
            self._log(f"Monotonicity: s increases with x (∂s/∂x > 0)")
        
        # Test 4: Increase y by 1%, s should increase
        perturbed_y = orig_y * 1.01
        new_s = math.sqrt(orig_x**2 + perturbed_y**2)
        if not new_s > self.s:
            success = self._log(
                f"Monotonicity failed: s didn't increase when y increased from {orig_y} to {perturbed_y}",
                False
            )
        else:
            self._log(f"Monotonicity: s increases with y (∂s/∂y > 0)")
        
        # Restore original values
        self.v_x = orig_v_x
        self.v_y = orig_v_y
        self.x = orig_x
        self.y = orig_y
        
        return success
        
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success flag with logs"""
        self.logs = []  # Reset logs
        
        # Run tests in sequence
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        overall_success = True
        for test_name, test_func in tests:
            self._log(f"=== Running {test_name} Test ===", True)
            if not test_func():
                overall_success = False
            self._log("", True)  # Empty line for readability
        
        # Summary
        self._log(f"=== Overall Result: {'PASS' if overall_success else 'FAIL'} ===", overall_success)
        
        return overall_success, self.logs

# Optional: Example usage guard
if __name__ == "__main__":
    # Mock solution instance for testing
    class MockSolution:
        def __init__(self):
            self.results = {
                'v_x': 4.0,
                'v_y': 6.0,
                'v_2': 2*math.sqrt(13),  # ~7.211
                'x': 8.0,
                'y': 6.0,
                's': 10.0
            }
    
    mock_solution = MockSolution()
    checker = PhysicsChecker(mock_solution)
    success, logs = checker.run_all_tests()
    
    for log in logs:
        if log:  # Skip empty lines
            print(log)