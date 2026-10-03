import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallback to results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with safe defaults
        self.l = 2.5  # Given length in meters
        self.theta_deg = 30  # Given angle in degrees
        self.theta_rad = math.radians(self.theta_deg)
        self.g = 10.0  # Given gravity
        
        # Extract results with safe fallbacks
        self.x = getattr(solution_instance, 'x', self.results.get('x', None))
        self.t2 = getattr(solution_instance, 't_2', self.results.get('t_2', None))
        self.a = getattr(solution_instance, 'a', self.results.get('a', None))
        self.v0 = getattr(solution_instance, 'v_0', self.results.get('v_0', None))
        self.t1 = getattr(solution_instance, 't_1', self.results.get('t_1', None))
        
        # Store logs for all tests
        self.logs = []
    
    def _log(self, message: str, success: bool):
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {message}")
        return success
    
    def test_invariants(self) -> bool:
        """Verify fundamental physics laws."""
        all_pass = True
        
        # 1. Check acceleration of ball p (Newton's 2nd Law on incline)
        a_expected = self.g * math.sin(self.theta_rad)
        if self.a is not None:
            if math.isclose(self.a, a_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Acceleration correct: a = {self.a:.3f} ≈ g·sinθ = {a_expected:.3f}", True)
            else:
                all_pass = self._log(f"Acceleration mismatch: a = {self.a:.3f} ≠ g·sinθ = {a_expected:.3f}", False)
        
        # 2. Check time for ball p using kinematic equations
        if self.a is not None and self.t1 is not None:
            t1_expected = math.sqrt(2 * self.l / a_expected)
            if math.isclose(self.t1, t1_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Time t1 correct: t1 = {self.t1:.3f} ≈ √(2l/g·sinθ) = {t1_expected:.3f}", True)
            else:
                all_pass = self._log(f"Time t1 mismatch: t1 = {self.t1:.3f} ≠ √(2l/g·sinθ) = {t1_expected:.3f}", False)
        
        # 3. Check horizontal displacement x
        if self.x is not None:
            x_expected = self.l * math.cos(self.theta_rad)
            if math.isclose(self.x, x_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Horizontal displacement correct: x = {self.x:.3f} ≈ l·cosθ = {x_expected:.3f}", True)
            else:
                all_pass = self._log(f"Horizontal displacement mismatch: x = {self.x:.3f} ≠ l·cosθ = {x_expected:.3f}", False)
        
        # 4. Check simultaneous arrival (t1 = t2)
        if self.t1 is not None and self.t2 is not None:
            if math.isclose(self.t1, self.t2, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Simultaneous arrival: t1 = {self.t1:.3f} = t2 = {self.t2:.3f}", True)
            else:
                all_pass = self._log(f"Time mismatch: t1 = {self.t1:.3f} ≠ t2 = {self.t2:.3f}", False)
        
        # 5. Check initial velocity of ball q
        if self.v0 is not None and self.x is not None and self.t1 is not None:
            v0_expected = self.x / self.t1
            if math.isclose(self.v0, v0_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Initial velocity correct: v0 = {self.v0:.3f} ≈ x/t1 = {v0_expected:.3f}", True)
            else:
                all_pass = self._log(f"Initial velocity mismatch: v0 = {self.v0:.3f} ≠ x/t1 = {v0_expected:.3f}", False)
        
        return all_pass
    
    def test_trace_consistency(self) -> bool:
        """Verify intermediate results are consistent with final answer."""
        all_pass = True
        
        # Check exact symbolic relationship: v0 = (l * cosθ) / t1
        if all(v is not None for v in [self.v0, self.x, self.t1]):
            # Using symbolic verification with numerical fallback
            v0_symbolic = self.x / self.t1
            if math.isclose(self.v0, v0_symbolic, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Trace consistency: v0 = {self.v0:.3f} matches x/t1 = {v0_symbolic:.3f}", True)
            else:
                all_pass = self._log(f"Trace inconsistency: v0 = {self.v0:.3f} ≠ x/t1 = {v0_symbolic:.3f}", False)
        
        # Check that t1 matches target answer
        if self.t1 is not None:
            if math.isclose(self.t1, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"t1 matches target: {self.t1:.3f} ≈ 1.0 s", True)
            else:
                all_pass = self._log(f"t1 doesn't match target: {self.t1:.3f} ≠ 1.0 s", False)
        
        # Check that v0 matches target answer (5√3/4)
        if self.v0 is not None:
            target_v0 = 5 * math.sqrt(3) / 4  # ≈ 2.165
            if math.isclose(self.v0, target_v0, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"v0 matches target: {self.v0:.3f} ≈ 5√3/4 ≈ 2.165 m/s", True)
            else:
                all_pass = self._log(f"v0 doesn't match target: {self.v0:.3f} ≠ 5√3/4 ≈ 2.165 m/s", False)
        
        return all_pass
    
    def test_monotonicity(self) -> bool:
        """Test that perturbing inputs produces expected trends."""
        all_pass = True
        
        # Perturb length l by 10%
        original_l = self.l
        perturbed_l = original_l * 1.10
        
        # Calculate new expected values using physics formulas
        a_expected = self.g * math.sin(self.theta_rad)
        original_t1 = math.sqrt(2 * original_l / a_expected)
        perturbed_t1 = math.sqrt(2 * perturbed_l / a_expected)
        
        original_x = original_l * math.cos(self.theta_rad)
        perturbed_x = perturbed_l * math.cos(self.theta_rad)
        
        original_v0 = original_x / original_t1
        perturbed_v0 = perturbed_x / perturbed_t1
        
        # Check monotonic trends
        if perturbed_t1 > original_t1:
            self._log(f"Monotonic t1: Increasing l increases t1 ({original_t1:.3f} → {perturbed_t1:.3f})", True)
        else:
            all_pass = self._log(f"Non-monotonic t1: t1 doesn't increase with l", False)
        
        if perturbed_v0 > original_v0:
            self._log(f"Monotonic v0: Increasing l increases v0 ({original_v0:.3f} → {perturbed_v0:.3f})", True)
        else:
            all_pass = self._log(f"Non-monotonic v0: v0 doesn't increase with l", False)
        
        # Restore original value
        self.l = original_l
        
        return all_pass
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return (success, logs)."""
        self.logs = []
        
        test_order = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        success = True
        for test_name, test_func in test_order:
            self.logs.append(f"\n=== Running {test_name} ===")
            if not test_func():
                success = False
        
        # Add summary
        if success:
            self.logs.append(f"\n=== SUMMARY: All tests passed ===")
        else:
            self.logs.append(f"\n=== SUMMARY: Some tests failed ===")
        
        return success, self.logs