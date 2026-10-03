import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    """
    Physics auditor for free-fall motion problem.
    Validates solution consistency with physics laws and stepwise logic.
    """
    
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution instance.
        """
        # Constants from problem statement
        self.g = 10.0  # m/s²
        self.h_total = 80.0  # m
        self.v0 = 0.0  # m/s (released from rest)
        
        # Safely extract results with fallbacks
        self.t = self._safe_get(solution_instance, 't', 0.0)
        self.v = self._safe_get(solution_instance, 'v', 0.0)
        self.h_3 = self._safe_get(solution_instance, 'h_3', 0.0)
        self.Deltah = self._safe_get(solution_instance, 'Deltah', 0.0)
        
        # Log storage
        self.logs = []
        
    def _safe_get(self, obj, key, default):
        """Safe attribute access with dictionary fallback."""
        value = getattr(obj, key, None)
        if value is None and hasattr(obj, 'results'):
            value = obj.results.get(key, default)
        return float(value) if value is not None else default
    
    def _log(self, message: str, success: bool = True):
        """Append test result to logs."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {message}")
        return success
    
    def test_invariants(self) -> bool:
        """
        Verify fundamental physics laws:
        1. Energy conservation
        2. Kinematic equations
        """
        tests_passed = True
        
        # 1. Energy conservation: (1/2)mv² = mgh_total
        # Cancel mass m from both sides
        expected_v2 = 2 * self.g * self.h_total
        computed_v2 = self.v ** 2
        
        if not math.isclose(computed_v2, expected_v2, rel_tol=0.02, abs_tol=0.02):
            tests_passed = self._log(
                f"Energy conservation violated: v² = {computed_v2:.2f}, expected 2gh = {expected_v2:.2f}",
                False
            )
        else:
            tests_passed = self._log(
                f"Energy conservation satisfied: v² = {computed_v2:.2f} ≈ 2gh = {expected_v2:.2f}",
                True
            )
        
        # 2. Kinematic equation: v = g*t
        expected_v_from_t = self.g * self.t
        
        if not math.isclose(self.v, expected_v_from_t, rel_tol=0.02, abs_tol=0.02):
            tests_passed = self._log(
                f"Kinematic equation v=gt violated: v={self.v:.2f}, g*t={expected_v_from_t:.2f}",
                False
            ) and tests_passed
        else:
            tests_passed = self._log(
                f"Kinematic equation v=gt satisfied: v={self.v:.2f} ≈ g*t={expected_v_from_t:.2f}",
                True
            ) and tests_passed
            
        # 3. Displacement after 3s: h_3 = 0.5*g*(3)²
        expected_h3 = 0.5 * self.g * 9.0  # 3² = 9
        
        if not math.isclose(self.h_3, expected_h3, rel_tol=0.02, abs_tol=0.02):
            tests_passed = self._log(
                f"Displacement formula violated: h_3={self.h_3:.2f}, expected 0.5*g*9={expected_h3:.2f}",
                False
            ) and tests_passed
        else:
            tests_passed = self._log(
                f"Displacement formula satisfied: h_3={self.h_3:.2f} ≈ 0.5*g*9={expected_h3:.2f}",
                True
            ) and tests_passed
            
        return tests_passed
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check consistency between intermediate results and final answers.
        """
        tests_passed = True
        
        # 1. Time from free-fall: t = √(2h_total/g)
        expected_t = math.sqrt(2 * self.h_total / self.g)
        
        if not math.isclose(self.t, expected_t, rel_tol=0.02, abs_tol=0.02):
            tests_passed = self._log(
                f"Time inconsistency: t={self.t:.2f}, expected √(2h/g)={expected_t:.2f}",
                False
            )
        else:
            tests_passed = self._log(
                f"Time consistent: t={self.t:.2f} ≈ √(2h/g)={expected_t:.2f}",
                True
            )
        
        # 2. Check that Δh = h_total - h_3
        expected_Deltah = self.h_total - self.h_3
        
        if not math.isclose(self.Deltah, expected_Deltah, rel_tol=0.02, abs_tol=0.02):
            tests_passed = self._log(
                f"Height difference inconsistent: Δh={self.Deltah:.2f}, expected h_total-h_3={expected_Deltah:.2f}",
                False
            ) and tests_passed
        else:
            tests_passed = self._log(
                f"Height difference consistent: Δh={self.Deltah:.2f} ≈ h_total-h_3={expected_Deltah:.2f}",
                True
            ) and tests_passed
            
        # 3. Check that h_3 corresponds to 3 seconds of fall
        # Using kinematic: h_3 should be 0.5*g*t1² where t1=3s
        t1 = 3.0
        h3_from_t1 = 0.5 * self.g * t1**2
        
        if not math.isclose(self.h_3, h3_from_t1, rel_tol=0.02, abs_tol=0.02):
            tests_passed = self._log(
                f"h_3 inconsistent with 3s fall: h_3={self.h_3:.2f}, expected 0.5*g*3²={h3_from_t1:.2f}",
                False
            ) and tests_passed
        else:
            tests_passed = self._log(
                f"h_3 consistent with 3s fall: h_3={self.h_3:.2f} ≈ 0.5*g*3²={h3_from_t1:.2f}",
                True
            ) and tests_passed
            
        return tests_passed
    
    def test_monotonicity(self) -> bool:
        """
        Perturb initial height and verify expected trends in outputs.
        """
        tests_passed = True
        
        # Original values
        h_original = self.h_total
        t_original = self.t
        v_original = self.v
        h3_original = self.h_3
        deltah_original = self.Deltah
        
        # Perturbation: increase height by 10%
        h_perturbed = h_original * 1.1  # 88 m
        
        # Expected new values based on formulas
        t_expected = math.sqrt(2 * h_perturbed / self.g)
        v_expected = self.g * t_expected
        h3_expected = 0.5 * self.g * 9.0  # Unchanged (3s is fixed)
        deltah_expected = h_perturbed - h3_expected
        
        # Check monotonic trends
        # 1. Time should increase with height
        if t_expected <= t_original:
            tests_passed = self._log(
                f"Monotonicity violated: t should increase with h, but t_expected={t_expected:.2f} ≤ t_original={t_original:.2f}",
                False
            )
        else:
            tests_passed = self._log(
                f"Monotonic trend OK: t increases with h (t_original={t_original:.2f} → t_expected={t_expected:.2f})",
                True
            )
            
        # 2. Velocity should increase with height
        if v_expected <= v_original:
            tests_passed = self._log(
                f"Monotonicity violated: v should increase with h, but v_expected={v_expected:.2f} ≤ v_original={v_original:.2f}",
                False
            ) and tests_passed
        else:
            tests_passed = self._log(
                f"Monotonic trend OK: v increases with h (v_original={v_original:.2f} → v_expected={v_expected:.2f})",
                True
            ) and tests_passed
            
        # 3. h_3 should remain constant (depends only on fixed 3s interval)
        if not math.isclose(h3_expected, h3_original, rel_tol=0.02, abs_tol=0.02):
            tests_passed = self._log(
                f"Monotonicity violated: h_3 should be constant, but changed from {h3_original:.2f} to {h3_expected:.2f}",
                False
            ) and tests_passed
        else:
            tests_passed = self._log(
                f"Monotonic trend OK: h_3 constant with h change (remains {h3_original:.2f})",
                True
            ) and tests_passed
            
        # 4. Δh should increase with height
        if deltah_expected <= deltah_original:
            tests_passed = self._log(
                f"Monotonicity violated: Δh should increase with h, but Δh_expected={deltah_expected:.2f} ≤ Δh_original={deltah_original:.2f}",
                False
            ) and tests_passed
        else:
            tests_passed = self._log(
                f"Monotonic trend OK: Δh increases with h (Δh_original={deltah_original:.2f} → Δh_expected={deltah_expected:.2f})",
                True
            ) and tests_passed
            
        # Restore original values
        self.h_total = h_original
        
        return tests_passed
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all physics validation tests.
        Returns (success_flag, logs).
        """
        self.logs = []
        self.logs.append("=" * 50)
        self.logs.append("Physics Audit Report")
        self.logs.append("=" * 50)
        
        # Run tests
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        # Summary
        all_passed = test1 and test2 and test3
        
        self.logs.append("=" * 50)
        self.logs.append(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
        self.logs.append("=" * 50)
        
        return all_passed, self.logs


# Example usage demonstration
if __name__ == "__main__":
    # Mock solution instance for testing
    class MockSolution:
        def __init__(self):
            self.results = {
                't': 4.0,
                'v': 40.0,
                'h_3': 45.0,
                'Deltah': 35.0
            }
    
    # Create checker and run tests
    checker = PhysicsChecker(MockSolution())
    success, logs = checker.run_all_tests()
    
    # Print results
    for log in logs:
        print(log)