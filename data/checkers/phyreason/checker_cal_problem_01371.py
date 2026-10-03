import math
import sympy as sp
from typing import Tuple, List, Any, Optional

class PhysicsChecker:
    """
    Robust physics auditor for horizontal projectile motion problem.
    Validates solution for: ball thrown horizontally from h=5m, v0=10m/s, g=10m/s².
    """
    
    def __init__(self, solution_instance: Any):
        """
        Safely extract all inputs and results without direct attribute access.
        """
        # Safe extraction of solution attributes
        self.results = getattr(solution_instance, 'results', {})
        
        # Input parameters (with defaults if missing)
        self.h = getattr(solution_instance, 'h', 5.0)
        self.v0 = getattr(solution_instance, 'v0', 10.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Results with safe fallbacks
        self.t = self.results.get('t', 0.0)
        self.x = self.results.get('x', 0.0)
        self.v_y = self.results.get('v_y', 0.0)
        self.v = self.results.get('v', 0.0)
        self.alpha = self.results.get('alpha', 0.0)  # in degrees
        
        # Constants for tolerances
        self.REL_TOL = 1e-2
        self.ABS_TOL = 1e-2
        
        # Logging storage
        self.logs = []
        
    def _log(self, message: str) -> None:
        """Helper for logging."""
        self.logs.append(message)
    
    def test_invariants(self) -> bool:
        """
        Verify fundamental physical invariants:
        1. Energy conservation (KE_i + PE_i = KE_f)
        2. Time of flight from vertical motion
        3. Pythagorean relation for final velocity
        """
        passed = True
        
        # 1. Energy conservation: ½v0² + g*h = ½v²
        E_initial = 0.5 * self.v0**2 + self.g * self.h
        E_final = 0.5 * self.v**2
        
        if not math.isclose(E_initial, E_final, rel_tol=self.REL_TOL, abs_tol=self.ABS_TOL):
            self._log(f"Energy conservation failed: E_initial={E_initial:.2f}, E_final={E_final:.2f}")
            passed = False
        else:
            self._log("✓ Energy conservation satisfied")
        
        # 2. Time of flight from vertical free fall: t = √(2h/g)
        t_expected = math.sqrt(2 * self.h / self.g)
        if not math.isclose(self.t, t_expected, rel_tol=self.REL_TOL, abs_tol=self.ABS_TOL):
            self._log(f"Time of flight mismatch: expected {t_expected:.2f}, got {self.t:.2f}")
            passed = False
        else:
            self._log(f"✓ Time of flight correct: t = √(2h/g) = {t_expected:.2f}s")
        
        # 3. Pythagorean theorem: v² = v0² + v_y²
        v_squared = self.v0**2 + self.v_y**2
        if not math.isclose(self.v**2, v_squared, rel_tol=self.REL_TOL, abs_tol=self.ABS_TOL):
            self._log(f"Velocity components don't satisfy Pythagorean theorem: v²={self.v**2:.2f}, v0²+v_y²={v_squared:.2f}")
            passed = False
        else:
            self._log("✓ Velocity components satisfy v² = v0² + v_y²")
        
        # 4. Vertical velocity from free fall: v_y = g*t
        v_y_expected = self.g * self.t
        if not math.isclose(self.v_y, v_y_expected, rel_tol=self.REL_TOL, abs_tol=self.ABS_TOL):
            self._log(f"Vertical velocity mismatch: expected g*t = {v_y_expected:.2f}, got {self.v_y:.2f}")
            passed = False
        else:
            self._log(f"✓ Vertical velocity correct: v_y = g*t = {v_y_expected:.2f} m/s")
        
        return passed
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answer and inputs.
        Based on the steps analysis.
        """
        passed = True
        
        # Step 1: Check t = sqrt(2h/g) (already done, but verify with calculated t)
        t_calc = math.sqrt(2 * self.h / self.g)
        if not math.isclose(self.t, t_calc, rel_tol=self.REL_TOL, abs_tol=self.ABS_TOL):
            self._log(f"Trace inconsistency in step 1: t should be √(2h/g)={t_calc:.2f}, got {self.t:.2f}")
            passed = False
        
        # Step 2: Check x = v0 * t
        x_expected = self.v0 * self.t
        if not math.isclose(self.x, x_expected, rel_tol=self.REL_TOL, abs_tol=self.ABS_TOL):
            self._log(f"Trace inconsistency in step 2: x should be v0*t={x_expected:.2f}, got {self.x:.2f}")
            passed = False
        
        # Step 3: Check v_y = g*t and v = √(v0² + v_y²)
        v_y_expected = self.g * self.t
        v_expected = math.sqrt(self.v0**2 + v_y_expected**2)
        
        if not math.isclose(self.v_y, v_y_expected, rel_tol=self.REL_TOL, abs_tol=self.ABS_TOL):
            self._log(f"Trace inconsistency in step 3: v_y should be g*t={v_y_expected:.2f}, got {self.v_y:.2f}")
            passed = False
        
        if not math.isclose(self.v, v_expected, rel_tol=self.REL_TOL, abs_tol=self.ABS_TOL):
            self._log(f"Trace inconsistency in step 3: v should be √(v0²+v_y²)={v_expected:.2f}, got {self.v:.2f}")
            passed = False
        
        # Step 4: Check α = arctan(v_y / v0) in degrees
        alpha_expected_rad = math.atan2(self.v_y, self.v0)
        alpha_expected_deg = math.degrees(alpha_expected_rad)
        
        if not math.isclose(self.alpha, alpha_expected_deg, rel_tol=self.REL_TOL, abs_tol=1.0):  # 1 degree tolerance
            self._log(f"Trace inconsistency in step 4: α should be arctan(v_y/v0)={alpha_expected_deg:.1f}°, got {self.alpha:.1f}°")
            passed = False
        
        if passed:
            self._log("✓ All trace steps are internally consistent")
        
        return passed
    
    def test_monotonicity(self) -> bool:
        """
        Perturb input parameters and verify expected trends.
        Restores original values after testing.
        """
        passed = True
        
        # Store original values
        original_h = self.h
        original_v0 = self.v0
        original_g = self.g
        
        # Test 1: Increase height h -> increase flight time t and horizontal distance x
        self.h = original_h * 1.1  # +10%
        t1 = math.sqrt(2 * self.h / self.g)
        x1 = self.v0 * t1
        
        if t1 <= self.t or x1 <= self.x:
            self._log(f"Monotonicity violation: increasing h should increase t and x, but t: {self.t:.2f}->{t1:.2f}, x: {self.x:.2f}->{x1:.2f}")
            passed = False
        
        # Restore h
        self.h = original_h
        
        # Test 2: Increase initial velocity v0 -> increase x and v
        self.v0 = original_v0 * 1.1  # +10%
        x2 = self.v0 * self.t
        v_y2 = self.g * self.t
        v2 = math.sqrt(self.v0**2 + v_y2**2)
        
        if x2 <= self.x or v2 <= self.v:
            self._log(f"Monotonicity violation: increasing v0 should increase x and v, but x: {self.x:.2f}->{x2:.2f}, v: {self.v:.2f}->{v2:.2f}")
            passed = False
        
        # Restore v0
        self.v0 = original_v0
        
        # Test 3: Increase g -> decrease t and x, increase v_y and v
        self.g = original_g * 1.1  # +10%
        t3 = math.sqrt(2 * self.h / self.g)
        x3 = self.v0 * t3
        v_y3 = self.g * t3
        v3 = math.sqrt(self.v0**2 + v_y3**2)
        
        if t3 >= self.t or x3 >= self.x or v_y3 <= self.v_y or v3 <= self.v:
            self._log(f"Monotonicity violation: increasing g should decrease t and x, increase v_y and v")
            self._log(f"  t: {self.t:.2f}->{t3:.2f}, x: {self.x:.2f}->{x3:.2f}, v_y: {self.v_y:.2f}->{v_y3:.2f}, v: {self.v:.2f}->{v3:.2f}")
            passed = False
        
        # Restore all original values
        self.h = original_h
        self.v0 = original_v0
        self.g = original_g
        
        if passed:
            self._log("✓ Monotonicity tests passed for perturbations of h, v0, g")
        
        return passed
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all validation tests and return success status with logs.
        """
        self.logs = []
        self._log("=== Physics Checker Started ===")
        self._log(f"Inputs: h={self.h}m, v0={self.v0}m/s, g={self.g}m/s²")
        self._log(f"Results: t={self.t}s, x={self.x}m, v_y={self.v_y}m/s, v={self.v}m/s, α={self.alpha}°")
        
        tests = [
            ("Physical Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        all_passed = True
        for test_name, test_func in tests:
            self._log(f"\n--- {test_name} ---")
            try:
                if not test_func():
                    all_passed = False
            except Exception as e:
                self._log(f"Test '{test_name}' crashed: {e}")
                all_passed = False
        
        self._log(f"\n=== Summary: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'} ===")
        return all_passed, self.logs

# Optional: Add a simple test to verify the checker works
if __name__ == "__main__":
    # Mock solution class for testing
    class MockSolution:
        def __init__(self):
            self.h = 5.0
            self.v0 = 10.0
            self.g = 10.0
            self.results = {
                't': 1.0,
                'x': 10.0,
                'v_y': 10.0,
                'v': 14.14,
                'alpha': 45.0
            }
    
    solution = MockSolution()
    checker = PhysicsChecker(solution)
    success, logs = checker.run_all_tests()
    
    for log in logs:
        print(log)