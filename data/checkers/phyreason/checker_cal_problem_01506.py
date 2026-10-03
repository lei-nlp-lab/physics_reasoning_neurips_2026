import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction
        self.y = getattr(solution_instance, 'y', 0.8)
        self.x = getattr(solution_instance, 'x', 0.8)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Extract results with safe fallbacks - handle both 'tanθ' and 'tantheta' keys
        results_dict = getattr(solution_instance, 'results', {})
        self.t = results_dict.get('t', 0.0)
        self.v0 = results_dict.get('v_0', 0.0)
        self.v_y = results_dict.get('v_y', 0.0)
        
        # Try both possible keys for tangent
        self.tan_theta = results_dict.get('tanθ', results_dict.get('tantheta', 0.0))
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Verify projectile motion kinematic equations."""
        logs = []
        success = True
        
        # 1. Check vertical motion: y = 0.5*g*t² (since initial vertical velocity = 0)
        t_calc = math.sqrt(2 * self.y / self.g)
        if not math.isclose(self.t, t_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Vertical motion invariant violated: t = {self.t}, expected ≈{t_calc}")
            success = False
        
        # 2. Check horizontal motion: x = v0*t
        x_calc = self.v0 * self.t
        if not math.isclose(self.x, x_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Horizontal motion invariant violated: v0*t = {x_calc}, expected x ≈{self.x}")
            success = False
        
        # 3. Check vertical velocity: v_y = g*t (since initial v_y0 = 0)
        v_y_calc = self.g * self.t
        if not math.isclose(self.v_y, v_y_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Vertical velocity invariant violated: v_y = {self.v_y}, expected g*t ≈{v_y_calc}")
            success = False
        
        # 4. Check tangent relation: tanθ = v_y / v0
        if self.v0 != 0:
            tan_theta_calc = self.v_y / self.v0
            if not math.isclose(self.tan_theta, tan_theta_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Tangent relation violated: tanθ = {self.tan_theta}, expected v_y/v0 ≈{tan_theta_calc}")
                success = False
        
        # 5. Energy conservation check (kinetic + potential)
        # Initial energy: E_i = 0.5*m*v0² + m*g*y
        # Final energy: E_f = 0.5*m*(v0² + v_y²) (since y=0 at ground)
        # Should have E_i = E_f => 0.5*v0² + g*y = 0.5*(v0² + v_y²)
        left = 0.5*self.v0**2 + self.g*self.y
        right = 0.5*(self.v0**2 + self.v_y**2)
        if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Energy conservation check: initial energy per mass = {left}, final = {right}")
            success = False
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Cross-check intermediate results from STEPS ANALYSIS."""
        logs = []
        success = True
        
        # Step 3 values: t=0.4, v0=2
        if not math.isclose(self.t, 0.4, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Step 3 inconsistency: t = {self.t}, expected ≈0.4")
            success = False
        
        if not math.isclose(self.v0, 2.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Step 3 inconsistency: v0 = {self.v0}, expected ≈2.0")
            success = False
        
        # Step 4: v_y = 4
        if not math.isclose(self.v_y, 4.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Step 4 inconsistency: v_y = {self.v_y}, expected ≈4.0")
            success = False
        
        # Step 5: tanθ = 2
        if not math.isclose(self.tan_theta, 2.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Step 5 inconsistency: tanθ = {self.tan_theta}, expected ≈2.0")
            success = False
        
        # Check that v_y/v0 matches tanθ
        if self.v0 != 0 and not math.isclose(self.tan_theta, self.v_y/self.v0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Trace inconsistency: tanθ({self.tan_theta}) ≠ v_y/v0({self.v_y/self.v0})")
            success = False
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Perturb input (y) and check trends."""
        logs = []
        success = True
        
        # Store original values
        y_orig = self.y
        t_orig = self.t
        v0_orig = self.v0
        
        # Perturb y by +10%
        y_perturbed = y_orig * 1.10
        
        # Recalculate using projectile formulas
        t_new = math.sqrt(2 * y_perturbed / self.g)
        v0_new = self.x / t_new
        
        # Expected trends from KB:
        # 1. As y increases, t should increase (t ∝ √y)
        if t_new <= t_orig:
            logs.append(f"Monotonicity failure: t should increase with y (t_orig={t_orig:.3f}, t_new={t_new:.3f})")
            success = False
        
        # 2. As y increases, v0 should decrease (v0 = x/t, and t increases)
        if v0_new >= v0_orig:
            logs.append(f"Monotonicity failure: v0 should decrease with y (v0_orig={v0_orig:.3f}, v0_new={v0_new:.3f})")
            success = False
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return results."""
        logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                test_success, test_logs = test_func()
                logs.append(f"{test_name}: {'PASS' if test_success else 'FAIL'}")
                logs.extend([f"  {log}" for log in test_logs])
                if not test_success:
                    overall_success = False
            except Exception as e:
                logs.append(f"{test_name}: ERROR - {str(e)}")
                overall_success = False
        
        return overall_success, logs