import sympy as sp
import math
import traceback
from typing import Tuple, List, Any, Dict

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """Initialize checker with safe attribute access."""
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Safe extraction of inputs with defaults
        self.m = getattr(solution_instance, 'm', 1.0)  # symbolic mass
        self.mu1 = getattr(solution_instance, 'mu1', 0.1)
        self.mu2 = getattr(solution_instance, 'mu2', 0.05)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.v0 = getattr(solution_instance, 'v0', 3.0)
        
        # Safe extraction of results with fallbacks
        self.v_1 = self.results.get('v_1')
        self.v_A = self.results.get('v_A')
        self.v_C = self.results.get('v_C')
        self.t_C = self.results.get('t_C')
        self.x_C = self.results.get('x_C')
        self.v_2 = self.results.get('v_2')
        self.t_1 = self.results.get('t_1')
        self.t_2 = self.results.get('t_2')
        self.t = self.results.get('t')
        
        # Physics constants
        self.decel_B = -self.mu1 * self.g  # B on A
        self.decel_C = -self.mu2 * self.g  # C on track
        
        self.logs = []
        
    def _log(self, msg: str, success: bool):
        """Helper to log test results."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {msg}")
        return success
        
    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Fuzzy numerical comparison."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> bool:
        """Verify conservation laws and kinematic relationships."""
        success = True
        
        # Step 1: Conservation of momentum for A+B system
        if self.v_1 is not None:
            # Symbolic: 2m*v0 = (2m + m)*v_1 => v_1 = (2/3)*v0
            expected_v1 = (2/3) * self.v0
            if not self._is_close(self.v_1, expected_v1):
                success = self._log(
                    f"Step 1 momentum: v_1={self.v_1:.3f}, expected {expected_v1:.3f}", 
                    False
                )
            else:
                self._log(f"Step 1 momentum: v_1={self.v_1:.3f} correct", True)
        
        # Steps 2-4: Elastic collision between A and C
        if all(v is not None for v in [self.v_1, self.v_A, self.v_C]):
            # Momentum conservation: m*v_1 = m*v_A + 3m*v_C => v_1 = v_A + 3*v_C
            momentum_eq = self.v_1 - (self.v_A + 3*self.v_C)
            if not self._is_close(momentum_eq, 0.0):
                success = self._log(
                    f"Momentum in collision: v_1 - (v_A + 3v_C) = {momentum_eq:.3e}", 
                    False
                )
            else:
                self._log("Momentum in collision satisfied", True)
            
            # Kinetic energy conservation: 0.5*m*v_1^2 = 0.5*m*v_A^2 + 0.5*3m*v_C^2
            # => v_1^2 = v_A^2 + 3*v_C^2
            energy_eq = self.v_1**2 - (self.v_A**2 + 3*self.v_C**2)
            if not self._is_close(energy_eq, 0.0, abs_tol=0.1):  # Energy tolerances slightly larger
                success = self._log(
                    f"Energy in collision: v_1^2 - (v_A^2 + 3v_C^2) = {energy_eq:.3e}", 
                    False
                )
            else:
                self._log("Energy in collision satisfied", True)
            
            # Derived relationship: v_A = -v_1/2, v_C = v_1/2
            if not (self._is_close(self.v_A, -self.v_1/2) and self._is_close(self.v_C, self.v_1/2)):
                self._log(
                    f"Derived relations: v_A={self.v_A:.3f} (expected {-self.v_1/2:.3f}), "
                    f"v_C={self.v_C:.3f} (expected {self.v_1/2:.3f})", 
                    False
                )
                success = False
            else:
                self._log("Derived collision relations correct", True)
        
        # Step 5: Kinematics for C
        if self.v_C is not None and self.t_C is not None:
            # t_C = v_C / (μ2*g) for constant deceleration to 0
            expected_tC = self.v_C / (self.mu2 * self.g)
            if not self._is_close(self.t_C, expected_tC):
                success = self._log(
                    f"Step 5: t_C={self.t_C:.3f}, expected {expected_tC:.3f}", 
                    False
                )
            else:
                self._log(f"Step 5: t_C={self.t_C:.3f} correct", True)
        
        # Step 6: Displacement of C
        if self.v_C is not None and self.x_C is not None:
            # x_C = v_C^2 / (2*μ2*g)
            expected_xC = self.v_C**2 / (2 * self.mu2 * self.g)
            if not self._is_close(self.x_C, expected_xC):
                success = self._log(
                    f"Step 6: x_C={self.x_C:.3f}, expected {expected_xC:.3f}", 
                    False
                )
            else:
                self._log(f"Step 6: x_C={self.x_C:.3f} correct", True)
        
        # Step 7: Momentum for A+B after collision
        if all(v is not None for v in [self.v_1, self.v_A, self.v_2]):
            # 2m*v_1 + m*v_A = 3m*v_2 => v_2 = (2*v_1 + v_A)/3
            expected_v2 = (2*self.v_1 + self.v_A) / 3
            if not self._is_close(self.v_2, expected_v2):
                success = self._log(
                    f"Step 7: v_2={self.v_2:.3f}, expected {expected_v2:.3f}", 
                    False
                )
            else:
                self._log(f"Step 7: v_2={self.v_2:.3f} correct", True)
        
        # Step 8: Kinematics for A+B relative motion
        if all(v is not None for v in [self.v_1, self.v_2, self.t_1]):
            # Relative deceleration between B and A is μ1*g
            # Time to equalize: t_1 = (v_1 - v_2) / (μ1*g)
            expected_t1 = (self.v_1 - self.v_2) / (self.mu1 * self.g)
            if not self._is_close(self.t_1, expected_t1):
                success = self._log(
                    f"Step 8: t_1={self.t_1:.3f}, expected {expected_t1:.3f}", 
                    False
                )
            else:
                self._log(f"Step 8: t_1={self.t_1:.3f} correct", True)
        
        # Step 9: Uniform motion of A+B
        if all(v is not None for v in [self.x_C, self.v_2, self.t_2]):
            # t_2 = x_C / v_2
            expected_t2 = self.x_C / self.v_2
            if not self._is_close(self.t_2, expected_t2):
                success = self._log(
                    f"Step 9: t_2={self.t_2:.3f}, expected {expected_t2:.3f}", 
                    False
                )
            else:
                self._log(f"Step 9: t_2={self.t_2:.3f} correct", True)
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Check intermediate results are consistent with final answer."""
        success = True
        
        # Check t = t_C + t_1 + t_2
        if all(t_val is not None for t_val in [self.t, self.t_C, self.t_1, self.t_2]):
            sum_t = self.t_C + self.t_1 + self.t_2
            if not self._is_close(self.t, sum_t):
                success = self._log(
                    f"Time sum: t={self.t:.3f}, but t_C+t_1+t_2={sum_t:.3f}", 
                    False
                )
            else:
                self._log(f"Time sum consistent: t={self.t:.3f}", True)
        
        # Check displacement consistency for C
        if self.t_C is not None and self.v_C is not None:
            # Using kinematics: x_C_alt = v_C*t_C - 0.5*μ2*g*t_C^2
            x_C_alt = self.v_C * self.t_C + 0.5 * self.decel_C * self.t_C**2
            if self.x_C is not None and not self._is_close(self.x_C, x_C_alt):
                success = self._log(
                    f"xC kinematic check: {self.x_C:.3f} vs {x_C_alt:.3f}", 
                    False
                )
        
        # Check v_2 from energy considerations during friction phase
        if all(v is not None for v in [self.v_1, self.v_A, self.v_2]):
            # Work-energy: friction does negative work on B, positive on A
            # Change in KE of B = -μ1*2m*g * displacement_B_rel
            # Change in KE of A = +μ1*2m*g * displacement_A_rel
            # The relative displacement calculation is complex, but we can check
            # that v_2 is between v_A and v_1
            if not (min(self.v_A, self.v_1) <= self.v_2 <= max(self.v_A, self.v_1)):
                self._log(
                    f"v_2={self.v_2:.3f} not between v_A={self.v_A:.3f} and v_1={self.v_1:.3f}", 
                    False
                )
                success = False
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb an input parameter and verify expected trends.
        Returns to original state.
        """
        try:
            # Save original state
            orig_v0 = self.v0
            
            # Perturb v0 by +5%
            perturb_factor = 1.05
            self.v0 *= perturb_factor
            
            # Recalculate expected chain using simplified formulas
            # v_1 = (2/3)*v0
            v_1_new = (2/3) * self.v0
            
            # Collision: v_A = -v_1/2, v_C = v_1/2
            v_A_new = -v_1_new / 2
            v_C_new = v_1_new / 2
            
            # t_C = v_C / (μ2*g)
            t_C_new = v_C_new / (self.mu2 * self.g)
            
            # x_C = v_C^2 / (2*μ2*g)
            x_C_new = v_C_new**2 / (2 * self.mu2 * self.g)
            
            # v_2 = (2*v_1 + v_A)/3 = v_1/2
            v_2_new = (2*v_1_new + v_A_new) / 3  # Should equal v_1/2
            
            # t_1 = (v_1 - v_2) / (μ1*g)
            t_1_new = (v_1_new - v_2_new) / (self.mu1 * self.g)
            
            # t_2 = x_C / v_2
            t_2_new = x_C_new / v_2_new
            
            # t_total = t_C + t_1 + t_2
            t_total_new = t_C_new + t_1_new + t_2_new
            
            # Check monotonic trends
            trends_ok = True
            
            # v0 increased, so v_1 should increase
            if v_1_new <= self.v_1:
                self._log(f"Monotonicity: v_1 did not increase with v0", False)
                trends_ok = False
            
            # v_C should increase with v0
            if v_C_new <= self.v_C:
                self._log(f"Monotonicity: v_C did not increase with v0", False)
                trends_ok = False
            
            # x_C should increase with v0 (quadratic)
            if x_C_new <= self.x_C:
                self._log(f"Monotonicity: x_C did not increase with v0", False)
                trends_ok = False
            
            # t_total should increase with v0 (linear relationship)
            if t_total_new <= self.t:
                self._log(
                    f"Monotonicity: t_total={t_total_new:.3f} <= original {self.t:.3f}", 
                    False
                )
                trends_ok = False
            
            # Restore original state
            self.v0 = orig_v0
            
            if trends_ok:
                self._log("Monotonicity tests passed: trends as expected", True)
            
            return trends_ok
            
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {str(e)}", False)
            # Restore original state on error
            self.v0 = getattr(self.solution, 'v0', 3.0)
            return False
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return results."""
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for test_name, test_func in tests:
            try:
                if not test_func():
                    success = False
                    self.logs.append(f"{test_name} test failed")
            except Exception as e:
                success = False
                self.logs.append(f"{test_name} test raised exception: {str(e)}")
                self.logs.append(traceback.format_exc())
        
        # Summary
        if success:
            self.logs.append("All physics checks passed")
        else:
            self.logs.append("Some physics checks failed")
        
        return success, self.logs