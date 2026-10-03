import math
import sympy as sp
import traceback
from typing import Tuple, List, Optional, Dict, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """Safe initialization extracting all inputs and results."""
        self.solution = solution_instance
        
        # Safe extraction of input parameters
        self.H = self._safe_get('H', solution_instance)
        self.h = self._safe_get('h', solution_instance)
        self.r = self._safe_get('r', solution_instance)
        self.theta = self._safe_get('theta', solution_instance)
        self.g = self._safe_get('g', solution_instance, 9.8)
        self.rho = self._safe_get('rho', solution_instance, 1000.0)
        self.S = self._safe_get('S', solution_instance, 1.0)
        self.dm = self._safe_get('dm', solution_instance, 1.0)  # Δm
        
        # Safe extraction of results from KB mapping
        self.v_m = self._safe_get('v_m', solution_instance)
        self.v_c = self._safe_get('v_c', solution_instance)
        self.F_N1 = self._safe_get('F_{N1}', solution_instance)
        self.F_N2 = self._safe_get('F_{N2}', solution_instance)
        self.F_N = self._safe_get('F_N', solution_instance)
        self.t = self._safe_get('t', solution_instance)
        self.x_DE = self._safe_get('x_DE', solution_instance)
        
        self.logs = []
        
    def _safe_get(self, key: str, instance: Any, default: float = None) -> Optional[float]:
        """Safe attribute access with fallback to results dictionary."""
        try:
            value = getattr(instance, key, None)
            if value is None and hasattr(instance, 'results'):
                value = instance.results.get(key, default)
            return float(value) if value is not None else default
        except (AttributeError, KeyError, TypeError, ValueError):
            return default
    
    def _log(self, message: str, error: bool = False):
        """Add message to logs."""
        self.logs.append(f"{'ERROR' if error else 'INFO'}: {message}")
    
    def test_invariants(self) -> bool:
        """Test energy conservation, Newton's laws, and centripetal force."""
        success = True
        
        try:
            # 1. Energy conservation from tank to point B (v_m)
            if None not in (self.v_m, self.H, self.h, self.r, self.theta, self.g):
                expected_v_m = math.sqrt(2 * self.g * (self.H - self.h + self.r * math.sin(self.theta)))
                if not math.isclose(self.v_m, expected_v_m, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Energy invariant failed: v_m={self.v_m}, expected={expected_v_m}", error=True)
                    success = False
                else:
                    self._log(f"Energy conservation (v_m) check passed")
            
            # 2. Energy conservation from tank to point C (v_c)
            if None not in (self.v_c, self.H, self.h, self.g):
                expected_v_c = math.sqrt(2 * self.g * (self.H - self.h))
                if not math.isclose(self.v_c, expected_v_c, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Energy invariant failed: v_c={self.v_c}, expected={expected_v_c}", error=True)
                    success = False
                else:
                    self._log(f"Energy conservation (v_c) check passed")
            
            # 3. Newton's 2nd law at point B: centripetal force balance
            if None not in (self.F_N1, self.dm, self.g, self.theta, self.v_m, self.r):
                # F_N1 - Δmg sinθ = Δm v_m² / r
                centripetal_force = self.dm * self.v_m**2 / self.r
                gravity_component = self.dm * self.g * math.sin(self.theta)
                expected_F_N1 = centripetal_force + gravity_component
                if not math.isclose(self.F_N1, expected_F_N1, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Newton's 2nd law failed: F_N1={self.F_N1}, expected={expected_F_N1}", error=True)
                    success = False
                else:
                    self._log(f"Newton's 2nd law (centripetal) check passed")
            
            # 4. Tangential force at point B
            if None not in (self.F_N2, self.dm, self.g, self.theta):
                expected_F_N2 = self.dm * self.g * math.cos(self.theta)
                if not math.isclose(self.F_N2, expected_F_N2, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Tangential force failed: F_N2={self.F_N2}, expected={expected_F_N2}", error=True)
                    success = False
                else:
                    self._log(f"Tangential force check passed")
            
            # 5. Pythagorean theorem for total normal force
            if None not in (self.F_N, self.F_N1, self.F_N2):
                expected_F_N = math.sqrt(self.F_N1**2 + self.F_N2**2)
                if not math.isclose(self.F_N, expected_F_N, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Pythagorean theorem failed: F_N={self.F_N}, expected={expected_F_N}", error=True)
                    success = False
                else:
                    self._log(f"Vector sum (Pythagorean) check passed")
            
            # 6. Free fall time from height h
            if None not in (self.t, self.h, self.g):
                expected_t = math.sqrt(2 * self.h / self.g)
                if not math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Free fall time failed: t={self.t}, expected={expected_t}", error=True)
                    success = False
                else:
                    self._log(f"Free fall time check passed")
            
            # 7. Projectile motion: horizontal distance
            if None not in (self.x_DE, self.v_c, self.t):
                expected_x_DE = self.v_c * self.t
                if not math.isclose(self.x_DE, expected_x_DE, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Projectile motion failed: x_DE={self.x_DE}, expected={expected_x_DE}", error=True)
                    success = False
                else:
                    self._log(f"Projectile motion (range) check passed")
            
            # 8. Alternative: x_DE = 2√[(H-h)h] from combined formulas
            if None not in (self.x_DE, self.H, self.h):
                alt_expected = 2 * math.sqrt((self.H - self.h) * self.h)
                if not math.isclose(self.x_DE, alt_expected, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Alternative x_DE formula mismatch: {self.x_DE} vs {alt_expected}", error=True)
                    success = False
                else:
                    self._log(f"Alternative x_DE formula check passed")
                    
        except Exception as e:
            self._log(f"Exception in test_invariants: {str(e)}\n{traceback.format_exc()}", error=True)
            success = False
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final answer."""
        success = True
        
        try:
            # Check that v_m and v_c are consistent with their energy relations
            if None not in (self.v_m, self.v_c, self.r, self.theta, self.g):
                # v_m² - v_c² = 2g r sinθ
                expected_diff = 2 * self.g * self.r * math.sin(self.theta)
                actual_diff = self.v_m**2 - self.v_c**2
                if not math.isclose(actual_diff, expected_diff, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Trace consistency v_m²-v_c² failed: {actual_diff} vs {expected_diff}", error=True)
                    success = False
                else:
                    self._log(f"v_m and v_c energy relation check passed")
            
            # Check that F_N1 expression matches the derived formula from steps
            if None not in (self.F_N1, self.dm, self.g, self.H, self.h, self.r, self.theta):
                expected_F_N1_formula = 2 * self.dm * self.g * (self.H - self.h) / self.r + 3 * self.dm * self.g * math.sin(self.theta)
                if not math.isclose(self.F_N1, expected_F_N1_formula, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"F_N1 formula consistency failed: {self.F_N1} vs {expected_F_N1_formula}", error=True)
                    success = False
                else:
                    self._log(f"F_N1 formula consistency check passed")
            
            # Check that final F_N matches the target expression
            if None not in (self.F_N, self.dm, self.g, self.H, self.h, self.r, self.theta):
                term1 = 2 * self.dm * self.g * (self.H - self.h) / self.r + 3 * self.dm * self.g * math.sin(self.theta)
                term2 = self.dm * self.g * math.cos(self.theta)
                expected_F_N = math.sqrt(term1**2 + term2**2)
                if not math.isclose(self.F_N, expected_F_N, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"F_N formula consistency failed: {self.F_N} vs {expected_F_N}", error=True)
                    success = False
                else:
                    self._log(f"F_N formula consistency check passed")
            
            # Check that x_DE matches the target expression 2√[(H-h)h]
            if None not in (self.x_DE, self.H, self.h):
                expected_x_DE_formula = 2 * math.sqrt((self.H - self.h) * self.h)
                if not math.isclose(self.x_DE, expected_x_DE_formula, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"x_DE formula consistency failed: {self.x_DE} vs {expected_x_DE_formula}", error=True)
                    success = False
                else:
                    self._log(f"x_DE formula consistency check passed")
                    
        except Exception as e:
            self._log(f"Exception in test_trace_consistency: {str(e)}\n{traceback.format_exc()}", error=True)
            success = False
        
        return success
    
    def test_monotonicity(self) -> bool:
        """Perturb input parameters and check expected trends."""
        success = True
        
        try:
            # Test 1: Increase H should increase v_m, v_c, F_N1, x_DE
            if None not in (self.H, self.h, self.r, self.theta, self.g, self.dm, self.v_m, self.v_c, self.F_N1, self.x_DE):
                perturbed_H = self.H * 1.01  # +1%
                
                # Recalculate using formulas (not solution methods)
                new_v_m = math.sqrt(2 * self.g * (perturbed_H - self.h + self.r * math.sin(self.theta)))
                new_v_c = math.sqrt(2 * self.g * (perturbed_H - self.h))
                new_F_N1 = 2 * self.dm * self.g * (perturbed_H - self.h) / self.r + 3 * self.dm * self.g * math.sin(self.theta)
                new_x_DE = 2 * math.sqrt((perturbed_H - self.h) * self.h)
                
                if not (new_v_m > self.v_m):
                    self._log(f"Monotonicity failed: v_m should increase with H", error=True)
                    success = False
                if not (new_v_c > self.v_c):
                    self._log(f"Monotonicity failed: v_c should increase with H", error=True)
                    success = False
                if not (new_F_N1 > self.F_N1):
                    self._log(f"Monotonicity failed: F_N1 should increase with H", error=True)
                    success = False
                if not (new_x_DE > self.x_DE):
                    self._log(f"Monotonicity failed: x_DE should increase with H", error=True)
                    success = False
                else:
                    self._log(f"H monotonicity check passed")
            
            # Test 2: Increase h should decrease v_m, v_c, and have non-monotonic effect on x_DE
            if None not in (self.H, self.h, self.r, self.theta, self.g, self.dm, self.v_m, self.v_c, self.x_DE):
                if self.h > 0:
                    perturbed_h = self.h * 1.01  # +1%
                    
                    new_v_m = math.sqrt(2 * self.g * (self.H - perturbed_h + self.r * math.sin(self.theta)))
                    new_v_c = math.sqrt(2 * self.g * (self.H - perturbed_h))
                    new_x_DE = 2 * math.sqrt((self.H - perturbed_h) * perturbed_h)
                    
                    if not (new_v_m < self.v_m):
                        self._log(f"Monotonicity failed: v_m should decrease with h", error=True)
                        success = False
                    if not (new_v_c < self.v_c):
                        self._log(f"Monotonicity failed: v_c should decrease with h", error=True)
                        success = False
                    
                    # x_DE peaks at h = H/2, so check if we're on correct side
                    if self.h < self.H/2:
                        if not (new_x_DE > self.x_DE):
                            self._log(f"Monotonicity failed: x_DE should increase with h when h < H/2", error=True)
                            success = False
                    else:
                        if not (new_x_DE < self.x_DE):
                            self._log(f"Monotonicity failed: x_DE should decrease with h when h > H/2", error=True)
                            success = False
                    self._log(f"h monotonicity check passed")
            
            # Test 3: Increase θ (sinθ increases for 0° to 90°) should increase v_m and F_N1
            if None not in (self.H, self.h, self.r, self.theta, self.g, self.dm, self.v_m, self.F_N1):
                if 0 <= self.theta < math.pi/2:  # Only test if in valid range
                    perturbed_theta = self.theta + 0.01  # +0.01 rad
                    
                    new_v_m = math.sqrt(2 * self.g * (self.H - self.h + self.r * math.sin(perturbed_theta)))
                    new_F_N1 = 2 * self.dm * self.g * (self.H - self.h) / self.r + 3 * self.dm * self.g * math.sin(perturbed_theta)
                    
                    if not (new_v_m > self.v_m):
                        self._log(f"Monotonicity failed: v_m should increase with θ", error=True)
                        success = False
                    if not (new_F_N1 > self.F_N1):
                        self._log(f"Monotonicity failed: F_N1 should increase with θ", error=True)
                        success = False
                    else:
                        self._log(f"θ monotonicity check passed")
            
            # Test 4: Increase r should decrease F_N1 (since (H-h)/r term decreases)
            if None not in (self.H, self.h, self.r, self.theta, self.g, self.dm, self.F_N1):
                perturbed_r = self.r * 1.01  # +1%
                
                new_F_N1 = 2 * self.dm * self.g * (self.H - self.h) / perturbed_r + 3 * self.dm * self.g * math.sin(self.theta)
                
                if not (new_F_N1 < self.F_N1):
                    self._log(f"Monotonicity failed: F_N1 should decrease with r", error=True)
                    success = False
                else:
                    self._log(f"r monotonicity check passed")
                    
        except Exception as e:
            self._log(f"Exception in test_monotonicity: {str(e)}\n{traceback.format_exc()}", error=True)
            success = False
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all physics audits and return success status and logs."""
        self.logs = []
        
        # Run tests
        test1_success = self.test_invariants()
        test2_success = self.test_trace_consistency()
        test3_success = self.test_monotonicity()
        
        overall_success = test1_success and test2_success and test3_success
        
        # Summary
        self._log(f"\n=== SUMMARY ===")
        self._log(f"Invariants test: {'PASSED' if test1_success else 'FAILED'}")
        self._log(f"Trace consistency: {'PASSED' if test2_success else 'FAILED'}")
        self._log(f"Monotonicity: {'PASSED' if test3_success else 'FAILED'}")
        self._log(f"Overall: {'PASSED' if overall_success else 'FAILED'}")
        
        return overall_success, self.logs