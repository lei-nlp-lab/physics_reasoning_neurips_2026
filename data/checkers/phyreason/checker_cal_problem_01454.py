import math
import sympy as sp
import traceback
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        self.solution_instance = solution_instance
        
        # Input parameters
        self.v_belt = 2.0  # m/s
        self.m = 1.0      # kg
        self.mu = 0.2     # coefficient of kinetic friction
        self.L = 4.0      # m
        self.g = 10.0     # m/s²
        
        # First scenario results (belt speed = 2 m/s)
        self.a = self._safe_get('a', 0.0)
        self.t1 = self._safe_get('t₁', 0.0)
        self.x1 = self._safe_get('x₁', 0.0)
        self.x2 = self._safe_get('x₂', 0.0)
        self.t2 = self._safe_get('t₂', 0.0)
        self.t_total = self._safe_get('t', 0.0)  # 2.5 s for first scenario
        
        # Second scenario results (minimum belt speed)
        self.v_min = self._safe_get('v_min', 0.0)  # 4 m/s
        self.t_short = 0.0
        # Try to get shortest time - might be stored in same 't' key or different
        if hasattr(solution_instance, 't_short'):
            self.t_short = getattr(solution_instance, 't_short', 0.0)
        elif 't_short' in self.results:
            self.t_short = self.results.get('t_short', 0.0)
        else:
            # Fallback: if t_total is close to 2.0, it might be the second scenario time
            if math.isclose(self.t_total, 2.0, rel_tol=0.05, abs_tol=0.1):
                self.t_short = self.t_total
        
        # Belt speed for first scenario
        self.v = self._safe_get('v', self.v_belt)
        
    def _safe_get(self, key: str, default: float) -> float:
        """Safely get attribute from solution instance or results dict."""
        value = getattr(self.solution_instance, key, 
                       self.results.get(key, default))
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    
    def test_invariants(self) -> List[str]:
        """Test fundamental physics invariants."""
        logs = []
        
        try:
            # 1. Check acceleration from friction
            a_expected = self.mu * self.g  # 2.0 m/s²
            if not math.isclose(self.a, a_expected, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: Acceleration {self.a} != μ·g = {a_expected}")
            else:
                logs.append(f"PASS: Acceleration a = μ·g = {a_expected:.1f} m/s²")
            
            # 2. Check Newton's 2nd Law for accelerated phase
            F_friction = self.mu * self.m * self.g
            a_from_newton = F_friction / self.m
            if not math.isclose(self.a, a_from_newton, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"FAIL: a={self.a} doesn't match F/m={a_from_newton}")
            else:
                logs.append(f"PASS: Newton's 2nd Law: F_friction/m = {a_from_newton:.1f} m/s²")
            
            # 3. Check kinematics for first scenario accelerated phase
            if self.t1 > 0:
                # v = a·t₁
                v_from_accel = self.a * self.t1
                if not math.isclose(self.v_belt, v_from_accel, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: v_belt={self.v_belt} != a·t₁={v_from_accel}")
                else:
                    logs.append(f"PASS: Kinematics: v = a·t₁ = {v_from_accel:.1f} m/s")
                
                # x₁ = ½a·t₁²
                x1_expected = 0.5 * self.a * self.t1**2
                if not math.isclose(self.x1, x1_expected, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: x₁={self.x1} != ½a·t₁²={x1_expected}")
                else:
                    logs.append(f"PASS: x₁ = ½a·t₁² = {x1_expected:.1f} m")
            
            # 4. Check energy-work theorem for friction
            if self.x1 > 0:
                work_friction = -self.mu * self.m * self.g * self.x1
                delta_KE = 0.5 * self.m * self.v_belt**2 - 0
                if not math.isclose(abs(work_friction), delta_KE, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: |W_friction|={abs(work_friction):.2f} ≠ ΔKE={delta_KE:.2f}")
                else:
                    logs.append(f"PASS: Work-Energy: |W_friction| = ΔKE = {delta_KE:.2f} J")
            
            # 5. Check second scenario invariants
            if self.v_min > 0:
                # v_min² = 2·a·L
                v_min_sq_expected = 2 * self.a * self.L
                v_min_expected = math.sqrt(v_min_sq_expected)
                if not math.isclose(self.v_min, v_min_expected, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: v_min={self.v_min} ≠ √(2aL)={v_min_expected:.2f}")
                else:
                    logs.append(f"PASS: v_min = √(2aL) = {v_min_expected:.1f} m/s")
                
                # t_short = v_min / a
                if self.t_short > 0:
                    t_short_expected = self.v_min / self.a
                    if not math.isclose(self.t_short, t_short_expected, rel_tol=0.02, abs_tol=0.02):
                        logs.append(f"FAIL: t_short={self.t_short} ≠ v_min/a={t_short_expected:.2f}")
                    else:
                        logs.append(f"PASS: t_short = v_min/a = {t_short_expected:.1f} s")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def test_trace_consistency(self) -> List[str]:
        """Check consistency between intermediate and final results."""
        logs = []
        
        try:
            # First scenario consistency checks
            if all(v > 0 for v in [self.x1, self.x2, self.L]):
                # x₁ + x₂ = L
                if not math.isclose(self.x1 + self.x2, self.L, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: x₁+x₂={self.x1+self.x2} ≠ L={self.L}")
                else:
                    logs.append(f"PASS: x₁ + x₂ = L = {self.L} m")
            
            if all(v > 0 for v in [self.t1, self.t2, self.t_total]):
                # t₁ + t₂ = t_total
                if not math.isclose(self.t1 + self.t2, self.t_total, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: t₁+t₂={self.t1+self.t2} ≠ t_total={self.t_total}")
                else:
                    logs.append(f"PASS: t₁ + t₂ = t_total = {self.t_total:.1f} s")
            
            if self.t2 > 0 and self.x2 > 0:
                # t₂ = x₂ / v
                t2_expected = self.x2 / self.v_belt
                if not math.isclose(self.t2, t2_expected, rel_tol=0.02, abs_tol=0.02):
                    logs.append(f"FAIL: t₂={self.t2} ≠ x₂/v={t2_expected:.2f}")
                else:
                    logs.append(f"PASS: t₂ = x₂/v = {t2_expected:.1f} s")
            
            # Cross-check between scenarios
            if self.v_min > 0 and self.v_belt > 0:
                if self.v_min <= self.v_belt:
                    logs.append(f"WARN: v_min={self.v_min} ≤ v_belt={self.v_belt} - should be greater for shortest time")
            
            if self.t_short > 0 and self.t_total > 0:
                if self.t_short >= self.t_total:
                    logs.append(f"WARN: t_short={self.t_short} ≥ t_total={self.t_total} - should be shorter")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def test_monotonicity(self) -> List[str]:
        """Test how results change with parameter perturbations."""
        logs = []
        
        try:
            # Perturb coefficient of friction
            mu_perturbed = self.mu * 1.1  # 10% increase
            
            # Expected trends:
            # 1. Acceleration increases: a' = μ'·g
            a_perturbed = mu_perturbed * self.g
            if a_perturbed <= self.a:
                logs.append(f"FAIL: a should increase with μ, got {a_perturbed} ≤ {self.a}")
            else:
                logs.append(f"PASS: a increases with μ: {self.a:.1f} → {a_perturbed:.1f} m/s²")
            
            # 2. Time t₁ decreases: t₁' = v/a'
            if self.t1 > 0:
                t1_perturbed = self.v_belt / a_perturbed
                if t1_perturbed >= self.t1:
                    logs.append(f"FAIL: t₁ should decrease with μ, got {t1_perturbed} ≥ {self.t1}")
                else:
                    logs.append(f"PASS: t₁ decreases with μ: {self.t1:.1f} → {t1_perturbed:.1f} s")
            
            # 3. Distance x₁ decreases: x₁' = ½a'·t₁'²
            if self.x1 > 0 and self.t1 > 0:
                t1_perturbed = self.v_belt / a_perturbed
                x1_perturbed = 0.5 * a_perturbed * t1_perturbed**2
                if x1_perturbed >= self.x1:
                    logs.append(f"FAIL: x₁ should decrease with μ, got {x1_perturbed} ≥ {self.x1}")
                else:
                    logs.append(f"PASS: x₁ decreases with μ: {self.x1:.1f} → {x1_perturbed:.1f} m")
            
            # 4. Total time for first scenario decreases
            if self.t_total > 0:
                # t_total = t₁ + (L - x₁)/v
                t1_perturbed = self.v_belt / a_perturbed
                x1_perturbed = 0.5 * a_perturbed * t1_perturbed**2
                x2_perturbed = self.L - x1_perturbed
                t2_perturbed = x2_perturbed / self.v_belt
                t_total_perturbed = t1_perturbed + t2_perturbed
                
                if t_total_perturbed >= self.t_total:
                    logs.append(f"FAIL: t_total should decrease with μ, got {t_total_perturbed} ≥ {self.t_total}")
                else:
                    logs.append(f"PASS: t_total decreases with μ: {self.t_total:.1f} → {t_total_perturbed:.1f} s")
            
            # 5. For second scenario, v_min increases: v_min' = √(2·μ'·g·L)
            if self.v_min > 0:
                v_min_perturbed = math.sqrt(2 * mu_perturbed * self.g * self.L)
                if v_min_perturbed <= self.v_min:
                    logs.append(f"FAIL: v_min should increase with μ, got {v_min_perturbed} ≤ {self.v_min}")
                else:
                    logs.append(f"PASS: v_min increases with μ: {self.v_min:.1f} → {v_min_perturbed:.1f} m/s")
            
            # 6. Shortest time decreases: t_short' = v_min'/a'
            if self.t_short > 0:
                v_min_perturbed = math.sqrt(2 * mu_perturbed * self.g * self.L)
                t_short_perturbed = v_min_perturbed / a_perturbed
                if t_short_perturbed >= self.t_short:
                    logs.append(f"FAIL: t_short should decrease with μ, got {t_short_perturbed} ≥ {self.t_short}")
                else:
                    logs.append(f"PASS: t_short decreases with μ: {self.t_short:.1f} → {t_short_perturbed:.1f} s")
                    
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status and logs."""
        all_logs = []
        success = True
        
        try:
            # Run each test
            all_logs.append("=== Physics Invariants Test ===")
            inv_logs = self.test_invariants()
            all_logs.extend(inv_logs)
            
            all_logs.append("\n=== Trace Consistency Test ===")
            trace_logs = self.test_trace_consistency()
            all_logs.extend(trace_logs)
            
            all_logs.append("\n=== Monotonicity Test ===")
            mono_logs = self.test_monotonicity()
            all_logs.extend(mono_logs)
            
            # Check for critical failures
            critical_failures = [log for log in all_logs if log.startswith("FAIL:")]
            if critical_failures:
                success = False
                all_logs.append(f"\n❌ Found {len(critical_failures)} critical failures")
            else:
                all_logs.append("\n✅ All physics checks passed")
                
        except Exception as e:
            success = False
            all_logs.append(f"\n❌ Exception in test suite: {str(e)}")
            all_logs.append(traceback.format_exc())
        
        return success, all_logs

# Example usage if run directly
if __name__ == "__main__":
    # Mock solution class for testing
    class MockSolution:
        def __init__(self):
            self.results = {
                'a': 2.0,
                't₁': 1.0,
                'x₁': 1.0,
                'x₂': 3.0,
                't₂': 1.5,
                't': 2.5,
                'v': 2.0,
                'v_min': 4.0,
                't_short': 2.0
            }
    
    mock_solution = MockSolution()
    checker = PhysicsChecker(mock_solution)
    success, logs = checker.run_all_tests()
    
    print(f"Success: {success}")
    for log in logs:
        print(log)