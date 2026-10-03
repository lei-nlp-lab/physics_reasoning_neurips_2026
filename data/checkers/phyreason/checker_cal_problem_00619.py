import math
import sympy as sp
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        # Safe attribute access using getattr with fallback to results dict
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract all known quantities with safe defaults
        self.m = 50.0  # kg (given stationary reading)
        self.g = 10.0  # m/s² (given)
        self.t1 = 10.0  # s (given acceleration phase duration)
        self.t2 = 80.0  # s (given constant speed duration)
        
        # Extract computed results with safe access
        self.a1 = self._safe_get('a_1', 0.0)
        self.F = self._safe_get('F', 0.0)
        self.t3 = self._safe_get('t_3', 0.0)
        self.a2 = self._safe_get('a_2', 0.0)
        self.v = self._safe_get('v', 0.0)
        self.h = self._safe_get('h', 0.0)
        self.F_prime = self._safe_get("F'", 0.0)  # F' during deceleration
        
        # For monotonicity test: store original state
        self._original_state = {
            'm': self.m,
            'a1': self.a1,
            'v': self.v,
            'h': self.h,
            't3': self.t3
        }
        
    def _safe_get(self, key: str, default: float) -> float:
        """Safely get value from results with fallback."""
        return self.results.get(key, default)
    
    def _is_close(self, a: float, b: float) -> bool:
        """Robust fuzzy comparison for physics values."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Test Newton's Laws and other invariants."""
        logs = []
        success = True
        
        # 1. Newton's 2nd Law for acceleration phase
        # Apparent weight formula: N = m(g + a1) = 52.5 * g
        expected_F = (52.5 - 50.0) * self.g  # Should be 25 N
        if not self._is_close(self.F, expected_F):
            logs.append(f"FAIL: Startup resultant force F = {self.F}, expected {expected_F}")
            success = False
        else:
            logs.append(f"PASS: Startup resultant force F = {self.F} ≈ {expected_F}")
        
        # Check a1 = F/m
        expected_a1 = expected_F / self.m  # Should be 0.5 m/s²
        if not self._is_close(self.a1, expected_a1):
            logs.append(f"FAIL: Startup acceleration a1 = {self.a1}, expected {expected_a1}")
            success = False
        else:
            logs.append(f"PASS: Startup acceleration a1 = {self.a1} ≈ {expected_a1}")
        
        # 2. Newton's 2nd Law for deceleration phase
        # Apparent weight: N = m(g + a2) = 45 * g
        expected_F_prime = (45.0 - 50.0) * self.g  # Should be -50 N
        if not self._is_close(self.F_prime, expected_F_prime):
            logs.append(f"FAIL: Deceleration resultant force F' = {self.F_prime}, expected {expected_F_prime}")
            success = False
        else:
            logs.append(f"PASS: Deceleration resultant force F' = {self.F_prime} ≈ {expected_F_prime}")
        
        # Check a2 = F'/m
        expected_a2 = expected_F_prime / self.m  # Should be -1 m/s²
        if not self._is_close(self.a2, expected_a2):
            logs.append(f"FAIL: Deceleration acceleration a2 = {self.a2}, expected {expected_a2}")
            success = False
        else:
            logs.append(f"PASS: Deceleration acceleration a2 = {self.a2} ≈ {expected_a2}")
        
        # 3. Check velocity from kinematics: v = a1 * t1
        expected_v = expected_a1 * self.t1  # Should be 5 m/s
        if not self._is_close(self.v, expected_v):
            logs.append(f"FAIL: Velocity v = {self.v}, expected {expected_v}")
            success = False
        else:
            logs.append(f"PASS: Velocity v = {self.v} ≈ {expected_v}")
        
        # 4. Check deceleration time: t3 = v / |a2|
        expected_t3 = abs(expected_v / expected_a2)  # Should be 5 s
        if not self._is_close(self.t3, expected_t3):
            logs.append(f"FAIL: Deceleration time t3 = {self.t3}, expected {expected_t3}")
            success = False
        else:
            logs.append(f"PASS: Deceleration time t3 = {self.t3} ≈ {expected_t3}")
        
        # 5. Energy conservation check (work-energy theorem)
        # Total work done = ΔKE + ΔPE = 0 at start and end (v=0 at both ends)
        # But during motion, we can check consistency:
        # Height should match area under v-t curve
        total_time = self.t1 + self.t2 + self.t3
        avg_vel = self.v / 2  # for acceleration and deceleration phases
        # Area = rectangle + two triangles
        area_acc = 0.5 * self.t1 * self.v
        area_const = self.t2 * self.v
        area_dec = 0.5 * self.t3 * self.v
        expected_h = area_acc + area_const + area_dec
        
        if not self._is_close(self.h, expected_h):
            logs.append(f"FAIL: Height h = {self.h}, expected from v-t area {expected_h}")
            success = False
        else:
            logs.append(f"PASS: Height h = {self.h} ≈ {expected_h} (v-t area)")
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Check consistency between intermediate and final results."""
        logs = []
        success = True
        
        # 1. Verify v = a1 * t1 (already checked in invariants, but double-check)
        if not self._is_close(self.v, self.a1 * self.t1):
            logs.append(f"FAIL: v = {self.v} ≠ a1*t1 = {self.a1 * self.t1}")
            success = False
        else:
            logs.append(f"PASS: v = a1*t1 = {self.v}")
        
        # 2. Verify t3 = v / |a2|
        if not self._is_close(self.t3, abs(self.v / self.a2)):
            logs.append(f"FAIL: t3 = {self.t3} ≠ v/|a2| = {abs(self.v / self.a2)}")
            success = False
        else:
            logs.append(f"PASS: t3 = v/|a2| = {self.t3}")
        
        # 3. Verify F = m*a1 and F' = m*a2
        if not self._is_close(self.F, self.m * self.a1):
            logs.append(f"FAIL: F = {self.F} ≠ m*a1 = {self.m * self.a1}")
            success = False
        else:
            logs.append(f"PASS: F = m*a1 = {self.F}")
        
        if not self._is_close(self.F_prime, self.m * self.a2):
            logs.append(f"FAIL: F' = {self.F_prime} ≠ m*a2 = {self.m * self.a2}")
            success = False
        else:
            logs.append(f"PASS: F' = m*a2 = {self.F_prime}")
        
        # 4. Check height formula consistency
        # Using the formula from step12: h = (t_uniform + t_total)*v/2
        # Where t_uniform = t2 = 80s, t_total = t1 + t2 + t3
        t_total = self.t1 + self.t2 + self.t3
        expected_h_formula = (self.t2 + t_total) * self.v / 2
        
        if not self._is_close(self.h, expected_h_formula):
            logs.append(f"FAIL: h = {self.h} ≠ (t2 + t_total)*v/2 = {expected_h_formula}")
            success = False
        else:
            logs.append(f"PASS: h = (t2 + t_total)*v/2 = {self.h}")
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Perturb mass and check if results change in expected direction.
        Restore original state after test.
        """
        logs = []
        success = True
        
        # Perturb mass by +10%
        perturbed_m = self.m * 1.10
        
        # Recalculate expected values using formulas
        # Acceleration phase
        F_perturbed = (52.5 - perturbed_m) * self.g
        a1_perturbed = F_perturbed / perturbed_m
        v_perturbed = a1_perturbed * self.t1
        
        # Deceleration phase
        F_prime_perturbed = (45.0 - perturbed_m) * self.g
        a2_perturbed = F_prime_perturbed / perturbed_m
        t3_perturbed = abs(v_perturbed / a2_perturbed)
        
        # Height calculation
        area_acc = 0.5 * self.t1 * v_perturbed
        area_const = self.t2 * v_perturbed
        area_dec = 0.5 * t3_perturbed * v_perturbed
        h_perturbed = area_acc + area_const + area_dec
        
        # Check monotonic trends
        # 1. If mass increases, acceleration should decrease (F = (m_app - m)g)
        if not (a1_perturbed < self.a1):
            logs.append(f"FAIL: a1 should decrease with increased mass. Original: {self.a1}, Perturbed: {a1_perturbed}")
            success = False
        else:
            logs.append(f"PASS: a1 decreases with increased mass: {self.a1} → {a1_perturbed}")
        
        # 2. If mass increases, velocity should decrease (v = a1*t1)
        if not (v_perturbed < self.v):
            logs.append(f"FAIL: v should decrease with increased mass. Original: {self.v}, Perturbed: {v_perturbed}")
            success = False
        else:
            logs.append(f"PASS: v decreases with increased mass: {self.v} → {v_perturbed}")
        
        # 3. If mass increases, deceleration magnitude should change
        # |a2| = |(45-m)g/m| = |(45/m - 1)g|
        # For m=50: |45/50 - 1|*10 = 1
        # For m=55: |45/55 - 1|*10 ≈ 1.818
        # So |a2| should increase with increased mass
        if not (abs(a2_perturbed) > abs(self.a2)):
            logs.append(f"FAIL: |a2| should increase with increased mass. Original: {abs(self.a2)}, Perturbed: {abs(a2_perturbed)}")
            success = False
        else:
            logs.append(f"PASS: |a2| increases with increased mass: {abs(self.a2)} → {abs(a2_perturbed)}")
        
        # 4. If mass increases and velocity decreases, height should decrease
        if not (h_perturbed < self.h):
            logs.append(f"FAIL: h should decrease with increased mass. Original: {self.h}, Perturbed: {h_perturbed}")
            success = False
        else:
            logs.append(f"PASS: h decreases with increased mass: {self.h} → {h_perturbed}")
        
        # 5. Check deceleration time trend: t3 = v/|a2|
        # v decreases, |a2| increases, so t3 should decrease
        if not (t3_perturbed < self.t3):
            logs.append(f"FAIL: t3 should decrease with increased mass. Original: {self.t3}, Perturbed: {t3_perturbed}")
            success = False
        else:
            logs.append(f"PASS: t3 decreases with increased mass: {self.t3} → {t3_perturbed}")
        
        # Restore original state (not strictly necessary as we didn't modify self)
        # but included for completeness
        self.m = self._original_state['m']
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        # Test 1: Invariants
        success1, logs1 = self.test_invariants()
        all_logs.append("=== TEST 1: Invariants ===")
        all_logs.extend(logs1)
        overall_success = overall_success and success1
        
        # Test 2: Trace Consistency
        success2, logs2 = self.test_trace_consistency()
        all_logs.append("\n=== TEST 2: Trace Consistency ===")
        all_logs.extend(logs2)
        overall_success = overall_success and success2
        
        # Test 3: Monotonicity
        success3, logs3 = self.test_monotonicity()
        all_logs.append("\n=== TEST 3: Monotonicity ===")
        all_logs.extend(logs3)
        overall_success = overall_success and success3
        
        # Final summary
        all_logs.append(f"\n=== OVERALL RESULT: {'PASS' if overall_success else 'FAIL'} ===")
        
        return overall_success, all_logs

# Example usage (commented out for safety):
# checker = PhysicsChecker(solution)
# success, logs = checker.run_all_tests()
# for log in logs:
#     print(log)