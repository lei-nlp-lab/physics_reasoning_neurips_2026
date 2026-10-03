import math
import sympy as sp
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Initialize with safe attribute access."""
        self.solution = solution_instance
        
        # Extract inputs from problem context (fixed)
        self.total_distance = 60.0  # AC distance
        self.total_time = 11.0      # A to C time
        self.dist_last_2_sec = 4.5  # distance in 2nd second before stopping
        
        # Safe extraction of results
        results = getattr(solution_instance, 'results', {})
        
        self.a = self._safe_get('a', results, 0.0)          # acceleration (should be negative for deceleration)
        self.x_prime = self._safe_get("x'", results, 0.0)   # BC distance
        self.x = self._safe_get('x', results, 0.0)          # total distance (should be 60)
        self.t1 = self._safe_get('t₁', results, 0.0)        # AB time
        self.t2 = self._safe_get('t₂', results, 0.0)        # BC time
        self.t = self._safe_get('t', results, 0.0)          # total time
        self.v = self._safe_get('v', results, 0.0)          # constant velocity AB
        
        # Derived quantities for checks
        self.abs_a = abs(self.a) if self.a < 0 else abs(self.a)
        
    def _safe_get(self, key: str, results: dict, default: float) -> float:
        """Safely get value from results or attributes."""
        # Try results dict first
        val = results.get(key, default)
        if val is None:
            val = default
        # Convert sympy to float if needed
        if hasattr(val, 'evalf'):
            val = float(val.evalf())
        return float(val)
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Test fundamental kinematic invariants."""
        logs = []
        success = True
        
        # 1. Check acceleration magnitude from step1 formula
        expected_a_mag = 2 * self.dist_last_2_sec / (2**2 - 1**2)  # = 3 m/s²
        if not math.isclose(self.abs_a, expected_a_mag, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: |a| = {self.abs_a}, expected {expected_a_mag}")
            success = False
        else:
            logs.append(f"PASS: |a| = {self.abs_a} ≈ {expected_a_mag}")
        
        # 2. Check acceleration sign (should be negative for deceleration)
        if self.a >= 0:
            logs.append("WARN: Acceleration a should be negative (deceleration)")
            # Not fatal, but note
        
        # 3. Check total time consistency
        if not math.isclose(self.t, self.total_time, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Total time t = {self.t}, expected {self.total_time}")
            success = False
        else:
            logs.append(f"PASS: Total time t = {self.t}")
        
        # 4. Check t1 + t2 = total time
        t_sum = self.t1 + self.t2
        if not math.isclose(t_sum, self.total_time, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: t1 + t2 = {t_sum}, expected {self.total_time}")
            success = False
        else:
            logs.append(f"PASS: t1 + t2 = {t_sum} ≈ {self.total_time}")
        
        # 5. Check total distance
        if not math.isclose(self.x, self.total_distance, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Total distance x = {self.x}, expected {self.total_distance}")
            success = False
        else:
            logs.append(f"PASS: Total distance x = {self.x}")
        
        # 6. Check x' (BC distance) from step6
        if not math.isclose(self.x_prime, 6.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: BC distance x' = {self.x_prime}, expected 6.0")
            success = False
        else:
            logs.append(f"PASS: BC distance x' = {self.x_prime} ≈ 6.0")
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Cross-check intermediate results with kinematic equations."""
        logs = []
        success = True
        
        # Symbolic setup
        t2, a, v = sp.symbols('t2 a v', real=True)
        
        # 1. Check v = |a| * t2 (from step3, since stops at C: 0 = v + a*t2)
        expected_v = self.abs_a * self.t2
        if not math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: v = {self.v}, expected |a|*t2 = {expected_v}")
            success = False
        else:
            logs.append(f"PASS: v = {self.v} ≈ |a|*t2 = {expected_v}")
        
        # 2. Check total distance formula: x = v*t1 + 0.5*|a|*t2² (step4)
        calc_distance = self.v * self.t1 + 0.5 * self.abs_a * self.t2**2
        if not math.isclose(calc_distance, self.total_distance, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: v*t1 + 0.5|a|t2² = {calc_distance}, expected {self.total_distance}")
            success = False
        else:
            logs.append(f"PASS: v*t1 + 0.5|a|t2² = {calc_distance} ≈ {self.total_distance}")
        
        # 3. Check x' = 0.5*|a|*t2² (step5)
        calc_x_prime = 0.5 * self.abs_a * self.t2**2
        if not math.isclose(self.x_prime, calc_x_prime, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: x' = {self.x_prime}, calculated 0.5|a|t2² = {calc_x_prime}")
            success = False
        else:
            logs.append(f"PASS: x' = 0.5|a|t2² = {calc_x_prime}")
        
        # 4. Verify the 2nd-second-before-stopping condition
        # Distance in interval [t2-2, t2-1] should be 4.5 m
        # Using kinematic equation: Δx = v0*t + 0.5*a*t² where t=1, v0 = velocity at (t2-2)
        # Velocity at (t2-2): v_t2_minus_2 = v + a*(t2-2)
        # But v = -a*t2 (since stops at t2), so v_t2_minus_2 = -a*t2 + a*(t2-2) = -2a
        # Then Δx = (-2a)*1 + 0.5*a*1² = -2a + 0.5a = -1.5a
        # Since a is negative, -1.5a is positive
        calc_dist_2nd_last = -1.5 * self.a  # a is negative, so this is positive
        if not math.isclose(calc_dist_2nd_last, self.dist_last_2_sec, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: 2nd-last second distance = {calc_dist_2nd_last}, expected {self.dist_last_2_sec}")
            success = False
        else:
            logs.append(f"PASS: 2nd-last second distance = {calc_dist_2nd_last} ≈ {self.dist_last_2_sec}")
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Perturb input (distance in 2nd second before stopping) and check trends.
        Restores original values after test.
        """
        logs = []
        success = True
        
        # Store original values
        original_a = self.a
        original_x_prime = self.x_prime
        original_t1 = self.t1
        original_t2 = self.t2
        original_v = self.v
        
        try:
            # Perturb the input: increase distance in 2nd second by 10%
            perturbed_dist = self.dist_last_2_sec * 1.1
            
            # Recalculate acceleration magnitude from step1 formula
            perturbed_a_mag = 2 * perturbed_dist / (2**2 - 1**2)  # = 2*perturbed_dist/3
            perturbed_a = -perturbed_a_mag  # negative for deceleration
            
            # Solve for t2 using system of equations:
            # 1) t1 + t2 = 11
            # 2) v = |a|*t2
            # 3) 60 = v*t1 + 0.5*|a|*t2²
            # Substitute (1) and (2) into (3):
            # 60 = |a|*t2*(11 - t2) + 0.5*|a|*t2²
            # 60 = |a|*t2*(11 - t2 + 0.5*t2)
            # 60 = |a|*t2*(11 - 0.5*t2)
            
            # Solve quadratic: |a|*t2*(11 - 0.5*t2) = 60
            # 0.5*|a|*t2² - 11*|a|*t2 + 60 = 0
            a = perturbed_a_mag
            coeffs = [0.5*a, -11*a, 60]
            
            # Solve quadratic: 0.5a*t2² - 11a*t2 + 60 = 0
            discriminant = (-11*a)**2 - 4*(0.5*a)*60
            if discriminant < 0:
                logs.append("FAIL: No real solution for perturbed system")
                success = False
                return success, logs
            
            sqrt_disc = math.sqrt(discriminant)
            t2_solutions = [
                (11*a - sqrt_disc) / (2 * 0.5 * a),
                (11*a + sqrt_disc) / (2 * 0.5 * a)
            ]
            
            # Select solution where 0 < t2 < 11
            perturbed_t2 = None
            for sol in t2_solutions:
                if 0 < sol < 11:
                    perturbed_t2 = sol
                    break
            
            if perturbed_t2 is None:
                logs.append("FAIL: No valid t2 in (0,11) for perturbed system")
                success = False
                return success, logs
            
            perturbed_t1 = 11 - perturbed_t2
            perturbed_v = perturbed_a_mag * perturbed_t2
            perturbed_x_prime = 0.5 * perturbed_a_mag * perturbed_t2**2
            
            # Check trends:
            # 1. More distance in 2nd second -> larger |a| (more negative a)
            if not (abs(perturbed_a) > abs(original_a)):
                logs.append(f"FAIL: |a| should increase with larger 2nd-second distance")
                logs.append(f"  Original |a| = {abs(original_a)}, Perturbed |a| = {abs(perturbed_a)}")
                success = False
            else:
                logs.append(f"PASS: |a| increases: {abs(original_a)} → {abs(perturbed_a)}")
            
            # 2. With larger |a|, deceleration phase distance x' should decrease
            # (faster deceleration means shorter stopping distance for same initial velocity)
            if not (perturbed_x_prime < original_x_prime):
                logs.append(f"FAIL: x' should decrease with larger |a|")
                logs.append(f"  Original x' = {original_x_prime}, Perturbed x' = {perturbed_x_prime}")
                success = False
            else:
                logs.append(f"PASS: x' decreases: {original_x_prime} → {perturbed_x_prime}")
            
            # 3. With larger |a|, t2 should decrease (stops faster)
            if not (perturbed_t2 < original_t2):
                logs.append(f"WARN: t2 expected to decrease but: {original_t2} → {perturbed_t2}")
                # Not necessarily failure due to coupled equations
            else:
                logs.append(f"INFO: t2 decreases: {original_t2} → {perturbed_t2}")
            
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {str(e)}")
            success = False
        
        finally:
            # Restore original values
            self.a = original_a
            self.x_prime = original_x_prime
            self.t1 = original_t1
            self.t2 = original_t2
            self.v = original_v
            self.abs_a = abs(original_a)
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success and logs."""
        all_logs = ["=== Physics Checker ==="]
        overall_success = True
        
        # Test 1: Invariants
        success1, logs1 = self.test_invariants()
        all_logs.append("\n1. Invariant Tests:")
        all_logs.extend(logs1)
        overall_success = overall_success and success1
        
        # Test 2: Trace Consistency
        success2, logs2 = self.test_trace_consistency()
        all_logs.append("\n2. Trace Consistency Tests:")
        all_logs.extend(logs2)
        overall_success = overall_success and success2
        
        # Test 3: Monotonicity
        success3, logs3 = self.test_monotonicity()
        all_logs.append("\n3. Monotonicity Tests:")
        all_logs.extend(logs3)
        overall_success = overall_success and success3
        
        # Summary
        all_logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        
        return overall_success, all_logs

# Example usage (commented out for the sandbox):
# if __name__ == "__main__":
#     # Mock solution class for testing
#     class MockSolution:
#         def __init__(self):
#             self.results = {
#                 'a': -3.0,
#                 "x'": 6.0,
#                 'x': 60.0,
#                 't₁': 9.0,
#                 't₂': 2.0,
#                 't': 11.0,
#                 'v': 6.0
#             }
#     
#     mock_sol = MockSolution()
#     checker = PhysicsChecker(mock_sol)
#     success, logs = checker.run_all_tests()
#     for log in logs:
#         print(log)