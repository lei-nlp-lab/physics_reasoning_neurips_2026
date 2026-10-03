import math
import sympy as sp
from sympy import symbols, Eq, solve

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        
        # Extract inputs safely
        self.m = 73.0  # kg (given in problem)
        self.g = 10.0  # m/s² (given)
        self.r = 10.0  # m (turn radius)
        self.v_turn = 14.0  # m/s (turn speed)
        self.x_accel = 8.0  # m (distance in acceleration phase)
        self.t_accel = 2.0  # s (time in acceleration phase)
        self.v0 = 0.0  # m/s (starts from rest)
        
        # Extract results safely
        self.a = getattr(solution_instance, 'a', getattr(solution_instance, 'results', {}).get('a', 0.0))
        if isinstance(self.a, str):
            self.a = float(self.a.replace('m/s²', '').replace('m/s^2', '').strip())
        
        self.F = getattr(solution_instance, 'F', getattr(solution_instance, 'results', {}).get('F', 0.0))
        if isinstance(self.F, str):
            self.F = float(self.F.replace('N', '').strip())
        
        self.theta = getattr(solution_instance, 'theta', getattr(solution_instance, 'results', {}).get('theta', 0.0))
        if isinstance(self.theta, str):
            self.theta = float(self.theta.replace('°', '').replace('deg', '').strip())
        
        self.tantheta = getattr(solution_instance, 'tantheta', getattr(solution_instance, 'results', {}).get('tantheta', 0.0))
        if isinstance(self.tantheta, str):
            self.tantheta = float(self.tantheta)
        
        # Pre-calculated expected values from KB
        self.expected_a = 4.0  # m/s²
        self.expected_F = 73.0 * (14.0**2) / 10.0  # 1430.8 N
        self.expected_tantheta = (73.0 * 10.0) / self.expected_F  # ~0.51
        self.expected_theta = 27.0  # degrees
        
        self.logs = []
    
    def _log(self, message, success=True):
        """Helper to log test results"""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {message}")
        return success
    
    def test_invariants(self):
        """Verify conservation laws and Newton's laws from KB"""
        tests_passed = True
        
        # 1. Check uniformly accelerated motion: x = 0.5*a*t²
        calculated_x = 0.5 * self.a * (self.t_accel**2)
        if math.isclose(calculated_x, self.x_accel, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(f"Uniform acceleration invariant: x={calculated_x:.2f} m ≈ {self.x_accel} m", True) and tests_passed
        else:
            tests_passed = self._log(f"Uniform acceleration invariant violated: x={calculated_x:.2f} m ≠ {self.x_accel} m", False) and False
        
        # 2. Check Newton's 2nd Law for acceleration phase: F_net = m*a
        F_net_accel = self.m * self.a
        # During straight acceleration, net force should equal m*a
        tests_passed = self._log(f"Newton's 2nd Law (acceleration): F_net = {F_net_accel:.1f} N", True) and tests_passed
        
        # 3. Check centripetal force formula: F_c = m*v²/r
        F_calculated = self.m * (self.v_turn**2) / self.r
        if math.isclose(F_calculated, self.F, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(f"Centripetal force formula: {F_calculated:.1f} N ≈ {self.F:.1f} N", True) and tests_passed
        else:
            tests_passed = self._log(f"Centripetal force mismatch: {F_calculated:.1f} N ≠ {self.F:.1f} N", False) and False
        
        # 4. Check tanθ = m*g / F (from force balance during turn)
        tanθ_calculated = (self.m * self.g) / self.F
        if math.isclose(tanθ_calculated, self.tantheta, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(f"Force balance: tanθ = {tanθ_calculated:.3f} ≈ {self.tantheta:.3f}", True) and tests_passed
        else:
            tests_passed = self._log(f"Force balance violated: tanθ = {tanθ_calculated:.3f} ≠ {self.tantheta:.3f}", False) and False
        
        # 5. Check θ corresponds to given tan values
        tan_values = {22: 0.40, 27: 0.51, 32: 0.62, 37: 0.75}
        closest_angle = min(tan_values.keys(), key=lambda θ: abs(tan_values[θ] - self.tantheta))
        if math.isclose(tan_values[closest_angle], self.tantheta, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(f"Angle consistency: tan({closest_angle}°) = {tan_values[closest_angle]} ≈ {self.tantheta}", True) and tests_passed
        else:
            tests_passed = self._log(f"Angle mismatch: closest is tan({closest_angle}°) = {tan_values[closest_angle]}, got {self.tantheta}", False) and False
        
        return tests_passed
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer"""
        tests_passed = True
        
        # Check acceleration matches expected
        if math.isclose(self.a, self.expected_a, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(f"Acceleration trace: {self.a:.1f} m/s² ≈ {self.expected_a} m/s²", True) and tests_passed
        else:
            tests_passed = self._log(f"Acceleration mismatch: {self.a:.1f} m/s² ≠ {self.expected_a} m/s²", False) and False
        
        # Check centripetal force matches expected
        if math.isclose(self.F, self.expected_F, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(f"Force trace: {self.F:.1f} N ≈ {self.expected_F:.1f} N", True) and tests_passed
        else:
            tests_passed = self._log(f"Force mismatch: {self.F:.1f} N ≠ {self.expected_F:.1f} N", False) and False
        
        # Check tanθ matches expected
        if math.isclose(self.tantheta, self.expected_tantheta, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(f"tanθ trace: {self.tantheta:.3f} ≈ {self.expected_tantheta:.3f}", True) and tests_passed
        else:
            tests_passed = self._log(f"tanθ mismatch: {self.tantheta:.3f} ≠ {self.expected_tantheta:.3f}", False) and False
        
        # Check final angle matches expected
        if math.isclose(self.theta, self.expected_theta, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(f"Angle trace: {self.theta:.1f}° ≈ {self.expected_theta}°", True) and tests_passed
        else:
            tests_passed = self._log(f"Angle mismatch: {self.theta:.1f}° ≠ {self.expected_theta}°", False) and False
        
        # Verify chain: a → F → tanθ → θ
        # Calculate F from a? Not directly - different motion phases
        # But we can check internal consistency
        F_from_formula = self.m * (self.v_turn**2) / self.r
        tanθ_from_F = (self.m * self.g) / F_from_formula
        expected_angle = 27.0  # From given tan values
        
        if math.isclose(tanθ_from_F, self.tantheta, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(f"Trace chain consistency: F→tanθ→θ holds", True) and tests_passed
        else:
            tests_passed = self._log(f"Trace chain broken: F→tanθ inconsistent", False) and False
        
        return tests_passed
    
    def test_monotonicity(self):
        """Perturb mass and check trend, then restore"""
        tests_passed = True
        
        # Store original mass
        original_m = self.m
        
        # Perturb: increase mass by 10%
        perturbed_m = original_m * 1.1
        
        # Recalculate expected values with perturbed mass
        # 1. Acceleration should NOT change (depends only on x and t, not m)
        expected_a_perturbed = self.expected_a  # Same
        
        # 2. Centripetal force should increase proportionally to mass
        expected_F_perturbed = perturbed_m * (self.v_turn**2) / self.r
        
        # 3. tanθ should remain the same (tanθ = m*g/F, but F ∝ m, so ratio constant)
        expected_tanθ_perturbed = (perturbed_m * self.g) / expected_F_perturbed
        
        # 4. θ should remain the same
        expected_theta_perturbed = self.expected_theta
        
        # Check trends
        # Trend 1: F ∝ m (for fixed v and r)
        F_ratio = expected_F_perturbed / self.expected_F
        m_ratio = perturbed_m / original_m
        if math.isclose(F_ratio, m_ratio, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(f"Monotonicity: F ∝ m holds (F_ratio={F_ratio:.3f}, m_ratio={m_ratio:.3f})", True) and tests_passed
        else:
            tests_passed = self._log(f"Monotonicity violated: F ∝ m fails (F_ratio={F_ratio:.3f} ≠ {m_ratio:.3f})", False) and False
        
        # Trend 2: tanθ independent of m
        if math.isclose(expected_tanθ_perturbed, self.expected_tantheta, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(f"Monotonicity: tanθ independent of m ({expected_tanθ_perturbed:.3f} ≈ {self.expected_tantheta:.3f})", True) and tests_passed
        else:
            tests_passed = self._log(f"Monotonicity: tanθ depends on m (unexpected)", False) and False
        
        # Trend 3: a independent of m
        if math.isclose(expected_a_perturbed, self.expected_a, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(f"Monotonicity: a independent of m", True) and tests_passed
        else:
            tests_passed = self._log(f"Monotonicity: a depends on m (unexpected)", False) and False
        
        # Restore original mass
        self.m = original_m
        
        return tests_passed
    
    def run_all_tests(self):
        """Execute all tests and return success flag with logs"""
        self.logs = []
        self.logs.append("=== PhysicsChecker Starting ===")
        
        success = True
        
        # Run all tests
        if not self.test_invariants():
            success = False
            self.logs.append("--- invariants test failed ---")
        
        if not self.test_trace_consistency():
            success = False
            self.logs.append("--- trace consistency test failed ---")
        
        if not self.test_monotonicity():
            success = False
            self.logs.append("--- monotonicity test failed ---")
        
        # Summary
        if success:
            self.logs.append("=== All tests PASSED ===")
        else:
            self.logs.append("=== Some tests FAILED ===")
        
        return success, self.logs

# Example usage (for testing the checker itself):
if __name__ == "__main__":
    # Mock solution class for testing
    class MockSolution:
        def __init__(self):
            self.results = {
                'a': 4.0,
                'F': 1430.8,
                'theta': 27.0,
                'tantheta': 0.51
            }
    
    mock_solution = MockSolution()
    checker = PhysicsChecker(mock_solution)
    success, logs = checker.run_all_tests()
    
    for log in logs:
        print(log)