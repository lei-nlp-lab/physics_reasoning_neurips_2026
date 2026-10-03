import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction for inputs
        self.m1 = getattr(solution_instance, 'm1', 1.0)  # kg
        self.m2 = getattr(solution_instance, 'm2', 4.0)  # kg
        self.k = getattr(solution_instance, 'k', 20.0)   # N/m
        self.v0 = getattr(solution_instance, 'v0', 1.25) # m/s (5/4)
        self.μ = getattr(solution_instance, 'μ', 0.1)
        self.g = getattr(solution_instance, 'g', 10.0)   # m/s²
        
        # Safe extraction for results using the key mapping
        results = getattr(solution_instance, 'results', {})
        self.v1 = results.get('v_1', 0.0)
        self.a1 = results.get('a_1', 0.0)
        self.x1 = results.get('x_1', 0.0)
        self.a_all = results.get('a_all', 0.0)
        self.a2 = results.get('a_2', 0.0)
        self.x2 = results.get('x_2', 0.0)
        self.v2 = results.get('v_2', 0.0)
        self.v3 = results.get('v_3', 0.0)
        self.deltaU = results.get('DeltaU', 0.0)
        
        self.logs = []
        
    def log(self, message, success):
        """Helper to log test results"""
        self.logs.append(f"{'✓' if success else '✗'} {message}")
        return success
        
    def test_invariants(self):
        """Verify fundamental physics laws"""
        try:
            successes = []
            
            # 1. Conservation of Momentum (Step 1)
            p_initial = self.m2 * self.v0
            p_final = (self.m1 + self.m2) * self.v1
            momentum_ok = math.isclose(p_initial, p_final, rel_tol=1e-2, abs_tol=1e-2)
            successes.append(self.log(
                f"Momentum conservation: {p_initial:.3f} = {p_final:.3f}", 
                momentum_ok
            ))
            
            # 2. Newton's Second Law for board (Step 3)
            expected_a1 = self.μ * self.m2 * self.g / self.m1
            a1_ok = math.isclose(self.a1, expected_a1, rel_tol=1e-2, abs_tol=1e-2)
            successes.append(self.log(
                f"Board acceleration: {self.a1:.3f} = μ·m₂·g/m₁ = {expected_a1:.3f}", 
                a1_ok
            ))
            
            # 3. Kinematic equation (Step 4)
            expected_x1 = self.v1**2 / (2 * self.a1) if self.a1 != 0 else 0
            x1_ok = math.isclose(self.x1, expected_x1, rel_tol=1e-2, abs_tol=1e-2)
            successes.append(self.log(
                f"Distance to spring: {self.x1:.3f} = v₁²/(2a₁) = {expected_x1:.3f}", 
                x1_ok
            ))
            
            # 4. Hooke's Law & Newton's Second Law (Step 5)
            # When x = x2, system acceleration should satisfy k*x2 = (m1+m2)*a_all
            if self.x2 != 0 and self.a_all != 0:
                lhs = self.k * self.x2
                rhs = (self.m1 + self.m2) * self.a_all
                hooke_newton_ok = math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2)
                successes.append(self.log(
                    f"Hooke+Newton: k·x₂ = {lhs:.3f}, (m₁+m₂)·a_all = {rhs:.3f}", 
                    hooke_newton_ok
                ))
            
            # 5. Block acceleration (Step 6)
            expected_a2 = self.μ * self.g
            a2_ok = math.isclose(self.a2, expected_a2, rel_tol=1e-2, abs_tol=1e-2)
            successes.append(self.log(
                f"Block acceleration: {self.a2:.3f} = μ·g = {expected_a2:.3f}", 
                a2_ok
            ))
            
            # 6. Work-Energy Theorem (Step 8)
            left_side = -0.5 * self.k * self.x2**2
            right_side = 0.5 * (self.m1 + self.m2) * (self.v2**2 - self.v1**2)
            work_energy_ok = math.isclose(left_side, right_side, rel_tol=1e-2, abs_tol=1e-2)
            successes.append(self.log(
                f"Work-Energy: -½k·x₂² = {left_side:.3f}, ΔKE = {right_side:.3f}", 
                work_energy_ok
            ))
            
            return all(successes)
            
        except Exception as e:
            self.logs.append(f"✗ Invariant test failed: {str(e)}")
            return False
            
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        try:
            successes = []
            
            # Check if v1 makes sense from momentum
            expected_v1 = (self.m2 * self.v0) / (self.m1 + self.m2)
            v1_ok = math.isclose(self.v1, expected_v1, rel_tol=1e-2, abs_tol=1e-2)
            successes.append(self.log(
                f"v₁ consistency: {self.v1:.3f} ≈ {expected_v1:.3f}", 
                v1_ok
            ))
            
            # Check x2 from force balance when block is about to slide
            # Condition: μ·m₂·g = m₂·a_all, so a_all = μ·g
            expected_x2 = (self.m1 + self.m2) * self.μ * self.g / self.k
            x2_ok = math.isclose(self.x2, expected_x2, rel_tol=1e-2, abs_tol=1e-2)
            successes.append(self.log(
                f"x₂ consistency: {self.x2:.3f} ≈ (m₁+m₂)·μ·g/k = {expected_x2:.3f}", 
                x2_ok
            ))
            
            # Check v2 from energy conservation
            # ½k·x₂² = ½(m₁+m₂)(v₁² - v₂²) => v₂² = v₁² - k·x₂²/(m₁+m₂)
            if self.x2 != 0:
                expected_v2_sq = self.v1**2 - (self.k * self.x2**2) / (self.m1 + self.m2)
                expected_v2 = math.sqrt(max(0, expected_v2_sq))
                v2_ok = math.isclose(self.v2, expected_v2, rel_tol=1e-2, abs_tol=1e-2)
                successes.append(self.log(
                    f"v₂ consistency: {self.v2:.3f} ≈ √(v₁² - k·x₂²/(m₁+m₂)) = {expected_v2:.3f}", 
                    v2_ok
                ))
            
            return all(successes)
            
        except Exception as e:
            self.logs.append(f"✗ Trace consistency failed: {str(e)}")
            return False
            
    def test_monotonicity(self):
        """Check that computed values are physically reasonable"""
        try:
            successes = []
            
            # Check that v1 is between 0 and v0
            v1_range_ok = (0 <= self.v1 <= self.v0)
            successes.append(self.log(
                f"v₁ range: 0 ≤ {self.v1:.3f} ≤ {self.v0:.3f}", 
                v1_range_ok
            ))
            
            # Check that a1 is positive
            a1_positive = self.a1 > 0
            successes.append(self.log(
                f"a₁ positive: {self.a1:.3f} > 0", 
                a1_positive
            ))
            
            # Check that x1 is positive
            x1_positive = self.x1 > 0
            successes.append(self.log(
                f"x₁ positive: {self.x1:.3f} > 0", 
                x1_positive
            ))
            
            # Check that v2 is less than v1 (spring slows system)
            v2_less_than_v1 = self.v2 < self.v1
            successes.append(self.log(
                f"v₂ < v₁: {self.v2:.3f} < {self.v1:.3f}", 
                v2_less_than_v1
            ))
            
            # Check that x2 is positive
            x2_positive = self.x2 > 0
            successes.append(self.log(
                f"x₂ positive: {self.x2:.3f} > 0", 
                x2_positive
            ))
            
            return all(successes)
            
        except Exception as e:
            self.logs.append(f"✗ Monotonicity test failed: {str(e)}")
            return False
            
    def run_all_tests(self):
        """Execute all tests and return summary"""
        self.logs = []  # Reset logs
        
        try:
            # Run tests
            inv_ok = self.test_invariants()
            trace_ok = self.test_trace_consistency()
            mono_ok = self.test_monotonicity()
            
            success = inv_ok and trace_ok and mono_ok
            
            # Add summary
            self.logs.append("\n" + "="*50)
            self.logs.append(f"SUMMARY: {'ALL TESTS PASSED' if success else 'SOME TESTS FAILED'}")
            self.logs.append(f"  Invariants: {'PASS' if inv_ok else 'FAIL'}")
            self.logs.append(f"  Trace Consistency: {'PASS' if trace_ok else 'FAIL'}")
            self.logs.append(f"  Monotonicity: {'PASS' if mono_ok else 'FAIL'}")
            
            return success, self.logs
            
        except Exception as e:
            error_msg = f"Runtime error: {str(e)}\n{traceback.format_exc()}"
            self.logs.append(error_msg)
            return False, self.logs