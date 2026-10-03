import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs and results
        self.solution = solution_instance
        
        # Extract inputs with safe defaults
        self.L = getattr(solution_instance, 'L', 11.0)
        self.h = getattr(solution_instance, 'h', 3.3)
        self.m = getattr(solution_instance, 'm', 0.1)
        self.M = getattr(solution_instance, 'M', 0.2)
        self.v_B0 = getattr(solution_instance, 'v_B0', 10.0)
        self.mu = getattr(solution_instance, 'mu', math.sqrt(7/13))
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Extract results from solution_instance.results with safe fallbacks
        results = getattr(solution_instance, 'results', {})
        
        # Essential results with fallbacks
        self.sintheta = getattr(solution_instance, 'sintheta', results.get('sintheta', 0.3))
        self.a_1 = getattr(solution_instance, 'a_1', results.get('a_1', 3.0))
        self.a_2 = getattr(solution_instance, 'a_2', results.get('a_2', 10.0))
        self.t_B = getattr(solution_instance, 't_B', results.get('t_B', 1.0))
        self.x_B = getattr(solution_instance, 'x_B', results.get('x_B', 5.0))
        self.t_A = getattr(solution_instance, 't_A', results.get('t_A', 2.0))
        self.x_0 = getattr(solution_instance, 'x_0', results.get('x_0', 6.0))
        self.v_0 = getattr(solution_instance, 'v_0', results.get('v_0', 6.0))
        self.v_A1 = getattr(solution_instance, 'v_A1', results.get('v_A1', -2.0))
        self.v_B1 = getattr(solution_instance, 'v_B1', results.get('v_B1', 4.0))
        self.v_A1_magnitude = abs(getattr(solution_instance, 'v_A1_magnitude', results.get('v_A1_magnitude', 2.0)))
        self.v_B1_magnitude = abs(getattr(solution_instance, 'v_B1_magnitude', results.get('v_B1_magnitude', 4.0)))
        self.x_1 = getattr(solution_instance, 'x_1', results.get('x_1', 4.0))
        self.x_2 = getattr(solution_instance, 'x_2', results.get('x_2', 4.0*(4/9)))
        self.x_3 = getattr(solution_instance, 'x_3', results.get('x_3', 4.0*(4/9)**2))
        self.x_4 = getattr(solution_instance, 'x_4', results.get('x_4', 4.0*(4/9)**3))
        self.x_total = getattr(solution_instance, 'x', results.get('x', 7.2))
        self.v_A20 = getattr(solution_instance, 'v_A20', results.get('v_A20', 4.0))
        self.v_B2 = getattr(solution_instance, 'v_B2', results.get('v_B2', (2/3)*4.0))
        self.v_B3 = getattr(solution_instance, 'v_B3', results.get('v_B3', (2/3)*(2/3)*4.0))
        self.v_B4 = getattr(solution_instance, 'v_B4', results.get('v_B4', (2/3)**3 * 4.0))
        
        # Derived quantities
        self.costheta = math.sqrt(1 - self.sintheta**2)
        
        self.logs = []
        
    def log(self, message, success=True):
        """Helper to log messages"""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"[{status}] {message}")
        return success
        
    def test_invariants(self):
        """Test fundamental physics invariants"""
        try:
            # 1. Check trigonometric consistency
            expected_sintheta = self.h / self.L
            if not math.isclose(self.sintheta, expected_sintheta, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"sinθ mismatch: got {self.sintheta}, expected {expected_sintheta}", False)
                return False
            self.log(f"Trigonometric check: sinθ = h/L = {self.sintheta}")
            
            # 2. Check Newton's Second Law for block A
            # a_1 should be g sinθ
            expected_a1 = self.g * self.sintheta
            if not math.isclose(self.a_1, expected_a1, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Block A acceleration mismatch: got {self.a_1}, expected {expected_a1}", False)
                return False
            self.log(f"Newton's 2nd Law for block A: a_1 = g sinθ = {self.a_1}")
            
            # 3. Check Newton's Second Law for block B
            # Force equation: μMg cosθ + Mg sinθ = M a_2
            expected_a2 = self.g * (self.mu * self.costheta + self.sintheta)
            if not math.isclose(self.a_2, expected_a2, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Block B acceleration mismatch: got {self.a_2}, expected {expected_a2}", False)
                return False
            self.log(f"Newton's 2nd Law for block B: a_2 = g(μ cosθ + sinθ) = {self.a_2}")
            
            # 4. Check kinematics for block B's initial motion
            # t_B = v_B0 / a_2
            expected_tB = self.v_B0 / self.a_2
            if not math.isclose(self.t_B, expected_tB, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Time for B to stop mismatch: got {self.t_B}, expected {expected_tB}", False)
                return False
            self.log(f"Kinematics: t_B = v_B0 / a_2 = {self.t_B}")
            
            # x_B = v_B0^2 / (2 * a_2)
            expected_xB = self.v_B0**2 / (2 * self.a_2)
            if not math.isclose(self.x_B, expected_xB, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Displacement x_B mismatch: got {self.x_B}, expected {expected_xB}", False)
                return False
            self.log(f"Kinematics: x_B = v_B0^2 / (2a_2) = {self.x_B}")
            
            # 5. Check elastic collision conservation laws
            # Momentum: m*v_0 = m*v_A1 + M*v_B1
            momentum_before = self.m * self.v_0
            momentum_after = self.m * self.v_A1 + self.M * self.v_B1
            if not math.isclose(momentum_before, momentum_after, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Momentum not conserved in collision: before={momentum_before}, after={momentum_after}", False)
                return False
            self.log(f"Momentum conserved: {momentum_before:.3f} = {momentum_after:.3f}")
            
            # Energy: 0.5*m*v_0^2 = 0.5*m*v_A1^2 + 0.5*M*v_B1^2
            energy_before = 0.5 * self.m * self.v_0**2
            energy_after = 0.5 * self.m * self.v_A1**2 + 0.5 * self.M * self.v_B1**2
            if not math.isclose(energy_before, energy_after, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Energy not conserved in collision: before={energy_before}, after={energy_after}", False)
                return False
            self.log(f"Energy conserved: {energy_before:.3f} = {energy_after:.3f}")
            
            # 6. Check the pattern from elastic collision formulas
            # For elastic collision with m and M where M = 2m:
            # v_A1 = (m - M)/(m + M) * v_0 = (0.1 - 0.2)/0.3 * v_0 = -1/3 * v_0
            expected_vA1 = -self.v_0 / 3
            if not math.isclose(self.v_A1, expected_vA1, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"v_A1 mismatch: got {self.v_A1}, expected {expected_vA1}", False)
                return False
            self.log(f"Elastic collision formula for v_A1: v_A1 = -v_0/3 = {self.v_A1}")
            
            # v_B1 = (2m)/(m + M) * v_0 = (0.2)/0.3 * v_0 = 2/3 * v_0
            expected_vB1 = 2 * self.v_0 / 3
            if not math.isclose(self.v_B1, expected_vB1, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"v_B1 mismatch: got {self.v_B1}, expected {expected_vB1}", False)
                return False
            self.log(f"Elastic collision formula for v_B1: v_B1 = 2v_0/3 = {self.v_B1}")
            
            # 7. Check magnitudes match signs
            if not math.isclose(abs(self.v_A1), self.v_A1_magnitude, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"v_A1 magnitude mismatch: abs({self.v_A1}) ≠ {self.v_A1_magnitude}", False)
                return False
            if not math.isclose(abs(self.v_B1), self.v_B1_magnitude, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"v_B1 magnitude mismatch: abs({self.v_B1}) ≠ {self.v_B1_magnitude}", False)
                return False
            self.log(f"Velocity magnitudes consistent: |v_A1|={self.v_A1_magnitude}, |v_B1|={self.v_B1_magnitude}")
            
            # 8. Check geometric series consistency for subsequent collisions
            # x_2 should be (4/9) * x_1
            expected_x2 = (4/9) * self.x_1
            if not math.isclose(self.x_2, expected_x2, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"x_2 mismatch: got {self.x_2}, expected {expected_x2}", False)
                return False
            self.log(f"Geometric pattern: x_2 = (4/9) * x_1 = {self.x_2}")
            
            # 9. Check total displacement from geometric series
            # x_total = x_1 + x_2 + x_3 + ... = x_1 / (1 - 4/9) = x_1 / (5/9) = (9/5) * x_1
            expected_total = self.x_1 / (1 - 4/9)
            if not math.isclose(self.x_total, expected_total, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Total displacement mismatch: got {self.x_total}, expected {expected_total}", False)
                return False
            self.log(f"Geometric series sum: x_total = x_1 / (1 - 4/9) = {self.x_total}")
            
            return True
            
        except Exception as e:
            self.log(f"Error in test_invariants: {str(e)}\n{traceback.format_exc()}", False)
            return False
    
    def test_trace_consistency(self):
        """Test consistency between intermediate results"""
        try:
            # 1. Check collision point consistency
            # x_0 should be 0.5 * a_1 * t_A^2
            expected_x0 = 0.5 * self.a_1 * self.t_A**2
            if not math.isclose(self.x_0, expected_x0, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Collision point x_0 mismatch: got {self.x_0}, expected {expected_x0}", False)
                return False
            self.log(f"Collision point consistent: x_0 = 0.5*a_1*t_A² = {self.x_0}")
            
            # 2. Check velocity of A at collision
            # v_0 should be a_1 * t_A (not g*t_A as misstated in step_8)
            expected_v0 = self.a_1 * self.t_A
            if not math.isclose(self.v_0, expected_v0, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"v_0 mismatch: got {self.v_0}, expected {expected_v0}", False)
                return False
            self.log(f"Velocity at collision: v_0 = a_1*t_A = {self.v_0}")
            
            # 3. Check that after collision, both blocks travel until next collision
            # The distance x_1 should satisfy v_B1^2 = 2*a_2*x_1
            expected_x1_from_B = self.v_B1**2 / (2 * self.a_2)
            if not math.isclose(self.x_1, expected_x1_from_B, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"x_1 from v_B1^2=2*a_2*x_1 mismatch: got {self.x_1}, expected {expected_x1_from_B}", False)
                return False
            self.log(f"First post-collision displacement consistent: v_B1²=2*a_2*x_1 gives x_1={self.x_1}")
            
            # 4. Check velocity of A before second collision (v_A20)
            # Should equal v_B1 as stated in step_15
            if not math.isclose(self.v_A20, self.v_B1, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"v_A20 mismatch: got {self.v_A20}, expected {self.v_B1}", False)
                return False
            self.log(f"v_A20 = v_B1 = {self.v_A20}")
            
            # 5. Check velocity of B after second collision
            # v_B2 = (2/3) * v_A20 = (2/3) * v_B1
            expected_vB2 = (2/3) * self.v_B1
            if not math.isclose(self.v_B2, expected_vB2, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"v_B2 mismatch: got {self.v_B2}, expected {expected_vB2}", False)
                return False
            self.log(f"Second collision: v_B2 = (2/3)*v_A20 = {self.v_B2}")
            
            # 6. Check displacement x_2 from v_B2
            expected_x2_from_vB2 = self.v_B2**2 / (2 * self.a_2)
            if not math.isclose(self.x_2, expected_x2_from_vB2, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"x_2 mismatch: got {self.x_2}, expected {expected_x2_from_vB2}", False)
                return False
            self.log(f"x_2 from kinematics: v_B2²=2*a_2*x_2 gives {self.x_2}")
            
            # 7. Check the pattern for subsequent velocities and displacements
            # v_B3 = (2/3)^2 * v_B1
            expected_vB3 = (2/3)**2 * self.v_B1
            if not math.isclose(self.v_B3, expected_vB3, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"v_B3 mismatch: got {self.v_B3}, expected {expected_vB3}", False)
                return False
            self.log(f"Third collision: v_B3 = (2/3)²*v_B1 = {self.v_B3}")
            
            # v_B4 = (2/3)^3 * v_B1
            expected_vB4 = (2/3)**3 * self.v_B1
            if not math.isclose(self.v_B4, expected_vB4, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"v_B4 mismatch: got {self.v_B4}, expected {expected_vB4}", False)
                return False
            self.log(f"Fourth collision: v_B4 = (2/3)³*v_B1 = {self.v_B4}")
            
            # 8. Check x_3 from v_B3
            expected_x3_from_vB3 = self.v_B3**2 / (2 * self.a_2)
            if not math.isclose(self.x_3, expected_x3_from_vB3, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"x_3 mismatch: got {self.x_3}, expected {expected_x3_from_vB3}", False)
                return False
            self.log(f"x_3 from kinematics: {self.x_3}")
            
            # 9. Check x_4 from v_B4
            expected_x4_from_vB4 = self.v_B4**2 / (2 * self.a_2)
            if not math.isclose(self.x_4, expected_x4_from_vB4, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"x_4 mismatch: got {self.x_4}, expected {expected_x4_from_vB4}", False)
                return False
            self.log(f"x_4 from kinematics: {self.x_4}")
            
            return True
            
        except Exception as e:
            self.log(f"Error in test_trace_consistency: {str(e)}\n{traceback.format_exc()}", False)
            return False
    
    def test_monotonicity(self):
        """Test that perturbing inputs produces expected trends"""
        try:
            # Save original values
            original_m = self.m
            
            # Test 1: Increase mass of block A
            self.m = original_m * 1.1  # Increase by 10%
            
            # Recompute collision velocities using elastic collision formulas
            # v_A1_new = (m - M)/(m + M) * v_0
            # v_B1_new = (2m)/(m + M) * v_0
            
            v_A1_new = (self.m - self.M) / (self.m + self.M) * self.v_0
            v_B1_new = (2 * self.m) / (self.m + self.M) * self.v_0
            
            # Since M > m originally, v_A1 is negative. As m increases toward M,
            # |v_A1| should decrease (less negative)
            if abs(v_A1_new) > abs(self.v_A1):
                self.log(f"Monotonicity fail: |v_A1| increased when m increased", False)
                self.m = original_m
                return False
            
            # v_B1 should increase as m increases (since numerator 2m increases)
            if v_B1_new < self.v_B1:
                self.log(f"Monotonicity fail: v_B1 decreased when m increased", False)
                self.m = original_m
                return False
            
            self.log(f"Monotonicity check passed: increasing m reduces |v_A1| and increases v_B1")
            
            # Restore original value
            self.m = original_m
            
            # Test 2: Increase friction coefficient
            original_mu = self.mu
            self.mu = original_mu * 1.1
            
            # a_2 should increase with increased friction
            a_2_new = self.g * (self.mu * self.costheta + self.sintheta)
            if a_2_new < self.a_2:
                self.log(f"Monotonicity fail: a_2 decreased when μ increased", False)
                self.mu = original_mu
                return False
            
            # x_B should decrease with increased a_2 (since x_B = v_B0^2/(2*a_2))
            x_B_new = self.v_B0**2 / (2 * a_2_new)
            if x_B_new > self.x_B:
                self.log(f"Monotonicity fail: x_B increased when μ increased", False)
                self.mu = original_mu
                return False
            
            self.log(f"Monotonicity check passed: increasing μ increases a_2 and decreases x_B")
            
            # Restore
            self.mu = original_mu
            
            return True
            
        except Exception as e:
            self.log(f"Error in test_monotonicity: {str(e)}\n{traceback.format_exc()}", False)
            # Try to restore on error
            self.m = original_m if 'original_m' in locals() else self.m
            self.mu = original_mu if 'original_mu' in locals() else self.mu
            return False
    
    def run_all_tests(self):
        """Run all tests and return success and logs"""
        self.logs = []
        
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        success = True
        for test_name, test_func in tests:
            self.log(f"Running {test_name}...")
            if not test_func():
                success = False
                self.log(f"{test_name} failed", False)
            else:
                self.log(f"{test_name} passed")
        
        return success, self.logs