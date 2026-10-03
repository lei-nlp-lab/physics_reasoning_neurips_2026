import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs
        self.theta = math.radians(30)  # 30 degrees in radians
        self.sin_theta = math.sin(self.theta)
        self.cos_theta = math.cos(self.theta)
        
        # Given constants (hardcoded from problem)
        self.mu1 = math.sqrt(3)/2
        self.mu2 = math.sqrt(3)/15
        self.v0 = 4.0
        self.g = 10.0
        
        # Safe extraction of results from solution instance
        self.results = {}
        expected_keys = [
            'L₁', 'a₃', 'Δt₁', 'S', 'L₂', 'T', 'Δt₂', 'x₁', 'x₃', 'x₄',
            'x_i+1', 'v_i+1', 'a₁', 'x₂', 'v₁', 'k', 'a₂'
        ]
        
        # Try to get results from solution_instance.results or attributes
        for key in expected_keys:
            # Convert to safe Python variable names
            safe_key = key.replace('₁', '1').replace('₂', '2').replace('₃', '3')\
                        .replace('Δ', 'delta_').replace('₄', '4')
            if hasattr(solution_instance, 'results'):
                self.results[key] = getattr(solution_instance.results, safe_key, 
                                          solution_instance.results.get(safe_key, None))
            else:
                self.results[key] = getattr(solution_instance, safe_key, None)
        
        # For compatibility, also try direct attribute access with fallback
        self.L1 = self.results.get('L₁', getattr(solution_instance, 'L1', None))
        self.a1 = self.results.get('a₁', getattr(solution_instance, 'a1', None))
        self.a2 = self.results.get('a₂', getattr(solution_instance, 'a2', None))
        self.a3 = self.results.get('a₃', getattr(solution_instance, 'a3', None))
        self.L2 = self.results.get('L₂', getattr(solution_instance, 'L2', None))
        self.delta_t1 = self.results.get('Δt₁', getattr(solution_instance, 'delta_t1', None))
        self.delta_t2 = self.results.get('Δt₂', getattr(solution_instance, 'delta_t2', None))
        self.T = self.results.get('T', getattr(solution_instance, 'T', None))
        self.x1 = self.results.get('x₁', getattr(solution_instance, 'x1', None))
        self.x2 = self.results.get('x₂', getattr(solution_instance, 'x2', None))
        self.x3 = self.results.get('x₃', getattr(solution_instance, 'x3', None))
        self.x4 = self.results.get('x₄', getattr(solution_instance, 'x4', None))
        self.v1 = self.results.get('v₁', getattr(solution_instance, 'v1', None))
        self.k = self.results.get('k', getattr(solution_instance, 'k', None))
        self.S = self.results.get('S', getattr(solution_instance, 'S', None))
        self.x_i_plus_1 = self.results.get('x_i+1', None)
        self.v_i_plus_1 = self.results.get('v_i+1', None)
        
        # Initialize logs
        self.logs = []
        
    def log(self, message):
        self.logs.append(message)
    
    def fuzzy_equal(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison with tolerances."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_newton_laws(self):
        """Verify Newton's Second Law for all three accelerations."""
        tests_passed = True
        
        # Test a1: μ1*g*cosθ - g*sinθ = a1
        expected_a1 = self.mu1 * self.g * self.cos_theta - self.g * self.sin_theta
        if not self.fuzzy_equal(self.a1, expected_a1):
            self.log(f"FAIL: a1 mismatch. Expected {expected_a1:.3f}, got {self.a1}")
            tests_passed = False
        else:
            self.log(f"PASS: a1 = {self.a1} matches Newton's Law")
        
        # Test a2: -g*sinθ - μ2*g*cosθ = a2
        expected_a2 = -self.g * self.sin_theta - self.mu2 * self.g * self.cos_theta
        if not self.fuzzy_equal(self.a2, expected_a2):
            self.log(f"FAIL: a2 mismatch. Expected {expected_a2:.3f}, got {self.a2}")
            tests_passed = False
        else:
            self.log(f"PASS: a2 = {self.a2} matches Newton's Law")
        
        # Test a3: g*sinθ - μ2*g*cosθ = a3
        expected_a3 = self.g * self.sin_theta - self.mu2 * self.g * self.cos_theta
        if not self.fuzzy_equal(self.a3, expected_a3):
            self.log(f"FAIL: a3 mismatch. Expected {expected_a3:.3f}, got {self.a3}")
            tests_passed = False
        else:
            self.log(f"PASS: a3 = {self.a3} matches Newton's Law")
        
        return tests_passed
    
    def test_kinematics_consistency(self):
        """Verify kinematic relationships between quantities."""
        tests_passed = True
        
        # L1 = v0^2 / (2*a1)
        if self.a1 and self.a1 != 0:
            expected_L1 = self.v0**2 / (2 * self.a1)
            if not self.fuzzy_equal(self.L1, expected_L1):
                self.log(f"FAIL: L1 kinematics mismatch. Expected {expected_L1:.3f}, got {self.L1}")
                tests_passed = False
            else:
                self.log(f"PASS: L1 = {self.L1} matches v0^2/(2*a1)")
        
        # L2 = -v0^2 / (2*a2)  [since a2 is negative]
        if self.a2 and self.a2 != 0:
            expected_L2 = -self.v0**2 / (2 * self.a2)
            if not self.fuzzy_equal(self.L2, expected_L2):
                self.log(f"FAIL: L2 kinematics mismatch. Expected {expected_L2:.3f}, got {self.L2}")
                tests_passed = False
            else:
                self.log(f"PASS: L2 = {self.L2} matches -v0^2/(2*a2)")
        
        # Δt1 = v0 / a1
        if self.a1 and self.a1 != 0:
            expected_dt1 = self.v0 / self.a1
            if not self.fuzzy_equal(self.delta_t1, expected_dt1):
                self.log(f"FAIL: Δt1 mismatch. Expected {expected_dt1:.3f}, got {self.delta_t1}")
                tests_passed = False
            else:
                self.log(f"PASS: Δt1 = {self.delta_t1} matches v0/a1")
        
        # Δt2 = -v0 / a2
        if self.a2 and self.a2 != 0:
            expected_dt2 = -self.v0 / self.a2
            if not self.fuzzy_equal(self.delta_t2, expected_dt2):
                self.log(f"FAIL: Δt2 mismatch. Expected {expected_dt2:.3f}, got {self.delta_t2}")
                tests_passed = False
            else:
                self.log(f"PASS: Δt2 = {self.delta_t2} matches -v0/a2")
        
        # T = Δt1 + Δt2
        if self.delta_t1 is not None and self.delta_t2 is not None:
            expected_T = self.delta_t1 + self.delta_t2
            if not self.fuzzy_equal(self.T, expected_T):
                self.log(f"FAIL: T mismatch. Expected {expected_T:.3f}, got {self.T}")
                tests_passed = False
            else:
                self.log(f"PASS: T = {self.T} matches Δt1 + Δt2")
        
        # k = -a3/a2
        if self.a2 and self.a2 != 0 and self.a3 is not None:
            expected_k = -self.a3 / self.a2
            if not self.fuzzy_equal(self.k, expected_k):
                self.log(f"FAIL: k mismatch. Expected {expected_k:.3f}, got {self.k}")
                tests_passed = False
            else:
                self.log(f"PASS: k = {self.k} matches -a3/a2")
        
        # x2 from kinematics: x2 = v1^2 / (2*|a2|) but also x2 = (a3/|a2|)*x1
        if self.a2 and self.a2 != 0 and self.x1 is not None and self.a3 is not None:
            expected_x2 = (self.a3 / abs(self.a2)) * self.x1
            if self.x2 is not None and not self.fuzzy_equal(self.x2, expected_x2):
                self.log(f"FAIL: x2 mismatch. Expected {expected_x2:.3f}, got {self.x2}")
                tests_passed = False
            elif self.x2 is not None:
                self.log(f"PASS: x2 = {self.x2} matches (a3/|a2|)*x1")
        
        # x3 = x2
        if self.x2 is not None and self.x3 is not None:
            if not self.fuzzy_equal(self.x3, self.x2):
                self.log(f"FAIL: x3 should equal x2. x2={self.x2}, x3={self.x3}")
                tests_passed = False
            else:
                self.log(f"PASS: x3 = {self.x3} equals x2")
        
        # x4 = k*x3
        if self.k is not None and self.x3 is not None and self.x4 is not None:
            expected_x4 = self.k * self.x3
            if not self.fuzzy_equal(self.x4, expected_x4):
                self.log(f"FAIL: x4 mismatch. Expected {expected_x4:.3f}, got {self.x4}")
                tests_passed = False
            else:
                self.log(f"PASS: x4 = {self.x4} matches k*x3")
        
        # Geometric series sum: S = x1 + 2*x2/(1-k) if infinite series
        if self.x1 is not None and self.x2 is not None and self.k is not None:
            # For infinite geometric series: S = x1 + 2*x2*(1 + k + k² + ...)
            # Sum = x1 + 2*x2/(1-k)
            if abs(self.k) < 1:  # Series converges
                expected_S = self.x1 + 2 * self.x2 / (1 - self.k)
                if not self.fuzzy_equal(self.S, expected_S):
                    self.log(f"FAIL: S mismatch from geometric series. Expected {expected_S:.3f}, got {self.S}")
                    tests_passed = False
                else:
                    self.log(f"PASS: S = {self.S} matches geometric series sum")
        
        return tests_passed
    
    def test_energy_conservation(self):
        """
        Verify energy considerations.
        On conveyor belt: Work done by friction and gravity equals change in KE.
        On wooden board: Similar energy balance for oscillatory motion.
        """
        tests_passed = True
        
        # For conveyor belt segment (A to B):
        # Initial KE = 0, final KE = 0 (relative to belt)
        # Work done by friction: μ1*m*g*cosθ*L1 (positive since friction helps motion)
        # Work done by gravity: -m*g*sinθ*L1
        # Net work = m*a1*L1 = change in KE = 0? Wait, object gains speed relative to ground.
        # Actually, the object starts at 0 and reaches v0 relative to ground.
        # So (1/2)*m*v0^2 = m*(μ1*g*cosθ - g*sinθ)*L1 = m*a1*L1
        if self.L1 is not None and self.a1 is not None:
            # Check: v0^2 should equal 2*a1*L1
            lhs = self.v0**2
            rhs = 2 * self.a1 * self.L1
            if not self.fuzzy_equal(lhs, rhs):
                self.log(f"FAIL: Energy balance on belt. v0²={lhs:.3f}, 2*a1*L1={rhs:.3f}")
                tests_passed = False
            else:
                self.log(f"PASS: Energy balance on belt: v0² = 2*a1*L1")
        
        # For wooden board segment (B to D, first pass):
        # Initial KE = (1/2)*m*v0^2, final KE = 0 at D
        # Work done by friction: -μ2*m*g*cosθ*L2 (negative)
        # Work done by gravity: -m*g*sinθ*L2
        # So (1/2)*m*v0^2 + m*(-g*sinθ - μ2*g*cosθ)*L2 = 0
        # => v0^2 = 2*(g*sinθ + μ2*g*cosθ)*L2 = -2*a2*L2 (since a2 is negative)
        if self.L2 is not None and self.a2 is not None:
            lhs = self.v0**2
            rhs = -2 * self.a2 * self.L2  # a2 is negative, so -a2 is positive
            if not self.fuzzy_equal(lhs, rhs):
                self.log(f"FAIL: Energy balance on board (first pass). v0²={lhs:.3f}, -2*a2*L2={rhs:.3f}")
                tests_passed = False
            else:
                self.log(f"PASS: Energy balance on board: v0² = -2*a2*L2")
        
        return tests_passed
    
    def test_monotonicity(self):
        """
        Perturb an input parameter and check if results change in expected direction.
        We'll perturb μ1 and check L1 (should decrease if μ1 increases).
        """
        original_mu1 = self.mu1
        tests_passed = True
        
        try:
            # Increase μ1 by 5%
            perturbed_mu1 = original_mu1 * 1.05
            
            # Recalculate a1 with perturbed μ1
            perturbed_a1 = perturbed_mu1 * self.g * self.cos_theta - self.g * self.sin_theta
            
            # Original a1 and L1
            original_a1 = self.a1
            original_L1 = self.L1
            
            # Recalculate L1 using kinematics: L1 = v0^2/(2*a1)
            perturbed_L1 = self.v0**2 / (2 * perturbed_a1)
            
            # Check monotonicity: if μ1 increases, a1 should increase, L1 should decrease
            if perturbed_mu1 > original_mu1:
                if perturbed_a1 > original_a1 and perturbed_L1 < original_L1:
                    self.log(f"PASS: Monotonicity test - Increasing μ1 increases a1 and decreases L1")
                else:
                    self.log(f"FAIL: Monotonicity test - Expected a1↑ and L1↓ when μ1↑")
                    tests_passed = False
            else:
                self.log("WARN: Monotonicity test skipped - perturbation didn't increase μ1")
            
        except Exception as e:
            self.log(f"ERROR in monotonicity test: {str(e)}")
            tests_passed = False
        
        # Restore original μ1 (not strictly needed since we didn't modify instance)
        self.mu1 = original_mu1
        
        return tests_passed
    
    def test_target_answers(self):
        """Verify the three target answers."""
        tests_passed = True
        
        # Target 1: L1 = 3.2 m
        if not self.fuzzy_equal(self.L1, 3.2):
            self.log(f"FAIL: L1 target mismatch. Expected ~3.2, got {self.L1}")
            tests_passed = False
        else:
            self.log(f"PASS: L1 = {self.L1} ≈ 3.2 m")
        
        # Target 2: T = 34/15 ≈ 2.2667 s
        expected_T = 34/15  # Approximately 2.2667
        if not self.fuzzy_equal(self.T, expected_T):
            self.log(f"FAIL: T target mismatch. Expected ~{expected_T:.3f}, got {self.T}")
            tests_passed = False
        else:
            self.log(f"PASS: T = {self.T} ≈ 34/15 s")
        
        # Target 3: S = 20/3 ≈ 6.6667 m
        expected_S = 20/3  # Approximately 6.6667
        if not self.fuzzy_equal(self.S, expected_S):
            self.log(f"FAIL: S target mismatch. Expected ~{expected_S:.3f}, got {self.S}")
            tests_passed = False
        else:
            self.log(f"PASS: S = {self.S} ≈ 20/3 m")
        
        return tests_passed
    
    def run_all_tests(self):
        """Execute all tests and return success status with logs."""
        self.logs = []
        all_passed = True
        
        try:
            self.log("=== PhysicsChecker Started ===")
            
            # Test 1: Newton's Laws
            self.log("\n--- Testing Newton's Laws ---")
            if not self.test_newton_laws():
                all_passed = False
            
            # Test 2: Kinematics Consistency
            self.log("\n--- Testing Kinematics Consistency ---")
            if not self.test_kinematics_consistency():
                all_passed = False
            
            # Test 3: Energy Conservation
            self.log("\n--- Testing Energy Conservation ---")
            if not self.test_energy_conservation():
                all_passed = False
            
            # Test 4: Monotonicity
            self.log("\n--- Testing Monotonicity ---")
            if not self.test_monotonicity():
                all_passed = False
            
            # Test 5: Target Answers
            self.log("\n--- Testing Target Answers ---")
            if not self.test_target_answers():
                all_passed = False
            
            # Summary
            self.log("\n=== Summary ===")
            if all_passed:
                self.log("All tests PASSED!")
            else:
                self.log("Some tests FAILED!")
            
        except Exception as e:
            self.log(f"ERROR during testing: {str(e)}")
            self.log(traceback.format_exc())
            all_passed = False
        
        return all_passed, self.logs