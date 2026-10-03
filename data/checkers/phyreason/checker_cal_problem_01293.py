import math
import sympy as sp
from sympy.abc import t

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from solution_instance."""
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract all known variables with safe fallbacks
        self.d = getattr(solution_instance, 'd', self.results.get('d', 0.0))
        self.a_1 = getattr(solution_instance, 'a_1', self.results.get('a_1', 0.0))
        self.x_car = getattr(solution_instance, 'x_car', self.results.get('x_car', 0.0))
        self.t_2 = getattr(solution_instance, 't_2', self.results.get('t_2', 0.0))
        self.t = getattr(solution_instance, 't', self.results.get('t', 0.0))
        self.x_1 = getattr(solution_instance, 'x_1', self.results.get('x_1', 0.0))
        self.x_stone = getattr(solution_instance, 'x_stone', self.results.get('x_stone', 0.0))
        self.v_0 = getattr(solution_instance, 'v_0', self.results.get('v_0', 0.0))
        self.a_2 = getattr(solution_instance, 'a_2', self.results.get('a_2', 0.0))
        self.t_3 = getattr(solution_instance, 't_3', self.results.get('t_3', 0.0))
        self.t_1 = getattr(solution_instance, 't_1', self.results.get('t_1', 0.0))
        self.v_1 = getattr(solution_instance, 'v_1', self.results.get('v_1', 0.0))
        self.a_car = getattr(solution_instance, 'a_car', self.results.get('a_car', 0.0))
        
        # Fixed parameters from problem statement
        self.t_reaction = 1.0  # driver's reaction time
        self.g = 10.0  # gravitational acceleration
        
        # For monotonicity test
        self.original_a_1 = self.a_1
        
        self.logs = []

    def _log(self, message, success):
        """Helper to log test results."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {message}")
        return success

    def test_invariants(self):
        """Verify kinematic equations and energy conservation."""
        tests_passed = True
        
        # 1. Check v_1 = sqrt(2*a_1*x_1)
        expected_v_1 = math.sqrt(2 * self.a_1 * self.x_1)
        if not math.isclose(self.v_1, expected_v_1, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(
                f"v_1 invariant failed: {self.v_1} ≠ sqrt(2*{self.a_1}*{self.x_1}) = {expected_v_1}",
                False
            )
        else:
            self._log(f"v_1 = sqrt(2*a_1*x_1) satisfied: {self.v_1:.2f} ≈ {expected_v_1:.2f}", True)
        
        # 2. Check t_1 = v_1 / a_1
        expected_t_1 = self.v_1 / self.a_1 if self.a_1 != 0 else 0
        if not math.isclose(self.t_1, expected_t_1, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(
                f"t_1 invariant failed: {self.t_1} ≠ {self.v_1}/{self.a_1} = {expected_t_1}",
                False
            )
        else:
            self._log(f"t_1 = v_1/a_1 satisfied: {self.t_1:.2f} ≈ {expected_t_1:.2f}", True)
        
        # 3. Check t_2 = t_1 - t_reaction
        expected_t_2 = self.t_1 - self.t_reaction
        if not math.isclose(self.t_2, expected_t_2, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(
                f"t_2 invariant failed: {self.t_2} ≠ {self.t_1} - {self.t_reaction} = {expected_t_2}",
                False
            )
        else:
            self._log(f"t_2 = t_1 - t_reaction satisfied: {self.t_2:.2f} ≈ {expected_t_2:.2f}", True)
        
        # 4. Check d = 0.5 * a_car * t_2^2
        expected_d = 0.5 * self.a_car * self.t_2**2
        if not math.isclose(self.d, expected_d, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(
                f"d invariant failed: {self.d} ≠ 0.5*{self.a_car}*{self.t_2}^2 = {expected_d}",
                False
            )
        else:
            self._log(f"d = 0.5*a_car*t_2^2 satisfied: {self.d:.2f} ≈ {expected_d:.2f}", True)
        
        # 5. Check v_0 = a_car * t_2
        expected_v_0 = self.a_car * self.t_2
        if not math.isclose(self.v_0, expected_v_0, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(
                f"v_0 invariant failed: {self.v_0} ≠ {self.a_car}*{self.t_2} = {expected_v_0}",
                False
            )
        else:
            self._log(f"v_0 = a_car*t_2 satisfied: {self.v_0:.2f} ≈ {expected_v_0:.2f}", True)
        
        # 6. Check t from v_0 + a_car*t = v_1 - a_2*t
        expected_t = (self.v_1 - self.v_0) / (self.a_car + self.a_2) if (self.a_car + self.a_2) != 0 else 0
        if not math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(
                f"t invariant failed: {self.t} ≠ ({self.v_1}-{self.v_0})/({self.a_car}+{self.a_2}) = {expected_t}",
                False
            )
        else:
            self._log(f"t from equal speeds satisfied: {self.t:.2f} ≈ {expected_t:.2f}", True)
        
        # 7. Check x_stone = v_1*t - 0.5*a_2*t^2
        expected_x_stone = self.v_1 * self.t - 0.5 * self.a_2 * self.t**2
        if not math.isclose(self.x_stone, expected_x_stone, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(
                f"x_stone invariant failed: {self.x_stone} ≠ {self.v_1}*{self.t} - 0.5*{self.a_2}*{self.t}^2 = {expected_x_stone}",
                False
            )
        else:
            self._log(f"x_stone kinematic satisfied: {self.x_stone:.2f} ≈ {expected_x_stone:.2f}", True)
        
        # 8. Check x_car = v_0*t + 0.5*a_car*t^2
        expected_x_car = self.v_0 * self.t + 0.5 * self.a_car * self.t**2
        if not math.isclose(self.x_car, expected_x_car, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(
                f"x_car invariant failed: {self.x_car} ≠ {self.v_0}*{self.t} + 0.5*{self.a_car}*{self.t}^2 = {expected_x_car}",
                False
            )
        else:
            self._log(f"x_car kinematic satisfied: {self.x_car:.2f} ≈ {expected_x_car:.2f}", True)
        
        # 9. Verify t_3 satisfies collision equation: v_1*(t_3 - t_2) - 0.5*a_2*(t_3-t_2)^2 = 0.5*a_car*t_3^2
        if self.t_3 > 0:
            lhs = self.v_1 * (self.t_3 - self.t_2) - 0.5 * self.a_2 * (self.t_3 - self.t_2)**2
            rhs = 0.5 * self.a_car * self.t_3**2
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                tests_passed = self._log(
                    f"t_3 collision equation failed: LHS={lhs:.3f}, RHS={rhs:.3f}", False
                )
            else:
                self._log(f"t_3 satisfies collision equation: LHS={lhs:.3f} ≈ RHS={rhs:.3f}", True)
        
        return tests_passed

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        tests_passed = True
        
        # 1. Verify that the two target answers match our computed values
        # First target: "8 m/s, 4s" should match v_1 and t_1
        if not (math.isclose(self.v_1, 8.0, rel_tol=1e-2, abs_tol=1e-2) and 
                math.isclose(self.t_1, 4.0, rel_tol=1e-2, abs_tol=1e-2)):
            tests_passed = self._log(
                f"First target answer mismatch: v_1={self.v_1}, t_1={self.t_1} vs expected 8 m/s, 4s",
                False
            )
        else:
            self._log(f"First target answer matches: v_1={self.v_1:.1f} m/s, t_1={self.t_1:.1f} s", True)
        
        # Second target: t_3 = (5 - √10/5) ≈ (5 - 0.6324555) ≈ 4.3675
        expected_t_3 = 5 - math.sqrt(10)/5
        if not math.isclose(self.t_3, expected_t_3, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed = self._log(
                f"Second target answer mismatch: t_3={self.t_3} vs expected (5-√10/5)≈{expected_t_3:.4f}",
                False
            )
        else:
            self._log(f"Second target answer matches: t_3={self.t_3:.4f} ≈ {expected_t_3:.4f}", True)
        
        # 2. Check that when stone and car have same speed (at time t), stone is ahead of car
        # This is implied by x_stone > x_car + d? Actually, at time t after stone reaches bottom,
        # the stone's total distance from start is x_1 + x_stone = 16 + x_stone
        # Car's total distance from start is d + x_car
        # We should have: 16 + x_stone > d + x_car (stone is ahead when speeds equal)
        stone_total = self.x_1 + self.x_stone
        car_total = self.d + self.x_car
        if not stone_total > car_total - 0.1:  # Allow small tolerance
            tests_passed = self._log(
                f"Stone not ahead when speeds equal: stone_total={stone_total}, car_total={car_total}",
                False
            )
        else:
            self._log(f"Stone ahead when speeds equal: stone_total={stone_total:.2f} > car_total={car_total:.2f}", True)
        
        # 3. Check that collision happens after equal speeds time
        if self.t_3 <= self.t:
            tests_passed = self._log(
                f"Collision time t_3={self.t_3} should be > equal speeds time t={self.t}", False
            )
        else:
            self._log(f"Collision occurs after equal speeds: t_3={self.t_3:.2f} > t={self.t:.2f}", True)
        
        return tests_passed

    def test_monotonicity(self):
        """Perturb a_1 and check trend in t_3."""
        tests_passed = True
        
        # Save original state
        original_a_1 = self.a_1
        
        # Perturb a_1 by +10%
        perturbed_a_1 = original_a_1 * 1.1
        
        # Recalculate intermediate values based on perturbed a_1
        # Step 1: v_1 = sqrt(2*a_1*x_1)
        perturbed_v_1 = math.sqrt(2 * perturbed_a_1 * self.x_1)
        
        # Step 2: t_1 = v_1 / a_1
        perturbed_t_1 = perturbed_v_1 / perturbed_a_1
        
        # Step 3: t_2 = t_1 - t_reaction
        perturbed_t_2 = perturbed_t_1 - self.t_reaction
        
        # Step 4: v_0 = a_car * t_2
        perturbed_v_0 = self.a_car * perturbed_t_2
        
        # Step 5: Solve for t_3 from collision equation
        # Equation: v_1*(t_3 - t_2) - 0.5*a_2*(t_3-t_2)^2 = 0.5*a_car*t_3^2
        # Rearrange: 0.5*a_car*t_3^2 - v_1*(t_3 - t_2) + 0.5*a_2*(t_3-t_2)^2 = 0
        # Solve quadratic for t_3
        
        # Coefficients for quadratic in t_3: A*t_3^2 + B*t_3 + C = 0
        A = 0.5 * self.a_car - 0.5 * self.a_2
        B = -perturbed_v_1 + self.a_2 * perturbed_t_2
        C = perturbed_v_1 * perturbed_t_2 - 0.5 * self.a_2 * perturbed_t_2**2
        
        # Solve quadratic
        discriminant = B**2 - 4*A*C
        if discriminant < 0:
            self._log("Monotonicity test: No real solution for perturbed t_3", False)
            return False
        
        sqrt_disc = math.sqrt(discriminant)
        t3_candidate1 = (-B + sqrt_disc) / (2*A)
        t3_candidate2 = (-B - sqrt_disc) / (2*A)
        
        # Choose positive root greater than t_2
        perturbed_t_3 = None
        for candidate in [t3_candidate1, t3_candidate2]:
            if candidate > perturbed_t_2 and candidate > 0:
                perturbed_t_3 = candidate
                break
        
        if perturbed_t_3 is None:
            self._log("Monotonicity test: No valid positive root found", False)
            return False
        
        # Check monotonicity: Increasing a_1 should DECREASE t_3 (stone faster → earlier collision)
        if perturbed_t_3 < self.t_3:
            self._log(
                f"Monotonicity holds: a_1↑ ({original_a_1}→{perturbed_a_1}) → t_3↓ ({self.t_3:.3f}→{perturbed_t_3:.3f})",
                True
            )
        else:
            tests_passed = self._log(
                f"Monotonicity violated: a_1↑ but t_3 not decreasing ({self.t_3:.3f}→{perturbed_t_3:.3f})",
                False
            )
        
        # Restore original value
        self.a_1 = original_a_1
        
        return tests_passed

    def run_all_tests(self):
        """Execute all tests and return summary."""
        self.logs = []
        
        overall_success = True
        
        # Run individual test suites
        test_functions = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency Tests", self.test_trace_consistency),
            ("Monotonicity Test", self.test_monotonicity)
        ]
        
        for test_name, test_func in test_functions:
            self.logs.append(f"\n--- Running {test_name} ---")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self.logs.append(f"ERROR in {test_name}: {str(e)}")
                overall_success = False
        
        # Summary
        self.logs.append(f"\n=== Overall Result: {'PASS' if overall_success else 'FAIL'} ===")
        
        return overall_success, self.logs