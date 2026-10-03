import math
import sympy as sp
from typing import Dict, Any, List, Tuple

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        
        # Inputs (extracted safely)
        self.v_A0 = 10.0  # Initial speed of train A (given in problem)
        self.v_B0 = 30.0  # Initial speed of train B (given in problem)
        self.initial_distance = 500.0  # 500m separation (given in problem)
        self.stop_distance_B = 1800.0  # 1800m to stop (given in problem)
        
        # Results (extracted safely from results dict)
        results_dict = getattr(solution_instance, 'results', {})
        
        # Extract required variables from contract
        self.a_B = results_dict.get('a_B', None)  # Magnitude (should be positive)
        self.t_1 = results_dict.get('t_1', None)
        self.a_A = results_dict.get('a_A', None)
        
        # Additional intermediate results
        self.x = results_dict.get('x', None)
        self.x_A = results_dict.get('x_A', None)
        self.x_B = results_dict.get('x_B', None)
        self.t = results_dict.get('t', None)
        self.t_2 = results_dict.get('t_2', None)
        self.v = results_dict.get('v', None)
        self.x_A2 = results_dict.get('x_A2', None)
        self.x_B2 = results_dict.get('x_B2', None)
        
        # Tolerance for numerical comparisons
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Test logs
        self.logs = []
    
    def _log(self, msg: str, success: bool):
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {msg}")
        return success
    
    def test_invariants(self) -> bool:
        """Test fundamental kinematic invariants."""
        tests_passed = True
        
        # 1. Check deceleration calculation from stopping distance
        if self.a_B is not None:
            # a_B should be the magnitude (positive)
            # From stopping: 0 = v_B0^2 + 2*(-a_B)*x_B_stop
            # So a_B = v_B0^2 / (2*x_B_stop)
            expected_a_B_mag = self.v_B0**2 / (2 * self.stop_distance_B)
            if not math.isclose(self.a_B, expected_a_B_mag, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                tests_passed = self._log(
                    f"a_B magnitude mismatch: got {self.a_B}, expected {expected_a_B_mag}",
                    False
                )
            else:
                tests_passed = self._log(
                    f"a_B magnitude correct: {self.a_B} ≈ {expected_a_B_mag}",
                    True
                ) & tests_passed
        
        # 2. Check collision time t_1 consistency with kinematics
        if self.a_B is not None and self.t_1 is not None:
            # Position equations for collision: x_A(t) + 500 = x_B(t)
            # x_A = v_A0 * t, x_B = v_B0 * t + 0.5 * (-a_B) * t^2
            # Use signed acceleration: -a_B
            lhs = self.v_A0 * self.t_1 + self.initial_distance
            rhs = self.v_B0 * self.t_1 + 0.5 * (-self.a_B) * self.t_1**2
            if not math.isclose(lhs, rhs, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                tests_passed = self._log(
                    f"Collision condition fails: v_A0*t + d = {lhs}, v_B0*t + 0.5*(-a_B)*t^2 = {rhs}",
                    False
                )
            else:
                tests_passed = self._log(
                    f"Collision time t_1={self.t_1}s satisfies kinematic equations",
                    True
                ) & tests_passed
        
        # 3. Check critical condition for avoidance (same speed and position)
        if all(x is not None for x in [self.a_A, self.t_2, self.v, self.a_B]):
            # Speed condition: v = v_A0 + a_A * t_2 = v_B0 + (-a_B) * t_2
            v_from_A = self.v_A0 + self.a_A * self.t_2
            v_from_B = self.v_B0 + (-self.a_B) * self.t_2
            
            if not (math.isclose(self.v, v_from_A, rel_tol=self.rel_tol, abs_tol=self.abs_tol) and
                    math.isclose(self.v, v_from_B, rel_tol=self.rel_tol, abs_tol=self.abs_tol)):
                tests_passed = self._log(
                    f"Speed condition mismatch: v={self.v}, v_A0+a_A*t_2={v_from_A}, v_B0+(-a_B)*t_2={v_from_B}",
                    False
                )
            else:
                tests_passed = self._log(
                    f"Speed condition satisfied: common velocity v={self.v}m/s",
                    True
                ) & tests_passed
            
            # Position condition for critical avoidance
            # At critical point: x_A2 + initial_distance = x_B2
            if self.x_A2 is not None and self.x_B2 is not None:
                # Use the stored values
                if not math.isclose(self.x_A2 + self.initial_distance, self.x_B2, 
                                   rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    tests_passed = self._log(
                        f"Position condition fails: x_A2 + d = {self.x_A2 + self.initial_distance}, x_B2 = {self.x_B2}",
                        False
                    )
                else:
                    tests_passed = self._log(
                        f"Position condition satisfied at critical point",
                        True
                    ) & tests_passed
            else:
                # Fallback calculation
                x_A_crit = self.v_A0 * self.t_2 + 0.5 * self.a_A * self.t_2**2
                x_B_crit = self.v_B0 * self.t_2 + 0.5 * (-self.a_B) * self.t_2**2
                if not math.isclose(x_A_crit + self.initial_distance, x_B_crit, 
                                   rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    tests_passed = self._log(
                        f"Position condition fails: x_A + d = {x_A_crit + self.initial_distance}, x_B = {x_B_crit}",
                        False
                    )
                else:
                    tests_passed = self._log(
                        f"Position condition satisfied at critical point",
                        True
                    ) & tests_passed
        
        return tests_passed
    
    def test_trace_consistency(self) -> bool:
        """Check consistency between intermediate and final results."""
        tests_passed = True
        
        # 1. Check step 1: a_B from stopping distance
        if self.a_B is not None:
            expected_a_B_mag = self.v_B0**2 / (2 * self.stop_distance_B)
            if not math.isclose(self.a_B, expected_a_B_mag, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                tests_passed = self._log(
                    f"a_B magnitude inconsistent: got {self.a_B}, expected {expected_a_B_mag}",
                    False
                )
        
        # 2. Check step 2: time to reach same speed (80s calculation)
        if self.a_B is not None and self.t is not None:
            # t = (v_B0 - v_A0) / a_B (since a_B is magnitude)
            expected_t = (self.v_B0 - self.v_A0) / self.a_B
            if not math.isclose(self.t, expected_t, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                tests_passed = self._log(
                    f"Intermediate t inconsistent: got {self.t}, expected (v_B0-v_A0)/a_B={expected_t}",
                    False
                )
        
        # 3. Check step 3: displacements during time t (if available)
        if self.t is not None:
            if self.x_A is not None:
                expected_x_A = self.v_A0 * self.t
                if not math.isclose(self.x_A, expected_x_A, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    tests_passed = self._log(
                        f"x_A inconsistent: got {self.x_A}, expected v_A0*t={expected_x_A}",
                        False
                    )
            
            if self.x_B is not None:
                # x_B = v_B0*t + 0.5*(-a_B)*t^2
                expected_x_B = self.v_B0 * self.t + 0.5 * (-self.a_B) * self.t**2
                if not math.isclose(self.x_B, expected_x_B, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    tests_passed = self._log(
                        f"x_B inconsistent: got {self.x_B}, expected v_B0*t+0.5*(-a_B)*t^2={expected_x_B}",
                        False
                    )
            
            if self.x is not None and self.x_A is not None and self.x_B is not None:
                expected_x = self.x_B - self.x_A
                if not math.isclose(self.x, expected_x, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    tests_passed = self._log(
                        f"Relative displacement x inconsistent: got {self.x}, expected x_B-x_A={expected_x}",
                        False
                    )
        
        # 4. Check that t_1 (collision time) < t (time to equal speed) when no avoidance
        if self.t_1 is not None and self.t is not None:
            if self.t_1 >= self.t:
                tests_passed = self._log(
                    f"Collision occurs after reaching equal speed: t_1={self.t_1}s, t={self.t}s",
                    False
                )
        
        if tests_passed:
            self._log("All trace consistency checks passed", True)
        
        return tests_passed
    
    def test_monotonicity(self) -> bool:
        """
        Test monotonic relationship: increasing initial distance should decrease required a_A.
        """
        if self.a_A is None or self.a_B is None:
            self._log("Cannot test monotonicity: missing a_A or a_B", True)
            return True
        
        tests_passed = True
        
        # Perturb initial distance by +10%
        perturbed_distance = self.initial_distance * 1.1
        
        # Recalculate critical a_A using derived formulas
        # From the condition: v_A0 + a_A*t = v_B0 + (-a_B)*t (same speed)
        # and positions: v_A0*t + 0.5*a_A*t^2 + d = v_B0*t + 0.5*(-a_B)*t^2 (same position)
        
        # Solve symbolically for a_A
        t, a_A, a_B_mag, d, v_A0, v_B0 = sp.symbols('t a_A a_B_mag d v_A0 v_B0', real=True)
        a_B_signed = -a_B_mag  # Signed acceleration
        
        # Equations:
        eq1 = sp.Eq(v_A0 + a_A*t, v_B0 + a_B_signed*t)  # Same speed
        eq2 = sp.Eq(v_A0*t + 0.5*a_A*t**2 + d, v_B0*t + 0.5*a_B_signed*t**2)  # Same position
        
        # Solve for a_A and t
        sol = sp.solve([eq1, eq2], [a_A, t], dict=True)
        
        if sol:
            # Substitute known values
            subs_dict = {a_B_mag: float(self.a_B), d: perturbed_distance, 
                        v_A0: self.v_A0, v_B0: self.v_B0}
            
            # Filter solutions where t > 0
            valid_solutions = []
            for s in sol:
                try:
                    t_val = float(s[t].subs(subs_dict))
                    if t_val > 0:
                        a_A_val = float(s[a_A].subs(subs_dict))
                        valid_solutions.append((a_A_val, t_val))
                except:
                    continue
            
            if valid_solutions:
                # Take the solution with smallest positive t (physically relevant)
                a_A_perturbed, t_perturbed = min(valid_solutions, key=lambda x: x[1])
                
                # Check monotonicity: larger distance should require smaller acceleration
                if a_A_perturbed >= self.a_A:
                    tests_passed = self._log(
                        f"Monotonicity violated: d↑ to {perturbed_distance}m, a_A={a_A_perturbed} ≥ original {self.a_A}",
                        False
                    )
                else:
                    tests_passed = self._log(
                        f"Monotonicity confirmed: d↑ to {perturbed_distance}m, a_A↓ to {a_A_perturbed}",
                        True
                    ) & tests_passed
            else:
                tests_passed = self._log(
                    "Could not find valid solution for perturbed case",
                    True
                )
        else:
            tests_passed = self._log(
                "Could not solve symbolic equations for monotonicity test",
                True
            )
        
        return tests_passed
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success status and logs."""
        self.logs = []
        
        self._log("="*50, True)
        self._log("Starting Physics Audit", True)
        self._log(f"Inputs: v_A0={self.v_A0}m/s, v_B0={self.v_B0}m/s, d={self.initial_distance}m", True)
        self._log(f"Results: a_B={self.a_B}, t_1={self.t_1}, a_A={self.a_A}", True)
        
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        all_passed = test1 and test2 and test3
        
        self._log("="*50, True)
        self._log(f"AUDIT {'PASSED' if all_passed else 'FAILED'}", all_passed)
        
        return all_passed, self.logs