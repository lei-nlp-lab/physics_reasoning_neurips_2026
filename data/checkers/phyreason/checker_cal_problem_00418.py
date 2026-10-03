import math
import sympy as sp
from typing import Tuple, List, Union

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize PhysicsChecker with safe attribute access.
        solution_instance: The solution object from the student's code.
        """
        # Extract inputs (safe access with defaults)
        self.v_final = getattr(solution_instance, 'v_final', 30.0)
        self.t_accel = getattr(solution_instance, 't_accel', 15.0)
        self.x_brake = getattr(solution_instance, 'x_brake', 90.0)
        
        # Extract results (safe access via results dictionary or attributes)
        self.a1 = getattr(solution_instance, 'a1', 
                         getattr(solution_instance, 'results', {}).get('a_1', 0.0))
        self.a2 = getattr(solution_instance, 'a2',
                         getattr(solution_instance, 'results', {}).get('a_2', 0.0))
        
        # Expected values from steps analysis
        self.expected_a1 = 2.0
        self.expected_a2 = 5.0
        
        # Store logs
        self.logs = []
        
        # Define symbols for symbolic checks
        self.v0, self.v1, self.v2, self.t, self.x, self.a = sp.symbols('v0 v1 v2 t x a')
        
    def log(self, message: str) -> None:
        """Helper to log messages."""
        self.logs.append(message)
        
    def fuzzy_equal(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Fuzzy numerical comparison with tolerance."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> bool:
        """
        Verify kinematic invariants using symbolic and numerical checks.
        Returns: bool success
        """
        success = True
        
        # 1. Check acceleration phase: v = a1 * t (since starts from rest)
        calc_a1 = self.v_final / self.t_accel
        if not self.fuzzy_equal(self.a1, calc_a1):
            self.log(f"Invariant failed: a1 should be v/t = {calc_a1:.3f}, got {self.a1:.3f}")
            success = False
        else:
            self.log(f"Acceleration phase invariant OK: a1 = v/t = {calc_a1:.3f}")
            
        # 2. Check braking phase: 0 = v_final^2 + 2*a2*x_brake (since final v=0)
        #    So a2 = -v_final^2 / (2*x_brake)
        calc_a2 = -self.v_final**2 / (2 * self.x_brake)
        if not self.fuzzy_equal(abs(self.a2), abs(calc_a2)):
            self.log(f"Invariant failed: |a2| should be v^2/(2x) = {abs(calc_a2):.3f}, got |{self.a2:.3f}|")
            success = False
        else:
            self.log(f"Braking phase invariant OK: |a2| = v^2/(2x) = {abs(calc_a2):.3f}")
            
        # 3. Check sign convention: a1 positive, a2 negative
        if self.a1 <= 0:
            self.log(f"Warning: a1 should be positive (acceleration), got {self.a1:.3f}")
        if self.a2 >= 0:
            self.log(f"Warning: a2 should be negative (deceleration), got {self.a2:.3f}")
            
        # 4. Symbolic check: Verify equations hold symbolically
        try:
            # Acceleration phase: v1 = 0 + a1*t
            eq1 = sp.Eq(self.v1, self.a * self.t)
            sol1 = sp.solve(eq1.subs({self.v1: self.v_final, self.t: self.t_accel}), self.a)
            if not any(self.fuzzy_equal(self.a1, float(sol)) for sol in sol1):
                self.log(f"Symbolic check failed for acceleration phase")
                success = False
                
            # Braking phase: 0^2 = v_final^2 + 2*a2*x
            eq2 = sp.Eq(0, self.v1**2 + 2 * self.a * self.x)
            sol2 = sp.solve(eq2.subs({self.v1: self.v_final, self.x: self.x_brake}), self.a)
            if not any(self.fuzzy_equal(self.a2, float(sol)) for sol in sol2):
                self.log(f"Symbolic check failed for braking phase")
                success = False
                
        except Exception as e:
            self.log(f"Symbolic check error: {e}")
            
        return success
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check with steps analysis and expected values.
        Returns: bool success
        """
        success = True
        
        # Check against expected values from steps analysis
        if not self.fuzzy_equal(self.a1, self.expected_a1):
            self.log(f"Trace inconsistency: a1 expected {self.expected_a1}, got {self.a1}")
            success = False
        else:
            self.log(f"a1 matches expected value {self.expected_a1}")
            
        if not self.fuzzy_equal(abs(self.a2), self.expected_a2):
            self.log(f"Trace inconsistency: |a2| expected {self.expected_a2}, got |{self.a2}|")
            success = False
        else:
            self.log(f"|a2| matches expected value {self.expected_a2}")
            
        # Check that using a1 and t_accel gives correct v_final
        v_calc = self.a1 * self.t_accel
        if not self.fuzzy_equal(v_calc, self.v_final):
            self.log(f"Trace inconsistency: a1*t should be {self.v_final}, got {v_calc}")
            success = False
            
        # Check that using a2 and x_brake gives correct final velocity (0)
        # v_final^2 = 30^2 + 2*a2*90 should be ~0
        v2_sq_calc = self.v_final**2 + 2 * self.a2 * self.x_brake
        if not self.fuzzy_equal(v2_sq_calc, 0.0, abs_tol=1.0):  # Allow some tolerance
            self.log(f"Trace inconsistency: v^2 from braking should be ~0, got {v2_sq_calc}")
            success = False
            
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb inputs and check expected trends.
        Returns: bool success
        """
        success = True
        
        # Perturb acceleration time: increase by 10%
        t_perturbed = self.t_accel * 1.1
        
        # Recalculate a1 = v_final / t_perturbed
        a1_new = self.v_final / t_perturbed
        
        # Since a1 = v/t, as t increases, a1 should decrease
        if not a1_new < self.a1:
            self.log(f"Monotonicity failed: a1 should decrease when t increases")
            self.log(f"  Original: t={self.t_accel}, a1={self.a1}")
            self.log(f"  Perturbed: t={t_perturbed}, a1_new={a1_new}")
            success = False
        else:
            self.log(f"Monotonicity OK: a1 decreases when t increases")
            
        # Perturb braking distance: decrease by 10%
        x_perturbed = self.x_brake * 0.9
        
        # Recalculate a2 = -v_final^2 / (2*x_perturbed)
        a2_new = -self.v_final**2 / (2 * x_perturbed)
        
        # Since |a2| = v^2/(2x), as x decreases, |a2| should increase
        if not abs(a2_new) > abs(self.a2):
            self.log(f"Monotonicity failed: |a2| should increase when x decreases")
            self.log(f"  Original: x={self.x_brake}, |a2|={abs(self.a2)}")
            self.log(f"  Perturbed: x={x_perturbed}, |a2_new|={abs(a2_new)}")
            success = False
        else:
            self.log(f"Monotonicity OK: |a2| increases when x decreases")
            
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and return overall success and logs.
        Returns: (success_bool, logs_list)
        """
        self.logs = []  # Reset logs
        
        self.log("=== Physics Checker Starting ===")
        self.log(f"Inputs: v_final={self.v_final}, t_accel={self.t_accel}, x_brake={self.x_brake}")
        self.log(f"Results: a1={self.a1}, a2={self.a2}")
        
        # Run tests
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        overall_success = test1 and test2 and test3
        
        self.log("=== Physics Checker Finished ===")
        self.log(f"Overall success: {overall_success}")
        
        return overall_success, self.logs