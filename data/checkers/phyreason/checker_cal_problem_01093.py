import math
import sympy as sp
from typing import Tuple, List, Dict, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        """
        self.solution = solution_instance
        
        # Constants from problem statement
        self.v0 = 20.0  # m/s
        self.t_braking = 4.0  # s (time to reduce speed by half)
        self.t_target = 10.0  # s (time for second part of problem)
        
        # Extract results SAFELY (prevent AttributeError)
        self.results = {}
        result_keys = ['x', '|a|', 'a', 'v', 't_1']
        
        for key in result_keys:
            # Try to get from solution_instance.attr, then solution_instance.results
            if hasattr(solution_instance, key):
                self.results[key] = getattr(solution_instance, key, None)
            elif hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
                self.results[key] = solution_instance.results.get(key, None)
            else:
                self.results[key] = None
        
        # Extract specific values with fallbacks
        self.x = self.results.get('x', 0.0)
        self.a_mag = self.results.get('|a|', 0.0)
        self.a = self.results.get('a', 0.0)
        self.v = self.results.get('v', 0.0)
        self.t1 = self.results.get('t_1', 0.0)
        
        # Derived constants
        self.v_half = 10.0  # m/s (half of initial speed)
        
        # For symbolic calculations
        self.v0_sym = sp.Symbol('v0', positive=True)
        self.t_sym = sp.Symbol('t', positive=True)
        self.a_sym = sp.Symbol('a', negative=True)
        self.v_sym = sp.Symbol('v', nonnegative=True)
        self.x_sym = sp.Symbol('x', nonnegative=True)
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """
        Verify fundamental physics invariants:
        1. Kinematic equations for uniformly accelerated motion
        2. Consistency between acceleration and magnitude
        3. Stopping time calculation
        """
        logs = []
        success = True
        
        # Test 1: Verify v = v0 + a*t for t=4s (braking to half speed)
        v_calc_4s = self.v0 + self.a * self.t_braking
        if not math.isclose(v_calc_4s, self.v_half, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: v(t=4s) = v0 + a*t gives {v_calc_4s:.2f} m/s, expected {self.v_half:.2f} m/s")
            success = False
        else:
            logs.append(f"PASS: v(t=4s) = {v_calc_4s:.2f} m/s matches expected {self.v_half:.2f} m/s")
        
        # Test 2: Verify |a| = abs(a)
        if not math.isclose(self.a_mag, abs(self.a), rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: |a| = {self.a_mag:.2f} ≠ abs(a) = {abs(self.a):.2f}")
            success = False
        else:
            logs.append(f"PASS: |a| = {self.a_mag:.2f} = abs(a)")
        
        # Test 3: Verify acceleration is negative (braking)
        if self.a >= 0:
            logs.append(f"FAIL: Acceleration should be negative (braking), got a = {self.a:.2f}")
            success = False
        else:
            logs.append(f"PASS: Acceleration is negative (a = {self.a:.2f} m/s²)")
        
        # Test 4: Verify stopping time t1 = -v0 / a
        t1_calc = -self.v0 / self.a
        if not math.isclose(self.t1, t1_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Stopping time t1 = {self.t1:.2f} s, expected -v0/a = {t1_calc:.2f} s")
            success = False
        else:
            logs.append(f"PASS: Stopping time t1 = {self.t1:.2f} s matches -v0/a")
        
        # Test 5: Verify displacement at t1 using average velocity formula
        x_calc_t1 = (self.v0 / 2) * self.t1
        if not math.isclose(self.x, x_calc_t1, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Displacement x = {self.x:.2f} m, expected (v0/2)*t1 = {x_calc_t1:.2f} m")
            success = False
        else:
            logs.append(f"PASS: Displacement x = {self.x:.2f} m matches (v0/2)*t1")
        
        # Test 6: Verify velocity at t_target (should be 0 since t_target > t1)
        if self.t_target > self.t1:
            if not math.isclose(self.v, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: At t={self.t_target}s > t1={self.t1}s, velocity should be 0, got v={self.v:.2f}")
                success = False
            else:
                logs.append(f"PASS: At t={self.t_target}s > t1={self.t1}s, v=0 m/s (car stopped)")
        else:
            v_calc = self.v0 + self.a * self.t_target
            if not math.isclose(self.v, v_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: At t={self.t_target}s, v should be {v_calc:.2f}, got {self.v:.2f}")
                success = False
            else:
                logs.append(f"PASS: Velocity at t={self.t_target}s matches v0 + a*t")
        
        return success, logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """
        Cross-check intermediate results with final answer.
        """
        logs = []
        success = True
        
        # Check step 1: acceleration calculation
        a_calc = (self.v_half - self.v0) / self.t_braking
        if not math.isclose(self.a, a_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Step 1: a should be (10-20)/4 = {a_calc:.2f}, got {self.a:.2f}")
            success = False
        else:
            logs.append(f"PASS: Step 1: a = (10-20)/4 = {self.a:.2f} m/s²")
        
        # Check step 2: stopping time
        t1_calc = -self.v0 / self.a
        if not math.isclose(self.t1, t1_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Step 2: t1 should be -20/a = {t1_calc:.2f}, got {self.t1:.2f}")
            success = False
        else:
            logs.append(f"PASS: Step 2: t1 = -20/a = {self.t1:.2f} s")
        
        # Check step 3 & 5: velocity at 10s
        if self.t_target > self.t1:
            if not math.isclose(self.v, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 3/5: At t=10s > t1={self.t1}s, v should be 0, got {self.v:.2f}")
                success = False
            else:
                logs.append(f"PASS: Step 3/5: At t=10s, v=0 m/s (car already stopped)")
        
        # Check step 4 & 5: displacement
        x_calc = (self.v0 / 2) * self.t1
        if not math.isclose(self.x, x_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Step 4/5: x should be (20/2)*{self.t1} = {x_calc:.2f}, got {self.x:.2f}")
            success = False
        else:
            logs.append(f"PASS: Step 4/5: x = (20/2)*{self.t1} = {self.x:.2f} m")
        
        return success, logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Perturb initial velocity and check trend in stopping time and displacement.
        Then restore original values.
        """
        logs = []
        success = True
        
        # Store original values
        v0_original = self.v0
        a_original = self.a
        t1_original = self.t1
        x_original = self.x
        
        # Test 1: Increase initial velocity by 10%
        v0_perturbed = v0_original * 1.1
        
        # Calculate new stopping time (using same acceleration magnitude)
        t1_perturbed = -v0_perturbed / a_original
        
        # Calculate new displacement at t_target
        if self.t_target > t1_perturbed:
            x_perturbed = (v0_perturbed / 2) * t1_perturbed
        else:
            x_perturbed = v0_perturbed * self.t_target + 0.5 * a_original * self.t_target**2
        
        # Check monotonicity: larger v0 should lead to larger stopping time and displacement
        if not t1_perturbed > t1_original:
            logs.append(f"FAIL: Increasing v0 from {v0_original} to {v0_perturbed} should increase t1, but {t1_perturbed} ≤ {t1_original}")
            success = False
        else:
            logs.append(f"PASS: t1 increases from {t1_original:.2f} to {t1_perturbed:.2f} when v0 increases")
        
        if not x_perturbed > x_original:
            logs.append(f"FAIL: Increasing v0 should increase displacement, but {x_perturbed} ≤ {x_original}")
            success = False
        else:
            logs.append(f"PASS: Displacement increases from {x_original:.2f} to {x_perturbed:.2f} when v0 increases")
        
        # Test 2: Decrease initial velocity by 10%
        v0_perturbed = v0_original * 0.9
        t1_perturbed = -v0_perturbed / a_original
        
        if self.t_target > t1_perturbed:
            x_perturbed = (v0_perturbed / 2) * t1_perturbed
        else:
            x_perturbed = v0_perturbed * self.t_target + 0.5 * a_original * self.t_target**2
        
        if not t1_perturbed < t1_original:
            logs.append(f"FAIL: Decreasing v0 should decrease t1, but {t1_perturbed} ≥ {t1_original}")
            success = False
        else:
            logs.append(f"PASS: t1 decreases from {t1_original:.2f} to {t1_perturbed:.2f} when v0 decreases")
        
        if not x_perturbed < x_original:
            logs.append(f"FAIL: Decreasing v0 should decrease displacement, but {x_perturbed} ≥ {x_original}")
            success = False
        else:
            logs.append(f"PASS: Displacement decreases from {x_original:.2f} to {x_perturbed:.2f} when v0 decreases")
        
        # Restore original values
        self.v0 = v0_original
        self.a = a_original
        self.t1 = t1_original
        self.x = x_original
        
        return success, logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and return overall success and consolidated logs.
        """
        all_logs = ["=== Physics Checker Results ==="]
        overall_success = True
        
        # Run invariant tests
        success, logs = self.test_invariants()
        all_logs.append("\n1. Invariant Tests:")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Run trace consistency tests
        success, logs = self.test_trace_consistency()
        all_logs.append("\n2. Trace Consistency Tests:")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Run monotonicity tests
        success, logs = self.test_monotonicity()
        all_logs.append("\n3. Monotonicity Tests:")
        all_logs.extend(logs)
        if not success:
            overall_success = False
        
        # Summary
        all_logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        
        return overall_success, all_logs