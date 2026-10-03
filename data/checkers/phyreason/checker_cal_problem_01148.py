import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the PhysicsChecker with safe attribute access.
        """
        # Extract input parameters with safe defaults
        self.v0 = getattr(solution_instance, 'v0', 10.0)
        self.theta_deg = getattr(solution_instance, 'theta_deg', 30.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Convert theta to radians
        self.theta = math.radians(self.theta_deg)
        
        # Extract results from solution_instance.results (or as attributes)
        results = getattr(solution_instance, 'results', {})
        
        # Use getattr for each key, with fallback to results dictionary
        self.x = getattr(solution_instance, 'x', results.get('x', 0.0))
        self.v_y = getattr(solution_instance, 'v_y', results.get('v_y', 0.0))
        self.t = getattr(solution_instance, 't', results.get('t', 0.0))
        self.h = getattr(solution_instance, 'h', results.get('h', 0.0))
        self.h_AB = getattr(solution_instance, 'h_AB', results.get('h_AB', 0.0))
        
        # Store logs for reporting
        self.logs = []

    def _log(self, message: str):
        """Helper to append log messages."""
        self.logs.append(message)

    def _is_close(self, a: float, b: float) -> bool:
        """Robust fuzzy comparison with 1% tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

    def test_invariants(self) -> bool:
        """
        Verify fundamental physics invariants:
        1. Perpendicular impact condition.
        2. Kinematic equations for vertical and horizontal motion.
        3. Energy conservation.
        """
        success = True
        
        # 1. Perpendicular impact condition: |v_y| = v0 * cot(theta)
        cot_theta = 1.0 / math.tan(self.theta)
        expected_vy_mag = self.v0 * cot_theta
        if not self._is_close(self.v_y, expected_vy_mag):
            self._log(f"Perpendicular condition failed: |v_y| = {self.v_y}, expected {expected_vy_mag}")
            success = False
        else:
            self._log("Perpendicular condition satisfied.")
        
        # 2. Time from vertical motion: t = v_y / g (since initial vertical velocity is 0)
        expected_t = self.v_y / self.g
        if not self._is_close(self.t, expected_t):
            self._log(f"Time consistency failed: t = {self.t}, expected {expected_t}")
            success = False
        else:
            self._log("Time consistency satisfied.")
        
        # 3. Vertical displacement: h = 0.5 * g * t^2
        expected_h = 0.5 * self.g * self.t**2
        if not self._is_close(self.h, expected_h):
            self._log(f"Vertical displacement failed: h = {self.h}, expected {expected_h}")
            success = False
        else:
            self._log("Vertical displacement consistency satisfied.")
        
        # 4. Horizontal displacement: x = v0 * t
        expected_x = self.v0 * self.t
        if not self._is_close(self.x, expected_x):
            self._log(f"Horizontal displacement failed: x = {self.x}, expected {expected_x}")
            success = False
        else:
            self._log("Horizontal displacement consistency satisfied.")
        
        # 5. Geometric relationship: h_AB = h + x * tan(theta)
        expected_h_AB = self.h + self.x * math.tan(self.theta)
        if not self._is_close(self.h_AB, expected_h_AB):
            self._log(f"Height AB failed: h_AB = {self.h_AB}, expected {expected_h_AB}")
            success = False
        else:
            self._log("Height AB consistency satisfied.")
        
        # 6. Energy conservation (optional but insightful)
        # Initial total mechanical energy (relative to point B): E_i = m*g*h_AB + 0.5*m*v0^2
        # Final total mechanical energy at impact: E_f = 0.5*m*(v0^2 + v_y^2) + m*g*y_impact
        # where y_impact = x * tan(theta) = self.h_AB - self.h
        y_impact = self.h_AB - self.h
        # Energy per unit mass (mass cancels out)
        E_i_per_m = self.g * self.h_AB + 0.5 * self.v0**2
        E_f_per_m = 0.5 * (self.v0**2 + self.v_y**2) + self.g * y_impact
        if not self._is_close(E_i_per_m, E_f_per_m):
            self._log(f"Energy conservation warning: E_i/m = {E_i_per_m}, E_f/m = {E_f_per_m}")
            # This might be a warning rather than a failure due to rounding, but we note it.
        else:
            self._log("Energy conservation satisfied.")
        
        return success

    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with each other and final answers.
        """
        success = True
        
        # Recompute h from t and g
        h_from_t = 0.5 * self.g * self.t**2
        if not self._is_close(self.h, h_from_t):
            self._log(f"Trace h inconsistent: stored {self.h}, computed from t {h_from_t}")
            success = False
        
        # Recompute x from v0 and t
        x_from_t = self.v0 * self.t
        if not self._is_close(self.x, x_from_t):
            self._log(f"Trace x inconsistent: stored {self.x}, computed from t {x_from_t}")
            success = False
        
        # Recompute v_y from t and g
        v_y_from_t = self.g * self.t
        if not self._is_close(self.v_y, v_y_from_t):
            self._log(f"Trace v_y inconsistent: stored {self.v_y}, computed from t {v_y_from_t}")
            success = False
        
        # Recompute h_AB from h and x
        h_AB_from_geo = self.h + self.x * math.tan(self.theta)
        if not self._is_close(self.h_AB, h_AB_from_geo):
            self._log(f"Trace h_AB inconsistent: stored {self.h_AB}, computed from h and x {h_AB_from_geo}")
            success = False
        
        if success:
            self._log("All trace consistency checks passed.")
        return success

    def test_monotonicity(self) -> bool:
        """
        Perturb v0 by +1% and verify that results increase as expected.
        Uses the known formulas to predict new values.
        """
        success = True
        perturb_factor = 1.01
        v0_perturbed = self.v0 * perturb_factor
        
        # Compute expected results for perturbed v0 using the derived formulas
        cot_theta = 1.0 / math.tan(self.theta)
        t_perturbed = (v0_perturbed * cot_theta) / self.g
        h_perturbed = 0.5 * self.g * t_perturbed**2
        x_perturbed = v0_perturbed * t_perturbed
        h_AB_perturbed = h_perturbed + x_perturbed * math.tan(self.theta)
        
        # Check monotonic increase (all quantities are positive and should increase with v0)
        if not (t_perturbed > self.t):
            self._log(f"Monotonicity failed: t does not increase with v0")
            success = False
        if not (h_perturbed > self.h):
            self._log(f"Monotonicity failed: h does not increase with v0")
            success = False
        if not (x_perturbed > self.x):
            self._log(f"Monotonicity failed: x does not increase with v0")
            success = False
        if not (h_AB_perturbed > self.h_AB):
            self._log(f"Monotonicity failed: h_AB does not increase with v0")
            success = False
        
        if success:
            self._log("Monotonicity check passed: all quantities increase with v0.")
        return success

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all tests and return overall success and logs.
        """
        self.logs.clear()
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for test_name, test_func in tests:
            self._log(f"Running {test_name}...")
            try:
                if not test_func():
                    overall_success = False
                    self._log(f"{test_name} failed.")
            except Exception as e:
                self._log(f"{test_name} raised exception: {e}")
                overall_success = False
        
        self._log(f"Overall test result: {'PASS' if overall_success else 'FAIL'}")
        return overall_success, self.logs