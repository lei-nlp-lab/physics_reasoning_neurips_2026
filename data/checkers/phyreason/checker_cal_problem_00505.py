import math
import sympy as sp
from typing import Tuple, List, Any, Optional

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Initialize the checker by extracting all necessary inputs and results.
        """
        # Extract results safely from the solution instance's results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Input parameters - use default values of 1.0 if not found
        self.m = self.results.get('m', 1.0)
        self.q = self.results.get('q', 1.0)
        self.v0 = self.results.get('v0', 1.0)
        self.R = self.results.get('R', 1.0)
        
        # Computed results - use None if not found
        self.E = self.results.get('E', None)
        self.a = self.results.get('a', None)
        self.delta_E_k1 = self.results.get('delta_E_k1', None)
        self.delta_E_k2 = self.results.get('delta_E_k2', None)
        self.y = self.results.get('y', None)
        self.X_c1 = self.results.get('X_c1', None)
        self.X_c2 = self.results.get('X_c2', None)
        self.v_c1 = self.results.get('v_c1', None)
        self.v_c2 = self.results.get('v_c2', None)
        
        # Constants
        self.sqrt2 = math.sqrt(2)
        self.sqrt3 = math.sqrt(3)
        self.sqrt6 = math.sqrt(6)
        
        # Logs
        self.logs: List[str] = []
    
    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Robust numerical comparison with tolerance."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> bool:
        """Test fundamental physics invariants."""
        success = True
        
        # Check 1: Newton's Second Law: qE = m a
        if all(v is not None for v in [self.q, self.E, self.m, self.a]):
            lhs = self.q * self.E
            rhs = self.m * self.a
            if not self._is_close(lhs, rhs):
                self.logs.append(f"Invariant violation: qE = {lhs}, m a = {rhs}")
                success = False
        else:
            self.logs.append("Skipping Newton's Law check due to missing parameters.")
        
        # Check 2: Kinematics for point B: a = 2 v0^2 / R (from t1 = R/v0 and R = 0.5 a t1^2)
        if all(v is not None for v in [self.a, self.v0, self.R]):
            expected_a = 2 * self.v0**2 / self.R
            if not self._is_close(self.a, expected_a):
                self.logs.append(f"Invariant violation: a = {self.a}, expected 2 v0^2/R = {expected_a}")
                success = False
        else:
            self.logs.append("Skipping kinematics check for point B due to missing parameters.")
        
        # Check 3: Work-energy theorem for point B: ΔE_k1 = q E R
        if all(v is not None for v in [self.delta_E_k1, self.q, self.E, self.R]):
            expected_work_b = self.q * self.E * self.R  # vertical displacement magnitude is R
            if not self._is_close(self.delta_E_k1, expected_work_b):
                self.logs.append(f"Invariant violation: ΔE_k1 = {self.delta_E_k1}, expected q E R = {expected_work_b}")
                success = False
        else:
            self.logs.append("Skipping work-energy check for point B due to missing parameters.")
        
        # Check 4: Work-energy theorem for point C: ΔE_k2 = q E |y|, with |y| = R/2
        if all(v is not None for v in [self.delta_E_k2, self.q, self.E, self.y]):
            # y should be negative, so take absolute value for displacement magnitude
            displacement_c = abs(self.y) if self.y is not None else self.R/2
            expected_work_c = self.q * self.E * displacement_c
            if not self._is_close(self.delta_E_k2, expected_work_c):
                self.logs.append(f"Invariant violation: ΔE_k2 = {self.delta_E_k2}, expected q E |y| = {expected_work_c}")
                success = False
        else:
            self.logs.append("Skipping work-energy check for point C due to missing parameters.")
        
        # Check 5: Ratio of kinetic energy increments should be 2 (since |y_B|/|y_C| = R/(R/2) = 2)
        if all(v is not None for v in [self.delta_E_k1, self.delta_E_k2]):
            if not self._is_close(self.delta_E_k1 / self.delta_E_k2, 2.0):
                self.logs.append(f"Invariant violation: ΔE_k1/ΔE_k2 = {self.delta_E_k1/self.delta_E_k2}, expected 2.0")
                success = False
        
        # Check 6: Circle geometry for point C: (X_c - R)^2 + y^2 = R^2
        if all(v is not None for v in [self.X_c1, self.y, self.R]):
            lhs1 = (self.X_c1 - self.R)**2 + self.y**2
            if not self._is_close(lhs1, self.R**2):
                self.logs.append(f"Circle violation for C1: (X_c1-R)^2+y^2 = {lhs1}, expected R^2 = {self.R**2}")
                success = False
        if all(v is not None for v in [self.X_c2, self.y, self.R]):
            lhs2 = (self.X_c2 - self.R)**2 + self.y**2
            if not self._is_close(lhs2, self.R**2):
                self.logs.append(f"Circle violation for C2: (X_c2-R)^2+y^2 = {lhs2}, expected R^2 = {self.R**2}")
                success = False
        
        # Check 7: Consistency of y with R/2
        if self.y is not None and self.R is not None:
            if not self._is_close(self.y, -self.R/2):  # y should be negative
                self.logs.append(f"Invariant violation: y = {self.y}, expected -R/2 = {-self.R/2}")
                success = False
        
        # Check 8: Formula for electric field: E = 2 m v0^2 / (q R)
        if all(v is not None for v in [self.E, self.m, self.v0, self.q, self.R]):
            expected_E = 2 * self.m * self.v0**2 / (self.q * self.R)
            if not self._is_close(self.E, expected_E):
                self.logs.append(f"Invariant violation: E = {self.E}, expected 2 m v0^2/(q R) = {expected_E}")
                success = False
        
        # Check 9: Formulas for v_c1 and v_c2
        if all(v is not None for v in [self.v_c1, self.v0]):
            expected_v_c1 = (2*self.sqrt2 + self.sqrt6) / 2 * self.v0
            if not self._is_close(self.v_c1, expected_v_c1):
                self.logs.append(f"Invariant violation: v_c1 = {self.v_c1}, expected (2√2+√6)/2 v0 = {expected_v_c1}")
                success = False
        if all(v is not None for v in [self.v_c2, self.v0]):
            expected_v_c2 = (2*self.sqrt2 - self.sqrt6) / 2 * self.v0
            if not self._is_close(self.v_c2, expected_v_c2):
                self.logs.append(f"Invariant violation: v_c2 = {self.v_c2}, expected (2√2-√6)/2 v0 = {expected_v_c2}")
                success = False
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Check that intermediate results are consistent with final answers."""
        success = True
        
        # Trace 1: From a and v0, compute t1 = R/v0, then check that R = 0.5 a t1^2
        if all(v is not None for v in [self.a, self.v0, self.R]):
            t1 = self.R / self.v0
            vertical_from_kinematics = 0.5 * self.a * t1**2
            if not self._is_close(vertical_from_kinematics, self.R):
                self.logs.append(f"Trace inconsistency: 0.5 a t1^2 = {vertical_from_kinematics}, expected R = {self.R}")
                success = False
        
        # Trace 2: For point C, from y = R/2 and a, compute t2 = sqrt(R/a)
        if all(v is not None for v in [self.a, self.R]):
            t2 = math.sqrt(self.R / self.a)  # since |y| = R/2 = 0.5 a t2^2
            # Then horizontal displacements: X_c1 = v_c1 * t2, X_c2 = v_c2 * t2
            if self.v_c1 is not None and self.X_c1 is not None:
                x_from_v1 = self.v_c1 * t2
                if not self._is_close(x_from_v1, self.X_c1):
                    self.logs.append(f"Trace inconsistency: v_c1 * t2 = {x_from_v1}, expected X_c1 = {self.X_c1}")
                    success = False
            if self.v_c2 is not None and self.X_c2 is not None:
                x_from_v2 = self.v_c2 * t2
                if not self._is_close(x_from_v2, self.X_c2):
                    self.logs.append(f"Trace inconsistency: v_c2 * t2 = {x_from_v2}, expected X_c2 = {self.X_c2}")
                    success = False
        
        # Trace 3: Check that the computed coordinates C are consistent with X_c1, X_c2, and y
        if self.y is not None and self.X_c1 is not None and self.X_c2 is not None:
            # Expected coordinates from step 9
            coord1 = (self.X_c1, self.y)
            coord2 = (self.X_c2, self.y)
            # Check that X_c1 and X_c2 match the formulas (2±√3)R/2
            if self.R is not None:
                expected_X_c1 = (2 + self.sqrt3) * self.R / 2
                expected_X_c2 = (2 - self.sqrt3) * self.R / 2
                if not self._is_close(self.X_c1, expected_X_c1):
                    self.logs.append(f"Trace inconsistency: X_c1 = {self.X_c1}, expected (2+√3)R/2 = {expected_X_c1}")
                    success = False
                if not self._is_close(self.X_c2, expected_X_c2):
                    self.logs.append(f"Trace inconsistency: X_c2 = {self.X_c2}, expected (2-√3)R/2 = {expected_X_c2}")
                    success = False
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb v0 and verify that E, v_c1, v_c2 change monotonically and proportionally.
        """
        success = True
        
        # Require all necessary parameters
        required = [self.m, self.q, self.v0, self.R, self.E, self.X_c1, self.X_c2, self.v_c1, self.v_c2]
        if any(v is None for v in required):
            self.logs.append("Skipping monotonicity test due to missing parameters.")
            return False
        
        original_v0 = self.v0
        perturbed_v0 = original_v0 * 1.01  # Increase by 1%
        
        # Recompute using physics formulas (independent of solution methods)
        # New acceleration: a = 2 v0^2 / R
        new_a = 2 * perturbed_v0**2 / self.R
        
        # New electric field: E = 2 m v0^2 / (q R)
        new_E = 2 * self.m * perturbed_v0**2 / (self.q * self.R)
        
        # New time for point C: t2 = sqrt(R / a) = R / (sqrt(2) * v0)
        new_t2 = self.R / (math.sqrt(2) * perturbed_v0)
        
        # New horizontal velocities: v_c = X_c / t2
        new_v_c1 = self.X_c1 / new_t2
        new_v_c2 = self.X_c2 / new_t2
        
        # Check monotonicity: if v0 increases, E, v_c1, v_c2 should increase
        if not new_E > self.E:
            self.logs.append(f"Monotonicity violation: E did not increase when v0 increased. Original E={self.E}, new E={new_E}")
            success = False
        if not new_v_c1 > self.v_c1:
            self.logs.append(f"Monotonicity violation: v_c1 did not increase when v0 increased. Original v_c1={self.v_c1}, new v_c1={new_v_c1}")
            success = False
        if not new_v_c2 > self.v_c2:
            self.logs.append(f"Monotonicity violation: v_c2 did not increase when v0 increased. Original v_c2={self.v_c2}, new v_c2={new_v_c2}")
            success = False
        
        # Check proportionality (scaling factors)
        v0_ratio = perturbed_v0 / original_v0
        expected_E_ratio = v0_ratio**2
        actual_E_ratio = new_E / self.E
        if not self._is_close(actual_E_ratio, expected_E_ratio):
            self.logs.append(f"Scaling violation for E: actual ratio={actual_E_ratio}, expected {expected_E_ratio}")
            success = False
        
        expected_v_ratio = v0_ratio
        actual_v_c1_ratio = new_v_c1 / self.v_c1
        if not self._is_close(actual_v_c1_ratio, expected_v_ratio):
            self.logs.append(f"Scaling violation for v_c1: actual ratio={actual_v_c1_ratio}, expected {expected_v_ratio}")
            success = False
        
        actual_v_c2_ratio = new_v_c2 / self.v_c2
        if not self._is_close(actual_v_c2_ratio, expected_v_ratio):
            self.logs.append(f"Scaling violation for v_c2: actual ratio={actual_v_c2_ratio}, expected {expected_v_ratio}")
            success = False
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success and logs."""
        self.logs.clear()
        
        # Run each test
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        overall_success = test1 and test2 and test3
        
        # Summary log
        if overall_success:
            self.logs.append("All tests passed.")
        else:
            self.logs.append("Some tests failed.")
        
        return overall_success, self.logs