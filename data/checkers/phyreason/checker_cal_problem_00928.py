import math
import sympy as sp
from typing import List, Tuple, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        Uses getattr to avoid AttributeError in sandbox environment.
        """
        # Extract inputs safely
        self.U = getattr(solution_instance, 'U', getattr(solution_instance, 'inputs', {}).get('U', 1.0))
        self.B = getattr(solution_instance, 'B', getattr(solution_instance, 'inputs', {}).get('B', 1.0))
        self.d = getattr(solution_instance, 'd', getattr(solution_instance, 'inputs', {}).get('d', 1.0))
        
        # Extract results safely from results dictionary
        results = getattr(solution_instance, 'results', {})
        self.fracqm = results.get('fracqm', 0.0)      # q/m
        self.r = results.get('r', 0.0)               # radius
        self.S_2 = results.get('S_2', 0.0)           # straight path length
        self.t = results.get('t', 0.0)               # total time
        self.v_0 = results.get('v_0', 0.0)           # initial velocity
        self.S_1 = results.get('S_1', 0.0)           # arc length
        
        # Store for monotonicity test perturbation
        self._original_U = self.U
        self._original_d = self.d
        
        # Tolerance for numerical comparisons
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Logs
        self.logs = []
        
    def _log(self, message: str, success: bool = True):
        """Helper to log messages with success/failure indicator."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"[{status}] {message}")
        return success
        
    def test_invariants(self) -> bool:
        """
        Verify fundamental physics invariants using symbolic and numerical checks.
        Returns: True if all invariants pass, False otherwise.
        """
        success = True
        
        # 1. Work-energy theorem: v0 = sqrt(2*(q/m)*U)
        try:
            v0_expected = math.sqrt(2 * abs(self.fracqm) * self.U)  # abs for safety
            if not math.isclose(self.v_0, v0_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                success = False
                self._log(f"Work-energy invariant failed: v0={self.v_0}, expected sqrt(2*(q/m)*U)={v0_expected}", False)
            else:
                self._log(f"Work-energy invariant passed: v0 ≈ sqrt(2*(q/m)*U)")
        except Exception as e:
            success = False
            self._log(f"Work-energy invariant error: {e}", False)
        
        # 2. Lorentz force as centripetal force: r = m*v0/(q*B) = v0/((q/m)*B)
        try:
            if abs(self.fracqm) > 1e-12:  # Avoid division by zero
                r_expected = self.v_0 / (abs(self.fracqm) * self.B)
                if not math.isclose(self.r, r_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    success = False
                    self._log(f"Lorentz force invariant failed: r={self.r}, expected v0/((q/m)*B)={r_expected}", False)
                else:
                    self._log(f"Lorentz force invariant passed: r ≈ v0/((q/m)*B)")
        except Exception as e:
            success = False
            self._log(f"Lorentz force invariant error: {e}", False)
        
        # 3. Geometric relationship: r = (sqrt(2)/2) * d
        try:
            r_geom_expected = (math.sqrt(2) / 2) * self.d
            if not math.isclose(self.r, r_geom_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                success = False
                self._log(f"Geometric invariant failed: r={self.r}, expected (√2/2)*d={r_geom_expected}", False)
            else:
                self._log(f"Geometric invariant passed: r ≈ (√2/2)*d")
        except Exception as e:
            success = False
            self._log(f"Geometric invariant error: {e}", False)
        
        # 4. Time consistency: t = (S1 + S2)/v0 (both segments at constant speed v0)
        try:
            if self.v_0 > 1e-12:  # Avoid division by zero
                t_expected = (self.S_1 + self.S_2) / self.v_0
                if not math.isclose(self.t, t_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    success = False
                    self._log(f"Time consistency invariant failed: t={self.t}, expected (S1+S2)/v0={t_expected}", False)
                else:
                    self._log(f"Time consistency invariant passed: t ≈ (S1+S2)/v0")
        except Exception as e:
            success = False
            self._log(f"Time consistency invariant error: {e}", False)
        
        return success
        
    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answers and steps analysis.
        Returns: True if all cross-checks pass, False otherwise.
        """
        success = True
        
        # 1. Check specific charge: q/m = 4U/(d^2*B^2)
        try:
            fracqm_expected = 4 * self.U / (self.d**2 * self.B**2)
            if not math.isclose(self.fracqm, fracqm_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                success = False
                self._log(f"Specific charge trace failed: q/m={self.fracqm}, expected 4U/(d^2*B^2)={fracqm_expected}", False)
            else:
                self._log(f"Specific charge trace passed: q/m ≈ 4U/(d^2*B^2)")
        except Exception as e:
            success = False
            self._log(f"Specific charge trace error: {e}", False)
        
        # 2. Check time: t = (π/8 + √3/12) * (d^2 * B / U)
        try:
            t_expected = (math.pi/8 + math.sqrt(3)/12) * (self.d**2 * self.B / self.U)
            if not math.isclose(self.t, t_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                success = False
                self._log(f"Time trace failed: t={self.t}, expected (π/8+√3/12)*(d^2*B/U)={t_expected}", False)
            else:
                self._log(f"Time trace passed: t ≈ (π/8+√3/12)*(d^2*B/U)")
        except Exception as e:
            success = False
            self._log(f"Time trace error: {e}", False)
        
        # 3. Check arc length S1 = (√2/4)*π*d
        try:
            S1_expected = (math.sqrt(2)/4) * math.pi * self.d
            if not math.isclose(self.S_1, S1_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                success = False
                self._log(f"Arc length trace failed: S1={self.S_1}, expected (√2/4)*π*d={S1_expected}", False)
            else:
                self._log(f"Arc length trace passed: S1 ≈ (√2/4)*π*d")
        except Exception as e:
            success = False
            self._log(f"Arc length trace error: {e}", False)
        
        # 4. Check straight path S2 = (√6/6)*d = r*tan(30°)
        try:
            S2_expected = (math.sqrt(6)/6) * self.d
            if not math.isclose(self.S_2, S2_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                success = False
                self._log(f"Straight path trace failed: S2={self.S_2}, expected (√6/6)*d={S2_expected}", False)
            else:
                self._log(f"Straight path trace passed: S2 ≈ (√6/6)*d")
            
            # Alternative check: S2 = r * tan(30°)
            S2_alt_expected = self.r * math.tan(math.radians(30))
            if not math.isclose(self.S_2, S2_alt_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                success = False
                self._log(f"Straight path alternative trace failed: S2={self.S_2}, expected r*tan(30°)={S2_alt_expected}", False)
        except Exception as e:
            success = False
            self._log(f"Straight path trace error: {e}", False)
        
        return success
        
    def test_monotonicity(self) -> bool:
        """
        Perturb inputs and check monotonic trends.
        Returns: True if all monotonicity tests pass, False otherwise.
        """
        success = True
        
        # Test 1: Increase U by 10%, q/m should increase (proportional relationship)
        try:
            U_perturbed = self._original_U * 1.1
            fracqm_perturbed = 4 * U_perturbed / (self.d**2 * self.B**2)
            if not fracqm_perturbed > self.fracqm:
                success = False
                self._log(f"Monotonicity test 1 failed: q/m should increase when U increases", False)
            else:
                self._log(f"Monotonicity test 1 passed: q/m ∝ U")
        except Exception as e:
            success = False
            self._log(f"Monotonicity test 1 error: {e}", False)
        
        # Test 2: Increase d by 10%, q/m should decrease (inverse square relationship)
        try:
            d_perturbed = self._original_d * 1.1
            fracqm_perturbed = 4 * self.U / (d_perturbed**2 * self.B**2)
            if not fracqm_perturbed < self.fracqm:
                success = False
                self._log(f"Monotonicity test 2 failed: q/m should decrease when d increases", False)
            else:
                self._log(f"Monotonicity test 2 passed: q/m ∝ 1/d^2")
        except Exception as e:
            success = False
            self._log(f"Monotonicity test 2 error: {e}", False)
        
        # Test 3: Increase U by 10%, time t should decrease (t ∝ 1/U)
        try:
            U_perturbed = self._original_U * 1.1
            t_perturbed = (math.pi/8 + math.sqrt(3)/12) * (self.d**2 * self.B / U_perturbed)
            if not t_perturbed < self.t:
                success = False
                self._log(f"Monotonicity test 3 failed: t should decrease when U increases", False)
            else:
                self._log(f"Monotonicity test 3 passed: t ∝ 1/U")
        except Exception as e:
            success = False
            self._log(f"Monotonicity test 3 error: {e}", False)
        
        # Test 4: Increase d by 10%, time t should increase (t ∝ d^2)
        try:
            d_perturbed = self._original_d * 1.1
            t_perturbed = (math.pi/8 + math.sqrt(3)/12) * (d_perturbed**2 * self.B / self.U)
            if not t_perturbed > self.t:
                success = False
                self._log(f"Monotonicity test 4 failed: t should increase when d increases", False)
            else:
                self._log(f"Monotonicity test 4 passed: t ∝ d^2")
        except Exception as e:
            success = False
            self._log(f"Monotonicity test 4 error: {e}", False)
        
        # Test 5: Increase B by 10%, radius r should decrease (r ∝ 1/B)
        try:
            B_perturbed = self.B * 1.1
            r_perturbed = self.v_0 / (abs(self.fracqm) * B_perturbed)
            if not r_perturbed < self.r:
                success = False
                self._log(f"Monotonicity test 5 failed: r should decrease when B increases", False)
            else:
                self._log(f"Monotonicity test 5 passed: r ∝ 1/B")
        except Exception as e:
            success = False
            self._log(f"Monotonicity test 5 error: {e}", False)
        
        return success
        
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all tests and return success status with logs.
        Returns: (success_bool, logs_list)
        """
        self.logs = []
        self.logs.append("=== PhysicsChecker starting ===")
        
        # Run tests
        invariant_success = self.test_invariants()
        trace_success = self.test_trace_consistency()
        monotonicity_success = self.test_monotonicity()
        
        # Overall success
        overall_success = invariant_success and trace_success and monotonicity_success
        
        # Summary
        self.logs.append("=== Summary ===")
        self.logs.append(f"Invariants test: {'PASS' if invariant_success else 'FAIL'}")
        self.logs.append(f"Trace consistency test: {'PASS' if trace_success else 'FAIL'}")
        self.logs.append(f"Monotonicity test: {'PASS' if monotonicity_success else 'FAIL'}")
        self.logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, self.logs

# Example usage (commented out for safety):
# if __name__ == "__main__":
#     class MockSolution:
#         def __init__(self):
#             self.inputs = {'U': 100.0, 'B': 0.5, 'd': 0.1}
#             self.results = {
#                 'fracqm': 160000.0,  # 4*100/(0.1^2*0.5^2) = 160000
#                 'r': 0.070710678,    # sqrt(2)/2 * 0.1 = 0.070710678
#                 'S_2': 0.040824829,  # sqrt(6)/6 * 0.1 = 0.040824829
#                 't': 0.000355902,    # (π/8+√3/12)*(0.1^2*0.5/100) = 0.000355902
#                 'v_0': 5656.854249,  # sqrt(2*160000*100) = 5656.854249
#                 'S_1': 0.055581728   # sqrt(2)/4 * π * 0.1 = 0.055581728
#             }
#     
#     solution = MockSolution()
#     checker = PhysicsChecker(solution)
#     success, logs = checker.run_all_tests()
#     for log in logs:
#         print(log)