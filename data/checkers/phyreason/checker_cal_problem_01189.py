import math
import traceback
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize with safe attribute access.
        """
        self.solution = solution_instance
        
        # Extract inputs with safe fallbacks
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'results', {}).get('m', 1.0))
        self.alpha = getattr(solution_instance, 'alpha', getattr(solution_instance, 'results', {}).get('alpha', 0.0))
        self.g = getattr(solution_instance, 'g', getattr(solution_instance, 'results', {}).get('g', 9.8))
        
        # Extract results with safe fallbacks
        results = getattr(solution_instance, 'results', {})
        self.T1 = results.get('T1', results.get('T_1', getattr(solution_instance, 'T1', 0.0)))
        self.T2 = results.get('T2', results.get('T_2', getattr(solution_instance, 'T2', 0.0)))
        self.T_res = results.get('T', 0.0)  # Resultant force magnitude
        self.res_direction = results.get('N/A', '')  # Resultant force direction
        
        self.logs = []
        
    def _log(self, msg: str, success: bool):
        """Helper to log messages."""
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {msg}")
        return success
        
    def test_invariants(self) -> bool:
        """
        Verify force equilibrium conditions.
        Returns True if all invariant checks pass.
        """
        try:
            # Check 1: Newton's 1st Law - Net force components must be zero
            # x-component: T2 = T1 * sin(alpha)
            Fx_expected = self.T1 * math.sin(self.alpha)
            Fx_check = math.isclose(self.T2, Fx_expected, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"x-equilibrium: T2={self.T2:.3f}, T1*sin(α)={Fx_expected:.3f}", Fx_check)
            
            # y-component: T1 * cos(alpha) = mg
            Fy_expected = self.m * self.g
            Fy_actual = self.T1 * math.cos(self.alpha)
            Fy_check = math.isclose(Fy_actual, Fy_expected, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"y-equilibrium: T1*cos(α)={Fy_actual:.3f}, mg={Fy_expected:.3f}", Fy_check)
            
            # Check 2: Resultant force magnitude and direction
            if self.T_res > 0:
                T_res_calculated = math.sqrt(self.T1**2 + self.T2**2 + 2*self.T1*self.T2*math.cos(math.pi/2 + self.alpha))
                # Actually, simpler: resultant = mg from step_1
                T_res_expected = self.m * self.g
                res_mag_check = math.isclose(self.T_res, T_res_expected, rel_tol=1e-2, abs_tol=1e-2)
                self._log(f"Resultant magnitude: T={self.T_res:.3f}, mg={T_res_expected:.3f}", res_mag_check)
            
            # Check direction is "vertically upward"
            if self.res_direction:
                dir_check = "vertically upward" in str(self.res_direction).lower()
                self._log(f"Resultant direction: {self.res_direction}", dir_check)
            
            return Fx_check and Fy_check
            
        except Exception as e:
            self._log(f"Invariant test failed with error: {str(e)}", False)
            return False
            
    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answers.
        """
        try:
            # From step_2: T1 = mg/cos(alpha)
            T1_expected = self.m * self.g / math.cos(self.alpha)
            T1_check = math.isclose(self.T1, T1_expected, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"T1 consistency: {self.T1:.3f} vs mg/cos(α)={T1_expected:.3f}", T1_check)
            
            # From step_2: T2 = mg * tan(alpha)
            T2_expected = self.m * self.g * math.tan(self.alpha)
            T2_check = math.isclose(self.T2, T2_expected, rel_tol=1e-2, abs_tol=1e-2)
            self._log(f"T2 consistency: {self.T2:.3f} vs mg*tan(α)={T2_expected:.3f}", T2_check)
            
            return T1_check and T2_check
            
        except Exception as e:
            self._log(f"Trace consistency test failed: {str(e)}", False)
            return False
            
    def test_monotonicity(self) -> bool:
        """
        Perturb mass and verify tension trends.
        """
        original_m = self.m
        
        try:
            # Perturb mass by +10%
            perturbed_m = original_m * 1.10
            
            # Calculate expected tensions with perturbed mass
            T1_perturbed = perturbed_m * self.g / math.cos(self.alpha)
            T2_perturbed = perturbed_m * self.g * math.tan(self.alpha)
            
            # Check monotonic increase (both tensions should increase by 10%)
            T1_ratio = T1_perturbed / self.T1
            T2_ratio = T2_perturbed / self.T2
            
            T1_mono = math.isclose(T1_ratio, 1.10, rel_tol=1e-2, abs_tol=1e-2)
            T2_mono = math.isclose(T2_ratio, 1.10, rel_tol=1e-2, abs_tol=1e-2)
            
            self._log(f"T1 monotonicity: ratio={T1_ratio:.3f} (expected ~1.100)", T1_mono)
            self._log(f"T2 monotonicity: ratio={T2_ratio:.3f} (expected ~1.100)", T2_mono)
            
            # Restore original mass (conceptually)
            self.m = original_m
            
            return T1_mono and T2_mono
            
        except Exception as e:
            self._log(f"Monotonicity test failed: {str(e)}", False)
            self.m = original_m  # Ensure restoration even on error
            return False
            
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all tests and return overall success with logs.
        """
        self.logs.clear()
        
        try:
            success = True
            
            # Run individual tests
            if not self.test_invariants():
                success = False
            if not self.test_trace_consistency():
                success = False
            if not self.test_monotonicity():
                success = False
                
            # Summary log
            if success:
                self.logs.append("SUMMARY: All physics checks passed")
            else:
                self.logs.append("SUMMARY: Some physics checks failed")
                
            return success, self.logs
            
        except Exception as e:
            error_msg = f"Critical error in run_all_tests: {traceback.format_exc()}"
            self.logs.append(error_msg)
            return False, self.logs