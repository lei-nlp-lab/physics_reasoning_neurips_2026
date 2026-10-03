import math
import traceback
from typing import Tuple, List, Any, Optional

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Initialize the physics checker with safe attribute access.
        """
        self.solution = solution_instance
        
        # Extract inputs with safe defaults
        self.T_A = getattr(solution_instance, 'T_A', 1.0)
        self.T_B = getattr(solution_instance, 'T_B', 8.0)
        
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract computed quantities with safe access
        self.omega_ratio = self.results.get('omega_A/omega_B', None)
        self.radius_ratio = self.results.get('r_A/r_B', None)
        self.accel_ratio = self.results.get('a_A/a_B', None)
        self.k = self.results.get('k', None)
        
        # Log storage
        self.logs = []
    
    def _log(self, message: str, status: str = "INFO"):
        """Helper to log messages."""
        self.logs.append(f"[{status}] {message}")
    
    def test_invariants(self) -> bool:
        """
        Verify fundamental physics invariants using numeric checks only.
        """
        try:
            self._log("Testing physics invariants...")
            all_passed = True
            
            # 1. Kepler's Third Law: T² ∝ r³ (for same central mass)
            # Numerical check only
            expected_radius_ratio = (self.T_A / self.T_B) ** (2/3)
            if self.radius_ratio is not None:
                if not math.isclose(self.radius_ratio, expected_radius_ratio, rel_tol=0.02, abs_tol=0.02):
                    self._log(f"Radius ratio mismatch: got {self.radius_ratio}, expected {expected_radius_ratio:.4f}", "FAIL")
                    all_passed = False
                else:
                    self._log(f"Radius ratio correct: {self.radius_ratio} ≈ {expected_radius_ratio:.4f}", "PASS")
            
            # 2. Angular velocity vs period: ω = 2π/T
            expected_omega_ratio = self.T_B / self.T_A  # ω_A/ω_B = T_B/T_A
            if self.omega_ratio is not None:
                if not math.isclose(self.omega_ratio, expected_omega_ratio, rel_tol=0.02, abs_tol=0.02):
                    self._log(f"Angular velocity ratio mismatch: got {self.omega_ratio}, expected {expected_omega_ratio}", "FAIL")
                    all_passed = False
                else:
                    self._log(f"Angular velocity ratio correct: {self.omega_ratio} ≈ {expected_omega_ratio}", "PASS")
            
            # 3. Centripetal acceleration: a = ω²r
            # Check consistency between ω ratio, r ratio, and a ratio
            if all(v is not None for v in [self.omega_ratio, self.radius_ratio, self.accel_ratio]):
                # Derived from a_A/a_B = (ω_A/ω_B)² * (r_A/r_B)
                derived_accel_ratio = (self.omega_ratio ** 2) * self.radius_ratio
                if not math.isclose(self.accel_ratio, derived_accel_ratio, rel_tol=0.02, abs_tol=0.02):
                    self._log(f"Acceleration ratio inconsistency: got {self.accel_ratio}, derived {derived_accel_ratio:.4f}", "FAIL")
                    all_passed = False
                else:
                    self._log(f"Acceleration ratios are consistent: {self.accel_ratio} ≈ {derived_accel_ratio:.4f}", "PASS")
            
            # 4. Check target answers: ω_A/ω_B = 8, a_A/a_B = 16
            if self.omega_ratio is not None:
                if not math.isclose(self.omega_ratio, 8.0, rel_tol=0.02, abs_tol=0.02):
                    self._log(f"ω ratio should be 8:1, got {self.omega_ratio}:1", "FAIL")
                    all_passed = False
            
            if self.accel_ratio is not None:
                if not math.isclose(self.accel_ratio, 16.0, rel_tol=0.02, abs_tol=0.02):
                    self._log(f"a ratio should be 16:1, got {self.accel_ratio}:1", "FAIL")
                    all_passed = False
            
            return all_passed
            
        except Exception as e:
            self._log(f"Error in test_invariants: {str(e)}\n{traceback.format_exc()}", "ERROR")
            return False
    
    def test_trace_consistency(self) -> bool:
        """
        Verify that intermediate steps lead to consistent final results.
        """
        try:
            self._log("Testing trace consistency...")
            all_consistent = True
            
            # Check that all computed ratios follow from the periods
            if self.omega_ratio is not None:
                # From step 2: ω_A/ω_B = T_B/T_A
                expected = self.T_B / self.T_A
                if not math.isclose(self.omega_ratio, expected, rel_tol=0.02, abs_tol=0.02):
                    self._log(f"ω ratio inconsistent with periods: {self.omega_ratio} vs {expected}", "FAIL")
                    all_consistent = False
                else:
                    self._log("ω ratio consistent with period ratio", "PASS")
            
            if self.radius_ratio is not None:
                # From step 4: r_A/r_B = (T_A/T_B)^(2/3)
                expected = (self.T_A / self.T_B) ** (2/3)
                if not math.isclose(self.radius_ratio, expected, rel_tol=0.02, abs_tol=0.02):
                    self._log(f"Radius ratio inconsistent: {self.radius_ratio} vs {expected:.4f}", "FAIL")
                    all_consistent = False
                else:
                    self._log("Radius ratio consistent with Kepler's Law", "PASS")
            
            if self.accel_ratio is not None:
                # From step 6: a_A/a_B = (T_B/T_A)^(4/3)
                expected = (self.T_B / self.T_A) ** (4/3)
                if not math.isclose(self.accel_ratio, expected, rel_tol=0.02, abs_tol=0.02):
                    self._log(f"Acceleration ratio inconsistent: {self.accel_ratio} vs {expected:.4f}", "FAIL")
                    all_consistent = False
                else:
                    self._log("Acceleration ratio consistent with derivation", "PASS")
            
            return all_consistent
            
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {str(e)}\n{traceback.format_exc()}", "ERROR")
            return False
    
    def test_monotonicity(self) -> bool:
        """
        Perturb input periods and check if results change in the expected direction.
        Restores original values after test.
        """
        try:
            self._log("Testing monotonicity...")
            
            # Store original values
            original_T_B = self.T_B
            
            # Test 1: Increase T_B and check trends
            perturbed_T_B = original_T_B * 1.1  # 10% increase
            
            # Expected trends (derived from formulas):
            # 1. ω_A/ω_B = T_B/T_A → increases with T_B
            # 2. r_A/r_B = (T_A/T_B)^(2/3) → decreases with T_B
            # 3. a_A/a_B = (T_B/T_A)^(4/3) → increases with T_B
            
            # Calculate expected changes
            original_omega_ratio = original_T_B / self.T_A
            perturbed_omega_ratio = perturbed_T_B / self.T_A
            
            original_radius_ratio = (self.T_A / original_T_B) ** (2/3)
            perturbed_radius_ratio = (self.T_A / perturbed_T_B) ** (2/3)
            
            original_accel_ratio = (original_T_B / self.T_A) ** (4/3)
            perturbed_accel_ratio = (perturbed_T_B / self.T_A) ** (4/3)
            
            # Check trends
            trends_ok = True
            
            # ω ratio should increase
            if perturbed_omega_ratio <= original_omega_ratio:
                self._log(f"ω ratio monotonicity failed: {perturbed_omega_ratio:.4f} ≤ {original_omega_ratio:.4f}", "FAIL")
                trends_ok = False
            else:
                self._log(f"ω ratio increases as T_B increases: {original_omega_ratio:.4f} → {perturbed_omega_ratio:.4f}", "PASS")
            
            # Radius ratio should decrease
            if perturbed_radius_ratio >= original_radius_ratio:
                self._log(f"Radius ratio monotonicity failed: {perturbed_radius_ratio:.4f} ≥ {original_radius_ratio:.4f}", "FAIL")
                trends_ok = False
            else:
                self._log(f"Radius ratio decreases as T_B increases: {original_radius_ratio:.4f} → {perturbed_radius_ratio:.4f}", "PASS")
            
            # Acceleration ratio should increase
            if perturbed_accel_ratio <= original_accel_ratio:
                self._log(f"Acceleration ratio monotonicity failed: {perturbed_accel_ratio:.4f} ≤ {original_accel_ratio:.4f}", "FAIL")
                trends_ok = False
            else:
                self._log(f"Acceleration ratio increases as T_B increases: {original_accel_ratio:.4f} → {perturbed_accel_ratio:.4f}", "PASS")
            
            # Restore original value
            self.T_B = original_T_B
            
            return trends_ok
            
        except Exception as e:
            self._log(f"Error in test_monotonicity: {str(e)}\n{traceback.format_exc()}", "ERROR")
            # Try to restore original value even if error
            if 'original_T_B' in locals():
                self.T_B = original_T_B
            return False
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and return overall success and logs.
        """
        self.logs = []  # Reset logs
        overall_success = True
        
        try:
            # Run individual tests
            test1 = self.test_invariants()
            test2 = self.test_trace_consistency()
            test3 = self.test_monotonicity()
            
            # Check final answers against expected targets
            final_check = True
            if self.omega_ratio is not None and self.accel_ratio is not None:
                if not (math.isclose(self.omega_ratio, 8.0, rel_tol=0.02, abs_tol=0.02) and
                       math.isclose(self.accel_ratio, 16.0, rel_tol=0.02, abs_tol=0.02)):
                    self._log(f"Final answers incorrect: ω_A/ω_B = {self.omega_ratio}:1, a_A/a_B = {self.accel_ratio}:1", "FAIL")
                    final_check = False
                else:
                    self._log(f"Final answers correct: ω_A/ω_B = {self.omega_ratio}:1, a_A/a_B = {self.accel_ratio}:1", "PASS")
            else:
                self._log("Missing final ratio values", "WARNING")
                final_check = False
            
            # Summarize
            overall_success = test1 and test2 and test3 and final_check
            
            if overall_success:
                self._log("All tests passed!", "PASS")
            else:
                self._log("Some tests failed", "FAIL")
            
            self._log(f"Test summary: Invariants={test1}, Trace={test2}, Monotonicity={test3}, Final={final_check}", "INFO")
            
        except Exception as e:
            self._log(f"Critical error in run_all_tests: {str(e)}\n{traceback.format_exc()}", "ERROR")
            overall_success = False
        
        return overall_success, self.logs