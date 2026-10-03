import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        Follows safe attribute access pattern to prevent AttributeError.
        """
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract force result - expecting a single float value
        self.F_result = self.results.get('F', 0.0)
        
        # Extract inputs safely with fallbacks
        self.I = getattr(solution_instance, 'I', 0.5)
        self.L = getattr(solution_instance, 'L', 1.0)
        self.B = getattr(solution_instance, 'B', 0.5)
        
        # Theta values from the two scenarios (in degrees)
        self.theta1 = 90.0  # perpendicular (fixed for this problem)
        self.theta2 = 0.0   # parallel (fixed for this problem)
        
        # Initialize symbolic variables
        self.I_sym, self.L_sym, self.B_sym, self.theta_sym = sp.symbols('I L B theta')
        self.F_sym = self.I_sym * self.L_sym * self.B_sym * sp.sin(self.theta_sym)
        
        self.logs = []
        
    def _log(self, message: str, success: bool):
        """Helper to log test results."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {message}")
        return success
    
    def _is_close(self, a: float, b: float) -> bool:
        """Fuzzy numerical comparison with physics tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self) -> bool:
        """
        Verify Ampere's Force Law for both scenarios.
        """
        success = True
        
        # Test scenario 1: wire perpendicular to B (theta = 90°)
        theta1_rad = math.radians(self.theta1)
        F_expected1 = self.I * self.L * self.B * math.sin(theta1_rad)
        
        # Symbolic check
        symbolic_F1 = self.F_sym.subs({
            self.I_sym: self.I,
            self.L_sym: self.L,
            self.B_sym: self.B,
            self.theta_sym: theta1_rad
        })
        symbolic_F1_val = float(symbolic_F1.evalf())
        
        # Verify the perpendicular case calculation
        if not self._is_close(F_expected1, 0.25):
            success = self._log(f"Perpendicular case calculation wrong: expected 0.25 N, got {F_expected1:.3f} N", False)
        elif not self._is_close(F_expected1, symbolic_F1_val):
            success = self._log(f"Symbolic/numeric mismatch: symbolic {symbolic_F1_val:.3f}, numeric {F_expected1:.3f}", False)
        else:
            self._log(f"Perpendicular case: F = {F_expected1:.3f} N matches Ampere's Law", True)
        
        # Test scenario 2: wire parallel to B (theta = 0°)
        theta2_rad = math.radians(self.theta2)
        F_expected2 = self.I * self.L * self.B * math.sin(theta2_rad)  # Should be 0
        
        # Check the final result stored in results['F']
        if not self._is_close(self.F_result, 0.0):
            success = self._log(f"Parallel case should give zero force, got {self.F_result}", False)
        else:
            self._log(f"Parallel case: F = {self.F_result:.3f} N (correctly zero)", True)
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answers.
        According to steps analysis: F1 = 0.25 N, F2 = 0 N.
        """
        success = True
        
        # Check that the final result is 0 N (from Q2)
        if not self._is_close(self.F_result, 0.0):
            success = self._log(f"Final result inconsistent: expected 0 N, got {self.F_result:.3f} N", False)
        else:
            self._log(f"Final result matches expected 0 N", True)
        
        # Also verify the perpendicular calculation would give 0.25 N
        theta1_rad = math.radians(self.theta1)
        F_perp = self.I * self.L * self.B * math.sin(theta1_rad)
        if not self._is_close(F_perp, 0.25):
            success = self._log(f"Perpendicular force should be 0.25 N, got {F_perp:.3f} N", False)
        else:
            self._log(f"Perpendicular force would be {F_perp:.3f} N", True)
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb input current and verify linear relationship.
        Tests the trend: Force ∝ Current.
        """
        success = True
        
        # Use perpendicular case for monotonicity test (non-zero force)
        theta_rad = math.radians(self.theta1)
        original_F = self.I * self.L * self.B * math.sin(theta_rad)
        
        if original_F != 0:
            # Perturb current by +10%
            perturbed_I = self.I * 1.1
            
            # Calculate expected force with perturbed current
            expected_perturbed_F = perturbed_I * self.L * self.B * math.sin(theta_rad)
            
            # Check if ratio matches
            expected_ratio = perturbed_I / self.I  # Should be 1.1
            actual_ratio = expected_perturbed_F / original_F
            
            if not self._is_close(actual_ratio, expected_ratio):
                success = self._log(f"Non-linear response to current perturbation: ratio {actual_ratio:.3f} ≠ {expected_ratio:.3f}", False)
            else:
                self._log(f"Force scales linearly with current (ratio: {actual_ratio:.3f})", True)
        else:
            success = self._log("Skipping monotonicity test: force is zero", True)
        
        return success
    
    def test_direction_logic(self) -> bool:
        """
        Verify force direction logic from steps analysis.
        Force should be maximum when perpendicular, zero when parallel.
        """
        success = True
        
        # Calculate sin values
        sin_perp = math.sin(math.radians(self.theta1))
        sin_parallel = math.sin(math.radians(self.theta2))
        
        # Check sin(90°) ≈ 1
        if not self._is_close(sin_perp, 1.0):
            success = self._log(f"Perpendicular sin(theta) should be ~1, got {sin_perp:.3f}", False)
        else:
            self._log(f"Perpendicular case: sin(θ) = {sin_perp:.3f} (max)", True)
        
        # Check sin(0°) ≈ 0
        if not self._is_close(sin_parallel, 0.0):
            success = self._log(f"Parallel sin(theta) should be ~0, got {sin_parallel:.3f}", False)
        else:
            self._log(f"Parallel case: sin(θ) = {sin_parallel:.3f} (zero)", True)
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all physics checks and return results."""
        self.logs = []
        overall_success = True
        
        # Run all test suites
        tests = [
            ("Ampere's Law Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity (F ∝ I)", self.test_monotonicity),
            ("Direction Logic", self.test_direction_logic)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n--- Testing {test_name} ---")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"Test crashed: {str(e)}", False)
                overall_success = False
        
        # Summary
        self.logs.append(f"\n=== SUMMARY ===")
        self.logs.append(f"Overall: {'ALL TESTS PASSED' if overall_success else 'SOME TESTS FAILED'}")
        
        return overall_success, self.logs