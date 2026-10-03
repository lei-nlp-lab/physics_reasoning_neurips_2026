import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from solution_instance."""
        # Problem-given constants (from context)
        self.L = 10.0  # m
        self.g = 10.0  # m/s²
        
        # Extract results from solution_instance with safe fallbacks
        self.T = getattr(solution_instance, 'T', 
                         getattr(solution_instance, 'results', {}).get('T', 0.0))
        
        # Additional symbolic variables for checks
        self.pi = math.pi
        
        # Store logs for test results
        self.logs = []

    def _log(self, message: str, success: bool):
        """Helper to log test results."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {message}")
        return success

    def test_invariants(self) -> bool:
        """Verify physical invariants using KB formulas."""
        success = True
        
        # Check 1: Period formula T = 2π√(L/g)
        expected_T = 2 * self.pi * math.sqrt(self.L / self.g)
        if not math.isclose(self.T, expected_T, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Period formula violated: T={self.T:.3f}, expected={expected_T:.3f}", False)
            success = False
        else:
            self._log(f"Period formula check passed: T={self.T:.3f} ≈ 2π√(L/g)={expected_T:.3f}", True)
        
        # Check 2: Dimensional consistency (T should have units of seconds)
        # Numerical check: T should be positive and ~2π ≈ 6.28
        if not (self.T > 0 and 5.0 < self.T < 7.0):  # Reasonable bounds for 2π
            self._log(f"Suspicious period value: T={self.T:.3f}s (should be ~6.28s)", False)
            success = False
        
        # Check 3: Small-angle approximation validity (angle < 5° = 0.0873 rad)
        # Using energy ratio from KB: energy_ratio = 1 - cos(θ_avg)
        # For small angles, energy_ratio ≈ θ_avg²/2, so period correction negligible
        # Since problem states angle < 5°, period should be very close to 2π√(L/g)
        # We check that T doesn't deviate by more than 1% from small-angle formula
        rel_error = abs(self.T - expected_T) / expected_T
        if rel_error > 0.01:
            self._log(f"Period deviates >1% from small-angle formula: error={rel_error:.2%}", False)
            success = False
        else:
            self._log(f"Small-angle approximation valid: period error={rel_error:.2%}", True)
        
        return success

    def test_trace_consistency(self) -> bool:
        """Cross-check with intermediate results from STEPS ANALYSIS."""
        success = True
        
        # From STEPS ANALYSIS: step_1 gives T = 6.28 s
        step_T = 6.28
        if not math.isclose(self.T, step_T, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Trace inconsistency: T={self.T:.3f}s ≠ step result {step_T}s", False)
            success = False
        else:
            self._log(f"Trace consistency passed: T={self.T:.3f}s ≈ step result {step_T}s", True)
        
        # Also check that T matches target answer 2π
        target_T = 2 * self.pi
        if not math.isclose(self.T, target_T, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Does not match target answer: T={self.T:.3f}s ≠ 2π={target_T:.3f}s", False)
            success = False
        else:
            self._log(f"Matches target answer: T={self.T:.3f}s ≈ 2π={target_T:.3f}s", True)
        
        return success

    def test_monotonicity(self) -> bool:
        """Test how period changes with perturbations in L and g."""
        success = True
        
        # Store original values
        L_orig = self.L
        g_orig = self.g
        T_orig = self.T
        
        # Test 1: Increase L by 10% → T should increase (T ∝ √L)
        self.L = L_orig * 1.10
        T_new_L = 2 * self.pi * math.sqrt(self.L / g_orig)
        if not (T_new_L > T_orig - 1e-2):  # Allow small tolerance
            self._log(f"Monotonicity violated: T({self.L:.1f}m)={T_new_L:.3f}s ≤ T({L_orig:.1f}m)={T_orig:.3f}s", False)
            success = False
        else:
            self._log(f"L↑ → T↑: T({L_orig:.1f}m)={T_orig:.3f}s → T({self.L:.1f}m)={T_new_L:.3f}s", True)
        
        # Restore L
        self.L = L_orig
        
        # Test 2: Increase g by 10% → T should decrease (T ∝ 1/√g)
        self.g = g_orig * 1.10
        T_new_g = 2 * self.pi * math.sqrt(L_orig / self.g)
        if not (T_new_g < T_orig + 1e-2):  # Allow small tolerance
            self._log(f"Monotonicity violated: T(g={self.g:.1f}m/s²)={T_new_g:.3f}s ≥ T(g={g_orig:.1f}m/s²)={T_orig:.3f}s", False)
            success = False
        else:
            self._log(f"g↑ → T↓: T(g={g_orig:.1f}m/s²)={T_orig:.3f}s → T(g={self.g:.1f}m/s²)={T_new_g:.3f}s", True)
        
        # Restore g
        self.g = g_orig
        
        return success

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return success status and logs."""
        self.logs = []
        
        # Run tests
        test_results = [
            ("Invariants", self.test_invariants()),
            ("Trace Consistency", self.test_trace_consistency()),
            ("Monotonicity", self.test_monotonicity()),
        ]
        
        # Summary
        all_passed = all(result for _, result in test_results)
        self._log(f"All tests {'PASSED' if all_passed else 'FAILED'}", all_passed)
        
        return all_passed, self.logs