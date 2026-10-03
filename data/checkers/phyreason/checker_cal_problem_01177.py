import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with safe defaults
        self.M = getattr(solution_instance, 'M', 1.5)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.sin53 = getattr(solution_instance, 'sin53', 0.8)
        self.cos53 = getattr(solution_instance, 'cos53', 0.6)
        
        # Extract results with safe fallbacks
        self.m = self.results.get('m', 0.0)
        self.F_BC = self.results.get('F_BC', 0.0)
        self.F = self.results.get('F', 0.0)
        self.F_AC = self.results.get('F_AC', 0.0)
        
        # Intermediate calculations from STEPS ANALYSIS
        self.sin37 = 0.6  # cos53
        self.cos37 = 0.8  # sin53
        
        self.logs = []
        
    def _log(self, message: str, status: str = "INFO"):
        """Helper to log messages with status."""
        self.logs.append(f"[{status}] {message}")
        
    def _isclose(self, a, b):
        """Robust fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self) -> bool:
        """Test fundamental physical invariants."""
        success = True
        
        # Test 1: Vertical force equilibrium at point C (ball)
        # According to step_3: F_AC*sin37 + F_BC*sin53 = m*g
        left_side = self.F_AC * self.sin37 + self.F_BC * self.sin53
        right_side = self.m * self.g
        if not self._isclose(left_side, right_side):
            self._log(f"Vertical force equilibrium at ball violated: {left_side} ≠ {right_side}", "ERROR")
            success = False
        else:
            self._log(f"Vertical force equilibrium at ball holds: {left_side} ≈ {right_side}")
        
        # Test 2: Horizontal force equilibrium at point C (ball)
        # According to step_2: F_AC*cos37 = F_BC*cos53
        left_horiz = self.F_AC * self.cos37
        right_horiz = self.F_BC * self.cos53
        if not self._isclose(left_horiz, right_horiz):
            self._log(f"Horizontal force equilibrium at ball violated: {left_horiz} ≠ {right_horiz}", "ERROR")
            success = False
        else:
            self._log(f"Horizontal force equilibrium at ball holds: {left_horiz} ≈ {right_horiz}")
        
        # Test 3: Force equilibrium at mass M (step_1)
        # F_AC = M*g (since only vertical forces)
        if not self._isclose(self.F_AC, self.M * self.g):
            self._log(f"Force equilibrium at mass M violated: {self.F_AC} ≠ {self.M*self.g}", "ERROR")
            success = False
        else:
            self._log(f"Force equilibrium at mass M holds: {self.F_AC} ≈ {self.M*self.g}")
        
        # Test 4: Minimum pulling force F (step_4)
        # F = m*g - F_AC*sin37
        expected_F = self.m * self.g - self.F_AC * self.sin37
        if not self._isclose(self.F, expected_F):
            self._log(f"Minimum pulling force equation violated: {self.F} ≠ {expected_F}", "ERROR")
            success = False
        else:
            self._log(f"Minimum pulling force equation holds: {self.F} ≈ {expected_F}")
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Test consistency between intermediate and final results."""
        success = True
        
        # Recompute F_AC from step_1
        expected_F_AC = self.M * self.g
        if not self._isclose(self.F_AC, expected_F_AC):
            self._log(f"F_AC inconsistent with M: {self.F_AC} ≠ {expected_F_AC}", "ERROR")
            success = False
        else:
            self._log(f"F_AC consistent with M: {self.F_AC} ≈ {expected_F_AC}")
        
        # Recompute F_BC from step_2
        expected_F_BC = self.F_AC * self.cos37 / self.cos53
        if not self._isclose(self.F_BC, expected_F_BC):
            self._log(f"F_BC inconsistent with F_AC: {self.F_BC} ≠ {expected_F_BC}", "ERROR")
            success = False
        else:
            self._log(f"F_BC consistent with F_AC: {self.F_BC} ≈ {expected_F_BC}")
        
        # Recompute m from step_3
        expected_m = (self.F_AC * self.sin37 + self.F_BC * self.sin53) / self.g
        if not self._isclose(self.m, expected_m):
            self._log(f"m inconsistent with forces: {self.m} ≠ {expected_m}", "ERROR")
            success = False
        else:
            self._log(f"m consistent with forces: {self.m} ≈ {expected_m}")
        
        # Recompute F from step_4
        expected_F = self.m * self.g - self.F_AC * self.sin37
        if not self._isclose(self.F, expected_F):
            self._log(f"F inconsistent with m and F_AC: {self.F} ≠ {expected_F}", "ERROR")
            success = False
        else:
            self._log(f"F consistent with m and F_AC: {self.F} ≈ {expected_F}")
        
        # Cross-check with target answers
        target_answers = [20.0, 2.5, 16.0]
        computed = [self.F_BC, self.m, self.F]
        
        for i, (target, computed_val) in enumerate(zip(target_answers, computed)):
            if not self._isclose(target, computed_val):
                self._log(f"Target answer {i+1} mismatch: {computed_val} ≠ {target}", "ERROR")
                success = False
            else:
                self._log(f"Target answer {i+1} matches: {computed_val} ≈ {target}")
        
        return success
    
    def test_monotonicity(self) -> bool:
        """Test how results change when M is perturbed, then restore."""
        success = True
        
        # Save original values
        original_M = self.M
        original_F_AC = self.F_AC
        original_F_BC = self.F_BC
        original_m = self.m
        original_F = self.F
        
        try:
            # Perturb M by +10%
            perturbed_M = original_M * 1.1
            
            # Recompute chain using formulas from steps
            perturbed_F_AC = perturbed_M * self.g
            perturbed_F_BC = perturbed_F_AC * self.cos37 / self.cos53
            perturbed_m = (perturbed_F_AC * self.sin37 + perturbed_F_BC * self.sin53) / self.g
            perturbed_F = perturbed_m * self.g - perturbed_F_AC * self.sin37
            
            # Check monotonic relationships
            # F_AC should increase with M
            if not perturbed_F_AC > original_F_AC:
                self._log(f"F_AC not monotonic with M: {perturbed_F_AC} ≤ {original_F_AC}", "WARNING")
                success = False
            else:
                self._log(f"F_AC increases with M: {perturbed_F_AC} > {original_F_AC}")
            
            # F_BC should increase with F_AC (and thus with M)
            if not perturbed_F_BC > original_F_BC:
                self._log(f"F_BC not monotonic with M: {perturbed_F_BC} ≤ {original_F_BC}", "WARNING")
                success = False
            else:
                self._log(f"F_BC increases with M: {perturbed_F_BC} > {original_F_BC}")
            
            # m should increase with both tensions
            if not perturbed_m > original_m:
                self._log(f"m not monotonic with M: {perturbed_m} ≤ {original_m}", "WARNING")
                success = False
            else:
                self._log(f"m increases with M: {perturbed_m} > {original_m}")
            
            # F should increase with m but decrease with F_AC*sin37
            # The net effect from formula: F = m*g - F_AC*sin37
            # Since both m and F_AC increase, need to check sign
            delta_F = perturbed_F - original_F
            if delta_F <= 0:
                self._log(f"F not strictly increasing with M: {perturbed_F} ≤ {original_F}", "WARNING")
                # Not necessarily an error, just log
            else:
                self._log(f"F increases with M: {perturbed_F} > {original_F}")
                
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {str(e)}", "ERROR")
            success = False
        finally:
            # Restore original values
            self.M = original_M
            self.F_AC = original_F_AC
            self.F_BC = original_F_BC
            self.m = original_m
            self.F = original_F
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success and logs."""
        self.logs.clear()
        
        self._log("Starting PhysicsChecker tests...")
        
        # Run tests
        invariants_ok = self.test_invariants()
        trace_ok = self.test_trace_consistency()
        monotonicity_ok = self.test_monotonicity()
        
        overall_success = invariants_ok and trace_ok
        
        # Summarize
        self._log("\n=== TEST SUMMARY ===")
        self._log(f"Invariants test: {'PASS' if invariants_ok else 'FAIL'}")
        self._log(f"Trace consistency: {'PASS' if trace_ok else 'FAIL'}")
        self._log(f"Monotonicity check: {'PASS' if monotonicity_ok else 'WARNINGS'}")
        
        if overall_success:
            self._log("All critical tests passed!", "SUCCESS")
        else:
            self._log("Some critical tests failed!", "ERROR")
        
        return overall_success, self.logs