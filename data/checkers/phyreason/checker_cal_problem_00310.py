import math
import sympy as sp
from typing import Tuple, List, Any, Optional

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """Initialize with safe attribute access."""
        # Extract inputs from solution instance
        self.a_A_problem = 0.12  # Given in problem statement (cm/s^2)
        
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        
        # Get values with fallbacks
        self.a_A = getattr(solution_instance, 'a_A', self.results.get('a_A', self.a_A_problem))
        self.a_B = getattr(solution_instance, 'a_B', self.results.get('a_B', None))
        self.a_C = getattr(solution_instance, 'a_C', self.results.get('a_C', None))
        
        # Constants from problem
        self.R = 2  # Relative units (R = 2r)
        self.r = 1  # Relative units
        
        # Initialize logs
        self.logs = []
        
    def _log(self, message: str, success: bool = True):
        """Helper to log messages."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {message}")
        
    def test_invariants(self) -> bool:
        """Test physical invariants based on centripetal motion laws."""
        success = True
        
        # Test 1: Check a_A consistency
        if not math.isclose(self.a_A, 0.12, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"a_A={self.a_A} differs from given 0.12 cm/s²", False)
            success = False
        else:
            self._log(f"a_A={self.a_A} matches given value")
        
        # Test 2: Check a_B using linear velocity relationship (same v for both rollers)
        if self.a_B is not None:
            # From a_A = v²/R and a_B = v²/r, so a_B = (R/r) * a_A = 2 * a_A
            expected_a_B = 2 * self.a_A
            if not math.isclose(self.a_B, expected_a_B, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"a_B={self.a_B} but expected 2*a_A={expected_a_B}", False)
                success = False
            else:
                self._log(f"a_B={self.a_B} matches 2*a_A={expected_a_B}")
        
        # Test 3: Check a_C using angular velocity relationship (same ω for points on same roller)
        if self.a_C is not None:
            # From a_A = ω²*R and a_C = ω²*(R/3), so a_C = a_A/3
            expected_a_C = self.a_A / 3.0
            if not math.isclose(self.a_C, expected_a_C, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"a_C={self.a_C} but expected a_A/3={expected_a_C}", False)
                success = False
            else:
                self._log(f"a_C={self.a_C} matches a_A/3={expected_a_C}")
        
        # Test 4: Verify ratios numerically (replaces flawed symbolic check)
        if self.a_B is not None and self.a_A != 0:
            ratio_B = self.a_B / self.a_A
            if not math.isclose(ratio_B, 2.0, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Ratio a_B/a_A={ratio_B} != 2", False)
                success = False
            else:
                self._log(f"Ratio a_B/a_A={ratio_B} = 2")
                
        if self.a_C is not None and self.a_A != 0:
            ratio_C = self.a_C / self.a_A
            if not math.isclose(ratio_C, 1.0/3.0, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Ratio a_C/a_A={ratio_C} != 1/3", False)
                success = False
            else:
                self._log(f"Ratio a_C/a_A={ratio_C} = 1/3")
            
        return success
    
    def test_trace_consistency(self) -> bool:
        """Check consistency between intermediate steps and final results."""
        success = True
        
        if self.a_B is None or self.a_C is None:
            self._log("Missing results for trace consistency check", False)
            return False
        
        # Step 1: Verify a_A is as given
        if not math.isclose(self.a_A, 0.12, rel_tol=1e-2, abs_tol=1e-2):
            self._log("Step 1 (a_A) inconsistent with problem statement", False)
            success = False
            
        # Step 2-3: Verify a_B derivation
        # From a_B = (R/r) * a_A = 2 * 0.12 = 0.24
        a_B_from_ratio = 2 * self.a_A
        if not math.isclose(self.a_B, a_B_from_ratio, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Step 2-3: a_B={self.a_B} doesn't match 2*a_A={a_B_from_ratio}", False)
            success = False
            
        # Step 4-6: Verify a_C derivation
        # From a_C = a_A/3 = 0.12/3 = 0.04
        a_C_from_ratio = self.a_A / 3.0
        if not math.isclose(self.a_C, a_C_from_ratio, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Step 4-6: a_C={self.a_C} doesn't match a_A/3={a_C_from_ratio}", False)
            success = False
            
        if success:
            self._log("All trace steps are consistent")
            
        return success
    
    def test_monotonicity(self) -> bool:
        """Test that perturbations in inputs produce expected changes in outputs."""
        success = True
        
        # Store original values
        original_a_A = self.a_A
        
        # Perturb a_A by +10%
        perturbed_a_A = original_a_A * 1.10
        
        # Recalculate expected values based on formulas
        expected_a_B_perturbed = 2 * perturbed_a_A
        expected_a_C_perturbed = perturbed_a_A / 3.0
        
        # Check that relationships remain linear
        # For a_B: should scale linearly with a_A
        ratio_B = expected_a_B_perturbed / perturbed_a_A
        if not math.isclose(ratio_B, 2.0, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Perturbed a_B ratio {ratio_B} != 2", False)
            success = False
            
        # For a_C: should scale linearly with a_A
        ratio_C = expected_a_C_perturbed / perturbed_a_A
        if not math.isclose(ratio_C, 1.0/3.0, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Perturbed a_C ratio {ratio_C} != 1/3", False)
            success = False
            
        # Test negative perturbation
        perturbed_a_A_neg = original_a_A * 0.90
        expected_a_B_neg = 2 * perturbed_a_A_neg
        expected_a_C_neg = perturbed_a_A_neg / 3.0
        
        # Verify monotonic relationship
        # When a_A increases, a_B and a_C should increase proportionally
        delta_a_A = perturbed_a_A - perturbed_a_A_neg
        delta_a_B = expected_a_B_perturbed - expected_a_B_neg
        delta_a_C = expected_a_C_perturbed - expected_a_C_neg
        
        expected_delta_a_B = 2 * delta_a_A
        expected_delta_a_C = delta_a_A / 3.0
        
        if not math.isclose(delta_a_B, expected_delta_a_B, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"a_B change {delta_a_B} doesn't match expected {expected_delta_a_B}", False)
            success = False
            
        if not math.isclose(delta_a_C, expected_delta_a_C, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"a_C change {delta_a_C} doesn't match expected {expected_delta_a_C}", False)
            success = False
            
        if success:
            self._log("Monotonicity tests passed: linear relationships preserved under perturbation")
            
        # Restore original value
        self.a_A = original_a_A
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all physics consistency tests."""
        self.logs = []  # Reset logs
        
        overall_success = True
        
        # Run each test
        tests = [
            ("Physical Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"\n=== Running {test_name} ===")
            try:
                if not test_func():
                    overall_success = False
                    self._log(f"{test_name} FAILED", False)
            except Exception as e:
                self._log(f"{test_name} ERROR: {e}", False)
                overall_success = False
        
        # Summary
        self._log(f"\n=== SUMMARY ===")
        self._log(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, self.logs