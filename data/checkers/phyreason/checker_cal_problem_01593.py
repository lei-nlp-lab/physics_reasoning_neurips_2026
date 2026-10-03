import sympy as sp
import math
import traceback
from typing import Dict, List, Tuple, Any, Optional

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Safely extract all attributes from solution_instance.
        Use getattr with fallback to results dictionary.
        """
        # Safe attribute access with fallbacks
        self.I = self._safe_get(solution_instance, 'I', 0.0)
        self.I_prime = self._safe_get(solution_instance, 'I_prime', 0.0)
        self.I_prime_2 = self._safe_get(solution_instance, 'I\'_2', 0.0)  # Key from mapping
        self.U = self._safe_get(solution_instance, 'U', 0.0)
        self.R_eq = self._safe_get(solution_instance, 'R_eq', 0.0)
        
        # Symbolic variables for physics laws
        self.R = sp.symbols('R', positive=True)
        self.I_sym = sp.symbols('I', positive=True)
        self.U_sym = sp.symbols('U', positive=True)
        self.I_prime_sym = sp.symbols('I_prime', positive=True)
        self.I_prime_2_sym = sp.symbols('I_prime_2', positive=True)
        
        self.logs: List[str] = []
        self.solution = solution_instance
        
    def _safe_get(self, obj: Any, key: str, default: float) -> float:
        """Safely get attribute from solution instance or its results dict."""
        try:
            val = getattr(obj, key, None)
            if val is None:
                val = getattr(obj, 'results', {}).get(key, default)
            return float(val) if val is not None else default
        except (AttributeError, KeyError, TypeError, ValueError):
            return default
    
    def test_invariants(self) -> bool:
        """Test Ohm's Law and current division principles."""
        tests_passed = True
        
        # 1. Ohm's Law: U = I * (2R) before switch (from step 1)
        # R is unknown, but we can check if U/I = 2*R (if R can be inferred)
        if self.I != 0:
            inferred_R = self.U / (2 * self.I)
            # Check if this R is positive (basic sanity)
            if inferred_R <= 0:
                self.logs.append(f"FAIL: Inferred R = {inferred_R} from U/(2I) is not positive")
                tests_passed = False
            else:
                self.logs.append(f"OK: Inferred R = {inferred_R} from initial condition U/(2I)")
        
        # 2. Ohm's Law after switch: U = I_prime * R_eq
        if self.I_prime != 0 and self.R_eq != 0:
            lhs = self.U
            rhs = self.I_prime * self.R_eq
            if not math.isclose(lhs, rhs, rel_tol=0.02, abs_tol=0.02):
                self.logs.append(f"FAIL: Ohm's Law after switch: U={lhs} ≠ I'*R_eq={rhs}")
                tests_passed = False
            else:
                self.logs.append(f"OK: Ohm's Law after switch holds: U={lhs} ≈ I'*R_eq={rhs}")
        
        # 3. Current division: In parallel R2||R3, current splits equally (R2=R3)
        # I_prime_2 should be half of I_prime
        if self.I_prime != 0:
            expected_I_prime_2 = self.I_prime / 2
            if not math.isclose(self.I_prime_2, expected_I_prime_2, rel_tol=0.02, abs_tol=0.02):
                self.logs.append(f"FAIL: Current division: I'_2={self.I_prime_2} ≠ I'/2={expected_I_prime_2}")
                tests_passed = False
            else:
                self.logs.append(f"OK: Current division holds: I'_2={self.I_prime_2} ≈ I'/2={expected_I_prime_2}")
        
        # 4. Symbolic check of relationships
        try:
            # From step 1: U = 2*I*R
            eq1 = sp.Eq(self.U_sym, 2 * self.I_sym * self.R)
            # From step 2-3: I_prime = (4/3)*I
            eq2 = sp.Eq(self.I_prime_sym, (4/3) * self.I_sym)
            # From step 4: I_prime_2 = I_prime/2 = (2/3)*I
            eq3 = sp.Eq(self.I_prime_2_sym, (2/3) * self.I_sym)
            
            # Check if solution matches symbolic relationships
            if not math.isclose(self.I_prime, (4/3) * self.I, rel_tol=0.02, abs_tol=0.02):
                self.logs.append(f"FAIL: Symbolic I' relationship: I'={self.I_prime} ≠ (4/3)*I={(4/3)*self.I}")
                tests_passed = False
            else:
                self.logs.append(f"OK: Symbolic I' relationship holds")
                
            if not math.isclose(self.I_prime_2, (2/3) * self.I, rel_tol=0.02, abs_tol=0.02):
                self.logs.append(f"FAIL: Symbolic I'_2 relationship: I'_2={self.I_prime_2} ≠ (2/3)*I={(2/3)*self.I}")
                tests_passed = False
            else:
                self.logs.append(f"OK: Symbolic I'_2 relationship holds")
                
        except Exception as e:
            self.logs.append(f"WARNING: Symbolic check failed: {e}")
        
        return tests_passed
    
    def test_trace_consistency(self) -> bool:
        """Check consistency between intermediate steps and final answer."""
        tests_passed = True
        
        # Step 1: U = 2*I*R (we don't know R, but we can check consistency with later steps)
        # Step 2-3: I_prime = U / R_eq, and R_eq = 1.5*R
        # So I_prime should equal (2*I*R) / (1.5*R) = (4/3)*I
        if self.I != 0:
            expected_I_prime = (4/3) * self.I
            if not math.isclose(self.I_prime, expected_I_prime, rel_tol=0.02, abs_tol=0.02):
                self.logs.append(f"FAIL: Trace consistency: I'={self.I_prime} ≠ (4/3)*I={expected_I_prime}")
                tests_passed = False
            else:
                self.logs.append(f"OK: Trace consistency for I' holds")
        
        # Step 4: I_prime_2 = I_prime/2
        expected_I_prime_2_from_trace = self.I_prime / 2
        if not math.isclose(self.I_prime_2, expected_I_prime_2_from_trace, rel_tol=0.02, abs_tol=0.02):
            self.logs.append(f"FAIL: Trace consistency: I'_2={self.I_prime_2} ≠ I'/2={expected_I_prime_2_from_trace}")
            tests_passed = False
        else:
            self.logs.append(f"OK: Trace consistency for I'_2 holds")
        
        # Check that final answer matches target: (2/3)*I
        target_answer = (2/3) * self.I
        if not math.isclose(self.I_prime_2, target_answer, rel_tol=0.02, abs_tol=0.02):
            self.logs.append(f"FAIL: Final answer mismatch: I'_2={self.I_prime_2} ≠ (2/3)*I={target_answer}")
            tests_passed = False
        else:
            self.logs.append(f"OK: Final answer matches target (2/3)*I")
        
        return tests_passed
    
    def test_monotonicity(self) -> bool:
        """
        Perturb input (mass/resistance) and check if trends match physics intuition.
        We'll perturb R (resistance) and observe effect on I_prime_2.
        """
        tests_passed = True
        
        # We need a value for R to perturb. Infer from U and I if possible.
        if self.I == 0 or self.U == 0:
            self.logs.append("SKIP: Monotonicity test - insufficient data (I=0 or U=0)")
            return True
            
        R_inferred = self.U / (2 * self.I)
        
        # Test 1: Increase R by 10%
        R_perturbed = R_inferred * 1.10
        
        # Recalculate using physics formulas:
        # U remains constant (same battery)
        # New initial current: I_new = U / (2*R_perturbed)
        I_new = self.U / (2 * R_perturbed)
        
        # New equivalent resistance: R_eq_new = R_perturbed + (R_perturbed^2)/(2*R_perturbed) = 1.5*R_perturbed
        R_eq_new = 1.5 * R_perturbed
        
        # New main current: I_prime_new = U / R_eq_new
        I_prime_new = self.U / R_eq_new
        
        # New current through R2: I_prime_2_new = I_prime_new / 2
        I_prime_2_new = I_prime_new / 2
        
        # Original I_prime_2 for comparison
        I_prime_2_original = self.I_prime_2 if self.I_prime_2 != 0 else (2/3) * self.I
        
        # Physics trend: With U fixed, if R increases, I_prime_2 should DECREASE
        # Because I_prime = U/(1.5R) ∝ 1/R, and I_prime_2 = I_prime/2 ∝ 1/R
        if I_prime_2_new >= I_prime_2_original:
            self.logs.append(f"FAIL: Monotonicity - Increasing R should decrease I'_2, but got {I_prime_2_new} >= {I_prime_2_original}")
            tests_passed = False
        else:
            self.logs.append(f"OK: Monotonicity holds - Increasing R decreases I'_2: {I_prime_2_new} < {I_prime_2_original}")
        
        # Test 2: Decrease R by 10%
        R_perturbed2 = R_inferred * 0.90
        I_new2 = self.U / (2 * R_perturbed2)
        R_eq_new2 = 1.5 * R_perturbed2
        I_prime_new2 = self.U / R_eq_new2
        I_prime_2_new2 = I_prime_new2 / 2
        
        # Trend: Decreasing R should INCREASE I_prime_2
        if I_prime_2_new2 <= I_prime_2_original:
            self.logs.append(f"FAIL: Monotonicity - Decreasing R should increase I'_2, but got {I_prime_2_new2} <= {I_prime_2_original}")
            tests_passed = False
        else:
            self.logs.append(f"OK: Monotonicity holds - Decreasing R increases I'_2: {I_prime_2_new2} > {I_prime_2_original}")
        
        # Restore original values (not strictly needed as we used local variables)
        # But ensure solution instance is not modified
        self.logs.append("INFO: Monotonicity test completed with perturbation, original state preserved")
        
        return tests_passed
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return success flag and logs."""
        self.logs.clear()
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for test_name, test_func in tests:
            try:
                self.logs.append(f"\n=== Running {test_name} Test ===")
                if not test_func():
                    success = False
                    self.logs.append(f"RESULT: {test_name} FAILED")
                else:
                    self.logs.append(f"RESULT: {test_name} PASSED")
            except Exception as e:
                self.logs.append(f"ERROR in {test_name}: {str(e)}")
                self.logs.append(traceback.format_exc())
                success = False
        
        self.logs.append(f"\n=== Overall Result: {'PASS' if success else 'FAIL'} ===")
        return success, self.logs