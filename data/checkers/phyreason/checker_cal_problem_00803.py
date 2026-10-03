import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all required inputs and results from the solution instance.
        Never access attributes directly; use getattr with fallback to results dict.
        """
        # Physical constants and inputs
        self.m1 = 40.0  # Sui Wenjing's mass (kg)
        self.m2 = 60.0  # Han Cong's mass (kg)
        self.v0 = 1.0   # Initial common velocity (m/s)
        self.v1 = 4.0   # Sui's final velocity (m/s)
        self.delta_t = 2.0  # Push duration (s)
        
        # Results from the solution (with safe access)
        self.v2 = getattr(solution_instance, 'v_2', 
                         getattr(solution_instance, 'results', {}).get('v_2', None))
        self.F = getattr(solution_instance, 'F', 
                        getattr(solution_instance, 'results', {}).get('F', None))
        
        # Tolerance for fuzzy matching (1-2% as per rules)
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Logs for reporting
        self.logs = []
    
    def _log(self, message, success):
        """Helper to append log messages with status."""
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {message}")
        return success
    
    def test_invariants(self):
        """
        Verify fundamental physics invariants:
        1. Conservation of momentum (external force = 0 in horizontal direction)
        2. Impulse-momentum theorem for both skaters
        """
        tests_passed = True
        
        # 1. Conservation of momentum (initial = final)
        p_initial = (self.m1 + self.m2) * self.v0
        p_final = self.m1 * self.v1 + self.m2 * self.v2
        
        if not math.isclose(p_initial, p_final, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            tests_passed = self._log(
                f"Momentum not conserved: initial={p_initial:.3f}, final={p_final:.3f}", 
                False
            )
        else:
            tests_passed = self._log(
                f"Momentum conserved: {p_initial:.3f} ≈ {p_final:.3f}", 
                True
            ) and tests_passed
        
        # 2. Impulse-momentum for Han Cong (subject to force F)
        if self.F is not None and self.v2 is not None:
            impulse_Han = self.F * self.delta_t
            delta_p_Han = self.m2 * (self.v2 - self.v0)
            
            if not math.isclose(impulse_Han, delta_p_Han, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                tests_passed = self._log(
                    f"Impulse-momentum mismatch for Han: FΔt={impulse_Han:.3f}, Δp={delta_p_Han:.3f}", 
                    False
                )
            else:
                tests_passed = self._log(
                    f"Impulse-momentum for Han OK: {impulse_Han:.3f} ≈ {delta_p_Han:.3f}", 
                    True
                ) and tests_passed
            
            # 3. Impulse-momentum for Sui Wenjing (Newton's 3rd law: force = -F)
            delta_p_Sui = self.m1 * (self.v1 - self.v0)
            impulse_Sui = -self.F * self.delta_t
            
            if not math.isclose(impulse_Sui, delta_p_Sui, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                tests_passed = self._log(
                    f"Impulse-momentum mismatch for Sui: -FΔt={impulse_Sui:.3f}, Δp={delta_p_Sui:.3f}", 
                    False
                )
            else:
                tests_passed = self._log(
                    f"Impulse-momentum for Sui OK: {impulse_Sui:.3f} ≈ {delta_p_Sui:.3f}", 
                    True
                ) and tests_passed
        
        return tests_passed
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers using step-by-step formulas.
        """
        tests_passed = True
        
        # Step 1: Conservation of momentum to find v2
        v2_expected = ((self.m1 + self.m2) * self.v0 - self.m1 * self.v1) / self.m2
        
        if self.v2 is not None:
            if not math.isclose(self.v2, v2_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                tests_passed = self._log(
                    f"v2 inconsistent: expected {v2_expected:.3f}, got {self.v2:.3f}", 
                    False
                )
            else:
                tests_passed = self._log(
                    f"v2 correct: {self.v2:.3f} ≈ {v2_expected:.3f}", 
                    True
                ) and tests_passed
        else:
            tests_passed = self._log("v2 missing", False) and tests_passed
        
        # Step 2: Impulse-momentum for Han to find F
        if self.v2 is not None:
            F_expected = self.m2 * (self.v2 - self.v0) / self.delta_t
            
            if self.F is not None:
                if not math.isclose(self.F, F_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    tests_passed = self._log(
                        f"F inconsistent: expected {F_expected:.3f}, got {self.F:.3f}", 
                        False
                    )
                else:
                    tests_passed = self._log(
                        f"F correct: {self.F:.3f} ≈ {F_expected:.3f}", 
                        True
                    ) and tests_passed
            else:
                tests_passed = self._log("F missing", False) and tests_passed
        
        return tests_passed
    
    def test_monotonicity(self):
        """
        Perturb an input parameter and verify the expected trend in output.
        We'll increase Sui's mass (m1) and check that the force magnitude increases.
        """
        tests_passed = True
        
        if self.F is None or self.v2 is None:
            return self._log("Cannot test monotonicity without results", False)
        
        # Perturb m1 upward by 10%
        m1_perturbed = self.m1 * 1.1
        
        # Recalculate using physics formulas (conservation of momentum first)
        v2_perturbed = ((m1_perturbed + self.m2) * self.v0 - m1_perturbed * self.v1) / self.m2
        F_perturbed = self.m2 * (v2_perturbed - self.v0) / self.delta_t
        
        # Expected trend: With larger m1, Sui has more inertia to accelerate,
        # so the force required (magnitude) should increase.
        # Note: F is negative (opposite to motion), so compare absolute values.
        if abs(F_perturbed) > abs(self.F):
            tests_passed = self._log(
                f"Monotonicity OK: |F| increases with m1 (|F|={abs(self.F):.3f} → {abs(F_perturbed):.3f})", 
                True
            )
        else:
            tests_passed = self._log(
                f"Monotonicity violation: |F| should increase with m1", 
                False
            )
        
        return tests_passed
    
    def run_all_tests(self):
        """
        Execute all tests and return overall success with logs.
        """
        self.logs = []  # Reset logs
        success = True
        
        try:
            success = self.test_invariants() and success
            success = self.test_trace_consistency() and success
            success = self.test_monotonicity() and success
        except Exception as e:
            self.logs.append(f"ERROR: {traceback.format_exc()}")
            success = False
        
        return success, self.logs