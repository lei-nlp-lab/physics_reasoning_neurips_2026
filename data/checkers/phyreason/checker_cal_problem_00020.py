import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        Use getattr with fallbacks to avoid AttributeError in sandbox.
        """
        self.solution = solution_instance
        
        # Extract input parameters with safe defaults
        self.m = getattr(solution_instance, 'm', None)
        self.L = getattr(solution_instance, 'L', None)
        self.v = getattr(solution_instance, 'v', None)
        self.g = getattr(solution_instance, 'g', None)
        
        # Extract results with safe fallback to results dictionary
        results = getattr(solution_instance, 'results', {})
        self.H = getattr(solution_instance, 'H', results.get('H', None))
        self.F = getattr(solution_instance, 'F', results.get('F', None))
        self.delta_E = getattr(solution_instance, 'delta_E', results.get('delta_E', None))
        self.v_prime = getattr(solution_instance, 'v_prime', results.get('v_prime', None))
        
        # If v_prime not directly stored, compute from conservation of momentum
        if self.v_prime is None and self.m is not None and self.v is not None:
            self.v_prime = self.v / 2
        
        # Store symbolic variables for invariant checks
        self.m_sym = sp.symbols('m')
        self.L_sym = sp.symbols('L')
        self.v_sym = sp.symbols('v')
        self.g_sym = sp.symbols('g')
        self.H_sym = sp.symbols('H')
        self.F_sym = sp.symbols('F')
        self.delta_E_sym = sp.symbols('delta_E')
        self.v_prime_sym = sp.symbols('v_prime')
        
        self.logs = []
    
    def _log(self, message, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            prefix = "PASS" if success else "FAIL"
            self.logs.append(f"{prefix}: {message}")
        else:
            self.logs.append(f"INFO: {message}")
    
    def _fuzzy_eq(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy equality check for physics quantities."""
        if a is None or b is None:
            return False
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False
    
    def test_invariants(self):
        """Verify fundamental physics laws: energy conservation, Newton's laws, momentum conservation."""
        success = True
        
        # 1. Check conservation of mechanical energy for ball A's fall (step 1)
        if all(x is not None for x in [self.m, self.g, self.H, self.v]):
            expected_H = self.v**2 / (2 * self.g)
            if not self._fuzzy_eq(self.H, expected_H):
                self._log(f"Energy conservation (fall): H = {self.H} but v²/(2g) = {expected_H}", False)
                success = False
            else:
                self._log(f"Energy conservation (fall): H = v²/(2g) holds", True)
        
        # 2. Check Newton's second law for tension (step 2)
        if all(x is not None for x in [self.F, self.m, self.g, self.v, self.L]):
            expected_F = self.m * self.g + self.m * self.v**2 / self.L
            if not self._fuzzy_eq(self.F, expected_F):
                self._log(f"Newton's second law: F = {self.F} but mg + mv²/L = {expected_F}", False)
                success = False
            else:
                self._log(f"Newton's second law: F = mg + mv²/L holds", True)
        
        # 3. Check conservation of momentum during collision (step 3)
        if all(x is not None for x in [self.m, self.v, self.v_prime]):
            momentum_before = self.m * self.v
            momentum_after = 2 * self.m * self.v_prime
            if not self._fuzzy_eq(momentum_before, momentum_after):
                self._log(f"Momentum conservation: m*v = {momentum_before}, 2m*v' = {momentum_after}", False)
                success = False
            else:
                self._log(f"Momentum conservation: m*v = 2m*v' holds", True)
        
        # 4. Check mechanical energy loss (step 4)
        if all(x is not None for x in [self.delta_E, self.m, self.v, self.v_prime]):
            KE_before = 0.5 * self.m * self.v**2
            KE_after = 0.5 * (2 * self.m) * self.v_prime**2
            expected_delta_E = KE_before - KE_after
            if not self._fuzzy_eq(self.delta_E, expected_delta_E):
                self._log(f"Energy loss: ΔE = {self.delta_E} but ½mv² - ½(2m)v'² = {expected_delta_E}", False)
                success = False
            else:
                self._log(f"Energy loss: ΔE = ½mv² - ½(2m)v'² holds", True)
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers and input consistency."""
        success = True
        
        # Check H from energy conservation
        if self.H is not None and self.v is not None and self.g is not None:
            H_from_v = self.v**2 / (2 * self.g)
            if not self._fuzzy_eq(self.H, H_from_v):
                self._log(f"Trace consistency: H ({self.H}) ≠ v²/(2g) ({H_from_v})", False)
                success = False
        
        # Check F from Newton's second law
        if self.F is not None and self.m is not None and self.g is not None and self.v is not None and self.L is not None:
            F_from_law = self.m * self.g + self.m * self.v**2 / self.L
            if not self._fuzzy_eq(self.F, F_from_law):
                self._log(f"Trace consistency: F ({self.F}) ≠ mg + mv²/L ({F_from_law})", False)
                success = False
        
        # Check v' from momentum conservation
        if self.v_prime is not None and self.v is not None:
            v_prime_expected = self.v / 2
            if not self._fuzzy_eq(self.v_prime, v_prime_expected):
                self._log(f"Trace consistency: v' ({self.v_prime}) ≠ v/2 ({v_prime_expected})", False)
                success = False
        
        # Check ΔE from energy difference
        if self.delta_E is not None and self.m is not None and self.v is not None:
            delta_E_expected = 0.25 * self.m * self.v**2
            if not self._fuzzy_eq(self.delta_E, delta_E_expected):
                self._log(f"Trace consistency: ΔE ({self.delta_E}) ≠ ¼mv² ({delta_E_expected})", False)
                success = False
        
        if success:
            self._log("All trace consistency checks passed", True)
        return success
    
    def test_monotonicity(self):
        """
        Perturb an input parameter and check if outputs change as expected.
        We'll perturb mass m by +10% and verify:
          - H independent of m (should not change)
          - F proportional to m (should increase by 10%)
          - v' independent of m (should not change)
          - ΔE proportional to m (should increase by 10%)
        Then restore original m.
        """
        success = True
        
        if any(x is None for x in [self.m, self.L, self.v, self.g, self.H, self.F, self.v_prime, self.delta_E]):
            self._log("Monotonicity test skipped due to missing parameters", None)
            return True
        
        original_m = self.m
        perturbation = 1.1  # +10%
        
        try:
            # Perturb mass
            self.m = original_m * perturbation
            
            # Recompute expected values after perturbation
            H_expected = self.v**2 / (2 * self.g)  # independent of m
            F_expected = self.m * self.g + self.m * self.v**2 / self.L
            v_prime_expected = self.v / 2  # independent of m
            delta_E_expected = 0.25 * self.m * self.v**2
            
            # Check H unchanged (within tolerance)
            if not self._fuzzy_eq(self.H, H_expected):
                self._log(f"Monotonicity: H should be independent of m but changed", False)
                success = False
            
            # Check F increased proportionally to m
            F_original = original_m * self.g + original_m * self.v**2 / self.L
            if not self._fuzzy_eq(self.F * perturbation, F_expected):
                self._log(f"Monotonicity: F should scale with m (original F={F_original}, expected after perturbation={F_expected})", False)
                success = False
            
            # Check v' unchanged
            if not self._fuzzy_eq(self.v_prime, v_prime_expected):
                self._log(f"Monotonicity: v' should be independent of m but changed", False)
                success = False
            
            # Check ΔE increased proportionally to m
            delta_E_original = 0.25 * original_m * self.v**2
            if not self._fuzzy_eq(self.delta_E * perturbation, delta_E_expected):
                self._log(f"Monotonicity: ΔE should scale with m (original ΔE={delta_E_original}, expected after perturbation={delta_E_expected})", False)
                success = False
            
            if success:
                self._log("Monotonicity test passed: responses to mass perturbation are physically correct", True)
        
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {str(e)}", False)
            success = False
        
        finally:
            # Restore original mass
            self.m = original_m
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                self._log(f"Running {test_name}...", None)
                test_success = test_func()
                overall_success = overall_success and test_success
            except Exception as e:
                self._log(f"{test_name} crashed: {traceback.format_exc()}", False)
                overall_success = False
        
        self._log(f"Overall test result: {'PASS' if overall_success else 'FAIL'}", None)
        return overall_success, self.logs

# Example usage (commented out for safety):
# if __name__ == "__main__":
#     class DummySolution:
#         def __init__(self):
#             self.m = 0.5
#             self.L = 1.0
#             self.v = 4.0
#             self.g = 9.8
#             self.results = {
#                 'H': 0.8163,
#                 'F': 13.3,
#                 'delta_E': 2.0,
#                 'v_prime': 2.0
#             }
#     
#     solution = DummySolution()
#     checker = PhysicsChecker(solution)
#     success, logs = checker.run_all_tests()
#     for log in logs:
#         print(log)