import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from the solution instance."""
        self.solution = solution_instance
        
        # Input parameters (from problem context)
        self.M = 200.0  # kg, total mass of person+boat
        self.m = 5.0    # kg, mass of each ball
        self.N = 20     # number of balls
        self.v0 = 1.0   # m/s, initial speed
        
        # From ground truth: v = 6 m/s is ball speed relative to ground for Q1
        # For Q2, ball is thrown with relative speed 6 m/s to boat
        self.v = 6.0    # m/s, ball speed (relative to ground for Q1, relative to boat for Q2)
        
        # Extracted results with safe fallbacks
        results = getattr(solution_instance, 'results', {})
        self.v_1 = results.get('v_1', 0.0)
        self.v_n = results.get('v_n', 0.0)
        self.delta_v = results.get('delta_v', 0.0)
        self.n = results.get('n', 0)
        
        # Additional derived quantities for checking
        self.total_mass_initial = self.M + self.N * self.m  # 300 kg
        self.logs = []
    
    def _log(self, msg, success):
        """Helper to log test results."""
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {msg}")
        return success
    
    def test_invariants(self):
        """Verify momentum conservation at key steps."""
        tests_passed = True
        
        try:
            # 1. Check momentum conservation for first throw (Q1 physics)
            # According to ground truth step_1: (M + N*m)*v0 = (M + (N-1)*m)*v_1 + m*v
            # where v = 6 m/s is ball speed relative to ground
            initial_momentum = self.total_mass_initial * self.v0
            final_momentum = (self.M + (self.N - 1) * self.m) * self.v_1 + self.m * self.v
            
            if not math.isclose(initial_momentum, final_momentum, rel_tol=1e-2, abs_tol=1e-2):
                tests_passed = self._log(
                    f"Momentum not conserved in first throw: "
                    f"initial={initial_momentum:.3f}, final={final_momentum:.3f}",
                    False
                )
            else:
                tests_passed = self._log(
                    f"Momentum conserved in first throw: "
                    f"Δp={abs(initial_momentum-final_momentum):.2e}",
                    True
                ) and tests_passed
                
            # 2. Check general momentum conservation formula for nth throw (Q1)
            # From step_3: 300*v0 = (300 - n*m)*v_n + n*m*v
            # Test with n=5 as example
            n_test = 5
            v_n_test = ((self.M + self.N * self.m) * self.v0 - n_test * self.m * self.v) / \
                      (self.M + self.N * self.m - n_test * self.m)
            
            # Verify using the formula from solution
            # The solution should have computed v_n for n=16, we can check consistency
            if hasattr(self.solution, 'v_n_expr'):
                # If solution stored the expression, use it
                v_n_from_expr = float(self.solution.v_n_expr.subs({'n': n_test}))
                if not math.isclose(v_n_test, v_n_from_expr, rel_tol=1e-2):
                    tests_passed = self._log(
                        f"General formula test failed for n={n_test}: "
                        f"expected {v_n_test:.4f}, got {v_n_from_expr:.4f}",
                        False
                    )
                else:
                    tests_passed = self._log(f"General formula consistent for n={n_test}", True) and tests_passed
            else:
                tests_passed = self._log("General formula check skipped (no expression)", True) and tests_passed
                
        except Exception as e:
            tests_passed = self._log(f"Invariant test failed with error: {str(e)}", False)
            
        return tests_passed
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        tests_passed = True
        
        try:
            # 1. Check v_1 = 270/295 ≈ 0.9153 (from ground truth)
            expected_v1 = 270/295
            if not math.isclose(self.v_1, expected_v1, rel_tol=1e-2, abs_tol=1e-2):
                tests_passed = self._log(
                    f"v_1 mismatch: expected {expected_v1:.4f}, got {self.v_1:.4f}",
                    False
                )
            else:
                tests_passed = self._log(f"v_1 consistent with expected {expected_v1:.4f}", True) and tests_passed
                
            # 2. Check n (direction reversal)
            # From ground truth step_5: n = 11
            n_expected = 11
            if not math.isclose(self.n, n_expected, abs_tol=0.5):
                tests_passed = self._log(
                    f"Reversal n mismatch: expected {n_expected}, got {self.n}",
                    False
                )
            else:
                tests_passed = self._log(f"Reversal n={self.n} correct", True) and tests_passed
                
            # 3. Check Δv for 16th ball
            # From ground truth step_6: Δv = 2/15 ≈ 0.13333
            expected_delta_v = 2/15
            if not math.isclose(self.delta_v, expected_delta_v, rel_tol=1e-2, abs_tol=1e-2):
                tests_passed = self._log(
                    f"Δv mismatch: expected {expected_delta_v:.4f}, got {self.delta_v:.4f}",
                    False
                )
            else:
                tests_passed = self._log(f"Δv={self.delta_v:.4f} consistent with 2/15", True) and tests_passed
                
        except Exception as e:
            tests_passed = self._log(f"Trace consistency test failed: {str(e)}", False)
            
        return tests_passed
    
    def test_monotonicity(self):
        """Perturb mass m and verify expected trends."""
        original_m = self.m
        tests_passed = True
        
        try:
            # Increase m by 10%
            perturbed_m = original_m * 1.1
            
            # Recalculate v_1 with perturbed m using Q1 physics
            # (M + N*m)*v0 = (M + (N-1)*m)*v_1 + m*v
            # Solve for v_1: v_1 = [(M + N*m)*v0 - m*v] / [M + (N-1)*m]
            perturbed_v1 = ((self.M + self.N * perturbed_m) * self.v0 - perturbed_m * self.v) / \
                          (self.M + (self.N - 1) * perturbed_m)
            original_v1 = ((self.M + self.N * original_m) * self.v0 - original_m * self.v) / \
                         (self.M + (self.N - 1) * original_m)
            
            # Since m appears in both numerator (negatively) and denominator,
            # increasing m should decrease v_1
            if perturbed_v1 >= original_v1:
                tests_passed = self._log(
                    f"Monotonicity violated: increasing m from {original_m} to {perturbed_m} "
                    f"should decrease v_1, but {perturbed_v1:.4f} >= {original_v1:.4f}",
                    False
                )
            else:
                tests_passed = self._log(
                    f"Monotonicity OK: v_1 decreases from {original_v1:.4f} to {perturbed_v1:.4f} "
                    f"when m increases by 10%",
                    True
                ) and tests_passed
                
            # Recalculate reversal n with perturbed m
            # From v_n < 0 condition: (M + N*m)*v0 - n*m*v < 0
            # So n > (M + N*m)*v0/(m*v)
            perturbed_n_crit = (self.M + self.N * perturbed_m) * self.v0 / (perturbed_m * self.v)
            original_n_crit = (self.M + self.N * original_m) * self.v0 / (original_m * self.v)
            
            # First integer n where reversal occurs
            perturbed_n = int(perturbed_n_crit) + 1
            original_n = int(original_n_crit) + 1
            
            # Check it's not wildly different
            if abs(perturbed_n - original_n) > 2:
                tests_passed = self._log(
                    f"Reversal n change too large: {original_n} -> {perturbed_n} "
                    f"with 10% m increase",
                    False
                )
            else:
                tests_passed = self._log(
                    f"Reversal n changes reasonably: {original_n} -> {perturbed_n}",
                    True
                ) and tests_passed
                
        except Exception as e:
            tests_passed = self._log(f"Monotonicity test failed: {str(e)}", False)
            
        return tests_passed
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = ["Physics Checker started"]
        success = True
        
        try:
            success = self.test_invariants() and success
            success = self.test_trace_consistency() and success
            success = self.test_monotonicity() and success
        except Exception as e:
            self.logs.append(f"ERROR: Critical failure in test execution: {str(e)}")
            self.logs.append(traceback.format_exc())
            success = False
            
        overall_msg = "ALL TESTS PASSED" if success else "SOME TESTS FAILED"
        self.logs.append(f"=== {overall_msg} ===")
        
        return success, self.logs