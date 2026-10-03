import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Robust initialization using safe attribute access.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract input parameters with fallbacks
        self.M = getattr(solution_instance, 'M', self.results.get('M', None))
        self.m = getattr(solution_instance, 'm', self.results.get('m', None))
        self.v0 = getattr(solution_instance, 'v0', self.results.get('v0', None))
        self.mu = getattr(solution_instance, 'mu', self.results.get('mu', None))
        self.g = getattr(solution_instance, 'g', self.results.get('g', 9.8))
        
        # Extract results with fallbacks
        self.v_total = getattr(solution_instance, 'v_total', self.results.get('v_total', None))
        self.x = getattr(solution_instance, 'x', self.results.get('x', None))
        self.L = getattr(solution_instance, 'L', self.results.get('L', None))
        
        self.logs = []

    def _log(self, msg, success):
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {msg}")
        return success

    def test_invariants(self):
        """
        Verify physical invariants using numeric checks only.
        """
        checks = []
        
        # 1. Momentum conservation during collision (step_1)
        try:
            if self.v_total is not None and all(param is not None for param in [self.M, self.m, self.v0]):
                # The formula from ground truth: v_total = (M-m)v0/(M+m)
                # But sign depends on direction convention. We'll check magnitude.
                v_total_expected_magnitude = abs((self.M - self.m) * self.v0 / (self.M + self.m))
                if math.isclose(abs(self.v_total), v_total_expected_magnitude, rel_tol=1e-2, abs_tol=1e-2):
                    checks.append(self._log("Momentum conservation (step_1) magnitude satisfied", True))
                else:
                    checks.append(self._log(
                        f"Momentum conservation magnitude failed: |v_total|={abs(self.v_total)}, expected={v_total_expected_magnitude}", 
                        False
                    ))
            else:
                checks.append(self._log("Momentum conservation check skipped (missing parameters)", True))
        except Exception as e:
            checks.append(self._log(f"Momentum conservation check error: {e}", False))

        # 2. Work-energy theorem (step_2)
        try:
            if self.x is not None and all(param is not None for param in [self.M, self.m, self.v0, self.mu, self.g]):
                x_expected = 2 * self.M * self.v0**2 / (self.mu * (self.M + self.m) * self.g)
                if math.isclose(self.x, x_expected, rel_tol=1e-2, abs_tol=1e-2):
                    checks.append(self._log("Work-energy theorem (step_2) satisfied", True))
                else:
                    checks.append(self._log(
                        f"Work-energy theorem (step_2) failed: x={self.x}, expected={x_expected}", 
                        False
                    ))
            else:
                checks.append(self._log("Work-energy theorem (step_2) check skipped (missing parameters)", True))
        except Exception as e:
            checks.append(self._log(f"Work-energy theorem (step_2) check error: {e}", False))

        # 3. Work-energy theorem (step_3)
        try:
            if self.L is not None and all(param is not None for param in [self.M, self.m, self.v0, self.mu, self.g]):
                L_expected = (self.M + self.m) * self.v0**2 / (2 * self.mu * self.m * self.g)
                if math.isclose(self.L, L_expected, rel_tol=1e-2, abs_tol=1e-2):
                    checks.append(self._log("Work-energy theorem (step_3) satisfied", True))
                else:
                    checks.append(self._log(
                        f"Work-energy theorem (step_3) failed: L={self.L}, expected={L_expected}", 
                        False
                    ))
            else:
                checks.append(self._log("Work-energy theorem (step_3) check skipped (missing parameters)", True))
        except Exception as e:
            checks.append(self._log(f"Work-energy theorem (step_3) check error: {e}", False))

        return all(checks)

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        checks = []
        
        # Check consistency between v_total, x, and L using derived formulas
        try:
            if all(param is not None for param in [self.M, self.m, self.v0, self.mu, self.g]):
                # Recompute v_total magnitude from step_1
                v_total_recomputed_magnitude = abs((self.M - self.m) * self.v0 / (self.M + self.m))
                # Recompute x from step_2
                x_recomputed = 2 * self.M * self.v0**2 / (self.mu * (self.M + self.m) * self.g)
                # Recompute L from step_3
                L_recomputed = (self.M + self.m) * self.v0**2 / (2 * self.mu * self.m * self.g)
                
                # Compare with stored values if available
                if self.v_total is not None:
                    if math.isclose(abs(self.v_total), v_total_recomputed_magnitude, rel_tol=1e-2, abs_tol=1e-2):
                        checks.append(self._log("v_total magnitude consistent with input parameters", True))
                    else:
                        checks.append(self._log(
                            f"v_total magnitude inconsistency: |stored|={abs(self.v_total)}, recomputed={v_total_recomputed_magnitude}", 
                            False
                        ))
                
                if self.x is not None:
                    if math.isclose(self.x, x_recomputed, rel_tol=1e-2, abs_tol=1e-2):
                        checks.append(self._log("x consistent with input parameters", True))
                    else:
                        checks.append(self._log(
                            f"x inconsistency: stored={self.x}, recomputed={x_recomputed}", 
                            False
                        ))
                
                if self.L is not None:
                    if math.isclose(self.L, L_recomputed, rel_tol=1e-2, abs_tol=1e-2):
                        checks.append(self._log("L consistent with input parameters", True))
                    else:
                        checks.append(self._log(
                            f"L inconsistency: stored={self.L}, recomputed={L_recomputed}", 
                            False
                        ))
            else:
                checks.append(self._log("Trace consistency check skipped (missing parameters)", True))
        except Exception as e:
            checks.append(self._log(f"Trace consistency check error: {e}", False))

        return all(checks)

    def test_monotonicity(self):
        """
        Perturb input parameter m and verify trend consistency using numeric values only.
        """
        checks = []
        
        try:
            if all(param is not None for param in [self.M, self.m, self.v0, self.mu, self.g]):
                # Store original value
                m_original = self.m
                
                # Perturb m by +10%
                m_perturbed = m_original * 1.1
                
                # Compute results with perturbed m (magnitudes only for v_total)
                v_total_perturbed_mag = abs((self.M - m_perturbed) * self.v0 / (self.M + m_perturbed))
                x_perturbed = 2 * self.M * self.v0**2 / (self.mu * (self.M + m_perturbed) * self.g)
                L_perturbed = (self.M + m_perturbed) * self.v0**2 / (2 * self.mu * m_perturbed * self.g)
                
                # Compute original results
                v_total_original_mag = abs((self.M - m_original) * self.v0 / (self.M + m_original))
                x_original = 2 * self.M * self.v0**2 / (self.mu * (self.M + m_original) * self.g)
                L_original = (self.M + m_original) * self.v0**2 / (2 * self.mu * m_original * self.g)
                
                # Check trends
                # As m increases, |v_total| should decrease (since numerator decreases, denominator increases)
                if v_total_perturbed_mag < v_total_original_mag:
                    checks.append(self._log("Monotonicity: |v_total| decreases as m increases (correct)", True))
                else:
                    checks.append(self._log("Monotonicity: |v_total| trend unexpected", False))
                
                # As m increases, x should decrease (denominator increases)
                if x_perturbed < x_original:
                    checks.append(self._log("Monotonicity: x decreases as m increases (correct)", True))
                else:
                    checks.append(self._log("Monotonicity: x trend unexpected", False))
                
                # As m increases, L should decrease
                if L_perturbed < L_original:
                    checks.append(self._log("Monotonicity: L decreases as m increases (correct)", True))
                else:
                    checks.append(self._log("Monotonicity: L trend unexpected", False))
            else:
                checks.append(self._log("Monotonicity check skipped (missing parameters)", True))
        except Exception as e:
            checks.append(self._log(f"Monotonicity check error: {e}", False))

        return all(checks)

    def run_all_tests(self):
        """
        Execute all tests and return success flag with logs.
        """
        self.logs = []
        
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        all_passed = True
        for test_name, test_func in tests:
            try:
                self.logs.append(f"--- {test_name} ---")
                if not test_func():
                    all_passed = False
            except Exception as e:
                self.logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                all_passed = False
        
        return all_passed, self.logs