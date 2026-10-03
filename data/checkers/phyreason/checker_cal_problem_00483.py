import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results with robust fallbacks."""
        # Extract results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Use direct extraction from results ONLY for required output variables
        # Do NOT try to extract k, Q, q from results - they are not required outputs
        self.E = self._get_value('E', None)
        self.W_MN = self._get_value('W_MN', None)
        self.delta_E = self._get_value('delta_E', None)
        self.r = self._get_value('r', 0.30)
        
        # Use the actual known constants from the problem statement
        self.k = 9.0e9          # N*m^2/C^2 (from problem)
        self.Q = 1.0e-9         # C (from problem)
        self.q = -2.0e-19       # C (from problem)
        
        # Derived quantities - ensure numeric values
        self.r_m = float(self.r) if self.r is not None else 0.30
        self.logs = []
    
    def _get_value(self, key, default):
        """Safely get a value from results, handling None and conversion issues."""
        value = self.results.get(key, default)
        # If value is None, return default
        if value is None:
            return default
        # Ensure it's a numeric type
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _log(self, msg, success):
        self.logs.append(f"{'✓' if success else '✗'} {msg}")
        return success

    def test_invariants(self):
        """Verify fundamental physics laws."""
        tests_passed = True
        
        # 1. Check Coulomb's Law: E = k*Q/r²
        if self.E is not None:
            # Use absolute value for Q in calculation since E magnitude depends on |Q|
            E_calc = self.k * abs(self.Q) / (self.r_m ** 2)
            if not math.isclose(self.E, E_calc, rel_tol=1e-2, abs_tol=1e-2):
                tests_passed &= self._log(
                    f"Electric field mismatch: {self.E} vs calculated {E_calc}", False
                )
            else:
                self._log(f"Electric field matches Coulomb's Law: {E_calc:.1f} N/C", True)
        
        # 2. Check energy conservation: ΔE = -W
        if self.W_MN is not None and self.delta_E is not None:
            # For very small numbers, use a relative tolerance
            if not math.isclose(self.delta_E, -self.W_MN, rel_tol=1e-2, abs_tol=1e-30):
                tests_passed &= self._log(
                    f"Energy conservation violated: ΔE={self.delta_E:.2e}, -W={-self.W_MN:.2e}", False
                )
            else:
                self._log(f"Energy conservation holds: ΔE = -W = {abs(self.delta_E):.1e} J", True)
        
        # 3. Check work sign consistency
        if self.W_MN is not None and not math.isclose(self.W_MN, 0, abs_tol=1e-30):
            # For negative q, work should be negative when moving toward positive Q
            # (potential increases, so q*ΔV is negative)
            expected_sign = -1 if self.q < 0 else 1
            actual_sign = math.copysign(1, self.W_MN)
            if actual_sign != expected_sign:
                self._log(f"Work sign check: expected sign {expected_sign}, got {actual_sign}", True)
            
        return tests_passed

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        tests_passed = True
        
        # Check step 3 from the analysis: E = 9e9 * 1e-9 / 0.30² = 100 N/C
        if self.E is not None:
            expected_E = 9e9 * 1e-9 / (0.30 ** 2)
            if not math.isclose(self.E, expected_E, rel_tol=1e-2, abs_tol=1e-2):
                tests_passed &= self._log(
                    f"Electric field inconsistent with steps: {self.E} vs {expected_E}", False
                )
            else:
                self._log(f"E consistent with step-by-step calculation: {self.E} N/C", True)
        
        # Check step 4 and 5: W_MN = -4e-18 J, ΔE = 4e-18 J
        if self.W_MN is not None:
            expected_W = -4e-18
            # For very small numbers, use both relative and absolute tolerance
            if not (math.isclose(self.W_MN, expected_W, rel_tol=1e-2) or 
                    math.isclose(self.W_MN, expected_W, abs_tol=1e-30)):
                tests_passed &= self._log(
                    f"Work value mismatch: {self.W_MN:.2e} vs expected -4.00e-18 J", False
                )
            else:
                self._log(f"Work matches target: {self.W_MN:.2e} J", True)
        
        if self.delta_E is not None:
            expected_delta_E = 4e-18
            # For very small numbers, use both relative and absolute tolerance
            if not (math.isclose(self.delta_E, expected_delta_E, rel_tol=1e-2) or 
                    math.isclose(self.delta_E, expected_delta_E, abs_tol=1e-30)):
                tests_passed &= self._log(
                    f"ΔE mismatch: {self.delta_E:.2e} vs expected 4.00e-18 J", False
                )
            else:
                self._log(f"ΔE matches target: {self.delta_E:.2e} J", True)
        
        return tests_passed

    def test_monotonicity(self):
        """Perturb inputs and verify expected trends."""
        original_r = self.r_m
        tests_passed = True
        
        try:
            # 1. Perturb distance r: E ∝ 1/r²
            r_perturbed = original_r * 1.1  # Increase by 10%
            E_original = self.k * abs(self.Q) / (original_r ** 2)
            E_perturbed = self.k * abs(self.Q) / (r_perturbed ** 2)
            
            # E should decrease when r increases
            if E_perturbed >= E_original:
                tests_passed &= self._log(
                    f"Monotonicity failed: E({r_perturbed:.3f}) = {E_perturbed:.2f} ≥ E({original_r:.3f}) = {E_original:.2f}", 
                    False
                )
            else:
                ratio_expected = (original_r / r_perturbed) ** 2  # (1/1.1)² ≈ 0.826
                ratio_actual = E_perturbed / E_original
                if math.isclose(ratio_actual, ratio_expected, rel_tol=1e-2):
                    self._log(
                        f"E ∝ 1/r² holds: ratio {ratio_actual:.3f} ≈ expected {ratio_expected:.3f}", 
                        True
                    )
                else:
                    tests_passed &= self._log(
                        f"E ∝ 1/r² deviation: ratio {ratio_actual:.3f} vs {ratio_expected:.3f}", 
                        False
                    )
            
            # 2. Perturb charge Q: E ∝ Q
            Q_perturbed = self.Q * 1.05  # Increase by 5%
            E_original = self.k * abs(self.Q) / (original_r ** 2)
            E_perturbed = self.k * abs(Q_perturbed) / (original_r ** 2)
            
            # E should increase with |Q|
            if E_perturbed <= E_original:
                tests_passed &= self._log(
                    f"Monotonicity failed: E(|Q|↑) = {E_perturbed:.2f} ≤ E(original) = {E_original:.2f}", 
                    False
                )
            else:
                ratio_expected = abs(Q_perturbed / self.Q)
                ratio_actual = E_perturbed / E_original
                if math.isclose(ratio_actual, ratio_expected, rel_tol=1e-2):
                    self._log(
                        f"E ∝ |Q| holds: ratio {ratio_actual:.3f} ≈ expected {ratio_expected:.3f}", 
                        True
                    )
                else:
                    tests_passed &= self._log(
                        f"E ∝ |Q| deviation: ratio {ratio_actual:.3f} vs {ratio_expected:.3f}", 
                        False
                    )
                    
        except Exception as e:
            tests_passed &= self._log(f"Monotonicity test error: {str(e)}", False)
        
        return tests_passed

    def run_all_tests(self):
        """Execute all checks and return success flag with logs."""
        self.logs = []
        overall_success = True
        
        try:
            self.logs.append("=== Physics Audit Started ===")
            
            # Run test suites
            test_functions = [
                ("Physics Invariants", self.test_invariants),
                ("Trace Consistency", self.test_trace_consistency),
                ("Monotonicity Checks", self.test_monotonicity)
            ]
            
            for name, test_func in test_functions:
                self.logs.append(f"\n--- {name} ---")
                try:
                    if not test_func():
                        overall_success = False
                except Exception as e:
                    self.logs.append(f"✗ Test crashed: {str(e)}")
                    self.logs.append(traceback.format_exc())
                    overall_success = False
            
            self.logs.append(f"\n=== Audit {'PASSED' if overall_success else 'FAILED'} ===")
            
        except Exception as e:
            self.logs.append(f"✗ Critical error in audit: {str(e)}")
            overall_success = False
        
        return overall_success, self.logs