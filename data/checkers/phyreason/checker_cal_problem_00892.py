import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extracts all required inputs and results from the solution instance.
        Uses robust getattr with fallback to .results dictionary.
        """
        self.solution = solution_instance
        
        # Extract problem constants (safe access)
        self.N = getattr(solution_instance, 'N', 200)
        self.S = getattr(solution_instance, 'S', 1.0)
        self.r = getattr(solution_instance, 'r', 10.0)
        self.R_ext = 50.0  # External resistor from problem statement
        
        # Extract results (safe access)
        self.E = getattr(solution_instance, 'E', getattr(solution_instance, 'results', {}).get('E', 0.0))
        self.I = getattr(solution_instance, 'I', getattr(solution_instance, 'results', {}).get('I', 0.0))
        self.I_1 = getattr(solution_instance, 'I_1', getattr(solution_instance, 'results', {}).get('I_1', 0.0))
        self.R_max = getattr(solution_instance, 'R', getattr(solution_instance, 'results', {}).get('R', 0.0))
        self.P = getattr(solution_instance, 'P', getattr(solution_instance, 'results', {}).get('P', 0.0))
        
        # Convert any sympy expressions to float if needed
        self._convert_to_float()
        
    def _convert_to_float(self):
        """Convert sympy expressions to float for numerical comparison."""
        for attr in ['E', 'I', 'I_1', 'R_max', 'P', 'N', 'S', 'r', 'R_ext']:
            value = getattr(self, attr)
            if hasattr(value, 'evalf'):
                setattr(self, attr, float(value.evalf()))
    
    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison with default 1% tolerance."""
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """
        Test fundamental physics laws:
        1. Ohm's Law for both circuit configurations
        2. Power conservation and formulas
        """
        logs = []
        
        # Test 1: Ohm's Law for initial circuit (R_ext = 50Ω)
        expected_I = self.E / (self.R_ext + self.r)
        if self._isclose(self.I, expected_I):
            logs.append(f"✓ Ohm's Law satisfied: I = {self.I:.4f} ≈ E/(R+r) = {expected_I:.4f}")
        else:
            logs.append(f"✗ Ohm's Law violation: I = {self.I:.4f}, but E/(R+r) = {expected_I:.4f}")
        
        # Test 2: Ohm's Law for maximum power configuration (R = r)
        expected_I1 = self.E / (2 * self.r)
        if self._isclose(self.I_1, expected_I1):
            logs.append(f"✓ Ohm's Law for max power: I_1 = {self.I_1:.4f} ≈ E/(2r) = {expected_I1:.4f}")
        else:
            logs.append(f"✗ Ohm's Law violation for max power: I_1 = {self.I_1:.4f}, but E/(2r) = {expected_I1:.4f}")
        
        # Test 3: Power formula consistency
        expected_P = self.I_1**2 * self.r
        if self._isclose(self.P, expected_P):
            logs.append(f"✓ Power formula: P = {self.P:.4f} ≈ I_1²·r = {expected_P:.4f}")
        else:
            logs.append(f"✗ Power formula violation: P = {self.P:.4f}, but I_1²·r = {expected_P:.4f}")
        
        # Test 4: Maximum power transfer theorem (R should equal r)
        if self._isclose(self.R_max, self.r):
            logs.append(f"✓ Maximum power condition: R_max = {self.R_max:.4f} ≈ r = {self.r:.4f}")
        else:
            logs.append(f"✗ Max power condition violation: R_max = {self.R_max:.4f}, but r = {self.r:.4f}")
        
        # Test 5: Alternative power formula P_max = E²/(4r)
        alt_P = self.E**2 / (4 * self.r)
        if self._isclose(self.P, alt_P):
            logs.append(f"✓ Alternative power formula: P = {self.P:.4f} ≈ E²/(4r) = {alt_P:.4f}")
        else:
            logs.append(f"✗ Alternative power formula mismatch: P = {self.P:.4f}, E²/(4r) = {alt_P:.4f}")
        
        # All tests must pass for invariant validation
        all_passed = len([l for l in logs if l.startswith('✗')]) == 0
        return all_passed, logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results against final answer.
        Based on the STEPS ANALYSIS provided.
        """
        logs = []
        
        # Step 1: EMF from Faraday's Law (we don't have ΔB/Δt, but can check relative values)
        # E should be 10V according to steps
        if self._isclose(self.E, 10.0):
            logs.append(f"✓ Step 1 EMF: E = {self.E:.4f} ≈ 10V")
        else:
            logs.append(f"✗ Step 1 EMF mismatch: E = {self.E:.4f}, expected ~10V")
        
        # Step 2: Initial current I = 1/6 A ≈ 0.1667 A
        expected_I = 1.0/6.0
        if self._isclose(self.I, expected_I):
            logs.append(f"✓ Step 2 current: I = {self.I:.4f} ≈ 1/6 A")
        else:
            logs.append(f"✗ Step 2 current mismatch: I = {self.I:.4f}, expected ~{expected_I:.4f}")
        
        # Step 3: Maximum power resistance R = r = 10Ω
        if self._isclose(self.R_max, 10.0):
            logs.append(f"✓ Step 3 resistance: R_max = {self.R_max:.4f} ≈ 10Ω")
        else:
            logs.append(f"✗ Step 3 resistance mismatch: R_max = {self.R_max:.4f}, expected ~10Ω")
        
        # Step 4: Current for max power I_1 = E/(2r) = 10/20 = 0.5 A
        expected_I1 = self.E / (2 * self.r)
        if self._isclose(self.I_1, expected_I1):
            logs.append(f"✓ Step 4 current: I_1 = {self.I_1:.4f} ≈ E/(2r) = {expected_I1:.4f}")
        else:
            logs.append(f"✗ Step 4 current mismatch: I_1 = {self.I_1:.4f}, expected ~{expected_I1:.4f}")
        
        # Step 5: Maximum power P = 2.5 W
        if self._isclose(self.P, 2.5):
            logs.append(f"✓ Step 5 power: P = {self.P:.4f} ≈ 2.5W")
        else:
            logs.append(f"✗ Step 5 power mismatch: P = {self.P:.4f}, expected ~2.5W")
        
        # Check chain consistency: if E is off, everything else should scale appropriately
        # This is a stronger consistency check
        if not self._isclose(self.E, 10.0):
            # Recompute expected values based on actual E
            expected_I = self.E / (self.R_ext + self.r)
            expected_R_max = self.r
            expected_I1 = self.E / (2 * self.r)
            expected_P = self.E**2 / (4 * self.r)
            
            chain_ok = (
                self._isclose(self.I, expected_I) and
                self._isclose(self.R_max, expected_R_max) and
                self._isclose(self.I_1, expected_I1) and
                self._isclose(self.P, expected_P)
            )
            if chain_ok:
                logs.append("✓ Results are internally consistent despite EMF deviation")
            else:
                logs.append("✗ Internal inconsistency in the computation chain")
        
        all_passed = len([l for l in logs if l.startswith('✗')]) == 0
        return all_passed, logs
    
    def test_monotonicity(self):
        """
        Verify monotonic relationships using numerical reasoning.
        Since we don't have ΔB/Δt to recompute with different N,
        we verify the proportionality relationships are correct.
        """
        logs = []
        
        try:
            logs.append("Monotonicity test: Checking proportionality relationships")
            
            # From physics laws:
            # 1. E ∝ N (Faraday's Law: E = N * S * (ΔB/Δt))
            # 2. I ∝ E ∝ N (Ohm's Law: I = E/(R+r))
            # 3. I_1 ∝ E ∝ N (Ohm's Law for max power: I_1 = E/(2r))
            # 4. P ∝ E² ∝ N² (Power: P = E²/(4r))
            
            # We can't test with different N values, but we can verify
            # that the relationships between computed values are correct
            
            # Check that E is positive (monotonicity condition)
            if self.E > 0:
                logs.append(f"✓ E = {self.E:.4f} > 0 (positive EMF)")
            else:
                logs.append(f"✗ E = {self.E:.4f} ≤ 0 (non-positive EMF)")
            
            # Check that I and I_1 are positive
            if self.I > 0:
                logs.append(f"✓ I = {self.I:.4f} > 0 (positive current)")
            else:
                logs.append(f"✗ I = {self.I:.4f} ≤ 0 (non-positive current)")
                
            if self.I_1 > 0:
                logs.append(f"✓ I_1 = {self.I_1:.4f} > 0 (positive max-power current)")
            else:
                logs.append(f"✗ I_1 = {self.I_1:.4f} ≤ 0 (non-positive max-power current)")
            
            # Check that P is positive
            if self.P > 0:
                logs.append(f"✓ P = {self.P:.4f} > 0 (positive power)")
            else:
                logs.append(f"✗ P = {self.P:.4f} ≤ 0 (non-positive power)")
            
            # Verify the proportionality chain
            # If we hypothetically double N, E should double, I should double, etc.
            # We can check that the ratios are consistent with the formulas
            
            # Check I/E ratio = 1/(R_ext + r)
            expected_I_over_E = 1.0 / (self.R_ext + self.r)
            actual_I_over_E = self.I / self.E if self.E != 0 else 0
            if self._isclose(actual_I_over_E, expected_I_over_E):
                logs.append(f"✓ I/E ratio correct: {actual_I_over_E:.4f} ≈ 1/(R+r) = {expected_I_over_E:.4f}")
            else:
                logs.append(f"✗ I/E ratio incorrect: {actual_I_over_E:.4f} ≠ 1/(R+r) = {expected_I_over_E:.4f}")
            
            # Check I_1/E ratio = 1/(2r)
            expected_I1_over_E = 1.0 / (2 * self.r)
            actual_I1_over_E = self.I_1 / self.E if self.E != 0 else 0
            if self._isclose(actual_I1_over_E, expected_I1_over_E):
                logs.append(f"✓ I_1/E ratio correct: {actual_I1_over_E:.4f} ≈ 1/(2r) = {expected_I1_over_E:.4f}")
            else:
                logs.append(f"✗ I_1/E ratio incorrect: {actual_I1_over_E:.4f} ≠ 1/(2r) = {expected_I1_over_E:.4f}")
            
            # Check P/E² ratio = 1/(4r)
            expected_P_over_E2 = 1.0 / (4 * self.r)
            actual_P_over_E2 = self.P / (self.E**2) if self.E != 0 else 0
            if self._isclose(actual_P_over_E2, expected_P_over_E2):
                logs.append(f"✓ P/E² ratio correct: {actual_P_over_E2:.4f} ≈ 1/(4r) = {expected_P_over_E2:.4f}")
            else:
                logs.append(f"✗ P/E² ratio incorrect: {actual_P_over_E2:.4f} ≠ 1/(4r) = {expected_P_over_E2:.4f}")
            
            # All positivity checks must pass
            all_passed = (self.E > 0 and self.I > 0 and self.I_1 > 0 and self.P > 0)
            
            # Check ratio consistency
            ratio_checks = [
                self._isclose(actual_I_over_E, expected_I_over_E),
                self._isclose(actual_I1_over_E, expected_I1_over_E),
                self._isclose(actual_P_over_E2, expected_P_over_E2)
            ]
            
            if all(ratio_checks):
                logs.append("✓ All proportionality relationships are correct")
                all_passed = all_passed and True
            else:
                logs.append("✗ Some proportionality relationships are incorrect")
                all_passed = False
                
        except Exception as e:
            logs.append(f"✗ Monotonicity test failed with error: {str(e)}")
            all_passed = False
            
        return all_passed, logs
    
    def run_all_tests(self):
        """
        Execute all physics audits and return aggregated results.
        Returns: (success_bool, logs_list)
        """
        all_logs = []
        all_passed = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            try:
                passed, logs = test_method()
                all_logs.append(f"\n=== {test_name} ===")
                all_logs.extend(logs)
                if not passed:
                    all_passed = False
            except Exception as e:
                all_logs.append(f"\n=== {test_name} ===")
                all_logs.append(f"✗ Test crashed: {str(e)}")
                all_logs.append(traceback.format_exc())
                all_passed = False
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        if all_passed:
            all_logs.append("✓ All physics checks passed")
        else:
            all_logs.append("✗ Some physics checks failed")
        
        return all_passed, all_logs