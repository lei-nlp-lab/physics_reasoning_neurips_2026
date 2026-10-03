import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        """
        # Extract inputs safely with defaults
        self.Q = getattr(solution_instance, 'Q', 5.0)  # m³/s
        self.h = getattr(solution_instance, 'h', 10.0)  # m
        self.U1 = getattr(solution_instance, 'U1', 250.0)  # V
        self.r = getattr(solution_instance, 'r', 4.0)  # Ω
        self.n3_over_n4 = getattr(solution_instance, 'n3_over_n4', 190/11)  # ratio
        self.U4 = getattr(solution_instance, 'U4', 220.0)  # V
        self.P4 = getattr(solution_instance, 'P4', 1.9e5)  # W
        self.rho = getattr(solution_instance, 'rho', 1e3)  # kg/m³
        self.g = getattr(solution_instance, 'g', 10.0)  # m/s²
        
        # Extract results safely - use correct keys from contract
        results = getattr(solution_instance, 'results', {})
        
        # Expected keys with fallbacks - match contract variable names exactly
        self.P3 = results.get('P_3', 0.0)  # Contract has P_3, not P3
        self.delta_P = results.get('delta_P', 0.0)
        self.n1_over_n2 = results.get('n_1/n_2', 0.0)
        self.P0 = results.get('P_0', 0.0)
        self.eta = results.get('η', 0.0)  # Contract has η (Greek eta)
        self.P1 = results.get('P_1', 0.0)
        self.U2 = results.get('U_2', 0.0)
        self.U3 = results.get('U_3', 0.0)
        self.delta_U = results.get('delta_U', 0.0)
        self.I2 = results.get('I_2', 0.0)
        
        self.logs = []

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def _log(self, message):
        """Helper for logging."""
        self.logs.append(message)

    def test_invariants(self):
        """Verify physical invariants using the KB reference."""
        try:
            # 1. Mechanical power from potential energy: P0 = ρ * Q * g * h
            P0_expected = self.rho * self.Q * self.g * self.h
            if not self._is_close(self.P0, P0_expected):
                self._log(f"Invariant violated: P0 = {self.P0}, expected {P0_expected}")
                return False
            self._log("✓ Mechanical power invariant passed.")

            # 2. Power conservation in ideal transformers: P3 = P4, P1 = P3 + ΔP
            if not self._is_close(self.P3, self.P4):
                self._log(f"Invariant violated: P3 = {self.P3}, P4 = {self.P4}")
                return False
            P1_expected = self.P3 + self.delta_P
            if not self._is_close(self.P1, P1_expected):
                self._log(f"Invariant violated: P1 = {self.P1}, expected {P1_expected}")
                return False
            self._log("✓ Transformer power conservation invariant passed.")

            # 3. Ohm's Law for transmission line: ΔU = I2 * r, ΔP = I2² * r
            delta_U_expected = self.I2 * self.r
            if not self._is_close(self.delta_U, delta_U_expected):
                self._log(f"Invariant violated: ΔU = {self.delta_U}, expected {delta_U_expected}")
                return False
            delta_P_expected = self.I2**2 * self.r
            if not self._is_close(self.delta_P, delta_P_expected):
                self._log(f"Invariant violated: ΔP = {self.delta_P}, expected {delta_P_expected}")
                return False
            self._log("✓ Ohm's Law invariants passed.")

            # 4. Ideal transformer voltage ratios: U3 = U4 * (n3/n4), U1/U2 = n1/n2
            U3_expected = self.U4 * self.n3_over_n4
            if not self._is_close(self.U3, U3_expected):
                self._log(f"Invariant violated: U3 = {self.U3}, expected {U3_expected}")
                return False
            n1_over_n2_expected = self.U1 / self.U2 if self.U2 != 0 else 0
            if not self._is_close(self.n1_over_n2, n1_over_n2_expected):
                self._log(f"Invariant violated: n1/n2 = {self.n1_over_n2}, expected {n1_over_n2_expected}")
                return False
            self._log("✓ Transformer voltage ratio invariants passed.")

            # 5. Series voltage addition: U2 = U3 + ΔU
            U2_expected = self.U3 + self.delta_U
            if not self._is_close(self.U2, U2_expected):
                self._log(f"Invariant violated: U2 = {self.U2}, expected {U2_expected}")
                return False
            self._log("✓ Series voltage addition invariant passed.")

            # 6. Efficiency: η = (P1 / P0) * 100% - note η is stored as decimal in results
            eta_expected = (self.P1 / self.P0) * 100 if self.P0 != 0 else 0
            # The code stores η as decimal (0.4), but checker expects percentage (40.0)
            # Convert stored decimal to percentage for comparison
            eta_stored_percent = self.eta * 100
            if not self._is_close(eta_stored_percent, eta_expected):
                self._log(f"Invariant violated: η = {eta_stored_percent}%, expected {eta_expected}%")
                return False
            self._log("✓ Efficiency invariant passed.")

            return True
        except Exception as e:
            self._log(f"Error in test_invariants: {traceback.format_exc()}")
            return False

    def test_trace_consistency(self):
        """Cross-check intermediate results with STEPS ANALYSIS."""
        try:
            # Step 1: U3 = U4 * (n3/n4)
            U3_expected = self.U4 * self.n3_over_n4
            if not self._is_close(self.U3, U3_expected):
                self._log(f"Step 1 mismatch: U3 = {self.U3}, expected {U3_expected}")
                return False

            # Step 2: P3 = P4
            if not self._is_close(self.P3, self.P4):
                self._log(f"Step 2 mismatch: P3 = {self.P3}, expected {self.P4}")
                return False

            # Step 3: I2 = P3 / U3
            I2_expected = self.P3 / self.U3 if self.U3 != 0 else 0
            if not self._is_close(self.I2, I2_expected):
                self._log(f"Step 3 mismatch: I2 = {self.I2}, expected {I2_expected}")
                return False

            # Step 4: ΔU = I2 * r
            delta_U_expected = self.I2 * self.r
            if not self._is_close(self.delta_U, delta_U_expected):
                self._log(f"Step 4 mismatch: ΔU = {self.delta_U}, expected {delta_U_expected}")
                return False

            # Step 5: U2 = U3 + ΔU
            U2_expected = self.U3 + self.delta_U
            if not self._is_close(self.U2, U2_expected):
                self._log(f"Step 5 mismatch: U2 = {self.U2}, expected {U2_expected}")
                return False

            # Step 6: n1/n2 = U1 / U2
            n1_over_n2_expected = self.U1 / self.U2 if self.U2 != 0 else 0
            if not self._is_close(self.n1_over_n2, n1_over_n2_expected):
                self._log(f"Step 6 mismatch: n1/n2 = {self.n1_over_n2}, expected {n1_over_n2_expected}")
                return False

            # Step 7: ΔP = I2² * r
            delta_P_expected = self.I2**2 * self.r
            if not self._is_close(self.delta_P, delta_P_expected):
                self._log(f"Step 7 mismatch: ΔP = {self.delta_P}, expected {delta_P_expected}")
                return False

            # Step 8: P1 = P3 + ΔP
            P1_expected = self.P3 + self.delta_P
            if not self._is_close(self.P1, P1_expected):
                self._log(f"Step 8 mismatch: P1 = {self.P1}, expected {P1_expected}")
                return False

            # Step 9: P0 = ρ * Q * g * h
            P0_expected = self.rho * self.Q * self.g * self.h
            if not self._is_close(self.P0, P0_expected):
                self._log(f"Step 9 mismatch: P0 = {self.P0}, expected {P0_expected}")
                return False

            # Step 10: η = (P1 / P0) * 100% - convert stored decimal to percentage
            eta_expected = (self.P1 / self.P0) * 100 if self.P0 != 0 else 0
            eta_stored_percent = self.eta * 100
            if not self._is_close(eta_stored_percent, eta_expected):
                self._log(f"Step 10 mismatch: η = {eta_stored_percent}%, expected {eta_expected}%")
                return False

            self._log("✓ All trace consistency checks passed.")
            return True
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {traceback.format_exc()}")
            return False

    def test_monotonicity(self):
        """
        Perturb input (head h) and verify monotonic trend in efficiency η.
        η = P1 / P0, P0 ∝ h, so if h increases, η decreases (since P1 is fixed).
        """
        try:
            # Save original values - note η is stored as decimal
            original_h = self.h
            original_eta = self.eta  # decimal (0.4)
            original_P0 = self.P0

            # Perturb h by +10%
            perturbed_h = original_h * 1.1
            perturbed_P0 = self.rho * self.Q * self.g * perturbed_h
            # P1 remains constant (see STEPS ANALYSIS: P1 depends on P4 and transformer ratios, not h)
            perturbed_eta = (self.P1 / perturbed_P0)  # decimal

            # Check monotonicity: if h increases, η should decrease
            if perturbed_h > original_h and perturbed_eta < original_eta:
                self._log("✓ Monotonicity passed: increasing h decreases η as expected.")
                return True
            else:
                self._log(f"Monotonicity failed: h={original_h}->{perturbed_h}, η={original_eta}->{perturbed_eta}")
                return False
        except Exception as e:
            self._log(f"Error in test_monotonicity: {traceback.format_exc()}")
            return False

    def run_all_tests(self):
        """Execute all tests and return success and logs."""
        self.logs.clear()
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        success = True
        for test_name, test_func in tests:
            self._log(f"\n--- Running {test_name} Test ---")
            if not test_func():
                success = False
                self._log(f"✗ {test_name} test failed.")
            else:
                self._log(f"✓ {test_name} test passed.")
        
        return success, self.logs