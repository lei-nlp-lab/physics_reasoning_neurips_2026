import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract all expected results from the solution instance
        self.results = {}
        expected_keys = ['P', 'P_N', 'U', 'P_chemical', 'R', 'I', 'P_Q', 'I_N']
        for key in expected_keys:
            # First try to get from solution_instance.attr, then from results dict
            val = getattr(solution_instance, key, None)
            if val is None:
                # If not an attribute, try to get from results dictionary
                if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
                    val = solution_instance.results.get(key, 0.0)
                else:
                    val = 0.0
            self.results[key] = float(val) if isinstance(val, (int, float)) else 0.0

        # Extract given constants (should be present in solution_instance)
        self.E = getattr(solution_instance, 'E', 12.0)
        self.r = getattr(solution_instance, 'r', 0.5)
        self.I1 = getattr(solution_instance, 'I1', 4.0)
        self.I2 = getattr(solution_instance, 'I2', 8.0)
        self.r_A = getattr(solution_instance, 'r_A', 1.0)

        # Expected intermediate values from steps (for trace consistency)
        self.expected = {
            'R': 2.5,
            'P_N_step2': 40.0,
            'U': 8.0,
            'I_N': 3.2,
            'P_N_step5': 25.6,
            'I': 4.8,
            'P': 38.4,
            'P_Q': 23.04,
            'P_chemical': 15.36
        }

        # Tolerances for fuzzy matching
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2

        # Logs
        self.logs = []

    def _is_close(self, a, b):
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)

    def _log(self, msg, success):
        self.logs.append((msg, success))
        return success

    def test_invariants(self):
        """Test physical invariants: Ohm's law and energy conservation."""
        success = True

        # 1. Ohm's law for the first circuit (only S1 closed)
        R_calc = self.E / self.I1 - self.r
        if not self._is_close(R_calc, self.results.get('R', 0.0)):
            success = self._log(
                f"Ohm's law (first circuit): R_calc = {R_calc}, stored R = {self.results.get('R', 0.0)}",
                False
            )
        else:
            self._log("Ohm's law (first circuit) satisfied.", True)

        # 2. Ohm's law for the whole circuit when both switches closed
        U_calc = self.E - self.I2 * self.r
        if not self._is_close(U_calc, self.results.get('U', 0.0)):
            success = self._log(
                f"Ohm's law (both closed): U_calc = {U_calc}, stored U = {self.results.get('U', 0.0)}",
                False
            )
        else:
            self._log("Ohm's law (both closed) satisfied.", True)

        # 3. Ohm's law for the furnace when both switches closed
        if 'R' in self.results and 'U' in self.results:
            I_N_calc = self.results['U'] / self.results['R']
            if not self._is_close(I_N_calc, self.results.get('I_N', 0.0)):
                success = self._log(
                    f"Ohm's law (furnace): I_N_calc = {I_N_calc}, stored I_N = {self.results.get('I_N', 0.0)}",
                    False
                )
            else:
                self._log("Ohm's law (furnace) satisfied.", True)

        # 4. Energy conservation in the electrolytic cell: P = P_Q + P_chemical
        P = self.results.get('P', 0.0)
        P_Q = self.results.get('P_Q', 0.0)
        P_chemical = self.results.get('P_chemical', 0.0)
        if not self._is_close(P, P_Q + P_chemical):
            success = self._log(
                f"Energy conservation in cell: P = {P}, P_Q + P_chemical = {P_Q + P_chemical}",
                False
            )
        else:
            self._log("Energy conservation in cell satisfied.", True)

        # 5. Check power formulas for the furnace (both cases)
        # Case 1: only S1 closed
        P_N1_calc = self.I1**2 * self.results.get('R', 0.0)
        # We don't have a separate key for P_N in first case, but it is part of first answer.
        # We'll check against expected value from steps.
        if not self._is_close(P_N1_calc, self.expected['P_N_step2']):
            success = self._log(
                f"Power (furnace, only S1): calculated {P_N1_calc}, expected {self.expected['P_N_step2']}",
                False
            )
        else:
            self._log("Power formula (furnace, only S1) satisfied.", True)

        # Case 2: both closed
        if 'U' in self.results and 'I_N' in self.results:
            P_N2_calc = self.results['U'] * self.results['I_N']
            if not self._is_close(P_N2_calc, self.expected['P_N_step5']):
                success = self._log(
                    f"Power (furnace, both closed): calculated {P_N2_calc}, expected {self.expected['P_N_step5']}",
                    False
                )
            else:
                self._log("Power formula (furnace, both closed) satisfied.", True)

        return success

    def test_trace_consistency(self):
        """Check that intermediate results match the expected values from steps."""
        success = True
        for key, expected_val in self.expected.items():
            # Handle special keys: P_N_step2 and P_N_step5 are not directly in results
            if key == 'P_N_step2':
                # This is the power of the furnace when only S1 is closed.
                # It should be consistent with the first part of the answer.
                # We'll check if the solution stored it as P_N? But note: the solution might store P_N as the second power.
                # We'll skip this check because it's ambiguous.
                continue
            if key == 'P_N_step5':
                # This is the power of the furnace when both are closed.
                # The results might store it as P_N? But the problem asks for two different P_N.
                # We'll check if the stored P_N matches either? We'll skip.
                continue

            stored_val = self.results.get(key, None)
            if stored_val is not None:
                if not self._is_close(stored_val, expected_val):
                    success = self._log(
                        f"Trace consistency: {key} = {stored_val}, expected {expected_val}",
                        False
                    )
                else:
                    self._log(f"Trace consistency: {key} matches.", True)
        return success

    def test_monotonicity(self):
        """Perturb internal resistance r and check that powers decrease."""
        success = True

        # Store original r
        r_original = self.r

        # Perturb r by 10% increase
        r_perturbed = r_original * 1.10

        # Recalculate all quantities using the same steps as the solution
        # Step 1: R = E/I1 - r
        R_new = self.E / self.I1 - r_perturbed
        # Step 2: P_N (first) = I1^2 * R
        P_N1_new = self.I1**2 * R_new
        # Step 3: U = E - I2 * r
        U_new = self.E - self.I2 * r_perturbed
        # Step 4: I_N = U / R
        I_N_new = U_new / R_new
        # Step 5: P_N (second) = U * I_N
        P_N2_new = U_new * I_N_new
        # Step 6: I = I2 - I_N
        I_new = self.I2 - I_N_new
        # Step 7: P = U * I
        P_new = U_new * I_new
        # Step 8: P_Q = I^2 * r_A
        P_Q_new = I_new**2 * self.r_A
        # Step 9: P_chemical = P - P_Q
        P_chemical_new = P_new - P_Q_new

        # Original values (from expected)
        P_N1_orig = self.expected['P_N_step2']
        P_N2_orig = self.expected['P_N_step5']
        P_chemical_orig = self.expected['P_chemical']

        # Check monotonicity: increasing r should decrease all powers
        if not (P_N1_new < P_N1_orig):
            success = self._log(
                f"Monotonicity (P_N1): {P_N1_new} should be less than {P_N1_orig} after increasing r",
                False
            )
        else:
            self._log("Monotonicity (P_N1) satisfied.", True)

        if not (P_N2_new < P_N2_orig):
            success = self._log(
                f"Monotonicity (P_N2): {P_N2_new} should be less than {P_N2_orig} after increasing r",
                False
            )
        else:
            self._log("Monotonicity (P_N2) satisfied.", True)

        if not (P_chemical_new < P_chemical_orig):
            success = self._log(
                f"Monotonicity (P_chemical): {P_chemical_new} should be less than {P_chemical_orig} after increasing r",
                False
            )
        else:
            self._log("Monotonicity (P_chemical) satisfied.", True)

        # Restore original r (just in case)
        self.r = r_original

        return success

    def run_all_tests(self):
        """Run all tests and return success flag and logs."""
        self.logs = []

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        overall_success = True
        for test_name, test_func in tests:
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"Test {test_name} raised exception: {e}", False)
                overall_success = False

        return overall_success, self.logs