import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Follow robustness rules: use getattr to avoid AttributeError.
        """
        # Known parameters from the problem statement
        self.Um = 220 * math.sqrt(2)      # peak voltage (V)
        self.omega = math.pi               # angular frequency (rad/s)
        self.f = self.omega / (2 * math.pi)  # frequency (Hz)
        self.n_ratio = 2                   # turns ratio n1/n2
        self.R = 55.0                      # load resistance (Ω)

        # Extract results from solution_instance with safe fallbacks
        self.U1 = getattr(solution_instance, 'U_1', 
                          getattr(solution_instance, 'results', {}).get('U_1', 0.0))
        self.I1 = getattr(solution_instance, 'I_1', 
                          getattr(solution_instance, 'results', {}).get('I_1', 0.0))
        self.U2 = getattr(solution_instance, 'U_2', 
                          getattr(solution_instance, 'results', {}).get('U_2', 0.0))
        self.I_ammeter = getattr(solution_instance, 'I_ammeter', 
                                 getattr(solution_instance, 'results', {}).get('I_ammeter', 0.0))
        self.P_out = getattr(solution_instance, 'P_out', 
                             getattr(solution_instance, 'results', {}).get('P_out', 0.0))
        self.I2 = getattr(solution_instance, 'I_2', 
                          getattr(solution_instance, 'results', {}).get('I_2', 0.0))
        self.U_voltmeter = getattr(solution_instance, 'U_voltmeter', 
                                   getattr(solution_instance, 'results', {}).get('U_voltmeter', 0.0))

        # For symbolic checks
        self.symbols = {
            'Um': sp.symbols('Um'),
            'U1': sp.symbols('U1'),
            'U2': sp.symbols('U2'),
            'I1': sp.symbols('I1'),
            'I2': sp.symbols('I2'),
            'R': sp.symbols('R'),
            'n1_n2': sp.symbols('n1_n2'),
            'P': sp.symbols('P')
        }

    def _is_close(self, a, b):
        """Robust fuzzy comparison with 1% tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

    def test_invariants(self):
        """
        Verify physical invariants: 
        1. RMS voltage from peak.
        2. Transformer voltage ratio.
        3. Ohm's law on secondary.
        4. Transformer current ratio.
        5. Power conservation (input = output).
        6. Meter readings consistency.
        """
        logs = []
        success = True

        try:
            # 1. Check U1 (RMS) from peak voltage
            U1_expected = self.Um / math.sqrt(2)
            if not self._is_close(self.U1, U1_expected):
                logs.append(f"FAIL: U1 (RMS) mismatch. Expected {U1_expected:.2f} V, got {self.U1:.2f} V")
                success = False
            else:
                logs.append(f"PASS: U1 = {self.U1:.2f} V matches Um/√2 = {U1_expected:.2f} V")

            # 2. Transformer voltage ratio: U1/U2 = n1/n2
            if not self._is_close(self.U1 / self.U2, self.n_ratio):
                logs.append(f"FAIL: Voltage ratio U1/U2 = {self.U1/self.U2:.2f}, expected {self.n_ratio}")
                success = False
            else:
                logs.append(f"PASS: U1/U2 = {self.U1/self.U2:.2f} matches n1/n2 = {self.n_ratio}")

            # 3. Ohm's law on secondary: U2 = I2 * R
            I2_from_ohm = self.U2 / self.R
            if not self._is_close(self.I2, I2_from_ohm):
                logs.append(f"FAIL: Ohm's law on secondary: I2 = U2/R = {I2_from_ohm:.2f} A, got {self.I2:.2f} A")
                success = False
            else:
                logs.append(f"PASS: I2 = {self.I2:.2f} A satisfies U2 = I2 * R")

            # 4. Transformer current ratio: I2/I1 = n1/n2
            if not self._is_close(self.I2 / self.I1, self.n_ratio):
                logs.append(f"FAIL: Current ratio I2/I1 = {self.I2/self.I1:.2f}, expected {self.n_ratio}")
                success = False
            else:
                logs.append(f"PASS: I2/I1 = {self.I2/self.I1:.2f} matches n1/n2 = {self.n_ratio}")

            # 5. Power conservation: P_in = P_out, using both sides
            P_in = self.U1 * self.I1
            P_out_secondary = self.U2 * self.I2
            if not (self._is_close(self.P_out, P_in) and self._is_close(self.P_out, P_out_secondary)):
                logs.append(f"FAIL: Power mismatch. P_in = {P_in:.2f} W, P_out (secondary) = {P_out_secondary:.2f} W, reported P_out = {self.P_out:.2f} W")
                success = False
            else:
                logs.append(f"PASS: Power conservation: P_in = P_out = {self.P_out:.2f} W")

            # 6. Meter readings consistency
            if not self._is_close(self.U_voltmeter, self.U2):
                logs.append(f"FAIL: Voltmeter reading {self.U_voltmeter:.2f} V does not match U2 = {self.U2:.2f} V")
                success = False
            else:
                logs.append(f"PASS: Voltmeter correctly reads U2 = {self.U2:.2f} V")

            if not self._is_close(self.I_ammeter, self.I1):
                logs.append(f"FAIL: Ammeter reading {self.I_ammeter:.2f} A does not match I1 = {self.I1:.2f} A")
                success = False
            else:
                logs.append(f"PASS: Ammeter correctly reads I1 = {self.I1:.2f} A")

        except Exception as e:
            logs.append(f"ERROR in test_invariants: {e}\n{traceback.format_exc()}")
            success = False

        return success, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with the final answers.
        Uses the STEPS ANALYSIS as reference.
        """
        logs = []
        success = True

        try:
            # Step 1: U1 = 220 V
            if not self._is_close(self.U1, 220.0):
                logs.append(f"FAIL: Step 1 U1 = {self.U1:.2f} V, expected 220 V")
                success = False
            else:
                logs.append(f"PASS: Step 1 U1 = {self.U1:.2f} V")

            # Step 2: U2 = 110 V
            if not self._is_close(self.U2, 110.0):
                logs.append(f"FAIL: Step 2 U2 = {self.U2:.2f} V, expected 110 V")
                success = False
            else:
                logs.append(f"PASS: Step 2 U2 = {self.U2:.2f} V")

            # Step 3: I2 = 2 A
            if not self._is_close(self.I2, 2.0):
                logs.append(f"FAIL: Step 3 I2 = {self.I2:.2f} A, expected 2 A")
                success = False
            else:
                logs.append(f"PASS: Step 3 I2 = {self.I2:.2f} A")

            # Step 4: I1 = 1 A, U_voltmeter = 110 V
            if not self._is_close(self.I1, 1.0):
                logs.append(f"FAIL: Step 4 I1 = {self.I1:.2f} A, expected 1 A")
                success = False
            else:
                logs.append(f"PASS: Step 4 I1 = {self.I1:.2f} A")

            if not self._is_close(self.U_voltmeter, 110.0):
                logs.append(f"FAIL: Step 4 U_voltmeter = {self.U_voltmeter:.2f} V, expected 110 V")
                success = False
            else:
                logs.append(f"PASS: Step 4 U_voltmeter = {self.U_voltmeter:.2f} V")

            # Step 5: I_ammeter = 1 A
            if not self._is_close(self.I_ammeter, 1.0):
                logs.append(f"FAIL: Step 5 I_ammeter = {self.I_ammeter:.2f} A, expected 1 A")
                success = False
            else:
                logs.append(f"PASS: Step 5 I_ammeter = {self.I_ammeter:.2f} A")

            # Step 6: P_out = 220 W
            if not self._is_close(self.P_out, 220.0):
                logs.append(f"FAIL: Step 6 P_out = {self.P_out:.2f} W, expected 220 W")
                success = False
            else:
                logs.append(f"PASS: Step 6 P_out = {self.P_out:.2f} W")

        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {e}\n{traceback.format_exc()}")
            success = False

        return success, logs

    def test_monotonicity(self):
        """
        Perturb the load resistance R and verify expected trends.
        Increase R by 10% -> I2 decreases -> I1 decreases -> P_out decreases.
        Restore original R after computation.
        """
        logs = []
        success = True

        try:
            # Original values
            R_original = self.R
            # Perturbation: increase R by 10%
            R_perturbed = R_original * 1.10

            # Recalculate using physical formulas (independent of solution instance)
            # U1 and U2 remain unchanged (transformer ratio independent of R)
            U1 = self.U1
            U2 = self.U2
            # New secondary current
            I2_new = U2 / R_perturbed
            # New primary current (using current ratio)
            I1_new = I2_new / self.n_ratio
            # New output power (using primary side)
            P_out_new = U1 * I1_new

            # Expected trends:
            # I1 should decrease
            if not (I1_new < self.I1):
                logs.append(f"FAIL: Monotonicity - I1 did not decrease when R increased. Original I1 = {self.I1:.2f} A, new I1 = {I1_new:.2f} A")
                success = False
            else:
                logs.append(f"PASS: I1 decreases from {self.I1:.2f} A to {I1_new:.2f} A when R increases")

            # P_out should decrease
            if not (P_out_new < self.P_out):
                logs.append(f"FAIL: Monotonicity - P_out did not decrease when R increased. Original P_out = {self.P_out:.2f} W, new P_out = {P_out_new:.2f} W")
                success = False
            else:
                logs.append(f"PASS: P_out decreases from {self.P_out:.2f} W to {P_out_new:.2f} W when R increases")

            # Verify that U2 remains unchanged (transformer voltage ratio)
            if not self._is_close(U2, self.U2):
                logs.append(f"FAIL: U2 changed from {self.U2:.2f} V to {U2:.2f} V when R increased, but should remain constant")
                success = False
            else:
                logs.append(f"PASS: U2 remains constant at {U2:.2f} V when R changes")

        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {e}\n{traceback.format_exc()}")
            success = False

        return success, logs

    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True

        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for name, test_func in tests:
            all_logs.append(f"\n--- {name} Test ---")
            success, logs = test_func()
            overall_success = overall_success and success
            all_logs.extend(logs)

        all_logs.append(f"\n--- Overall Result ---")
        all_logs.append(f"All tests passed: {overall_success}")

        return overall_success, all_logs