import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction using getattr and results dictionary fallback
        results = getattr(solution_instance, 'results', {})
        self.U1 = getattr(solution_instance, 'U_1', results.get('U_1', 0.0))
        self.fracn_1n_2 = getattr(solution_instance, 'fracn_1n_2', results.get('fracn_1n_2', 0.0))
        self.P_in = getattr(solution_instance, 'P_in', results.get('P_in', 0.0))
        self.I_1 = getattr(solution_instance, 'I_1', results.get('I_1', 0.0))
        
        # Constants from problem statement
        self.U2 = 220.0           # Motor rated voltage (V)
        self.P_out = 22000.0      # Total power of two motors (W)
        self.tol_rel = 1e-2       # Relative tolerance for fuzzy matching
        self.tol_abs = 1e-2       # Absolute tolerance

    def test_invariants(self):
        logs = []
        success = True

        # 1. Transformer voltage ratio: U1/U2 = n1/n2
        expected_ratio = self.U1 / self.U2
        if not math.isclose(self.fracn_1n_2, expected_ratio, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
            logs.append(f"FAIL: Transformer voltage ratio mismatch. Computed U1/U2 = {expected_ratio}, got n1/n2 = {self.fracn_1n_2}")
            success = False
        else:
            logs.append(f"PASS: Voltage ratio holds: n1/n2 = {self.fracn_1n_2} ≈ U1/U2 = {expected_ratio}")

        # 2. Power conservation in ideal transformer: P_in = P_out
        if not math.isclose(self.P_in, self.P_out, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
            logs.append(f"FAIL: Power conservation violated. P_in = {self.P_in}, expected P_out = {self.P_out}")
            success = False
        else:
            logs.append(f"PASS: Power conserved: P_in = {self.P_in} ≈ P_out = {self.P_out}")

        # 3. Primary current: I1 = P_in / U1
        expected_I1 = self.P_in / self.U1
        if not math.isclose(self.I_1, expected_I1, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
            logs.append(f"FAIL: Primary current mismatch. Computed P_in/U1 = {expected_I1}, got I1 = {self.I_1}")
            success = False
        else:
            logs.append(f"PASS: Primary current consistent: I1 = {self.I_1} ≈ P_in/U1 = {expected_I1}")

        return success, logs

    def test_trace_consistency(self):
        logs = []
        success = True

        # Expected values from steps analysis and target answers
        expected_values = {
            'U1': 1100.0,
            'n1/n2': 5.0,
            'P_in': 22000.0,
            'I1': 20.0
        }

        checks = [
            ('U1', self.U1, expected_values['U1']),
            ('n1/n2', self.fracn_1n_2, expected_values['n1/n2']),
            ('P_in', self.P_in, expected_values['P_in']),
            ('I1', self.I_1, expected_values['I1'])
        ]

        for name, actual, expected in checks:
            if not math.isclose(actual, expected, rel_tol=self.tol_rel, abs_tol=self.tol_abs):
                logs.append(f"FAIL: {name} = {actual}, expected {expected}")
                success = False
            else:
                logs.append(f"PASS: {name} = {actual} matches expected value {expected}")

        return success, logs

    def test_monotonicity(self):
        logs = []
        success = True

        # Perturb magnetic field B by ±1% and observe changes
        # Assume U1 ∝ B (from generator formula), all else follows
        factors = [1.01, 0.99]  # Increase and decrease
        original = {
            'U1': self.U1,
            'ratio': self.fracn_1n_2,
            'I1': self.I_1
        }

        for factor in factors:
            U1_new = original['U1'] * factor
            ratio_new = U1_new / self.U2
            I1_new = self.P_out / U1_new  # P_out fixed

            if factor > 1:
                # B increased: U1 and ratio should increase, I1 decrease
                if not (U1_new > original['U1']):
                    logs.append(f"FAIL: U1 did not increase when B increased (factor {factor})")
                    success = False
                if not (ratio_new > original['ratio']):
                    logs.append(f"FAIL: Turns ratio did not increase when B increased (factor {factor})")
                    success = False
                if not (I1_new < original['I1']):
                    logs.append(f"FAIL: I1 did not decrease when B increased (factor {factor})")
                    success = False
            else:
                # B decreased: opposite trends
                if not (U1_new < original['U1']):
                    logs.append(f"FAIL: U1 did not decrease when B decreased (factor {factor})")
                    success = False
                if not (ratio_new < original['ratio']):
                    logs.append(f"FAIL: Turns ratio did not decrease when B decreased (factor {factor})")
                    success = False
                if not (I1_new > original['I1']):
                    logs.append(f"FAIL: I1 did not increase when B decreased (factor {factor})")
                    success = False

        if success:
            logs.append("PASS: All monotonicity trends hold for ±1% changes in B.")

        return success, logs

    def run_all_tests(self):
        all_logs = []
        overall_success = True

        tests = [
            ('Invariants', self.test_invariants),
            ('Trace Consistency', self.test_trace_consistency),
            ('Monotonicity', self.test_monotonicity)
        ]

        for name, test_func in tests:
            try:
                all_logs.append(f"=== {name} Test ===")
                success, logs = test_func()
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR in {name}: {traceback.format_exc()}")
                overall_success = False

        return overall_success, all_logs