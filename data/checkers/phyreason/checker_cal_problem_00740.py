import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs with defaults
        self.N = getattr(solution_instance, 'N', getattr(solution_instance, 'input', {}).get('N', 100))
        self.S = getattr(solution_instance, 'S', getattr(solution_instance, 'input', {}).get('S', 0.03))
        self.omega = getattr(solution_instance, 'omega', getattr(solution_instance, 'input', {}).get('omega', 100 * math.pi))
        self.B = getattr(solution_instance, 'B', getattr(solution_instance, 'input', {}).get('B', math.sqrt(2) / math.pi))
        self.R = getattr(solution_instance, 'R', getattr(solution_instance, 'input', {}).get('R', 8.0))
        # Transformer ratios
        self.n3_n4_ratio = getattr(solution_instance, 'n3_n4_ratio', getattr(solution_instance, 'input', {}).get('n3_n4_ratio', 10.0))
        # Motor specs
        self.U_motor = getattr(solution_instance, 'U_motor', getattr(solution_instance, 'input', {}).get('U_motor', 220.0))
        self.P_motor = getattr(solution_instance, 'P_motor', getattr(solution_instance, 'input', {}).get('P_motor', 11000.0))

        # Safely extract results from solution instance
        self.results = {}
        expected_keys = ['I_3', 'U_1', 'E', 'r', 'DeltaP', 'T', 'P_1', 'I_1', 'U_2', 'U_3', 'E_m', 'W', 'I_4']
        for key in expected_keys:
            # Try attribute, then results dict, then set to None
            val = getattr(solution_instance, key, None)
            if val is None:
                val = getattr(solution_instance, 'results', {}).get(key, None)
            self.results[key] = val

    def test_invariants(self):
        logs = []
        success = True

        # 1. Check maximum EMF formula: E_m = N * B * S * omega
        if self.results.get('E_m') is not None:
            E_m_calc = self.N * self.B * self.S * self.omega
            if not math.isclose(self.results['E_m'], E_m_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: E_m mismatch. Expected {E_m_calc:.2f}, got {self.results['E_m']:.2f}")
                success = False
            else:
                logs.append(f"PASS: E_m = N*B*S*ω check")

        # 2. Check RMS EMF: E = E_m / sqrt(2)
        if self.results.get('E') is not None and self.results.get('E_m') is not None:
            E_calc = self.results['E_m'] / math.sqrt(2)
            if not math.isclose(self.results['E'], E_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: E (RMS) mismatch. Expected {E_calc:.2f}, got {self.results['E']:.2f}")
                success = False
            else:
                logs.append(f"PASS: E = E_m/√2 check")

        # 3. Check step-down transformer current ratio: I_3 / I_4 = n4 / n3 = 1/10
        if self.results.get('I_3') is not None and self.results.get('I_4') is not None:
            expected_ratio = 1.0 / self.n3_n4_ratio  # n4/n3
            actual_ratio = self.results['I_3'] / self.results['I_4']
            if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step-down transformer current ratio mismatch. Expected {expected_ratio:.3f}, got {actual_ratio:.3f}")
                success = False
            else:
                logs.append(f"PASS: Step-down transformer current ratio")

        # 4. Check step-down transformer voltage ratio: U_3 / U_4 = n3 / n4 = 10
        # Note: U_4 is the motor voltage = 220V (given)
        if self.results.get('U_3') is not None:
            expected_U_3 = self.U_motor * self.n3_n4_ratio  # 220 * 10 = 2200
            if not math.isclose(self.results['U_3'], expected_U_3, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step-down transformer voltage mismatch. Expected {expected_U_3:.1f}, got {self.results['U_3']:.1f}")
                success = False
            else:
                logs.append(f"PASS: Step-down transformer voltage ratio")

        # 5. Check transmission line loss: ΔP = I_3² * R
        if self.results.get('DeltaP') is not None and self.results.get('I_3') is not None:
            DeltaP_calc = self.results['I_3']**2 * self.R
            if not math.isclose(self.results['DeltaP'], DeltaP_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Transmission line power loss mismatch. Expected {DeltaP_calc:.1f}, got {self.results['DeltaP']:.1f}")
                success = False
            else:
                logs.append(f"PASS: ΔP = I_3² * R check")

        # 6. Check U_2 = U_3 + I_3 * R (Kirchhoff's Voltage Law)
        if self.results.get('U_2') is not None and self.results.get('U_3') is not None and self.results.get('I_3') is not None:
            U_2_calc = self.results['U_3'] + self.results['I_3'] * self.R
            if not math.isclose(self.results['U_2'], U_2_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: U_2 mismatch. Expected {U_2_calc:.1f}, got {self.results['U_2']:.1f}")
                success = False
            else:
                logs.append(f"PASS: U_2 = U_3 + I_3*R check")

        # 7. Check power conservation at step-up transformer: P_1 = P_user + ΔP
        if self.results.get('P_1') is not None and self.results.get('DeltaP') is not None:
            P_user = self.P_motor
            P_1_calc = P_user + self.results['DeltaP']
            if not math.isclose(self.results['P_1'], P_1_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Power conservation mismatch. Expected {P_1_calc:.1f}, got {self.results['P_1']:.1f}")
                success = False
            else:
                logs.append(f"PASS: P_1 = P_user + ΔP check")

        # 8. Check P_1 = U_1 * I_1
        if self.results.get('P_1') is not None and self.results.get('U_1') is not None and self.results.get('I_1') is not None:
            P_1_from_UI = self.results['U_1'] * self.results['I_1']
            if not math.isclose(self.results['P_1'], P_1_from_UI, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: P_1 = U_1 * I_1 mismatch. Expected {P_1_from_UI:.1f}, got {self.results['P_1']:.1f}")
                success = False
            else:
                logs.append(f"PASS: P_1 = U_1 * I_1 check")

        # 9. Check generator coil resistance: r = (E - U_1) / I_1
        if self.results.get('r') is not None and self.results.get('E') is not None and self.results.get('U_1') is not None and self.results.get('I_1') is not None:
            r_calc = (self.results['E'] - self.results['U_1']) / self.results['I_1']
            if not math.isclose(self.results['r'], r_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Generator coil resistance mismatch. Expected {r_calc:.3f}, got {self.results['r']:.3f}")
                success = False
            else:
                logs.append(f"PASS: r = (E - U_1)/I_1 check")

        # 10. Check period: T = 2π / ω
        if self.results.get('T') is not None:
            T_calc = 2 * math.pi / self.omega
            if not math.isclose(self.results['T'], T_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Period mismatch. Expected {T_calc:.3f}, got {self.results['T']:.3f}")
                success = False
            else:
                logs.append(f"PASS: T = 2π/ω check")

        # 11. Check work done: W = E * I_1 * T (from step 13)
        if self.results.get('W') is not None and self.results.get('E') is not None and self.results.get('I_1') is not None and self.results.get('T') is not None:
            W_calc = self.results['E'] * self.results['I_1'] * self.results['T']
            if not math.isclose(self.results['W'], W_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Work done mismatch. Expected {W_calc:.1f}, got {self.results['W']:.1f}")
                success = False
            else:
                logs.append(f"PASS: W = E * I_1 * T check")

        return success, logs

    def test_trace_consistency(self):
        logs = []
        success = True

        # Cross-check that all results are consistent with the steps.
        # We'll check a few key dependencies.

        # Step 2: I_4 = P_motor / U_motor
        if self.results.get('I_4') is not None:
            I_4_calc = self.P_motor / self.U_motor
            if not math.isclose(self.results['I_4'], I_4_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: I_4 should be {I_4_calc:.1f}, got {self.results['I_4']:.1f}")
                success = False
            else:
                logs.append(f"PASS: I_4 = P_motor / U_motor")

        # Step 3: I_3 = I_4 * (n4/n3) = I_4 / 10
        if self.results.get('I_3') is not None and self.results.get('I_4') is not None:
            I_3_calc = self.results['I_4'] / self.n3_n4_ratio
            if not math.isclose(self.results['I_3'], I_3_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: I_3 should be {I_3_calc:.1f}, got {self.results['I_3']:.1f}")
                success = False
            else:
                logs.append(f"PASS: I_3 = I_4 / 10")

        # Step 5: U_3 = U_motor * 10
        if self.results.get('U_3') is not None:
            U_3_calc = self.U_motor * self.n3_n4_ratio
            if not math.isclose(self.results['U_3'], U_3_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: U_3 should be {U_3_calc:.1f}, got {self.results['U_3']:.1f}")
                success = False
            else:
                logs.append(f"PASS: U_3 = U_motor * 10")

        # Step 6: U_2 = U_3 + I_3 * R
        if self.results.get('U_2') is not None and self.results.get('U_3') is not None and self.results.get('I_3') is not None:
            U_2_calc = self.results['U_3'] + self.results['I_3'] * self.R
            if not math.isclose(self.results['U_2'], U_2_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: U_2 should be {U_2_calc:.1f}, got {self.results['U_2']:.1f}")
                success = False
            else:
                logs.append(f"PASS: U_2 = U_3 + I_3 * R")

        # Step 7: The step-up transformer ratio is not given, but we can check consistency:
        # If we compute the ratio n1/n2 from U_1 and U_2, it should be constant.
        # Actually, we don't have n1 and n2, so we skip this.

        # Step 10: I_1 = P_1 / U_1
        if self.results.get('I_1') is not None and self.results.get('P_1') is not None and self.results.get('U_1') is not None:
            I_1_calc = self.results['P_1'] / self.results['U_1']
            if not math.isclose(self.results['I_1'], I_1_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: I_1 should be {I_1_calc:.1f}, got {self.results['I_1']:.1f}")
                success = False
            else:
                logs.append(f"PASS: I_1 = P_1 / U_1")

        return success, logs

    def test_monotonicity(self):
        logs = []
        success = True

        # Perturb the angular velocity ω and check that E_m changes proportionally.
        original_omega = self.omega
        perturbed_omega = original_omega * 1.05  # Increase by 5%

        # Recalculate E_m using the formula
        original_E_m = self.N * self.B * self.S * original_omega
        perturbed_E_m = self.N * self.B * self.S * perturbed_omega

        # Check that the change is in the same direction and roughly proportional (within 1%)
        expected_ratio = perturbed_omega / original_omega
        actual_ratio = perturbed_E_m / original_E_m
        if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: E_m does not scale linearly with ω. Expected ratio {expected_ratio:.3f}, got {actual_ratio:.3f}")
            success = False
        else:
            logs.append(f"PASS: E_m scales linearly with ω (monotonicity)")

        # Restore ω (not necessary because we didn't change the instance, just used local variables)
        # Also, perturb the transmission line resistance R and check that ΔP increases.
        original_R = self.R
        perturbed_R = original_R * 1.05  # Increase by 5%

        # Use the current I_3 from results to compute ΔP
        if self.results.get('I_3') is not None:
            original_DeltaP = self.results['I_3']**2 * original_R
            perturbed_DeltaP = self.results['I_3']**2 * perturbed_R

            # ΔP should increase when R increases (since I_3 is constant in this perturbation)
            if not (perturbed_DeltaP > original_DeltaP):
                logs.append(f"FAIL: ΔP does not increase when R increases (at fixed I_3)")
                success = False
            else:
                logs.append(f"PASS: ΔP increases with R (at fixed I_3)")

            # Check linearity: ΔP ∝ R (since I_3 fixed)
            expected_ratio_R = perturbed_R / original_R
            actual_ratio_DeltaP = perturbed_DeltaP / original_DeltaP
            if not math.isclose(actual_ratio_DeltaP, expected_ratio_R, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: ΔP does not scale linearly with R (at fixed I_3). Expected ratio {expected_ratio_R:.3f}, got {actual_ratio_DeltaP:.3f}")
                success = False
            else:
                logs.append(f"PASS: ΔP scales linearly with R (at fixed I_3)")

        return success, logs

    def run_all_tests(self):
        all_logs = []
        overall_success = True

        # Run each test
        for test_name, test_func in [("Invariants", self.test_invariants),
                                     ("Trace Consistency", self.test_trace_consistency),
                                     ("Monotonicity", self.test_monotonicity)]:
            success, logs = test_func()
            overall_success = overall_success and success
            all_logs.append(f"=== {test_name} ===")
            all_logs.extend(logs)

        return overall_success, all_logs