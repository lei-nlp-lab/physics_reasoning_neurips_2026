import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        # Input parameters (with default fallbacks)
        self.r_gen = 1.0  # internal resistance (Ω)
        self.R = 4.0      # transmission line resistance (Ω)
        self.n_classes = 22
        self.lamps_per_class = 6
        self.V_lamp = 220.0  # V
        self.P_lamp = 40.0   # W
        self.turns_up = (1, 4)    # step-up transformer ratio
        self.turns_down = (4, 1)  # step-down transformer ratio
        
        # Safely extract results using getattr/dict.get pattern
        results = getattr(solution_instance, 'results', {})
        
        # Key mappings (with safe defaults) - handle multiple naming conventions
        self.I2 = results.get('I_2', 0.0)
        self.I1 = results.get('I_1', 0.0)
        self.nP_light = results.get("n'P_light", results.get('P_light_half', 0.0))
        self.U1 = results.get('U_1', 0.0)
        self.eta = results.get('η', 0.0)
        self.P_out_prime = results.get("P_out'", results.get('P_out_prime', 0.0))
        self.P_out = results.get('P_out', 0.0)
        self.U2_prime = results.get("U_2'", results.get('U_2_prime', 0.0))
        self.I2_prime = results.get("I_2'", results.get('I_2_prime', 0.0))
        self.U2 = results.get('U_2', 0.0)
        
        # Derived quantities for testing
        self.total_lamps = self.n_classes * self.lamps_per_class
        self.P_total_lamps = self.total_lamps * self.P_lamp
        self.half_lamps = self.total_lamps // 2
        self.P_half_lamps = self.half_lamps * self.P_lamp

    def _is_close(self, a, b):
        """Robust fuzzy comparison with 2% tolerance."""
        return math.isclose(a, b, rel_tol=0.02, abs_tol=0.02)

    # ------------------------------------------------------------
    # PHYSICS INVARIANT TESTS
    # ------------------------------------------------------------
    def test_power_conservation_transformers(self):
        """Verify power conservation at transformers."""
        logs = []
        success = True
        
        # Step-up transformer: U1*I1 = U2*I2
        if all(v != 0 for v in [self.U1, self.I1, self.U2, self.I2]):
            lhs = self.U1 * self.I1
            rhs = self.U2 * self.I2
            if not self._is_close(lhs, rhs):
                logs.append(f"FAIL: Step-up transformer power conservation: U1*I1={lhs} ≠ U2*I2={rhs}")
                success = False
            else:
                logs.append(f"PASS: Step-up transformer power conservation: {lhs:.1f} ≈ {rhs:.1f} W")
        else:
            logs.append("SKIP: Step-up transformer check (missing values)")
        
        # Step-down transformer: U2'*I2 = total lamp power
        expected_power = self.P_total_lamps
        if self.U2_prime != 0 and self.I2 != 0:
            actual_power = self.U2_prime * self.I2
            if not self._is_close(actual_power, expected_power):
                logs.append(f"FAIL: Step-down transformer power: U2'*I2={actual_power} ≠ P_lamps={expected_power}")
                success = False
            else:
                logs.append(f"PASS: Step-down transformer power: {actual_power:.0f} ≈ {expected_power:.0f} W")
        else:
            logs.append("SKIP: Step-down transformer check (missing values)")
        
        return success, logs

    def test_transformer_ratios(self):
        """Verify transformer voltage and current ratios."""
        logs = []
        success = True
        
        # Step-up transformer: U2/U1 = 4 (1:4 turns ratio)
        if self.U1 != 0 and self.U2 != 0:
            expected_ratio = self.turns_up[1] / self.turns_up[0]  # = 4
            actual_ratio = self.U2 / self.U1
            if not self._is_close(actual_ratio, expected_ratio):
                logs.append(f"FAIL: Step-up voltage ratio: U2/U1={actual_ratio:.2f} ≠ {expected_ratio}")
                success = False
            else:
                logs.append(f"PASS: Step-up voltage ratio: {actual_ratio:.2f} ≈ {expected_ratio}")
        
        # Step-down transformer: U2'/U3 = 4 (4:1 turns ratio)
        if self.U2_prime != 0:
            expected_ratio = self.turns_down[0] / self.turns_down[1]  # = 4
            actual_ratio = self.U2_prime / self.V_lamp
            if not self._is_close(actual_ratio, expected_ratio):
                logs.append(f"FAIL: Step-down voltage ratio: U2'/U3={actual_ratio:.2f} ≠ {expected_ratio}")
                success = False
            else:
                logs.append(f"PASS: Step-down voltage ratio: {actual_ratio:.2f} ≈ {expected_ratio}")
        
        # Current ratios: For step-up transformer, I1/I2 = N2/N1 = 4/1 = 4
        if self.I2 != 0 and self.I1 != 0:
            expected_current_ratio_up = self.turns_up[1] / self.turns_up[0]  # = 4
            actual_current_ratio_up = self.I1 / self.I2
            if not self._is_close(actual_current_ratio_up, expected_current_ratio_up):
                logs.append(f"FAIL: Step-up current ratio: I1/I2={actual_current_ratio_up:.3f} ≠ {expected_current_ratio_up:.3f}")
                success = False
            else:
                logs.append(f"PASS: Step-up current ratio: {actual_current_ratio_up:.3f} ≈ {expected_current_ratio_up:.3f}")
        
        return success, logs

    def test_transmission_line(self):
        """Verify Ohm's law and power loss on transmission line."""
        logs = []
        success = True
        
        # Voltage drop: U2 = U2' + I2*R
        if self.U2_prime != 0 and self.I2 != 0:
            expected_U2 = self.U2_prime + self.I2 * self.R
            if self.U2 != 0:
                if not self._is_close(self.U2, expected_U2):
                    logs.append(f"FAIL: Transmission voltage: U2={self.U2} ≠ U2'+I2*R={expected_U2}")
                    success = False
                else:
                    logs.append(f"PASS: Transmission voltage: U2={self.U2} ≈ U2'+I2*R={expected_U2}")
        
        # Power loss: P_loss = I2²*R
        if self.I2 != 0:
            P_loss = self.I2**2 * self.R
            expected_P_out = self.P_total_lamps + P_loss
            if self.P_out != 0:
                if not self._is_close(self.P_out, expected_P_out):
                    logs.append(f"FAIL: Generator output power: P_out={self.P_out} ≠ P_lamps+P_loss={expected_P_out}")
                    success = False
                else:
                    logs.append(f"PASS: Generator output power: {self.P_out} ≈ {expected_P_out} W")
        
        return success, logs

    def test_efficiency(self):
        """Verify transmission efficiency calculation."""
        logs = []
        success = True
        
        if self.P_out != 0 and self.P_total_lamps != 0:
            expected_eta = self.P_total_lamps / self.P_out
            if not self._is_close(self.eta, expected_eta):
                logs.append(f"FAIL: Efficiency: η={self.eta} ≠ P_lamps/P_out={expected_eta}")
                success = False
            else:
                logs.append(f"PASS: Efficiency: η={self.eta:.3f} ≈ {expected_eta:.3f}")
        
        return success, logs

    def test_half_load_consistency(self):
        """Verify calculations for half the lamps."""
        logs = []
        success = True
        
        # Power for half lamps
        if self.nP_light != 0:
            if not self._is_close(self.nP_light, self.P_half_lamps):
                logs.append(f"FAIL: Half-load power: n'P_light={self.nP_light} ≠ expected={self.P_half_lamps}")
                success = False
            else:
                logs.append(f"PASS: Half-load power: {self.nP_light} ≈ {self.P_half_lamps} W")
        
        # Current for half lamps: I2' = P_half / U2'
        if self.U2_prime != 0 and self.I2_prime != 0:
            expected_I2_prime = self.P_half_lamps / self.U2_prime
            if not self._is_close(self.I2_prime, expected_I2_prime):
                logs.append(f"FAIL: Half-load current: I2'={self.I2_prime} ≠ P_half/U2'={expected_I2_prime}")
                success = False
            else:
                logs.append(f"PASS: Half-load current: {self.I2_prime} ≈ {expected_I2_prime} A")
        
        # Generator output for half lamps
        if self.P_out_prime != 0 and self.I2_prime != 0:
            expected_P_out_prime = self.P_half_lamps + self.I2_prime**2 * self.R
            if not self._is_close(self.P_out_prime, expected_P_out_prime):
                logs.append(f"FAIL: Half-load generator power: P_out'={self.P_out_prime} ≠ P_half+I2'²R={expected_P_out_prime}")
                success = False
            else:
                logs.append(f"PASS: Half-load generator power: {self.P_out_prime} ≈ {expected_P_out_prime} W")
        
        # Verify "reduced by more than half" statement
        if self.P_out != 0 and self.P_out_prime != 0:
            half_P_out = self.P_out / 2
            if self.P_out_prime >= half_P_out:
                logs.append(f"WARNING: Half-load power {self.P_out_prime} should be < half of full load {half_P_out}")
            else:
                logs.append(f"PASS: Half-load power ({self.P_out_prime}) < half of full load ({half_P_out})")
        
        return success, logs

    # ------------------------------------------------------------
    # TEST BUNDLES
    # ------------------------------------------------------------
    def test_invariants(self):
        """Run all physics invariant tests."""
        logs = []
        success = True
        
        tests = [
            self.test_power_conservation_transformers,
            self.test_transformer_ratios,
            self.test_transmission_line,
            self.test_efficiency,
            self.test_half_load_consistency
        ]
        
        for test in tests:
            try:
                test_success, test_logs = test()
                logs.extend(test_logs)
                success = success and test_success
            except Exception as e:
                logs.append(f"ERROR in {test.__name__}: {str(e)}")
                success = False
        
        return success, logs

    def test_trace_consistency(self):
        """Check intermediate results against final answers."""
        logs = []
        success = True
        
        # Check target answer 1: Generator output power = 5424W
        if not self._is_close(self.P_out, 5424.0):
            logs.append(f"FAIL: P_out={self.P_out} ≠ 5424W (target answer 1)")
            success = False
        else:
            logs.append(f"PASS: P_out={self.P_out} ≈ 5424W")
        
        # Check target answer 2: Generator EMF = 250V (U1 + I1*r_gen)
        emf = self.U1 + self.I1 * self.r_gen
        if not self._is_close(emf, 250.0):
            logs.append(f"FAIL: Generator EMF={emf} ≠ 250V (target answer 2)")
            success = False
        else:
            logs.append(f"PASS: Generator EMF={emf} ≈ 250V")
        
        # Check target answer 3: Efficiency = 97%
        if not self._is_close(self.eta, 0.97):
            logs.append(f"FAIL: η={self.eta} ≠ 97% (target answer 3)")
            success = False
        else:
            logs.append(f"PASS: η={self.eta*100:.1f}% ≈ 97%")
        
        # Check target answer 4: Statement about power reduction
        if self.P_out != 0 and self.P_out_prime != 0:
            if self.P_out_prime >= self.P_out / 2:
                logs.append("FAIL: Generator output power is NOT reduced by more than half")
                success = False
            else:
                logs.append("PASS: Generator output power reduced by more than half")
        
        return success, logs

    def test_monotonicity(self):
        """Perturb input and check expected trends."""
        logs = []
        success = True
        
        original_lamps = self.total_lamps
        
        try:
            # Test 1: Increase lamps by 10%
            perturbed_lamps = int(original_lamps * 1.1)
            perturbed_power = perturbed_lamps * self.P_lamp
            
            # Current in transmission line (using same U2')
            if self.U2_prime == 0:
                logs.append("SKIP: Monotonicity test (U2_prime is zero)")
                return success, logs
                
            I2_perturbed = perturbed_power / self.U2_prime
            P_loss_perturbed = I2_perturbed**2 * self.R
            P_out_perturbed = perturbed_power + P_loss_perturbed
            
            # Efficiency should decrease (more loss)
            eta_perturbed = perturbed_power / P_out_perturbed
            
            if eta_perturbed >= self.eta:
                logs.append(f"FAIL: Efficiency should decrease with more lamps, but η={eta_perturbed:.3f} >= {self.eta:.3f}")
                success = False
            else:
                logs.append(f"PASS: Efficiency decreases with more lamps: {eta_perturbed:.3f} < {self.eta:.3f}")
            
            # Test 2: Decrease transmission line resistance
            R_perturbed = self.R * 0.5  # halve resistance
            
            # Recalculate with original lamps
            I2_original = self.P_total_lamps / self.U2_prime
            P_loss_new = I2_original**2 * R_perturbed
            P_out_new = self.P_total_lamps + P_loss_new
            eta_new = self.P_total_lamps / P_out_new
            
            if eta_new <= self.eta:
                logs.append(f"FAIL: Efficiency should increase with lower R, but η={eta_new:.3f} <= {self.eta:.3f}")
                success = False
            else:
                logs.append(f"PASS: Efficiency increases with lower R: {eta_new:.3f} > {self.eta:.3f}")
                
        except ZeroDivisionError:
            logs.append("SKIP: Monotonicity test (division by zero)")
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {str(e)}")
            success = False
        
        return success, logs

    # ------------------------------------------------------------
    # MAIN ENTRY POINT
    # ------------------------------------------------------------
    def run_all_tests(self):
        """Run all tests and return overall success and logs."""
        logs = ["Physics Audit Report", "="*50]
        overall_success = True
        
        # Test bundles
        test_suites = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for suite_name, test_func in test_suites:
            logs.append(f"\n{suite_name}")
            logs.append("-"*len(suite_name))
            try:
                success, suite_logs = test_func()
                logs.extend(suite_logs)
                if not success:
                    overall_success = False
                    logs.append(f"❌ {suite_name} FAILED")
                else:
                    logs.append(f"✅ {suite_name} PASSED")
            except Exception as e:
                logs.append(f"ERROR in {suite_name}: {str(e)}")
                logs.append(traceback.format_exc())
                overall_success = False
        
        logs.append("\n" + "="*50)
        logs.append(f"OVERALL: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, logs