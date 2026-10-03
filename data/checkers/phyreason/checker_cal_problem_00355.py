import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction
        self.results = getattr(solution_instance, 'results', {})
        self.E = getattr(solution_instance, 'E', 3.0)
        self.r = getattr(solution_instance, 'r', 1.5)
        self.R1 = getattr(solution_instance, 'R1', 0.5)
        self.R2_max = 5.0
        
        # Extract results with safe fallbacks
        self.P = self.results.get('P', 0.0)
        self.U_1 = self.results.get('U_1', 0.0)
        self.delta_Q = self.results.get('delta_Q', 0.0)
        self.U_1_prime = self.results.get('U_1_prime', 0.0)
        self.P_1 = self.results.get('P_1', 0.0)
        self.U_AB = self.results.get('U_AB', 0.0)
        self.C_1 = self.results.get('C_1', 0.0)
        self.delta_U = self.results.get('delta_U', 0.0)
        self.R_2 = self.results.get('R_2', 0.0)
        self.I = self.results.get('I', 0.0)
        
        self.logs = []
        
    def _log(self, message, success):
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {message}")
        return success
        
    def _is_close(self, a, b):
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        """Verify fundamental physics laws"""
        successes = []
        
        # 1. Ohm's Law for entire circuit when R2=0 (max power condition)
        try:
            I_expected = self.E / (self.R1 + self.r)
            successes.append(self._log(
                f"Ohm's Law: I = E/(R1+r) = {I_expected:.3f} A, got {self.I:.3f} A",
                self._is_close(self.I, I_expected)
            ))
        except Exception as e:
            successes.append(self._log(f"Ohm's Law failed: {e}", False))
        
        # 2. Power formula for R1 (P1 = I² * R1)
        try:
            P1_expected = self.I**2 * self.R1
            successes.append(self._log(
                f"Power formula: P1 = I²*R1 = {P1_expected:.3f} W, got {self.P_1:.3f} W",
                self._is_close(self.P_1, P1_expected)
            ))
        except Exception as e:
            successes.append(self._log(f"Power formula failed: {e}", False))
        
        # 3. Maximum power theorem: For max power in R1, R2 should be 0
        try:
            R2_expected = 0.0
            successes.append(self._log(
                f"Max power condition: R2 should be {R2_expected} Ω, got {self.R_2} Ω",
                self._is_close(self.R_2, R2_expected)
            ))
        except Exception as e:
            successes.append(self._log(f"Max power condition failed: {e}", False))
        
        # 4. Open circuit voltage (when switch open)
        try:
            U_AB_expected = self.E
            successes.append(self._log(
                f"Open circuit: U_AB should equal E = {U_AB_expected:.1f} V, got {self.U_AB:.1f} V",
                self._is_close(self.U_AB, U_AB_expected)
            ))
        except Exception as e:
            successes.append(self._log(f"Open circuit check failed: {e}", False))
        
        # 5. Capacitor voltage divider when switch closed - CORRECTED
        # For capacitor analysis, R2 = 1.0 Ω (from step 8 in ground truth)
        try:
            R2_cap = 1.0  # Correct value for capacitor analysis
            U1_prime_expected = (self.R1 / (self.R1 + R2_cap + self.r)) * self.E
            successes.append(self._log(
                f"Voltage divider (R2=1Ω): U1' = R1/(R1+R2+r)*E = {U1_prime_expected:.2f} V, got {self.U_1_prime:.2f} V",
                self._is_close(self.U_1_prime, U1_prime_expected)
            ))
        except Exception as e:
            successes.append(self._log(f"Voltage divider check failed: {e}", False))
        
        # 6. Capacitance formula
        try:
            if abs(self.delta_U) > 1e-12:
                C1_expected = self.delta_Q / self.delta_U
                successes.append(self._log(
                    f"Capacitance: C1 = ΔQ/ΔU = {C1_expected:.3e} F, got {self.C_1:.3e} F",
                    self._is_close(self.C_1, C1_expected)
                ))
        except Exception as e:
            successes.append(self._log(f"Capacitance check failed: {e}", False))
        
        return all(successes)
    
    def test_trace_consistency(self):
        """Cross-check intermediate steps with final answers"""
        successes = []
        
        # Target answers from problem
        target_power = "0, 1.125W"
        target_capacitance = 3e-3  # 3×10⁻³ F
        
        # Check first answer: R2=0 and P=1.125W
        try:
            # Parse target power (format: "0, 1.125W")
            target_R2 = 0.0
            target_P = 1.125
            
            successes.append(self._log(
                f"R2 consistency: target={target_R2} Ω, got={self.R_2:.3f} Ω",
                self._is_close(self.R_2, target_R2)
            ))
            successes.append(self._log(
                f"Power consistency: target={target_P} W, got={self.P:.3f} W",
                self._is_close(self.P, target_P)
            ))
        except Exception as e:
            successes.append(self._log(f"Power answer check failed: {e}", False))
        
        # Check second answer: capacitance
        try:
            successes.append(self._log(
                f"Capacitance consistency: target={target_capacitance:.3e} F, got={self.C_1:.3e} F",
                self._is_close(self.C_1, target_capacitance)
            ))
        except Exception as e:
            successes.append(self._log(f"Capacitance answer check failed: {e}", False))
        
        # Check step-by-step values from STEPS ANALYSIS
        # Step 4: I = 1.5 A
        try:
            successes.append(self._log(
                f"Step 4 current: expected=1.5 A, got={self.I:.2f} A",
                self._is_close(self.I, 1.5)
            ))
        except Exception as e:
            successes.append(self._log(f"Step 4 check failed: {e}", False))
        
        # Step 6: U_AB = 3.0 V
        try:
            successes.append(self._log(
                f"Step 6 voltage: expected=3.0 V, got={self.U_AB:.1f} V",
                self._is_close(self.U_AB, 3.0)
            ))
        except Exception as e:
            successes.append(self._log(f"Step 6 check failed: {e}", False))
        
        # Step 7: U_1 = 3.0 V
        try:
            successes.append(self._log(
                f"Step 7 capacitor voltage (open): expected=3.0 V, got={self.U_1:.1f} V",
                self._is_close(self.U_1, 3.0)
            ))
        except Exception as e:
            successes.append(self._log(f"Step 7 check failed: {e}", False))
        
        # Step 8: U_1' = 0.5 V
        try:
            successes.append(self._log(
                f"Step 8 capacitor voltage (closed): expected=0.5 V, got={self.U_1_prime:.1f} V",
                self._is_close(self.U_1_prime, 0.5)
            ))
        except Exception as e:
            successes.append(self._log(f"Step 8 check failed: {e}", False))
        
        # Step 9: ΔU = 2.5 V
        try:
            delta_U_calc = self.U_1 - self.U_1_prime
            successes.append(self._log(
                f"Step 9 voltage change: expected=2.5 V, got={self.delta_U:.1f} V",
                self._is_close(self.delta_U, 2.5)
            ))
        except Exception as e:
            successes.append(self._log(f"Step 9 check failed: {e}", False))
        
        # Step 10: ΔQ = 7.5e-3 C
        try:
            successes.append(self._log(
                f"Step 10 charge change: expected=7.5e-3 C, got={self.delta_Q:.3e} C",
                self._is_close(self.delta_Q, 7.5e-3)
            ))
        except Exception as e:
            successes.append(self._log(f"Step 10 check failed: {e}", False))
        
        return all(successes)
    
    def test_monotonicity(self):
        """Perturb inputs and check trends match expected physics"""
        successes = []
        
        # Save original values
        orig_R1 = self.R1
        orig_R_2 = self.R_2
        orig_E = self.E
        
        try:
            # Test 1: If we increase R1 (with R2=0 fixed), current should decrease
            R1_perturbed = orig_R1 * 1.1  # +10%
            I_perturbed = self.E / (R1_perturbed + self.r)
            current_decreased = I_perturbed < self.I
            successes.append(self._log(
                f"Monotonicity (R1↑→I↓): R1={orig_R1:.3f}→{R1_perturbed:.3f} Ω, "
                f"I={self.I:.3f}→{I_perturbed:.3f} A, decreased={current_decreased}",
                current_decreased
            ))
        except Exception as e:
            successes.append(self._log(f"R1 monotonicity test failed: {e}", False))
        
        try:
            # Test 2: For fixed R2=0, if E increases, power in R1 should increase (P ∝ E²)
            E_perturbed = orig_E * 1.1  # +10%
            P_perturbed = (E_perturbed**2 * self.R1) / (self.R1 + self.r)**2
            power_increased = P_perturbed > self.P_1
            successes.append(self._log(
                f"Monotonicity (E↑→P↑): E={orig_E:.1f}→{E_perturbed:.1f} V, "
                f"P={self.P_1:.3f}→{P_perturbed:.3f} W, increased={power_increased}",
                power_increased
            ))
        except Exception as e:
            successes.append(self._log(f"E monotonicity test failed: {e}", False))
        
        try:
            # Test 3: For R2 > 0, power should be less than maximum (at R2=0)
            if abs(orig_R_2) < 1e-9:  # Only test if original R2 is 0
                R2_perturbed = 1.0  # Non-zero
                I_perturbed = self.E / (self.R1 + R2_perturbed + self.r)
                P_perturbed = I_perturbed**2 * self.R1
                power_decreased = P_perturbed < self.P_1
                successes.append(self._log(
                    f"Monotonicity (R2>0→P↓): R2=0→{R2_perturbed} Ω, "
                    f"P={self.P_1:.3f}→{P_perturbed:.3f} W, decreased={power_decreased}",
                    power_decreased
                ))
        except Exception as e:
            successes.append(self._log(f"R2 monotonicity test failed: {e}", False))
        
        # Restore original values
        self.R1 = orig_R1
        self.R_2 = orig_R_2
        self.E = orig_E
        
        return all(successes)
    
    def run_all_tests(self):
        """Execute all tests and return summary"""
        self.logs = []
        overall_success = True
        
        try:
            self.logs.append("=== Physics Checker Started ===")
            
            # Run tests
            invariants_ok = self.test_invariants()
            self.logs.append(f"Invariants test: {'PASSED' if invariants_ok else 'FAILED'}")
            
            trace_ok = self.test_trace_consistency()
            self.logs.append(f"Trace consistency test: {'PASSED' if trace_ok else 'FAILED'}")
            
            monotonicity_ok = self.test_monotonicity()
            self.logs.append(f"Monotonicity test: {'PASSED' if monotonicity_ok else 'FAILED'}")
            
            overall_success = invariants_ok and trace_ok and monotonicity_ok
            self.logs.append(f"\nOverall: {'ALL TESTS PASSED' if overall_success else 'SOME TESTS FAILED'}")
            
        except Exception as e:
            self.logs.append(f"CRITICAL ERROR: {e}")
            self.logs.append(traceback.format_exc())
            overall_success = False
        
        return overall_success, self.logs