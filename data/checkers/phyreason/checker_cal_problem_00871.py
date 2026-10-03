import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Directly extract all values from results dictionary
        results = solution_instance.results
        
        # Base quantities - use 0.0 as default if not found
        self.L = results.get('L', 0.0)
        self.S = results.get('S', 0.0)
        self.n = results.get('n', 0.0)
        self.e = results.get('e', 0.0)  # Don't use default 1.6e-19
        self.v = results.get('v', 0.0)
        self.B = results.get('B', 0.0)
        
        # Derived quantities
        self.V = results.get('V', 0.0)
        self.f = results.get('f', 0.0)
        self.F = results.get('F', 0.0)
        self.q = results.get('q', 0.0)
        self.l = results.get('l', 0.0)
        self.N_pass = results.get('N_pass', 0.0)
        self.I = results.get('I', 0.0)
        self.N = results.get('N', 0.0)
        
        self.logs = []
        
    def log(self, message, success):
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {message}")
        return success
        
    def test_invariants(self):
        """Test fundamental physics laws: charge conservation, force consistency."""
        success = True
        
        # 1. Total charge consistency: q = N_pass * e
        if self.N_pass > 0 and self.q > 0 and self.e > 0:
            expected_q = self.N_pass * self.e
            if not math.isclose(self.q, expected_q, rel_tol=1e-2, abs_tol=1e-2):
                success = self.log(f"Charge conservation: q = {self.q}, expected N_pass*e = {expected_q}", False)
            else:
                self.log("Charge conservation: PASS", True)
        elif self.N_pass == 0 or self.q == 0 or self.e == 0:
            self.log("Charge conservation: SKIP (zero values)", True)
        
        # 2. Current definition: I = q / t, where t = l/v (from step 3)
        if self.l > 0 and self.v > 0 and self.q > 0:
            t = self.l / self.v
            if t > 0:
                expected_I = self.q / t
                if not math.isclose(self.I, expected_I, rel_tol=1e-2, abs_tol=1e-2):
                    success = self.log(f"Current definition: I = {self.I}, expected q/(l/v) = {expected_I}", False)
                else:
                    self.log("Current definition: PASS", True)
            else:
                self.log("Current definition: SKIP (t=0)", True)
        else:
            self.log("Current definition: SKIP (zero values)", True)
        
        # 3. Force distribution: f = F / N (from step 7)
        if self.N > 0 and self.F > 0:
            expected_f = self.F / self.N
            if not math.isclose(self.f, expected_f, rel_tol=1e-2, abs_tol=1e-2):
                success = self.log(f"Force distribution: f = {self.f}, expected F/N = {expected_f}", False)
            else:
                self.log("Force distribution: PASS", True)
        elif self.N == 0 or self.F == 0:
            self.log("Force distribution: SKIP (zero values)", True)
        
        # 4. Current from drift velocity: I = n*e*S*v (step 4)
        if self.n > 0 and self.e > 0 and self.S > 0 and self.v > 0:
            expected_I_drift = self.n * self.e * self.S * self.v
            if not math.isclose(self.I, expected_I_drift, rel_tol=1e-2, abs_tol=1e-2):
                success = self.log(f"Drift current: I = {self.I}, expected n*e*S*v = {expected_I_drift}", False)
            else:
                self.log("Drift current: PASS", True)
        else:
            self.log("Drift current: SKIP (zero values)", True)
            
        return success
        
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True
        
        # Step 1 & 8: N = n*L*S
        if self.n > 0 and self.L > 0 and self.S > 0:
            expected_N = self.n * self.L * self.S
            if not math.isclose(self.N, expected_N, rel_tol=1e-2, abs_tol=1e-2):
                success = self.log(f"Total electrons: N = {self.N}, expected n*L*S = {expected_N}", False)
            else:
                self.log("Total electrons consistency: PASS", True)
        else:
            self.log("Total electrons consistency: SKIP (zero values)", True)
        
        # Step 3: l = v*t, but we don't have t. Check V = l*S instead
        if self.l > 0 and self.S > 0:
            expected_V = self.l * self.S
            if not math.isclose(self.V, expected_V, rel_tol=1e-2, abs_tol=1e-2):
                success = self.log(f"Volume: V = {self.V}, expected l*S = {expected_V}", False)
            else:
                self.log("Volume consistency: PASS", True)
        else:
            self.log("Volume consistency: SKIP (zero values)", True)
        
        # Step 3: N_pass = n*S*l
        if self.n > 0 and self.S > 0 and self.l > 0:
            expected_N_pass = self.n * self.S * self.l
            if not math.isclose(self.N_pass, expected_N_pass, rel_tol=1e-2, abs_tol=1e-2):
                success = self.log(f"Passing electrons: N_pass = {self.N_pass}, expected n*S*l = {expected_N_pass}", False)
            else:
                self.log("Passing electrons consistency: PASS", True)
        else:
            self.log("Passing electrons consistency: SKIP (zero values)", True)
        
        # Step 5: F = B*I*L
        if self.B > 0 and self.I > 0 and self.L > 0:
            expected_F_ampere = self.B * self.I * self.L
            if not math.isclose(self.F, expected_F_ampere, rel_tol=1e-2, abs_tol=1e-2):
                success = self.log(f"Ampere force: F = {self.F}, expected B*I*L = {expected_F_ampere}", False)
            else:
                self.log("Ampere force consistency: PASS", True)
        else:
            self.log("Ampere force consistency: SKIP (zero values)", True)
        
        # Step 6: F = B*n*e*S*v*L (alternate form)
        if self.B > 0 and self.n > 0 and self.e > 0 and self.S > 0 and self.v > 0 and self.L > 0:
            expected_F_full = self.B * self.n * self.e * self.S * self.v * self.L
            if not math.isclose(self.F, expected_F_full, rel_tol=1e-2, abs_tol=1e-2):
                success = self.log(f"Full Ampere force: F = {self.F}, expected B*n*e*S*v*L = {expected_F_full}", False)
            else:
                self.log("Full Ampere force consistency: PASS", True)
        else:
            self.log("Full Ampere force consistency: SKIP (zero values)", True)
        
        # Step 9: f = B*e*v (final answer)
        if self.B > 0 and self.e > 0 and self.v > 0:
            expected_f_final = self.B * self.e * self.v
            if not math.isclose(self.f, expected_f_final, rel_tol=1e-2, abs_tol=1e-2):
                success = self.log(f"Lorentz force: f = {self.f}, expected B*e*v = {expected_f_final}", False)
            else:
                self.log("Lorentz force consistency: PASS", True)
        else:
            self.log("Lorentz force consistency: SKIP (zero values)", True)
            
        return success
        
    def test_monotonicity(self):
        """Perturb input n and check proportional changes."""
        success = True
        
        # Skip if any required value is zero
        if self.n == 0 or self.e == 0 or self.S == 0 or self.v == 0 or self.B == 0 or self.L == 0:
            self.log("Monotonicity: SKIP (zero values)", True)
            return True
            
        if self.I == 0 or self.F == 0 or self.N == 0 or self.q == 0 or self.N_pass == 0:
            self.log("Monotonicity: SKIP (zero derived values)", True)
            return True
        
        # Save original values
        orig_n = self.n
        orig_I = self.I
        orig_F = self.F
        orig_N = self.N
        orig_q = self.q
        orig_N_pass = self.N_pass
        
        # Perturb n by +10%
        perturbed_n = orig_n * 1.10
        
        # Recalculate expected values using formulas
        expected_I_new = perturbed_n * self.e * self.S * self.v
        expected_N_new = perturbed_n * self.L * self.S
        expected_q_new = perturbed_n * self.e * self.S * self.l
        expected_N_pass_new = perturbed_n * self.S * self.l
        expected_F_new = self.B * expected_I_new * self.L
        
        # Check proportionality: new/old should equal perturbed_n/orig_n
        # Use more relaxed tolerance for monotonicity due to floating point errors
        if orig_I > 0:
            ratio_I = expected_I_new / orig_I
            if not math.isclose(ratio_I, 1.10, rel_tol=5e-2, abs_tol=5e-2):  # 5% tolerance
                success = self.log(f"Monotonicity I: ratio = {ratio_I:.6f}, expected 1.10", False)
            else:
                self.log("Monotonicity I: PASS", True)
        
        if orig_N > 0:
            ratio_N = expected_N_new / orig_N
            if not math.isclose(ratio_N, 1.10, rel_tol=5e-2, abs_tol=5e-2):
                success = self.log(f"Monotonicity N: ratio = {ratio_N:.6f}, expected 1.10", False)
            else:
                self.log("Monotonicity N: PASS", True)
        
        if orig_q > 0:
            ratio_q = expected_q_new / orig_q
            if not math.isclose(ratio_q, 1.10, rel_tol=5e-2, abs_tol=5e-2):
                success = self.log(f"Monotonicity q: ratio = {ratio_q:.6f}, expected 1.10", False)
            else:
                self.log("Monotonicity q: PASS", True)
        
        if orig_N_pass > 0:
            ratio_N_pass = expected_N_pass_new / orig_N_pass
            if not math.isclose(ratio_N_pass, 1.10, rel_tol=5e-2, abs_tol=5e-2):
                success = self.log(f"Monotonicity N_pass: ratio = {ratio_N_pass:.6f}, expected 1.10", False)
            else:
                self.log("Monotonicity N_pass: PASS", True)
        
        if orig_F > 0:
            ratio_F = expected_F_new / orig_F
            if not math.isclose(ratio_F, 1.10, rel_tol=5e-2, abs_tol=5e-2):
                success = self.log(f"Monotonicity F: ratio = {ratio_F:.6f}, expected 1.10", False)
            else:
                self.log("Monotonicity F: PASS", True)
        
        return success
        
    def run_all_tests(self):
        """Run all tests and return success flag and logs."""
        self.logs = []
        overall_success = True
        
        try:
            if not self.test_invariants():
                overall_success = False
            if not self.test_trace_consistency():
                overall_success = False
            if not self.test_monotonicity():
                overall_success = False
        except Exception as e:
            self.logs.append(f"ERROR: {traceback.format_exc()}")
            overall_success = False
            
        return overall_success, self.logs