import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs with defaults from problem statement
        self.n = getattr(solution_instance, 'n', 100)
        self.L = getattr(solution_instance, 'L', 0.2)  # 20cm -> 0.2m
        self.B = getattr(solution_instance, 'B', 0.5)
        self.omega = getattr(solution_instance, 'omega', 100 * math.pi)
        self.r = getattr(solution_instance, 'r', 1)
        self.R = getattr(solution_instance, 'R', 99)
        
        # Safely extract results from solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract with fallbacks for all expected keys
        self.E_m = getattr(solution_instance, 'E_m', self.results.get('E_m', 0.0))
        self.I_m = getattr(solution_instance, 'I_m', self.results.get('I_m', 0.0))
        self.i_expr = getattr(solution_instance, 'i', self.results.get('i', ''))
        self.I_eff = getattr(solution_instance, 'I', self.results.get('I', 0.0))
        self.P_R = getattr(solution_instance, 'P_R', self.results.get('P_R', 0.0))
        self.P_out = getattr(solution_instance, 'P_out', self.results.get('P_out', 0.0))
        
        # Derived quantities
        self.S = self.L ** 2  # coil area in m²
        
        # Initialize logs
        self.logs = []
        
    def _log(self, message, success=None):
        """Helper to format log messages with status."""
        if success is None:
            self.logs.append(f"INFO: {message}")
        else:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {message}")
    
    def test_invariants(self):
        """Test fundamental physics relationships."""
        try:
            # 1. Check Faraday's law: E_m = n * B * S * ω
            expected_E_m = self.n * self.B * self.S * self.omega
            if not math.isclose(self.E_m, expected_E_m, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Faraday's law violated: E_m = {self.E_m}, expected = {expected_E_m}", False)
                return False
            self._log(f"Faraday's law satisfied: E_m = {self.E_m:.2f} ≈ n·B·S·ω = {expected_E_m:.2f}", True)
            
            # 2. Check Ohm's law for maximum current: I_m = E_m / (R + r)
            expected_I_m = self.E_m / (self.R + self.r)
            if not math.isclose(self.I_m, expected_I_m, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Ohm's law violated: I_m = {self.I_m}, expected = {expected_I_m}", False)
                return False
            self._log(f"Ohm's law satisfied: I_m = {self.I_m:.2f} ≈ E_m/(R+r) = {expected_I_m:.2f}", True)
            
            # 3. Check effective current: I_eff = I_m / √2
            expected_I_eff = self.I_m / math.sqrt(2)
            if not math.isclose(self.I_eff, expected_I_eff, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Effective current relationship violated: I = {self.I_eff}, expected I_m/√2 = {expected_I_eff}", False)
                return False
            self._log(f"Effective current correct: I = {self.I_eff:.2f} ≈ I_m/√2 = {expected_I_eff:.2f}", True)
            
            # 4. Check Joule's law for external resistor: P_R = I_eff² * R
            expected_P_R = self.I_eff ** 2 * self.R
            if not math.isclose(self.P_R, expected_P_R, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Joule's law (external) violated: P_R = {self.P_R}, expected I²·R = {expected_P_R}", False)
                return False
            self._log(f"Joule's law (external) satisfied: P_R = {self.P_R:.2f} ≈ I²·R = {expected_P_R:.2f}", True)
            
            # 5. Check total power: P_out = I_eff² * (R + r) = P_R + I_eff² * r
            expected_P_out = self.I_eff ** 2 * (self.R + self.r)
            if not math.isclose(self.P_out, expected_P_out, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Total power violated: P_out = {self.P_out}, expected I²·(R+r) = {expected_P_out}", False)
                return False
            self._log(f"Total power correct: P_out = {self.P_out:.2f} ≈ I²·(R+r) = {expected_P_out:.2f}", True)
            
            # 6. Check power consistency: P_out should equal P_R + internal losses
            internal_loss = self.I_eff ** 2 * self.r
            if not math.isclose(self.P_out, self.P_R + internal_loss, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Power balance violated: P_out = {self.P_out}, P_R + I²·r = {self.P_R + internal_loss}", False)
                return False
            self._log(f"Power balance satisfied: P_out = P_R + I²·r = {self.P_out:.2f}", True)
            
            return True
        except Exception as e:
            self._log(f"Invariant test failed with error: {str(e)}", False)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # 1. Check that instantaneous current expression has correct amplitude
            if isinstance(self.i_expr, str):
                # Extract amplitude from string (crude but functional for this format)
                import re
                match = re.search(r'([\d.]+)\s*sin', self.i_expr)
                if match:
                    amplitude = float(match.group(1))
                    if not math.isclose(amplitude, self.I_m, rel_tol=1e-2, abs_tol=1e-2):
                        self._log(f"Instantaneous current amplitude mismatch: i has {amplitude}, I_m = {self.I_m}", False)
                        return False
                    self._log(f"Instantaneous current amplitude consistent: {amplitude:.2f} ≈ I_m = {self.I_m:.2f}", True)
            
            # 2. Check P_R against target answer 198π²
            expected_P_R_symbolic = 198 * (math.pi ** 2)
            if not math.isclose(self.P_R, expected_P_R_symbolic, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"P_R doesn't match target 198π²: {self.P_R:.2f} vs {expected_P_R_symbolic:.2f}", False)
                return False
            self._log(f"P_R matches target 198π²: {self.P_R:.2f}", True)
            
            # 3. Check P_out against target answer 200π²
            expected_P_out_symbolic = 200 * (math.pi ** 2)
            if not math.isclose(self.P_out, expected_P_out_symbolic, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"P_out doesn't match target 200π²: {self.P_out:.2f} vs {expected_P_out_symbolic:.2f}", False)
                return False
            self._log(f"P_out matches target 200π²: {self.P_out:.2f}", True)
            
            return True
        except Exception as e:
            self._log(f"Trace consistency test failed with error: {str(e)}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb an input and check trend consistency."""
        original_B = self.B
        
        try:
            # Double the magnetic field
            self.B = 2 * original_B
            
            # Recalculate using formulas
            new_E_m = self.n * self.B * self.S * self.omega
            new_I_m = new_E_m / (self.R + self.r)
            new_I_eff = new_I_m / math.sqrt(2)
            new_P_R = new_I_eff ** 2 * self.R
            new_P_out = new_I_eff ** 2 * (self.R + self.r)
            
            # Check trends
            # E_m should double
            if not math.isclose(new_E_m / self.E_m, 2.0, rel_tol=1e-2):
                self._log(f"E_m doesn't double when B doubles: ratio = {new_E_m/self.E_m:.2f}", False)
                self.B = original_B
                return False
            
            # I_m should double
            if not math.isclose(new_I_m / self.I_m, 2.0, rel_tol=1e-2):
                self._log(f"I_m doesn't double when B doubles: ratio = {new_I_m/self.I_m:.2f}", False)
                self.B = original_B
                return False
            
            # I_eff should double
            if not math.isclose(new_I_eff / self.I_eff, 2.0, rel_tol=1e-2):
                self._log(f"I_eff doesn't double when B doubles: ratio = {new_I_eff/self.I_eff:.2f}", False)
                self.B = original_B
                return False
            
            # Powers should quadruple (since P ∝ I² ∝ B²)
            if not math.isclose(new_P_R / self.P_R, 4.0, rel_tol=1e-2):
                self._log(f"P_R doesn't quadruple when B doubles: ratio = {new_P_R/self.P_R:.2f}", False)
                self.B = original_B
                return False
            
            if not math.isclose(new_P_out / self.P_out, 4.0, rel_tol=1e-2):
                self._log(f"P_out doesn't quadruple when B doubles: ratio = {new_P_out/self.P_out:.2f}", False)
                self.B = original_B
                return False
            
            self._log("Monotonicity test passed: All quantities scale correctly with B", True)
            
            # Restore original value
            self.B = original_B
            return True
            
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {str(e)}", False)
            self.B = original_B  # Ensure restoration even on error
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success flag with logs."""
        self.logs = []  # Clear previous logs
        
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        overall_success = True
        
        for test_name, test_func in tests:
            self._log(f"Running {test_name}...")
            try:
                success = test_func()
                if not success:
                    overall_success = False
            except Exception as e:
                self._log(f"{test_name} crashed: {traceback.format_exc()}", False)
                overall_success = False
        
        return overall_success, self.logs