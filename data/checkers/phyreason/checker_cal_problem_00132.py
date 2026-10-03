import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        
        # Extract given parameters safely
        self.B = getattr(solution_instance, 'B', 0.2)  # Magnetic field (T)
        self.N = getattr(solution_instance, 'N', 100)  # Number of turns
        self.S = getattr(solution_instance, 'S', 0.5)  # Area (m^2)
        self.R = getattr(solution_instance, 'R', 0.6)  # Resistance (Ω)
        self.omega = getattr(solution_instance, 'omega', 90.0)  # Angular velocity (rad/s)
        self.I = getattr(solution_instance, 'I', 50.0)  # Current (A)
        
        # Extract results safely
        results_dict = getattr(solution_instance, 'results', {})
        self.E = results_dict.get('E', getattr(solution_instance, 'E', 0.0))
        self.P = results_dict.get('P', getattr(solution_instance, 'P', 0.0))
        self.U = results_dict.get('U', getattr(solution_instance, 'U', 0.0))
        
        # Expected values from physics formulas (not rounded approximations)
        self.E_expected = (self.N * self.B * self.S * self.omega) / math.sqrt(2)
        self.P_expected = self.I * (self.E_expected - self.I * self.R)
        
        self.logs = []
        
    def _log(self, msg, level="INFO"):
        self.logs.append(f"[{level}] {msg}")
        
    def test_invariants(self):
        """Verify conservation laws and fundamental relationships"""
        try:
            # 1. Check Faraday's Law: E = N*B*S*ω/√2 (RMS for sinusoidal)
            E_calculated = (self.N * self.B * self.S * self.omega) / math.sqrt(2)
            
            if not math.isclose(self.E, E_calculated, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Faraday's law violation: E={self.E} vs calculated={E_calculated}", "WARNING")
                return False
            
            self._log(f"Faraday's law check passed: E={self.E} ≈ {E_calculated}")
            
            # 2. Check Ohm's Law and voltage relationships: U = E - I*R
            U_calculated = self.E - self.I * self.R
            
            if hasattr(self.solution, 'U') or 'U' in getattr(self.solution, 'results', {}):
                if not math.isclose(self.U, U_calculated, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Ohm's law violation: U={self.U} vs E-I*R={U_calculated}", "WARNING")
                    return False
                self._log(f"Ohm's law check passed: U={self.U} ≈ E - I*R = {U_calculated}")
            
            # 3. Check power relationships: P = I*U = I*(E - I*R)
            P_calculated = self.I * U_calculated
            
            if not math.isclose(self.P, P_calculated, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Power formula violation: P={self.P} vs I*(E-I*R)={P_calculated}", "WARNING")
                return False
            
            self._log(f"Power formula check passed: P={self.P} ≈ I*(E-I*R) = {P_calculated}")
            
            # 4. Energy conservation (power balance)
            # Electrical power output + thermal power = generated power
            P_thermal = self.I**2 * self.R
            P_generated = self.I * self.E
            
            if not math.isclose(self.P + P_thermal, P_generated, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Energy conservation violation: P_out+P_thermal={self.P+P_thermal} vs P_gen={P_generated}", "WARNING")
                return False
            
            self._log(f"Energy conservation check passed: P_out+P_thermal={self.P+P_thermal} ≈ P_gen={P_generated}")
            
            return True
            
        except Exception as e:
            self._log(f"Error in test_invariants: {str(e)}", "ERROR")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        try:
            # Check step 1 result matches final E
            E_step1 = (self.N * self.B * self.S * self.omega) / math.sqrt(2)
            
            if not math.isclose(self.E, E_step1, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Step 1 inconsistency: E={self.E} vs step1 formula={E_step1}", "WARNING")
                return False
            
            self._log(f"Step 1 consistency: E={self.E} matches formula")
            
            # Check step 4 result matches final P
            P_step4 = self.I * (self.E - self.I * self.R)
            
            if not math.isclose(self.P, P_step4, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Step 4 inconsistency: P={self.P} vs step4 formula={P_step4}", "WARNING")
                return False
            
            self._log(f"Step 4 consistency: P={self.P} matches formula")
            
            return True
            
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {str(e)}", "ERROR")
            return False
    
    def test_monotonicity(self):
        """Check that perturbations to inputs produce expected trends"""
        try:
            # Store original values
            orig_omega = self.omega
            orig_E = self.E
            orig_P = self.P
            
            # Test 1: Increase angular velocity by 10%
            new_omega = orig_omega * 1.1
            
            # Recalculate E with new omega
            E_new = (self.N * self.B * self.S * new_omega) / math.sqrt(2)
            
            # According to Faraday's law, E should increase with omega
            if not (E_new > orig_E):
                self._log(f"Monotonicity violation: E didn't increase when ω increased", "WARNING")
                return False
            
            self._log(f"Monotonicity check 1: E increases with ω (E_new={E_new} > E_orig={orig_E})")
            
            # Recalculate P with new E
            P_new = self.I * (E_new - self.I * self.R)
            
            # Since E increased and I,R constant, P should increase
            if not (P_new > orig_P):
                self._log(f"Monotonicity violation: P didn't increase when ω increased", "WARNING")
                return False
            
            self._log(f"Monotonicity check 2: P increases with ω (P_new={P_new} > P_orig={orig_P})")
            
            # Test 2: Increase resistance by 10%
            orig_R = self.R
            new_R = orig_R * 1.1
            
            # Recalculate P with increased resistance
            P_new_R = self.I * (orig_E - self.I * new_R)
            
            # Since P = I*(E - I*R), increasing R decreases P
            if not (P_new_R < orig_P):
                self._log(f"Monotonicity violation: P didn't decrease when R increased", "WARNING")
                return False
            
            self._log(f"Monotonicity check 3: P decreases with R (P_new={P_new_R} < P_orig={orig_P})")
            
            return True
            
        except Exception as e:
            self._log(f"Error in test_monotonicity: {str(e)}", "ERROR")
            return False
    
    def test_expected_values(self):
        """Check against correct physics formulas"""
        try:
            # Calculate expected values from physics formulas
            E_expected = (self.N * self.B * self.S * self.omega) / math.sqrt(2)
            P_expected = self.I * (E_expected - self.I * self.R)
            
            # Check E against expected
            if not math.isclose(self.E, E_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"E mismatch: {self.E} vs physics formula {E_expected}", "WARNING")
                return False
            
            self._log(f"E matches physics formula: {self.E} ≈ {E_expected}")
            
            # Check P against expected
            if not math.isclose(self.P, P_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"P mismatch: {self.P} vs physics formula {P_expected}", "WARNING")
                return False
            
            self._log(f"P matches physics formula: {self.P} ≈ {P_expected}")
            
            return True
            
        except Exception as e:
            self._log(f"Error in test_expected_values: {str(e)}", "ERROR")
            return False
    
    def run_all_tests(self):
        """Execute all tests and return results"""
        self.logs = []
        success = True
        
        tests = [
            ("Expected Values", self.test_expected_values),
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"Running {test_name}...")
            try:
                if not test_func():
                    success = False
                    self._log(f"{test_name} failed", "WARNING")
                else:
                    self._log(f"{test_name} passed", "INFO")
            except Exception as e:
                success = False
                self._log(f"{test_name} crashed: {traceback.format_exc()}", "ERROR")
        
        return success, self.logs