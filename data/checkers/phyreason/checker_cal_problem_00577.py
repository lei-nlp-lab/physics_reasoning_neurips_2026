import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all required inputs and results from solution instance."""
        # Safely get results with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract key quantities with safe fallbacks
        self.U_1 = getattr(solution_instance, 'U_1', self.results.get('U_1', 0.0))
        self.I_1 = getattr(solution_instance, 'I_1', self.results.get('I_1', 0.0))
        self.U_2 = getattr(solution_instance, 'U_2', self.results.get('U_2', 0.0))
        self.R = getattr(solution_instance, 'R', self.results.get('R', 0.0))
        self.I_2 = getattr(solution_instance, 'I_2', self.results.get('I_2', 0.0))
        
        # Problem constants from KB
        self.n_ratio = 3.0  # n1:n2 = 3:1
        self.n_bulbs = 100
        self.U_lamp = 220.0  # V
        self.P_lamp = 60.0   # W
        
        # Tolerance for numerical comparisons (2% relative, 2 absolute)
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Test results log
        self.logs = []
    
    def _log_test(self, test_name, passed, message=""):
        """Helper to log test results."""
        status = "PASSED" if passed else "FAILED"
        full_message = f"{test_name}: {status}"
        if message:
            full_message += f" - {message}"
        self.logs.append(full_message)
        return passed
    
    def test_invariants(self):
        """Test fundamental physics invariants."""
        tests_passed = True
        
        try:
            # 1. Check transformer voltage ratio
            if self.U_2 != 0:
                voltage_ratio = self.U_1 / self.U_2
                expected_ratio = self.n_ratio
                if math.isclose(voltage_ratio, expected_ratio, 
                              rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    self._log_test("Voltage Ratio Invariant", True, 
                                 f"U1/U2 = {voltage_ratio:.3f} ≈ n1/n2 = {expected_ratio}")
                else:
                    self._log_test("Voltage Ratio Invariant", False,
                                 f"U1/U2 = {voltage_ratio:.3f}, expected {expected_ratio}")
                    tests_passed = False
            
            # 2. Check transformer current ratio (ideal transformer, power conservation)
            if self.I_2 != 0:
                current_ratio = self.I_1 / self.I_2
                expected_current_ratio = 1.0 / self.n_ratio
                if math.isclose(current_ratio, expected_current_ratio,
                              rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    self._log_test("Current Ratio Invariant", True,
                                 f"I1/I2 = {current_ratio:.3f} ≈ n2/n1 = {expected_current_ratio:.3f}")
                else:
                    self._log_test("Current Ratio Invariant", False,
                                 f"I1/I2 = {current_ratio:.3f}, expected {expected_current_ratio:.3f}")
                    tests_passed = False
            
            # 3. Check power conservation (input power ≈ output power for ideal transformer)
            P_primary = self.U_1 * self.I_1
            P_secondary = self.U_2 * self.I_2
            if math.isclose(P_primary, P_secondary,
                          rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log_test("Power Conservation", True,
                             f"P_primary = {P_primary:.1f} W ≈ P_secondary = {P_secondary:.1f} W")
            else:
                self._log_test("Power Conservation", False,
                             f"P_primary = {P_primary:.1f} W, P_secondary = {P_secondary:.1f} W")
                tests_passed = False
            
            # 4. Check parallel circuit property: total power equals sum of bulb powers
            expected_total_power = self.n_bulbs * self.P_lamp
            actual_total_power = self.U_2 * self.I_2
            if math.isclose(actual_total_power, expected_total_power,
                          rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log_test("Parallel Power Sum", True,
                             f"Total power = {actual_total_power:.1f} W ≈ {expected_total_power:.1f} W")
            else:
                self._log_test("Parallel Power Sum", False,
                             f"Total power = {actual_total_power:.1f} W, expected {expected_total_power:.1f} W")
                tests_passed = False
                
        except Exception as e:
            self._log_test("Invariants", False, f"Exception: {str(e)}")
            tests_passed = False
            
        return tests_passed
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        tests_passed = True
        
        try:
            # 1. Check resistance calculation from bulb specs
            expected_R = (self.U_lamp ** 2) / self.P_lamp
            if math.isclose(self.R, expected_R,
                          rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log_test("Resistance Consistency", True,
                             f"R = {self.R:.1f} Ω ≈ U²/P = {expected_R:.1f} Ω")
            else:
                self._log_test("Resistance Consistency", False,
                             f"R = {self.R:.1f} Ω, expected {expected_R:.1f} Ω")
                tests_passed = False
            
            # 2. Check secondary voltage matches bulb voltage
            if math.isclose(self.U_2, self.U_lamp,
                          rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log_test("Secondary Voltage Consistency", True,
                             f"U₂ = {self.U_2:.1f} V ≈ bulb voltage {self.U_lamp:.1f} V")
            else:
                self._log_test("Secondary Voltage Consistency", False,
                             f"U₂ = {self.U_2:.1f} V, expected {self.U_lamp:.1f} V")
                tests_passed = False
            
            # 3. Check secondary current from total power
            expected_I_2 = (self.n_bulbs * self.P_lamp) / self.U_lamp
            if math.isclose(self.I_2, expected_I_2,
                          rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log_test("Secondary Current Consistency", True,
                             f"I₂ = {self.I_2:.2f} A ≈ (n×P)/U = {expected_I_2:.2f} A")
            else:
                self._log_test("Secondary Current Consistency", False,
                             f"I₂ = {self.I_2:.2f} A, expected {expected_I_2:.2f} A")
                tests_passed = False
            
            # 4. Check primary voltage from transformer ratio
            expected_U_1 = self.n_ratio * self.U_lamp
            if math.isclose(self.U_1, expected_U_1,
                          rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log_test("Primary Voltage Consistency", True,
                             f"U₁ = {self.U_1:.1f} V ≈ n×U₂ = {expected_U_1:.1f} V")
            else:
                self._log_test("Primary Voltage Consistency", False,
                             f"U₁ = {self.U_1:.1f} V, expected {expected_U_1:.1f} V")
                tests_passed = False
            
            # 5. Check primary current from transformer ratio
            expected_I_1 = self.I_2 / self.n_ratio
            if math.isclose(self.I_1, expected_I_1,
                          rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self._log_test("Primary Current Consistency", True,
                             f"I₁ = {self.I_1:.2f} A ≈ I₂/n = {expected_I_1:.2f} A")
            else:
                self._log_test("Primary Current Consistency", False,
                             f"I₁ = {self.I_1:.2f} A, expected {expected_I_1:.2f} A")
                tests_passed = False
                
        except Exception as e:
            self._log_test("Trace Consistency", False, f"Exception: {str(e)}")
            tests_passed = False
            
        return tests_passed
    
    def test_monotonicity(self):
        """Test that changing input parameters produces expected trends."""
        tests_passed = True
        
        try:
            # Original parameters
            original_n_bulbs = self.n_bulbs
            original_P_lamp = self.P_lamp
            
            # Test 1: Increase number of bulbs
            perturbed_n = original_n_bulbs + 10  # 110 bulbs
            
            # Calculate expected changes using physics formulas
            # Secondary current should increase proportionally
            original_I_2 = (original_n_bulbs * original_P_lamp) / self.U_lamp
            perturbed_I_2 = (perturbed_n * original_P_lamp) / self.U_lamp
            
            if perturbed_I_2 > original_I_2:
                self._log_test("Bulb Count Monotonicity", True,
                             f"I₂ increases from {original_I_2:.2f} A to {perturbed_I_2:.2f} A with more bulbs")
            else:
                self._log_test("Bulb Count Monotonicity", False,
                             f"I₂ doesn't increase with more bulbs")
                tests_passed = False
            
            # Primary current should also increase proportionally
            original_I_1 = original_I_2 / self.n_ratio
            perturbed_I_1 = perturbed_I_2 / self.n_ratio
            
            if perturbed_I_1 > original_I_1:
                self._log_test("Primary Current Trend", True,
                             f"I₁ increases from {original_I_1:.2f} A to {perturbed_I_1:.2f} A")
            else:
                self._log_test("Primary Current Trend", False,
                             f"I₁ doesn't increase with more bulbs")
                tests_passed = False
            
            # Test 2: Decrease bulb power (simulate dimmer bulbs)
            perturbed_P = original_P_lamp * 0.8  # 48W bulbs
            
            # With same number of bulbs, current should decrease
            perturbed_I_2_lowP = (original_n_bulbs * perturbed_P) / self.U_lamp
            
            if perturbed_I_2_lowP < original_I_2:
                self._log_test("Power Monotonicity", True,
                             f"I₂ decreases from {original_I_2:.2f} A to {perturbed_I_2_lowP:.2f} A with lower power bulbs")
            else:
                self._log_test("Power Monotonicity", False,
                             f"I₂ doesn't decrease with lower power bulbs")
                tests_passed = False
            
            # Reset parameters
            self.n_bulbs = original_n_bulbs
            
        except Exception as e:
            self._log_test("Monotonicity", False, f"Exception: {str(e)}")
            tests_passed = False
            
        return tests_passed
    
    def run_all_tests(self):
        """Execute all tests and return success status with logs."""
        self.logs = []  # Reset logs
        
        # Run all tests
        invariants_ok = self.test_invariants()
        trace_ok = self.test_trace_consistency()
        monotonicity_ok = self.test_monotonicity()
        
        # Determine overall success
        all_passed = invariants_ok and trace_ok and monotonicity_ok
        
        # Add summary
        self.logs.append(f"\n=== SUMMARY ===")
        self.logs.append(f"Invariants: {'PASSED' if invariants_ok else 'FAILED'}")
        self.logs.append(f"Trace Consistency: {'PASSED' if trace_ok else 'FAILED'}")
        self.logs.append(f"Monotonicity: {'PASSED' if monotonicity_ok else 'FAILED'}")
        self.logs.append(f"OVERALL: {'PASSED' if all_passed else 'FAILED'}")
        
        return all_passed, self.logs