import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs and results
        self.E = 9.0
        self.r = 1.0
        self.R = 2.0
        
        # Extract results safely from solution_instance
        results = getattr(solution_instance, 'results', {})
        
        self.I = getattr(solution_instance, 'I', results.get('I', 0.0))
        self.U = getattr(solution_instance, 'U', results.get('U', 0.0))
        
        # For power, check for both output and total power
        self.P_output = getattr(solution_instance, 'P_output', 
                               results.get('P_output', results.get('P', 0.0)))
        self.P_total = getattr(solution_instance, 'P_total', 
                              results.get('P_total', 0.0))
        
        self.logs = []
        
    def _log(self, message, success=None):
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {message}")
        else:
            self.logs.append(f"INFO: {message}")
    
    def _isclose(self, a, b):
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        """Verify physical conservation laws using symbolic and numerical checks"""
        try:
            I, U, P_out, P_tot = sp.symbols('I U P_out P_tot')
            E_val, r_val, R_val = self.E, self.r, self.R
            
            # Symbolic Ohm's Law for closed circuit
            ohm_law = sp.Eq(I, E_val/(R_val + r_val))
            # Symbolic Ohm's Law for terminal voltage
            terminal_voltage = sp.Eq(U, I * R_val)
            # Symbolic output power
            output_power = sp.Eq(P_out, U * I)
            # Symbolic total power
            total_power = sp.Eq(P_tot, E_val * I)
            
            # Numerical checks with fuzzy matching
            # 1. Check closed circuit current
            expected_I = self.E / (self.R + self.r)
            if self._isclose(self.I, expected_I):
                self._log(f"Ohm's Law (closed circuit): I = {self.I:.2f}A matches E/(R+r) = {expected_I:.2f}A", True)
            else:
                self._log(f"Ohm's Law (closed circuit): I = {self.I:.2f}A ≠ E/(R+r) = {expected_I:.2f}A", False)
                return False
            
            # 2. Check terminal voltage
            expected_U = self.I * self.R
            if self._isclose(self.U, expected_U):
                self._log(f"Ohm's Law (terminal): U = {self.U:.2f}V matches I*R = {expected_U:.2f}V", True)
            else:
                self._log(f"Ohm's Law (terminal): U = {self.U:.2f}V ≠ I*R = {expected_U:.2f}V", False)
                return False
            
            # 3. Check output power
            expected_P_out = self.U * self.I
            if self._isclose(self.P_output, expected_P_out):
                self._log(f"Output power: P = {self.P_output:.2f}W matches U*I = {expected_P_out:.2f}W", True)
            else:
                self._log(f"Output power: P = {self.P_output:.2f}W ≠ U*I = {expected_P_out:.2f}W", False)
                return False
            
            # 4. Check total power
            expected_P_tot = self.E * self.I
            if self._isclose(self.P_total, expected_P_tot):
                self._log(f"Total power: P = {self.P_total:.2f}W matches E*I = {expected_P_tot:.2f}W", True)
            else:
                self._log(f"Total power: P = {self.P_total:.2f}W ≠ E*I = {expected_P_tot:.2f}W", False)
                return False
            
            return True
            
        except Exception as e:
            self._log(f"Error in test_invariants: {str(e)}", False)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer"""
        try:
            # Verify the chain of calculations from steps analysis
            # Step 1: I = E/(R+r) = 9/(2+1) = 3A
            expected_I = self.E / (self.R + self.r)
            if not self._isclose(self.I, expected_I):
                self._log(f"Step 1 inconsistency: I = {self.I:.2f}A, expected {expected_I:.2f}A", False)
                return False
            
            # Step 2: U = I*R = 3*2 = 6V
            expected_U = self.I * self.R
            if not self._isclose(self.U, expected_U):
                self._log(f"Step 2 inconsistency: U = {self.U:.2f}V, expected {expected_U:.2f}V", False)
                return False
            
            # Step 3: P_output = U*I = 6*3 = 18W
            expected_P_out = self.U * self.I
            if not self._isclose(self.P_output, expected_P_out):
                self._log(f"Step 3 inconsistency: P_output = {self.P_output:.2f}W, expected {expected_P_out:.2f}W", False)
                return False
            
            # Step 4: P_total = E*I = 9*3 = 27W
            expected_P_tot = self.E * self.I
            if not self._isclose(self.P_total, expected_P_tot):
                self._log(f"Step 4 inconsistency: P_total = {self.P_total:.2f}W, expected {expected_P_tot:.2f}W", False)
                return False
            
            self._log("All steps consistent with final results", True)
            return True
            
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {str(e)}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb resistance R and verify expected trends"""
        try:
            original_R = self.R
            
            # Test 1: Increase R
            self.R = original_R + 1.0  # Increase by 1Ω
            
            # Recalculate using formulas
            new_I = self.E / (self.R + self.r)
            new_U = new_I * self.R
            new_P_out = new_U * new_I
            new_P_tot = self.E * new_I
            
            # Check trends (based on Ohm's Law)
            # When R increases, current should decrease
            if new_I < self.I:
                trend_I = "decrease ✓"
                trend_I_pass = True
            else:
                trend_I = "increase ✗"
                trend_I_pass = False
            
            # When R increases and I decreases, U = I*R trend depends on values
            # For this specific circuit, let's check if the product I*R decreases
            # Since I decreases proportionally more than R increases (for R>r), U should decrease
            # Actually, U = E * R/(R+r). For R=2->3, U=6->6.75 (increases)
            # So U should increase when R increases
            if new_U > self.U:
                trend_U = "increase ✓"
                trend_U_pass = True
            else:
                trend_U = "decrease ✗"
                trend_U_pass = False
            
            # Output power P_out = U*I = E² * R/(R+r)²
            # Derivative shows maximum at R=r=1Ω. For R=2->3, power decreases
            if new_P_out < self.P_output:
                trend_P_out = "decrease ✓"
                trend_P_out_pass = True
            else:
                trend_P_out = "increase ✗"
                trend_P_out_pass = False
            
            # Total power P_tot = E*I = E²/(R+r) always decreases when R increases
            if new_P_tot < self.P_total:
                trend_P_tot = "decrease ✓"
                trend_P_tot_pass = True
            else:
                trend_P_tot = "increase ✗"
                trend_P_tot_pass = False
            
            # Restore original R
            self.R = original_R
            
            self._log(f"Monotonicity test (R↑): I should {trend_I}, U should {trend_U}, "
                     f"P_out should {trend_P_out}, P_tot should {trend_P_tot}", 
                     all([trend_I_pass, trend_U_pass, trend_P_out_pass, trend_P_tot_pass]))
            
            return all([trend_I_pass, trend_U_pass, trend_P_out_pass, trend_P_tot_pass])
            
        except Exception as e:
            self.R = original_R  # Ensure restoration even if error occurs
            self._log(f"Error in test_monotonicity: {str(e)}", False)
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success status with logs"""
        self.logs = []
        success = True
        
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"\n=== Running {test_name} ===", None)
            if not test_func():
                success = False
        
        return success, self.logs