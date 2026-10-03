import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extracts all inputs and results from the solution instance.
        Uses getattr and dictionary fallbacks to prevent AttributeError.
        """
        # Input parameters from problem statement
        self.E = getattr(solution_instance, 'E', 16.0)  # V
        self.r = getattr(solution_instance, 'r', 1.0)   # Ω
        self.R = getattr(solution_instance, 'R', 4.0)   # Ω
        self.U_L = getattr(solution_instance, 'U_L', 3.0)  # Bulb rated voltage (V)
        self.P_L = getattr(solution_instance, 'P_L', 4.5)  # Bulb rated power (W)
        self.r_motor = getattr(solution_instance, 'r_motor', 1.0)  # Ω
        
        # Results from solution instance (using safe access with fallback)
        results = getattr(solution_instance, 'results', {})
        self.I = getattr(solution_instance, 'I', results.get('I', 0.0))
        self.U_M = getattr(solution_instance, 'U_M', results.get('U_M', 0.0))
        self.P_total = getattr(solution_instance, 'P', results.get('P', 0.0))
        
        # For symbolic calculations
        self.I_sym = sp.symbols('I')
        self.U_M_sym = sp.symbols('U_M')
        self.P_total_sym = sp.symbols('P_total')
        
        self.logs = []
    
    def _log(self, message, success=True):
        """Helper to log messages with success status."""
        self.logs.append((message, success))
        return success
    
    def test_invariants(self):
        """Test fundamental physical invariants: Ohm's Law, KVL, Power balance."""
        success = True
        
        # 1. Ohm's Law for the bulb: P_L = U_L * I
        bulb_current = self.P_L / self.U_L  # Expected current through bulb
        if not math.isclose(self.I, bulb_current, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(
                f"Ohm's Law for bulb failed: I = {self.I}A, expected P_L/U_L = {bulb_current}A",
                False
            )
        else:
            self._log(f"Ohm's Law for bulb satisfied: I = {self.I}A")
        
        # 2. Kirchhoff's Voltage Law (KVL) around the loop
        # E = U_L + I*R + I*r + U_M + I*r_motor? Wait, motor has internal resistance r_motor
        # From step analysis: U_M = E - U_L - I*R - I*r (r_motor not in equation)
        # Actually, looking at step_2: "U_M = E - U_L - I R - I r" - they used r (source internal resistance)
        # But motor has internal resistance r'=1Ω. Where is that voltage drop?
        # In step_2 equation, they seem to treat U_M as the terminal voltage of the motor.
        # So the voltage drop across motor's internal resistance is I*r_motor.
        # Thus total voltage drop: U_L + I*R + I*r + (U_M + I*r_motor) = E
        # Rearranged: U_M = E - U_L - I*(R + r + r_motor)
        # Let's check both interpretations:
        
        # Interpretation from step analysis (which might be incomplete):
        U_M_kvl_simple = self.E - self.U_L - self.I * (self.R + self.r)
        # Complete interpretation:
        U_M_kvl_complete = self.E - self.U_L - self.I * (self.R + self.r + self.r_motor)
        
        # Check which one matches the reported U_M (with tolerance)
        if math.isclose(self.U_M, U_M_kvl_simple, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"KVL (simple) satisfied: U_M = {self.U_M}V matches {U_M_kvl_simple}V")
        elif math.isclose(self.U_M, U_M_kvl_complete, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"KVL (complete) satisfied: U_M = {self.U_M}V matches {U_M_kvl_complete}V")
        else:
            success = self._log(
                f"KVL failed: U_M = {self.U_M}V, expected from simple KVL: {U_M_kvl_simple}V, "
                f"expected from complete KVL: {U_M_kvl_complete}V",
                False
            )
        
        # 3. Power supply total power: P_total = E * I
        expected_P_total = self.E * self.I
        if not math.isclose(self.P_total, expected_P_total, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(
                f"Power supply formula failed: P_total = {self.P_total}W, expected E*I = {expected_P_total}W",
                False
            )
        else:
            self._log(f"Power supply formula satisfied: P_total = {self.P_total}W")
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results from steps with final answers."""
        success = True
        
        # Step 1: I = P_L / U_L = 4.5/3 = 1.5A
        expected_I = 1.5
        if not math.isclose(self.I, expected_I, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(
                f"Step 1 inconsistency: I = {self.I}A, expected 1.5A",
                False
            )
        else:
            self._log(f"Step 1 consistent: I = {self.I}A")
        
        # Step 2: U_M = E - U_L - I*R - I*r = 16 - 3 - 1.5*4 - 1.5*1 = 5.5V
        # Using the same equation as in step analysis
        expected_U_M = 16 - 3 - 1.5*4 - 1.5*1
        if not math.isclose(self.U_M, expected_U_M, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(
                f"Step 2 inconsistency: U_M = {self.U_M}V, expected {expected_U_M}V",
                False
            )
        else:
            self._log(f"Step 2 consistent: U_M = {self.U_M}V")
        
        # Step 3: P_total = E*I = 16*1.5 = 24W
        expected_P = 24.0
        if not math.isclose(self.P_total, expected_P, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(
                f"Step 3 inconsistency: P_total = {self.P_total}W, expected 24W",
                False
            )
        else:
            self._log(f"Step 3 consistent: P_total = {self.P_total}W")
        
        return success
    
    def test_monotonicity(self):
        """
        Perturb an input (R) and verify expected trends from Ohm's Law and KVL.
        Restores original values after check.
        """
        success = True
        
        # Store original R
        original_R = self.R
        
        # Increase R by 10%
        perturbed_R = original_R * 1.1
        self.R = perturbed_R
        
        # Recalculate expected current - but wait, bulb is operating normally
        # So U_L and P_L are fixed, meaning I remains the same? Actually, if bulb operates
        # normally, it must have exactly 3V and draw 4.5W, so I = P_L/U_L = 1.5A is fixed.
        # So I should not change when R changes (as long as circuit can adjust).
        
        # Recalculate U_M using the same formula as in step analysis
        # U_M' = E - U_L - I*(R' + r)
        recalculated_U_M = self.E - self.U_L - self.I * (self.R + self.r)
        
        # Recalculate total power: P_total' = E*I (I unchanged)
        recalculated_P = self.E * self.I
        
        # Check trends:
        # 1. Current I should remain constant (bulb operating normally)
        if not math.isclose(self.I, 1.5, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(
                f"Monotonicity check failed: I changed from 1.5A to {self.I}A when R increased",
                False
            )
        
        # 2. U_M should decrease when R increases (more voltage drop across R)
        if recalculated_U_M >= self.U_M:  # Should be strictly less
            success = self._log(
                f"Monotonicity check failed: U_M increased or stayed same when R increased. "
                f"Original U_M = {self.U_M}V, new U_M = {recalculated_U_M}V",
                False
            )
        else:
            self._log(f"Monotonicity: U_M decreased from {self.U_M}V to {recalculated_U_M}V as R increased ✓")
        
        # 3. Total power should remain constant (since I constant)
        if not math.isclose(recalculated_P, self.P_total, rel_tol=1e-2, abs_tol=1e-2):
            success = self._log(
                f"Monotonicity check failed: P_total changed from {self.P_total}W to {recalculated_P}W",
                False
            )
        else:
            self._log(f"Monotonicity: P_total remained constant at {self.P_total}W as R increased ✓")
        
        # Restore original R
        self.R = original_R
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        self.logs = []
        all_success = True
        
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency Tests", self.test_trace_consistency),
            ("Monotonicity Tests", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                self._log(f"--- Running {test_name} ---")
                if not test_func():
                    all_success = False
            except Exception as e:
                all_success = False
                self._log(f"{test_name} crashed: {str(e)}\n{traceback.format_exc()}", False)
        
        return all_success, self.logs