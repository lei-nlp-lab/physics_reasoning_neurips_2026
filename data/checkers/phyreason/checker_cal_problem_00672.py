import math
import traceback
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access: use getattr with fallback to results dict
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract results with safe defaults
        self.E = getattr(solution_instance, 'E', self.results.get('E', 0.0))
        self.R_sj = getattr(solution_instance, 'R_sj', self.results.get('R_sj', 0.0))
        self.I_M = getattr(solution_instance, 'I_M', self.results.get('I_M', 0.0))
        self.r_0 = getattr(solution_instance, 'r_0', self.results.get('r_0', 0.0))
        self.U_prime = getattr(solution_instance, 'U_prime', self.results.get("U'", 0.0))
        
        # Input parameters from problem statement
        self.I1 = 10.0  # A, ammeter reading with only S1 closed
        self.V1 = 12.0  # V, voltmeter reading with only S1 closed
        self.I2 = 8.0   # A, ammeter reading with both switches closed
        self.r_b = 0.05 # Ω, battery internal resistance (given)
        
        # Expected values from steps analysis
        self.expected_E = 12.5  # V
        self.expected_R_sj = 1.2  # Ω
        self.expected_I_M = 50.0  # A
        self.expected_r_0 = 0.04  # Ω
        
        self.logs = []
    
    def log(self, message):
        self.logs.append(message)
    
    def test_invariants(self):
        """Test fundamental physical laws"""
        success = True
        
        # 1. Test Ohm's Law for first scenario (only headlight)
        # Headlight resistance from V1 and I1
        R_sj_calc = self.V1 / self.I1
        if not math.isclose(R_sj_calc, self.expected_R_sj, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: Headlight resistance from Ohm's Law: {R_sj_calc} Ω, expected {self.expected_R_sj} Ω")
            success = False
        else:
            self.log(f"OK: Headlight resistance {R_sj_calc} Ω matches expected {self.expected_R_sj} Ω")
        
        # 2. Test battery EMF from first scenario
        # E = V + I*r (internal battery resistance)
        E_calc = self.V1 + self.I1 * self.r_b
        if not math.isclose(E_calc, self.expected_E, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: Battery EMF calculation: {E_calc} V, expected {self.expected_E} V")
            success = False
        else:
            self.log(f"OK: Battery EMF {E_calc} V matches expected {self.expected_E} V")
        
        # 3. Test Kirchhoff's Current Law for second scenario
        # When both switches closed, current through headlight is I2 = 8A
        # Current through motor is I_M
        # Total current from battery should be I2 + I_M
        # Voltage across headlight: V_headlight = I2 * R_sj_calc
        V_headlight = self.I2 * R_sj_calc
        
        # Voltage across motor terminals (same as headlight since parallel)
        # From battery perspective: E = V_headlight + (I2 + I_M) * r_b
        I_M_calc = (self.expected_E - V_headlight) / self.r_b - self.I2
        if not math.isclose(I_M_calc, self.expected_I_M, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: Motor current from KCL: {I_M_calc} A, expected {self.expected_I_M} A")
            success = False
        else:
            self.log(f"OK: Motor current {I_M_calc} A matches expected {self.expected_I_M} A")
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        success = True
        
        # Check if computed results match expected values (with tolerance)
        if hasattr(self, 'E') and not math.isclose(self.E, self.expected_E, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: Computed E = {self.E} V ≠ expected {self.expected_E} V")
            success = False
        else:
            self.log(f"OK: Computed E = {self.E} V matches expected")
        
        if hasattr(self, 'R_sj') and not math.isclose(self.R_sj, self.expected_R_sj, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: Computed R_sj = {self.R_sj} Ω ≠ expected {self.expected_R_sj} Ω")
            success = False
        else:
            self.log(f"OK: Computed R_sj = {self.R_sj} Ω matches expected")
        
        if hasattr(self, 'I_M') and not math.isclose(self.I_M, self.expected_I_M, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: Computed I_M = {self.I_M} A ≠ expected {self.expected_I_M} A")
            success = False
        else:
            self.log(f"OK: Computed I_M = {self.I_M} A matches expected")
        
        if hasattr(self, 'r_0') and not math.isclose(self.r_0, self.expected_r_0, rel_tol=1e-2, abs_tol=1e-2):
            self.log(f"FAIL: Computed r_0 = {self.r_0} Ω ≠ expected {self.expected_r_0} Ω")
            success = False
        else:
            self.log(f"OK: Computed r_0 = {self.r_0} Ω matches expected")
        
        # Verify internal consistency: E should equal V1 + I1*r_b
        if hasattr(self, 'E'):
            E_check = self.V1 + self.I1 * self.r_b
            if not math.isclose(self.E, E_check, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"WARNING: E ({self.E} V) doesn't match V1 + I1*r_b = {E_check} V")
        
        return success
    
    def test_monotonicity(self):
        """Perturb an input and check trend"""
        success = True
        
        # Perturb the first ammeter reading (I1) and observe changes
        perturbed_I1 = self.I1 * 1.05  # Increase by 5%
        
        # Recalculate using formulas (not accessing solution_instance)
        # Headlight resistance should remain constant (given as constant)
        R_sj_constant = self.V1 / self.I1
        
        # Recalculate EMF with perturbed I1
        E_perturbed = self.V1 + perturbed_I1 * self.r_b
        
        # Check monotonicity: If I1 increases, E should increase (E = V + I*r)
        if E_perturbed <= self.expected_E:
            self.log(f"FAIL: E doesn't increase when I1 increases. E_original={self.expected_E} V, E_perturbed={E_perturbed} V")
            success = False
        else:
            self.log(f"OK: E increases from {self.expected_E} V to {E_perturbed} V when I1 increases")
        
        # Restore original values (conceptual)
        self.log("Monotonicity test complete - original values restored")
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return success status with logs"""
        self.logs = []
        
        try:
            overall_success = True
            
            self.log("=== Starting PhysicsChecker ===")
            
            # Run individual tests
            test1 = self.test_invariants()
            if not test1:
                overall_success = False
                self.log("FAIL: test_invariants")
            
            test2 = self.test_trace_consistency()
            if not test2:
                overall_success = False
                self.log("FAIL: test_trace_consistency")
            
            test3 = self.test_monotonicity()
            if not test3:
                overall_success = False
                self.log("FAIL: test_monotonicity")
            
            if overall_success:
                self.log("=== All tests passed ===")
            else:
                self.log("=== Some tests failed ===")
            
            return overall_success, self.logs
            
        except Exception as e:
            self.log(f"ERROR during testing: {str(e)}")
            self.log(traceback.format_exc())
            return False, self.logs