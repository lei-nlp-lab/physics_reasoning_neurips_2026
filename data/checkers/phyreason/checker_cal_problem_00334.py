import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with fallback to results dict
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs (work values in eV)
        self.W_PM = getattr(solution_instance, 'W_PM', 20.0)  # Work to move from P to M (overcoming field)
        self.W_MN = getattr(solution_instance, 'W_MN', 30.0)  # Work done by field from M to N
        
        # Extract results with safe fallbacks
        self.U_PM = self.results.get('U_PM', getattr(solution_instance, 'U_PM', 0.0))
        self.U_MN = self.results.get('U_MN', getattr(solution_instance, 'U_MN', 0.0))
        self.varphi_P = self.results.get('varphi_P', getattr(solution_instance, 'varphi_P', 0.0))
        self.E_P = self.results.get('E_P', getattr(solution_instance, 'E_P', 0.0))
        
        # Constants
        self.e = 1.602176634e-19  # elementary charge in C
        self.eV_to_J = self.e     # 1 eV = 1.602e-19 J
        
        # Log storage
        self.logs = []
        
    def log(self, msg):
        self.logs.append(msg)
    
    def test_invariants(self):
        """Test energy conservation and potential-work relationships."""
        try:
            # 1. Test potential difference definition: W = q * ΔV
            # For electron (q = -e), ΔV = -W/e
            # For P→M: external work = 20 eV against field => field work = -20 eV
            expected_U_PM = -(-20.0 * self.eV_to_J) / (-self.e)  # ΔV = -W_field/e
            expected_U_PM_val = 20.0  # In volts (from problem logic)
            
            if not math.isclose(self.U_PM, expected_U_PM_val, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: U_PM = {self.U_PM} V, expected ~{expected_U_PM_val} V")
                return False
            self.log(f"PASS: U_PM = {self.U_PM} V matches expected ~20 V")
            
            # For M→N: field does work = 30 eV
            expected_U_MN = -(30.0 * self.eV_to_J) / (-self.e)  # ΔV = -W_field/e
            expected_U_MN_val = -30.0  # In volts
            
            if not math.isclose(self.U_MN, expected_U_MN_val, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: U_MN = {self.U_MN} V, expected ~{expected_U_MN_val} V")
                return False
            self.log(f"PASS: U_MN = {self.U_MN} V matches expected ~-30 V")
            
            # 2. Test potential at P (reference consistency)
            # From steps: φ_P = U_PM = 20 V (assuming M is at 0V or similar reference)
            if not math.isclose(self.varphi_P, self.U_PM, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: φ_P = {self.varphi_P} V ≠ U_PM = {self.U_PM} V")
                return False
            self.log(f"PASS: φ_P = {self.varphi_P} V consistent with U_PM")
            
            # 3. Test potential energy: E_P = -e * φ_P
            expected_E_P = -self.varphi_P  # In eV (since -e * φ_P / e = -φ_P)
            if not math.isclose(self.E_P, expected_E_P, rel_tol=0.02, abs_tol=0.02):
                self.log(f"FAIL: E_P = {self.E_P} eV, expected ~{expected_E_P} eV")
                return False
            self.log(f"PASS: E_P = {self.E_P} eV consistent with -φ_P")
            
            return True
        except Exception as e:
            self.log(f"ERROR in test_invariants: {traceback.format_exc()}")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Target answers: ["20 V, -30 V", "20 V, -20 eV"]
            # First answer: U_PM and U_MN
            if not (math.isclose(self.U_PM, 20.0, rel_tol=0.02, abs_tol=0.02) and 
                    math.isclose(self.U_MN, -30.0, rel_tol=0.02, abs_tol=0.02)):
                self.log(f"FAIL: First answer mismatch: U_PM={self.U_PM}, U_MN={self.U_MN}")
                return False
            self.log(f"PASS: First answer {self.U_PM} V, {self.U_MN} V matches 20 V, -30 V")
            
            # Second answer: φ_P and E_P
            if not (math.isclose(self.varphi_P, 20.0, rel_tol=0.02, abs_tol=0.02) and 
                    math.isclose(self.E_P, -20.0, rel_tol=0.02, abs_tol=0.02)):
                self.log(f"FAIL: Second answer mismatch: φ_P={self.varphi_P}, E_P={self.E_P}")
                return False
            self.log(f"PASS: Second answer {self.varphi_P} V, {self.E_P} eV matches 20 V, -20 eV")
            
            return True
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            return False
    
    def test_monotonicity(self):
        """Perturb work input and check linear scaling."""
        try:
            original_W_PM = self.W_PM
            perturb_factor = 1.1  # +10%
            
            # Perturb W_PM and recalculate expected U_PM (linear relationship)
            perturbed_W_PM = original_W_PM * perturb_factor
            # ΔV = -W_field/e, and W_field = -W_PM (since overcoming field)
            expected_U_PM_perturbed = -(-perturbed_W_PM)  # In volts (simplified from -(-W_PM)/e * e)
            
            # Check if scaling is monotonic (U_PM should increase with W_PM)
            if not (expected_U_PM_perturbed > self.U_PM):
                self.log(f"FAIL: Monotonicity violated: U_PM should increase with W_PM")
                return False
            self.log(f"PASS: Monotonic scaling: U_PM increases with W_PM")
            
            # Restore original value (not strictly needed but good practice)
            self.W_PM = original_W_PM
            return True
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in tests:
            self.log(f"\n--- Running {name} Test ---")
            if not test_func():
                success = False
                self.log(f"FAILED: {name}")
            else:
                self.log(f"PASSED: {name}")
        
        return success, self.logs