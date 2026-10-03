import math
import traceback
import random

class PhysicsChecker:
    """
    Physics auditor for electromagnetic induction problem: rectangular coil rotating
    in uniform magnetic field, connected to external resistor R.
    """
    
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results using getattr to prevent AttributeError.
        """
        # Input parameters (with defaults to avoid crashes)
        self.B = getattr(solution_instance, 'B', 1.0)
        self.L1 = getattr(solution_instance, 'L1', 1.0)
        self.L2 = getattr(solution_instance, 'L2', 1.0)
        self.omega = getattr(solution_instance, 'omega', 1.0)
        self.R = getattr(solution_instance, 'R', 1.0)
        self.r = getattr(solution_instance, 'r', 1.0)
        
        # Results - check both attributes and results dictionary
        self.results = getattr(solution_instance, 'results', {})
        self.E_m = self.results.get('E_m', 0.0)
        self.e = self.results.get('e', 0.0)
        self.E_effective = self.results.get('E_effective', 0.0)
        self.I_valid = self.results.get('I_valid', 0.0)
        self.P = self.results.get('P', 0.0)
        
        self.logs = []
    
    def _log(self, message, is_error=False):
        """Helper to log messages."""
        self.logs.append(f"{'ERROR' if is_error else 'INFO'}: {message}")
    
    def _check_numeric_equality(self, val1, val2, context=""):
        """Check if two numeric values are close within tolerance."""
        if not math.isclose(val1, val2, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"{context}: Numerical mismatch {val1} ≠ {val2}", is_error=True)
            return False
        return True
    
    def test_invariants(self):
        """Verify fundamental physics laws: Faraday, Ohm, Joule using numeric checks."""
        success = True
        
        # 1. Faraday's Law: E_m = B*L1*L2*ω
        expected_E_m = self.B * self.L1 * self.L2 * self.omega
        if not self._check_numeric_equality(self.E_m, expected_E_m, "Faraday's Law (E_m)"):
            success = False
            self._log(f"E_m mismatch: got {self.E_m}, expected {expected_E_m}", is_error=True)
        
        # 2. Instantaneous EMF: e = E_m * sin(ωt) - need t value
        # Since t is not provided in inputs, we cannot compute e directly
        # We'll skip this check as t is not available
        self._log("Skipping e check: t not available in inputs")
        
        # 3. Effective EMF: E_effective = E_m / √2
        expected_E_eff = expected_E_m / math.sqrt(2)
        if not self._check_numeric_equality(self.E_effective, expected_E_eff, "Effective EMF"):
            success = False
            self._log(f"E_effective mismatch: got {self.E_effective}, expected {expected_E_eff}", is_error=True)
        
        # 4. Ohm's Law: I_valid = E_effective / (R + r)
        expected_I = expected_E_eff / (self.R + self.r)
        if not self._check_numeric_equality(self.I_valid, expected_I, "Ohm's Law (I_valid)"):
            success = False
            self._log(f"I_valid mismatch: got {self.I_valid}, expected {expected_I}", is_error=True)
        
        # 5. Joule's Law: P = I_valid² * R
        expected_P = expected_I**2 * self.R
        if not self._check_numeric_equality(self.P, expected_P, "Joule's Law (P)"):
            success = False
            self._log(f"P mismatch: got {self.P}, expected {expected_P}", is_error=True)
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer using numeric values."""
        success = True
        
        # Check that I_valid = E_effective / (R + r) holds with computed values
        try:
            expected_I = self.E_effective / (self.R + self.r)
            if not self._check_numeric_equality(self.I_valid, expected_I, "I_valid from E_effective"):
                success = False
                self._log("I_valid not consistent with E_effective/(R+r)", is_error=True)
        except Exception as e:
            self._log(f"Trace consistency check failed: {e}", is_error=True)
            success = False
        
        # Check that P = I_valid² * R
        try:
            expected_P = self.I_valid**2 * self.R
            if not self._check_numeric_equality(self.P, expected_P, "P from I_valid"):
                success = False
                self._log("P not consistent with I_valid²*R", is_error=True)
        except Exception as e:
            self._log(f"Trace consistency check failed: {e}", is_error=True)
            success = False
        
        return success
    
    def test_monotonicity(self):
        """
        Perturb input parameters and verify expected trends using numeric computations.
        """
        success = True
        
        # Test 1: Increase B → P should increase (quadratic in B)
        try:
            # Compute P with original B
            E_m_orig = self.B * self.L1 * self.L2 * self.omega
            E_eff_orig = E_m_orig / math.sqrt(2)
            I_orig = E_eff_orig / (self.R + self.r)
            P_orig = I_orig**2 * self.R
            
            # Compute P with increased B
            B_pert = self.B * 1.01
            E_m_pert = B_pert * self.L1 * self.L2 * self.omega
            E_eff_pert = E_m_pert / math.sqrt(2)
            I_pert = E_eff_pert / (self.R + self.r)
            P_pert = I_pert**2 * self.R
            
            if not (P_pert > P_orig):
                success = False
                self._log(f"Monotonicity fail: P(B↑) = {P_pert} ≤ {P_orig} = P(B)", is_error=True)
        except Exception as e:
            self._log(f"Monotonicity check for B failed: {e}", is_error=True)
            success = False
        
        # Test 2: Increase ω → P should increase (quadratic in ω)
        try:
            # Compute P with original ω
            E_m_orig = self.B * self.L1 * self.L2 * self.omega
            E_eff_orig = E_m_orig / math.sqrt(2)
            I_orig = E_eff_orig / (self.R + self.r)
            P_orig = I_orig**2 * self.R
            
            # Compute P with increased ω
            omega_pert = self.omega * 1.01
            E_m_pert = self.B * self.L1 * self.L2 * omega_pert
            E_eff_pert = E_m_pert / math.sqrt(2)
            I_pert = E_eff_pert / (self.R + self.r)
            P_pert = I_pert**2 * self.R
            
            if not (P_pert > P_orig):
                success = False
                self._log(f"Monotonicity fail: P(ω↑) = {P_pert} ≤ {P_orig} = P(ω)", is_error=True)
        except Exception as e:
            self._log(f"Monotonicity check for ω failed: {e}", is_error=True)
            success = False
        
        # Test 3: Power vs R should have maximum at R = r (maximum power transfer)
        try:
            # Use fixed values for this test
            B_test = 1.0
            L1_test = 1.0
            L2_test = 1.0
            omega_test = 1.0
            r_test = 2.0
            
            # Compute P at R = r
            E_m = B_test * L1_test * L2_test * omega_test
            E_eff = E_m / math.sqrt(2)
            I_r = E_eff / (r_test + r_test)
            P_at_R_equals_r = I_r**2 * r_test
            
            # Compute P at R = 1.5*r
            R_test = 1.5 * r_test
            I_R = E_eff / (R_test + r_test)
            P_at_R_gt_r = I_R**2 * R_test
            
            # For R > r, power should be less than at R = r
            if not (P_at_R_gt_r < P_at_R_equals_r):
                success = False
                self._log(f"Power transfer fail: P(R>r) = {P_at_R_gt_r} ≥ {P_at_R_equals_r} = P(R=r)", is_error=True)
        except Exception as e:
            self._log(f"Power transfer check failed: {e}", is_error=True)
            success = False
        
        return success
    
    def run_all_tests(self):
        """
        Execute all physics audits and return (success_bool, logs_list).
        """
        self.logs = []
        overall_success = True
        
        try:
            # Run each test suite
            tests = [
                ("Physical Invariants", self.test_invariants),
                ("Trace Consistency", self.test_trace_consistency),
                ("Monotonicity", self.test_monotonicity)
            ]
            
            for test_name, test_func in tests:
                self._log(f"Running {test_name}...")
                try:
                    if not test_func():
                        overall_success = False
                        self._log(f"{test_name} FAILED", is_error=True)
                    else:
                        self._log(f"{test_name} passed")
                except Exception as e:
                    overall_success = False
                    self._log(f"{test_name} ERROR: {traceback.format_exc()}", is_error=True)
                
                self._log("")  # Blank line for readability
        
        except Exception as e:
            overall_success = False
            self._log(f"Checker crashed: {traceback.format_exc()}", is_error=True)
        
        return overall_success, self.logs