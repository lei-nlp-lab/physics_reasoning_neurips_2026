import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all required inputs and results from the solution instance.
        """
        # Safely access attributes with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract key results
        self.mu = getattr(solution_instance, 'mu', self.results.get('mu', None))
        self.Fm = getattr(solution_instance, 'Fm', self.results.get('Fm', None))
        self.F = getattr(solution_instance, 'F', self.results.get('F', None))
        
        # Additional data from steps analysis (for validation)
        self.G = 5.0  # weight of each book in Newtons
        
        # Logs for test results
        self.logs = []
        
    def _log(self, message, success):
        """Helper to log test results."""
        if success is None:
            self.logs.append(f"INFO: {message}")
        else:
            self.logs.append(f"{'PASS' if success else 'FAIL'}: {message}")
        
    def test_invariants(self):
        """Check physical invariants: force equilibrium and friction formula."""
        try:
            # Convert mu to float if it's not already
            mu_val = float(self.mu) if self.mu is not None else None
            if mu_val is None:
                self._log("μ not found in results.", False)
                return False
            
            # 1. Check that the empirical formula matches the physical model: F = 5μ(2m-1)
            # From ground truth: F = 3*m - 1.5
            # Physical model: F = 5 * μ * (2*m - 1)
            # These should be equivalent, so: 3*m - 1.5 = 5μ(2m - 1)
            # For this to hold for all m, coefficients must match:
            # Coefficient of m: 3 = 10μ => μ = 0.3
            # Constant term: -1.5 = -5μ => μ = 0.3
            # So if μ = 0.3, the formulas are equivalent
            
            if math.isclose(mu_val, 0.3, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"μ = {mu_val} makes empirical formula F=3m-1.5 equivalent to physical model F=5μ(2m-1)", True)
            else:
                self._log(f"μ = {mu_val} does not make formulas equivalent", False)
                return False
            
            # 2. Check force equilibrium for m=2
            # For m=2, F should be 4.5 from empirical formula
            expected_F_m2 = 3*2 - 1.5  # = 4.5
            # Also from physical model with μ=0.3: 5*0.3*(2*2-1) = 1.5*3 = 4.5
            if math.isclose(expected_F_m2, 5 * mu_val * (2*2 - 1), rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Force equilibrium for m=2 holds: F={expected_F_m2} = 5μ(2m-1) = {5 * mu_val * (2*2 - 1)}", True)
            else:
                self._log(f"Force equilibrium for m=2 fails: {expected_F_m2} vs {5 * mu_val * (2*2 - 1)}", False)
                return False
                
            return True
        except Exception as e:
            self._log(f"Error in test_invariants: {e}\n{traceback.format_exc()}", False)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            mu_val = float(self.mu) if self.mu is not None else None
            F_val = float(self.F) if self.F is not None else None
            Fm_val = float(self.Fm) if self.Fm is not None else None
            
            # 1. Check that μ = 0.3
            if mu_val is not None and math.isclose(mu_val, 0.3, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"μ = {mu_val} matches expected 0.3", True)
            else:
                self._log(f"μ = {mu_val} does not match expected 0.3", False)
                return False
            
            # 2. Check that F for m=32 is 94.5
            if F_val is not None and math.isclose(F_val, 94.5, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"F(m=32) = {F_val} matches expected 94.5 N", True)
            else:
                self._log(f"F(m=32) = {F_val} does not match expected 94.5 N", False)
                return False
            
            # 3. Check that F for m=2 is 4.5
            if Fm_val is not None and math.isclose(Fm_val, 4.5, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"F(m=2) = {Fm_val} matches expected 4.5 N", True)
            else:
                self._log(f"F(m=2) = {Fm_val} does not match expected 4.5 N", False)
                return False
            
            # 4. Verify empirical formula F = 3*m - 1.5
            # Check for m=32: 3*32 - 1.5 = 96 - 1.5 = 94.5
            if math.isclose(3*32 - 1.5, 94.5, rel_tol=1e-2, abs_tol=1e-2):
                self._log("Empirical formula F=3m-1.5 gives correct F for m=32.", True)
            else:
                self._log("Empirical formula F=3m-1.5 gives incorrect F for m=32.", False)
                return False
            
            return True
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {e}\n{traceback.format_exc()}", False)
            return False
    
    def test_monotonicity(self):
        """
        Perturb the number of parts m and verify that the required force F increases.
        Uses the empirical formula or physical model.
        """
        try:
            mu_val = float(self.mu) if self.mu is not None else None
            if mu_val is None:
                self._log("μ not available for monotonicity test.", False)
                return False
            
            # Use empirical formula: F = 3*m - 1.5
            def F_empirical(m_val):
                return 3 * m_val - 1.5
            
            # Test with m = 30 and 31 (should increase)
            F30 = F_empirical(30)
            F31 = F_empirical(31)
            if F31 > F30:
                self._log(f"Monotonicity holds: F(31)={F31:.2f} > F(30)={F30:.2f}", True)
            else:
                self._log(f"Monotonicity fails: F(31)={F31:.2f} <= F(30)={F30:.2f}", False)
                return False
            
            # Also test with m=32 and 33
            F32 = F_empirical(32)
            F33 = F_empirical(33)
            if F33 > F32:
                self._log(f"Monotonicity holds: F(33)={F33:.2f} > F(32)={F32:.2f}", True)
            else:
                self._log(f"Monotonicity fails: F(33)={F33:.2f} <= F(32)={F32:.2f}", False)
                return False
            
            return True
        except Exception as e:
            self._log(f"Error in test_monotonicity: {e}\n{traceback.format_exc()}", False)
            return False
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = []  # Reset logs
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        all_passed = True
        for name, test_func in tests:
            self._log(f"--- {name} Test ---", None)
            if not test_func():
                all_passed = False
        
        return all_passed, self.logs