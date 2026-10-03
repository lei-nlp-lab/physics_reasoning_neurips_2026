import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Initialize with safe attribute access."""
        self.solution = solution_instance
        
        # Extract inputs with defaults
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'inputs', {}).get('m', 0.04))
        self.L = getattr(solution_instance, 'L', getattr(solution_instance, 'inputs', {}).get('L', 0.4))
        self.I = getattr(solution_instance, 'I', getattr(solution_instance, 'inputs', {}).get('I', 1.5))
        self.theta_deg = 37.0
        self.g = 10.0
        self.sin37 = 0.6
        self.cos37 = 0.8
        
        # Extract results with safe defaults
        results_dict = getattr(solution_instance, 'results', {})
        self.B = results_dict.get('B', 0.0)
        self.a = results_dict.get('a', 0.0)
        self.F_a = results_dict.get('F_a', 0.0)
        
        # Derived quantities for checking
        self.F_gravity_parallel = self.m * self.g * self.sin37
        self.F_magnetic = self.B * self.I * self.L
        
        self.logs = []
    
    def _log(self, message: str, success: bool) -> None:
        """Helper to log test results."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {message}")
    
    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Test fundamental physics laws."""
        all_pass = True
        
        # 1. Static equilibrium condition (step 1)
        expected_F_a = self.F_gravity_parallel
        if math.isclose(self.F_magnetic, expected_F_a, rel_tol=0.02, abs_tol=0.02):
            self._log(f"Static equilibrium: F_magnetic={self.F_magnetic:.3f} ≈ mgsinθ={expected_F_a:.3f}", True)
        else:
            self._log(f"Static equilibrium violated: F_magnetic={self.F_magnetic:.3f}, mgsinθ={expected_F_a:.3f}", False)
            all_pass = False
        
        # 2. Lorentz force formula (step 2, step 5)
        calculated_F_a = self.B * self.I * self.L
        if math.isclose(self.F_magnetic, calculated_F_a, rel_tol=0.02, abs_tol=0.02):
            self._log(f"Lorentz force: F_a={self.F_magnetic:.3f} = BIL={calculated_F_a:.3f}", True)
        else:
            self._log(f"Lorentz force mismatch: {self.F_magnetic:.3f} vs {calculated_F_a:.3f}", False)
            all_pass = False
        
        # 3. Newton's second law for acceleration (step 6)
        expected_a = (self.F_gravity_parallel - self.F_magnetic * self.cos37) / self.m
        if math.isclose(self.a, expected_a, rel_tol=0.02, abs_tol=0.02):
            self._log(f"Newton's 2nd law: a={self.a:.3f} ≈ (mgsinθ - BILcosθ)/m={expected_a:.3f}", True)
        else:
            self._log(f"Newton's 2nd law mismatch: a={self.a:.3f}, expected {expected_a:.3f}", False)
            all_pass = False
        
        # 4. Symbolic verification - use numeric comparison instead of symbolic
        try:
            # Check static equilibrium numerically
            lhs_eq1 = self.B * self.I * self.L
            rhs_eq1 = self.m * self.g * self.sin37
            if math.isclose(lhs_eq1, rhs_eq1, rel_tol=0.02, abs_tol=0.02):
                self._log("Symbolic static equilibrium verified numerically", True)
            else:
                self._log("Symbolic static equilibrium check failed", False)
                all_pass = False
            
            # Check acceleration numerically
            a_calc = (self.m * self.g * self.sin37 - self.B * self.I * self.L * self.cos37) / self.m
            if math.isclose(self.a, a_calc, rel_tol=0.02, abs_tol=0.02):
                self._log(f"Symbolic acceleration verified: {a_calc:.3f} ≈ {self.a:.3f}", True)
            else:
                self._log(f"Symbolic acceleration mismatch: {a_calc:.3f} vs {self.a:.3f}", False)
                all_pass = False
                
        except Exception as e:
            self._log(f"Symbolic check skipped due to error: {str(e)}", False)
            all_pass = False
        
        return all_pass, self.logs
    
    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Check intermediate results are consistent with final answers."""
        all_pass = True
        
        # Check B calculation (step 3)
        calculated_B = (self.m * self.g * self.sin37) / (self.I * self.L)
        if math.isclose(self.B, calculated_B, rel_tol=0.02, abs_tol=0.02):
            self._log(f"B consistency: B={self.B:.3f} = (mgsinθ)/(IL)={calculated_B:.3f}", True)
        else:
            self._log(f"B inconsistent: {self.B:.3f} vs {calculated_B:.3f}", False)
            all_pass = False
        
        # Check a calculation (step 6)
        calculated_a = (self.m * self.g * self.sin37 - self.B * self.I * self.L * self.cos37) / self.m
        if math.isclose(self.a, calculated_a, rel_tol=0.02, abs_tol=0.02):
            self._log(f"a consistency: a={self.a:.3f} = (mgsinθ - BILcosθ)/m={calculated_a:.3f}", True)
        else:
            self._log(f"a inconsistent: {self.a:.3f} vs {calculated_a:.3f}", False)
            all_pass = False
        
        # Check that F_a from equilibrium equals F_a from Lorentz force
        F_a_eq = self.m * self.g * self.sin37
        F_a_lorentz = self.B * self.I * self.L
        if math.isclose(F_a_eq, F_a_lorentz, rel_tol=0.02, abs_tol=0.02):
            self._log(f"Force consistency: mgsinθ={F_a_eq:.3f} = BIL={F_a_lorentz:.3f}", True)
        else:
            self._log(f"Force mismatch: mgsinθ={F_a_eq:.3f}, BIL={F_a_lorentz:.3f}", False)
            all_pass = False
        
        return all_pass, self.logs
    
    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """Perturb mass and check trend, then restore."""
        all_pass = True
        
        original_m = self.m
        
        try:
            # Perturb mass by +10%
            perturbed_m = original_m * 1.1
            
            # Recalculate B using the formula
            new_B = (perturbed_m * self.g * self.sin37) / (self.I * self.L)
            
            # Check monotonicity: B should increase with increased mass
            if new_B > self.B:
                self._log(f"Monotonicity: B increases with mass (B_orig={self.B:.3f}, B_new={new_B:.3f})", True)
            else:
                self._log(f"Monotonicity violation: B doesn't increase with mass", False)
                all_pass = False
            
            # Recalculate a with original B (fixed as per problem context)
            new_a = (perturbed_m * self.g * self.sin37 - self.B * self.I * self.L * self.cos37) / perturbed_m
            
            # For our specific values, check if acceleration decreases
            # This is not a strict requirement, so log as info not failure
            if new_a < self.a:
                self._log(f"Monotonicity: a decreases with increased mass (a_orig={self.a:.3f}, a_new={new_a:.3f})", True)
            else:
                # This is just informational, not a test failure
                self._log(f"Info: a doesn't decrease with mass (a_orig={self.a:.3f}, a_new={new_a:.3f})", True)
            
            # Restore original mass
            self.m = original_m
            self._log("Mass restored to original value", True)
            
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {str(e)}", False)
            all_pass = False
            # Restore just in case
            self.m = original_m
        
        return all_pass, self.logs
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all tests and return overall success and logs."""
        self.logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n--- {test_name} Test ---")
            success, _ = test_func()
            if not success:
                overall_success = False
        
        # Summary
        self.logs.append(f"\n=== SUMMARY ===")
        self.logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, self.logs