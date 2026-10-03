import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs with defaults
        self.m = getattr(solution_instance, 'm', 5.0)
        self.F = getattr(solution_instance, 'F', 25.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Safely extract results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Safely extract each result with fallback
        self.F_N1 = self.results.get('F_N1', None)
        self.F_N2 = self.results.get('F_N2', None)
        self.F_x = self.results.get('F_x', None)
        self.F_y = self.results.get('F_y', None)
        self.mu = self.results.get('mu', None)
        self.F_m = self.results.get('F_m', None)
        self.F_2 = self.results.get('F_2', None)
        
        # Compute theta from components if available
        self.theta = None
        if self.F_x is not None and self.F_y is not None and self.F is not None:
            if self.F != 0:
                self.theta = math.atan2(self.F_y, self.F_x)
            else:
                self.theta = 0.0
        
        # Tolerance for numerical comparisons (1-2% relative)
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
    def is_close(self, a, b):
        """Safe numerical comparison with tolerance."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
    
    def test_invariants(self):
        """Check fundamental physics laws."""
        logs = []
        success = True
        
        try:
            # 1. First case: constant speed => F = friction = mu * m*g
            if self.mu is not None and self.F_N1 is not None:
                expected_mu = self.F / (self.m * self.g)
                if not self.is_close(self.mu, expected_mu):
                    logs.append(f"Invariant violation: mu should be {expected_mu:.3f}, got {self.mu}")
                    success = False
                # Check F_N1 = m*g
                expected_F_N1 = self.m * self.g
                if not self.is_close(self.F_N1, expected_F_N1):
                    logs.append(f"Invariant violation: F_N1 should be {expected_F_N1:.3f}, got {self.F_N1}")
                    success = False
            else:
                logs.append("Missing data for first case invariant check")
                success = False
            
            # 2. Second case: force decomposition and equilibrium
            if self.F_x is not None and self.F_y is not None and self.F is not None:
                # Check Pythagorean theorem for force components
                F_mag_sq = self.F_x**2 + self.F_y**2
                if not self.is_close(F_mag_sq, self.F**2):
                    logs.append(f"Invariant violation: F_x^2 + F_y^2 = {F_mag_sq:.3f}, F^2 = {self.F**2:.3f}")
                    success = False
            
            if self.F_N2 is not None and self.F_y is not None:
                # Check normal force: F_N2 = m*g + F_y
                expected_F_N2 = self.m * self.g + self.F_y
                if not self.is_close(self.F_N2, expected_F_N2):
                    logs.append(f"Invariant violation: F_N2 should be {expected_F_N2:.3f}, got {self.F_N2}")
                    success = False
            
            if self.F_m is not None and self.mu is not None and self.F_N2 is not None:
                # Check maximum static friction: F_m = mu * F_N2
                expected_F_m = self.mu * self.F_N2
                if not self.is_close(self.F_m, expected_F_m):
                    logs.append(f"Invariant violation: F_m should be {expected_F_m:.3f}, got {self.F_m}")
                    success = False
            
            if self.F_2 is not None and self.F_x is not None:
                # Static friction equals horizontal component if at rest
                if not self.is_close(self.F_2, self.F_x):
                    logs.append(f"Invariant violation: F_2 should equal F_x ({self.F_x:.3f}), got {self.F_2}")
                    success = False
                # Check that F_x <= F_m (static condition)
                if self.F_m is not None and self.F_x > self.F_m and not self.is_close(self.F_x, self.F_m):
                    logs.append(f"Invariant violation: F_x ({self.F_x}) > F_m ({self.F_m}) but block is at rest")
                    success = False
            
        except Exception as e:
            logs.append(f"Error in test_invariants: {traceback.format_exc()}")
            success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        """Check consistency between intermediate and final results."""
        logs = []
        success = True
        
        try:
            # Check step sequence from the analysis
            # Step 1: F_N1 = m*g
            if self.F_N1 is not None:
                expected = self.m * self.g
                if not self.is_close(self.F_N1, expected):
                    logs.append(f"Trace inconsistency: Step 1 F_N1 = {self.F_N1}, expected {expected}")
                    success = False
            
            # Steps 2-4: mu = F / (m*g)
            if self.mu is not None:
                expected = self.F / (self.m * self.g)
                if not self.is_close(self.mu, expected):
                    logs.append(f"Trace inconsistency: mu = {self.mu}, expected {expected}")
                    success = False
            
            # Steps 5-6: Force decomposition (already checked in invariants)
            if self.F_x is not None and self.F_y is not None and self.F is not None:
                if not self.is_close(self.F_x**2 + self.F_y**2, self.F**2):
                    logs.append("Trace inconsistency: Force decomposition fails Pythagorean theorem")
                    success = False
            
            # Step 7: F_N2 = m*g + F_y
            if self.F_N2 is not None and self.F_y is not None:
                expected = self.m * self.g + self.F_y
                if not self.is_close(self.F_N2, expected):
                    logs.append(f"Trace inconsistency: Step 7 F_N2 = {self.F_N2}, expected {expected}")
                    success = False
            
            # Steps 8-9: F_m = mu * F_N2
            if self.F_m is not None and self.mu is not None and self.F_N2 is not None:
                expected = self.mu * self.F_N2
                if not self.is_close(self.F_m, expected):
                    logs.append(f"Trace inconsistency: Step 9 F_m = {self.F_m}, expected {expected}")
                    success = False
            
            # Step 11: F_2 = F_x
            if self.F_2 is not None and self.F_x is not None:
                if not self.is_close(self.F_2, self.F_x):
                    logs.append(f"Trace inconsistency: Step 11 F_2 = {self.F_2}, expected F_x = {self.F_x}")
                    success = False
            
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {traceback.format_exc()}")
            success = False
        
        return success, logs
    
    def test_monotonicity(self):
        """Perturb input force F and check expected trends."""
        logs = []
        success = True
        
        try:
            # We'll perturb F by +10% and check that F_2 increases
            if None in (self.F, self.m, self.g, self.F_x, self.F_y, self.mu, self.F_2):
                logs.append("Missing data for monotonicity test")
                return False, logs
            
            # Original values
            F_orig = self.F
            m_orig = self.m
            g_orig = self.g
            F_x_orig = self.F_x
            F_y_orig = self.F_y
            mu_orig = self.mu
            F_2_orig = self.F_2
            
            # Perturb F upward by 10%
            F_new = F_orig * 1.1
            
            # Recompute using the same physical formulas as the solution
            # Step 1: mu_new from first case (since surface same, but actually mu changes because F changes)
            # Note: In the problem, mu is derived from the first case: mu = F/(m*g)
            mu_new = F_new / (m_orig * g_orig)
            
            # Force decomposition with same angle (using ratios)
            if F_orig != 0:
                ratio_x = F_x_orig / F_orig
                ratio_y = F_y_orig / F_orig
            else:
                ratio_x = 0
                ratio_y = 0
            F_x_new = F_new * ratio_x
            F_y_new = F_new * ratio_y
            
            # Normal force in second case
            F_N2_new = m_orig * g_orig + F_y_new
            
            # Maximum static friction
            F_m_new = mu_new * F_N2_new
            
            # Determine if block is at rest (should be true for this perturbation)
            if F_x_new < F_m_new or self.is_close(F_x_new, F_m_new):
                F_2_new = F_x_new
            else:
                # Would be sliding, but for our perturbation we expect static
                F_2_new = None
                logs.append("Monotonicity warning: Perturbation caused sliding, not static")
            
            # Check trend: F increased, so F_2 should increase
            if F_2_new is not None and F_2_orig is not None:
                if F_2_new <= F_2_orig and not self.is_close(F_2_new, F_2_orig):
                    logs.append(f"Monotonicity violation: F increased from {F_orig} to {F_new}, but F_2 decreased from {F_2_orig} to {F_2_new}")
                    success = False
                else:
                    logs.append(f"Monotonicity passed: F_2 increased from {F_2_orig} to {F_2_new} when F increased by 10%")
            else:
                logs.append("Could not compute F_2 for perturbed case")
                success = False
            
            # Optional: Also test negative perturbation (decrease F by 10%)
            F_new2 = F_orig * 0.9
            mu_new2 = F_new2 / (m_orig * g_orig)
            F_x_new2 = F_new2 * ratio_x
            F_y_new2 = F_new2 * ratio_y
            F_N2_new2 = m_orig * g_orig + F_y_new2
            F_m_new2 = mu_new2 * F_N2_new2
            
            if F_x_new2 < F_m_new2 or self.is_close(F_x_new2, F_m_new2):
                F_2_new2 = F_x_new2
            else:
                F_2_new2 = None
            
            if F_2_new2 is not None and F_2_orig is not None:
                if F_2_new2 >= F_2_orig and not self.is_close(F_2_new2, F_2_orig):
                    logs.append(f"Monotonicity violation: F decreased from {F_orig} to {F_new2}, but F_2 increased from {F_2_orig} to {F_2_new2}")
                    success = False
                else:
                    logs.append(f"Monotonicity passed: F_2 decreased from {F_2_orig} to {F_2_new2} when F decreased by 10%")
            
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {traceback.format_exc()}")
            success = False
        
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        all_logs = []
        overall_success = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            success, logs = test_method()
            all_logs.append(f"=== {test_name} Test ===")
            if logs:
                all_logs.extend(logs)
            all_logs.append(f"Result: {'PASS' if success else 'FAIL'}")
            overall_success = overall_success and success
        
        return overall_success, all_logs

# Example usage (commented out):
# checker = PhysicsChecker(solution_instance)
# success, logs = checker.run_all_tests()
# for log in logs:
#     print(log)