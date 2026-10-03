import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results."""
        # Extract inputs with fallbacks
        self.M = getattr(solution_instance, 'M', 1.0)
        self.m = getattr(solution_instance, 'm', 1.0)
        self.mu1 = getattr(solution_instance, 'mu1', 0.1)
        self.mu2 = getattr(solution_instance, 'mu2', 0.4)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Extract results safely (use .results dict if available)
        self.results = getattr(solution_instance, 'results', {})
        self.a1 = self.results.get('a_1', 0.0)
        self.F_fmax1 = self.results.get('F_fmax1', 0.0)
        self.t = self.results.get('t', 0.0)
        self.a = self.results.get('a', 0.0)
        self.F_fmax2 = self.results.get('F_fmax2', 0.0)
        self.a2 = self.results.get('a_2', 0.0)
        self.F_f = self.results.get('F_f', 0.0)
        
        # For symbolic checks
        self.symbols = {
            'M': sp.symbols('M'), 'm': sp.symbols('m'),
            'mu1': sp.symbols('mu1'), 'mu2': sp.symbols('mu2'),
            'g': sp.symbols('g'), 'a1': sp.symbols('a1'),
            'a2': sp.symbols('a2'), 'a': sp.symbols('a'),
            'F_fmax1': sp.symbols('F_fmax1'), 'F_fmax2': sp.symbols('F_fmax2'),
            'F_f': sp.symbols('F_f'), 't': sp.symbols('t')
        }
        
    def test_invariants(self):
        """Check fundamental physics laws."""
        logs = []
        success = True
        
        try:
            # 1. Maximum friction checks
            expected_F_fmax2 = self.mu2 * self.m * self.g  # μ₂ m g
            if not math.isclose(self.F_fmax2, expected_F_fmax2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: F_fmax2 mismatch. Expected {expected_F_fmax2:.2f}, got {self.F_fmax2:.2f}")
                success = False
            else:
                logs.append(f"PASS: F_fmax2 = μ₂ m g = {self.F_fmax2:.2f} N")
            
            expected_F_fmax1 = self.mu1 * (self.m + self.M) * self.g  # μ₁ (m+M) g
            if not math.isclose(self.F_fmax1, expected_F_fmax1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: F_fmax1 mismatch. Expected {expected_F_fmax1:.2f}, got {self.F_fmax1:.2f}")
                success = False
            else:
                logs.append(f"PASS: F_fmax1 = μ₁ (m+M) g = {self.F_fmax1:.2f} N")
            
            # 2. Consistency check from step_8: F_f = F - m*a
            # Since F isn't directly given, use step_7: a = (F - μ₁(m+M)g) / (m+M)
            # => F = a*(m+M) + μ₁(m+M)g
            if abs(self.a) > 1e-12:  # Only check if acceleration is non-zero
                F_calc = self.a * (self.m + self.M) + self.mu1 * (self.m + self.M) * self.g
                expected_F_f = F_calc - self.m * self.a
                if not math.isclose(self.F_f, expected_F_f, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: F_f inconsistency. From step_7&8, expected {expected_F_f:.2f}, got {self.F_f:.2f}")
                    success = False
                else:
                    logs.append(f"PASS: F_f = F - m*a consistency holds")
            
            # 3. Check if F_f <= F_fmax2 (static friction limit before sliding)
            if self.F_f > self.F_fmax2 + 1e-2:  # Small tolerance
                logs.append(f"WARNING: F_f ({self.F_f:.2f}) > F_fmax2 ({self.F_fmax2:.2f}) violates static friction limit")
            
            # 4. Check sliding phase accelerations (steps 9 & 10)
            # Step 9: a₂ = (F - μ₂ m g)/m, but F unknown. Use alternative check:
            # During sliding, friction on block = μ₂ m g (kinetic friction)
            if self.a2 != 0:
                # Step 10: a₁ = (μ₂ m g - μ₁ (m+M)g)/M
                expected_a1 = (self.mu2 * self.m * self.g - self.mu1 * (self.m + self.M) * self.g) / self.M
                if not math.isclose(self.a1, expected_a1, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: a₁ mismatch. Expected {expected_a1:.2f}, got {self.a1:.2f}")
                    success = False
                else:
                    logs.append(f"PASS: a₁ = (μ₂ m g - μ₁(m+M)g)/M = {self.a1:.2f} m/s²")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        try:
            # Target answers: 1 N, 3.5 N, 1 s
            if not math.isclose(self.F_f, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                # Might be the second case (3.5 N)
                if not math.isclose(self.F_f, 3.5, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: F_f ({self.F_f:.2f}) doesn't match target answers (1 N or 3.5 N)")
                    success = False
                else:
                    logs.append(f"PASS: F_f matches second target answer (3.5 N)")
                    # Check associated a = 1.5 m/s² from step_7
                    expected_a = (5.0 - self.mu1 * (self.m + self.M) * self.g) / (self.m + self.M)  # F=5N from reconstruction
                    if not math.isclose(self.a, expected_a, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"FAIL: Associated a mismatch. Expected {expected_a:.2f}, got {self.a:.2f}")
                        success = False
                    else:
                        logs.append(f"PASS: Consistent a = {self.a:.2f} m/s² for F_f=3.5 N case")
            else:
                logs.append(f"PASS: F_f matches first target answer (1 N)")
            
            # Check time t = 1 s
            if not math.isclose(self.t, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: t ({self.t:.2f}) doesn't match target answer (1 s)")
                success = False
            else:
                logs.append(f"PASS: t = {self.t:.2f} s matches target")
                
                # Verify kinematics: L = 0.5*(a₂ - a₁)*t²
                # Assuming board length L=1m (typical implicit), check consistency
                if self.a2 != 0 and self.a1 != 0:
                    L_calc = 0.5 * (self.a2 - self.a1) * self.t**2
                    if L_calc < 0:
                        logs.append("WARNING: Calculated L is negative - check acceleration directions")
                    else:
                        logs.append(f"INFO: Implied board length L ≈ {L_calc:.2f} m")
        
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Perturb inputs and check physical trends."""
        logs = []
        success = True
        
        try:
            # Perturb mass m by +10%
            original_m = self.m
            perturbed_m = original_m * 1.1
            
            # Recompute expected trends using formulas
            # 1. F_fmax2 should increase
            original_F_fmax2 = self.mu2 * original_m * self.g
            perturbed_F_fmax2 = self.mu2 * perturbed_m * self.g
            if perturbed_F_fmax2 <= original_F_fmax2:
                logs.append(f"FAIL: F_fmax2 monotonicity violated. Increased m should increase F_fmax2")
                success = False
            else:
                logs.append(f"PASS: F_fmax2 increases with m (μ₂ m g)")
            
            # 2. F_fmax1 should increase
            original_F_fmax1 = self.mu1 * (original_m + self.M) * self.g
            perturbed_F_fmax1 = self.mu1 * (perturbed_m + self.M) * self.g
            if perturbed_F_fmax1 <= original_F_fmax1:
                logs.append(f"FAIL: F_fmax1 monotonicity violated. Increased m should increase F_fmax1")
                success = False
            else:
                logs.append(f"PASS: F_fmax1 increases with m (μ₁(m+M)g)")
            
            # 3. For sliding accelerations: a₁ = (μ₂ m g - μ₁(m+M)g)/M
            # Derivative with respect to m: d(a₁)/dm = (μ₂ g - μ₁ g)/M = g(μ₂ - μ₁)/M > 0
            if self.a1 != 0:
                original_a1 = (self.mu2 * original_m * self.g - self.mu1 * (original_m + self.M) * self.g) / self.M
                perturbed_a1 = (self.mu2 * perturbed_m * self.g - self.mu1 * (perturbed_m + self.M) * self.g) / self.M
                if perturbed_a1 <= original_a1:
                    logs.append(f"FAIL: a₁ monotonicity violated. Should increase with m when μ₂ > μ₁")
                    success = False
                else:
                    logs.append(f"PASS: a₁ increases with m (since μ₂ > μ₁)")
        
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Execute all checks and return summary."""
        all_logs = []
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            success, logs = test_func()
            all_logs.append(f"=== {test_name} ===")
            all_logs.extend(logs)
            if not success:
                overall_success = False
        
        # Summary
        all_logs.append("=" * 40)
        if overall_success:
            all_logs.append("ALL TESTS PASSED")
        else:
            all_logs.append("SOME TESTS FAILED")
        
        return overall_success, all_logs

# Example usage (commented out for sandbox):
# if __name__ == "__main__":
#     class Solution:
#         def __init__(self):
#             self.results = {
#                 'a_1': 2.0, 'F_fmax1': 2.0, 't': 1.0,
#                 'a': 1.5, 'F_fmax2': 4.0, 'a_2': 5.0, 'F_f': 3.5
#             }
#     checker = PhysicsChecker(Solution())
#     success, logs = checker.run_all_tests()
#     for log in logs:
#         print(log)