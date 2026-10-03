import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction
        self.I = getattr(solution_instance, 'I', solution_instance.results.get('I', 0.0))
        self.R = getattr(solution_instance, 'R', solution_instance.results.get('R', 0.0))
        
        # Given problem parameters
        self.t = 2.0  # seconds
        self.q = 1.0  # coulombs
        self.Q = 25.0 # joules
        
        # For symbolic calculations
        self.t_sym = sp.symbols('t')
        self.q_sym = sp.symbols('q')
        self.Q_sym = sp.symbols('Q')
        self.I_sym = sp.symbols('I')
        self.R_sym = sp.symbols('R')
        
        # Store original values for perturbation tests
        self.original_I = float(self.I)
        self.original_R = float(self.R)
        
        self.logs = []
        
    def test_invariants(self):
        """Verify basic physical laws"""
        logs = []
        
        try:
            # 1. Check current definition: I = q/t
            expected_I = self.q / self.t
            if not math.isclose(float(self.I), expected_I, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: I = {self.I} ≠ q/t = {expected_I}")
            else:
                logs.append(f"PASS: Current definition I = q/t satisfied")
                
            # 2. Check Joule's Law: Q = I²·R·t
            expected_Q = (float(self.I)**2) * float(self.R) * self.t
            if not math.isclose(self.Q, expected_Q, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Q = {self.Q} ≠ I²·R·t = {expected_Q}")
            else:
                logs.append(f"PASS: Joule's Law Q = I²·R·t satisfied")
                
            # 3. Check non-negativity
            if float(self.I) < 0:
                logs.append("WARN: Current is negative - check sign convention")
            if float(self.R) < 0:
                logs.append("FAIL: Resistance cannot be negative")
            else:
                logs.append("PASS: All physical quantities non-negative")
                
        except Exception as e:
            logs.append(f"ERROR in invariants: {traceback.format_exc()}")
            
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate calculations with final answer"""
        logs = []
        
        try:
            # Step 1 consistency: I = 0.5 A
            step1_I = 0.5
            if not math.isclose(float(self.I), step1_I, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Final I = {self.I} ≠ step1 I = {step1_I}")
            else:
                logs.append(f"PASS: Step 1 current calculation consistent")
                
            # Step 2 consistency: R = 50 Ω
            step2_R = 50.0
            if not math.isclose(float(self.R), step2_R, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Final R = {self.R} ≠ step2 R = {step2_R}")
            else:
                logs.append(f"PASS: Step 2 resistance calculation consistent")
                
            # Verify step-by-step derivation
            # Step 1: I = q/t = 1/2 = 0.5
            derived_I = self.q / self.t
            if not math.isclose(derived_I, step1_I, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"WARN: Step 1 derivation mismatch: q/t = {derived_I} ≠ {step1_I}")
                
            # Step 2: R = Q/(I²·t) = 25/(0.5²·2) = 50
            derived_R = self.Q / ((derived_I**2) * self.t)
            if not math.isclose(derived_R, step2_R, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"WARN: Step 2 derivation mismatch: Q/(I²·t) = {derived_R} ≠ {step2_R}")
            else:
                logs.append("PASS: Step-by-step derivation consistent")
                
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {traceback.format_exc()}")
            
        return logs
    
    def test_monotonicity(self):
        """Perturb inputs and check expected trends"""
        logs = []
        
        try:
            # Test 1: Increase charge q by 10%
            perturbed_q = self.q * 1.1
            
            # Calculate new I from perturbed q (using formula I = q/t)
            new_I = perturbed_q / self.t
            
            # Calculate new R from perturbed I (using formula R = Q/(I²·t))
            new_R = self.Q / (new_I**2 * self.t)
            
            # Check trends from KB:
            # 1. If delta_q increases while t constant, I should increase
            if new_I <= self.original_I:
                logs.append(f"FAIL: I should increase when q increases, but {new_I} ≤ {self.original_I}")
            else:
                logs.append(f"PASS: I increases with increased q ({new_I:.3f} > {self.original_I:.3f})")
                
            # 2. When I increases, with Q and t fixed, R should decrease (from R = Q/(I²·t))
            if new_R >= self.original_R:
                logs.append(f"FAIL: R should decrease when I increases, but {new_R} ≥ {self.original_R}")
            else:
                logs.append(f"PASS: R decreases with increased I ({new_R:.3f} < {self.original_R:.3f})")
                
            # Test 2: Decrease time t by 10%
            perturbed_t = self.t * 0.9
            
            # Calculate new I from original q and perturbed t
            new_I2 = self.q / perturbed_t
            
            # Calculate new R
            new_R2 = self.Q / (new_I2**2 * perturbed_t)
            
            # Check trends from KB:
            # 1. If t decreases while q constant, I should increase
            if new_I2 <= self.original_I:
                logs.append(f"FAIL: I should increase when t decreases, but {new_I2} ≤ {self.original_I}")
            else:
                logs.append(f"PASS: I increases with decreased t ({new_I2:.3f} > {self.original_I:.3f})")
                
            # R = Q/(I²·t): If I increases (quadratic in denominator) and t decreases (linear in denominator),
            # R should decrease more significantly
            expected_decrease = (self.original_R - new_R2) / self.original_R
            if expected_decrease <= 0:
                logs.append(f"FAIL: R should decrease when t decreases, but change is {expected_decrease:.1%}")
            else:
                logs.append(f"PASS: R decreases with decreased t ({new_R2:.3f} < {self.original_R:.3f})")
                
        except Exception as e:
            logs.append(f"ERROR in monotonicity: {traceback.format_exc()}")
            
        return logs
    
    def run_all_tests(self):
        """Run all checks and return success status with logs"""
        all_logs = []
        success = True
        
        # Run each test
        for test_name, test_func in [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity Tests", self.test_monotonicity)
        ]:
            all_logs.append(f"\n{'='*50}")
            all_logs.append(f"Running {test_name}")
            all_logs.append('='*50)
            
            test_logs = test_func()
            all_logs.extend(test_logs)
            
            # Check for FAIL messages
            if any("FAIL:" in log for log in test_logs):
                success = False
                
        # Final summary
        all_logs.append(f"\n{'='*50}")
        all_logs.append("SUMMARY")
        all_logs.append('='*50)
        all_logs.append(f"Overall Status: {'PASS' if success else 'FAIL'}")
        all_logs.append(f"Extracted Results: I = {self.I} A, R = {self.R} Ω")
        all_logs.append(f"Expected: I = 0.5 A, R = 50 Ω")
        
        return success, all_logs

# Example usage (commented out):
# if __name__ == "__main__":
#     class Solution:
#         def __init__(self):
#             self.results = {'I': 0.5, 'R': 50.0}
#     
#     solution = Solution()
#     checker = PhysicsChecker(solution)
#     success, logs = checker.run_all_tests()
#     for log in logs:
#         print(log)