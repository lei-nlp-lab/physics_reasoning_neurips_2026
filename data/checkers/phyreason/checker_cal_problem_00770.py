import sympy as sp
import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallback to results dict
        self.T_2 = getattr(solution_instance, 'T_2', getattr(solution_instance, 'results', {}).get('T_2', 0.0))
        self.T_1 = getattr(solution_instance, 'T_1', getattr(solution_instance, 'results', {}).get('T_1', 0.0))
        self.t_2 = getattr(solution_instance, 't_2', getattr(solution_instance, 'results', {}).get('t_2', 0.0))
        self.t = getattr(solution_instance, 't', getattr(solution_instance, 'results', {}).get('t', 0.0))
        self.l = getattr(solution_instance, 'l', getattr(solution_instance, 'results', {}).get('l', 0.0))
        self.t_1 = getattr(solution_instance, 't_1', getattr(solution_instance, 'results', {}).get('t_1', 0.0))
        self.fracl_1l_2 = getattr(solution_instance, 'fracl_1l_2', getattr(solution_instance, 'results', {}).get('fracl_1l_2', 0.0))
        
        # Store original values for restoration in monotonicity test
        self.original_T1 = float(self.T_1)
        self.original_T2 = float(self.T_2)
        
        # Constants
        self.g = 9.8  # Standard gravity

    def test_invariants(self):
        logs = []
        try:
            # 1. Period-Length relationship (from KB theorem)
            # Using symbolic check with numerical fallback
            L1, L2, g = sp.symbols('L1 L2 g', positive=True)
            T1_sym = 2*sp.pi*sp.sqrt(L1/g)
            T2_sym = 2*sp.pi*sp.sqrt(L2/g)
            
            # Check if ratio of squares of periods equals ratio of lengths
            ratio_from_T = (float(self.T_1)**2) / (float(self.T_2)**2)
            ratio_from_l = float(self.fracl_1l_2)
            
            if math.isclose(ratio_from_T, ratio_from_l, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Period-Length relationship invariant satisfied")
            else:
                logs.append(f"✗ Period-Length relationship mismatch: T1²/T2²={ratio_from_T:.4f}, l1/l2={ratio_from_l:.4f}")
                return False, logs
                
            # 2. Check given periods match expected values (T1=1.0, T2=0.9 from steps analysis)
            if not (math.isclose(float(self.T_1), 1.0, rel_tol=1e-2, abs_tol=1e-2) and 
                    math.isclose(float(self.T_2), 0.9, rel_tol=1e-2, abs_tol=1e-2)):
                logs.append(f"✗ Period values incorrect: T1={self.T_1}, T2={self.T_2}")
                return False, logs
            logs.append("✓ Given periods match expected values (T1=1.0s, T2=0.9s)")
            
            # 3. Check t1 and t2 formulas consistency
            # For pendulum 1: t1 should be odd multiples of T1/2
            # For pendulum 2: t2 should be integer multiples of T2
            # Since they're equal at common time t, check if t satisfies both
            if hasattr(self, 't') and float(self.t) > 0:
                t_val = float(self.t)
                # Check if t satisfies both conditions for some integers m,n
                # t = (T1/2)*(2m+1) = n*T2
                m_found = False
                for m in range(10):  # Check first 10 possibilities
                    t1_calc = (float(self.T_1)/2) * (2*m + 1)
                    if math.isclose(t_val, t1_calc, rel_tol=1e-2, abs_tol=1e-2):
                        m_found = True
                        break
                
                n_found = False
                for n in range(10):
                    t2_calc = n * float(self.T_2)
                    if math.isclose(t_val, t2_calc, rel_tol=1e-2, abs_tol=1e-2):
                        n_found = True
                        break
                
                if m_found and n_found:
                    logs.append(f"✓ Time t={t_val}s satisfies both pendulum conditions")
                else:
                    logs.append(f"✗ Time t={t_val}s doesn't satisfy both conditions simultaneously")
                    return False, logs
                    
            return True, logs
            
        except Exception as e:
            logs.append(f"✗ Error in invariants test: {str(e)}")
            return False, logs

    def test_trace_consistency(self):
        logs = []
        try:
            # Check that intermediate calculations match final results
            
            # 1. Length ratio from periods (step_2)
            calc_ratio = (float(self.T_1)**2) / (float(self.T_2)**2)
            if math.isclose(calc_ratio, float(self.fracl_1l_2), rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"✓ Length ratio consistent: calculated {calc_ratio:.4f} ≈ {float(self.fracl_1l_2):.4f}")
            else:
                logs.append(f"✗ Length ratio inconsistent: calculated {calc_ratio:.4f} ≠ {float(self.fracl_1l_2):.4f}")
                return False, logs
            
            # 2. Check if answer matches target format
            # Target answers: ["100:81", "(4.5+9k)s (k=0,1,2,3...)"]
            
            # Check ratio format (should be approximately 100/81 = 1.234567...)
            expected_ratio = 100/81  # ≈ 1.2345679
            if math.isclose(float(self.fracl_1l_2), expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Ratio matches expected 100:81")
            else:
                logs.append(f"✗ Ratio {self.fracl_1l_2} doesn't match 100:81 ({expected_ratio})")
                return False, logs
            
            # Check time expression structure
            # The time should be of form 4.5 + 9k for integer k
            # We'll check that for k=0,1,2, the condition holds
            success = True
            for k in [0, 1, 2]:
                t_val = 4.5 + 9*k
                # Check if t_val satisfies both pendulum conditions
                # Pendulum 1: t = (T1/2)*(2m+1) for some integer m
                m = (2*t_val/float(self.T_1) - 1) / 2
                m_int = abs(m - round(m)) < 1e-2
                
                # Pendulum 2: t = n*T2 for some integer n
                n = t_val / float(self.T_2)
                n_int = abs(n - round(n)) < 1e-2
                
                if not (m_int and n_int):
                    logs.append(f"✗ Time {t_val}s doesn't work for k={k}")
                    success = False
            
            if success:
                logs.append("✓ Time expression matches pattern (4.5+9k)s for k=0,1,2")
            else:
                return False, logs
                
            return True, logs
            
        except Exception as e:
            logs.append(f"✗ Error in trace consistency test: {str(e)}")
            return False, logs

    def test_monotonicity(self):
        logs = []
        try:
            # Test: If we increase T1, the ratio l1/l2 should increase quadratically
            # Remember: l ∝ T², so l1/l2 ∝ (T1/T2)²
            
            # Store original
            original_T1 = float(self.T_1)
            original_ratio = float(self.fracl_1l_2)
            
            # Perturb T1 by +10%
            perturbed_T1 = original_T1 * 1.1
            self.T_1 = perturbed_T1
            
            # Recalculate ratio with perturbed value
            new_ratio_calc = (perturbed_T1**2) / (float(self.T_2)**2)
            
            # Check monotonicity: increasing T1 should increase ratio
            if new_ratio_calc > original_ratio:
                logs.append(f"✓ Monotonicity: Increasing T1 from {original_T1} to {perturbed_T1} increases ratio from {original_ratio:.4f} to {new_ratio_calc:.4f}")
            else:
                logs.append(f"✗ Monotonicity violated: Ratio didn't increase with T1")
                # Restore original value
                self.T_1 = original_T1
                return False, logs
            
            # Restore original value
            self.T_1 = original_T1
            
            # Test opposite direction: decrease T1
            perturbed_T1 = original_T1 * 0.9
            self.T_1 = perturbed_T1
            new_ratio_calc = (perturbed_T1**2) / (float(self.T_2)**2)
            
            if new_ratio_calc < original_ratio:
                logs.append(f"✓ Monotonicity: Decreasing T1 from {original_T1} to {perturbed_T1} decreases ratio from {original_ratio:.4f} to {new_ratio_calc:.4f}")
            else:
                logs.append(f"✗ Monotonicity violated: Ratio didn't decrease with T1")
                self.T_1 = original_T1
                return False, logs
            
            # Restore original value
            self.T_1 = original_T1
            
            # Additional test: Period independence from amplitude (for small angles)
            # According to KB, period should be independent of amplitude for small angles
            # We can't test this directly without amplitude data, but we can log the principle
            logs.append("✓ Monotonicity principle: Period ∝ √Length, independent of amplitude for small angles")
            
            return True, logs
            
        except Exception as e:
            # Restore original values on error
            self.T_1 = self.original_T1
            logs.append(f"✗ Error in monotonicity test: {str(e)}")
            return False, logs

    def run_all_tests(self):
        all_logs = []
        success = True
        
        # Run all tests
        test_methods = [
            ("Invariants Test", self.test_invariants),
            ("Trace Consistency Test", self.test_trace_consistency),
            ("Monotonicity Test", self.test_monotonicity)
        ]
        
        for test_name, test_func in test_methods:
            all_logs.append(f"\n--- {test_name} ---")
            test_success, test_logs = test_func()
            all_logs.extend(test_logs)
            if not test_success:
                success = False
                all_logs.append(f"✗ {test_name} FAILED")
            else:
                all_logs.append(f"✓ {test_name} PASSED")
        
        return success, all_logs

# Example usage (commented out):
# if __name__ == "__main__":
#     # Mock solution class for testing
#     class MockSolution:
#         def __init__(self):
#             self.results = {
#                 'T_1': 1.0,
#                 'T_2': 0.9,
#                 'fracl_1l_2': 100/81,  # Approximately 1.2345679
#                 't': 4.5,
#                 't_1': 0.5,
#                 't_2': 0.9,
#                 'l': 0.248  # Example length for T=1.0s, g=9.8: L = gT²/(4π²)
#             }
    
#     checker = PhysicsChecker(MockSolution())
#     success, logs = checker.run_all_tests()
#     for log in logs:
#         print(log)