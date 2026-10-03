import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from solution_instance."""
        try:
            # Extract results with safe fallbacks
            self.angle1 = getattr(solution_instance, 'angle1', 
                                 getattr(solution_instance, 'results', {}).get('angle1', 0.0))
            self.angle2 = getattr(solution_instance, 'angle2', 
                                 getattr(solution_instance, 'results', {}).get('angle2', 0.0))
            self.angle3 = getattr(solution_instance, 'angle3', 
                                 getattr(solution_instance, 'results', {}).get('angle3', 0.0))
            self.n = getattr(solution_instance, 'n', 
                            getattr(solution_instance, 'results', {}).get('n', 0.0))
            self.v = getattr(solution_instance, 'v', 
                            getattr(solution_instance, 'results', {}).get('v', 0.0))
            self.s = getattr(solution_instance, 's', 
                            getattr(solution_instance, 'results', {}).get('s', 0.0))
            self.t = getattr(solution_instance, 't', 
                            getattr(solution_instance, 'results', {}).get('t', 0.0))
            
            # Extract problem parameters with safe defaults
            self.R = getattr(solution_instance, 'R', 1.0)  # Assume R=1 for relative checks
            self.c = getattr(solution_instance, 'c', 1.0)  # Assume c=1 for dimensionless checks
            
        except Exception as e:
            print(f"Warning: Initialization error - {e}")
            # Set defaults to avoid crashes
            self.angle1 = self.angle2 = self.angle3 = 0.0
            self.n = self.v = self.s = self.t = 0.0
            self.R = 1.0
            self.c = 1.0
    
    def test_invariants(self):
        """Verify fundamental physics laws from KB."""
        logs = []
        success = True
        
        try:
            # 1. Triangle sum invariant (angles in degrees)
            angle_sum = self.angle1 + self.angle2 + self.angle3
            if not math.isclose(angle_sum, 180.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Angle sum {angle_sum} ≠ 180°")
                success = False
            else:
                logs.append(f"PASS: Triangle angle sum = {angle_sum}°")
            
            # 2. Critical angle consistency (∠1 = ∠2 = ∠3 from step_1)
            if not (math.isclose(self.angle1, self.angle2, rel_tol=1e-2, abs_tol=1e-2) and
                    math.isclose(self.angle2, self.angle3, rel_tol=1e-2, abs_tol=1e-2)):
                logs.append(f"FAIL: Angles not equal: ∠1={self.angle1}°, ∠2={self.angle2}°, ∠3={self.angle3}°")
                success = False
            else:
                logs.append(f"PASS: All angles equal to {self.angle1}°")
            
            # 3. Refractive index from critical angle: n = 1/sin(C)
            C_rad = math.radians(self.angle1)
            n_expected = 1.0 / math.sin(C_rad) if math.sin(C_rad) > 0 else float('inf')
            
            if not math.isclose(self.n, n_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: n={self.n} ≠ 1/sin({self.angle1}°)={n_expected}")
                success = False
            else:
                logs.append(f"PASS: n = 1/sin(C) holds")
            
            # 4. Speed from refractive index: v = c/n
            v_expected = self.c / self.n if self.n > 0 else 0.0
            if not math.isclose(self.v, v_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v={self.v} ≠ c/n={v_expected} (c={self.c})")
                success = False
            else:
                logs.append(f"PASS: v = c/n holds")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        try:
            # 1. Distance formula: s = 2R + 2R cos(∠1)
            C_rad = math.radians(self.angle1)
            s_expected = 2 * self.R + 2 * self.R * math.cos(C_rad)
            
            if not math.isclose(self.s, s_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: s={self.s} ≠ 2R + 2R cos({self.angle1}°)={s_expected} (R={self.R})")
                success = False
            else:
                logs.append(f"PASS: s = 2R(1+cos(C)) holds")
            
            # 2. Time formula: t = s/v
            t_expected = self.s / self.v if self.v > 0 else 0.0
            
            if not math.isclose(self.t, t_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: t={self.t} ≠ s/v={t_expected}")
                success = False
            else:
                logs.append(f"PASS: t = s/v holds")
            
            # 3. Check target answer forms (symbolic)
            # Target 1: n = 2√3/3 ≈ 1.155
            target_n = 2 * math.sqrt(3) / 3
            if not math.isclose(self.n, target_n, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"WARN: n={self.n} not close to target 2√3/3≈{target_n}")
            
            # Target 2: t = 2√3 R/c
            target_t = 2 * math.sqrt(3) * self.R / self.c
            if not math.isclose(self.t, target_t, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"WARN: t={self.t} not close to target 2√3 R/c≈{target_t}")
            else:
                logs.append(f"PASS: t matches target expression")
            
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Perturb refractive index and verify expected trend in time."""
        logs = []
        success = True
        
        try:
            # Save original state
            n_original = float(self.n)
            
            # Perturb n upwards by 5%
            n_perturbed = n_original * 1.05
            
            # Recompute dependent quantities using physics formulas
            C_perturbed = math.asin(1.0 / n_perturbed)  # C = arcsin(1/n)
            s_perturbed = 2 * self.R * (1 + math.cos(C_perturbed))
            v_perturbed = self.c / n_perturbed
            t_perturbed = s_perturbed / v_perturbed
            
            # From symbolic analysis: t = (2R/c)(n + √(n²-1)), which increases with n for n>1
            # So t_perturbed > t_original when n increases
            if t_perturbed > self.t:
                logs.append(f"PASS: t increases when n increases ({t_perturbed:.4f} > {self.t:.4f})")
            else:
                logs.append(f"FAIL: t doesn't increase with n ({t_perturbed:.4f} ≤ {self.t:.4f})")
                success = False
            
            # Restore original state
            self.n = n_original
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Execute all tests and return consolidated results."""
        all_logs = []
        overall_success = True
        
        # Run each test
        for test_name, test_func in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"\n=== {test_name} ===")
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                all_logs.extend(logs)
            except Exception as e:
                all_logs.append(f"CRASH: {traceback.format_exc()}")
                overall_success = False
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, all_logs

# Example usage (for testing):
if __name__ == "__main__":
    class MockSolution:
        def __init__(self):
            self.results = {
                'angle1': 60.0,
                'angle2': 60.0,
                'angle3': 60.0,
                'n': 2 * math.sqrt(3) / 3,
                'v': 0.8660254,
                's': 3.0,
                't': 2 * math.sqrt(3)
            }
            self.R = 1.0
            self.c = 1.0
    
    checker = PhysicsChecker(MockSolution())
    success, logs = checker.run_all_tests()
    for log in logs:
        print(log)