import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Inputs from problem statement (with safe defaults)
        self.h_person = getattr(solution_instance, 'h_person', 1.6)  # m
        self.L = getattr(solution_instance, 'L', 0.6)  # m (horizontal distance)
        self.d_hole = getattr(solution_instance, 'd_hole', 0.01)  # m (diameter)
        self.t_hole = getattr(solution_instance, 't_hole', 0.014)  # m (depth/thickness)
        
        # Extract results with safe fallbacks
        self.n = self.results.get('n', 0.0)
        self.n_prime = self.results.get("n'", 0.0)
        self.tan_theta = self.results.get('tantheta', 0.0)
        self.sin_theta = self.results.get('sintheta', 0.0)
        self.tan_alpha = self.results.get('tanalpha', 0.0)
        self.sin_alpha = self.results.get('sinalpha', 0.0)
        
        # Derived geometric quantities
        self.h_hole = 0.5 * self.h_person  # hole at half person's height
        
    def test_invariants(self):
        """Verify Snell's Law and geometric relationships"""
        logs = []
        success = True
        
        try:
            # 1. Geometric consistency: tanθ = (h/2) / L (since hole at h/2)
            expected_tan_theta = (self.h_person/2) / self.L  # 0.8/0.6 = 4/3
            if not math.isclose(self.tan_theta, expected_tan_theta, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: tanθ mismatch. Expected ~{expected_tan_theta:.3f}, got {self.tan_theta:.3f}")
                success = False
            else:
                logs.append(f"PASS: tanθ = {self.tan_theta:.3f} ≈ 4/3")
            
            # 2. Geometric consistency: tanα = diameter / depth
            expected_tan_alpha = self.d_hole / self.t_hole  # 0.01/0.014 ≈ 0.714
            if not math.isclose(self.tan_alpha, expected_tan_alpha, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: tanα mismatch. Expected ~{expected_tan_alpha:.3f}, got {self.tan_alpha:.3f}")
                success = False
            else:
                logs.append(f"PASS: tanα = {self.tan_alpha:.3f} ≈ 1/1.4")
            
            # 3. Trigonometric identity: sinθ from tanθ
            expected_sin_theta = math.sin(math.atan(self.tan_theta))
            if not math.isclose(self.sin_theta, expected_sin_theta, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: sinθ mismatch. Expected ~{expected_sin_theta:.3f}, got {self.sin_theta:.3f}")
                success = False
            else:
                logs.append(f"PASS: sinθ = {self.sin_theta:.3f} ≈ 0.8")
            
            # 4. Trigonometric identity: sinα from tanα
            expected_sin_alpha = math.sin(math.atan(self.tan_alpha))
            if not math.isclose(self.sin_alpha, expected_sin_alpha, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: sinα mismatch. Expected ~{expected_sin_alpha:.3f}, got {self.sin_alpha:.3f}")
                success = False
            else:
                logs.append(f"PASS: sinα = {self.sin_alpha:.3f} ≈ 1/√2.96")
            
            # 5. Snell's Law: n = sinθ / sinα
            if self.sin_alpha != 0:
                expected_n = self.sin_theta / self.sin_alpha
                if not math.isclose(self.n, expected_n, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: n mismatch via Snell. Expected ~{expected_n:.3f}, got {self.n:.3f}")
                    success = False
                else:
                    logs.append(f"PASS: n = sinθ/sinα = {self.n:.3f} ≈ 1.38")
            
            # 6. Critical angle condition: n' = 1 / sinα
            if self.sin_alpha != 0:
                expected_n_prime = 1.0 / self.sin_alpha
                if not math.isclose(self.n_prime, expected_n_prime, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: n' mismatch. Expected ~{expected_n_prime:.3f}, got {self.n_prime:.3f}")
                    success = False
                else:
                    logs.append(f"PASS: n' = 1/sinα = {self.n_prime:.3f}")
            
        except Exception as e:
            logs.append(f"ERROR in invariants: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Check intermediate results lead to final answers"""
        logs = []
        success = True
        
        try:
            # Recompute n from intermediate values
            if self.sin_alpha != 0:
                computed_n = self.sin_theta / self.sin_alpha
                if not math.isclose(self.n, computed_n, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: n inconsistent. From intermediates: {computed_n:.3f}, stored: {self.n:.3f}")
                    success = False
                else:
                    logs.append(f"PASS: n consistent with sinθ and sinα")
            
            # Check n' > n (critical index should be higher)
            if self.n_prime <= self.n:
                logs.append(f"FAIL: n' ({self.n_prime:.3f}) should be > n ({self.n:.3f}) for grazing exit")
                success = False
            else:
                logs.append(f"PASS: n' > n as expected")
                
            # Verify approximate values are reasonable
            # n should be around 1.38 ± 0.02
            if not (1.36 <= self.n <= 1.40):
                logs.append(f"FAIL: n outside reasonable range 1.36-1.40 (got {self.n:.3f})")
                success = False
            else:
                logs.append(f"PASS: n ≈ {self.n:.3f} is reasonable")
                
            # n' should be around 1.72 ± 0.02
            if not (1.70 <= self.n_prime <= 1.74):
                logs.append(f"FAIL: n' outside reasonable range 1.70-1.74 (got {self.n_prime:.3f})")
                success = False
            else:
                logs.append(f"PASS: n' ≈ {self.n_prime:.3f} is reasonable")
                
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Perturb inputs and check physics trends"""
        logs = []
        success = True
        
        original_d_hole = self.d_hole
        original_t_hole = self.t_hole
        
        try:
            # Test 1: Increase hole diameter → increase sinα → decrease n
            self.d_hole *= 1.1  # +10%
            new_tan_alpha = self.d_hole / self.t_hole
            new_sin_alpha = math.sin(math.atan(new_tan_alpha))
            new_n = self.sin_theta / new_sin_alpha
            
            if new_n >= self.n:
                logs.append(f"FAIL: n should decrease when diameter increases (was {self.n:.3f}, becomes {new_n:.3f})")
                success = False
            else:
                logs.append(f"PASS: n decreases with larger hole (monotonicity OK)")
            
            # Test 2: Decrease hole depth → increase tanα → increase sinα → decrease n
            self.d_hole = original_d_hole  # restore
            self.t_hole *= 0.9  # -10%
            new_tan_alpha = self.d_hole / self.t_hole
            new_sin_alpha = math.sin(math.atan(new_tan_alpha))
            new_n = self.sin_theta / new_sin_alpha
            
            if new_n >= self.n:
                logs.append(f"FAIL: n should decrease when depth decreases (was {self.n:.3f}, becomes {new_n:.3f})")
                success = False
            else:
                logs.append(f"PASS: n decreases with shallower hole (monotonicity OK)")
            
            # Restore original values
            self.t_hole = original_t_hole
            self.d_hole = original_d_hole
            
        except Exception as e:
            logs.append(f"ERROR in monotonicity: {str(e)}")
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Execute all checks and return summary"""
        all_logs = []
        overall_success = True
        
        # Run each test suite
        for test_name, test_func in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            try:
                success, logs = test_func()
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
                    all_logs.append(f"Result: FAILED")
                else:
                    all_logs.append(f"Result: PASSED")
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False
        
        return overall_success, all_logs