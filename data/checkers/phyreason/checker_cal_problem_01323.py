import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        Use getattr with fallbacks to avoid AttributeError in sandbox.
        """
        # Given constants from problem statement
        self.P_m = getattr(solution_instance, 'P_m', 8e3)  # W
        self.m = getattr(solution_instance, 'm', 1000.0)   # kg
        self.g = getattr(solution_instance, 'g', 10.0)     # m/s²
        
        # Expected results from solution
        self.results = {}
        # Extract from solution_instance.results if available, else from attributes
        results_dict = getattr(solution_instance, 'results', {})
        self.results['v_m'] = getattr(solution_instance, 'v_m', results_dict.get('v_m', 0.0))
        self.results['P'] = getattr(solution_instance, 'P', results_dict.get('P', 0.0))
        self.results['F'] = getattr(solution_instance, 'F', results_dict.get('F', 0.0))
        self.results['t'] = getattr(solution_instance, 't', results_dict.get('t', 0.0))
        self.results['v'] = getattr(solution_instance, 'v', results_dict.get('v', 0.0))
        self.results['a'] = getattr(solution_instance, 'a', results_dict.get('a', 0.0))
        
        # Derived constants for checks
        self.mg = self.m * self.g  # Weight force in N
        
        # Tolerance for fuzzy matching (1-2% as per robustness rules)
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Log storage
        self.logs = []
        
    def log(self, msg):
        """Append message to logs."""
        self.logs.append(msg)
        
    def test_invariants(self):
        """
        Verify physical invariants based on KB theorems.
        Checks energy conservation (work-energy), Newton's laws, and power relationships.
        """
        try:
            # 1. Maximum speed at constant velocity (step 1-2): P_m = m*g*v_m
            expected_v_m = self.P_m / (self.m * self.g)
            if not math.isclose(self.results['v_m'], expected_v_m, 
                               rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self.log(f"FAIL: v_m = {self.results['v_m']:.4f}, expected {expected_v_m:.4f} from P_m/(m*g)")
                return False
            self.log(f"PASS: v_m matches P_m/(m*g) within tolerance")
            
            # 2. Force balance at constant velocity v=0.4 m/s (step 3-4)
            # From problem context, when velocity is constant (v=0.4 m/s), F = m*g
            expected_F_const = self.mg
            # The solution should have F=10000 N from step 3
            if not math.isclose(self.results['F'], expected_F_const,
                               rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self.log(f"FAIL: Traction force F = {self.results['F']:.1f}, expected {expected_F_const:.1f} (m*g) for constant velocity")
                return False
            self.log(f"PASS: Constant velocity force balance F = m*g")
            
            # 3. Power at constant velocity (step 4): P = F*v with v=0.4 m/s
            v_const = 0.4  # m/s (from problem context for step 4)
            expected_P_const = expected_F_const * v_const
            if not math.isclose(self.results['P'], expected_P_const,
                               rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self.log(f"FAIL: Power P = {self.results['P']:.1f}, expected {expected_P_const:.1f} (F*v with v=0.4 m/s)")
                return False
            self.log(f"PASS: Power P = F*v for constant velocity case")
            
            # 4. Newton's second law for accelerated case (step 5-6)
            # When accelerating at a=2 m/s² (from step 6: F_acc = m*g + m*a)
            # We need the acceleration value. From step 6, a=2 m/s².
            a_accel = 2.0  # m/s² (from problem context for accelerated motion)
            expected_F_accel = self.mg + self.m * a_accel
            # Note: The solution's F is for constant velocity case, not accelerated case
            # We should NOT check results['F'] against expected_F_accel
            # Instead, we verify the physics by checking if v and t are consistent
            self.log(f"INFO: F is constant-velocity force ({self.results['F']:.1f} N), not accelerated force")
            
            # 5. Maximum power constraint during acceleration (step 7)
            # P_m = F_acc * v, so v = P_m / F_acc
            expected_v_accel = self.P_m / expected_F_accel
            if not math.isclose(self.results['v'], expected_v_accel,
                               rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self.log(f"FAIL: Final velocity v = {self.results['v']:.4f}, expected {expected_v_accel:.4f} (P_m/(m*g + m*a))")
                return False
            self.log(f"PASS: Maximum power constraint P_m = F_acc*v gives correct v")
            
            # 6. Kinematics for acceleration time (step 8)
            # t = v / a (from v = a*t starting from rest)
            expected_t = expected_v_accel / a_accel
            if not math.isclose(self.results['t'], expected_t,
                               rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self.log(f"FAIL: Time t = {self.results['t']:.4f}, expected {expected_t:.4f} (v/a)")
                return False
            self.log(f"PASS: Kinematics t = v/a consistent")
            
            return True
            
        except Exception as e:
            self.log(f"ERROR in test_invariants: {str(e)}")
            self.log(traceback.format_exc())
            return False
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        Ensure the solution steps are logically consistent.
        """
        try:
            # Check that all three target answers match the computed values
            # Target answers: v_m=0.8 m/s, P=4000 W, t=1/3 ≈ 0.333 s
            
            # 1. Check v_m
            if not math.isclose(self.results['v_m'], 0.8,
                               rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self.log(f"FAIL: v_m = {self.results['v_m']:.4f} doesn't match target 0.8 m/s")
                return False
            self.log(f"PASS: v_m matches target answer 0.8 m/s")
            
            # 2. Check P (should be 4000 W for constant velocity case)
            if not math.isclose(self.results['P'], 4000.0,
                               rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self.log(f"FAIL: P = {self.results['P']:.1f} doesn't match target 4000 W")
                return False
            self.log(f"PASS: P matches target answer 4000 W")
            
            # 3. Check t (should be 1/3 ≈ 0.3333 s)
            if not math.isclose(self.results['t'], 1.0/3.0,
                               rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self.log(f"FAIL: t = {self.results['t']:.4f} doesn't match target 1/3 s ≈ 0.3333")
                return False
            self.log(f"PASS: t matches target answer 1/3 s")
            
            # Additional consistency: v from step 7 should be 2/3 ≈ 0.667 m/s
            expected_v = 2.0/3.0
            if not math.isclose(self.results['v'], expected_v,
                               rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self.log(f"INFO: v = {self.results['v']:.4f}, expected 2/3 ≈ 0.667 m/s")
                # Don't fail for this as it's an intermediate check
                
            # F should be 10000 N for constant velocity case
            if not math.isclose(self.results['F'], 10000.0,
                               rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self.log(f"INFO: F = {self.results['F']:.1f}, expected 10000 N for constant velocity case")
                # Don't fail for this as it's an intermediate check
                
            return True
            
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {str(e)}")
            self.log(traceback.format_exc())
            return False
    
    def test_monotonicity(self):
        """
        Perturb an input (mass) and verify expected trends.
        Important: Restore original values after perturbation.
        """
        original_m = self.m
        
        try:
            # Test 1: Increase mass by 10%
            self.m = original_m * 1.10
            self.mg = self.m * self.g
            
            # Expected trends:
            # 1. v_m = P_m/(m*g) should decrease
            new_v_m = self.P_m / (self.m * self.g)
            if new_v_m >= self.results['v_m']:
                self.log(f"FAIL: v_m should decrease when mass increases, but got {new_v_m:.4f} vs original {self.results['v_m']:.4f}")
                return False
            self.log(f"PASS: v_m decreases when mass increases (monotonic)")
            
            # 2. Power at constant velocity v=0.4 m/s: P = m*g*v should increase
            v_const = 0.4
            new_P_const = self.mg * v_const
            if new_P_const <= self.results['P']:
                self.log(f"FAIL: P at constant v should increase when mass increases, but got {new_P_const:.1f} vs original {self.results['P']:.1f}")
                return False
            self.log(f"PASS: P at constant v increases when mass increases (monotonic)")
            
            # 3. Time for accelerated motion with same acceleration a=2 m/s²
            # t = v/a = (P_m/(m*g + m*a))/a
            a_accel = 2.0
            new_F_accel = self.mg + self.m * a_accel
            new_v_accel = self.P_m / new_F_accel
            new_t = new_v_accel / a_accel
            
            # Since m increases, F_accel increases, v_accel decreases, t decreases
            original_F_accel = original_m * self.g + original_m * a_accel
            original_v_accel = self.P_m / original_F_accel
            original_t = original_v_accel / a_accel
            
            if new_t >= original_t:
                self.log(f"FAIL: Time t should decrease when mass increases (with same a), but got {new_t:.4f} vs original {original_t:.4f}")
                return False
            self.log(f"PASS: t decreases when mass increases (with same acceleration)")
            
            # Restore original mass
            self.m = original_m
            self.mg = self.m * self.g
            
            return True
            
        except Exception as e:
            # Restore original mass on error
            self.m = original_m
            self.mg = self.m * self.g
            self.log(f"ERROR in test_monotonicity: {str(e)}")
            self.log(traceback.format_exc())
            return False
    
    def run_all_tests(self):
        """
        Run all tests and return success flag and logs.
        """
        self.logs = []  # Reset logs
        
        tests = [
            ("Invariant checks", self.test_invariants),
            ("Trace consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        success = True
        for test_name, test_func in tests:
            self.log(f"\n--- Running {test_name} ---")
            test_result = test_func()
            if not test_result:
                success = False
                self.log(f"{test_name} FAILED")
            else:
                self.log(f"{test_name} PASSED")
        
        return success, self.logs