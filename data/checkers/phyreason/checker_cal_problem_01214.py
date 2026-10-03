import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        self.logs = []
        
        # Inputs (with defaults if not present)
        self.AB = getattr(solution_instance, 'AB', 0.15)  # meters
        self.BC = getattr(solution_instance, 'BC', 0.20)  # meters
        self.t_interval = getattr(solution_instance, 't_interval', 0.1)  # seconds
        
        # Get results dictionary (default empty)
        results = getattr(solution_instance, 'results', {})
        
        # Extract results from dictionary
        self.t_B = results.get('t_B', None)
        self.t_A = results.get('t_A', None)
        self.a = results.get('a', None)
        self.v_B = results.get('v_B', None)
        # The number of balls above A is stored as 'num_above_A' in the code
        self.n_balls = results.get('num_above_A', None)
        
        # Validate critical values exist
        self.critical_missing = []
        for name, val in [('t_B', self.t_B), ('t_A', self.t_A), 
                         ('a', self.a), ('v_B', self.v_B), ('n_balls', self.n_balls)]:
            if val is None:
                self.critical_missing.append(name)
        
        if self.critical_missing:
            self.logs.append(f"WARNING: Missing results: {self.critical_missing}")
    
    def test_invariants(self):
        """
        Verify kinematic equations and physical invariants.
        """
        logs = []
        success = True
        
        # 1. Check v_B = (AB + BC) / (2t)
        expected_v_B = (self.AB + self.BC) / (2 * self.t_interval)
        if self.v_B is not None:
            if not math.isclose(self.v_B, expected_v_B, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_B mismatch. Expected {expected_v_B:.3f}, got {self.v_B:.3f}")
                success = False
            else:
                logs.append(f"PASS: v_B = (AB+BC)/(2t) check")
        
        # 2. Check a = (BC - AB) / t²
        expected_a = (self.BC - self.AB) / (self.t_interval ** 2)
        if self.a is not None:
            if not math.isclose(self.a, expected_a, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: a mismatch. Expected {expected_a:.3f}, got {self.a:.3f}")
                success = False
            else:
                logs.append(f"PASS: a = (BC-AB)/t² check")
        
        # 3. Check t_B = v_B / a (from v = a*t starting from rest)
        if all(x is not None for x in [self.v_B, self.a, self.t_B]):
            expected_t_B = self.v_B / self.a
            if not math.isclose(self.t_B, expected_t_B, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: t_B mismatch. Expected {expected_t_B:.3f}, got {self.t_B:.3f}")
                success = False
            else:
                logs.append(f"PASS: t_B = v_B/a check")
        
        # 4. Check t_A = t_B - t
        if all(x is not None for x in [self.t_B, self.t_A]):
            expected_t_A = self.t_B - self.t_interval
            if not math.isclose(self.t_A, expected_t_A, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: t_A mismatch. Expected {expected_t_A:.3f}, got {self.t_A:.3f}")
                success = False
            else:
                logs.append(f"PASS: t_A = t_B - t check")
        
        # 5. Energy consistency (for incline with gravity component)
        # Assuming g = 9.8 m/s², sinθ = a/g
        g = 9.8
        if self.a is not None:
            sin_theta = self.a / g
            if not (0 <= sin_theta <= 1):
                logs.append(f"WARNING: a/g = {sin_theta:.3f} suggests unrealistic incline")
        
        self.logs.extend(logs)
        return success
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        logs = []
        success = True
        
        # Check that v_B matches target answer 1.75 m/s
        if self.v_B is not None:
            if not math.isclose(self.v_B, 1.75, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_B ≠ 1.75 m/s. Got {self.v_B:.3f}")
                success = False
            else:
                logs.append(f"PASS: v_B matches target 1.75 m/s")
        
        # Check that n_balls matches target answer 2
        if self.n_balls is not None:
            if not math.isclose(self.n_balls, 2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: n_balls ≠ 2. Got {self.n_balls}")
                success = False
            else:
                logs.append(f"PASS: n_balls matches target 2")
        
        # Verify n_balls calculation from t_A and t_interval
        if self.t_A is not None and self.n_balls is not None:
            # Number of balls released after first ball before time t_A
            expected_n = int(self.t_A / self.t_interval)
            if not math.isclose(self.n_balls, expected_n, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: n_balls calculation. Expected {expected_n}, got {self.n_balls}")
                success = False
        
        self.logs.extend(logs)
        return success
    
    def test_monotonicity(self):
        """
        Perturb inputs and check monotonic trends.
        """
        logs = []
        success = True
        
        # Store original values
        orig_AB = self.AB
        
        # Test 1: Increase AB slightly, v_B should increase
        try:
            self.AB = orig_AB * 1.01
            new_v_B = (self.AB + self.BC) / (2 * self.t_interval)
            if not new_v_B > self.v_B:
                logs.append(f"FAIL: v_B not monotonic with AB")
                success = False
            else:
                logs.append(f"PASS: v_B increases with AB (monotonic)")
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {e}")
        
        # Restore AB
        self.AB = orig_AB
        
        # Test 2: Increase BC, a should increase
        orig_BC = self.BC
        try:
            self.BC = orig_BC * 1.01
            new_a = (self.BC - self.AB) / (self.t_interval ** 2)
            if not new_a > self.a:
                logs.append(f"FAIL: a not monotonic with BC")
                success = False
            else:
                logs.append(f"PASS: a increases with BC (monotonic)")
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {e}")
        
        # Restore BC
        self.BC = orig_BC
        
        self.logs.extend(logs)
        return success
    
    def run_all_tests(self):
        """
        Execute all tests and return summary.
        """
        self.logs = []  # Reset logs
        
        # Run tests only if critical values exist
        if self.critical_missing:
            self.logs.append("SKIP: Missing critical results, cannot run tests")
            return False, self.logs
        
        try:
            test1 = self.test_invariants()
            test2 = self.test_trace_consistency()
            test3 = self.test_monotonicity()
            
            overall = test1 and test2 and test3
            
            self.logs.append(f"\nSUMMARY:")
            self.logs.append(f"  Invariants: {'PASS' if test1 else 'FAIL'}")
            self.logs.append(f"  Trace consistency: {'PASS' if test2 else 'FAIL'}")
            self.logs.append(f"  Monotonicity: {'PASS' if test3 else 'FAIL'}")
            self.logs.append(f"  OVERALL: {'PASS' if overall else 'FAIL'}")
            
            return overall, self.logs
            
        except Exception as e:
            self.logs.append(f"EXCEPTION in run_all_tests: {traceback.format_exc()}")
            return False, self.logs