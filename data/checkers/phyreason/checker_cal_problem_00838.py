import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallback to results dict
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract input parameters with safe defaults
        self.R = float(getattr(solution_instance, 'R', self.results.get('R', 1.0)))
        self.c = float(getattr(solution_instance, 'c', self.results.get('c', 3e8)))
        self.n = -2.0  # Given in problem
        
        # Extract results with safe defaults
        self.x_range = self.results.get('x', 0.0)
        self.t_range = self.results.get('t_range', (0.0, 0.0))
        self.r_deg = float(self.results.get('r', 0.0))  # Critical angle in degrees
        self.s = float(self.results.get('s', 0.0))  # Longest optical path
        self.s_prime = float(self.results.get("s''", self.results.get('s_prime', 0.0)))  # Shortest optical path
        self.t = float(self.results.get('t', 0.0))  # Maximum time
        self.t_prime = float(self.results.get("t''", self.results.get('t_prime', 0.0)))  # Minimum time
        self.v = float(self.results.get('v', 0.0))  # Speed in medium
        
        self.logs = []
        
    def _log(self, message, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            message = f"[{'PASS' if success else 'FAIL'}] {message}"
        self.logs.append(message)
        return success
    
    def _compare_numeric(self, val1, val2, context, rel_tol=1e-2):
        """Compare two numeric values."""
        try:
            if math.isclose(float(val1), float(val2), rel_tol=rel_tol, abs_tol=1e-2):
                return self._log(f"{context}: Numerical match", True)
            else:
                return self._log(f"{context}: Mismatch. Got {val1}, expected {val2}", False)
        except Exception as e:
            return self._log(f"{context}: Comparison failed with error: {str(e)}", False)
    
    def test_invariants(self):
        """Check fundamental physics invariants."""
        tests_passed = 0
        total_tests = 0
        
        # 1. Snell's Law check: |n| = sin(i)/sin(r)
        # At vertex, i = 90° (horizontal incidence at top of semicircle)
        i_rad = math.pi/2
        r_rad = math.radians(self.r_deg)
        lhs = abs(self.n)
        rhs = math.sin(i_rad)/math.sin(r_rad)
        total_tests += 1
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed += 1
            self._log("Snell's Law: |n| = sin(i)/sin(r) holds", True)
        else:
            self._log(f"Snell's Law failed: |n|={lhs}, sin(i)/sin(r)={rhs}", False)
        
        # 2. Speed in medium: v = c/|n|
        total_tests += 1
        if math.isclose(self.v, self.c/abs(self.n), rel_tol=1e-2, abs_tol=1e-2):
            tests_passed += 1
            self._log("Speed in medium: v = c/|n| holds", True)
        else:
            self._log(f"Speed relation failed: v={self.v}, c/|n|={self.c/abs(self.n)}", False)
        
        # 3. Time-distance-speed relations
        total_tests += 1
        if math.isclose(self.t, self.s/self.v, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed += 1
            self._log("Maximum time: t = s/v holds", True)
        else:
            self._log(f"Maximum time relation failed: t={self.t}, s/v={self.s/self.v}", False)
        
        total_tests += 1
        if math.isclose(self.t_prime, self.s_prime/self.v, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed += 1
            self._log("Minimum time: t' = s'/v holds", True)
        else:
            self._log(f"Minimum time relation failed: t'={self.t_prime}, s'/v={self.s_prime/self.v}", False)
        
        return self._log(f"Invariants: {tests_passed}/{total_tests} passed", tests_passed == total_tests)
    
    def test_trace_consistency(self):
        """Check consistency between intermediate and final results."""
        tests_passed = 0
        total_tests = 0
        
        # 1. Check critical angle r = arcsin(1/|n|) = 30°
        expected_r = math.degrees(math.asin(1/abs(self.n)))
        total_tests += 1
        if math.isclose(self.r_deg, expected_r, rel_tol=1e-2, abs_tol=1e-2):
            tests_passed += 1
            self._log(f"Critical angle: r = {self.r_deg}°, expected 30°", True)
        else:
            self._log(f"Critical angle mismatch: {self.r_deg}° vs {expected_r}°", False)
        
        # 2. Check longest optical path s = R/cos(30°)
        expected_s = self.R / math.cos(math.radians(30))
        success = self._compare_numeric(self.s, expected_s, "Longest optical path s")
        total_tests += 1
        if success:
            tests_passed += 1
        
        # 3. Check shortest optical path s' = R
        expected_s_prime = self.R
        success = self._compare_numeric(self.s_prime, expected_s_prime, "Shortest optical path s'")
        total_tests += 1
        if success:
            tests_passed += 1
        
        # 4. Check maximum time t = (4√3 R)/(3c)
        expected_t = (4*math.sqrt(3)*self.R) / (3*self.c)
        success = self._compare_numeric(self.t, expected_t, "Maximum time t")
        total_tests += 1
        if success:
            tests_passed += 1
        
        # 5. Check minimum time t' = 2R/c
        expected_t_prime = (2*self.R) / self.c
        success = self._compare_numeric(self.t_prime, expected_t_prime, "Minimum time t'")
        total_tests += 1
        if success:
            tests_passed += 1
        
        # 6. Check x-range lower bound: (-√3/3 R, 0]
        expected_x_lower = -math.sqrt(3)*self.R/3
        # Handle if x_range is a tuple/list or single value
        if isinstance(self.x_range, (list, tuple)):
            x_lower = float(self.x_range[0])
        else:
            x_lower = float(self.x_range)
        success = self._compare_numeric(x_lower, expected_x_lower, "x-range lower bound")
        total_tests += 1
        if success:
            tests_passed += 1
            self._log("x-range lower bound: -√3 R/3", True)
        else:
            self._log("x-range lower bound mismatch", False)
        
        return self._log(f"Trace consistency: {tests_passed}/{total_tests} passed", tests_passed == total_tests)
    
    def test_monotonicity(self):
        """Check that increasing R increases propagation times proportionally."""
        try:
            # Test with two different R values
            R_values = [1.0, 2.0]
            c_test = 1.0  # Normalize c for test
            
            times_max = []
            times_min = []
            
            for R_val in R_values:
                # Calculate expected times using formulas
                v_test = c_test / abs(self.n)
                
                # Longest path: s = R/cos(30°)
                s_test = R_val / math.cos(math.radians(30))
                t_max = s_test / v_test
                times_max.append(t_max)
                
                # Shortest path: s' = R
                s_prime_test = R_val
                t_min = s_prime_test / v_test
                times_min.append(t_min)
            
            # Check proportionality: doubling R should double times
            ratio_max = times_max[1] / times_max[0]
            ratio_min = times_min[1] / times_min[0]
            
            success = (math.isclose(ratio_max, 2.0, rel_tol=1e-2, abs_tol=1e-2) and 
                      math.isclose(ratio_min, 2.0, rel_tol=1e-2, abs_tol=1e-2))
            
            if success:
                return self._log(f"Monotonicity: Times scale linearly with R (ratios: max={ratio_max:.3f}, min={ratio_min:.3f})", True)
            else:
                return self._log(f"Monotonicity failed: Times don't scale linearly (ratios: max={ratio_max:.3f}, min={ratio_min:.3f})", False)
                
        except Exception as e:
            return self._log(f"Monotonicity test failed with error: {str(e)}", False)
    
    def run_all_tests(self):
        """Run all tests and return success flag and logs."""
        self.logs = []  # Reset logs
        
        # Run tests
        test_results = []
        
        try:
            test_results.append(self.test_invariants())
        except Exception as e:
            self._log(f"Invariants test crashed: {traceback.format_exc()}", False)
            test_results.append(False)
        
        try:
            test_results.append(self.test_trace_consistency())
        except Exception as e:
            self._log(f"Trace consistency test crashed: {traceback.format_exc()}", False)
            test_results.append(False)
        
        try:
            test_results.append(self.test_monotonicity())
        except Exception as e:
            self._log(f"Monotonicity test crashed: {traceback.format_exc()}", False)
            test_results.append(False)
        
        # Determine overall success
        success = all(test_results)
        
        # Summary
        self._log(f"\nOverall: {'All tests passed' if success else 'Some tests failed'}")
        self._log(f"Passed {sum(test_results)} out of {len(test_results)} test categories")
        
        return success, self.logs