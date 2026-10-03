import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from solution_instance.
        """
        # First try to get results from the instance
        self.results = {}
        if hasattr(solution_instance, 'results'):
            self.results = solution_instance.results
        elif hasattr(solution_instance, '__dict__'):
            # Check if results is in __dict__
            self.results = solution_instance.__dict__.get('results', {})
        
        # Define the exact variables we need to extract
        required_vars = ['m', 'e', 'U', 'T', 'd', 'n', 'v_1', 'L_n', 'v_m', 'v_n', 'F', 'a', 'E']
        
        # Extract only the required variables, ignoring intermediate ones
        # Use default values if not found, but ensure they're floats
        filtered_results = {}
        for var in required_vars:
            if var in self.results:
                try:
                    # Get the value
                    val = self.results[var]
                    # Convert to float, handling various types
                    if isinstance(val, (int, float)):
                        filtered_results[var] = float(val)
                    else:
                        # Try to convert from string or other types
                        filtered_results[var] = float(val)
                except (ValueError, TypeError):
                    filtered_results[var] = 0.0
            else:
                filtered_results[var] = 0.0
        
        # Now assign from filtered_results
        self.m = filtered_results['m']
        self.e = filtered_results['e']
        self.U = filtered_results['U']
        self.T = filtered_results['T']
        self.d = filtered_results['d']
        self.n = filtered_results['n']
        self.v_1 = filtered_results['v_1']
        self.L_n = filtered_results['L_n']
        self.v_m = filtered_results['v_m']
        self.v_n = filtered_results['v_n']
        self.F = filtered_results['F']
        self.a = filtered_results['a']
        self.E = filtered_results['E']
        
        # Define tolerance for fuzzy matching
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Store logs
        self.logs = []

    def log(self, message):
        """Add message to logs."""
        self.logs.append(message)

    def test_invariants(self):
        """Test fundamental physics invariants: energy conservation, Newton's laws."""
        try:
            # Guard against division by zero or extremely small values
            # Use a more reasonable threshold for "too small"
            if abs(self.m) < 1e-30:  # Changed from 1e-50 to 1e-30
                self.log(f"Mass m is too small ({self.m}), cannot test energy conservation")
                return False
            
            if abs(self.e) < 1e-30:  # Changed from 1e-50 to 1e-30
                self.log(f"Charge e is too small ({self.e}), cannot test energy conservation")
                return False
            
            # Test 1: Energy conservation for v_1
            expected_v1 = math.sqrt(2 * self.U * self.e / self.m)
            if not math.isclose(self.v_1, expected_v1, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                self.log(f"Energy conservation violated for v_1: got {self.v_1}, expected {expected_v1}")
                return False
            self.log("✓ Energy conservation holds for v_1")
            
            # Test 2: Energy conservation for v_n
            if abs(self.v_n) > 1e-10:
                expected_vn = math.sqrt(2 * self.n * self.U * self.e / self.m)
                if not math.isclose(self.v_n, expected_vn, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    self.log(f"Energy conservation violated for v_n: got {self.v_n}, expected {expected_vn}")
                    return False
                self.log("✓ Energy conservation holds for v_n")
            
            # Test 3: Force-acceleration relation (Newton's 2nd Law)
            if abs(self.F) > 1e-30 and abs(self.a) > 1e-30:
                expected_F = self.m * self.a
                if not math.isclose(self.F, expected_F, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    self.log(f"Newton's 2nd Law violated: F = m*a: got F={self.F}, m*a={expected_F}")
                    return False
                self.log("✓ Newton's 2nd Law holds (F = m*a)")
            
            # Test 4: Electric field and force relation
            if abs(self.F) > 1e-30 and abs(self.E) > 1e-30:
                expected_F2 = self.e * self.E
                if not math.isclose(self.F, expected_F2, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    self.log(f"Electric force relation violated: F = e*E: got F={self.F}, e*E={expected_F2}")
                    return False
                self.log("✓ Electric force relation holds (F = e*E)")
            
            # Test 5: Electric field in gap
            if abs(self.E) > 1e-30:
                if abs(self.d) < 1e-100 or self.d == 0:
                    self.log("d is too small or zero, cannot compute E = U/d")
                    return False
                expected_E = self.U / self.d
                if not math.isclose(self.E, expected_E, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    self.log(f"Electric field relation violated: E = U/d: got E={self.E}, U/d={expected_E}")
                    return False
                self.log("✓ Electric field relation holds (E = U/d)")
            
            return True
            
        except Exception as e:
            self.log(f"Exception in test_invariants: {e}\n{traceback.format_exc()}")
            return False

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Check consistency between v_1, v_n, and L_n
            if abs(self.L_n) > 1e-10 and abs(self.v_n) > 1e-10:
                if abs(self.T) < 1e-100 or self.T == 0:
                    self.log("T is too small or zero, cannot compute expected L_n")
                    return False
                expected_Ln = self.v_n * (self.T / 2)
                if not math.isclose(self.L_n, expected_Ln, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    self.log(f"Inconsistency: L_n should be v_n*(T/2): got L_n={self.L_n}, expected {expected_Ln}")
                    return False
                self.log("✓ L_n consistent with v_n and T/2")
            
            # Check v_m derivation
            if abs(self.v_m) > 1e-10:
                if abs(self.d) < 1e-30 or self.d == 0 or abs(self.m) < 1e-30:  # Changed from 1e-50
                    self.log("d or m is too small, cannot compute expected v_m")
                    return False
                expected_a = (self.U * self.e) / (self.d * self.m)
                expected_vm = expected_a * (self.T / 2)
                if not math.isclose(self.v_m, expected_vm, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    self.log(f"Inconsistency: v_m should be (U*e*T)/(2*d*m): got v_m={self.v_m}, expected {expected_vm}")
                    return False
                self.log("✓ v_m consistent with acceleration and T/2")
            
            # Check that v_n for n=1 equals v_1
            if abs(self.v_n) > 1e-10 and self.n == 1:
                if not math.isclose(self.v_1, self.v_n, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    self.log(f"Inconsistency: v_n for n=1 should equal v_1: got v_1={self.v_1}, v_n={self.v_n}")
                    return False
                self.log("✓ v_1 equals v_n for n=1")
            
            return True
            
        except Exception as e:
            self.log(f"Exception in test_trace_consistency: {e}\n{traceback.format_exc()}")
            return False

    def test_monotonicity(self):
        """Perturb input parameters and check monotonic trends."""
        try:
            # Store original values
            orig_U = self.U
            orig_n = self.n
            orig_d = self.d
            orig_v1 = self.v_1
            orig_Ln = self.L_n
            orig_vm = self.v_m
            
            # Guard against zero or extremely small values
            if abs(self.m) < 1e-30:  # Changed from 1e-50
                self.log(f"Mass m is too small ({self.m}), cannot test monotonicity")
                return False
            
            if abs(self.e) < 1e-30:  # Changed from 1e-50
                self.log(f"Charge e is too small ({self.e}), cannot test monotonicity")
                return False
            
            if abs(self.T) < 1e-100 or self.T == 0:
                self.log(f"Period T is too small or zero ({self.T}), cannot test monotonicity")
                return False
            
            # Test 1: Increase voltage U should increase v_1
            self.U = orig_U * 1.1
            new_v1 = math.sqrt(2 * self.U * self.e / self.m)
            if new_v1 <= orig_v1:
                self.log(f"Monotonicity violated: Increasing U did not increase v_1")
                self.U = orig_U
                return False
            self.log("✓ v_1 increases with U")
            self.U = orig_U
            
            # Test 2: Increase n should increase L_n
            if abs(orig_Ln) > 1e-10:
                self.n = int(orig_n * 1.1) + 1
                new_Ln = math.sqrt(2 * self.n * self.U * self.e / self.m) * (self.T / 2)
                if new_Ln <= orig_Ln:
                    self.log(f"Monotonicity violated: Increasing n did not increase L_n")
                    self.n = orig_n
                    return False
                self.log("✓ L_n increases with n")
                self.n = orig_n
            
            # Test 3: Increase gap distance d should decrease v_m
            if abs(orig_vm) > 1e-10:
                if abs(self.d) < 1e-30 or self.d == 0:  # Changed from 1e-50
                    self.log("d is too small, cannot test monotonicity for v_m")
                    self.d = orig_d
                    return False
                self.d = orig_d * 1.1
                new_vm = (self.U * self.e * self.T) / (2 * self.d * self.m)
                if new_vm >= orig_vm:
                    self.log(f"Monotonicity violated: Increasing d did not decrease v_m")
                    self.d = orig_d
                    return False
                self.log("✓ v_m decreases with d")
                self.d = orig_d
            
            return True
            
        except Exception as e:
            self.log(f"Exception in test_monotonicity: {e}\n{traceback.format_exc()}")
            # Restore original values
            self.U = orig_U if 'orig_U' in locals() else self.U
            self.n = orig_n if 'orig_n' in locals() else self.n
            self.d = orig_d if 'orig_d' in locals() else self.d
            return False

    def run_all_tests(self):
        """Run all tests and return success flag with logs."""
        self.logs = []
        
        try:
            success = True
            
            if not self.test_invariants():
                success = False
                
            if not self.test_trace_consistency():
                success = False
                
            if not self.test_monotonicity():
                success = False
            
            if success:
                self.log("All physics checks passed ✓")
            else:
                self.log("Physics checks failed ✗")
                
            return success, self.logs
            
        except Exception as e:
            self.log(f"Critical error in run_all_tests: {e}\n{traceback.format_exc()}")
            return False, self.logs