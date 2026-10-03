import math
import traceback
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract all required inputs and results using getattr
        self.results = getattr(solution_instance, 'results', {})
        
        # Inputs (v_avg is given as 3 m/s from problem context)
        self.v_avg_target = 3.0  # From step_5
        self.v2 = 5.0  # Given in problem
        
        # Results from solution
        self.t_2 = getattr(solution_instance, 't_2', self.results.get('t_2', None))
        self.v_avg = getattr(solution_instance, 'v_avg', self.results.get('v_avg', None))
        self.t = getattr(solution_instance, 't', self.results.get('t', None))
        self.v = getattr(solution_instance, 'v', self.results.get('v', None))
        self.t_1 = getattr(solution_instance, 't_1', self.results.get('t_1', None))
        self.s = getattr(solution_instance, 's', self.results.get('s', None))
        
        # Symbolic variables for invariant checks
        self.s_sym = sp.symbols('s', positive=True)
        self.v_sym = sp.symbols('v', positive=True)
        self.v2_sym = sp.symbols('v2', positive=True)
        self.v_avg_sym = sp.symbols('v_avg', positive=True)
        
        self.logs = []
        
    def _log(self, message, is_error=False):
        """Helper to log messages"""
        entry = f"{'ERROR' if is_error else 'INFO'}: {message}"
        self.logs.append(entry)
        return is_error
        
    def test_invariants(self):
        """Test physical invariants using symbolic and numerical checks"""
        errors = []
        
        try:
            # Symbolic invariant: Time addition formula
            t1_expr = (2*self.s_sym/3) / self.v_sym
            t2_expr = (self.s_sym/3) / self.v2_sym
            t_total_expr = t1_expr + t2_expr
            
            # Symbolic invariant: Average speed definition
            v_avg_expr = self.s_sym / t_total_expr
            
            # Simplify the expression
            v_avg_simplified = sp.simplify(v_avg_expr)
            
            # Should equal: v_avg = 1 / (2/(3v) + 1/(3v2))
            expected_expr = 1 / (2/(3*self.v_sym) + 1/(3*self.v2_sym))
            
            if not sp.simplify(v_avg_simplified - expected_expr) == 0:
                errors.append(self._log("Symbolic average speed formula does not match expected form", True))
            else:
                self._log("Symbolic average speed formula is correct")
                
        except Exception as e:
            errors.append(self._log(f"Symbolic invariant check failed: {str(e)}", True))
        
        # Numerical invariant check (fallback)
        try:
            if self.s is not None and self.v is not None:
                # Compute expected times
                t1_expected = (2*self.s/3) / self.v
                t2_expected = (self.s/3) / self.v2
                t_total_expected = t1_expected + t2_expected
                v_avg_expected = self.s / t_total_expected
                
                # Check if provided results match computed ones
                if self.t_1 is not None:
                    if not math.isclose(self.t_1, t1_expected, rel_tol=0.02, abs_tol=0.02):
                        errors.append(self._log(f"t_1 mismatch: expected {t1_expected}, got {self.t_1}", True))
                
                if self.t_2 is not None:
                    if not math.isclose(self.t_2, t2_expected, rel_tol=0.02, abs_tol=0.02):
                        errors.append(self._log(f"t_2 mismatch: expected {t2_expected}, got {self.t_2}", True))
                
                if self.t is not None:
                    if not math.isclose(self.t, t_total_expected, rel_tol=0.02, abs_tol=0.02):
                        errors.append(self._log(f"Total time mismatch: expected {t_total_expected}, got {self.t}", True))
                
                if self.v_avg is not None:
                    if not math.isclose(self.v_avg, v_avg_expected, rel_tol=0.02, abs_tol=0.02):
                        errors.append(self._log(f"Average speed mismatch: expected {v_avg_expected}, got {self.v_avg}", True))
            else:
                self._log("Insufficient data for numerical invariant check")
                
        except Exception as e:
            errors.append(self._log(f"Numerical invariant check failed: {str(e)}", True))
            
        return len(errors) == 0
        
    def test_trace_consistency(self):
        """Check consistency between intermediate and final results"""
        errors = []
        
        # Check 1: Total time should be sum of t1 and t2
        if self.t_1 is not None and self.t_2 is not None and self.t is not None:
            if not math.isclose(self.t, self.t_1 + self.t_2, rel_tol=0.02, abs_tol=0.02):
                errors.append(self._log(f"Time addition violated: t={self.t}, t1+t2={self.t_1+self.t_2}", True))
            else:
                self._log("Time addition consistent")
        
        # Check 2: Average speed should equal s/t
        if self.s is not None and self.t is not None and self.v_avg is not None:
            if not math.isclose(self.v_avg, self.s/self.t, rel_tol=0.02, abs_tol=0.02):
                errors.append(self._log(f"Average speed definition violated: v_avg={self.v_avg}, s/t={self.s/self.t}", True))
            else:
                self._log("Average speed definition consistent")
        
        # Check 3: t1 should satisfy t1 = (2s/3)/v
        if self.t_1 is not None and self.s is not None and self.v is not None:
            expected_t1 = (2*self.s/3) / self.v
            if not math.isclose(self.t_1, expected_t1, rel_tol=0.02, abs_tol=0.02):
                errors.append(self._log(f"t1 formula violated: t1={self.t_1}, expected={expected_t1}", True))
        
        # Check 4: t2 should satisfy t2 = (s/3)/5
        if self.t_2 is not None and self.s is not None:
            expected_t2 = (self.s/3) / 5.0
            if not math.isclose(self.t_2, expected_t2, rel_tol=0.02, abs_tol=0.02):
                errors.append(self._log(f"t2 formula violated: t2={self.t_2}, expected={expected_t2}", True))
        
        # Check 5: Final v should produce v_avg = 3 when plugged into formulas
        if self.v is not None and self.s is not None:
            # Compute expected average speed with given v
            t1_comp = (2*self.s/3) / self.v
            t2_comp = (self.s/3) / 5.0
            t_comp = t1_comp + t2_comp
            v_avg_comp = self.s / t_comp
            
            if not math.isclose(v_avg_comp, self.v_avg_target, rel_tol=0.02, abs_tol=0.02):
                errors.append(self._log(f"Final v={self.v} does not yield v_avg=3: got {v_avg_comp}", True))
            else:
                self._log(f"Final v={self.v} correctly yields v_avg≈3")
        
        return len(errors) == 0
        
    def test_monotonicity(self):
        """Test that increasing v decreases total time (and increases average speed)"""
        try:
            # Use a sample distance if not provided
            s_test = self.s if self.s is not None else 100.0
            
            # Original v (if available) or use the target value
            v_original = self.v if self.v is not None else 2.5
            
            # Test with perturbed v
            v_perturbed = v_original * 1.1  # Increase by 10%
            
            # Compute times and average speeds
            t1_orig = (2*s_test/3) / v_original
            t2_orig = (s_test/3) / self.v2
            t_orig = t1_orig + t2_orig
            v_avg_orig = s_test / t_orig
            
            t1_pert = (2*s_test/3) / v_perturbed
            t2_pert = (s_test/3) / self.v2
            t_pert = t1_pert + t2_pert
            v_avg_pert = s_test / t_pert
            
            # Check monotonicity: increasing v should decrease total time
            if t_pert > t_orig:
                return self._log(f"Monotonicity violated: t increased from {t_orig} to {t_pert} when v increased", True)
            
            # Increasing v should increase average speed
            if v_avg_pert < v_avg_orig:
                return self._log(f"Monotonicity violated: v_avg decreased from {v_avg_orig} to {v_avg_pert} when v increased", True)
            
            self._log(f"Monotonicity check passed: t decreased from {t_orig:.3f} to {t_pert:.3f}, v_avg increased from {v_avg_orig:.3f} to {v_avg_pert:.3f}")
            return True
            
        except Exception as e:
            return self._log(f"Monotonicity test failed: {str(e)}", True)
        
    def run_all_tests(self):
        """Run all tests and return success status and logs"""
        self.logs = []  # Reset logs
        success = True
        
        try:
            self._log("Starting PhysicsChecker tests")
            
            # Test 1: Invariants
            if not self.test_invariants():
                success = False
                self._log("Invariant tests failed", True)
            else:
                self._log("Invariant tests passed")
            
            # Test 2: Trace consistency
            if not self.test_trace_consistency():
                success = False
                self._log("Trace consistency tests failed", True)
            else:
                self._log("Trace consistency tests passed")
            
            # Test 3: Monotonicity
            if not self.test_monotonicity():
                success = False
                self._log("Monotonicity test failed", True)
            else:
                self._log("Monotonicity test passed")
            
            # Additional check: Final answer should be approximately 2.5 m/s
            if self.v is not None:
                if math.isclose(self.v, 2.5, rel_tol=0.02, abs_tol=0.02):
                    self._log(f"Final v={self.v} matches target 2.5 m/s")
                else:
                    success = False
                    self._log(f"Final v={self.v} does not match target 2.5 m/s", True)
            
        except Exception as e:
            success = False
            self._log(f"Unexpected error in test suite: {str(e)}", True)
            self._log(traceback.format_exc(), True)
        
        self._log(f"All tests {'PASSED' if success else 'FAILED'}")
        return success, self.logs