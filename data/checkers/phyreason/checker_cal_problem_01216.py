import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallback to results dictionary
        self.solution = solution_instance
        
        # Inputs from problem statement (safe extraction with defaults)
        self.v1 = getattr(solution_instance, 'v1', 8.0)  # m/s, car A constant speed
        self.v2 = getattr(solution_instance, 'v2', 20.0) # m/s, car B initial speed
        self.x0 = getattr(solution_instance, 'x0', 28.0) # m, initial separation
        self.a_B = getattr(solution_instance, 'a', -2.0) # m/s², B's deceleration (negative)
        self.a_mag = abs(self.a_B)  # magnitude for formulas
        
        # Extract results using safe access pattern
        self.t1 = self._safe_get('t_1')
        self.xA = self._safe_get('x_A')
        self.xB = self._safe_get('x_B')
        self.Deltax_m = self._safe_get('Deltax_m')
        self.t0 = self._safe_get('t_0')
        self.xBprime = self._safe_get('x_Bprime')
        self.xAprime = self._safe_get('x_Aprime')
        self.t2 = self._safe_get('t_2')
        self.t_total = self._safe_get('t')
        self.a_A = self._safe_get('a_A')  # minimum acceleration of A
        
        # Tolerance for fuzzy matching (2% relative, 0.02 absolute)
        self.rtol = 1e-2
        self.atol = 1e-2
        
        # Log storage
        self.logs = []
        
    def _safe_get(self, key):
        """Safely get attribute from solution instance with fallback to results dict."""
        # Map LaTeX keys to variable names used in solution instance
        key_map = {
            't_1': 't_1',
            'x_A': 'x_A',
            'x_B': 'x_B',
            'Deltax_m': 'Deltax_m',
            't_0': 't_0',
            'x_Bprime': 'x_Bprime',
            'x_Aprime': 'x_Aprime',
            't_2': 't_2',
            't': 't',
            'a_A': 'a_A'
        }
        
        var_name = key_map.get(key, key)
        
        # Try direct attribute, then results dict, then default to None
        value = getattr(self.solution, var_name, 
                       getattr(self.solution, 'results', {}).get(var_name, None))
        
        # Convert to float if it's a sympy expression or string
        if hasattr(value, 'evalf'):
            value = float(value.evalf())
        elif isinstance(value, str):
            try:
                value = float(value)
            except:
                pass
                
        return value
    
    def _log(self, message, success=None):
        """Add message to log with optional success flag."""
        if success is not None:
            prefix = "✓" if success else "✗"
            self.logs.append(f"{prefix} {message}")
        else:
            self.logs.append(f"  {message}")
    
    def _is_close(self, a, b):
        """Fuzzy numerical comparison with tolerance."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=self.rtol, abs_tol=self.atol)
    
    def test_invariants(self):
        """Verify kinematic formulas and physical invariants."""
        self.logs.append("=== Testing Invariants ===")
        success = True
        
        try:
            # Check step 1: Time when speeds are equal
            if self.t1 is not None:
                t1_expected = (self.v2 - self.v1) / self.a_mag
                if self._is_close(self.t1, t1_expected):
                    self._log(f"t1 correct: {self.t1} ≈ {t1_expected}", True)
                else:
                    self._log(f"t1 mismatch: {self.t1} ≠ {t1_expected}", False)
                    success = False
            else:
                self._log("t1 not provided", None)
            
            # Check step 2: Displacements at t1
            if self.xA is not None:
                xA_expected = self.v1 * (self.t1 if self.t1 is not None else 6.0)
                if self._is_close(self.xA, xA_expected):
                    self._log(f"xA correct: {self.xA} ≈ {xA_expected}", True)
                else:
                    self._log(f"xA mismatch: {self.xA} ≠ {xA_expected}", False)
                    success = False
            
            if self.xB is not None:
                if self.t1 is not None:
                    xB_expected = self.v2 * self.t1 + 0.5 * self.a_B * self.t1**2
                    if self._is_close(self.xB, xB_expected):
                        self._log(f"xB correct: {self.xB} ≈ {xB_expected}", True)
                    else:
                        self._log(f"xB mismatch: {self.xB} ≠ {xB_expected}", False)
                        success = False
                else:
                    self._log("xB check skipped (needs t1)", None)
            
            # Check step 3: Maximum distance
            if self.Deltax_m is not None and self.xA is not None and self.xB is not None:
                Deltax_expected = self.xB + self.x0 - self.xA
                if self._is_close(self.Deltax_m, Deltax_expected):
                    self._log(f"Δx_m correct: {self.Deltax_m} ≈ {Deltax_expected}", True)
                else:
                    self._log(f"Δx_m mismatch: {self.Deltax_m} ≠ {Deltax_expected}", False)
                    success = False
            
            # Check step 4: Time for B to stop
            if self.t0 is not None:
                t0_expected = -self.v2 / self.a_B  # since a_B is negative
                if self._is_close(self.t0, t0_expected):
                    self._log(f"t0 correct: {self.t0} ≈ {t0_expected}", True)
                else:
                    self._log(f"t0 mismatch: {self.t0} ≠ {t0_expected}", False)
                    success = False
            
            # Check step 5: Displacement of B until stop
            if self.xBprime is not None:
                xBprime_expected = self.v2 * self.t0 + 0.5 * self.a_B * self.t0**2 if self.t0 else 100.0
                if self._is_close(self.xBprime, xBprime_expected):
                    self._log(f"xB' correct: {self.xBprime} ≈ {xBprime_expected}", True)
                else:
                    self._log(f"xB' mismatch: {self.xBprime} ≠ {xBprime_expected}", False)
                    success = False
            
            # Check step 6: Displacement of A until B stops
            if self.xAprime is not None:
                xAprime_expected = self.v1 * self.t0 if self.t0 else 80.0
                if self._is_close(self.xAprime, xAprime_expected):
                    self._log(f"xA' correct: {self.xAprime} ≈ {xAprime_expected}", True)
                else:
                    self._log(f"xA' mismatch: {self.xAprime} ≠ {xAprime_expected}", False)
                    success = False
            
            # Check step 9: Time for A to catch up after B stops
            if self.t2 is not None and self.xBprime is not None and self.xAprime is not None:
                t2_expected = (self.x0 + self.xBprime - self.xAprime) / self.v1
                if self._is_close(self.t2, t2_expected):
                    self._log(f"t2 correct: {self.t2} ≈ {t2_expected}", True)
                else:
                    self._log(f"t2 mismatch: {self.t2} ≠ {t2_expected}", False)
                    success = False
            
            # Check step 10: Total time
            if self.t_total is not None and self.t0 is not None and self.t2 is not None:
                t_total_expected = self.t0 + self.t2
                if self._is_close(self.t_total, t_total_expected):
                    self._log(f"Total time correct: {self.t_total} ≈ {t_total_expected}", True)
                else:
                    self._log(f"Total time mismatch: {self.t_total} ≠ {t_total_expected}", False)
                    success = False
            
            # Check step 12: Minimum acceleration of A
            if self.a_A is not None:
                # Expected value from problem: 8/31 ≈ 0.2580645
                a_A_expected = 8.0/31.0
                if self._is_close(self.a_A, a_A_expected):
                    self._log(f"a_A correct: {self.a_A} ≈ {a_A_expected}", True)
                else:
                    self._log(f"a_A mismatch: {self.a_A} ≠ {a_A_expected}", False)
                    success = False
            
        except Exception as e:
            self._log(f"Error in invariants: {str(e)}", False)
            success = False
            
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        self.logs.append("\n=== Testing Trace Consistency ===")
        success = True
        
        # Check that all required results are present
        required = ['Deltax_m', 't', 'a_A']
        missing = [key for key in required if self._safe_get(key) is None]
        if missing:
            self._log(f"Missing required results: {missing}", False)
            return False
        
        # Check target answers from problem statement
        target_Deltax_m = 64.0
        target_t = 16.0
        target_a_A = 8.0/31.0
        
        checks = [
            ("Δx_m", self.Deltax_m, target_Deltax_m),
            ("Total time t", self.t_total, target_t),
            ("a_A", self.a_A, target_a_A)
        ]
        
        for name, actual, expected in checks:
            if self._is_close(actual, expected):
                self._log(f"{name} matches target: {actual} ≈ {expected}", True)
            else:
                self._log(f"{name} doesn't match target: {actual} ≠ {expected}", False)
                success = False
        
        # Verify internal consistency: if t0 and t2 are given, check t = t0 + t2
        if self.t0 is not None and self.t2 is not None and self.t_total is not None:
            if self._is_close(self.t_total, self.t0 + self.t2):
                self._log(f"t = t0 + t2: {self.t_total} ≈ {self.t0} + {self.t2}", True)
            else:
                self._log(f"t ≠ t0 + t2: {self.t_total} ≠ {self.t0} + {self.t2}", False)
                success = False
        
        return success
    
    def test_monotonicity(self):
        """Perturb inputs and check expected trends."""
        self.logs.append("\n=== Testing Monotonicity ===")
        success = True
        
        try:
            # Store original values
            original_v1 = self.v1
            perturbations = [(1.0, "increase"), (-1.0, "decrease")]
            
            for delta, direction in perturbations:
                # Perturb v1
                self.v1 = original_v1 + delta
                
                # Recalculate expected changes using formulas
                # 1. t1 = (v2 - v1)/a_mag
                new_t1 = (self.v2 - self.v1) / self.a_mag
                
                # 2. Δx_m = xB + x0 - xA = (v2*t1 - 0.5*a_mag*t1²) + x0 - v1*t1
                # Simplify: Δx_m = (v2 - v1)*t1 - 0.5*a_mag*t1² + x0
                new_Deltax_m = (self.v2 - self.v1)*new_t1 - 0.5*self.a_mag*new_t1**2 + self.x0
                
                # 3. t0 remains same (depends only on v2 and a_B)
                # 4. t2 = (x0 + xB' - xA')/v1 = (x0 + v2²/(2*a_mag) - v1*t0)/v1
                new_t2 = (self.x0 + self.v2**2/(2*self.a_mag) - self.v1*self.t0) / self.v1
                new_t_total = self.t0 + new_t2
                
                # Check monotonic trends
                if delta > 0:  # v1 increased
                    # t1 should decrease (denominator increases)
                    if new_t1 < (self.t1 or 6.0):
                        self._log(f"t1 decreases when v1 increases (expected)", True)
                    else:
                        self._log(f"t1 doesn't decrease when v1 increases", False)
                        success = False
                    
                    # Δx_m should decrease (A is faster, catches up sooner)
                    if new_Deltax_m < (self.Deltax_m or 64.0):
                        self._log(f"Δx_m decreases when v1 increases (expected)", True)
                    else:
                        self._log(f"Δx_m doesn't decrease when v1 increases", False)
                        success = False
                    
                    # Total time should decrease
                    if new_t_total < (self.t_total or 16.0):
                        self._log(f"Total time decreases when v1 increases (expected)", True)
                    else:
                        self._log(f"Total time doesn't decrease when v1 increases", False)
                        success = False
                
                else:  # v1 decreased
                    # Opposite trends
                    if new_t1 > (self.t1 or 6.0):
                        self._log(f"t1 increases when v1 decreases (expected)", True)
                    else:
                        self._log(f"t1 doesn't increase when v1 decreases", False)
                        success = False
                    
                    if new_Deltax_m > (self.Deltax_m or 64.0):
                        self._log(f"Δx_m increases when v1 decreases (expected)", True)
                    else:
                        self._log(f"Δx_m doesn't increase when v1 decreases", False)
                        success = False
                    
                    if new_t_total > (self.t_total or 16.0):
                        self._log(f"Total time increases when v1 decreases (expected)", True)
                    else:
                        self._log(f"Total time doesn't increase when v1 decreases", False)
                        success = False
                
                # Restore original v1 for next iteration
                self.v1 = original_v1
            
        except Exception as e:
            self._log(f"Error in monotonicity test: {str(e)}", False)
            success = False
            # Restore original values on error
            self.v1 = original_v1
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return success flag with logs."""
        self.logs = []  # Clear previous logs
        
        try:
            # Run individual tests
            test1_success = self.test_invariants()
            test2_success = self.test_trace_consistency()
            test3_success = self.test_monotonicity()
            
            overall_success = test1_success and test2_success and test3_success
            
            # Summary
            self.logs.append("\n=== SUMMARY ===")
            self.logs.append(f"Invariants: {'PASS' if test1_success else 'FAIL'}")
            self.logs.append(f"Trace Consistency: {'PASS' if test2_success else 'FAIL'}")
            self.logs.append(f"Monotonicity: {'PASS' if test3_success else 'FAIL'}")
            self.logs.append(f"OVERALL: {'PASS' if overall_success else 'FAIL'}")
            
        except Exception as e:
            self.logs.append(f"\n=== ERROR ===")
            self.logs.append(f"Test execution failed: {str(e)}")
            self.logs.append(traceback.format_exc())
            overall_success = False
        
        return overall_success, self.logs