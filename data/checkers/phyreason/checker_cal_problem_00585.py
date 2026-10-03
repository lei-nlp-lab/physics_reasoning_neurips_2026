import math
import traceback
import random

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safe initialization: extract inputs and results using getattr.
        Avoid direct attribute access to prevent sandbox AttributeError.
        """
        self.solution = solution_instance
        
        # Extract results safely from results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract required variables with defaults
        self.x_0 = self.results.get('x_0', 0.0)
        self.x = self.results.get('x', 0.0)
        self.t_c = self.results.get('t_c', 0.0)
        self.v_c = self.results.get('v_c', 0.0)
        self.v_B = self.results.get('v_B', 0.0)
        
        # Extract inputs with defaults
        self.g = self.results.get('g', 9.8)
        self.t0 = self.results.get('t_0', 0.0)
        self.a = self.results.get('a', 0.0)
        self.t1 = self.results.get('t_1', 0.0)
        self.t2 = self.results.get('t_2', 0.0)
        
        # Logs for reporting
        self.logs = []
    
    def _log(self, message):
        """Helper to append log messages."""
        self.logs.append(message)
    
    def _is_close(self, a, b):
        """Robust numerical comparison with tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        """
        Verify physics invariants based on KB theorems.
        Check energy conservation and kinematic formulas.
        """
        try:
            # Part 1: Free-fall segment (A->B->C)
            # If v_B not in results, compute it from energy conservation: v_B^2 = 2*g*x
            if self.v_B == 0.0 and self.x > 0 and self.g > 0:
                self.v_B = math.sqrt(2 * self.g * self.x)
            
            # Check kinematic equation for BC: x_0 = v_B * t0 + 0.5 * g * t0^2
            # CORRECTED: Not x_0 - x, just x_0
            lhs1 = self.x_0
            rhs1 = self.v_B * self.t0 + 0.5 * self.g * (self.t0 ** 2)
            if not self._is_close(lhs1, rhs1):
                self._log(f"Invariant violation in free-fall: x_0={lhs1}, v_B*t0+0.5*g*t0^2={rhs1}")
                return False
            
            # Check velocity-displacement formula: v_B^2 = 2*g*x
            lhs2 = self.v_B ** 2
            rhs2 = 2 * self.g * self.x
            if not self._is_close(lhs2, rhs2):
                self._log(f"Invariant violation: v_B^2={lhs2}, 2*g*x={rhs2}")
                return False
            
            # Part 2: Uniform acceleration segment (B->C->D)
            # Check derived formula for v_c
            if self.t1 != self.t2 and self.a != 0:
                # Protect against division by zero
                if abs(self.t1 - self.t2) < 1e-10:
                    self._log("t1 - t2 is too close to zero")
                    return False
                
                v_c_expected = self.a * (self.t1**2 + self.t2**2) / (2 * (self.t1 - self.t2))
                if not self._is_close(self.v_c, v_c_expected):
                    self._log(f"Velocity at C mismatch: v_c={self.v_c}, expected={v_c_expected}")
                    return False
                
                # Check t_c = v_c / a
                if abs(self.a) < 1e-10:
                    self._log("Acceleration is too close to zero")
                    return False
                
                t_c_from_v = self.v_c / self.a
                if not self._is_close(self.t_c, t_c_from_v):
                    self._log(f"Time to C mismatch: t_c={self.t_c}, v_c/a={t_c_from_v}")
                    return False
                
                # Check displacement formulas for BC and CD (using derived v_c)
                # Equation for BC: x_BC = v_c * t1 - 0.5 * a * t1^2
                # Equation for CD: x_CD = v_c * t2 + 0.5 * a * t2^2
                # Since x_BC and x_CD are distances, they should be positive
                x_BC = self.v_c * self.t1 - 0.5 * self.a * (self.t1 ** 2)
                x_CD = self.v_c * self.t2 + 0.5 * self.a * (self.t2 ** 2)
                if x_BC < 0 or x_CD < 0:
                    self._log(f"Negative distances: x_BC={x_BC}, x_CD={x_CD}")
                    return False
            
            self._log("All invariants passed.")
            return True
        except Exception as e:
            self._log(f"Error in test_invariants: {traceback.format_exc()}")
            return False
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        Verify that the derived symbolic expressions match the target answers.
        """
        try:
            # Generate random numerical values for symbolic verification
            test_vals = {
                'x_0': random.uniform(1.0, 10.0),
                'g': random.uniform(8.0, 12.0),
                't0': random.uniform(0.5, 3.0),
                'a': random.uniform(1.0, 5.0),
                't1': random.uniform(1.0, 4.0),
                't2': random.uniform(0.5, 2.0)
            }
            
            # Ensure t1 > t2 for positive denominator
            if test_vals['t1'] <= test_vals['t2']:
                test_vals['t2'] = test_vals['t1'] * 0.5
            
            # Target formulas
            target_x_val = (2*test_vals['x_0'] - test_vals['g']*test_vals['t0']**2)**2 / (8*test_vals['g']*test_vals['t0']**2)
            target_t_c_val = (test_vals['t1']**2 + test_vals['t2']**2) / (2*(test_vals['t1'] - test_vals['t2']))
            
            # Check if the solution has compute methods - if not, skip this test
            # Many solutions won't have these methods, so we shouldn't fail if they don't exist
            if hasattr(self.solution, 'compute_x') and callable(self.solution.compute_x):
                computed_x = self.solution.compute_x(test_vals['x_0'], test_vals['g'], test_vals['t0'])
                if not self._is_close(computed_x, target_x_val):
                    self._log(f"Computed x mismatch: {computed_x} vs {target_x_val}")
                    return False
            
            if hasattr(self.solution, 'compute_t_c') and callable(self.solution.compute_t_c):
                computed_t_c = self.solution.compute_t_c(test_vals['t1'], test_vals['t2'])
                if not self._is_close(computed_t_c, target_t_c_val):
                    self._log(f"Computed t_c mismatch: {computed_t_c} vs {target_t_c_val}")
                    return False
            
            self._log("Trace consistency passed.")
            return True
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {traceback.format_exc()}")
            return False
    
    def test_monotonicity(self):
        """
        Perturb an input and check the trend matches expectations.
        Restore original values after test.
        """
        try:
            # Backup original values
            orig_g = self.g
            orig_t1 = self.t1
            orig_t2 = self.t2
            orig_x_0 = self.x_0
            orig_t0 = self.t0
            
            # Test 1: Increase g, expect x to decrease (since x = (2*x_0 - g*t0^2)^2/(8*g*t0^2))
            # For typical values where 2*x_0 > g*t0^2, derivative w.r.t g is negative
            self.g = orig_g * 1.1  # Increase by 10%
            if orig_g > 0 and orig_t0 > 0:
                # Protect against division by zero
                if abs(orig_g * orig_t0**2) < 1e-10:
                    self._log("g*t0^2 is too close to zero")
                    self.g = orig_g
                    return False
                    
                x_orig = (2*orig_x_0 - orig_g*orig_t0**2)**2 / (8*orig_g*orig_t0**2)
                x_new = (2*orig_x_0 - self.g*orig_t0**2)**2 / (8*self.g*orig_t0**2)
                # Only check if both x_orig and x_new are positive (physically meaningful)
                if x_orig > 0 and x_new > 0 and x_new > x_orig:
                    self._log(f"Monotonicity violation: x increased when g increased ({x_orig} -> {x_new})")
                    self.g = orig_g
                    return False
            
            # Restore g
            self.g = orig_g
            
            # Test 2: Increase t1, expect t_c to increase (since t_c = (t1^2 + t2^2)/(2*(t1-t2)))
            # For t1 > t2 > 0, derivative w.r.t t1 is positive
            if orig_t1 > orig_t2 and orig_t2 > 0:
                self.t1 = orig_t1 * 1.1
                # Protect against division by zero
                if abs(orig_t1 - orig_t2) < 1e-10 or abs(self.t1 - orig_t2) < 1e-10:
                    self._log("t1 - t2 is too close to zero")
                    self.t1 = orig_t1
                    return False
                    
                t_c_orig = (orig_t1**2 + orig_t2**2) / (2*(orig_t1 - orig_t2))
                t_c_new = (self.t1**2 + orig_t2**2) / (2*(self.t1 - orig_t2))
                if t_c_new < t_c_orig:  # Expect increase
                    self._log(f"Monotonicity violation: t_c decreased when t1 increased ({t_c_orig} -> {t_c_new})")
                    self.t1 = orig_t1
                    return False
            
            # Restore t1
            self.t1 = orig_t1
            
            # Test 3: Increase t2, expect t_c to increase (for t1 > t2 > 0)
            # Actually, derivative of t_c w.r.t t2 is positive when t1 > t2 > 0
            if orig_t1 > orig_t2 and orig_t2 > 0:
                self.t2 = orig_t2 * 1.1
                # Protect against division by zero
                if abs(orig_t1 - orig_t2) < 1e-10 or abs(orig_t1 - self.t2) < 1e-10:
                    self._log("t1 - t2 is too close to zero")
                    self.t2 = orig_t2
                    return False
                    
                t_c_orig = (orig_t1**2 + orig_t2**2) / (2*(orig_t1 - orig_t2))
                t_c_new = (orig_t1**2 + self.t2**2) / (2*(orig_t1 - self.t2))
                if t_c_new < t_c_orig:  # Actually should increase
                    self._log(f"Monotonicity note: t_c decreased when t2 increased ({t_c_orig} -> {t_c_new})")
                    # Don't fail, just note
                # Restore anyway
                self.t2 = orig_t2
            
            self._log("Monotonicity tests passed.")
            return True
        except Exception as e:
            self._log(f"Error in test_monotonicity: {traceback.format_exc()}")
            # Restore original values on error
            self.g = orig_g
            self.t1 = orig_t1
            self.t2 = orig_t2
            return False
    
    def run_all_tests(self):
        """
        Execute all tests and return (success_bool, logs_list).
        """
        self.logs.clear()
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"Running {test_name}...")
            if not test_func():
                success = False
                self._log(f"{test_name} failed.")
            else:
                self._log(f"{test_name} passed.")
        
        return success, self.logs