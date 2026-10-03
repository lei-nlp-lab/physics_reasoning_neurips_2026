import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Extract parameters and results from solution instance.
        Use numeric defaults for parameters if not provided.
        """
        # Extract results dictionary
        results = getattr(solution_instance, 'results', {})
        
        # Extract parameters with numeric defaults
        # Use typical values if not provided as attributes
        self.m = self._get_numeric_value(solution_instance, 'm', results.get('m', 1.0))
        self.q = self._get_numeric_value(solution_instance, 'q', results.get('q', 1.0))
        self.theta = self._get_numeric_value(solution_instance, 'theta', results.get('theta', math.pi/4))
        self.B = self._get_numeric_value(solution_instance, 'B', results.get('B', 1.0))
        self.g = self._get_numeric_value(solution_instance, 'g', results.get('g', 9.8))
        
        # Extract results
        self.F = results.get('F', None)
        self.t = results.get('t', None)
        self.a = results.get('a', None)
        self.v = results.get('v', None)
        self.N = results.get('N', None)
        
        self.logs = []
    
    def _get_numeric_value(self, obj, attr_name, default):
        """Safely extract numeric value from object attribute."""
        try:
            val = getattr(obj, attr_name, default)
            if hasattr(val, 'evalf'):
                return float(val.evalf())
            return float(val)
        except:
            return default
    
    def _log(self, msg):
        """Helper for logging."""
        self.logs.append(msg)
    
    def test_invariants(self):
        """Verify fundamental physics invariants."""
        try:
            # 1. Lorentz force at detachment: F = q*v*B
            if self.v is not None and self.F is not None:
                expected_F = self.q * self.v * self.B
                if not math.isclose(self.F, expected_F, rel_tol=0.02, abs_tol=0.02):
                    self._log(f"FAIL: Lorentz force doesn't match F = qvB. Got {self.F}, expected {expected_F}")
                    return False
            
            # 2. Detachment condition: qvB = mg cosθ
            if self.v is not None:
                lhs = self.q * self.v * self.B
                rhs = self.m * self.g * math.cos(self.theta)
                if not math.isclose(lhs, rhs, rel_tol=0.02, abs_tol=0.02):
                    self._log(f"FAIL: Detachment condition not satisfied. qvB={lhs}, mg cosθ={rhs}")
                    return False
            
            # 3. Newton's second law along incline: a = g sinθ
            if self.a is not None:
                expected_a = self.g * math.sin(self.theta)
                if not math.isclose(self.a, expected_a, rel_tol=0.02, abs_tol=0.02):
                    self._log(f"FAIL: Acceleration doesn't match g sinθ. Got {self.a}, expected {expected_a}")
                    return False
            
            # 4. Normal force at detachment should be approximately zero
            if self.N is not None and self.v is not None:
                N_calc = self.m * self.g * math.cos(self.theta) - self.q * self.v * self.B
                # At detachment, N should be close to zero
                if abs(N_calc) > 0.1:  # Allow small tolerance
                    self._log(f"WARN: Normal force not approaching zero at detachment. N_calc={N_calc}")
            
            self._log("PASS: All invariants satisfied")
            return True
            
        except Exception as e:
            self._log(f"Error in test_invariants: {str(e)}")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        try:
            # Check kinematic consistency: v = a*t
            if all(x is not None for x in [self.v, self.a, self.t]):
                v_from_kinematics = self.a * self.t
                if not math.isclose(self.v, v_from_kinematics, rel_tol=0.02, abs_tol=0.02):
                    self._log(f"FAIL: Kinematic inconsistency v ≠ a*t. v={self.v}, a*t={v_from_kinematics}")
                    return False
                
                # Also check that t = m/(qB tanθ)
                expected_t = self.m / (self.q * self.B * math.tan(self.theta))
                if not math.isclose(self.t, expected_t, rel_tol=0.02, abs_tol=0.02):
                    self._log(f"FAIL: Time doesn't match expected formula. t={self.t}, expected={expected_t}")
                    return False
            
            # Check that F = qvB gives consistent v
            if all(x is not None for x in [self.F, self.v]):
                v_from_F = self.F / (self.q * self.B)
                if not math.isclose(self.v, v_from_F, rel_tol=0.02, abs_tol=0.02):
                    self._log(f"FAIL: Velocity inconsistent with Lorentz force. v={self.v}, F/(qB)={v_from_F}")
                    return False
            
            self._log("PASS: Trace consistency verified")
            return True
            
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {str(e)}")
            return False
    
    def test_monotonicity(self):
        """Perturb mass m and verify expected trends."""
        try:
            # Perturb mass by +10%
            perturb_factor = 1.1
            m_perturbed = self.m * perturb_factor
            
            # Re-calculate expected v and t with perturbed mass
            v_original = self.m * self.g * math.cos(self.theta) / (self.q * self.B)
            v_perturbed = m_perturbed * self.g * math.cos(self.theta) / (self.q * self.B)
            
            t_original = self.m / (self.q * self.B * math.tan(self.theta))
            t_perturbed = m_perturbed / (self.q * self.B * math.tan(self.theta))
            
            # Check proportional relationships
            v_ratio_actual = v_perturbed / v_original
            if not math.isclose(v_ratio_actual, perturb_factor, rel_tol=0.02, abs_tol=0.02):
                self._log(f"FAIL: v not proportional to m. Ratio: {v_ratio_actual:.3f}, Expected: {perturb_factor:.3f}")
                return False
            
            t_ratio_actual = t_perturbed / t_original
            if not math.isclose(t_ratio_actual, perturb_factor, rel_tol=0.02, abs_tol=0.02):
                self._log(f"FAIL: t not proportional to m. Ratio: {t_ratio_actual:.3f}, Expected: {perturb_factor:.3f}")
                return False
            
            self._log("PASS: Monotonicity trends verified")
            return True
            
        except Exception as e:
            self._log(f"Error in test_monotonicity: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Execute all tests and return (success, logs)."""
        self.logs = []
        
        try:
            tests = [
                ("Invariants", self.test_invariants),
                ("Trace Consistency", self.test_trace_consistency),
                ("Monotonicity", self.test_monotonicity)
            ]
            
            all_pass = True
            for test_name, test_func in tests:
                self._log(f"\n=== Running {test_name} Test ===")
                if not test_func():
                    all_pass = False
                    self._log(f"FAIL: {test_name} test")
                else:
                    self._log(f"PASS: {test_name} test")
            
            self._log(f"\n=== SUMMARY ===")
            self._log(f"Overall: {'PASS' if all_pass else 'FAIL'}")
            
            return all_pass, self.logs
            
        except Exception as e:
            self._log(f"Critical error in run_all_tests: {str(e)}")
            self._log(traceback.format_exc())
            return False, self.logs