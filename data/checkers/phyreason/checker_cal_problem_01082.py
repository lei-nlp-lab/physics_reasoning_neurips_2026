import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs with fallbacks
        self.H = getattr(solution_instance, 'H', getattr(solution_instance, 'inputs', {}).get('H', 300.0))
        self.t1 = getattr(solution_instance, 't1', getattr(solution_instance, 'inputs', {}).get('t1', 6.0))
        self.g = getattr(solution_instance, 'g', getattr(solution_instance, 'inputs', {}).get('g', 10.0))
        
        # Safely extract results with fallbacks
        self.t_2 = getattr(solution_instance, 't_2', solution_instance.results.get('t_2', None))
        self.t = getattr(solution_instance, 't', solution_instance.results.get('t', None))
        self.v = getattr(solution_instance, 'v', solution_instance.results.get('v', None))
        self.h_1 = getattr(solution_instance, 'h_1', solution_instance.results.get('h_1', None))
        self.h_2 = getattr(solution_instance, 'h_2', solution_instance.results.get('h_2', None))
        
        self.logs = []
        
    def _log(self, msg, is_error=False):
        self.logs.append(f"{'ERROR' if is_error else 'INFO'}: {msg}")
        
    def _approx_equal(self, a, b, tol=1e-2):
        """Robust fuzzy comparison with 1% tolerance"""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)
    
    def test_invariants(self):
        """Verify fundamental physics laws are satisfied"""
        errors = []
        
        # Symbolic variables for validation
        h1, v, h2, t2, t, g, t1, H = sp.symbols('h1 v h2 t2 t g t1 H')
        
        # Check 1: Free fall displacement law h1 = 0.5*g*t1²
        try:
            expected_h1 = 0.5 * self.g * self.t1**2
            if not self._approx_equal(self.h_1, expected_h1):
                errors.append(f"Free fall displacement mismatch: h1={self.h_1}, expected {expected_h1}")
        except Exception as e:
            errors.append(f"Free fall displacement check failed: {str(e)}")
            
        # Check 2: Free fall velocity law v = g*t1
        try:
            expected_v = self.g * self.t1
            if not self._approx_equal(self.v, expected_v):
                errors.append(f"Free fall velocity mismatch: v={self.v}, expected {expected_v}")
        except Exception as e:
            errors.append(f"Free fall velocity check failed: {str(e)}")
            
        # Check 3: Height conservation H = h1 + h2
        try:
            if not self._approx_equal(self.H, self.h_1 + self.h_2):
                errors.append(f"Height conservation failed: H={self.H}, h1+h2={self.h_1 + self.h_2}")
        except Exception as e:
            errors.append(f"Height conservation check failed: {str(e)}")
            
        # Check 4: Uniform deceleration kinematics (v² = 2*a*h2 with a = v/t2)
        try:
            if self.t_2 and self.v and self.h_2:
                # From v_final = 0 and v_initial = v: v = a*t2 => a = v/t2
                # Also v² = 2*a*h2 => should satisfy v² = 2*(v/t2)*h2 => v = 2*h2/t2
                expected_v_from_h2 = 2 * self.h_2 / self.t_2
                if not self._approx_equal(self.v, expected_v_from_h2):
                    errors.append(f"Deceleration kinematics mismatch: v={self.v}, expected from h2/t2: {expected_v_from_h2}")
        except Exception as e:
            errors.append(f"Deceleration kinematics check failed: {str(e)}")
            
        # Check 5: Time addition t = t1 + t2
        try:
            if self.t and self.t1 and self.t_2:
                expected_t = self.t1 + self.t_2
                if not self._approx_equal(self.t, expected_t):
                    errors.append(f"Time addition mismatch: t={self.t}, expected t1+t2={expected_t}")
        except Exception as e:
            errors.append(f"Time addition check failed: {str(e)}")
            
        for err in errors:
            self._log(err, is_error=True)
            
        return len(errors) == 0
    
    def test_trace_consistency(self):
        """Verify intermediate calculations are consistent with final answer"""
        errors = []
        
        # Check that all steps from the analysis are internally consistent
        try:
            # Step 1: h1 should be 0.5*g*t1²
            expected_h1 = 0.5 * self.g * self.t1**2
            if not self._approx_equal(self.h_1, expected_h1):
                errors.append(f"Step 1 inconsistency: h1={self.h_1}, calculated {expected_h1}")
                
            # Step 2: v should be g*t1
            expected_v = self.g * self.t1
            if not self._approx_equal(self.v, expected_v):
                errors.append(f"Step 2 inconsistency: v={self.v}, calculated {expected_v}")
                
            # Step 3: h2 should be H - h1
            expected_h2 = self.H - self.h_1
            if not self._approx_equal(self.h_2, expected_h2):
                errors.append(f"Step 3 inconsistency: h2={self.h_2}, calculated {expected_h2}")
                
            # Step 4: t2 should satisfy h2 = 0.5*v*t2
            if self.h_2 and self.v:
                expected_t2 = 2 * self.h_2 / self.v if self.v != 0 else None
                if expected_t2 and not self._approx_equal(self.t_2, expected_t2):
                    errors.append(f"Step 4 inconsistency: t2={self.t_2}, calculated {expected_t2}")
                    
            # Step 5: t should be t1 + t2
            if self.t1 and self.t_2:
                expected_t = self.t1 + self.t_2
                if not self._approx_equal(self.t, expected_t):
                    errors.append(f"Step 5 inconsistency: t={self.t}, calculated {expected_t}")
                    
        except Exception as e:
            errors.append(f"Trace consistency check failed: {str(e)}")
            
        for err in errors:
            self._log(err, is_error=True)
            
        return len(errors) == 0
    
    def test_monotonicity(self):
        """Perturb inputs and check physical trends"""
        try:
            # Store original values
            orig_t1 = self.t1
            
            # Perturb t1 by +10%
            perturbed_t1 = self.t1 * 1.1
            
            # Calculate expected values with perturbed t1
            perturbed_h1 = 0.5 * self.g * perturbed_t1**2
            perturbed_v = self.g * perturbed_t1
            perturbed_h2 = self.H - perturbed_h1
            perturbed_t2 = 2 * perturbed_h2 / perturbed_v if perturbed_v != 0 else None
            perturbed_t = perturbed_t1 + perturbed_t2 if perturbed_t2 else None
            
            # Check monotonic trends
            if perturbed_h1 <= self.h_1:
                self._log(f"Monotonicity fail: h1 should increase with t1 (orig={self.h_1}, perturbed={perturbed_h1})", 
                         is_error=True)
                return False
                
            if perturbed_v <= self.v:
                self._log(f"Monotonicity fail: v should increase with t1 (orig={self.v}, perturbed={perturbed_v})", 
                         is_error=True)
                return False
                
            if perturbed_h2 >= self.h_2:
                self._log(f"Monotonicity fail: h2 should decrease with t1 (orig={self.h_2}, perturbed={perturbed_h2})", 
                         is_error=True)
                return False
                
            # Restore original value
            self.t1 = orig_t1
            
            self._log("Monotonicity test passed: All trends correct with t1 perturbation")
            return True
            
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {str(e)}", is_error=True)
            # Restore original value in case of error
            self.t1 = orig_t1 if 'orig_t1' in locals() else self.t1
            return False
    
    def run_all_tests(self):
        """Execute all tests and return summary"""
        self.logs = []
        success = True
        
        try:
            self._log("Starting Physics Audit...")
            
            # Run tests
            if not self.test_invariants():
                success = False
                self._log("Invariant tests failed")
                
            if not self.test_trace_consistency():
                success = False
                self._log("Trace consistency tests failed")
                
            if not self.test_monotonicity():
                success = False
                self._log("Monotonicity tests failed")
                
            if success:
                self._log("All physics checks passed!")
            else:
                self._log("Some physics checks failed", is_error=True)
                
        except Exception as e:
            self._log(f"Unhandled exception in physics audit: {str(e)}\n{traceback.format_exc()}", is_error=True)
            success = False
            
        return success, self.logs