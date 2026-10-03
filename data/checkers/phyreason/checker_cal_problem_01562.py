import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract parameters from results dictionary with defaults
        self.m = solution_instance.results.get('m', None)
        self.Q = solution_instance.results.get('Q', None)
        self.q = solution_instance.results.get('q', None)
        self.r = solution_instance.results.get('r', None)
        self.g = solution_instance.results.get('g', 9.8)
        self.k = solution_instance.results.get('k', 8.99e9)
        self.v0 = solution_instance.results.get('v0', None)
        
        # Extract results
        self.N = solution_instance.results.get('N', None)
        self.N_prime = solution_instance.results.get("N'", None)
        self.v = solution_instance.results.get('v', None)
        
        # Store original values for monotonicity tests
        self._original_values = {
            'm': self.m,
            'Q': self.Q,
            'q': self.q,
            'r': self.r,
            'g': self.g,
            'k': self.k,
            'v0': self.v0,
            'N': self.N,
            'N_prime': self.N_prime,
            'v': self.v
        }
        
        # Logs for test results
        self.logs = []
        
    def _log(self, message, success):
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {message}")
        return success
        
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison with tolerance."""
        if a is None or b is None:
            return False
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False
            
    def test_invariants(self):
        """Test fundamental physics laws: centripetal force balance at points A and B using numeric checks."""
        success = True
        
        try:
            # Check centripetal force balance at point B (lowest point)
            # From step_1: N + kQq/r² - mg = mv0²/r
            if None not in [self.N, self.v0, self.m, self.r, self.g, self.k, self.Q, self.q]:
                lhs_b = self.N + self.k * self.Q * self.q / self.r**2 - self.m * self.g
                rhs_b = self.m * self.v0**2 / self.r
                if not self._is_close(lhs_b, rhs_b):
                    success = self._log(f"Centripetal force balance at point B failed: lhs={lhs_b}, rhs={rhs_b}", False)
                else:
                    self._log("Centripetal force balance at point B passed", True)
            else:
                success = self._log("Centripetal force balance at point B check skipped (missing values)", True)
            
            # Check centripetal force balance at point A (highest point)
            # From step_3: kQq/r² + mg = mv²/r
            if None not in [self.v, self.m, self.r, self.g, self.k, self.Q, self.q]:
                lhs_a = self.k * self.Q * self.q / self.r**2 + self.m * self.g
                rhs_a = self.m * self.v**2 / self.r
                if not self._is_close(lhs_a, rhs_a):
                    success = self._log(f"Centripetal force balance at point A failed: lhs={lhs_a}, rhs={rhs_a}", False)
                else:
                    self._log("Centripetal force balance at point A passed", True)
            else:
                success = self._log("Centripetal force balance at point A check skipped (missing values)", True)
                
        except Exception as e:
            success = self._log(f"Invariant test failed with error: {str(e)}", False)
            
        return success
        
    def test_trace_consistency(self):
        """Check consistency between intermediate results and final answers."""
        success = True
        
        try:
            # Check N and N' are equal (Newton's 3rd Law)
            if self.N is not None and self.N_prime is not None:
                if not self._is_close(self.N, self.N_prime):
                    success = self._log(f"N and N' are not equal: N={self.N}, N'={self.N_prime}", False)
                else:
                    self._log("N and N' are consistent", True)
            else:
                success = self._log("N and N' check skipped (missing values)", True)
            
            # Check that v expression matches the derived formula
            if None not in [self.v, self.m, self.r, self.g, self.k, self.Q, self.q]:
                expected_v = math.sqrt(self.k * self.Q * self.q / (self.m * self.r) + self.g * self.r)
                if not self._is_close(self.v, expected_v):
                    success = self._log(f"Minimum speed v does not match derived formula: v={self.v}, expected={expected_v}", False)
                else:
                    self._log("Minimum speed v matches derived formula", True)
            else:
                success = self._log("v formula check skipped (missing values)", True)
                    
        except Exception as e:
            success = self._log(f"Trace consistency test failed with error: {str(e)}", False)
            
        return success
        
    def test_monotonicity(self):
        """Perturb mass m and check expected trends in N and v using numeric calculations."""
        success = True
        
        try:
            # Only proceed if we have all necessary values
            if None in [self.m, self.Q, self.q, self.r, self.g, self.k, self.v0, self.N, self.v]:
                self._log("Monotonicity test skipped due to missing values", True)
                return True
                
            # Perturb mass by +1%
            perturbed_m = self.m * 1.01
            
            # Calculate expected N with perturbed mass (keeping v0 constant)
            # N = m*v0²/r + m*g - kQq/r²
            N_perturbed = perturbed_m * (self.v0**2 / self.r + self.g) - self.k * self.Q * self.q / self.r**2
            
            # Calculate expected v with perturbed mass
            # v = sqrt(kQq/(m*r) + g*r)
            v_perturbed = math.sqrt(self.k * self.Q * self.q / (perturbed_m * self.r) + self.g * self.r)
            
            # Check trends
            # For fixed v0, N should increase with m (since dN/dm = v0²/r + g > 0)
            if N_perturbed <= self.N:
                success = self._log(f"N should increase with mass but didn't: N={self.N}, N_perturbed={N_perturbed}", False)
            else:
                self._log("N increases with mass as expected", True)
                
            # v should decrease with increasing m (since dv/dm < 0 for positive parameters)
            if v_perturbed >= self.v:
                success = self._log(f"v should decrease with mass but didn't: v={self.v}, v_perturbed={v_perturbed}", False)
            else:
                self._log("v decreases with mass as expected", True)
                
            # Restore original values
            self.m = self._original_values['m']
            
        except Exception as e:
            success = self._log(f"Monotonicity test failed with error: {str(e)}", False)
            # Restore original values even if test fails
            if 'm' in self._original_values:
                self.m = self._original_values['m']
                
        return success
        
    def run_all_tests(self):
        """Run all tests and return success boolean and logs."""
        self.logs = []  # Reset logs
        
        overall_success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"Running {test_name} test...")
            if not test_func():
                overall_success = False
                
        return overall_success, self.logs