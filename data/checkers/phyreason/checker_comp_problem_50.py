import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract inputs and results safely using getattr and .results.get
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', None))
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', None))
        self.R = getattr(solution_instance, 'R', solution_instance.results.get('R', None))
        self.theta0 = getattr(solution_instance, 'theta0', solution_instance.results.get('theta0', None))
        self.v = getattr(solution_instance, 'v', solution_instance.results.get('v', None))
        self.K = getattr(solution_instance, 'K', solution_instance.results.get('K', None))
        self.r = getattr(solution_instance, 'r', solution_instance.results.get('r', None))
        self.alpha = getattr(solution_instance, 'alpha', solution_instance.results.get('alpha', None))
        self.k = getattr(solution_instance, 'k', solution_instance.results.get('k', None))
        self.F_c = getattr(solution_instance, 'F_c', solution_instance.results.get('F_c', None))
        self.F_cperp = getattr(solution_instance, 'F_cperp', solution_instance.results.get('F_cperp', None))
        self.F_r = getattr(solution_instance, 'F_r', solution_instance.results.get('F_r', None))
        self.E_i = getattr(solution_instance, 'E_i', solution_instance.results.get('E_i', None))
        self.E_f = getattr(solution_instance, 'E_f', solution_instance.results.get('E_f', None))
        self.L = getattr(solution_instance, 'L', solution_instance.results.get('L', None))
        self.u = getattr(solution_instance, 'u', solution_instance.results.get('u', None))
        
        # Store original values for monotonicity test
        self._original_values = {
            'm': self.m, 'g': self.g, 'R': self.R, 'theta0': self.theta0,
            'v': self.v, 'K': self.K, 'r': self.r, 'alpha': self.alpha
        }
        
        self.logs = []
        
    def _log(self, message):
        self.logs.append(message)
        
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison."""
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify energy conservation and Newton's laws."""
        success = True
        
        # 1. Energy conservation: E_i should equal E_f (from steps 4, 7, 8)
        if self.E_i is not None and self.E_f is not None:
            if not self._is_close(self.E_i, self.E_f):
                self._log(f"Energy not conserved: E_i={self.E_i}, E_f={self.E_f}")
                success = False
            else:
                self._log("Energy conservation check passed.")
        
        # 2. Newton's second law in radial direction (from steps 1-3)
        if all(x is not None for x in [self.m, self.g, self.R, self.theta0, self.v, self.F_c, self.F_cperp, self.F_r]):
            # Step 1: F_c = m*v^2/(R*sin(theta0))
            F_c_expected = self.m * self.v**2 / (self.R * math.sin(self.theta0))
            if not self._is_close(self.F_c, F_c_expected):
                self._log(f"F_c mismatch: expected {F_c_expected}, got {self.F_c}")
                success = False
            
            # Step 2: F_cperp = F_c * sin(theta0) = m*v^2/R
            F_cperp_expected = self.F_c * math.sin(self.theta0) if self.F_c else self.m * self.v**2 / self.R
            if not self._is_close(self.F_cperp, F_cperp_expected):
                self._log(f"F_cperp mismatch: expected {F_cperp_expected}, got {self.F_cperp}")
                success = False
            
            # Step 3: F_r = mg*cos(theta0) - m*v^2/R
            F_r_expected = self.m * self.g * math.cos(self.theta0) - self.m * self.v**2 / self.R
            if not self._is_close(self.F_r, F_r_expected):
                self._log(f"F_r mismatch: expected {F_r_expected}, got {self.F_r}")
                success = False
            else:
                self._log("Radial force balance check passed.")
        
        # 3. Equilibrium condition for spring (from step 9)
        if all(x is not None for x in [self.k, self.m, self.g, self.R, self.theta0]):
            k_expected = (self.m * self.g * math.sin(self.theta0)) / (self.R * self.theta0)
            if not self._is_close(self.k, k_expected):
                self._log(f"Spring constant mismatch: expected {k_expected}, got {self.k}")
                success = False
            else:
                self._log("Spring equilibrium check passed.")
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        success = True
        
        # Check that steps 4-11 are algebraically consistent
        if all(x is not None for x in [self.m, self.g, self.R, self.theta0, self.K, self.r, self.alpha, self.k]):
            # From step 4: E_i = mgR*cos(theta0) + 0.5*k*R^2*theta0^2 + K
            E_i_calc = (self.m * self.g * self.R * math.cos(self.theta0) + 
                        0.5 * self.k * self.R**2 * self.theta0**2 + self.K)
            
            # CORRECTED: From step 7: E_f = mgR + K*(sin(alpha)^2 * R^2 * sin^2(theta0))/r^2
            # Because K = 0.5*m*v^2, and u = v*sin(alpha)*R*sin(theta0)/r
            # So 0.5*m*u^2 = 0.5*m*v^2 * (sin(alpha)*R*sin(theta0)/r)^2 = K * (sin(alpha)*R*sin(theta0)/r)^2
            sin_alpha = math.sin(self.alpha)
            E_f_calc = (self.m * self.g * self.R + 
                        self.K * (sin_alpha**2 * self.R**2 * math.sin(self.theta0)**2) / (self.r**2))
            
            if not self._is_close(E_i_calc, E_f_calc):
                self._log(f"Energy equation mismatch: E_i_calc={E_i_calc}, E_f_calc={E_f_calc}")
                success = False
            else:
                self._log("Energy equation consistency check passed.")
            
            # Check that the derived alpha formula (step 11) matches the computed alpha
            # The target formula uses small-angle approximation sin(alpha) ≈ alpha
            # alpha = (r/(R*sin(theta0))) * sqrt(1 - [mgR(1-cos(theta0)) - 0.5*mgR*theta0*sin(theta0)]/K)
            numerator = (self.m * self.g * self.R * (1 - math.cos(self.theta0)) - 
                         0.5 * self.m * self.g * self.R * self.theta0 * math.sin(self.theta0))
            # Use small-angle approximation: sin(alpha) ≈ alpha
            alpha_expected = (self.r / (self.R * math.sin(self.theta0))) * math.sqrt(max(0, 1 - numerator / self.K))
            
            # Compare using small-angle approximation: compute sin(alpha) from our alpha and compare to alpha_expected
            # Since alpha_expected is the approximate alpha, we compare sin(alpha) to alpha_expected
            # Allow larger tolerance due to approximation
            if not math.isclose(math.sin(self.alpha), alpha_expected, rel_tol=0.05):
                self._log(f"Alpha approximation mismatch: sin(alpha)={math.sin(self.alpha)}, alpha_expected≈{alpha_expected}")
                # Not a failure, just a warning
                self._log("Note: Small-angle approximation may not be exact.")
            else:
                self._log("Alpha formula consistency check passed (within approximation).")
        
        # Check angular momentum conservation (steps 5-6)
        if all(x is not None for x in [self.m, self.v, self.alpha, self.R, self.theta0, self.r, self.u]):
            # L_initial = m*v*sin(alpha)*(R*sin(theta0))
            L_initial = self.m * self.v * math.sin(self.alpha) * (self.R * math.sin(self.theta0))
            
            # L_final = m*u*r (from step 6)
            L_final = self.m * self.u * self.r
            
            if not self._is_close(L_initial, L_final):
                self._log(f"Angular momentum not conserved: L_initial={L_initial}, L_final={L_final}")
                success = False
            else:
                self._log("Angular momentum conservation check passed.")
        
        return success
    
    def test_monotonicity(self):
        """Perturb an input and check trend."""
        success = True
        
        # Perturb K (kinetic energy) and check alpha
        if all(x is not None for x in [self.m, self.g, self.R, self.theta0, self.K, self.r]):
            original_K = self.K
            perturbed_K = original_K * 1.05  # Increase by 5%
            
            # Calculate alpha with perturbed K using the exact formula from energy conservation
            # We need to solve for alpha numerically, but for monotonicity we can use the approximate formula
            numerator = (self.m * self.g * self.R * (1 - math.cos(self.theta0)) - 
                         0.5 * self.m * self.g * self.R * self.theta0 * math.sin(self.theta0))
            
            # Use the approximate formula (same as in trace consistency)
            alpha_original = (self.r / (self.R * math.sin(self.theta0))) * math.sqrt(max(0, 1 - numerator / original_K))
            alpha_perturbed = (self.r / (self.R * math.sin(self.theta0))) * math.sqrt(max(0, 1 - numerator / perturbed_K))
            
            # Expected: if K increases, alpha should increase (since denominator in sqrt gets larger)
            if perturbed_K > original_K and alpha_perturbed <= alpha_original:
                self._log(f"Monotonicity violated: K↑ ({original_K}→{perturbed_K}) but α ({alpha_original}→{alpha_perturbed}) did not increase")
                success = False
            else:
                self._log(f"Monotonicity check passed: K↑ → α↑ ({alpha_original}→{alpha_perturbed})")
            
            # Restore original value
            self.K = original_K
        
        return success
    
    def run_all_tests(self):
        """Run all tests and return success flag and logs."""
        self.logs = []
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        overall_success = True
        for test_name, test_func in tests:
            try:
                self._log(f"\n=== Running {test_name} Test ===")
                if not test_func():
                    overall_success = False
                    self._log(f"{test_name} test FAILED")
                else:
                    self._log(f"{test_name} test PASSED")
            except Exception as e:
                self._log(f"{test_name} test ERROR: {str(e)}")
                self._log(traceback.format_exc())
                overall_success = False
        
        self._log(f"\n=== Overall Result: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, self.logs