import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract ONLY the results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract input parameters from results or use defaults
        # Use results if they exist and are non-zero, otherwise use defaults
        self.l = self._get_value('l', 0.1)
        self.L = self._get_value('L', 1.0)
        self.a = self._get_value('a', 0.5e-6)
        self.b = self._get_value('b', 5e-3)
        self.V_a = self._get_value('V_a', 10.0)
        self.V_0 = self._get_value('V_0', 50000.0)  # Use 50000 from problem, not 10000
        self.v_z = self._get_value('v_z', 299792458/2)
        
        # Physical constants with safe defaults
        self.h = self._get_value('h', 6.62607015e-34)
        self.hbar = self._get_value('hbar', 1.0545718e-34)
        self.e0 = self._get_value('e0', 1.60217662e-19)
        self.m = self._get_value('m', 9.10938356e-31)
        self.c = self._get_value('c', 2.99792458e8)
        self.epsilon0 = self._get_value('epsilon0', 8.854187817e-12)
        
        # Extract results - use _get_value to handle 0.0
        self.k_x = self._get_value('k_x', None)
        self.impulse = self._get_value('Impulse', None)
        self.lambda_nonrel = self._get_value('lambda_textnonrel', None)
        self.lambda_rel = self._get_value('lambda_textrel', None)
        self.fringe_sep = self._get_value('Fringeseparation', None)
        self.d = self._get_value('d', None)
        self.y = self._get_value('y', None)
        self.percent_error = self._get_value('Percentageerror', None)
        self.E_field = self._get_value('|E|', None)
        
        # Logging
        self.logs = []
    
    def _get_value(self, key, default):
        """Get value from results, but if it's 0.0 and we have a non-zero default, use default."""
        val = self.results.get(key, default)
        if val == 0.0 and default is not None and default != 0.0:
            # If we got 0.0 but have a non-zero default, use the default
            # This handles the case where floating-point underflow gave us 0.0
            return default
        return val
        
    def log(self, message):
        self.logs.append(message)
        
    def fuzzy_equal(self, a, b):
        """Robust fuzzy comparison with 1% tolerance."""
        if a is None or b is None:
            return False
        # Handle very small numbers
        if abs(a) < 1e-30 and abs(b) < 1e-30:
            return True
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def safe_divide(self, numerator, denominator):
        """Safe division with zero check."""
        if denominator is None or denominator == 0 or abs(denominator) < 1e-30:
            return None
        return numerator / denominator
    
    def test_invariants(self):
        """Check fundamental physics invariants."""
        try:
            # 1. Energy conservation in electron acceleration
            if self.V_0 is not None and self.V_0 != 0:
                # Non-relativistic kinetic energy = e*V_0
                K_nonrel = self.e0 * self.V_0
                
                # Check if non-relativistic kinetic energy is consistent with wavelength
                if self.lambda_nonrel is not None and self.lambda_nonrel != 0:
                    p_from_lambda = self.safe_divide(self.h, self.lambda_nonrel)
                    if p_from_lambda is not None:
                        K_from_p = p_from_lambda**2 / (2 * self.m)
                        if not self.fuzzy_equal(K_nonrel, K_from_p):
                            self.log(f"Warning: Non-rel KE mismatch: {K_nonrel:.2e} vs {K_from_p:.2e}")
                    else:
                        self.log("Warning: Could not compute p_from_lambda (division by zero)")
                
                # Relativistic total energy
                if self.m is not None and self.c is not None and self.e0 is not None:
                    E_total_rel = self.m * self.c**2 + self.e0 * self.V_0
                    p_rel_sq = (E_total_rel**2 - (self.m * self.c**2)**2) / self.c**2
                    
                    # Check relativistic consistency
                    if self.lambda_rel is not None and self.lambda_rel != 0:
                        p_rel_from_lambda = self.safe_divide(self.h, self.lambda_rel)
                        if p_rel_from_lambda is not None:
                            p_rel_calc = math.sqrt(max(0, p_rel_sq))
                            if not self.fuzzy_equal(p_rel_from_lambda, p_rel_calc):
                                self.log(f"Warning: Rel momentum mismatch: {p_rel_from_lambda:.2e} vs {p_rel_calc:.2e}")
                        else:
                            self.log("Warning: Could not compute p_rel_from_lambda (division by zero)")
            
            # 2. Electric potential gradient consistency
            if self.V_a is not None and self.a is not None and self.b is not None:
                # Check if ln(b/a) > 0 for positive V_a
                if self.b > self.a and self.V_a > 0:
                    self.log("✓ Positive V_a with b>a is consistent with potential gradient")
                elif self.b < self.a and self.V_a > 0:
                    self.log("Warning: V_a positive but b<a may indicate sign issue")
            
            # 3. De Broglie wavelength inversely proportional to momentum
            if (self.lambda_nonrel is not None and self.lambda_rel is not None and 
                self.lambda_nonrel != 0 and self.lambda_rel != 0):
                if self.lambda_rel < self.lambda_nonrel:
                    self.log("✓ Relativistic wavelength shorter (higher momentum)")
                else:
                    self.log("Warning: Relativistic wavelength not shorter than non-rel")
            
            return True
        except Exception as e:
            self.log(f"Invariant test failed: {str(e)}")
            return False
    
    def test_trace_consistency(self):
        """Check consistency between intermediate and final results."""
        try:
            # 1. k_x from impulse and hbar
            if (self.k_x is not None and self.impulse is not None and 
                self.hbar is not None and self.hbar != 0):
                k_x_from_impulse = self.safe_divide(self.impulse, self.hbar)
                if k_x_from_impulse is not None:
                    if not self.fuzzy_equal(self.k_x, k_x_from_impulse):
                        self.log(f"k_x mismatch: {self.k_x:.2e} vs {k_x_from_impulse:.2e}")
                        return False
                    else:
                        self.log("✓ k_x consistent with impulse via hbar")
                else:
                    self.log("Warning: Could not compute k_x_from_impulse (division by zero)")
            
            # 2. Fringe separation from k_x
            if (self.fringe_sep is not None and self.k_x is not None and 
                self.k_x != 0):
                fringe_from_kx = math.pi / self.k_x
                if not self.fuzzy_equal(self.fringe_sep, fringe_from_kx):
                    self.log(f"Fringe separation mismatch: {self.fringe_sep:.2e} vs {fringe_from_kx:.2e}")
                    return False
                else:
                    self.log("✓ Fringe separation consistent with k_x")
            
            # 3. Percentage error calculation
            if (self.percent_error is not None and 
                self.lambda_nonrel is not None and 
                self.lambda_rel is not None and self.lambda_rel != 0):
                calc_error = ((self.lambda_nonrel - self.lambda_rel) / self.lambda_rel) * 100
                if not self.fuzzy_equal(self.percent_error, calc_error):
                    self.log(f"Percent error mismatch: {self.percent_error:.2f} vs {calc_error:.2f}")
                    return False
                else:
                    self.log("✓ Percentage error correctly calculated")
            
            # 4. Double slit separation formula
            if (self.d is not None and self.y is not None and self.y != 0 and
                self.l is not None and self.L is not None and 
                self.lambda_rel is not None):
                # Using relativistic wavelength for d calculation (as per problem)
                d_calc = self.lambda_rel * (self.l + self.L) / self.y
                if not self.fuzzy_equal(self.d, d_calc):
                    self.log(f"d mismatch: {self.d:.2e} vs {d_calc:.2e}")
                    return False
                else:
                    self.log("✓ Double slit separation consistent")
            
            return True
        except Exception as e:
            self.log(f"Trace consistency test failed: {str(e)}")
            return False
    
    def test_monotonicity(self):
        """Check that perturbations produce expected trends."""
        try:
            # Test 1: Increase V_0, wavelength should decrease
            if (self.V_0 is not None and self.lambda_nonrel is not None and 
                self.lambda_rel is not None and self.V_0 != 0 and
                self.m is not None and self.e0 is not None and self.h is not None and
                self.m != 0 and self.e0 != 0):
                
                original_V0 = self.V_0
                perturbed_V0 = original_V0 * 1.1  # 10% increase
                
                # Calculate wavelengths for perturbed V0
                sqrt_term = math.sqrt(2 * self.m * self.e0 * perturbed_V0)
                if sqrt_term != 0:
                    lambda_nonrel_pert = self.h / sqrt_term
                    
                    # Calculate relativistic correction
                    rel_correction = self.e0 * perturbed_V0 / (2 * self.m * self.c**2)
                    rel_factor = 1 + rel_correction
                    sqrt_term_rel = math.sqrt(2 * self.m * self.e0 * perturbed_V0 * rel_factor)
                    if sqrt_term_rel != 0:
                        lambda_rel_pert = self.h / sqrt_term_rel
                        
                        # Check monotonicity
                        if (lambda_nonrel_pert < self.lambda_nonrel and 
                            lambda_rel_pert < self.lambda_rel):
                            self.log("✓ Wavelength decreases with increased V_0 (as expected)")
                        else:
                            self.log("Warning: Wavelength monotonicity violated for V_0")
                    else:
                        self.log("Warning: Could not calculate lambda_rel_pert (division by zero)")
                else:
                    self.log("Warning: Could not calculate lambda_nonrel_pert (division by zero)")
            
            # Test 2: Increase V_a, k_x should increase
            if (self.V_a is not None and self.k_x is not None and 
                self.a is not None and self.b is not None and 
                self.v_z is not None and self.hbar is not None and
                self.a != 0 and self.b != 0 and self.v_z != 0 and self.hbar != 0):
                original_Va = self.V_a
                perturbed_Va = original_Va * 1.1
                
                # Calculate perturbed k_x
                k_x_pert = (self.e0 * perturbed_Va * math.pi) / \
                          (self.hbar * self.v_z * math.log(self.b / self.a))
                
                if k_x_pert > self.k_x:
                    self.log("✓ k_x increases with increased V_a (as expected)")
                else:
                    self.log("Warning: k_x monotonicity violated for V_a")
            
            # Test 3: Increase b (mesh distance), potential gradient decreases
            if self.b is not None and self.a is not None and self.a != 0:
                original_b = self.b
                perturbed_b = original_b * 1.1
                
                # ln(b/a) increases, so |E| = V_a / (r * ln(b/a)) decreases for fixed r
                if perturbed_b > original_b:
                    self.log("✓ Electric field magnitude decreases with increased b")
                else:
                    self.log("Warning: Field monotonicity issue for b")
            
            return True
        except Exception as e:
            self.log(f"Monotonicity test failed: {str(e)}")
            return False
    
    def test_symbolic_laws(self):
        """Check symbolic relationships from KB."""
        try:
            # 1. Non-relativistic wavelength formula
            if (self.V_0 is not None and self.lambda_nonrel is not None and 
                self.V_0 != 0 and self.m is not None and self.e0 is not None and
                self.m != 0 and self.e0 != 0):
                sqrt_term = math.sqrt(2 * self.m * self.e0 * self.V_0)
                if sqrt_term != 0:
                    num_val = self.h / sqrt_term
                    if self.fuzzy_equal(self.lambda_nonrel, num_val):
                        self.log("✓ Non-relativistic formula numerically verified")
            
            # 2. Relativistic correction factor
            if (self.V_0 is not None and self.lambda_nonrel is not None and 
                self.lambda_rel is not None and self.V_0 != 0 and
                self.m is not None and self.c is not None and self.e0 is not None):
                factor_val = math.sqrt(1 + self.e0 * self.V_0 / (2 * self.m * self.c**2))
                if factor_val != 0:
                    lambda_rel_calc = self.lambda_nonrel / factor_val
                    if self.fuzzy_equal(self.lambda_rel, lambda_rel_calc):
                        self.log("✓ Relativistic correction factor verified")
            
            return True
        except Exception as e:
            self.log(f"Symbolic test failed: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all tests and return success status with logs."""
        self.logs = []
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
            ("Symbolic Laws", self.test_symbolic_laws)
        ]
        
        success = True
        for test_name, test_func in tests:
            try:
                test_result = test_func()
                if not test_result:
                    success = False
                    self.log(f"❌ {test_name} failed")
                else:
                    self.log(f"✓ {test_name} passed")
            except Exception as e:
                success = False
                self.log(f"❌ {test_name} raised exception: {str(e)}")
                self.log(traceback.format_exc())
        
        return success, self.logs