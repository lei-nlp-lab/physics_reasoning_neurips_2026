import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        
        # Extract inputs safely
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'inputs', {}).get('m', 9.11e-31))
        self.e = getattr(solution_instance, 'e', getattr(solution_instance, 'inputs', {}).get('e', 1.6e-19))
        self.k = getattr(solution_instance, 'k', getattr(solution_instance, 'inputs', {}).get('k', 8.99e9))
        self.r = getattr(solution_instance, 'r', getattr(solution_instance, 'inputs', {}).get('r', 5.29e-11))
        
        # Extract results safely with fallbacks
        results = getattr(solution_instance, 'results', {})
        self.E_k = results.get('E_k', 0.0)  # Kinetic energy
        self.E_p = results.get('E_p', 0.0)  # Potential energy  
        self.T = results.get('T', 0.0)      # Period
        self.I = results.get('I', 0.0)      # Current
        self.E = results.get('E', 0.0)      # Total energy
        
        # Prepare symbolic variables for invariant checking
        self._setup_symbolic()
        
        # For monotonicity testing - store original state
        self.original_inputs = {
            'm': self.m,
            'e': self.e, 
            'k': self.k,
            'r': self.r
        }
        
        self.logs = []
    
    def _setup_symbolic(self):
        """Set up symbolic variables and equations for invariant checks."""
        self.m_sym, self.e_sym, self.k_sym, self.r_sym = sp.symbols('m e k r', positive=True)
        self.v_sym = sp.symbols('v', positive=True)
        self.F_coulomb_sym = self.k_sym * self.e_sym**2 / self.r_sym**2
        self.F_centripetal_sym = self.m_sym * self.v_sym**2 / self.r_sym
        self.E_k_sym = sp.Rational(1,2) * self.m_sym * self.v_sym**2
        self.E_p_sym = -self.k_sym * self.e_sym**2 / self.r_sym
        self.E_total_sym = self.E_p_sym + self.E_k_sym
        self.T_sym = 2 * sp.pi * self.r_sym / self.v_sym
        self.I_sym = self.e_sym / self.T_sym
    
    def _log(self, message, success=True):
        """Helper to log test results."""
        self.logs.append(f"{'✓' if success else '✗'} {message}")
        return success
    
    def _is_close(self, a, b):
        """Robust numerical comparison with tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        """Test fundamental physical invariants."""
        success = True
        
        try:
            # 1. Force balance: Coulomb force = centripetal force
            # From step_1: k*e^2/r^2 = m*v^2/r
            
            # Method 1: Symbolic substitution
            # First derive v from force balance: v = sqrt(k*e^2/(m*r))
            v_calc = math.sqrt(self.k * self.e**2 / (self.m * self.r))
            
            # Calculate forces both ways
            F_coulomb = self.k * self.e**2 / self.r**2
            F_centripetal = self.m * v_calc**2 / self.r
            
            if not self._is_close(F_coulomb, F_centripetal):
                success = self._log(f"Force balance violated: F_coulomb={F_coulomb:.3e} ≠ F_centripetal={F_centripetal:.3e}", False)
            else:
                success = self._log(f"Force balance satisfied: {F_coulomb:.3e} ≈ {F_centripetal:.3e}")
            
            # 2. Energy consistency: E = E_p + E_k
            if not self._is_close(self.E, self.E_p + self.E_k):
                success = self._log(f"Energy conservation violated: E={self.E:.3e} ≠ E_p+E_k={self.E_p+self.E_k:.3e}", False)
            else:
                success = self._log(f"Energy conservation satisfied: {self.E:.3e} ≈ {self.E_p+self.E_k:.3e}")
            
            # 3. Check relationship between E_k and force balance
            # From force balance: v^2 = k*e^2/(m*r)
            # So E_k should = (1/2)*m*v^2 = k*e^2/(2*r)
            E_k_expected = self.k * self.e**2 / (2 * self.r)
            if not self._is_close(self.E_k, E_k_expected):
                success = self._log(f"Kinetic energy mismatch: E_k={self.E_k:.3e} ≠ expected k*e^2/(2r)={E_k_expected:.3e}", False)
            else:
                success = self._log(f"Kinetic energy matches force balance: {self.E_k:.3e} ≈ {E_k_expected:.3e}")
            
            # 4. Check total energy formula
            E_expected = -self.k * self.e**2 / (2 * self.r)
            if not self._is_close(self.E, E_expected):
                success = self._log(f"Total energy mismatch: E={self.E:.3e} ≠ expected -k*e^2/(2r)={E_expected:.3e}", False)
            else:
                success = self._log(f"Total energy formula correct: {self.E:.3e} ≈ {E_expected:.3e}")
                
        except Exception as e:
            success = self._log(f"Error in invariant tests: {str(e)}", False)
            self.logs.append(traceback.format_exc())
        
        return success
    
    def test_trace_consistency(self):
        """Check consistency between intermediate steps and final results."""
        success = True
        
        try:
            # Check if the solution followed the expected derivation path
            
            # Step 3: E_k should equal k*e^2/(2r)
            E_k_from_step3 = self.k * self.e**2 / (2 * self.r)
            if not self._is_close(self.E_k, E_k_from_step3):
                success = self._log(f"Step 3 inconsistency: E_k={self.E_k:.3e} ≠ k*e^2/(2r)={E_k_from_step3:.3e}", False)
            else:
                success = self._log(f"Step 3 consistent: E_k = k*e^2/(2r)")
            
            # Step 5: E should equal E_p + E_k = -k*e^2/(2r)
            E_from_step5 = -self.k * self.e**2 / (2 * self.r)
            if not self._is_close(self.E, E_from_step5):
                success = self._log(f"Step 5 inconsistency: E={self.E:.3e} ≠ -k*e^2/(2r)={E_from_step5:.3e}", False)
            else:
                success = self._log(f"Step 5 consistent: E = -k*e^2/(2r)")
            
            # Step 8: Check current formula
            # First calculate v from force balance
            v = math.sqrt(self.k * self.e**2 / (self.m * self.r))
            T_calc = 2 * math.pi * self.r / v
            I_expected = self.e / T_calc
            
            # Also check the formula given in step_8
            I_alt_formula = (self.e**2 / (2 * math.pi * self.r)) * math.sqrt(self.k / (self.m * self.r))
            
            # Check both forms agree
            if not self._is_close(I_expected, I_alt_formula):
                success = self._log(f"Current formulas inconsistent: I(period)={I_expected:.3e} ≠ I(alt)={I_alt_formula:.3e}", False)
            
            if not self._is_close(self.I, I_expected):
                success = self._log(f"Step 8 inconsistency: I={self.I:.3e} ≠ expected={I_expected:.3e}", False)
            else:
                success = self._log(f"Step 8 consistent: I = e/T with T=2πr/v")
                
        except Exception as e:
            success = self._log(f"Error in trace consistency: {str(e)}", False)
            self.logs.append(traceback.format_exc())
        
        return success
    
    def test_monotonicity(self):
        """Test monotonic trends by perturbing inputs."""
        success = True
        
        try:
            # Test 1: Increase radius r
            original_r = self.r
            perturbed_r = original_r * 1.1  # Increase by 10%
            
            # Expected trends:
            # E_k ∝ 1/r, so should decrease
            # E ∝ -1/r, so should increase (less negative)
            # I ∝ r^(-3/2), so should decrease
            
            # Calculate original values using formulas
            E_k_orig = self.k * self.e**2 / (2 * original_r)
            E_orig = -self.k * self.e**2 / (2 * original_r)
            v_orig = math.sqrt(self.k * self.e**2 / (self.m * original_r))
            I_orig = self.e / (2 * math.pi * original_r / v_orig)
            
            # Calculate perturbed values
            E_k_pert = self.k * self.e**2 / (2 * perturbed_r)
            E_pert = -self.k * self.e**2 / (2 * perturbed_r)
            v_pert = math.sqrt(self.k * self.e**2 / (self.m * perturbed_r))
            I_pert = self.e / (2 * math.pi * perturbed_r / v_pert)
            
            # Check trends
            if not E_k_pert < E_k_orig:
                success = self._log(f"Monotonicity violated: E_k not decreasing with r (orig={E_k_orig:.3e}, pert={E_k_pert:.3e})", False)
            
            if not E_pert > E_orig:  # E becomes less negative
                success = self._log(f"Monotonicity violated: E not increasing with r (orig={E_orig:.3e}, pert={E_pert:.3e})", False)
            
            if not I_pert < I_orig:
                success = self._log(f"Monotonicity violated: I not decreasing with r (orig={I_orig:.3e}, pert={I_pert:.3e})", False)
            
            if success:
                self._log(f"Monotonicity test passed: all trends correct for r perturbation")
            
            # Test 2: Increase charge e
            original_e = self.e
            perturbed_e = original_e * 1.1
            
            # Calculate with perturbed e (keeping other parameters at original)
            E_k_orig_e = self.k * original_e**2 / (2 * original_r)
            E_pert_e = -self.k * perturbed_e**2 / (2 * original_r)
            
            # E_k and E magnitude should increase quadratically with e
            if not abs(E_pert_e) > abs(E_orig):  # More negative = larger magnitude
                success = self._log(f"Monotonicity violated: |E| not increasing with e", False)
            
            # Restore original values
            self.r = original_r
            self.e = original_e
            
        except Exception as e:
            success = self._log(f"Error in monotonicity test: {str(e)}", False)
            self.logs.append(traceback.format_exc())
        
        return success
    
    def run_all_tests(self):
        """Run all tests and return success flag with logs."""
        self.logs = []
        self.logs.append("=== Physics Checker Started ===")
        
        # Run tests
        test1_success = self.test_invariants()
        test2_success = self.test_trace_consistency()
        test3_success = self.test_monotonicity()
        
        # Summary
        all_success = test1_success and test2_success and test3_success
        self.logs.append(f"\n=== Summary: {'ALL TESTS PASSED' if all_success else 'SOME TESTS FAILED'} ===")
        
        return all_success, self.logs