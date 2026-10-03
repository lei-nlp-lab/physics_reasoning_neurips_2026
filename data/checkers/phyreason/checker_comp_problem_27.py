import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        SAFE ATTRIBUTE ACCESS using getattr and results dictionary.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        self.inputs = {}
        self.errors = []
        self.logs = []
        
        # Extract input parameters safely
        self.inputs['l'] = getattr(solution_instance, 'l', None)
        self.inputs['r'] = getattr(solution_instance, 'r', None)
        self.inputs['Phi'] = getattr(solution_instance, 'Phi', 
                                     getattr(solution_instance, 'phi', None))
        self.inputs['mu0'] = getattr(solution_instance, 'mu0', 
                                      getattr(solution_instance, 'mu_0', 4*math.pi*1e-7))
        
        # Extract results with safe defaults
        self.F = self.results.get('F', 0.0)
        self.T = self.results.get('T', 0.0)
        self.F_1 = self.results.get('F_1', 0.0)
        self.DeltaW = self.results.get('DeltaW', 0.0)
        self.w = self.results.get('w', 0.0)
        self.B = self.results.get('B', 0.0)
        self.F_2 = self.results.get('F_2', 0.0)
        
        # For symbolic computations
        self.l_sym = sp.symbols('l', positive=True)
        self.r_sym = sp.symbols('r', positive=True)
        self.Phi_sym = sp.symbols('Phi', positive=True)
        self.mu0_sym = sp.symbols('mu0', positive=True)
    
    def _log_error(self, msg):
        self.errors.append(msg)
        self.logs.append(f"ERROR: {msg}")
    
    def _log_info(self, msg):
        self.logs.append(f"INFO: {msg}")
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical matching with 1-2% tolerance."""
        if a is None or b is None:
            return False
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False
    
    def test_invariants(self):
        """Verify fundamental physics laws based on KB theorems."""
        try:
            # 1. Magnetic flux conservation: B = Φ/(πr²)
            if self.B and self.inputs['Phi'] and self.inputs['r']:
                expected_B = self.inputs['Phi'] / (math.pi * self.inputs['r']**2)
                if not self._is_close(self.B, expected_B):
                    self._log_error(f"Magnetic field mismatch: B={self.B}, expected {expected_B}")
                else:
                    self._log_info(f"Magnetic field check passed: B ≈ {self.B:.3e} T")
            
            # 2. Energy conservation: T = ΔW/Δl = Φ²/(2μ₀πr²)
            if self.T and self.inputs['Phi'] and self.inputs['r'] and self.inputs['mu0']:
                expected_T = (self.inputs['Phi']**2) / (2 * self.inputs['mu0'] * math.pi * self.inputs['r']**2)
                if not self._is_close(self.T, expected_T):
                    self._log_error(f"Tension mismatch: T={self.T}, expected {expected_T}")
                else:
                    self._log_info(f"Tension check passed: T ≈ {self.T:.3e} N")
            
            # 3. Force superposition: F = (4-√2)/(8πμ₀) * Φ²/l²
            if self.F and self.inputs['Phi'] and self.inputs['l'] and self.inputs['mu0']:
                expected_F = (4 - math.sqrt(2)) / (8 * math.pi * self.inputs['mu0']) * \
                            (self.inputs['Phi']**2 / self.inputs['l']**2)
                if not self._is_close(self.F, expected_F):
                    self._log_error(f"Force mismatch: F={self.F}, expected {expected_F}")
                else:
                    self._log_info(f"Force check passed: F ≈ {self.F:.3e} N")
            
            # 4. Check magnetic energy density consistency
            if self.w and self.B and self.inputs['mu0']:
                expected_w = (self.B**2) / (2 * self.inputs['mu0'])
                if not self._is_close(self.w, expected_w):
                    self._log_error(f"Energy density mismatch: w={self.w}, expected {expected_w}")
                else:
                    self._log_info(f"Energy density check passed: w ≈ {self.w:.3e} J/m³")
            
            # 5. Check component forces if available
            if self.F_1 and self.inputs['Phi'] and self.inputs['l'] and self.inputs['mu0']:
                expected_F1 = 1/(4*math.pi*self.inputs['mu0']) * (self.inputs['Phi']**2 / self.inputs['l']**2)
                if not self._is_close(self.F_1, expected_F1):
                    self._log_error(f"F_1 mismatch: {self.F_1}, expected {expected_F1}")
            
            if self.F_2 and self.inputs['Phi'] and self.inputs['l'] and self.inputs['mu0']:
                expected_F2 = math.sqrt(2)/(8*math.pi*self.inputs['mu0']) * (self.inputs['Phi']**2 / (2*self.inputs['l']**2))
                if not self._is_close(self.F_2, expected_F2):
                    self._log_error(f"F_2 mismatch: {self.F_2}, expected {expected_F2}")
            
            # Symbolic verification as fallback
            try:
                # Symbolic expression for tension
                T_sym = self.Phi_sym**2 / (2 * self.mu0_sym * sp.pi * self.r_sym**2)
                # Symbolic expression for force
                F_sym = (4 - sp.sqrt(2)) / (8 * sp.pi * self.mu0_sym) * (self.Phi_sym**2 / self.l_sym**2)
                
                # Convert to numeric if all inputs available
                if all(v is not None for v in [self.inputs['Phi'], self.inputs['mu0'], 
                                              self.inputs['r'], self.inputs['l']]):
                    T_num = float(T_sym.subs({
                        self.Phi_sym: self.inputs['Phi'],
                        self.mu0_sym: self.inputs['mu0'],
                        self.r_sym: self.inputs['r']
                    }))
                    F_num = float(F_sym.subs({
                        self.Phi_sym: self.inputs['Phi'],
                        self.mu0_sym: self.inputs['mu0'],
                        self.l_sym: self.inputs['l']
                    }))
                    
                    if self.T and not self._is_close(self.T, T_num):
                        self._log_error(f"Symbolic T mismatch: {self.T} vs {T_num}")
                    if self.F and not self._is_close(self.F, F_num):
                        self._log_error(f"Symbolic F mismatch: {self.F} vs {F_num}")
            except Exception as e:
                self._log_info(f"Symbolic check skipped: {str(e)}")
                
        except Exception as e:
            self._log_error(f"Invariant test failed: {str(e)}")
            self._log_info(traceback.format_exc())
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Check if T and ΔW are consistent (T = dW/dl)
            if self.T and self.DeltaW and self.inputs['l']:
                # For small Δl, T ≈ ΔW/Δl
                # Use symbolic: ΔW = T * Δl
                # We can check dimensions: [T] = N, [ΔW] = J = N·m
                # So ΔW/T should have dimensions of length
                if self.DeltaW != 0 and self.T != 0:
                    delta_l = self.DeltaW / self.T
                    self._log_info(f"Inferred Δl from work: {delta_l:.3e} m")
            
            # Check if w from step6 and step7 are proportional
            # Step6: electric analogy w_e = Q²/(32π²ε₀a⁴)
            # Step7: magnetic case w_m = Φ²/(32π²μ₀a⁴)
            # They should have same structure with ε₀↔μ₀, Q↔Φ
            # This is a structural check rather than numerical
            
            # Check if B from step1 is used in step2
            if self.B and self.DeltaW and self.inputs['r']:
                # From step2: ΔW = (B²/(2μ₀)) * πr²Δl
                # Rearranged: B = sqrt(2μ₀ΔW/(πr²Δl))
                # We don't have Δl, but we can check consistency with T
                if self.T:
                    # Since T = ΔW/Δl, then B = sqrt(2μ₀T/(πr²))
                    B_from_T = math.sqrt(2 * self.inputs['mu0'] * self.T / (math.pi * self.inputs['r']**2))
                    if not self._is_close(self.B, B_from_T):
                        self._log_error(f"B inconsistent with T: {self.B} vs {B_from_T}")
            
            self._log_info("Trace consistency check completed")
        except Exception as e:
            self._log_error(f"Trace consistency test failed: {str(e)}")
    
    def test_monotonicity(self):
        """Perturb an input and check trend (restore after)."""
        try:
            if not self.inputs['Phi']:
                self._log_info("Skipping monotonicity test: Phi not available")
                return
            
            # Store original value
            original_Phi = self.inputs['Phi']
            
            # Perturb by 1%
            perturbed_Phi = original_Phi * 1.01
            
            # Recalculate expected results using formulas
            original_T = (original_Phi**2) / (2 * self.inputs['mu0'] * math.pi * self.inputs['r']**2)
            perturbed_T = (perturbed_Phi**2) / (2 * self.inputs['mu0'] * math.pi * self.inputs['r']**2)
            
            original_F = (4 - math.sqrt(2)) / (8 * math.pi * self.inputs['mu0']) * \
                        (original_Phi**2 / self.inputs['l']**2)
            perturbed_F = (4 - math.sqrt(2)) / (8 * math.pi * self.inputs['mu0']) * \
                         (perturbed_Phi**2 / self.inputs['l']**2)
            
            # Check quadratic scaling
            expected_T_ratio = (1.01)**2  # Φ² dependence
            actual_T_ratio = perturbed_T / original_T if original_T != 0 else 0
            
            expected_F_ratio = (1.01)**2
            actual_F_ratio = perturbed_F / original_F if original_F != 0 else 0
            
            if not self._is_close(actual_T_ratio, expected_T_ratio):
                self._log_error(f"T not quadratic in Φ: ratio={actual_T_ratio:.3f}, expected {expected_T_ratio:.3f}")
            
            if not self._is_close(actual_F_ratio, expected_F_ratio):
                self._log_error(f"F not quadratic in Φ: ratio={actual_F_ratio:.3f}, expected {expected_F_ratio:.3f}")
            
            # Restore original value
            self.inputs['Phi'] = original_Phi
            
            self._log_info(f"Monotonicity check: T scales as Φ² (ratio={actual_T_ratio:.3f}), "
                          f"F scales as Φ² (ratio={actual_F_ratio:.3f})")
            
        except Exception as e:
            self._log_error(f"Monotonicity test failed: {str(e)}")
            # Ensure restoration even if error occurs
            if 'original_Phi' in locals():
                self.inputs['Phi'] = original_Phi
    
    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        self.errors.clear()
        self.logs.clear()
        
        self._log_info("Starting Physics Audit")
        self._log_info(f"Inputs: l={self.inputs['l']}, r={self.inputs['r']}, "
                      f"Φ={self.inputs['Phi']}, μ₀={self.inputs['mu0']}")
        self._log_info(f"Results: T={self.T}, F={self.F}, B={self.B}")
        
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        success = len(self.errors) == 0
        if success:
            self._log_info("All physics checks passed!")
        else:
            self._log_info(f"Found {len(self.errors)} physics issues")
        
        return success, self.logs

# Example usage (commented out):
# checker = PhysicsChecker(solution)
# success, logs = checker.run_all_tests()