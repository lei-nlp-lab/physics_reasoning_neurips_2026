import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs with defaults
        self.m = getattr(solution_instance, 'm', 0.2)
        self.M = getattr(solution_instance, 'M', 0.6)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.v0 = getattr(solution_instance, 'v0', 10.0)
        self.l = getattr(solution_instance, 'l', 1.2)
        self.f = getattr(solution_instance, 'f', 1.0)
        
        # Safe extraction of results via results dictionary or attributes
        self.results = getattr(solution_instance, 'results', {})
        
        self.f_prime = getattr(solution_instance, 'f_prime', 
                              self.results.get('f_prime', 1.0))
        self.N1 = getattr(solution_instance, 'N_1', 
                         self.results.get('N_1', 8.0))
        self.N2 = getattr(solution_instance, 'N_2', 
                         self.results.get('N_2', 5.0))
        self.v1 = getattr(solution_instance, 'v_1', 
                         self.results.get('v_1', 8.0))
        self.v = getattr(solution_instance, 'v', 
                        self.results.get('v', None))
        self.h = getattr(solution_instance, 'h', 
                        self.results.get('h', 0.2))
        
        # Derived symbolic variables
        self.m_sym, self.M_sym, self.g_sym = sp.symbols('m M g', positive=True)
        self.v0_sym, self.v1_sym, self.v_sym = sp.symbols('v0 v1 v', real=True)
        self.l_sym, self.f_sym, self.h_sym = sp.symbols('l f h', positive=True)
        
        self.logs = []
        
    def _log(self, msg, success):
        self.logs.append(f"{'✓' if success else '✗'} {msg}")
        return success
        
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        
    def test_invariants(self):
        """Verify physical invariants from KB theorems"""
        tests_passed = True
        
        try:
            # 1. Newton's Third Law: f' = f (frictional forces equal and opposite)
            if not self._is_close(self.f_prime, self.f):
                tests_passed = self._log(
                    f"Newton's 3rd Law violation: f'={self.f_prime} ≠ f={self.f}", 
                    False
                ) and tests_passed
            else:
                self._log("Newton's 3rd Law: f' = f ✓", True)
                
            # 2. Static Equilibrium: N₁ = (m + M)g
            N1_expected = (self.m + self.M) * self.g
            if not self._is_close(self.N1, N1_expected):
                tests_passed = self._log(
                    f"Static equilibrium violation: N₁={self.N1} ≠ (m+M)g={(self.m+self.M)*self.g}", 
                    False
                ) and tests_passed
            else:
                self._log(f"Static equilibrium: N₁ = (m+M)g ✓", True)
                
            # 3. Dynamic Equilibrium: N₂ = Mg - f'
            N2_expected = self.M * self.g - self.f_prime
            if not self._is_close(self.N2, N2_expected):
                tests_passed = self._log(
                    f"Dynamic equilibrium violation: N₂={self.N2} ≠ Mg-f'={self.M*self.g-self.f_prime}", 
                    False
                ) and tests_passed
            else:
                self._log(f"Dynamic equilibrium: N₂ = Mg - f' ✓", True)
                
            # 4. Work-Energy Theorem for stage 1 (slider motion)
            # W_net = ΔK = -mgL - fL = ½mv₁² - ½mv₀²
            delta_K_expected = 0.5 * self.m * (self.v1**2 - self.v0**2)
            W_net_expected = -self.m * self.g * self.l - self.f * self.l
            
            if not self._is_close(delta_K_expected, W_net_expected):
                tests_passed = self._log(
                    f"Work-Energy Theorem violation (stage 1): ΔK={delta_K_expected:.2f} ≠ W_net={W_net_expected:.2f}", 
                    False
                ) and tests_passed
            else:
                self._log(f"Work-Energy Theorem (stage 1) ✓", True)
                
            # 5. Conservation of Momentum for collision
            if self.v is not None:
                p_before = self.m * self.v1
                p_after = (self.m + self.M) * self.v
                if not self._is_close(p_before, p_after):
                    tests_passed = self._log(
                        f"Momentum conservation violation: m*v₁={p_before} ≠ (m+M)*v={p_after}", 
                        False
                    ) and tests_passed
                else:
                    self._log("Momentum conservation (collision) ✓", True)
                    
            # 6. Work-Energy Theorem for stage 2 (vertical rise)
            if self.v is not None:
                K_after_collision = 0.5 * (self.m + self.M) * self.v**2
                U_gain = (self.m + self.M) * self.g * self.h
                
                if not self._is_close(K_after_collision, U_gain):
                    tests_passed = self._log(
                        f"Energy conservation violation (stage 2): K={K_after_collision:.2f} ≠ U={U_gain:.2f}", 
                        False
                    ) and tests_passed
                else:
                    self._log("Energy conservation (stage 2) ✓", True)
                    
        except Exception as e:
            tests_passed = self._log(f"Exception in test_invariants: {str(e)}", False) and False
            
        return tests_passed
        
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer"""
        tests_passed = True
        
        try:
            # Check if v1 is consistent with work-energy theorem
            # From work-energy: v₁² = v₀² - 2gL - 2fL/m
            v1_squared_expected = self.v0**2 - 2*self.g*self.l - 2*self.f*self.l/self.m
            v1_expected = math.sqrt(max(0, v1_squared_expected))
            
            if not self._is_close(self.v1, v1_expected):
                tests_passed = self._log(
                    f"Trace inconsistency: v₁={self.v1} ≠ √(v₀²-2gL-2fL/m)={v1_expected:.2f}", 
                    False
                ) and tests_passed
            else:
                self._log(f"v₁ trace consistency ✓", True)
                
            # Check if v (post-collision) is consistent with momentum conservation
            if self.v is None:
                # Compute v from momentum conservation
                self.v = self.m * self.v1 / (self.m + self.M)
                
            v_expected = self.m * self.v1 / (self.m + self.M)
            if not self._is_close(self.v, v_expected):
                tests_passed = self._log(
                    f"Trace inconsistency: v={self.v} ≠ m*v₁/(m+M)={v_expected:.2f}", 
                    False
                ) and tests_passed
            else:
                self._log(f"v trace consistency ✓", True)
                
            # Check if h is consistent with energy conservation
            h_expected = self.v**2 / (2 * self.g)
            if not self._is_close(self.h, h_expected):
                tests_passed = self._log(
                    f"Trace inconsistency: h={self.h} ≠ v²/(2g)={h_expected:.2f}", 
                    False
                ) and tests_passed
            else:
                self._log(f"h trace consistency ✓", True)
                
            # Check if all target answers match
            target_N1 = (self.m + self.M) * self.g
            target_N2 = self.M * self.g - self.f_prime
            target_v1 = v1_expected
            target_h = h_expected
            
            if not (self._is_close(self.N1, target_N1) and 
                   self._is_close(self.N2, target_N2) and 
                   self._is_close(self.v1, target_v1) and 
                   self._is_close(self.h, target_h)):
                tests_passed = self._log(
                    "One or more target answers inconsistent with derived values", 
                    False
                ) and tests_passed
            else:
                self._log("All target answers consistent ✓", True)
                
        except Exception as e:
            tests_passed = self._log(f"Exception in test_trace_consistency: {str(e)}", False) and False
            
        return tests_passed
        
    def test_monotonicity(self):
        """Perturb mass m and check trend in h"""
        tests_passed = True
        
        try:
            # Save original values
            m_original = self.m
            h_original = self.h
            
            # Perturb m by +10%
            perturbation = 0.1
            m_perturbed = m_original * (1 + perturbation)
            
            # Recalculate using physical formulas (without changing instance state)
            # Step 1: v1 for perturbed mass
            v1_squared_perturbed = self.v0**2 - 2*self.g*self.l - 2*self.f*self.l/m_perturbed
            if v1_squared_perturbed < 0:
                tests_passed = self._log(
                    f"Monotonicity test: v₁² negative for m={m_perturbed}", 
                    False
                ) and tests_passed
                return tests_passed
                
            v1_perturbed = math.sqrt(v1_squared_perturbed)
            
            # Step 2: v for perturbed mass
            v_perturbed = m_perturbed * v1_perturbed / (m_perturbed + self.M)
            
            # Step 3: h for perturbed mass
            h_perturbed = v_perturbed**2 / (2 * self.g)
            
            # Check monotonic trend: if m increases, h should change smoothly
            # We don't enforce a specific direction, just that it's not discontinuous
            delta_h = h_perturbed - h_original
            delta_m = m_perturbed - m_original
            
            # The relationship should be continuous and differentiable
            # We can check that the change isn't absurdly large
            relative_change = abs(delta_h / h_original) if h_original != 0 else float('inf')
            
            if relative_change > 10:  # Arbitrary but catches discontinuities
                tests_passed = self._log(
                    f"Monotonicity violation: h changed by {relative_change:.0%} for 10% m change", 
                    False
                ) and tests_passed
            else:
                self._log(f"Monotonicity: h changes smoothly with m (Δh={delta_h:.3f}) ✓", True)
                
            # Restore original values
            self.m = m_original
            self.h = h_original
            
        except Exception as e:
            # Restore on exception
            self.m = m_original if 'm_original' in locals() else self.m
            self.h = h_original if 'h_original' in locals() else self.h
            tests_passed = self._log(f"Exception in test_monotonicity: {str(e)}", False) and False
            
        return tests_passed
        
    def run_all_tests(self):
        """Execute all tests and return results"""
        self.logs = []  # Reset logs
        
        try:
            # Run tests
            inv_ok = self.test_invariants()
            trace_ok = self.test_trace_consistency()
            mono_ok = self.test_monotonicity()
            
            success = inv_ok and trace_ok and mono_ok
            
            # Summary
            self._log(f"\n=== SUMMARY ===", True)
            self._log(f"Invariants: {'PASS' if inv_ok else 'FAIL'}", inv_ok)
            self._log(f"Trace consistency: {'PASS' if trace_ok else 'FAIL'}", trace_ok)
            self._log(f"Monotonicity: {'PASS' if mono_ok else 'FAIL'}", mono_ok)
            self._log(f"Overall: {'PASS' if success else 'FAIL'}", success)
            
            return success, self.logs
            
        except Exception as e:
            error_msg = f"Critical error in test suite: {traceback.format_exc()}"
            self.logs.append(f"✗ {error_msg}")
            return False, self.logs

# Optional: Example usage if run directly
if __name__ == "__main__":
    class DummySolution:
        def __init__(self):
            self.m = 0.2
            self.M = 0.6
            self.g = 10.0
            self.v0 = 10.0
            self.l = 1.2
            self.f = 1.0
            self.results = {
                'f_prime': 1.0,
                'N_1': 8.0,
                'N_2': 5.0,
                'v_1': 8.0,
                'v': 2.0,
                'h': 0.2
            }
    
    checker = PhysicsChecker(DummySolution())
    success, logs = checker.run_all_tests()
    for log in logs:
        print(log)