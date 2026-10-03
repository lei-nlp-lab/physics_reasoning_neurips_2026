import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract inputs safely with defaults from problem context
        self.V0 = getattr(solution_instance, 'V0', 3000.0)  # in volts
        self.R = getattr(solution_instance, 'R', 0.05)      # in meters
        self.e_over_m = getattr(solution_instance, 'e_over_m', 1.76e11)  # C/kg
        
        # Extract results safely with defaults
        results = getattr(solution_instance, 'results', {})
        self.B = results.get('B', 0.0)
        self.B1 = results.get('B_1', 0.0)
        self.v0 = results.get('v_0', 0.0)
        self.r = results.get('r', 0.0)
        self.T = results.get('T', 0.0)
        self.b = results.get('b', 0.0)
        self.vecv = results.get('vecv', None)
        self.vecF = results.get('vecF', None)
        
        # Constants
        self.m_over_e = 1.0 / self.e_over_m  # kg/C
        self.e = 1.602176634e-19  # C (for explicit calculations)
        self.m = self.m_over_e * self.e      # kg (derived)
        
        # Tolerance for numerical comparisons (2% as per robustness rules)
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
    def test_invariants(self):
        logs = []
        success = True  # Assume success unless we find a failure
        
        try:
            # 1. Energy Conservation: ½mv₀² = eV₀
            kinetic_energy = 0.5 * self.m * self.v0**2
            potential_energy = self.e * self.V0
            if not math.isclose(kinetic_energy, potential_energy, 
                              rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"Energy conservation violation: KE={kinetic_energy:.3e}, eV₀={potential_energy:.3e}")
                success = False
            else:
                logs.append("✓ Energy conservation satisfied")
            
            # 2. Circular Motion Lorentz Force Balance (for B₁ case)
            # From step 11: m*v₀²/R = e*v₀*B₁ → B₁ = (m/e)*(v₀/R)
            if self.v0 > 0 and self.R > 0:
                B1_calc = (self.m_over_e) * (self.v0 / self.R)
                if not math.isclose(self.B1, B1_calc, 
                                  rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    logs.append(f"B₁ force balance mismatch: B₁={self.B1:.3e}, calculated={B1_calc:.3e}")
                    success = False
                else:
                    logs.append("✓ B₁ satisfies Lorentz/centripetal force balance")
            
            # 3. Cyclotron Period Consistency: T = 2πm/(eB)
            if self.B > 0:
                T_calc = 2 * math.pi * self.m / (self.e * self.B)
                if not math.isclose(self.T, T_calc, 
                                  rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    logs.append(f"Period T mismatch: T={self.T:.3e}, calculated={T_calc:.3e}")
                    success = False
                else:
                    logs.append("✓ Cyclotron period consistent with B")
            
            # 4. Velocity from Energy: v₀ = √(2(e/m)V₀)
            v0_calc = math.sqrt(2 * self.e_over_m * self.V0)
            if not math.isclose(self.v0, v0_calc, 
                              rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"v₀ mismatch: v₀={self.v0:.3e}, calculated={v0_calc:.3e}")
                success = False
            else:
                logs.append("✓ v₀ consistent with acceleration voltage")
                
        except Exception as e:
            logs.append(f"Error in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        logs = []
        success = True
        
        try:
            # Check that B formula from step 9 matches computed B
            # B = (4/R)√(2(m/e)V₀)
            B_calc = (4.0 / self.R) * math.sqrt(2 * self.m_over_e * self.V0)
            if not math.isclose(self.B, B_calc, 
                              rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"B formula mismatch: B={self.B:.3e}, step9 formula gives={B_calc:.3e}")
                success = False
            else:
                logs.append("✓ B matches step 9 formula")
            
            # Check that B₁ formula from step 12 matches computed B₁
            # B₁ = (1/R)√(2(m/e)V₀)
            B1_calc = (1.0 / self.R) * math.sqrt(2 * self.m_over_e * self.V0)
            if not math.isclose(self.B1, B1_calc, 
                              rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"B₁ formula mismatch: B₁={self.B1:.3e}, step12 formula gives={B1_calc:.3e}")
                success = False
            else:
                logs.append("✓ B₁ matches step 12 formula")
            
            # Check focusing distance b consistency
            # b = v₀T ≈ v₀*(2πm/(eB))
            if self.B > 0 and self.v0 > 0:
                b_from_T = self.v0 * (2 * math.pi * self.m / (self.e * self.B))
                # Also check formula from step 8: b = (2π/B)√(2(m/e)V₀)
                b_formula = (2 * math.pi / self.B) * math.sqrt(2 * self.m_over_e * self.V0)
                
                # Compare with extracted b if available
                if self.b > 0:
                    if not math.isclose(self.b, b_from_T, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                        logs.append(f"b mismatch: b={self.b:.3e}, from v₀T={b_from_T:.3e}")
                        success = False
                    else:
                        logs.append("✓ b consistent with v₀T")
                    
                    if not math.isclose(self.b, b_formula, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                        logs.append(f"b formula mismatch: b={self.b:.3e}, step8 formula={b_formula:.3e}")
                        success = False
                    else:
                        logs.append("✓ b matches step 8 formula")
                
                # CORRECTED: Check that b ≈ (2πR)/4 (focusing condition from problem)
                # The focusing condition is b = (2πR)/4, not b ≈ R
                b_expected = (2 * math.pi * self.R) / 4
                if not math.isclose(self.b, b_expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                    logs.append(f"Focusing condition: b={self.b:.3e} not close to (2πR)/4={b_expected:.3e}")
                    success = False
                else:
                    logs.append("✓ Focusing condition b=(2πR)/4 satisfied")
            
            # Check that B = 4*B₁ (from formulas)
            if not math.isclose(self.B, 4 * self.B1, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"B/B₁ ratio mismatch: B={self.B:.3e}, 4*B₁={4*self.B1:.3e}")
                success = False
            else:
                logs.append("✓ B = 4B₁ relationship holds")
                
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
        
        return success, logs
    
    def test_monotonicity(self):
        logs = []
        success = True
        
        try:
            # Perturb V₀ by +10% and check trends using ACTUAL formulas from the problem
            V0_perturbed = self.V0 * 1.1
            
            # Calculate new v₀ using energy conservation
            v0_new = math.sqrt(2 * self.e_over_m * V0_perturbed)
            
            # Calculate new B using the formula from step 9: B = (4/R)√(2(m/e)V₀)
            B_new = (4.0 / self.R) * math.sqrt(2 * self.m_over_e * V0_perturbed)
            
            # Calculate new B₁ using the formula from step 12: B₁ = (1/R)√(2(m/e)V₀)
            B1_new = (1.0 / self.R) * math.sqrt(2 * self.m_over_e * V0_perturbed)
            
            # v₀ should increase with √V₀
            if v0_new <= self.v0:
                logs.append(f"v₀ monotonicity violated: v₀({V0_perturbed})={v0_new:.3e} ≤ original={self.v0:.3e}")
                success = False
            else:
                logs.append("✓ v₀ increases with V₀")
            
            # B should increase with √V₀
            if B_new <= self.B:
                logs.append(f"B monotonicity violated: B({V0_perturbed})={B_new:.3e} ≤ original={self.B:.3e}")
                success = False
            else:
                logs.append("✓ B increases with V₀")
            
            # B₁ should increase with √V₀
            if B1_new <= self.B1:
                logs.append(f"B₁ monotonicity violated: B₁({V0_perturbed})={B1_new:.3e} ≤ original={self.B1:.3e}")
                success = False
            else:
                logs.append("✓ B₁ increases with V₀")
            
            # Perturb R by -10% and check inverse relationship
            R_perturbed = self.R * 0.9
            # Keep V₀ constant, recalculate B and B₁ with new R using the formulas
            B_new_R = (4.0 / R_perturbed) * math.sqrt(2 * self.m_over_e * self.V0)
            B1_new_R = (1.0 / R_perturbed) * math.sqrt(2 * self.m_over_e * self.V0)
            
            # B should increase as R decreases (inverse proportionality)
            if B_new_R <= self.B:
                logs.append(f"B-R inverse relation violated: B(R={R_perturbed})={B_new_R:.3e} ≤ original={self.B:.3e}")
                success = False
            else:
                logs.append("✓ B increases as R decreases (inverse relation)")
            
            if B1_new_R <= self.B1:
                logs.append(f"B₁-R inverse relation violated: B₁(R={R_perturbed})={B1_new_R:.3e} ≤ original={self.B1:.3e}")
                success = False
            else:
                logs.append("✓ B₁ increases as R decreases (inverse relation)")
            
            # Restore original values (conceptual - no actual mutation needed)
            logs.append("Original parameters restored")
                
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
        
        return success, logs
    
    def run_all_tests(self):
        all_logs = []
        overall_success = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"\n=== {test_name} Test ===")
            test_success, test_logs = test_method()
            all_logs.extend(test_logs)
            if not test_success:
                overall_success = False
        
        # Final summary
        all_logs.append(f"\n=== Summary ===")
        all_logs.append(f"Overall check {'PASSED' if overall_success else 'FAILED'}")
        
        return overall_success, all_logs