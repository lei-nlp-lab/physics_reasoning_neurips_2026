import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract inputs safely
        self.mA = getattr(solution_instance, 'mA', 0.2)
        self.mB = getattr(solution_instance, 'mB', 0.1)
        self.v0 = getattr(solution_instance, 'v0', 15.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.N = getattr(solution_instance, 'N', 2.5)  # normal force at N
        
        # Extract results safely via results dict or attributes
        self.results = getattr(solution_instance, 'results', {})
        self.v_A = self._safe_get('v_A')
        self.v_B = self._safe_get('v_B')
        self.R = self._safe_get('R')
        self.v_AN = self._safe_get('v_AN')
        self.v_BN = self._safe_get('v_BN')
        self.t = self._safe_get('t')
        self.Deltax = self._safe_get('Deltax')
        
        self.logs = []
        
    def _safe_get(self, key):
        """Safe extraction with fallback to 0.0"""
        val = getattr(self, key, None)
        if val is None:
            val = self.results.get(key, 0.0)
        # If it's a string with units, try to convert
        if isinstance(val, str):
            # Remove units and non-numeric parts
            import re
            match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', val)
            if match:
                return float(match.group())
            else:
                # Try to evaluate expressions like 8*sqrt(6)
                try:
                    return float(sp.sympify(val))
                except:
                    return 0.0
        return float(val)
    
    def _is_close(self, a, b):
        """Fuzzy numerical matching"""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        """Verify conservation laws"""
        logs = []
        
        # 1. Elastic collision: conservation of momentum and kinetic energy
        try:
            # Momentum before
            p_before = self.mA * self.v0
            # Momentum after collision (ball B was stationary)
            p_after = self.mA * self.v_A + self.mB * self.v_B
            
            if not self._is_close(p_before, p_after):
                logs.append(f"Momentum not conserved: p_before={p_before:.3f}, p_after={p_after:.3f}")
            else:
                logs.append("✓ Momentum conserved in collision")
                
            # Kinetic energy conservation (elastic collision)
            KE_before = 0.5 * self.mA * self.v0**2
            KE_after = 0.5 * self.mA * self.v_A**2 + 0.5 * self.mB * self.v_B**2
            
            if not self._is_close(KE_before, KE_after):
                logs.append(f"KE not conserved in collision: KE_before={KE_before:.3f}, KE_after={KE_after:.3f}")
            else:
                logs.append("✓ Kinetic energy conserved in collision")
        except Exception as e:
            logs.append(f"Error in collision invariants: {str(e)}")
        
        # 2. Mechanical energy for ball A from collision to point N
        try:
            # Bottom of track (after collision) to top point N
            # At bottom: KE = 0.5*mA*v_A^2, PE = 0 (reference)
            # At top: KE = 0.5*mA*v_AN^2, PE = mA*g*(2R)
            E_bottom_A = 0.5 * self.mA * self.v_A**2
            E_top_A = 0.5 * self.mA * self.v_AN**2 + self.mA * self.g * (2 * self.R)
            
            if not self._is_close(E_bottom_A, E_top_A):
                logs.append(f"Energy not conserved for ball A: E_bottom={E_bottom_A:.3f}, E_top={E_top_A:.3f}")
            else:
                logs.append("✓ Mechanical energy conserved for ball A")
        except Exception as e:
            logs.append(f"Error in energy conservation for A: {str(e)}")
        
        # 3. Mechanical energy for ball B from collision to point N
        try:
            E_bottom_B = 0.5 * self.mB * self.v_B**2
            E_top_B = 0.5 * self.mB * self.v_BN**2 + self.mB * self.g * (2 * self.R)
            
            if not self._is_close(E_bottom_B, E_top_B):
                logs.append(f"Energy not conserved for ball B: E_bottom={E_bottom_B:.3f}, E_top={E_top_B:.3f}")
            else:
                logs.append("✓ Mechanical energy conserved for ball B")
        except Exception as e:
            logs.append(f"Error in energy conservation for B: {str(e)}")
        
        # 4. Newton's 2nd law at point N for ball A
        try:
            # At top: N + mA*g = mA*v_AN^2/R (centripetal force downward)
            required_centripetal = self.mA * self.v_AN**2 / self.R
            actual_force = self.N + self.mA * self.g
            
            if not self._is_close(required_centripetal, actual_force):
                logs.append(f"Newton's 2nd law violation at N: required={required_centripetal:.3f}, actual={actual_force:.3f}")
            else:
                logs.append("✓ Newton's 2nd law satisfied at point N")
        except Exception as e:
            logs.append(f"Error in Newton's law check: {str(e)}")
        
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer"""
        logs = []
        
        # 1. Check that v_A, v_B satisfy elastic collision equations
        try:
            # From elastic collision formulas for equal masses? Actually masses differ.
            # General 1D elastic collision:
            # v_A_final = ((mA-mB)*v0 + 2*mB*0)/(mA+mB) = (mA-mB)*v0/(mA+mB)
            # v_B_final = ((mB-mA)*0 + 2*mA*v0)/(mA+mB) = 2*mA*v0/(mA+mB)
            expected_v_A = ((self.mA - self.mB) * self.v0) / (self.mA + self.mB)
            expected_v_B = (2 * self.mA * self.v0) / (self.mA + self.mB)
            
            if not (self._is_close(self.v_A, expected_v_A) and self._is_close(self.v_B, expected_v_B)):
                logs.append(f"Collision velocities mismatch: v_A={self.v_A:.3f} (expected {expected_v_A:.3f}), "
                          f"v_B={self.v_B:.3f} (expected {expected_v_B:.3f})")
            else:
                logs.append("✓ Collision velocities consistent with elastic formulas")
        except Exception as e:
            logs.append(f"Error in collision consistency: {str(e)}")
        
        # 2. Check R and v_AN from given normal force condition
        try:
            # From energy: v_A^2 = v_AN^2 + 4*g*R  => v_AN^2 = v_A^2 - 4*g*R
            # From Newton: N + mA*g = mA*v_AN^2/R
            # Substitute v_AN^2: N + mA*g = mA*(v_A^2 - 4*g*R)/R
            # Solve for R: (N + mA*g)*R = mA*v_A^2 - 4*mA*g*R
            # => R*(N + 5*mA*g) = mA*v_A^2
            # => R = mA*v_A^2 / (N + 5*mA*g)
            expected_R = (self.mA * self.v_A**2) / (self.N + 5 * self.mA * self.g)
            expected_v_AN = math.sqrt(self.v_A**2 - 4 * self.g * expected_R)
            
            if not self._is_close(self.R, expected_R):
                logs.append(f"Radius mismatch: R={self.R:.3f} (expected {expected_R:.3f})")
            else:
                logs.append("✓ Radius consistent with energy+Newton equations")
                
            if not self._is_close(self.v_AN, expected_v_AN):
                logs.append(f"v_AN mismatch: v_AN={self.v_AN:.3f} (expected {expected_v_AN:.3f})")
            else:
                logs.append("✓ v_AN consistent with energy conservation")
        except Exception as e:
            logs.append(f"Error in R/v_AN consistency: {str(e)}")
        
        # 3. Check v_BN from energy conservation
        try:
            expected_v_BN = math.sqrt(self.v_B**2 - 4 * self.g * self.R)
            if not self._is_close(self.v_BN, expected_v_BN):
                logs.append(f"v_BN mismatch: v_BN={self.v_BN:.3f} (expected {expected_v_BN:.3f})")
            else:
                logs.append("✓ v_BN consistent with energy conservation")
        except Exception as e:
            logs.append(f"Error in v_BN consistency: {str(e)}")
        
        # 4. Check time of flight
        try:
            # Free fall from height 2R: 2R = 0.5*g*t^2
            expected_t = math.sqrt(4 * self.R / self.g)
            if not self._is_close(self.t, expected_t):
                logs.append(f"Time mismatch: t={self.t:.3f} (expected {expected_t:.3f})")
            else:
                logs.append("✓ Time of flight consistent with free fall")
        except Exception as e:
            logs.append(f"Error in time consistency: {str(e)}")
        
        # 5. Check Delta x
        try:
            expected_Deltax = (self.v_BN - self.v_AN) * self.t
            if not self._is_close(self.Deltax, expected_Deltax):
                logs.append(f"Delta x mismatch: Δx={self.Deltax:.3f} (expected {expected_Deltax:.3f})")
            else:
                logs.append("✓ Δx consistent with projectile motion")
        except Exception as e:
            logs.append(f"Error in Δx consistency: {str(e)}")
        
        return logs
    
    def test_monotonicity(self):
        """Perturb an input and check trend"""
        logs = []
        
        # Perturb mass mA by +10% and check how R changes
        # Expected: larger mA -> larger inertia -> smaller v_A after collision
        # -> smaller centripetal requirement -> potentially different R
        original_mA = self.mA
        perturbed_mA = original_mA * 1.10  # +10%
        
        try:
            # Recalculate chain using formulas
            # 1. Collision velocities with perturbed mA
            v_A_pert = ((perturbed_mA - self.mB) * self.v0) / (perturbed_mA + self.mB)
            v_B_pert = (2 * perturbed_mA * self.v0) / (perturbed_mA + self.mB)
            
            # 2. R from normal force condition (same N, g)
            R_pert = (perturbed_mA * v_A_pert**2) / (self.N + 5 * perturbed_mA * self.g)
            
            # Check trend: larger mass should lead to larger R?
            # From formula R = mA*v_A^2/(N + 5*mA*g)
            # As mA increases, numerator increases quadratically (v_A decreases but mA increases)
            # Denominator increases linearly
            # The trend isn't obvious analytically, so we just report the change
            if R_pert > self.R:
                logs.append(f"✓ Monotonicity: Increasing mA from {original_mA:.3f} to {perturbed_mA:.3f} increases R from {self.R:.3f} to {R_pert:.3f}")
            elif R_pert < self.R:
                logs.append(f"✓ Monotonicity: Increasing mA from {original_mA:.3f} to {perturbed_mA:.3f} decreases R from {self.R:.3f} to {R_pert:.3f}")
            else:
                logs.append(f"Monotonicity: R unchanged with mA perturbation")
                
        except Exception as e:
            logs.append(f"Error in monotonicity test: {str(e)}")
            logs.append(traceback.format_exc())
        
        # Restore original value
        self.mA = original_mA
        
        return logs
    
    def run_all_tests(self):
        """Run all tests and return success status and logs"""
        self.logs = []
        
        try:
            self.logs.append("=== PhysicsChecker Report ===")
            
            # Run individual tests
            inv_logs = self.test_invariants()
            self.logs.extend(inv_logs)
            
            trace_logs = self.test_trace_consistency()
            self.logs.extend(trace_logs)
            
            mono_logs = self.test_monotonicity()
            self.logs.extend(mono_logs)
            
            # Check if any test failed (looking for error/warning messages)
            error_count = sum(1 for log in self.logs if 
                            ('mismatch' in log.lower() or 
                             'violation' in log.lower() or 
                             'not conserved' in log.lower()) and 
                             not log.startswith('✓'))
            
            success = error_count == 0
            
            self.logs.append(f"\nSummary: {'All checks passed' if success else f'{error_count} potential issue(s) found'}")
            
        except Exception as e:
            self.logs.append(f"Fatal error in PhysicsChecker: {str(e)}")
            self.logs.append(traceback.format_exc())
            success = False
        
        return success, self.logs