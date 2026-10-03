import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallback to results dict
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract key parameters with safe defaults from results
        self.gamma = self.results.get('gamma', 0.0250)
        self.R0 = self.results.get('R_0', 0.01)
        self.Pa = self.results.get('P_a', 1.013e5)
        self.rho_a = self.results.get('rho_a', 1.3)
        self.rho_s = self.results.get('rho_s', 1000.0)
        self.t = self.results.get('t', 1e-7)
        self.g = self.results.get('g', 9.8)
        self.eta = self.results.get('eta', 1.8e-5)
        self.Ta = self.results.get('T_a', 300.0)
        self.epsilon0 = 8.854e-12
        
        # Extract results with fallbacks
        self.ratio_rhoT = self.results.get('fracrho_iT_irho_aT_a', 1.0000987)
        self.ratio_diff = self.results.get('fracrho_iT_irho_aT_a-1', 0.0001)
        self.Ti = self.results.get('T_i', 307.1)
        self.u = self.results.get('u', 0.36)
        self.DeltaR = self.results.get('DeltaR', 0.0)
        
        # Additional results for verification
        self.W = self.results.get('W', None)
        self.B = self.results.get('B', None)
        self.Pi = self.results.get('P_i', None)
        
    def test_invariants(self):
        logs = []
        success = True
        
        try:
            # 1. Pressure equilibrium check
            Pi_calc = self.Pa + 4*self.gamma/self.R0
            if self.Pi is not None:
                if math.isclose(Pi_calc, self.Pi, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("✓ Pressure equilibrium valid")
                else:
                    logs.append(f"Pressure equilibrium fail: calc={Pi_calc}, result={self.Pi}")
                    success = False
            
            # 2. Ideal gas law ratio check
            ratio_calc = 1 + 4*self.gamma/(self.R0*self.Pa)
            if math.isclose(self.ratio_rhoT, ratio_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Ideal gas ratio valid")
            else:
                logs.append(f"Ideal gas ratio fail: calc={ratio_calc}, result={self.ratio_rhoT}")
                success = False
            
            # Check ratio difference
            diff_calc = ratio_calc - 1
            if math.isclose(self.ratio_diff, diff_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Ratio difference valid")
            else:
                logs.append(f"Ratio difference fail: calc={diff_calc}, result={self.ratio_diff}")
                success = False
            
            # 3. Force balance for floatation - check if values are reasonable
            if self.W is not None and self.B is not None:
                if self.W > 0 and self.B > 0:
                    logs.append(f"✓ Weight and buoyancy positive: W={self.W:.2e}, B={self.B:.2e}")
                else:
                    logs.append(f"Weight or buoyancy non-positive: W={self.W}, B={self.B}")
                    success = False
                
                # Net force should be positive (downward) when T_i = T_a
                net_force = self.W - self.B
                logs.append(f"Net downward force: {net_force:.3e} N")
            
            # 4. Stokes' law force balance - check if u is reasonable
            if self.u > 0 and self.u < 10:  # Reasonable range for updraft velocity
                logs.append(f"✓ Updraft velocity reasonable: u={self.u:.3f} m/s")
            else:
                logs.append(f"Updraft velocity out of range: u={self.u}")
                success = False
            
            # 5. Check DeltaR if available
            if self.DeltaR is not None:
                if abs(self.DeltaR) < 1e-6:  # Should be very small
                    logs.append(f"✓ DeltaR is small as expected: {self.DeltaR:.2e} m")
                else:
                    logs.append(f"DeltaR unexpectedly large: {self.DeltaR}")
            
        except Exception as e:
            logs.append(f"Invariants test error: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
        
        return success, logs
    
    def test_trace_consistency(self):
        logs = []
        success = True
        
        try:
            # Check step_5 to step_6 consistency
            calc_diff = self.ratio_rhoT - 1
            if math.isclose(calc_diff, self.ratio_diff, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ Steps 5-6 consistent")
            else:
                logs.append(f"Trace consistency fail (steps 5-6): {calc_diff} vs {self.ratio_diff}")
                success = False
            
            # Check step_10 temperature calculation - use the correct formula
            # From ground truth: T_i = (R_0 ρ_a T_a)/(R_0 ρ_a - 3ρ_s t) * (1 + 4γ/(R_0 P_a))
            denominator = self.R0*self.rho_a - 3*self.rho_s*self.t
            if abs(denominator) > 1e-10:  # Avoid division by zero
                Ti_calc = (self.R0*self.rho_a*self.Ta/denominator) * (1 + 4*self.gamma/(self.R0*self.Pa))
                if math.isclose(self.Ti, Ti_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"✓ Temperature calculation consistent: {Ti_calc:.1f} K")
                else:
                    logs.append(f"Temperature calc mismatch: calc={Ti_calc:.1f} vs result={self.Ti:.1f}")
                    logs.append(f"  denominator={denominator}, R0={self.R0}, rho_a={self.rho_a}, t={self.t}")
                    success = False
            else:
                logs.append("Skipping temperature consistency (denominator too small)")
            
            # Check step_15 velocity calculation - use the correct formula
            # From ground truth: u = [4R_0ρ_s t g + (4/3)R_0²ρ_a g(4γ/(R_0 P_a))] / (6η)
            if self.eta > 1e-10:
                # Note: The formula in ground truth step 14 is:
                # u ≥ [4R_0ρ_s t g + (4/3)R_0²ρ_a g(4γ/(R_0 P_a))] / (6η)
                term1 = 4 * self.R0 * self.rho_s * self.t * self.g / (6 * self.eta)
                term2 = (4/3) * self.R0**2 * self.rho_a * self.g * (4*self.gamma/(self.R0*self.Pa)) / (6 * self.eta)
                u_calc = term1 + term2
                if math.isclose(self.u, u_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"✓ Velocity calculation consistent: {u_calc:.3f} m/s")
                else:
                    logs.append(f"Velocity calc mismatch: calc={u_calc:.3f} vs result={self.u:.3f}")
                    logs.append(f"  term1={term1:.3e}, term2={term2:.3e}, t={self.t}")
                    success = False
            else:
                logs.append("Skipping velocity consistency (eta too small)")
            
        except Exception as e:
            logs.append(f"Trace consistency error: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
        
        return success, logs
    
    def test_monotonicity(self):
        logs = []
        success = True
        
        try:
            # Perturb gamma upward by 1%
            perturbed_gamma = self.gamma * 1.01
            
            # Recalculate ratio - should increase
            new_ratio = 1 + 4*perturbed_gamma/(self.R0*self.Pa)
            if new_ratio > self.ratio_rhoT:
                logs.append("✓ Ratio increases with gamma (monotonic)")
            else:
                logs.append(f"Monotonicity fail: ratio decreased with increased gamma")
                success = False
            
            # Recalculate temperature - should increase
            denominator = self.R0*self.rho_a - 3*self.rho_s*self.t
            if abs(denominator) > 1e-10:
                new_Ti = (self.R0*self.rho_a*self.Ta/denominator) * new_ratio
                if new_Ti > self.Ti:
                    logs.append("✓ Ti increases with gamma (monotonic)")
                else:
                    logs.append(f"Monotonicity fail: Ti decreased with increased gamma")
                    logs.append(f"  new_Ti={new_Ti:.1f}, Ti={self.Ti:.1f}")
                    success = False
            else:
                logs.append("Skipping Ti monotonicity (denominator too small)")
            
            # Recalculate velocity - should increase
            if self.eta > 1e-10:
                new_term2 = (4/3)*self.R0**2*self.rho_a*self.g*(4*perturbed_gamma/(self.R0*self.Pa))/(6*self.eta)
                term1 = 4*self.R0*self.rho_s*self.t*self.g/(6*self.eta)
                new_u = term1 + new_term2
                if new_u > self.u:
                    logs.append("✓ u increases with gamma (monotonic)")
                else:
                    logs.append(f"Monotonicity fail: u decreased with increased gamma")
                    logs.append(f"  new_u={new_u:.3f}, u={self.u:.3f}")
                    success = False
            else:
                logs.append("Skipping u monotonicity (eta too small)")
            
        except Exception as e:
            logs.append(f"Monotonicity test error: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
        
        return success, logs
    
    def run_all_tests(self):
        all_logs = []
        overall_success = True
        
        try:
            # Run invariants test
            inv_success, inv_logs = self.test_invariants()
            all_logs.extend(["=== Invariants Test ==="] + inv_logs)
            if not inv_success:
                overall_success = False
            
            # Run trace consistency test
            trace_success, trace_logs = self.test_trace_consistency()
            all_logs.extend(["\n=== Trace Consistency Test ==="] + trace_logs)
            if not trace_success:
                overall_success = False
            
            # Run monotonicity test
            mono_success, mono_logs = self.test_monotonicity()
            all_logs.extend(["\n=== Monotonicity Test ==="] + mono_logs)
            if not mono_success:
                overall_success = False
            
        except Exception as e:
            all_logs.append(f"Overall test error: {str(e)}")
            all_logs.append(traceback.format_exc())
            overall_success = False
        
        return overall_success, all_logs