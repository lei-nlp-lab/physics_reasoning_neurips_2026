import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for inputs
        self.R = getattr(solution_instance, 'R', solution_instance.results.get('R', 1.0))
        self.T1 = getattr(solution_instance, 'T1', solution_instance.results.get('T1', 1.0))
        self.T2 = getattr(solution_instance, 'T2', solution_instance.results.get('T2', 1.0))
        self.Rs = getattr(solution_instance, 'Rs', solution_instance.results.get('Rs', 1.0))
        self.Ts = getattr(solution_instance, 'Ts', solution_instance.results.get('Ts', 1.0))
        self.R0 = getattr(solution_instance, 'R0', solution_instance.results.get('R0', 1.0))
        self.sigma = 5.670374419e-8  # Stefan-Boltzmann constant (W/m^2/K^4)
        
        # Safe attribute access for results
        self.P = getattr(solution_instance, 'P', solution_instance.results.get('P', 0.0))
        self.I = getattr(solution_instance, 'I', solution_instance.results.get('I', 0.0))
        self.P_w = getattr(solution_instance, 'P_w', solution_instance.results.get('P_w', 0.0))
        self.Q1 = getattr(solution_instance, 'Q_1', solution_instance.results.get('Q_1', 0.0))
        self.Q2 = getattr(solution_instance, 'Q_2', solution_instance.results.get('Q_2', 0.0))
        self.ratio = getattr(solution_instance, 'P_w/P', solution_instance.results.get('P_w/P', 0.0))
        
        self.logs = []
        
    def _log(self, msg, success):
        self.logs.append((success, msg))
        
    def test_invariants(self):
        """Verify fundamental physics laws."""
        try:
            # 1. Inverse Square Law for Radiation
            I_expected = self.sigma * self.Ts**4 * (self.Rs**2 / self.R0**2)
            if not math.isclose(self.I, I_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Inverse square law violation: I={self.I}, expected={I_expected}", False)
            else:
                self._log("Inverse square law satisfied", True)
                
            # 2. Power = Intensity * Area (cross-sectional area πR^2)
            P_from_I = self.I * math.pi * self.R**2
            if not math.isclose(self.P, P_from_I, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Power-Intensity-Area mismatch: P={self.P}, I*πR^2={P_from_I}", False)
            else:
                self._log("Power = Intensity * Area satisfied", True)
                
            # 3. Energy Conservation: Q1 = P_w + Q2 (from step 3)
            if not math.isclose(self.Q1, self.P_w + self.Q2, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Energy conservation violation: Q1={self.Q1}, P_w+Q2={self.P_w+self.Q2}", False)
            else:
                self._log("Energy conservation (Q1 = P_w + Q2) satisfied", True)
                
            # 4. Stefan-Boltzmann for hemispheres (area 2πR^2 each)
            Q2_rad = 2 * math.pi * self.R**2 * self.sigma * self.T2**4
            if not math.isclose(self.Q2, Q2_rad, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Dark side radiation mismatch: Q2={self.Q2}, 2πR^2σT2^4={Q2_rad}", False)
            else:
                self._log("Dark side Stefan-Boltzmann law satisfied", True)
                
            # Bright side: P = Q1 + (2πR^2σ)T1^4 (from step 4)
            bright_rad = 2 * math.pi * self.R**2 * self.sigma * self.T1**4
            if not math.isclose(self.P, self.Q1 + bright_rad, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Bright side energy balance violation: P={self.P}, Q1+2πR^2σT1^4={self.Q1+bright_rad}", False)
            else:
                self._log("Bright side energy balance satisfied", True)
                
            # 5. Reversible heat engine condition: Q1/T1 = Q2/T2
            if self.T1 > 0 and self.T2 > 0:
                left = self.Q1 / self.T1
                right = self.Q2 / self.T2
                if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Reversible engine condition violation: Q1/T1={left}, Q2/T2={right}", False)
                else:
                    self._log("Reversible heat engine condition satisfied", True)
                    
        except Exception as e:
            self._log(f"Error in test_invariants: {traceback.format_exc()}", False)
            
    def test_trace_consistency(self):
        """Check intermediate results against final answers."""
        try:
            # Symbolic variables
            R, Rs, R0, Ts, T1, T2, sigma = sp.symbols('R Rs R0 Ts T1 T2 sigma', positive=True)
            x = sp.symbols('x', positive=True)
            
            # Step 2: P expression
            P_expr_step2 = sp.pi * sigma * Ts**4 * (R**2 * Rs**2) / R0**2
            # Step 7: P expression in terms of T1 and x = T2/T1
            x_val = self.T2 / self.T1 if self.T1 != 0 else 0
            P_expr_step7 = 2 * sp.pi * R**2 * sigma * T1**4 * (1 + x**3)
            
            # Evaluate numerically
            subs_dict = {
                R: self.R,
                Rs: self.Rs,
                R0: self.R0,
                Ts: self.Ts,
                T1: self.T1,
                T2: self.T2,
                sigma: self.sigma,
                x: x_val
            }
            
            P_step2 = float(P_expr_step2.subs(subs_dict))
            P_step7 = float(P_expr_step7.subs(subs_dict))
            
            # Check consistency between step 2 and step 7 expressions (they should match)
            if not math.isclose(P_step2, P_step7, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"P from step 2 ({P_step2}) and step 7 ({P_step7}) are inconsistent", False)
            else:
                self._log("P expressions from steps 2 and 7 are consistent", True)
                
            # Check against actual P
            if not math.isclose(self.P, P_step2, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"P result {self.P} doesn't match step 2 expression {P_step2}", False)
            else:
                self._log("P matches step 2 expression", True)
                
            # Step 8: P_w expression
            P_w_expr = (x**3 * (1 - x) / (1 + x**3)) * P_expr_step2
            P_w_calc = float(P_w_expr.subs(subs_dict))
            
            if not math.isclose(self.P_w, P_w_calc, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"P_w result {self.P_w} doesn't match step 8 expression {P_w_calc}", False)
            else:
                self._log("P_w matches step 8 expression", True)
                
            # Step 10: Ratio P_w/P ≈ 0.08 (for maximum)
            if self.P > 0:
                ratio_calc = self.P_w / self.P
                if not math.isclose(ratio_calc, self.ratio, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"P_w/P ratio mismatch: stored={self.ratio}, computed={ratio_calc}", False)
                else:
                    self._log("P_w/P ratio consistent", True)
                    
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {traceback.format_exc()}", False)
            
    def test_monotonicity(self):
        """Perturb inputs and check expected trends."""
        try:
            original_T1 = self.T1
            perturbed_T1 = original_T1 * 1.05  # +5%
            
            # Recalculate P_w using the formula from step 8 with perturbed T1
            # Note: x = T2/T1 changes because T1 changes (T2 fixed)
            x_original = self.T2 / original_T1 if original_T1 != 0 else 0
            x_perturbed = self.T2 / perturbed_T1 if perturbed_T1 != 0 else 0
            
            # Expression for P_w from step 8: P_w = [x^3(1-x)/(1+x^3)] * P
            # But P itself depends on T1 through step 7: P ∝ T1^4 * (1 + x^3)
            # So we need to recalculate P for perturbed T1
            
            # Original P (from step 7)
            P_original = 2 * math.pi * self.R**2 * self.sigma * original_T1**4 * (1 + x_original**3)
            P_w_original = (x_original**3 * (1 - x_original) / (1 + x_original**3)) * P_original
            
            # Perturbed P
            P_perturbed = 2 * math.pi * self.R**2 * self.sigma * perturbed_T1**4 * (1 + x_perturbed**3)
            P_w_perturbed = (x_perturbed**3 * (1 - x_perturbed) / (1 + x_perturbed**3)) * P_perturbed
            
            # The function P_w(T1) is not monotonic in general; it has a maximum.
            # Instead, check that the ratio P_w/P changes as expected when x changes.
            # For fixed T2, increasing T1 decreases x, which should initially increase P_w/P
            # up to a maximum then decrease. We can check the derivative sign numerically.
            
            # Compute derivative approximation
            delta_T1 = perturbed_T1 - original_T1
            delta_Pw_over_P = (P_w_perturbed/P_perturbed) - (P_w_original/P_original)
            
            # For the given problem, the maximum occurs at x ≈ 0.75 (from solving derivative=0)
            # If x_original > 0.75, increasing T1 (decreasing x) should increase P_w/P
            # If x_original < 0.75, increasing T1 (decreasing x) should decrease P_w/P
            x_critical = 0.75  # approximate maximum from derivative analysis
            
            if x_original > x_critical:
                expected_positive = delta_Pw_over_P > 0
                if expected_positive:
                    self._log(f"At x={x_original:.3f}>0.75, increasing T1 increases P_w/P (as expected)", True)
                else:
                    self._log(f"At x={x_original:.3f}>0.75, increasing T1 decreased P_w/P (unexpected)", False)
            elif x_original < x_critical:
                expected_negative = delta_Pw_over_P < 0
                if expected_negative:
                    self._log(f"At x={x_original:.3f}<0.75, increasing T1 decreases P_w/P (as expected)", True)
                else:
                    self._log(f"At x={x_original:.3f}<0.75, increasing T1 increased P_w/P (unexpected)", False)
            else:
                self._log(f"At x≈0.75 (critical point), change is small (expected)", True)
                
            # Restore original value (not strictly needed as we didn't modify instance)
            self.T1 = original_T1
            
        except Exception as e:
            self._log(f"Error in test_monotonicity: {traceback.format_exc()}", False)
            
    def run_all_tests(self):
        """Execute all checks and return success status and logs."""
        self.logs = []
        
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        # Determine overall success: all logs must be True (except errors which are False)
        success = all(log[0] for log in self.logs)
        
        return success, self.logs