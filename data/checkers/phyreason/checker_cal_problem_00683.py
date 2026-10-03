import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Input values from problem statement (fixed)
        self.R1 = 2.0
        self.R2 = 2.0
        self.R3 = 2.0
        self.I_a_measured = 3.0  # Ammeter reading when S2 at a
        self.I_b_measured = 2.5  # Ammeter reading when S2 at b
        
        # Safe extraction of results from solution instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract computed values with safe fallbacks
        self.r = getattr(solution_instance, 'r', self.results.get('r', 0.0))
        self.R_parallel = getattr(solution_instance, 'R_parallel', self.results.get('R_parallel', 0.0))
        self.I_b = getattr(solution_instance, 'I_b', self.results.get('I_b', 0.0))
        self.I = getattr(solution_instance, 'I', self.results.get('I', 0.0))
        self.I_a = getattr(solution_instance, 'I_a', self.results.get('I_a', 0.0))
        self.P_m = getattr(solution_instance, 'P_m', self.results.get('P_m', 0.0))
        self.E = getattr(solution_instance, 'E', self.results.get('E', 0.0))
        
        self.logs = []
        
    def _log(self, message, success):
        self.logs.append(f"{'✓' if success else '✗'} {message}")
        
    def test_invariants(self):
        """Test fundamental physics invariants (Ohm's Law, Energy Conservation)"""
        successes = []
        
        # 1. Check parallel resistance calculation
        R_parallel_expected = (self.R1 * self.R2) / (self.R1 + self.R2)
        if math.isclose(self.R_parallel, R_parallel_expected, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Parallel resistance correct: {self.R_parallel:.2f} Ω ≈ {R_parallel_expected:.2f} Ω", True)
            successes.append(True)
        else:
            self._log(f"Parallel resistance mismatch: {self.R_parallel:.2f} Ω ≠ {R_parallel_expected:.2f} Ω", False)
            successes.append(False)
        
        # 2. Check that both circuit equations give same E and r
        # Equation 1: E = I_a * (R_parallel + R3 + r)
        E_from_a = self.I_a_measured * (self.R_parallel + self.R3 + self.r)
        # Equation 2: E = I_b * (R2 + R3 + r)
        E_from_b = self.I_b_measured * (self.R2 + self.R3 + self.r)
        
        if math.isclose(E_from_a, E_from_b, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Circuit equations consistent: E_a={E_from_a:.2f}V ≈ E_b={E_from_b:.2f}V", True)
            successes.append(True)
        else:
            self._log(f"Circuit equations inconsistent: E_a={E_from_a:.2f}V ≠ E_b={E_from_b:.2f}V", False)
            successes.append(False)
            
        # 3. Check that computed E matches either equation
        if (math.isclose(self.E, E_from_a, rel_tol=1e-2, abs_tol=1e-2) or 
            math.isclose(self.E, E_from_b, rel_tol=1e-2, abs_tol=1e-2)):
            self._log(f"Computed E={self.E:.2f}V consistent with circuit equations", True)
            successes.append(True)
        else:
            self._log(f"Computed E={self.E:.2f}V doesn't match circuit equations", False)
            successes.append(False)
            
        # 4. Check Ohm's Law for configuration c: I = E/(R2 + r)
        I_expected = self.E / (self.R2 + self.r) if (self.R2 + self.r) != 0 else 0
        if math.isclose(self.I, I_expected, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Ohm's Law for config c: I={self.I:.2f}A ≈ E/(R2+r)={I_expected:.2f}A", True)
            successes.append(True)
        else:
            self._log(f"Ohm's Law violation for config c: I={self.I:.2f}A ≠ E/(R2+r)={I_expected:.2f}A", False)
            successes.append(False)
            
        return all(successes)
    
    def test_trace_consistency(self):
        """Check consistency between intermediate steps and final results"""
        successes = []
        
        # 1. Check step 1: R_parallel calculation
        R_parallel_calc = (self.R1 * self.R2) / (self.R1 + self.R2)
        if math.isclose(self.R_parallel, R_parallel_calc, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Step 1 consistent: R_parallel={self.R_parallel:.2f}Ω matches calculation", True)
            successes.append(True)
        else:
            self._log(f"Step 1 inconsistent: R_parallel={self.R_parallel:.2f}Ω ≠ {R_parallel_calc:.2f}Ω", False)
            successes.append(False)
            
        # 2. Check step 6: E and r from simultaneous equations
        # Solve symbolically: E = 3*(1+2+r) and E = 2.5*(2+2+r)
        r_symbolic, E_symbolic = sp.symbols('r E')
        eq1 = sp.Eq(E_symbolic, 3 * (3 + r_symbolic))  # R_parallel + R3 = 1+2=3
        eq2 = sp.Eq(E_symbolic, 2.5 * (4 + r_symbolic))  # R2 + R3 = 2+2=4
        
        try:
            sol = sp.solve((eq1, eq2), (r_symbolic, E_symbolic))
            if sol:
                r_expected = float(sol[r_symbolic])
                E_expected = float(sol[E_symbolic])
                
                if (math.isclose(self.r, r_expected, rel_tol=1e-2, abs_tol=1e-2) and
                    math.isclose(self.E, E_expected, rel_tol=1e-2, abs_tol=1e-2)):
                    self._log(f"Step 6 consistent: r={self.r:.2f}Ω, E={self.E:.2f}V matches simultaneous eq solution", True)
                    successes.append(True)
                else:
                    self._log(f"Step 6 inconsistent: r={self.r:.2f}Ω, E={self.E:.2f}V ≠ expected r={r_expected:.2f}Ω, E={E_expected:.2f}V", False)
                    successes.append(False)
        except:
            # Fallback to numerical check
            r_num = (self.I_b_measured * 4 - self.I_a_measured * 3) / (self.I_a_measured - self.I_b_measured)
            E_num = self.I_a_measured * (3 + r_num)
            if (math.isclose(self.r, r_num, rel_tol=1e-2, abs_tol=1e-2) and
                math.isclose(self.E, E_num, rel_tol=1e-2, abs_tol=1e-2)):
                self._log(f"Step 6 consistent (numerical): r={self.r:.2f}Ω, E={self.E:.2f}V", True)
                successes.append(True)
            else:
                self._log(f"Step 6 inconsistent: r={self.r:.2f}Ω, E={self.E:.2f}V ≠ expected", False)
                successes.append(False)
                
        # 3. Check step 9: Power calculation
        P_expected = self.I**2 * self.R2
        if math.isclose(self.P_m, P_expected, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Step 9 consistent: P_m={self.P_m:.3f}W = I²×R2={P_expected:.3f}W", True)
            successes.append(True)
        else:
            self._log(f"Step 9 inconsistent: P_m={self.P_m:.3f}W ≠ I²×R2={P_expected:.3f}W", False)
            successes.append(False)
            
        # 4. Check maximum power condition (step 8)
        if math.isclose(self.r, self.R2, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Maximum power condition satisfied: r={self.r:.2f}Ω = R2={self.R2:.2f}Ω", True)
            successes.append(True)
        else:
            self._log(f"Maximum power condition NOT satisfied: r={self.r:.2f}Ω ≠ R2={self.R2:.2f}Ω", False)
            successes.append(False)
            
        return all(successes)
    
    def test_monotonicity(self):
        """Test that power changes correctly when perturbing parameters"""
        successes = []
        
        # 1. Test maximum power theorem: power should decrease if load resistance deviates from source resistance
        original_r = self.r
        original_R2 = self.R2
        original_E = self.E
        
        # Calculate power at three points: R_L = r, R_L = r+0.5, R_L = r-0.5
        R_L_values = [original_r, original_r + 0.5, original_r - 0.5]
        
        # Filter out negative resistance
        R_L_values = [r for r in R_L_values if r > 0]
        
        powers = []
        for R_L in R_L_values:
            I_perturbed = original_E / (original_r + R_L)
            P_perturbed = I_perturbed**2 * R_L
            powers.append(P_perturbed)
        
        # Power at R_L = r should be maximum
        if powers:
            max_power = max(powers)
            power_at_match = powers[0]  # First element is R_L = r
            
            if math.isclose(power_at_match, max_power, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Maximum power theorem: P(r=R_L)={power_at_match:.3f}W is maximum", True)
                successes.append(True)
            else:
                self._log(f"Maximum power theorem violated: P(r=R_L)={power_at_match:.3f}W < max={max_power:.3f}W", False)
                successes.append(False)
        
        # 2. Test monotonicity of current with respect to E
        # If E increases, current should increase (Ohm's Law)
        E_perturbed = original_E * 1.1  # Increase by 10%
        I_perturbed = E_perturbed / (original_r + original_R2)
        
        if I_perturbed > self.I:
            self._log(f"Ohm's Law monotonicity: I increases with E (ΔE=+10% → ΔI≈{100*(I_perturbed/self.I-1):.1f}%)", True)
            successes.append(True)
        else:
            self._log(f"Ohm's Law monotonicity violated: I doesn't increase with E", False)
            successes.append(False)
            
        # 3. Test monotonicity of current with respect to total resistance
        # If total resistance increases, current should decrease
        r_perturbed = original_r * 1.1  # Increase by 10%
        I_perturbed2 = original_E / (r_perturbed + original_R2)
        
        if I_perturbed2 < self.I:
            self._log(f"Ohm's Law monotonicity: I decreases with r (Δr=+10% → ΔI≈{100*(I_perturbed2/self.I-1):.1f}%)", True)
            successes.append(True)
        else:
            self._log(f"Ohm's Law monotonicity violated: I doesn't decrease with r", False)
            successes.append(False)
            
        return all(successes)
    
    def run_all_tests(self):
        """Run all tests and return summary"""
        self.logs = []  # Reset logs
        
        try:
            invariants_ok = self.test_invariants()
            trace_ok = self.test_trace_consistency()
            monotonicity_ok = self.test_monotonicity()
            
            all_ok = invariants_ok and trace_ok and monotonicity_ok
            
            # Add summary log
            self.logs.append("\n" + "="*50)
            self.logs.append(f"SUMMARY:")
            self.logs.append(f"  Invariants: {'PASS' if invariants_ok else 'FAIL'}")
            self.logs.append(f"  Trace Consistency: {'PASS' if trace_ok else 'FAIL'}")
            self.logs.append(f"  Monotonicity: {'PASS' if monotonicity_ok else 'FAIL'}")
            self.logs.append(f"  OVERALL: {'PASS' if all_ok else 'FAIL'}")
            
            return all_ok, self.logs
            
        except Exception as e:
            error_msg = f"Error during testing: {str(e)}\n{traceback.format_exc()}"
            self.logs.append(error_msg)
            return False, self.logs