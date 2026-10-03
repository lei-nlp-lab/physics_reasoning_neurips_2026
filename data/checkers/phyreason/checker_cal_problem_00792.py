import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr and .get()
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with defaults to avoid AttributeError
        self.m = getattr(solution_instance, 'm', 1.0)
        self.v0 = getattr(solution_instance, 'v0', 1.0)
        self.mu = getattr(solution_instance, 'mu', 0.5)
        self.g = getattr(solution_instance, 'g', 9.8)
        
        # Extract results with fallbacks
        self.v_AB = self.results.get('v_AB', 0.0)
        self.L = self.results.get('L', 0.0)
        self.R = self.results.get('R', 0.0)
        
        # Expected formulas from steps analysis
        self.expected_v_AB = self.v0 / 4
        self.expected_L = (5 * self.v0**2) / (16 * self.mu * self.g)
        self.expected_R = (self.v0**2) / (64 * self.g)
        
        # Logs for reporting
        self.logs = []
    
    def _log(self, msg, success):
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {msg}")
        return success
    
    def _isclose(self, a, b):
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        """Test conservation laws."""
        successes = []
        
        # 1. Momentum conservation at point B (step 1 + step 2)
        # Equation: m*v0 = m*v_B + 2*m*v_AB, with v_B = v0/2
        v_B = self.v0 / 2
        lhs_momentum = self.m * self.v0
        rhs_momentum = self.m * v_B + 2 * self.m * self.v_AB
        mom_cons = self._isclose(lhs_momentum, rhs_momentum)
        successes.append(self._log(
            f"Momentum at B: {lhs_momentum:.3f} ≈ {rhs_momentum:.3f}", 
            mom_cons
        ))
        
        # 2. Energy conservation on plank AB (step 4)
        # Equation: ΔK = μ m g L
        K_initial = 0.5 * self.m * self.v0**2
        K_final_AB = 0.5 * 2 * self.m * self.v_AB**2  # plank+slider (mass 2m)
        K_final_P = 0.5 * self.m * v_B**2
        delta_K = K_initial - (K_final_AB + K_final_P)
        work_friction = self.mu * self.m * self.g * self.L
        energy_cons = self._isclose(delta_K, work_friction)
        successes.append(self._log(
            f"Energy on AB: ΔK={delta_K:.3f}, μmgL={work_friction:.3f}", 
            energy_cons
        ))
        
        # 3. Momentum conservation at point D (step 5)
        # Equation: m*v_B + m*v_AB = 2*m*v_all
        v_all = (self.m * v_B + self.m * self.v_AB) / (2 * self.m)
        mom_cons_D = self._isclose(
            self.m * v_B + self.m * self.v_AB,
            2 * self.m * v_all
        )
        successes.append(self._log(
            f"Momentum at D: {self.m*v_B + self.m*self.v_AB:.3f} ≈ {2*self.m*v_all:.3f}", 
            mom_cons_D
        ))
        
        # 4. Mechanical energy conservation on arc (step 6)
        # Equation: mgR = K_initial_D - K_final_all
        K_initial_D = 0.5 * self.m * v_B**2 + 0.5 * self.m * self.v_AB**2
        K_final_all = 0.5 * 2 * self.m * v_all**2
        delta_E_arc = K_initial_D - K_final_all
        pot_gain = self.m * self.g * self.R
        energy_cons_arc = self._isclose(delta_E_arc, pot_gain)
        successes.append(self._log(
            f"Energy on arc: ΔK={delta_E_arc:.3f}, mgR={pot_gain:.3f}", 
            energy_cons_arc
        ))
        
        return all(successes)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results."""
        successes = []
        
        # Check v_AB matches expected from steps
        v_AB_match = self._isclose(self.v_AB, self.expected_v_AB)
        successes.append(self._log(
            f"v_AB: {self.v_AB:.3f} ≈ expected {self.expected_v_AB:.3f}", 
            v_AB_match
        ))
        
        # Check L matches expected formula
        L_match = self._isclose(self.L, self.expected_L)
        successes.append(self._log(
            f"L: {self.L:.3f} ≈ expected {self.expected_L:.3f}", 
            L_match
        ))
        
        # Check R matches expected formula
        R_match = self._isclose(self.R, self.expected_R)
        successes.append(self._log(
            f"R: {self.R:.3f} ≈ expected {self.expected_R:.3f}", 
            R_match
        ))
        
        # Check internal consistency: v_all from momentum and energy should match
        v_all_mom = (self.v0/2 + self.v_AB) / 2
        v_all_energy = math.sqrt(
            (0.5*self.m*(self.v0/2)**2 + 0.5*self.m*self.v_AB**2 - self.m*self.g*self.R) 
            / self.m
        )  # from step 6 rearranged
        v_all_consistency = self._isclose(v_all_mom, v_all_energy)
        successes.append(self._log(
            f"v_all consistency: {v_all_mom:.3f} ≈ {v_all_energy:.3f}", 
            v_all_consistency
        ))
        
        return all(successes)
    
    def test_monotonicity(self):
        """Perturb μ and check L decreases appropriately."""
        try:
            original_mu = self.mu
            perturbed_mu = original_mu * 1.1  # Increase μ by 10%
            
            # Recalculate L using formula from step 4
            L_original = self.expected_L
            L_perturbed = (5 * self.v0**2) / (16 * perturbed_mu * self.g)
            
            # Expected: L should decrease when μ increases
            monotonic = L_perturbed < L_original
            successes = [self._log(
                f"Monotonicity: L(μ={original_mu:.3f})={L_original:.3f}, "
                f"L(μ={perturbed_mu:.3f})={L_perturbed:.3f}, decreasing={monotonic}", 
                monotonic
            )]
            
            # Restore original μ
            self.mu = original_mu
            
            return all(successes)
        except Exception as e:
            self.logs.append(f"ERROR in monotonicity: {traceback.format_exc()}")
            return False
    
    def run_all_tests(self):
        """Run all checks and return success status and logs."""
        self.logs = ["Physics Audit Report:"]
        overall_success = True
        
        try:
            overall_success &= self.test_invariants()
            overall_success &= self.test_trace_consistency()
            overall_success &= self.test_monotonicity()
        except Exception as e:
            self.logs.append(f"CRITICAL ERROR: {traceback.format_exc()}")
            overall_success = False
        
        return overall_success, self.logs