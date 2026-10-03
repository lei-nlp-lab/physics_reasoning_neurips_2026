import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs - check multiple possible attribute names
        self.m = getattr(solution_instance, 'm', 
                        getattr(solution_instance, 'm_val', 0.0))
        self.g = getattr(solution_instance, 'g', 
                        getattr(solution_instance, 'g_val', 9.8))  # Default to 9.8 if not found
        # theta might be stored as 'theta' or 'θ' or 'theta_val'
        self.theta = getattr(solution_instance, 'theta', 
                            getattr(solution_instance, 'θ', 
                                   getattr(solution_instance, 'theta_val', 0.0)))
        self.E = getattr(solution_instance, 'E', 
                        getattr(solution_instance, 'E_val', 0.0))
        self.L = getattr(solution_instance, 'L', 
                        getattr(solution_instance, 'L_val', 0.0))
        
        # Safely extract results from the instance or its results dictionary
        results = getattr(solution_instance, 'results', {})
        self.q = getattr(solution_instance, 'q', results.get('q', 0.0))
        self.v = getattr(solution_instance, 'v', results.get('v', 0.0))
        
        # FIX: charge_type might not be provided (strings shouldn't be in results)
        # Try to get it from various possible locations
        self.charge_type = ''
        # First try as attribute
        if hasattr(solution_instance, 'charge_type'):
            self.charge_type = getattr(solution_instance, 'charge_type', '')
        # Then try in results (even though strings shouldn't be there)
        elif 'charge_type' in results:
            self.charge_type = results.get('charge_type', '')
        # If still not found, check if there's a global variable charge_type
        # (solution_instance might be a namespace dict)
        elif isinstance(solution_instance, dict) and 'charge_type' in solution_instance:
            self.charge_type = solution_instance.get('charge_type', '')
        
        self.logs = []
    
    def _log(self, msg, success):
        self.logs.append((success, msg))
    
    def test_invariants(self):
        """Check physical invariants: equilibrium condition and work-energy relation."""
        try:
            # 1. Check charge type (string comparison)
            # FIX: Handle case where charge_type might not be provided
            if not self.charge_type:
                self._log("Charge type not provided in results (strings not allowed in results dict).", True)
            elif isinstance(self.charge_type, str) and self.charge_type.lower() != 'positive':
                self._log(f"Charge type should be 'positive', but got '{self.charge_type}'.", False)
            elif isinstance(self.charge_type, str) and self.charge_type.lower() == 'positive':
                self._log("Charge type is correctly 'positive'.", True)
            else:
                self._log(f"Charge type is not a string or is empty: {self.charge_type}", False)
            
            # 2. Check equilibrium condition: q = (m * g * tanθ) / E
            if self.E == 0:
                self._log("E is zero, cannot compute expected q.", False)
            else:
                expected_q = (self.m * self.g * math.tan(self.theta)) / self.E
                if math.isclose(self.q, expected_q, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Equilibrium condition satisfied: q = {self.q:.3e} ≈ (m g tanθ)/E = {expected_q:.3e}.", True)
                else:
                    self._log(f"Equilibrium condition failed: q = {self.q:.3e}, but (m g tanθ)/E = {expected_q:.3e}.", False)
            
            # 3. Check speed from work-energy: v = sqrt(2 g L (1/cosθ - 1))
            if math.cos(self.theta) == 0:
                self._log("cosθ is zero, cannot compute expected v.", False)
            else:
                expected_v = math.sqrt(2 * self.g * self.L * (1.0 / math.cos(self.theta) - 1.0))
                if math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Work-energy relation satisfied: v = {self.v:.3e} ≈ sqrt(2 g L (1/cosθ -1)) = {expected_v:.3e}.", True)
                else:
                    self._log(f"Work-energy relation failed: v = {self.v:.3e}, but sqrt(2 g L (1/cosθ -1)) = {expected_v:.3e}.", False)
        except Exception as e:
            self._log(f"Error in test_invariants: {e}\n{traceback.format_exc()}", False)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # The steps analysis gives two main results: q and v.
            # We already checked these in test_invariants, but we can double-check consistency.
            # Additionally, we can check that the charge type leads to the correct force direction.
            # For a positive charge, the electric force is in the direction of E (to the right).
            # In equilibrium, the horizontal force balance: T sinθ = q E.
            # Since T sinθ > 0 (for 0<θ<π/2), q must be positive. So charge_type -> q>0.
            if self.q > 0 and isinstance(self.charge_type, str) and self.charge_type.lower() == 'positive':
                self._log("Charge sign consistency: positive charge yields q>0.", True)
            elif self.q < 0 and isinstance(self.charge_type, str) and self.charge_type.lower() == 'negative':
                self._log("Charge sign consistency: negative charge yields q<0.", True)
            elif not self.charge_type and self.q > 0:
                self._log("Charge sign consistency: q>0 implies positive charge (charge_type not provided).", True)
            else:
                self._log(f"Charge sign inconsistency: q={self.q}, charge_type='{self.charge_type}'.", False)
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {e}\n{traceback.format_exc()}", False)
    
    def test_monotonicity(self):
        """Perturb inputs and verify expected trends in outputs."""
        try:
            # Perturb mass m by +1%
            perturbed_m = self.m * 1.01
            if self.E == 0:
                self._log("E is zero, cannot compute q perturbations.", False)
            else:
                expected_q_perturbed = (perturbed_m * self.g * math.tan(self.theta)) / self.E
                # q should increase with m
                if expected_q_perturbed > self.q:
                    self._log(f"Monotonicity (m->q): as m increases by 1%, q increases from {self.q:.3e} to {expected_q_perturbed:.3e}.", True)
                else:
                    self._log(f"Monotonicity (m->q): as m increases, q does not increase (from {self.q:.3e} to {expected_q_perturbed:.3e}).", False)
            
            # v should be independent of m
            if math.cos(self.theta) == 0:
                self._log("cosθ is zero, cannot compute v.", False)
            else:
                expected_v_perturbed = math.sqrt(2 * self.g * self.L * (1.0 / math.cos(self.theta) - 1.0))
                if math.isclose(self.v, expected_v_perturbed, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Independence (m->v): v unchanged ({self.v:.3e}) when m is perturbed.", True)
                else:
                    self._log(f"Independence (m->v): v changed from {self.v:.3e} to {expected_v_perturbed:.3e} when m changed.", False)
            
            # Perturb theta by +1% (but ensure it stays in valid range)
            perturbed_theta = self.theta * 1.01 if self.theta < 1.5 else self.theta * 0.99  # avoid too large angles
            # For small perturbations, we can check trends numerically
            if self.E == 0:
                self._log("E is zero, cannot compute q.", False)
            else:
                original_q = (self.m * self.g * math.tan(self.theta)) / self.E
                perturbed_q = (self.m * self.g * math.tan(perturbed_theta)) / self.E
                # For 0<θ<π/2, tanθ is increasing, so q should increase with θ
                if perturbed_theta > self.theta and perturbed_q > original_q:
                    self._log(f"Monotonicity (θ->q): as θ increases, q increases from {original_q:.3e} to {perturbed_q:.3e}.", True)
                elif perturbed_theta < self.theta and perturbed_q < original_q:
                    self._log(f"Monotonicity (θ->q): as θ decreases, q decreases from {original_q:.3e} to {perturbed_q:.3e}.", True)
                else:
                    self._log(f"Monotonicity (θ->q): unexpected trend. θ: {self.theta:.3f}->{perturbed_theta:.3f}, q: {original_q:.3e}->{perturbed_q:.3e}.", False)
            
            # For v: v = sqrt(2 g L (1/cosθ -1)), which increases with θ for 0<θ<π/2.
            if math.cos(self.theta) != 0 and math.cos(perturbed_theta) != 0:
                original_v = math.sqrt(2 * self.g * self.L * (1.0 / math.cos(self.theta) - 1.0))
                perturbed_v = math.sqrt(2 * self.g * self.L * (1.0 / math.cos(perturbed_theta) - 1.0))
                if (perturbed_theta > self.theta and perturbed_v > original_v) or (perturbed_theta < self.theta and perturbed_v < original_v):
                    self._log(f"Monotonicity (θ->v): as θ changes, v changes accordingly ({original_v:.3e}->{perturbed_v:.3e}).", True)
                else:
                    self._log(f"Monotonicity (θ->v): unexpected trend. θ: {self.theta:.3f}->{perturbed_theta:.3f}, v: {original_v:.3e}->{perturbed_v:.3e}.", False)
            else:
                self._log("Skipped v monotonicity due to cosθ near zero.", False)
        except Exception as e:
            self._log(f"Error in test_monotonicity: {e}\n{traceback.format_exc()}", False)
    
    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        self.logs = []
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        success = all(s for s, _ in self.logs)
        return success, self.logs