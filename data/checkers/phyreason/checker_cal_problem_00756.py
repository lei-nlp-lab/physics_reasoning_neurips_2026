import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from solution_instance."""
        self.solution = solution_instance
        
        # Inputs (with defaults if not present)
        self.m = getattr(solution_instance, 'm', 0.4)
        self.v = getattr(solution_instance, 'v', 2.0)
        self.M = getattr(solution_instance, 'M', 1.6)
        self.mu = getattr(solution_instance, 'mu', 0.4)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Results (prefer results dictionary, then direct attributes)
        if hasattr(solution_instance, 'results'):
            self.v_prime = solution_instance.results.get('v_prime', 
                         solution_instance.results.get('v\'', 0.0))
            self.s = solution_instance.results.get('s', 0.0)
        else:
            self.v_prime = getattr(solution_instance, 'v_prime', 
                         getattr(solution_instance, 'v\'', 0.0))
            self.s = getattr(solution_instance, 's', 0.0)
        
        # Store logs for reporting
        self.logs = []
    
    def _log(self, msg, is_error=False):
        """Helper to log messages."""
        self.logs.append(("ERROR" if is_error else "INFO", msg))
    
    def test_invariants(self):
        """Verify conservation of momentum and work-energy theorem."""
        try:
            # Conservation of momentum: initial momentum = final momentum
            p_initial = self.m * self.v
            p_final = (self.m + self.M) * self.v_prime
            if not math.isclose(p_initial, p_final, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Momentum not conserved: p_initial={p_initial:.3f}, p_final={p_final:.3f}", is_error=True)
                return False
            self._log("Momentum conservation verified.")
            
            # Work-energy theorem: friction work = change in kinetic energy of the system (block+cart)
            # Initial KE: only block moving
            KE_initial = 0.5 * self.m * self.v**2
            # Final KE: both moving together
            KE_final = 0.5 * (self.m + self.M) * self.v_prime**2
            # Work done by friction (negative): force = μ*m*g, displacement = s (relative displacement between block and cart)
            # The work done by friction on the system (block+cart) is actually the negative of the work done by friction on the block 
            # relative to the cart. The energy loss in the system equals the work done by friction (which is dissipated as heat).
            # So: KE_initial = KE_final + |work_friction|, where work_friction = μ*m*g*s (positive magnitude).
            work_friction = self.mu * self.m * self.g * self.s
            energy_loss = KE_initial - KE_final
            if not math.isclose(energy_loss, work_friction, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Work-energy mismatch: energy_loss={energy_loss:.3f}, work_friction={work_friction:.3f}", is_error=True)
                return False
            self._log("Work-energy theorem verified.")
            return True
        except Exception as e:
            self._log(f"Error in test_invariants: {str(e)}", is_error=True)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with formulas from steps."""
        try:
            # Step 1: v' from momentum conservation
            v_prime_expected = self.m * self.v / (self.m + self.M)
            if not math.isclose(self.v_prime, v_prime_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"v' mismatch: expected {v_prime_expected:.3f}, got {self.v_prime:.3f}", is_error=True)
                return False
            self._log(f"v' trace consistency verified: {self.v_prime:.3f} m/s.")
            
            # Step 2: s from work-energy
            numerator = 0.5 * self.m * self.v**2 - 0.5 * (self.m + self.M) * self.v_prime**2
            denominator = self.mu * self.m * self.g
            if denominator == 0:
                self._log("Denominator zero in s calculation.", is_error=True)
                return False
            s_expected = numerator / denominator
            if not math.isclose(self.s, s_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"s mismatch: expected {s_expected:.3f}, got {self.s:.3f}", is_error=True)
                return False
            self._log(f"s trace consistency verified: {self.s:.3f} m.")
            return True
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {str(e)}", is_error=True)
            return False
    
    def test_monotonicity(self):
        """Perturb mass m and verify trends in v' and s."""
        try:
            # Store original values
            m_original = self.m
            v_prime_original = self.v_prime
            s_original = self.s
            
            # Perturb m by +10%
            delta = 0.1
            self.m = m_original * (1 + delta)
            
            # Recompute v' and s using formulas (without modifying anything else)
            v_prime_perturbed = self.m * self.v / (self.m + self.M)
            numerator = 0.5 * self.m * self.v**2 - 0.5 * (self.m + self.M) * v_prime_perturbed**2
            denominator = self.mu * self.m * self.g
            s_perturbed = numerator / denominator
            
            # Check trends:
            # 1. v' should increase with m (since dv'/dm = v*M/(m+M)^2 > 0)
            if not v_prime_perturbed > v_prime_original:
                self._log(f"v' monotonicity failed: {v_prime_perturbed:.3f} <= {v_prime_original:.3f}", is_error=True)
                self.m = m_original  # Restore
                return False
            
            # 2. s should decrease with m? Let's analyze:
            # s = (0.5*m*v^2 - 0.5*(m+M)*v'^2) / (μ*m*g)
            # Substitute v' = m*v/(m+M)
            # Then s = [0.5*m*v^2 - 0.5*(m+M)*(m^2*v^2/(m+M)^2)] / (μ*m*g)
            # = [0.5*m*v^2 - 0.5*m^2*v^2/(m+M)] / (μ*m*g)
            # = [0.5*v^2 * (m - m^2/(m+M))] / (μ*m*g)
            # = [0.5*v^2 * (m*(m+M) - m^2) / (m+M)] / (μ*m*g)
            # = [0.5*v^2 * (m*M) / (m+M)] / (μ*m*g)
            # = (0.5*v^2 * M) / (μ*g * (m+M))
            # So s is decreasing in m (since denominator increases with m).
            if not s_perturbed < s_original:
                self._log(f"s monotonicity failed: {s_perturbed:.3f} >= {s_original:.3f}", is_error=True)
                self.m = m_original
                return False
            
            self._log("Monotonicity tests passed: v' increases and s decreases with increased m.")
            
            # Restore original values
            self.m = m_original
            return True
        except Exception as e:
            self._log(f"Error in test_monotonicity: {str(e)}", is_error=True)
            # Attempt to restore m in case of error
            if 'm_original' in locals():
                self.m = m_original
            return False
    
    def run_all_tests(self):
        """Execute all tests and return (success_bool, logs_list)."""
        self.logs.clear()
        success = True
        
        # Run each test, stopping early if any fails (optional)
        if not self.test_invariants():
            success = False
        if not self.test_trace_consistency():
            success = False
        if not self.test_monotonicity():
            success = False
        
        return success, self.logs

# Optional: Example usage (if run as script)
if __name__ == "__main__":
    # Mock a solution instance for testing
    class MockSolution:
        def __init__(self):
            self.m = 0.4
            self.v = 2.0
            self.M = 1.6
            self.mu = 0.4
            self.g = 10.0
            self.results = {
                'v_prime': 0.4,
                's': 0.4
            }
    
    mock_solution = MockSolution()
    checker = PhysicsChecker(mock_solution)
    success, logs = checker.run_all_tests()
    for level, msg in logs:
        print(f"{level}: {msg}")
    print(f"\nOverall success: {success}")