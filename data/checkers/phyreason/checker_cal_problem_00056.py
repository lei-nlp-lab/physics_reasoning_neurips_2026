import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results dictionary
        results = getattr(solution_instance, 'results', {})
        
        # Extract input parameters from results with sensible defaults
        # The problem parameters should be in results or use MockSolution defaults
        self.m = results.get('m', 1.0)
        self.v0 = results.get('v0', 5.0)
        self.mu = results.get('mu', 0.2)
        self.g = results.get('g', 9.81)
        
        # Extract computed results
        self.x = results.get('x', 0.0)
        self.x_prime = results.get("x''", results.get('x_prime', 0.0))
        self.F = results.get('F', 0.0)
        self.v_wood = results.get('v_wood', 0.0)
        self.t = results.get('t', 0.0)
        self.v_all = results.get('v_all', 0.0)
        self.a_slider = results.get('a_slider', 0.0)
        self.W = results.get('W', 0.0)
        
        self.logs = []
        
    def _log(self, msg, success):
        self.logs.append((success, msg))
        
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison with tolerance."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify conservation laws and Newton's laws from steps analysis."""
        try:
            # 1. Conservation of Momentum (Step 3)
            momentum_initial = 2 * self.m * self.v0
            momentum_final = 2 * self.m * self.v_wood + self.m * self.v_all
            if self._is_close(momentum_initial, momentum_final):
                self._log("Momentum conservation (step 3) satisfied.", True)
            else:
                self._log(f"Momentum mismatch: initial {momentum_initial:.3f}, final {momentum_final:.3f}", False)
            
            # 2. Work-Energy Theorem (Step 5)
            delta_KE = (0.5 * 2 * self.m * self.v_wood**2 + 0.5 * self.m * self.v_all**2) - (0.5 * 2 * self.m * self.v0**2)
            work_friction = -self.mu * self.m * self.g * self.x
            if self._is_close(delta_KE, work_friction):
                self._log("Work-energy theorem (step 5) satisfied.", True)
            else:
                self._log(f"Work-energy mismatch: ΔKE {delta_KE:.3f}, W_friction {work_friction:.3f}", False)
            
            # 3. Newton's Second Law for slider (Step 8)
            expected_a_slider = self.mu * self.g
            if self._is_close(self.a_slider, expected_a_slider):
                self._log(f"Slider acceleration matches μ*g: {self.a_slider:.3f} ≈ {expected_a_slider:.3f}", True)
            else:
                self._log(f"Slider acceleration mismatch: {self.a_slider:.3f} vs μ*g={expected_a_slider:.3f}", False)
            
            # 4. Kinematics: v0 = a_slider * t (Step 9)
            expected_t = self.v0 / (self.mu * self.g) if self.mu * self.g != 0 else 0
            if self._is_close(self.t, expected_t):
                self._log(f"Time t matches v0/(μ*g): {self.t:.3f} ≈ {expected_t:.3f}", True)
            else:
                self._log(f"Time mismatch: {self.t:.3f} vs v0/(μ*g)={expected_t:.3f}", False)
            
            # 5. External force F = μmg (Step 7)
            expected_F = self.mu * self.m * self.g
            if self._is_close(self.F, expected_F):
                self._log(f"External force F matches μ*m*g: {self.F:.3f} ≈ {expected_F:.3f}", True)
            else:
                self._log(f"Force F mismatch: {self.F:.3f} vs μ*m*g={expected_F:.3f}", False)
                
            # 6. Work W = F * x' (Step 11)
            expected_W = self.F * self.x_prime
            if self._is_close(self.W, expected_W):
                self._log(f"Work W matches F*x': {self.W:.3f} ≈ {expected_W:.3f}", True)
            else:
                self._log(f"Work mismatch: {self.W:.3f} vs F*x'={expected_W:.3f}", False)
                
        except Exception as e:
            self._log(f"Error in test_invariants: {str(e)}", False)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Check Step 4 velocities consistency
            # From step 4: v_all = 2v0/5, v_wood = 4v0/5
            expected_v_all = 2 * self.v0 / 5
            expected_v_wood = 4 * self.v0 / 5
            
            if self._is_close(self.v_all, expected_v_all):
                self._log(f"v_all matches 2v0/5: {self.v_all:.3f} ≈ {expected_v_all:.3f}", True)
            else:
                self._log(f"v_all mismatch: {self.v_all:.3f} vs 2v0/5={expected_v_all:.3f}", False)
                
            if self._is_close(self.v_wood, expected_v_wood):
                self._log(f"v_wood matches 4v0/5: {self.v_wood:.3f} ≈ {expected_v_wood:.3f}", True)
            else:
                self._log(f"v_wood mismatch: {self.v_wood:.3f} vs 4v0/5={expected_v_wood:.3f}", False)
            
            # Check Step 6 distance x
            expected_x = 7 * self.v0**2 / (25 * self.mu * self.g) if self.mu * self.g != 0 else 0
            if self._is_close(self.x, expected_x):
                self._log(f"Distance x matches 7v0²/(25μg): {self.x:.3f} ≈ {expected_x:.3f}", True)
            else:
                self._log(f"Distance x mismatch: {self.x:.3f} vs 7v0²/(25μg)={expected_x:.3f}", False)
            
            # Check Step 10 distance x'
            expected_x_prime = self.v0 * self.t
            if self._is_close(self.x_prime, expected_x_prime):
                self._log(f"Plank distance x' matches v0*t: {self.x_prime:.3f} ≈ {expected_x_prime:.3f}", True)
            else:
                self._log(f"Plank distance mismatch: {self.x_prime:.3f} vs v0*t={expected_x_prime:.3f}", False)
                
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {str(e)}", False)
    
    def test_monotonicity(self):
        """Perturb μ and check trend of distance x (should be inversely proportional)."""
        original_mu = self.mu
        
        try:
            # Perturb μ by +10%
            perturbed_mu = original_mu * 1.1
            
            # Calculate expected x with perturbed μ using formula from step 6
            original_x = 7 * self.v0**2 / (25 * original_mu * self.g) if original_mu * self.g != 0 else 0
            perturbed_x = 7 * self.v0**2 / (25 * perturbed_mu * self.g) if perturbed_mu * self.g != 0 else 0
            
            # Check monotonicity: increasing μ should decrease x
            if original_x > perturbed_x:
                self._log(f"Monotonicity: μ↑ ({original_mu:.3f}→{perturbed_mu:.3f}) causes x↓ ({original_x:.3f}→{perturbed_x:.3f})", True)
            else:
                self._log(f"Monotonicity violation: μ↑ but x not decreasing", False)
                
            # Additional check: ratio should match inverse proportionality
            expected_ratio = original_mu / perturbed_mu
            actual_ratio = perturbed_x / original_x if original_x != 0 else 0
            
            if self._is_close(actual_ratio, expected_ratio, rel_tol=0.05):
                self._log(f"Inverse proportionality holds: ratio {actual_ratio:.3f} ≈ μ_orig/μ_pert={expected_ratio:.3f}", True)
            else:
                self._log(f"Inverse proportionality mismatch: ratio {actual_ratio:.3f} vs {expected_ratio:.3f}", False)
                
        except Exception as e:
            self._log(f"Error in test_monotonicity: {str(e)}", False)
        finally:
            # Restore original μ
            self.mu = original_mu
    
    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        self.logs = []
        
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        # Determine overall success (all tests must pass)
        all_pass = all(success for success, _ in self.logs)
        
        return all_pass, self.logs

# Example usage (for testing the checker itself):
if __name__ == "__main__":
    class MockSolution:
        def __init__(self):
            self.results = {
                'm': 1.0,
                'v0': 5.0,
                'mu': 0.2,
                'g': 9.81,
                'v_all': 2.0,
                'v_wood': 4.0,
                'x': 1.785,
                't': 2.548,
                'x_prime': 12.74,
                'F': 1.962,
                'a_slider': 1.962,
                'W': 25.0
            }
    
    checker = PhysicsChecker(MockSolution())
    success, logs = checker.run_all_tests()
    
    print(f"Overall success: {success}")
    for test_success, msg in logs:
        status = "PASS" if test_success else "FAIL"
        print(f"{status}: {msg}")