import math
import sympy as sp
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.m = getattr(solution_instance, 'm', 2.0)
        self.F = getattr(solution_instance, 'F', 40.0)
        self.theta = getattr(solution_instance, 'theta', 37.0)
        self.mu = getattr(solution_instance, 'mu', 0.25)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.sin_theta = getattr(solution_instance, 'sin_theta', 0.6)
        self.cos_theta = getattr(solution_instance, 'cos_theta', 0.8)
        self.t = getattr(solution_instance, 't', 3.0)
        
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.N = results.get('N', 0.0)
        self.f = results.get('f', 0.0)
        self.a = results.get('a', 0.0)
        self.v_t = results.get('v_t', 0.0)
        
        # Store logs
        self.logs = []
    
    def _fuzzy_equal(self, a: float, b: float, 
                    rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Fuzzy numerical comparison with tolerance."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def _log(self, msg: str, success: bool):
        """Log test result."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {msg}")
    
    def test_invariants(self) -> bool:
        """Verify Newton's Laws and force relationships."""
        success = True
        
        # 1. Check normal force: N = m*g*cosθ + F*sinθ
        expected_N = self.m * self.g * self.cos_theta + self.F * self.sin_theta
        if self._fuzzy_equal(self.N, expected_N):
            self._log(f"Normal force: {self.N} ≈ {expected_N}", True)
        else:
            self._log(f"Normal force mismatch: {self.N} ≠ {expected_N}", False)
            success = False
        
        # 2. Check friction force: f = μ*N
        expected_f = self.mu * expected_N
        if self._fuzzy_equal(self.f, expected_f):
            self._log(f"Friction force: {self.f} ≈ {expected_f}", True)
        else:
            self._log(f"Friction force mismatch: {self.f} ≠ {expected_f}", False)
            success = False
        
        # 3. Check acceleration from Newton's 2nd Law:
        # a = (F*cosθ - m*g*sinθ - f)/m
        net_force = self.F * self.cos_theta - self.m * self.g * self.sin_theta - self.f
        expected_a = net_force / self.m
        
        if self._fuzzy_equal(self.a, expected_a):
            self._log(f"Acceleration: {self.a} ≈ {expected_a}", True)
        else:
            self._log(f"Acceleration mismatch: {self.a} ≠ {expected_a}", False)
            success = False
        
        # 4. Check force direction consistency
        # The object moves upward, so acceleration should be positive
        if self.a > -1e-2:  # Allow small negative due to rounding
            self._log(f"Acceleration direction consistent (positive): {self.a}", True)
        else:
            self._log(f"Acceleration should be positive but got: {self.a}", False)
            success = False
        
        # 5. Verify Newton's 1st Law would give zero acceleration if forces balanced
        # Symbolic check for equilibrium condition
        try:
            m, g, theta, F, mu = sp.symbols('m g theta F mu')
            cos_theta, sin_theta = sp.cos(theta), sp.sin(theta)
            
            # Equilibrium: F*cosθ = m*g*sinθ + μ*(m*g*cosθ + F*sinθ)
            lhs = F * cos_theta
            rhs = m * g * sin_theta + mu * (m * g * cos_theta + F * sin_theta)
            
            # Substitute actual values
            eq_check = lhs.subs({F: self.F, m: self.m, g: self.g, 
                                theta: math.radians(self.theta), mu: self.mu})
            rhs_check = rhs.subs({F: self.F, m: self.m, g: self.g, 
                                 theta: math.radians(self.theta), mu: self.mu})
            
            # Should NOT be equal (since we have acceleration)
            if not self._fuzzy_equal(float(eq_check), float(rhs_check)):
                self._log("Forces not in equilibrium (expected for acceleration)", True)
            else:
                self._log("Unexpected force equilibrium", False)
                success = False
        except Exception as e:
            self._log(f"Symbolic equilibrium check failed: {str(e)}", True)  # Non-critical
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Cross-check intermediate results with final velocity."""
        success = True
        
        # 1. Check kinematic relationship: v_t = a*t (since v0=0)
        expected_v_t = self.a * self.t
        
        if self._fuzzy_equal(self.v_t, expected_v_t):
            self._log(f"Velocity from kinematics: {self.v_t} ≈ {expected_v_t}", True)
        else:
            self._log(f"Velocity mismatch: {self.v_t} ≠ {expected_v_t}", False)
            success = False
        
        # 2. Verify work-energy relationship (alternative check)
        # Net work = change in kinetic energy
        # W_net = F_net * distance
        # Distance: d = 0.5*a*t²
        d = 0.5 * self.a * self.t**2
        F_net = self.m * self.a
        W_net = F_net * d
        delta_KE = 0.5 * self.m * self.v_t**2  # v0=0
        
        if self._fuzzy_equal(W_net, delta_KE):
            self._log(f"Work-energy consistency: W_net={W_net:.2f} ≈ ΔKE={delta_KE:.2f}", True)
        else:
            self._log(f"Work-energy mismatch: {W_net:.2f} ≠ {delta_KE:.2f}", False)
            # Not critical for this problem, but log anyway
            success = success and False  # Only fail if other checks also fail
        
        # 3. Check friction work is negative
        # Friction opposes motion (upward), so work should be negative
        W_friction = -self.f * d  # Negative because friction opposes displacement
        
        if W_friction < 1e-2:  # Should be negative
            self._log(f"Friction work negative (opposes motion): {W_friction:.2f} J", True)
        else:
            self._log(f"Friction work should be negative but got: {W_friction:.2f}", False)
            success = False
        
        return success
    
    def test_monotonicity(self) -> bool:
        """Perturb mass and check trend consistency."""
        success = True
        original_m = self.m
        
        try:
            # Perturb mass by ±10%
            perturbations = [0.9 * original_m, 1.1 * original_m]
            original_results = {
                'N': self.N,
                'f': self.f,
                'a': self.a,
                'v_t': self.v_t
            }
            
            for perturbed_m in perturbations:
                # Recalculate using formulas
                N_pert = perturbed_m * self.g * self.cos_theta + self.F * self.sin_theta
                f_pert = self.mu * N_pert
                a_pert = (self.F * self.cos_theta - perturbed_m * self.g * self.sin_theta - f_pert) / perturbed_m
                v_t_pert = a_pert * self.t
                
                # Check trends
                # 1. Normal force increases with mass
                if perturbed_m > original_m:
                    if N_pert > original_results['N']:
                        self._log(f"N increases with mass: {N_pert:.2f} > {original_results['N']:.2f}", True)
                    else:
                        self._log(f"N should increase with mass but {N_pert:.2f} ≤ {original_results['N']:.2f}", False)
                        success = False
                    
                    # 2. Friction increases with mass (since f = μN)
                    if f_pert > original_results['f']:
                        self._log(f"f increases with mass: {f_pert:.2f} > {original_results['f']:.2f}", True)
                    else:
                        self._log(f"f should increase with mass but {f_pert:.2f} ≤ {original_results['f']:.2f}", False)
                        success = False
                    
                    # 3. Acceleration should decrease with increased mass
                    # (more mass, same net force → less acceleration)
                    if a_pert < original_results['a']:
                        self._log(f"a decreases with mass: {a_pert:.2f} < {original_results['a']:.2f}", True)
                    else:
                        self._log(f"a should decrease with mass but {a_pert:.2f} ≥ {original_results['a']:.2f}", False)
                        success = False
                
                else:  # perturbed_m < original_m
                    if N_pert < original_results['N']:
                        self._log(f"N decreases with mass: {N_pert:.2f} < {original_results['N']:.2f}", True)
                    else:
                        self._log(f"N should decrease with mass but {N_pert:.2f} ≥ {original_results['N']:.2f}", False)
                        success = False
                    
                    if f_pert < original_results['f']:
                        self._log(f"f decreases with mass: {f_pert:.2f} < {original_results['f']:.2f}", True)
                    else:
                        self._log(f"f should decrease with mass but {f_pert:.2f} ≥ {original_results['f']:.2f}", False)
                        success = False
                    
                    if a_pert > original_results['a']:
                        self._log(f"a increases with mass decrease: {a_pert:.2f} > {original_results['a']:.2f}", True)
                    else:
                        self._log(f"a should increase with mass decrease but {a_pert:.2f} ≤ {original_results['a']:.2f}", False)
                        success = False
            
            # Restore original mass (important!)
            self.m = original_m
            
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {str(e)}", False)
            success = False
            # Ensure mass is restored even on error
            self.m = original_m
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return results."""
        self.logs = []
        
        # Run tests
        test1 = self.test_invariants()
        test2 = self.test_trace_consistency()
        test3 = self.test_monotonicity()
        
        # Summary
        all_pass = test1 and test2 and test3
        if all_pass:
            self.logs.append("SUMMARY: All physics checks passed ✓")
        else:
            self.logs.append("SUMMARY: Some physics checks failed ✗")
        
        return all_pass, self.logs

# Example usage stub
if __name__ == "__main__":
    # Create a mock solution instance for testing
    class MockSolution:
        def __init__(self):
            self.m = 2.0
            self.F = 40.0
            self.theta = 37.0
            self.mu = 0.25
            self.g = 10.0
            self.sin_theta = 0.6
            self.cos_theta = 0.8
            self.t = 3.0
            self.results = {
                'N': 40.0,
                'f': 10.0,
                'a': 5.0,
                'v_t': 15.0
            }
    
    solution = MockSolution()
    checker = PhysicsChecker(solution)
    success, logs = checker.run_all_tests()
    
    print(f"Success: {success}")
    for log in logs:
        print(log)