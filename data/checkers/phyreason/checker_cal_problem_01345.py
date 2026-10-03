import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        """
        self.m = getattr(solution_instance, 'm', 1.0)
        self.mu = getattr(solution_instance, 'mu', 0.2)
        self.g = getattr(solution_instance, 'g', 10.0)
        results = getattr(solution_instance, 'results', {})
        
        # Extract results with safe fallbacks
        self.s = results.get('s', 0.0)
        self.a = results.get('a', 0.0)
        self.W_F = results.get('W_F', 0.0)
        self.v_A2 = results.get('v_A2', 0.0)
        self.F_minus_mumg = results.get('F-mumg', 0.0)
        
        # Store logs for reporting
        self.logs = []
        
    def _log(self, message, success):
        """Append a test log."""
        self.logs.append((success, message))
        
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify fundamental physical laws."""
        try:
            # 1. Newton's Second Law for return trip (3s to 5s): F_net = m*a
            expected_F_net = self.m * self.a
            if self._is_close(self.F_minus_mumg, expected_F_net):
                self._log("Newton's Second Law holds for return trip.", True)
            else:
                self._log(f"Newton's Second Law violation: F_net = {self.F_minus_mumg}, expected m*a = {expected_F_net}", False)
                
            # 2. Work-Energy Theorem for entire trip (0s to 5s)
            # W_net = ΔK = 0.5*m*v_A^2 - 0
            W_friction = 2 * self.mu * self.m * self.g * self.s
            W_net = self.W_F - W_friction
            delta_K = 0.5 * self.m * self.v_A2
            if self._is_close(W_net, delta_K):
                self._log("Work-Energy Theorem holds for entire trip.", True)
            else:
                self._log(f"Work-Energy Theorem violation: W_net = {W_net}, ΔK = {delta_K}", False)
                
        except Exception as e:
            self._log(f"Error in test_invariants: {traceback.format_exc()}", False)
            
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # 1. Distance s from constant acceleration during return trip (t=2s, starting from rest)
            t_return = 2.0  # 3s to 5s
            s_expected = 0.5 * self.a * t_return**2
            if self._is_close(self.s, s_expected):
                self._log(f"Distance s consistent with kinematics: s = {self.s}, expected 0.5*a*t^2 = {s_expected}", True)
            else:
                self._log(f"Distance s inconsistent: s = {self.s}, expected 0.5*a*t^2 = {s_expected}", False)
                
            # 2. Velocity squared at A from constant acceleration: v_A^2 = 2*a*s
            v_A2_expected = 2 * self.a * self.s
            if self._is_close(self.v_A2, v_A2_expected):
                self._log(f"v_A^2 consistent: v_A^2 = {self.v_A2}, expected 2*a*s = {v_A2_expected}", True)
            else:
                self._log(f"v_A^2 inconsistent: v_A^2 = {self.v_A2}, expected 2*a*s = {v_A2_expected}", False)
                
            # 3. Work done by force W_F from energy balance: W_F = 2*μ*m*g*s + m*a*s
            W_F_expected = 2 * self.mu * self.m * self.g * self.s + self.m * self.a * self.s
            if self._is_close(self.W_F, W_F_expected):
                self._log(f"W_F consistent: W_F = {self.W_F}, expected 2*μ*m*g*s + m*a*s = {W_F_expected}", True)
            else:
                self._log(f"W_F inconsistent: W_F = {self.W_F}, expected 2*μ*m*g*s + m*a*s = {W_F_expected}", False)
                
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {traceback.format_exc()}", False)
            
    def test_monotonicity(self):
        """
        Perturb coefficient of friction μ and verify expected trends.
        Restore original μ after checking.
        """
        try:
            delta = 0.1  # small perturbation
            original_mu = self.mu
            
            # Compute force F during return trip from original results
            # F = (F - μmg) + μmg
            F = self.F_minus_mumg + original_mu * self.m * self.g
            
            # Helper to compute s and W_F for a given mu
            def compute_s_W_F(mu):
                a_new = (F - mu * self.m * self.g) / self.m
                s_new = 0.5 * a_new * (2.0)**2
                W_F_new = 2 * mu * self.m * self.g * s_new + self.m * a_new * s_new
                return s_new, W_F_new
            
            # Original values
            s_orig, W_F_orig = self.s, self.W_F
            
            # Increase mu
            mu_high = original_mu + delta
            s_high, W_F_high = compute_s_W_F(mu_high)
            
            # Decrease mu (but ensure it doesn't go negative)
            mu_low = max(0.0, original_mu - delta)
            s_low, W_F_low = compute_s_W_F(mu_low)
            
            # Check monotonicity of s with respect to mu: s should decrease as mu increases
            if s_high < s_orig < s_low:
                self._log(f"Monotonicity of s w.r.t μ holds: s(μ+{delta})={s_high:.3f} < s(μ)={s_orig:.3f} < s(μ-{delta})={s_low:.3f}", True)
            else:
                self._log(f"Monotonicity of s w.r.t μ violated: s(μ+{delta})={s_high:.3f}, s(μ)={s_orig:.3f}, s(μ-{delta})={s_low:.3f}", False)
                
            # Optional: Check monotonicity of W_F? Could be complex, so we skip or note it.
            # We'll just report the computed values for verification.
            self._log(f"Perturbation test: at μ={mu_high:.3f}, s={s_high:.3f}, W_F={W_F_high:.3f}; at μ={mu_low:.3f}, s={s_low:.3f}, W_F={W_F_low:.3f}", True)
            
        except Exception as e:
            self._log(f"Error in test_monotonicity: {traceback.format_exc()}", False)
            
    def run_all_tests(self):
        """Execute all tests and return success flag and logs."""
        self.logs = []
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        success = all(log[0] for log in self.logs)
        return success, self.logs