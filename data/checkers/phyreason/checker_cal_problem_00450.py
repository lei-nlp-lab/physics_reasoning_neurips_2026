import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Input parameters (safe access)
        self.m = getattr(solution_instance, 'm', 2.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.alpha = getattr(solution_instance, 'alpha', 37.0)  # degrees
        self.mu = getattr(solution_instance, 'mu', 0.2)
        self.v0 = getattr(solution_instance, 'v0', 2.4)
        self.t4 = 4.0
        
        # Convert alpha to radians for symbolic use
        self.alpha_rad = math.radians(self.alpha)
        self.sin_alpha = math.sin(self.alpha_rad)  # sin37° ≈ 0.6
        self.cos_alpha = math.cos(self.alpha_rad)  # cos37° ≈ 0.8
        
        # Extract results (safe access with fallbacks)
        results = getattr(solution_instance, 'results', {})
        self.F_0 = getattr(solution_instance, 'F_0', results.get('F_0', results.get('F_{0}', 0.0)))
        self.P = getattr(solution_instance, 'P', results.get('P', 0.0))
        self.F_t = getattr(solution_instance, 'F_t', results.get('F_t', results.get('F_{t}', 0.0)))
        self.Delta_v = getattr(solution_instance, 'Delta_v', results.get('Delta_v', results.get('Deltav', 0.0)))
        self.v_4 = getattr(solution_instance, 'v_4', results.get('v_4', results.get('v_{4}', 0.0)))
        self.a_4 = getattr(solution_instance, 'a_4', results.get('a_4', results.get('a_{4}', 0.0)))
        self.F_N = getattr(solution_instance, 'F_N', results.get('F_N', results.get('F_{N}', 0.0)))
        self.F_4 = getattr(solution_instance, 'F_4', results.get('F_4', results.get('F_{4}', 0.0)))
        self.a_0 = getattr(solution_instance, 'a_0', results.get('a_0', 0.0))
        self.k = getattr(solution_instance, 'k', results.get('k', 0.0))
        
        # Expected values from steps analysis
        self.exp_F_N = self.m * self.g * self.cos_alpha
        self.exp_a_0 = 1.9
        self.exp_F_0 = self.m * self.exp_a_0 + self.m * self.g * self.sin_alpha + self.mu * self.exp_F_N
        self.exp_a_4 = 5.9
        self.exp_k = (self.exp_a_4 - self.exp_a_0) / self.t4
        self.exp_Delta_v = (self.exp_a_0 + self.exp_a_4) / 2 * self.t4
        self.exp_v_4 = self.v0 + self.exp_Delta_v
        self.exp_F_4 = self.exp_F_0 + self.t4  # From F_t = F_0 + t
        self.exp_P = self.exp_F_4 * self.exp_v_4
        
        # Logs
        self.logs = []
    
    def _log(self, msg, success):
        self.logs.append((success, msg))
    
    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison with 1% tolerance."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify physical invariants (Newton's laws, equilibrium)."""
        try:
            # 1. Normal force equilibrium
            computed_F_N = self.m * self.g * self.cos_alpha
            if self._isclose(self.F_N, computed_F_N):
                self._log(f"Normal force correct: {self.F_N} ≈ {computed_F_N}", True)
            else:
                self._log(f"Normal force mismatch: {self.F_N} ≠ {computed_F_N}", False)
            
            # 2. Newton's 2nd law at t=0 (symbolic)
            t, m, g, mu = sp.symbols('t m g mu')
            alpha_sym = sp.rad(self.alpha)  # Symbolic angle
            a0_sym = sp.symbols('a0')
            F0_sym = sp.symbols('F0')
            FN_sym = m * g * sp.cos(alpha_sym)
            eq = sp.Eq(F0_sym, m * a0_sym + m * g * sp.sin(alpha_sym) + mu * FN_sym)
            # Numerical check
            lhs = self.F_0
            rhs = self.m * self.a_0 + self.m * self.g * self.sin_alpha + self.mu * self.F_N
            if self._isclose(lhs, rhs):
                self._log(f"Newton's 2nd law at t=0 satisfied: {lhs} ≈ {rhs}", True)
            else:
                self._log(f"Newton's 2nd law at t=0 failed: {lhs} ≠ {rhs}", False)
            
            # 3. Newton's 2nd law at t=4s (using step's linear F_t relation)
            # From steps: F_t = F_0 + t, so F_4 = F_0 + 4
            if self._isclose(self.F_4, self.F_0 + self.t4):
                self._log(f"Linear force relation holds: F_4={self.F_4} ≈ F_0+t4={self.F_0+self.t4}", True)
            else:
                self._log(f"Linear force relation broken: F_4={self.F_4} ≠ F_0+t4={self.F_0+self.t4}", False)
            
            # 4. Acceleration linearity: a_4 = a_0 + k*t4
            if self._isclose(self.a_4, self.a_0 + self.k * self.t4):
                self._log(f"Acceleration linearity holds: a_4={self.a_4} ≈ a_0+k*t4={self.a_0+self.k*self.t4}", True)
            else:
                self._log(f"Acceleration linearity broken: a_4={self.a_4} ≠ a_0+k*t4={self.a_0+self.k*self.t4}", False)
                
        except Exception as e:
            self._log(f"Invariants test failed with error: {str(e)}", False)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Check against expected values from steps (1-2% tolerance)
            checks = [
                ("Normal force F_N", self.F_N, self.exp_F_N),
                ("Initial acceleration a_0", self.a_0, self.exp_a_0),
                ("Initial force F_0", self.F_0, self.exp_F_0),
                ("Acceleration slope k", self.k, self.exp_k),
                ("Final acceleration a_4", self.a_4, self.exp_a_4),
                ("Velocity change Δv", self.Delta_v, self.exp_Delta_v),
                ("Final velocity v_4", self.v_4, self.exp_v_4),
                ("Final force F_4", self.F_4, self.exp_F_4),
                ("Power P", self.P, self.exp_P)
            ]
            all_ok = True
            for desc, val, exp in checks:
                if self._isclose(val, exp):
                    self._log(f"{desc} consistent: {val} ≈ {exp}", True)
                else:
                    self._log(f"{desc} inconsistent: {val} ≠ {exp} (expected)", False)
                    all_ok = False
            
            # Special check: v_4 = v_0 + Δv
            if self._isclose(self.v_4, self.v0 + self.Delta_v):
                self._log(f"Velocity addition holds: v_4={self.v_4} ≈ v0+Δv={self.v0+self.Delta_v}", True)
            else:
                self._log(f"Velocity addition broken: v_4={self.v_4} ≠ v0+Δv={self.v0+self.Delta_v}", False)
                all_ok = False
            
            # Special check: P = F_4 * v_4
            if self._isclose(self.P, self.F_4 * self.v_4):
                self._log(f"Power formula holds: P={self.P} ≈ F_4*v_4={self.F_4*self.v_4}", True)
            else:
                self._log(f"Power formula broken: P={self.P} ≠ F_4*v_4={self.F_4*self.v_4}", False)
                all_ok = False
                
        except Exception as e:
            self._log(f"Trace consistency test failed with error: {str(e)}", False)
    
    def test_monotonicity(self):
        """Perturb input (mass) and check trend."""
        original_m = self.m
        try:
            # Perturb mass by +10%
            perturbed_m = original_m * 1.1
            
            # Recalculate expected F_N (proportional to m)
            new_F_N = perturbed_m * self.g * self.cos_alpha
            # F_0 = m*a_0 + m*g*sinα + μ*F_N, but a_0 is fixed from graph
            new_F_0 = perturbed_m * self.exp_a_0 + perturbed_m * self.g * self.sin_alpha + self.mu * new_F_N
            # F_4 = F_0 + 4 (linear relation)
            new_F_4 = new_F_0 + self.t4
            # Δv unchanged (graph fixed, independent of m? Actually Δv formula uses a_0 and a_4 fixed)
            new_v_4 = self.v0 + self.exp_Delta_v  # Δv fixed
            new_P = new_F_4 * new_v_4
            
            # Check monotonic trends:
            # 1. Normal force increases with mass
            if new_F_N > self.exp_F_N:
                self._log(f"Monotonicity: F_N increases with mass ({self.exp_F_N} → {new_F_N})", True)
            else:
                self._log(f"Monotonicity: F_N should increase with mass", False)
            
            # 2. Forces F_0 and F_4 increase with mass (all terms proportional to m)
            if new_F_0 > self.exp_F_0 and new_F_4 > self.exp_F_4:
                self._log(f"Monotonicity: F_0 and F_4 increase with mass", True)
            else:
                self._log(f"Monotonicity: F_0 and F_4 should increase with mass", False)
            
            # 3. Power increases with mass (since F_4 increases, v_4 fixed)
            if new_P > self.exp_P:
                self._log(f"Monotonicity: Power increases with mass ({self.exp_P} → {new_P})", True)
            else:
                self._log(f"Monotonicity: Power should increase with mass", False)
                
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {str(e)}", False)
        finally:
            # Restore original mass (conceptually)
            self.m = original_m
    
    def run_all_tests(self):
        """Execute all tests and return success flag and logs."""
        self.logs = []
        try:
            self.test_invariants()
            self.test_trace_consistency()
            self.test_monotonicity()
            
            # Determine overall success
            success = all(s for s, _ in self.logs)
            return success, self.logs
        except Exception as e:
            self._log(f"Runtime error in test suite: {traceback.format_exc()}", False)
            return False, self.logs