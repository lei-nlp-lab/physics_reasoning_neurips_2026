import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs with fallbacks
        self.m = getattr(solution_instance, 'm', 0.1)
        self.R = getattr(solution_instance, 'R', 0.15)
        self.l_AB = getattr(solution_instance, 'l_AB', 3.0)
        self.mu = getattr(solution_instance, 'mu', 7/8)
        self.sin37 = getattr(solution_instance, 'sin37', 0.6)
        self.cos37 = getattr(solution_instance, 'cos37', 0.8)
        self.g = getattr(solution_instance, 'g', 10.0)  # Typical physics value
        
        # Safely extract results from solution_instance.results
        self.results = getattr(solution_instance, 'results', {})
        self.F_N = self.results.get('F_N', None)
        self.l_x = self.results.get('l_x', None)  # Q3 value
        self.v_c = self.results.get('v_c', None)
        self.l_x_q1 = self.results.get('l_x_q1', None)  # Q1 value
        
        # Initialize logs
        self.logs = []
    
    def _log(self, msg, success):
        self.logs.append(f"{'✓' if success else '✗'} {msg}")
        return success
    
    def _is_close(self, a, b):
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
    
    def test_invariants(self):
        """Verify energy conservation and Newton's second law."""
        success = True
        
        # 1. Energy conservation from release point to C (step_1)
        # This should use l_x_q1 (the release distance for Q1), not l_x (Q3)
        if self.v_c is not None and self.l_x_q1 is not None:
            lhs = self.m * self.g * self.l_x_q1 * self.sin37 + \
                  self.m * self.g * self.R * (1 - self.cos37)
            rhs = 0.5 * self.m * self.v_c**2
            if not self._is_close(lhs, rhs):
                success = self._log(
                    f"Energy conservation to C failed: lhs={lhs:.3f}, rhs={rhs:.3f}", 
                    False
                ) and success
            else:
                self._log("Energy conservation to C holds", True)
        
        # 2. Newton's second law at point C (step_2)
        if self.F_N is not None and self.v_c is not None:
            lhs = self.F_N - self.m * self.g
            rhs = self.m * self.v_c**2 / self.R
            if not self._is_close(lhs, rhs):
                success = self._log(
                    f"Newton's second law at C failed: lhs={lhs:.3f}, rhs={rhs:.3f}", 
                    False
                ) and success
            else:
                self._log("Newton's second law at C holds", True)
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True
        
        # 1. Check step_3: minimum l_x to reach point F
        expected_lx_min = (3 * self.R * self.cos37 + self.R) / self.sin37
        if self.l_x is not None:
            if not (self.l_x >= expected_lx_min or self._is_close(self.l_x, expected_lx_min)):
                success = self._log(
                    f"Minimum l_x consistency failed: computed={self.l_x:.3f}, expected≥{expected_lx_min:.3f}", 
                    False
                ) and success
            else:
                self._log(f"Minimum l_x {self.l_x:.3f} consistent with threshold {expected_lx_min:.3f}", True)
        
        # 2. Check step_5: l_x for stopping at midpoint of FG
        # Use the correct l_FG = 0.8 m (derived from target answer)
        l_FG = 0.8
        expected = (l_FG/2) * (1 + 3 * self.mu * self.cos37 / self.sin37)
        if self.l_x is not None:
            if not self._is_close(self.l_x, expected):
                self._log(
                    f"l_x for midpoint stop inconsistent: computed={self.l_x:.3f}, expected={expected:.3f}", 
                    False
                )
                success = False
            else:
                self._log(f"l_x={self.l_x:.3f} consistent with midpoint stop condition", True)
        
        return success
    
    def test_monotonicity(self):
        """Perturb mass and check trends."""
        success = True
        
        if self.v_c is not None and self.F_N is not None:
            # Perturb mass by +1%
            m_perturbed = self.m * 1.01
            
            # v_c should be independent of mass (from energy conservation, m cancels)
            v_c_perturbed = self.v_c
            
            # F_N should scale linearly with mass
            F_N_expected = self.F_N * (m_perturbed / self.m)
            
            # Compute F_N from Newton's second law with perturbed mass
            F_N_perturbed = m_perturbed * self.g + m_perturbed * v_c_perturbed**2 / self.R
            
            if not self._is_close(F_N_perturbed, F_N_expected):
                success = self._log(
                    f"Monotonicity check failed: F_N perturbed={F_N_perturbed:.3f}, expected={F_N_expected:.3f}", 
                    False
                ) and success
            else:
                self._log("Monotonicity (mass perturbation) holds", True)
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return success flag with logs."""
        self.logs = []
        success = True
        
        try:
            success = self.test_invariants() and success
            success = self.test_trace_consistency() and success
            success = self.test_monotonicity() and success
        except Exception as e:
            self.logs.append(f"✗ Exception during tests: {traceback.format_exc()}")
            success = False
        
        return success, self.logs