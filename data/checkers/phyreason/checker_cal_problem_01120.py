import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs with defaults
        self.m = getattr(solution_instance, 'm', 2.0)
        self.M = getattr(solution_instance, 'M', 1.0)
        self.v0 = getattr(solution_instance, 'v0', 3.0)
        self.mu = getattr(solution_instance, 'mu', 0.2)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Safe extraction of results from solution_instance or its results dict
        results = getattr(solution_instance, 'results', {})
        self.a_M = results.get('a_M', 0.0)
        self.t = results.get('t', 0.0)
        self.a_m = results.get('a_m', 0.0)
        self.F_f = results.get('F_f', 0.0)
        self.v = results.get('v', 0.0)
        self.x_2 = results.get('x_2', 0.0)
        self.x_1 = results.get('x_1', 0.0)
        self.L = results.get('L', 0.0)
        
        self.logs = []
    
    def _log(self, message, success=None):
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {message}")
        else:
            self.logs.append(f"INFO: {message}")
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        try:
            # 1. Newton's 2nd Law: F_f = m * a_m
            expected_a_m = self.mu * self.g
            if not self._is_close(self.a_m, expected_a_m):
                self._log(f"Newton 2nd Law (block): a_m={self.a_m}, expected {expected_a_m}", False)
                return False
            self._log(f"Newton 2nd Law (block) satisfied: a_m={self.a_m} ≈ μ*g={expected_a_m}", True)
            
            # 2. Newton's 2nd Law: F_f = M * a_M
            expected_a_M = self.mu * self.m * self.g / self.M
            if not self._is_close(self.a_M, expected_a_M):
                self._log(f"Newton 2nd Law (board): a_M={self.a_M}, expected {expected_a_M}", False)
                return False
            self._log(f"Newton 2nd Law (board) satisfied: a_M={self.a_M} ≈ μ*m*g/M={expected_a_M}", True)
            
            # 3. Friction force: F_f = μ * m * g
            expected_F_f = self.mu * self.m * self.g
            if not self._is_close(self.F_f, expected_F_f):
                self._log(f"Friction force: F_f={self.F_f}, expected μ*m*g={expected_F_f}", False)
                return False
            self._log(f"Friction force correct: F_f={self.F_f} ≈ μ*m*g={expected_F_f}", True)
            
            # 4. Momentum conservation (system isolated in horizontal direction)
            initial_momentum = self.m * self.v0
            final_momentum = (self.m + self.M) * self.v
            if not self._is_close(initial_momentum, final_momentum):
                self._log(f"Momentum conservation: initial={initial_momentum}, final={final_momentum}", False)
                return False
            self._log(f"Momentum conservation satisfied: {initial_momentum} ≈ {final_momentum}", True)
            
            # 5. Time from velocity equations: v = v0 - a_m*t and v = a_M*t
            t_from_block = (self.v0 - self.v) / self.a_m if self.a_m != 0 else 0
            t_from_board = self.v / self.a_M if self.a_M != 0 else 0
            if not (self._is_close(t_from_block, self.t) and self._is_close(t_from_board, self.t)):
                self._log(f"Time consistency: t={self.t}, from block={t_from_block}, from board={t_from_board}", False)
                return False
            self._log(f"Time consistent from both equations: t={self.t}", True)
            
            return True
        except Exception as e:
            self._log(f"Invariants test failed with error: {str(e)}", False)
            return False
    
    def test_trace_consistency(self):
        try:
            # Check intermediate results against steps analysis
            expected_values = {
                'F_f': 4.0,
                'a_m': 2.0,
                'a_M': 4.0,
                'v': 2.0,
                't': 0.5,
                'x_1': 0.5,
                'x_2': 1.25,
                'L': 0.75
            }
            
            all_pass = True
            for key, expected in expected_values.items():
                actual = getattr(self, key)
                if not self._is_close(actual, expected):
                    self._log(f"Trace mismatch for {key}: {actual} ≠ expected {expected}", False)
                    all_pass = False
                else:
                    self._log(f"Trace match for {key}: {actual} ≈ {expected}", True)
            
            # Check displacement relationship: L = x_2 - x_1
            if not self._is_close(self.L, self.x_2 - self.x_1):
                self._log(f"Displacement relation: L={self.L}, x_2-x_1={self.x_2-self.x_1}", False)
                all_pass = False
            else:
                self._log(f"Displacement relation satisfied: L={self.L} = x_2-x_1={self.x_2-self.x_1}", True)
            
            return all_pass
        except Exception as e:
            self._log(f"Trace consistency test failed with error: {str(e)}", False)
            return False
    
    def test_monotonicity(self):
        try:
            # The monotonicity test is problematic because L is not monotonic with μ
            # L = (v0²) / (2*(a_m + a_M)) * (a_M/a_m) = (v0² * M) / (2*g*(m+M)*μ)
            # This shows L is inversely proportional to μ!
            # When μ increases, L decreases, which is what we observed
            # So we should test that L decreases when μ increases
            
            perturbed_mu = self.mu * 1.1  # +10%
            
            # Recalculate using correct formulas
            a_m_pert = perturbed_mu * self.g
            a_M_pert = perturbed_mu * self.m * self.g / self.M
            v_pert = self.v0 * a_M_pert / (a_m_pert + a_M_pert)
            t_pert = v_pert / a_M_pert
            x1_pert = 0.5 * a_M_pert * t_pert**2
            x2_pert = self.v0 * t_pert - 0.5 * a_m_pert * t_pert**2
            L_pert = x2_pert - x1_pert
            
            # L should DECREASE when μ increases (more friction → less relative sliding)
            if L_pert < self.L:
                self._log(f"Monotonicity (increase μ): L decreases from {self.L:.4f} to {L_pert:.4f} as expected", True)
            else:
                self._log(f"Monotonicity (increase μ): L increases from {self.L:.4f} to {L_pert:.4f} (unexpected)", False)
                return False
            
            return True
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {str(e)}", False)
            return False
    
    def run_all_tests(self):
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"Running {test_name}...", None)
            try:
                if not test_func():
                    success = False
            except Exception as e:
                self._log(f"{test_name} crashed: {traceback.format_exc()}", False)
                success = False
        
        return success, self.logs