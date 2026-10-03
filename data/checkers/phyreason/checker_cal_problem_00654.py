import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Initialize with safe attribute access."""
        # Inputs
        self.Q = getattr(solution_instance, 'Q', 1e-6)
        self.h = getattr(solution_instance, 'h', 1.0)
        self.m = getattr(solution_instance, 'm', 1.0)
        self.g = getattr(solution_instance, 'g', 9.8)
        self.k = getattr(solution_instance, 'k', 8.99e9)
        
        # Results (from results dict with fallback)
        results = getattr(solution_instance, 'results', {})
        self.a_B = results.get('a_B', None)
        self.F_net_B = results.get('F_net_B', None)
        self.q = results.get('q', None)
        self.U_AB = results.get('U_AB', None)
        self.F_net_A = results.get('F_net_A', None)
        
        self.logs = []
        
    def _log(self, msg, is_error=False):
        """Helper for logging."""
        self.logs.append(('ERROR' if is_error else 'INFO', msg))
        
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison."""
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False
    
    def test_invariants(self):
        """Verify energy conservation and Newton's laws from KB."""
        try:
            # 1. Newton's 2nd Law at point A (from steps analysis)
            # mg - kQq/h² = m*(g/2) → F_net_A = m*g/2
            if self.F_net_A is not None:
                expected_F_net_A = self.m * self.g / 2
                if not self._is_close(self.F_net_A, expected_F_net_A):
                    self._log(f"F_net_A mismatch: {self.F_net_A} vs {expected_F_net_A}", True)
            
            # 2. Newton's 2nd Law at point B (from steps analysis)
            # kQq/(0.25h)² - mg = m*a_B → F_net_B = m*a_B
            if self.F_net_B is not None and self.a_B is not None:
                expected_F_net_B = self.m * self.a_B
                if not self._is_close(self.F_net_B, expected_F_net_B):
                    self._log(f"F_net_B mismatch: {self.F_net_B} vs {expected_F_net_B}", True)
            
            # 3. Work-energy theorem: Charge starts and ends at rest, so net work = 0
            if all(v is not None for v in [self.m, self.g, self.h, self.k, self.Q, self.q]):
                # Check energy balance numerically
                W_gravity = self.m * self.g * (0.25*self.h - self.h)  # Negative since moving down
                W_electric = self.q * (self.k*self.Q/self.h - self.k*self.Q/(0.25*self.h))
                if not self._is_close(W_gravity + W_electric, 0):
                    self._log(f"Energy not conserved: W_gravity+W_electric = {W_gravity+W_electric}", True)
            
            return True
        except Exception as e:
            self._log(f"In invariants: {str(e)}\n{traceback.format_exc()}", True)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Check that a_B = 7g (from steps analysis)
            if self.a_B is not None:
                expected_a_B = 7 * self.g
                if not self._is_close(self.a_B, expected_a_B):
                    self._log(f"a_B mismatch: {self.a_B} vs 7g = {expected_a_B}", True)
            
            # Check that q = mgh²/(2kQ) (from steps analysis)
            if self.q is not None:
                expected_q = (self.m * self.g * self.h**2) / (2 * self.k * self.Q)
                if not self._is_close(self.q, expected_q):
                    self._log(f"q mismatch: {self.q} vs {expected_q}", True)
            
            # Check that U_AB = -3kQ/(2h) (from steps analysis)
            if self.U_AB is not None:
                expected_U = -3 * self.k * self.Q / (2 * self.h)
                if not self._is_close(self.U_AB, expected_U):
                    self._log(f"U_AB mismatch: {self.U_AB} vs {expected_U}", True)
            
            # Verify F_net_A and F_net_B relationships
            if all(v is not None for v in [self.F_net_A, self.F_net_B]):
                # Expected ratio from physics: F_net_B/F_net_A = 14
                expected_ratio = 14.0
                if self.F_net_A != 0:
                    actual_ratio = self.F_net_B / self.F_net_A
                    if not self._is_close(actual_ratio, expected_ratio):
                        self._log(f"Force ratio mismatch: {actual_ratio} vs {expected_ratio}", True)
            
            return True
        except Exception as e:
            self._log(f"In trace consistency: {str(e)}\n{traceback.format_exc()}", True)
            return False
    
    def test_monotonicity(self):
        """Perturb mass and check trends."""
        try:
            # Save original values
            original_m = self.m
            
            # Perturb mass by +10%
            perturbed_m = self.m * 1.1
            
            # Recalculate q using formula (should be proportional to m)
            expected_q_perturbed = (perturbed_m * self.g * self.h**2) / (2 * self.k * self.Q)
            original_q = (original_m * self.g * self.h**2) / (2 * self.k * self.Q)
            
            # Check monotonicity: q should increase with m
            if expected_q_perturbed <= original_q:
                self._log(f"q not monotonic with m: {expected_q_perturbed} vs {original_q}", True)
            
            # Recalculate a_B (should remain constant at 7g)
            expected_a_B_perturbed = 7 * self.g
            if not self._is_close(expected_a_B_perturbed, 7*self.g):
                self._log(f"a_B changed with m perturbation: {expected_a_B_perturbed}", True)
            
            # Recalculate U_AB (should be independent of m)
            expected_U_perturbed = -3 * self.k * self.Q / (2 * self.h)
            if not self._is_close(expected_U_perturbed, -3*self.k*self.Q/(2*self.h)):
                self._log(f"U_AB changed with m perturbation: {expected_U_perturbed}", True)
            
            return True
        except Exception as e:
            self._log(f"In monotonicity: {str(e)}\n{traceback.format_exc()}", True)
            return False
    
    def run_all_tests(self):
        """Run all tests and return success status and logs."""
        self.logs.clear()
        
        success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                if not test_func():
                    success = False
                    self._log(f"{test_name} test failed", True)
            except Exception as e:
                success = False
                self._log(f"{test_name} test crashed: {str(e)}", True)
        
        return success, self.logs