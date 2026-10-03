import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs with defaults
        self.M = getattr(solution_instance, 'M', 1.5)
        self.L = getattr(solution_instance, 'L', 0.16)
        self.H = getattr(solution_instance, 'H', 0.06)
        self.h = getattr(solution_instance, 'h', 0.02)
        self.mu1 = getattr(solution_instance, 'mu1', 0.3)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Safely extract results with defaults
        self.results = getattr(solution_instance, 'results', {})
        self.m = self.results.get('m', 0.0)
        self.f = self.results.get('f', 0.0)
        self.F_N = self.results.get('F_N', 0.0)
        self.F = self.results.get('F', 0.0)  # Could be numeric or string expression
        self.f_desk = self.results.get('f_desk', 0.0)
        self.mu2 = self.results.get('mu₂', 0.0)
        
        # Derived symbolic variables
        self.x = sp.symbols('x', real=True, nonnegative=True)
        self.logs = []
    
    def _log(self, msg, success):
        self.logs.append((success, msg))
    
    def test_invariants(self):
        """Verify force balance, friction laws, and energy considerations."""
        success = True
        
        # 1. Check mass distribution for first part
        m_expected = (self.H - self.h) / self.H * self.M
        if not math.isclose(self.m, m_expected, rel_tol=1e-2, abs_tol=1e-2):
            msg = f"Mass m mismatch: expected {m_expected}, got {self.m}"
            self._log(msg, False)
            success = False
        else:
            self._log(f"Mass m correct: {self.m}", True)
        
        # 2. Check friction force from paper (two sides)
        f_expected = 2 * self.mu1 * self.m * self.g
        if not math.isclose(self.f, f_expected, rel_tol=1e-2, abs_tol=1e-2):
            msg = f"Friction f mismatch: expected {f_expected}, got {self.f}"
            self._log(msg, False)
            success = False
        else:
            self._log(f"Friction f correct: {self.f}", True)
        
        # 3. Check desk friction force
        f_desk_expected = self.mu2 * self.M * self.g
        if not math.isclose(self.f_desk, f_desk_expected, rel_tol=1e-2, abs_tol=1e-2):
            msg = f"Desk friction mismatch: expected {f_desk_expected}, got {self.f_desk}"
            self._log(msg, False)
            success = False
        else:
            self._log(f"Desk friction correct: {self.f_desk}", True)
        
        # 4. Force equilibrium at threshold (f = f_desk)
        if not math.isclose(self.f, self.f_desk, rel_tol=1e-2, abs_tol=1e-2):
            msg = f"Force equilibrium violated: f={self.f}, f_desk={self.f_desk}"
            self._log(msg, False)
            success = False
        else:
            self._log(f"Force equilibrium satisfied: f = f_desk = {self.f}", True)
        
        # 5. Check derived mu2 formula
        mu2_expected = 2 * self.mu1 * (self.H - self.h) / self.H
        if not math.isclose(self.mu2, mu2_expected, rel_tol=1e-2, abs_tol=1e-2):
            msg = f"μ₂ mismatch: expected {mu2_expected}, got {self.mu2}"
            self._log(msg, False)
            success = False
        else:
            self._log(f"μ₂ correct: {self.mu2}", True)
        
        # 6. Check second-part expression if provided as numeric for specific x
        if isinstance(self.F, (int, float)):
            # Assume x=0 for validation (fully inserted paper)
            F_expected_at_x0 = self.mu1 * self.M * self.g * (self.L - 0) / self.L
            if not math.isclose(self.F, F_expected_at_x0, rel_tol=1e-2, abs_tol=1e-2):
                msg = f"F at x=0 mismatch: expected {F_expected_at_x0}, got {self.F}"
                self._log(msg, False)
                success = False
            else:
                self._log(f"F at x=0 correct: {self.F}", True)
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True
        
        # 1. Verify mu2 from force balance
        mu2_from_balance = self.f / (self.M * self.g) if self.M * self.g != 0 else 0
        if not math.isclose(self.mu2, mu2_from_balance, rel_tol=1e-2, abs_tol=1e-2):
            msg = f"μ₂ inconsistent: from balance {mu2_from_balance}, stored {self.mu2}"
            self._log(msg, False)
            success = False
        else:
            self._log(f"μ₂ consistent from force balance: {self.mu2}", True)
        
        # 2. Verify m from geometry
        m_from_geom = (self.H - self.h) / self.H * self.M
        if not math.isclose(self.m, m_from_geom, rel_tol=1e-2, abs_tol=1e-2):
            msg = f"Mass m inconsistent: from geometry {m_from_geom}, stored {self.m}"
            self._log(msg, False)
            success = False
        else:
            self._log(f"Mass m consistent from geometry: {self.m}", True)
        
        # 3. Check normal force F_N if provided
        if self.F_N != 0:
            # At x=0, F_N should be M*g/2
            F_N_expected_at_x0 = self.M * self.g / 2
            if not math.isclose(self.F_N, F_N_expected_at_x0, rel_tol=1e-2, abs_tol=1e-2):
                msg = f"F_N at x=0 mismatch: expected {F_N_expected_at_x0}, got {self.F_N}"
                self._log(msg, False)
                success = False
            else:
                self._log(f"F_N at x=0 consistent: {self.F_N}", True)
        
        return success
    
    def test_monotonicity(self):
        """Perturb inputs and check output trends."""
        success = True
        
        # Perturb height h upward
        h_perturbed = self.h * 1.05  # +5%
        m_perturbed = (self.H - h_perturbed) / self.H * self.M
        f_perturbed = 2 * self.mu1 * m_perturbed * self.g
        mu2_perturbed = f_perturbed / (self.M * self.g)
        
        # Since (H-h)/H decreases as h increases, mu2 should decrease
        if mu2_perturbed >= self.mu2:
            msg = f"Monotonicity violated: μ₂ should decrease when h increases. Original μ₂={self.mu2}, perturbed μ₂={mu2_perturbed}"
            self._log(msg, False)
            success = False
        else:
            self._log(f"Monotonicity holds: μ₂ decreases from {self.mu2} to {mu2_perturbed} when h increases by 5%", True)
        
        # Perturb mu1 upward
        mu1_perturbed = self.mu1 * 1.05  # +5%
        f_from_mu1_perturbed = 2 * mu1_perturbed * self.m * self.g
        mu2_from_mu1_perturbed = f_from_mu1_perturbed / (self.M * self.g)
        
        if mu2_from_mu1_perturbed <= self.mu2:
            msg = f"Monotonicity violated: μ₂ should increase when μ₁ increases. Original μ₂={self.mu2}, perturbed μ₂={mu2_from_mu1_perturbed}"
            self._log(msg, False)
            success = False
        else:
            self._log(f"Monotonicity holds: μ₂ increases from {self.mu2} to {mu2_from_mu1_perturbed} when μ₁ increases by 5%", True)
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in tests:
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"Test '{name}' crashed: {traceback.format_exc()}", False)
                overall_success = False
        
        return overall_success, self.logs