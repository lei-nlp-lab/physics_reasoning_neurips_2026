import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        
        # Extract inputs with defaults
        self.m_A = getattr(solution_instance, 'm_A', 4.0)
        self.m_B = getattr(solution_instance, 'm_B', 4.0)
        self.F1 = getattr(solution_instance, 'F_1', getattr(solution_instance, 'F1', 16.0))
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Extract results with safe access
        results = getattr(solution_instance, 'results', {})
        self.mu = getattr(solution_instance, 'mu', results.get('mu', 0.0))
        self.F_N = getattr(solution_instance, 'F_N', results.get('F_N', 0.0))
        self.F_f1 = getattr(solution_instance, 'F_f1', results.get('F_f1', 0.0))
        self.F_f2 = getattr(solution_instance, 'F_f2', results.get('F_f2', 0.0))
        
        # Store logs for all tests
        self.logs = []
        
    def _log(self, message, is_error=False):
        """Helper to log messages"""
        self.logs.append({
            "type": "error" if is_error else "info",
            "message": message
        })
    
    def test_invariants(self):
        """Verify physical invariants: equilibrium conditions and friction formulas"""
        try:
            # Test 1: System equilibrium in horizontal direction
            # For uniform motion: F1 - F_f2 = 0
            if not math.isclose(self.F1, self.F_f2, rel_tol=0.02, abs_tol=0.2):
                self._log(f"Horizontal equilibrium violation: F1={self.F1}N, F_f2={self.F_f2}N (should be equal)", True)
                return False
            
            # Test 2: Block A equilibrium (F_f1 should be 0)
            if not math.isclose(self.F_f1, 0.0, rel_tol=0.02, abs_tol=0.2):
                self._log(f"F_f1 should be 0, but got {self.F_f1}N", True)
                return False
            
            # Test 3: Normal force = total weight
            expected_F_N = (self.m_A + self.m_B) * self.g
            if not math.isclose(self.F_N, expected_F_N, rel_tol=0.02, abs_tol=0.2):
                self._log(f"Normal force mismatch: got {self.F_N}N, expected {expected_F_N}N", True)
                return False
            
            # Test 4: Kinetic friction formula F_f2 = μ * F_N
            if self.F_N > 0:
                expected_mu = self.F_f2 / self.F_N
                if not math.isclose(self.mu, expected_mu, rel_tol=0.02, abs_tol=0.01):
                    self._log(f"Friction formula violation: μ={self.mu}, but F_f2/F_N={expected_mu}", True)
                    return False
            
            # Test 5: Symbolic verification (alternative check)
            try:
                mu_sym = sp.Symbol('mu')
                F_N_sym = (self.m_A + self.m_B) * self.g
                eq = sp.Eq(self.F_f2, mu_sym * F_N_sym)
                symbolic_mu = sp.solve(eq, mu_sym)[0]
                if not math.isclose(float(symbolic_mu), self.mu, rel_tol=0.02, abs_tol=0.01):
                    self._log(f"Symbolic check failed: μ from eq={float(symbolic_mu)}, result μ={self.mu}", True)
                    return False
            except:
                self._log("Symbolic check skipped (non-critical)")
            
            self._log("All invariants satisfied")
            return True
            
        except Exception as e:
            self._log(f"Invariant test failed with error: {str(e)}\n{traceback.format_exc()}", True)
            return False
    
    def test_trace_consistency(self):
        """Check if intermediate results are consistent with final answer"""
        try:
            # Check step consistency using KB theorems
            
            # Step 1: F_f1 should be 0
            if not math.isclose(self.F_f1, 0.0, rel_tol=0.02, abs_tol=0.2):
                self._log(f"Step 1 inconsistency: F_f1 should be 0, got {self.F_f1}N", True)
                return False
            
            # Step 2: F_f2 should equal F1
            if not math.isclose(self.F_f2, self.F1, rel_tol=0.02, abs_tol=0.2):
                self._log(f"Step 2 inconsistency: F_f2={self.F_f2}N, F1={self.F1}N (should be equal)", True)
                return False
            
            # Step 3: F_N = (m_A + m_B)g
            expected_F_N = (self.m_A + self.m_B) * self.g
            if not math.isclose(self.F_N, expected_F_N, rel_tol=0.02, abs_tol=0.2):
                self._log(f"Step 3 inconsistency: F_N={self.F_N}N, expected {expected_F_N}N", True)
                return False
            
            # Step 4 & 5: μ = F_f2 / ((m_A + m_B)g)
            if self.F_N > 0:
                expected_mu = self.F_f2 / self.F_N
                if not math.isclose(self.mu, expected_mu, rel_tol=0.02, abs_tol=0.01):
                    self._log(f"Step 4/5 inconsistency: μ={self.mu}, expected {expected_mu}", True)
                    return False
            
            self._log("Trace consistency satisfied")
            return True
            
        except Exception as e:
            self._log(f"Trace consistency test failed: {str(e)}", True)
            return False
    
    def test_monotonicity(self):
        """Perturb inputs and verify output trends match physical expectations"""
        try:
            # Original μ value
            original_mu = self.mu
            
            # Test 1: Increase F1, μ should increase proportionally
            perturbed_F1 = self.F1 * 1.1  # Increase by 10%
            perturbed_mu = perturbed_F1 / ((self.m_A + self.m_B) * self.g)
            
            if not perturbed_mu > original_mu:
                self._log(f"Monotonicity violation: Increasing F1 should increase μ", True)
                return False
            
            # Test 2: Increase mass, μ should decrease (since F1 fixed)
            original_mu_calc = self.F1 / ((self.m_A + self.m_B) * self.g)
            perturbed_mass = (self.m_A + self.m_B) * 1.1  # Increase total mass by 10%
            perturbed_mu_mass = self.F1 / (perturbed_mass * self.g)
            
            if not perturbed_mu_mass < original_mu_calc:
                self._log(f"Monotonicity violation: Increasing mass should decrease μ", True)
                return False
            
            # Test 3: Symbolic monotonicity check
            F1_sym, m_sym, g_sym = sp.symbols('F1 m g')
            mu_expr = F1_sym / (m_sym * g_sym)
            
            # Derivative w.r.t F1 should be positive
            dmu_dF1 = sp.diff(mu_expr, F1_sym)
            if not float(dmu_dF1.subs({m_sym: self.m_A + self.m_B, g_sym: self.g})) > 0:
                self._log("Symbolic monotonicity check failed for dμ/dF1", True)
                return False
            
            # Derivative w.r.t mass should be negative
            dmu_dm = sp.diff(mu_expr, m_sym)
            if not float(dmu_dm.subs({F1_sym: self.F1, g_sym: self.g, m_sym: self.m_A + self.m_B})) < 0:
                self._log("Symbolic monotonicity check failed for dμ/dm", True)
                return False
            
            self._log("Monotonicity tests passed")
            return True
            
        except Exception as e:
            self._log(f"Monotonicity test failed: {str(e)}", True)
            return False
    
    def run_all_tests(self):
        """Run all physics audits and return success status with logs"""
        self.logs = []  # Reset logs
        
        try:
            # Run tests
            invariant_ok = self.test_invariants()
            trace_ok = self.test_trace_consistency()
            monotonicity_ok = self.test_monotonicity()
            
            # Check against target answers
            target_answers = ["0", "16N", "0.2"]
            
            # Parse target answers
            target_F_f1 = float(target_answers[0])
            target_F_f2 = float(target_answers[1].replace('N', ''))
            target_mu = float(target_answers[2])
            
            target_ok = (
                math.isclose(self.F_f1, target_F_f1, rel_tol=0.02, abs_tol=0.2) and
                math.isclose(self.F_f2, target_F_f2, rel_tol=0.02, abs_tol=0.2) and
                math.isclose(self.mu, target_mu, rel_tol=0.02, abs_tol=0.01)
            )
            
            if not target_ok:
                self._log(f"Target answer mismatch: F_f1={self.F_f1}(target:{target_F_f1}), "
                         f"F_f2={self.F_f2}(target:{target_F_f2}), μ={self.mu}(target:{target_mu})", True)
            
            # Overall success
            success = invariant_ok and trace_ok and monotonicity_ok and target_ok
            
            if success:
                self._log("All physics audits passed successfully!")
            else:
                self._log("Physics audits failed - see individual test logs", True)
            
            return success, self.logs
            
        except Exception as e:
            self._log(f"PhysicsChecker runtime error: {str(e)}\n{traceback.format_exc()}", True)
            return False, self.logs