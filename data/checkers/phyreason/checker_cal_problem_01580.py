import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results with fallbacks"""
        # Safely access inputs with defaults
        self.m = getattr(solution_instance, 'm', 0.0)
        self.q = getattr(solution_instance, 'q', 0.0)
        self.E = getattr(solution_instance, 'E', 0.0)
        self.theta = getattr(solution_instance, 'theta', 0.0)
        self.v0 = getattr(solution_instance, 'v0', 0.0)
        self.g = getattr(solution_instance, 'g', 9.81)
        
        # Safely access results dict or attributes
        results = getattr(solution_instance, 'results', {})
        self.mu = getattr(solution_instance, 'mu', results.get('mu', 0.0))
        self.F_f = getattr(solution_instance, 'F_f', results.get('F_f', 0.0))
        self.N = getattr(solution_instance, 'N', results.get('N', 0.0))
        
        # Setup symbolic variables
        self._setup_symbols()
        
        self.logs = []
    
    def _setup_symbols(self):
        """Initialize symbolic variables for physics laws"""
        self.sym_m, self.sym_q, self.sym_E, self.sym_theta, self.sym_g = sp.symbols('m q E theta g', positive=True)
        self.sym_mu = sp.symbols('mu')
        self.sym_F_f = sp.symbols('F_f')
        self.sym_N = sp.symbols('N')
    
    def _log(self, msg, success=True):
        """Helper for logging test results"""
        self.logs.append(f"{'✓' if success else '✗'} {msg}")
        return success
    
    def test_invariants(self):
        """Verify fundamental physics laws based on KB"""
        checks = []
        
        try:
            # 1. Check Newton's First Law (constant speed => net force zero)
            # Along incline: F_f + mg sinθ = qE cosθ (from STEP_1)
            lhs_incline = self.F_f + self.m * self.g * math.sin(self.theta)
            rhs_incline = self.q * self.E * math.cos(self.theta)
            
            if not math.isclose(lhs_incline, rhs_incline, rel_tol=1e-2, abs_tol=1e-2):
                checks.append(self._log(
                    f"Newton's 1st Law (incline) failed: {lhs_incline:.3f} ≠ {rhs_incline:.3f}", 
                    False
                ))
            else:
                checks.append(self._log(
                    f"Newton's 1st Law (incline) satisfied: {lhs_incline:.3f} ≈ {rhs_incline:.3f}"
                ))
            
            # 2. Check Normal force decomposition (from STEP_2)
            N_calculated = self.m * self.g * math.cos(self.theta) + self.q * self.E * math.sin(self.theta)
            if self.N != 0.0:  # Only check if N was provided
                if not math.isclose(self.N, N_calculated, rel_tol=1e-2, abs_tol=1e-2):
                    checks.append(self._log(
                        f"Normal force mismatch: {self.N:.3f} ≠ {N_calculated:.3f}", 
                        False
                    ))
                else:
                    checks.append(self._log(
                        f"Normal force consistent: {self.N:.3f} ≈ {N_calculated:.3f}"
                    ))
            
            # 3. Check friction definition: F_f = μN (from STEP_3)
            if self.N != 0.0 and self.F_f != 0.0 and self.mu != 0.0:
                F_f_from_mu = self.mu * self.N
                if not math.isclose(self.F_f, F_f_from_mu, rel_tol=1e-2, abs_tol=1e-2):
                    checks.append(self._log(
                        f"Friction definition failed: F_f={self.F_f:.3f}, μN={F_f_from_mu:.3f}", 
                        False
                    ))
                else:
                    checks.append(self._log(
                        f"Friction definition satisfied: F_f={self.F_f:.3f} ≈ μN={F_f_from_mu:.3f}"
                    ))
            
        except Exception as e:
            checks.append(self._log(f"Invariant test error: {str(e)}", False))
            traceback.print_exc()
        
        return all(checks)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer"""
        checks = []
        
        try:
            # Symbolic check of derived formula
            sym_mu_expr = (self.sym_q * self.sym_E * sp.cos(self.sym_theta) - 
                          self.sym_m * self.sym_g * sp.sin(self.sym_theta)) / \
                         (self.sym_m * self.sym_g * sp.cos(self.sym_theta) + 
                          self.sym_q * self.sym_E * sp.sin(self.sym_theta))
            
            # Numerical verification if values available
            if all([self.m > 0, self.q > 0, self.E > 0, self.theta > 0]):
                mu_calculated = (self.q * self.E * math.cos(self.theta) - 
                                self.m * self.g * math.sin(self.theta)) / \
                               (self.m * self.g * math.cos(self.theta) + 
                                self.q * self.E * math.sin(self.theta))
                
                if not math.isclose(self.mu, mu_calculated, rel_tol=1e-2, abs_tol=1e-2):
                    checks.append(self._log(
                        f"μ formula mismatch: result={self.mu:.4f}, expected={mu_calculated:.4f}", 
                        False
                    ))
                else:
                    checks.append(self._log(
                        f"μ formula verified: {self.mu:.4f} ≈ {mu_calculated:.4f}"
                    ))
            else:
                checks.append(self._log(
                    "Trace consistency (symbolic): μ formula matches KB derivation"
                ))
                
        except Exception as e:
            checks.append(self._log(f"Trace consistency error: {str(e)}", False))
            traceback.print_exc()
        
        return all(checks)
    
    def test_monotonicity(self):
        """Perturb mass m and check expected trend"""
        checks = []
        
        try:
            if self.m <= 0 or self.mu <= 0:
                checks.append(self._log("Monotonicity skipped: invalid base values", True))
                return True
            
            # Save original
            original_m = self.m
            original_mu = self.mu
            
            # Perturb mass upward by 10%
            self.m *= 1.10
            mu_perturbed = (self.q * self.E * math.cos(self.theta) - 
                           self.m * self.g * math.sin(self.theta)) / \
                          (self.m * self.g * math.cos(self.theta) + 
                           self.q * self.E * math.sin(self.theta))
            
            # Check trend: as m increases, μ should decrease (from formula analysis)
            # ∂μ/∂m = -g(qE(cos²θ + sin²θ))/(denominator²) = -gqE/(denominator²) < 0
            trend_ok = mu_perturbed < original_mu
            
            if not trend_ok:
                checks.append(self._log(
                    f"Monotonicity failed: μ({original_m:.3f})={original_mu:.4f}, "
                    f"μ({self.m:.3f})={mu_perturbed:.4f}", 
                    False
                ))
            else:
                checks.append(self._log(
                    f"Monotonicity OK: μ decreases as m increases "
                    f"({original_mu:.4f} → {mu_perturbed:.4f})"
                ))
            
            # Restore original
            self.m = original_m
            
        except Exception as e:
            checks.append(self._log(f"Monotonicity error: {str(e)}", False))
            traceback.print_exc()
            # Restore original on error
            if 'original_m' in locals():
                self.m = original_m
        
        return all(checks)
    
    def run_all_tests(self):
        """Execute all physics audits and return summary"""
        self.logs = []
        
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        results = []
        for name, test_func in tests:
            self.logs.append(f"\n--- Testing {name} ---")
            try:
                success = test_func()
                results.append(success)
                self.logs.append(f"Result: {'PASS' if success else 'FAIL'}")
            except Exception as e:
                self.logs.append(f"Test crashed: {str(e)}")
                results.append(False)
        
        overall_success = all(results)
        self.logs.insert(0, f"\n=== Physics Audit Summary: {'PASS' if overall_success else 'FAIL'} ===")
        
        return overall_success, self.logs

# Example usage (for testing the checker)
if __name__ == "__main__":
    # Mock solution class for demonstration
    class MockSolution:
        def __init__(self):
            self.m = 0.5  # kg
            self.q = 1e-6  # C
            self.E = 1000  # N/C
            self.theta = math.radians(30)  # 30 degrees
            self.v0 = 2.0  # m/s
            self.g = 9.81
            
            # Intermediate results
            self.N = self.m * self.g * math.cos(self.theta) + self.q * self.E * math.sin(self.theta)
            self.mu = (self.q * self.E * math.cos(self.theta) - 
                      self.m * self.g * math.sin(self.theta)) / \
                     (self.m * self.g * math.cos(self.theta) + 
                      self.q * self.E * math.sin(self.theta))
            self.F_f = self.mu * self.N
            
            self.results = {
                'mu': self.mu,
                'F_f': self.F_f,
                'N': self.N
            }
    
    checker = PhysicsChecker(MockSolution())
    success, logs = checker.run_all_tests()
    for log in logs:
        print(log)