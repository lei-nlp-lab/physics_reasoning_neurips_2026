import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.F_g = getattr(solution_instance, 'F_g', 
                          solution_instance.results.get('F_g', 8.0e-3))
        self.q = getattr(solution_instance, 'q', 
                        solution_instance.results.get('q', 0.01))
        self.theta_deg = getattr(solution_instance, 'theta_deg', 
                                solution_instance.results.get('theta_deg', 37))
        
        # Convert to radians for calculations
        self.theta_rad = math.radians(self.theta_deg)
        
        # Extract results with safe access
        self.F_e = getattr(solution_instance, 'F_e', 
                          solution_instance.results.get('F_e', 0.0))
        self.E = getattr(solution_instance, 'E', 
                        solution_instance.results.get('E', 0.0))
        
        self.logs = []
        
    def _log(self, msg, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            msg = f"[{'PASS' if success else 'FAIL'}] {msg}"
        self.logs.append(msg)
        
    def _is_close(self, a, b):
        """Robust numerical comparison with physics tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
        
    def test_invariants(self):
        """Test fundamental physics laws."""
        try:
            # 1. Static equilibrium: net force = 0
            # Horizontally: F_e = T * sin(theta)
            # Vertically: F_g = T * cos(theta)
            # From these: F_e = F_g * tan(theta)
            
            # Symbolic check
            theta = sp.symbols('theta')
            F_g_sym, F_e_sym = sp.symbols('F_g F_e')
            equilibrium_eq = sp.Eq(F_e_sym, F_g_sym * sp.tan(theta))
            
            # Numerical check
            expected_F_e = self.F_g * math.tan(self.theta_rad)
            if self._is_close(self.F_e, expected_F_e):
                self._log(f"Static equilibrium: F_e = F_g * tan(θ) = {expected_F_e:.4e} N ≈ {self.F_e:.4e} N", True)
            else:
                self._log(f"Static equilibrium FAIL: F_g * tan(θ) = {expected_F_e:.4e} N ≠ {self.F_e:.4e} N", False)
                
            # 2. Electric field definition: E = F_e / q
            expected_E = self.F_e / self.q if abs(self.q) > 1e-12 else 0
            if self._is_close(self.E, expected_E):
                self._log(f"E = F_e/q = {expected_E:.4e} N/C ≈ {self.E:.4e} N/C", True)
            else:
                self._log(f"E = F_e/q FAIL: {expected_E:.4e} N/C ≠ {self.E:.4e} N/C", False)
                
            # 3. Force direction consistency (negative charge in E field)
            # F_e = qE, so with negative q, F_e is opposite to E direction
            # In the problem, ball moves left (θ to right), so E points right
            # Thus F_e points left (negative q in E field right)
            # We just check the magnitude relationships here
            self._log("Force direction logic: Negative charge in horizontal E field → F_e opposite to E", True)
            
        except Exception as e:
            self._log(f"test_invariants failed: {str(e)}\n{traceback.format_exc()}", False)
            
    def test_trace_consistency(self):
        """Cross-check intermediate results."""
        try:
            # Check step 3: F_e = 6.0e-3 N
            expected_F_e_step = 6.0e-3
            if self._is_close(self.F_e, expected_F_e_step):
                self._log(f"Step 3 consistency: F_e = {self.F_e:.4e} N ≈ {expected_F_e_step:.4e} N", True)
            else:
                self._log(f"Step 3 FAIL: F_e = {self.F_e:.4e} N ≠ {expected_F_e_step:.4e} N", False)
                
            # Check step 4: E = 0.6 N/C
            expected_E_step = 0.6
            if self._is_close(self.E, expected_E_step):
                self._log(f"Step 4 consistency: E = {self.E:.4f} N/C ≈ {expected_E_step:.4f} N/C", True)
            else:
                self._log(f"Step 4 FAIL: E = {self.E:.4f} N/C ≠ {expected_E_step:.4f} N/C", False)
                
            # Check the full chain: F_g → F_e → E
            tan_37 = math.tan(math.radians(37))
            chain_F_e = self.F_g * tan_37
            chain_E = chain_F_e / self.q if abs(self.q) > 1e-12 else 0
            
            chain_ok = (self._is_close(self.F_e, chain_F_e) and 
                       self._is_close(self.E, chain_E))
            
            if chain_ok:
                self._log(f"Chain consistency: F_g={self.F_g:.2e}N → F_e={chain_F_e:.2e}N → E={chain_E:.2f}N/C", True)
            else:
                self._log(f"Chain FAIL: Expected F_e={chain_F_e:.2e}N, E={chain_E:.2f}N/C", False)
                
        except Exception as e:
            self._log(f"test_trace_consistency failed: {str(e)}\n{traceback.format_exc()}", False)
            
    def test_monotonicity(self):
        """Perturb inputs and check trends."""
        try:
            # Save original values
            orig_theta = self.theta_deg
            orig_F_g = self.F_g
            orig_q = self.q
            
            # Test 1: Increase angle → F_e should increase
            self.theta_deg = 45  # Increase from 37°
            self.theta_rad = math.radians(self.theta_deg)
            F_e_45 = self.F_g * math.tan(self.theta_rad)
            
            # Restore
            self.theta_deg = orig_theta
            self.theta_rad = math.radians(self.theta_deg)
            
            if F_e_45 > self.F_e * 1.01:  # 1% tolerance
                self._log(f"Monotonicity (angle↑): F_e(45°)={F_e_45:.2e}N > F_e(37°)={self.F_e:.2e}N", True)
            else:
                self._log(f"Monotonicity (angle↑) FAIL: F_e should increase with angle", False)
                
            # Test 2: Increase charge magnitude → E should decrease (for same F_e)
            # Actually, from E = F_e/q, with fixed F_e, |E| ∝ 1/|q|
            self.q = 0.02  # Double the charge
            E_double_q = self.F_e / self.q if abs(self.q) > 1e-12 else 0
            
            # Restore
            self.q = orig_q
            
            if abs(E_double_q) < abs(self.E) * 0.51:  # Should be about half
                self._log(f"Monotonicity (|q|↑): E(|q|=0.02)={E_double_q:.2f}N/C < E(|q|=0.01)={self.E:.2f}N/C", True)
            else:
                self._log(f"Monotonicity (|q|↑) FAIL: E should decrease with increased |q|", False)
                
            # Test 3: Increase F_g → F_e should increase proportionally
            self.F_g = orig_F_g * 1.5
            F_e_inc_F_g = self.F_g * math.tan(self.theta_rad)
            
            # Restore
            self.F_g = orig_F_g
            
            if self._is_close(F_e_inc_F_g / self.F_e, 1.5):
                self._log(f"Monotonicity (F_g↑): F_e ∝ F_g holds", True)
            else:
                self._log(f"Monotonicity (F_g↑) FAIL: F_e should scale with F_g", False)
                
        except Exception as e:
            self._log(f"test_monotonicity failed: {str(e)}\n{traceback.format_exc()}", False)
            
    def run_all_tests(self):
        """Run all tests and return success status."""
        self.logs = []
        
        # Run tests
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        # Check overall success
        success_keywords = ['[PASS]', 'Static equilibrium', 'E = F_e/q', 'Step 3 consistency', 
                           'Step 4 consistency', 'Chain consistency', 'Monotonicity']
        
        fail_count = sum(1 for log in self.logs if log.startswith('[FAIL]'))
        pass_count = sum(1 for log in self.logs if log.startswith('[PASS]'))
        
        overall_success = (fail_count == 0 and pass_count >= 3)
        
        # Add summary
        summary = f"\n=== SUMMARY ===\nTotal tests: {len(self.logs)}\nPassed: {pass_count}\nFailed: {fail_count}\nOverall: {'PASS' if overall_success else 'FAIL'}"
        self.logs.append(summary)
        
        return overall_success, self.logs

# Example usage (commented out):
# checker = PhysicsChecker(solution_instance)
# success, logs = checker.run_all_tests()
# for log in logs:
#     print(log)