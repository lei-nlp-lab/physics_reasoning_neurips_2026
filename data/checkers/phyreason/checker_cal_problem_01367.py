import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for inputs
        self.R = getattr(solution_instance, 'R', getattr(solution_instance, 'results', {}).get('R', 0.0))
        self.h = getattr(solution_instance, 'h', getattr(solution_instance, 'results', {}).get('h', 0.0))
        self.x = getattr(solution_instance, 'x', getattr(solution_instance, 'results', {}).get('x', 0.0))
        self.v0 = getattr(solution_instance, 'v0', getattr(solution_instance, 'results', {}).get('v0', 0.0))
        self.G = getattr(solution_instance, 'G', getattr(solution_instance, 'results', {}).get('G', 0.0))
        
        # Safe attribute access for results
        self.g = getattr(solution_instance, 'g', getattr(solution_instance, 'results', {}).get('g', 0.0))
        self.sqrt_x2_minus_h2 = getattr(solution_instance, 'sqrt_x2_minus_h2', 
                                        getattr(solution_instance, 'results', {}).get('sqrt(x^2 - h^2)', 0.0))
        self.t = getattr(solution_instance, 't', getattr(solution_instance, 'results', {}).get('t', 0.0))
        self.v = getattr(solution_instance, 'v', getattr(solution_instance, 'results', {}).get('v', 0.0))
        self.F_g = getattr(solution_instance, 'F_g', getattr(solution_instance, 'results', {}).get('F_g', 0.0))
        self.F_c = getattr(solution_instance, 'F_c', getattr(solution_instance, 'results', {}).get('F_c', 0.0))
        
        # Symbolic variables for invariant checking
        self.sym_R, self.sym_h, self.sym_x, self.sym_v0, self.sym_G = sp.symbols('R h x v0 G', positive=True, real=True)
        self.sym_g, self.sym_t, self.sym_v, self.sym_sqrt = sp.symbols('g t v sqrt', positive=True, real=True)
        
        self.logs = []
        
    def _log(self, msg, success):
        self.logs.append((success, msg))
        
    def test_invariants(self):
        """Verify physical invariants from KB theorems."""
        try:
            # 1. Projectile motion consistency (horizontal velocity constant)
            # From step_1: sqrt(x^2 - h^2) = v0 * t
            if self.sqrt_x2_minus_h2 > 0 and self.t > 0:
                lhs = self.sqrt_x2_minus_h2
                rhs = self.v0 * self.t
                if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Projectile horizontal motion invariant violated: sqrt(x²-h²)={lhs:.3f}, v0*t={rhs:.3f}", False)
                else:
                    self._log("Projectile horizontal motion invariant satisfied", True)
            
            # 2. Vertical motion invariant: h = 0.5*g*t² (from step_2)
            if self.h > 0 and self.t > 0:
                lhs = self.h
                rhs = 0.5 * self.g * self.t**2
                if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Projectile vertical motion invariant violated: h={lhs:.3f}, 0.5*g*t²={rhs:.3f}", False)
                else:
                    self._log("Projectile vertical motion invariant satisfied", True)
            
            # 3. Gravitational law and centripetal force consistency (step_4 and step_5)
            # From step_4: F_g = m*g (but m cancels)
            # From step_5: F_c = m*v²/R = F_g when v is first cosmic velocity
            # So we expect: g = v²/R
            if self.v > 0 and self.R > 0:
                lhs = self.g
                rhs = self.v**2 / self.R
                if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Gravity-centripetal invariant violated: g={lhs:.3f}, v²/R={rhs:.3f}", False)
                else:
                    self._log("Gravity-centripetal invariant satisfied", True)
            
            # 4. Newton's 2nd Law consistency: gravitational force provides acceleration g
            # F_g = m*g (where m is arbitrary, so we can't check directly without m)
            # But we can check that g is consistent with the derived formula from step_3
            expected_g = 2 * self.h * self.v0**2 / (self.x**2 - self.h**2) if abs(self.x**2 - self.h**2) > 1e-9 else 0
            if expected_g > 0 and not math.isclose(self.g, expected_g, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Newton's 2nd Law consistency violated: g={self.g:.3f}, expected from projectile={expected_g:.3f}", False)
            else:
                self._log("Newton's 2nd Law consistency satisfied", True)
                
        except Exception as e:
            self._log(f"Invariant test failed with error: {str(e)}", False)
            
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        try:
            # 1. Check step_3: g = 2*h*v0²/(x²-h²)
            expected_g = 2 * self.h * self.v0**2 / (self.x**2 - self.h**2) if abs(self.x**2 - self.h**2) > 1e-9 else 0
            if not math.isclose(self.g, expected_g, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Step 3 (g calculation) inconsistent: g={self.g:.3f}, expected={expected_g:.3f}", False)
            else:
                self._log("Step 3 (g calculation) consistent", True)
                
            # 2. Check step_6: v = v0 * sqrt(2*h*R/(x²-h²))
            denom = self.x**2 - self.h**2
            if denom > 0:
                expected_v = self.v0 * math.sqrt(2 * self.h * self.R / denom)
                if not math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Step 6 (v calculation) inconsistent: v={self.v:.3f}, expected={expected_v:.3f}", False)
                else:
                    self._log("Step 6 (v calculation) consistent", True)
            else:
                self._log("Cannot check v consistency: denominator non-positive", False)
                
            # 3. Check time calculation from step_1: t = sqrt(x²-h²)/v0
            if self.sqrt_x2_minus_h2 > 0 and self.v0 > 0:
                expected_t = self.sqrt_x2_minus_h2 / self.v0
                if not math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Time calculation inconsistent: t={self.t:.3f}, expected={expected_t:.3f}", False)
                else:
                    self._log("Time calculation consistent", True)
                    
        except Exception as e:
            self._log(f"Trace consistency test failed with error: {str(e)}", False)
            
    def test_monotonicity(self):
        """Perturb input h and check output trends match physical intuition."""
        try:
            original_h = self.h
            
            # Perturb h by +1%
            perturbed_h = original_h * 1.01
            
            # Calculate expected v for perturbed h using formula from step_6
            denom = self.x**2 - original_h**2
            denom_perturbed = self.x**2 - perturbed_h**2
            
            if denom > 0 and denom_perturbed > 0:
                original_v = self.v0 * math.sqrt(2 * original_h * self.R / denom)
                perturbed_v = self.v0 * math.sqrt(2 * perturbed_h * self.R / denom_perturbed)
                
                # According to inverse square law and projectile motion:
                # As h increases, gravitational acceleration g increases (from step_3)
                # Therefore first cosmic velocity v should increase
                if perturbed_v > original_v:
                    self._log("Monotonicity test passed: v increases with h (as expected)", True)
                else:
                    self._log(f"Monotonicity test failed: v should increase with h but original={original_v:.3f}, perturbed={perturbed_v:.3f}", False)
                    
                # Also check g monotonicity
                original_g = 2 * original_h * self.v0**2 / denom
                perturbed_g = 2 * perturbed_h * self.v0**2 / denom_perturbed
                if perturbed_g > original_g:
                    self._log("Monotonicity test passed: g increases with h (as expected)", True)
                else:
                    self._log(f"Monotonicity test warning: g should increase with h but original={original_g:.3f}, perturbed={perturbed_g:.3f}", False)
            else:
                self._log("Monotonicity test skipped: invalid denominator", False)
                
        except Exception as e:
            self._log(f"Monotonicity test failed with error: {str(e)}", False)
            
    def run_all_tests(self):
        """Execute all tests and return success boolean and logs."""
        self.logs = []
        
        # Run tests in order
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        # Determine overall success (all tests must pass)
        success = all(log[0] for log in self.logs)
        
        return success, self.logs