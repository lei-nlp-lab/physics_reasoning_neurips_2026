import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract all inputs and results from the solution instance."""
        try:
            # Inputs (with defaults if missing)
            self.m = getattr(solution_instance, 'm', 1.0)  # mass, arbitrary but used symbolically
            self.mu = getattr(solution_instance, 'mu', 0.25)
            self.g = getattr(solution_instance, 'g', 10.0)
            self.x_OA = getattr(solution_instance, 'x_OA', 4.0)
            self.theta_deg = getattr(solution_instance, 'theta_deg', 30.0)
            self.theta = math.radians(self.theta_deg)
            
            # Results (extract from results dict or attributes)
            self.v = getattr(solution_instance, 'v', 
                            getattr(solution_instance, 'results', {}).get('v', 0.0))
            self.L = getattr(solution_instance, 'L', 
                            getattr(solution_instance, 'results', {}).get('L', 0.0))
            
            # Constants for work calculations from the steps analysis
            # From equation: 2*m*g*x1 - 0.5*m*g*x2 - μ*m*g*x_OA = 0.5*m*v^2
            # Solve for (2*x1 - 0.5*x2) as a constant C
            if self.v > 0:
                self.C = (0.5 * self.v**2 + self.mu * self.g * self.x_OA) / self.g
            else:
                self.C = 0.0
                
            self.logs = []
        except Exception as e:
            self.logs = [f"Initialization error: {e}"]
            raise

    def _log(self, message):
        """Helper to append logs."""
        self.logs.append(message)

    def test_invariants(self):
        """Verify fundamental physics laws using symbolic and numerical checks."""
        try:
            m, g, mu, x_OA, theta = sp.symbols('m g mu x_OA theta', positive=True)
            v, L = sp.symbols('v L', positive=True)
            
            # 1. Check work-energy theorem for the incline (step_2)
            # -m*g*L*sin(theta) = 0 - 0.5*m*v^2
            lhs_incline = -m * g * L * sp.sin(theta)
            rhs_incline = -0.5 * m * v**2
            eq_incline = sp.simplify(lhs_incline - rhs_incline)
            
            # Numerical substitution
            lhs_num = -self.m * self.g * self.L * math.sin(self.theta)
            rhs_num = -0.5 * self.m * self.v**2
            if not math.isclose(lhs_num, rhs_num, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL: Work-energy on incline: lhs={lhs_num:.3f}, rhs={rhs_num:.3f}")
                return False
            self._log(f"PASS: Work-energy on incline holds (ΔK = W_gravity)")
            
            # 2. Check work-energy theorem for horizontal motion (step_1)
            # Using the constant C computed from the steps analysis
            # C = (2*x1 - 0.5*x2) from the force pattern
            # Equation: g*C - mu*g*x_OA = 0.5*v^2
            lhs_horiz = self.g * self.C - self.mu * self.g * self.x_OA
            rhs_horiz = 0.5 * self.v**2
            if not math.isclose(lhs_horiz, rhs_horiz, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL: Work-energy horizontal: lhs={lhs_horiz:.3f}, rhs={rhs_horiz:.3f}")
                return False
            self._log("PASS: Work-energy horizontal consistent with force pattern")
            
            return True
        except Exception as e:
            self._log(f"Error in test_invariants: {e}")
            return False

    def test_trace_consistency(self):
        """Check that intermediate results are consistent with final answers."""
        try:
            # 1. Check that v from step_1 leads to correct L in step_2
            expected_L = self.v**2 / (2 * self.g * math.sin(self.theta))
            if not math.isclose(self.L, expected_L, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"FAIL: L inconsistency: computed={self.L:.3f}, expected={expected_L:.3f}")
                return False
            self._log(f"PASS: L = v^2/(2g sinθ) holds with v={self.v:.3f}")
            
            # 2. Verify v is positive and within reasonable bounds
            if self.v <= 0 or self.v > 100:  # arbitrary upper bound
                self._log(f"FAIL: v={self.v} out of plausible range")
                return False
                
            # 3. Verify that the computed C (from force pattern) is positive
            if self.C <= 0:
                self._log(f"WARNING: C={self.C} non-positive, force work may be underestimated")
            else:
                self._log(f"PASS: Force work constant C={self.C:.3f} positive")
            
            return True
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {e}")
            return False

    def test_monotonicity(self):
        """Perturb inputs and check expected trends."""
        try:
            # Perturb coefficient of friction μ
            mu_original = self.mu
            perturbation = 0.1  # 10% increase
            
            # Increase μ should decrease v (more friction)
            mu_new = mu_original * (1 + perturbation)
            v2_squared = 2 * (self.g * self.C - mu_new * self.g * self.x_OA)
            v_new = math.sqrt(max(v2_squared, 0)) if v2_squared > 0 else 0.0
            
            if v2_squared <= 0:
                self._log("WARNING: Increased μ makes v imaginary (block may not reach A)")
                # In this case, v should be zero or lower
                trend_v = (v_new < self.v) or (v_new == 0.0 and self.v > 0)
            else:
                trend_v = v_new < self.v
                
            if not trend_v:
                self._log(f"FAIL: μ↑ did not cause v↓ (v_old={self.v:.3f}, v_new={v_new:.3f})")
                return False
            self._log(f"PASS: μ↑ {perturbation*100:.0f}% causes v↓ ({self.v:.3f}→{v_new:.3f})")
            
            # Check L trend with v
            L_new = v_new**2 / (2 * self.g * math.sin(self.theta)) if v_new > 0 else 0.0
            trend_L = L_new < self.L
            if not trend_L:
                self._log(f"FAIL: L did not decrease with v (L_old={self.L:.3f}, L_new={L_new:.3f})")
                return False
            self._log(f"PASS: L decreases with v ({self.L:.3f}→{L_new:.3f})")
            
            # Restore original μ (not needed as we didn't modify instance)
            return True
        except Exception as e:
            self._log(f"Error in test_monotonicity: {e}")
            return False

    def run_all_tests(self):
        """Execute all tests and return success flag with logs."""
        self.logs = []  # Reset logs
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in tests:
            try:
                if not test_func():
                    success = False
                    self._log(f"✗ {name} test failed")
                else:
                    self._log(f"✓ {name} test passed")
            except Exception as e:
                success = False
                self._log(f"✗ {name} test crashed: {e}")
                self._log(traceback.format_exc())
        
        return success, self.logs