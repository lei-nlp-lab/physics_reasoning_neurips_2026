import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs from solution_instance
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'inputs', {}).get('m', 2.0))
        self.L = getattr(solution_instance, 'L', getattr(solution_instance, 'inputs', {}).get('L', 16.0))
        self.F = getattr(solution_instance, 'F', getattr(solution_instance, 'inputs', {}).get('F', 14.0))
        self.mu = getattr(solution_instance, 'mu', getattr(solution_instance, 'inputs', {}).get('mu', 0.5))
        self.theta = getattr(solution_instance, 'theta', getattr(solution_instance, 'inputs', {}).get('theta', 37.0))
        self.g = getattr(solution_instance, 'g', getattr(solution_instance, 'inputs', {}).get('g', 10.0))
        
        # Use given sin37=0.6 and cos37=0.8
        self.sin_theta = 0.6
        self.cos_theta = 0.8
        
        # Safely extract results from solution_instance
        results = getattr(solution_instance, 'results', {})
        self.a = results.get('a', 0.0)
        self.v_B = results.get('v_B', 0.0)
        self.a_1 = results.get('a_1', 0.0)
        self.x = results.get('x', 0.0)
        
        self.logs = []
    
    def _log(self, message, success):
        self.logs.append({"message": message, "success": success})
    
    def test_invariants(self):
        """Check Newton's Second Law for both segments and energy/work consistency."""
        try:
            # 1. Horizontal segment: F - μmg = m*a
            a_expected = (self.F - self.mu * self.m * self.g) / self.m
            if not math.isclose(self.a, a_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Horizontal acceleration mismatch: expected {a_expected:.3f}, got {self.a:.3f}", False)
                return False
            else:
                self._log(f"Horizontal acceleration correct: {self.a:.3f}", True)
            
            # 2. Inclined segment: mg sinθ + μ mg cosθ = m * a_1
            a1_expected = self.g * self.sin_theta + self.mu * self.g * self.cos_theta
            if not math.isclose(self.a_1, a1_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Inclined acceleration mismatch: expected {a1_expected:.3f}, got {self.a_1:.3f}", False)
                return False
            else:
                self._log(f"Inclined acceleration correct: {self.a_1:.3f}", True)
            
            # 3. Work-energy check: total work should equal change in kinetic energy (0)
            # Work from M to B: (F - μmg) * L
            W_MB = (self.F - self.mu * self.m * self.g) * self.L
            # Work from B to highest point: -(mg sinθ + μmg cosθ) * x
            W_Bmax = -(self.m * self.g * self.sin_theta + self.mu * self.m * self.g * self.cos_theta) * self.x
            W_total = W_MB + W_Bmax
            
            # Change in kinetic energy from start (M) to highest point: 0 - 0 = 0
            delta_KE = 0.0
            
            if not math.isclose(W_total, delta_KE, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Work-energy check failed: total work {W_total:.3f} != ΔKE {delta_KE:.3f}", False)
                return False
            else:
                self._log(f"Work-energy check passed: total work = ΔKE = {W_total:.3f}", True)
            
            return True
        except Exception as e:
            self._log(f"Exception in test_invariants: {e}", False)
            return False
    
    def test_trace_consistency(self):
        """Check that intermediate results are consistent with each other."""
        try:
            # 1. Horizontal kinematics: v_B^2 = 2 * a * L
            v_B_sq_expected = 2 * self.a * self.L
            if not math.isclose(self.v_B**2, v_B_sq_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Velocity at B mismatch: expected v_B^2={v_B_sq_expected:.3f}, got {self.v_B**2:.3f}", False)
                return False
            else:
                self._log(f"Velocity at B consistent with horizontal acceleration: v_B^2={self.v_B**2:.3f}", True)
            
            # 2. Inclined kinematics: 0 - v_B^2 = 2 * (-a_1) * x  => x = v_B^2 / (2 * a_1)
            if self.a_1 == 0:
                self._log("Inclined acceleration is zero, cannot compute x", False)
                return False
            x_expected = self.v_B**2 / (2 * self.a_1)
            if not math.isclose(self.x, x_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Distance x mismatch: expected {x_expected:.3f}, got {self.x:.3f}", False)
                return False
            else:
                self._log(f"Distance x consistent with v_B and a_1: x={self.x:.3f}", True)
            
            return True
        except Exception as e:
            self._log(f"Exception in test_trace_consistency: {e}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb mass m and check the trend of the final distance x."""
        try:
            # Helper function to compute x from given parameters
            def compute_x(m, L, F, mu, g, sin_th, cos_th):
                # Horizontal acceleration
                a = (F - mu * m * g) / m
                # Velocity at B
                v_B = math.sqrt(2 * a * L)
                # Inclined acceleration
                a_1 = g * sin_th + mu * g * cos_th
                # Distance x
                x = v_B**2 / (2 * a_1)
                return x
            
            # Original x
            x_orig = compute_x(self.m, self.L, self.F, self.mu, self.g, self.sin_theta, self.cos_theta)
            
            # Perturb m by +1%
            m_perturbed = self.m * 1.01
            x_perturbed = compute_x(m_perturbed, self.L, self.F, self.mu, self.g, self.sin_theta, self.cos_theta)
            
            # Check trend: increasing m should decrease x (since a decreases, v_B decreases, a_1 constant)
            if x_perturbed < x_orig:
                self._log(f"Monotonicity passed: increasing m by 1% decreases x from {x_orig:.3f} to {x_perturbed:.3f}", True)
                return True
            else:
                self._log(f"Monotonicity failed: increasing m by 1% did not decrease x (original {x_orig:.3f}, perturbed {x_perturbed:.3f})", False)
                return False
        except Exception as e:
            self._log(f"Exception in test_monotonicity: {e}", False)
            return False
    
    def run_all_tests(self):
        """Run all tests and return success and logs."""
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                if not test_func():
                    success = False
                    self._log(f"Test '{test_name}' failed.", False)
                else:
                    self._log(f"Test '{test_name}' passed.", True)
            except Exception as e:
                self._log(f"Test '{test_name}' raised exception: {traceback.format_exc()}", False)
                success = False
        
        return success, self.logs