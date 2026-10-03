import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results with fallbacks."""
        # Input parameters with defaults from problem statement
        self.l = getattr(solution_instance, 'l', 15.0)
        self.theta_deg = getattr(solution_instance, 'theta', 30.0)
        self.theta = math.radians(self.theta_deg)
        self.m = getattr(solution_instance, 'm', 2.0)
        self.F = getattr(solution_instance, 'F', 30.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Extract results with safe access
        self.results = getattr(solution_instance, 'results', {})
        
        self.f = self.results.get('f', 0.0)
        self.F_net = self.results.get('F_net', 0.0)
        self.v2 = self.results.get('v2', 0.0)  # v^2
        self.t = self.results.get('t', 0.0)
        self.a = self.results.get('a', 0.0)
        self.mu = self.results.get('mu', 0.0)
        self.N = self.results.get('N', 0.0)
        self.a_prime = self.results.get('a\'', 0.0)
        self.x1 = self.results.get('x_1', 0.0)
        self.v = self.results.get('v', 0.0)
        
        # Precomputed constants
        self.sin_theta = math.sin(self.theta)
        self.cos_theta = math.cos(self.theta)
        
        # For symbolic checks
        self.m_sym, self.g_sym, self.F_sym = sp.symbols('m g F')
        self.theta_sym, self.mu_sym = sp.symbols('theta mu')
        self.a_sym, self.a_prime_sym = sp.symbols('a a_prime')
        
        self.logs = []
    
    def log(self, message, success):
        self.logs.append((message, success))
    
    def test_invariants(self):
        """Verify physical invariants: equilibrium, Newton's 2nd law, friction formula."""
        try:
            # 1. Check normal force: N = mg cosθ
            N_expected = self.m * self.g * self.cos_theta
            if math.isclose(self.N, N_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Normal force check passed: N = {self.N:.3f} ≈ mg cosθ = {N_expected:.3f}", True)
            else:
                self.log(f"Normal force mismatch: N = {self.N:.3f} vs mg cosθ = {N_expected:.3f}", False)
            
            # 2. Check equilibrium during constant speed descent: mg sinθ = f
            mg_sin_theta = self.m * self.g * self.sin_theta
            if math.isclose(self.f, mg_sin_theta, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Equilibrium check passed: f = {self.f:.3f} ≈ mg sinθ = {mg_sin_theta:.3f}", True)
            else:
                self.log(f"Equilibrium mismatch: f = {self.f:.3f} vs mg sinθ = {mg_sin_theta:.3f}", False)
            
            # 3. Check friction formula: f = μN
            mu_N = self.mu * self.N
            if math.isclose(self.f, mu_N, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Friction formula check passed: f = {self.f:.3f} ≈ μN = {mu_N:.3f}", True)
            else:
                self.log(f"Friction formula mismatch: f = {self.f:.3f} vs μN = {mu_N:.3f}", False)
            
            # 4. Check μ = tanθ (from steps analysis)
            mu_from_tan = math.tan(self.theta)
            if math.isclose(self.mu, mu_from_tan, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"μ = tanθ check passed: μ = {self.mu:.3f} ≈ tanθ = {mu_from_tan:.3f}", True)
            else:
                self.log(f"μ mismatch: μ = {self.mu:.3f} vs tanθ = {mu_from_tan:.3f}", False)
            
            # 5. Check Newton's 2nd law for upward motion: F_net = F - mg sinθ - f = ma
            F_net_expected = self.F - self.m * self.g * self.sin_theta - self.f
            if math.isclose(self.F_net, F_net_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Newton's 2nd law (upward) check passed: F_net = {self.F_net:.3f} ≈ {F_net_expected:.3f}", True)
            else:
                self.log(f"Newton's 2nd law mismatch: F_net = {self.F_net:.3f} vs expected {F_net_expected:.3f}", False)
            
            # 6. Check acceleration: a = F_net / m
            a_from_F_net = self.F_net / self.m if self.m != 0 else 0.0
            if math.isclose(self.a, a_from_F_net, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Acceleration check passed: a = {self.a:.3f} ≈ F_net/m = {a_from_F_net:.3f}", True)
            else:
                self.log(f"Acceleration mismatch: a = {self.a:.3f} vs F_net/m = {a_from_F_net:.3f}", False)
            
            # 7. Check deceleration after force removal: a' = g sinθ + μ g cosθ
            a_prime_expected = self.g * self.sin_theta + self.mu * self.g * self.cos_theta
            if math.isclose(self.a_prime, a_prime_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Deceleration check passed: a' = {self.a_prime:.3f} ≈ g sinθ + μ g cosθ = {a_prime_expected:.3f}", True)
            else:
                self.log(f"Deceleration mismatch: a' = {self.a_prime:.3f} vs expected {a_prime_expected:.3f}", False)
            
            return all(success for _, success in self.logs[-7:])
        except Exception as e:
            self.log(f"Error in test_invariants: {e}\n{traceback.format_exc()}", False)
            return False
    
    def test_trace_consistency(self):
        """Verify consistency between intermediate results and final answer."""
        try:
            # 1. Check kinematics: x1 = 0.5 * a * t^2
            x1_from_kinematics = 0.5 * self.a * self.t**2
            if math.isclose(self.x1, x1_from_kinematics, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Kinematics x1 check passed: {self.x1:.3f} ≈ 0.5*a*t² = {x1_from_kinematics:.3f}", True)
            else:
                self.log(f"Kinematics x1 mismatch: {self.x1:.3f} vs 0.5*a*t² = {x1_from_kinematics:.3f}", False)
            
            # 2. Check velocity at time t: v = a * t
            v_from_kinematics = self.a * self.t
            if math.isclose(self.v, v_from_kinematics, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Velocity check passed: v = {self.v:.3f} ≈ a*t = {v_from_kinematics:.3f}", True)
            else:
                self.log(f"Velocity mismatch: v = {self.v:.3f} vs a*t = {v_from_kinematics:.3f}", False)
            
            # 3. Check v^2 = 2*a'*x2, where x2 = l - x1
            x2 = self.l - self.x1
            v2_from_deceleration = 2 * self.a_prime * x2
            if math.isclose(self.v2, v2_from_deceleration, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"v² during deceleration check passed: {self.v2:.3f} ≈ 2*a'*x2 = {v2_from_deceleration:.3f}", True)
            else:
                self.log(f"v² mismatch: {self.v2:.3f} vs 2*a'*x2 = {v2_from_deceleration:.3f}", False)
            
            # 4. Check total distance constraint: x1 + x2 = l
            total_distance = self.x1 + x2
            if math.isclose(total_distance, self.l, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Total distance check passed: x1+x2 = {total_distance:.3f} ≈ l = {self.l:.3f}", True)
            else:
                self.log(f"Total distance mismatch: x1+x2 = {total_distance:.3f} vs l = {self.l:.3f}", False)
            
            # 5. Verify the derived time t satisfies both motion segments
            # From kinematics: t = sqrt(2*x1/a)
            t_from_x1_a = math.sqrt(2 * self.x1 / self.a) if self.a > 0 else 0.0
            # From deceleration: t = v/a, and v from v^2 = 2*a'*x2
            v_from_deceleration = math.sqrt(2 * self.a_prime * x2)
            t_from_v_a = v_from_deceleration / self.a if self.a > 0 else 0.0
            # Both should match the reported t
            t_checks = [
                math.isclose(self.t, t_from_x1_a, rel_tol=1e-2, abs_tol=1e-2),
                math.isclose(v_from_deceleration, self.v, rel_tol=1e-2, abs_tol=1e-2),
            ]
            if all(t_checks):
                self.log(f"Time consistency check passed: t = {self.t:.3f}, from kinematics: {t_from_x1_a:.3f}, v consistency: {v_from_deceleration:.3f} ≈ {self.v:.3f}", True)
            else:
                self.log(f"Time inconsistency: t = {self.t:.3f}, from kinematics: {t_from_x1_a:.3f}, v from deceleration: {v_from_deceleration:.3f}", False)
            
            return all(success for _, success in self.logs[-5:])
        except Exception as e:
            self.log(f"Error in test_trace_consistency: {e}\n{traceback.format_exc()}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb mass m and check expected trends in acceleration and time."""
        try:
            # Store original values
            m_original = self.m
            a_original = self.a
            t_original = self.t
            
            # Perturb mass by +10%
            self.m = m_original * 1.10
            
            # Recompute derived quantities using formulas
            # 1. Recompute normal force
            N_new = self.m * self.g * self.cos_theta
            # 2. Recompute friction force (using same μ)
            f_new = self.mu * N_new
            # 3. Recompute net force and acceleration
            F_net_new = self.F - self.m * self.g * self.sin_theta - f_new
            a_new = F_net_new / self.m if self.m != 0 else 0.0
            # 4. Recompute deceleration a' (independent of m)
            a_prime_new = self.g * self.sin_theta + self.mu * self.g * self.cos_theta
            # 5. Recompute time t from total distance constraint
            # Derivation: x1 + x2 = l, with x1 = 0.5*a*t², x2 = v²/(2*a') = (a²t²)/(2*a')
            # So: 0.5*a*t² + (a²t²)/(2*a') = l
            # t² * [0.5*a + a²/(2*a')] = l
            # t = sqrt(2l / [a*(1 + a/a')])
            if a_new > 0 and a_prime_new > 0:
                t_new = math.sqrt(2 * self.l / (a_new * (1 + a_new / a_prime_new)))
            else:
                t_new = 0.0
            
            # Expected trends:
            # Since a = (F/m) - g(sinθ + μ cosθ), increasing m should decrease a
            trend_a = a_new < a_original
            # With larger m and smaller a, time t should increase for same distance
            trend_t = t_new > t_original
            
            if trend_a and trend_t:
                self.log(f"Monotonicity check passed: increasing m reduces a ({a_original:.3f} → {a_new:.3f}) and increases t ({t_original:.3f} → {t_new:.3f})", True)
            else:
                self.log(f"Monotonicity mismatch: a trend {a_original:.3f} → {a_new:.3f} (should decrease), t trend {t_original:.3f} → {t_new:.3f} (should increase)", False)
            
            # Restore original values
            self.m = m_original
            
            return trend_a and trend_t
        except Exception as e:
            self.log(f"Error in test_monotonicity: {e}\n{traceback.format_exc()}", False)
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        self.logs = []  # Reset logs
        success = True
        
        # Run tests
        tests = [
            ("Physical Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for test_name, test_func in tests:
            self.log(f"\n=== Running {test_name} ===", True)
            if not test_func():
                success = False
        
        return success, self.logs