import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction with fallbacks
        self.results = {}
        # Extract results safely
        expected_keys = ['f', 'x', 'a_1', 'x_2', 'mu', 'x_1']
        for key in expected_keys:
            # Try direct attribute, then results dictionary
            val = getattr(solution_instance, key, None)
            if val is None:
                val = getattr(solution_instance, 'results', {}).get(key, 0.0)
            self.results[key] = val
        
        # Constants from problem
        self.m = 1.0  # kg
        self.theta = 37 * math.pi / 180  # radians
        self.sin_theta = 0.6
        self.cos_theta = 0.8
        self.g = 10.0  # m/s² (standard assumption)
        self.t_total = 2.0  # s
        
        # Velocities from graph description (t=0 to t=2s)
        self.v0 = 0.0  # m/s
        self.v1 = 10.0  # m/s at t=1s (from v-t graph analysis)
        self.v2 = 12.0  # m/s at t=2s (from v-t graph analysis)
        
        # Logs for reporting
        self.logs = []
        
    def log(self, message):
        self.logs.append(message)
        
    def test_invariants(self):
        """Test fundamental physics laws"""
        try:
            # Test 1: Newton's second law for segment 1-2s
            # From steps: mg sinθ - f = m*a_1
            a1_expected = self.results.get('a_1', 0.0)
            f_expected = self.results.get('f', 0.0)
            lhs = self.m * self.g * self.sin_theta - f_expected
            rhs = self.m * a1_expected
            
            if not math.isclose(lhs, rhs, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Newton's second law violated: mg sinθ - f = {lhs}, m*a_1 = {rhs}")
                return False
                
            # Test 2: Friction formula f = μ * N
            mu_expected = self.results.get('mu', 0.0)
            N = self.m * self.g * self.cos_theta
            f_from_mu = mu_expected * N
            
            if not math.isclose(f_expected, f_from_mu, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Friction formula violated: f = {f_expected}, μ*N = {f_from_mu}")
                return False
                
            # Test 3: Kinematic consistency for segment 0-1s
            # a0 = (v1 - v0)/1 = 10 m/s²
            a0_expected = 10.0
            x1_expected = self.results.get('x_1', 0.0)
            x1_calc = 0.5 * a0_expected * (1.0)**2  # Using t=1s
            
            if not math.isclose(x1_expected, x1_calc, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Segment 0-1s kinematic violation: x1={x1_expected}, calc={x1_calc}")
                return False
                
            self.log("All invariants passed")
            return True
            
        except Exception as e:
            self.log(f"Invariant test error: {traceback.format_exc()}")
            return False
            
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers"""
        try:
            # Check 1: Total displacement = x1 + x2
            x1 = self.results.get('x_1', 0.0)
            x2 = self.results.get('x_2', 0.0)
            x_total = self.results.get('x', 0.0)
            
            if not math.isclose(x_total, x1 + x2, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Displacement sum violation: x={x_total}, x1+x2={x1+x2}")
                return False
                
            # Check 2: Acceleration a1 from velocities
            a1_calc = (self.v2 - self.v1) / (self.t_total - 1.0)  # 2s - 1s
            a1_expected = self.results.get('a_1', 0.0)
            
            if not math.isclose(a1_expected, a1_calc, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Acceleration a1 mismatch: expected={a1_expected}, calc={a1_calc}")
                return False
                
            # Check 3: x2 using trapezoid rule
            x2_calc = 0.5 * (self.v1 + self.v2) * (self.t_total - 1.0)
            if not math.isclose(x2, x2_calc, rel_tol=0.02, abs_tol=0.02):
                self.log(f"x2 calculation mismatch: expected={x2}, calc={x2_calc}")
                return False
                
            # Check 4: f from Newton's second law
            f_calc = self.m * self.g * self.sin_theta - self.m * a1_expected
            if not math.isclose(self.results.get('f', 0.0), f_calc, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Friction force mismatch: expected={self.results.get('f')}, calc={f_calc}")
                return False
                
            # Check 5: μ from friction formula
            mu_calc = self.results.get('f', 0.0) / (self.m * self.g * self.cos_theta)
            if not math.isclose(self.results.get('mu', 0.0), mu_calc, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Friction coefficient mismatch: expected={self.results.get('mu')}, calc={mu_calc}")
                return False
                
            self.log("All trace consistency checks passed")
            return True
            
        except Exception as e:
            self.log(f"Trace consistency error: {traceback.format_exc()}")
            return False
            
    def test_monotonicity(self):
        """Test behavior when mass is perturbed"""
        try:
            original_mass = self.m
            perturbation = 1.1  # 10% increase
            
            # Save original values
            original_f = self.results.get('f', 0.0)
            original_mu = self.results.get('mu', 0.0)
            
            # Perturb mass (but keep same acceleration from graph - independent of mass)
            # This assumes the v-t graph remains the same, which is a reasonable test assumption
            new_mass = original_mass * perturbation
            a1 = self.results.get('a_1', 0.0)  # Same acceleration
            
            # Calculate new friction force
            new_f = new_mass * self.g * self.sin_theta - new_mass * a1
            
            # Calculate new coefficient (should remain the same)
            new_mu = new_f / (new_mass * self.g * self.cos_theta)
            
            # Check 1: f should scale linearly with mass (since a1 constant)
            expected_f_ratio = perturbation
            actual_f_ratio = new_f / original_f if original_f != 0 else 0
            
            if not math.isclose(actual_f_ratio, expected_f_ratio, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Force scaling violation: f_ratio={actual_f_ratio}, expected={expected_f_ratio}")
                return False
                
            # Check 2: μ should remain constant (surface property)
            if not math.isclose(new_mu, original_mu, rel_tol=0.02, abs_tol=0.02):
                self.log(f"μ non-constant: new_μ={new_mu}, original_μ={original_mu}")
                return False
                
            # Check 3: Displacement should remain same (same kinematics)
            # x1 and x2 depend only on kinematics, not mass
            x1 = self.results.get('x_1', 0.0)
            x2 = self.results.get('x_2', 0.0)
            x_total = x1 + x2
            
            # Using same velocities and times
            x1_new = 0.5 * 10.0 * (1.0)**2  # a0 = 10 m/s² still
            x2_new = 0.5 * (self.v1 + self.v2) * (self.t_total - 1.0)
            x_total_new = x1_new + x2_new
            
            if not math.isclose(x_total, x_total_new, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Displacement changed with mass: should remain {x_total}, got {x_total_new}")
                return False
                
            self.log("Monotonicity test passed: f scales with mass, μ constant, displacement unchanged")
            return True
            
        except Exception as e:
            self.log(f"Monotonicity test error: {traceback.format_exc()}")
            return False
            
    def test_target_answers(self):
        """Verify final answers match expected format"""
        try:
            # Convert results to target answer format
            x_total = self.results.get('x', 0.0)
            f_value = self.results.get('f', 0.0)
            mu_value = self.results.get('mu', 0.0)
            
            # Check 1: Total displacement ≈ 16m
            if not math.isclose(x_total, 16.0, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Displacement mismatch: expected ~16m, got {x_total}m")
                return False
                
            # Check 2: Friction force ≈ 4N
            if not math.isclose(f_value, 4.0, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Friction force mismatch: expected ~4N, got {f_value}N")
                return False
                
            # Check 3: μ ≈ 0.5
            if not math.isclose(mu_value, 0.5, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Friction coefficient mismatch: expected ~0.5, got {mu_value}")
                return False
                
            # Check direction: friction should oppose motion (upward along belt)
            a1 = self.results.get('a_1', 0.0)
            # If a1 < g sinθ, friction is upward (positive in our equation)
            if a1 >= self.g * self.sin_theta:
                self.log("Friction direction suspicious: should be upward along belt")
                return False
                
            self.log("All target answers match expected values")
            return True
            
        except Exception as e:
            self.log(f"Target answer test error: {traceback.format_exc()}")
            return False
            
    def run_all_tests(self):
        """Execute all tests and return success status with logs"""
        self.logs = []  # Reset logs
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
            ("Target Answers", self.test_target_answers)
        ]
        
        success = True
        for test_name, test_func in tests:
            self.log(f"--- Running {test_name} Test ---")
            if not test_func():
                success = False
                self.log(f"{test_name} test FAILED")
            else:
                self.log(f"{test_name} test PASSED")
                
        return success, self.logs