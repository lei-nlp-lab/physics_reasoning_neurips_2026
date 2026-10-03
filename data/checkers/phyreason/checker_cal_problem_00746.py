import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction from solution instance (robust to missing attributes)
        self.mass = getattr(solution_instance, 'mass', 0.5)
        self.theta = getattr(solution_instance, 'theta', 30.0)  # in degrees
        self.v0 = getattr(solution_instance, 'v0', 10.0)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Extract results safely using the provided key mapping
        self.results = {}
        result_keys = ['v_2', 'Deltap', 'p_1', 'a', 'p_0', 'p_2', 'v_1']
        for key in result_keys:
            self.results[key] = getattr(solution_instance, key, None)
        
        # Fallback: if results are in a dictionary attribute
        if hasattr(solution_instance, 'results'):
            for key in result_keys:
                if key in solution_instance.results:
                    self.results[key] = solution_instance.results[key]
        
        # Convert theta to radians for calculations
        self.theta_rad = math.radians(self.theta)
        
        # Expected values from steps analysis
        self.expected = {
            'a': -5.0,
            'v_1': 5.0,
            'p_1': 2.5,
            'v_2': -5.0,
            'p_2': -2.5,
            'p_0': 5.0,
            'Deltap': -7.5
        }
        
        self.logs = []
        
    def log(self, message, level="INFO"):
        self.logs.append(f"[{level}] {message}")
    
    def test_invariants(self):
        """Test physical invariants using KB formulas."""
        try:
            # 1. Check acceleration from Newton's Second Law: a = -g * sin(theta)
            expected_a = -self.g * math.sin(self.theta_rad)
            if not math.isclose(self.results['a'], expected_a, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Acceleration invariant failed: got {self.results['a']}, expected {expected_a}", "WARNING")
                return False
            
            # 2. Check momentum definitions: p = m*v at all times
            # p0 = m * v0
            p0_calc = self.mass * self.v0
            if not math.isclose(self.results['p_0'], p0_calc, rel_tol=0.02, abs_tol=0.02):
                self.log(f"p0 definition failed: got {self.results['p_0']}, calculated {p0_calc}", "WARNING")
                return False
            
            # p1 = m * v1
            p1_calc = self.mass * self.results['v_1']
            if not math.isclose(self.results['p_1'], p1_calc, rel_tol=0.02, abs_tol=0.02):
                self.log(f"p1 definition failed: got {self.results['p_1']}, calculated {p1_calc}", "WARNING")
                return False
            
            # p2 = m * v2
            p2_calc = self.mass * self.results['v_2']
            if not math.isclose(self.results['p_2'], p2_calc, rel_tol=0.02, abs_tol=0.02):
                self.log(f"p2 definition failed: got {self.results['p_2']}, calculated {p2_calc}", "WARNING")
                return False
            
            # 3. Check change in momentum: Δp = p2 - p0
            delta_p_calc = self.results['p_2'] - self.results['p_0']
            if not math.isclose(self.results['Deltap'], delta_p_calc, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Δp definition failed: got {self.results['Deltap']}, calculated {delta_p_calc}", "WARNING")
                return False
            
            # 4. Check kinematics consistency: v = v0 + a*t
            # At t=1s
            v1_calc = self.v0 + self.results['a'] * 1.0
            if not math.isclose(self.results['v_1'], v1_calc, rel_tol=0.02, abs_tol=0.02):
                self.log(f"v1 kinematics failed: got {self.results['v_1']}, calculated {v1_calc}", "WARNING")
                return False
            
            # At t=3s
            v2_calc = self.v0 + self.results['a'] * 3.0
            if not math.isclose(self.results['v_2'], v2_calc, rel_tol=0.02, abs_tol=0.02):
                self.log(f"v2 kinematics failed: got {self.results['v_2']}, calculated {v2_calc}", "WARNING")
                return False
            
            # 5. Check impulse-momentum theorem: Δp = F_avg * Δt
            # F_avg = m * a (constant acceleration)
            F_avg = self.mass * self.results['a']
            delta_t = 3.0
            impulse_calc = F_avg * delta_t
            if not math.isclose(self.results['Deltap'], impulse_calc, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Impulse-momentum failed: Δp={self.results['Deltap']}, F_avg*Δt={impulse_calc}", "WARNING")
                return False
            
            self.log("All physical invariants satisfied.", "INFO")
            return True
            
        except Exception as e:
            self.log(f"Invariant test failed with error: {e}\n{traceback.format_exc()}", "ERROR")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Target answers from problem statement
            target_p1 = 2.5  # kg·m/s
            target_delta_p = -7.5  # kg·m/s
            
            # Check against results
            if not math.isclose(self.results['p_1'], target_p1, rel_tol=0.02, abs_tol=0.02):
                self.log(f"p1 mismatch: got {self.results['p_1']}, target {target_p1}", "WARNING")
                return False
            
            if not math.isclose(self.results['Deltap'], target_delta_p, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Δp mismatch: got {self.results['Deltap']}, target {target_delta_p}", "WARNING")
                return False
            
            # Check that all results match expected values from steps analysis
            for key, expected_val in self.expected.items():
                if self.results[key] is not None:
                    if not math.isclose(self.results[key], expected_val, rel_tol=0.02, abs_tol=0.02):
                        self.log(f"Result {key} mismatch: got {self.results[key]}, expected {expected_val}", "WARNING")
                        return False
            
            self.log("Trace consistency test passed.", "INFO")
            return True
            
        except Exception as e:
            self.log(f"Trace consistency test failed with error: {e}\n{traceback.format_exc()}", "ERROR")
            return False
    
    def test_monotonicity(self):
        """Perturb mass and check trend in final momentum values."""
        try:
            original_mass = self.mass
            perturbed_mass = original_mass * 1.1  # Increase by 10%
            
            # Calculate expected results with perturbed mass using formulas
            # Acceleration remains the same (independent of mass)
            a = -self.g * math.sin(self.theta_rad)
            
            # Calculate velocities (unchanged, independent of mass)
            v1 = self.v0 + a * 1.0
            v2 = self.v0 + a * 3.0
            
            # Calculate momenta with perturbed mass
            p0_pert = perturbed_mass * self.v0
            p1_pert = perturbed_mass * v1
            p2_pert = perturbed_mass * v2
            delta_p_pert = p2_pert - p0_pert
            
            # Original momenta (from results or calculated)
            p0_orig = self.results['p_0']
            p1_orig = self.results['p_1']
            delta_p_orig = self.results['Deltap']
            
            # Check monotonicity: if mass increases, all momenta should increase in magnitude
            # Since v1 and v2 can be positive or negative, we check absolute values
            if not (abs(p1_pert) > abs(p1_orig) and abs(delta_p_pert) > abs(delta_p_orig)):
                self.log(f"Monotonicity failed: p1_pert={p1_pert}, p1_orig={p1_orig}, "
                        f"Δp_pert={delta_p_pert}, Δp_orig={delta_p_orig}", "WARNING")
                return False
            
            # Check linear proportionality: p1_pert / p1_orig ≈ mass_pert / mass_orig
            mass_ratio = perturbed_mass / original_mass
            p1_ratio = p1_pert / p1_orig
            delta_p_ratio = delta_p_pert / delta_p_orig
            
            if not (math.isclose(p1_ratio, mass_ratio, rel_tol=0.02, abs_tol=0.02) and
                    math.isclose(delta_p_ratio, mass_ratio, rel_tol=0.02, abs_tol=0.02)):
                self.log(f"Linearity failed: p1_ratio={p1_ratio}, Δp_ratio={delta_p_ratio}, "
                        f"expected {mass_ratio}", "WARNING")
                return False
            
            # Restore original mass
            self.mass = original_mass
            
            self.log("Monotonicity test passed.", "INFO")
            return True
            
        except Exception as e:
            self.log(f"Monotonicity test failed with error: {e}\n{traceback.format_exc()}", "ERROR")
            return False
    
    def run_all_tests(self):
        """Run all tests and return success and logs."""
        self.logs = []  # Reset logs
        success = True
        
        # Run tests in order
        tests = [
            ("Trace Consistency", self.test_trace_consistency),
            ("Physical Invariants", self.test_invariants),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.log(f"Running {test_name}...", "INFO")
            if not test_func():
                success = False
                self.log(f"{test_name} failed.", "ERROR")
        
        if success:
            self.log("All tests passed successfully.", "INFO")
        else:
            self.log("Some tests failed.", "WARNING")
        
        return success, self.logs