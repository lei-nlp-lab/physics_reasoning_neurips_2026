import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Never access solution_instance.attr directly—use getattr with fallback.
        """
        self.solution = solution_instance
        # Inputs: displacements and time intervals (with defaults from problem statement)
        self.dx = getattr(solution_instance, 'dx', 6.0)
        self.dy = getattr(solution_instance, 'dy', 8.0)
        self.dt1 = getattr(solution_instance, 'dt1', 1.0)
        self.dt2 = getattr(solution_instance, 'dt2', 1.0)
        
        # Results: expected keys from the mapping
        # Use results dict if available, otherwise fallback to attributes
        results_dict = getattr(solution_instance, 'results', {})
        self.x = getattr(solution_instance, 'x', results_dict.get('x', 0.0))       # displacement magnitude
        self.v = getattr(solution_instance, 'v', results_dict.get('v', 0.0))       # average velocity magnitude
        self.theta = getattr(solution_instance, 'theta', results_dict.get('theta', 0.0))  # direction in degrees
        
        # Tolerance for fuzzy matching (1-2% relative/absolute)
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # For symbolic checks
        self.x_sym, self.y_sym, self.t_sym = sp.symbols('x y t', positive=True, real=True)
        self.v_avg_sym = sp.sqrt(self.x_sym**2 + self.y_sym**2) / (2*self.t_sym)  # t = dt1 = dt2 = 1, total = 2
        
    def fuzzy_equal(self, a, b):
        """Robust numerical comparison with tolerances."""
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
    
    def test_invariants(self):
        """
        Verify physical invariants:
        1. Pythagorean theorem for displacement magnitude.
        2. Trigonometry for direction (arctan).
        3. Definition of average velocity (displacement / total time).
        """
        logs = []
        try:
            # 1. Displacement magnitude: sqrt(dx^2 + dy^2)
            expected_x = math.hypot(self.dx, self.dy)
            if self.fuzzy_equal(self.x, expected_x):
                logs.append(f"✓ Displacement magnitude invariant: {self.x} ≈ {expected_x}")
            else:
                logs.append(f"✗ Displacement magnitude mismatch: {self.x} ≠ {expected_x}")
                return False, logs
            
            # 2. Direction: θ = arctan(dy/dx) in degrees
            expected_theta = math.degrees(math.atan2(self.dy, self.dx))
            if self.fuzzy_equal(self.theta, expected_theta):
                logs.append(f"✓ Direction invariant: {self.theta}° ≈ {expected_theta}°")
            else:
                logs.append(f"✗ Direction mismatch: {self.theta}° ≠ {expected_theta}°")
                return False, logs
            
            # 3. Average velocity magnitude: v = displacement / total_time
            total_time = self.dt1 + self.dt2
            expected_v = expected_x / total_time
            if self.fuzzy_equal(self.v, expected_v):
                logs.append(f"✓ Average velocity invariant: {self.v} ≈ {expected_v}")
            else:
                logs.append(f"✗ Average velocity mismatch: {self.v} ≠ {expected_v}")
                return False, logs
            
            logs.append("All invariants passed.")
            return True, logs
        except Exception as e:
            logs.append(f"Error in test_invariants: {e}")
            return False, logs
    
    def test_trace_consistency(self):
        """
        Cross-check intermediate results (if available) with final answers.
        If intermediates aren't provided, use the steps from the KB analysis.
        """
        logs = []
        try:
            # Try to extract intermediate step results from solution instance
            step1 = getattr(self.solution, 'step1_result', None)  # displacement magnitude
            step2 = getattr(self.solution, 'step2_result', None)  # direction
            step3 = getattr(self.solution, 'step3_result', None)  # average velocity
            
            # Step 1: displacement magnitude = sqrt(6^2 + 8^2) = 10
            expected_step1 = math.hypot(self.dx, self.dy)
            if step1 is not None:
                if self.fuzzy_equal(step1, expected_step1):
                    logs.append(f"✓ Step 1 (displacement magnitude) consistent: {step1} ≈ {expected_step1}")
                else:
                    logs.append(f"✗ Step 1 mismatch: {step1} ≠ {expected_step1}")
                    return False, logs
            else:
                logs.append("ℹ Step 1 result not available; skipping consistency check.")
            
            # Step 2: direction θ = arctan(8/6) ≈ 53°
            expected_step2 = math.degrees(math.atan2(self.dy, self.dx))
            if step2 is not None:
                if self.fuzzy_equal(step2, expected_step2):
                    logs.append(f"✓ Step 2 (direction) consistent: {step2}° ≈ {expected_step2}°")
                else:
                    logs.append(f"✗ Step 2 mismatch: {step2}° ≠ {expected_step2}°")
                    return False, logs
            else:
                logs.append("ℹ Step 2 result not available; skipping consistency check.")
            
            # Step 3: average velocity = 10 / 2 = 5
            expected_step3 = expected_step1 / (self.dt1 + self.dt2)
            if step3 is not None:
                if self.fuzzy_equal(step3, expected_step3):
                    logs.append(f"✓ Step 3 (average velocity) consistent: {step3} ≈ {expected_step3}")
                else:
                    logs.append(f"✗ Step 3 mismatch: {step3} ≠ {expected_step3}")
                    return False, logs
            else:
                logs.append("ℹ Step 3 result not available; skipping consistency check.")
            
            logs.append("Trace consistency checks passed.")
            return True, logs
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {e}")
            return False, logs
    
    def test_monotonicity(self):
        """
        Perturb an input (dx) and verify that the output changes according to the theorem:
        - If displacement magnitude increases (time fixed), average velocity magnitude increases.
        """
        logs = []
        original_dx = self.dx
        perturbation = 1.0  # increase by 1 m
        
        try:
            # Original values
            original_disp = math.hypot(original_dx, self.dy)
            original_vel = original_disp / (self.dt1 + self.dt2)
            
            # Perturbed scenario
            perturbed_dx = original_dx + perturbation
            perturbed_disp = math.hypot(perturbed_dx, self.dy)
            perturbed_vel = perturbed_disp / (self.dt1 + self.dt2)
            
            # Check monotonicity: both displacement and velocity should increase
            if perturbed_disp > original_disp and perturbed_vel > original_vel:
                logs.append(f"✓ Monotonicity: dx ↑ → displacement ↑ ({original_disp:.2f} → {perturbed_disp:.2f})")
                logs.append(f"✓ Monotonicity: dx ↑ → average velocity ↑ ({original_vel:.2f} → {perturbed_vel:.2f})")
                logs.append("Monotonicity test passed.")
                return True, logs
            else:
                logs.append(f"✗ Monotonicity violation:")
                logs.append(f"  Original: displacement={original_disp:.2f}, velocity={original_vel:.2f}")
                logs.append(f"  Perturbed: displacement={perturbed_disp:.2f}, velocity={perturbed_vel:.2f}")
                return False, logs
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {e}")
            return False, logs
        finally:
            # Restore original dx (important for later use if needed)
            self.dx = original_dx
    
    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n--- {test_name} Test ---")
            success, logs = test_func()
            all_logs.extend(logs)
            if not success:
                overall_success = False
        
        all_logs.append(f"\n=== Overall Result: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, all_logs