import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for inputs and results
        self.solution = solution_instance
        
        # Inputs from the problem statement
        self.L = 40.0  # field length (m)
        self.W = 20.0  # field width (m)
        self.v1 = 5.0  # initial ball velocity (m/s)
        self.a1 = 0.5  # ball deceleration magnitude (m/s^2)
        self.delay = 1.0  # player delay (s)
        self.a2 = 1.0  # player acceleration (m/s^2)
        self.v2_max = 6.0  # player max speed (m/s)
        
        # Extract results safely - ONLY from results dictionary
        self.results = {}
        keys = ['delta_x', 'x', 'x''', 'x_3', 't_2', 't_4', 't_3', 'v', 'x_2', 'v_1''', 't_1', 'x_0', 'x_1']
        
        # First, get the results dictionary from the solution instance
        # Use getattr with default empty dict if results attribute doesn't exist
        solution_results = getattr(solution_instance, 'results', {})
        
        for key in keys:
            # Handle the double-prime keys by mapping to Python-safe names
            if key == "x''":
                # Map x'' to x_double_prime in the results dict
                self.results[key] = solution_results.get('x_double_prime', 0.0)
            elif key == "v_1''":
                # Map v_1'' to v_1_double_prime in the results dict
                self.results[key] = solution_results.get('v_1_double_prime', 0.0)
            else:
                # For all other keys, get directly from results dict
                self.results[key] = solution_results.get(key, 0.0)
        
        # Alias for easier access
        self.t1 = self.results.get('t_1', 0.0)
        self.delta_x = self.results.get('delta_x', 0.0)
        self.t4 = self.results.get('t_4', 0.0)
        self.v1_final = self.results.get('v_1''', 0.0)  # This should now get the correct value
        
        # Intermediate results from steps analysis for cross-checking
        self.step_values = {
            't_1': 3.0,
            'x': 16.0,  # ball displacement to max separation
            'x\'': 4.5,  # player displacement to max separation
            'delta_x': 11.5,
            't_2': 10.0,
            't_3': 6.0,
            'x_2': 18.0,
            'x_1': 22.75,
            'x_0': 25.0,
            'x_3': 18.0,
            't_4': 7.0,
            'v_1\'': 1.0
        }
        
        self.logs = []

    def log(self, message):
        self.logs.append(message)

    def test_invariants(self):
        """Verify kinematic invariants using numeric checks only."""
        try:
            # Numerical checks on provided results
            # Check ball stopping time: t_2 = v1 / a1
            if self.a1 == 0:
                self.log("Division by zero: a1 cannot be zero")
                return False
            t2_calc = self.v1 / self.a1
            if not math.isclose(self.results.get('t_2', 0.0), t2_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Ball stopping time mismatch: {self.results.get('t_2')} vs {t2_calc}")
                return False
            
            # Check player time to max speed: t_3 = v2_max / a2
            if self.a2 == 0:
                self.log("Division by zero: a2 cannot be zero")
                return False
            t3_calc = self.v2_max / self.a2
            if not math.isclose(self.results.get('t_3', 0.0), t3_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Player time to max speed mismatch: {self.results.get('t_3')} vs {t3_calc}")
                return False
            
            # Check ball displacement until stop: x_0 = v1^2 / (2*a1)
            if self.a1 == 0:
                self.log("Division by zero: a1 cannot be zero")
                return False
            x0_calc = self.v1**2 / (2 * self.a1)
            if not math.isclose(self.results.get('x_0', 0.0), x0_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Ball displacement until stop mismatch: {self.results.get('x_0')} vs {x0_calc}")
                return False
            
            # Check player displacement during acceleration: x_2 = 0.5*a2*t3^2
            x2_calc = 0.5 * self.a2 * self.results.get('t_3', 0.0)**2
            if not math.isclose(self.results.get('x_2', 0.0), x2_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Player acceleration displacement mismatch: {self.results.get('x_2')} vs {x2_calc}")
                return False
            
            self.log("Numerical invariants passed.")
            return True
            
        except Exception as e:
            self.log(f"Invariant test failed with error: {e}")
            return False

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Compare with step_values from the KB
            for key, expected in self.step_values.items():
                actual = self.results.get(key)
                if actual is not None:
                    if not math.isclose(actual, expected, rel_tol=1e-2, abs_tol=1e-2):
                        self.log(f"Trace mismatch for {key}: {actual} vs {expected}")
                        return False
            
            # Additional consistency checks between steps
            
            # 1. At maximum separation, ball and player velocities should be equal
            v_ball_at_t1 = self.v1 - self.a1 * (self.delay + self.t1)
            v_player_at_t1 = self.a2 * self.t1
            if not math.isclose(v_ball_at_t1, v_player_at_t1, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Velocity mismatch at max separation: ball {v_ball_at_t1}, player {v_player_at_t1}")
                return False
            
            # 2. Maximum separation Δx = x_ball - x_player
            x_ball_t1 = self.v1 * (self.delay + self.t1) - 0.5 * self.a1 * (self.delay + self.t1)**2
            x_player_t1 = 0.5 * self.a2 * self.t1**2
            delta_x_calc = x_ball_t1 - x_player_t1
            if not math.isclose(self.delta_x, delta_x_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Max separation mismatch: {self.delta_x} vs {delta_x_calc}")
                return False
            
            # 3. At catch-up time t4, positions must be equal
            t_total = self.delay + self.t4
            x_ball_catch = self.v1 * t_total - 0.5 * self.a1 * t_total**2
            if self.t4 <= self.results.get('t_3', 0.0):
                x_player_catch = 0.5 * self.a2 * self.t4**2
            else:
                x_player_catch = 0.5 * self.a2 * self.results['t_3']**2 + self.v2_max * (self.t4 - self.results['t_3'])
            
            if not math.isclose(x_ball_catch, x_player_catch, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Catch-up position mismatch: ball {x_ball_catch}, player {x_player_catch}")
                return False
            
            # 4. Ball speed at catch-up: v1' = v1 - a1*(t0+t4)
            v1_final_calc = self.v1 - self.a1 * (self.delay + self.t4)
            if not math.isclose(self.v1_final, v1_final_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Ball final speed mismatch: {self.v1_final} vs {v1_final_calc}")
                return False
            
            self.log("Trace consistency passed.")
            return True
            
        except Exception as e:
            self.log(f"Trace consistency test failed with error: {e}")
            return False

    def test_monotonicity(self):
        """Perturb an input and check the trend, then restore."""
        original_a1 = self.a1
        success = True
        
        try:
            # Increase ball deceleration by 10%
            self.a1 *= 1.1
            
            # Recalculate maximum separation time t1
            if (self.a1 + self.a2) == 0:
                self.log("Division by zero in monotonicity test")
                self.a1 = original_a1
                return False
                
            t1_new = (self.v1 - self.a1 * self.delay) / (self.a1 + self.a2)
            
            # The original t1 was 3 s. With increased deceleration, t1 should decrease.
            if t1_new >= self.step_values['t_1']:
                self.log(f"Monotonicity failed: increased a1 should decrease t1, but {t1_new} >= {self.step_values['t_1']}")
                success = False
            
            # Recalculate maximum separation distance
            x_ball_new = self.v1 * (self.delay + t1_new) - 0.5 * self.a1 * (self.delay + t1_new)**2
            x_player_new = 0.5 * self.a2 * t1_new**2
            delta_x_new = x_ball_new - x_player_new
            
            # With increased deceleration, the ball slows faster, so maximum separation should decrease.
            if delta_x_new >= self.step_values['delta_x']:
                self.log(f"Monotonicity failed: increased a1 should decrease delta_x, but {delta_x_new} >= {self.step_values['delta_x']}")
                success = False
            
            # Restore original value
            self.a1 = original_a1
            
            if success:
                self.log("Monotonicity test passed.")
            return success
            
        except Exception as e:
            self.a1 = original_a1
            self.log(f"Monotonicity test failed with error: {e}")
            return False

    def run_all_tests(self):
        """Execute all tests and return success flag and logs."""
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
                    self.log(f"FAILED: {test_name}")
                    success = False
                else:
                    self.log(f"PASSED: {test_name}")
            except Exception as e:
                self.log(f"ERROR in {test_name}: {traceback.format_exc()}")
                success = False
        
        return success, self.logs