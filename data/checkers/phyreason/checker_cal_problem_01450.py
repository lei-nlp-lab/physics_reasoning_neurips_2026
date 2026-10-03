import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from the solution instance."""
        # Constants from the problem statement
        self.v_bike = 6.0  # m/s (constant)
        self.v_car0 = 18.0  # m/s (initial car speed)
        self.a_car = -2.0   # m/s² (deceleration magnitude is 2, direction negative)
        
        # Results extracted with safe access (using .results dictionary or direct attributes)
        self.t0 = getattr(solution_instance, 't_0', getattr(solution_instance, 'results', {}).get('t_0', 0.0))
        self.t1 = getattr(solution_instance, 't_1', getattr(solution_instance, 'results', {}).get('t_1', 0.0))
        self.x1 = getattr(solution_instance, 'x_1', getattr(solution_instance, 'results', {}).get('x_1', 0.0))
        self.x2 = getattr(solution_instance, 'x_2', getattr(solution_instance, 'results', {}).get('x_2', 0.0))
        self.delta_x = getattr(solution_instance, 'Δx', getattr(solution_instance, 'delta_x', 
                                 getattr(solution_instance, 'results', {}).get('Δx', 
                                 getattr(solution_instance, 'results', {}).get('delta_x', 0.0))))
        self.x2_prime = getattr(solution_instance, 'x_2\'', getattr(solution_instance, 'x_2\'\'', 
                                 getattr(solution_instance, 'results', {}).get('x_2\'', 
                                 getattr(solution_instance, 'results', {}).get('x_2\'\'', 0.0))))
        self.x1_prime = getattr(solution_instance, 'x_1\'', getattr(solution_instance, 'x_1\'\'',
                                 getattr(solution_instance, 'results', {}).get('x_1\'', 
                                 getattr(solution_instance, 'results', {}).get('x_1\'\'', 0.0))))
        self.t_meet = getattr(solution_instance, 't', getattr(solution_instance, 'results', {}).get('t', 0.0))
        
        # For monotonicity perturbation (store original for restoration)
        self.original_v_car0 = self.v_car0
        
    def test_invariants(self):
        """Verify kinematic laws and conservation principles."""
        logs = []
        try:
            # 1. Time for car to stop: t0 = (0 - v_car0) / a_car
            expected_t0 = (0 - self.v_car0) / self.a_car
            if not math.isclose(self.t0, expected_t0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: t0 mismatch. Expected {expected_t0:.3f}, got {self.t0:.3f}")
            else:
                logs.append(f"PASS: t0 = {self.t0:.3f} matches kinematic equation.")
            
            # 2. Time when speeds equal: t1 = (v_bike - v_car0) / a_car
            expected_t1 = (self.v_bike - self.v_car0) / self.a_car
            if not math.isclose(self.t1, expected_t1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: t1 mismatch. Expected {expected_t1:.3f}, got {self.t1:.3f}")
            else:
                logs.append(f"PASS: t1 = {self.t1:.3f} matches kinematic equation.")
            
            # 3. Bicycle displacement at t1: x1 = v_bike * t1
            expected_x1 = self.v_bike * self.t1
            if not math.isclose(self.x1, expected_x1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: x1 mismatch. Expected {expected_x1:.3f}, got {self.x1:.3f}")
            else:
                logs.append(f"PASS: x1 = {self.x1:.3f} matches uniform motion.")
            
            # 4. Car displacement at t1: x2 = v_car0*t1 + 0.5*a_car*t1²
            expected_x2 = self.v_car0 * self.t1 + 0.5 * self.a_car * self.t1**2
            if not math.isclose(self.x2, expected_x2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: x2 mismatch. Expected {expected_x2:.3f}, got {self.x2:.3f}")
            else:
                logs.append(f"PASS: x2 = {self.x2:.3f} matches uniformly varied motion.")
            
            # 5. Maximum distance Δx = x2 - x1
            expected_delta_x = self.x2 - self.x1
            if not math.isclose(self.delta_x, expected_delta_x, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Δx mismatch. Expected {expected_delta_x:.3f}, got {self.delta_x:.3f}")
            else:
                logs.append(f"PASS: Δx = {self.delta_x:.3f} equals x2 - x1.")
            
            # 6. Car displacement at stop: x2' = v_car0*t0 + 0.5*a_car*t0²
            expected_x2_prime = self.v_car0 * self.t0 + 0.5 * self.a_car * self.t0**2
            if not math.isclose(self.x2_prime, expected_x2_prime, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: x2' mismatch. Expected {expected_x2_prime:.3f}, got {self.x2_prime:.3f}")
            else:
                logs.append(f"PASS: x2' = {self.x2_prime:.3f} matches stopping distance.")
            
            # 7. Bicycle displacement when car stops: x1' = v_bike * t0
            expected_x1_prime = self.v_bike * self.t0
            if not math.isclose(self.x1_prime, expected_x1_prime, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: x1' mismatch. Expected {expected_x1_prime:.3f}, got {self.x1_prime:.3f}")
            else:
                logs.append(f"PASS: x1' = {self.x1_prime:.3f} matches uniform motion.")
            
            # 8. Time to meet again: t_meet = t0 + (x2' - x1') / v_bike
            # (After car stops, bicycle must cover the gap at constant speed)
            gap = self.x2_prime - self.x1_prime
            expected_t_meet = self.t0 + gap / self.v_bike
            if not math.isclose(self.t_meet, expected_t_meet, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: t_meet mismatch. Expected {expected_t_meet:.3f}, got {self.t_meet:.3f}")
            else:
                logs.append(f"PASS: t_meet = {self.t_meet:.3f} consistent with gap closure.")
            
        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
        
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # The three target answers: t0, Δx, t_meet
            # These should match the extracted results
            target_t0 = 9.0
            target_delta_x = 36.0
            target_t_meet = 13.5
            
            checks = [
                ("t0", self.t0, target_t0),
                ("Δx", self.delta_x, target_delta_x),
                ("t_meet", self.t_meet, target_t_meet)
            ]
            
            for name, value, target in checks:
                if math.isclose(value, target, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"PASS: {name} = {value:.3f} matches target answer.")
                else:
                    logs.append(f"FAIL: {name} = {value:.3f} differs from target {target}.")
            
            # Additionally, check that t1 = 6.0 (from steps)
            if math.isclose(self.t1, 6.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("PASS: t1 = 6.0 as expected.")
            else:
                logs.append(f"WARNING: t1 = {self.t1:.3f} (expected 6.0).")
                
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
        
        return logs
    
    def test_monotonicity(self):
        """
        Perturb initial car speed, recompute key outputs using kinematic formulas,
        check expected trends, and restore original value.
        """
        logs = []
        original = self.v_car0
        
        try:
            # Increase initial car speed by 10%
            perturbed_v_car0 = original * 1.10
            self.v_car0 = perturbed_v_car0
            
            # Recompute using kinematic formulas (numerical, not calling solution methods)
            perturbed_t0 = (0 - self.v_car0) / self.a_car
            perturbed_t1 = (self.v_bike - self.v_car0) / self.a_car
            perturbed_x1 = self.v_bike * perturbed_t1
            perturbed_x2 = self.v_car0 * perturbed_t1 + 0.5 * self.a_car * perturbed_t1**2
            perturbed_delta_x = perturbed_x2 - perturbed_x1
            perturbed_x2_prime = self.v_car0 * perturbed_t0 + 0.5 * self.a_car * perturbed_t0**2
            perturbed_x1_prime = self.v_bike * perturbed_t0
            perturbed_gap = perturbed_x2_prime - perturbed_x1_prime
            perturbed_t_meet = perturbed_t0 + perturbed_gap / self.v_bike
            
            # Check monotonic trends (all should increase with increased initial speed)
            trends = [
                ("t0", self.t0, perturbed_t0, perturbed_t0 > self.t0),
                ("Δx", self.delta_x, perturbed_delta_x, perturbed_delta_x > self.delta_x),
                ("t_meet", self.t_meet, perturbed_t_meet, perturbed_t_meet > self.t_meet)
            ]
            
            for name, orig, pert, expected in trends:
                if expected:
                    logs.append(f"PASS: {name} increases with increased v_car0 (from {orig:.3f} to {pert:.3f}).")
                else:
                    logs.append(f"FAIL: {name} does not show expected monotonic trend.")
            
            # Restore original value
            self.v_car0 = original
            
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            self.v_car0 = original  # Ensure restoration even on error
        
        return logs
    
    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        logs = []
        all_passed = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            logs.append(f"\n=== {test_name} ===")
            test_logs = test_method()
            logs.extend(test_logs)
            
            # Check for any FAIL or ERROR in the test logs
            for entry in test_logs:
                if entry.startswith("FAIL") or entry.startswith("ERROR"):
                    all_passed = False
        
        # Final summary
        logs.append(f"\n=== SUMMARY ===")
        if all_passed:
            logs.append("SUCCESS: All physics checks passed.")
        else:
            logs.append("FAILURE: Some physics checks failed.")
        
        return all_passed, logs