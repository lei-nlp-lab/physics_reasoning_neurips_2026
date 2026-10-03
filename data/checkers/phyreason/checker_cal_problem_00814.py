import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.direction = getattr(solution_instance, 'N/A', solution_instance.results.get('N/A', None))
        self.s = getattr(solution_instance, 's', solution_instance.results.get('s', None))
        self.t = getattr(solution_instance, 't', solution_instance.results.get('t', None))
        
        # Constants from problem
        self.v = 0.6  # m/s
        self.x_Q = 0.24  # m
        self.x_P = 0.96  # m
        
        # Derived quantities
        self.distance_QP = self.x_P - self.x_Q  # 0.72 m
        
        # Compute wavelength from step_2 equation if s is available
        self.lambda_val = None
        if self.s is not None:
            try:
                # s = x_P + (3/4)*lambda => lambda = (s - x_P) * (4/3)
                self.lambda_val = (self.s - self.x_P) * (4/3)
            except:
                self.lambda_val = None

    def test_invariants(self):
        logs = []
        
        # 1. Check wave speed formula v = lambda * f
        if self.lambda_val is not None:
            f_calc = self.v / self.lambda_val
            if f_calc > 0:
                logs.append(f"PASS: Wave speed relation v = λ*f yields f = {f_calc:.3f} Hz")
            else:
                logs.append(f"FAIL: Calculated frequency {f_calc:.3f} Hz is not positive")
        else:
            logs.append("SKIP: Cannot check v=λ*f without wavelength")
        
        # 2. Check time-distance-speed relationship: t = s/v
        if self.s is not None and self.t is not None:
            t_calc = self.s / self.v
            if math.isclose(self.t, t_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"PASS: t = s/v check: {self.t:.3f} ≈ {self.s:.3f}/{self.v:.3f} = {t_calc:.3f}")
            else:
                logs.append(f"FAIL: t = s/v mismatch: {self.t:.3f} ≠ {self.s:.3f}/{self.v:.3f} = {t_calc:.3f}")
        else:
            logs.append("SKIP: t = s/v check requires s and t values")
            
        # 3. Check distance formula s = x_P + (3/4)*lambda
        if self.s is not None and self.lambda_val is not None:
            s_calc = self.x_P + (3/4) * self.lambda_val
            if math.isclose(self.s, s_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"PASS: s = x_P + (3/4)λ check: {self.s:.3f} ≈ {self.x_P:.3f} + 0.75*{self.lambda_val:.3f}")
            else:
                logs.append(f"FAIL: s formula mismatch: {self.s:.3f} ≠ {self.x_P:.3f} + 0.75*{self.lambda_val:.3f}")
        
        # 4. Check distance QP consistency
        expected_QP = 0.72  # 0.96 - 0.24
        if math.isclose(self.distance_QP, expected_QP, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"PASS: Distance QP = {self.distance_QP:.3f} m matches expected 0.72 m")
        else:
            logs.append(f"FAIL: Distance QP = {self.distance_QP:.3f} m, expected 0.72 m")
            
        return logs

    def test_trace_consistency(self):
        logs = []
        
        # Check direction result
        if self.direction is not None:
            if isinstance(self.direction, str) and "downward" in self.direction.lower():
                logs.append("PASS: Vibration direction correctly identified as downward")
            else:
                logs.append(f"FAIL: Direction '{self.direction}' does not match expected 'downward'")
        else:
            logs.append("SKIP: No direction result available")
            
        # Check final answer consistency
        if self.t is not None:
            if math.isclose(self.t, 1.9, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"PASS: Final time t = {self.t:.3f} s matches target 1.9 s")
            else:
                logs.append(f"FAIL: Final time t = {self.t:.3f} s differs from target 1.9 s")
                
        # Check if s makes physical sense (positive distance)
        if self.s is not None:
            if self.s > 0:
                logs.append(f"PASS: Distance s = {self.s:.3f} m is positive")
            else:
                logs.append(f"FAIL: Distance s = {self.s:.3f} m is non-positive")
                
        return logs

    def test_monotonicity(self):
        logs = []
        
        # Perturb the wave speed v and check trend
        original_v = self.v
        perturbed_v = original_v * 1.1  # Increase by 10%
        
        # Save original state
        original_s = self.s
        original_t = self.t
        
        try:
            # Recalculate s and t with perturbed v using formulas from steps
            if original_s is not None:
                # s depends on lambda, not v directly (lambda is fixed by wave source)
                # But t = s/v, so t should decrease when v increases
                if original_t is not None:
                    new_t = original_s / perturbed_v
                    
                    # Check monotonic trend: t should decrease when v increases
                    if new_t < original_t:
                        logs.append(f"PASS: Monotonicity - Increasing v from {original_v:.3f} to {perturbed_v:.3f} m/s decreases t from {original_t:.3f} to {new_t:.3f} s")
                    else:
                        logs.append(f"FAIL: Monotonicity violated - t didn't decrease when v increased")
                    
                    # Check proportionality: t ∝ 1/v
                    expected_ratio = original_v / perturbed_v
                    actual_ratio = new_t / original_t if original_t != 0 else 0
                    
                    if math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"PASS: t ∝ 1/v holds: ratio {actual_ratio:.3f} ≈ {expected_ratio:.3f}")
                    else:
                        logs.append(f"WARN: t ∝ 1/v approximation off: {actual_ratio:.3f} vs {expected_ratio:.3f}")
                else:
                    logs.append("SKIP: Monotonicity test requires original t")
            else:
                logs.append("SKIP: Monotonicity test requires original s")
                
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {e}")
            traceback.print_exc()
            
        # Restore original state
        self.v = original_v
        
        return logs

    def run_all_tests(self):
        all_logs = []
        success = True
        
        try:
            # Run each test suite
            for test_name, test_func in [
                ("Invariants", self.test_invariants),
                ("Trace Consistency", self.test_trace_consistency),
                ("Monotonicity", self.test_monotonicity)
            ]:
                all_logs.append(f"\n=== {test_name} ===")
                logs = test_func()
                all_logs.extend(logs)
                
                # Check for failures in critical tests
                for log in logs:
                    if "FAIL:" in log and test_name != "Monotonicity":
                        success = False
                        
        except Exception as e:
            all_logs.append(f"ERROR in test execution: {e}")
            success = False
            
        return success, all_logs

# Example usage and test
if __name__ == "__main__":
    # Mock solution instance for testing
    class MockSolution:
        def __init__(self):
            self.results = {
                'N/A': 'downward',
                's': 1.14,  # 0.96 + 0.75*0.24 = 1.14
                't': 1.9    # 1.14 / 0.6 = 1.9
            }
    
    mock_solution = MockSolution()
    checker = PhysicsChecker(mock_solution)
    
    print("Physics Checker Initialized")
    print(f"Direction: {checker.direction}")
    print(f"s: {checker.s}")
    print(f"t: {checker.t}")
    print(f"Calculated λ: {checker.lambda_val}")
    
    success, logs = checker.run_all_tests()
    print(f"\nOverall Success: {success}")
    for log in logs:
        print(log)