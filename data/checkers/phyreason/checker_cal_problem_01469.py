import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize with safe attribute access.
        solution_instance should have a .results dictionary containing the computed values.
        """
        self.solution = solution_instance
        
        # Inputs from problem statement
        self.s1 = 40.0  # m, north
        self.s2 = 30.0  # m, west
        self.t1 = 5.0   # s
        self.t2 = 5.0   # s
        self.total_t = self.t1 + self.t2  # 10 s
        
        # Extract results safely with fallbacks
        results = getattr(solution_instance, 'results', {})
        
        self.x = results.get('x', 0.0)
        self.v_avg = results.get('v_avg', 0.0)
        self.direction = results.get('direction', 0.0)  # Could be numeric or string
        self.theta = results.get('theta', 0.0)
        self.v2 = results.get('v2', 0.0)
        self.s = results.get('s', 0.0)
        self.v1 = results.get('v1', 0.0)
        
        # Additional derived quantities for checking
        self.expected_x = math.hypot(self.s1, self.s2)  # 50 m
        self.expected_theta = math.degrees(math.atan2(self.s2, self.s1))  # ~36.87°
        self.expected_v_avg = self.expected_x / self.total_t  # 5 m/s
        self.expected_s = self.s1 + self.s2  # 70 m
        self.expected_v_avg_speed = self.expected_s / self.total_t  # 7 m/s
        self.expected_v1 = self.s1 / self.t1  # 8 m/s
        self.expected_v2 = self.s2 / self.t2  # 6 m/s
        
        self.tol_rel = 1e-2
        self.tol_abs = 1e-2
        
    def is_close(self, a, b):
        """Robust fuzzy comparison."""
        return math.isclose(a, b, rel_tol=self.tol_rel, abs_tol=self.tol_abs)
    
    def test_invariants(self):
        """Verify fundamental physical laws and definitions."""
        logs = []
        
        # 1. Pythagorean theorem for displacement
        if not self.is_close(self.x, self.expected_x):
            logs.append(f"FAIL: Displacement x = {self.x} m, expected {self.expected_x} m")
        else:
            logs.append(f"PASS: Displacement x = {self.x} m matches Pythagorean theorem.")
        
        # 2. Trigonometry for angle (arctan(s2/s1))
        if not self.is_close(self.theta, self.expected_theta):
            logs.append(f"FAIL: Angle theta = {self.theta}°, expected {self.expected_theta}°")
        else:
            logs.append(f"PASS: Angle theta = {self.theta}° matches trigonometry.")
        
        # 3. Definition of average velocity (vector)
        if not self.is_close(self.v_avg, self.expected_v_avg):
            logs.append(f"FAIL: Average velocity magnitude = {self.v_avg} m/s, expected {self.expected_v_avg} m/s")
        else:
            logs.append(f"PASS: Average velocity magnitude = {self.v_avg} m/s matches definition.")
        
        # Check direction - handle both numeric and string cases
        if isinstance(self.direction, (int, float)):
            # Numeric angle in degrees
            if not self.is_close(self.direction, self.expected_theta):
                logs.append(f"WARNING: Direction angle {self.direction}° differs from expected {self.expected_theta}°")
            else:
                logs.append(f"PASS: Direction angle {self.direction}° matches expected.")
        else:
            # String description
            expected_dir = "37 west of north"
            if expected_dir not in str(self.direction).lower():
                logs.append(f"WARNING: Direction string '{self.direction}' may not match expected '{expected_dir}'")
            else:
                logs.append(f"PASS: Direction description seems correct.")
        
        # 4. Total distance (scalar addition)
        if not self.is_close(self.s, self.expected_s):
            logs.append(f"FAIL: Total distance s = {self.s} m, expected {self.expected_s} m")
        else:
            logs.append(f"PASS: Total distance s = {self.s} m matches scalar addition.")
        
        # 5. Average speed (scalar)
        v_avg_speed = self.s / self.total_t
        if not self.is_close(v_avg_speed, self.expected_v_avg_speed):
            logs.append(f"FAIL: Average speed = {v_avg_speed} m/s, expected {self.expected_v_avg_speed} m/s")
        else:
            logs.append(f"PASS: Average speed = {v_avg_speed} m/s matches definition.")
        
        # 6. First leg average speed
        if not self.is_close(self.v1, self.expected_v1):
            logs.append(f"FAIL: First leg speed v1 = {self.v1} m/s, expected {self.expected_v1} m/s")
        else:
            logs.append(f"PASS: First leg speed v1 = {self.v1} m/s matches definition.")
        
        # 7. Second leg average speed
        if not self.is_close(self.v2, self.expected_v2):
            logs.append(f"FAIL: Second leg speed v2 = {self.v2} m/s, expected {self.expected_v2} m/s")
        else:
            logs.append(f"PASS: Second leg speed v2 = {self.v2} m/s matches definition.")
        
        return logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        
        # The target answers are:
        # 1. 5 m/s, 37° west of north (average velocity)
        # 2. 7 m/s (average speed)
        # 3. 8 m/s, north (first leg speed)
        # 4. 6 m/s (second leg speed)
        
        # Check 1: average velocity magnitude
        if not self.is_close(self.v_avg, 5.0):
            logs.append(f"FAIL: Average velocity magnitude {self.v_avg} doesn't match target 5 m/s.")
        else:
            logs.append("PASS: Average velocity magnitude matches target 5 m/s.")
        
        # Check 2: average speed
        avg_speed = self.s / self.total_t
        if not self.is_close(avg_speed, 7.0):
            logs.append(f"FAIL: Average speed {avg_speed} doesn't match target 7 m/s.")
        else:
            logs.append("PASS: Average speed matches target 7 m/s.")
        
        # Check 3: first leg speed
        if not self.is_close(self.v1, 8.0):
            logs.append(f"FAIL: First leg speed {self.v1} doesn't match target 8 m/s.")
        else:
            logs.append("PASS: First leg speed matches target 8 m/s.")
        
        # Check 4: second leg speed
        if not self.is_close(self.v2, 6.0):
            logs.append(f"FAIL: Second leg speed {self.v2} doesn't match target 6 m/s.")
        else:
            logs.append("PASS: Second leg speed matches target 6 m/s.")
        
        return logs
    
    def test_monotonicity(self):
        """
        Perturb an input (s1) and verify that average velocity and speed change monotonically.
        """
        logs = []
        original_s1 = self.s1
        
        try:
            # Increase s1 by 10%
            perturbed_s1 = original_s1 * 1.10
            perturbed_x = math.hypot(perturbed_s1, self.s2)
            perturbed_v_avg = perturbed_x / self.total_t
            perturbed_s = perturbed_s1 + self.s2
            perturbed_avg_speed = perturbed_s / self.total_t
            
            # Check monotonic increase for both velocity and speed
            if perturbed_v_avg <= self.expected_v_avg:
                logs.append(f"FAIL: Increasing s1 did not increase average velocity (from {self.expected_v_avg} to {perturbed_v_avg}).")
            else:
                logs.append(f"PASS: Average velocity increases with s1 (from {self.expected_v_avg} to {perturbed_v_avg}).")
            
            if perturbed_avg_speed <= self.expected_v_avg_speed:
                logs.append(f"FAIL: Increasing s1 did not increase average speed (from {self.expected_v_avg_speed} to {perturbed_avg_speed}).")
            else:
                logs.append(f"PASS: Average speed increases with s1 (from {self.expected_v_avg_speed} to {perturbed_avg_speed}).")
            
            # Restore original s1
            self.s1 = original_s1
        except Exception as e:
            logs.append(f"ERROR during monotonicity test: {e}")
            self.s1 = original_s1  # Ensure restoration
        
        return logs
    
    def run_all_tests(self):
        """
        Run all checks and return (success_bool, logs_list).
        Success is defined as all critical tests passing (no FAIL logs).
        """
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
                # Check for any FAIL in this test suite
                if any("FAIL" in log for log in logs):
                    success = False
        except Exception as e:
            all_logs.append(f"\nCRITICAL: Checker crashed: {traceback.format_exc()}")
            success = False
        
        return success, all_logs