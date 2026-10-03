import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all necessary physics quantities from the solution instance.
        """
        self.solution_instance = solution_instance
        
        # Safe attribute access: use getattr and fallback to results dict
        results = getattr(solution_instance, 'results', {})
        
        # Problem constants (from context)
        self.v0 = 42.0  # m/s
        self.a_mag = 12.0  # m/s² (deceleration magnitude)
        self.a = -self.a_mag  # acceleration is negative for deceleration
        
        # Extracted results with safe fallbacks
        self.t_0 = self._safe_get(results, 't_0', 0.0)
        self.v = self._safe_get(results, 'v', 0.0)
        self.x = self._safe_get(results, 'x', 0.0)
        self.ramp_length = self._safe_get(results, 'N/A', 0.0)
        
        # Time for specific speed calculation (from step_2)
        self.t_specific = 2.0  # seconds
        
        # Initialize logs
        self.logs = []
        
    def _safe_get(self, results_dict, key, default):
        """Safely get value from results dict or return default."""
        val = results_dict.get(key, default)
        # Convert to float if it's a string or other type
        try:
            return float(val)
        except (ValueError, TypeError):
            return default
    
    def _log(self, message, success):
        """Helper to log test results."""
        self.logs.append({
            'test': message,
            'success': success,
            'details': None
        })
    
    def test_invariants(self):
        """
        Check fundamental physics invariants:
        1. Kinematic equations for uniformly accelerated motion.
        2. Stopping condition (v=0) consistency.
        """
        success = True
        
        # 1. Check time to stop using v = v0 + a*t (should give t_0 = 3.5s)
        t_calc = -self.v0 / self.a  # since v = 0 = v0 + a*t => t = -v0/a
        if not math.isclose(self.t_0, t_calc, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Time to stop invariant failed: got {self.t_0}, expected {t_calc}", False)
            success = False
        else:
            self._log(f"Time to stop invariant passed: {self.t_0} ≈ {t_calc}", True)
        
        # 2. Check speed at t=2s using v = v0 + a*t
        v_calc = self.v0 + self.a * self.t_specific
        if not math.isclose(self.v, v_calc, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Speed at t=2s invariant failed: got {self.v}, expected {v_calc}", False)
            success = False
        else:
            self._log(f"Speed at t=2s invariant passed: {self.v} ≈ {v_calc}", True)
        
        # 3. Check displacement using v² = v0² + 2*a*x
        x_calc = (0**2 - self.v0**2) / (2 * self.a)  # since v=0
        if not math.isclose(self.x, x_calc, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Displacement invariant failed: got {self.x}, expected {x_calc}", False)
            success = False
        else:
            self._log(f"Displacement invariant passed: {self.x} ≈ {x_calc}", True)
        
        # 4. Check ramp length equals displacement (from step_4)
        if not math.isclose(self.ramp_length, self.x, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Ramp length consistency failed: {self.ramp_length} ≠ {self.x}", False)
            success = False
        else:
            self._log(f"Ramp length consistency passed: {self.ramp_length} ≈ {self.x}", True)
        
        return success
    
    def test_trace_consistency(self):
        """
        Verify that intermediate results are consistent with each other.
        Cross-check step-by-step calculations.
        """
        success = True
        
        # From step_1: t_0 = 3.5s should match v0/|a|
        expected_t0 = self.v0 / self.a_mag
        if not math.isclose(self.t_0, expected_t0, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Trace step_1 failed: t_0={self.t_0}, expected {expected_t0}", False)
            success = False
        else:
            self._log(f"Trace step_1 passed: t_0={self.t_0} ≈ {expected_t0}", True)
        
        # From step_2: v = 18 m/s at t=2s
        expected_v = self.v0 - self.a_mag * self.t_specific
        if not math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Trace step_2 failed: v={self.v}, expected {expected_v}", False)
            success = False
        else:
            self._log(f"Trace step_2 passed: v={self.v} ≈ {expected_v}", True)
        
        # From step_3: x = 73.5 m
        expected_x = self.v0**2 / (2 * self.a_mag)
        if not math.isclose(self.x, expected_x, rel_tol=1e-2, abs_tol=1e-2):
            self._log(f"Trace step_3 failed: x={self.x}, expected {expected_x}", False)
            success = False
        else:
            self._log(f"Trace step_3 passed: x={self.x} ≈ {expected_x}", True)
        
        return success
    
    def test_monotonicity(self):
        """
        Perturb the deceleration magnitude and verify expected trends:
        - Larger deceleration => shorter stopping distance
        - Larger deceleration => shorter time to stop
        """
        success = True
        
        # Save original values
        original_a_mag = self.a_mag
        
        # Test 1: Increase deceleration magnitude by 10%
        perturbed_a_mag = original_a_mag * 1.10
        perturbed_a = -perturbed_a_mag
        
        # Calculate new stopping distance and time
        new_stopping_dist = self.v0**2 / (2 * perturbed_a_mag)
        new_stopping_time = self.v0 / perturbed_a_mag
        
        # Check that stopping distance decreases with increased deceleration
        if not (new_stopping_dist < self.x):
            self._log(f"Monotonicity test 1 failed: increased deceleration should decrease stopping distance", False)
            success = False
        else:
            self._log(f"Monotonicity test 1 passed: stopping distance decreases with increased deceleration", True)
        
        # Check that stopping time decreases with increased deceleration
        if not (new_stopping_time < self.t_0):
            self._log(f"Monotonicity test 2 failed: increased deceleration should decrease stopping time", False)
            success = False
        else:
            self._log(f"Monotonicity test 2 passed: stopping time decreases with increased deceleration", True)
        
        # Test 2: Decrease deceleration magnitude by 10%
        perturbed_a_mag = original_a_mag * 0.90
        new_stopping_dist = self.v0**2 / (2 * perturbed_a_mag)
        new_stopping_time = self.v0 / perturbed_a_mag
        
        # Check that stopping distance increases with decreased deceleration
        if not (new_stopping_dist > self.x):
            self._log(f"Monotonicity test 3 failed: decreased deceleration should increase stopping distance", False)
            success = False
        else:
            self._log(f"Monotonicity test 3 passed: stopping distance increases with decreased deceleration", True)
        
        # Check that stopping time increases with decreased deceleration
        if not (new_stopping_time > self.t_0):
            self._log(f"Monotonicity test 4 failed: decreased deceleration should increase stopping time", False)
            success = False
        else:
            self._log(f"Monotonicity test 4 passed: stopping time increases with decreased deceleration", True)
        
        return success
    
    def run_all_tests(self):
        """
        Execute all physics verification tests and return overall success.
        """
        self.logs = []  # Reset logs
        
        overall_success = True
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for test_name, test_func in tests:
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"{test_name} test crashed: {str(e)}", False)
                overall_success = False
        
        return overall_success, self.logs