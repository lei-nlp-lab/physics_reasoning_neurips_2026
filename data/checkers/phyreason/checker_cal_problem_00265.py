import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute extraction using getattr with fallback to results dictionary
        self.solution = solution_instance
        
        # Inputs (with safe defaults)
        self.v0 = 15.0  # m/s (given in problem)
        self.a_mag = 10.0  # m/s^2 (given magnitude)
        self.a = -self.a_mag  # actual acceleration (upward positive, so negative)
        self.x1 = 6.0  # m (displacement in last second)
        self.delta_t_last = 1.0  # s (last second interval)
        
        # Results (extracted safely) - using the exact contract symbol names
        self.t_total = self._get_result('t\'', 0.0)  # Note: escaped apostrophe
        self.v2 = self._get_result('v_2', 0.0)
        self.barv = self._get_result('barv', 0.0)
        self.h = self._get_result('h', 0.0)
        self.v1 = self._get_result('v_1', 0.0)
        
    def _get_result(self, key, default):
        """Safely extract result from solution instance."""
        # Try results dictionary first
        if hasattr(self.solution, 'results'):
            # Access the results dictionary directly
            results_dict = self.solution.results
            if isinstance(results_dict, dict):
                # Try the exact key first
                if key in results_dict:
                    value = results_dict[key]
                    return float(value) if value is not None else default
                
                # If not found, try alternative names for t_prime
                if key == 't\'' and 't_prime' in results_dict:
                    value = results_dict['t_prime']
                    return float(value) if value is not None else default
                    
        # Try direct attribute
        try:
            return float(getattr(self.solution, key, default))
        except (AttributeError, TypeError, ValueError):
            return default
    
    def _is_close(self, a, b):
        """Robust fuzzy comparison with 1% tolerance."""
        if a == 0 and b == 0:
            return True
        if a == 0 or b == 0:
            return math.isclose(a, b, abs_tol=0.02)
        return math.isclose(a, b, rel_tol=0.02, abs_tol=0.02)
    
    def test_invariants(self):
        """Verify fundamental physics laws."""
        logs = []
        
        # 1. Average velocity in last second
        barv_calc = self.x1 / self.delta_t_last
        if not self._is_close(self.barv, barv_calc):
            logs.append(f"FAIL: Average velocity mismatch: {self.barv} vs {barv_calc}")
        
        # 2. Midpoint velocity theorem (v_avg = v_mid for constant acceleration)
        # v1 should equal barv (velocity at midpoint of last second)
        if not self._is_close(self.v1, self.barv):
            logs.append(f"FAIL: Midpoint velocity theorem: v1={self.v1}, barv={self.barv}")
        
        # 3. Velocity-time relation in last second
        # v2 = v1 + a * (delta_t_last/2) since v1 is at midpoint
        v2_calc = self.v1 + self.a * (self.delta_t_last / 2.0)
        if not self._is_close(self.v2, v2_calc):
            logs.append(f"FAIL: Velocity-time in last second: v2={self.v2}, calc={v2_calc}")
        
        # 4. Total time from v0 to v2
        if self.a == 0:
            logs.append("ERROR: Acceleration is zero, cannot compute total time")
            return False, logs
            
        t_total_calc = (self.v2 - self.v0) / self.a
        if not self._is_close(self.t_total, t_total_calc):
            logs.append(f"FAIL: Total time: {self.t_total} vs {t_total_calc}")
        
        # 5. Velocity-displacement relation for entire motion
        if 2.0 * self.a == 0:
            logs.append("ERROR: 2*a is zero, cannot compute well depth")
            return False, logs
            
        h_calc = (self.v2**2 - self.v0**2) / (2.0 * self.a)
        if not self._is_close(self.h, h_calc):
            logs.append(f"FAIL: Well depth: {self.h} vs {h_calc}")
        
        # 6. Verify displacement in last second directly from kinematics
        s_t_total = self.v0 * self.t_total + 0.5 * self.a * self.t_total**2
        s_t_total_minus_1 = self.v0 * (self.t_total - 1.0) + 0.5 * self.a * (self.t_total - 1.0)**2
        x1_calc = s_t_total - s_t_total_minus_1
        if not self._is_close(x1_calc, self.x1):
            logs.append(f"FAIL: Last second displacement from kinematics: {x1_calc} vs given {self.x1}")
        
        return len(logs) == 0, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer."""
        logs = []
        
        # Expected values from steps analysis
        expected = {
            'barv': 6.0,
            'v1': 6.0,
            'v2': 1.0,
            't_total': 1.4,
            'h': 11.2
        }
        
        for key, exp_val in expected.items():
            actual = getattr(self, key)
            if not self._is_close(actual, exp_val):
                logs.append(f"TRACE MISMATCH: {key}: {actual} vs expected {exp_val}")
        
        # Additional consistency: v2 should be positive (caught on way up)
        if self.v2 <= 0:
            logs.append("WARNING: v2 should be positive (caught on ascent)")
        
        return len(logs) == 0, logs
    
    def test_monotonicity(self):
        """Perturb input and check output trend - CORRECTED PHYSICS."""
        logs = []
        original_v0 = self.v0
        
        try:
            # Perturb initial velocity by 1%
            perturbed_v0 = original_v0 * 1.01
            
            # CORRECTED: Solve the displacement equation properly for new t_total
            # Equation: s(t) - s(t-1) = x1
            # where s(t) = v0*t + 0.5*a*t^2
            # This simplifies to: v0 + a*(t - 0.5) = x1
            # So: t = (x1 - v0)/a + 0.5
            if self.a == 0:
                logs.append("ERROR: Acceleration is zero, cannot test monotonicity")
                return False, logs
                
            t_total_new = (self.x1 - perturbed_v0) / self.a + 0.5
            v2_new = perturbed_v0 + self.a * t_total_new
            
            if 2.0 * self.a == 0:
                logs.append("ERROR: 2*a is zero, cannot compute new well depth")
                return False, logs
                
            h_new = (v2_new**2 - perturbed_v0**2) / (2.0 * self.a)
            
            # PHYSICS CORRECTION: For this specific problem, increasing v0 actually INCREASES t_total
            # because the object goes higher and takes longer to return to the opening.
            # The relationship t = (x1 - v0)/a + 0.5 shows that since a is negative (-10),
            # increasing v0 makes (x1 - v0) more negative, and dividing by negative a gives
            # a larger positive value. So t_total should INCREASE with v0.
            if not (t_total_new > self.t_total):
                logs.append(f"MONOTONICITY WARNING: t_total({perturbed_v0})={t_total_new} not > t_total({original_v0})={self.t_total}")
            
            # Check h increases with v0 (deeper well)
            if not (h_new > self.h):
                logs.append(f"MONOTONICITY WARNING: h({perturbed_v0})={h_new} not > h({original_v0})={self.h}")
                
        except Exception as e:
            logs.append(f"Monotonicity test failed: {str(e)}")
            logs.append(traceback.format_exc())
        
        return len(logs) == 0, logs
    
    def run_all_tests(self):
        """Execute all tests and return overall success."""
        all_logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                test_success, logs = test_func()
                if not test_success:
                    success = False
                    all_logs.append(f"--- {test_name} Failed ---")
                else:
                    all_logs.append(f"✓ {test_name} Passed")
                all_logs.extend(logs)
            except Exception as e:
                success = False
                all_logs.append(f"ERROR in {test_name}: {str(e)}")
                all_logs.append(traceback.format_exc())
        
        return success, all_logs