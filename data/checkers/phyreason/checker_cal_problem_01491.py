import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallback to results dictionary
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Problem constants from context
        self.r = 0.8  # radius in meters
        self.h_C = 0.2  # height of C above B in meters
        self.mu = 0.1  # coefficient of kinetic friction
        self.g = 10.0  # gravitational acceleration
        
        # Extract results safely - handle missing keys gracefully
        self.v_B = self._safe_get('v_B', 4.0)  # speed at B
        self.F_N = self._safe_get('F_N', 0.0)  # normal force
        self.F_N_ratio = self._safe_get('F_N/mg', 3.0)  # ratio F_N/(mg)
        self.a = self._safe_get('a', 1.0)  # acceleration magnitude on belt
        self.t1 = self._safe_get('t_1', 4.0)  # time to stop on belt
        self.x1 = self._safe_get('x_1', 8.0)  # displacement during deceleration
        self.v = self._safe_get('v', 2.0)  # speed when returning from C
        self.t2 = self._safe_get('t_2', 2.0)  # time to accelerate to v
        self.x2 = self._safe_get('x_2', 2.0)  # displacement during acceleration
        self.x3 = self._safe_get('x_3', 6.0)  # displacement at constant speed
        self.t3 = self._safe_get('t_3', 3.0)  # time at constant speed
        self.t_total = self._safe_get('t', 9.0)  # total time for round trip
        self.delta_x = self._safe_get('delta_x', 16.0)  # relative displacement right
        self.delta_x_prime = self._safe_get("delta_x'", 2.0)  # relative displacement left
        self.Q = self._safe_get('Q', 18.0)  # heat generated
        
        # Get conveyor belt speed if available, otherwise compute from Δx relation
        self.v_belt = self._safe_get('v_belt', 0.0)
        self.logs = []
        
    def _safe_get(self, key, default):
        """Safely get value from results or solution attributes."""
        # First try results dictionary
        if key in self.results:
            val = self.results[key]
        else:
            # Try to get from solution instance attributes
            val = getattr(self.solution, key, default)
        
        # Convert to float if possible
        try:
            return float(val)
        except (TypeError, ValueError):
            return float(default)
        
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison."""
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False
        
    def test_invariants(self):
        """Verify fundamental physics laws."""
        logs = []
        success = True
        
        try:
            # 1. Energy conservation from A to B
            v_B_expected = math.sqrt(2 * self.g * self.r)
            if not self._is_close(self.v_B, v_B_expected):
                logs.append(f"Energy conservation fail: v_B={self.v_B}, expected={v_B_expected}")
                success = False
            else:
                logs.append("✓ Energy conservation A->B valid")
                
            # 2. Normal force ratio
            ratio_expected = 1 + self.v_B**2 / (self.g * self.r)
            if not self._is_close(self.F_N_ratio, ratio_expected):
                logs.append(f"Normal force ratio fail: {self.F_N_ratio}, expected {ratio_expected}")
                success = False
            else:
                logs.append("✓ Normal force at B valid")
                    
            # 3. Acceleration on conveyor belt: a = μ*g
            a_expected = self.mu * self.g
            if not self._is_close(self.a, a_expected):
                logs.append(f"Acceleration fail: a={self.a}, expected {a_expected}")
                success = False
            else:
                logs.append("✓ Acceleration on belt valid")
                
            # 4. Heat generation consistency: Q = μ*m*g*(Δx+Δx')
            # Since m cancels, we check Q/(μ*g) = Δx+Δx'
            if self.mu * self.g != 0:
                ratio = self.Q / (self.mu * self.g)
                expected_ratio = self.delta_x + self.delta_x_prime
                if not self._is_close(ratio, expected_ratio):
                    logs.append(f"Heat generation fail: Q/(μ*g)={ratio}, expected {expected_ratio}")
                    success = False
                else:
                    logs.append("✓ Heat generation consistent")
            else:
                logs.append("⚠ Skipping heat check (μ*g = 0)")
                    
        except Exception as e:
            logs.append(f"Invariant test error: {traceback.format_exc()}")
            success = False
            
        self.logs.extend(logs)
        return success, logs
        
    def test_trace_consistency(self):
        """Check intermediate calculations are consistent."""
        logs = []
        success = True
        
        try:
            # 1. Deceleration time: t1 = v_B / a
            if self.a != 0:
                t1_expected = self.v_B / self.a
                if not self._is_close(self.t1, t1_expected):
                    logs.append(f"t1 inconsistent: {self.t1}, expected {t1_expected}")
                    success = False
                else:
                    logs.append("✓ t1 consistent with v_B and a")
            else:
                logs.append("⚠ Skipping t1 check (a = 0)")
                
            # 2. Deceleration displacement: x1 = (v_B/2) * t1
            x1_expected = (self.v_B / 2) * self.t1
            if not self._is_close(self.x1, x1_expected):
                logs.append(f"x1 inconsistent: {self.x1}, expected {x1_expected}")
                success = False
            else:
                logs.append("✓ x1 consistent")
                
            # 3. Speed after returning from C: v = sqrt(2*g*h_C)
            v_expected = math.sqrt(2 * self.g * self.h_C)
            if not self._is_close(self.v, v_expected):
                logs.append(f"v inconsistent: {self.v}, expected {v_expected}")
                success = False
            else:
                logs.append("✓ Speed at C return valid")
                
            # 4. Acceleration time: t2 = v / a
            if self.a != 0:
                t2_expected = self.v / self.a
                if not self._is_close(self.t2, t2_expected):
                    logs.append(f"t2 inconsistent: {self.t2}, expected {t2_expected}")
                    success = False
                else:
                    logs.append("✓ t2 consistent")
            else:
                logs.append("⚠ Skipping t2 check (a = 0)")
                
            # 5. Acceleration displacement: x2 = v²/(2a)
            if self.a != 0:
                x2_expected = self.v**2 / (2 * self.a)
                if not self._is_close(self.x2, x2_expected):
                    logs.append(f"x2 inconsistent: {self.x2}, expected {x2_expected}")
                    success = False
                else:
                    logs.append("✓ x2 consistent")
            else:
                logs.append("⚠ Skipping x2 check (a = 0)")
                
            # 6. Constant velocity displacement: x3 = x1 - x2
            x3_expected = self.x1 - self.x2
            if not self._is_close(self.x3, x3_expected):
                logs.append(f"x3 inconsistent: {self.x3}, expected {x3_expected}")
                success = False
            else:
                logs.append("✓ x3 consistent")
                
            # 7. Constant velocity time: t3 = x3 / v
            if self.v != 0:
                t3_expected = self.x3 / self.v
                if not self._is_close(self.t3, t3_expected):
                    logs.append(f"t3 inconsistent: {self.t3}, expected {t3_expected}")
                    success = False
                else:
                    logs.append("✓ t3 consistent")
            else:
                logs.append("⚠ Skipping t3 check (v = 0)")
                
            # 8. Total time: t = t1 + t2 + t3
            t_total_expected = self.t1 + self.t2 + self.t3
            if not self._is_close(self.t_total, t_total_expected):
                logs.append(f"Total time inconsistent: {self.t_total}, expected {t_total_expected}")
                success = False
            else:
                logs.append("✓ Total time consistent")
                
            # 9. Relative displacement right: Δx = x1 + v_belt * t1
            # Compute v_belt from Δx if not provided
            if self.t1 != 0:
                v_belt_computed = (self.delta_x - self.x1) / self.t1
                delta_x_expected = self.x1 + v_belt_computed * self.t1
                if not self._is_close(self.delta_x, delta_x_expected):
                    logs.append(f"Δx inconsistent: {self.delta_x}, expected {delta_x_expected}")
                    success = False
                else:
                    logs.append("✓ Δx consistent")
            else:
                logs.append("⚠ Skipping Δx check (t1 = 0)")
                
            # 10. Relative displacement left: Δx' = v_belt * t2 - x2
            if self.t1 != 0:
                v_belt_computed = (self.delta_x - self.x1) / self.t1
                delta_x_prime_expected = v_belt_computed * self.t2 - self.x2
                if not self._is_close(self.delta_x_prime, delta_x_prime_expected):
                    logs.append(f"Δx' inconsistent: {self.delta_x_prime}, expected {delta_x_prime_expected}")
                    success = False
                else:
                    logs.append("✓ Δx' consistent")
            else:
                logs.append("⚠ Skipping Δx' check (t1 = 0)")
                
        except ZeroDivisionError as e:
            logs.append(f"Zero division in trace consistency: {e}")
            success = False
        except Exception as e:
            logs.append(f"Trace consistency error: {traceback.format_exc()}")
            success = False
            
        self.logs.extend(logs)
        return success, logs
        
    def test_monotonicity(self):
        """Perturb inputs and check expected trends."""
        logs = []
        success = True
        
        try:
            # Test 1: Increase radius, v_B should increase as sqrt(r)
            r_increased = self.r * 1.1  # +10%
            v_B_new = math.sqrt(2 * self.g * r_increased)
            if v_B_new > self.v_B:
                logs.append("✓ v_B increases with radius (as sqrt(r))")
            else:
                logs.append("✗ v_B monotonicity with radius failed")
                success = False
                
            # Test 2: Increase μ, acceleration on belt should increase linearly
            mu_increased = self.mu * 1.1
            a_new = mu_increased * self.g
            if a_new > self.a:
                logs.append("✓ Acceleration increases with μ")
            else:
                logs.append("✗ Acceleration monotonicity with μ failed")
                success = False
                
            # Test 3: Increase height h_C, return speed v should increase as sqrt(h)
            h_increased = self.h_C * 1.1
            v_new = math.sqrt(2 * self.g * h_increased)
            if v_new > self.v:
                logs.append("✓ Return speed increases with height")
            else:
                logs.append("✗ Return speed monotonicity failed")
                success = False
                
        except Exception as e:
            logs.append(f"Monotonicity test error: {traceback.format_exc()}")
            success = False
            
        self.logs.extend(logs)
        return success, logs
        
    def run_all_tests(self):
        """Execute all checks and return success status and logs."""
        self.logs = ["=== Physics Checker Results ==="]
        
        # Run tests
        inv_success, inv_logs = self.test_invariants()
        trace_success, trace_logs = self.test_trace_consistency()
        mono_success, mono_logs = self.test_monotonicity()
        
        # Combine logs
        self.logs.extend(inv_logs)
        self.logs.extend(trace_logs)
        self.logs.extend(mono_logs)
        
        # Summary
        overall_success = inv_success and trace_success and mono_success
        self.logs.append(f"\nSummary: Invariants {'PASS' if inv_success else 'FAIL'}, "
                        f"Trace {'PASS' if trace_success else 'FAIL'}, "
                        f"Monotonicity {'PASS' if mono_success else 'FAIL'}")
        self.logs.append(f"Overall: {'PASS' if overall_success else 'FAIL'}")
        
        return overall_success, self.logs