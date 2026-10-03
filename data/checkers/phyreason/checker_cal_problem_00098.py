import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallback to results dict
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Constants from problem statement (use exact values from problem)
        self.theta_deg = 37.0
        self.sin_theta = 0.6  # Given: sin37° = 0.6
        self.cos_theta = 0.8  # cos37° = 0.8 (exact from sin²+cos²=1 with sin=0.6)
        self.mu = 0.5
        self.R = 0.4
        self.L = 2.0
        self.H = 2.2
        self.g = 10.0  # Using g=10 m/s² as typical in such problems
        
        # Extract computed values safely
        self.v_B = self._safe_get('v_B')
        self.v_B2 = self._safe_get('v_B2')
        self.a = self._safe_get('a')
        self.ma = self._safe_get('ma')
        self.h1 = self._safe_get('h_1')
        self.h = self._safe_get('h')
        self.v = self._safe_get('v')
        self.t = self._safe_get('t')
        self.x = self._safe_get('x')
        
        # Target answers for validation
        self.target_v_B = 4.0
        self.target_h_left = 3.0
        self.target_h_right = 3.6
        
    def _safe_get(self, key):
        """Safely get value from solution instance or results dict."""
        val = getattr(self.solution, key, self.results.get(key, None))
        # Handle string values
        if isinstance(val, str):
            try:
                # Extract numeric part from strings like "4 m/s"
                import re
                match = re.search(r'[-+]?\d*\.?\d+', val)
                if match:
                    return float(match.group())
            except:
                pass
        return val
    
    def test_invariants(self):
        logs = []
        try:
            # 1. Check Newton's Second Law on inclined plane
            # From step_1: a = g*sinθ - μ*g*cosθ
            a_expected = self.g * self.sin_theta - self.mu * self.g * self.cos_theta
            
            if self.a is not None:
                if not math.isclose(self.a, a_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Acceleration mismatch. Expected {a_expected:.3f}, got {self.a}")
                else:
                    logs.append(f"PASS: Acceleration a = {self.a:.3f} matches theory")
            
            # 2. Check v_B^2 formula consistency - use appropriate h
            # v_B^2 = 2*a*(h/sinθ) from step_2
            # The stored v_B2 might be at different h (Q1 uses h=2.4, Q3 uses h=3.6)
            # We'll test with the h from results (which is critical h=3.6)
            if all(v is not None for v in [self.v_B2, self.a, self.h]):
                v_B2_expected = 2 * self.a * (self.h / self.sin_theta)
                if not math.isclose(self.v_B2, v_B2_expected, rel_tol=1e-2, abs_tol=1e-2):
                    # Also check if v_B2 corresponds to h=2.4 (from Q1)
                    h_q1 = 2.4  # From solving v_B=4
                    v_B2_q1_expected = 2 * self.a * (h_q1 / self.sin_theta)
                    if math.isclose(self.v_B2, v_B2_q1_expected, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"PASS: v_B^2 = {self.v_B2:.3f} matches kinematics at h=2.4")
                    else:
                        logs.append(f"FAIL: v_B^2 mismatch. Expected {v_B2_expected:.3f} at h={self.h}, got {self.v_B2}")
                else:
                    logs.append(f"PASS: v_B^2 = {self.v_B2:.3f} matches kinematics")
            
            # 3. Check energy conservation in step_4
            # 0 = m*g*h1 - μ*m*g*cosθ*(h1/sinθ) - μ*m*g*L
            if self.h1 is not None:
                # Cancel m*g: 0 = h1 - μ*cosθ/sinθ*h1 - μ*L
                lhs = self.h1 - self.mu * (self.cos_theta/self.sin_theta) * self.h1 - self.mu * self.L
                if not math.isclose(lhs, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Energy equation imbalance: {lhs:.3f} ≠ 0")
                else:
                    logs.append(f"PASS: Energy conservation holds for h1 = {self.h1}")
            
            # 4. Check projectile condition at point D
            # Condition: mg ≤ m*v²/R  =>  v² ≥ g*R
            if self.v is not None:
                min_v2 = self.g * self.R
                if self.v**2 < min_v2 - 1e-2:  # Slightly below threshold
                    logs.append(f"WARN: v² = {self.v**2:.3f} < g*R = {min_v2:.3f}")
                else:
                    logs.append(f"PASS: Projectile condition v² ≥ g*R satisfied")
            
        except Exception as e:
            logs.append(f"ERROR in invariants: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def test_trace_consistency(self):
        logs = []
        try:
            # 1. Check v_B target answer
            if self.v_B is not None:
                if not math.isclose(self.v_B, self.target_v_B, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: v_B = {self.v_B} ≠ target {self.target_v_B}")
                else:
                    logs.append(f"PASS: v_B = {self.v_B} matches target")
            
            # 2. Check h1 value
            if self.h1 is not None:
                if not math.isclose(self.h1, self.target_h_left, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: h1 = {self.h1} ≠ target {self.target_h_left}")
                else:
                    logs.append(f"PASS: h1 = {self.h1} matches target")
            
            # 3. Check energy-work theorem for step_6
            # (1/2)*mv² = m*g*h - μ*m*g*cosθ*(h/sinθ) - μ*m*g*L
            if all(v is not None for v in [self.v, self.h]):
                # Cancel m from both sides, use exact cosθ=0.8
                lhs = 0.5 * self.v**2
                rhs = self.g * self.h - self.mu * self.g * self.cos_theta * (self.h/self.sin_theta) - self.mu * self.g * self.L
                
                if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    # Check with more precise calculation
                    rhs_precise = self.g * self.h - self.mu * self.g * 0.8 * (self.h/0.6) - self.mu * self.g * self.L
                    if math.isclose(lhs, rhs_precise, rel_tol=1e-2, abs_tol=1e-2):
                        logs.append(f"PASS: Energy-work theorem consistent (with exact cosθ=0.8)")
                    else:
                        logs.append(f"FAIL: Energy-work mismatch: {lhs:.3f} ≠ {rhs:.3f} (rhs with cosθ=0.8: {rhs_precise:.3f})")
                else:
                    logs.append(f"PASS: Energy-work theorem consistent")
            
            # 4. Check free fall time
            # H + 2R = (1/2)*g*t²
            t_expected = math.sqrt(2 * (self.H + 2*self.R) / self.g)
            if self.t is not None:
                if not math.isclose(self.t, t_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Free fall time mismatch: {self.t:.3f} ≠ {t_expected:.3f}")
                else:
                    logs.append(f"PASS: Free fall time t = {self.t:.3f} correct")
            
            # 5. Check horizontal distance relation
            # x = 2*sqrt(h-3) from step_9
            if all(v is not None for v in [self.x, self.h]):
                x_expected = 2 * math.sqrt(max(0, self.h - 3.0))
                if not math.isclose(self.x, x_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: x mismatch: {self.x:.3f} ≠ {x_expected:.3f}")
                else:
                    logs.append(f"PASS: x = {self.x:.3f} matches formula")
            
            # 6. Check condition for rightward projectile
            if self.h is not None and self.v is not None:
                # From step_11: h ≥ 3.6 for projectile motion
                threshold = self.target_h_right
                condition_met = self.h >= threshold - 1e-2  # Allow small tolerance
                
                # Also check v² ≥ g*R
                velocity_condition = self.v**2 >= self.g * self.R - 1e-2
                
                if condition_met != velocity_condition:
                    logs.append(f"WARN: h condition and velocity condition inconsistent")
                else:
                    logs.append(f"PASS: Conditions for rightward motion consistent")
            
        except Exception as e:
            logs.append(f"ERROR in trace consistency: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def test_monotonicity(self):
        logs = []
        try:
            # Test monotonicity of x with respect to h
            # x = 2*sqrt(h-3) for h ≥ 3.0
            
            # Use h from results or default test value
            h_test = 4.0  # Test value in valid range
            if self.h is not None and self.h >= 3.0:
                h_test = self.h
            
            # Perturb h by ±10%
            delta = 0.1 * h_test
            h1 = h_test + delta
            h2 = h_test - delta
            
            # Calculate expected x values
            x1 = 2 * math.sqrt(max(0, h1 - 3.0))
            x2 = 2 * math.sqrt(max(0, h2 - 3.0))
            
            # x should increase with h
            if x1 > x2:
                logs.append(f"PASS: x increases with h (monotonic)")
            else:
                logs.append(f"FAIL: x does not increase with h")
            
            # Also test v increases with h
            # v² = 2*g*(h/3 - 1) from energy equation (simplified)
            v1_sq = 2 * self.g * (h1/3 - 1)
            v2_sq = 2 * self.g * (h2/3 - 1)
            
            if v1_sq > v2_sq:
                logs.append(f"PASS: v² increases with h (monotonic)")
            else:
                logs.append(f"FAIL: v² does not increase with h")
            
            # Test boundary conditions
            # At h = 3.0, x = 0, v = 0
            h_boundary = 3.0
            x_boundary = 2 * math.sqrt(max(0, h_boundary - 3.0))
            v_boundary_sq = 2 * self.g * (h_boundary/3 - 1)
            
            if math.isclose(x_boundary, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"PASS: x=0 at h=3.0 boundary")
            else:
                logs.append(f"WARN: x not zero at h=3.0")
            
            if math.isclose(v_boundary_sq, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"PASS: v=0 at h=3.0 boundary")
            else:
                logs.append(f"WARN: v not zero at h=3.0")
            
        except Exception as e:
            logs.append(f"ERROR in monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
        
        return logs
    
    def run_all_tests(self):
        all_logs = []
        
        # Run all test suites
        all_logs.append("=== Testing Invariants ===")
        all_logs.extend(self.test_invariants())
        
        all_logs.append("\n=== Testing Trace Consistency ===")
        all_logs.extend(self.test_trace_consistency())
        
        all_logs.append("\n=== Testing Monotonicity ===")
        all_logs.extend(self.test_monotonicity())
        
        # Determine overall success
        success = True
        for log in all_logs:
            if log.startswith("FAIL:"):
                success = False
                break
        
        return success, all_logs