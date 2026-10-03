import math
import traceback
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access: use getattr with fallback to results dict
        self.results = {}
        try:
            # Try to get results from solution_instance.results if it exists
            if hasattr(solution_instance, 'results'):
                self.results = solution_instance.results
            else:
                # Otherwise, try to extract attributes directly
                self.results = {
                    'a_1': getattr(solution_instance, 'a_1', None),
                    't_2': getattr(solution_instance, 't_2', None),
                    't': getattr(solution_instance, 't', None),
                    'v_C': getattr(solution_instance, 'v_C', None),
                    'a_2': getattr(solution_instance, 'a_2', None),
                    'a_4': getattr(solution_instance, 'a_4', None),
                    'v_B': getattr(solution_instance, 'v_B', None),
                    't_1': getattr(solution_instance, 't_1', None),
                    'a_3': getattr(solution_instance, 'a_3', None),
                    'x_1': getattr(solution_instance, 'x_1', None),
                }
        except Exception as e:
            self.results = {}
            print(f"Warning: Could not extract results: {e}")
        
        # Physical constants from the problem
        self.m = 1.0  # kg
        self.F = 7.0  # N
        self.L = 8.0  # m
        self.v_conv = 10.0  # m/s, conveyor speed
        self.mu = 0.5
        self.sin37 = 0.6
        self.cos37 = 0.8
        self.g = 10.0  # m/s^2
        
        # Expected symbolic/numerical values from KB and steps
        self.expected = {
            'a_1': 1.0,
            'v_B': 4.0,
            'a_2': 5.0,
            'v_C': 0.0,
            'a_3': 10.0,
            'x_1': 5.0,
            't_1': 1.0,
            'a_4': 2.0,
        }
        # t_2 and t are symbolic: (2*sqrt(7)-5) and (2*sqrt(7)-4)
        self.expected['t_2'] = 2*math.sqrt(7) - 5
        self.expected['t'] = 2*math.sqrt(7) - 4
        
        self.logs = []
    
    def _log(self, message):
        self.logs.append(message)
    
    def _is_close(self, a, b):
        """Fuzzy numerical matching with 2% tolerance."""
        return math.isclose(a, b, rel_tol=0.02, abs_tol=0.02)
    
    def _check_newton_second(self, acceleration, net_force_expr, step_name):
        """Check Newton's Second Law symbolically and numerically."""
        m, g, sin37, cos37, mu, F = sp.symbols('m g sin37 cos37 mu F')
        # Substitute known values
        subs_dict = {
            m: self.m,
            g: self.g,
            sin37: self.sin37,
            cos37: self.cos37,
            mu: self.mu,
            F: self.F
        }
        try:
            # Compute expected acceleration from net force equation
            net_force_val = net_force_expr.subs(subs_dict)
            a_expected = net_force_val / self.m
            a_given = acceleration
            if not self._is_close(float(a_expected), float(a_given)):
                self._log(f"FAIL {step_name}: Newton's Second Law mismatch. Expected a={float(a_expected):.3f}, got {float(a_given):.3f}")
                return False
            self._log(f"PASS {step_name}: Newton's Second Law holds.")
            return True
        except Exception as e:
            self._log(f"ERROR in {step_name}: {e}")
            return False
    
    def _check_kinematics(self, v0, v1, a, displacement, step_name):
        """Check kinematic equations v1^2 = v0^2 + 2*a*displacement."""
        try:
            left = v1**2
            right = v0**2 + 2*a*displacement
            if not self._is_close(float(left), float(right)):
                self._log(f"FAIL {step_name}: Kinematics mismatch. v1^2={float(left):.3f}, v0^2+2*a*x={float(right):.3f}")
                return False
            self._log(f"PASS {step_name}: Kinematics holds.")
            return True
        except Exception as e:
            self._log(f"ERROR in {step_name}: {e}")
            return False
    
    def test_invariants(self):
        """Verify physical invariants: Newton's Laws and kinematics from steps."""
        self._log("=== Testing Invariants ===")
        success = True
        
        # Extract results with safe defaults
        a_1 = self.results.get('a_1', 0.0)
        v_B = self.results.get('v_B', 0.0)
        a_2 = self.results.get('a_2', 0.0)
        v_C = self.results.get('v_C', 0.0)
        a_3 = self.results.get('a_3', 0.0)
        x_1 = self.results.get('x_1', 0.0)
        t_1 = self.results.get('t_1', 0.0)
        a_4 = self.results.get('a_4', 0.0)
        t_2 = self.results.get('t_2', 0.0)
        t_total = self.results.get('t', 0.0)
        
        # Step 1: A to B incline (Newton's Second)
        m, g, sin37, F = sp.symbols('m g sin37 F')
        net_force_1 = F - m*g*sin37
        if not self._check_newton_second(a_1, net_force_1, "Step 1 (AB incline)"):
            success = False
        
        # Step 2: Kinematics from A to B
        if not self._check_kinematics(0.0, v_B, a_1, self.L, "Step 2 (AB kinematics)"):
            success = False
        
        # Step 3: BC horizontal (Newton's Second)
        mu, cos37 = sp.symbols('mu cos37')
        net_force_2 = mu * m * g  # friction force magnitude
        if not self._check_newton_second(a_2, net_force_2, "Step 3 (BC horizontal)"):
            success = False
        
        # Step 4: Kinematics on BC (until stop at C)
        # v_C should be 0, so check v_C^2 = v_B^2 - 2*a_2*x (deceleration)
        # But distance x_BC is not given, so we check consistency: if v_C=0, then x_BC = v_B^2/(2*a_2)
        if not self._is_close(v_C, 0.0):
            self._log(f"FAIL Step 4: v_C should be 0, got {v_C:.3f}")
            success = False
        else:
            # Verify that with given v_B and a_2, the distance to stop is positive
            x_BC = v_B**2 / (2*a_2) if a_2 != 0 else 0
            if x_BC < 0:
                self._log(f"FAIL Step 4: Calculated distance x_BC={x_BC:.3f} is negative.")
                success = False
            else:
                self._log(f"PASS Step 4: v_C=0 consistent with deceleration a_2.")
        
        # Step 5: CD before reaching conveyor speed (Newton's Second)
        net_force_3 = m*g*sin37 + mu*m*g*cos37  # gravity + friction down incline
        if not self._check_newton_second(a_3, net_force_3, "Step 5 (CD before v_conv)"):
            success = False
        
        # Step 6: Kinematics on CD to reach conveyor speed
        if not self._check_kinematics(0.0, self.v_conv, a_3, x_1, "Step 6 (CD to v_conv)"):
            success = False
        
        # Step 7: Time to reach conveyor speed (kinematics)
        try:
            t_1_expected = self.v_conv / a_3
            if not self._is_close(t_1, t_1_expected):
                self._log(f"FAIL Step 7: t_1 mismatch. Expected {t_1_expected:.3f}, got {t_1:.3f}")
                success = False
            else:
                self._log(f"PASS Step 7: t_1 = v_conv / a_3 holds.")
        except ZeroDivisionError:
            self._log("ERROR Step 7: Division by zero in t_1 calculation.")
            success = False
        
        # Step 8: CD after reaching conveyor speed (Newton's Second)
        net_force_4 = m*g*sin37 - mu*m*g*cos37  # gravity down, friction up
        if not self._check_newton_second(a_4, net_force_4, "Step 8 (CD after v_conv)"):
            success = False
        
        # Step 9 and 10: Time addition and kinematics on remaining CD
        # We don't have the length of CD, so we cannot fully check step 9.
        # But we can check that t_total = t_1 + t_2
        if not self._is_close(t_total, t_1 + t_2):
            self._log(f"FAIL Step 10: t_total != t_1 + t_2. {t_total:.3f} vs {t_1+t_2:.3f}")
            success = False
        else:
            self._log(f"PASS Step 10: t = t_1 + t_2 holds.")
        
        return success
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers and expected values."""
        self._log("=== Testing Trace Consistency ===")
        success = True
        
        # Check each result against expected values from KB
        for key, expected_val in self.expected.items():
            val = self.results.get(key, None)
            if val is None:
                self._log(f"FAIL: {key} not found in results.")
                success = False
            elif not self._is_close(float(val), float(expected_val)):
                self._log(f"FAIL: {key} mismatch. Expected {expected_val:.3f}, got {float(val):.3f}")
                success = False
            else:
                self._log(f"PASS: {key} matches expected.")
        
        # Additional consistency: v_B should be 4 m/s, v_C should be 0
        v_B = self.results.get('v_B', None)
        v_C = self.results.get('v_C', None)
        if v_B is not None and not self._is_close(float(v_B), 4.0):
            self._log(f"FAIL: v_B should be 4 m/s, got {float(v_B):.3f}")
            success = False
        if v_C is not None and not self._is_close(float(v_C), 0.0):
            self._log(f"FAIL: v_C should be 0 m/s, got {float(v_C):.3f}")
            success = False
        
        return success
    
    def test_monotonicity(self):
        """Perturb mass m and check trend in v_B and t."""
        self._log("=== Testing Monotonicity ===")
        success = True
        
        # Store original mass
        m_original = self.m
        
        # Perturb mass by +10%
        self.m = m_original * 1.1
        
        # Recalculate expected v_B using formulas from steps
        a_1_perturbed = (self.F - self.m * self.g * self.sin37) / self.m
        v_B_perturbed = math.sqrt(2 * a_1_perturbed * self.L)
        
        # Restore original mass
        self.m = m_original
        
        # Get original v_B from results
        v_B_original = self.results.get('v_B', 0.0)
        
        # Check trend: increasing mass should decrease v_B (since a_1 decreases)
        if v_B_perturbed < v_B_original:
            self._log(f"PASS: v_B decreases when mass increases (from {v_B_original:.3f} to {v_B_perturbed:.3f}).")
        else:
            self._log(f"FAIL: v_B did not decrease as expected. Original {v_B_original:.3f}, perturbed {v_B_perturbed:.3f}")
            success = False
        
        # Also perturb mass downward to check opposite trend
        self.m = m_original * 0.9
        a_1_perturbed2 = (self.F - self.m * self.g * self.sin37) / self.m
        v_B_perturbed2 = math.sqrt(2 * a_1_perturbed2 * self.L)
        self.m = m_original
        
        if v_B_perturbed2 > v_B_original:
            self._log(f"PASS: v_B increases when mass decreases (from {v_B_original:.3f} to {v_B_perturbed2:.3f}).")
        else:
            self._log(f"FAIL: v_B did not increase as expected. Original {v_B_original:.3f}, perturbed {v_B_perturbed2:.3f}")
            success = False
        
        return success
    
    def run_all_tests(self):
        """Run all tests and return (success_bool, logs)."""
        self.logs = []
        overall_success = True
        
        try:
            if not self.test_invariants():
                overall_success = False
            if not self.test_trace_consistency():
                overall_success = False
            if not self.test_monotonicity():
                overall_success = False
        except Exception as e:
            self._log(f"ERROR during tests: {traceback.format_exc()}")
            overall_success = False
        
        self._log(f"=== Overall Result: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, self.logs