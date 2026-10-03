import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction from results dict
        results = getattr(solution_instance, 'results', {})
        
        # Get all required values from results
        self.v_p = results.get('v_p', 0.0)
        self.s_prime = results.get("s''", results.get('s_prime', 0.0))
        self.a = results.get('a', 0.0)
        self.t_1 = results.get('t_1', 0.0)
        self.x_1 = results.get('x_1', 0.0)
        self.t_2 = results.get('t_2', 0.0)
        self.x_2 = results.get('x_2', 0.0)
        self.W_conveyor = results.get('W_conveyor', 0.0)
        self.W_slide = results.get('W_slide', 0.0)
        self.s_total = results.get('s_total', 0.0)
        
        # Constants - use exact values as in problem
        self.m = 2.0
        self.theta = 37.0  # degrees
        self.sin_theta = 0.6  # exact sin37°
        self.cos_theta = 0.8  # exact cos37°
        self.s = 9.0
        self.L = 10.0
        self.mu1 = 0.5
        self.mu2 = 0.4
        self.g = 10.0
        self.v_conveyor = 2.0
        
        self.logs = []
    
    def _is_close(self, a, b, name=""):
        """Robust fuzzy comparison with tolerance."""
        close = math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
        if not close:
            self.logs.append(f"Numerical mismatch {name}: {a} vs {b}")
        return close
    
    def test_invariants(self):
        """Check fundamental physics laws using exact trig values."""
        success = True
        
        # 1. Work-energy theorem on incline for v_p
        # Using exact trig values: sin37° = 0.6, cos37° = 0.8
        W_gravity = self.m * self.g * self.s * self.sin_theta
        W_friction_incline = -self.mu1 * self.m * self.g * self.cos_theta * self.s
        W_net_incline = W_gravity + W_friction_incline
        ΔKE_incline = 0.5 * self.m * self.v_p**2
        
        if not self._is_close(W_net_incline, ΔKE_incline, "W_net_incline vs ΔKE_incline"):
            success = False
        
        # 2. Newton's 2nd law on conveyor belt: a = μ2*g (magnitude)
        a_expected = self.mu2 * self.g
        if not self._is_close(abs(self.a), a_expected, "|a| vs μ2*g"):
            success = False
        
        # 3. Work-energy on conveyor belt segment
        # W_conveyor should equal -μ2*m*g*(x1 + x2)
        W_conveyor_expected = -self.mu2 * self.m * self.g * (self.x_1 + self.x_2)
        if not self._is_close(self.W_conveyor, W_conveyor_expected, "W_conveyor vs -μ2mg(x1+x2)"):
            success = False
        
        # 4. Check s_total from ground truth
        s_total_expected = 9.5
        if not self._is_close(self.s_total, s_total_expected, "s_total vs expected 9.5"):
            success = False
        
        return success
    
    def test_trace_consistency(self):
        """Check intermediate results consistency."""
        success = True
        
        try:
            # 1. v_p from kinematics on incline (using exact trig)
            a_incline = self.g * (self.sin_theta - self.mu1 * self.cos_theta)
            v_p_expected = math.sqrt(2 * a_incline * self.s)
            if not self._is_close(self.v_p, v_p_expected, "v_p from kinematics"):
                success = False
            
            # 2. s' from work-energy on belt: v_p^2/(2*|a|)
            if abs(self.a) > 1e-10:
                s_prime_expected = self.v_p**2 / (2 * abs(self.a))
                if not self._is_close(self.s_prime, s_prime_expected, "s' calculation"):
                    success = False
            
            # 3. t_1 = v_p / |a|
            if abs(self.a) > 1e-10:
                t1_expected = self.v_p / abs(self.a)
                if not self._is_close(self.t_1, t1_expected, "t_1 calculation"):
                    success = False
            
            # 4. x_1 = v_conveyor * t_1
            x1_expected = self.v_conveyor * self.t_1
            if not self._is_close(self.x_1, x1_expected, "x_1 calculation"):
                success = False
            
            # 5. t_2 = v_conveyor / |a|
            if abs(self.a) > 1e-10:
                t2_expected = self.v_conveyor / abs(self.a)
                if not self._is_close(self.t_2, t2_expected, "t_2 calculation"):
                    success = False
            
            # 6. x_2 = v_conveyor * t_2
            x2_expected = self.v_conveyor * self.t_2
            if not self._is_close(self.x_2, x2_expected, "x_2 calculation"):
                success = False
            
            # 7. W_slide = 0.5*m*(v_conveyor^2 - v_p^2)
            W_slide_expected = 0.5 * self.m * (self.v_conveyor**2 - self.v_p**2)
            if not self._is_close(self.W_slide, W_slide_expected, "W_slide calculation"):
                success = False
            
            # 8. s_total from correct work-energy equation
            # -μ1*m*g*cosθ*(s_total - s) = -0.5*m*v_conveyor^2
            # => s_total = s + (0.5*v_conveyor^2) / (μ1*g*cosθ)
            s_total_calc = self.s + (0.5 * self.v_conveyor**2) / (self.mu1 * self.g * self.cos_theta)
            if not self._is_close(self.s_total, s_total_calc, "s_total from work-energy"):
                success = False
                
        except ZeroDivisionError as e:
            self.logs.append(f"ZeroDivisionError in trace consistency: {str(e)}")
            success = False
        except ValueError as e:
            self.logs.append(f"ValueError in trace consistency: {str(e)}")
            success = False
        
        return success
    
    def test_monotonicity(self):
        """Perturb input and check trend consistency."""
        success = True
        
        try:
            # Store original values
            original_mu1 = self.mu1
            
            # Perturb μ1 by +10%
            self.mu1 *= 1.1
            
            # Recalculate v_p using work-energy with exact trig
            # v_p^2 = 2*g*s*(sinθ - μ1*cosθ)
            term = self.sin_theta - self.mu1 * self.cos_theta
            if term > 0:
                v_p_perturbed = math.sqrt(2 * self.g * self.s * term)
                
                # Check monotonic trend: higher μ1 → lower v_p
                # Use <= to handle equality due to rounding
                if not (v_p_perturbed <= self.v_p + 1e-10):
                    self.logs.append(f"Monotonicity violation: μ1↑ should cause v_p↓, but {v_p_perturbed} > {self.v_p}")
                    success = False
            else:
                self.logs.append("Monotonicity test skipped: term <= 0")
            
            # Restore original μ1
            self.mu1 = original_mu1
            
            # Perturb μ2 by +10%
            original_mu2 = self.mu2
            self.mu2 *= 1.1
            
            # Recalculate a and then W_conveyor
            a_perturbed = self.mu2 * self.g
            t1_perturbed = self.v_p / a_perturbed if a_perturbed > 1e-10 else 0
            x1_perturbed = self.v_conveyor * t1_perturbed
            t2_perturbed = self.v_conveyor / a_perturbed if a_perturbed > 1e-10 else 0
            x2_perturbed = self.v_conveyor * t2_perturbed
            W_conveyor_perturbed = -self.mu2 * self.m * self.g * (x1_perturbed + x2_perturbed)
            
            # Check monotonic trend: higher μ2 → more negative W_conveyor
            # Use <= to handle equality due to rounding
            if not (W_conveyor_perturbed <= self.W_conveyor + 1e-10):
                self.logs.append(f"Monotonicity violation: μ2↑ should cause W_conveyor↓, but {W_conveyor_perturbed} > {self.W_conveyor}")
                success = False
            
            # Restore original μ2
            self.mu2 = original_mu2
            
        except Exception as e:
            self.logs.append(f"Monotonicity test error: {str(e)}")
            success = False
        
        return success
    
    def run_all_tests(self):
        """Execute all tests and return results."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                success = test_func()
                if success:
                    self.logs.append(f"✓ {test_name} passed")
                else:
                    self.logs.append(f"✗ {test_name} failed")
                    overall_success = False
            except Exception as e:
                self.logs.append(f"✗ {test_name} error: {str(e)}")
                overall_success = False
        
        return overall_success, self.logs