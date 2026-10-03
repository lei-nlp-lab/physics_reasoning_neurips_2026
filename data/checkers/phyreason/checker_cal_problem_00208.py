import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.n = getattr(solution_instance, 'n', solution_instance.results.get('n', 1.5))
        # Extract key results with safe defaults
        self.TIR_E = getattr(solution_instance, 'TIR_E', 
                           solution_instance.results.get('TIR_E', 
                           solution_instance.results.get('total_internal_reflection', False)))
        self.sin_r_prime = getattr(solution_instance, 'sin_r_prime', 
                                 solution_instance.results.get('sinr\'', 
                                 solution_instance.results.get('sin_r_prime', None)))
        self.theta = getattr(solution_instance, 'theta', 
                           solution_instance.results.get('theta', None))
        self.r = getattr(solution_instance, 'r', 
                        solution_instance.results.get('r', None))
        self.i_prime = getattr(solution_instance, 'i_prime', 
                             solution_instance.results.get('i\'', None))
        
        self.logs = []
        self.success = True
        
    def _log(self, msg, is_error=False):
        self.logs.append(msg)
        if is_error:
            self.success = False
    
    def _check_snell_at_D(self):
        """Check Snell's Law at point D"""
        try:
            i = math.radians(30)  # Given in problem
            sin_i = math.sin(i)
            r_computed = math.asin(sin_i / self.n)
            
            # If r was provided, check consistency
            if self.r is not None:
                if not math.isclose(r_computed, self.r, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Snell at D: Computed r={r_computed:.4f}, Provided r={self.r:.4f}", True)
            else:
                self.r = r_computed  # Store for later use
                
            return r_computed
        except Exception as e:
            self._log(f"Snell at D check failed: {str(e)}", True)
            return None
    
    def _check_theta_geometry(self, r_val):
        """Check geometric relationship for theta"""
        try:
            # θ = 90° - (30° - r) = 60° + r
            theta_computed = math.radians(60) + r_val
            
            if self.theta is not None:
                if not math.isclose(theta_computed, self.theta, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Theta geometry: Computed θ={theta_computed:.4f}, Provided θ={self.theta:.4f}", True)
            else:
                self.theta = theta_computed
                
            return theta_computed
        except Exception as e:
            self._log(f"Theta geometry check failed: {str(e)}", True)
            return None
    
    def _check_TIR_condition(self, theta_val):
        """Verify total internal reflection condition at point E"""
        try:
            # Critical angle
            critical_angle = math.asin(1 / self.n)
            
            # Check if theta > critical angle
            TIR_condition = theta_val > critical_angle
            
            # Check the stronger condition from step_2: sinθ > sin60° > 1/n
            sin_theta = math.sin(theta_val)
            sin_60 = math.sin(math.radians(60))
            condition_step2 = (sin_theta > sin_60) and (sin_60 > 1/self.n)
            
            if not TIR_condition:
                self._log(f"TIR condition failed: θ={theta_val:.4f} ≤ critical_angle={critical_angle:.4f}", True)
            
            # Compare with provided TIR_E if available
            if isinstance(self.TIR_E, bool) and self.TIR_E != TIR_condition:
                self._log(f"TIR mismatch: Expected {TIR_condition}, Got {self.TIR_E}", True)
                
            return TIR_condition, condition_step2
        except Exception as e:
            self._log(f"TIR condition check failed: {str(e)}", True)
            return False, False
    
    def _check_snell_at_F(self, theta_val):
        """Check Snell's Law at point F and compute sin(r')"""
        try:
            # i' = 90° - θ
            i_prime_computed = math.pi/2 - theta_val
            
            if self.i_prime is not None:
                if not math.isclose(i_prime_computed, self.i_prime, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"i' geometry: Computed i'={i_prime_computed:.4f}, Provided i'={self.i_prime:.4f}", True)
            
            # sin(r') = n * sin(i')
            sin_r_prime_computed = self.n * math.sin(i_prime_computed)
            
            # Expected value from problem
            sin_r_prime_expected = (2*math.sqrt(2) - math.sqrt(3)) / 4
            
            if self.sin_r_prime is not None:
                if not math.isclose(sin_r_prime_computed, self.sin_r_prime, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"sin(r'): Computed {sin_r_prime_computed:.6f}, Provided {self.sin_r_prime:.6f}", True)
            
            # Check against expected value
            if not math.isclose(sin_r_prime_computed, sin_r_prime_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"sin(r'): Computed {sin_r_prime_computed:.6f}, Expected {sin_r_prime_expected:.6f}", True)
                
            return sin_r_prime_computed
        except Exception as e:
            self._log(f"Snell at F check failed: {str(e)}", True)
            return None
    
    def test_invariants(self):
        """Test physics invariants: Snell's Laws and TIR condition"""
        self._log("=== Testing Physics Invariants ===")
        
        # 1. Check Snell's Law at D
        r_val = self._check_snell_at_D()
        if r_val is None:
            return
        
        # 2. Check theta geometry
        theta_val = self._check_theta_geometry(r_val)
        if theta_val is None:
            return
        
        # 3. Check TIR condition at E
        TIR_cond, step2_cond = self._check_TIR_condition(theta_val)
        
        # 4. Check Snell's Law at F
        sin_r_prime_val = self._check_snell_at_F(theta_val)
        
        # 5. Verify step_2 condition: sinθ > sin60° > 1/n
        if theta_val is not None:
            sin_theta = math.sin(theta_val)
            sin_60 = math.sin(math.radians(60))
            if not (sin_theta > sin_60 and sin_60 > 1/self.n):
                self._log("Step_2 condition failed: sinθ > sin60° > 1/n not satisfied", True)
    
    def test_trace_consistency(self):
        """Check consistency between intermediate results and final answer"""
        self._log("=== Testing Trace Consistency ===")
        
        try:
            # Reconstruct from known values
            if self.r is None or self.theta is None:
                # Try to compute from scratch
                i = math.radians(30)
                r_computed = math.asin(math.sin(i) / self.n)
                theta_computed = math.radians(60) + r_computed
            else:
                r_computed = self.r
                theta_computed = self.theta
            
            # Compute sin(r') from reconstructed values
            i_prime_computed = math.pi/2 - theta_computed
            sin_r_prime_reconstructed = self.n * math.sin(i_prime_computed)
            
            # Expected value
            sin_r_prime_expected = (2*math.sqrt(2) - math.sqrt(3)) / 4
            
            if not math.isclose(sin_r_prime_reconstructed, sin_r_prime_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Trace inconsistency: Reconstructed sin(r')={sin_r_prime_reconstructed:.6f}, Expected {sin_r_prime_expected:.6f}", True)
            
            # Check if TIR would occur
            critical_angle = math.asin(1 / self.n)
            if not (theta_computed > critical_angle):
                self._log("Trace inconsistency: TIR should occur but condition not met", True)
                
        except Exception as e:
            self._log(f"Trace consistency check failed: {str(e)}", True)
    
    def test_monotonicity(self):
        """Perturb refractive index and check expected trends"""
        self._log("=== Testing Monotonicity ===")
        
        try:
            original_n = self.n
            perturbed_n = original_n + 0.1  # Increase n
            
            # Compute for original n
            i = math.radians(30)
            r_orig = math.asin(math.sin(i) / original_n)
            theta_orig = math.radians(60) + r_orig
            critical_angle_orig = math.asin(1 / original_n)
            TIR_orig = theta_orig > critical_angle_orig
            sin_r_prime_orig = original_n * math.sin(math.pi/2 - theta_orig)
            
            # Compute for perturbed n
            r_pert = math.asin(math.sin(i) / perturbed_n)
            theta_pert = math.radians(60) + r_pert
            critical_angle_pert = math.asin(1 / perturbed_n)
            TIR_pert = theta_pert > critical_angle_pert
            sin_r_prime_pert = perturbed_n * math.sin(math.pi/2 - theta_pert)
            
            # Check trends
            # 1. As n increases, critical angle decreases
            if not (critical_angle_pert < critical_angle_orig):
                self._log("Monotonicity: Critical angle should decrease as n increases", True)
            
            # 2. As n increases, r decreases
            if not (r_pert < r_orig):
                self._log("Monotonicity: r should decrease as n increases", True)
            
            # 3. As n increases, TIR condition becomes stronger (easier to satisfy)
            # Actually, check if both satisfy TIR (they should)
            if not (TIR_orig and TIR_pert):
                self._log("Monotonicity: TIR should occur for both n values", True)
            
            # 4. Change in sin(r')
            delta_sin = sin_r_prime_pert - sin_r_prime_orig
            self._log(f"Monotonicity: Δsin(r') = {delta_sin:.6f} when n increased by 0.1")
            
            # Restore original n
            self.n = original_n
            
        except Exception as e:
            self._log(f"Monotonicity check failed: {str(e)}", True)
            # Restore n even on error
            self.n = original_n
    
    def run_all_tests(self):
        """Run all tests and return success status and logs"""
        try:
            self.test_invariants()
            self.test_trace_consistency()
            self.test_monotonicity()
        except Exception as e:
            self._log(f"Unexpected error in test suite: {str(e)}", True)
        
        return self.success, self.logs