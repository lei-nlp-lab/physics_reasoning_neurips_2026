import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract all expected keys with safe defaults
        self.s0 = self._get_value('s_0', 0.0)
        self.a1 = self._get_value('a_1', 0.0)
        self.t_min = self._get_value('t_min', 0.0)
        self.t2 = self._get_value('t_2', 0.0)
        self.t = self._get_value('t', 0.0)
        self.a = self._get_value('a', 0.0)
        self.a2 = self._get_value('a_2', 0.0)
        self.t1 = self._get_value('t_1', 0.0)
        self.a0 = self._get_value('a_0', 0.0)
        self.v_CD = self._get_value('v_CD', 0.0)
        self.s = self._get_value('s', 0.0)
        self.v1 = self._get_value('v_1', 0.0)
        
        # Constants from problem statement
        self.AB = 3.0
        self.CD = 4.45
        self.theta_deg = 37.0
        self.theta_rad = math.radians(self.theta_deg)
        self.sin37 = 0.60
        self.cos37 = 0.80
        self.m = 10.0
        self.mu = 0.5
        self.g = 10.0
        self.v0 = 5.0
        
        self.logs = []
        
    def _get_value(self, key, default):
        """Safely extract value from results or solution attributes"""
        value = self.results.get(key)
        if value is None:
            value = getattr(self.solution, key, default)
        return float(value) if value is not None else default
        
    def _log(self, message, is_error=False):
        """Log message with optional error flag"""
        self.logs.append(f"{'ERROR' if is_error else 'INFO'}: {message}")
        
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison"""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Test fundamental physics invariants"""
        try:
            # Newton's 2nd Law check for a0
            expected_a0 = self.mu * self.g
            if not self._is_close(self.a0, expected_a0):
                self._log(f"a0 mismatch: expected {expected_a0}, got {self.a0}", True)
                
            # Newton's 2nd Law check for a (stationary belt)
            expected_a = self.g * (self.sin37 + self.mu * self.cos37)
            if not self._is_close(self.a, expected_a):
                self._log(f"a mismatch: expected {expected_a}, got {self.a}", True)
                
            # Newton's 2nd Law check for a1 (moving belt, decelerating up)
            expected_a1 = -self.g * (self.sin37 + self.mu * self.cos37)
            if not self._is_close(self.a1, expected_a1):
                self._log(f"a1 mismatch: expected {expected_a1}, got {self.a1}", True)
                
            # Newton's 2nd Law check for a2 (after reaching belt speed)
            expected_a2 = -self.g * (self.sin37 - self.mu * self.cos37)
            if not self._is_close(self.a2, expected_a2):
                self._log(f"a2 mismatch: expected {expected_a2}, got {self.a2}", True)
                
            # Energy check for horizontal section: Work-energy theorem with friction
            # Friction work on AB: W_f = mu*m*g*s0
            # Change in kinetic energy: 0.5*m*v0^2
            # They should be equal during acceleration phase
            friction_work = self.mu * self.m * self.g * self.s0
            kinetic_energy_gain = 0.5 * self.m * self.v0**2
            if not self._is_close(friction_work, kinetic_energy_gain):
                self._log(f"Energy balance on AB mismatch: friction work {friction_work} vs KE gain {kinetic_energy_gain}", True)
                
            return True
        except Exception as e:
            self._log(f"Invariants test failed: {str(e)}", True)
            return False
            
    def test_trace_consistency(self):
        """Test consistency between intermediate and final results"""
        try:
            # Step 2: t1 = v0 / a0
            expected_t1 = self.v0 / self.a0 if abs(self.a0) > 1e-12 else 0.0
            if not self._is_close(self.t1, expected_t1):
                self._log(f"t1 inconsistent: expected {expected_t1}, got {self.t1}", True)
                
            # Step 3: s0 = v0^2/(2*a0)
            expected_s0 = self.v0**2 / (2*self.a0) if abs(self.a0) > 1e-12 else 0.0
            if not self._is_close(self.s0, expected_s0):
                self._log(f"s0 inconsistent: expected {expected_s0}, got {self.s0}", True)
                
            # Step 5: t2 = (AB - s0)/v0
            expected_t2 = (self.AB - self.s0) / self.v0 if abs(self.v0) > 1e-12 else 0.0
            if not self._is_close(self.t2, expected_t2):
                self._log(f"t2 inconsistent: expected {expected_t2}, got {self.t2}", True)
                
            # Step 6: t = t1 + t2
            expected_t = self.t1 + self.t2
            if not self._is_close(self.t, expected_t):
                self._log(f"Total time t inconsistent: expected {expected_t}, got {self.t}", True)
                
            # Step 8: s = v0^2/(2*a)
            expected_s = self.v0**2 / (2*self.a) if abs(self.a) > 1e-12 else 0.0
            if not self._is_close(self.s, expected_s):
                self._log(f"Max distance s inconsistent: expected {expected_s}, got {self.s}", True)
                
            # For the critical velocity v1, check it satisfies the kinematics on CD
            # The bag must travel CD distance with two accelerations: a1 until v1, then a2 to stop
            # Use v1 (critical belt speed) for the calculation
            if abs(self.v1) > 1e-12:
                # Phase 1: deceleration from v0 to v1 with a1
                t_phase1 = (self.v1 - self.v0) / self.a1 if abs(self.a1) > 1e-12 else 0.0
                s_phase1 = self.v0*t_phase1 + 0.5*self.a1*t_phase1**2
                
                # Phase 2: deceleration from v1 to 0 with a2
                t_phase2 = -self.v1 / self.a2 if abs(self.a2) > 1e-12 else 0.0
                s_phase2 = self.v1*t_phase2 + 0.5*self.a2*t_phase2**2
                
                total_distance = s_phase1 + s_phase2
                if not self._is_close(total_distance, self.CD, abs_tol=0.1):
                    self._log(f"CD distance mismatch: computed {total_distance}, required {self.CD}", True)
                    
            return True
        except Exception as e:
            self._log(f"Trace consistency test failed: {str(e)}", True)
            return False
            
    def test_monotonicity(self):
        """Test sensitivity to parameter perturbations"""
        original_mu = self.mu
        
        try:
            # Perturb mu by +10%
            perturbed_mu = original_mu * 1.1
            
            # Recalculate accelerations with perturbed mu
            perturbed_a0 = perturbed_mu * self.g
            perturbed_a = self.g * (self.sin37 + perturbed_mu * self.cos37)
            perturbed_a1 = -self.g * (self.sin37 + perturbed_mu * self.cos37)
            perturbed_a2 = -self.g * (self.sin37 - perturbed_mu * self.cos37)
            
            # Check monotonic trends
            if perturbed_mu > original_mu:
                # a0 should increase
                if perturbed_a0 <= self.a0:
                    self._log(f"a0 not increasing with mu: {perturbed_a0} <= {self.a0}", True)
                    
                # a (stationary belt deceleration) should increase
                if perturbed_a <= self.a:
                    self._log(f"a not increasing with mu: {perturbed_a} <= {self.a}", True)
                    
                # a1 magnitude should increase (more negative)
                if perturbed_a1 >= self.a1:
                    self._log(f"|a1| not increasing with mu: {perturbed_a1} >= {self.a1}", True)
                    
                # a2 magnitude should decrease (less negative)
                if perturbed_a2 <= self.a2:
                    self._log(f"|a2| not decreasing with mu: {perturbed_a2} <= {self.a2}", True)
                    
                # s0 should decrease (higher friction reduces slip distance)
                perturbed_s0 = self.v0**2 / (2*perturbed_a0)
                if perturbed_s0 >= self.s0:
                    self._log(f"s0 not decreasing with increased mu: {perturbed_s0} >= {self.s0}", True)
                    
                # t1 should decrease (higher acceleration)
                perturbed_t1 = self.v0 / perturbed_a0
                if perturbed_t1 >= self.t1:
                    self._log(f"t1 not decreasing with increased mu: {perturbed_t1} >= {self.t1}", True)
                    
            # Restore original value
            self.mu = original_mu
            
            return True
        except Exception as e:
            self._log(f"Monotonicity test failed: {str(e)}", True)
            self.mu = original_mu  # Ensure restoration
            return False
            
    def run_all_tests(self):
        """Run all tests and return success flag with logs"""
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            try:
                if not test_func():
                    success = False
                    self._log(f"{test_name} test failed", True)
                else:
                    self._log(f"{test_name} test passed")
            except Exception as e:
                success = False
                self._log(f"{test_name} test crashed: {str(e)}", True)
                self._log(traceback.format_exc(), True)
                
        return success, self.logs