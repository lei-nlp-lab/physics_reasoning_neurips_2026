import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr with fallback to results dict
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Problem constants
        self.m = 60.0  # kg
        self.x1 = 54.0  # m (inclined track)
        self.x2 = 40.5  # m (horizontal track)
        self.t_total = 10.5  # s
        self.theta = 37.0  # degrees
        self.sin_theta = 0.6
        self.g = 10.0  # m/s^2 (implied by the calculation)
        
        # Extract results safely
        self.v_avg = getattr(solution_instance, 'v_avg', self.results.get('v_avg', 0.0))
        self.v_m = getattr(solution_instance, 'v_m', self.results.get('v_m', 0.0))
        self.t1 = getattr(solution_instance, 't_1', self.results.get('t_1', 0.0))
        self.a = getattr(solution_instance, 'a', self.results.get('a', 0.0))
        self.F_f = getattr(solution_instance, 'F_f', self.results.get('F_f', 0.0))
        
        # Derived quantities
        self.t2 = self.t_total - self.t1 if self.t1 > 0 else 0.0
        self.a_horiz = -self.v_m / self.t2 if self.t2 > 0 else 0.0  # horizontal deceleration
        
        # Tolerances for fuzzy matching
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Logs
        self.logs = []
    
    def _log(self, message, success=None):
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {message}")
        else:
            self.logs.append(f"INFO: {message}")
    
    def _is_close(self, a, b):
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
    
    def test_invariants(self):
        """Test physical invariants: Newton's 2nd law and energy/work consistency."""
        success = True
        
        # 1. Newton's 2nd law on incline: mg sinθ - F_f = m*a
        net_force_incline = self.m * self.g * self.sin_theta - self.F_f
        expected_net = self.m * self.a
        
        if not self._is_close(net_force_incline, expected_net):
            self._log(f"Newton's 2nd law on incline failed: {net_force_incline:.2f} != {expected_net:.2f}", False)
            success = False
        else:
            self._log(f"Newton's 2nd law on incline satisfied: {net_force_incline:.2f} = {expected_net:.2f}", True)
        
        # 2. Kinematic consistency for horizontal motion: v_m^2 = 2 * |a_horiz| * x2
        # Since it decelerates uniformly from v_m to 0: v_m^2 = 2 * a_horiz_mag * x2
        a_horiz_mag = abs(self.a_horiz)
        expected_vm_sq = 2 * a_horiz_mag * self.x2
        actual_vm_sq = self.v_m ** 2
        
        if not self._is_close(actual_vm_sq, expected_vm_sq):
            self._log(f"Horizontal kinematic consistency failed: v_m^2={actual_vm_sq:.2f}, 2*a*x2={expected_vm_sq:.2f}", False)
            success = False
        else:
            self._log(f"Horizontal kinematic consistency satisfied: v_m^2={actual_vm_sq:.2f} = 2*a*x2={expected_vm_sq:.2f}", True)
        
        # 3. Work-energy: Net work = ΔKE
        # On incline: (mg sinθ - F_f)*x1 = 0.5*m*v_m^2 - 0
        net_work_incline = (self.m * self.g * self.sin_theta - self.F_f) * self.x1
        delta_ke = 0.5 * self.m * self.v_m ** 2
        
        if not self._is_close(net_work_incline, delta_ke):
            self._log(f"Work-energy on incline failed: {net_work_incline:.2f} != {delta_ke:.2f}", False)
            success = False
        else:
            self._log(f"Work-energy on incline satisfied: {net_work_incline:.2f} = {delta_ke:.2f}", True)
        
        return success
    
    def test_trace_consistency(self):
        """Check internal consistency between intermediate results."""
        success = True
        
        # 1. Average velocity from total distance and time
        expected_v_avg = (self.x1 + self.x2) / self.t_total
        if not self._is_close(self.v_avg, expected_v_avg):
            self._log(f"v_avg inconsistency: {self.v_avg:.2f} != {expected_v_avg:.2f}", False)
            success = False
        else:
            self._log(f"v_avg consistent with total distance/time: {self.v_avg:.2f} = {expected_v_avg:.2f}", True)
        
        # 2. v_m = 2*v_avg (for motion with constant acceleration from rest to v_m then to rest)
        expected_vm_from_avg = 2 * self.v_avg
        if not self._is_close(self.v_m, expected_vm_from_avg):
            self._log(f"v_m vs v_avg failed: {self.v_m:.2f} != 2*{self.v_avg:.2f}={expected_vm_from_avg:.2f}", False)
            success = False
        else:
            self._log(f"v_m consistent with v_avg: {self.v_m:.2f} = 2*{self.v_avg:.2f}", True)
        
        # 3. t1 from inclined motion: x1 = (v_m/2)*t1 => t1 = 2*x1/v_m
        expected_t1 = 2 * self.x1 / self.v_m if self.v_m > 0 else 0
        if not self._is_close(self.t1, expected_t1):
            self._log(f"t1 inconsistency: {self.t1:.2f} != 2*x1/v_m={expected_t1:.2f}", False)
            success = False
        else:
            self._log(f"t1 consistent with x1 and v_m: {self.t1:.2f} = 2*x1/v_m={expected_t1:.2f}", True)
        
        # 4. Acceleration on incline: a = v_m/t1
        expected_a = self.v_m / self.t1 if self.t1 > 0 else 0
        if not self._is_close(self.a, expected_a):
            self._log(f"Acceleration inconsistency: {self.a:.2f} != v_m/t1={expected_a:.2f}", False)
            success = False
        else:
            self._log(f"Acceleration consistent: {self.a:.2f} = v_m/t1={expected_a:.2f}", True)
        
        # 5. Check horizontal time consistency
        if not self._is_close(self.t1 + self.t2, self.t_total):
            self._log(f"Time sum inconsistency: t1+t2={self.t1+self.t2:.2f} != total={self.t_total:.2f}", False)
            success = False
        else:
            self._log(f"Time sum consistent: {self.t1:.2f}+{self.t2:.2f}={self.t_total:.2f}", True)
        
        return success
    
    def test_monotonicity(self):
        """Perturb mass and check expected trends in F_f."""
        success = True
        
        # Store original values
        original_m = self.m
        original_F_f = self.F_f
        
        try:
            # Perturb mass by +10%
            perturbed_m = original_m * 1.1
            
            # Recalculate F_f using Newton's 2nd law formula: F_f = m*g*sinθ - m*a
            # Note: a is kinematic and shouldn't change with mass for same motion
            # But here, we assume the same acceleration (from kinematic constraints)
            perturbed_F_f = perturbed_m * self.g * self.sin_theta - perturbed_m * self.a
            
            # Check monotonicity: F_f should increase with mass (since g*sinθ > a)
            if not perturbed_F_f > original_F_f:
                self._log(f"Monotonicity failed: F_f({perturbed_m:.1f})={perturbed_F_f:.2f} ≤ F_f({original_m:.1f})={original_F_f:.2f}", False)
                success = False
            else:
                self._log(f"Monotonicity passed: F_f increases with mass: {original_F_f:.2f} → {perturbed_F_f:.2f}", True)
            
            # Perturb mass by -10%
            perturbed_m2 = original_m * 0.9
            perturbed_F_f2 = perturbed_m2 * self.g * self.sin_theta - perturbed_m2 * self.a
            
            if not perturbed_F_f2 < original_F_f:
                self._log(f"Monotonicity failed: F_f({perturbed_m2:.1f})={perturbed_F_f2:.2f} ≥ F_f({original_m:.1f})={original_F_f:.2f}", False)
                success = False
            else:
                self._log(f"Monotonicity passed: F_f decreases with mass: {original_F_f:.2f} → {perturbed_F_f2:.2f}", True)
                
        except Exception as e:
            self._log(f"Monotonicity test error: {traceback.format_exc()}", False)
            success = False
        
        # Restore original values (though they weren't actually modified)
        self.m = original_m
        self.F_f = original_F_f
        
        return success
    
    def run_all_tests(self):
        """Run all tests and return success flag and logs."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Trace Consistency", self.test_trace_consistency),
            ("Invariants", self.test_invariants),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self._log(f"--- Running {test_name} ---", None)
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"Test {test_name} crashed: {traceback.format_exc()}", False)
                overall_success = False
        
        # Final summary
        self._log(f"=== Overall: {'PASS' if overall_success else 'FAIL'} ===", None)
        
        return overall_success, self.logs