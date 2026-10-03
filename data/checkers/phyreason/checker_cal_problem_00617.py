import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize with safe attribute access from solution_instance.
        solution_instance should have attributes or a results dictionary.
        """
        # Problem constants (from context)
        self.theta = math.radians(30)  # 30 degrees in radians
        self.sin_theta = math.sin(self.theta)  # 0.5
        self.cos_theta = math.cos(self.theta)  # sqrt(3)/2 ≈ 0.8660
        self.v = 2.0  # m/s, conveyor belt speed
        self.v0 = 6.0  # m/s, initial block speed
        self.mu = math.sqrt(3) / 5  # friction coefficient ≈ 0.3464
        self.g = 10.0  # m/s²
        
        # Extract results safely from solution_instance
        # Use getattr with fallback to results dict if available
        self.results = {}
        result_keys = [
            'a_1', 't_2', 's_prime', 'l_0', 't_3', 'a_2', 'x_2', 
            't_1', 's', 'x_1'
        ]
        
        # First try to get from results dict
        if hasattr(solution_instance, 'results'):
            for key in result_keys:
                self.results[key] = solution_instance.results.get(key, 0.0)
        else:
            # Fallback to attributes
            for key in result_keys:
                attr_name = key
                if hasattr(solution_instance, attr_name):
                    self.results[key] = getattr(solution_instance, attr_name, 0.0)
                else:
                    self.results[key] = 0.0
        
        # For convenience, map keys to variables
        self.a1 = self.results.get('a_1', 0.0)
        self.a2 = self.results.get('a_2', 0.0)
        self.t1 = self.results.get('t_1', 0.0)
        self.t2 = self.results.get('t_2', 0.0)
        self.t3 = self.results.get('t_3', 0.0)
        self.x1 = self.results.get('x_1', 0.0)
        self.x2 = self.results.get('x_2', 0.0)
        self.l0 = self.results.get('l_0', 0.0)
        self.s = self.results.get('s', 0.0)  # scratch length
        self.s_prime = self.results.get('s_prime', 0.0)  # belt distance
        
        # Store logs for reporting
        self.logs = []
    
    def _log(self, message: str, success: bool = True):
        """Helper to log test results."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{status}: {message}")
    
    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Robust numerical comparison with physics tolerance."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self) -> bool:
        """Test fundamental physics laws: Newton's 2nd Law and Energy."""
        success = True
        
        # 1. Check Newton's 2nd Law for first phase (a1)
        # From step_1: a1 = -g*(sinθ + μ cosθ)
        expected_a1 = -self.g * (self.sin_theta + self.mu * self.cos_theta)
        if not self._is_close(self.a1, expected_a1):
            self._log(f"a1 fails Newton's 2nd Law: expected {expected_a1:.3f}, got {self.a1:.3f}", False)
            success = False
        else:
            self._log(f"a1 Newton's 2nd Law check: {self.a1:.3f} ≈ {expected_a1:.3f}")
        
        # 2. Check Newton's 2nd Law for second phase (a2)
        # From step_2: a2 = g*(μ cosθ - sinθ)
        expected_a2 = self.g * (self.mu * self.cos_theta - self.sin_theta)
        if not self._is_close(self.a2, expected_a2):
            self._log(f"a2 fails Newton's 2nd Law: expected {expected_a2:.3f}, got {self.a2:.3f}", False)
            success = False
        else:
            self._log(f"a2 Newton's 2nd Law check: {self.a2:.3f} ≈ {expected_a2:.3f}")
        
        # 3. Check that both accelerations are negative (deceleration uphill)
        if self.a1 >= 0:
            self._log(f"a1 should be negative (deceleration uphill), got {self.a1:.3f}", False)
            success = False
        if self.a2 >= 0:
            self._log(f"a2 should be negative (deceleration uphill), got {self.a2:.3f}", False)
            success = False
        
        # 4. Energy conservation check (numeric, not symbolic)
        # For first phase: ΔKE = 0.5*(v² - v0²)
        delta_ke1 = 0.5 * (self.v**2 - self.v0**2)
        # Work by gravity: -g*sinθ*x1
        work_gravity1 = -self.g * self.sin_theta * self.x1
        # Work by friction: -μ*g*cosθ*x1
        work_friction1 = -self.mu * self.g * self.cos_theta * self.x1
        total_work1 = work_gravity1 + work_friction1
        
        if not self._is_close(delta_ke1, total_work1):
            self._log(f"Energy conservation (first phase) failed: ΔKE={delta_ke1:.3f}, total work={total_work1:.3f}", False)
            success = False
        else:
            self._log(f"Energy conservation (first phase): ΔKE={delta_ke1:.3f} ≈ total work={total_work1:.3f}")
        
        return success
    
    def test_trace_consistency(self) -> bool:
        """Check consistency between intermediate results and final answers."""
        success = True
        
        # 1. Check kinematic equations for t1
        # t1 = (v - v0) / a1
        expected_t1 = (self.v - self.v0) / self.a1 if abs(self.a1) > 1e-12 else 0.0
        if not self._is_close(self.t1, expected_t1):
            self._log(f"t1 kinematic inconsistent: expected {expected_t1:.3f}, got {self.t1:.3f}", False)
            success = False
        else:
            self._log(f"t1 kinematic check: {self.t1:.3f} ≈ {expected_t1:.3f}")
        
        # 2. Check kinematic for t2
        # t2 = (0 - v) / a2
        expected_t2 = (0 - self.v) / self.a2 if abs(self.a2) > 1e-12 else 0.0
        if not self._is_close(self.t2, expected_t2):
            self._log(f"t2 kinematic inconsistent: expected {expected_t2:.3f}, got {self.t2:.3f}", False)
            success = False
        else:
            self._log(f"t2 kinematic check: {self.t2:.3f} ≈ {expected_t2:.3f}")
        
        # 3. Check x1 = (v0 + v)/2 * t1
        expected_x1 = (self.v0 + self.v) / 2 * self.t1
        if not self._is_close(self.x1, expected_x1):
            self._log(f"x1 kinematic inconsistent: expected {expected_x1:.3f}, got {self.x1:.3f}", False)
            success = False
        else:
            self._log(f"x1 kinematic check: {self.x1:.3f} ≈ {expected_x1:.3f}")
        
        # 4. Check x2 = v/2 * t2
        expected_x2 = self.v / 2 * self.t2
        if not self._is_close(self.x2, expected_x2):
            self._log(f"x2 kinematic inconsistent: expected {expected_x2:.3f}, got {self.x2:.3f}", False)
            success = False
        else:
            self._log(f"x2 kinematic check: {self.x2:.3f} ≈ {expected_x2:.3f}")
        
        # 5. Check l0 = x1 + x2
        expected_l0 = self.x1 + self.x2
        if not self._is_close(self.l0, expected_l0):
            self._log(f"l0 sum inconsistent: expected {expected_l0:.3f}, got {self.l0:.3f}", False)
            success = False
        else:
            self._log(f"l0 sum check: {self.l0:.3f} ≈ {expected_l0:.3f}")
        
        # 6. Check t3 = sqrt(2*l0/|a2|)
        # Note: a2 is negative, use absolute value for acceleration magnitude
        expected_t3 = math.sqrt(2 * self.l0 / abs(self.a2)) if self.a2 != 0 else 0.0
        if not self._is_close(self.t3, expected_t3):
            self._log(f"t3 kinematic inconsistent: expected {expected_t3:.3f}, got {self.t3:.3f}", False)
            success = False
        else:
            self._log(f"t3 kinematic check: {self.t3:.3f} ≈ {expected_t3:.3f}")
        
        # 7. Check belt distance s' = v*(t1 + t2 + t3)
        total_time = self.t1 + self.t2 + self.t3
        expected_s_prime = self.v * total_time
        if not self._is_close(self.s_prime, expected_s_prime):
            self._log(f"s' inconsistent: expected {expected_s_prime:.3f}, got {self.s_prime:.3f}", False)
            success = False
        else:
            self._log(f"s' check: {self.s_prime:.3f} ≈ {expected_s_prime:.3f}")
        
        # 8. Check scratch length s (should be min(s', 6) or 2*l0 depending on interpretation)
        # According to problem, s should be 6.0 m (the total conveyor belt length)
        # since s' > 6.0 m
        expected_s = 6.0  # From ground truth
        if not self._is_close(self.s, expected_s):
            self._log(f"s inconsistent: expected {expected_s:.3f}, got {self.s:.3f}", False)
            success = False
        else:
            self._log(f"s check: {self.s:.3f} ≈ {expected_s:.3f}")
        
        return success
    
    def test_monotonicity(self) -> bool:
        """
        Perturb an input and check that results change in expected direction.
        We'll perturb v0 (initial speed) and check t1.
        """
        success = True
        
        # Original t1
        t1_original = self.t1
        
        # Perturb v0 by +10%
        v0_perturbed = self.v0 * 1.1  # Increase by 10%
        
        # Recalculate t1 using kinematic formula: t1 = (v - v0)/a1
        # a1 remains constant (depends only on g, θ, μ)
        a1 = -self.g * (self.sin_theta + self.mu * self.cos_theta)
        t1_perturbed = (self.v - v0_perturbed) / a1
        
        # Since a1 is negative, and v0 increases, (v - v0) becomes more negative
        # Dividing by negative a1 gives positive increase in t1
        # So t1 should increase when v0 increases
        if t1_perturbed <= t1_original:
            self._log(f"Monotonicity failed: t1 did not increase when v0 increased. Original: {t1_original:.3f}, Perturbed: {t1_perturbed:.3f}", False)
            success = False
        else:
            self._log(f"Monotonicity: t1 increases from {t1_original:.3f} to {t1_perturbed:.3f} when v0 increases")
        
        # Additional check: l0 should increase when v0 increases
        # l0 = x1 + x2, x1 depends on v0
        # x1_original = (v0 + v)/2 * t1
        # For perturbed: x1_perturbed = (v0_perturbed + v)/2 * t1_perturbed
        x1_perturbed = (v0_perturbed + self.v) / 2 * t1_perturbed
        x2_perturbed = self.x2  # x2 doesn't depend on v0 (depends on v and t2, which depends on a2)
        l0_perturbed = x1_perturbed + x2_perturbed
        
        if l0_perturbed <= self.l0:
            self._log(f"Monotonicity failed: l0 did not increase when v0 increased. Original: {self.l0:.3f}, Perturbed: {l0_perturbed:.3f}", False)
            success = False
        else:
            self._log(f"Monotonicity: l0 increases from {self.l0:.3f} to {l0_perturbed:.3f} when v0 increases")
        
        return success
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return overall success and logs."""
        self.logs = []
        overall_success = True
        
        tests = [
            ("Physics Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for test_name, test_func in tests:
            try:
                self.logs.append(f"=== Running {test_name} ===")
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"Test '{test_name}' crashed: {e}\n{traceback.format_exc()}", False)
                overall_success = False
        
        # Final summary
        if overall_success:
            self.logs.append("=== ALL TESTS PASSED ===")
        else:
            self.logs.append("=== SOME TESTS FAILED ===")
        
        return overall_success, self.logs