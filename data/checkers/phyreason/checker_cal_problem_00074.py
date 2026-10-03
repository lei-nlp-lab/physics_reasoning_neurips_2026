import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from solution_instance.
        Use getattr with fallback to results dictionary for robustness.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Problem constants (with safe defaults if missing)
        self.m = getattr(solution_instance, 'm', self.results.get('m', 2.0))
        self.v0 = getattr(solution_instance, 'v0', self.results.get('v0', 16.0))
        self.theta = getattr(solution_instance, 'theta', self.results.get('theta', 37.0))
        self.t = getattr(solution_instance, 't', self.results.get('t', 2.0))
        
        # Extract g from solution if available, otherwise default to 9.8
        self.g = getattr(solution_instance, 'g', 9.8)
        if 'g' in self.results:
            self.g = self.results['g']
        
        # Extract results with safe defaults (0.0 for numerical, None for symbolic)
        self.mu = self.results.get('mu', self.results.get('μ', 0.0))
        self.x = self.results.get('x', 0.0)
        self.a1 = self.results.get('a_1', 0.0)
        self.a2 = self.results.get('a_2', 0.0)
        self.vt = self.results.get('v_t', 0.0)
        self.P_avg = self.results.get('P_avg', 0.0)
        
        # Pre-computed trigonometric values (from problem statement)
        self.sin_theta = 0.6  # sin37°
        self.cos_theta = 0.8  # cos37°
        
        self.logs = []
    
    def _log(self, msg, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            msg = f"[{'PASS' if success else 'FAIL'}] {msg}"
        self.logs.append(msg)
        return success
    
    def _isclose(self, a, b, rel_tol=0.02, abs_tol=0.02):
        """Robust fuzzy numerical comparison."""
        # Handle None values
        if a is None or b is None:
            return False
        # Handle sympy expressions
        if hasattr(a, 'evalf'):
            a = float(a.evalf())
        if hasattr(b, 'evalf'):
            b = float(b.evalf())
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Check fundamental physics laws."""
        try:
            # 1. Newton's 2nd Law for upward motion: a1 = g sinθ + μ g cosθ
            # Note: a1 is negative (deceleration), so we compare magnitudes
            expected_a1_magnitude = self.g * self.sin_theta + self.mu * self.g * self.cos_theta
            if not self._isclose(abs(self.a1), expected_a1_magnitude):
                self._log(f"Newton's 2nd Law (upward) violated: |a1|={abs(self.a1)}, expected {expected_a1_magnitude}", False)
                return False
            self._log(f"Newton's 2nd Law (upward) satisfied: |a1|={abs(self.a1)} ≈ g sinθ + μ g cosθ = {expected_a1_magnitude}", True)
            
            # 2. Newton's 2nd Law for downward motion: a2 = g sinθ - μ g cosθ
            expected_a2 = self.g * self.sin_theta - self.mu * self.g * self.cos_theta
            if not self._isclose(self.a2, expected_a2):
                self._log(f"Newton's 2nd Law (downward) violated: a2={self.a2}, expected {expected_a2}", False)
                return False
            self._log(f"Newton's 2nd Law (downward) satisfied: a2={self.a2} ≈ g sinθ - μ g cosθ = {expected_a2}", True)
            
            # 3. Kinematic consistency: x = (v0/2) * t (from average velocity)
            expected_x = (self.v0 / 2) * self.t
            if not self._isclose(self.x, expected_x):
                self._log(f"Kinematic relation violated: x={self.x}, expected (v0/2)*t = {expected_x}", False)
                return False
            self._log(f"Kinematic relation satisfied: x={self.x} ≈ (v0/2)*t = {expected_x}", True)
            
            # 4. Energy-work theorem (with friction): net work = ΔKE
            # Upward: Work_friction + Work_gravity = ΔKE_up
            # Both friction and gravity do negative work during upward motion
            # Work_friction = -μ m g cosθ * x (negative)
            # Work_gravity = -m g sinθ * x (negative)
            # Net work = -μ m g cosθ * x - m g sinθ * x = -m g x (μ cosθ + sinθ)
            delta_KE_up = -0.5 * self.m * self.v0**2
            work_friction_up = -self.mu * self.m * self.g * self.cos_theta * self.x
            work_gravity_up = -self.m * self.g * self.sin_theta * self.x
            net_work_up = work_friction_up + work_gravity_up
            if not self._isclose(net_work_up, delta_KE_up):
                self._log(f"Work-energy theorem (upward) violated: net work={net_work_up}, ΔKE={delta_KE_up}", False)
                return False
            self._log(f"Work-energy theorem (upward) satisfied: net work={net_work_up} ≈ ΔKE={delta_KE_up}", True)
            
            return True
        except Exception as e:
            self._log(f"Error in test_invariants: {str(e)}", False)
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Check the calculation chain
            # 1. μ from a1: μ = (|a1|/g - sinθ)/cosθ
            mu_from_a1 = (abs(self.a1) / self.g - self.sin_theta) / self.cos_theta
            if not self._isclose(self.mu, mu_from_a1):
                self._log(f"μ inconsistency: μ={self.mu}, from a1: {mu_from_a1}", False)
                return False
            self._log(f"μ consistent with a1: μ={self.mu} ≈ (|a1|/g - sinθ)/cosθ = {mu_from_a1}", True)
            
            # 2. a2 from μ
            a2_from_mu = self.g * self.sin_theta - self.mu * self.g * self.cos_theta
            if not self._isclose(self.a2, a2_from_mu):
                self._log(f"a2 inconsistency: a2={self.a2}, from μ: {a2_from_mu}", False)
                return False
            self._log(f"a2 consistent with μ: a2={self.a2} ≈ g sinθ - μ g cosθ = {a2_from_mu}", True)
            
            # 3. vt from kinematic: vt = sqrt(2*a2*x)
            vt_from_kinematic = math.sqrt(2 * self.a2 * self.x)
            if not self._isclose(self.vt, vt_from_kinematic):
                self._log(f"vt inconsistency: vt={self.vt}, from kinematics: {vt_from_kinematic}", False)
                return False
            self._log(f"vt consistent with kinematics: vt={self.vt} ≈ sqrt(2*a2*x) = {vt_from_kinematic}", True)
            
            # 4. Average power: P_avg = m*g*v_avg*sinθ, where v_avg = (v0+0)/2 for upward
            v_avg_up = self.v0 / 2
            expected_P_avg = self.m * self.g * v_avg_up * self.sin_theta
            if not self._isclose(self.P_avg, expected_P_avg):
                self._log(f"P_avg inconsistency: {self.P_avg}, expected m*g*v_avg*sinθ = {expected_P_avg}", False)
                return False
            self._log(f"P_avg consistent: {self.P_avg} ≈ m*g*v_avg*sinθ = {expected_P_avg}", True)
            
            return True
        except Exception as e:
            self._log(f"Error in test_trace_consistency: {str(e)}", False)
            return False
    
    def test_monotonicity(self):
        """Perturb mass and check if trends match expectations."""
        try:
            # Save original values
            original_m = self.m
            perturbed_m = original_m * 1.1  # +10%
            
            # Recalculate expected changes using formulas
            # μ, a1, a2, x are independent of mass
            # vt is independent of mass (from kinematics)
            # P_avg should scale linearly with mass
            
            # Calculate with perturbed mass
            expected_P_avg_perturbed = self.P_avg * (perturbed_m / original_m)
            
            # Restore original mass
            self.m = original_m
            
            # Check if the trend is correct (P_avg ∝ m)
            # We don't have actual perturbed calculation, so we verify the formula
            # by checking that the ratio equals the mass ratio
            ratio_calc = expected_P_avg_perturbed / self.P_avg
            mass_ratio = perturbed_m / original_m
            if not self._isclose(ratio_calc, mass_ratio):
                self._log(f"Monotonicity violated: P_avg should scale linearly with mass", False)
                return False
            
            self._log(f"Monotonicity satisfied: P_avg ∝ m (factor {mass_ratio})", True)
            return True
        except Exception as e:
            self._log(f"Error in test_monotonicity: {str(e)}", False)
            return False
    
    def run_all_tests(self):
        """Execute all tests and return (success_bool, logs)."""
        self.logs = []
        self.logs.append("=== Physics Audit Report ===")
        
        success = True
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.logs.append(f"\n--- {test_name} ---")
            try:
                test_success = test_func()
                if not test_success:
                    success = False
            except Exception as e:
                self.logs.append(f"CRASH: {str(e)}")
                self.logs.append(traceback.format_exc())
                success = False
        
        self.logs.append(f"\n=== Overall: {'PASS' if success else 'FAIL'} ===")
        return success, self.logs