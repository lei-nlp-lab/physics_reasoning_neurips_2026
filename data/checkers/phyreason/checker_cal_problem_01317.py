import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access: never directly access solution_instance.attribute
        self.results = getattr(solution_instance, 'results', {})
        self.solution = solution_instance
        
        # Extract inputs with safe defaults
        self.v0 = getattr(solution_instance, 'v0', 2.0)
        self.s = getattr(solution_instance, 's', 30.0)
        self.mu1 = getattr(solution_instance, 'mu1', 0.008)
        self.mu2 = getattr(solution_instance, 'mu2', 0.004)
        self.g = getattr(solution_instance, 'g', 10.0)
        
        # Extract results from KB
        self.a = self.results.get('a', 0.0)
        self.t = self.results.get('t', 0.0)
        self.d = self.results.get('d', 0.0)
        
        self.logs = []
        
    def log(self, msg):
        self.logs.append(msg)
    
    def test_invariants(self):
        """Verify fundamental physical laws"""
        try:
            # Test 1: Verify Newton's 2nd Law for acceleration without brushing
            a_expected = self.mu1 * self.g
            if not math.isclose(self.a, a_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Newton's 2nd Law mismatch. Expected a={a_expected:.4f}, got a={self.a:.4f}")
                return False
            self.log(f"PASS: Newton's 2nd Law: a = μ₁g = {a_expected:.4f} m/s²")
            
            # Test 2: Verify work-energy theorem for the entire motion
            # Work done by friction should equal initial kinetic energy
            KE_initial = 0.5 * self.v0**2  # mass cancels
            work_friction = self.mu2 * self.g * self.d + self.mu1 * self.g * (self.s - self.d)
            
            if not math.isclose(KE_initial, work_friction, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Work-Energy Theorem mismatch. KE_initial={KE_initial:.4f}, work_friction={work_friction:.4f}")
                return False
            self.log(f"PASS: Work-Energy Theorem: KE_initial = work_friction = {KE_initial:.4f} J/kg")
            
            # Test 3: Verify time calculation from kinematics
            # Time to stop without brushing: t = v0 / a
            t_expected = self.v0 / self.a if abs(self.a) > 1e-12 else float('inf')
            if not math.isclose(self.t, t_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Kinematics mismatch. Expected t={t_expected:.4f}, got t={self.t:.4f}")
                return False
            self.log(f"PASS: Kinematics: t = v₀/a = {t_expected:.4f} s")
            
            # Test 4: Verify d is within physical bounds [0, s]
            if not (0 <= self.d <= self.s):
                self.log(f"FAIL: Brushed distance d={self.d} is outside bounds [0, {self.s}]")
                return False
            self.log(f"PASS: Brushed distance d={self.d} m is within bounds")
            
            return True
            
        except Exception as e:
            self.log(f"ERROR in test_invariants: {traceback.format_exc()}")
            return False
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer"""
        try:
            # Test 1: Check consistency of acceleration with step_1
            a_step1 = self.mu1 * self.g
            if not math.isclose(self.a, a_step1, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Acceleration a={self.a:.4f} doesn't match step_1 value {a_step1:.4f}")
                return False
            self.log(f"PASS: Acceleration consistent with step_1")
            
            # Test 2: Check consistency of time with step_2
            t_step2 = self.v0 / self.a
            if not math.isclose(self.t, t_step2, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Time t={self.t:.4f} doesn't match step_2 value {t_step2:.4f}")
                return False
            self.log(f"PASS: Time consistent with step_2")
            
            # Test 3: Check consistency of d with step_3 equation
            # From work-energy: μ₁mg(s-d) + μ₂mgd = ½mv₀²
            lhs = self.mu1 * self.g * (self.s - self.d) + self.mu2 * self.g * self.d
            rhs = 0.5 * self.v0**2
            
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"FAIL: Step_3 equation mismatch. LHS={lhs:.4f}, RHS={rhs:.4f}")
                return False
            self.log(f"PASS: Brushed distance d consistent with work-energy theorem")
            
            return True
            
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            return False
    
    def test_monotonicity(self):
        """Perturb input parameters and check physical trends - CORRECTED VERSION"""
        try:
            original_d = self.d
            
            # CORRECTED FORMULA for d from work-energy theorem:
            # μ₁g(s-d) + μ₂gd = ½v₀²
            # => d = (½v₀² - μ₁gs) / (g(μ₂ - μ₁))
            
            # Helper function to compute d correctly
            def compute_d(v0_val, mu1_val, mu2_val, g_val, s_val):
                numerator = 0.5 * v0_val**2 - mu1_val * g_val * s_val
                denominator = g_val * (mu2_val - mu1_val)
                if abs(denominator) < 1e-12:
                    return float('inf')
                return numerator / denominator
            
            # Test 1: If μ₂ decreases (brushing becomes even less effective), d should DECREASE
            perturbed_mu2 = self.mu2 * 0.8  # 20% decrease
            d_new = compute_d(self.v0, self.mu1, perturbed_mu2, self.g, self.s)
            
            if not (d_new < original_d):
                self.log(f"FAIL: When μ₂ decreases, d should decrease. Got d_new={d_new:.4f}, original={original_d:.4f}")
                return False
            self.log(f"PASS: Monotonicity 1: d decreases when μ₂ decreases")
            
            # Test 2: If μ₁ increases (more friction without brushing), d should INCREASE
            perturbed_mu1 = self.mu1 * 1.2  # 20% increase
            d_new2 = compute_d(self.v0, perturbed_mu1, self.mu2, self.g, self.s)
            
            if not (d_new2 > original_d):
                self.log(f"FAIL: When μ₁ increases, d should increase. Got d_new={d_new2:.4f}, original={original_d:.4f}")
                return False
            self.log(f"PASS: Monotonicity 2: d increases when μ₁ increases")
            
            # Test 3: CORRECTED - If initial velocity increases (within physical limits), d should DECREASE
            # Maximum possible v0 for stopping in distance s with all high friction (no brushing):
            # 0.5*v0_max² = μ₁*g*s => v0_max = sqrt(2*μ₁*g*s)
            v0_max = math.sqrt(2 * self.mu1 * self.g * self.s)
            if self.v0 >= v0_max:
                self.log(f"SKIP: Original v0={self.v0} is already at or above maximum for stopping in {self.s}m")
                return True
                
            # Use 5% increase instead of 20% to ensure we stay physically possible
            perturbed_v0 = self.v0 * 1.05
            if perturbed_v0 >= v0_max:
                perturbed_v0 = v0_max * 0.99  # Just below the maximum
                
            d_new3 = compute_d(perturbed_v0, self.mu1, self.mu2, self.g, self.s)
            
            # CORRECTED EXPECTATION: d should DECREASE when v0 increases
            if not (d_new3 < original_d):
                self.log(f"FAIL: When v₀ increases (within physical limits), d should decrease. Got d_new={d_new3:.4f}, original={original_d:.4f}")
                return False
            self.log(f"PASS: Monotonicity 3: d decreases when v₀ increases (within physical limits)")
            
            # Restore original value
            self.d = original_d
            
            return True
            
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            return False
    
    def run_all_tests(self):
        """Execute all tests and return success flag with logs"""
        self.logs = []
        success = True
        
        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity Tests", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            self.log(f"\n--- Running {test_name} ---")
            try:
                if not test_func():
                    success = False
                    self.log(f"{test_name} FAILED")
                else:
                    self.log(f"{test_name} PASSED")
            except Exception as e:
                success = False
                self.log(f"{test_name} ERROR: {traceback.format_exc()}")
        
        # Final summary
        self.log(f"\n=== Overall Result: {'PASS' if success else 'FAIL'} ===")
        return success, self.logs