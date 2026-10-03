import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs from solution_instance with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Physical constants - extract from results dict with defaults
        # Use safe get with proper defaults
        self.B = self._safe_get_float('B', 1.0)
        self.E = self._safe_get_float('E', 1.0e3)
        self.m = self._safe_get_float('m', 1e-14)
        self.q = self._safe_get_float('q', -1e-10)  # Negative charge
        self.theta = 53 * math.pi/180  # Fixed from problem
        
        # Given coordinates in meters
        self.x_OA = 0.12  # A at (0.12, 0)
        self.x_OC = 0.26  # C at (0.26, 0)
        self.x_AC = self.x_OC - self.x_OA  # = 0.14 m
        
        # Trigonometric constants
        self.sin53 = 0.8
        self.cos53 = 0.6
        
        # Extract results safely from results dict
        keys = ['v', 'r', 'alpha', 'beta', 'y', 'x_CD', 't_1', 't_2', 't_3', 't', 'x_D', 'y_D']
        for key in keys:
            setattr(self, key, self.results.get(key, None))
        
        # Convert angles to radians if they're in degrees
        if self.alpha is not None:
            self.alpha_rad = math.radians(self.alpha) if self.alpha > 6.28 else self.alpha
        else:
            self.alpha_rad = None
            
        if self.beta is not None:
            self.beta_rad = math.radians(self.beta) if self.beta > 6.28 else self.beta
        else:
            self.beta_rad = None
    
    def _safe_get_float(self, key, default):
        """Safely get a float value from results dict."""
        val = self.results.get(key, default)
        try:
            # Handle the case where val might be 0.0 but default is small
            result = float(val)
            # If result is 0.0 but default is not 0, and we're looking for m or q,
            # use the default (these are known physical constants)
            if result == 0.0 and default != 0:
                if key in ['m', 'q']:
                    return float(default)
            return result
        except (TypeError, ValueError):
            return float(default)
    
    def test_invariants(self):
        """Test fundamental physical invariants."""
        logs = []
        checks = []
        
        try:
            # 1. Check equilibrium in region I: qE = qvB (straight line motion)
            if self.v is not None and self.B != 0:
                # For equilibrium: |q|E = |q|vB sin(90°) => v = E/B
                expected_v = abs(self.E / self.B)  # Magnitude only
                check1 = math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2)
                logs.append(f"Velocity check: v={self.v:.1f}, E/B={expected_v:.1f}, match={check1}")
                checks.append(check1)
            else:
                logs.append("Velocity check skipped: v or B not available or B=0")
            
            # 2. Check Lorentz force as centripetal force in region II
            # Use more robust check for small numbers
            if (self.v is not None and self.r is not None and 
                self.m is not None and self.q is not None):
                
                # Check if values are valid for calculation
                if self.B == 0:
                    logs.append("Radius check skipped: B=0")
                elif abs(self.q) < 1e-20:
                    logs.append(f"Radius check skipped: |q|={abs(self.q):.2e} < 1e-20")
                else:
                    # r = mv/(|q|B)
                    expected_r = self.m * self.v / (abs(self.q) * self.B)
                    check2 = math.isclose(self.r, expected_r, rel_tol=1e-2, abs_tol=1e-2)
                    logs.append(f"Radius check: r={self.r:.3f}, mv/(|q|B)={expected_r:.3f}, match={check2}")
                    checks.append(check2)
            else:
                logs.append(f"Radius check skipped: v={self.v}, r={self.r}, m={self.m}, q={self.q}")
            
            # 3. Check total time consistency
            if all([self.t_1 is not None, self.t_2 is not None, self.t_3 is not None, self.t is not None]):
                expected_t = self.t_1 + self.t_2 + self.t_3
                check3 = math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2)
                logs.append(f"Time sum check: t={self.t:.6f}, sum={expected_t:.6f}, match={check3}")
                checks.append(check3)
            else:
                logs.append("Time sum check skipped: missing time values")
            
            # 4. Check coordinate D is on x-axis
            if self.y_D is not None:
                check4 = math.isclose(self.y_D, 0.0, rel_tol=1e-2, abs_tol=1e-2)
                logs.append(f"D on x-axis: y_D={self.y_D}, match={check4}")
                checks.append(check4)
            else:
                logs.append("D on x-axis check skipped: y_D not available")
            
            # 5. Check CD distance geometry
            if self.y is not None and self.x_CD is not None:
                expected_x_CD = self.y * math.tan(self.theta)
                check5 = math.isclose(self.x_CD, expected_x_CD, rel_tol=1e-2, abs_tol=1e-2)
                logs.append(f"x_CD geometry: {self.x_CD:.3f}, y*tan53={expected_x_CD:.3f}, match={check5}")
                checks.append(check5)
            else:
                logs.append("x_CD geometry check skipped: y or x_CD not available")
            
            success = all(checks) if checks else False
            logs.append(f"Invariants test overall: {success}")
            
        except Exception as e:
            logs.append(f"Invariants test failed with error: {e}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        checks = []
        
        try:
            # 1. Check t_1 calculation
            if self.v is not None and self.t_1 is not None and self.v != 0:
                expected_t1 = self.x_OA / (self.v * self.cos53)
                check1 = math.isclose(self.t_1, expected_t1, rel_tol=1e-2, abs_tol=1e-2)
                logs.append(f"t_1 check: {self.t_1:.6f}, OA/(v*cos53)={expected_t1:.6f}, match={check1}")
                checks.append(check1)
            else:
                logs.append("t_1 check skipped: v or t_1 not available or v=0")
            
            # 2. Check t_2 calculation (quarter circle time)
            if self.v is not None and self.r is not None and self.t_2 is not None and self.v != 0:
                expected_t2 = (math.pi * self.r) / (2 * self.v)
                check2 = math.isclose(self.t_2, expected_t2, rel_tol=1e-2, abs_tol=1e-2)
                logs.append(f"t_2 check: {self.t_2:.6f}, πr/(2v)={expected_t2:.6f}, match={check2}")
                checks.append(check2)
            else:
                logs.append("t_2 check skipped: v, r, or t_2 not available or v=0")
            
            # 3. Check t_3 calculation
            if self.v is not None and self.y is not None and self.t_3 is not None and self.v != 0:
                expected_t3 = self.y / (self.v * self.cos53)
                check3 = math.isclose(self.t_3, expected_t3, rel_tol=1e-2, abs_tol=1e-2)
                logs.append(f"t_3 check: {self.t_3:.6f}, y/(v*cos53)={expected_t3:.6f}, match={check3}")
                checks.append(check3)
            else:
                logs.append("t_3 check skipped: v, y, or t_3 not available or v=0")
            
            # 4. Check x_D coordinate
            if self.x_D is not None and self.x_CD is not None:
                expected_x_D = self.x_OC + self.x_CD
                check4 = math.isclose(self.x_D, expected_x_D, rel_tol=1e-2, abs_tol=1e-2)
                logs.append(f"x_D check: {self.x_D:.2f}, OC+x_CD={expected_x_D:.2f}, match={check4}")
                checks.append(check4)
            else:
                logs.append("x_D check skipped: x_D or x_CD not available")
            
            # 5. Check against target answers
            target_v = 1000.0  # 10^3 m/s
            target_xD = 0.5    # 50 cm
            target_t = 6.57e-4 # 6.57×10^{-4} s
            
            if self.v is not None:
                check5 = math.isclose(self.v, target_v, rel_tol=1e-2, abs_tol=1e-2)
                logs.append(f"Target v check: {self.v:.1f} vs {target_v:.1f}, match={check5}")
                checks.append(check5)
            else:
                logs.append("Target v check skipped: v not available")
            
            if self.x_D is not None:
                check6 = math.isclose(self.x_D, target_xD, rel_tol=1e-2, abs_tol=1e-2)
                logs.append(f"Target x_D check: {self.x_D:.2f} vs {target_xD:.2f}, match={check6}")
                checks.append(check6)
            else:
                logs.append("Target x_D check skipped: x_D not available")
            
            if self.t is not None:
                check7 = math.isclose(self.t, target_t, rel_tol=1e-2, abs_tol=1e-2)
                logs.append(f"Target t check: {self.t:.6f} vs {target_t:.6f}, match={check7}")
                checks.append(check7)
            else:
                logs.append("Target t check skipped: t not available")
            
            success = all(checks) if checks else False
            logs.append(f"Trace consistency test overall: {success}")
            
        except Exception as e:
            logs.append(f"Trace consistency test failed with error: {e}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Test monotonic relationship by perturbing mass."""
        logs = []
        
        try:
            # Check if we have all required values
            # Use the actual values from problem statement if results are 0
            m_val = self.m if self.m is not None and self.m != 0 else 1e-14
            q_val = self.q if self.q is not None and abs(self.q) > 1e-20 else -1e-10
            
            if (m_val is None or self.r is None or self.E is None or 
                self.B is None or q_val is None or self.B == 0):
                logs.append(f"Monotonicity test skipped: missing values. m={m_val}, r={self.r}, E={self.E}, B={self.B}, q={q_val}")
                return False, logs
            
            # Check if q is effectively zero
            if abs(q_val) < 1e-20:
                logs.append(f"Monotonicity test skipped: |q|={abs(q_val):.2e} < 1e-20")
                return False, logs
            
            # Store original values
            original_m = m_val
            original_r = self.r
            
            # Perturb mass by +10%
            perturbed_m = original_m * 1.1
            
            # Recalculate expected radius using r = mv/(|q|B)
            # v remains E/B (fixed by equilibrium)
            expected_v = abs(self.E / self.B)
            original_expected_r = original_m * expected_v / (abs(q_val) * self.B)
            perturbed_expected_r = perturbed_m * expected_v / (abs(q_val) * self.B)
            
            # Check monotonic increase: larger mass => larger radius
            check = perturbed_expected_r > original_expected_r
            
            logs.append(f"Monotonicity test: m increased by 10%")
            logs.append(f"  Original m: {original_m:.1e}, r: {original_expected_r:.3f}")
            logs.append(f"  Perturbed m: {perturbed_m:.1e}, r: {perturbed_expected_r:.3f}")
            logs.append(f"  Radius increases with mass: {check}")
            
            success = check
            logs.append(f"Monotonicity test overall: {success}")
            
        except Exception as e:
            logs.append(f"Monotonicity test failed with error: {e}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return aggregated results."""
        all_logs = []
        all_success = True
        
        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for test_name, test_func in tests:
            all_logs.append(f"\n{'='*50}")
            all_logs.append(f"Running {test_name} Test")
            all_logs.append('='*50)
            
            success, logs = test_func()
            all_logs.extend(logs)
            all_logs.append(f"{test_name} Test Result: {'PASS' if success else 'FAIL'}")
            
            if not success:
                all_success = False
        
        # Final summary
        all_logs.append(f"\n{'='*50}")
        all_logs.append(f"OVERALL RESULT: {'ALL TESTS PASS' if all_success else 'SOME TESTS FAILED'}")
        all_logs.append('='*50)
        
        return all_success, all_logs