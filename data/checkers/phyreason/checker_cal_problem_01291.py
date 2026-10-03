import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        self.U = getattr(solution_instance, 'U', 0.0)
        
        # Extract results with safe defaults
        self.x = self.results.get('x', 0.0)
        self.y = self.results.get('y', 0.0)
        self.v0 = self.results.get('v_0', 0.0)
        self.t = self.results.get('t', 0.0)
        self.tantheta = self.results.get('tantheta', 0.0)
        
        # Given constants
        self.R = 0.75
        self.g = 10.0
        self.cos53 = 0.6  # cos53° = sin37° = 0.6
        self.sin53 = 0.8  # sin53° = cos37° = 0.8
        self.tan37 = 0.75  # tan37° = 0.6/0.8 = 0.75
        
        # Expected values from steps analysis
        self.expected_tantheta = 3/8  # 0.375
        self.expected_x = self.R * (1 + self.cos53)  # 1.2
        self.expected_y = self.expected_x * self.expected_tantheta  # 0.45
        self.expected_t = math.sqrt(2 * self.expected_y / self.g)  # 0.3
        self.expected_v0 = self.expected_x / self.expected_t  # 4.0
        
        # Heights for energy conservation
        self.h_C = self.R * self.sin53  # height of point C above ground
        self.h_B = self.h_C + self.expected_y  # height of point B above ground
        
        self.logs = []
        
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def _log_test(self, test_name, passed, message=""):
        self.logs.append(f"{test_name}: {'PASS' if passed else 'FAIL'} - {message}")
        return passed
    
    def test_step1_tantheta(self):
        """Check Step 1: tanθ = tan37°/2"""
        try:
            passed = self._is_close(self.tantheta, self.expected_tantheta)
            return self._log_test(
                "Step1_tantheta",
                passed,
                f"Got {self.tantheta:.3f}, expected {self.expected_tantheta:.3f}"
            )
        except Exception as e:
            return self._log_test("Step1_tantheta", False, f"Error: {str(e)}")
    
    def test_step2_geometry(self):
        """Check Step 2: Geometric relationships"""
        try:
            # Check x = R(1 + cos53°)
            passed_x = self._is_close(self.x, self.expected_x)
            self._log_test(
                "Step2_x",
                passed_x,
                f"Got {self.x:.3f}, expected {self.expected_x:.3f}"
            )
            
            # Check y = x * tanθ
            passed_y = self._is_close(self.y, self.expected_y)
            self._log_test(
                "Step2_y",
                passed_y,
                f"Got {self.y:.3f}, expected {self.expected_y:.3f}"
            )
            
            # Also check consistency: if tanθ is correct, y/x should match
            if abs(self.x) > 1e-9:
                ratio = self.y / self.x
                passed_ratio = self._is_close(ratio, self.expected_tantheta)
                self._log_test(
                    "Step2_y/x_ratio",
                    passed_ratio,
                    f"Got {ratio:.3f}, expected {self.expected_tantheta:.3f}"
                )
                return passed_x and passed_y and passed_ratio
            
            return passed_x and passed_y
        except Exception as e:
            return self._log_test("Step2_geometry", False, f"Error: {str(e)}")
    
    def test_step3_time(self):
        """Check Step 3: t = sqrt(2y/g)"""
        try:
            # Numerical check
            passed = self._is_close(self.t, self.expected_t)
            self._log_test(
                "Step3_time",
                passed,
                f"Got {self.t:.3f}, expected {self.expected_t:.3f}"
            )
            
            return passed
        except Exception as e:
            return self._log_test("Step3_time", False, f"Error: {str(e)}")
    
    def test_step4_velocity(self):
        """Check Step 4: v0 = x/t"""
        try:
            passed = self._is_close(self.v0, self.expected_v0)
            self._log_test(
                "Step4_velocity",
                passed,
                f"Got {self.v0:.3f}, expected {self.expected_v0:.3f}"
            )
            
            # Check relationship x = v0*t
            product = self.v0 * self.t
            passed_product = self._is_close(product, self.expected_x)
            self._log_test(
                "Step4_x=v0*t",
                passed_product,
                f"Got v0*t = {product:.3f}, expected x = {self.expected_x:.3f}"
            )
            
            return passed and passed_product
        except Exception as e:
            return self._log_test("Step4_velocity", False, f"Error: {str(e)}")
    
    def test_energy_conservation(self):
        """Check energy conservation from B to C - CORRECTED VERSION"""
        try:
            # Heights for energy conservation (using actual computed values)
            h_C = self.R * self.sin53  # height of point C above ground
            # y is vertical displacement from B to C, so height of B = h_C + y
            h_B = h_C + self.y
            
            # At point B: kinetic + potential energy (m=1 assumed)
            E_B = 0.5 * self.v0**2 + self.g * h_B
            
            # At point C: kinetic + potential energy
            v_x = self.v0  # horizontal component unchanged
            v_y = -self.g * self.t  # downward, since initial vertical velocity = 0
            speed_sq = v_x**2 + v_y**2
            E_C = 0.5 * speed_sq + self.g * h_C
            
            passed = self._is_close(E_B, E_C)
            return self._log_test(
                "Energy_conservation",
                passed,
                f"E_B={E_B:.3f} (KE={0.5*self.v0**2:.3f}+PE={self.g*h_B:.3f}), "
                f"E_C={E_C:.3f} (KE={0.5*speed_sq:.3f}+PE={self.g*h_C:.3f})"
            )
        except Exception as e:
            return self._log_test("Energy_conservation", False, f"Error: {str(e)}")
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answer"""
        try:
            # Check that using the calculated v0 and t gives back x and y
            x_from_v0 = self.v0 * self.t
            y_from_t = 0.5 * self.g * self.t**2  # for initial vertical velocity = 0
            
            passed_x = self._is_close(x_from_v0, self.x)
            passed_y = self._is_close(y_from_t, self.y)
            
            self._log_test(
                "Trace_x_consistency",
                passed_x,
                f"v0*t = {x_from_v0:.3f}, x = {self.x:.3f}"
            )
            self._log_test(
                "Trace_y_consistency",
                passed_y,
                f"0.5*g*t^2 = {y_from_t:.3f}, y = {self.y:.3f}"
            )
            
            return passed_x and passed_y
        except Exception as e:
            return self._log_test("Trace_consistency", False, f"Error: {str(e)}")
    
    def test_monotonicity(self):
        """Perturb R and check trends"""
        try:
            original_R = self.R
            
            # Test 1: Increase R by 10%
            perturbed_R = original_R * 1.1
            x_perturbed = perturbed_R * (1 + self.cos53)
            y_perturbed = x_perturbed * self.expected_tantheta
            t_perturbed = math.sqrt(2 * y_perturbed / self.g)
            v0_perturbed = x_perturbed / t_perturbed
            
            # Check monotonic relationships
            # x should increase with R
            passed_x_mono = x_perturbed > self.expected_x
            # y should increase with R (since y = x*tanθ)
            passed_y_mono = y_perturbed > self.expected_y
            # t should increase with R (since y increases)
            passed_t_mono = t_perturbed > self.expected_t
            # v0 relationship: v0 = x/t = x/√(2y/g) = x/√(2x*tanθ/g) = √(g*x/(2*tanθ))
            # So v0 should increase with √x, thus with R
            passed_v0_mono = v0_perturbed > self.expected_v0
            
            self._log_test(
                "Monotonicity_R_increase",
                passed_x_mono and passed_y_mono and passed_t_mono and passed_v0_mono,
                f"Perturbed: x={x_perturbed:.3f}, y={y_perturbed:.3f}, t={t_perturbed:.3f}, v0={v0_perturbed:.3f}"
            )
            
            return passed_x_mono and passed_y_mono and passed_t_mono and passed_v0_mono
        except Exception as e:
            return self._log_test("Monotonicity", False, f"Error: {str(e)}")
    
    def test_graze_condition(self):
        """Verify that velocity at C is tangent to circle"""
        try:
            # Velocity components at C
            v_x = self.v0
            v_y = -self.g * self.t  # downward
            
            # Position of C relative to O
            # From geometry: C is at angle 53° from horizontal
            C_x_rel = self.R * self.cos53  # = 0.75*0.6 = 0.45
            C_y_rel = self.R * self.sin53  # = 0.75*0.8 = 0.60
            
            # For tangent condition: velocity vector perpendicular to radius vector
            # Dot product should be zero: v·r = v_x*C_x_rel + v_y*C_y_rel = 0
            dot_product = v_x * C_x_rel + v_y * C_y_rel
            
            # Also check using slopes: v_y/v_x should equal negative reciprocal of C_y_rel/C_x_rel
            slope_v = v_y / v_x if abs(v_x) > 1e-9 else float('inf')
            slope_r = C_y_rel / C_x_rel if abs(C_x_rel) > 1e-9 else float('inf')
            
            # Check if slope_v * slope_r ≈ -1
            slope_check = self._is_close(slope_v * slope_r, -1.0)
            dot_check = self._is_close(dot_product, 0.0, abs_tol=0.1)
            
            return self._log_test(
                "Graze_condition",
                slope_check or dot_check,
                f"slope_v={slope_v:.3f}, slope_r={slope_r:.3f}, dot={dot_product:.3f}"
            )
        except Exception as e:
            return self._log_test("Graze_condition", False, f"Error: {str(e)}")
    
    def run_all_tests(self):
        """Run all physics checks and return (success, logs)"""
        self.logs = []
        all_passed = True
        
        try:
            tests = [
                self.test_step1_tantheta,
                self.test_step2_geometry,
                self.test_step3_time,
                self.test_step4_velocity,
                self.test_energy_conservation,
                self.test_trace_consistency,
                self.test_monotonicity,
                self.test_graze_condition
            ]
            
            for test in tests:
                if not test():
                    all_passed = False
            
            return all_passed, self.logs
            
        except Exception as e:
            self.logs.append(f"CRITICAL_ERROR: {traceback.format_exc()}")
            return False, self.logs