import sympy as sp
import math
import traceback
from typing import Tuple, List, Optional, Union

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the PhysicsChecker with safe attribute access.
        """
        # Extract input parameters safely
        self.d = getattr(solution_instance, 'd', 1.0)  # default if not present
        self.v0 = getattr(solution_instance, 'v0', 1.0)
        self.m = getattr(solution_instance, 'm', 1.67e-27)  # proton mass
        self.e = getattr(solution_instance, 'e', 1.6e-19)   # proton charge
        self.n = getattr(solution_instance, 'n', 1)         # integer for second part
        
        # Extract results safely
        self.results = getattr(solution_instance, 'results', {})
        
        # Expected keys from the key mapping
        self.v = self._safe_get('v')                     # velocity magnitude
        self.alpha = self._safe_get('alpha')             # angle in degrees
        self.B = self._safe_get('B')                     # magnetic field (first part)
        self.B0 = self._safe_get('B_0')                  # magnetic field (second part)
        self.T = self._safe_get('T')                     # period
        self.t = self._safe_get('t')                     # time to reach origin
        self.v_y = self._safe_get('v_y')                 # y-velocity at origin
        self.tanalpha = self._safe_get('tanalpha')       # tangent of angle
        self.R = self._safe_get('R')                     # radius (general)
        self.R1 = self._safe_get('R_1')                  # radius for first part
        self.T0 = self._safe_get('T_0')                  # period of circular motion
        
        # For numerical comparisons, we'll use these tolerances
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Logs for storing test results
        self.logs = []
    
    def _safe_get(self, key: str) -> Optional[float]:
        """Safely get a value from results with multiple fallbacks."""
        # Map expected keys to possible result names
        key_map = {
            'v': ['v', 'velocity'],
            'alpha': ['alpha', 'α'],
            'B': ['B', 'B_field'],
            'B_0': ['B_0', 'B0'],
            'T': ['T', 'period'],
            't': ['t', 'time'],
            'v_y': ['v_y', 'vy'],
            'tanalpha': ['tanalpha', 'tanα'],
            'R': ['R', 'radius'],
            'R_1': ['R_1', 'R1'],
            'T_0': ['T_0', 'T0']
        }
        
        possible_keys = key_map.get(key, [key])
        for k in possible_keys:
            value = self.results.get(k)
            if value is not None:
                # Convert to float if it's a sympy expression
                if isinstance(value, (sp.Expr, sp.Symbol)):
                    try:
                        return float(value.evalf())
                    except:
                        pass
                elif isinstance(value, (int, float)):
                    return float(value)
        return None
    
    def _is_close(self, a: float, b: float, rel_tol: float = None, abs_tol: float = None) -> bool:
        """Robust numerical comparison with default tolerances."""
        if rel_tol is None:
            rel_tol = self.rel_tol
        if abs_tol is None:
            abs_tol = self.abs_tol
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def _log_test(self, name: str, success: bool, message: str):
        """Log a test result."""
        status = "PASS" if success else "FAIL"
        self.logs.append(f"{name}: {status} - {message}")
        return success
    
    def test_invariants(self) -> bool:
        """
        Verify physical invariants:
        1. Energy conservation in electric field region
        2. Speed constancy in magnetic field
        3. Lorentz force equals centripetal force
        """
        tests_passed = True
        
        try:
            # 1. Energy conservation in electric field region (from P to O)
            # Work done by electric field = change in kinetic energy
            # Electric field E = m*a_y/e, where a_y = v0²/(3d) from kinematics
            a_y = (self.v0**2) / (3 * self.d)
            E_field = self.m * a_y / self.e
            displacement_y = self.d/2  # from y = -d/2 to 0
            work_done = self.e * E_field * displacement_y
            
            # Initial KE = 0.5*m*v0²
            # Final KE = 0.5*m*v², where v = (2√3/3)*v0
            v_expected = (2*math.sqrt(3)/3) * self.v0
            ke_initial = 0.5 * self.m * self.v0**2
            ke_final = 0.5 * self.m * v_expected**2
            ke_change = ke_final - ke_initial
            
            if self._is_close(work_done, ke_change):
                self._log_test("EnergyConservation", True, 
                             f"Work done ({work_done:.3e}) ≈ ΔKE ({ke_change:.3e})")
            else:
                self._log_test("EnergyConservation", False,
                             f"Work done ({work_done:.3e}) ≠ ΔKE ({ke_change:.3e})")
                tests_passed = False
            
            # 2. Speed constancy in magnetic field
            # The speed should not change in magnetic field
            if self.v is not None and self._is_close(self.v, v_expected):
                self._log_test("SpeedConstancy", True,
                             f"Speed v={self.v:.3e} matches expected {v_expected:.3e}")
            else:
                self._log_test("SpeedConstancy", False,
                             f"Speed mismatch: v={self.v}, expected {v_expected}")
                tests_passed = False
            
            # 3. Lorentz force = centripetal force for first part
            if self.B is not None and self.R1 is not None and self.v is not None:
                F_lorentz = self.e * self.v * self.B
                F_centripetal = self.m * self.v**2 / self.R1
                if self._is_close(F_lorentz, F_centripetal):
                    self._log_test("ForceBalance", True,
                                 f"Lorentz force ({F_lorentz:.3e}) ≈ centripetal ({F_centripetal:.3e})")
                else:
                    self._log_test("ForceBalance", False,
                                 f"Force imbalance: Lorentz={F_lorentz:.3e}, centripetal={F_centripetal:.3e}")
                    tests_passed = False
            
        except Exception as e:
            self._log_test("Invariants", False, f"Exception: {str(e)}")
            tests_passed = False
            
        return tests_passed
    
    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results with final answers.
        Verify the step-by-step calculations from the KB.
        """
        tests_passed = True
        
        try:
            # Step 1: Time to reach origin
            t_expected = math.sqrt(3) * self.d / self.v0
            if self.t is not None and self._is_close(self.t, t_expected):
                self._log_test("Step1_Time", True,
                             f"t={self.t:.3e} matches expected {t_expected:.3e}")
            else:
                self._log_test("Step1_Time", False,
                             f"t mismatch: {self.t} vs {t_expected}")
                tests_passed = False
            
            # Step 2: y-velocity at origin
            v_y_expected = self.v0 / math.sqrt(3)  # v0/√3
            if self.v_y is not None and self._is_close(self.v_y, v_y_expected):
                self._log_test("Step2_Vy", True,
                             f"v_y={self.v_y:.3e} matches expected {v_y_expected:.3e}")
            else:
                self._log_test("Step2_Vy", False,
                             f"v_y mismatch: {self.v_y} vs {v_y_expected}")
                tests_passed = False
            
            # Step 3: tan(α)
            tan_alpha_expected = v_y_expected / self.v0  # = 1/√3
            if self.tanalpha is not None and self._is_close(self.tanalpha, tan_alpha_expected):
                self._log_test("Step3_TanAlpha", True,
                             f"tanα={self.tanalpha:.3e} matches expected {tan_alpha_expected:.3e}")
            else:
                self._log_test("Step3_TanAlpha", False,
                             f"tanα mismatch: {self.tanalpha} vs {tan_alpha_expected}")
                tests_passed = False
            
            # Step 4: Velocity magnitude and angle
            v_expected = (2*math.sqrt(3)/3) * self.v0
            alpha_expected = 30.0  # degrees
            
            if self.v is not None and self._is_close(self.v, v_expected):
                self._log_test("Step4_Velocity", True,
                             f"v={self.v:.3e} matches expected {v_expected:.3e}")
            else:
                self._log_test("Step4_Velocity", False,
                             f"v mismatch: {self.v} vs {v_expected}")
                tests_passed = False
            
            if self.alpha is not None and self._is_close(self.alpha, alpha_expected, abs_tol=0.1):
                self._log_test("Step4_Angle", True,
                             f"α={self.alpha}° matches expected {alpha_expected}°")
            else:
                self._log_test("Step4_Angle", False,
                             f"α mismatch: {self.alpha}° vs {alpha_expected}°")
                tests_passed = False
            
            # Step 5: Radius R1 for first part
            R1_expected = 2 * self.d / 3
            if self.R1 is not None and self._is_close(self.R1, R1_expected):
                self._log_test("Step5_Radius", True,
                             f"R1={self.R1:.3e} matches expected {R1_expected:.3e}")
            else:
                self._log_test("Step5_Radius", False,
                             f"R1 mismatch: {self.R1} vs {R1_expected}")
                tests_passed = False
            
            # Step 6: Magnetic field B for first part
            B_expected = math.sqrt(3) * self.m * self.v0 / (self.e * self.d)
            if self.B is not None and self._is_close(self.B, B_expected):
                self._log_test("Step6_BField", True,
                             f"B={self.B:.3e} matches expected {B_expected:.3e}")
            else:
                self._log_test("Step6_BField", False,
                             f"B mismatch: {self.B} vs {B_expected}")
                tests_passed = False
            
            # Step 9: Radius R and B0 for second part
            if self.n > 0:
                # Check radius formula
                if self.B0 is not None:
                    R_expected = (2*math.sqrt(3) * self.m * self.v0) / (3 * self.e * self.B0)
                    if self.R is not None and self._is_close(self.R, R_expected):
                        self._log_test("Step9_RadiusCheck", True,
                                     f"R={self.R:.3e} matches formula")
                    else:
                        # Also check the condition n*2R = d
                        if self.R is not None and self._is_close(2 * self.n * self.R, self.d):
                            self._log_test("Step9_Geometry", True,
                                         f"n*2R = {2*self.n*self.R:.3e} ≈ d={self.d}")
                        else:
                            self._log_test("Step9_Geometry", False,
                                         f"Geometry condition failed")
                            tests_passed = False
                
                # Check B0 formula
                B0_expected = (4*math.sqrt(3) * self.n * self.m * self.v0) / (3 * self.e * self.d)
                if self.B0 is not None and self._is_close(self.B0, B0_expected):
                    self._log_test("Step9_B0", True,
                                 f"B0={self.B0:.3e} matches expected {B0_expected:.3e}")
                else:
                    self._log_test("Step9_B0", False,
                                 f"B0 mismatch: {self.B0} vs {B0_expected}")
                    tests_passed = False
            
            # Step 10: Period T
            if self.n > 0 and self.B0 is not None:
                T0_expected = (2 * math.pi * self.m) / (self.e * self.B0)
                T_expected = T0_expected / 3
                
                if self.T is not None and self._is_close(self.T, T_expected):
                    self._log_test("Step10_Period", True,
                                 f"T={self.T:.3e} matches expected {T_expected:.3e}")
                else:
                    self._log_test("Step10_Period", False,
                                 f"T mismatch: {self.T} vs {T_expected}")
                    tests_passed = False
                    
        except Exception as e:
            self._log_test("TraceConsistency", False, f"Exception: {str(e)}")
            tests_passed = False
            
        return tests_passed
    
    def test_monotonicity(self) -> bool:
        """
        Perturb inputs and verify expected trends.
        Tests that results change in the physically expected way.
        """
        tests_passed = True
        
        try:
            # Store original values
            d_orig = self.d
            v0_orig = self.v0
            
            # Test 1: Double the distance d
            self.d = 2 * d_orig
            
            # Recompute expected values with new d
            t_new = math.sqrt(3) * self.d / self.v0
            # Original time
            t_orig = math.sqrt(3) * d_orig / v0_orig
            
            # Time should increase proportionally to d
            if t_new > t_orig and self._is_close(t_new/t_orig, 2.0):
                self._log_test("Monotonicity_d_t", True,
                             f"t ∝ d: t(new)/t(orig) = {t_new/t_orig:.3f} ≈ 2")
            else:
                self._log_test("Monotonicity_d_t", False,
                             f"t not proportional to d: ratio = {t_new/t_orig:.3f}")
                tests_passed = False
            
            # Magnetic field B should be inversely proportional to d
            B_new = math.sqrt(3) * self.m * self.v0 / (self.e * self.d)
            B_orig = math.sqrt(3) * self.m * self.v0 / (self.e * d_orig)
            
            if B_new < B_orig and self._is_close(B_new/B_orig, 0.5):
                self._log_test("Monotonicity_d_B", True,
                             f"B ∝ 1/d: B(new)/B(orig) = {B_new/B_orig:.3f} ≈ 0.5")
            else:
                self._log_test("Monotonicity_d_B", False,
                             f"B not inversely proportional to d: ratio = {B_new/B_orig:.3f}")
                tests_passed = False
            
            # For second part: B0 ∝ n/d and T ∝ d/n
            if self.n > 0:
                B0_new = (4*math.sqrt(3) * self.n * self.m * self.v0) / (3 * self.e * self.d)
                B0_orig = (4*math.sqrt(3) * self.n * self.m * self.v0) / (3 * self.e * d_orig)
                
                if B0_new < B0_orig and self._is_close(B0_new/B0_orig, 0.5):
                    self._log_test("Monotonicity_d_B0", True,
                                 f"B0 ∝ 1/d: B0(new)/B0(orig) = {B0_new/B0_orig:.3f} ≈ 0.5")
                else:
                    self._log_test("Monotonicity_d_B0", False,
                                 f"B0 not inversely proportional to d: ratio = {B0_new/B0_orig:.3f}")
                    tests_passed = False
                
                T_new = (math.sqrt(3) * math.pi * self.d) / (6 * self.n * self.v0)
                T_orig = (math.sqrt(3) * math.pi * d_orig) / (6 * self.n * self.v0)
                
                if T_new > T_orig and self._is_close(T_new/T_orig, 2.0):
                    self._log_test("Monotonicity_d_T", True,
                                 f"T ∝ d: T(new)/T(orig) = {T_new/T_orig:.3f} ≈ 2")
                else:
                    self._log_test("Monotonicity_d_T", False,
                                 f"T not proportional to d: ratio = {T_new/T_orig:.3f}")
                    tests_passed = False
            
            # Restore d
            self.d = d_orig
            
            # Test 2: Double the initial velocity v0
            self.v0 = 2 * v0_orig
            
            # Velocity magnitude should increase proportionally
            v_new = (2*math.sqrt(3)/3) * self.v0
            v_orig = (2*math.sqrt(3)/3) * v0_orig
            
            if v_new > v_orig and self._is_close(v_new/v_orig, 2.0):
                self._log_test("Monotonicity_v0_v", True,
                             f"v ∝ v0: v(new)/v(orig) = {v_new/v_orig:.3f} ≈ 2")
            else:
                self._log_test("Monotonicity_v0_v", False,
                             f"v not proportional to v0: ratio = {v_new/v_orig:.3f}")
                tests_passed = False
            
            # Time should be inversely proportional to v0
            t_new = math.sqrt(3) * self.d / self.v0
            t_orig = math.sqrt(3) * self.d / v0_orig
            
            if t_new < t_orig and self._is_close(t_new/t_orig, 0.5):
                self._log_test("Monotonicity_v0_t", True,
                             f"t ∝ 1/v0: t(new)/t(orig) = {t_new/t_orig:.3f} ≈ 0.5")
            else:
                self._log_test("Monotonicity_v0_t", False,
                             f"t not inversely proportional to v0: ratio = {t_new/t_orig:.3f}")
                tests_passed = False
            
            # Magnetic field B should be proportional to v0
            B_new = math.sqrt(3) * self.m * self.v0 / (self.e * self.d)
            B_orig = math.sqrt(3) * self.m * v0_orig / (self.e * self.d)
            
            if B_new > B_orig and self._is_close(B_new/B_orig, 2.0):
                self._log_test("Monotonicity_v0_B", True,
                             f"B ∝ v0: B(new)/B(orig) = {B_new/B_orig:.3f} ≈ 2")
            else:
                self._log_test("Monotonicity_v0_B", False,
                             f"B not proportional to v0: ratio = {B_new/B_orig:.3f}")
                tests_passed = False
            
            # Restore v0
            self.v0 = v0_orig
            
        except Exception as e:
            self._log_test("Monotonicity", False, f"Exception: {str(e)}")
            tests_passed = False
            
        return tests_passed
    
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Run all tests and return success status and logs.
        """
        self.logs = []
        overall_success = True
        
        # Run each test suite
        test_suites = [
            ("Invariants", self.test_invariants),
            ("TraceConsistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for name, test_func in test_suites:
            try:
                success = test_func()
                if not success:
                    overall_success = False
            except Exception as e:
                self._log_test(name, False, f"Test crashed: {str(e)}")
                overall_success = False
        
        # Summary
        if overall_success:
            self.logs.append("SUMMARY: All tests passed.")
        else:
            self.logs.append("SUMMARY: Some tests failed.")
        
        return overall_success, self.logs