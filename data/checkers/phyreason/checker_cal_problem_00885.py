import math
import sympy as sp
from typing import Tuple, List, Optional

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract inputs with safe defaults
        self.m = getattr(solution_instance, 'm', self.results.get('m', 1.0))
        self.q = getattr(solution_instance, 'q', self.results.get('q', 1.0))
        self.v0 = getattr(solution_instance, 'v0', self.results.get('v0', 1.0))
        self.h = getattr(solution_instance, 'h', self.results.get('h', 1.0))
        self.L = getattr(solution_instance, 'L', self.results.get('L', 1.0))
        
        # Extract calculated results with safe defaults
        self.E = getattr(solution_instance, 'E', self.results.get('E', 0.0))
        self.v = getattr(solution_instance, 'v', self.results.get('v', 0.0))
        self.theta_deg = getattr(solution_instance, 'theta', self.results.get('theta', 0.0))
        self.B = getattr(solution_instance, 'B', self.results.get('B', 0.0))
        self.R = getattr(solution_instance, 'R', self.results.get('R', 0.0))
        self.v_y = getattr(solution_instance, 'v_y', self.results.get('v_y', 0.0))
        self.tan_theta = getattr(solution_instance, 'tan_theta', self.results.get('tantheta', 0.0))
        
        # Derived quantities
        self.theta_rad = math.radians(self.theta_deg)
        self.qE = self.q * self.E
        self.a_electric = self.qE / self.m  # Acceleration in electric field
        
        # Target answers for reference
        self.target_E = self.m * self.v0**2 / (2 * self.q * self.h)
        self.target_v = math.sqrt(2) * self.v0
        self.target_B = 2 * self.m * self.v0 / (self.q * self.L)
        
        # Sympy symbols for symbolic checks
        self.symbols = {
            'm': sp.symbols('m', positive=True),
            'q': sp.symbols('q', positive=True),
            'v0': sp.symbols('v0', positive=True),
            'h': sp.symbols('h', positive=True),
            'L': sp.symbols('L', positive=True),
            'E': sp.symbols('E', positive=True),
            'B': sp.symbols('B', positive=True),
            'v': sp.symbols('v', positive=True),
            'vy': sp.symbols('vy', positive=True),
            'θ': sp.symbols('theta', positive=True),
            'R': sp.symbols('R', positive=True),
            't': sp.symbols('t', positive=True),
            'a': sp.symbols('a', positive=True)
        }

    def test_invariants(self) -> List[str]:
        logs = []
        
        try:
            # 1. Check energy conservation from start to point a
            # Initial KE: 0.5*m*v0²
            # Final KE: 0.5*m*(v0² + v_y²) = 0.5*m*(2*v0²) = m*v0²
            # Work done by electric field: qE*h (force in y-direction, displacement = h)
            # Work-Energy: ΔKE = W = qE*h
            expected_ΔKE = self.m * self.v0**2 / 2  # From (1/2)m(v² - v0²) = (1/2)m(2v0² - v0²) = (1/2)m*v0²
            actual_ΔKE = self.qE * self.h
            
            if not math.isclose(expected_ΔKE, actual_ΔKE, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Energy conservation violated. Expected ΔKE={expected_ΔKE:.3e}, got W={actual_ΔKE:.3e}")
            else:
                logs.append(f"PASS: Energy conservation holds")
                
        except Exception as e:
            logs.append(f"ERROR in energy check: {str(e)}")
            
        try:
            # 2. Check Newton's 2nd law in electric field
            # From kinematics: h = 0.5*a*t² and 2h = v0*t
            # => t = 2h/v0, then a = 2h/t² = v0²/(2h)
            expected_a = self.v0**2 / (2 * self.h)
            
            # From force: a = qE/m
            actual_a = self.qE / self.m
            
            if not math.isclose(expected_a, actual_a, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Newton's 2nd law mismatch. From kinematics a={expected_a:.3e}, from force a={actual_a:.3e}")
            else:
                logs.append(f"PASS: Newton's 2nd law consistent")
                
        except Exception as e:
            logs.append(f"ERROR in Newton's law check: {str(e)}")
            
        try:
            # 3. Check Lorentz force = centripetal force
            # For magnetic field: qvB = m*v²/R
            # => B = m*v/(q*R)
            expected_B = self.m * self.v / (self.q * self.R)
            
            if not math.isclose(self.B, expected_B, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Lorentz force mismatch. Expected B={expected_B:.3e}, got B={self.B:.3e}")
            else:
                logs.append(f"PASS: Lorentz force equals centripetal force")
                
        except Exception as e:
            logs.append(f"ERROR in Lorentz force check: {str(e)}")
            
        return logs

    def test_trace_consistency(self) -> List[str]:
        logs = []
        
        try:
            # Check step 1: Time of flight in electric field
            t_calc = 2 * self.h / self.v0
            # From kinematics: h = 0.5*a*t² and a = qE/m
            # Solving: t = sqrt(2h/a) = sqrt(2h*m/(qE))
            t_from_a = math.sqrt(2 * self.h * self.m / self.qE)
            
            if not math.isclose(t_calc, t_from_a, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Time consistency. From x=2h: t={t_calc:.3e}, from kinematics: t={t_from_a:.3e}")
            else:
                logs.append(f"PASS: Time in electric field consistent")
                
        except Exception as e:
            logs.append(f"ERROR in time check: {str(e)}")
            
        try:
            # Check step 4: Vertical velocity at point a
            # v_y = a*t = (qE/m) * (2h/v0)
            expected_v_y = self.a_electric * (2 * self.h / self.v0)
            
            if not math.isclose(self.v_y, expected_v_y, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Vertical velocity mismatch. Expected v_y={expected_v_y:.3e}, got v_y={self.v_y:.3e}")
            else:
                logs.append(f"PASS: Vertical velocity consistent")
                
        except Exception as e:
            logs.append(f"ERROR in vertical velocity check: {str(e)}")
            
        try:
            # Check step 5: Speed at point a
            expected_v = math.sqrt(self.v0**2 + self.v_y**2)
            
            if not math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Speed mismatch. Expected v={expected_v:.3e}, got v={self.v:.3e}")
            else:
                logs.append(f"PASS: Speed magnitude consistent")
                
        except Exception as e:
            logs.append(f"ERROR in speed check: {str(e)}")
            
        try:
            # Check step 6: Angle at point a
            expected_tan = self.v_y / self.v0
            expected_theta = math.degrees(math.atan(expected_tan))
            
            if not math.isclose(self.tan_theta, expected_tan, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: tanθ mismatch. Expected tanθ={expected_tan:.3e}, got tanθ={self.tan_theta:.3e}")
            elif not math.isclose(self.theta_deg, expected_theta, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Angle mismatch. Expected θ={expected_theta:.1f}°, got θ={self.theta_deg:.1f}°")
            else:
                logs.append(f"PASS: Angle θ=45° consistent")
                
        except Exception as e:
            logs.append(f"ERROR in angle check: {str(e)}")
            
        try:
            # Check step 8: Radius in magnetic field
            # From geometry: R = (√2/2)L for minimum B
            expected_R = (math.sqrt(2) / 2) * self.L
            
            if not math.isclose(self.R, expected_R, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Radius mismatch. Expected R={expected_R:.3e}, got R={self.R:.3e}")
            else:
                logs.append(f"PASS: Radius R consistent with geometry")
                
        except Exception as e:
            logs.append(f"ERROR in radius check: {str(e)}")
            
        return logs

    def test_monotonicity(self) -> List[str]:
        logs = []
        
        try:
            # Perturb mass and check trend in B
            original_m = self.m
            perturb_factor = 1.1  # 10% increase
            
            # Perturb mass
            self.m *= perturb_factor
            
            # Recalculate B using the formula B = 2*m*v0/(q*L)
            new_B = 2 * self.m * self.v0 / (self.q * self.L)
            
            # Check if B increased with increased mass
            if new_B > self.B:
                logs.append(f"PASS: B increases with mass (B_old={self.B:.3e}, B_new={new_B:.3e})")
            else:
                logs.append(f"FAIL: B does not increase with mass")
                
            # Restore original mass
            self.m = original_m
            
        except Exception as e:
            logs.append(f"ERROR in monotonicity check: {str(e)}")
            # Try to restore mass even if error occurred
            self.m = original_m if 'original_m' in locals() else self.m
            
        try:
            # Perturb v0 and check trend in E
            original_v0 = self.v0
            perturb_factor = 1.1
            
            # Perturb v0
            self.v0 *= perturb_factor
            
            # Recalculate E using E = m*v0²/(2*q*h)
            new_E = self.m * self.v0**2 / (2 * self.q * self.h)
            
            # Check if E increased with increased v0
            if new_E > self.E:
                logs.append(f"PASS: E increases with v0 (E_old={self.E:.3e}, E_new={new_E:.3e})")
            else:
                logs.append(f"FAIL: E does not increase with v0")
                
            # Restore original v0
            self.v0 = original_v0
            
        except Exception as e:
            logs.append(f"ERROR in v0 monotonicity check: {str(e)}")
            self.v0 = original_v0 if 'original_v0' in locals() else self.v0
            
        try:
            # Perturb q and check trend in B
            original_q = self.q
            perturb_factor = 1.1
            
            # Perturb q
            self.q *= perturb_factor
            
            # Recalculate B
            new_B = 2 * self.m * self.v0 / (self.q * self.L)
            
            # Check if B decreases with increased q
            if new_B < self.B:
                logs.append(f"PASS: B decreases with charge (B_old={self.B:.3e}, B_new={new_B:.3e})")
            else:
                logs.append(f"FAIL: B does not decrease with charge")
                
            # Restore original q
            self.q = original_q
            
        except Exception as e:
            logs.append(f"ERROR in charge monotonicity check: {str(e)}")
            self.q = original_q if 'original_q' in locals() else self.q
            
        return logs

    def test_target_answers(self) -> List[str]:
        logs = []
        
        try:
            # Check E against target
            if not math.isclose(self.E, self.target_E, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: E mismatch. Target: {self.target_E:.3e}, Got: {self.E:.3e}")
            else:
                logs.append(f"PASS: E = {self.E:.3e} matches target")
                
        except Exception as e:
            logs.append(f"ERROR in E target check: {str(e)}")
            
        try:
            # Check v against target
            if not math.isclose(self.v, self.target_v, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v mismatch. Target: {self.target_v:.3e}, Got: {self.v:.3e}")
            elif not math.isclose(self.theta_deg, 45.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: θ mismatch. Target: 45.0°, Got: {self.theta_deg:.1f}°")
            else:
                logs.append(f"PASS: v = {self.v:.3e} at θ = {self.theta_deg:.1f}° matches target")
                
        except Exception as e:
            logs.append(f"ERROR in v target check: {str(e)}")
            
        try:
            # Check B against target
            if not math.isclose(self.B, self.target_B, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: B mismatch. Target: {self.target_B:.3e}, Got: {self.B:.3e}")
            else:
                logs.append(f"PASS: B = {self.B:.3e} matches target")
                
        except Exception as e:
            logs.append(f"ERROR in B target check: {str(e)}")
            
        return logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        all_logs = []
        success = True
        
        # Run all test suites
        test_suites = [
            ("Target Answers", self.test_target_answers),
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]
        
        for suite_name, test_func in test_suites:
            all_logs.append(f"\n=== {suite_name} ===")
            try:
                logs = test_func()
                all_logs.extend(logs)
                
                # Check for failures in logs
                for log in logs:
                    if log.startswith("FAIL:") or log.startswith("ERROR:"):
                        success = False
                        
            except Exception as e:
                all_logs.append(f"CRITICAL ERROR in {suite_name}: {str(e)}")
                success = False
        
        # Summary
        all_logs.append(f"\n=== SUMMARY ===")
        all_logs.append(f"Overall: {'PASS' if success else 'FAIL'}")
        
        return success, all_logs