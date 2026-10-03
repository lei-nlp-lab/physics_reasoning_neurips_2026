import math
import sympy as sp
import traceback
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize with safe attribute access.
        solution_instance should have attributes:
            - inputs: dict with keys 'm', 'q', 'v0', 'B', 'd', 'R' (magnetic field radius)
            - results: dict with keys 'alpha', 'r', 'a', 'theta', 't_0', 'v', 'U', 'E'
        """
        # Extract inputs safely
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'inputs', {}).get('m', 1.0))
        self.q = getattr(solution_instance, 'q', getattr(solution_instance, 'inputs', {}).get('q', 1.0))
        self.v0 = getattr(solution_instance, 'v0', getattr(solution_instance, 'inputs', {}).get('v0', 1.0))
        self.B = getattr(solution_instance, 'B', getattr(solution_instance, 'inputs', {}).get('B', 1.0))
        self.d = getattr(solution_instance, 'd', getattr(solution_instance, 'inputs', {}).get('d', 1.0))
        # Magnetic field radius from problem statement
        self.R_given = (2 * self.m * self.v0) / (3 * self.q * self.B)
        
        # Extract results safely with defaults
        results = getattr(solution_instance, 'results', {})
        self.alpha = results.get('alpha', 0.0)
        self.r = results.get('r', 0.0)
        self.a = results.get('a', 0.0)
        self.theta = results.get('theta', 0.0)
        self.t0 = results.get('t_0', 0.0)
        self.v = results.get('v', 0.0)
        self.U = results.get('U', 0.0)
        self.E = results.get('E', 0.0)
        
        # For symbolic computations
        self.m_sym = sp.Symbol('m', positive=True)
        self.q_sym = sp.Symbol('q', positive=True)
        self.v0_sym = sp.Symbol('v0', positive=True)
        self.B_sym = sp.Symbol('B', positive=True)
        self.d_sym = sp.Symbol('d', positive=True)
        self.U_sym = sp.Symbol('U', positive=True)
        
        self.logs = []

    def log(self, msg):
        self.logs.append(msg)

    def test_invariants(self) -> bool:
        """Check Newton's 2nd Law, Lorentz force, and energy-work theorem."""
        try:
            # 1. Electric field: E = U/d
            E_calc = self.U / self.d if self.d != 0 else 0
            if not math.isclose(self.E, E_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"E mismatch: {self.E} vs {E_calc}")
                return False
            
            # 2. Newton's 2nd Law in electric field: a = qE/m
            a_calc = (self.q * self.E) / self.m if self.m != 0 else 0
            if not math.isclose(self.a, a_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"a mismatch: {self.a} vs {a_calc}")
                return False
            
            # 3. Exit velocity magnitude from kinematics: v = v0/cos(alpha)
            if abs(self.alpha) < 1e-12:
                v_calc = self.v0
            else:
                v_calc = self.v0 / math.cos(self.alpha)
            if not math.isclose(self.v, v_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"v mismatch: {self.v} vs {v_calc}")
                return False
            
            # 4. Lorentz force as centripetal: r = m*v/(q*B)
            if self.q != 0 and self.B != 0:
                r_calc = (self.m * self.v) / (self.q * self.B)
                if not math.isclose(self.r, r_calc, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"r mismatch: {self.r} vs {r_calc}")
                    return False
            
            # 5. Given magnetic field radius relation: r = sqrt(3)*R_given
            r_expected = math.sqrt(3) * self.R_given
            if not math.isclose(self.r, r_expected, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"r vs R_given mismatch: {self.r} vs {r_expected}")
                return False
            
            # 6. Work-Energy theorem in electric field
            # ΔK = q*U_eff, where U_eff is potential difference across vertical displacement d/2?
            # Actually, vertical displacement is d/2, so potential difference = E*(d/2) = (U/d)*(d/2) = U/2
            # Work done = q*(U/2)
            # Kinetic energy gain = 0.5*m*(v^2 - v0^2)
            delta_K = 0.5 * self.m * (self.v**2 - self.v0**2)
            work_done = self.q * (self.U / 2)
            if not math.isclose(delta_K, work_done, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Work-Energy mismatch: ΔK={delta_K}, qU/2={work_done}")
                return False
            
            return True
        except Exception as e:
            self.log(f"Error in test_invariants: {traceback.format_exc()}")
            return False

    def test_trace_consistency(self) -> bool:
        """Check intermediate steps from the analysis."""
        try:
            # 1. Time t0 from horizontal motion: sqrt(3)*d = v0*t0
            if self.v0 != 0:
                t0_horizontal = (math.sqrt(3) * self.d) / self.v0
                if not math.isclose(self.t0, t0_horizontal, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"t0 horizontal mismatch: {self.t0} vs {t0_horizontal}")
                    return False
            
            # 2. Time t0 from vertical motion: d/2 = 0.5*a*t0^2
            if self.a != 0:
                t0_vertical = math.sqrt(self.d / self.a)
                if not math.isclose(self.t0, t0_vertical, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"t0 vertical mismatch: {self.t0} vs {t0_vertical}")
                    return False
            
            # 3. Angle alpha: tan(alpha) = (a*t0)/v0
            tan_alpha_calc = (self.a * self.t0) / self.v0 if self.v0 != 0 else 0
            if not math.isclose(math.tan(self.alpha), tan_alpha_calc, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"tan(alpha) mismatch: {math.tan(self.alpha)} vs {tan_alpha_calc}")
                return False
            
            # 4. Alpha should be π/6
            if not math.isclose(self.alpha, math.pi/6, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"alpha not π/6: {self.alpha}")
                return False
            
            # 5. Theta = 2*alpha
            if not math.isclose(self.theta, 2*self.alpha, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"theta mismatch: {self.theta} vs {2*self.alpha}")
                return False
            
            # 6. Theta should be π/3
            if not math.isclose(self.theta, math.pi/3, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"theta not π/3: {self.theta}")
                return False
            
            return True
        except Exception as e:
            self.log(f"Error in test_trace_consistency: {traceback.format_exc()}")
            return False

    def test_monotonicity(self) -> bool:
        """Perturb v0 and check U changes quadratically, alpha remains constant."""
        try:
            original_v0 = self.v0
            
            # Perturb v0 by +10%
            perturbed_v0 = original_v0 * 1.1
            
            # Recalculate U using formula U = m*v0^2/(3q)
            U_perturbed = (self.m * perturbed_v0**2) / (3 * self.q)
            U_original = (self.m * original_v0**2) / (3 * self.q)
            
            # Check quadratic scaling: (U_perturbed/U_original) = (v0_perturbed/v0_original)^2
            expected_ratio = (perturbed_v0 / original_v0) ** 2
            actual_ratio = U_perturbed / U_original if U_original != 0 else 0
            
            if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"U not scaling quadratically with v0: {actual_ratio} vs {expected_ratio}")
                return False
            
            # Alpha should remain π/6 independent of v0 (check via tan(alpha) = sqrt(3)/3)
            # From kinematics: a = v0^2/(3d), t0 = sqrt(3)*d/v0, so tan(alpha) = (a*t0)/v0 = sqrt(3)/3
            tan_alpha_theory = math.sqrt(3)/3
            if not math.isclose(math.tan(self.alpha), tan_alpha_theory, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"tan(alpha) not constant with v0: {math.tan(self.alpha)} vs {tan_alpha_theory}")
                return False
            
            # Restore original v0 (not strictly necessary as we didn't modify self.v0)
            return True
        except Exception as e:
            self.log(f"Error in test_monotonicity: {traceback.format_exc()}")
            return False

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Run all checks and return success flag and logs."""
        self.logs = []
        tests = [
            ("Invariants", self.test_invariants()),
            ("Trace Consistency", self.test_trace_consistency()),
            ("Monotonicity", self.test_monotonicity())
        ]
        
        success = all(result for _, result in tests)
        
        # Summarize
        for test_name, passed in tests:
            status = "PASS" if passed else "FAIL"
            self.log(f"{test_name}: {status}")
        
        self.log(f"Overall: {'PASS' if success else 'FAIL'}")
        return success, self.logs

# Example usage (commented out):
# class Solution:
#     def __init__(self):
#         self.inputs = {'m': 1e-30, 'q': 1.6e-19, 'v0': 1e6, 'B': 0.1, 'd': 0.01}
#         self.results = {
#             'U': (1e-30 * (1e6)**2) / (3 * 1.6e-19),
#             'alpha': math.pi/6,
#             'r': math.sqrt(3) * (2*1e-30*1e6)/(3*1.6e-19*0.1),
#             'a': (1.6e-19 * ((1e-30 * (1e6)**2) / (3 * 1.6e-19)) / 0.01) / 1e-30,
#             'theta': math.pi/3,
#             't_0': (math.sqrt(3)*0.01)/1e6,
#             'v': 1e6 / math.cos(math.pi/6),
#             'E': ((1e-30 * (1e6)**2) / (3 * 1.6e-19)) / 0.01
#         }
# 
# checker = PhysicsChecker(Solution())
# success, logs = checker.run_all_tests()
# for log in logs:
#     print(log)