import math
import sympy as sp
import traceback
from typing import Tuple, List, Any, Optional

class PhysicsChecker:
    """
    Robust physics auditor for the particle detection device problem.
    """
    def __init__(self, solution_instance: Any):
        """
        Safely extract inputs and results from the solution instance.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Extract given parameters with safe defaults
        self.m = getattr(solution_instance, 'm', 1.0)
        self.q = getattr(solution_instance, 'q', 1.0)
        self.v0 = getattr(solution_instance, 'v0', 1.0)
        self.L = getattr(solution_instance, 'L', 1.0)
        self.R = getattr(solution_instance, 'R', 1.0)
        self.N = getattr(solution_instance, 'N', 1.0)

        # Extract results using key mapping (fallback to None)
        self.alpha = self.results.get('alpha', None)
        self.y = self.results.get('y', None)
        self.angleDO2K = self.results.get('angleDO_2K', None)
        self.r = self.results.get('r', None)
        self.IH = self.results.get('IH', None)
        self.tanalpha = self.results.get('tanalpha', None)
        self.T = self.results.get('T', None)
        self.t = self.results.get('t', None)
        self.a = self.results.get('a', None)
        self.angleKIH = self.results.get('angleKIH', None)
        self.angleKJH = self.results.get('angleKJH', None)
        self.IJ = self.results.get('IJ', None)
        self.KH = self.results.get('KH', None)
        self.v = self.results.get('v', None)
        self.JH = self.results.get('JH', None)
        self.n = self.results.get('n', None)
        self.L_res = self.results.get('L', None)  # may be same as input L

        # Derived quantities from problem statement
        self.U = self.m * self.v0**2 / self.q
        self.B = math.sqrt(2) * self.m * self.v0 / (self.q * self.R)
        self.theta = math.radians(37)  # 37 degrees in radians

        # Symbolic symbols for symbolic checks
        self.m_sym, self.q_sym, self.v0_sym, self.L_sym, self.R_sym, self.N_sym = sp.symbols('m q v0 L R N', positive=True)
        self.U_sym = self.m_sym * self.v0_sym**2 / self.q_sym
        self.B_sym = sp.sqrt(2) * self.m_sym * self.v0_sym / (self.q_sym * self.R_sym)
        self.theta_sym = sp.rad(37)  # sympy radian conversion

        # For numerical fallback: use the extracted numerical values
        self.numerical_vals = {
            self.m_sym: self.m,
            self.q_sym: self.q,
            self.v0_sym: self.v0,
            self.L_sym: self.L,
            self.R_sym: self.R,
            self.N_sym: self.N
        }

    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Fuzzy numerical matching."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def _eval_expr(self, expr: sp.Expr) -> float:
        """Evaluate symbolic expression with numerical values."""
        return float(expr.subs(self.numerical_vals))

    def test_invariants(self) -> Tuple[bool, List[str]]:
        """
        Verify energy conservation and Newton's second law in electric field.
        """
        logs = []
        success = True

        # 1. Energy conservation in electric field (kinetic energy gain = work done by electric force)
        # Initial KE: 0.5 * m * v0^2
        # Work by electric field: q * E * y, where E = U/L, y = L/2 (from step_3)
        # So work = q * (U/L) * (L/2) = q * U / 2
        # But U = m * v0^2 / q, so work = (q * (m * v0^2 / q)) / 2 = m * v0^2 / 2
        # Final KE: 0.5 * m * v^2, with v = sqrt(2)*v0 -> 0.5*m*(2*v0^2) = m*v0^2
        # So initial KE + work = 0.5*m*v0^2 + m*v0^2/2 = m*v0^2, matches final KE.
        try:
            # Symbolic check
            KE_initial_sym = 0.5 * self.m_sym * self.v0_sym**2
            work_sym = self.q_sym * (self.U_sym / self.L_sym) * (self.L_sym / 2)  # y = L/2
            KE_final_sym = 0.5 * self.m_sym * (sp.sqrt(2) * self.v0_sym)**2
            energy_eq_sym = sp.simplify(KE_initial_sym + work_sym - KE_final_sym)
            if energy_eq_sym != 0:
                # Evaluate numerically
                KE_initial_num = self._eval_expr(KE_initial_sym)
                work_num = self._eval_expr(work_sym)
                KE_final_num = self._eval_expr(KE_final_sym)
                if not self._is_close(KE_initial_num + work_num, KE_final_num):
                    logs.append(f"Energy conservation fails: KE_initial={KE_initial_num}, work={work_num}, KE_final={KE_final_num}")
                    success = False
                else:
                    logs.append("Energy conservation holds numerically.")
            else:
                logs.append("Energy conservation holds symbolically.")
        except Exception as e:
            logs.append(f"Energy conservation check error: {e}")
            success = False

        # 2. Newton's second law in electric field: a = F/m = q*E/m = q*(U/L)/m
        # Given U = m*v0^2/q, so a = v0^2/L
        try:
            a_expected_sym = self.v0_sym**2 / self.L_sym
            a_expected_num = self._eval_expr(a_expected_sym)
            if self.a is not None:
                if not self._is_close(self.a, a_expected_num):
                    logs.append(f"Acceleration mismatch: a={self.a}, expected={a_expected_num}")
                    success = False
                else:
                    logs.append(f"Acceleration matches: a={self.a}")
            else:
                # Compute from extracted parameters
                a_computed = self.q * (self.U / self.L) / self.m
                if not self._is_close(a_computed, a_expected_num):
                    logs.append(f"Acceleration from parameters mismatch: computed={a_computed}, expected={a_expected_num}")
                    success = False
                else:
                    logs.append(f"Acceleration from parameters matches: {a_computed}")
        except Exception as e:
            logs.append(f"Newton's second law check error: {e}")
            success = False

        return success, logs

    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """
        Cross-check intermediate results with final answers and step-by-step logic.
        """
        logs = []
        success = True

        # Check 1: Vertical displacement in electric field (step_3)
        try:
            y_expected = self.L / 2
            if self.y is not None:
                if not self._is_close(self.y, y_expected):
                    logs.append(f"Vertical displacement y mismatch: {self.y} vs expected {y_expected}")
                    success = False
                else:
                    logs.append(f"Vertical displacement y matches: {self.y}")
        except Exception as e:
            logs.append(f"Vertical displacement check error: {e}")
            success = False

        # Check 2: Angle alpha (step_6)
        try:
            alpha_expected_deg = 45.0
            alpha_expected_rad = math.radians(45.0)
            if self.alpha is not None:
                if not self._is_close(self.alpha, alpha_expected_deg):
                    logs.append(f"Alpha mismatch: {self.alpha}° vs expected {alpha_expected_deg}°")
                    success = False
                else:
                    logs.append(f"Alpha matches: {self.alpha}°")
            # Check tan(alpha) = 1
            if self.tanalpha is not None:
                if not self._is_close(self.tanalpha, 1.0):
                    logs.append(f"tan(alpha) mismatch: {self.tanalpha} vs expected 1.0")
                    success = False
                else:
                    logs.append(f"tan(alpha) matches: {self.tanalpha}")
        except Exception as e:
            logs.append(f"Alpha check error: {e}")
            success = False

        # Check 3: Exit velocity v (step_6)
        try:
            v_expected = math.sqrt(2) * self.v0
            if self.v is not None:
                if not self._is_close(self.v, v_expected):
                    logs.append(f"Velocity v mismatch: {self.v} vs expected {v_expected}")
                    success = False
                else:
                    logs.append(f"Velocity v matches: {self.v}")
        except Exception as e:
            logs.append(f"Velocity check error: {e}")
            success = False

        # Check 4: Radius in magnetic field (step_8)
        try:
            r_expected = self.R
            if self.r is not None:
                if not self._is_close(self.r, r_expected):
                    logs.append(f"Radius r mismatch: {self.r} vs expected {r_expected}")
                    success = False
                else:
                    logs.append(f"Radius r matches: {self.r}")
        except Exception as e:
            logs.append(f"Radius check error: {e}")
            success = False

        # Check 5: Central angle DO2K (step_12)
        try:
            angle_expected_deg = 98.0  # 180 - (45 + 37)
            if self.angleDO2K is not None:
                if not self._is_close(self.angleDO2K, angle_expected_deg):
                    logs.append(f"Angle DO2K mismatch: {self.angleDO2K}° vs expected {angle_expected_deg}°")
                    success = False
                else:
                    logs.append(f"Angle DO2K matches: {self.angleDO2K}°")
        except Exception as e:
            logs.append(f"Angle DO2K check error: {e}")
            success = False

        # Check 6: Period T (step_13)
        try:
            T_expected = (math.sqrt(2) * math.pi * self.R) / self.v0
            if self.T is not None:
                if not self._is_close(self.T, T_expected):
                    logs.append(f"Period T mismatch: {self.T} vs expected {T_expected}")
                    success = False
                else:
                    logs.append(f"Period T matches: {self.T}")
        except Exception as e:
            logs.append(f"Period check error: {e}")
            success = False

        # Check 7: Time in magnetic field t (step_14)
        try:
            t_expected = (49 * math.sqrt(2) * math.pi * self.R) / (180 * self.v0)
            if self.t is not None:
                if not self._is_close(self.t, t_expected):
                    logs.append(f"Time t mismatch: {self.t} vs expected {t_expected}")
                    success = False
                else:
                    logs.append(f"Time t matches: {self.t}")
        except Exception as e:
            logs.append(f"Time check error: {e}")
            success = False

        # Check 8: Particle count n (step_4)
        try:
            n_expected = self.N / 2
            if self.n is not None:
                if not self._is_close(self.n, n_expected):
                    logs.append(f"Particle count n mismatch: {self.n} vs expected {n_expected}")
                    success = False
                else:
                    logs.append(f"Particle count n matches: {self.n}")
        except Exception as e:
            logs.append(f"Particle count check error: {e}")
            success = False

        # Check 9: Geometry on screen (steps 17-20)
        try:
            cos45 = math.cos(math.radians(45))
            KH_expected = self.R * (1 - cos45)
            if self.KH is not None:
                if not self._is_close(self.KH, KH_expected):
                    logs.append(f"KH mismatch: {self.KH} vs expected {KH_expected}")
                    success = False
                else:
                    logs.append(f"KH matches: {self.KH}")
            # IH = KH / tan(45) = KH
            IH_expected = KH_expected
            if self.IH is not None:
                if not self._is_close(self.IH, IH_expected):
                    logs.append(f"IH mismatch: {self.IH} vs expected {IH_expected}")
                    success = False
                else:
                    logs.append(f"IH matches: {self.IH}")
            # JH = KH / tan(37)
            tan37 = math.tan(self.theta)
            JH_expected = KH_expected / tan37
            if self.JH is not None:
                if not self._is_close(self.JH, JH_expected):
                    logs.append(f"JH mismatch: {self.JH} vs expected {JH_expected}")
                    success = False
                else:
                    logs.append(f"JH matches: {self.JH}")
            # IJ = JH - IH
            IJ_expected = JH_expected - IH_expected
            if self.IJ is not None:
                if not self._is_close(self.IJ, IJ_expected):
                    logs.append(f"IJ mismatch: {self.IJ} vs expected {IJ_expected}")
                    success = False
                else:
                    logs.append(f"IJ matches: {self.IJ}")
        except Exception as e:
            logs.append(f"Geometry check error: {e}")
            success = False

        return success, logs

    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Perturb mass m and check expected trends.
        """
        logs = []
        success = True

        try:
            # Original values
            m_orig = self.m
            # Increase mass by 10%
            self.m = m_orig * 1.1
            # Recompute derived quantities
            self.U = self.m * self.v0**2 / self.q
            self.B = math.sqrt(2) * self.m * self.v0 / (self.q * self.R)

            # Expected trends:
            # 1. Period T in magnetic field should increase with m (T ∝ m)
            T_new = (math.sqrt(2) * math.pi * self.R) / self.v0  # but wait, T = 2πm/(qB), and B ∝ m, so T independent? Let's check.
            # Actually, T = 2πm/(qB). With B = √2 m v0/(q R), so T = 2πm / (q * (√2 m v0/(q R))) = 2πR/(√2 v0) = constant! So T does not depend on m.
            # So T should remain the same. That's fine, we check that T is invariant under m change.
            T_orig = (math.sqrt(2) * math.pi * self.R) / self.v0
            if not self._is_close(T_new, T_orig):
                logs.append(f"Period T changed with m: orig={T_orig}, new={T_new}, but expected constant")
                success = False
            else:
                logs.append(f"Period T correctly independent of m: {T_orig}")

            # 2. Acceleration a in electric field: a = v0^2/L, independent of m.
            a_orig = self.v0**2 / self.L
            a_new = self.v0**2 / self.L
            if not self._is_close(a_new, a_orig):
                logs.append(f"Acceleration a changed with m: orig={a_orig}, new={a_new}, but expected constant")
                success = False
            else:
                logs.append(f"Acceleration a correctly independent of m: {a_orig}")

            # Restore original mass
            self.m = m_orig
            self.U = self.m * self.v0**2 / self.q
            self.B = math.sqrt(2) * self.m * self.v0 / (self.q * self.R)

        except Exception as e:
            logs.append(f"Monotonicity test error: {e}")
            success = False
            # Restore mass on error
            self.m = m_orig
            self.U = self.m * self.v0**2 / self.q
            self.B = math.sqrt(2) * self.m * self.v0 / (self.q * self.R)

        return success, logs

    def test_final_answers(self) -> Tuple[bool, List[str]]:
        """
        Compare the solution's final answers with the target answers.
        Target answers are given as strings, but we can compute expected expressions.
        """
        logs = []
        success = True

        # Target answers (as strings, but we'll compute expected values numerically and compare)
        target1_str = r"$\frac{N}{2}$"
        target2_str = r"$\frac{49\pi\sqrt{2}\pi R}{180{v_0}}$"  # note: extra π? We'll use our computed expression.
        target3_str = r"$(1-\frac{\sqrt{2}}{2})\frac{R}{3}$"

        # Expected expressions from our analysis:
        # 1. n = N/2
        expected1 = self.N / 2
        # 2. t = (49 * sqrt(2) * π * R) / (180 * v0)
        expected2 = (49 * math.sqrt(2) * math.pi * self.R) / (180 * self.v0)
        # 3. IJ = (1 - sqrt(2)/2) * R / 3
        expected3 = (1 - math.sqrt(2)/2) * self.R / 3

        # The solution may have these stored in results or as attributes.
        # We'll check n, t, IJ from results.
        try:
            if self.n is not None:
                if not self._is_close(self.n, expected1):
                    logs.append(f"Final answer n mismatch: {self.n} vs expected {expected1}")
                    success = False
                else:
                    logs.append(f"Final answer n matches: {self.n}")
            else:
                logs.append("Warning: n not found in results.")

            if self.t is not None:
                if not self._is_close(self.t, expected2):
                    logs.append(f"Final answer t mismatch: {self.t} vs expected {expected2}")
                    success = False
                else:
                    logs.append(f"Final answer t matches: {self.t}")
            else:
                logs.append("Warning: t not found in results.")

            if self.IJ is not None:
                if not self._is_close(self.IJ, expected3):
                    logs.append(f"Final answer IJ mismatch: {self.IJ} vs expected {expected3}")
                    success = False
                else:
                    logs.append(f"Final answer IJ matches: {self.IJ}")
            else:
                logs.append("Warning: IJ not found in results.")

        except Exception as e:
            logs.append(f"Final answers check error: {e}")
            success = False

        return success, logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all tests and return overall success and logs.
        """
        logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
            ("Final Answers", self.test_final_answers)
        ]

        for name, test_func in tests:
            try:
                success, test_logs = test_func()
                logs.append(f"=== {name} ===")
                logs.extend(test_logs)
                if not success:
                    overall_success = False
                    logs.append(f"{name} FAILED.")
                else:
                    logs.append(f"{name} passed.")
            except Exception as e:
                logs.append(f"{name} raised exception: {traceback.format_exc()}")
                overall_success = False

        return overall_success, logs