import math
import sympy as sp
import traceback
from typing import Tuple, List


class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution = solution_instance
        self.results_dict = getattr(solution_instance, "results", {}) or {}
        self.inputs_dict = getattr(solution_instance, "inputs", {}) or {}

        # -----------------------------
        # Constants from the problem
        # -----------------------------
        self.u_to_MeV = 931.502
        self.MeV_to_J = 1.602e-13

        self.abundance_u235 = 0.00720
        self.rho_uo2 = 10600.0          # kg/m^3
        self.M_w_uo2 = 0.270            # kg/mol
        self.N_A = 6.022e23             # mol^-1
        self.sigma_f = 5.400e-26        # m^2
        self.phi = 2.000e18             # m^-2 s^-1
        self.heat_fraction = 0.80

        self.lambda_ = 3.28             # W/(m K)
        self.T_c_max = 3138.0           # K
        self.T_s = 577.0                # K

        self.A_d2o = 20.0
        self.k1 = 1.021e-2              # m
        self.k2 = 8.787e-3              # m^-1
        self.channel_pitch = 0.286      # m
        self.channel_radius = 3.617e-2  # m

        # Needed isotope masses
        self.m_U235 = 235.04397
        self.m_Zr94 = 93.9063
        self.m_Ce140 = 139.905
        self.m_n = 1.00867

        # -----------------------------
        # Reference answers
        # -----------------------------
        self.ref_Delta_m = self.m_U235 - self.m_Zr94 - self.m_Ce140 - self.m_n
        self.ref_DeltaE = self.ref_Delta_m * self.u_to_MeV

        self.ref_N1 = self.rho_uo2 * self.N_A / self.M_w_uo2
        self.ref_N = self.abundance_u235 * self.ref_N1

        self.ref_E_f_MeV = self.heat_fraction * self.ref_DeltaE
        self.ref_E_f_J = self.ref_E_f_MeV * self.MeV_to_J
        self.ref_Q = self.ref_N * self.sigma_f * self.phi * self.ref_E_f_J

        self.ref_T_diff = self.T_c_max - self.T_s
        self.ref_a_u2 = 4 * self.lambda_ * self.ref_T_diff / self.ref_Q
        self.ref_a_u = math.sqrt(self.ref_a_u2)

        self.ref_alpha = ((self.A_d2o - 1) / (self.A_d2o + 1)) ** 2
        self.ref_f_l = 1 - self.ref_alpha

        self.ref_H = math.sqrt(3 * self.k1 * math.pi**2 / self.k2)
        self.ref_R = 2.405 * self.ref_H / (math.sqrt(2) * math.pi)

        self.ref_F_n = math.pi * self.ref_R**2 / (self.channel_pitch**2)
        self.ref_F_n_int = round(self.ref_F_n)

        self.ref_M = (
            self.ref_F_n_int
            * math.pi
            * self.channel_radius**2
            * self.ref_H
            * self.rho_uo2
        )

    # =========================================================
    # Helpers
    # =========================================================
    def _aliases(self):
        return {
            "Delta_m": ["Delta_m", "Deltam", "$\\Delta m$"],
            "DeltaE": ["DeltaE", "$\\Delta E$"],
            "N1": ["N1", "N_1", "$N_1$"],
            "N": ["N", "$N$"],
            "E_f": ["E_f", "$E_f$"],
            "Q": ["Q", "$Q$"],
            "T_diff_formula": ["T_diff_formula", "F", "$F$", "$T_c - T_s$"],
            "a_u2": ["a_u2", "$a_u^2$"],
            "a_u": ["a_u", "$a_u$"],
            "v_formula": ["v_formula", "v", "$v$"],
            "V_formula": ["V_formula", "V", "$V$"],
            "G_formula": ["G_formula", "G", "$G(\\alpha, \\theta)$"],
            "alpha": ["alpha", "$\\alpha$"],
            "f_l": ["f_l", "$f_l$"],
            "R": ["R", "$R$"],
            "H": ["H", "$H$"],
            "F_n": ["F_n", "$F_n$"],
            "M": ["M", "$M$"],
        }

    def _get(self, canonical_name):
        aliases = self._aliases().get(canonical_name, [canonical_name])

        for key in aliases:
            if hasattr(self.solution, key):
                value = getattr(self.solution, key)
                if value is not None:
                    return value

        for key in aliases:
            if key in self.results_dict and self.results_dict[key] is not None:
                return self.results_dict[key]

        for key in aliases:
            if key in self.inputs_dict and self.inputs_dict[key] is not None:
                return self.inputs_dict[key]

        return None

    def _to_float(self, x):
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        if hasattr(x, "evalf"):
            return float(x.evalf())
        return float(x)

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-6):
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)

    def _parse_expr(self, expr_str, symbol_mode="generic"):
        if expr_str is None:
            return None
        if not isinstance(expr_str, str):
            return expr_str

        s = expr_str.strip()
        s = s.replace("^", "**")
        s = s.replace("λ", "lambda_")
        s = s.replace("\\lambda", "lambda_")
        s = s.replace("cos ", "cos")
        s = s.replace("sin ", "sin")

        if "=" in s:
            s = s.split("=", 1)[1].strip()

        # Symbols used in formulas
        Q, a, lambda_ = sp.symbols("Q a lambda_")
        A, v_b = sp.symbols("A v_b")
        alpha, theta = sp.symbols("alpha theta")

        local_dict = {
            "Q": Q,
            "a": a,
            "lambda_": lambda_,
            "A": A,
            "v_b": v_b,
            "alpha": alpha,
            "theta": theta,
            "cos": sp.cos,
            "sin": sp.sin,
        }

        try:
            return sp.simplify(sp.sympify(s, locals=local_dict))
        except Exception:
            return None

    # =========================================================
    # Tests
    # =========================================================
    def test_invariants(self) -> Tuple[bool, List[str]]:
        logs = []
        success = True

        try:
            # ΔE
            DeltaE = self._to_float(self._get("DeltaE"))
            if DeltaE is None:
                logs.append("FAIL: Missing DeltaE")
                success = False
            elif self._is_close(DeltaE, self.ref_DeltaE, rel_tol=1e-3, abs_tol=1e-3):
                logs.append("PASS: DeltaE matches mass-energy calculation")
            else:
                logs.append(f"FAIL: DeltaE mismatch. Expected {self.ref_DeltaE:.3f}, got {DeltaE:.3f}")
                success = False

            # N1
            N1 = self._to_float(self._get("N1"))
            if N1 is not None:
                if self._is_close(N1, self.ref_N1, rel_tol=1e-3, abs_tol=1e24):
                    logs.append("PASS: N1 consistent with density and molar mass")
                else:
                    logs.append(f"FAIL: N1 mismatch. Expected {self.ref_N1:.3e}, got {N1:.3e}")
                    success = False

            # N
            N = self._to_float(self._get("N"))
            if N is None:
                logs.append("FAIL: Missing N")
                success = False
            else:
                if self._is_close(N, self.ref_N, rel_tol=1e-3, abs_tol=1e23):
                    logs.append("PASS: N consistent with natural abundance")
                else:
                    logs.append(f"FAIL: N mismatch. Expected {self.ref_N:.3e}, got {N:.3e}")
                    success = False

            # E_f
            E_f = self._to_float(self._get("E_f"))
            if E_f is not None:
                if self._is_close(E_f, self.ref_E_f_J, rel_tol=1e-3, abs_tol=1e-14):
                    logs.append("PASS: E_f correctly converted to Joules")
                else:
                    logs.append(f"FAIL: E_f mismatch. Expected {self.ref_E_f_J:.3e}, got {E_f:.3e}")
                    success = False

            # Q
            Q_val = self._to_float(self._get("Q"))
            if Q_val is None:
                logs.append("FAIL: Missing Q")
                success = False
            elif self._is_close(Q_val, self.ref_Q, rel_tol=1e-3, abs_tol=1e5):
                logs.append("PASS: Q consistent with fission heating rate")
            else:
                logs.append(f"FAIL: Q mismatch. Expected {self.ref_Q:.3e}, got {Q_val:.3e}")
                success = False

            # a_u
            a_u = self._to_float(self._get("a_u"))
            if a_u is None:
                logs.append("FAIL: Missing a_u")
                success = False
            elif self._is_close(a_u, self.ref_a_u, rel_tol=1e-3, abs_tol=1e-5):
                logs.append("PASS: a_u consistent with conduction limit")
            else:
                logs.append(f"FAIL: a_u mismatch. Expected {self.ref_a_u:.3e}, got {a_u:.3e}")
                success = False

            # alpha and f_l
            alpha_val = self._to_float(self._get("alpha"))
            if alpha_val is None:
                logs.append("FAIL: Missing alpha")
                success = False
            elif self._is_close(alpha_val, self.ref_alpha, rel_tol=1e-3, abs_tol=1e-4):
                logs.append("PASS: alpha correct for A=20")
            else:
                logs.append(f"FAIL: alpha mismatch. Expected {self.ref_alpha:.3f}, got {alpha_val:.3f}")
                success = False

            f_l = self._to_float(self._get("f_l"))
            if f_l is None:
                logs.append("FAIL: Missing f_l")
                success = False
            elif self._is_close(f_l, self.ref_f_l, rel_tol=1e-3, abs_tol=1e-4):
                logs.append("PASS: f_l correct")
            else:
                logs.append(f"FAIL: f_l mismatch. Expected {self.ref_f_l:.3f}, got {f_l:.3f}")
                success = False

            # Reactor geometry
            R = self._to_float(self._get("R"))
            H = self._to_float(self._get("H"))
            if R is None or H is None:
                logs.append("FAIL: Missing R or H")
                success = False
            else:
                lhs = (2.405 / R) ** 2
                rhs = 2 * (math.pi / H) ** 2
                if self._is_close(lhs, rhs, rel_tol=1e-3, abs_tol=1e-6):
                    logs.append("PASS: Reactor geometry satisfies minimum-leakage condition")
                else:
                    logs.append(f"FAIL: Reactor geometry relation mismatch. LHS={lhs:.6f}, RHS={rhs:.6f}")
                    success = False

            # F_n
            F_n = self._to_float(self._get("F_n"))
            if F_n is None:
                logs.append("FAIL: Missing F_n")
                success = False
            elif math.isclose(F_n, self.ref_F_n_int, rel_tol=1e-3, abs_tol=1.0):
                logs.append("PASS: F_n consistent with square lattice estimate")
            else:
                logs.append(f"FAIL: F_n mismatch. Expected {self.ref_F_n_int}, got {F_n}")
                success = False

            # M
            M_val = self._to_float(self._get("M"))
            if M_val is None:
                logs.append("FAIL: Missing M")
                success = False
            elif self._is_close(M_val, self.ref_M, rel_tol=1e-3, abs_tol=1e2):
                logs.append("PASS: M consistent with channel geometry and UO2 density")
            else:
                logs.append(f"FAIL: M mismatch. Expected {self.ref_M:.3e}, got {M_val:.3e}")
                success = False

        except Exception as e:
            logs.append(f"ERROR: test_invariants crashed: {str(e)}")
            logs.append(traceback.format_exc())
            success = False

        return success, logs

    def test_formula_consistency(self) -> Tuple[bool, List[str]]:
        logs = []
        success = True

        try:
            # T_c - T_s formula
            T_formula = self._get("T_diff_formula")
            expr = self._parse_expr(T_formula)
            Q, a, lambda_ = sp.symbols("Q a lambda_")
            expected_T = sp.simplify(Q * a**2 / (4 * lambda_))

            if expr is None:
                logs.append("FAIL: Missing or unparsable T_diff_formula")
                success = False
            elif sp.simplify(expr - expected_T) == 0:
                logs.append("PASS: Temperature-difference formula correct")
            else:
                logs.append(f"FAIL: Temperature-difference formula incorrect. Got {expr}, expected {expected_T}")
                success = False

            # v formula
            v_formula = self._get("v_formula")
            expr_v = self._parse_expr(v_formula)
            A, v_b = sp.symbols("A v_b")
            expected_v = sp.simplify(A * v_b / (A + 1))

            if expr_v is None:
                logs.append("FAIL: Missing or unparsable v formula")
                success = False
            elif sp.simplify(expr_v - expected_v) == 0:
                logs.append("PASS: v formula correct")
            else:
                logs.append(f"FAIL: v formula incorrect. Got {expr_v}, expected {expected_v}")
                success = False

            # V formula
            V_formula = self._get("V_formula")
            expr_V = self._parse_expr(V_formula)
            expected_V = sp.simplify(v_b / (A + 1))

            if expr_V is None:
                logs.append("FAIL: Missing or unparsable V formula")
                success = False
            elif sp.simplify(expr_V - expected_V) == 0:
                logs.append("PASS: V formula correct")
            else:
                logs.append(f"FAIL: V formula incorrect. Got {expr_V}, expected {expected_V}")
                success = False

            # G formula
            G_formula = self._get("G_formula")
            expr_G = self._parse_expr(G_formula)
            alpha, theta = sp.symbols("alpha theta")
            expected_G = sp.simplify(1 - (1 - alpha) * (1 - sp.cos(theta)) / 2)

            if expr_G is None:
                logs.append("FAIL: Missing or unparsable G formula")
                success = False
            elif sp.simplify(expr_G - expected_G) == 0:
                logs.append("PASS: G(alpha, theta) formula correct")
            else:
                logs.append(f"FAIL: G formula incorrect. Got {expr_G}, expected {expected_G}")
                success = False

        except Exception as e:
            logs.append(f"ERROR: test_formula_consistency crashed: {str(e)}")
            logs.append(traceback.format_exc())
            success = False

        return success, logs

    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        logs = []
        success = True

        try:
            checks = [
                ("DeltaE", self.ref_DeltaE, 1e-3, 1e-3),
                ("N", self.ref_N, 1e-3, 1e23),
                ("Q", self.ref_Q, 1e-3, 1e5),
                ("a_u", self.ref_a_u, 1e-3, 1e-5),
                ("f_l", self.ref_f_l, 1e-3, 1e-4),
                ("H", self.ref_H, 1e-3, 1e-3),
                ("R", self.ref_R, 1e-3, 1e-3),
                ("F_n", self.ref_F_n_int, 1e-3, 1.0),
                ("M", self.ref_M, 1e-3, 1e2),
            ]

            for name, ref, rel_tol, abs_tol in checks:
                val = self._to_float(self._get(name))
                if val is None:
                    logs.append(f"FAIL: Missing {name}")
                    success = False
                elif not math.isclose(val, ref, rel_tol=rel_tol, abs_tol=abs_tol):
                    logs.append(f"FAIL: {name} differs from reference. Expected {ref}, got {val}")
                    success = False

            if success:
                logs.append("PASS: All key reference values are consistent")

        except Exception as e:
            logs.append(f"ERROR: test_trace_consistency crashed: {str(e)}")
            logs.append(traceback.format_exc())
            success = False

        return success, logs

    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        logs = []
        success = True

        try:
            # rho up => N up => Q up
            rho2 = 1.1 * self.rho_uo2
            N1_2 = rho2 * self.N_A / self.M_w_uo2
            N2 = self.abundance_u235 * N1_2
            Q2 = N2 * self.sigma_f * self.phi * self.ref_E_f_J
            if N2 > self.ref_N and Q2 > self.ref_Q:
                logs.append("PASS: rho increase leads to larger N and Q")
            else:
                logs.append("FAIL: rho monotonicity broken")
                success = False

            # phi up => Q up
            phi2 = 1.1 * self.phi
            Qphi = self.ref_N * self.sigma_f * phi2 * self.ref_E_f_J
            if Qphi > self.ref_Q:
                logs.append("PASS: phi increase leads to larger Q")
            else:
                logs.append("FAIL: phi monotonicity broken")
                success = False

            # a up => ΔT up
            a1 = self.ref_a_u
            a2 = 1.1 * a1
            dT1 = self.ref_Q * a1**2 / (4 * self.lambda_)
            dT2 = self.ref_Q * a2**2 / (4 * self.lambda_)
            if dT2 > dT1:
                logs.append("PASS: larger pin radius gives larger temperature rise")
            else:
                logs.append("FAIL: radius monotonicity broken")
                success = False

            # A up => alpha up, f_l down
            A1 = 20.0
            A2 = 22.0
            alpha1 = ((A1 - 1) / (A1 + 1)) ** 2
            alpha2 = ((A2 - 1) / (A2 + 1)) ** 2
            f1 = 1 - alpha1
            f2 = 1 - alpha2
            if alpha2 > alpha1 and f2 < f1:
                logs.append("PASS: larger moderator mass ratio increases alpha and reduces max fractional loss")
            else:
                logs.append("FAIL: collision monotonicity broken")
                success = False

        except Exception as e:
            logs.append(f"ERROR: test_monotonicity crashed: {str(e)}")
            logs.append(traceback.format_exc())
            success = False

        return success, logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        all_logs = []
        overall_success = True

        sections = [
            ("=== Testing Physical Invariants ===", self.test_invariants),
            ("\n=== Testing Formula Consistency ===", self.test_formula_consistency),
            ("\n=== Testing Trace Consistency ===", self.test_trace_consistency),
            ("\n=== Testing Monotonicity ===", self.test_monotonicity),
        ]

        for title, fn in sections:
            all_logs.append(title)
            ok, logs = fn()
            all_logs.extend(logs)
            overall_success = overall_success and ok

        return overall_success, all_logs