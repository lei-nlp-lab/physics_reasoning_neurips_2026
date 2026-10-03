import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {}) or {}

        def get_res(*keys, default=None):
            for key in keys:
                if hasattr(self.solution, key):
                    val = getattr(self.solution, key)
                    if val is not None:
                        return val
                if key in self.results and self.results[key] is not None:
                    return self.results[key]
            return default

        # Extract results
        self.v_P1 = get_res('v_P1')
        self.v_Q1 = get_res('v_Q1')

        self.speed_P1 = get_res('speed_P1', 'abs_v_P1', '|v_P1|')
        self.speed_Q1 = get_res('speed_Q1', 'abs_v_Q1', '|v_Q1|')

        self.h1 = get_res('h_1', 'h1')
        self.h2 = get_res('h_2', 'h2')
        self.h3 = get_res('h_3', 'h3')
        self.h_n = get_res('h_n', 'hn')

        self.v02 = get_res('v_02', 'v02')
        self.v_P2 = get_res('v_P2')
        self.v_Q2 = get_res('v_Q2')

        self.v03 = get_res('v_03', 'v03')
        self.v_P3 = get_res('v_P3')
        self.v_Q3 = get_res('v_Q3')

        self.H = get_res('H')
        self.s = get_res('s')

        # Checker default constants for numerical verification
        self.m = 1.0
        self.v0 = 1.0
        self.g = 9.8
        self.theta = math.pi / 6  # 30 degrees
        self.sin_theta = math.sin(self.theta)

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        if a is None or b is None:
            return False
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except Exception:
            return False

    def _parse_hn(self, value, n):
        """
        Allow either:
        1. a callable h_n(n)
        2. a string formula using n, v0, g
        3. numeric value when n is fixed externally (not used here)
        """
        if value is None:
            return None

        if callable(value):
            try:
                return float(value(n))
            except Exception:
                return None

        if isinstance(value, str):
            expr = value.strip().replace('^', '**')
            local_vars = {
                'n': n,
                'v0': self.v0,
                'g': self.g,
                'math': math,
                'sqrt': math.sqrt,
            }
            try:
                return float(eval(expr, {"__builtins__": {}}, local_vars))
            except Exception:
                return None

        return None

    def test_invariants(self) -> Tuple[bool, List[str]]:
        logs = []
        success = True

        try:
            # 1) First collision
            expected_vP1 = -3 * self.v0 / 5
            expected_vQ1 =  2 * self.v0 / 5
            expected_speedP1 = 3 * self.v0 / 5
            expected_speedQ1 = 2 * self.v0 / 5

            if self.v_P1 is not None and self.v_Q1 is not None:
                lhs_mom = self.m * self.v0
                rhs_mom = self.m * self.v_P1 + 4 * self.m * self.v_Q1
                if not self._is_close(lhs_mom, rhs_mom):
                    logs.append(f"FAIL: Momentum not conserved in first collision. LHS={lhs_mom}, RHS={rhs_mom}")
                    success = False
                else:
                    logs.append("PASS: Momentum conserved in first collision")

                lhs_ke = 0.5 * self.m * self.v0**2
                rhs_ke = 0.5 * self.m * self.v_P1**2 + 0.5 * 4 * self.m * self.v_Q1**2
                if not self._is_close(lhs_ke, rhs_ke):
                    logs.append(f"FAIL: KE not conserved in first collision. LHS={lhs_ke}, RHS={rhs_ke}")
                    success = False
                else:
                    logs.append("PASS: KE conserved in first collision")

                if not self._is_close(self.v_P1, expected_vP1):
                    logs.append(f"FAIL: v_P1 mismatch. Expected signed {expected_vP1}, got {self.v_P1}")
                    success = False
                else:
                    logs.append("PASS: v_P1 signed value correct")

                if not self._is_close(self.v_Q1, expected_vQ1):
                    logs.append(f"FAIL: v_Q1 mismatch. Expected {expected_vQ1}, got {self.v_Q1}")
                    success = False
                else:
                    logs.append("PASS: v_Q1 signed value correct")

            else:
                # Allow magnitude-only answers for subquestion 1
                if self.speed_P1 is not None:
                    if not self._is_close(self.speed_P1, expected_speedP1):
                        logs.append(f"FAIL: |v_P1| mismatch. Expected {expected_speedP1}, got {self.speed_P1}")
                        success = False
                    else:
                        logs.append("PASS: |v_P1| correct")
                else:
                    logs.append("FAIL: Missing v_P1 or |v_P1|")
                    success = False

                if self.speed_Q1 is not None:
                    if not self._is_close(self.speed_Q1, expected_speedQ1):
                        logs.append(f"FAIL: |v_Q1| mismatch. Expected {expected_speedQ1}, got {self.speed_Q1}")
                        success = False
                    else:
                        logs.append("PASS: |v_Q1| correct")
                else:
                    logs.append("FAIL: Missing v_Q1 or |v_Q1|")
                    success = False

            # 2) Q after first collision: h1 = v_Q1^2 / (4g)
            if self.h1 is not None:
                expected_h1 = self.v0**2 / (25 * self.g)
                if not self._is_close(self.h1, expected_h1):
                    logs.append(f"FAIL: h1 mismatch. Expected {expected_h1}, got {self.h1}")
                    success = False
                else:
                    logs.append("PASS: h1 correct")

            # 3) P before second collision
            if self.v02 is not None:
                expected_v02 = math.sqrt(7) * self.v0 / 5
                if not self._is_close(self.v02, expected_v02):
                    logs.append(f"FAIL: v02 mismatch. Expected {expected_v02}, got {self.v02}")
                    success = False
                else:
                    logs.append("PASS: v02 correct")

            # 4) Second collision
            if self.v02 is not None and self.v_P2 is not None and self.v_Q2 is not None:
                lhs_mom2 = self.m * self.v02
                rhs_mom2 = self.m * self.v_P2 + 4 * self.m * self.v_Q2
                if not self._is_close(lhs_mom2, rhs_mom2):
                    logs.append(f"FAIL: Momentum not conserved in second collision. LHS={lhs_mom2}, RHS={rhs_mom2}")
                    success = False
                else:
                    logs.append("PASS: Momentum conserved in second collision")

                lhs_ke2 = 0.5 * self.m * self.v02**2
                rhs_ke2 = 0.5 * self.m * self.v_P2**2 + 0.5 * 4 * self.m * self.v_Q2**2
                if not self._is_close(lhs_ke2, rhs_ke2):
                    logs.append(f"FAIL: KE not conserved in second collision. LHS={lhs_ke2}, RHS={rhs_ke2}")
                    success = False
                else:
                    logs.append("PASS: KE conserved in second collision")

            # 5) h2 / h1 = 7/25
            if self.h1 is not None and self.h2 is not None and self.h1 != 0:
                ratio = self.h2 / self.h1
                if not self._is_close(ratio, 7/25):
                    logs.append(f"FAIL: h2/h1 mismatch. Expected {7/25}, got {ratio}")
                    success = False
                else:
                    logs.append("PASS: h2/h1 ratio correct")

            # 6) h3 / h2 = 7/25
            if self.h2 is not None and self.h3 is not None and self.h2 != 0:
                ratio = self.h3 / self.h2
                if not self._is_close(ratio, 7/25):
                    logs.append(f"FAIL: h3/h2 mismatch. Expected {7/25}, got {ratio}")
                    success = False
                else:
                    logs.append("PASS: h3/h2 ratio correct")

            # 7) General h_n formula
            if self.h_n is not None:
                expected_h1 = (7/25)**0 * self.v0**2 / (25*self.g)
                expected_h2 = (7/25)**1 * self.v0**2 / (25*self.g)
                expected_h3 = (7/25)**2 * self.v0**2 / (25*self.g)

                h1_from_formula = self._parse_hn(self.h_n, 1)
                h2_from_formula = self._parse_hn(self.h_n, 2)
                h3_from_formula = self._parse_hn(self.h_n, 3)

                if h1_from_formula is None or h2_from_formula is None or h3_from_formula is None:
                    logs.append("FAIL: Could not evaluate h_n formula")
                    success = False
                elif (
                    self._is_close(h1_from_formula, expected_h1) and
                    self._is_close(h2_from_formula, expected_h2) and
                    self._is_close(h3_from_formula, expected_h3)
                ):
                    logs.append("PASS: h_n formula correct")
                else:
                    logs.append(
                        f"FAIL: h_n formula mismatch. "
                        f"Expected ({expected_h1}, {expected_h2}, {expected_h3}), "
                        f"got ({h1_from_formula}, {h2_from_formula}, {h3_from_formula})"
                    )
                    success = False

            # 8) Overall energy balance -> H = v0^2/(18g)
            if self.H is not None:
                expected_H = self.v0**2 / (18*self.g)
                if not self._is_close(self.H, expected_H):
                    logs.append(f"FAIL: H mismatch. Expected {expected_H}, got {self.H}")
                    success = False
                else:
                    logs.append("PASS: H correct")

            # 9) Minimum bumper distance
            if self.s is not None:
                expected_s = (8*math.sqrt(7)-13)*self.v0**2/(200*self.g*self.sin_theta)
                if not self._is_close(self.s, expected_s):
                    logs.append(f"FAIL: s mismatch. Expected {expected_s}, got {self.s}")
                    success = False
                else:
                    logs.append("PASS: s correct")

        except Exception:
            logs.append(f"ERROR in test_invariants:\n{traceback.format_exc()}")
            success = False

        return success, logs

    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        logs = []
        success = True

        try:
            if self.v_P2 is not None:
                expected = -3 * math.sqrt(7) / 25 * self.v0
                if not self._is_close(self.v_P2, expected):
                    logs.append(f"FAIL: v_P2 mismatch. Expected {expected}, got {self.v_P2}")
                    success = False
                else:
                    logs.append("PASS: v_P2 correct")

            if self.v_Q2 is not None:
                expected = 2 * math.sqrt(7) / 25 * self.v0
                if not self._is_close(self.v_Q2, expected):
                    logs.append(f"FAIL: v_Q2 mismatch. Expected {expected}, got {self.v_Q2}")
                    success = False
                else:
                    logs.append("PASS: v_Q2 correct")

            if self.v03 is not None:
                expected = (math.sqrt(7)/5)**2 * self.v0
                if not self._is_close(self.v03, expected):
                    logs.append(f"FAIL: v03 mismatch. Expected {expected}, got {self.v03}")
                    success = False
                else:
                    logs.append("PASS: v03 correct")

            if self.v_P3 is not None:
                expected = -3/5 * (math.sqrt(7)/5)**2 * self.v0
                if not self._is_close(self.v_P3, expected):
                    logs.append(f"FAIL: v_P3 mismatch. Expected {expected}, got {self.v_P3}")
                    success = False
                else:
                    logs.append("PASS: v_P3 correct")

            if self.v_Q3 is not None:
                expected = 2/5 * (math.sqrt(7)/5)**2 * self.v0
                if not self._is_close(self.v_Q3, expected):
                    logs.append(f"FAIL: v_Q3 mismatch. Expected {expected}, got {self.v_Q3}")
                    success = False
                else:
                    logs.append("PASS: v_Q3 correct")

            if self.h2 is not None:
                expected = (7/25) * self.v0**2 / (25*self.g)
                if not self._is_close(self.h2, expected):
                    logs.append(f"FAIL: h2 mismatch. Expected {expected}, got {self.h2}")
                    success = False
                else:
                    logs.append("PASS: h2 correct")

            if self.h3 is not None:
                expected = (7/25)**2 * self.v0**2 / (25*self.g)
                if not self._is_close(self.h3, expected):
                    logs.append(f"FAIL: h3 mismatch. Expected {expected}, got {self.h3}")
                    success = False
                else:
                    logs.append("PASS: h3 correct")

            if success and not logs:
                logs.append("PASS: Trace consistency checks passed")

        except Exception:
            logs.append(f"ERROR in test_trace_consistency:\n{traceback.format_exc()}")
            success = False

        return success, logs

    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        logs = []
        success = True

        try:
            factor = 1.01
            v0_new = self.v0 * factor
            expected_scale = factor**2

            H_old = self.v0**2 / (18*self.g)
            H_new = v0_new**2 / (18*self.g)
            if not self._is_close(H_new/H_old, expected_scale, rel_tol=1e-3, abs_tol=1e-3):
                logs.append(f"FAIL: H does not scale as v0^2. Got {H_new/H_old}, expected {expected_scale}")
                success = False
            else:
                logs.append("PASS: H scales as v0^2")

            h1_old = self.v0**2 / (25*self.g)
            h1_new = v0_new**2 / (25*self.g)
            if not self._is_close(h1_new/h1_old, expected_scale, rel_tol=1e-3, abs_tol=1e-3):
                logs.append(f"FAIL: h1 does not scale as v0^2. Got {h1_new/h1_old}, expected {expected_scale}")
                success = False
            else:
                logs.append("PASS: h1 scales as v0^2")

            s_old = (8*math.sqrt(7)-13)*self.v0**2/(200*self.g*self.sin_theta)
            s_new = (8*math.sqrt(7)-13)*v0_new**2/(200*self.g*self.sin_theta)
            if not self._is_close(s_new/s_old, expected_scale, rel_tol=1e-3, abs_tol=1e-3):
                logs.append(f"FAIL: s does not scale as v0^2. Got {s_new/s_old}, expected {expected_scale}")
                success = False
            else:
                logs.append("PASS: s scales as v0^2")

        except Exception:
            logs.append(f"ERROR in test_monotonicity:\n{traceback.format_exc()}")
            success = False

        return success, logs

    def run_all_tests(self):
        all_logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]

        for test_name, test_func in tests:
            all_logs.append(f"Running {test_name} test...")
            success, logs = test_func()
            overall_success = overall_success and success
            all_logs.extend(logs)

        return overall_success, all_logs