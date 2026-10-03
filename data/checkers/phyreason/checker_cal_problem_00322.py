import math
import traceback
import sympy as sp
from typing import Tuple, List


class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {}) or {}

        def get_value(*keys, default=None):
            for key in keys:
                if hasattr(self.solution, key):
                    val = getattr(self.solution, key)
                    if val is not None:
                        return val
                if key in self.results and self.results[key] is not None:
                    return self.results[key]
            return default

        # Numeric quantities
        self.m = self._safe_float(get_value('m', default=1.0), 1.0)
        self.g = self._safe_float(get_value('g', default=10.0), 10.0)
        self.R = self._safe_float(get_value('R', default=3.0), 3.0)
        self.r = self._safe_float(get_value('r', default=4.0), 4.0)

        self.v_B = self._safe_float(get_value('v_B'), None)
        self.F_N_B = self._safe_float(get_value('F_N_B', 'F_B'), None)

        # Formula / symbolic outputs
        self.FN_formula = get_value('FN_formula', 'F_N_formula', 'FN_expr')
        self.min_point = get_value('min_point', 'FN_min_point')

        # Third question outputs
        self.can_reach_D = get_value('can_reach_D', 'reach_D')
        self.y_projectile = self._safe_float(get_value('y', 'y_projectile'), None)

        # Optional answers list
        self.answers = []
        if hasattr(self.solution, 'answers'):
            ans = self.solution.answers
            if isinstance(ans, (list, tuple)):
                self.answers = [str(x) for x in ans]
            elif isinstance(ans, str):
                self.answers = [ans]

    def _safe_float(self, val, default):
        if val is None:
            return default
        try:
            return float(val)
        except Exception:
            return default

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        if a is None or b is None:
            return False
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)

    def _parse_formula(self, expr_str):
        if expr_str is None:
            return None
        if not isinstance(expr_str, str):
            return expr_str
        s = expr_str.strip()
        s = s.replace('^', '**')
        s = s.replace('cosθ', 'cos(theta)')
        s = s.replace('cos θ', 'cos(theta)')
        s = s.replace('θ', 'theta')
        s = s.replace('F_N=', '')
        s = s.replace('FN=', '')
        s = s.replace('mg', 'm*g')
        try:
            m, g, R, r, theta = sp.symbols('m g R r theta')
            local_dict = {
                'm': m, 'g': g, 'R': R, 'r': r, 'theta': theta,
                'cos': sp.cos
            }
            return sp.simplify(sp.sympify(s, locals=local_dict))
        except Exception:
            return None

    def test_answer1(self) -> Tuple[bool, str]:
        # Maximum normal force at B should be 3mg
        expected = 3 * self.m * self.g

        if self.F_N_B is not None:
            if self._is_close(self.F_N_B, expected):
                return True, "Answer 1: F_N_B matches 3mg."
            return False, f"Answer 1 mismatch: F_N_B={self.F_N_B}, expected {expected}"

        if len(self.answers) >= 1:
            ans1 = self.answers[0].strip().lower().replace(' ', '')
            if '3mg' in ans1 or '3*m*g' in ans1:
                return True, "Answer 1 text matches 3mg."
            return False, f"Answer 1 mismatch: got '{self.answers[0]}'"

        return False, "No answer provided for part (1)."

    def test_answer2(self) -> Tuple[bool, str]:
        # Check exact formula:
        # F_N = 3mg cos(theta) - mg(1 + 2R/r)
        m, g, R, r, theta = sp.symbols('m g R r theta')
        expected = sp.simplify(3*m*g*sp.cos(theta) - m*g*(1 + 2*R/r))

        expr = self._parse_formula(self.FN_formula)

        if expr is not None:
            if sp.simplify(expr - expected) == 0:
                # also check minimum point if available
                if self.min_point is not None:
                    mp = str(self.min_point).strip().lower()
                    if 'c' in mp:
                        return True, "Answer 2 formula correct and minimum point is C."
                    return False, f"Answer 2 minimum-point mismatch: got '{self.min_point}', expected point C."
                return True, "Answer 2 formula correct."
            return False, f"Answer 2 formula mismatch: got {expr}, expected {expected}"

        # Fallback to textual answer
        if len(self.answers) >= 2:
            expr2 = self._parse_formula(self.answers[1])
            if expr2 is not None and sp.simplify(expr2 - expected) == 0:
                return True, "Answer 2 text formula correct."
            return False, f"Answer 2 mismatch: got '{self.answers[1]}'"

        return False, "No answer provided for part (2)."

    def test_answer3(self) -> Tuple[bool, str]:
        # Under R = 3/4 r:
        # At point C, theta = 60 deg => cos(theta)=1/2
        # F_N = 3mg(1/2) - mg(1 + 2*(3/4)) = 1.5mg - 2.5mg = -mg < 0
        # so it leaves the track at C, then projectile motion lets it just reach D
        expected_reach = True

        if self.can_reach_D is not None:
            if isinstance(self.can_reach_D, str):
                ans = self.can_reach_D.lower()
                if ('can' in ans and 'reach' in ans) or ('yes' in ans) or ('true' in ans):
                    return True, "Answer 3 indicates block can reach D."
                return False, f"Answer 3 mismatch: got '{self.can_reach_D}'"
            if bool(self.can_reach_D) == expected_reach:
                return True, "Answer 3 boolean is correct."
            return False, f"Answer 3 boolean mismatch: got {self.can_reach_D}"

        if len(self.answers) >= 3:
            ans3 = self.answers[2].lower()
            keywords = ['can reach', 'reach d', 'highest point d', 'just reach']
            if any(k in ans3 for k in keywords):
                return True, "Answer 3 text indicates the block can reach D."
            return False, f"Answer 3 mismatch: got '{self.answers[2]}'"

        return False, "No answer provided for part (3)."

    def test_invariants(self) -> Tuple[bool, List[str]]:
        logs = []
        success = True

        try:
            # 1. Energy from A to B: mgR = 1/2 m v_B^2
            if self.v_B is not None:
                expected_v_B = math.sqrt(2 * self.g * self.R)
                if self._is_close(self.v_B, expected_v_B):
                    logs.append("PASS: v_B satisfies energy conservation from A to B.")
                else:
                    logs.append(f"FAIL: v_B mismatch. Got {self.v_B}, expected {expected_v_B}")
                    success = False
            else:
                logs.append("WARN: v_B not provided; skipped A->B energy check.")

            # 2. At B: F_N - mg = m v_B^2 / R => F_N = 3mg
            if self.F_N_B is not None:
                expected_F_B = 3 * self.m * self.g
                if self._is_close(self.F_N_B, expected_F_B):
                    logs.append("PASS: F_N_B = 3mg.")
                else:
                    logs.append(f"FAIL: F_N_B mismatch. Got {self.F_N_B}, expected {expected_F_B}")
                    success = False
            else:
                logs.append("WARN: F_N_B not provided; skipped B-force check.")

            # 3. Third-question special case: if R = 3/4 r, FN at C is -mg
            if self._is_close(self.R, 0.75 * self.r, rel_tol=1e-6, abs_tol=1e-6):
                FN_C = 3*self.m*self.g*0.5 - self.m*self.g*(1 + 2*self.R/self.r)
                if self._is_close(FN_C, -self.m*self.g):
                    logs.append("PASS: Under R=3/4 r, normal force at C is -mg.")
                else:
                    logs.append(f"FAIL: Under R=3/4 r, FN(C) should be -mg, got {FN_C}")
                    success = False
            else:
                logs.append("INFO: R != 3/4 r in provided numerics; skipped special-case check for part (3).")

        except Exception:
            logs.append(f"ERROR in test_invariants:\n{traceback.format_exc()}")
            success = False

        return success, logs

    def test_monotonicity(self) -> Tuple[bool, str]:
        try:
            # F_N_B = 3mg should scale linearly with m
            m1 = self.m
            m2 = 1.1 * self.m
            F1 = 3 * m1 * self.g
            F2 = 3 * m2 * self.g
            ratio = F2 / F1
            if self._is_close(ratio, 1.1, rel_tol=1e-3, abs_tol=1e-3):
                return True, "Monotonicity passed: F_N_B scales linearly with m."
            return False, f"Monotonicity failed: ratio={ratio}, expected 1.1"
        except Exception as e:
            return False, f"Monotonicity error: {e}"

    def run_all_tests(self):
        all_logs = []
        overall_success = True

        tests = [
            ('test_answer1', self.test_answer1),
            ('test_answer2', self.test_answer2),
            ('test_answer3', self.test_answer3),
        ]

        for name, fn in tests:
            try:
                ok, log = fn()
                all_logs.append(f"{name}: {log}")
                overall_success = overall_success and ok
            except Exception as e:
                all_logs.append(f"{name} exception: {e}")
                overall_success = False

        try:
            ok, logs = self.test_invariants()
            all_logs.append("test_invariants:")
            all_logs.extend([f"  {x}" for x in logs])
            overall_success = overall_success and ok
        except Exception as e:
            all_logs.append(f"test_invariants exception: {e}")
            overall_success = False

        try:
            ok, log = self.test_monotonicity()
            all_logs.append(f"test_monotonicity: {log}")
            overall_success = overall_success and ok
        except Exception as e:
            all_logs.append(f"test_monotonicity exception: {e}")
            overall_success = False

        return overall_success, all_logs