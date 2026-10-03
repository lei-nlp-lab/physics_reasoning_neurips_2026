import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs and results
        self.results = getattr(solution_instance, 'results', {})
        self.m = 2.0  # kg
        self.M = 4.0  # kg
        self.r = 2.0  # m
        self.g = 10.0  # m/s^2
        self.L = 6.0  # m

        # Extract quantities with safe fallbacks
        self.a = self._safe_get('a')
        self.b = self._safe_get('b')
        self.h = self._safe_get('h')
        self.v_A = self._safe_get('v_A')
        self.v_B = self._safe_get('v_B')
        self.v_Q = self._safe_get('v_Q')
        self.F = self._safe_get('F')
        self.F_plus_mg = self._safe_get('F+mg')

        # Derived quantities (if needed)
        if self.h is not None:
            self.v_B_from_energy = math.sqrt(2 * self.g * (self.h - 2 * self.r))
        else:
            self.v_B_from_energy = None

    def _safe_get(self, key):
        """Safely retrieve a value from solution instance or results dict."""
        value = getattr(self, key, None)
        if value is None:
            value = getattr(self.solution_instance, key, None) if hasattr(self, 'solution_instance') else None
        if value is None and self.results:
            value = self.results.get(key, None)
        return value

    def test_invariants(self):
        logs = []
        success = True

        # 1. Check F-h relation: F = 20h - 100
        if self.a is not None and self.b is not None:
            if not (math.isclose(self.a, -100.0, rel_tol=1e-2, abs_tol=1e-2) and 
                    math.isclose(self.b, 20.0, rel_tol=1e-2, abs_tol=1e-2)):
                logs.append(f"FAIL: F-h relation coefficients: a={self.a}, b={self.b}, expected a=-100, b=20")
                success = False
            else:
                logs.append("PASS: F-h relation coefficients match expected.")
        else:
            logs.append("SKIP: Missing a or b for F-h relation check.")

        # 2. Energy conservation from release to point B
        if self.h is not None and self.v_B is not None:
            lhs = self.m * self.g * self.h
            rhs = 0.5 * self.m * self.v_B**2 + self.m * self.g * 2 * self.r
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Energy conservation (release to B): mgh={lhs}, (1/2)mv_B^2 + mg(2r)={rhs}")
                success = False
            else:
                logs.append("PASS: Energy conservation (release to B) holds.")
        else:
            logs.append("SKIP: Missing h or v_B for energy conservation check.")

        # 3. Newton's second law at point B: F + mg = m * v_B^2 / r
        if self.F_plus_mg is not None and self.v_B is not None:
            lhs = self.F_plus_mg
            rhs = self.m * self.v_B**2 / self.r
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Newton's second law at B: F+mg={lhs}, m*v_B^2/r={rhs}")
                success = False
            else:
                logs.append("PASS: Newton's second law at B holds.")
        else:
            logs.append("SKIP: Missing F+mg or v_B for Newton's second law check.")

        # 4. Elastic collision: momentum and kinetic energy conservation
        if self.v_A is not None and self.v_Q is not None:
            # Momentum conservation: m * v_A = m * v_A' + M * v_Q, but v_A' is not given.
            # Instead, use standard elastic collision formulas.
            # Compute v_A' from momentum and energy equations symbolically.
            m, M, v_A, v_Q = sp.symbols('m M v_A v_Q', positive=True)
            v_Ap = sp.symbols('v_Ap')
            eq1 = sp.Eq(m * v_A, m * v_Ap + M * v_Q)
            eq2 = sp.Eq(0.5 * m * v_A**2, 0.5 * m * v_Ap**2 + 0.5 * M * v_Q**2)
            sol = sp.solve((eq1, eq2), (v_Ap, v_A), dict=True)
            # There are two solutions; pick the one with v_A > 0 and v_Ap != v_A.
            for s in sol:
                if s[v_A] > 0 and s[v_Ap] != s[v_A]:
                    v_A_expected = float(s[v_A].subs({m: self.m, M: self.M, v_Q: self.v_Q}))
                    break
            if math.isclose(self.v_A, v_A_expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("PASS: Elastic collision momentum and energy consistent.")
            else:
                logs.append(f"FAIL: v_A={self.v_A}, expected from collision formulas={v_A_expected}")
                success = False
        else:
            logs.append("SKIP: Missing v_A or v_Q for elastic collision check.")

        return success, logs

    def test_trace_consistency(self):
        logs = []
        success = True

        # 1. Check v_B computed from energy matches v_B from Newton's law (if both available)
        if self.v_B is not None and self.v_B_from_energy is not None:
            if not math.isclose(self.v_B, self.v_B_from_energy, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_B mismatch: from solution={self.v_B}, from energy={self.v_B_from_energy}")
                success = False
            else:
                logs.append("PASS: v_B consistent between energy and solution.")
        else:
            logs.append("SKIP: Missing v_B or h for trace consistency (v_B).")

        # 2. Check h computed from v_A matches given h
        if self.v_A is not None and self.h is not None:
            h_from_v_A = self.v_A**2 / (2 * self.g)
            if not math.isclose(self.h, h_from_v_A, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: h mismatch: given={self.h}, from v_A={h_from_v_A}")
                success = False
            else:
                logs.append("PASS: h consistent with v_A via energy.")
        else:
            logs.append("SKIP: Missing v_A or h for trace consistency (h).")

        # 3. Check F from F-h relation matches if h and F are given
        if self.h is not None and self.F is not None and self.a is not None and self.b is not None:
            F_from_relation = self.a + self.b * self.h
            if not math.isclose(self.F, F_from_relation, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: F mismatch: given={self.F}, from F-h relation={F_from_relation}")
                success = False
            else:
                logs.append("PASS: F consistent with F-h relation.")
        else:
            logs.append("SKIP: Missing h, F, a, or b for F-h relation consistency.")

        return success, logs

    def test_monotonicity(self):
        logs = []
        success = True

        # Perturb mass m by +0.1 kg and observe change in h (which is derived from v_A)
        if self.v_Q is not None and self.h is not None:
            m_original = self.m
            m_perturbed = m_original + 0.1
            # v_Q is independent of m (determined by friction on Q)
            v_Q = self.v_Q
            # Compute v_A for perturbed mass using elastic collision formula
            v_A_perturbed = (m_perturbed + self.M) / (2 * m_perturbed) * v_Q
            h_perturbed = v_A_perturbed**2 / (2 * self.g)
            # Since m increases, v_A decreases (because (m+M)/(2m) decreases), so h decreases.
            if h_perturbed < self.h:
                logs.append(f"PASS: Monotonicity: increasing m from {m_original} to {m_perturbed} decreases h from {self.h:.2f} to {h_perturbed:.2f}.")
            else:
                logs.append(f"FAIL: Monotonicity: expected h to decrease when m increases, but h_perturbed={h_perturbed:.2f} >= h_original={self.h:.2f}")
                success = False
        else:
            logs.append("SKIP: Missing v_Q or h for monotonicity test.")

        return success, logs

    def run_all_tests(self):
        all_logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"--- {test_name} ---")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False

        return overall_success, all_logs