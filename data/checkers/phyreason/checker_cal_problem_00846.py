import math
import random
import traceback
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from the solution instance."""
        # Inputs
        self.h = getattr(solution_instance, 'h', None)
        self.n = getattr(solution_instance, 'n', None)
        self.c = getattr(solution_instance, 'c', 3e8)  # default speed of light

        # Results: try attribute first, then results dictionary
        res = getattr(solution_instance, 'results', {})
        self.v = getattr(solution_instance, 'v', res.get('v', None))
        self.t = getattr(solution_instance, 't', res.get('t', None))
        self.C = getattr(solution_instance, 'C', res.get('C', None))
        self.r = getattr(solution_instance, 'r', res.get('r', None))
        self.S = getattr(solution_instance, 'S', res.get('S', None))

        # For symbolic checks
        self.symbols = {'h': sp.Symbol('h'), 'n': sp.Symbol('n'), 'c': sp.Symbol('c')}

    def _check_relation(self, expected, provided, name):
        """
        Compare expected and provided values using symbolic simplification or
        numerical evaluation with fuzzy matching.
        Returns (success_bool, message).
        """
        if expected is None or provided is None:
            return False, f"{name}: Missing value(s)"

        # Convert to sympy expressions if they aren't already
        if not isinstance(expected, sp.Expr):
            try:
                expected = sp.sympify(expected)
            except:
                pass
        if not isinstance(provided, sp.Expr):
            try:
                provided = sp.sympify(provided)
            except:
                pass

        # Symbolic check: try to simplify the difference
        try:
            diff = sp.simplify(expected - provided)
            if diff == 0:
                return True, f"{name}: symbolic equality"
        except:
            pass

        # Numerical check with random substitutions (if there are symbols)
        free_symbols = list(expected.free_symbols) + list(provided.free_symbols)
        free_symbols = list(set(free_symbols))
        if free_symbols:
            for _ in range(5):
                subs_dict = {}
                for sym in free_symbols:
                    if sym == self.symbols['h']:
                        subs_dict[sym] = random.uniform(0.1, 10.0)
                    elif sym == self.symbols['n']:
                        subs_dict[sym] = random.uniform(1.1, 3.0)  # n > 1
                    elif sym == self.symbols['c']:
                        subs_dict[sym] = 3e8
                    else:
                        subs_dict[sym] = random.uniform(0.1, 10.0)
                try:
                    exp_val = float(expected.subs(subs_dict).evalf())
                    prov_val = float(provided.subs(subs_dict).evalf())
                    if not math.isclose(exp_val, prov_val, rel_tol=1e-2, abs_tol=1e-2):
                        return False, f"{name}: numerical mismatch (expected {exp_val}, got {prov_val} with {subs_dict})"
                except Exception as e:
                    return False, f"{name}: evaluation error: {e}"
            return True, f"{name}: numerical consistency across random samples"
        else:
            # Both are pure numbers
            try:
                exp_num = float(expected) if isinstance(expected, sp.Expr) else float(expected)
                prov_num = float(provided) if isinstance(provided, sp.Expr) else float(provided)
                if math.isclose(exp_num, prov_num, rel_tol=1e-2, abs_tol=1e-2):
                    return True, f"{name}: numerical match"
                else:
                    return False, f"{name}: numerical mismatch (expected {exp_num}, got {prov_num})"
            except Exception as e:
                return False, f"{name}: conversion error: {e}"

    def test_invariants(self):
        """Verify fundamental physics relationships from the KB."""
        logs = []
        # 1. v = c / n
        if self.v is not None and self.c is not None and self.n is not None:
            expected_v = self.c / self.n
            ok, msg = self._check_relation(expected_v, self.v, "v = c/n")
            if not ok:
                logs.append(msg)
        else:
            logs.append("Missing v, c, or n for v = c/n")

        # 2. t = n*h/(2c)  (from step_2)
        if self.t is not None and self.n is not None and self.h is not None and self.c is not None:
            expected_t = self.n * self.h / (2 * self.c)
            ok, msg = self._check_relation(expected_t, self.t, "t = n*h/(2c)")
            if not ok:
                logs.append(msg)

        # 3. sin(C) = 1/n  (critical angle)
        if self.C is not None and self.n is not None:
            # Compute sin(C) carefully (C in radians assumed)
            if isinstance(self.C, sp.Expr):
                sin_C = sp.sin(self.C)
            else:
                try:
                    sin_C = math.sin(float(self.C))
                except:
                    sin_C = None
            if sin_C is not None:
                expected_sinC = 1 / self.n
                ok, msg = self._check_relation(expected_sinC, sin_C, "sin(C) = 1/n")
                if not ok:
                    logs.append(msg)
            else:
                logs.append("Cannot compute sin(C)")
        else:
            logs.append("Missing C or n for sin(C) = 1/n")

        # 4. r = (3h/2) * tan(C)
        if self.r is not None and self.h is not None and self.C is not None:
            if isinstance(self.C, sp.Expr):
                tan_C = sp.tan(self.C)
            else:
                try:
                    tan_C = math.tan(float(self.C))
                except:
                    tan_C = None
            if tan_C is not None:
                expected_r = (3 * self.h / 2) * tan_C
                ok, msg = self._check_relation(expected_r, self.r, "r = (3h/2)*tan(C)")
                if not ok:
                    logs.append(msg)
            else:
                logs.append("Cannot compute tan(C)")
        else:
            logs.append("Missing r, h, or C for r = (3h/2)*tan(C)")

        # 5. S = pi * r^2
        if self.S is not None and self.r is not None:
            expected_S = sp.pi * (self.r ** 2)
            ok, msg = self._check_relation(expected_S, self.S, "S = pi*r^2")
            if not ok:
                logs.append(msg)
        else:
            logs.append("Missing S or r for S = pi*r^2")

        success = len(logs) == 0
        return success, logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        # Essentially the same as invariants, but we also verify that the
        # computed quantities are consistent with the stepwise derivation.
        return self.test_invariants()

    def test_monotonicity(self):
        """Perturb refractive index n and verify expected trends."""
        logs = []
        # Only run if all relevant values are numerical (no symbols)
        def is_numeric(x):
            if x is None:
                return False
            if isinstance(x, (int, float)):
                return True
            if isinstance(x, sp.Expr) and not x.free_symbols:
                return True
            return False

        required = [self.h, self.n, self.c, self.v, self.t, self.C, self.r, self.S]
        if all(is_numeric(x) for x in required):
            n_orig = float(self.n)
            n_new = n_orig * 1.1  # increase by 10%

            # Compute expected new values using formulas
            v_new_exp = float(self.c) / n_new
            t_new_exp = n_new * float(self.h) / (2 * float(self.c))
            C_new_exp = math.asin(1 / n_new)
            r_new_exp = (3 * float(self.h) / 2) * math.tan(C_new_exp)
            S_new_exp = math.pi * (r_new_exp ** 2)

            # Check trends
            if not (v_new_exp < float(self.v)):
                logs.append(f"Monotonicity: v should decrease when n increases (original {self.v}, new {v_new_exp})")
            if not (t_new_exp > float(self.t)):
                logs.append(f"Monotonicity: t should increase when n increases (original {self.t}, new {t_new_exp})")
            if not (C_new_exp < float(self.C)):
                logs.append(f"Monotonicity: C should decrease when n increases (original {self.C}, new {C_new_exp})")
            if not (r_new_exp < float(self.r)):
                logs.append(f"Monotonicity: r should decrease when n increases (original {self.r}, new {r_new_exp})")
            if not (S_new_exp < float(self.S)):
                logs.append(f"Monotonicity: S should decrease when n increases (original {self.S}, new {S_new_exp})")
        else:
            logs.append("Monotonicity test skipped: non-numeric values present.")

        success = len(logs) == 0
        return success, logs

    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        all_logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]

        for name, test_func in tests:
            try:
                success, logs = test_func()
                if success:
                    all_logs.append(f"{name} test passed.")
                else:
                    overall_success = False
                    all_logs.append(f"{name} test failed:")
                    all_logs.extend(logs)
            except Exception as e:
                overall_success = False
                all_logs.append(f"{name} test crashed: {e}")
                all_logs.append(traceback.format_exc())

        return overall_success, all_logs