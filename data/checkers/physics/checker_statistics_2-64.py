import sympy as sp
import math

class PhysicsChecker:
    """
    Auditor for Bose‑Einstein condensation and photon gas problem.
    Uses safe defaults, theorem‑based checks, and explicit monotonicity.
    Invariant tests are self‑consistency checks without hardcoded targets.
    """

    def __init__(self, solution_instance):
        self.solution_instance = solution_instance
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = solution_instance.get('results', {}) if isinstance(solution_instance, dict) else {}

        # Extract the three key results; safe defaults
        self.u_expr = self._parse_expr(self.results.get('u', None))
        self.c_expr = self._parse_expr(self.results.get('c', None))
        self.u_ph_expr = self._parse_expr(self.results.get('u_ph', None))

        # Symbols for evaluation
        self.T = sp.Symbol('T', positive=True)
        self.m = sp.Symbol('m', positive=True)
        self.h = sp.Symbol('h', positive=True)
        self.k = sp.Symbol('k', positive=True)
        self.c_light = sp.Symbol('c', positive=True)

    @staticmethod
    def _parse_expr(expr):
        if expr is None:
            return None
        if isinstance(expr, sp.Basic):
            return expr
        try:
            return sp.sympify(expr)
        except Exception:
            return None

    def _numeric_eval(self, expr, subs_dict):
        """Evaluate expression numerically, replacing any Integral with 1.0.
        Returns float or None on failure."""
        if not isinstance(expr, sp.Basic):
            return None
        try:
            expr_no_int = expr.replace(
                lambda e: isinstance(e, sp.Integral),
                lambda e: sp.Integer(1)
            )
            val = sp.N(expr_no_int.subs(subs_dict))
            return float(val)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_u_exists(self):
        if 'u' not in self.results or self.u_expr is None:
            return (False, "Missing result key 'u'")
        return (True, "Key 'u' present")

    def test_interface_c_exists(self):
        if 'c' not in self.results or self.c_expr is None:
            return (False, "Missing result key 'c'")
        return (True, "Key 'c' present")

    def test_interface_u_ph_exists(self):
        if 'u_ph' not in self.results or self.u_ph_expr is None:
            return (False, "Missing result key 'u_ph'")
        return (True, "Key 'u_ph' present")

    def test_interface_u_type(self):
        if not isinstance(self.u_expr, (sp.Basic, str)):
            return (False, "'u' is not a sympy expression or string")
        return (True, "'u' is valid type")

    def test_interface_c_type(self):
        if not isinstance(self.c_expr, (sp.Basic, str)):
            return (False, "'c' is not a sympy expression or string")
        return (True, "'c' is valid type")

    def test_interface_u_ph_type(self):
        if not isinstance(self.u_ph_expr, (sp.Basic, str)):
            return (False, "'u_ph' is not a sympy expression or string")
        return (True, "'u_ph' is valid type")

    # ------------------------------------------------------------------
    # Invariant tests (theorem‑based, self‑consistency, no hardcoded targets)
    # ------------------------------------------------------------------
    def test_invariant_u_power_law_consistency(self):
        """
        For massive bosons, energy density follows a power law in T.
        The logarithmic derivative d(ln u)/d(ln T) should be constant
        (independent of T). This is a pure self‑consistency check.
        """
        if self.u_expr is None:
            return (False, "SKIP: u not available")
        try:
            subs1 = {self.T: 1.0, self.m: 1.0, self.h: 1.0, self.k: 1.0}
            subs2 = {self.T: 1.1, self.m: 1.0, self.h: 1.0, self.k: 1.0}
            subs3 = {self.T: 1.21, self.m: 1.0, self.h: 1.0, self.k: 1.0}

            u1 = self._numeric_eval(self.u_expr, subs1)
            u2 = self._numeric_eval(self.u_expr, subs2)
            u3 = self._numeric_eval(self.u_expr, subs3)

            if any(v is None for v in (u1, u2, u3)):
                return (False, "Cannot evaluate u at all required T points")

            log_ratio_T = math.log(1.1)  # same for both intervals
            exponent1 = math.log(u2 / u1) / log_ratio_T
            exponent2 = math.log(u3 / u2) / log_ratio_T

            if math.isclose(exponent1, exponent2, rel_tol=0.02):
                return (True, f"u power law consistent: exponents {exponent1:.3f} and {exponent2:.3f}")
            else:
                return (False, f"u exponents differ: {exponent1:.3f} vs {exponent2:.3f}")
        except Exception as e:
            return (False, f"u invariant test error: {e}")

    def test_invariant_u_ph_power_law_consistency(self):
        """
        For photons, energy density follows a power law in T.
        The logarithmic derivative d(ln u_ph)/d(ln T) should be constant.
        """
        if self.u_ph_expr is None:
            return (False, "SKIP: u_ph not available")
        try:
            subs1 = {self.T: 1.0, self.h: 1.0, self.k: 1.0, self.c_light: 1.0}
            subs2 = {self.T: 1.1, self.h: 1.0, self.k: 1.0, self.c_light: 1.0}
            subs3 = {self.T: 1.21, self.h: 1.0, self.k: 1.0, self.c_light: 1.0}

            u1 = self._numeric_eval(self.u_ph_expr, subs1)
            u2 = self._numeric_eval(self.u_ph_expr, subs2)
            u3 = self._numeric_eval(self.u_ph_expr, subs3)

            if any(v is None for v in (u1, u2, u3)):
                return (False, "Cannot evaluate u_ph at all required T points")

            log_ratio_T = math.log(1.1)
            exponent1 = math.log(u2 / u1) / log_ratio_T
            exponent2 = math.log(u3 / u2) / log_ratio_T

            if math.isclose(exponent1, exponent2, rel_tol=0.02):
                return (True, f"u_ph power law consistent: exponents {exponent1:.3f} and {exponent2:.3f}")
            else:
                return (False, f"u_ph exponents differ: {exponent1:.3f} vs {exponent2:.3f}")
        except Exception as e:
            return (False, f"u_ph invariant test error: {e}")

    def test_invariant_c_equals_derivative(self):
        """Heat capacity c must equal du/dT (energy density derivative)."""
        if self.u_expr is None or self.c_expr is None:
            return (False, "SKIP: u or c not available")
        try:
            du_dT = sp.diff(self.u_expr, self.T)
            diff = sp.simplify(du_dT - self.c_expr)
            if diff == 0:
                return (True, "c = du/dT holds exactly")
            # fallback numeric check
            subs = {self.T: 1.0, self.m: 1.0, self.h: 1.0, self.k: 1.0}
            du_num = self._numeric_eval(du_dT, subs)
            c_num = self._numeric_eval(self.c_expr, subs)
            if du_num is not None and c_num is not None and math.isclose(du_num, c_num, rel_tol=0.05):
                return (True, "c ≈ du/dT numerically")
            else:
                return (False, f"c differs from du/dT: {du_num} vs {c_num}")
        except Exception as e:
            return (False, f"Invariant test error: {e}")

    # ------------------------------------------------------------------
    # Modeling tests (structural checks)
    # ------------------------------------------------------------------
    def test_modeling_contains_integral_massive(self):
        """u must contain an Integral object (dimensionless integration)."""
        if self.u_expr is None:
            return (False, "SKIP: u not available")
        try:
            from sympy import Integral
            for sub in sp.preorder_traversal(self.u_expr):
                if isinstance(sub, Integral):
                    return (True, "u contains an integral")
            return (False, "No integral found in u expression")
        except Exception as e:
            return (False, f"Modeling test error: {e}")

    def test_modeling_contains_integral_photon(self):
        """u_ph must contain an Integral object."""
        if self.u_ph_expr is None:
            return (False, "SKIP: u_ph not available")
        try:
            from sympy import Integral
            for sub in sp.preorder_traversal(self.u_ph_expr):
                if isinstance(sub, Integral):
                    return (True, "u_ph contains an integral")
            return (False, "No integral found in u_ph expression")
        except Exception as e:
            return (False, f"Modeling test error: {e}")

    def test_modeling_no_chemical_potential_massive(self):
        """Massive boson expression must not contain μ (μ=0 approximation)."""
        if self.u_expr is None:
            return (False, "SKIP: u not available")
        try:
            if 'mu' in [str(s) for s in self.u_expr.free_symbols]:
                return (False, "Chemical potential symbol found")
            return (True, "No chemical potential present; μ=0 assumed")
        except Exception as e:
            return (False, f"Chemical potential check error: {e}")

    def test_modeling_no_chemical_potential_photon(self):
        """Photon expression must not contain μ (exact μ=0)."""
        if self.u_ph_expr is None:
            return (False, "SKIP: u_ph not available")
        try:
            if 'mu' in [str(s) for s in self.u_ph_expr.free_symbols]:
                return (False, "Chemical potential symbol found")
            return (True, "No chemical potential present; μ=0 for photons")
        except Exception as e:
            return (False, f"Chemical potential check error: {e}")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit perturbation: baseline T=1, perturbed T=1.1)
    # ------------------------------------------------------------------
    def test_monotonicity_u_with_T(self):
        """u should increase when T is increased (perturb T from 1.0 to 1.1)."""
        if self.u_expr is None:
            return (False, "SKIP: u not available")
        try:
            # Baseline: T=1.0
            u_base = self._numeric_eval(self.u_expr, {self.T: 1.0, self.m: 1.0, self.h: 1.0, self.k: 1.0})
            # Perturbed: T=1.1
            u_pert = self._numeric_eval(self.u_expr, {self.T: 1.1, self.m: 1.0, self.h: 1.0, self.k: 1.0})
            if u_base is None or u_pert is None:
                return (False, "Cannot evaluate u numerically")
            if u_pert > u_base:
                return (True, f"u increases with T: baseline={u_base:.3f}, perturbed={u_pert:.3f}")
            else:
                return (False, f"u does not increase: baseline={u_base:.3f}, perturbed={u_pert:.3f}")
        except Exception as e:
            return (False, f"Monotonicity test error: {e}")

    def test_monotonicity_u_ph_with_T(self):
        """u_ph should increase when T is increased (perturb T from 1.0 to 1.1)."""
        if self.u_ph_expr is None:
            return (False, "SKIP: u_ph not available")
        try:
            subs_base = {self.T: 1.0, self.h: 1.0, self.k: 1.0, self.c_light: 1.0}
            subs_pert = {self.T: 1.1, self.h: 1.0, self.k: 1.0, self.c_light: 1.0}
            u_base = self._numeric_eval(self.u_ph_expr, subs_base)
            u_pert = self._numeric_eval(self.u_ph_expr, subs_pert)
            if u_base is None or u_pert is None:
                return (False, "Cannot evaluate u_ph numerically")
            if u_pert > u_base:
                return (True, f"u_ph increases with T: baseline={u_base:.3f}, perturbed={u_pert:.3f}")
            else:
                return (False, f"u_ph does not increase: baseline={u_base:.3f}, perturbed={u_pert:.3f}")
        except Exception as e:
            return (False, f"Photon monotonicity test error: {e}")

    # ------------------------------------------------------------------
    # Single weak answer‑sanity test (positivity only; no numeric targets)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """All available quantities are positive for positive parameters."""
        checks = []
        if self.u_expr is not None:
            val = self._numeric_eval(self.u_expr, {self.T: 1.0, self.m: 1.0, self.h: 1.0, self.k: 1.0})
            if val is None:
                checks.append("u not evaluable")
            elif val <= 0:
                checks.append("u non‑positive")
        if self.c_expr is not None:
            val = self._numeric_eval(self.c_expr, {self.T: 1.0, self.m: 1.0, self.h: 1.0, self.k: 1.0})
            if val is None:
                checks.append("c not evaluable")
            elif val <= 0:
                checks.append("c non‑positive")
        if self.u_ph_expr is not None:
            subs = {self.T: 1.0, self.h: 1.0, self.k: 1.0, self.c_light: 1.0}
            val = self._numeric_eval(self.u_ph_expr, subs)
            if val is None:
                checks.append("u_ph not evaluable")
            elif val <= 0:
                checks.append("u_ph non‑positive")
        if checks:
            return (False, "; ".join(checks))
        return (True, "All available quantities positive for positive parameters")

    # ------------------------------------------------------------------
    # Main test runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)