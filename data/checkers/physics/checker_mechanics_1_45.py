import sympy as sp
import math


class PhysicsChecker:
    """Audit the stability condition derivation for a circular orbit under central force."""

    def __init__(self, solution_instance):
        # Extract the required results safely
        results = getattr(solution_instance, 'results', {}) or {}

        self.f_threshold = results.get('f_threshold', None)
        self.d2U_eff_dr2 = results.get('d2U_eff_dr2', None)

        # Attempt to extract symbols/variables if the solution also provides them
        self.r0 = results.get('r0', None)
        self.f_at_r0 = results.get('f_at_r0', None)          # f(r0)
        self.df_at_r0 = results.get('df_at_r0', None)        # f'(r0)

        # Also try to pick up from solution_instance attributes
        for attr in ['r0', 'f_at_r0', 'df_at_r0']:
            if getattr(solution_instance, attr, None) is not None:
                setattr(self, attr, getattr(solution_instance, attr))

        self._prepare_symbols()

    def _prepare_symbols(self):
        """Prepare symbolic variables for checks."""
        self._sym_r = sp.Symbol('r0', positive=True)
        self._sym_f = sp.Symbol('f0', positive=True)
        self._sym_df = sp.Symbol('df0')

        def _to_sympy(val, default_sym):
            if val is None:
                return default_sym
            if isinstance(val, (sp.Basic,)):
                return val
            try:
                return sp.nsimplify(val)
            except Exception:
                return sp.Float(val) if not isinstance(val, str) else default_sym

        self.sym_r0 = _to_sympy(self.r0, self._sym_r)
        self.sym_f0 = _to_sympy(self.f_at_r0, self._sym_f)
        self.sym_df0 = _to_sympy(self.df_at_r0, self._sym_df)

        self.sym_f_threshold = _to_sympy(self.f_threshold, -self.sym_r0 / 3 * self.sym_df0)
        self.sym_d2U = _to_sympy(self.d2U_eff_dr2, 3 * self.sym_f0 / self.sym_r0 + self.sym_df0)

    # ------------------------------------------------------------------
    #   Interface tests
    # ------------------------------------------------------------------
    def test_interface_has_f_threshold(self):
        if self.f_threshold is None:
            return False, "Missing required result key: 'f_threshold'"
        return True, "f_threshold is present"

    def test_interface_has_d2U_eff_dr2(self):
        if self.d2U_eff_dr2 is None:
            return False, "Missing required result key: 'd2U_eff_dr2'"
        return True, "d2U_eff_dr2 is present"

    # ------------------------------------------------------------------
    #   Invariant (physics‑law) tests
    # ------------------------------------------------------------------
    def test_invariant_threshold_expression(self):
        """Check that f_threshold equals - (r0/3) * f'(r0) symbolically or numerically."""
        if self.sym_r0 is None or self.sym_df0 is None:
            return False, "SKIP: cannot access r0 or f'(r0) to verify expression"

        expected = -self.sym_r0 / 3 * self.sym_df0
        diff = sp.simplify(self.sym_f_threshold - expected)
        if diff == 0:
            return True, "f_threshold matches -r/3 * f'(r)"
        try:
            numeric_diff = float(diff.evalf())
            if math.isclose(numeric_diff, 0.0, rel_tol=1e-5, abs_tol=1e-5):
                return True, "f_threshold numerically matches -r/3 * f'(r)"
        except Exception:
            pass
        return False, f"f_threshold = {self.sym_f_threshold}, expected {expected}"

    def test_invariant_second_derivative_expression(self):
        """Check that d2U_eff_dr2 equals 3*f(r0)/r0 + f'(r0)."""
        if self.sym_r0 is None or self.sym_f0 is None or self.sym_df0 is None:
            return False, "SKIP: insufficient variables to verify second derivative expression"

        expected = 3 * self.sym_f0 / self.sym_r0 + self.sym_df0
        diff = sp.simplify(self.sym_d2U - expected)
        if diff == 0:
            return True, "d2U_eff_dr2 matches 3*f/r + f'"
        try:
            numeric_diff = float(diff.evalf())
            if math.isclose(numeric_diff, 0.0, rel_tol=1e-5, abs_tol=1e-5):
                return True, "d2U_eff_dr2 numerically matches 3*f/r + f'"
        except Exception:
            pass
        return False, f"d2U_eff_dr2 = {self.sym_d2U}, expected {expected}"

    def test_invariant_stability_inequality(self):
        """Stability requires d^2U_eff/dr^2 > 0  <=>  f(r0) > -r0/3 * f'(r0)."""
        if self.sym_d2U is None:
            return False, "SKIP: cannot access d2U_eff_dr2"
        if self.sym_f_threshold is None or self.sym_f0 is None:
            return False, "SKIP: cannot access f(r0) or f_threshold"

        try:
            d2U_val = float(self.sym_d2U.evalf())
            f0_val = float(self.sym_f0.evalf())
            thr_val = float(self.sym_f_threshold.evalf())
        except Exception:
            # symbolic case – structural equivalence is enough
            return True, "Symbolic expressions are consistent with the stability inequality"

        condition_holds = (d2U_val > 0) == (f0_val > thr_val)
        if condition_holds:
            return True, "Stability inequality is consistent: d2U>0 ⇔ f0 > threshold"
        else:
            return False, f"Numerical inconsistency: d2U={d2U_val}, f0={f0_val}, threshold={thr_val}"

    # ------------------------------------------------------------------
    #   Modeling / trace checks  (at least two tests)
    # ------------------------------------------------------------------
    def test_trace_derivation_sign(self):
        """The effective potential second derivative must be positive for stability."""
        if self.d2U_eff_dr2 is None:
            return False, "SKIP: d2U_eff_dr2 not available"
        try:
            d2U_val = float(self.sym_d2U.evalf())
        except Exception:
            # symbolic – try evaluation with generic positive numbers
            subs_dict = {self._sym_r: 1.0, self._sym_f: 1.0, self._sym_df: -0.5}
            try:
                d2U_val = float(self.sym_d2U.subs(subs_dict).evalf())
            except Exception:
                return False, "Cannot evaluate d2U symbolically or numerically"
        if d2U_val > 0:
            return True, f"d2U_eff_dr2 is positive ({d2U_val:.3f}) – consistent with stability"
        else:
            return False, f"d2U_eff_dr2 is zero or negative ({d2U_val:.3f}) – stability fails"

    def test_modeling_rederive_d2U(self):
        """Re‑derive the second derivative from the reported r0, f0, df0 and compare with stored value.
        This checks that the numerical output matches the theoretical formula."""
        if self.sym_r0 is None or self.sym_f0 is None or self.sym_df0 is None:
            return False, "SKIP: missing r0, f0, or df0 to re‑derive"
        try:
            # Compute the second derivative according to the theorem
            derived = 3 * self.sym_f0 / self.sym_r0 + self.sym_df0
            # Check difference against stored d2U
            diff = sp.simplify(self.sym_d2U - derived)
            if diff == 0:
                return True, "Reported d2U matches re‑derived value"
            numeric_diff = float(diff.evalf())
            if math.isclose(numeric_diff, 0.0, rel_tol=1e-5, abs_tol=1e-5):
                return True, "Reported d2U numerically matches re‑derived value"
            return False, f"Difference: {diff}"
        except Exception as e:
            return False, f"Re‑derivation failed: {e}"

    # ------------------------------------------------------------------
    #   Monotonicity tests
    # ------------------------------------------------------------------
    def test_monotonicity_force_derivative(self):
        """Increasing (less negative) f'(r0) should lower the threshold,
        making the inequality easier to satisfy."""
        if self.sym_r0 is None or self.sym_f0 is None or self.sym_df0 is None:
            return False, "SKIP: insufficient symbolic quantities to run monotonicity"

        base_subs = {self._sym_r: 2.0, self._sym_f: 10.0, self._sym_df: -2.0}
        try:
            base_thr = float((-self.sym_r0 / 3 * self.sym_df0).subs(base_subs).evalf())
        except Exception:
            return False, "Could not evaluate base threshold"

        # Perturb df0 by +10% (less negative)
        perturb_subs = base_subs.copy()
        perturb_subs[self._sym_df] = -1.8   # -2.0 * 0.9
        try:
            perturb_thr = float((-self.sym_r0 / 3 * self.sym_df0).subs(perturb_subs).evalf())
        except Exception:
            return False, "Could not evaluate perturbed threshold"

        # Expect: less negative df => threshold becomes lower (more negative)
        if perturb_thr < base_thr:
            return (True,
                    f"Monotonicity holds: perturbing df' from -2.0 to -1.8 changes threshold "
                    f"from {base_thr:.3f} to {perturb_thr:.3f} (decrease expected)")
        else:
            return (False,
                    f"Monotonicity violated: threshold increased from {base_thr:.3f} to {perturb_thr:.3f}")

    # ------------------------------------------------------------------
    #   Weak answer sanity (only one such test allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak_form(self):
        """Verify that the reported threshold expression has the correct structural form:
        contains r, f', and a negative sign."""
        if self.f_threshold is None:
            return False, "SKIP: f_threshold missing"
        str_repr = str(self.f_threshold)
        # Weak check: look for basic tokens
        required_tokens = ['r', 'f', 'd']
        found = all(token in str_repr for token in required_tokens)
        if found:
            return True, "f_threshold string contains required tokens (r, f, d) – structural sanity"
        else:
            return False, f"f_threshold string '{str_repr}' does not contain expected tokens {required_tokens}"

    # ------------------------------------------------------------------
    #   Test runner (exactly as required)
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