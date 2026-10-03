import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for the satellite-towing problem.
    Extracts key results and provides unit-test-like methods.
    """

    def __init__(self, solution_instance):
        # Extract results (expected keys: F, T_x, t_catch)
        results = getattr(solution_instance, 'results', {})
        self.F = results.get('F', None)
        self.T_x = results.get('T_x', None)
        self.t_catch = results.get('t_catch', None)

        # Attempt to extract givens (optional, for deeper invariant checks)
        self.m_s = getattr(solution_instance, 'm_s', None) or results.get('m_s', None)
        self.L = getattr(solution_instance, 'L', None) or results.get('L', None)
        self.lambda_ = getattr(solution_instance, 'lambda', None) or results.get('lambda', None)
        self.a1 = getattr(solution_instance, 'a1', None) or results.get('a1', None)
        self.a3 = getattr(solution_instance, 'a3', None) or results.get('a3', None)

        # Precompute rope mass if possible
        self.m_r_calced = None
        if self.lambda_ is not None and self.L is not None:
            self.m_r_calced = self.lambda_ * self.L

        # Parse T_x expression if it's a string
        self.T_expr = None
        self.T_func = None
        if isinstance(self.T_x, str):
            try:
                x = sp.Symbol('x')
                self.T_expr = sp.sympify(self.T_x)
                self.T_func = sp.lambdify(x, self.T_expr, 'numpy')
            except Exception:
                self.T_expr = None
                self.T_func = None
        elif self.T_x is not None:
            try:
                x = sp.Symbol('x')
                self.T_expr = sp.sympify(self.T_x)
                self.T_func = sp.lambdify(x, self.T_expr, 'numpy')
            except Exception:
                self.T_expr = None
                self.T_func = None

    def _eval_T(self, x_val):
        """Evaluate T at a given x (numeric). Returns None if unavailable."""
        if self.T_func is not None:
            try:
                return float(self.T_func(x_val))
            except Exception:
                return None
        return None

    # ------------------------------------------------------------------
    # Interface tests: required keys and types
    # ------------------------------------------------------------------
    def test_interface_F_present(self):
        if self.F is None:
            return (False, "F (force on rope) is missing in results.")
        return (True, "F is present.")

    def test_interface_T_x_present(self):
        if self.T_x is None:
            return (False, "T_x (tension expression) is missing in results.")
        return (True, "T_x is present.")

    def test_interface_t_catch_present(self):
        if self.t_catch is None:
            return (False, "t_catch (catch time) is missing in results.")
        return (True, "t_catch is present.")

    def test_interface_T_x_type(self):
        if self.T_x is not None and not isinstance(self.T_x, str):
            return (False, f"T_x should be a string, got {type(self.T_x).__name__}.")
        if self.T_x is None:
            return (False, "SKIP: T_x is missing.")
        return (True, "T_x is a string.")

    def test_interface_t_catch_numeric(self):
        if self.t_catch is not None:
            try:
                float(self.t_catch)
            except (ValueError, TypeError):
                return (False, f"t_catch is not numeric: {self.t_catch}.")
        else:
            return (False, "SKIP: t_catch is missing.")
        return (True, "t_catch is numeric.")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based, using math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_F_total(self):
        """F = (m_s + m_r) * a1 if givens present."""
        if any(v is None for v in [self.F, self.m_s, self.m_r_calced, self.a1]):
            return (False, "SKIP: missing F/m_s/m_r/a1.")
        expected_F = (self.m_s + self.m_r_calced) * self.a1
        if math.isclose(self.F, expected_F, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"F ≈ {expected_F} N (calculated).")
        return (False, f"F = {self.F}, expected ≈ {expected_F} N.")

    def test_invariant_tension_at_zero_equals_F(self):
        """T(0) should equal F."""
        if self.F is None or self.T_func is None:
            return (False, "SKIP: missing F or T expression.")
        T0 = self._eval_T(0)
        if T0 is None:
            return (False, "SKIP: could not evaluate T(0).")
        if math.isclose(T0, self.F, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"T(0) ≈ F = {self.F} N.")
        return (False, f"T(0) = {T0}, F = {self.F}.")

    def test_invariant_tension_at_L(self):
        """T(L) should equal m_s * a1 (only satellite mass)."""
        if any(v is None for v in [self.T_func, self.m_s, self.a1, self.L]):
            return (False, "SKIP: missing T_func/m_s/a1/L.")
        TL = self._eval_T(self.L)
        expected = self.m_s * self.a1
        if TL is None:
            return (False, "SKIP: could not evaluate T(L).")
        if math.isclose(TL, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"T(L) ≈ {expected} N.")
        return (False, f"T(L) = {TL}, expected ≈ {expected} N.")

    def test_invariant_tension_linearity(self):
        """Slope dT/dx should be -lambda * a1."""
        if any(v is None for v in [self.T_func, self.lambda_, self.a1]):
            return (False, "SKIP: missing T_func/lambda/a1.")
        x1, x2 = 10.0, 20.0
        T1 = self._eval_T(x1)
        T2 = self._eval_T(x2)
        if T1 is None or T2 is None:
            return (False, "SKIP: could not evaluate T at two points.")
        slope = (T2 - T1) / (x2 - x1)
        expected_slope = -self.lambda_ * self.a1
        if math.isclose(slope, expected_slope, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Slope ≈ {expected_slope} N/m.")
        return (False, f"Slope = {slope}, expected ≈ {expected_slope}.")

    # ------------------------------------------------------------------
    # Modeling / trace tests (theorem-based checks on the expression)
    # ------------------------------------------------------------------
    def test_modeling_tension_structural(self):
        """
        Check that T(x) satisfies the structure:
          T(x) = (m_s + lambda*(L - x)) * a1
        by verifying:
         - T(0) = (m_s + lambda*L)*a1
         - T(L) = m_s*a1
         - slope = -lambda*a1
        This avoids hardcoding a numeric target.
        """
        if any(v is None for v in [self.T_func, self.m_s, self.lambda_, self.L, self.a1]):
            return (False, "SKIP: missing T_func or givens (m_s, lambda, L, a1).")

        T0 = self._eval_T(0)
        expected_T0 = (self.m_s + self.lambda_ * self.L) * self.a1
        if T0 is None or not math.isclose(T0, expected_T0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"T(0) = {T0}, expected {expected_T0}.")

        TL = self._eval_T(self.L)
        expected_TL = self.m_s * self.a1
        if TL is None or not math.isclose(TL, expected_TL, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"T(L) = {TL}, expected {expected_TL}.")

        x1, x2 = 10.0, 20.0
        T1 = self._eval_T(x1)
        T2 = self._eval_T(x2)
        if T1 is None or T2 is None:
            return (False, "SKIP: could not evaluate slope.")
        slope = (T2 - T1) / (x2 - x1)
        expected_slope = -self.lambda_ * self.a1
        if not math.isclose(slope, expected_slope, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Slope = {slope}, expected {expected_slope}.")

        return (True, "T(x) structurally matches the physical model.")

    def test_modeling_t_catch_derivation(self):
        """Check that t_catch is consistent with relative acceleration and L."""
        if any(v is None for v in [self.t_catch, self.L, self.a3]):
            return (False, "SKIP: missing t_catch/L/a3.")
        a_rel = 0 - self.a3   # a_sat - a_ship
        expected_t = math.sqrt(2 * self.L / a_rel)
        if math.isclose(self.t_catch, expected_t, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"t_catch ≈ {expected_t} s from sqrt(2L / a_rel).")
        return (False, f"t_catch = {self.t_catch}, expected ≈ {expected_t} s.")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit input perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_tension_decreases_with_x(self):
        """
        Perturb x (position along rope) by +1 and verify T decreases.
        This tests the monotonic decrease of tension with x.
        """
        if self.T_func is None:
            return (False, "SKIP: T_func not available.")
        x_base = 15.0
        T_base = self._eval_T(x_base)
        if T_base is None:
            return (False, "SKIP: could not evaluate T at baseline x.")
        x_pert = x_base + 1.0
        T_pert = self._eval_T(x_pert)
        if T_pert is None:
            return (False, "SKIP: could not evaluate T at perturbed x.")
        if T_pert < T_base:
            return (True, f"T decreases from {T_base} N at x={x_base} to {T_pert} N at x={x_pert}.")
        return (False, f"T does not decrease: baseline {T_base}, perturbed {T_pert}.")

    def test_monotonicity_force_increases_with_satellite_mass(self):
        """
        Perturb satellite mass m_s by +10% and verify that F (force on rope)
        increases. We compute the expected F under the perturbed mass using
        the physical relation F = (m_s + m_r)*a1.
        """
        if any(v is None for v in [self.F, self.m_s, self.m_r_calced, self.a1]):
            return (False, "SKIP: missing F/m_s/m_r/a1.")
        # Baseline (should be consistent with self.F)
        F_baseline = (self.m_s + self.m_r_calced) * self.a1
        if not math.isclose(self.F, F_baseline, rel_tol=1e-2, abs_tol=1e-2):
            # Fallback: use actual F as baseline
            F_baseline = self.F
        # Perturb m_s by +10%
        m_s_pert = self.m_s * 1.1
        F_pert = (m_s_pert + self.m_r_calced) * self.a1
        if F_pert > F_baseline:
            return (True, f"F increases from {F_baseline} N to {F_pert} N when m_s increased by 10%.")
        return (False, f"F does not increase: baseline {F_baseline}, perturbed {F_pert}.")

    # ------------------------------------------------------------------
    # Weak answer sanity test (at most one)
    # ------------------------------------------------------------------
    def test_answer_sanity_basic(self):
        """Weak checks: F positive, t_catch positive and reasonable."""
        if self.F is not None and self.F <= 0:
            return (False, f"F is not positive: {self.F}.")
        if self.t_catch is not None:
            try:
                tc = float(self.t_catch)
                if tc <= 0:
                    return (False, f"t_catch is not positive: {tc}.")
                if tc > 100:
                    return (False, f"t_catch unreasonably large: {tc} > 100 s.")
            except (ValueError, TypeError):
                return (False, f"t_catch not numeric: {self.t_catch}.")
        if self.F is None and self.t_catch is None:
            return (False, "SKIP: no answer data available.")
        return (True, "Basic sanity checks passed (F>0, t_catch>0, t_catch<100).")

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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)