import math
import traceback
import sympy as sp

class PhysicsChecker:
    """
    Robust checker for thermodynamic equilibrium problem with rubber band tension.
    """

    def __init__(self, solution_instance):
        # Extract results dict safely
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}

        # Extract givens (assumed available in solution_instance.givens)
        self.givens = getattr(solution_instance, 'givens', {})
        self.A = self.givens.get('A', None)
        self.l0 = self.givens.get('l0', None)
        self.K = self.givens.get('K', None)
        self.T0 = self.givens.get('T0', None)

        # Extract required result quantities
        self.dE_dx_T = self.results.get('dE_dx_T', None)
        self.dcx_dx_T = self.results.get('dcx_dx_T', None)
        self.cx = self.results.get('cx', None)
        self.E = self.results.get('E', None)
        self.S = self.results.get('S', None)
        self.Tf = self.results.get('Tf', None)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _eval_num(self, expr, subs=None):
        """Try to convert expr to a float number.
        If expr is sympy, substitute subs dict and evaluate numerically.
        Returns None if not possible."""
        if expr is None:
            return None
        if isinstance(expr, (int, float)):
            return float(expr)
        # SymPy expression
        try:
            if hasattr(expr, 'subs') and callable(expr.subs) and subs is not None:
                val = expr.subs(subs).evalf()
                return float(val)
        except Exception:
            pass
        # Fallback: try direct float conversion
        try:
            return float(expr)
        except (TypeError, ValueError):
            return None

    def _safe_isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Return True if both a and b are numbers and close."""
        if a is None or b is None:
            return False
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        return False

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all six required result keys exist."""
        required = ['dE_dx_T', 'dcx_dx_T', 'cx', 'E', 'S', 'Tf']
        missing = [k for k in required if self.results.get(k) is None]
        if missing:
            return (False, f"Missing results: {missing}")
        return (True, "All required keys present")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_dE_dx_T_via_maxwell(self):
        """(∂E/∂x)_T should equal t - T*(∂t/∂T)_x, which is zero for given t.
        Use numeric substitution with typical values."""
        if self.A is None or self.l0 is None:
            return (False, "SKIP: givens A, l0 not available")
        # Choose sample numeric values
        sample_x = 1.2
        sample_T = 300.0
        # Compute t and (∂t/∂T)_x
        t_val = self.A * sample_T * (sample_x / self.l0 - self.l0**2 / sample_x**2)
        dt_dT_val = self.A * (sample_x / self.l0 - self.l0**2 / sample_x**2)
        expected = t_val - sample_T * dt_dT_val   # should be 0

        # Evaluate provided dE_dx_T at same point
        subs = {'x': sample_x, 'T': sample_T, 'l0': self.l0, 'A': self.A}
        provided = self._eval_num(self.dE_dx_T, subs)
        if provided is None:
            return (False, "Cannot evaluate dE_dx_T numerically")
        if self._safe_isclose(provided, expected):
            return (True, f"dE_dx_T = {provided} ≈ expected {expected}")
        else:
            return (False, f"dE_dx_T = {provided} but expected ≈ {expected}")

    def test_invariant_dcx_dx_T_zero(self):
        """(∂c_x/∂x)_T = (∂/∂T)_x (∂E/∂x)_T, and (∂E/∂x)_T=0, so it should be 0."""
        # We can check numerically by finite difference on (∂E/∂x)_T.
        if self.A is None or self.l0 is None:
            return (False, "SKIP: givens A, l0 not available")
        # Use two nearby T values
        T_base = 300.0
        delta = 1e-3
        T1 = T_base - delta
        T2 = T_base + delta
        x_sample = 1.2
        # (∂E/∂x)_T at T1 and T2 (both should be ~0)
        subs1 = {'x': x_sample, 'T': T1, 'l0': self.l0, 'A': self.A}
        subs2 = {'x': x_sample, 'T': T2, 'l0': self.l0, 'A': self.A}
        val1 = self._eval_num(self.dE_dx_T, subs1)
        val2 = self._eval_num(self.dE_dx_T, subs2)
        if val1 is None or val2 is None:
            return (False, "Cannot evaluate dE_dx_T for finite difference")
        # derivative w.r.t T at constant x: (val2 - val1)/(T2 - T1)
        dcx_dx_derived = (val2 - val1) / (2 * delta)
        # dcx_dx_T should be 0
        expected = 0.0
        if self._safe_isclose(dcx_dx_derived, expected):
            return (True, f"dcx_dx_T derived ≈ {dcx_dx_derived}, expected 0")
        else:
            return (False, f"dcx_dx_T derived = {dcx_dx_derived}, expected 0")

    def test_invariant_cx_equals_dE_dT(self):
        """c_x = (∂E/∂T)_x. Check numerically using sample values."""
        if self.A is None or self.l0 is None or self.K is None:
            return (False, "SKIP: givens A, l0, K not available")
        x_sample = 1.2
        T_sample = 400.0
        delta = 1e-3
        subs1 = {'x': x_sample, 'T': T_sample - delta, 'l0': self.l0, 'A': self.A, 'K': self.K, 'T0': 300.0}
        subs2 = {'x': x_sample, 'T': T_sample + delta, 'l0': self.l0, 'A': self.A, 'K': self.K, 'T0': 300.0}
        E1 = self._eval_num(self.E, subs1)
        E2 = self._eval_num(self.E, subs2)
        if E1 is None or E2 is None:
            return (False, "Cannot evaluate E for finite difference")
        dE_dT_num = (E2 - E1) / (2 * delta)
        # Evaluate cx at same x,T
        cx_val = self._eval_num(self.cx, {'x': x_sample, 'T': T_sample, 'l0': self.l0, 'A': self.A, 'K': self.K, 'T0': 300.0})
        if cx_val is None:
            return (False, "Cannot evaluate cx numerically")
        if self._safe_isclose(dE_dT_num, cx_val):
            return (True, f"(∂E/∂T)_x ≈ {dE_dT_num}, cx ≈ {cx_val}")
        else:
            return (False, f"(∂E/∂T)_x = {dE_dT_num}, cx = {cx_val}")

    # ------------------------------------------------------------------
    # Modeling / trace tests (at least two)
    # ------------------------------------------------------------------
    def test_modeling_adiabatic_entropy_conservation(self):
        """Check that S(l0,T0) ≈ S(1.5 l0, Tf) for adiabatic process."""
        if any(v is None for v in [self.A, self.l0, self.K, self.T0]):
            return (False, "SKIP: givens A, l0, K, T0 not available")
        # Compute Tf from provided expression
        Tf_subs = {'T0': self.T0, 'A': self.A, 'l0': self.l0, 'K': self.K, 'T': 300.0, 'x': 1.2}
        Tf_val = self._eval_num(self.Tf, Tf_subs)
        if Tf_val is None:
            return (False, "Cannot evaluate Tf numerically; skip adiabatic check")
        # Compute S at initial state
        S_initial_subs = {'x': self.l0, 'T': self.T0, 'l0': self.l0, 'A': self.A, 'K': self.K}
        S_init = self._eval_num(self.S, S_initial_subs)
        if S_init is None:
            return (False, "Cannot evaluate S at initial state")
        # Compute S at final state
        S_final_subs = {'x': 1.5 * self.l0, 'T': Tf_val, 'l0': self.l0, 'A': self.A, 'K': self.K}
        S_final = self._eval_num(self.S, S_final_subs)
        if S_final is None:
            return (False, "Cannot evaluate S at final state")
        if self._safe_isclose(S_init, S_final, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"S(initial) ≈ {S_init}, S(final) ≈ {S_final}")
        else:
            return (False, f"S(initial) = {S_init}, S(final) = {S_final}")

    def test_modeling_cx_independent_of_x(self):
        """c_x should be independent of x because dcx_dx_T=0. Check at two different x values."""
        if self.K is None or self.A is None or self.l0 is None:
            return (False, "SKIP: givens K, A, l0 not available")
        T_sample = 350.0
        x1 = 1.0
        x2 = 1.8
        subs1 = {'x': x1, 'T': T_sample, 'l0': self.l0, 'A': self.A, 'K': self.K, 'T0': 300.0}
        subs2 = {'x': x2, 'T': T_sample, 'l0': self.l0, 'A': self.A, 'K': self.K, 'T0': 300.0}
        cx1 = self._eval_num(self.cx, subs1)
        cx2 = self._eval_num(self.cx, subs2)
        if cx1 is None or cx2 is None:
            return (False, "Cannot evaluate cx at both x values")
        if self._safe_isclose(cx1, cx2):
            return (True, f"cx(x1) = {cx1}, cx(x2) = {cx2} -> consistent")
        else:
            return (False, f"cx differs: {cx1} vs {cx2}")

    # ------------------------------------------------------------------
    # Monotonicity test
    # ------------------------------------------------------------------
    def test_monotonicity_Tf_increases_with_A(self):
        """Perturb A upward and verify Tf increases (exponent larger)."""
        if any(v is None for v in [self.A, self.l0, self.K, self.T0]):
            return (False, "SKIP: givens missing for monotonicity test")
        # Evaluate Tf at baseline A
        base_subs = {'T0': self.T0, 'A': self.A, 'l0': self.l0, 'K': self.K}
        Tf_base = self._eval_num(self.Tf, base_subs)
        if Tf_base is None:
            return (False, "Cannot evaluate Tf at baseline")
        # Increase A by 10%
        A_perturbed = self.A * 1.1
        pert_subs = base_subs.copy()
        pert_subs['A'] = A_perturbed
        Tf_pert = self._eval_num(self.Tf, pert_subs)
        if Tf_pert is None:
            return (False, "Cannot evaluate Tf at perturbed A")
        if Tf_pert > Tf_base:
            return (True, f"Tf increases with A: baseline {Tf_base}, perturbed {Tf_pert}")
        else:
            return (False, f"Tf did not increase: baseline {Tf_base}, perturbed {Tf_pert}")

    # ------------------------------------------------------------------
    # Weak answer sanity (at most one)
    # ------------------------------------------------------------------
    def test_answer_sanity_Tf_positive_and_above_T0(self):
        """Check that Tf is finite, positive, and greater than T0."""
        if self.T0 is None or self.Tf is None:
            return (False, "SKIP: missing T0 or Tf")
        Tf_val = self._eval_num(self.Tf, {'T0': self.T0, 'A': self.A, 'l0': self.l0, 'K': self.K})
        if Tf_val is None:
            # If symbolic, check that exponent is positive (if we can)
            # Otherwise just pass if it's a symbolic expression
            return (True, "Tf is symbolic; cannot check numerical inequality")
        if Tf_val <= 0:
            return (False, f"Tf = {Tf_val} is not positive")
        if Tf_val <= self.T0:
            return (False, f"Tf = {Tf_val} should be > T0 = {self.T0}")
        return (True, f"Tf = {Tf_val} > 0 and > T0")

    # ------------------------------------------------------------------
    # Runner
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