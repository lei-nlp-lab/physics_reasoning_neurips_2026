import math
import traceback

class PhysicsChecker:
    """
    Audit the quality and physical correctness of a solution to the thermodynamics problem
    involving the Clausius-Clapeyron equation and surface thermodynamic relations.
    """

    def __init__(self, solution_instance):
        # ------------------------------------------------------------
        # Extract input givens (safe defaults)
        # ------------------------------------------------------------
        self.l = getattr(solution_instance, 'l', None)
        self.R = getattr(solution_instance, 'R', None)
        self.T = getattr(solution_instance, 'T', None)
        self.p = getattr(solution_instance, 'p', None)
        self.tau = getattr(solution_instance, 'tau', None)
        self.dtau_dT = getattr(solution_instance, 'dtau_dT', None)
        self.d2tau_dT2 = getattr(solution_instance, 'd2tau_dT2', None)
        self.A = getattr(solution_instance, 'A', None)
        # Surface entropy density (may be given or derived)
        self.s = getattr(solution_instance, 's', None)

        # ------------------------------------------------------------
        # Extract result values (from a dictionary, with safe defaults)
        # ------------------------------------------------------------
        results_dict = getattr(solution_instance, 'results', {})
        self.dp_dT = results_dict.get('dp_dT', None)
        self.dlnp_dT = results_dict.get('dlnp_dT', None)
        self.u = results_dict.get('u', None)
        self.dT_dA = results_dict.get('dT_dA', None)

        # additional convenience: also look for s in results
        if self.s is None:
            self.s = results_dict.get('s', None)

    # ------------------------------------------------------------------
    # Helper: try to convert a value to float (only if it is numeric)
    # ------------------------------------------------------------------
    def _numeric_or_none(self, val):
        """Return a float if val is a number, else None."""
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        try:
            # could be sympy number? if so, evalf
            import sympy
            if isinstance(val, sympy.Basic):
                if val.is_number:
                    return float(val.evalf())
        except ImportError:
            pass
        return None

    # ------------------------------------------------------------------
    # Interface tests – check that the expected result keys exist
    # ------------------------------------------------------------------
    def test_interface_result_keys_present(self):
        """Verify that all four expected result keys are present."""
        keys = ['dp_dT', 'dlnp_dT', 'u', 'dT_dA']
        missing = [k for k in keys if getattr(self, k, None) is None]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required result keys present")

    def test_interface_result_values_not_none(self):
        """Verify that all result values are not None."""
        keys = ['dp_dT', 'dlnp_dT', 'u', 'dT_dA']
        nones = [k for k in keys if getattr(self, k, None) is None]
        if nones:
            return (False, f"Result values are None for keys: {nones}")
        return (True, "All result values are non‑None")

    # ------------------------------------------------------------------
    # Invariant / consistency tests (fuzzy numerical matching allowed)
    # ------------------------------------------------------------------
    def test_invariant_clausius_clapeyron(self):
        """
        Check that dp_dT ≈ p * dlnp_dT (i.e., the logarithmic derivative relation holds).
        Requires numeric values for dp_dT, dlnp_dT, and p.
        """
        dp = self._numeric_or_none(self.dp_dT)
        dlnp = self._numeric_or_none(self.dlnp_dT)
        p = self._numeric_or_none(self.p)
        if dp is None or dlnp is None or p is None:
            return (False, "SKIP: dp_dT, dlnp_dT, or p not numeric")
        if not math.isclose(dp, p * dlnp, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"dp_dT = {dp} not equal to p*dlnp_dT = {p*dlnp}")
        return (True, "dp_dT = p * dlnp_dT within tolerance")

    def test_invariant_u_relation(self):
        """
        Check that u ≈ tau - T * dtau_dT (the surface energy density relation).
        Requires numeric values for u, tau, T, dtau_dT.
        """
        u = self._numeric_or_none(self.u)
        tau = self._numeric_or_none(self.tau)
        T = self._numeric_or_none(self.T)
        dtau = self._numeric_or_none(self.dtau_dT)
        if u is None or tau is None or T is None or dtau is None:
            return (False, "SKIP: u, tau, T, or dtau_dT not numeric")
        expected = tau - T * dtau
        if not math.isclose(u, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"u = {u} not equal to tau - T*dtau_dT = {expected}")
        return (True, "u = tau - T*dtau_dT within tolerance")

    def test_invariant_entropy_density(self):
        """
        Check that s ≈ -dtau_dT (definition of surface entropy density).
        Requires numeric s and dtau_dT.
        """
        s = self._numeric_or_none(self.s)
        dtau = self._numeric_or_none(self.dtau_dT)
        if s is None or dtau is None:
            return (False, "SKIP: s or dtau_dT not numeric")
        if not math.isclose(s, -dtau, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"s = {s} not equal to -dtau_dT = {-dtau}")
        return (True, "s = -dtau_dT within tolerance")

    # ------------------------------------------------------------------
    # Modeling / trace checks
    # ------------------------------------------------------------------
    def test_modeling_sign_dT_dA(self):
        """
        For part (b)(ii): if dtau_dT < 0 and d2tau_dT2 > 0, then dT/dA > 0.
        We check either that a numeric dT_dA is positive, or that the
        logical conclusion from dtau_dT and d2tau_dT2 is positive.
        """
        dtau = self._numeric_or_none(self.dtau_dT)
        d2tau = self._numeric_or_none(self.d2tau_dT2)
        dTdA = self._numeric_or_none(self.dT_dA)

        # Case 1: dT_dA is numeric – directly test sign
        if dTdA is not None:
            if dTdA <= 0:
                return (False, f"dT_dA = {dTdA} should be > 0")
            return (True, "dT_dA > 0 (numeric check)")

        # Case 2: only have the sign conditions (dtau<0, d2tau>0)
        if dtau is not None and d2tau is not None:
            if dtau < 0 and d2tau > 0:
                # s = -dtau > 0, so dT/dA = s/(A*d2tau) > 0
                return (True, "Logic: dtau<0 and d2tau>0 imply dT/dA>0")
            else:
                return (False, f"Given dtau={dtau}, d2tau={d2tau}, sign condition not satisfied")
        return (False, "SKIP: neither dT_dA numeric nor dtau/d2tau given")

    def test_modeling_clausius_clapeyron_linearity(self):
        """
        Check that dp/dT is proportional to p/T^2 (i.e., dlnp/dT = l/(RT^2)).
        This is a modeling check: if l, R, T are known, we can compute expected dlnp_dT.
        """
        dlnp = self._numeric_or_none(self.dlnp_dT)
        l = self._numeric_or_none(self.l)
        R = self._numeric_or_none(self.R)
        T = self._numeric_or_none(self.T)
        if dlnp is None or l is None or R is None or T is None:
            return (False, "SKIP: insufficient numeric data for this check")
        expected = l / (R * T**2)
        if not math.isclose(dlnp, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"dlnp/dT = {dlnp}, expected {expected} (l/(RT^2))")
        return (True, "dlnp_dT matches l/(R T^2)")

    # ------------------------------------------------------------------
    # Monotonicity test (choose one input and perturb)
    # ------------------------------------------------------------------
    def test_monotonicity_dlnp_dT_with_temperature(self):
        """
        Perturb T upward and verify that dlnp/dT decreases.
        Requires numeric l, R, T, and that dlnp_dT is given as a function of T
        (we rely on the result being numeric for one T value, but we need another).
        We'll use the known formula to compute dlnp_dT at T+ε and compare.
        """
        l = self._numeric_or_none(self.l)
        R = self._numeric_or_none(self.R)
        T = self._numeric_or_none(self.T)
        dlnp = self._numeric_or_none(self.dlnp_dT)  # not strictly needed for derivative
        if l is None or R is None or T is None or T <= 0:
            return (False, "SKIP: missing l, R, or valid T")
        # compute dlnp/dT at T and T+delta
        delta = 0.01 * T if T > 0 else 0.01  # 1% perturbation
        if delta <= 0:
            delta = 0.01
        dlnp1 = l / (R * T**2)
        dlnp2 = l / (R * (T + delta)**2)
        if dlnp2 >= dlnp1:
            return (False, f"dlnp/dT does not decrease when T increases: {dlnp1} -> {dlnp2}")
        return (True, "dlnp/dT decreases with increasing T (consistent with l/(RT^2))")

    # ------------------------------------------------------------------
    # Weak answer sanity check (only one allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_finite(self):
        """
        Verify that all result values are finite (not None, not NaN, not inf).
        This is a light sanity check – no formulas re‑derived.
        """
        keys = ['dp_dT', 'dlnp_dT', 'u', 'dT_dA']
        for key in keys:
            val = getattr(self, key, None)
            if val is None:
                return (False, f"Result '{key}' is None")
            try:
                fval = float(val) if isinstance(val, (int,float)) else None
            except:
                # if it's a symbolic expression, we cannot evaluate—pass
                continue
            if fval is not None:
                if math.isnan(fval) or math.isinf(fval):
                    return (False, f"Result '{key}' is NaN or Inf")
        return (True, "All result values are finite")

    # ------------------------------------------------------------------
    # Run all tests (discovery + execution)
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