import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Audits a solution for a 3-level molecule's statistical mechanics.
    Checks: interface completeness, Boltzmann invariants, modeling of E_avg and C_v,
    monotonicity trends via explicit input perturbation, and weak sanity of derived expressions.
    """

    def __init__(self, solution_instance):
        # ---------- extract givens safely (use None for missing) ----------
        self.epsilon = getattr(solution_instance, 'epsilon', None)
        self.k = getattr(solution_instance, 'k', None)
        self.N_A = getattr(solution_instance, 'N_A', None)
        self.R = getattr(solution_instance, 'R', None)          # may be computed from NA*k
        if self.R is None and self.k is not None and self.N_A is not None:
            self.R = self.N_A * self.k

        # ---------- extract result expressions (strings expected) ----------
        results = getattr(solution_instance, 'results', {})
        self.T_c_expr = results.get('T_c', None)
        self.E_avg_expr = results.get('E_avg', None)
        self.C_v_full_expr = results.get('C_v_full', None)
        self.C_v_high_expr = results.get('C_v_high', None)
        self.C_v_low_expr = results.get('C_v_low', None)

        # ---------- symbolic infrastructure ----------
        self.eps_s = sp.Symbol('epsilon', positive=True)
        self.k_s = sp.Symbol('k', positive=True)
        self.T_s = sp.Symbol('T', positive=True)
        self.NA_s = sp.Symbol('N_A', positive=True)
        self.R_s = sp.Symbol('R', positive=True)
        self.beta_s = sp.Symbol('beta', positive=True)

        # parse expressions into sympy objects if strings exist
        self.E_avg_sym = self._parse_sym(self.E_avg_expr)
        self.C_v_full_sym = self._parse_sym(self.C_v_full_expr)
        self.C_v_high_sym = self._parse_sym(self.C_v_high_expr)
        self.C_v_low_sym = self._parse_sym(self.C_v_low_expr)
        self.T_c_sym = self._parse_sym(self.T_c_expr)

        # capability flag: can we run numeric tests?
        self.can_evaluate = (self.epsilon is not None and self.k is not None and self.N_A is not None)

        # convenience: numeric epsilon/k
        if self.can_evaluate:
            self.eps_over_k = self.epsilon / self.k
        else:
            self.eps_over_k = None

    def _parse_sym(self, expr_str):
        """Convert string to sympy expression, returning None on failure."""
        if expr_str is None or not isinstance(expr_str, str):
            return None
        try:
            return sp.sympify(expr_str)
        except Exception:
            return None

    def _numeric(self, expr, T_val):
        """
        Evaluate a sympy expression at a given T (numeric).
        Returns None if constants are missing, evaluation fails, or expr is None.
        """
        if expr is None or not self.can_evaluate:
            return None
        subs_dict = {
            self.eps_s: self.epsilon,
            self.k_s: self.k,
            self.NA_s: self.N_A,
            self.R_s: self.R,
            self.beta_s: 1.0 / (self.k * T_val),
            self.T_s: T_val
        }
        try:
            val = expr.subs(subs_dict).evalf()
            return float(val)
        except Exception:
            return None

    # ===================== INTERFACE TESTS =====================
    def test_interface_T_c_exists(self):
        if self.T_c_expr is None:
            return (False, "Missing result key 'T_c'")
        if not isinstance(self.T_c_expr, str):
            return (False, "'T_c' is not a string")
        return (True, "T_c result present and is a string")

    def test_interface_E_avg_exists(self):
        if self.E_avg_expr is None:
            return (False, "Missing result key 'E_avg'")
        if not isinstance(self.E_avg_expr, str):
            return (False, "'E_avg' is not a string")
        return (True, "E_avg present")

    def test_interface_C_v_full_exists(self):
        if self.C_v_full_expr is None:
            return (False, "Missing result key 'C_v_full'")
        if not isinstance(self.C_v_full_expr, str):
            return (False, "'C_v_full' is not a string")
        return (True, "C_v_full present")

    def test_interface_C_v_high_exists(self):
        if self.C_v_high_expr is None:
            return (False, "Missing result key 'C_v_high'")
        if not isinstance(self.C_v_high_expr, str):
            return (False, "'C_v_high' is not a string")
        return (True, "C_v_high present")

    def test_interface_C_v_low_exists(self):
        if self.C_v_low_expr is None:
            return (False, "Missing result key 'C_v_low'")
        if not isinstance(self.C_v_low_expr, str):
            return (False, "'C_v_low' is not a string")
        return (True, "C_v_low present")

    # ===================== INVARIANT / THEOREM TESTS =====================
    def test_invariant_avg_energy_positive(self):
        """At any finite T, average energy should be positive."""
        if not self.can_evaluate:
            return (False, "SKIP: missing constants epsilon, k, N_A")
        T1 = 0.1 * self.eps_over_k
        val = self._numeric(self.E_avg_sym, T1)
        if val is None:
            return (False, "Cannot evaluate E_avg")
        if val <= 0:
            return (False, f"E_avg at T={T1:.2e} is non-positive: {val}")
        return (True, f"E_avg positive at T={T1:.2e}: {val}")

    def test_invariant_C_v_positive(self):
        """Specific heat should be positive for any T."""
        if not self.can_evaluate:
            return (False, "SKIP: missing constants")
        T1 = 0.5 * self.eps_over_k
        val = self._numeric(self.C_v_full_sym, T1)
        if val is None:
            return (False, "Cannot evaluate C_v_full")
        if val <= 0:
            return (False, f"C_v_full at T={T1:.2e} is non-positive: {val}")
        return (True, f"C_v_full positive: {val}")

    def test_invariant_low_T_E3_negligible(self):
        """At T<<epsilon/k, Boltzmann factor ratio exp(-10βε)/exp(-βε) << 1."""
        if not self.can_evaluate:
            return (False, "SKIP: missing constants")
        T_low = 0.05 * self.eps_over_k
        beta = 1.0 / (self.k * T_low)
        ratio = math.exp(-10 * beta * self.epsilon) / math.exp(-beta * self.epsilon)
        if not math.isclose(ratio, 0.0, abs_tol=1e-2):
            return (False, f"Ratio at low T is not negligible: {ratio}")
        return (True, f"Low T ratio ≈ {ratio:.3e} << 1")

    def test_invariant_high_T_C_v_tends_zero(self):
        """At very high T, C_v should approach zero as 1/T^2."""
        if not self.can_evaluate:
            return (False, "SKIP: missing constants")
        T_high = 100.0 * self.eps_over_k
        val = self._numeric(self.C_v_full_sym, T_high)
        if val is None:
            return (False, "Cannot evaluate C_v_full at high T")
        if val > 1e-3 * self.R:
            return (False, f"High T C_v not small: {val} (R={self.R})")
        return (True, f"High T C_v = {val:.3e} is small")

    # ===================== MODELING / TRACE CONSISTENCY TESTS =====================
    def test_modeling_E_avg_via_Boltzmann_sum(self):
        """
        Check that the derived E_avg expression matches the explicit sum
          ⟨E⟩ = ( ε e^{-βε} + 10ε e^{-10βε} ) / (1 + e^{-βε} + e^{-10βε} )
        at a couple of temperatures.
        """
        if not self.can_evaluate:
            return (False, "SKIP: missing constants")
        temps = [0.1 * self.eps_over_k, 0.5 * self.eps_over_k, 2.0 * self.eps_over_k]
        for T_val in temps:
            expr_val = self._numeric(self.E_avg_sym, T_val)
            if expr_val is None:
                return (False, f"Cannot evaluate E_avg at T={T_val}")
            beta = 1.0 / (self.k * T_val)
            Z = 1 + math.exp(-beta*self.epsilon) + math.exp(-10*beta*self.epsilon)
            expected = (self.epsilon * math.exp(-beta*self.epsilon) + 10*self.epsilon*math.exp(-10*beta*self.epsilon)) / Z
            if not math.isclose(expr_val, expected, rel_tol=1e-2, abs_tol=1e-12):
                return (False, f"At T={T_val}: E_avg={expr_val} vs expected={expected}")
        return (True, "E_avg matches Boltzmann sum at tested T")

    def test_modeling_C_v_from_E_derivative(self):
        """
        Check that C_v_full is consistent with N_A * d⟨E⟩/dT at a middle T.
        Use simple finite difference because symbolic derivative may differ in form.
        """
        if not self.can_evaluate:
            return (False, "SKIP: missing constants")
        T_mid = 0.42 * self.eps_over_k   # near Schottky peak
        delta = 1e-6 * self.eps_over_k
        E_plus = self._numeric(self.E_avg_sym, T_mid + delta)
        E_minus = self._numeric(self.E_avg_sym, T_mid - delta)
        if E_plus is None or E_minus is None:
            return (False, "Cannot evaluate E_avg for finite difference")
        deriv = (E_plus - E_minus) / (2 * delta)
        C_v_from_deriv = self.N_A * deriv
        C_actual = self._numeric(self.C_v_full_sym, T_mid)
        if C_actual is None:
            return (False, "Cannot evaluate C_v_full")
        if not math.isclose(C_actual, C_v_from_deriv, rel_tol=5e-2, abs_tol=1e-12):
            return (False, f"At T={T_mid}: C_v={C_actual}, N_A*dE/dT ≈ {C_v_from_deriv}")
        return (True, "C_v_full matches N_A dE_avg/dT within 5%")

    def test_modeling_low_T_approximation(self):
        """At T << epsilon/k, C_v_low should approximate C_v_full."""
        if not self.can_evaluate:
            return (False, "SKIP: missing constants")
        T_low = 0.05 * self.eps_over_k
        C_full = self._numeric(self.C_v_full_sym, T_low)
        C_low = self._numeric(self.C_v_low_sym, T_low)
        if C_full is None or C_low is None:
            return (False, "Cannot evaluate at low T")
        if not math.isclose(C_low, C_full, rel_tol=5e-1, abs_tol=1e-12):
            return (False, f"At low T: C_full={C_full}, C_low={C_low}")
        return (True, f"Low T approx ok: C_full={C_full}, C_low={C_low}")

    def test_modeling_high_T_approximation(self):
        """At T >> epsilon/k, C_v_high should approximate C_v_full."""
        if not self.can_evaluate:
            return (False, "SKIP: missing constants")
        T_high = 50.0 * self.eps_over_k
        C_full = self._numeric(self.C_v_full_sym, T_high)
        C_high = self._numeric(self.C_v_high_sym, T_high)
        if C_full is None or C_high is None:
            return (False, "Cannot evaluate at high T")
        if not math.isclose(C_high, C_full, rel_tol=5e-1, abs_tol=1e-12):
            return (False, f"At high T: C_full={C_full}, C_high={C_high}")
        return (True, f"High T approx ok: C_full={C_full}, C_high={C_high}")

    # ===================== MONOTONICITY TESTS (with explicit perturbation) =====================
    def test_monotonicity_E_avg_increases_with_T(self):
        """
        Perturb T upward by 1% from a baseline and verify that E_avg increases.
        This tests monotonic increase of average energy with temperature.
        """
        if not self.can_evaluate:
            return (False, "SKIP: missing constants")
        # baseline at a moderate temperature
        T_base = 0.5 * self.eps_over_k
        baseline_val = self._numeric(self.E_avg_sym, T_base)
        if baseline_val is None:
            return (False, "Cannot evaluate E_avg at baseline")
        # perturb T upward by 1%
        delta = 0.01 * T_base
        T_perturbed = T_base + delta
        perturbed_val = self._numeric(self.E_avg_sym, T_perturbed)
        if perturbed_val is None:
            return (False, "Cannot evaluate E_avg at perturbed T")
        if not (perturbed_val > baseline_val):
            return (False, f"E_avg did not increase: baseline {baseline_val}, perturbed {perturbed_val}")
        return (True, f"E_avg increased from {baseline_val} to {perturbed_val} when T perturbed +1%")

    def test_monotonicity_C_v_low_T_increasing(self):
        """
        At a low temperature (below the Schottky peak), perturb T upward by 1%
        and verify that C_v_full increases (monotonic increase before peak).
        """
        if not self.can_evaluate:
            return (False, "SKIP: missing constants")
        # Choose T well below the peak (peak near 0.42*eps/k)
        T_base = 0.1 * self.eps_over_k
        baseline_val = self._numeric(self.C_v_full_sym, T_base)
        if baseline_val is None:
            return (False, "Cannot evaluate C_v_full at baseline low T")
        delta = 0.01 * T_base
        T_perturbed = T_base + delta
        perturbed_val = self._numeric(self.C_v_full_sym, T_perturbed)
        if perturbed_val is None:
            return (False, "Cannot evaluate C_v_full at perturbed low T")
        if not (perturbed_val > baseline_val):
            return (False, f"C_v_full did not increase: baseline {baseline_val}, perturbed {perturbed_val}")
        return (True, f"C_v_full increased from {baseline_val} to {perturbed_val} when T perturbed +1% (low T regime)")

    def test_monotonicity_C_v_high_T_decreasing(self):
        """
        At a high temperature (above the Schottky peak), perturb T upward by 1%
        and verify that C_v_full decreases (monotonic decrease after peak).
        """
        if not self.can_evaluate:
            return (False, "SKIP: missing constants")
        # Choose T well above the peak
        T_base = 2.0 * self.eps_over_k
        baseline_val = self._numeric(self.C_v_full_sym, T_base)
        if baseline_val is None:
            return (False, "Cannot evaluate C_v_full at baseline high T")
        delta = 0.01 * T_base
        T_perturbed = T_base + delta
        perturbed_val = self._numeric(self.C_v_full_sym, T_perturbed)
        if perturbed_val is None:
            return (False, "Cannot evaluate C_v_full at perturbed high T")
        if not (perturbed_val < baseline_val):
            return (False, f"C_v_full did not decrease: baseline {baseline_val}, perturbed {perturbed_val}")
        return (True, f"C_v_full decreased from {baseline_val} to {perturbed_val} when T perturbed +1% (high T regime)")

    # ===================== WEAK ANSWER SANITY (only one allowed) =====================
    def test_answer_sanity_finite_nonnegative(self):
        """
        Very weak sanity: ensure all computed expressions can be evaluated
        to finite, non-negative values at some representative T.
        No exact value checking, no math.isclose.
        """
        if not self.can_evaluate:
            return (False, "SKIP: missing constants")
        T_test = 0.5 * self.eps_over_k
        for name, sym in [("E_avg", self.E_avg_sym),
                           ("C_v_full", self.C_v_full_sym),
                           ("C_v_low", self.C_v_low_sym),
                           ("C_v_high", self.C_v_high_sym)]:
            if sym is None:
                continue
            val = self._numeric(sym, T_test)
            if val is None:
                return (False, f"{name} gives None at T={T_test}")
            if not math.isfinite(val):
                return (False, f"{name} is non-finite: {val}")
            if val < 0:
                return (False, f"{name} is negative: {val}")
        # also check T_c if possible
        if self.T_c_sym is not None and self.can_evaluate:
            try:
                subs_tc = {self.NA_s: self.N_A, self.eps_s: self.epsilon, self.k_s: self.k}
                tc_val = self.T_c_sym.subs(subs_tc).evalf()
                tc_val = float(tc_val)
                if not math.isfinite(tc_val) or tc_val <= 0:
                    return (False, f"T_c not positive finite: {tc_val}")
            except Exception:
                return (False, "Cannot evaluate T_c")
        return (True, "All evaluated results are finite, non-NaN, non-negative")

    # ===================== RUNNER =====================
    def run_all_tests(self):
        logs = []
        all_pass = True
        # discover all test_* methods
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)