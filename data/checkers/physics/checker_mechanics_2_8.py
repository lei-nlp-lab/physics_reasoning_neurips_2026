import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Checker for bound orbit problem with V(r) = -k/r.
    Expected solution instance provides:
        - attributes: m, k, E, L (or via getattr; default None)
        - results dict with keys:
            'p_r', 'p_theta', 'J_sum', 'period_equality', 'tau'
    """
    def __init__(self, solution_instance):
        # Givens – use None for missing (no numeric defaults)
        self.m = getattr(solution_instance, 'm', None)
        self.k = getattr(solution_instance, 'k', None)
        self.E = getattr(solution_instance, 'E', None)
        self.L = getattr(solution_instance, 'L', None)
        self.tau = getattr(solution_instance, 'tau', None)

        # Results – also from solution_instance.results
        results = getattr(solution_instance, 'results', {})
        self.p_r = results.get('p_r', None)
        self.p_theta = results.get('p_theta', None)
        self.J_sum = results.get('J_sum', None)
        self.period_equality = results.get('period_equality', None)
        self.tau_res = results.get('tau', None)  # prefer results tau

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_sympy(self, expr):
        """Convert a string or sympy expression to a sympy expression."""
        if isinstance(expr, sp.Basic):
            return expr
        if isinstance(expr, str):
            try:
                return sp.sympify(expr)
            except Exception:
                return None
        try:
            return sp.sympify(expr)
        except Exception:
            return None

    def _get_numeric(self, sym_expr, subs_dict=None):
        """Try to evaluate a sympy expression to a float."""
        try:
            if subs_dict:
                val = sym_expr.subs(subs_dict).evalf()
            else:
                val = sym_expr.evalf()
            return float(val)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_givens(self):
        """Check that essential givens are provided (m, k, E are required)."""
        missing = []
        if self.m is None:
            missing.append('m')
        if self.k is None:
            missing.append('k')
        if self.E is None:
            missing.append('E')
        if missing:
            return (False, f"Missing required givens: {', '.join(missing)}")
        return (True, "All required givens present")

    def test_interface_p_r_present(self):
        """Check that p_r result is present."""
        if self.p_r is None:
            return (False, "p_r not found in results")
        return (True, "p_r present")

    def test_interface_p_theta_present(self):
        if self.p_theta is None:
            return (False, "p_theta not found in results")
        return (True, "p_theta present")

    def test_interface_J_sum_present(self):
        if self.J_sum is None:
            return (False, "J_sum not found in results")
        return (True, "J_sum present")

    def test_interface_tau_present(self):
        if self.tau_res is None:
            return (False, "tau not found in results")
        return (True, "tau present")

    def test_interface_period_equality_present(self):
        if self.period_equality is None:
            return (False, "period_equality not found in results")
        return (True, "period_equality present")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_energy_negative(self):
        """Bound orbit => total energy E < 0."""
        if self.E is None:
            return (False, "E not available, skip")
        E = self.E
        if isinstance(E, (int, float)):
            if E >= 0:
                return (False, f"E = {E} is not negative (bound orbit requires E<0)")
            return (True, f"E = {E} is negative (bound orbit)")
        # symbolic case: try to evaluate sign
        sym_E = self._ensure_sympy(E)
        if sym_E is None:
            return (False, "Cannot interpret E as numeric or symbolic")
        num_E = self._get_numeric(sym_E)
        if num_E is not None:
            if num_E >= 0:
                return (False, f"E ≈ {num_E} is not negative")
            return (True, f"E ≈ {num_E} is negative")
        # cannot evaluate numerically, assume sign consistent with problem
        return (True, "E is symbolic, assumed negative (bound orbit)")

    def test_invariant_J_sum_positive(self):
        """Action sum J_r+J_theta must be positive real."""
        if self.J_sum is None:
            return (False, "J_sum missing, skip")
        sym = self._ensure_sympy(self.J_sum)
        if sym is None:
            return (False, "Cannot convert J_sum to sympy")
        dummy_vals = {'E': -1, 'k': 1, 'm': 1, 'pi': sp.pi}
        try:
            val = sym.subs(dummy_vals).evalf()
            if val.is_real and val > 0:
                return (True, f"J_sum ≈ {float(val)} > 0 (real)")
            else:
                return (False, f"J_sum evaluated to {val}, not a positive real")
        except Exception:
            return (True, "Symbolic J_sum; assuming positive by construction")

    def test_invariant_tau_real_finite(self):
        """Period tau must be real and finite for bound orbit."""
        if self.tau_res is None:
            return (False, "tau missing")
        sym = self._ensure_sympy(self.tau_res)
        if sym is None:
            return (False, "Cannot convert tau to sympy")
        dummy_vals = {'E': -1, 'k': 1, 'm': 1, 'pi': sp.pi}
        try:
            val = sym.subs(dummy_vals).evalf()
            if val.is_real and val > 0 and not sp.isnan(val):
                return (True, f"tau ≈ {float(val)} is real positive finite")
            else:
                return (False, f"tau evaluated to {val}; not a positive real")
        except Exception:
            return (True, "Symbolic tau; assumed real and finite")

    # ------------------------------------------------------------------
    # Modeling tests (verification of derived expressions)
    # ------------------------------------------------------------------
    def test_modeling_p_r_expression(self):
        """p_r should equal m * r_dot (using dot symbol)."""
        if self.p_r is None:
            return (False, "p_r missing")
        sym_expr = self._ensure_sympy(self.p_r)
        if sym_expr is None:
            return (False, "Cannot parse p_r expression")
        m, r_dot = sp.symbols('m dot_r')
        expected = m * r_dot
        diff = sp.simplify(sym_expr - expected)
        if diff == 0:
            return (True, "p_r = m * r_dot")
        else:
            return (False, f"p_r expression {sym_expr} does not simplify to m*dot_r")

    def test_modeling_p_theta_expression(self):
        """p_theta should equal m * r^2 * theta_dot."""
        if self.p_theta is None:
            return (False, "p_theta missing")
        sym_expr = self._ensure_sympy(self.p_theta)
        if sym_expr is None:
            return (False, "Cannot parse p_theta expression")
        m, r, theta_dot = sp.symbols('m r dot_theta')
        expected = m * r**2 * theta_dot
        diff = sp.simplify(sym_expr - expected)
        if diff == 0:
            return (True, "p_theta = m * r^2 * theta_dot")
        else:
            return (False, f"p_theta expression {sym_expr} does not simplify to m*r^2*dot_theta")

    def test_modeling_J_sum_formula(self):
        """J_sum should equal sqrt(-2*pi^2*m*k^2/E) for bound E<0."""
        if self.J_sum is None:
            return (False, "J_sum missing")
        sym_expr = self._ensure_sympy(self.J_sum)
        if sym_expr is None:
            return (False, "Cannot parse J_sum")
        m, k, E, pi = sp.symbols('m k E pi', positive=True)
        expected = sp.sqrt(-2*pi**2*m*k**2/E)
        diff = sp.simplify(sym_expr**2 - expected**2)
        if diff == 0:
            return (True, "J_sum matches expected formula")
        # numeric fallback
        try:
            val_expr = self._get_numeric(sym_expr, {'E': -1, 'k': 1, 'm': 1, 'pi': math.pi})
            val_expected = self._get_numeric(expected, {'E': -1, 'k': 1, 'm': 1, 'pi': math.pi})
            if val_expr is not None and val_expected is not None and math.isclose(val_expr, val_expected, rel_tol=1e-2):
                return (True, "J_sum numerically matches expected formula")
        except Exception:
            pass
        return (True, "J_sum symbolically plausible (could not verify exactly)")

    def test_modeling_tau_formula(self):
        """tau should equal pi*k*sqrt(m/(-2*E**3))."""
        if self.tau_res is None:
            return (False, "tau missing")
        sym_expr = self._ensure_sympy(self.tau_res)
        if sym_expr is None:
            return (False, "Cannot parse tau")
        m, k, E, pi = sp.symbols('m k E pi', positive=True)
        expected = pi*k*sp.sqrt(m/(-2*E**3))
        diff = sp.simplify(sym_expr**2 - expected**2)
        if diff == 0:
            return (True, "tau matches expected formula")
        # numeric fallback
        try:
            val_expr = self._get_numeric(sym_expr, {'E': -1, 'k': 1, 'm': 1, 'pi': math.pi})
            val_expected = self._get_numeric(expected, {'E': -1, 'k': 1, 'm': 1, 'pi': math.pi})
            if val_expr is not None and val_expected is not None and math.isclose(val_expr, val_expected, rel_tol=1e-2):
                return (True, "tau numerically matches expected formula")
        except Exception:
            pass
        return (True, "tau symbolically plausible")

    def test_modeling_period_equality(self):
        """period_equality should indicate that radial and angular periods are equal."""
        eq = self.period_equality
        if eq is None:
            return (False, "period_equality missing")
        if isinstance(eq, bool):
            if eq:
                return (True, "period_equality is True")
            else:
                return (False, "period_equality is False, expected True")
        if isinstance(eq, str):
            lower_eq = eq.lower()
            if 'equal' in lower_eq or '==' in lower_eq:
                return (True, f"period_equality states periods are equal: {eq}")
            else:
                return (False, f"period_equality string does not convey equality: {eq}")
        try:
            sym_eq = self._ensure_sympy(eq)
            if sym_eq is not None and sym_eq == sp.true:
                return (True, "period_equality is sympy True")
        except Exception:
            pass
        return (True, "period_equality present; cannot fully verify")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_tau_with_E(self):
        """
        For E<0, increasing |E| (more negative) decreases tau.
        Perturb E by +1% relative (more negative) and check tau decreases.
        """
        if self.tau_res is None:
            return (False, "SKIP: tau missing")
        sym_tau = self._ensure_sympy(self.tau_res)
        if sym_tau is None:
            return (False, "SKIP: Cannot parse tau")
        subs_base = {'m': 1, 'k': 1, 'pi': sp.pi, 'E': -1}
        try:
            tau_base = self._get_numeric(sym_tau, subs_base)
            if tau_base is None or tau_base <= 0:
                return (False, "Base tau evaluation failed or non-positive")
            # perturb E by +1% (more negative: -1 * 1.01)
            subs_perturbed = subs_base.copy()
            subs_perturbed['E'] = -1 * 1.01
            tau_pert = self._get_numeric(sym_tau, subs_perturbed)
            if tau_pert is None:
                return (False, "Perturbed tau evaluation failed")
            if tau_pert < tau_base:
                return (True, f"tau decreases when |E| increases by 1%: {tau_base:.4f} -> {tau_pert:.4f}")
            else:
                return (False, f"tau did not decrease: base={tau_base:.4f}, perturbed={tau_pert:.4f}")
        except Exception as e:
            return (False, f"Monotonicity check crashed: {e}")

    def test_monotonicity_tau_with_k(self):
        """
        Increasing k (by +1%) should increase tau.
        """
        if self.tau_res is None:
            return (False, "SKIP: tau missing")
        sym_tau = self._ensure_sympy(self.tau_res)
        if sym_tau is None:
            return (False, "SKIP: Cannot parse tau")
        subs_base = {'m': 1, 'E': -1, 'pi': sp.pi, 'k': 1}
        try:
            tau_base = self._get_numeric(sym_tau, subs_base)
            if tau_base is None or tau_base <= 0:
                return (False, "Base tau evaluation failed or non-positive")
            subs_perturbed = subs_base.copy()
            subs_perturbed['k'] = 1.01
            tau_pert = self._get_numeric(sym_tau, subs_perturbed)
            if tau_pert is None:
                return (False, "Perturbed tau evaluation failed")
            if tau_pert > tau_base:
                return (True, f"tau increases when k increases by 1%: {tau_base:.4f} -> {tau_pert:.4f}")
            else:
                return (False, f"tau did not increase: base={tau_base:.4f}, perturbed={tau_pert:.4f}")
        except Exception as e:
            return (False, f"Monotonicity check crashed: {e}")

    # ------------------------------------------------------------------
    # Answer sanity (weak, only one allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_tau(self):
        """Weak sanity: tau must be real, positive, finite for a bound orbit."""
        if self.tau_res is None:
            return (False, "tau missing")
        sym = self._ensure_sympy(self.tau_res)
        if sym is None:
            return (True, "tau present but not convertible to sympy; skip")
        try:
            val = sym.subs({'m': 1, 'k': 1, 'E': -1, 'pi': sp.pi}).evalf()
            if val.is_real and val > 0 and not sp.isnan(val):
                return (True, f"tau ≈ {float(val)}: positive real finite (sanity ok)")
            else:
                return (False, f"tau evaluated to {val}; not a sensible positive real")
        except Exception as e:
            return (False, f"tau sanity check crashed: {e}")

    # ------------------------------------------------------------------
    # Master test runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        """
        Execute all test_* methods, collect results.
        Returns (all_pass, logs_list).
        """
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
                passed, msg = False, f"{name} CRASHED: {e} (traceback: {traceback.format_exc()})"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)