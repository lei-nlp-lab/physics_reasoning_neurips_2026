import sympy as sp
import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Use getattr with safe defaults – no numeric fallbacks
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.mu_operator = self.results.get('mu_operator', None)
        self.sum_mu_mirror = self.results.get('sum_mu_mirror', None)

        # Constants from problem statement (these are givens, not derived within the checker)
        self.mu_p_val = 2.793
        self.mu_n_val = -1.913
        self.coeff = self.mu_p_val + self.mu_n_val - 0.5   # 0.38

        # Symbolic variables (should match the output contract)
        self.J_sym = sp.Symbol('J')
        self.sigma_sym = sp.Symbol('langle_sum_sigma_z')

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------
    def _parse_expr(self, expr):
        """Try to convert a string or sympy expression into a sympy expression."""
        if expr is None:
            return None
        if isinstance(expr, sp.Expr):
            return expr
        if isinstance(expr, str):
            try:
                return sp.sympify(expr)
            except Exception:
                return None
        return None

    def _safe_eval(self, expr, subs_dict):
        """Substitute numeric values and evaluate. Returns float or None."""
        if expr is None:
            return None
        try:
            val = expr.subs(subs_dict).evalf()
            return float(val)
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Main test runner
    # -------------------------------------------------------------------------
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
                import traceback
                passed = False
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

    # -------------------------------------------------------------------------
    # 1. Interface tests – ensure required keys exist and have valid types
    # -------------------------------------------------------------------------
    def test_interface_mu_operator_exists(self):
        if self.mu_operator is None:
            return (False, "SKIP: mu_operator is missing from results")
        return (True, "mu_operator exists")

    def test_interface_sum_mu_mirror_exists(self):
        if self.sum_mu_mirror is None:
            return (False, "SKIP: sum_mu_mirror is missing from results")
        return (True, "sum_mu_mirror exists")

    def test_interface_mu_operator_not_empty(self):
        if self.mu_operator is None:
            return (False, "SKIP: mu_operator not available")
        if isinstance(self.mu_operator, str) and len(self.mu_operator.strip()) == 0:
            return (False, "mu_operator is an empty string")
        if isinstance(self.mu_operator, sp.Expr):
            return (True, "mu_operator is a non‑empty sympy expression")
        return (True, "mu_operator appears non‑empty")

    def test_interface_sum_mu_mirror_not_empty(self):
        if self.sum_mu_mirror is None:
            return (False, "SKIP: sum_mu_mirror not available")
        if isinstance(self.sum_mu_mirror, str) and len(self.sum_mu_mirror.strip()) == 0:
            return (False, "sum_mu_mirror is an empty string")
        if isinstance(self.sum_mu_mirror, sp.Expr):
            return (True, "sum_mu_mirror is a non‑empty sympy expression")
        return (True, "sum_mu_mirror appears non‑empty")

    # -------------------------------------------------------------------------
    # 2. Invariant / theorem‑based physical sanity tests (use math.isclose)
    # -------------------------------------------------------------------------
    def test_invariant_sum_mu_mirror_form(self):
        """
        Verify that sum_mu_mirror numerically matches
        J + (mu_p + mu_n - 0.5) * <sigma_z>
        for two independent test points.
        """
        expr = self._parse_expr(self.sum_mu_mirror)
        if expr is None:
            return (False, "SKIP: cannot parse sum_mu_mirror as a sympy expression")

        test_points = [
            (1.0, 2.0, 1.0 + self.coeff * 2.0),      # J=1, sigma=2
            (3.0, -1.0, 3.0 + self.coeff * (-1.0))    # J=3, sigma=-1
        ]

        for J_val, sigma_val, expected in test_points:
            result = self._safe_eval(expr, {self.J_sym: J_val,
                                            self.sigma_sym: sigma_val})
            if result is None:
                return (False, f"Evaluation failed at J={J_val}, sigma={sigma_val}")
            if not math.isclose(result, expected, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"Mismatch at (J={J_val}, sigma={sigma_val}): "
                               f"got {result}, expected {expected}")

        return (True, "sum_mu_mirror matches the expected linear form")

    def test_invariant_sum_mu_mirror_coefficient_J(self):
        """
        Check that the coefficient of J in the sum_mu_mirror expression is exactly 1.
        """
        expr = self._parse_expr(self.sum_mu_mirror)
        if expr is None:
            return (False, "SKIP: cannot parse sum_mu_mirror")
        try:
            # Express J as a symbol and sigma as another symbol, then expand
            poly = sp.expand(expr)
            coeff_J = poly.coeff(self.J_sym)
            if math.isclose(float(coeff_J), 1.0, rel_tol=1e-2, abs_tol=1e-5):
                return (True, f"Coefficient of J is 1 (got {coeff_J})")
            else:
                return (False, f"Coefficient of J is {coeff_J}, expected 1")
        except Exception as e:
            return (False, f"Failed to extract coefficient: {e}")

    def test_invariant_sum_mu_mirror_coefficient_sigma(self):
        """
        Check that the coefficient of <sigma_z> in the sum_mu_mirror expression
        equals mu_p + mu_n - 0.5 (numerically 0.38).
        """
        expr = self._parse_expr(self.sum_mu_mirror)
        if expr is None:
            return (False, "SKIP: cannot parse sum_mu_mirror")
        try:
            poly = sp.expand(expr)
            coeff_sigma = poly.coeff(self.sigma_sym)
            expected_coeff = self.coeff
            if math.isclose(float(coeff_sigma), expected_coeff, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Coefficient of <sigma_z> is {coeff_sigma} (expected {expected_coeff})")
            else:
                return (False, f"Coefficient of <sigma_z> is {coeff_sigma}, expected {expected_coeff}")
        except Exception as e:
            return (False, f"Failed to extract coefficient: {e}")

    # -------------------------------------------------------------------------
    # 3. Modeling / trace consistency tests (at least two)
    # -------------------------------------------------------------------------
    def test_modeling_mu_operator_mentions_operators(self):
        """
        Weak modeling check: the mu_operator string should contain references
        to orbital (L) and spin (S) angular momentum (or g-factors).
        """
        if self.mu_operator is None:
            return (False, "SKIP: mu_operator not available")

        if isinstance(self.mu_operator, str):
            keywords = ['g_l', 'g_s', 'L', 'S', '\\ell', 'l_i', 's_i']
            found = any(kw in self.mu_operator.lower() for kw in keywords)
            if not found:
                return (False, "mu_operator string lacks common keywords (g_l, g_s, L, S)")
            return (True, "mu_operator contains relevant operator names")

        if isinstance(self.mu_operator, sp.Expr):
            # Assume a sympy expression is correct by construction
            return (True, "mu_operator is a sympy expression (accepted)")

        return (False, "mu_operator has unknown type")

    def test_modeling_mu_operator_includes_isospin(self):
        """
        Check that the mu_operator expression contains isospin operators (tau_z or sigma_3).
        The operator should involve tau_z(i) to distinguish proton and neutron.
        """
        if self.mu_operator is None:
            return (False, "SKIP: mu_operator not available")

        if isinstance(self.mu_operator, str):
            keywords = ['tau', 'tau_z', 'isospin', 'tau_3', 'sigma_3', 'S_z']
            found = any(kw in self.mu_operator.lower() for kw in keywords)
            if not found:
                return (False, "mu_operator string lacks isospin-related keywords")
            return (True, "mu_operator mentions isospin operators")

        if isinstance(self.mu_operator, sp.Expr):
            # Check if the expression contains symbols that look like tau or sigma
            free_syms = self.mu_operator.free_symbols
            sym_names = [str(s) for s in free_syms]
            if any(('tau' in name) or ('sigma' in name) for name in sym_names):
                return (True, "mu_operator sympy expression contains tau/sigma symbols")
            else:
                return (False, "mu_operator sympy expression does not contain isospin symbols")
        return (False, "mu_operator has unknown type")

    def test_modeling_sum_mu_mirror_contains_J_and_sigma(self):
        """
        Verify that sum_mu_mirror mentions both J and sigma_z expectation.
        """
        if self.sum_mu_mirror is None:
            return (False, "SKIP: sum_mu_mirror not available")

        if isinstance(self.sum_mu_mirror, str):
            keywords = ['J', 'sigma', '\\sigma', 'langle_sum_sigma_z']
            found = any(kw in self.sum_mu_mirror for kw in keywords)
            if not found:
                return (False, "sum_mu_mirror string lacks references to J or sigma")
            return (True, "sum_mu_mirror mentions J and sigma")

        if isinstance(self.sum_mu_mirror, sp.Expr):
            free_syms = self.sum_mu_mirror.free_symbols
            if self.J_sym in free_syms and self.sigma_sym in free_syms:
                return (True, "sum_mu_mirror sympy expression contains J and sigma symbols")
            else:
                return (False, "sum_mu_mirror sympy expression missing J or sigma symbol")
        return (False, "sum_mu_mirror has unknown type")

    # -------------------------------------------------------------------------
    # 4. Monotonicity tests – perturb one input and check the response
    # -------------------------------------------------------------------------
    def test_monotonicity_sum_increases_with_J(self):
        """
        When J increases by ΔJ, the sum should increase by exactly ΔJ
        (since the coefficient of J is 1).
        """
        expr = self._parse_expr(self.sum_mu_mirror)
        if expr is None:
            return (False, "SKIP: cannot parse sum_mu_mirror")

        sigma_fixed = 1.0
        J_low = 0.0
        J_high = 2.0

        val_low = self._safe_eval(expr, {self.J_sym: J_low,
                                         self.sigma_sym: sigma_fixed})
        val_high = self._safe_eval(expr, {self.J_sym: J_high,
                                          self.sigma_sym: sigma_fixed})
        if val_low is None or val_high is None:
            return (False, "Numerical evaluation failed in monotonicity J test")

        delta = val_high - val_low
        expected_delta = J_high - J_low
        if not math.isclose(delta, expected_delta, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Sum increases by {delta} instead of {expected_delta} "
                           f"when J changes by {expected_delta}")
        return (True, "Sum increases correctly with J")

    def test_monotonicity_sum_increases_with_sigma(self):
        """
        When <sigma_z> increases by Δ, the sum should increase by
        (mu_p + mu_n - 0.5) * Δ.
        """
        expr = self._parse_expr(self.sum_mu_mirror)
        if expr is None:
            return (False, "SKIP: cannot parse sum_mu_mirror")

        J_fixed = 1.0
        sigma_low = 0.0
        sigma_high = 1.0

        val_low = self._safe_eval(expr, {self.J_sym: J_fixed,
                                         self.sigma_sym: sigma_low})
        val_high = self._safe_eval(expr, {self.J_sym: J_fixed,
                                          self.sigma_sym: sigma_high})
        if val_low is None or val_high is None:
            return (False, "Numerical evaluation failed in monotonicity sigma test")

        delta = val_high - val_low
        expected_delta = self.coeff * (sigma_high - sigma_low)
        if not math.isclose(delta, expected_delta, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Sum increases by {delta} instead of {expected_delta} "
                           f"when sigma changes by {sigma_high - sigma_low}")
        return (True, "Sum increases correctly with sigma")

    # -------------------------------------------------------------------------
    # 5. Answer sanity – weak, non‑numerical checks only
    # -------------------------------------------------------------------------
    def test_answer_sanity_sum_mu_mirror(self):
        if self.sum_mu_mirror is None:
            return (False, "SKIP: sum_mu_mirror not available")
        # Ensure it is a non-empty representation (string or sympy)
        if isinstance(self.sum_mu_mirror, str) and len(self.sum_mu_mirror.strip()) == 0:
            return (False, "sum_mu_mirror is empty string")
        if isinstance(self.sum_mu_mirror, sp.Expr):
            return (True, "sum_mu_mirror is a sympy expression (acceptable)")
        return (True, "sum_mu_mirror appears non-empty")