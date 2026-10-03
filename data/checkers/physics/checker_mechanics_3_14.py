import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Auditor for gravitational redshift derivation in weak static field.
    Checks consistency with p0 conservation, proper time relation, and weak-field expansion.
    """

    def __init__(self, solution_instance):
        # Extract results safely (use None for missing)
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}
        self.givens = getattr(solution_instance, 'givens', {})
        if not isinstance(self.givens, dict):
            self.givens = {}

        self.redshift_formula = self.results.get('redshift_formula', None)
        self.Phi_s = self.givens.get('Phi_s', None)
        self.Phi_e = self.givens.get('Phi_e', None)
        self.c = self.givens.get('c', None)

    def _parse_formula_expression(self):
        """
        Try to extract the symbolic expression from the stored formula string.
        Expected format: '... (nu_e - nu_s)/nu_s ~= (Phi_s - Phi_e)/c^2'
        Returns the sympy expression after '~=', or None if parsing fails.
        """
        if self.redshift_formula is None:
            return None
        formula_str = str(self.redshift_formula)
        if '~=' not in formula_str:
            return None
        _, expr_str = formula_str.split('~=', 1)
        expr_str = expr_str.strip()
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
        transformations = (standard_transformations + (implicit_multiplication_application,))
        try:
            expr_str_clean = expr_str.replace('\\Phi', 'Phi')
            expr = parse_expr(expr_str_clean, transformations=transformations)
            return expr
        except Exception:
            return None

    def _compute_redshift(self, Phi_s_val, Phi_e_val, c_val):
        """Compute redshift using either the parsed formula or the weak-field approximation."""
        expr = self._parse_formula_expression()
        if expr is not None:
            try:
                z_val = float(expr.subs({sp.Symbol('Phi_s'): Phi_s_val,
                                         sp.Symbol('Phi_e'): Phi_e_val,
                                         sp.Symbol('c'): c_val}).evalf())
                return z_val
            except Exception:
                pass
        # Fallback to approximate formula
        return (Phi_s_val - Phi_e_val) / c_val**2

    # ---------------------------------------------------
    # Interface tests
    # ---------------------------------------------------
    def test_interface_redshift_formula_key(self):
        """Check that results['redshift_formula'] exists and is not None."""
        if 'redshift_formula' not in self.results:
            return (False, "Missing key 'redshift_formula'")
        if self.redshift_formula is None:
            return (False, "'redshift_formula' is None")
        return (True, "Key 'redshift_formula' present and non-None")

    # ---------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ---------------------------------------------------
    def test_invariant_weak_field_consistency(self):
        """
        Verify that the redshift expression satisfies (1+z)*sqrt(g00(e)) - sqrt(g00(s)) = 0
        to first order in Phi/c^2 under the weak-field expansion.
        Uses sympy on the parsed formula.
        """
        expr = self._parse_formula_expression()
        if expr is None:
            return (False, "SKIP: Cannot parse redshift formula for symbolic check")
        Phi_s, Phi_e, c = sp.symbols('Phi_s Phi_e c', real=True)
        g00_s = 1 + 2*Phi_s/c**2
        g00_e = 1 + 2*Phi_e/c**2
        z = expr
        invariant = (1+z)*sp.sqrt(g00_e) - sp.sqrt(g00_s)
        epsilon = sp.symbols('epsilon')
        invariant_eps = invariant.subs({1/c**2: epsilon})
        inv_series = sp.series(invariant_eps, epsilon, 0, 2).removeO()
        inv_expanded = inv_series.subs(epsilon, 1/c**2)
        inv_expanded_simplified = sp.simplify(inv_expanded)
        if inv_expanded_simplified == 0:
            return (True, "Weak-field invariant satisfied to leading order")
        else:
            return (False, f"Invariant not zero after expansion: {inv_expanded_simplified}")

    def test_invariant_numeric_ratio(self):
        """
        Choose typical numerical values (weak-field regime) and check that the
        exact ratio nu_e/nu_s = sqrt(g00(s))/sqrt(g00(e)) holds approximately
        when using the redshift formula.
        """
        expr = self._parse_formula_expression()
        if expr is None:
            return (False, "SKIP: Cannot parse formula for numeric test")
        Phi_s_val = -1.9e11
        Phi_e_val = -6.3e7
        c_val = 3.0e8
        if abs(Phi_s_val)/c_val**2 > 0.01 or abs(Phi_e_val)/c_val**2 > 0.01:
            return (False, "Numeric values violate weak-field assumption")
        z_subs = expr.subs({sp.Symbol('Phi_s'): Phi_s_val,
                            sp.Symbol('Phi_e'): Phi_e_val,
                            sp.Symbol('c'): c_val})
        try:
            z_val = float(z_subs.evalf())
        except Exception:
            return (False, "Could not evaluate formula numerically")
        g00_s = 1 + 2*Phi_s_val / c_val**2
        g00_e = 1 + 2*Phi_e_val / c_val**2
        ratio_exact = math.sqrt(g00_s / g00_e)
        ratio_from_z = 1 + z_val
        diff = abs(ratio_exact - ratio_from_z)
        rel_tol = 0.02
        if diff <= rel_tol * max(abs(ratio_exact), abs(ratio_from_z), 1e-12):
            return (True, f"Exact ratio {ratio_exact:.6f} differs from 1+z ({ratio_from_z:.6f}) by {diff:.2e} (within tolerance)")
        else:
            return (False, f"Exact ratio {ratio_exact:.6f} vs 1+z {ratio_from_z:.6f} differ by {diff:.2e} (exceeds tolerance)")

    # ---------------------------------------------------
    # Modeling/Trace tests (consistency with derivation steps)
    # ---------------------------------------------------
    def test_modeling_redshift_definition(self):
        """Check that the formula is defined as (nu_e - nu_s)/nu_s."""
        if self.redshift_formula is None:
            return (False, "SKIP: No formula to check")
        formula_str = str(self.redshift_formula)
        if '(nu_e - nu_s)/nu_s' in formula_str or '(ν_e - ν_s)/ν_s' in formula_str:
            return (True, "Redshift defined as (nu_e - nu_s)/nu_s")
        else:
            return (False, "Formula does not appear to define redshift as (nu_e - nu_s)/nu_s")

    def test_modeling_weak_field_approx_used(self):
        """Check that formula uses approximation sign (~=) indicating weak-field expansion."""
        if self.redshift_formula is None:
            return (False, "SKIP: No formula")
        if '~=' in str(self.redshift_formula):
            return (True, "Approximation sign present, consistent with weak-field expansion")
        else:
            return (False, "Missing approximation sign (expected ~=)")

    # ---------------------------------------------------
    # Monotonicity tests (explicit perturbation and baseline comparison)
    # ---------------------------------------------------
    def test_monotonicity_vary_phi_s(self):
        """
        Increase Phi_s (make it less negative) by a small percentage and check that
        redshift increases (becomes less negative). This tests perturbs an input and
        compares the perturbed output to the baseline.
        """
        # Use provided numeric values or reasonable defaults if missing
        Phi_s_base = self.Phi_s if self.Phi_s is not None else -1.9e11
        Phi_e_base = self.Phi_e if self.Phi_e is not None else -6.3e7
        c_base = self.c if self.c is not None else 3.0e8

        baseline_z = self._compute_redshift(Phi_s_base, Phi_e_base, c_base)

        # Perturb Phi_s by +1% (increase, less negative)
        perturbation_factor = 1.01
        Phi_s_perturbed = Phi_s_base * perturbation_factor
        perturbed_z = self._compute_redshift(Phi_s_perturbed, Phi_e_base, c_base)

        # Check direction: increasing Phi_s should cause redshift to increase (become larger, less negative)
        if perturbed_z > baseline_z:
            return (True, f"Baseline z={baseline_z:.6e}, perturbed z={perturbed_z:.6e}: increasing Phi_s increases redshift (correct)")
        else:
            return (False, f"Baseline z={baseline_z:.6e}, perturbed z={perturbed_z:.6e}: increasing Phi_s did NOT increase redshift")

    def test_monotonicity_vary_phi_e(self):
        """
        Increase Phi_e (make it less negative) by a small percentage and check that
        redshift decreases (becomes more negative). This tests perturbs an input and
        compares the perturbed output to the baseline.
        """
        Phi_s_base = self.Phi_s if self.Phi_s is not None else -1.9e11
        Phi_e_base = self.Phi_e if self.Phi_e is not None else -6.3e7
        c_base = self.c if self.c is not None else 3.0e8

        baseline_z = self._compute_redshift(Phi_s_base, Phi_e_base, c_base)

        # Perturb Phi_e by +1% (increase, less negative)
        perturbation_factor = 1.01
        Phi_e_perturbed = Phi_e_base * perturbation_factor
        perturbed_z = self._compute_redshift(Phi_s_base, Phi_e_perturbed, c_base)

        # Check direction: increasing Phi_e should cause redshift to decrease (become smaller, more negative)
        if perturbed_z < baseline_z:
            return (True, f"Baseline z={baseline_z:.6e}, perturbed z={perturbed_z:.6e}: increasing Phi_e decreases redshift (correct)")
        else:
            return (False, f"Baseline z={baseline_z:.6e}, perturbed z={perturbed_z:.6e}: increasing Phi_e did NOT decrease redshift")

    # ---------------------------------------------------
    # Weak answer sanity test (only one, not dominating)
    # ---------------------------------------------------
    def test_answer_sanity_sign_typical(self):
        """
        If numeric potentials are provided, check that redshift is negative
        (frequency decreases from sun to earth). This is a weak sanity only;
        not a precise value match.
        """
        if self.Phi_s is None or self.Phi_e is None or self.c is None:
            return (False, "SKIP: Numeric potentials not available")
        z_val = self._compute_redshift(self.Phi_s, self.Phi_e, self.c)
        if z_val < 0:
            return (True, f"Redshift is negative ({z_val:.3e}), consistent with expected decrease in frequency")
        else:
            return (False, f"Redshift is non-negative ({z_val:.3e}), but expected negative for sun->earth")

    # ---------------------------------------------------
    # Main runner
    # ---------------------------------------------------
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