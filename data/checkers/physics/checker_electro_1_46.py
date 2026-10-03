import math
import sympy as sym
from sympy.parsing.sympy_parser import parse_expr

class PhysicsChecker:
    """
    Audits the solution for a cylindrical coaxial capacitor with a radially
    varying dielectric constant such that the energy density is constant.
    """

    def __init__(self, solution_instance):
        # Extract results dictionary safely
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}

        self.K_form = results.get('K_form', None)   # expected: "k / r**2" or similar
        self.E = results.get('E', None)              # expected: "-2*V*r/(b**2 - a**2)" or similar

        # Extract given parameters (may be on the solution instance directly)
        self.a = getattr(solution_instance, 'a', None)
        self.b = getattr(solution_instance, 'b', None)
        self.V = getattr(solution_instance, 'V', None)
        self.epsilon0 = getattr(solution_instance, 'epsilon0', None)

        # Define symbols for parsing
        self.r_sym = sym.Symbol('r', positive=True)
        self.a_sym = sym.Symbol('a', positive=True)
        self.b_sym = sym.Symbol('b', positive=True)
        self.V_sym = sym.Symbol('V')
        self.eps0_sym = sym.Symbol('epsilon0', positive=True)
        self.k_sym = sym.Symbol('k')  # constant in K(r) = k / r^2

    def _parse_expr(self, expr_input):
        """Convert a string or sympy expression to a sympy expression."""
        if expr_input is None:
            return None
        if isinstance(expr_input, sym.Basic):
            return expr_input
        try:
            # Replace common textual forms like "K(r)" with just "k / r**2"
            expr_str = str(expr_input)
            # Remove any "K(r) = " prefix if present
            if '=' in expr_str:
                expr_str = expr_str.split('=', 1)[1].strip()
            return parse_expr(expr_str)
        except Exception:
            return None

    def _get_K_expr(self):
        """Return a sympy expression for K(r) or None."""
        return self._parse_expr(self.K_form)

    def _get_E_expr(self):
        """Return a sympy expression for E(r) or None."""
        return self._parse_expr(self.E)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that required result keys exist and are not None."""
        if self.K_form is None or self.E is None:
            return (False, "Missing required result key: K_form or E")
        return (True, "Both K_form and E present")

    def test_interface_K_form_parseable(self):
        """Check that K_form can be parsed to a sympy expression."""
        K = self._get_K_expr()
        if K is None:
            return (False, "K_form cannot be parsed")
        return (True, "K_form parsed successfully")

    def test_interface_E_form_parseable(self):
        """Check that E can be parsed to a sympy expression."""
        E = self._get_E_expr()
        if E is None:
            return (False, "E cannot be parsed")
        return (True, "E parsed successfully")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_energy_density_constant(self):
        """
        Verify that the energy density u = 0.5*epsilon0*K*E^2 is independent of r.
        Use numerical evaluation if numeric parameters are available.
        """
        K = self._get_K_expr()
        E = self._get_E_expr()
        if K is None or E is None:
            return (False, "SKIP: K or E not available")

        # Check if numeric parameters are present
        if self.a is not None and self.b is not None and self.V is not None:
            try:
                a_val = float(self.a)
                b_val = float(self.b)
                V_val = float(self.V)
                eps0_val = 1.0 if self.epsilon0 is None else float(self.epsilon0)
                # Substitute symbolic parameters with numbers
                subs_K = K.subs({self.a_sym: a_val, self.b_sym: b_val,
                                 self.V_sym: V_val, self.eps0_sym: eps0_val})
                subs_E = E.subs({self.a_sym: a_val, self.b_sym: b_val,
                                 self.V_sym: V_val, self.eps0_sym: eps0_val})
                # If symbols remain (e.g., k), we cannot evaluate numerically.
                # Instead, check symbolically that u is independent of r.
                u = 0.5 * self.eps0_sym * K * E**2
                # Differentiate with respect to r
                du_dr = sym.diff(u, self.r_sym)
                simplified = sym.simplify(du_dr)
                if simplified == 0:
                    return (True, "Energy density is constant (du/dr = 0)")
                else:
                    # Try numerical check using arbitrary numeric values for remaining symbols
                    # For K = k/r**2 and E = -2Vr/(b**2-a**2), symbolically it should be constant.
                    # If not, it's a failure.
                    return (False, f"Energy density derivative not zero: {simplified}")
            except Exception as e:
                return (False, f"Error in energy density test: {e}")
        else:
            # Symbolic check only
            try:
                u = 0.5 * self.eps0_sym * K * E**2
                du_dr = sym.diff(u, self.r_sym)
                simplified = sym.simplify(du_dr)
                if simplified == 0:
                    return (True, "Energy density is constant (du/dr = 0)")
                else:
                    return (False, f"Energy density derivative not zero: {simplified}")
            except Exception as e:
                return (False, f"Error in symbolic energy density test: {e}")

    def test_invariant_voltage_integral(self):
        """
        Check that V = -∫_a^b E(r) dr holds.
        """
        E = self._get_E_expr()
        if E is None:
            return (False, "SKIP: E not available")
        try:
            # Integrate E from a to b
            integral = sym.integrate(E, (self.r_sym, self.a_sym, self.b_sym))
            # The voltage integral should equal -V (assuming minus sign convention)
            expected = -self.V_sym
            difference = sym.simplify(integral - expected)
            if difference == 0:
                return (True, "Voltage integral condition satisfied: ∫E dr = -V")
            else:
                # Try numeric check if numeric parameters available
                if self.a is not None and self.b is not None and self.V is not None:
                    a_val = float(self.a)
                    b_val = float(self.b)
                    V_val = float(self.V)
                    subs = {self.a_sym: a_val, self.b_sym: b_val, self.V_sym: V_val}
                    integral_num = float(integral.subs(subs))
                    if math.isclose(integral_num, -V_val, rel_tol=1e-2, abs_tol=1e-2):
                        return (True, f"Voltage integral numerically matches: {integral_num} ≈ {-V_val}")
                    else:
                        return (False, f"Voltage integral mismatch: {integral_num} vs expected {-V_val}")
                else:
                    return (False, f"Voltage integral condition fails symbolically: ∫E dr - (-V) = {difference}")
        except Exception as e:
            return (False, f"Error in voltage integral test: {e}")

    def test_invariant_K_times_r_squared_constant(self):
        """
        Check that K(r) * r^2 is independent of r (constant).
        """
        K = self._get_K_expr()
        if K is None:
            return (False, "SKIP: K not available")
        try:
            expr = sym.simplify(K * self.r_sym**2)
            # Check that derivative with respect to r is zero
            d_expr_dr = sym.diff(expr, self.r_sym)
            if sym.simplify(d_expr_dr) == 0:
                return (True, "K(r)*r^2 is constant")
            else:
                # Try numerical check
                if self.a is not None and self.b is not None and self.V is not None:
                    a_val = float(self.a)
                    b_val = float(self.b)
                    V_val = float(self.V)
                    subs = {self.a_sym: a_val, self.b_sym: b_val, self.V_sym: V_val}
                    r1, r2 = (a_val + b_val)/2, (a_val + 3*b_val)/4
                    val1 = float(expr.subs({self.r_sym: r1, **subs}))
                    val2 = float(expr.subs({self.r_sym: r2, **subs}))
                    if math.isclose(val1, val2, rel_tol=1e-2, abs_tol=1e-2):
                        return (True, f"K(r)*r^2 numerically constant: {val1} ≈ {val2}")
                    else:
                        return (False, f"K(r)*r^2 not constant: {val1} vs {val2}")
                else:
                    return (False, f"K(r)*r^2 derivative not zero: {d_expr_dr}")
        except Exception as e:
            return (False, f"Error in K_r2 test: {e}")

    def test_invariant_E_over_r_constant(self):
        """
        Check that E(r)/r is independent of r.
        """
        E = self._get_E_expr()
        if E is None:
            return (False, "SKIP: E not available")
        try:
            expr = sym.simplify(E / self.r_sym)
            d_expr_dr = sym.diff(expr, self.r_sym)
            if sym.simplify(d_expr_dr) == 0:
                return (True, "E(r)/r is constant")
            # numeric fallback
            if self.a is not None and self.b is not None and self.V is not None:
                a_val = float(self.a)
                b_val = float(self.b)
                V_val = float(self.V)
                subs = {self.a_sym: a_val, self.b_sym: b_val, self.V_sym: V_val}
                r1, r2 = (a_val + b_val)/2, (a_val + 3*b_val)/4
                val1 = float(expr.subs({self.r_sym: r1, **subs}))
                val2 = float(expr.subs({self.r_sym: r2, **subs}))
                if math.isclose(val1, val2, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, f"E(r)/r numerically constant: {val1} ≈ {val2}")
                else:
                    return (False, f"E(r)/r not constant: {val1} vs {val2}")
            else:
                return (False, f"E(r)/r derivative not zero: {d_expr_dr}")
        except Exception as e:
            return (False, f"Error in E_over_r test: {e}")

    # ------------------------------------------------------------------
    # Modeling tests (quality checks based on explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_K_form_inverse_r_squared(self):
        """
        Check that K(r) has the form constant * r^{-2}.
        """
        K = self._get_K_expr()
        if K is None:
            return (False, "SKIP: K not available")
        try:
            expr = sym.simplify(K * self.r_sym**2)
            if expr.is_constant():
                return (True, "K(r) = constant / r^2")
            else:
                return (False, f"K(r) * r^2 is not constant: {expr}")
        except Exception as e:
            return (False, f"Error in K_form modeling: {e}")

    def test_modeling_E_form_proportional_r(self):
        """
        Check that E(r) is proportional to r, i.e., E(r)/r is constant.
        """
        E = self._get_E_expr()
        if E is None:
            return (False, "SKIP: E not available")
        try:
            expr = sym.simplify(E / self.r_sym)
            if expr.is_constant():
                return (True, "E(r) ∝ r")
            else:
                return (False, f"E(r)/r is not constant: {expr}")
        except Exception as e:
            return (False, f"Error in E_form modeling: {e}")

    def test_modeling_E_scaling_with_V(self):
        """
        Check that E is linear in V (as expected).
        """
        E = self._get_E_expr()
        if E is None:
            return (False, "SKIP: E not available")
        try:
            dE_dV = sym.diff(E, self.V_sym)
            # It should not contain V (i.e., derivative is constant with respect to V)
            if not dE_dV.has(self.V_sym):
                return (True, "E is linear in V")
            else:
                return (False, f"dE/dV depends on V: {dE_dV}")
        except Exception as e:
            return (False, f"Error in E_scaling test: {e}")

    # ------------------------------------------------------------------
    # Weak answer sanity test (only one, no isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_E_sign(self):
        """
        Weak sanity: For positive V, E should be negative at the inner conductor.
        Only perform if numeric parameters are available.
        """
        if self.a is None or self.b is None or self.V is None:
            return (True, "SKIP: no numeric parameters")
        E = self._get_E_expr()
        if E is None:
            return (False, "SKIP: E not available")
        try:
            V_val = float(self.V)
            if V_val <= 0:
                return (True, "SKIP: non-positive V - cannot perform sign check")
            a_val = float(self.a)
            b_val = float(self.b)
            subs = {self.a_sym: a_val, self.b_sym: b_val, self.V_sym: V_val, self.r_sym: a_val}
            E_a = float(E.subs(subs))
            if E_a < 0:
                return (True, f"E(a) = {E_a} < 0 for V>0, acceptable sign")
            else:
                return (False, f"E(a) = {E_a} should be negative when V>0")
        except Exception as e:
            return (False, f"Error in sign sanity test: {e}")

    # ------------------------------------------------------------------
    # Monotonicity test (explicit perturbation of V, compare perturbed E to baseline)
    # ------------------------------------------------------------------
    def test_monotonicity_E_with_V(self):
        """
        Check that E(r) decreases monotonically as V increases.
        Perturb V by +5% and +10% and verify that the resulting E is less than the baseline.
        Only runs if numeric a, b, V are available.
        """
        if self.a is None or self.b is None or self.V is None:
            return (True, "SKIP: numeric a, b, V needed for perturbation test")
        E = self._get_E_expr()
        if E is None:
            return (False, "SKIP: E not available")
        try:
            a_val = float(self.a)
            b_val = float(self.b)
            V_base = float(self.V)
            r_test = (a_val + b_val) / 2.0   # any interior point

            # Compute baseline E at V_base
            subs_base = {self.a_sym: a_val, self.b_sym: b_val,
                         self.r_sym: r_test, self.V_sym: V_base}
            E_base = float(E.subs(subs_base))

            # Define perturbation percentages
            perturbations = [5, 10]  # +5% and +10%
            for pct in perturbations:
                V_pert = V_base * (1 + pct / 100.0)
                subs_pert = {self.a_sym: a_val, self.b_sym: b_val,
                             self.r_sym: r_test, self.V_sym: V_pert}
                E_pert = float(E.subs(subs_pert))

                # For a physically correct solution (E = -2Vr/(b^2-a^2)),
                # increasing V makes E more negative (decreases).
                if not (E_pert < E_base):
                    return (False, f"E not monotonic: at +{pct}% perturbation, "
                                   f"E({V_pert}) = {E_pert} is not less than "
                                   f"E({V_base}) = {E_base}")

            return (True, f"E is monotonically decreasing in V: verified with +5% and +10% perturbations")
        except Exception as e:
            return (False, f"Error in monotonicity test: {e}")

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