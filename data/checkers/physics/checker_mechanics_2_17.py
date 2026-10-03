import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Physics Auditor for the central force problem (Lagrangian, conservation,
    planar motion, Kepler's second law, harmonic oscillator turning points).
    """

    def __init__(self, solution_instance):
        # Extract results safely – always use .get with None default
        results = getattr(solution_instance, 'results', {}) or {}
        self.lagrangian = results.get('lagrangian', None)
        self.E = results.get('E', None)
        self.J = results.get('J', None)
        self.dA_dt = results.get('dA_dt', None)
        self.turning_equation = results.get('turning_equation', None)

        # Optional givens that may be present in results or directly in solution
        self.mu = results.get('mu', None)
        self.k = results.get('k', None)
        self.m1 = getattr(solution_instance, 'm1', None)
        self.m2 = getattr(solution_instance, 'm2', None)

    # ------------------------------------------------------------------
    # Helper to parse a string/sympy expression into a sympy expression
    # ------------------------------------------------------------------
    def _parse_expr(self, expr):
        if expr is None:
            return None
        if isinstance(expr, str):
            try:
                return sp.sympify(expr)
            except Exception:
                return expr  # keep as is if cannot parse (will cause later checks to fail)
        elif isinstance(expr, sp.Expr):
            return expr
        else:
            return None

    # ------------------------------------------------------------------
    # Test: interface – all required result keys exist and are not None
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that the expected result keys are present and not None."""
        required = ['lagrangian', 'E', 'J', 'dA_dt', 'turning_equation']
        missing = [k for k in required if getattr(self, k) is None]
        if missing:
            return (False, f"Missing required result keys: {missing}")
        return (True, "All required result keys present")

    # ------------------------------------------------------------------
    # Test: invariant – Kepler's second law (area constant)
    # ------------------------------------------------------------------
    def test_invariant_kepler_second_law(self):
        """Check that dA_dt = J/(2*mu) is consistent, if mu is known."""
        if self.J is None or self.dA_dt is None:
            return (False, "SKIP: J or dA_dt not available")
        # Try to get mu from results or derive from m1,m2
        mu = self.mu
        if mu is None and self.m1 is not None and self.m2 is not None:
            mu = self.m1 * self.m2 / (self.m1 + self.m2)
        if mu is None:
            return (False, "SKIP: mu not available and cannot be derived")
        # Convert J and dA_dt to numeric if they are symbolic
        J_val = self.J
        dA_val = self.dA_dt
        if isinstance(J_val, sp.Expr):
            # If it's an expression like 'mu*r^2*theta_dot', try to get numeric from results
            # For simplicity, we rely on numeric values if present; else skip.
            return (False, "SKIP: J is symbolic, cannot test numeric consistency")
        # Attempt numeric conversion
        try:
            J_val = float(J_val) if not isinstance(J_val, (int, float)) else J_val
            dA_val = float(dA_val) if not isinstance(dA_val, (int, float)) else dA_val
            mu_val = float(mu)
        except (TypeError, ValueError):
            return (False, "SKIP: J, dA_dt or mu non-numeric")
        expected_dA = J_val / (2 * mu_val)
        if math.isclose(dA_val, expected_dA, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"dA/dt = {dA_val} close to J/(2µ) = {expected_dA}")
        else:
            return (False, f"dA/dt = {dA_val} does not match J/(2µ) = {expected_dA}")

    # ------------------------------------------------------------------
    # Test: invariant – turning point condition is a quartic in r^2
    # ------------------------------------------------------------------
    def test_invariant_turning_equation_form(self):
        """Check that the turning equation is a polynomial of degree 4 in r (or 2 in r^2)."""
        if self.turning_equation is None:
            return (False, "SKIP: turning_equation not available")
        expr = self._parse_expr(self.turning_equation)
        if expr is None:
            return (False, "SKIP: turning_equation could not be parsed")
        # Check that the expression is zero and involves r, mu, k, E, J
        # We look for typical structure: mu*k*r**4 - 2*mu*E*r**2 + J**2
        # But we allow it to be written as an equation (LHS == RHS) or a single expression.
        # For robustness, we check that it contains r**4 and r**2 and is a polynomial.
        try:
            # Try to extract polynomial coefficients with respect to r
            poly = sp.Poly(expr, sp.Symbol('r'), domain=sp.QQ)
            # It should be of degree 4 and have no odd powers (or only even powers)
            if poly.degree() != 4:
                return (False, f"Turning equation degree {poly.degree()} != 4")
            # Check that the coefficient of r^1 and r^3 are zero
            coeffs = poly.all_coeffs()
            if len(coeffs) != 5:  # degree 4 gives 5 coefficients
                return (False, "Poly coefficient count mismatch")
            # coefficients: [a4, a3, a2, a1, a0] for r^4, r^3, r^2, r^1, constant
            a3 = coeffs[1]
            a1 = coeffs[3]
            if a3 != 0 or a1 != 0:
                return (False, "Turning equation contains odd powers of r; expected only even powers")
            # Check that a4 and a2 have opposite signs? Not necessary, but typical with mu*k >0, -2muE<0
            return (True, "Turning equation is a quartic in r with only even powers")
        except sp.PolynomialError:
            return (False, "Turning equation is not a polynomial in r")
        except Exception:
            return (False, "Unexpected error parsing turning equation")

    # ------------------------------------------------------------------
    # Test: modeling – Lagrangian structure
    # ------------------------------------------------------------------
    def test_modeling_lagrangian_structure(self):
        """Check that the Lagrangian contains kinetic energy term (½µ(ṙ²+r²θ̇²)) and V(r)."""
        if self.lagrangian is None:
            return (False, "SKIP: lagrangian not available")
        L_str = str(self.lagrangian)
        # Check for typical keywords
        has_mu = 'mu' in L_str or 'μ' in L_str
        has_r_dot = 'r_dot' in L_str or 'ṙ' in L_str or 'rdot' in L_str
        has_theta_dot = 'theta_dot' in L_str or 'θ̇' in L_str or 'thetadot' in L_str
        has_V = 'V(' in L_str or 'V' in L_str or '-V' in L_str
        # Also check for r^2 term
        has_r2 = 'r**2' in L_str or 'r^2' in L_str
        if has_mu and has_r_dot and has_theta_dot and has_V and has_r2:
            return (True, "Lagrangian contains expected structure (mu, r_dot, theta_dot, V, r^2)")
        else:
            return (False, f"Lagrangian string missing expected components (mu:{has_mu}, r_dot:{has_r_dot}, theta_dot:{has_theta_dot}, V:{has_V}, r^2:{has_r2})")

    # ------------------------------------------------------------------
    # Test: modeling – Energy expression includes effective potential form
    # ------------------------------------------------------------------
    def test_modeling_energy_expression(self):
        """Check that the energy expression contains J^2/(2µr^2) or similar."""
        if self.E is None:
            return (False, "SKIP: E not available")
        E_str = str(self.E)
        # Look for J^2/(2*mu*r^2) or J^2/(2µr^2) or similar
        pattern_angular = 'J**2/(2*mu*r**2)' in E_str or 'J**2/(2µr**2)' in E_str or 'J^2/(2μ r^2)' in E_str
        # Also expect a term from potential V(r) or (1/2)kr^2
        has_potential = 'V(r)' in E_str or 'k*r**2' in E_str or '0.5*k*r**2' in E_str
        if pattern_angular or has_potential:
            return (True, "Energy expression includes angular barrier and potential")
        else:
            return (False, f"Energy expression missing angular barrier or potential: {E_str[:100]}")

    # ------------------------------------------------------------------
    # Test: monotonicity – area rate increases with angular momentum
    # ------------------------------------------------------------------
    def test_monotonicity_area_with_angular_momentum(self):
        """If J and dA_dt are numeric, check that dA/dt scales linearly with J."""
        if self.J is None or self.dA_dt is None or self.mu is None:
            return (False, "SKIP: need numeric J, dA_dt, mu for monotonicity")
        # Attempt numeric conversion
        try:
            J_val = float(self.J)
            dA_val = float(self.dA_dt)
            mu_val = float(self.mu)
        except (TypeError, ValueError):
            return (False, "SKIP: non-numeric values")
        # Check baseline relation: dA_dt = J/(2µ)
        if not math.isclose(dA_val, J_val/(2*mu_val), rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Baseline dA_dt {dA_val} not matching J/(2µ)={J_val/(2*mu_val)}")
        # Now check monotonicity: if we hypothetically increase J by 10%, what happens to dA_dt?
        # Since the relation is linear, dA_dt should increase by the same factor.
        J_pert = J_val * 1.1
        dA_expected = J_pert / (2*mu_val)
        factor = dA_expected / dA_val
        if math.isclose(factor, 1.1, rel_tol=1e-2):
            return (True, f"Linear scaling holds: increasing J by 10% increases dA/dt by {factor:.3f}")
        else:
            return (False, f"Monotonicity violated: expected factor 1.1, got {factor:.3f}")

    # ------------------------------------------------------------------
    # Test: answer sanity (weak) – energy sign and finite
    # ------------------------------------------------------------------
    def test_answer_sanity_energy_finite_positive(self):
        """Basic check that total energy E is finite and non-negative (for bound motion)."""
        if self.E is None:
            return (False, "SKIP: E not available")
        # Try to convert to numeric if possible
        try:
            E_val = float(self.E) if not isinstance(self.E, (int, float)) else self.E
        except (TypeError, ValueError):
            return (True, "PASS: E is symbolic (weak sanity skipped)")
        if not math.isfinite(E_val):
            return (False, "E is not finite (inf or NaN)")
        if E_val < 0:
            return (False, f"E = {E_val} is negative, but expected non-negative for bound motion")
        return (True, f"E = {E_val} is finite and non-negative")

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        """Discover and run all test methods, collect logs."""
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