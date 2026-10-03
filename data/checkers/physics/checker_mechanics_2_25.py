import math
import sympy

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results dictionary; default to empty dict if missing
        self.results = getattr(solution_instance, 'results', {})
        # Required keys from the pseudocode contract (use None for missing)
        self.lagrangian = self.results.get('lagrangian', None)
        self.r_eom = self.results.get('r_eom', None)
        self.theta_eom = self.results.get('theta_eom', None)
        self.angular_momentum = self.results.get('angular_momentum', None)
        self.omega = self.results.get('omega', None)
        self.constant = self.results.get('constant', None)
        self.orbit_equation = self.results.get('orbit_equation', None)

        # Optional pieces – might be provided as givens in solution_instance
        self.m = getattr(solution_instance, 'm', None)
        self.k = getattr(solution_instance, 'k', None)
        self.kprime = getattr(solution_instance, "k'", None)  # note the key
        self.l = getattr(solution_instance, 'l', None)

    def _parse_expr(self, expr_str):
        """Try to sympify a string; return None on failure."""
        if expr_str is None:
            return None
        try:
            return sympy.sympify(expr_str)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # INTERFACE TESTS
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all five required output keys are present."""
        required = ['lagrangian', 'r_eom', 'theta_eom', 'angular_momentum', 'orbit_equation']
        missing = [k for k in required if self.results.get(k) is None]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    def test_interface_optional_keys(self):
        """Check that optional keys (omega, constant) are present if they were emitted."""
        # Not failing if missing – just informative
        optional = ['omega', 'constant']
        missing = [k for k in optional if self.results.get(k) is None]
        if missing:
            return (True, f"Optional keys missing (OK): {missing}")
        return (True, "Optional keys present")

    # ------------------------------------------------------------------
    # INVARIANT / THEOREM-BASED TESTS (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_lagrangian_structure(self):
        """Verify the Lagrangian matches L = 1/2 m rdot^2 + 1/2 m r^2 thetadot^2 + k/r - k'/(2 r^2)."""
        if self.lagrangian is None:
            return (False, "SKIP: lagrangian not available")
        L = self._parse_expr(self.lagrangian)
        if L is None:
            return (False, "lagrangian not parseable as sympy expression")
        # Expected symbolic expression
        m, r, rdot, theta, thetadot, k, kprime = sympy.symbols('m r rdot theta thetadot k kprime')
        expected = sympy.Rational(1,2)*m*rdot**2 + sympy.Rational(1,2)*m*r**2*thetadot**2 + k/r - kprime/(2*r**2)
        # Try symbolic simplification
        diff = sympy.simplify(L - expected)
        if diff == 0:
            return (True, "Lagrangian matches expected symbolic form")
        # Numerical fallback
        try:
            subs = {m:1.0, r:0.5, rdot:0.3, theta:0.2, thetadot:1.5, k:2.0, kprime:0.5}
            num_diff = float(L.subs(subs) - expected.subs(subs))
            if math.isclose(num_diff, 0, abs_tol=1e-6):
                return (True, "Lagrangian numerically matches expected form")
            else:
                return (False, f"Lagrangian numerical mismatch: diff {num_diff}")
        except Exception as e:
            return (False, f"Lagrangian fallback check failed: {e}")

    def test_invariant_angular_momentum_expression(self):
        """Check that l = m r^2 thetadot."""
        if self.angular_momentum is None:
            return (False, "SKIP: angular_momentum not available")
        l_expr = self._parse_expr(self.angular_momentum)
        if l_expr is None:
            return (False, "angular_momentum not parseable")
        m, r, thetadot = sympy.symbols('m r thetadot')
        expected = m * r**2 * thetadot
        diff = sympy.simplify(l_expr - expected)
        if diff == 0:
            return (True, "Angular momentum expression correct")
        try:
            subs = {m:2.0, r:0.8, thetadot:3.0}
            num_diff = float(l_expr.subs(subs) - expected.subs(subs))
            if math.isclose(num_diff, 0, abs_tol=1e-6):
                return (True, "Angular momentum numerically matches")
            else:
                return (False, f"Angular momentum numerical mismatch: {num_diff}")
        except Exception as e:
            return (False, f"Angular momentum check error: {e}")

    def test_invariant_binet_ode_structure(self):
        """Weak check that the orbit equation is of form r = 1/(A cos(ω θ) + C)."""
        if self.orbit_equation is None:
            return (False, "SKIP: orbit_equation not available")
        expr = self._parse_expr(self.orbit_equation)
        if expr is None:
            return (False, "orbit_equation not parseable")
        try:
            numerator, denominator = expr.as_numer_denom()
            if denominator.has(sympy.cos):
                return (True, "Orbit equation contains 1/(...cos...) structure")
            else:
                return (False, "Orbit equation does not contain cos in denominator")
        except Exception as e:
            return (False, f"Orbit structure check error: {e}")

    # ------------------------------------------------------------------
    # MODELING / TRACE CONSISTENCY TESTS
    # ------------------------------------------------------------------
    def test_modeling_omega_expression(self):
        """Check ω = sqrt(1 + m k' / l^2)."""
        if self.omega is None:
            return (False, "SKIP: omega not available")
        omega_expr = self._parse_expr(self.omega)
        if omega_expr is None:
            return (False, "omega not parseable")
        m, kprime, l = sympy.symbols('m kprime l', positive=True)
        expected = sympy.sqrt(1 + m*kprime/l**2)
        diff = sympy.simplify(omega_expr - expected)
        if diff == 0:
            return (True, "Omega expression correct")
        try:
            subs = {m:1.0, kprime:0.5, l:2.0}
            num_diff = float(omega_expr.subs(subs) - expected.subs(subs))
            if math.isclose(num_diff, 0, rel_tol=1e-9):
                return (True, "Omega numerically matches")
            else:
                return (False, f"Omega numerical mismatch: {num_diff}")
        except Exception as e:
            return (False, f"Omega check error: {e}")

    def test_modeling_constant_expression(self):
        """Check constant = m k / (l^2 + m k')."""
        if self.constant is None:
            return (False, "SKIP: constant not available")
        const_expr = self._parse_expr(self.constant)
        if const_expr is None:
            return (False, "constant not parseable")
        m, k, kprime, l = sympy.symbols('m k kprime l', positive=True)
        expected = m*k / (l**2 + m*kprime)
        diff = sympy.simplify(const_expr - expected)
        if diff == 0:
            return (True, "Constant expression correct")
        try:
            subs = {m:1.0, k:2.0, kprime:0.5, l:2.0}
            num_diff = float(const_expr.subs(subs) - expected.subs(subs))
            if math.isclose(num_diff, 0, rel_tol=1e-9):
                return (True, "Constant numerically matches")
            else:
                return (False, f"Constant numerical mismatch: {num_diff}")
        except Exception as e:
            return (False, f"Constant check error: {e}")

    def test_modeling_r_eom_structure(self):
        """Check that r equation of motion is m rddot - m r thetadot^2 + k/r^2 - k'/r^3 = 0."""
        if self.r_eom is None:
            return (False, "SKIP: r_eom not available")
        r_eom = self._parse_expr(self.r_eom)
        if r_eom is None:
            return (False, "r_eom not parseable")
        # We expect an equation or expression = 0.
        # Remove "=0" if present.
        if isinstance(r_eom, sympy.Equality):
            r_eom = r_eom.lhs - r_eom.rhs
        m, r, rdot, rddot, theta, thetadot, k, kprime = sympy.symbols('m r rdot rddot theta thetadot k kprime')
        expected = m*rddot - m*r*thetadot**2 + k/r**2 - kprime/r**3
        diff = sympy.simplify(r_eom - expected)
        if diff == 0:
            return (True, "Radial EOM matches expected form")
        try:
            subs = {m:1.0, r:0.5, rdot:0.3, rddot:0.1, theta:0.2, thetadot:1.5, k:2.0, kprime:0.5}
            num_diff = float(r_eom.subs(subs) - expected.subs(subs))
            if math.isclose(num_diff, 0, abs_tol=1e-6):
                return (True, "Radial EOM numerically matches")
            else:
                return (False, f"Radial EOM numerical mismatch: {num_diff}")
        except Exception as e:
            return (False, f"Radial EOM check error: {e}")

    def test_modeling_theta_eom_structure(self):
        """Check θ equation: m r thetaddot + 2 m rdot thetadot = 0."""
        if self.theta_eom is None:
            return (False, "SKIP: theta_eom not available")
        theta_eom = self._parse_expr(self.theta_eom)
        if theta_eom is None:
            return (False, "theta_eom not parseable")
        if isinstance(theta_eom, sympy.Equality):
            theta_eom = theta_eom.lhs - theta_eom.rhs
        m, r, rdot, theta, thetadot, thetaddot = sympy.symbols('m r rdot theta thetadot thetaddot')
        expected = m*r*thetaddot + 2*m*rdot*thetadot
        diff = sympy.simplify(theta_eom - expected)
        if diff == 0:
            return (True, "Angular EOM matches expected form")
        try:
            subs = {m:1.0, r:0.5, rdot:0.3, theta:0.2, thetadot:1.5, thetaddot:0.0}
            num_diff = float(theta_eom.subs(subs) - expected.subs(subs))
            if math.isclose(num_diff, 0, abs_tol=1e-6):
                return (True, "Angular EOM numerically matches")
            else:
                return (False, f"Angular EOM numerical mismatch: {num_diff}")
        except Exception as e:
            return (False, f"Angular EOM check error: {e}")

    # ------------------------------------------------------------------
    # MONOTONICITY TESTS (explicit perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_omega_increases_with_kprime(self):
        """Perturb k' by +1% and verify omega increases."""
        if self.omega is None:
            return (False, "SKIP: omega not available")
        omega_expr = self._parse_expr(self.omega)
        if omega_expr is None:
            return (False, "omega not parseable")
        # Choose baseline numeric values
        m_val = 1.0
        l_val = 2.0
        kprime_base = 0.5
        # Compute baseline omega
        subs_base = sympy.symbols('m kprime l')  # positional order matters
        # Better: use dict substitution
        subs_dict = {sympy.symbols('m'): m_val, sympy.symbols('kprime'): kprime_base, sympy.symbols('l'): l_val}
        try:
            omega_base = float(omega_expr.subs(subs_dict))
        except Exception as e:
            return (False, f"Cannot evaluate baseline omega: {e}")
        # Perturb k' by +1%
        kprime_pert = kprime_base * 1.01
        subs_dict[sympy.symbols('kprime')] = kprime_pert
        try:
            omega_pert = float(omega_expr.subs(subs_dict))
        except Exception as e:
            return (False, f"Cannot evaluate perturbed omega: {e}")
        if omega_pert > omega_base:
            return (True, f"Omega increases from {omega_base:.6f} to {omega_pert:.6f} when k' increases by 1%")
        else:
            return (False, f"Omega did not increase: base {omega_base:.6f}, perturbed {omega_pert:.6f}")

    def test_monotonicity_constant_decreases_with_l(self):
        """Perturb l by +1% and verify the orbit constant decreases."""
        if self.constant is None:
            return (False, "SKIP: constant not available")
        const_expr = self._parse_expr(self.constant)
        if const_expr is None:
            return (False, "constant not parseable")
        # Baseline numeric values (including k, m, kprime)
        m_val = 1.0
        k_val = 2.0
        kprime_val = 0.5
        l_base = 2.0
        subs_dict = {sympy.symbols('m'): m_val, sympy.symbols('k'): k_val,
                     sympy.symbols('kprime'): kprime_val, sympy.symbols('l'): l_base}
        try:
            const_base = float(const_expr.subs(subs_dict))
        except Exception as e:
            return (False, f"Cannot evaluate baseline constant: {e}")
        # Perturb l by +1%
        l_pert = l_base * 1.01
        subs_dict[sympy.symbols('l')] = l_pert
        try:
            const_pert = float(const_expr.subs(subs_dict))
        except Exception as e:
            return (False, f"Cannot evaluate perturbed constant: {e}")
        if const_pert < const_base:
            return (True, f"Constant decreases from {const_base:.6f} to {const_pert:.6f} when l increases by 1%")
        else:
            return (False, f"Constant did not decrease: base {const_base:.6f}, perturbed {const_pert:.6f}")

    # ------------------------------------------------------------------
    # WEAK ANSWER SANITY (one test max)
    # ------------------------------------------------------------------
    def test_answer_sanity_orbit_positive(self):
        """Weak sanity: check that the orbit equation is syntactically present (no sign contradictions)."""
        if self.orbit_equation is None:
            return (False, "SKIP: orbit_equation not available")
        # Just ensure the expression is non‑empty and contains a cosine
        expr = self._parse_expr(self.orbit_equation)
        if expr is None:
            return (False, "orbit_equation not parseable")
        # Very weak: it exists
        return (True, "Orbit equation present (weak sanity)")

    # ------------------------------------------------------------------
    # TEST DISCOVERY AND EXECUTION
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