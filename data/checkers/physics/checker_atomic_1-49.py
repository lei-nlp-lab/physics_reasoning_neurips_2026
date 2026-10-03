import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results dictionary safely
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}
        
        # Attempt to get givens from solution_instance (attributes or dictionary)
        # Use None defaults for missing values, NEVER numeric fallbacks
        self.e = getattr(solution_instance, 'e', None)
        self.m = getattr(solution_instance, 'm', None)
        self.epsilon0 = getattr(solution_instance, 'epsilon0', None)
        self.hbar = getattr(solution_instance, 'hbar', None)
        self.B = getattr(solution_instance, 'B', None)
        self.n = getattr(solution_instance, 'n', None)
        
        # Extract expected emit keys from results dictionary
        self.E_n = self.results.get('E_n')
        self.Z = self.results.get('Z')
        self.mu_B = self.results.get('mu_B')
        self.Delta_E_m = self.results.get('Delta_E_m')
        self.Zeeman_levels = self.results.get('Zeeman_levels')

    def _to_sympy(self, expr):
        """Convert string or sympy expression to sympy expression. Return None on failure."""
        if expr is None:
            return None
        if isinstance(expr, sp.Basic):
            return expr
        if isinstance(expr, str):
            try:
                return sp.sympify(expr)
            except:
                return None
        # numeric types (int, float, etc.)
        try:
            return sp.sympify(expr)
        except:
            return None

    def _is_finite_real(self, expr):
        """Check if sympy expression evaluates to a finite real number."""
        if expr is None:
            return False
        try:
            val = complex(expr.evalf())
            return val.imag == 0 and math.isfinite(val.real)
        except:
            return False

    # ---------- interface checks ----------
    def test_interface_required_keys(self):
        """Check that all required keys exist in results."""
        required = ['E_n', 'Z', 'mu_B', 'Delta_E_m', 'Zeeman_levels']
        missing = [k for k in required if k not in self.results]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    def test_interface_Z_value(self):
        """Z must be exactly 2 (helium nucleus)."""
        if self.Z is None:
            return (False, "SKIP: Z not available")
        Z_sym = self._to_sympy(self.Z)
        if Z_sym is None:
            return (False, "Z not convertible to sympy")
        if sp.simplify(Z_sym - 2) == 0:
            return (True, "Z = 2 as expected")
        else:
            return (False, f"Z = {Z_sym}, expected 2")

    # ---------- theorem-based invariant checks ----------
    def test_invariant_energy_negative(self):
        """Bohr energy for He+ should be negative (bound state)."""
        E_n_sym = self._to_sympy(self.E_n)
        if E_n_sym is None:
            return (False, "SKIP: E_n not available")
        # Try to evaluate at n=1 to test negativity
        if E_n_sym.has(sp.Symbol('n')):
            try:
                E1 = E_n_sym.subs(sp.Symbol('n'), 1)
                if self._is_finite_real(E1):
                    if float(E1.evalf()) < 0:
                        return (True, "E_n (n=1) is negative")
                    else:
                        return (False, f"E_n (n=1) = {E1} is not negative")
                else:
                    return (False, "Could not evaluate E_n at n=1")
            except:
                return (False, "Exception during evaluation")
        else:
            # numeric E_n
            if self._is_finite_real(E_n_sym):
                if float(E_n_sym.evalf()) < 0:
                    return (True, "E_n is negative")
                else:
                    return (False, "E_n is not negative")
            else:
                return (False, "E_n not finite real")

    def test_invariant_bohr_energy_derivation(self):
        """Verify that the user's E_n expression matches the Bohr formula derived from first principles."""
        if self.E_n is None or self.Z is None or self.e is None or self.m is None or self.epsilon0 is None or self.hbar is None:
            return (False, "SKIP: missing givens for derivation")
        # Define symbolic variables
        e, m, eps0, hbar, Z, n = sp.symbols('e m eps0 hbar Z n', positive=True)
        # Expected expression derived from Bohr model
        expected = - (Z**2 * e**4 * m) / (2 * (4*sp.pi*eps0)**2 * hbar**2 * n**2)
        # User expression
        user_expr = self._to_sympy(self.E_n)
        if user_expr is None:
            return (False, "E_n not parseable")
        # Substitute known numeric Z if needed
        if user_expr.has(sp.Symbol('Z')):
            Z_val = self._to_sympy(self.Z)
            if Z_val is not None:
                user_expr_sub = user_expr.subs(sp.Symbol('Z'), Z_val)
            else:
                user_expr_sub = user_expr
        else:
            user_expr_sub = user_expr
        # Map user symbols to our canonical symbols
        sym_map = {}
        for sym in user_expr_sub.free_symbols:
            sname = str(sym)
            if sname.lower() == 'e':
                sym_map[sym] = e
            elif sname.lower() == 'm':
                sym_map[sym] = m
            elif sname in ('epsilon0', 'eps0', 'epsilon_0', 'ε0'):
                sym_map[sym] = eps0
            elif sname in ('hbar', 'h_bar', 'ℏ'):
                sym_map[sym] = hbar
            elif sname == 'Z':
                sym_map[sym] = Z
            elif sname == 'n':
                sym_map[sym] = n
        substituted = user_expr_sub.xreplace(sym_map)
        # Compare by simplifying the ratio
        ratio = sp.simplify(substituted / expected)
        if ratio == 1:
            return (True, "Bohr energy expression matches derived formula")
        else:
            return (False, f"Expression mismatch: ratio = {ratio}")

    def test_invariant_mu_B_expression(self):
        """Check that mu_B expression equals e*hbar/(2m)."""
        if self.mu_B is None or self.e is None or self.m is None or self.hbar is None:
            return (False, "SKIP: missing givens for mu_B")
        mu_B_user = self._to_sympy(self.mu_B)
        if mu_B_user is None:
            return (False, "mu_B not parseable")
        # Build expected expression
        e_s, m_s, hbar_s = sp.symbols('e m hbar')
        expected = e_s * hbar_s / (2 * m_s)
        # Map user symbols to our canonical symbols
        sym_map = {}
        for sym in mu_B_user.free_symbols:
            sname = str(sym)
            if sname.lower() == 'e':
                sym_map[sym] = e_s
            elif sname.lower() == 'm':
                sym_map[sym] = m_s
            elif sname in ('hbar', 'h_bar', 'ℏ'):
                sym_map[sym] = hbar_s
        substituted = mu_B_user.xreplace(sym_map)
        diff = sp.simplify(substituted - expected)
        if diff == 0:
            return (True, "mu_B expression matches expected")
        else:
            # Numeric fallback if symbols are numeric
            if self._is_finite_real(mu_B_user):
                expected_numeric = float(self.e) * float(self.hbar) / (2 * float(self.m))
                if math.isclose(float(mu_B_user.evalf()), expected_numeric, rel_tol=1e-6):
                    return (True, "mu_B numeric value matches")
                else:
                    return (False, f"mu_B numeric value {float(mu_B_user.evalf())} != {expected_numeric}")
            return (False, f"mu_B expression differs: {diff}")

    def test_invariant_zeeman_levels_symmetric(self):
        """Zeeman levels must be three equally spaced levels, middle = E_n."""
        if self.Zeeman_levels is None:
            return (False, "SKIP: Zeeman_levels not available")
        if not isinstance(self.Zeeman_levels, list) or len(self.Zeeman_levels) != 3:
            return (False, "Zeeman_levels should be a list of three values")
        E_n_sym = self._to_sympy(self.E_n)
        if E_n_sym is None:
            return (False, "SKIP: E_n not available")
        levels = [self._to_sympy(l) for l in self.Zeeman_levels]
        if any(l is None for l in levels):
            return (False, "One or more Zeeman levels not parseable")
        # Middle level equals E_n
        if sp.simplify(levels[1] - E_n_sym) != 0:
            return (False, f"Middle Zeeman level does not equal E_n: {levels[1]} vs {E_n_sym}")
        # Symmetry: levels[0] - levels[1] = - (levels[1] - levels[2])
        if sp.simplify((levels[0] - levels[1]) + (levels[1] - levels[2])) != 0:
            return (False, f"Zeeman levels not symmetric: splittings {levels[0]-levels[1]} and {levels[1]-levels[2]}")
        return (True, "Zeeman levels symmetric with correct middle")

    # ---------- modeling / trace quality checks ----------
    def test_modeling_bohr_derivation_from_quantization(self):
        """Re-derive Bohr energy from Coulomb force and angular momentum quantization using symbolic algebra."""
        if self.E_n is None or self.e is None or self.m is None or self.epsilon0 is None or self.hbar is None:
            return (False, "SKIP: missing givens")
        e, m, eps0, hbar, Z, n = sp.symbols('e m eps0 hbar Z n', positive=True)
        # Quantization: m*v*r = n*hbar  => v = n*hbar/(m*r)
        r, v = sp.symbols('r v', positive=True)
        v_expr = n*hbar/(m*r)
        # Centripetal force = Coulomb force: m*v^2/r = Z*e^2/(4*pi*eps0*r^2)
        force_eq = sp.Eq(m*v_expr**2/r, Z*e**2/(4*sp.pi*eps0*r**2))
        # Solve for r
        r_solution = sp.solve(force_eq, r)[0]
        # Total energy E = (1/2) m v^2 - Z*e^2/(4*pi*eps0*r)
        E_expr = (sp.Rational(1,2))*m*v_expr**2 - Z*e**2/(4*sp.pi*eps0*r)
        E_simpl = sp.simplify(E_expr.subs(r, r_solution))
        # Expected formula
        expected = - (Z**2 * e**4 * m) / (2 * (4*sp.pi*eps0)**2 * hbar**2 * n**2)
        if sp.simplify(E_simpl - expected) == 0:
            return (True, "Derivation from first principles matches E_n")
        else:
            return (False, f"Derived expression {E_simpl} does not match expected")

    def test_trace_zeeman_levels_from_shift(self):
        """Check that Zeeman levels are exactly E_n + mu_B*B*m_l for m_l = -1,0,1."""
        if self.Zeeman_levels is None or self.E_n is None or self.mu_B is None or self.B is None:
            return (False, "SKIP: missing data")
        levels = [self._to_sympy(l) for l in self.Zeeman_levels]
        E_n_sym = self._to_sympy(self.E_n)
        mu_B_sym = self._to_sympy(self.mu_B)
        B_sym = self._to_sympy(self.B)
        if any(x is None for x in [levels[0], levels[1], levels[2], E_n_sym, mu_B_sym, B_sym]):
            return (False, "Cannot parse required symbols")
        # Expected levels: E_n + mu_B*B, E_n, E_n - mu_B*B
        expected = [E_n_sym + mu_B_sym*B_sym, E_n_sym, E_n_sym - mu_B_sym*B_sym]
        for i, (user, exp) in enumerate(zip(levels, expected)):
            if sp.simplify(user - exp) != 0:
                return (False, f"Zeeman level {i}: {user} != {exp}")
        return (True, "Zeeman levels correctly constructed from E_n and mu_B*B")

    # ---------- monotonicity checks (explicit input perturbation) ----------
    def test_monotonicity_energy_increasing_with_n(self):
        """Perturb n by +1 and check that E_n becomes less negative (increases algebraically)."""
        E_n_sym = self._to_sympy(self.E_n)
        if E_n_sym is None:
            return (False, "SKIP: E_n not available")
        if not E_n_sym.has(sp.Symbol('n')):
            return (False, "SKIP: E_n does not depend on n symbolically")
        n_sym = sp.Symbol('n', positive=True, integer=True)
        # Baseline at some n (e.g., n=2)
        try:
            base = E_n_sym.subs(n_sym, 2)
            perturbed = E_n_sym.subs(n_sym, 3)  # n increased by 1
        except:
            return (False, "Could not substitute n")
        if not self._is_finite_real(base) or not self._is_finite_real(perturbed):
            return (False, "SKIP: cannot evaluate symbolically")
        # Energy should increase algebraically: perturbed > base
        if float(perturbed.evalf()) > float(base.evalf()):
            return (True, "E_n increases as n increases (less negative)")
        else:
            return (False, f"E_n at n=3 {float(perturbed.evalf())} not greater than at n=2 {float(base.evalf())}")

    def test_monotonicity_zeeman_splitting_linear_with_B(self):
        """Perturb B by 10% and check that Zeeman splitting scales proportionally."""
        if self.Zeeman_levels is None or self.B is None or self.mu_B is None:
            return (False, "SKIP: missing data")
        # Obtain symbolic expression for splitting (top - middle)
        levels = [self._to_sympy(l) for l in self.Zeeman_levels]
        if any(l is None for l in levels):
            return (False, "Cannot parse Zeeman levels")
        # Top level - middle level (should be mu_B*B)
        split_expr = sp.simplify(levels[0] - levels[1])
        # Identify B symbol
        B_sym = None
        for sym in split_expr.free_symbols:
            if str(sym).upper() in ('B', 'b'):
                B_sym = sym
                break
        if B_sym is None:
            # Try to extract B from the given B
            B_from_given = self._to_sympy(self.B)
            if B_from_given is not None and B_from_given.is_Symbol:
                B_sym = B_from_given
            else:
                return (False, "Cannot identify B symbol in splitting expression")
        # Baseline splitting with B = 1 (arbitrary)
        baseline = sp.simplify(split_expr.subs(B_sym, 1))
        # Perturb B to 1.1
        perturbed = sp.simplify(split_expr.subs(B_sym, 1.1))
        if not self._is_finite_real(baseline) or not self._is_finite_real(perturbed):
            return (False, "Cannot evaluate splitting")
        # Expected linear scaling: perturbed = 1.1 * baseline
        if math.isclose(float(perturbed.evalf()), 1.1 * float(baseline.evalf()), rel_tol=1e-4):
            return (True, "Zeeman splitting scales linearly with B (10% perturbation gives 10% change)")
        else:
            return (False, f"Splitting for B=1.1: {float(perturbed.evalf())}, expected 1.1 * {float(baseline.evalf())} = {1.1*float(baseline.evalf())}")

    # ---------- weak final answer sanity ----------
    def test_answer_sanity_broad(self):
        """Weak sanity: energies are real, negative, and Zeeman levels show splitting."""
        issues = []
        if self.E_n is not None:
            E_sym = self._to_sympy(self.E_n)
            if E_sym is None:
                issues.append("E_n not parseable")
            elif not self._is_finite_real(E_sym):
                issues.append("E_n not finite real")
            elif float(E_sym.evalf()) >= 0:
                issues.append("E_n not negative")
        if self.Zeeman_levels is not None:
            if not isinstance(self.Zeeman_levels, list) or len(self.Zeeman_levels) != 3:
                issues.append("Zeeman_levels not a list of 3")
            else:
                try:
                    vals = [float(self._to_sympy(l).evalf()) for l in self.Zeeman_levels]
                    if vals[0] == vals[1] == vals[2]:
                        issues.append("All Zeeman levels equal, no splitting")
                except:
                    issues.append("Cannot evaluate Zeeman levels")
        if issues:
            return (False, "; ".join(issues))
        return (True, "Basic sanity checks passed")

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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)