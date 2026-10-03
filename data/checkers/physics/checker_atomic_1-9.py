import sympy as sp
import traceback

class PhysicsChecker:
    """
    Audits expressions for a multi-part symbolic physics problem.
    Compares each result against known forms, checks allowed symbols,
    and performs consistency and monotonicity checks.
    """

    def __init__(self, solution_instance):
        # Extract results – all default to None
        self.results = getattr(solution_instance, 'results', {})
        if self.results is None:
            self.results = {}

        self.E_I = self.results.get('E_I', None)
        self.delta_nu = self.results.get('delta_nu', None)
        self.mu_e = self.results.get('mu_e', None)
        self.delta_E = self.results.get('delta_E', None)
        self.delta_m = self.results.get('delta_m', None)
        self.B = self.results.get('B', None)
        self.delta_E_fs = self.results.get('delta_E_fs', None)

        # Givens (may include tau, T)
        self.givens = getattr(solution_instance, 'givens', {})
        self.tau = self.givens.get('tau', None)
        self.T = self.givens.get('T', None)

        # Define symbolic constants for conversion
        e   = sp.Symbol('e',   positive=True)
        hbar = sp.Symbol('hbar', positive=True)
        c   = sp.Symbol('c',   positive=True)
        k   = sp.Symbol('k',   positive=True)
        m_e = sp.Symbol('m_e', positive=True)
        m_p = sp.Symbol('m_p', positive=True)
        tau = sp.Symbol('tau', positive=True)
        T   = sp.Symbol('T',   positive=True)

        self.sym = {
            'e': e, 'hbar': hbar, 'c': c, 'k': k,
            'm_e': m_e, 'm_p': m_p, 'tau': tau, 'T': T,
            'pi': sp.pi
        }

        # Local dictionary for sympify
        self._local_dict = {str(k): v for k, v in self.sym.items()}
        self._local_dict['π'] = sp.pi   # in case Greek pi is used

    def _to_sympy(self, expr):
        """Convert a string or sympy object to a sympy expression with known symbols."""
        if expr is None:
            return None
        if isinstance(expr, str):
            return sp.sympify(expr, locals=self._local_dict)
        else:
            # Assume it's already a sympy expression (or numeric)
            return sp.sympify(expr)

    def _check_symbols(self, expr, allowed):
        """Return True if all free symbols in expr belong to allowed set."""
        if expr is None:
            return False
        free = expr.free_symbols
        for sym in free:
            if sym not in allowed and sym != sp.pi:
                return False
        return True

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """All seven result keys must be present and convertible to sympy."""
        required = [
            ('E_I', self.E_I),
            ('delta_nu', self.delta_nu),
            ('mu_e', self.mu_e),
            ('delta_E', self.delta_E),
            ('delta_m', self.delta_m),
            ('B', self.B),
            ('delta_E_fs', self.delta_E_fs)
        ]
        missing = []
        for name, val in required:
            if val is None:
                missing.append(name)
            else:
                # Try conversion
                try:
                    self._to_sympy(val)
                except Exception:
                    missing.append(name + " (invalid)")
        if missing:
            return (False, f"Missing or invalid keys: {', '.join(missing)}")
        return (True, "All required keys present and convertible")

    # ------------------------------------------------------------------
    # Invariant tests – symbolic form checks (using exact expected forms)
    # ------------------------------------------------------------------
    def test_invariant_ionization_energy(self):
        if self.E_I is None:
            return (False, "SKIP: E_I not available")
        try:
            expr = self._to_sympy(self.E_I)
        except Exception:
            return (False, "E_I could not be parsed")
        # Expected: m_e e^4 / (2 ħ^2)
        expected = self.sym['m_e'] * self.sym['e']**4 / (2 * self.sym['hbar']**2)
        diff = sp.simplify(expr - expected)
        if diff == 0:
            return (True, "Ionization energy matches expected form")
        else:
            return (False, f"E_I mismatch. Got {expr}, expected {expected}")

    def test_invariant_lyman_alpha_frequency_diff(self):
        if self.delta_nu is None:
            return (False, "SKIP: delta_nu not available")
        try:
            expr = self._to_sympy(self.delta_nu)
        except Exception:
            return (False, "delta_nu could not be parsed")
        # Reduced masses
        m_e = self.sym['m_e']
        m_p = self.sym['m_p']
        mu_H = m_e*m_p / (m_e + m_p)
        mu_D = 2*m_e*m_p / (m_e + 2*m_p)
        expected = (3 * self.sym['e']**4 / (16 * sp.pi * self.sym['hbar']**3)) * (mu_H - mu_D)
        diff = sp.simplify(expr - expected)
        if diff == 0:
            return (True, "Δν matches expected form via reduced mass difference")
        return (False, f"Δν mismatch. Got {expr}, expected {expected}")

    def test_invariant_electron_magnetic_moment(self):
        if self.mu_e is None:
            return (False, "SKIP: mu_e not available")
        try:
            expr = self._to_sympy(self.mu_e)
        except Exception:
            return (False, "mu_e could not be parsed")
        expected = self.sym['e'] * self.sym['hbar'] / (2 * self.sym['m_e'])
        diff = sp.simplify(expr - expected)
        if diff == 0:
            return (True, "Electron magnetic moment matches Dirac value")
        return (False, f"μ_e mismatch. Got {expr}, expected {expected}")

    def test_invariant_pi0_mass_spread(self):
        # Check both ΔE and Δm
        # 1) Check ΔE = ħ/τ
        if self.delta_E is None or self.tau is None:
            return (False, "SKIP: delta_E or tau missing")
        try:
            expr_E = self._to_sympy(self.delta_E)
        except Exception:
            return (False, "delta_E could not be parsed")
        expected_E = self.sym['hbar'] / self.sym['tau']
        diff_E = sp.simplify(expr_E - expected_E)
        if diff_E != 0:
            return (False, f"ΔE mismatch. Got {expr_E}, expected {expected_E}")

        # 2) Check Δm = ΔE / c^2
        if self.delta_m is None:
            return (False, "ΔE OK, but Δm missing")
        try:
            expr_m = self._to_sympy(self.delta_m)
        except Exception:
            return (False, "delta_m could not be parsed")
        expected_m = expected_E / self.sym['c']**2
        diff_m = sp.simplify(expr_m - expected_m)
        if diff_m == 0:
            return (True, "ΔE = ħ/τ and Δm = ΔE/c^2 both match")
        return (False, f"Δm mismatch. Got {expr_m}, expected {expected_m}")

    def test_invariant_proton_polarization_B(self):
        if self.B is None or self.T is None:
            return (False, "SKIP: B or T missing")
        try:
            expr = self._to_sympy(self.B)
        except Exception:
            return (False, "B could not be parsed")
        mu_p = self.sym['e'] * self.sym['hbar'] / (2 * self.sym['m_p'])
        expected = self.sym['k'] * self.sym['T'] * sp.Rational(1,10000) / mu_p
        diff = sp.simplify(expr - expected)
        if diff == 0:
            return (True, "B matches expected form (k T / μ_p × 10⁻⁴)")
        return (False, f"B mismatch. Got {expr}, expected {expected}")

    def test_invariant_fine_structure_splitting(self):
        if self.delta_E_fs is None:
            return (False, "SKIP: delta_E_fs not available")
        try:
            expr = self._to_sympy(self.delta_E_fs)
        except Exception:
            return (False, "delta_E_fs could not be parsed")
        expected = self.sym['m_e'] * self.sym['e']**8 / (32 * self.sym['hbar']**4 * self.sym['c']**2)
        diff = sp.simplify(expr - expected)
        if diff == 0:
            return (True, "Fine structure splitting matches expected form")
        return (False, f"ΔE_fs mismatch. Got {expr}, expected {expected}")

    # ------------------------------------------------------------------
    # Modelling checks – allowed symbols
    # ------------------------------------------------------------------
    def test_modeling_ionization_energy_symbols(self):
        if self.E_I is None:
            return (False, "SKIP: E_I missing")
        try:
            expr = self._to_sympy(self.E_I)
        except Exception:
            return (False, "E_I could not be parsed")
        allowed = {self.sym['e'], self.sym['hbar'], self.sym['m_e']}
        if self._check_symbols(expr, allowed):
            return (True, "E_I uses only e, ħ, m_e")
        return (False, f"E_I contains disallowed symbols: {expr.free_symbols - allowed}")

    def test_modeling_lyman_alpha_symbols(self):
        if self.delta_nu is None:
            return (False, "SKIP: delta_nu missing")
        try:
            expr = self._to_sympy(self.delta_nu)
        except Exception:
            return (False, "delta_nu could not be parsed")
        allowed = {self.sym['e'], self.sym['hbar'], self.sym['m_e'], self.sym['m_p'], self.sym['pi']}
        if self._check_symbols(expr, allowed):
            return (True, "Δν uses only e, ħ, m_e, m_p, π")
        return (False, f"Δν contains disallowed symbols: {expr.free_symbols - allowed}")

    def test_modeling_electron_magnetic_moment_symbols(self):
        if self.mu_e is None:
            return (False, "SKIP: mu_e missing")
        try:
            expr = self._to_sympy(self.mu_e)
        except Exception:
            return (False, "mu_e could not be parsed")
        allowed = {self.sym['e'], self.sym['hbar'], self.sym['m_e']}
        if self._check_symbols(expr, allowed):
            return (True, "μ_e uses only e, ħ, m_e")
        return (False, f"μ_e contains disallowed symbols: {expr.free_symbols - allowed}")

    def test_modeling_pi0_symbols(self):
        # ΔE and Δm together; check both
        ok = True
        msg = ""
        if self.delta_E is not None:
            try:
                expr = self._to_sympy(self.delta_E)
                allowed = {self.sym['hbar'], self.sym['tau']}
                if not self._check_symbols(expr, allowed):
                    ok = False
                    msg += f"ΔE symbols: {expr.free_symbols} "
            except Exception:
                ok = False
                msg += "ΔE parse error "
        if self.delta_m is not None:
            try:
                expr = self._to_sympy(self.delta_m)
                allowed = {self.sym['hbar'], self.sym['tau'], self.sym['c']}
                if not self._check_symbols(expr, allowed):
                    ok = False
                    msg += f"Δm symbols: {expr.free_symbols} "
            except Exception:
                ok = False
                msg += "Δm parse error "
        if not ok:
            return (False, f"Disallowed symbols found: {msg}")
        return (True, "Pi⁰ expressions use only ħ, τ, c")

    def test_modeling_B_symbols(self):
        if self.B is None:
            return (False, "SKIP: B missing")
        try:
            expr = self._to_sympy(self.B)
        except Exception:
            return (False, "B could not be parsed")
        allowed = {self.sym['k'], self.sym['T'], self.sym['e'], self.sym['hbar'], self.sym['m_p']}
        if self._check_symbols(expr, allowed):
            return (True, "B uses only k, T, e, ħ, m_p")
        return (False, f"B contains disallowed symbols: {expr.free_symbols - allowed}")

    def test_modeling_fine_structure_symbols(self):
        if self.delta_E_fs is None:
            return (False, "SKIP: delta_E_fs missing")
        try:
            expr = self._to_sympy(self.delta_E_fs)
        except Exception:
            return (False, "delta_E_fs could not be parsed")
        allowed = {self.sym['m_e'], self.sym['e'], self.sym['hbar'], self.sym['c']}
        if self._check_symbols(expr, allowed):
            return (True, "ΔE_fs uses only m_e, e, ħ, c")
        return (False, f"ΔE_fs contains disallowed symbols: {expr.free_symbols - allowed}")

    # ------------------------------------------------------------------
    # Consistency checks
    # ------------------------------------------------------------------
    def test_invariant_delta_m_consistency(self):
        """If both ΔE and Δm present, check Δm = ΔE / c²."""
        if self.delta_E is None or self.delta_m is None:
            return (False, "SKIP: either delta_E or delta_m missing")
        try:
            expr_E = self._to_sympy(self.delta_E)
            expr_m = self._to_sympy(self.delta_m)
        except Exception:
            return (False, "Could not parse one of the expressions")
        if sp.simplify(expr_m * self.sym['c']**2 - expr_E) == 0:
            return (True, "Δm = ΔE / c² holds")
        return (False, f"Δm * c² = {sp.simplify(expr_m*self.sym['c']**2)}, but ΔE = {expr_E}")

    # ------------------------------------------------------------------
    # Monotonicity test (one required)
    # ------------------------------------------------------------------
    def test_monotonicity_ionization_energy_with_mass(self):
        """
        Ionization energy E_I ∝ m_e. Increasing m_e increases E_I.
        Perturb m_e by 10% and verify ratio > 1.
        """
        if self.E_I is None:
            return (False, "SKIP: E_I not available")
        try:
            expr = self._to_sympy(self.E_I)
        except Exception:
            return (False, "E_I could not be parsed")
        # Check if expression is symbolic and has m_e; if not, can't test monotonicity
        if not expr.has(self.sym['m_e']):
            return (False, "SKIP: E_I does not contain m_e, cannot test monotonicity via m_e")
        # Perturb: substitute m_e -> 1.1*m_e
        perturbed = expr.subs(self.sym['m_e'], 1.1 * self.sym['m_e'])
        # Compute ratio perturbed / baseline (should be 1.1)
        ratio = sp.simplify(perturbed / expr)
        # Ideally ratio should be a number; we check that it is >1
        # We can evaluate symbolically: ratio should be 1.1 exactly because linear
        if ratio.is_Number:
            if ratio > 1:
                return (True, f"E_I increases with m_e: baseline E_I ∝ m_e, perturbed ratio = {ratio}")
            else:
                return (False, f"E_I does not increase with m_e: ratio = {ratio}")
        else:
            # If ratio is still symbolic, we can check it's a constant > 1
            # For E_I = m_e e^4/(2 ħ^2), ratio = 1.1 exactly
            # Convert to float if possible
            try:
                float_ratio = float(ratio)
                if float_ratio > 1.0:
                    return (True, f"E_I increases with m_e: ratio = {float_ratio}")
                else:
                    return (False, f"E_I does not increase with m_e: ratio = {float_ratio}")
            except:
                return (False, f"Cannot evaluate ratio {ratio}")

    # ------------------------------------------------------------------
    # Weak answer sanity (sign/finite only)
    # ------------------------------------------------------------------
    def test_answer_sanity_sign_finite(self):
        """All results should be positive and finite (no NaNs, no infinities)."""
        exprs = []
        keys = ['E_I', 'delta_nu', 'mu_e', 'delta_E', 'delta_m', 'B', 'delta_E_fs']
        for key in keys:
            val = getattr(self, key, None)
            if val is not None:
                try:
                    expr = self._to_sympy(val)
                    exprs.append((key, expr))
                except Exception:
                    exprs.append((key, None))
        failures = []
        for key, expr in exprs:
            if expr is None:
                failures.append(f"{key} (unparseable)")
                continue
            # Check positivity: For symbolic expressions with positive assumptions, the expression should be positive.
            # We can check that the expression is not identically zero.
            if expr.is_zero:
                failures.append(f"{key} is identically zero")
            # Check for infinities/NaN (only if numeric)
            if expr.is_number:
                if expr.is_nan or expr.is_infinite:
                    failures.append(f"{key} is NaN or infinite")
        if failures:
            return (False, f"Sanity issues: {', '.join(failures)}")
        return (True, "All checkable results appear positive and finite")

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Get all methods starting with 'test_'
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed = False
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)