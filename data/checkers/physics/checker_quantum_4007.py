import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.results = getattr(solution_instance, 'results', {})
        self.givens = getattr(solution_instance, 'givens', {}) or {}
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', [])

        # Emitted results (strings)
        self.energy_str = self.results.get('energy', None)
        self.degeneracy_str = self.results.get('degeneracy_per_area', None)
        self.wavefunction_str = self.results.get('wavefunction', None)

        # Givens (safe None defaults)
        self.L = self.givens.get('L', None)
        self.H0 = self.givens.get('H0', None)
        self.m = self.givens.get('m', None)
        self.e = self.givens.get('e', None)
        self.hbar = self.givens.get('hbar', None)
        self.c = self.givens.get('c', None)

        # Common SymPy symbols for parsing (no numeric defaults)
        self.symbols = {
            'hbar': sp.Symbol('hbar', positive=True),
            'm': sp.Symbol('m', positive=True),
            'e': sp.Symbol('e', real=True),           # real, can be negative
            'c': sp.Symbol('c', positive=True),
            'H0': sp.Symbol('H0', positive=True),
            'h': sp.Symbol('h', positive=True),
            'n': sp.Symbol('n', integer=True, nonnegative=True),
            'k_y': sp.Symbol('k_y', real=True),
            'k_z': sp.Symbol('k_z', real=True),
            'x': sp.Symbol('x'),
            'y': sp.Symbol('y'),
            'z': sp.Symbol('z'),
            'x0': sp.Symbol('x0', real=True),
            'C_n': sp.Symbol('C_n'),
        }

    def _parse(self, expr_str):
        """Safely parse a string into a SymPy expression."""
        if expr_str is None:
            return None
        try:
            return sp.sympify(expr_str, locals=self.symbols)
        except Exception:
            return None

    # ---------- Interface tests ----------
    def test_interface_energy(self):
        if self.energy_str is None:
            return (False, "SKIP: energy key missing or None")
        return (True, "energy key present")

    def test_interface_degeneracy(self):
        if self.degeneracy_str is None:
            return (False, "SKIP: degeneracy_per_area key missing or None")
        return (True, "degeneracy_per_area key present")

    def test_interface_wavefunction(self):
        if self.wavefunction_str is None:
            return (False, "SKIP: wavefunction key missing or None")
        return (True, "wavefunction key present")

    # ---------- Invariant tests (theorem-based structural sanity) ----------
    def test_invariant_energy_form(self):
        if self.energy_str is None:
            return (False, "SKIP: no energy string")
        expr = self._parse(self.energy_str)
        if expr is None:
            return (False, "energy string cannot be parsed")
        # Expected: hbar^2 * k_z^2/(2m) + (n+1/2) * hbar * e * H0 / m
        hbar, m, e, H0, n, k_z = [self.symbols[s] for s in ['hbar','m','e','H0','n','k_z']]
        expected = hbar**2 * k_z**2/(2*m) + (n + sp.Rational(1,2)) * hbar * e * H0 / m
        if sp.simplify(expr - expected) == 0:
            return (True, "energy matches Landau level formula")
        else:
            return (False, f"energy expression does not match expected form: {expr}")

    def test_invariant_degeneracy_form(self):
        if self.degeneracy_str is None:
            return (False, "SKIP: no degeneracy string")
        expr = self._parse(self.degeneracy_str)
        if expr is None:
            return (False, "degeneracy string cannot be parsed")
        e, H0, h = [self.symbols[s] for s in ['e','H0','h']]
        expected = e * H0 / h
        if sp.simplify(expr - expected) == 0:
            return (True, "degeneracy_per_area matches e*H0/h")
        else:
            return (False, f"degeneracy_per_area expression does not match e*H0/h: {expr}")

    def test_invariant_wavefunction_form(self):
        if self.wavefunction_str is None:
            return (False, "SKIP: no wavefunction string")
        expr = self._parse(self.wavefunction_str)
        if expr is None:
            return (False, "wavefunction string cannot be parsed")
        i = sp.I
        k_y, k_z, y, z = [self.symbols[s] for s in ['k_y','k_z','y','z']]
        x, x0, n = [self.symbols[s] for s in ['x','x0','n']]
        e, H0, hbar, c = [self.symbols[s] for s in ['e','H0','hbar','c']]
        plane_factor = sp.exp(i*(k_y*y + k_z*z))
        gaussian_factor = sp.exp( e*H0/(2*hbar*c) * (x - x0)**2 )
        missing = []
        if not expr.has(plane_factor):
            missing.append("plane-wave factor exp(i(k_y y + k_z z))")
        if not expr.has(gaussian_factor):
            missing.append("Gaussian factor exp(eH0/(2 hbar c)*(x-x0)^2)")
        if not expr.has(sp.hermite):
            missing.append("Hermite polynomial")
        if missing:
            return (False, f"Wavefunction missing: {', '.join(missing)}")
        return (True, "Wavefunction contains plane-wave, Gaussian, and Hermite factors")

    # ---------- Modeling tests (specific feature checks) ----------
    def test_modeling_energy_landau_level_structure(self):
        """Energy formula explicitly contains (n+1/2) factor."""
        if self.energy_str is None:
            return (False, "SKIP: no energy string")
        expr = self._parse(self.energy_str)
        if expr is None:
            return (False, "cannot parse energy expression")
        n = self.symbols['n']
        coeff_n = sp.diff(expr, n)
        if coeff_n == 0:
            return (False, "energy does not depend on Landau level index n")
        hbar, e, H0, m = [self.symbols[s] for s in ['hbar','e','H0','m']]
        expected_coeff = hbar*e*H0/m
        if sp.simplify(coeff_n - expected_coeff) == 0:
            return (True, "energy contains (n+1/2) factor with correct coefficient")
        else:
            return (False, f"coefficient of n is {coeff_n}, expected {expected_coeff}")

    def test_modeling_wavefunction_separation(self):
        """Wavefunction is separable as product of y- and z-plane waves times x-dependent part."""
        if self.wavefunction_str is None:
            return (False, "SKIP: no wavefunction string")
        expr = self._parse(self.wavefunction_str)
        if expr is None:
            return (False, "cannot parse wavefunction expression")
        i = sp.I
        k_y, k_z, y, z = [self.symbols[s] for s in ['k_y','k_z','y','z']]
        factor_y = sp.exp(i*k_y*y)
        factor_z = sp.exp(i*k_z*z)
        remainder = sp.simplify(expr / (factor_y * factor_z))
        if remainder.has(sp.Symbol('y')) or remainder.has(sp.Symbol('z')):
            return (False, "wavefunction cannot be separated into product of y- and z-dependent parts")
        free_syms = remainder.free_symbols
        x = sp.Symbol('x')
        allowed_params = {sp.Symbol('n'), sp.Symbol('x0'), sp.Symbol('C_n'),
                          sp.Symbol('hbar'), sp.Symbol('m'), sp.Symbol('e'),
                          sp.Symbol('H0'), sp.Symbol('c')}
        extra = free_syms - allowed_params - {x}
        if extra:
            return (False, f"remainder contains unexpected symbols: {extra}")
        return (True, "wavefunction is separable in y and z as plane waves")

    # ---------- Monotonicity tests (explicit input perturbation) ----------
    def test_monotonicity_energy_with_H0(self):
        """Energy decreases when H0 increases (electron charge e < 0)."""
        # Need numeric values for givens; if missing, skip.
        if self.energy_str is None:
            return (False, "SKIP: no energy string")
        if None in (self.hbar, self.m, self.e, self.H0):
            return (False, "SKIP: missing required givens (hbar, m, e, H0)")
        # Use given numeric values (e should be negative, e.g., -1.602e-19)
        hbar_val = float(self.hbar)
        m_val = float(self.m)
        e_val = float(self.e)
        if e_val >= 0:
            return (False, "SKIP: electron charge e is not negative; cannot test monotonicity")
        expr = self._parse(self.energy_str)
        if expr is None:
            return (False, "cannot parse energy expression")
        # Substitute n=0, k_z=0 for baseline
        subst = {
            self.symbols['hbar']: hbar_val,
            self.symbols['m']: m_val,
            self.symbols['e']: e_val,
            self.symbols['n']: 0,
            self.symbols['k_z']: 0
        }
        H0_sym = self.symbols['H0']
        H0_baseline = float(self.H0)
        delta = 0.1 * H0_baseline   # +10%
        H0_perturbed = H0_baseline + delta
        try:
            E_base = float(expr.subs(subst, {H0_sym: H0_baseline}).evalf())
            E_pert = float(expr.subs(subst, {H0_sym: H0_perturbed}).evalf())
        except Exception as e:
            return (False, f"Numeric evaluation failed: {e}")
        if E_pert < E_base:
            return (True, f"Energy decreases when H0 increases: baseline {E_base:.3e} -> perturbed {E_pert:.3e}")
        else:
            return (False, f"Energy does NOT decrease with H0: baseline {E_base:.3e}, perturbed {E_pert:.3e}")

    def test_monotonicity_degeneracy_with_H0(self):
        """Signed degeneracy becomes more negative when H0 increases (since e < 0)."""
        if self.degeneracy_str is None:
            return (False, "SKIP: no degeneracy string")
        if None in (self.e, self.H0):
            return (False, "SKIP: missing required givens (e, H0)")
        expr = self._parse(self.degeneracy_str)
        if expr is None:
            return (False, "cannot parse degeneracy expression")
        e_val = float(self.e)
        if e_val >= 0:
            return (False, "SKIP: electron charge e is not negative; cannot test monotonicity")
        # Use h=1 for relative check; the sign and linearity are what matter.
        h_val = 1.0
        subst = {
            self.symbols['e']: e_val,
            self.symbols['h']: h_val
        }
        H0_sym = self.symbols['H0']
        H0_baseline = float(self.H0)
        delta = 0.1 * H0_baseline
        H0_perturbed = H0_baseline + delta
        try:
            D_base = float(expr.subs(subst, {H0_sym: H0_baseline}).evalf())
            D_pert = float(expr.subs(subst, {H0_sym: H0_perturbed}).evalf())
        except Exception as e:
            return (False, f"Numeric evaluation failed: {e}")
        # Since e<0, D should decrease (become more negative)
        if D_pert < D_base:
            return (True, f"Degeneracy (signed) decreases when H0 increases: {D_base:.3e} -> {D_pert:.3e}")
        else:
            return (False, f"Degeneracy does not monotonically decrease: baseline {D_base:.3e}, perturbed {D_pert:.3e}")

    # ---------- Weak answer sanity test (at most one, no hardcoded target values) ----------
    def test_answer_sanity_weak(self):
        """Check that emitted strings are non-empty and parseable."""
        issues = []
        if self.energy_str is None:
            issues.append("energy string missing")
        elif self._parse(self.energy_str) is None:
            issues.append("energy string not parseable")
        if self.degeneracy_str is None:
            issues.append("degeneracy_per_area string missing")
        elif self._parse(self.degeneracy_str) is None:
            issues.append("degeneracy_per_area string not parseable")
        if self.wavefunction_str is None:
            issues.append("wavefunction string missing")
        elif self._parse(self.wavefunction_str) is None:
            issues.append("wavefunction string not parseable")
        if issues:
            return (False, f"Sanity issues: {', '.join(issues)}")
        return (True, "All three emitted strings are present and parseable")

    # ---------- run_all_tests ----------
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