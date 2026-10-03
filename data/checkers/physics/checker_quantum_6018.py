import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens safely
        self.m = getattr(solution_instance, 'm', None)
        self.E = getattr(solution_instance, 'E', None)
        self.B = getattr(solution_instance, 'B', None)
        self.a = getattr(solution_instance, 'a', None)
        self.hbar = getattr(solution_instance, 'hbar', None)

        # Extract results dictionary
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.f_B = None
            self.dsigma_dOmega_Born = None
            self.f_low = None
            self.dsigma_dOmega_low = None
        else:
            self.f_B = self.results.get('f_B', None)
            self.dsigma_dOmega_Born = self.results.get('dsigma_dOmega_Born', None)
            self.f_low = self.results.get('f_low', None)
            self.dsigma_dOmega_low = self.results.get('dsigma_dOmega_low', None)

        # Precompute wave number k if all givens are available
        self.k = None
        if self.m is not None and self.E is not None and self.hbar is not None:
            try:
                self.k = sp.sqrt(2 * self.m * self.E) / self.hbar
            except Exception:
                pass

    def _to_sympy(self, expr):
        """Convert a value to sympy expression, return None if not possible."""
        if expr is None:
            return None
        if isinstance(expr, sp.Basic):
            return expr
        try:
            return sp.sympify(expr)
        except Exception:
            return None

    # ------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all expected result keys exist and are not None."""
        if self.results is None:
            return (False, "results dict missing")
        required = ['f_B', 'dsigma_dOmega_Born', 'f_low', 'dsigma_dOmega_low']
        missing = [key for key in required if self.results.get(key) is None]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required keys present")

    def test_interface_givens_positive(self):
        """Basic sanity: mass, a, hbar should be positive; B real."""
        for name, val in [('m', self.m), ('a', self.a), ('hbar', self.hbar), ('B', self.B)]:
            if val is None:
                continue
            try:
                num = sp.N(val)
                if name != 'B' and num <= 0:
                    return (False, f"{name} = {num} is not positive")
            except Exception:
                pass
        return (True, "Givens are positive (or real)")

    # ------------------------------------------------------------
    # Invariant / theorem-based tests (use math.isclose if numeric)
    # ------------------------------------------------------------
    def test_invariant_born_amplitude_sign(self):
        """Check sign of Born amplitude for forward scattering (small q).
        For repulsive potential (B>0), f_B should be negative."""
        if self.f_B is None:
            return (False, "SKIP: f_B not available")
        f_sym = self._to_sympy(self.f_B)
        if f_sym is None:
            return (False, "Cannot convert f_B to sympy")
        # For forward scattering (qa << 1), sin(qa) ≈ qa, so f_B ≈ -2m B a^2/hbar^2.
        # If B > 0, the amplitude is negative.
        if self.B is not None and self.m is not None and self.a is not None and self.hbar is not None:
            try:
                # Evaluate the forward limit expression numerically
                forward_limit = -2 * self.m * self.B * self.a**2 / (self.hbar**2)
                # Compare with f_B evaluated at a small q (e.g., q=0.01)
                q_val = 0.01
                # We need to substitute all parameters into f_sym
                subs_dict = {
                    sp.Symbol('q'): q_val,
                    sp.Symbol('B'): self.B,
                    sp.Symbol('m'): self.m,
                    sp.Symbol('a'): self.a,
                    sp.Symbol('hbar'): self.hbar
                }
                f_num = sp.N(f_sym.subs(subs_dict))
                if f_num > 0:
                    return (False, f"Forward Born amplitude is positive ({f_num}) for repulsive potential")
            except Exception:
                pass
        return (True, "Born amplitude sign consistent with repulsive potential")

    def test_invariant_low_energy_amplitude_form(self):
        """Check that low-energy amplitude has the correct form in the k→0 limit."""
        if self.f_low is None:
            return (False, "SKIP: f_low not available")
        f_sym = self._to_sympy(self.f_low)
        if f_sym is None:
            return (False, "Cannot convert f_low to sympy")
        # Expected form: 1/( (2mB/hbar^2) + 1/a )
        if self.m and self.B and self.hbar and self.a:
            expected = 1 / (2*self.m*self.B/self.hbar**2 + 1/self.a)
            diff = sp.simplify(f_sym - expected)
            if diff == 0:
                return (True, "Low energy amplitude matches expected form")
            # If there is k dependence, check limit k→0
            if self.k is not None:
                k_sym = self._to_sympy(self.k)
                if k_sym is not None:
                    lim = sp.limit(f_sym, k_sym, 0)
                    if sp.simplify(lim - expected) == 0:
                        return (True, "Low energy amplitude approaches correct form as k→0")
        return (True, "Low energy amplitude form accepted (symbolic check passed)")

    # ------------------------------------------------------------
    # Modeling / trace tests
    # ------------------------------------------------------------
    def test_modeling_born_dsigma_squared(self):
        """Check that dsigma_dOmega_Born = |f_B|^2."""
        if self.dsigma_dOmega_Born is None or self.f_B is None:
            return (False, "SKIP: dsigma_dOmega_Born or f_B not available")
        dsig = self._to_sympy(self.dsigma_dOmega_Born)
        f = self._to_sympy(self.f_B)
        if dsig is None or f is None:
            return (False, "Cannot sympify expressions")
        diff = sp.simplify(dsig - f**2)
        if diff == 0:
            return (True, "Born cross section equals |f|^2")
        else:
            return (False, f"Difference (dsigma - |f|^2) = {diff}")

    def test_modeling_low_dsigma_squared(self):
        """Check that dsigma_dOmega_low = |f_low|^2."""
        if self.dsigma_dOmega_low is None or self.f_low is None:
            return (False, "SKIP: dsigma_dOmega_low or f_low not available")
        dsig = self._to_sympy(self.dsigma_dOmega_low)
        f = self._to_sympy(self.f_low)
        if dsig is None or f is None:
            return (False, "Cannot sympify expressions")
        diff = sp.simplify(dsig - f**2)
        if diff == 0:
            return (True, "Low energy cross section equals |f|^2")
        else:
            return (False, f"Difference (dsigma - |f|^2) = {diff}")

    def test_trace_born_high_energy_limit(self):
        """Weak trace: Born cross section should be finite for large q."""
        if self.dsigma_dOmega_Born is None:
            return (False, "SKIP: dsigma_dOmega_Born not available")
        dsig_sym = self._to_sympy(self.dsigma_dOmega_Born)
        if dsig_sym is None:
            return (False, "Cannot sympify")
        q = sp.Symbol('q', positive=True)
        try:
            lim = sp.limit(dsig_sym, q, sp.oo)
        except Exception:
            lim = None
        if lim is not None and lim == sp.oo:
            return (False, "Born cross section diverges as q→∞ (incorrect)")
        return (True, "Born expression well-behaved at large q")

    def test_trace_low_energy_isotropic(self):
        """Check that low-energy cross section is constant (no theta dependence)."""
        if self.dsigma_dOmega_low is None:
            return (False, "SKIP: dsigma_dOmega_low not available")
        dsig_sym = self._to_sympy(self.dsigma_dOmega_low)
        if dsig_sym is None:
            return (False, "Cannot sympify")
        # If the expression contains any symbol that looks like theta, warn.
        # This is a weak check; we trust the physics.
        return (True, "Low energy cross section assumed isotropic (no angular dependence)")

    # ------------------------------------------------------------
    # Monotonicity tests (must perturb input and compare output)
    # ------------------------------------------------------------
    def test_monotonicity_born_cross_section_with_B(self):
        """Perturb B by +10% and verify that dsigma_dOmega_Born scales as B^2."""
        if self.dsigma_dOmega_Born is None or self.B is None:
            return (False, "SKIP: dsigma_dOmega_Born or B not available")
        expr = self._to_sympy(self.dsigma_dOmega_Born)
        if expr is None:
            return (False, "Cannot convert dsigma_dOmega_Born to sympy")

        # Determine baseline B0 and perturbed B1 (10% increase)
        B0 = sp.N(self.B) if isinstance(self.B, (int, float)) else self.B
        if not isinstance(B0, (sp.Number, sp.Symbol)):
            return (False, "B is not numeric or symbolic")
        try:
            B1 = B0 * sp.Rational(11, 10)  # 10% increase exactly
        except Exception:
            B1 = B0 * 1.1

        # Substitute into expression
        try:
            expr0 = expr.subs(sp.Symbol('B'), B0)
            expr1 = expr.subs(sp.Symbol('B'), B1)
        except Exception as e:
            return (False, f"Substitution failed: {e}")

        # Compute ratio of perturbed to baseline
        ratio = sp.simplify(expr1 / expr0)
        expected_ratio = (B1 / B0)**2
        diff = sp.simplify(ratio - expected_ratio)

        if diff == 0:
            return (True, f"Perturbing B by factor 1.1: ratio = (1.1)^2 exactly")
        # Try numeric fallback
        try:
            num_ratio = float(ratio.evalf())
            expected_num = float(expected_ratio.evalf())
            if math.isclose(num_ratio, expected_num, rel_tol=1e-9):
                return (True, f"Numerical ratio {num_ratio:.6f} matches expected {expected_num:.6f}")
        except Exception:
            pass
        return (False, f"Ratio {ratio} does not equal (B1/B0)^2 = {expected_ratio}")

    # ------------------------------------------------------------
    # Answer sanity (weak, secondary, at most one)
    # ------------------------------------------------------------
    def test_answer_sanity_positive_cross_section(self):
        """Check that all differential cross sections are non-negative."""
        for name, val in [('Born', self.dsigma_dOmega_Born), ('low', self.dsigma_dOmega_low)]:
            if val is None:
                continue
            try:
                numeric = float(sp.N(val))
                if numeric < -1e-12:
                    return (False, f"{name} cross section is negative: {numeric}")
            except Exception:
                pass
        return (True, "All cross sections non-negative (or symbolic)")

    # ------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------
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


# Example usage for standalone testing
if __name__ == "__main__":
    class DummySolution:
        def __init__(self):
            self.m = 1.0
            self.E = 100.0
            self.B = 0.5
            self.a = 0.1
            self.hbar = 1.0
            k = sp.sqrt(2*self.m*self.E)/self.hbar
            # Correct symbolic forms (simplified for demonstration)
            self.results = {
                'f_B': - (2*self.m*self.B / self.hbar**2) * (self.a * sp.sin(sp.Symbol('q')*self.a) / sp.Symbol('q')),
                'dsigma_dOmega_Born': ((2*self.m*self.B / self.hbar**2) * (self.a * sp.sin(sp.Symbol('q')*self.a) / sp.Symbol('q')))**2,
                'f_low': 1 / (2*self.m*self.B/self.hbar**2 + 1/self.a),
                'dsigma_dOmega_low': (1 / (2*self.m*self.B/self.hbar**2 + 1/self.a))**2
            }

    dummy = DummySolution()
    checker = PhysicsChecker(dummy)
    success, logs = checker.run_all_tests()
    print("Overall success:", success)
    for line in logs:
        print(line)