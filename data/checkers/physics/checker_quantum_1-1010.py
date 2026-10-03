import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Physics Checker for the free particle Gaussian wavepacket problem.
    Checks interface completeness, physics invariants (normalization, Fourier relations),
    modeling quality, monotonicity, and a weak answer sanity.
    """

    def __init__(self, solution_instance):
        """
        Extract inputs and results safely, defaulting to None if missing.
        """
        # Access givens; expected keys: 'm', 'hbar', 'a', 'A' (optional)
        self.givens = getattr(solution_instance, 'givens', {})
        if self.givens is None:
            self.givens = {}
        self.m = self.givens.get('m', None)
        self.hbar = self.givens.get('hbar', None)
        self.a = self.givens.get('a', None)
        self.A = self.givens.get('A', None)  # may be derived by user, but we can recompute

        # Access results; expected keys: 'phi_p0', 'psi_xt'
        self.results = getattr(solution_instance, 'results', {})
        if self.results is None:
            self.results = {}
        self.phi_p0 = self.results.get('phi_p0', None)
        self.psi_xt = self.results.get('psi_xt', None)

        # Symbols for symbolic manipulation
        self.x, self.p, self.t = sp.symbols('x p t', real=True)
        self.i = sp.I

        # Derived normalization constant if needed
        if self.a is not None and self.A is None:
            # A = (2/pi)^(1/4) / sqrt(a)
            self.A_norm = (2/sp.pi)**(sp.Rational(1,4)) / sp.sqrt(self.a)
        else:
            self.A_norm = None

    # ---------- Interface Checks ----------
    def test_interface_phi_p0_key(self):
        """Check that results contains 'phi_p0' and it's not None."""
        if 'phi_p0' not in self.results:
            return (False, "Missing key 'phi_p0' in results")
        if self.phi_p0 is None:
            return (False, "phi_p0 is None")
        return (True, "phi_p0 present and non-None")

    def test_interface_psi_xt_key(self):
        """Check that results contains 'psi_xt' and it's not None."""
        if 'psi_xt' not in self.results:
            return (False, "Missing key 'psi_xt' in results")
        if self.psi_xt is None:
            return (False, "psi_xt is None")
        return (True, "psi_xt present and non-None")

    def test_interface_givens_present(self):
        """Check that essential givens (m, hbar, a) are provided."""
        if self.m is None:
            return (False, "Given 'm' missing")
        if self.hbar is None:
            return (False, "Given 'hbar' missing")
        if self.a is None:
            return (False, "Given 'a' missing")
        return (True, "All essential givens present")

    # ---------- Invariant Checks (theorem-based, use isclose) ----------
    def test_invariant_normalization_initial(self):
        """Check that the initial wavefunction is normalized: ∫|ψ(x,0)|^2 dx = 1."""
        if self.a is None:
            return (False, "SKIP: a not available")
        # Use computed A from normalization if A not explicitly given
        if self.A is not None:
            A = sp.sympify(self.A) if not isinstance(self.A, sp.Basic) else self.A
        elif self.A_norm is not None:
            A = self.A_norm
        else:
            return (False, "SKIP: cannot obtain A")

        psi0 = A * sp.exp(-self.x**2 / self.a**2)
        norm_sq = sp.integrate(sp.conjugate(psi0)*psi0, (self.x, -sp.oo, sp.oo))
        # Simplify assuming real positive a
        norm_sq_simpl = sp.simplify(norm_sq)
        if norm_sq_simpl.is_number:
            val = float(norm_sq_simpl)
            if math.isclose(val, 1.0, rel_tol=1e-2):
                return (True, f"Normalization integral = {val} (close to 1)")
            else:
                return (False, f"Normalization integral = {val} (not 1)")
        # fallback: numerical evaluation with sample values
        try:
            a_val = 1.0  # sample
            A_val = float(A.subs(self.a, a_val))
            norm_num = sp.N(sp.integrate((A_val*sp.exp(-self.x**2/a_val**2))**2, (self.x, -sp.oo, sp.oo)))
            if math.isclose(float(norm_num), 1.0, rel_tol=1e-2):
                return (True, "Numerical normalization check passed")
            else:
                return (False, f"Numerical normalization = {norm_num}")
        except Exception:
            return (False, "Cannot evaluate normalization")

    def test_invariant_fourier_relation(self):
        """Check that φ(p,0) is the Fourier transform of ψ(x,0)."""
        if any(v is None for v in [self.a, self.hbar, self.phi_p0]):
            return (False, "SKIP: insufficient data")
        if self.A is not None:
            A = sp.sympify(self.A) if not isinstance(self.A, sp.Basic) else self.A
        elif self.A_norm is not None:
            A = self.A_norm
        else:
            return (False, "SKIP: cannot obtain A")
        # psi(x,0)
        psi0 = A * sp.exp(-self.x**2 / self.a**2)
        # Fourier transform definition: φ(p,0) = (1/√(2πħ)) ∫ ψ(x,0) e^{-ipx/ħ} dx
        factor = 1/sp.sqrt(2*sp.pi*self.hbar)
        ft_expr = factor * sp.integrate(psi0 * sp.exp(-sp.I*self.p*self.x/self.hbar),
                                        (self.x, -sp.oo, sp.oo))
        ft_simpl = sp.simplify(ft_expr)
        # Compare with provided phi_p0
        phi_provided = sp.sympify(self.phi_p0) if not isinstance(self.phi_p0, sp.Basic) else self.phi_p0
        diff = sp.simplify(ft_simpl - phi_provided)
        # Check if difference is zero symbolically
        if diff == 0:
            return (True, "Fourier transform relation holds symbolically")
        # Numerical equality test on a few points
        try:
            a_val, hbar_val = 1.0, 1.0
            subs_dict = {self.a: a_val, self.hbar: hbar_val}
            ft_num = sp.N(ft_simpl.subs(subs_dict).subs(self.p, 1.0))
            phi_num = sp.N(phi_provided.subs(subs_dict).subs(self.p, 1.0))
            if math.isclose(float(ft_num), float(phi_num), rel_tol=1e-2):
                return (True, "Fourier transform matches numerically")
            else:
                return (False, f"FT = {ft_num}, provided = {phi_num}")
        except Exception as e:
            return (False, f"Fourier check failed: {e}")

    def test_invariant_time_evolution_via_momentum(self):
        """Check that ψ(x,t) is the inverse Fourier transform of φ(p,0) * exp(-i p^2 t/(2mħ))."""
        if any(v is None for v in [self.m, self.hbar, self.a, self.phi_p0, self.psi_xt]):
            return (False, "SKIP: missing data")
        # Obtain expressions
        phi_p0 = sp.sympify(self.phi_p0) if not isinstance(self.phi_p0, sp.Basic) else self.phi_p0
        # Time-evolved momentum wavefunction
        phi_pt = phi_p0 * sp.exp(-sp.I * self.p**2 * self.t / (2 * self.m * self.hbar))
        # Inverse transform: ψ(x,t) = (1/√(2πħ)) ∫ φ(p,t) e^{ipx/ħ} dp
        inv_factor = 1/sp.sqrt(2*sp.pi*self.hbar)
        inv_expr = inv_factor * sp.integrate(phi_pt * sp.exp(sp.I * self.p * self.x / self.hbar),
                                             (self.p, -sp.oo, sp.oo))
        inv_simpl = sp.simplify(inv_expr)
        # Compare with provided psi_xt
        psi_provided = sp.sympify(self.psi_xt) if not isinstance(self.psi_xt, sp.Basic) else self.psi_xt
        diff = sp.simplify(inv_simpl - psi_provided)
        if diff == 0:
            return (True, "Time evolution (momentum space) matches symbolically")
        # Numerical test for a sample (x,t)
        try:
            a_val, hbar_val, m_val, x_val, t_val = 1.0, 1.0, 1.0, 0.5, 0.2
            subs = {self.a: a_val, self.hbar: hbar_val, self.m: m_val, self.x: x_val, self.t: t_val}
            inv_num = sp.N(inv_simpl.subs(subs))
            psi_num = sp.N(psi_provided.subs(subs))
            if math.isclose(float(inv_num), float(psi_num), rel_tol=1e-2):
                return (True, "Time evolution matches numerically")
            else:
                return (False, f"InvFT = {inv_num}, provided = {psi_num}")
        except Exception as e:
            return (False, f"Time evolution check failed: {e}")

    # ---------- Modeling / Trace Checks ----------
    def test_modeling_normalization_constant(self):
        """Check that the amplitude A (if given) matches normalization condition."""
        if self.a is None:
            return (False, "SKIP: a missing")
        if self.A is None:
            return (False, "SKIP: A not provided")
        # Compute expected A from normalization
        expected_A = sp.simplify((2/sp.pi)**(sp.Rational(1,4)) / sp.sqrt(self.a))
        provided_A = sp.sympify(self.A) if not isinstance(self.A, sp.Basic) else self.A
        if sp.simplify(expected_A - provided_A) == 0:
            return (True, "Amplitude A matches normalization")
        # Numerical check
        try:
            a_val = 1.0
            expected_val = float(expected_A.subs(self.a, a_val))
            provided_val = float(provided_A.subs(self.a, a_val))
            if math.isclose(expected_val, provided_val, rel_tol=1e-2):
                return (True, "A matches normalization numerically")
            else:
                return (False, f"Expected A = {expected_val}, got {provided_val}")
        except Exception:
            return (False, "Cannot compare A")

    def test_modeling_momentum_at_zero(self):
        """Check that φ(p=0,0) = A a / √(2ħ)."""
        if any(v is None for v in [self.a, self.hbar, self.phi_p0]):
            return (False, "SKIP: missing data")
        phi = sp.sympify(self.phi_p0) if not isinstance(self.phi_p0, sp.Basic) else self.phi_p0
        phi_at_zero = sp.simplify(phi.subs(self.p, 0))
        # Expected value: A*a/sqrt(2*hbar)
        if self.A is not None:
            A = sp.sympify(self.A)
        elif self.A_norm is not None:
            A = self.A_norm
        else:
            return (False, "SKIP: cannot obtain A")
        expected = sp.simplify(A * self.a / sp.sqrt(2*self.hbar))
        if sp.simplify(phi_at_zero - expected) == 0:
            return (True, "φ(p=0) matches expected value")
        try:
            a_val, hbar_val = 1.0, 1.0
            subs = {self.a: a_val, self.hbar: hbar_val}
            phi_val = float(phi_at_zero.subs(subs))
            exp_val = float(expected.subs(subs))
            if math.isclose(phi_val, exp_val, rel_tol=1e-2):
                return (True, "φ(p=0) matches expected numerically")
            else:
                return (False, f"φ(0) = {phi_val}, expected = {exp_val}")
        except Exception:
            return (False, "Cannot evaluate φ(p=0)")

    # ---------- Monotonicity Checks (trend with parameter variation) ----------
    def test_monotonicity_p0_width_with_a(self):
        """
        As a (width in position space) increases, the width of |φ(p,0)|^2 should decrease.
        We compare the value at p=0 (peak) relative to a fixed off-peak p.
        """
        if any(v is None for v in [self.a, self.hbar, self.phi_p0]):
            return (False, "SKIP: missing data")

        # Use two values of a: a1 < a2
        a1 = 1.0
        a2 = 2.0
        hbar_val = 1.0
        # Evaluate |φ(p,0)|^2 at p=0 and at p=1 for both a values
        phi_expr = sp.sympify(self.phi_p0) if not isinstance(self.phi_p0, sp.Basic) else self.phi_p0
        # We need to evaluate phi_expr numerically, handling A properly.
        # If A is a symbol, we substitute it with its normalized value.
        def eval_phi_sq(p_val, a_val):
            """Return |φ(p,0)|^2 for given p and a, using normalized A."""
            # Build substitution dictionary
            subs = {self.p: p_val, self.a: a_val, self.hbar: hbar_val}
            # If A is a symbol in the expression, replace it with normalized value.
            if self.A is not None and isinstance(self.A, sp.Basic):
                A_norm_val = (2/sp.pi)**(sp.Rational(1,4)) / sp.sqrt(a_val)
                subs[self.A] = A_norm_val
            # Evaluate expression numerically
            val = phi_expr.subs(subs)
            # If expression contains complex numbers, take absolute square.
            val_complex = sp.N(val)
            return float(abs(val_complex)**2)

        try:
            # For a1
            peak_sq_a1 = eval_phi_sq(0.0, a1)
            off_sq_a1 = eval_phi_sq(1.0, a1)
            ratio1 = off_sq_a1 / peak_sq_a1 if peak_sq_a1 != 0 else float('inf')
            # For a2
            peak_sq_a2 = eval_phi_sq(0.0, a2)
            off_sq_a2 = eval_phi_sq(1.0, a2)
            ratio2 = off_sq_a2 / peak_sq_a2 if peak_sq_a2 != 0 else float('inf')

            if ratio2 < ratio1:
                return (True, f"Width decreases as a increases: ratio1={ratio1:.4f}, ratio2={ratio2:.4f}")
            else:
                return (False, f"Expected width decrease, but ratio increased: {ratio1:.4f} -> {ratio2:.4f}")
        except Exception as e:
            return (False, f"Monotonicity test crashed: {e}")

    # ---------- Weak Answer Sanity (no isclose, no re-derivation) ----------
    def test_answer_sanity_expr_finite(self):
        """Check that φ(p,0) and ψ(x,t) are finite for typical finite arguments (not infinite)."""
        if self.phi_p0 is None or self.psi_xt is None:
            return (False, "SKIP: results missing")
        try:
            phi_expr = sp.sympify(self.phi_p0) if not isinstance(self.phi_p0, sp.Basic) else self.phi_p0
            psi_expr = sp.sympify(self.psi_xt) if not isinstance(self.psi_xt, sp.Basic) else self.psi_xt
            # Use some dummy values; if they contain undefined symbols, it's okay.
            subs = {self.p:1.0, self.x:0.5, self.t:0.1}
            # If m, hbar, a are known, include them.
            if self.m is not None:
                subs[self.m] = 1.0
            if self.hbar is not None:
                subs[self.hbar] = 1.0
            if self.a is not None:
                subs[self.a] = 1.0
            # Evaluate
            phi_val = sp.N(phi_expr.subs(subs))
            psi_val = sp.N(psi_expr.subs(subs))
            if math.isfinite(float(phi_val)) and math.isfinite(float(psi_val)):
                return (True, "Expressions yield finite values at sample points")
            else:
                return (False, "Expressions produced non-finite (NaN/inf) values")
        except Exception:
            return (False, "Could not evaluate expressions")

    # ---------- Run All Tests ----------
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