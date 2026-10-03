import math
import sympy as sp

class PhysicsChecker:
    """
    Audits solutions for the problem:
    "A charged oscillating system losing energy by radiation.
    ν(E) = α (E/E₀)^{-β}. Compute E_n for large n."
    
    Uses the correspondence principle: dE/dn = h ν(E).
    """
    def __init__(self, solution_instance):
        # Safely extract givens and results
        givens = getattr(solution_instance, 'givens', {})
        results = getattr(solution_instance, 'results', {})

        # Physical constants and parameters – use None if missing
        self.alpha = givens.get('alpha', None)
        self.beta = givens.get('beta', None)
        self.E0 = givens.get('E0', None)
        self.h = givens.get('h', None)
        # Note: 'n' may be provided as a symbol or a numeric value
        self.n = givens.get('n', sp.Symbol('n', positive=True, real=True))

        # Computed result
        self.E_n = results.get('E_n', None)

        # For numeric checks we define a positive test value of n and constants.
        self._test_n = 100   # large n
        self._test_alpha = 3.0
        self._test_beta = 0.5
        self._test_E0 = 2.0
        self._test_h = 6.626e-34

    # ------- Interface Tests -------
    def test_interface_E_n_exists(self):
        if self.E_n is None:
            return (False, "SKIP: results['E_n'] missing")
        return (True, "E_n is present")

    def test_interface_givens_positive(self):
        """Check that all required physical givens are present and positive."""
        for name, val in [('alpha', self.alpha),
                          ('beta', self.beta),
                          ('E0', self.E0),
                          ('h', self.h)]:
            if val is None:
                return (False, f"SKIP: given '{name}' missing")
            if val <= 0:
                return (False, f"given '{name}' must be positive, got {val}")
        return (True, "All givens are positive")

    # ------- Invariant / Physics Consistency Tests -------
    def test_invariant_correspondence_equation(self):
        """Check symbolic equality: dE/dn = h α (E/E₀)^{-β}."""
        if self.E_n is None or None in (self.alpha, self.beta, self.E0, self.h):
            return (False, "SKIP: insufficient data")

        # Build the equation symbolically
        n_sym = self.n if isinstance(self.n, sp.Basic) else sp.Symbol('n', positive=True)
        E_expr = self.E_n if isinstance(self.E_n, sp.Basic) else sp.sympify(self.E_n)

        # Left-hand side: derivative dE/dn
        dE_dn = sp.diff(E_expr, n_sym)

        # Right-hand side: h * alpha * (E/E0)^{-beta}
        lhs = dE_dn
        rhs = self.h * self.alpha * (E_expr / self.E0) ** (-self.beta)

        # Simplify the difference
        diff = sp.simplify(lhs - rhs)
        if diff == 0:
            return (True, "dE/dn ≡ h α (E/E₀)^{-β} holds symbolically")
        else:
            # Attempt numeric test as fallback for small residue
            try:
                subs = {n_sym: self._test_n, self.alpha: self._test_alpha,
                        self.beta: self._test_beta, self.E0: self._test_E0,
                        self.h: self._test_h, sp.Symbol('h'): self._test_h}
                num_diff = diff.subs(subs).evalf()
                if abs(num_diff) < 1e-6:
                    return (True, f"dE/dn matches numerically (diff = {num_diff})")
                else:
                    return (False, f"dE/dn discrepancy (diff = {num_diff})")
            except Exception:
                return (False, f"Symbolic test failed and numeric fallback not possible")

    def test_invariant_energy_positive(self):
        """Check that E_n > 0 for positive parameters and n."""
        if self.E_n is None:
            return (False, "SKIP: no E_n")

        # Try numeric evaluation with positive test values
        try:
            E_expr = sp.sympify(self.E_n)
            n_sym = self.n if isinstance(self.n, sp.Basic) else sp.Symbol('n')
            subs = {n_sym: self._test_n, sp.Symbol('alpha'): self._test_alpha,
                    sp.Symbol('beta'): self._test_beta, sp.Symbol('E0'): self._test_E0,
                    sp.Symbol('h'): self._test_h}
            # Also replace any existing symbols in E_expr
            val = E_expr.subs(subs).evalf()
            if val > 0:
                return (True, f"E_n = {val} > 0 (positive test parameters)")
            else:
                return (False, f"E_n = {val} is not positive")
        except Exception:
            return (False, "Could not evaluate E_n numerically")

    # ------- Modeling / Trace Tests -------
    def test_modeling_integral_relation(self):
        """Check that ∫₀ᴱ E^β dE = h α E₀^β n."""
        if self.E_n is None or None in (self.alpha, self.beta, self.E0, self.h):
            return (False, "SKIP: insufficient data")

        # Build the integral relation: left = (1/(β+1)) E^{β+1}
        E_expr = sp.sympify(self.E_n)
        n_sym = self.n if isinstance(self.n, sp.Basic) else sp.Symbol('n', positive=True)

        left = (1/(self.beta + 1)) * (E_expr ** (self.beta + 1))
        right = self.h * self.alpha * (self.E0 ** self.beta) * n_sym

        diff = sp.simplify(left - right)
        if diff == 0:
            return (True, "Integral relation holds symbolically")
        else:
            # Numeric fallback
            try:
                subs = {n_sym: self._test_n, sp.Symbol('alpha'): self._test_alpha,
                        sp.Symbol('beta'): self._test_beta, sp.Symbol('E0'): self._test_E0,
                        sp.Symbol('h'): self._test_h}
                num_diff = abs(diff.subs(subs).evalf())
                if num_diff < 1e-6:
                    return (True, f"Integral relation holds numerically (diff={num_diff})")
                else:
                    return (False, f"Integral relation fails numerically (diff={num_diff})")
            except Exception:
                return (False, "Symbolic and numeric checks inconclusive")

    def test_modeling_beta_zero_limit(self):
        """For β=0, ν = α constant, E_n = h α n."""
        if self.E_n is None:
            return (False, "SKIP: no E_n")

        # Replace beta with 0 in E_n expression and simplify
        E_expr = sp.sympify(self.E_n)
        # Substitute beta=0
        try:
            E_beta0 = E_expr.subs(sp.Symbol('beta'), 0)
            expected = self.h * self.alpha * sp.Symbol('n')  # n is symbolic
            if E_beta0.equals(expected):
                return (True, "For β=0, E_n reduces to h α n")
            else:
                # Numeric test
                subs = {sp.Symbol('n'): self._test_n, sp.Symbol('alpha'): self._test_alpha,
                        sp.Symbol('h'): self._test_h}
                numeric_diff = abs(E_beta0.subs(subs).evalf() - (self._test_h * self._test_alpha * self._test_n))
                if numeric_diff < 1e-6:
                    return (True, f"For β=0, E_n numerically matches h α n (diff={numeric_diff})")
                else:
                    return (False, f"For β=0, E_n does not match h α n (diff={numeric_diff})")
        except Exception:
            return (False, "Could not perform β=0 substitution")

    # ------- Monotonicity Tests (must perturb input and compare) -------
    def test_monotonicity_energy_increases_with_n(self):
        """Check that E_n increases with n (explicit perturbation of n by +5%)."""
        if self.E_n is None or None in (self.alpha, self.beta, self.E0, self.h):
            return (False, "SKIP: insufficient data")

        # Use numeric evaluation: increase n by 5% and check increase.
        try:
            E_expr = sp.sympify(self.E_n)
            n_sym = self.n if isinstance(self.n, sp.Basic) else sp.Symbol('n')
            subs_base = {n_sym: self._test_n, sp.Symbol('alpha'): self._test_alpha,
                         sp.Symbol('beta'): self._test_beta, sp.Symbol('E0'): self._test_E0,
                         sp.Symbol('h'): self._test_h}
            E_base = E_expr.subs(subs_base).evalf()

            # 5% increase in n
            n_pert = self._test_n * 1.05
            subs_pert = dict(subs_base)
            subs_pert[n_sym] = n_pert
            E_pert = E_expr.subs(subs_pert).evalf()

            if E_pert > E_base:
                return (True, f"E_n increases: E({self._test_n})={E_base} < E({n_pert})={E_pert}")
            else:
                return (False, f"E_n does not increase: E({self._test_n})={E_base} >= E({n_pert})={E_pert}")
        except Exception as e:
            return (False, f"Monotonicity test failed: {e}")

    def test_monotonicity_energy_decreases_with_beta(self):
        """Check that increasing beta (steeper decay) reduces E_n (explicit perturbation of beta by +10%)."""
        if self.E_n is None or None in (self.alpha, self.beta, self.E0, self.h):
            return (False, "SKIP: insufficient data")

        # Use numeric evaluation: increase beta by 10% and check that E_n decreases.
        try:
            E_expr = sp.sympify(self.E_n)
            n_sym = self.n if isinstance(self.n, sp.Basic) else sp.Symbol('n')
            subs_base = {n_sym: self._test_n, sp.Symbol('alpha'): self._test_alpha,
                         sp.Symbol('beta'): self._test_beta, sp.Symbol('E0'): self._test_E0,
                         sp.Symbol('h'): self._test_h}
            E_base = E_expr.subs(subs_base).evalf()

            # 10% increase in beta
            beta_pert = self._test_beta * 1.10
            subs_pert = dict(subs_base)
            subs_pert[sp.Symbol('beta')] = beta_pert
            E_pert = E_expr.subs(subs_pert).evalf()

            if E_pert < E_base:
                return (True, f"E_n decreases with beta: E(β={self._test_beta})={E_base} > E(β={beta_pert})={E_pert}")
            else:
                return (False, f"E_n does not decrease with beta: E(β={self._test_beta})={E_base} <= E(β={beta_pert})={E_pert}")
        except Exception as e:
            return (False, f"Monotonicity test (beta) failed: {e}")

    # ------- Weak Answer Sanity (optional, max one) -------
    def test_answer_sanity_broad_range(self):
        """Weak final check: E_n must be finite, positive, and not absurdly large."""
        if self.E_n is None:
            return (False, "SKIP: no E_n")
        try:
            E_expr = sp.sympify(self.E_n)
            n_sym = self.n if isinstance(self.n, sp.Basic) else sp.Symbol('n')
            subs = {n_sym: self._test_n, sp.Symbol('alpha'): self._test_alpha,
                    sp.Symbol('beta'): self._test_beta, sp.Symbol('E0'): self._test_E0,
                    sp.Symbol('h'): self._test_h}
            val = float(E_expr.subs(subs).evalf())
            if val <= 0 or not math.isfinite(val):
                return (False, f"E_n = {val} not positive/finite")
            # Check not absurd: for typical values, E_n ~ 1e-30 to 1e+30 is plausible
            if val > 1e30 or val < 1e-30:
                return (False, f"E_n = {val} appears out of physical range")
            return (True, f"E_n = {val} passes broad sanity")
        except Exception:
            return (False, "Could not evaluate E_n for sanity check")

    # ------- Test Runner -------
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