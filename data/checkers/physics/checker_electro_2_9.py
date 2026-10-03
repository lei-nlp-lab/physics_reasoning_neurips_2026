import sympy as sp
import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results safely using getattr with None defaults
        self.results = getattr(solution_instance, 'results', {})
        if self.results is None:
            self.results = {}

        # Convert function strings to sympy expressions if possible
        self.B_inside_func = self._safe_sympify(self.results.get('B_inside_func', None))
        self.E_inside_func = self._safe_sympify(self.results.get('E_inside_func', None))
        self.E_outside_func = self._safe_sympify(self.results.get('E_outside_func', None))
        self.E_inside_at_t0 = self._safe_sympify(self.results.get('E_inside_at_t0', None))
        self.E_outside_at_t0 = self._safe_sympify(self.results.get('E_outside_at_t0', None))

        # Define symbolic variables commonly used
        self.mu0, self.n, self.i0, self.omega, self.b, self.s, self.t = sp.symbols('mu0 n i0 omega b s t')
        self.t0 = 2*sp.pi / self.omega

        # For numerical checks, define a sample parameter set
        self.test_params = {
            self.mu0: 4*sp.pi*1e-7,
            self.n: 1000,
            self.i0: 5,
            self.omega: 2*sp.pi*60,
            self.b: 0.1,
            self.s: 0.05,
            self.t: 0.001
        }
        self.test_params_outside = self.test_params.copy()
        self.test_params_outside[self.s] = 0.2  # s > b

    def _safe_sympify(self, expr):
        """Try to convert a string or sympy expression to sympy; return None if fails."""
        if expr is None:
            return None
        if isinstance(expr, sp.Basic):
            return expr
        try:
            # Pre-define known symbols to avoid 'could not parse' errors
            local_dict = {'mu0': self.mu0, 'n': self.n, 'i0': self.i0,
                          'omega': self.omega, 'b': self.b, 's': self.s, 't': self.t}
            return sp.sympify(str(expr), locals=local_dict)
        except Exception:
            return None

    def _eval_expr(self, expr, subs_dict):
        """Safely evaluate a sympy expression; return None on error."""
        if expr is None:
            return None
        try:
            return float(expr.subs(subs_dict).evalf())
        except Exception:
            return None

    def _get_faraday_inside(self, s_val, t_val):
        """Compute dΦ_B/dt inside: d/dt (B * pi s^2) = pi s^2 * dB/dt.
           Returns numerical value if possible, else None."""
        if self.B_inside_func is None:
            return None
        subs = self.test_params.copy()
        subs[self.s] = s_val
        subs[self.t] = t_val
        B_val = self._eval_expr(self.B_inside_func, subs)
        if B_val is None:
            return None
        # dB/dt = B.diff(t)
        dBdt_expr = sp.diff(self.B_inside_func, self.t)
        dBdt_val = self._eval_expr(dBdt_expr, subs)
        if dBdt_val is None:
            return None
        return - math.pi * s_val**2 * dBdt_val

    def _get_faraday_outside(self, s_val, t_val):
        """For s > b, flux = B * pi b^2, so dΦ/dt = pi b^2 dB/dt."""
        if self.B_inside_func is None:
            return None
        subs = self.test_params_outside.copy()
        subs[self.s] = s_val
        subs[self.t] = t_val
        B_val = self._eval_expr(self.B_inside_func, subs)
        if B_val is None:
            return None
        dBdt_expr = sp.diff(self.B_inside_func, self.t)
        dBdt_val = self._eval_expr(dBdt_expr, subs)
        if dBdt_val is None:
            return None
        return - math.pi * self.test_params[self.b]**2 * dBdt_val

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

    # ------------------------------------------------------------------
    # Interface tests: required keys exist and are not None
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        if self.B_inside_func is None:
            return (False, "B_inside_func missing or invalid")
        if self.E_inside_func is None:
            return (False, "E_inside_func missing or invalid")
        if self.E_outside_func is None:
            return (False, "E_outside_func missing or invalid")
        if self.E_inside_at_t0 is None:
            return (False, "E_inside_at_t0 missing or invalid")
        if self.E_outside_at_t0 is None:
            return (False, "E_outside_at_t0 missing or invalid")
        return (True, "All required result keys present and parseable")

    # ------------------------------------------------------------------
    # Invariant tests: theorem-based physical sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_faraday_inside(self):
        """Check that E_inside * 2πs = - dΦ_B/dt for s<b."""
        if self.E_inside_func is None:
            return (False, "SKIP: E_inside_func not available")
        s_val = 0.05  # inside
        t_val = 0.001
        subs = self.test_params.copy()
        subs[self.s] = s_val
        subs[self.t] = t_val
        E_val = self._eval_expr(self.E_inside_func, subs)
        if E_val is None:
            return (False, "Cannot evaluate E_inside_func")
        lhs = E_val * 2 * math.pi * s_val
        rhs = self._get_faraday_inside(s_val, t_val)
        if rhs is None:
            return (False, "Cannot compute dΦ_B/dt inside")
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Faraday inside mismatch: LHS={lhs}, RHS={rhs}")
        return (True, "Faraday inside verified")

    def test_invariant_faraday_outside(self):
        """Check that E_outside * 2πs = - dΦ_B/dt for s>b."""
        if self.E_outside_func is None:
            return (False, "SKIP: E_outside_func not available")
        s_val = 0.2  # outside
        t_val = 0.001
        subs = self.test_params_outside.copy()
        subs[self.s] = s_val
        subs[self.t] = t_val
        E_val = self._eval_expr(self.E_outside_func, subs)
        if E_val is None:
            return (False, "Cannot evaluate E_outside_func")
        lhs = E_val * 2 * math.pi * s_val
        rhs = self._get_faraday_outside(s_val, t_val)
        if rhs is None:
            return (False, "Cannot compute dΦ_B/dt outside")
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Faraday outside mismatch: LHS={lhs}, RHS={rhs}")
        return (True, "Faraday outside verified")

    def test_invariant_continuity_at_b(self):
        """At s=b, E_inside and E_outside should be equal."""
        if self.E_inside_func is None or self.E_outside_func is None:
            return (False, "SKIP: One or both E_func not available")
        b_val = self.test_params[self.b]
        t_val = 0.001
        subs_in = self.test_params.copy()
        subs_in[self.s] = b_val
        subs_in[self.t] = t_val
        subs_out = self.test_params_outside.copy()
        subs_out[self.s] = b_val
        subs_out[self.t] = t_val
        E_in = self._eval_expr(self.E_inside_func, subs_in)
        E_out = self._eval_expr(self.E_outside_func, subs_out)
        if E_in is None or E_out is None:
            return (False, "Cannot evaluate at s=b")
        if not math.isclose(E_in, E_out, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Continuity at b violated: E_in={E_in}, E_out={E_out}")
        return (True, "Continuity at s=b holds")

    # ------------------------------------------------------------------
    # Modeling tests: specific form checks (using analytical derivation)
    # ------------------------------------------------------------------
    def test_modeling_E_inside_form(self):
        """Check that E_inside = -(s/2) * dB/dt, where dB/dt = mu0 n i0 omega cos(omega t)."""
        if self.E_inside_func is None:
            return (False, "SKIP: E_inside_func not available")
        # Derive expected form using given parameters
        dBdt = self.mu0 * self.n * self.i0 * self.omega * sp.cos(self.omega * self.t)
        expected = - (self.s / 2) * dBdt
        # Compare with E_inside_func symbolically (after simplification)
        diff = sp.simplify(self.E_inside_func + expected)  # should be 0 if equal
        # Substitute sample numbers to avoid tricky symbolic comparisons
        subs = self.test_params.copy()
        diff_num = self._eval_expr(diff, subs)
        if diff_num is None:
            return (False, "Cannot evaluate difference")
        if not math.isclose(diff_num, 0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E_inside form mismatch: difference = {diff_num}")
        return (True, "E_inside has correct form")

    def test_modeling_E_outside_form(self):
        """Check that E_outside = -(b^2/(2s)) * dB/dt."""
        if self.E_outside_func is None:
            return (False, "SKIP: E_outside_func not available")
        dBdt = self.mu0 * self.n * self.i0 * self.omega * sp.cos(self.omega * self.t)
        expected = - (self.b**2 / (2 * self.s)) * dBdt
        diff = sp.simplify(self.E_outside_func + expected)
        subs = self.test_params_outside.copy()
        diff_num = self._eval_expr(diff, subs)
        if diff_num is None:
            return (False, "Cannot evaluate difference")
        if not math.isclose(diff_num, 0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E_outside form mismatch: difference = {diff_num}")
        return (True, "E_outside has correct form")

    # ------------------------------------------------------------------
    # Monotonicity tests: explicitly perturb an input and compare perturbed output to baseline
    # ------------------------------------------------------------------
    def test_monotonicity_E_inside_with_s(self):
        """Perturb s upward; |E_inside| should increase proportionally to s."""
        if self.E_inside_func is None:
            return (False, "SKIP: E_inside_func not available")
        t_val = 0.001
        s0 = 0.04          # baseline
        s1 = 0.05          # perturbed (larger)
        subs0 = self.test_params.copy()
        subs0[self.s] = s0
        subs0[self.t] = t_val
        subs1 = self.test_params.copy()
        subs1[self.s] = s1
        subs1[self.t] = t_val
        E0 = abs(self._eval_expr(self.E_inside_func, subs0))
        E1 = abs(self._eval_expr(self.E_inside_func, subs1))
        if E0 is None or E1 is None:
            return (False, "Cannot evaluate")
        # Baseline output is E0, perturbed output is E1
        # Expected: E1/E0 ≈ s1/s0 = 1.25
        if not (E1 > E0):
            return (False, f"E_inside magnitude did not increase: E0={E0}, E1={E1}")
        expected_ratio = s1 / s0
        actual_ratio = E1 / E0
        if not math.isclose(actual_ratio, expected_ratio, rel_tol=0.1):
            return (False, f"Proportional increase violated: ratio={actual_ratio}, expected={expected_ratio}")
        return (True, "E_inside magnitude increases linearly with s (perturbation verified)")

    def test_monotonicity_E_outside_with_s(self):
        """Perturb s upward; |E_outside| should decrease as 1/s."""
        if self.E_outside_func is None:
            return (False, "SKIP: E_outside_func not available")
        t_val = 0.001
        s0 = 0.2           # baseline (outside)
        s1 = 0.3           # perturbed (larger)
        subs0 = self.test_params_outside.copy()
        subs0[self.s] = s0
        subs0[self.t] = t_val
        subs1 = self.test_params_outside.copy()
        subs1[self.s] = s1
        subs1[self.t] = t_val
        E0 = abs(self._eval_expr(self.E_outside_func, subs0))
        E1 = abs(self._eval_expr(self.E_outside_func, subs1))
        if E0 is None or E1 is None:
            return (False, "Cannot evaluate")
        # Baseline output is E0, perturbed output is E1
        # Expected: E1/E0 ≈ s0/s1 = 2/3 ≈ 0.6667
        if not (E1 < E0):
            return (False, f"E_outside magnitude did not decrease: E0={E0}, E1={E1}")
        expected_ratio = s0 / s1
        actual_ratio = E1 / E0
        if not math.isclose(actual_ratio, expected_ratio, rel_tol=0.1):
            return (False, f"Inverse proportionality violated: ratio={actual_ratio}, expected={expected_ratio}")
        return (True, "E_outside magnitude decreases as 1/s (perturbation verified)")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one allowed, no re-derivation, no math.isclose)
    # ------------------------------------------------------------------
    def test_answer_sanity_sign_and_range(self):
        """Weak check: fields are finite and sign is reasonable for a sample case."""
        # Check that B_inside evaluates to a finite number
        if self.B_inside_func is not None:
            subs = self.test_params.copy()
            subs[self.t] = 0.001  # some time where sin is non-zero
            B_val = self._eval_expr(self.B_inside_func, subs)
            if B_val is None or not math.isfinite(B_val):
                return (False, "B_inside gives non-finite value")
            # Sign check: for i0>0 and sin(omega*t)>0, B should be positive
            if B_val <= 0:
                return (False, "B_inside should be positive for chosen t and positive i0")
        else:
            return (False, "SKIP: B_inside_func not available")

        # Check that E_inside is finite
        if self.E_inside_func is not None:
            subs = self.test_params.copy()
            E_in = self._eval_expr(self.E_inside_func, subs)
            if E_in is None or not math.isfinite(E_in):
                return (False, "E_inside gives non-finite value")
        else:
            return (False, "SKIP: E_inside_func not available")

        # Check that E_outside is finite
        if self.E_outside_func is not None:
            subs_out = self.test_params_outside.copy()
            subs_out[self.t] = 0.001
            E_out = self._eval_expr(self.E_outside_func, subs_out)
            if E_out is None or not math.isfinite(E_out):
                return (False, "E_outside gives non-finite value")
        else:
            return (False, "SKIP: E_outside_func not available")

        return (True, "Basic sanity: fields finite and B sign reasonable")