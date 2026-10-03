import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Audits a solution for the GRIN lens problem: n(r) = n0 - r^2/(2 d F).
    """

    def __init__(self, solution_instance):
        # Extract givens – try both attribute and dict access
        self.a = getattr(solution_instance, 'a', None)
        self.d = getattr(solution_instance, 'd', None)
        self.F = getattr(solution_instance, 'F', None)
        self.n0 = getattr(solution_instance, 'n0', None)

        # Fallback to .givens dict if available
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            givens = solution_instance.givens
            self.a = self.a if self.a is not None else givens.get('a', None)
            self.d = self.d if self.d is not None else givens.get('d', None)
            self.F = self.F if self.F is not None else givens.get('F', None)
            self.n0 = self.n0 if self.n0 is not None else givens.get('n0', None)

        # Extract result (n_r expression)
        self.n_r = None
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.n_r = solution_instance.results.get('n_r', None)

        # Try to convert n_r to a sympy expression
        self._n_r_expr = None
        if self.n_r is not None:
            try:
                if isinstance(self.n_r, str):
                    self._n_r_expr = sp.sympify(self.n_r)
                else:
                    self._n_r_expr = sp.sympify(self.n_r)
            except Exception:
                self._n_r_expr = None

        # Symbolic variables for substitution
        self._r = sp.Symbol('r', real=True, positive=True)
        self._d = sp.Symbol('d', real=True, positive=True)
        self._F = sp.Symbol('F', real=True, positive=True)
        self._n0 = sp.Symbol('n0', real=True)

    # ------------------------------------------------------------------
    # Helper to evaluate n(r) at a given numerical r (using the expression)
    # ------------------------------------------------------------------
    def _eval_n(self, r_val, override_d=None, override_F=None, override_n0=None):
        """Evaluate the n_r expression at a given r_val with optional parameter overrides."""
        if self._n_r_expr is None:
            return None
        try:
            subs_dict = {self._r: r_val}
            if self.d is not None:
                subs_dict[self._d] = self.d if override_d is None else override_d
            if self.F is not None:
                subs_dict[self._F] = self.F if override_F is None else override_F
            if self.n0 is not None:
                subs_dict[self._n0] = self.n0 if override_n0 is None else override_n0
            val = self._n_r_expr.subs(subs_dict).evalf()
            return float(val)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_result_key_exists(self):
        """Check that results['n_r'] exists."""
        if self.n_r is None:
            return (False, "results['n_r'] is missing")
        return (True, "results['n_r'] present")

    def test_interface_given_a_present(self):
        if self.a is None:
            return (False, "Given 'a' (disk radius) not found")
        return (True, f"a = {self.a}")

    def test_interface_given_d_present(self):
        if self.d is None:
            return (False, "Given 'd' (lens thickness) not found")
        return (True, f"d = {self.d}")

    def test_interface_given_F_present(self):
        if self.F is None:
            return (False, "Given 'F' (focal length) not found")
        return (True, f"F = {self.F}")

    def test_interface_given_n0_present(self):
        if self.n0 is None:
            return (False, "Given 'n0' (center index) not found")
        return (True, f"n0 = {self.n0}")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_center_value(self):
        """n(r=0) should equal n0."""
        if self._n_r_expr is None or self.n0 is None:
            return (False, "SKIP: expression or n0 missing")
        n_center = self._eval_n(0.0)
        if n_center is None:
            return (False, "Could not evaluate n at r=0")
        if math.isclose(n_center, self.n0, rel_tol=1e-2):
            return (True, f"n(0) = {n_center} ≈ n0 = {self.n0}")
        else:
            return (False, f"n(0) = {n_center} but expected {self.n0}")

    def test_invariant_parabolic_form(self):
        """Check that (n(r) - n0) * 2 * d * F == -r^2 (symbolic test)."""
        if self._n_r_expr is None or any(v is None for v in [self.d, self.F, self.n0]):
            return (False, "SKIP: expression or givens incomplete")
        delta = (self._n_r_expr - self._n0) * 2 * self._d * self._F
        # substitute known numeric values for d, F, n0 in delta, then replace r with a numeric value
        subs_delta = delta.subs({self._d: self.d, self._F: self.F, self._n0: self.n0})
        # evaluate at a couple of radii
        for r_test in [0.1, 0.2, 0.5]:
            expected = - r_test**2
            computed = subs_delta.subs({self._r: r_test}).evalf()
            if computed is None:
                return (False, f"Computation failed at r={r_test}")
            if not math.isclose(float(computed), expected, rel_tol=1e-2):
                return (False, f"At r={r_test}: (n-n0)*2*d*F = {float(computed):.4f}, expected -r^2 = {expected:.4f}")
        return (True, "Parabolic form (n-n0)*2*d*F = -r^2 holds")

    def test_invariant_vs_exact_small_angle(self):
        """For small r, parabolic approximation should match exact sqrt formula."""
        if self._n_r_expr is None or any(v is None for v in [self.a, self.d, self.F, self.n0]):
            return (False, "SKIP: expression or givens missing")
        # Choose a small radius (r << F)
        r_small = 0.05 * self.F
        if r_small > self.a:
            r_small = 0.1 * self.a
        n_parabolic = self._eval_n(r_small)
        # Exact formula from explanation steps: n_exact = n0 - (sqrt(F^2 + r^2) - F)/d
        n_exact = self.n0 - (math.sqrt(self.F**2 + r_small**2) - self.F) / self.d
        if n_parabolic is None or math.isnan(n_parabolic):
            return (False, "Could not compute parabolic n")
        if math.isclose(n_parabolic, n_exact, rel_tol=1e-2):
            return (True, f"For r={r_small:.4f}: parabolic n={n_parabolic:.6f}, exact n={n_exact:.6f}")
        else:
            return (False, f"For r={r_small:.4f}: parabolic n={n_parabolic:.6f} differs from exact n={n_exact:.6f}")

    def test_invariant_derivative_negative(self):
        """Derivative dn/dr should be negative for r>0."""
        if self._n_r_expr is None:
            return (False, "SKIP: no expression")
        try:
            deriv = sp.diff(self._n_r_expr, self._r)
            # substitute numeric r=some positive value
            if self.d is not None and self.F is not None:
                num_deriv = deriv.subs({self._r: 0.1, self._d: self.d, self._F: self.F, self._n0: self.n0}).evalf()
                if float(num_deriv) >= 0:
                    return (False, f"Derivative dn/dr at r=0.1 is {float(num_deriv):.6f} (should be negative)")
                return (True, f"Derivative dn/dr at r=0.1 = {float(num_deriv):.6f} (negative)")
            else:
                # symbolic sign test: derivative should be -r/(d*F) which is negative for r>0
                simplified = sp.simplify(deriv)
                # expecting -r/(d*F)
                target = -self._r/(self._d*self._F)
                if sp.simplify(simplified - target) == 0:
                    return (True, f"Derivative symbolically matches -r/(d*F): {simplified}")
                else:
                    return (False, f"Derivative {simplified} does not match expected -r/(d*F)")
        except Exception as e:
            return (False, f"Derivative check crashed: {e}")

    # ------------------------------------------------------------------
    # Modeling / trace consistency tests
    # ------------------------------------------------------------------
    def test_modeling_phase_condition(self):
        """Check that the phase difference: k * n(r) * d - k * n0 * d equals - k * r^2/(2F)."""
        if self._n_r_expr is None or any(v is None for v in [self.d, self.F, self.n0]):
            return (False, "SKIP: expression or givens missing")
        # Without k, we need (n(r) - n0)*d = -r^2/(2F)
        condition = (self._n_r_expr - self._n0) * self._d + self._r**2 / (2 * self._F)
        simplified = sp.simplify(condition)
        # For the parabolic expression, this should be identically zero
        # But we can test numerically
        r_test = 0.3
        val = condition.subs({self._r: r_test, self._d: self.d, self._F: self.F, self._n0: self.n0}).evalf()
        if val is None:
            return (False, "Evaluation failed")
        if math.isclose(float(val), 0.0, abs_tol=1e-2):
            return (True, f"Phase condition holds: (n-n0)*d + r^2/(2F) ≈ 0 at r={r_test}")
        else:
            return (False, f"Phase condition deviates: (n-n0)*d + r^2/(2F) = {float(val):.6f} at r={r_test}")

    def test_modeling_thin_lens_consistency(self):
        """Check that n(r) > 1 for small r (glass requirement)."""
        if self._n_r_expr is None or any(v is None for v in [self.a, self.d, self.F, self.n0]):
            return (False, "SKIP: missing data")
        # Evaluate at r=0 and at r=0.1*a (small radius to ensure n>1)
        for r_frac in [0.0, 0.1]:
            r_val = r_frac * self.a
            n_val = self._eval_n(r_val)
            if n_val is None or not math.isfinite(n_val):
                return (False, f"Could not evaluate n(r) at r={r_val}")
            # Note: For sensible parameters n0 should be >1. We only flag if n_val < 1
            if n_val < 1.0:
                return (False, f"n(r) = {n_val} at r={r_val} < 1 (unphysical for glass)")
        return (True, "n(r) stays above 1 for small radii (within typical parameters)")

    # ------------------------------------------------------------------
    # Monotonicity tests (perturb an input and compare perturbed output to baseline)
    # ------------------------------------------------------------------
    def test_monotonicity_n_decreases_with_r(self):
        """Perturb d and check that the decrease Δn = n(r1)-n(r2) becomes smaller."""
        if self._n_r_expr is None or any(v is None for v in [self.a, self.d, self.F, self.n0]):
            return (False, "SKIP: expression or givens missing")
        r1 = 0.1 * self.a
        r2 = 0.5 * self.a
        # Baseline Δn
        n1_base = self._eval_n(r1)
        n2_base = self._eval_n(r2)
        if n1_base is None or n2_base is None:
            return (False, "Could not evaluate baseline n")
        delta_base = n1_base - n2_base
        # Perturb d (increase by 10%)
        d_new = self.d * 1.1
        n1_pert = self._eval_n(r1, override_d=d_new)
        n2_pert = self._eval_n(r2, override_d=d_new)
        if n1_pert is None or n2_pert is None:
            return (False, "Could not evaluate perturbed n")
        delta_pert = n1_pert - n2_pert
        # Since n(r) = n0 - r^2/(2dF), a larger d reduces the decrease.
        if delta_pert < delta_base:
            return (True, f"Delta_n baseline {delta_base:.4f}, perturbed (d*1.1) {delta_pert:.4f}: decrease is smaller (consistent with physics)")
        else:
            return (False, f"Delta_n baseline {delta_base:.4f}, perturbed {delta_pert:.4f}: expected decrease to become smaller, but it increased")

    def test_monotonicity_effect_of_d(self):
        """Increasing d should increase n(r) for fixed r (less index variation)."""
        if self._n_r_expr is None or any(v is None for v in [self.a, self.d, self.F, self.n0]):
            return (False, "SKIP: expression or givens missing")
        r_test = 0.3 * self.a
        d_original = self.d
        d_new = d_original * 1.1  # +10%
        n_orig = self._eval_n(r_test, override_d=d_original)
        n_new = self._eval_n(r_test, override_d=d_new)
        if n_orig is None or n_new is None:
            return (False, "Could not evaluate n under perturbation")
        if n_new > n_orig:
            return (True, f"Increasing d from {d_original} to {d_new:.4f}, n(r) rises from {n_orig:.4f} to {n_new:.4f}")
        else:
            return (False, f"Increasing d should make n(r) larger, but got n_orig={n_orig:.4f}, n_new={n_new:.4f}")

    def test_monotonicity_effect_of_F(self):
        """Increasing F should increase n(r) (less focusing required)."""
        if self._n_r_expr is None or any(v is None for v in [self.a, self.d, self.F, self.n0]):
            return (False, "SKIP: expression or givens missing")
        r_test = 0.3 * self.a
        F_original = self.F
        F_new = F_original * 1.2  # +20%
        n_orig = self._eval_n(r_test, override_F=F_original)
        n_new = self._eval_n(r_test, override_F=F_new)
        if n_orig is None or n_new is None:
            return (False, "Could not evaluate n under perturbation")
        if n_new > n_orig:
            return (True, f"Increasing F from {F_original} to {F_new:.4f}, n(r) rises from {n_orig:.4f} to {n_new:.4f}")
        else:
            return (False, f"Increasing F should make n(r) larger, but got n_orig={n_orig:.4f}, n_new={n_new:.4f}")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one, and must not compute expected value)
    # ------------------------------------------------------------------
    def test_answer_sanity_bounds(self):
        """Check that n(r) is real and finite for all r in [0,a] (weak)."""
        if self._n_r_expr is None or any(v is None for v in [self.a, self.d, self.F, self.n0]):
            return (False, "SKIP: missing data")
        # Test a few points
        for frac in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            r_val = frac * self.a
            n_val = self._eval_n(r_val)
            if n_val is None:
                return (False, f"Evaluation at r={r_val} returned None")
            if not math.isfinite(n_val):
                return (False, f"n(r) at r={r_val} is not finite (NaN/Inf)")
            # n should be real (already ensured by evalf)
        return (True, "n(r) is finite and real for all sampled radii")

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Gather all test methods starting with 'test_'
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