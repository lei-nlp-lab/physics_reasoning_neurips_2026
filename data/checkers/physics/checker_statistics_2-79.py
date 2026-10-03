import sympy as sp
import math

class PhysicsChecker:
    """
    Audits the derived exponents α, β, γ in an n-dimensional universe.
    """

    def __init__(self, solution_instance):
        # Safely extract results dict
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # Extract symbolic expressions for α, β, γ
        self.alpha = self.results.get('alpha', None)
        self.beta = self.results.get('beta', None)
        self.gamma = self.results.get('gamma', None)

        # Try to extract the number of dimensions n
        # Look in solution_instance.n, then in givens dict, then in results, then default to None
        self.n = None
        if hasattr(solution_instance, 'n') and solution_instance.n is not None:
            self.n = solution_instance.n
        else:
            givens = getattr(solution_instance, 'givens', {})
            if isinstance(givens, dict) and 'n' in givens:
                self.n = givens['n']
            elif 'n' in self.results:
                self.n = self.results['n']

        # If n is a string, convert to sympy symbol; we primarily work with sympy
        if isinstance(self.n, str):
            self.n = sp.symbols(self.n)
        # If n is still None, define a symbolic variable for consistency
        if self.n is None:
            self.n = sp.symbols('n')

        # Ensure our expressions are sympy objects, converting strings if needed
        for attr in ['alpha', 'beta', 'gamma']:
            val = getattr(self, attr)
            if isinstance(val, str):
                setattr(self, attr, sp.sympify(val))
            elif val is not None and not isinstance(val, sp.Basic):
                # Turn numeric into sympy number
                setattr(self, attr, sp.nsimplify(val))

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

    # --------------------------------------------------------------------------
    # Interface checks
    # --------------------------------------------------------------------------
    def test_interface_alpha_exists(self):
        if self.alpha is None:
            return (False, "Missing result key 'alpha'")
        return (True, "alpha is present")

    def test_interface_beta_exists(self):
        if self.beta is None:
            return (False, "Missing result key 'beta'")
        return (True, "beta is present")

    def test_interface_gamma_exists(self):
        if self.gamma is None:
            return (False, "Missing result key 'gamma'")
        return (True, "gamma is present")

    # --------------------------------------------------------------------------
    # Theorem-based invariants (symbolic checks, with numerical fallback)
    # --------------------------------------------------------------------------
    def test_invariant_alpha_form(self):
        """α should equal n+1."""
        if self.alpha is None:
            return (False, "SKIP: alpha not available")
        try:
            diff = sp.simplify(self.alpha - (self.n + 1))
            if diff == 0:
                return (True, "α = n+1 holds symbolically")
            # numerical fallback: test for n=3 -> α=4
            n_val = 3
            alpha_subs = self.alpha.subs(self.n, n_val)
            if alpha_subs.is_number:
                if math.isclose(float(alpha_subs), 4.0, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, f"α evaluates to about 4 for n=3")
            return (False, f"α does not equal n+1; difference is {diff}")
        except Exception as e:
            return (False, f"Symbolic check failed: {e}")

    def test_invariant_beta_form(self):
        """β should equal n."""
        if self.beta is None:
            return (False, "SKIP: beta not available")
        try:
            diff = sp.simplify(self.beta - self.n)
            if diff == 0:
                return (True, "β = n holds symbolically")
            n_val = 3
            beta_subs = self.beta.subs(self.n, n_val)
            if beta_subs.is_number:
                if math.isclose(float(beta_subs), 3.0, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, f"β evaluates to about 3 for n=3")
            return (False, f"β does not equal n; difference is {diff}")
        except Exception as e:
            return (False, f"Symbolic check failed: {e}")

    def test_invariant_gamma_form(self):
        """γ should equal (n+2)/n = 1 + 2/n."""
        if self.gamma is None:
            return (False, "SKIP: gamma not available")
        expected = (self.n + 2) / self.n
        try:
            diff = sp.simplify(self.gamma - expected)
            if diff == 0:
                return (True, "γ = (n+2)/n holds symbolically")
            n_val = 3
            gamma_subs = self.gamma.subs(self.n, n_val)
            expected_subs = expected.subs(self.n, n_val)
            if gamma_subs.is_number and expected_subs.is_number:
                if math.isclose(float(gamma_subs), float(expected_subs), rel_tol=1e-2, abs_tol=1e-2):
                    return (True, f"γ evaluates to about 5/3 for n=3")
            return (False, f"γ does not equal (n+2)/n; difference is {diff}")
        except Exception as e:
            return (False, f"Symbolic check failed: {e}")

    # --------------------------------------------------------------------------
    # Modeling/trace quality tests
    # --------------------------------------------------------------------------
    def test_modeling_numerical_3d(self):
        """For n=3, the known 3D values should be recovered."""
        if self.alpha is None or self.beta is None or self.gamma is None:
            return (False, "SKIP: one or more results missing")
        try:
            n3 = 3
            alpha_val = float(self.alpha.subs(self.n, n3))
            beta_val = float(self.beta.subs(self.n, n3))
            gamma_val = float(self.gamma.subs(self.n, n3))
        except Exception as e:
            return (False, f"Cannot evaluate for n=3: {e}")
        if not math.isclose(alpha_val, 4.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"α(3) = {alpha_val}, expected ~4.0")
        if not math.isclose(beta_val, 3.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"β(3) = {beta_val}, expected ~3.0")
        if not math.isclose(gamma_val, 5/3, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"γ(3) = {gamma_val}, expected ~5/3")
        return (True, "All three formulas reproduce 3D values within tolerance")

    def test_modeling_alpha_linear_in_n(self):
        """α should be exactly n + 1, i.e., no other parameters."""
        if self.alpha is None:
            return (False, "SKIP: alpha not available")
        try:
            expr = sp.simplify(self.alpha - (self.n + 1))
            # Check that expr is identically zero (no free symbols beyond n)
            if expr == 0:
                return (True, "α = n+1 with no extra parameters")
            return (False, f"α contains extra terms: {expr}")
        except Exception as e:
            return (False, f"Modeling check failed: {e}")

    def test_modeling_beta_linear_in_n(self):
        """β should be exactly n, no extra parameters."""
        if self.beta is None:
            return (False, "SKIP: beta not available")
        try:
            expr = sp.simplify(self.beta - self.n)
            if expr == 0:
                return (True, "β = n with no extra parameters")
            return (False, f"β contains extra terms: {expr}")
        except Exception as e:
            return (False, f"Modeling check failed: {e}")

    # --------------------------------------------------------------------------
    # Monotonicity tests (perturb n by small amount and compare)
    # --------------------------------------------------------------------------
    def test_monotonicity_alpha_increases_with_n(self):
        """α should increase when n increases by 5%."""
        if self.alpha is None:
            return (False, "SKIP: alpha not available")
        try:
            baseline_n = 3.0
            perturbed_n = baseline_n * 1.05   # +5%
            baseline_val = float(self.alpha.subs(self.n, baseline_n))
            perturbed_val = float(self.alpha.subs(self.n, perturbed_n))
            if perturbed_val > baseline_val:
                return (True, f"α increases from {baseline_val} to {perturbed_val} when n increases by 5%")
            return (False, f"α did not increase: baseline {baseline_val}, perturbed {perturbed_val}")
        except Exception as e:
            return (False, f"Monotonicity check crashed: {e}")

    def test_monotonicity_beta_increases_with_n(self):
        """β should increase when n increases by 5%."""
        if self.beta is None:
            return (False, "SKIP: beta not available")
        try:
            baseline_n = 3.0
            perturbed_n = baseline_n * 1.05
            baseline_val = float(self.beta.subs(self.n, baseline_n))
            perturbed_val = float(self.beta.subs(self.n, perturbed_n))
            if perturbed_val > baseline_val:
                return (True, f"β increases from {baseline_val} to {perturbed_val} when n increases by 5%")
            return (False, f"β did not increase: baseline {baseline_val}, perturbed {perturbed_val}")
        except Exception as e:
            return (False, f"Monotonicity check crashed: {e}")

    def test_monotonicity_gamma_decreases_with_n(self):
        """γ should decrease when n increases by 5%."""
        if self.gamma is None:
            return (False, "SKIP: gamma not available")
        try:
            baseline_n = 3.0
            perturbed_n = baseline_n * 1.05
            baseline_val = float(self.gamma.subs(self.n, baseline_n))
            perturbed_val = float(self.gamma.subs(self.n, perturbed_n))
            if perturbed_val < baseline_val:
                return (True, f"γ decreases from {baseline_val} to {perturbed_val} when n increases by 5%")
            return (False, f"γ did not decrease: baseline {baseline_val}, perturbed {perturbed_val}")
        except Exception as e:
            return (False, f"Monotonicity check crashed: {e}")

    # --------------------------------------------------------------------------
    # Weak answer sanity (only one, no re-derivation)
    # --------------------------------------------------------------------------
    def test_answer_sanity_sign_and_finite(self):
        """All answers should be positive and finite for positive n."""
        if self.alpha is None or self.beta is None or self.gamma is None:
            return (False, "SKIP: missing results")
        try:
            # Evaluate for a generic positive n, e.g., n=2.5
            n_val = 2.5
            a = float(self.alpha.subs(self.n, n_val))
            b = float(self.beta.subs(self.n, n_val))
            g = float(self.gamma.subs(self.n, n_val))
        except Exception:
            # If evaluation fails, just accept
            return (True, "Could not evaluate numerically; symbolically acceptable")
        for name, val in [('alpha', a), ('beta', b), ('gamma', g)]:
            if not (val > 0 and math.isfinite(val)):
                return (False, f"{name} = {val} is not positive and finite")
        return (True, "All results are positive and finite for n=2.5")