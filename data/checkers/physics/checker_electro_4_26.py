import sympy as sp
import math

class PhysicsChecker:
    """
    Auditor for plane-wave electromagnetic problem:
    derive H field and refractive index n from Maxwell's equations.
    """
    def __init__(self, solution_instance):
        # Extract results (primary outputs)
        self.results_H = getattr(solution_instance, 'H', None)
        if self.results_H is None and hasattr(solution_instance, 'results'):
            self.results_H = solution_instance.results.get('H', None)
        self.results_n = getattr(solution_instance, 'n', None)
        if self.results_n is None and hasattr(solution_instance, 'results'):
            self.results_n = solution_instance.results.get('n', None)

        # Extract givens – use getattr with None default; no numeric defaults
        self.E = getattr(solution_instance, 'E', None)
        self.E0 = getattr(solution_instance, 'E0', None)
        self.K = getattr(solution_instance, 'K', None)
        self.n = getattr(solution_instance, 'n', None)          # may be symbolic ref
        self.omega = getattr(solution_instance, 'omega', None)
        self.c = getattr(solution_instance, 'c', None)
        self.mu = getattr(solution_instance, 'mu', None)
        self.epsilon = getattr(solution_instance, 'epsilon', None)
        self.sigma = getattr(solution_instance, 'sigma', None)
        self.mu0 = getattr(solution_instance, 'mu0', None)
        self.epsilon0 = getattr(solution_instance, 'epsilon0', None)

        # Convert string expressions to sympy if possible
        self._convert_to_sympy()

    def _convert_to_sympy(self):
        """Try to convert LaTeX strings to sympy; leave as is if not possible."""
        for attr in ['results_H', 'results_n', 'E', 'E0', 'K', 'n',
                      'omega', 'c', 'mu', 'epsilon', 'sigma', 'mu0', 'epsilon0']:
            val = getattr(self, attr, None)
            if isinstance(val, str):
                try:
                    setattr(self, attr, sp.sympify(val))
                except (sp.SympifyError, TypeError):
                    pass  # keep as string

    def _get_vector(self, obj):
        """If obj is a vector (sympy Matrix or list/tuple), return Matrix; else None."""
        if isinstance(obj, (sp.Matrix, list, tuple)):
            return sp.Matrix(obj)
        return None

    # ------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------
    def test_interface_H_exists(self):
        if self.results_H is None:
            return False, "Missing required result 'H'"
        return True, "H result is present"

    def test_interface_n_exists(self):
        if self.results_n is None:
            return False, "Missing required result 'n'"
        return True, "n result is present"

    def test_interface_H_type(self):
        if self.results_H is None:
            return False, "SKIP: H not available"
        return True, f"H result type: {type(self.results_H).__name__}"

    def test_interface_n_type(self):
        if self.results_n is None:
            return False, "SKIP: n not available"
        return True, f"n result type: {type(self.results_n).__name__}"

    # ------------------------------------------------------------
    # Invariant tests (theorem‑based physical sanity)
    # ------------------------------------------------------------
    def test_invariant_H_orthogonal_to_K(self):
        """H · K = 0 from Maxwell's equations."""
        H_vec = self._get_vector(self.results_H)
        K_vec = self._get_vector(self.K)
        if H_vec is None or K_vec is None:
            return False, "SKIP: H or K not available as vectors"
        if H_vec.shape != K_vec.shape:
            return False, "Vector dimensions mismatch"
        dot = H_vec.dot(K_vec)
        if sp.simplify(dot) != 0:
            return False, f"H·K simplified to {dot}, expected 0"
        return True, "H is orthogonal to K"

    def test_invariant_H_orthogonal_to_E(self):
        """H · E = 0 from Maxwell's equations."""
        H_vec = self._get_vector(self.results_H)
        E_vec = self._get_vector(self.E)
        if H_vec is None or E_vec is None:
            return False, "SKIP: H or E not available as vectors"
        if H_vec.shape != E_vec.shape:
            return False, "Vector dimensions mismatch"
        dot = H_vec.dot(E_vec)
        if sp.simplify(dot) != 0:
            return False, f"H·E simplified to {dot}, expected 0"
        return True, "H is orthogonal to E"

    def test_invariant_n_dispersion_relation(self):
        """
        Check that (n ω/c)^2 = ω^2 μ (ε - i σ/ω) holds analytically or numerically.
        """
        if self.results_n is None:
            return False, "SKIP: n not available"
        if any(v is None for v in [self.omega, self.c, self.mu, self.epsilon, self.sigma]):
            return False, "SKIP: missing required medium parameters"

        # Build symbolic representation with fresh symbols for substitution
        omega = sp.Symbol('omega', positive=True, real=True)
        c = sp.Symbol('c', positive=True, real=True)
        mu = sp.Symbol('mu', positive=True, real=True)
        epsilon = sp.Symbol('epsilon', positive=True, real=True)
        sigma = sp.Symbol('sigma', real=True)

        n_sym = self.results_n if isinstance(self.results_n, sp.Basic) else sp.Symbol('n')
        lhs = (n_sym * omega / c) ** 2
        rhs = omega**2 * mu * (epsilon - sp.I * sigma / omega)
        expr = sp.simplify(lhs - rhs)

        # Check analytic identity
        if expr == 0:
            return True, "Dispersion relation holds analytically"

        # Fallback: numeric substitution
        subs = {}
        if self.omega is not None and not isinstance(self.omega, sp.Basic):
            subs[omega] = self.omega
        if self.c is not None and not isinstance(self.c, sp.Basic):
            subs[c] = self.c
        if self.mu is not None and not isinstance(self.mu, sp.Basic):
            subs[mu] = self.mu
        if self.epsilon is not None and not isinstance(self.epsilon, sp.Basic):
            subs[epsilon] = self.epsilon
        if self.sigma is not None and not isinstance(self.sigma, sp.Basic):
            subs[sigma] = self.sigma
        if subs:
            try:
                numeric_val = sp.N(expr.subs(subs))
                if abs(numeric_val) < 1e-10:
                    return True, "Dispersion relation numerically satisfied (tolerance 1e-10)"
            except Exception:
                pass

        return False, f"Dispersion relation not satisfied; expression = {expr}"

    def test_invariant_transverse_condition(self):
        """E0 · K = 0 (transverse wave)."""
        E0_vec = self._get_vector(self.E0)
        K_vec = self._get_vector(self.K)
        if E0_vec is None or K_vec is None:
            return False, "SKIP: E0 and K not available as vectors"
        if E0_vec.shape != K_vec.shape:
            return False, "Vector dimensions mismatch"
        dot = E0_vec.dot(K_vec)
        if sp.simplify(dot) != 0:
            return False, f"E0·K = {dot}, expected 0"
        return True, "Transverse condition holds: E0 perpendicular to K"

    # ------------------------------------------------------------
    # Modeling/trace tests (consistency with explanation steps)
    # ------------------------------------------------------------
    def test_modeling_H_from_Faraday(self):
        """
        From Faraday's law: curl E = -∂B/∂t.
        For the given plane wave: H = (n/(μ c)) K × E.
        Check that provided H matches this derived form.
        """
        if self.results_H is None:
            return False, "SKIP: H not available"
        if any(v is None for v in [self.n, self.mu, self.c, self.K, self.E]):
            return False, "SKIP: missing required quantities (n, mu, c, K, E)"

        # Build expected expression symbolically
        n = self.n if isinstance(self.n, sp.Basic) else sp.Symbol('n')
        mu = self.mu if isinstance(self.mu, sp.Basic) else sp.Symbol('mu')
        c = self.c if isinstance(self.c, sp.Basic) else sp.Symbol('c')
        K_vec = self._get_vector(self.K)
        E_vec = self._get_vector(self.E)
        if K_vec is None or E_vec is None:
            return False, "SKIP: K or E not vectorial"
        expected_H = (n / (mu * c)) * K_vec.cross(E_vec)

        actual_H = self._get_vector(self.results_H)
        if actual_H is None:
            return False, "Provided H is not a vector expression"
        if sp.simplify(actual_H - expected_H) != sp.zeros(actual_H.shape[0], 1):
            return False, "H does not match expected expression from Faraday's law"
        return True, "H derivation from Faraday consistent"

    def test_modeling_K_unit_vector(self):
        """
        The problem statement and explanation steps require that K is a unit vector.
        Check |K| = 1.
        """
        K_vec = self._get_vector(self.K)
        if K_vec is None:
            return False, "SKIP: K not available as vector"
        norm_sq = K_vec.dot(K_vec)
        # Simplify to see if it's identically 1
        try:
            simplified = sp.simplify(norm_sq)
            if simplified == 1:
                return True, "K is a unit vector"
            # Fallback: numeric substitution if possible
            if all(isinstance(v, (int, float)) for v in K_vec):
                norm = math.sqrt(sum(v**2 for v in K_vec))
                if abs(norm - 1.0) < 1e-10:
                    return True, "K is numerically unit"
            return False, f"K magnitude squared = {simplified}, expected 1"
        except Exception:
            return False, "Could not simplify magnitude of K"

    # ------------------------------------------------------------
    # Monotonicity tests (must perturb an input and compare)
    # ------------------------------------------------------------
    def test_monotonicity_imaginary_n_increases_with_sigma(self):
        """
        Im(n) should become more negative (magnitude increases) when sigma increases.
        Perturb sigma by +10% and compare baseline Im(n) with perturbed Im(n).
        """
        numeric = all(
            isinstance(getattr(self, a), (int, float))
            for a in ['sigma', 'epsilon', 'omega', 'mu', 'mu0', 'epsilon0']
        )
        if not numeric or self.results_n is None:
            return False, "SKIP: insufficient numeric parameters or n result missing"

        sigma_base = self.sigma
        sigma_pert = sigma_base * 1.1   # +10%

        # Build expression for n using the analytical formula from explanation steps
        n_expr = sp.sqrt(self.mu * self.epsilon / (2 * self.mu0 * self.epsilon0)) * (
            sp.sqrt(1 + sp.sqrt(1 + (self.sigma / (self.epsilon * self.omega))**2)) -
            sp.I * sp.sqrt(sp.sqrt(1 + (self.sigma / (self.epsilon * self.omega))**2) - 1)
        )
        try:
            n_base = complex(sp.N(n_expr.subs({sp.Symbol('sigma'): sigma_base})))
            n_pert = complex(sp.N(n_expr.subs({sp.Symbol('sigma'): sigma_pert})))
        except Exception as e:
            return False, f"Numeric evaluation failed: {e}"

        im_base = n_base.imag
        im_pert = n_pert.imag

        # For a passive conductor, Im(n) ≤ 0 and becomes more negative with increasing sigma.
        if im_base <= 0 and im_pert < im_base:
            return True, (
                f"Baseline Im(n) = {im_base:.6f}, with +10% sigma Im(n) = {im_pert:.6f}. "
                "Monotonicity holds: Im(n) becomes more negative."
            )
        else:
            return False, (
                f"Baseline Im(n) = {im_base:.6f}, perturbed Im(n) = {im_pert:.6f}. "
                "Expected to become more negative (im_pert < im_base)."
            )

    def test_monotonicity_real_n_decreases_with_sigma(self):
        """
        Re(n) should decrease when sigma increases (dispersion).
        Perturb sigma by +10% and compare baseline Re(n) with perturbed Re(n).
        """
        numeric = all(
            isinstance(getattr(self, a), (int, float))
            for a in ['sigma', 'epsilon', 'omega', 'mu', 'mu0', 'epsilon0']
        )
        if not numeric or self.results_n is None:
            return False, "SKIP: insufficient numeric parameters or n result missing"

        sigma_base = self.sigma
        sigma_pert = sigma_base * 1.1

        n_expr = sp.sqrt(self.mu * self.epsilon / (2 * self.mu0 * self.epsilon0)) * (
            sp.sqrt(1 + sp.sqrt(1 + (self.sigma / (self.epsilon * self.omega))**2)) -
            sp.I * sp.sqrt(sp.sqrt(1 + (self.sigma / (self.epsilon * self.omega))**2) - 1)
        )
        try:
            n_base = complex(sp.N(n_expr.subs({sp.Symbol('sigma'): sigma_base})))
            n_pert = complex(sp.N(n_expr.subs({sp.Symbol('sigma'): sigma_pert})))
        except Exception as e:
            return False, f"Numeric evaluation failed: {e}"

        re_base = n_base.real
        re_pert = n_pert.real

        if re_base > 0 and re_pert < re_base:
            return True, (
                f"Baseline Re(n) = {re_base:.6f}, with +10% sigma Re(n) = {re_pert:.6f}. "
                "Monotonicity holds: Re(n) decreases."
            )
        else:
            return False, (
                f"Baseline Re(n) = {re_base:.6f}, perturbed Re(n) = {re_pert:.6f}. "
                "Expected decreasing Re(n) for increasing sigma."
            )

    # ------------------------------------------------------------
    # Weak answer sanity (only one allowed)
    # ------------------------------------------------------------
    def test_answer_sanity_simple(self):
        """Weak check: H and n are present and finite (not NaN)."""
        if self.results_H is None:
            return False, "H is missing"
        if self.results_n is None:
            return False, "n is missing"
        # Attempt numeric evaluation if scalar
        if isinstance(self.results_H, (int, float)) and math.isnan(self.results_H):
            return False, "H is NaN"
        if isinstance(self.results_n, (int, float)) and math.isnan(self.results_n):
            return False, "n is NaN"
        return True, "Basic sanity: H and n present and finite"

    # ------------------------------------------------------------
    # Main test runner
    # ------------------------------------------------------------
    def run_all_tests(self):
        """Discover all test_* methods, run them, return (all_pass, logs)."""
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
        return all_pass, logs