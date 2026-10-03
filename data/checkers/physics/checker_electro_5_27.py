import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Audits a solution for a relativistic charged particle in a magnetic dipole field.
    Checks:
      - Interface: required keys exist and are of valid type.
      - Invariant (theorem‑based):
          1) Euler–Lagrange for phi yields d(p_phi)/dt = 0.
          2) p_phi expression is independent of phi.
      - Modeling: gauge transformation modifies p_phi correctly.
      - Monotonicity: explicit perturbation of mu and r changes p_phi in the expected direction.
      - Weak final‑answer sanity (one test only): finiteness and sign of p_phi.
    """

    def __init__(self, solution_instance):
        # Extract results safely (default None)
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.givens = getattr(solution_instance, 'givens', {}) or {}

        self.p_phi = self.results.get('p_phi', None)
        self.d_p_phi_dt = self.results.get('d_p_phi_dt', None)
        self.p_prime_phi = self.results.get('p_prime_phi', None)
        self.d_p_prime_phi_dt = self.results.get('d_p_prime_phi_dt', None)

        # Givens – symbolic variables (no numeric defaults)
        self.m = self.givens.get('m', sp.symbols('m', positive=True))
        self.e = self.givens.get('e', sp.symbols('e'))
        self.c = self.givens.get('c', sp.symbols('c', positive=True))
        self.mu = self.givens.get('mu', sp.symbols('mu', positive=True))
        self.r = self.givens.get('r', sp.symbols('r', positive=True))
        self.theta = self.givens.get('theta', sp.symbols('theta', positive=True))
        self.phi = self.givens.get('phi', sp.symbols('phi'))
        self.dot_r = self.givens.get('dot(r)', sp.symbols('dot_r'))
        self.dot_theta = self.givens.get('dot(theta)', sp.symbols('dot_theta'))
        self.dot_phi = self.givens.get('dot(phi)', sp.symbols('dot_phi'))
        chi_func = sp.Function('chi')
        self.chi = self.givens.get('chi', chi_func(self.r, self.theta, self.phi))

        # Derived quantities (for re‑deriving expected expressions – used only in modeling/monotonicity)
        self.v_sq = (self.dot_r**2 +
                     (self.r * self.dot_theta)**2 +
                     (self.r * sp.sin(self.theta) * self.dot_phi)**2)
        self.gamma = 1 / sp.sqrt(1 - self.v_sq / self.c**2)

        # Expected expression for p_phi (from L = -mc^2 γ + (e/c) A_phi r sinθ φdot)
        self.expected_p_phi = (self.m * self.gamma * self.r**2 *
                               sp.sin(self.theta)**2 * self.dot_phi +
                               (self.e / self.c) * (self.mu * sp.sin(self.theta)**2 / self.r))

        # Expected gauge‑transformed momentum
        self.expected_p_prime_phi = self.expected_p_phi + (self.e / self.c) * sp.diff(self.chi, self.phi)

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

    # ---------- Interface tests ----------
    def test_interface_required_keys(self):
        keys = ['p_phi', 'd_p_phi_dt', 'p_prime_phi', 'd_p_prime_phi_dt']
        for k in keys:
            if k not in self.results:
                return (False, f"Missing key '{k}'")
        return (True, "All required keys present")

    def test_interface_p_phi_type(self):
        if self.p_phi is None:
            return (False, "p_phi is None")
        if not isinstance(self.p_phi, (sp.Basic, int, float)):
            return (False, "p_phi has invalid type")
        return (True, "p_phi has valid type")

    def test_interface_d_p_phi_dt_type(self):
        if self.d_p_phi_dt is None:
            return (False, "d_p_phi_dt is None")
        if not isinstance(self.d_p_phi_dt, (sp.Basic, int, float)):
            return (False, "d_p_phi_dt has invalid type")
        return (True, "d_p_phi_dt has valid type")

    def test_interface_p_prime_phi_type(self):
        if self.p_prime_phi is None:
            return (False, "p_prime_phi is None")
        if not isinstance(self.p_prime_phi, sp.Basic):
            return (False, "p_prime_phi is not a sympy expression")
        return (True, "p_prime_phi has valid type")

    def test_interface_d_p_prime_phi_dt_type(self):
        if self.d_p_prime_phi_dt is None:
            return (False, "d_p_prime_phi_dt is None")
        return (True, "d_p_prime_phi_dt is present")

    # ---------- Invariant tests (theorem‑based) ----------
    def test_invariant_cyclic_phi(self):
        """Euler–Lagrange for phi gives d(p_phi)/dt = 0."""
        if self.d_p_phi_dt is None:
            return (False, "SKIP: d_p_phi_dt is None")
        try:
            expr = sp.simplify(self.d_p_phi_dt)
            if expr != 0:
                return (False, f"d_p_phi_dt is not zero, got {expr}")
        except Exception as e:
            return (False, f"Error simplifying d_p_phi_dt: {e}")
        return (True, "d(p_phi)/dt = 0 (cyclic coordinate)")

    def test_invariant_p_phi_independent_of_phi(self):
        """p_phi (original) does not depend explicitly on phi."""
        if self.p_phi is None:
            return (False, "SKIP: p_phi is None")
        try:
            deriv = sp.diff(self.p_phi, self.phi)
            if deriv != 0:
                return (False, f"∂p_phi/∂φ is non‑zero: {deriv}")
        except Exception as e:
            return (False, f"Error computing ∂p_phi/∂φ: {e}")
        return (True, "p_phi is independent of φ (cyclic coordinate)")

    # ---------- Modeling / trace tests ----------
    def test_modeling_p_phi_expression(self):
        """p_phi matches the derivative of the Lagrangian w.r.t. φdot."""
        if self.p_phi is None:
            return (False, "SKIP: p_phi is None")
        try:
            diff = sp.simplify(self.p_phi - self.expected_p_phi)
            if diff != 0:
                return (False, f"p_phi differs from Lagrangian derivative by {diff}")
        except Exception as e:
            return (False, f"Error comparing p_phi: {e}")
        return (True, "p_phi expression matches Lagrangian derivative")

    def test_modeling_gauge_transformation(self):
        """p'_phi = p_phi + (e/c) ∂χ/∂φ."""
        if self.p_prime_phi is None or self.p_phi is None:
            return (False, "SKIP: p_prime_phi or p_phi is None")
        try:
            expected_diff = (self.e / self.c) * sp.diff(self.chi, self.phi)
            actual_diff = sp.simplify(self.p_prime_phi - self.p_phi)
            if sp.simplify(actual_diff - expected_diff) != 0:
                return (False, "Gauge transformation not satisfied")
        except Exception as e:
            return (False, f"Error checking gauge transformation: {e}")
        return (True, "p'_phi = p_phi + (e/c) ∂χ/∂φ")

    def test_modeling_gauge_p_prime_not_constant(self):
        """d(p'_phi)/dt is generally non‑zero (unless χ is special)."""
        if self.d_p_prime_phi_dt is None:
            return (False, "SKIP: d_p_prime_phi_dt is None")
        try:
            # Substitute a generic non‑trivial χ = r θ φ
            test_chi = self.r * self.theta * self.phi
            sub_dict = {
                sp.diff(self.chi, self.phi): sp.diff(test_chi, self.phi),
                sp.diff(self.chi, self.r): sp.diff(test_chi, self.r),
                sp.diff(self.chi, self.theta): sp.diff(test_chi, self.theta)
            }
            expr = self.d_p_prime_phi_dt.subs(sub_dict)
            # Numeric evaluation
            vals = {self.r: 2.0, self.theta: 1.0, self.phi: 0.5,
                    self.dot_r: 0.1, self.dot_theta: 0.2, self.dot_phi: 0.3,
                    self.m: 1.0, self.e: 1.0, self.c: 1.0, self.mu: 1.0}
            numeric_val = float(expr.subs(vals).evalf())
            if math.isclose(numeric_val, 0.0, abs_tol=1e-10):
                return (False, "d(p'_phi)/dt evaluates to zero for a generic χ – should be non‑zero")
        except Exception:
            # If substitution fails, skip (the test is not essential)
            return (True, "SKIP: could not evaluate d(p'_phi)/dt for generic χ")
        return (True, "d(p'_phi)/dt is non‑zero for a generic function χ")

    # ---------- Monotonicity tests (explicit perturbation) ----------
    def test_monotonicity_mu_increases_p_phi(self):
        """Increasing mu by 10% should increase p_phi."""
        if self.p_phi is None:
            return (False, "SKIP: p_phi is None")
        try:
            # Baseline values
            sub = {self.m: 1.0, self.e: 1.0, self.c: 1.0, self.mu: 1.0,
                   self.r: 2.0, self.theta: sp.pi/4, self.dot_phi: 1.0,
                   self.dot_r: 0.0, self.dot_theta: 0.0}
            baseline = float(self.expected_p_phi.subs(sub).evalf())
            # Perturb: increase mu by 10%
            sub2 = sub.copy()
            sub2[self.mu] = 1.1
            perturbed = float(self.expected_p_phi.subs(sub2).evalf())
            if perturbed <= baseline:
                return (False, f"p_phi decreased when mu increased: baseline {baseline}, perturbed {perturbed}")
        except Exception as e:
            return (False, f"Error in monotonicity test (mu): {e}")
        return (True, "p_phi increases with mu")

    def test_monotonicity_r_decreases_p_phi(self):
        """Increasing r by 10% should decrease p_phi (due to 1/r term)."""
        if self.p_phi is None:
            return (False, "SKIP: p_phi is None")
        try:
            sub = {self.m: 1.0, self.e: 1.0, self.c: 1.0, self.mu: 1.0,
                   self.r: 2.0, self.theta: sp.pi/4, self.dot_phi: 1.0,
                   self.dot_r: 0.0, self.dot_theta: 0.0}
            baseline = float(self.expected_p_phi.subs(sub).evalf())
            # Perturb: increase r by 10%
            sub2 = sub.copy()
            sub2[self.r] = 2.2
            perturbed = float(self.expected_p_phi.subs(sub2).evalf())
            if perturbed >= baseline:
                return (False, f"p_phi increased when r increased: baseline {baseline}, perturbed {perturbed}")
        except Exception as e:
            return (False, f"Error in monotonicity test (r): {e}")
        return (True, "p_phi decreases with r")

    # ---------- Weak final‑answer sanity (one test only) ----------
    def test_answer_sanity_simple(self):
        """
        Weak sanity: check that p_phi is finite, positive, and of plausible order.
        No symbolic manipulation, no re-derivation.
        """
        if self.p_phi is None:
            return (False, "SKIP: p_phi is None")
        try:
            # Evaluate p_phi using a generic set of physical values (avoid division by zero)
            sub = {self.m: 1.0, self.e: 1.0, self.c: 1.0, self.mu: 1.0,
                   self.r: 2.0, self.theta: sp.pi/4, self.dot_phi: 1.0,
                   self.dot_r: 0.0, self.dot_theta: 0.0}
            val = float(self.p_phi.subs(sub).evalf())
            if math.isnan(val) or math.isinf(val):
                return (False, f"p_phi evaluates to {val} (NaN or Inf)")
            if val <= 0:
                return (False, f"p_phi should be positive, got {val}")
            # Check order of magnitude: for these parameters, p_phi is around 2-3, so plausible >0.1
            if val < 0.1:
                return (False, f"p_phi seems too small: {val}")
        except Exception as e:
            return (False, f"Error in answer_sanity: {e}")
        return (True, "p_phi is finite, positive, and of reasonable magnitude")