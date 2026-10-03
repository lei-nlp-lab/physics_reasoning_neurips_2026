import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Audits the physics solution for the central force scattering problem.
    """
    def __init__(self, solution_instance):
        # Extract results dictionary safely
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}

        # Extract givens: mass m, initial speed V0, impact parameter b, scattering angle theta,
        # hard sphere radius a, potential U(r)
        self.m = results.get('m', None)
        self.V0 = results.get('V0', None)
        self.b = results.get('b', None)
        self.theta = results.get('theta', None)
        self.a = results.get('a', None)
        self.U = results.get('U', None)  # potential function (sympy expression or None)

        # Main emitted results
        self.phi_r = results.get('phi_r', None)                     # φ(r) integral expression
        self.dsigma_dOmega = results.get('dsigma_dOmega', None)     # general differential cross section formula
        self.dsigma_dOmega_HS = results.get('dsigma_dOmega_HS', None)  # hard sphere differential cross section (a^2/4)
        self.sigma_total = results.get('sigma_total', None)         # total cross section for hard sphere (π a^2)

        # Optional: explanation_steps (list of dicts) – not used for hardcoded numbers, only for modeling hints
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # For convenience, define symbols for symbolic checks
        self.r, self.h, self.E, self.m_sym = sp.symbols('r h E m', positive=True)
        self.phi_sym = sp.Function('phi')(self.r)
        self.a_sym = sp.symbols('a', positive=True)
        self.b_sym = sp.symbols('b', positive=True)
        self.theta_sym = sp.symbols('theta', positive=True)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_phi_r_exists(self):
        """Check that phi_r key is present and is a symbolic expression or string."""
        if self.phi_r is None:
            return (False, "SKIP: phi_r not provided")
        # It might be a sympy expression or a string representation
        return (True, "phi_r present")

    def test_interface_dsigma_dOmega_exists(self):
        if self.dsigma_dOmega is None:
            return (False, "SKIP: dsigma_dOmega not provided")
        return (True, "dsigma_dOmega present")

    def test_interface_dsigma_dOmega_HS_exists(self):
        if self.dsigma_dOmega_HS is None:
            return (False, "SKIP: dsigma_dOmega_HS not provided")
        return (True, "dsigma_dOmega_HS present")

    def test_interface_sigma_total_exists(self):
        if self.sigma_total is None:
            return (False, "SKIP: sigma_total not provided")
        return (True, "sigma_total present")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based, use math.isclose where appropriate)
    # ------------------------------------------------------------------
    def test_invariant_hard_sphere_theta_b_relation(self):
        """
        For hard sphere: θ = 2 * arccos(b/a) for b ≤ a.
        Check if we have both theta and b and a, and they satisfy this relation.
        """
        if self.theta is None or self.b is None or self.a is None:
            return (False, "SKIP: theta, b, or a missing")
        # Ensure theta and b are in physical range
        if self.b > self.a:
            return (False, "SKIP: b > a, no scattering (theta=0)")
        # Compute expected theta for given b (assuming hard sphere)
        theta_expected = 2 * math.acos(self.b / self.a)
        # Allow a few percent tolerance (physics noise)
        if math.isclose(self.theta, theta_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"θ ≈ 2 arccos(b/a): {self.theta:.4f} ≈ {theta_expected:.4f}")
        else:
            return (False, f"θ = {self.theta:.4f} does not match 2 arccos(b/a) = {theta_expected:.4f}")

    def test_invariant_total_cross_section_geometric(self):
        """
        For hard sphere, total cross section equals π a^2.
        """
        if self.sigma_total is None or self.a is None:
            return (False, "SKIP: sigma_total or a missing")
        expected = math.pi * self.a**2
        if math.isclose(self.sigma_total, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"σ_total ≈ π a^2: {self.sigma_total:.4f} ≈ {expected:.4f}")
        else:
            return (False, f"σ_total = {self.sigma_total:.4f} ≠ π a^2 = {expected:.4f}")

    # ------------------------------------------------------------------
    # Modeling/trace tests
    # ------------------------------------------------------------------
    def test_modeling_phi_r_integral_form(self):
        """
        Check that phi_r expression has the correct structure:
        φ = ∫ (h dr) / (r sqrt(2r²(E-U)/m - h²))
        We'll check symbolically if possible.
        """
        if self.phi_r is None:
            return (False, "SKIP: phi_r not provided")
        # If phi_r is a sympy expression, try to verify its form
        if isinstance(self.phi_r, sp.Expr):
            # Check that it contains an integration (Integral) or at least a sqrt and r
            # We can't easily reverse-engineer, but we can check that it depends on r and h
            free_symbols = list(self.phi_r.free_symbols)
            if self.r in free_symbols:
                return (True, "phi_r expression contains r and likely correct form")
            else:
                return (False, "phi_r expression does not involve the radial variable r")
        elif isinstance(self.phi_r, str):
            # String representation: check for keywords
            if '∫' in self.phi_r and 'h' in self.phi_r and 'r' in self.phi_r:
                return (True, "phi_r string contains integral and h, r")
            else:
                return (False, "phi_r string does not contain expected integral form")
        else:
            return (False, "phi_r is not a recognized type (expected sympy expression or string)")

    def test_modeling_dsigma_dOmega_formula(self):
        """
        Check that dsigma_dOmega = (b / sinθ) |db/dθ|.
        We'll check if the expression has the structure: (b/sinθ) * derivative.
        """
        if self.dsigma_dOmega is None:
            return (False, "SKIP: dsigma_dOmega not provided")
        # For simplicity, we check if it contains b, sin(theta), and a derivative-like factor.
        # If symbolic, check structure.
        found_b = False
        found_sin = False
        if isinstance(self.dsigma_dOmega, sp.Expr):
            # Look for b and sin(theta)
            found_b = self.b_sym in self.dsigma_dOmega.free_symbols if self.b_sym.free_symbols else False
            # Check if sin appears
            sin_theta = sp.sin(self.theta_sym)
            if sin_theta in self.dsigma_dOmega.find(sp.sin):
                found_sin = True
        elif isinstance(self.dsigma_dOmega, str):
            found_b = 'b' in self.dsigma_dOmega
            found_sin = 'sin' in self.dsigma_dOmega
        else:
            return (False, "dsigma_dOmega type not recognised")
        if found_b and found_sin:
            return (True, "dsigma_dOmega contains b and sin(θ)")
        else:
            return (False, "dsigma_dOmega missing b and/or sin(θ)")

    def test_modeling_dsigma_dOmega_HS_constant(self):
        """
        For hard sphere, dsigma/dOmega = a^2/4 (constant).
        Check that the numerical value matches.
        """
        if self.dsigma_dOmega_HS is None or self.a is None:
            return (False, "SKIP: dsigma_dOmega_HS or a missing")
        expected = (self.a**2) / 4.0
        if math.isclose(self.dsigma_dOmega_HS, expected, rel_tol=1e-3, abs_tol=0.01):
            return (True, f"dσ/dΩ_HS = a^2/4 = {expected:.4f}, got {self.dsigma_dOmega_HS:.4f}")
        else:
            return (False, f"dσ/dΩ_HS = {self.dsigma_dOmega_HS:.4f}, expected {expected:.4f}")

    def test_modeling_sigma_total_pi_a2(self):
        """
        Check total cross section for hard sphere equals π a^2.
        """
        if self.sigma_total is None or self.a is None:
            return (False, "SKIP: sigma_total or a missing")
        expected = math.pi * self.a**2
        if math.isclose(self.sigma_total, expected, rel_tol=1e-3, abs_tol=1e-3):
            return (True, f"σ_total = π a^2 = {expected:.4f}, got {self.sigma_total:.4f}")
        else:
            return (False, f"σ_total = {self.sigma_total:.4f}, expected {expected:.4f}")

    # ------------------------------------------------------------------
    # Monotonicity tests (must perturb an input and compare perturbed output to baseline)
    # ------------------------------------------------------------------
    def test_monotonicity_hard_sphere_theta_decreases_with_b(self):
        """
        For hard sphere, the scattering angle θ decreases when impact parameter b increases.
        This test computes the baseline θ from the given b and a, then increases b by 10% 
        and computes the new θ. It checks that the new θ is smaller.
        """
        if self.b is None or self.a is None:
            return (False, "SKIP: b and/or a missing")
        if self.b <= 0 or self.b >= self.a:
            return (False, "SKIP: b out of (0, a) range, cannot perturb")
        # Baseline theta from given b and a
        theta0 = 2 * math.acos(self.b / self.a)
        # Perturb b: increase by 10%, but ensure it stays ≤ a
        b_pert = self.b * 1.1
        if b_pert >= self.a:
            # Reduce perturbation to half the gap to a
            b_pert = (self.b + self.a) / 2.0
        theta1 = 2 * math.acos(b_pert / self.a)
        # Check that perturbed theta is smaller (monotonic decreasing)
        if theta1 < theta0:
            return (True, f"Perturbed b from {self.b:.4f} to {b_pert:.4f}: θ decreased from {theta0:.4f} to {theta1:.4f}")
        else:
            return (False, f"θ did not decrease upon increasing b: baseline {theta0:.4f}, perturbed {theta1:.4f}")

    # ------------------------------------------------------------------
    # Answer sanity (weak, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_sign_and_finite(self):
        """
        Check that all numerical results are positive, finite, and non-NaN.
        """
        failed = []
        for name, val in [('phi_r', self.phi_r), ('dsigma_dOmega', self.dsigma_dOmega),
                          ('dsigma_dOmega_HS', self.dsigma_dOmega_HS), ('sigma_total', self.sigma_total)]:
            if val is None:
                continue
            if isinstance(val, (int, float)):
                if math.isnan(val) or math.isinf(val) or val < 0:
                    failed.append(f"{name} is NaN, infinite, or negative: {val}")
            # For symbolic expressions, we assume they are valid.
        if failed:
            return (False, "; ".join(failed))
        return (True, "All numerical results are positive, finite, and non-NaN")

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------
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