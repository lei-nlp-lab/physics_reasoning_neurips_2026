import math
import traceback

class PhysicsChecker:
    """
    Audits the solution for the spherical shell potential problem.
    Tests focus on Laplace's equation, boundary conditions, and charge density.
    """
    def __init__(self, solution_instance):
        # Extract results dictionary and other attributes safely
        self.results = getattr(solution_instance, 'results', {})
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.reasoning_plan = getattr(solution_instance, 'reasoning_plan', None)
        self.givens = getattr(solution_instance, 'givens', None)

        # Extract expected result keys (use None if missing)
        self.A_n = self.results.get('A_n', None)
        self.B_n = self.results.get('B_n', None)
        self.sigma_general = self.results.get('sigma_general', None)
        self.A0 = self.results.get('A0', None)
        self.A2 = self.results.get('A2', None)
        self.sigma = self.results.get('sigma', None)

        # Extract problem constants (if provided)
        self.R = getattr(solution_instance, 'R', None)
        self.V0 = getattr(solution_instance, 'V0', None)
        self.epsilon0 = getattr(solution_instance, 'epsilon0', None)

        # If not directly available, try to parse from givens dict
        if self.R is None and self.givens is not None:
            self.R = self.givens.get('R', None)
        if self.V0 is None and self.givens is not None:
            self.V0 = self.givens.get('V0', None)
        if self.epsilon0 is None and self.givens is not None:
            self.epsilon0 = self.givens.get('epsilon0', None)

        # Store Legendre polynomials as functions
        def P0(x): return 1.0
        def P1(x): return x
        def P2(x): return (3*x*x - 1)/2.0
        self.legendre = {0: P0, 1: P1, 2: P2}  # only first three needed

    # -----------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------
    def _get_V_at_theta(self, theta):
        """Return V(theta) = V0 * cos^2(theta) if V0 is known, else None."""
        if self.V0 is None:
            return None
        return self.V0 * (math.cos(theta) ** 2)

    def _get_P_n(self, n, x):
        """Return P_n(x) for n=0,1,2."""
        if n in self.legendre:
            return self.legendre[n](x)
        else:
            return None

    def _compute_V_in(self, r, theta):
        """Compute V_in(r,theta) = A0 + A2 * r^2 * P2(cosθ) (only n=0,2 nonzero)."""
        if self.A0 is None or self.A2 is None:
            return None
        costheta = math.cos(theta)
        return self.A0 + self.A2 * (r ** 2) * self._get_P_n(2, costheta)

    def _compute_V_out(self, r, theta):
        """Compute V_out(r,theta) = B0/r + B2/r^3 * P2(cosθ) with B0=R*A0, B2=R^5*A2."""
        if self.A0 is None or self.A2 is None or self.R is None:
            return None
        B0 = self.R * self.A0
        B2 = (self.R ** 5) * self.A2
        costheta = math.cos(theta)
        return B0 / r + B2 / (r ** 3) * self._get_P_n(2, costheta)

    # -----------------------------------------------------------------
    # Interface tests
    # -----------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all six required keys are present in results."""
        required = ['A_n', 'B_n', 'sigma_general', 'A0', 'A2', 'sigma']
        missing = [k for k in required if self.results.get(k) is None]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required keys present")

    def test_interface_constants_available(self):
        """Check that R, V0, and epsilon0 are available for numerical tests."""
        if self.R is None or self.V0 is None or self.epsilon0 is None:
            return (False, "SKIP: one or more constants (R, V0, epsilon0) missing")
        return (True, "Constants available")

    # -----------------------------------------------------------------
    # Invariant tests (physical sanity, use math.isclose)
    # -----------------------------------------------------------------
    def test_invariant_boundary_continuity(self):
        """At r=R, V_in(R,theta) should equal V_out(R,theta) and equal V(theta)."""
        if None in (self.R, self.V0, self.A0, self.A2):
            return (False, "SKIP: missing constants or coefficients")
        # Check at three theta values
        thetas = [0.0, math.pi/4, math.pi/2]
        for theta in thetas:
            Vin = self._compute_V_in(self.R, theta)
            Vout = self._compute_V_out(self.R, theta)
            Vtheta = self._get_V_at_theta(theta)
            if None in (Vin, Vout, Vtheta):
                return (False, "SKIP: could not compute potentials")
            if not math.isclose(Vin, Vtheta, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"V_in({theta}) = {Vin} != V(theta) = {Vtheta}")
            if not math.isclose(Vout, Vtheta, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"V_out({theta}) = {Vout} != V(theta) = {Vtheta}")
        return (True, "Boundary continuity holds at sampled points")

    def test_invariant_charge_density_jump(self):
        """Check that sigma = ε0 (∂V_in/∂r - ∂V_out/∂r) at r=R.
        For part (b): sigma = (ε0 V0/(3R)) * (1 + 5 P_2(cosθ)).
        """
        if None in (self.R, self.V0, self.epsilon0, self.A0, self.A2, self.sigma):
            return (False, "SKIP: missing data for charge density test")
        # Compute expected sigma from coefficients: sigma = ε0 Σ (2n+1) A_n R^{n-1} P_n
        # For n=0: 1 * A0 * R^{-1} * P0 = A0/R
        # For n=2: 5 * A2 * R^{1} * P2 = 5 * A2 * R * P2
        thetas = [0.0, math.pi/4, math.pi/2]
        for theta in thetas:
            costheta = math.cos(theta)
            expected_sigma = self.epsilon0 * (
                self.A0 / self.R +
                5.0 * self.A2 * self.R * self._get_P_n(2, costheta)
            )
            # Actual sigma: try to evaluate if it is callable or sympy expression
            actual_sigma = None
            if hasattr(self.sigma, 'subs'):
                try:
                    actual_sigma = float(self.sigma.subs({'theta': theta}))
                except:
                    pass
            elif callable(self.sigma):
                actual_sigma = self.sigma(theta)
            else:
                # fallback: assume numeric constant (not angular-dependent) – skip
                return (False, "SKIP: sigma cannot be evaluated")
            if actual_sigma is None:
                return (False, "SKIP: sigma could not be evaluated")
            if not math.isclose(actual_sigma, expected_sigma, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"sigma(theta={theta}) = {actual_sigma} != expected {expected_sigma}")
        return (True, "Charge density jump relation holds")

    # -----------------------------------------------------------------
    # Modeling/trace tests (check coefficient relations)
    # -----------------------------------------------------------------
    def test_modeling_coefficient_relation_Bn_from_An(self):
        """Check B_n = R^{2n+1} A_n for n=0 and n=2."""
        if None in (self.R, self.A0, self.A2, self.B_n):
            return (False, "SKIP: missing coefficients or R")
        # For n=0: B0 = R^1 * A0
        expected_B0 = self.R * self.A0
        # For n=2: B2 = R^5 * A2
        expected_B2 = (self.R ** 5) * self.A2
        # Need actual B_n as a dict with keys 0 and 2
        if isinstance(self.B_n, dict):
            actual_B0 = self.B_n.get(0)
            actual_B2 = self.B_n.get(2)
        else:
            return (False, "SKIP: B_n is not a dict with keys 0,2")
        if actual_B0 is None or actual_B2 is None:
            return (False, "SKIP: B_n missing entries for n=0,2")
        if not math.isclose(expected_B0, actual_B0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"B0 = {actual_B0} != expected {expected_B0}")
        if not math.isclose(expected_B2, actual_B2, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"B2 = {actual_B2} != expected {expected_B2}")
        return (True, "Coefficient relation holds for n=0 and n=2")

    def test_modeling_specific_coefficients_part_b(self):
        """For V(theta)=V0 cos^2theta, check A0 = V0/3 and A2 = 2V0/(3R^2)."""
        if None in (self.V0, self.R, self.A0, self.A2):
            return (False, "SKIP: missing V0, R, or coefficients")
        expected_A0 = self.V0 / 3.0
        expected_A2 = (2.0 * self.V0) / (3.0 * self.R * self.R)
        if not math.isclose(self.A0, expected_A0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"A0 = {self.A0} != expected {expected_A0}")
        if not math.isclose(self.A2, expected_A2, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"A2 = {self.A2} != expected {expected_A2}")
        return (True, "Coefficients A0 and A2 match analytic values")

    # -----------------------------------------------------------------
    # Monotonicity test (explicit perturbation)
    # -----------------------------------------------------------------
    def test_monotonicity_outside_potential_decays_with_r(self):
        """Perturb r by +10% and check that V_out decreases (monotonic decay)."""
        if None in (self.R, self.A0, self.A2):
            return (False, "SKIP: missing R or coefficients for V_out")
        # Baseline: at theta=0, r = 1.5*R (outside the sphere)
        theta = 0.0
        r_baseline = 1.5 * self.R
        V_baseline = self._compute_V_out(r_baseline, theta)
        if V_baseline is None:
            return (False, "SKIP: could not compute baseline V_out")
        # Perturb r by +10%
        r_perturbed = 1.1 * r_baseline
        V_perturbed = self._compute_V_out(r_perturbed, theta)
        if V_perturbed is None:
            return (False, "SKIP: could not compute perturbed V_out")
        # Compare: potential should decrease as r increases
        if V_perturbed >= V_baseline:
            return (False, f"V_out(r_perturbed) = {V_perturbed} >= V_baseline = {V_baseline}")
        return (True, f"V_out decreases with r: baseline={V_baseline:.6f}, perturbed={V_perturbed:.6f}")

    # -----------------------------------------------------------------
    # Answer sanity test (weak, no math.isclose, no re-derivation)
    # -----------------------------------------------------------------
    def test_answer_sanity_sigma_finite(self):
        """Verify that sigma is finite (not NaN or inf) for typical theta values."""
        if self.sigma is None:
            return (False, "SKIP: sigma not provided")
        thetas = [0.0, math.pi/4, math.pi/2]
        for theta in thetas:
            if hasattr(self.sigma, 'subs'):
                try:
                    val = float(self.sigma.subs({'theta': theta}))
                except:
                    return (False, "SKIP: cannot evaluate sigma")
            elif callable(self.sigma):
                val = self.sigma(theta)
            else:
                return (False, "SKIP: sigma format unknown")
            if not math.isfinite(val):
                return (False, f"sigma(theta={theta}) = {val} is not finite")
        return (True, "sigma is finite at sampled points")

    # -----------------------------------------------------------------
    # Runner
    # -----------------------------------------------------------------
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