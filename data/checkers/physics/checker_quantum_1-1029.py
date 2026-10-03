import math

class PhysicsChecker:
    """Audits the solution for the minimum energy of a simple harmonic oscillator
    using uncertainty relations."""

    def __init__(self, solution_instance):
        # Safely extract results and givens; never use numeric defaults
        self.results = getattr(solution_instance, 'results', None) or {}
        self.givens = getattr(solution_instance, 'givens', None) or {}
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None) or []

        # Key result: minimum energy
        self.E_min = self.results.get('E_min', None)

        # Optional intermediate quantities (may be present for deeper checks)
        self.Delta_x = self.results.get('Delta_x', None)
        self.Delta_p = self.results.get('Delta_p', None)
        self.m = self.results.get('m', None)
        self.omega = self.results.get('omega', None)
        self.hbar = self.results.get('hbar', None)

    # ------------------------------------------------------------------
    # Interface tests: required keys exist
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        if self.E_min is None:
            return (False, "Missing required result key 'E_min'")
        return (True, "E_min present")

    # ------------------------------------------------------------------
    # Theorem-based invariant checks (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_uncertainty_relation(self):
        """Check Δx·Δp = ℏ/2 if both Δx and Δp are provided."""
        if self.Delta_x is None or self.Delta_p is None or self.hbar is None:
            return (False, "SKIP: Δx, Δp, or ℏ not available")
        product = self.Delta_x * self.Delta_p
        expected = self.hbar / 2.0
        if math.isclose(product, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Uncertainty product {product} ≈ {expected}")
        return (False, f"Δx·Δp = {product}, expected {expected}")

    def test_invariant_energy_expression(self):
        """Check that E_min satisfies E = (Δp)^2/(2m) + ½ m ω² (Δx)²."""
        if any(v is None for v in [self.Delta_x, self.Delta_p, self.m, self.omega, self.hbar]):
            return (False, "SKIP: missing intermediate variables")
        E_computed = self.Delta_p**2/(2*self.m) + 0.5*self.m*self.omega**2*self.Delta_x**2
        if math.isclose(E_computed, self.E_min, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Energy expression consistent")
        return (False, f"E from expression = {E_computed}, E_min = {self.E_min}")

    def test_invariant_optimal_deltax(self):
        """Check that Δx_opt² = ℏ/(2 m ω)."""
        if any(v is None for v in [self.Delta_x, self.m, self.omega, self.hbar]):
            return (False, "SKIP: missing variables for optimal Δx check")
        expected_sq = self.hbar / (2 * self.m * self.omega)
        actual_sq = self.Delta_x**2
        if math.isclose(actual_sq, expected_sq, rel_tol=1e-2):
            return (True, f"Δx² = {actual_sq} ≈ {expected_sq}")
        return (False, f"Δx² = {actual_sq}, expected {expected_sq}")

    # ------------------------------------------------------------------
    # Modeling / trace quality checks (use isclose)
    # ------------------------------------------------------------------
    def test_modeling_derivative_zero(self):
        """At the optimum, dE/d(Δx) = 0."""
        if any(v is None for v in [self.Delta_x, self.m, self.omega, self.hbar]):
            return (False, "SKIP: missing variables for derivative check")
        # Analytical derivative of E with respect to Δx:
        deriv = -self.hbar**2/(4*self.m*self.Delta_x**3) + self.m*self.omega**2*self.Delta_x
        if math.isclose(deriv, 0, abs_tol=1e-2):
            return (True, "Derivative condition satisfied")
        return (False, f"Derivative = {deriv}, expected ~0")

    def test_modeling_energy_partition(self):
        """At the minimum, kinetic energy equals potential energy (a=b condition)."""
        if any(v is None for v in [self.Delta_p, self.Delta_x, self.m, self.omega]):
            return (False, "SKIP: missing variables for energy partition check")
        KE = self.Delta_p**2/(2*self.m)
        PE = 0.5*self.m*self.omega**2*self.Delta_x**2
        if math.isclose(KE, PE, rel_tol=1e-2):
            return (True, f"KE = {KE}, PE = {PE}")
        return (False, f"KE ≠ PE: KE = {KE}, PE = {PE}")

    # ------------------------------------------------------------------
    # Monotonicity test: perturb Δx and check energy increases
    # ------------------------------------------------------------------
    def test_monotonicity_energy_minimum(self):
        """Increase Δx by 10% and verify that E increases."""
        if any(v is None for v in [self.Delta_x, self.m, self.omega, self.hbar]):
            return (False, "SKIP: missing variables for monotonicity check")
        E_opt = self.E_min
        Delta_x_pert = self.Delta_x * 1.1
        # Energy for perturbed Δx (using the relation E = ℏ²/(8 m Δx²) + ½ m ω² Δx²)
        E_pert = self.hbar**2/(8*self.m*Delta_x_pert**2) + 0.5*self.m*self.omega**2*Delta_x_pert**2
        if E_pert > E_opt and not math.isclose(E_pert, E_opt, rel_tol=1e-2):
            return (True, "Energy increases when Δx is perturbed from optimum")
        return (False, "Energy does not increase away from optimum (suspect minimum not found)")

    # ------------------------------------------------------------------
    # Weak answer sanity test (no isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_sign_and_finite(self):
        """Check that E_min is positive and finite."""
        if self.E_min is None:
            return (False, "SKIP: E_min not available")
        if not math.isfinite(self.E_min):
            return (False, "E_min is not finite")
        if self.E_min <= 0:
            return (False, "E_min should be positive")
        return (True, "E_min positive and finite")

    # ------------------------------------------------------------------
    # Run all tests
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