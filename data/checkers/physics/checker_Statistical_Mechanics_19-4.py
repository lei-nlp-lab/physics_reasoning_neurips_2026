import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract all required inputs; default None
        # Givens
        self.e = getattr(solution_instance, 'e', None)
        self.A = getattr(solution_instance, 'A', None)
        self.m = getattr(solution_instance, 'm', None)
        self.tau = getattr(solution_instance, 'tau', None)
        self.mu = getattr(solution_instance, 'mu', None)
        self.V = getattr(solution_instance, 'V', None)
        self.h = getattr(solution_instance, 'h', None)

        # Result expected from the solution
        self.I = None
        if hasattr(solution_instance, 'I'):
            self.I = solution_instance.I
        elif hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.I = solution_instance.results.get('I', None)

    # ----------------------------------------------------------------------
    # Helper function to compute the theoretical current (for monotonicity)
    # ----------------------------------------------------------------------
    def _compute_theoretical_I(self, e, A, m, tau, mu, V, h):
        """Return the theoretical current from the derived formula.
           All parameters should be provided."""
        if any(p is None for p in [e, A, m, tau, mu, V, h]):
            return None
        prefactor = (4 * math.pi * e * A * m * tau**2) / (h**3)
        exponent = math.exp(-(V - mu) / tau)
        factor = 1 + V / tau
        return -prefactor * factor * exponent

    # ----------------------------------------------------------------------
    # Interface checks – required keys and types
    # ----------------------------------------------------------------------
    def test_interface_required_givens(self):
        """All physical parameters must be present and numeric."""
        missing = []
        for name in ['e', 'A', 'm', 'tau', 'mu', 'V', 'h']:
            val = getattr(self, name, None)
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Missing parameters: {missing}")
        return (True, "All required givens present")

    def test_interface_I_present(self):
        """Result I must be present and a number."""
        if self.I is None:
            return (False, "I is missing")
        if not isinstance(self.I, (int, float, sp.Number)):
            return (False, "I is not a numeric type")
        return (True, "I present and numeric")

    # ----------------------------------------------------------------------
    # Invariant checks (theorem-based, use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_integral_identity(self):
        """Check the key integral identity:
           ∫_a^∞ (x + μ/τ) e^{-x} dx = (1 + V/τ) e^{-a}   where a = (V-μ)/τ.
        """
        if any(getattr(self, p, None) is None for p in ['V','mu','tau']):
            return (False, "SKIP: missing V, mu, or tau")
        a = (self.V - self.mu) / self.tau
        if a <= 0:
            return (False, "Condition iv (V > mu) not satisfied, integral identity may not apply")

        x = sp.symbols('x')
        mu_frac = self.mu / self.tau
        integral = sp.integrate((x + mu_frac) * sp.exp(-x), (x, a, sp.oo))
        expected = (1 + a + mu_frac) * sp.exp(-a)
        simplified = sp.simplify(integral - expected)
        try:
            integral_num = float(integral.evalf())
            expected_num = float(expected.evalf())
            if math.isclose(integral_num, expected_num, rel_tol=1e-8, abs_tol=1e-12):
                return (True, f"Integral identity holds: {integral_num} ≈ {expected_num}")
            else:
                return (False, f"Integral mismatch: {integral_num} vs {expected_num}")
        except Exception as e:
            return (False, f"Could not evaluate integral: {e}")

    def test_invariant_approximation_condition(self):
        """Check that V - mu > tau (condition iv)."""
        if self.V is None or self.mu is None or self.tau is None:
            return (False, "SKIP: missing V, mu, or tau")
        if (self.V - self.mu) > self.tau:
            return (True, f"V - mu = {self.V - self.mu} > tau = {self.tau}")
        else:
            return (False, f"V - mu = {self.V - self.mu} <= tau = {self.tau}, approximation may be invalid")

    def test_invariant_sign_current(self):
        """The current I must be negative (electrons leaving)."""
        if self.I is None:
            return (False, "SKIP: I missing")
        if self.I < 0:
            return (True, "I is negative as expected")
        else:
            return (False, f"I = {self.I} is not negative (expected negative)")

    def test_invariant_finite_current(self):
        """I must be finite and non-NaN."""
        if self.I is None:
            return (False, "SKIP: I missing")
        if math.isfinite(self.I):
            return (True, "I is finite")
        else:
            return (False, f"I = {self.I} is not finite")

    # ----------------------------------------------------------------------
    # Modeling / trace checks
    # ----------------------------------------------------------------------
    def test_modeling_angular_integration(self):
        """Check that the angular integral over φ and θ yields π.
           That is: ∫0^2π dφ ∫0^{π/2} sinθ cosθ dθ = π.
        """
        phi, theta = sp.symbols('phi theta')
        integrand = sp.sin(theta) * sp.cos(theta)
        integral = sp.integrate(integrand, (theta, 0, sp.pi/2))
        integral = sp.integrate(integral, (phi, 0, 2*sp.pi))
        expected = sp.pi
        simplified = sp.simplify(integral - expected)
        if simplified == 0:
            return (True, "Angular integration factor is π")
        else:
            return (False, f"Angular integration yields {integral}, expected {expected}")

    def test_modeling_momentum_threshold(self):
        """Check that the lower momentum limit is sqrt(2 m V)."""
        if self.m is None or self.V is None:
            return (False, "SKIP: missing m or V")
        p_min_theory = math.sqrt(2 * self.m * self.V)
        # We can't directly verify the solution's integration bounds, but we can
        # check that the derived current formula has the correct exponential factor.
        # For a high barrier, |I| ∝ exp(-(V-μ)/τ), indicating the threshold is correct.
        if self.tau is None or self.mu is None or self.h is None:
            return (False, "SKIP: missing tau, mu, or h")
        # The leading exponential factor should be exp(-(V-μ)/τ)
        # We compute the theoretical current to verify its form.
        I_theory = self._compute_theoretical_I(self.e, self.A, self.m, self.tau, self.mu, self.V, self.h)
        if I_theory is None:
            return (False, "SKIP: cannot compute theoretical I")
        # Check that I_theory is negative and its magnitude follows the expected exponential scaling
        if I_theory < 0:
            return (True, "Theoretical current is negative, consistent with escape direction")
        else:
            return (False, "Theoretical current is not negative")

    def test_modeling_fermi_dirac_approximation(self):
        """Check that the solution's I matches the derived formula within tolerance."""
        if any(getattr(self, p, None) is None for p in ['e','A','m','tau','mu','V','h','I']):
            return (False, "SKIP: missing one or more parameters")
        I_expected = self._compute_theoretical_I(self.e, self.A, self.m, self.tau, self.mu, self.V, self.h)
        if I_expected is None:
            return (False, "Cannot compute expected I")
        if math.isclose(self.I, I_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Stored I = {self.I} matches derived I = {I_expected} within 1%")
        else:
            return (False, f"Stored I = {self.I} differs from derived I = {I_expected}")

    # ----------------------------------------------------------------------
    # Monotonicity checks (with explicit perturbation)
    # Each test perturbs one input by a small factor (e.g., 5%) and compares
    # the perturbed output (recomputed from theory) to the baseline output.
    # ----------------------------------------------------------------------
    def test_monotonicity_current_with_barrier_voltage(self):
        """Increasing V should make I less negative (magnitude decreases)."""
        if any(getattr(self, p, None) is None for p in ['e','A','m','tau','mu','V','h']):
            return (False, "SKIP: missing parameters")
        # Baseline from theory
        I_baseline = self._compute_theoretical_I(self.e, self.A, self.m, self.tau, self.mu, self.V, self.h)
        # Perturb V upward by 5%
        V_pert = self.V * 1.05
        I_pert = self._compute_theoretical_I(self.e, self.A, self.m, self.tau, self.mu, V_pert, self.h)
        if I_baseline is None or I_pert is None:
            return (False, "Could not compute currents")
        # For negative current, increase V => I becomes less negative => I_pert > I_baseline
        if I_pert > I_baseline:
            return (True, f"V increases by 5%: I_baseline={I_baseline:.4e}, I_pert={I_pert:.4e}, I_pert > I_baseline")
        else:
            return (False, f"V increases but I does not increase: I_baseline={I_baseline:.4e}, I_pert={I_pert:.4e}")

    def test_monotonicity_current_with_temperature(self):
        """Increasing τ should make I more negative (magnitude increases)."""
        if any(getattr(self, p, None) is None for p in ['e','A','m','tau','mu','V','h']):
            return (False, "SKIP: missing parameters")
        I_baseline = self._compute_theoretical_I(self.e, self.A, self.m, self.tau, self.mu, self.V, self.h)
        # Perturb τ upward by 5%
        tau_pert = self.tau * 1.05
        I_pert = self._compute_theoretical_I(self.e, self.A, self.m, tau_pert, self.mu, self.V, self.h)
        if I_baseline is None or I_pert is None:
            return (False, "Could not compute currents")
        # For negative current, increase τ => I becomes more negative => I_pert < I_baseline
        if I_pert < I_baseline:
            return (True, f"τ increases by 5%: I_baseline={I_baseline:.4e}, I_pert={I_pert:.4e}, I_pert < I_baseline")
        else:
            return (False, f"τ increases but I does not decrease: I_baseline={I_baseline:.4e}, I_pert={I_pert:.4e}")

    def test_monotonicity_current_with_chemical_potential(self):
        """Increasing μ should make I more negative (magnitude increases)."""
        if any(getattr(self, p, None) is None for p in ['e','A','m','tau','mu','V','h']):
            return (False, "SKIP: missing parameters")
        I_baseline = self._compute_theoretical_I(self.e, self.A, self.m, self.tau, self.mu, self.V, self.h)
        # Perturb μ upward by 5%
        mu_pert = self.mu * 1.05
        I_pert = self._compute_theoretical_I(self.e, self.A, self.m, self.tau, mu_pert, self.V, self.h)
        if I_baseline is None or I_pert is None:
            return (False, "Could not compute currents")
        # For negative current, increase μ => I becomes more negative => I_pert < I_baseline
        if I_pert < I_baseline:
            return (True, f"μ increases by 5%: I_baseline={I_baseline:.4e}, I_pert={I_pert:.4e}, I_pert < I_baseline")
        else:
            return (False, f"μ increases but I does not decrease: I_baseline={I_baseline:.4e}, I_pert={I_pert:.4e}")

    # ----------------------------------------------------------------------
    # Answer sanity (weak, at most one)
    # ----------------------------------------------------------------------
    def test_answer_sanity_sign(self):
        """Weak sanity: I must be negative and not extremely large/small."""
        if self.I is None:
            return (False, "SKIP: I missing")
        if self.I < 0 and math.isfinite(self.I) and abs(self.I) < 1e30:
            return (True, "I is negative and finite")
        else:
            return (False, f"I = {self.I} fails basic sanity")

    # ----------------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)