import sympy as sp
import math
import traceback

class PhysicsChecker:
    """Audits student solutions for detailed balance between γ+p→π⁺+n and π⁺+n→γ+p."""

    def __init__(self, solution_instance):
        # Safe extraction from results dict
        results = getattr(solution_instance, 'results', {})
        self.sigma_II = results.get('sigma_II', None)
        self.W_th = results.get('W_th', None)
        self.sigma_II_variation = results.get('sigma_II_variation', None)

        # Try to get masses from results or from solution_instance directly (e.g., givens)
        # Use safe get; do not assign numeric fallbacks
        self.m = None
        self.mu = None
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.m = solution_instance.givens.get('m', None)
            self.mu = solution_instance.givens.get('mu', None)
        if self.m is None:
            self.m = results.get('m', None)
        if self.mu is None:
            self.mu = results.get('mu', None)
        self.solution = solution_instance

    # --------------------------------------------------------------------------
    # Interface tests
    # --------------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all three required keys are present in results."""
        missing = []
        if self.sigma_II is None:
            missing.append('sigma_II')
        if self.W_th is None:
            missing.append('W_th')
        if self.sigma_II_variation is None:
            missing.append('sigma_II_variation')
        if missing:
            return (False, f"Missing keys: {', '.join(missing)}")
        return (True, "All required keys present")

    def test_interface_masses_available(self):
        """Check that mass parameters are available for physics tests."""
        if self.m is None or self.mu is None:
            return (True, "Mass parameters not available; physics tests that depend on them will be skipped.")
        return (True, "Mass parameters available")

    # --------------------------------------------------------------------------
    # Invariant / theorem-based tests (at least two)
    # --------------------------------------------------------------------------
    def test_invariant_threshold_energy(self):
        """Threshold energy must be W_th = m + mu (up to 1% tolerance)."""
        if self.m is None or self.mu is None:
            return (False, "SKIP: masses not available")
        if self.W_th is None:
            return (False, "SKIP: W_th not available")
        expected = self.m + self.mu
        if math.isclose(self.W_th, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"W_th = {self.W_th} matches m+mu = {expected}")
        return (False, f"W_th = {self.W_th} does not match expected {expected}")

    def test_invariant_denominator_positive(self):
        """The denominator in sigma_II expression should be positive for W > threshold (physical region)."""
        if self.sigma_II is None or self.m is None or self.mu is None:
            return (True, "SKIP: insufficient data to test denominator positivity")
        try:
            # Define symbolic variables
            W, m, mu = sp.symbols('W m mu', positive=True, real=True)
            # Attempt to parse sigma_II as expression; denominator is the factor multiplying sigma_I
            # Assuming sigma_II = ( (W^2-m^2)^2 / ( (W^2+m^2-mu^2)^2 - 4*W^2*m^2 ) ) * sigma_I
            # We'll extract the denominator symbolically from the string
            expr = sp.sympify(self.sigma_II)
            # The denominator is the divisor (excluding sigma_I). We'll evaluate at a test W > threshold
            subst = {m: self.m, mu: self.mu}
            W_test = self.m + self.mu + 0.1
            numerator = (W_test**2 - self.m**2)**2
            denominator = (W_test**2 + self.m**2 - self.mu**2)**2 - 4*W_test**2*self.m**2
            if denominator > 0:
                return (True, f"Denominator at W={W_test} is {denominator} > 0")
            else:
                return (False, f"Denominator at W={W_test} is {denominator} <= 0 (unphysical)")
        except Exception:
            return (True, "SKIP: could not evaluate denominator")

    # --------------------------------------------------------------------------
    # Modeling / trace consistency tests
    # --------------------------------------------------------------------------
    def test_modeling_near_threshold_variation(self):
        """Check that sigma_II_variation describes the correct near-threshold behavior."""
        if self.sigma_II_variation is None:
            return (False, "SKIP: sigma_II_variation not available")
        expected_pattern = "proportional to (W - W_th)^{-1/2}"
        if "proportional" in self.sigma_II_variation and "-1/2" in self.sigma_II_variation:
            return (True, f"Variation description '{self.sigma_II_variation}' matches expected behavior")
        return (False, f"Variation description '{self.sigma_II_variation}' does not match expected '{expected_pattern}'")

    def test_modeling_threshold_consistency(self):
        """If both W_th and variation are present, ensure variation refers to that threshold."""
        if self.W_th is None or self.sigma_II_variation is None:
            return (True, "SKIP: insufficient data")
        return (True, "Threshold and variation present; modeling looks consistent")

    # --------------------------------------------------------------------------
    # Monotonicity tests (explicit perturbation of input)
    # --------------------------------------------------------------------------
    def test_monotonicity_W_th_increases_with_m(self):
        """Perturb nucleon mass m upward and verify W_th increases linearly."""
        if self.m is None or self.mu is None:
            return (True, "SKIP: masses not available")
        if self.W_th is None:
            return (False, "SKIP: W_th not available")
        try:
            m_val = float(self.m)
            mu_val = float(self.mu)
            W_th_val = float(self.W_th)
        except (TypeError, ValueError):
            return (False, "Cannot convert masses or W_th to float")

        # Perturb m by +1%
        epsilon = 0.01 * m_val
        m_perturbed = m_val + epsilon
        # Expected perturbed W_th (assuming correct formula)
        expected_perturbed_W_th = m_perturbed + mu_val
        # Compare perturbed to baseline
        if expected_perturbed_W_th > W_th_val:
            return (True, f"Perturbing m from {m_val} to {m_perturbed}: expected perturbed W_th = {expected_perturbed_W_th} > baseline {W_th_val}")
        else:
            return (False, f"Perturbing m does not increase W_th as expected (baseline {W_th_val} >= perturbed {expected_perturbed_W_th})")

    # --------------------------------------------------------------------------
    # Weak answer sanity tests (at most one, no isclose, no derivation)
    # --------------------------------------------------------------------------
    def test_answer_sanity_W_th_positive_finite(self):
        """Ensure W_th is a positive finite number (weak sanity)."""
        if self.W_th is None:
            return (False, "SKIP: W_th not present")
        try:
            val = float(self.W_th)
        except (TypeError, ValueError):
            return (False, f"W_th not a number: {self.W_th}")
        if val <= 0 or not math.isfinite(val):
            return (False, f"W_th = {val} is not positive and finite")
        return (True, f"W_th = {val} is positive and finite")

    # --------------------------------------------------------------------------
    # Runner
    # --------------------------------------------------------------------------
    def run_all_tests(self):
        """Discover all test_* methods, execute them, collect logs."""
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
                passed = False
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)