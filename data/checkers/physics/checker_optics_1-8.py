import sympy
import math
import traceback

class PhysicsChecker:
    """
    Audits solutions for a diverging lens problem with parallel incident rays.
    Checks interface completeness, lens formula invariants, modeling consistency,
    sign conventions, and weak answer sanity.
    """

    def __init__(self, solution_instance):
        # Safely extract results dictionary
        self.results = getattr(solution_instance, 'results', {}) or {}
        # Safely extract givens dictionary (inputs)
        self.givens = getattr(solution_instance, 'givens', {}) or {}
        # Safely extract explanation steps (list of strings or similar)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', []) or []

        # Extract output quantities - use None if missing
        self.v = self.results.get('v', None)
        self.y_prime = self.results.get('y_prime', None)

        # Extract input quantities - use None if missing
        self.f = self.givens.get('f', None)          # focal length, should be -20 cm
        self.theta = self.givens.get('theta', None)  # incident angle in degrees, should be 5

        # Helper to convert theta to radians if present
        self.theta_rad = None
        if self.theta is not None:
            try:
                self.theta_rad = math.radians(self.theta)
            except (TypeError, ValueError):
                pass

    def run_all_tests(self):
        """Discover and execute all test_ methods, collecting results."""
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
    # Interface tests – ensure required keys exist and have correct types
    # ------------------------------------------------------------------
    def test_interface_results_dict(self):
        if not isinstance(self.results, dict):
            return (False, "'results' is not a dict or is empty")
        return (True, "results is a valid dict")

    def test_interface_v_present(self):
        if self.v is None:
            return (False, "results['v'] missing or None")
        return (True, "v present")

    def test_interface_y_prime_present(self):
        if self.y_prime is None:
            return (False, "results['y_prime'] missing or None")
        return (True, "y_prime present")

    def test_interface_v_numeric(self):
        if not isinstance(self.v, (int, float)):
            return (False, f"v is not numeric: {type(self.v)}")
        return (True, "v is numeric")

    def test_interface_y_prime_numeric(self):
        if not isinstance(self.y_prime, (int, float)):
            return (False, f"y_prime is not numeric: {type(self.y_prime)}")
        return (True, "y_prime is numeric")

    # ------------------------------------------------------------------
    # Invariant tests – theorem based physical sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_lens_equation(self):
        """Check v = f for parallel rays (u = ∞)."""
        if self.v is None or self.f is None:
            return (False, "SKIP: v or f missing")
        # Both v and f should be negative for diverging lens
        if not math.isclose(self.v, self.f, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"v ({self.v}) not close to f ({self.f})")
        return (True, "v ≈ f (lens equation for parallel rays)")

    def test_invariant_image_height(self):
        """Check y' = |f| * tan(θ) using absolute focal length."""
        if self.y_prime is None or self.f is None or self.theta_rad is None:
            return (False, "SKIP: y_prime, f or theta missing")
        expected = abs(self.f) * math.tan(self.theta_rad)
        if not math.isclose(self.y_prime, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"y' ({self.y_prime}) vs expected {expected:.4f}")
        return (True, "y' ≈ |f| tan(θ)")

    def test_invariant_sign_convention(self):
        """Virtual image: v < 0 ; u = -∞ ; upright image: y' > 0."""
        if self.v is None or self.y_prime is None:
            return (False, "SKIP: v or y_prime missing")
        if not (self.v < 0):
            return (False, f"v should be negative, got {self.v}")
        if not (self.y_prime > 0):
            return (False, f"y_prime should be positive, got {self.y_prime}")
        return (True, "v negative (virtual), y_prime positive (upright)")

    # ------------------------------------------------------------------
    # Modeling / trace tests – consistency with explanation steps
    # ------------------------------------------------------------------
    def test_modeling_v_is_f(self):
        """Check that v exactly equals the given focal length (from pseudocode step)."""
        if self.v is None or self.f is None:
            return (False, "SKIP: v or f missing")
        # For parallel rays, v = f by a well‑known limit of the thin‑lens equation.
        # We do not use isclose here because it's a direct assignment.
        if self.v != self.f:
            return (False, f"v ({self.v}) should equal f ({self.f}) exactly")
        return (True, "v equals f (as required by parallel rays)")

    def test_modeling_trace_tan_derivation(self):
        """Check that y' / |f| matches tan(θ) within tolerance."""
        if self.f is None or self.y_prime is None or self.theta_rad is None:
            return (False, "SKIP: required quantities missing")
        ratio = self.y_prime / abs(self.f)
        tan_theta = math.tan(self.theta_rad)
        if not math.isclose(ratio, tan_theta, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"y'/|f| = {ratio:.4f} vs tan(θ) = {tan_theta:.4f}")
        return (True, "y'/|f| ≈ tan(θ)")

    # ------------------------------------------------------------------
    # Monotonicity test – verify that output varies correctly with input
    # ------------------------------------------------------------------
    def test_monotonicity_y_prime_increases_with_theta(self):
        """
        Perturb the incident angle by +10% and check that the expected
        image height (according to |f| tan θ) increases.
        """
        if self.theta_rad is None or self.y_prime is None or self.f is None:
            return (False, "SKIP: theta, y_prime, or f missing")
        # Baseline values
        baseline_y = self.y_prime
        # Perturb theta by 10% (add 10% of the original angle, but angle small)
        delta = 0.1 * self.theta_rad
        perturbed_theta = self.theta_rad + delta
        # Expected new height (using the physical relation)
        new_y_expected = abs(self.f) * math.tan(perturbed_theta)
        # For small angles, tan is increasing, so new_y_expected > baseline_y
        if not (new_y_expected > baseline_y):
            return (False, f"Perturbed y' ({new_y_expected:.4f}) not > baseline ({baseline_y:.4f})")
        return (True, "y' increases when θ increases (consistent with tan)")

    # ------------------------------------------------------------------
    # Weak answer sanity – only one, secondary (no isclose, no re‑deriving)
    # ------------------------------------------------------------------
    def test_answer_sanity_plausible_values(self):
        """
        Vague checks: v is negative and within a few tens of cm,
        y_prime is positive and under, say, 10 cm.
        """
        if self.v is None or self.y_prime is None:
            return (False, "SKIP: v or y_prime missing")
        # v should be on the order of focal length
        if not (-50 < self.v < 0):
            return (False, f"v out of plausible range: {self.v}")
        # y' for a 5° tilt and 20 cm lens is around 1.75 cm
        if not (0 < self.y_prime < 10):
            return (False, f"y_prime out of plausible range: {self.y_prime}")
        return (True, "v and y_prime within plausible ranges")