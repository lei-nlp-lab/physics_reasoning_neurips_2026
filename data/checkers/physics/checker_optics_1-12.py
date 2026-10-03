import math
import sympy as sp

class PhysicsChecker:
    """
    Audits a solution for the depth-of-field derivation problem.
    """

    def __init__(self, solution_instance):
        # --- Extract givens safely ---
        # Try to get from solution_instance.results, then from attributes
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.results = solution_instance.results
        else:
            self.results = {}

        # Required parameters: f, N, l, q, Delta_q
        self.f = getattr(solution_instance, 'f', None) or self.results.get('f', None)
        self.N = getattr(solution_instance, 'N', None) or self.results.get('N', None)
        self.l = getattr(solution_instance, 'l', None) or self.results.get('l', None)
        self.q = getattr(solution_instance, 'q', None) or self.results.get('q', None)
        self.Delta_q = getattr(solution_instance, 'Delta_q', None) or self.results.get('Delta_q', None)

        # Optional: explanation steps (for context, not used directly in tests)
        self.explanation_steps = (
            getattr(solution_instance, 'explanation_steps', None)
            or self.results.get('explanation_steps', None)
        )

        # For monotonicity tests we may need to compute small perturbations
        # Baseline values are the actual provided numbers (if present)
        self.baseline_Delta_q = self.Delta_q  # may be None

    # ----------------------------------------------------------------------
    # Helper: safely perturb a parameter
    # ----------------------------------------------------------------------
    def _perturbed_Delta_q(self, param_name, factor):
        """
        Compute Delta_q with a perturbed parameter (f, N, l, or q) by scaling
        it by `factor`. Returns (new_Delta_q, message) or raises ValueError.
        """
        if any(v is None for v in [self.f, self.N, self.l, self.q]):
            raise ValueError("Not enough givens to compute perturbed Delta_q.")
        new_val = None
        if param_name == 'f':
            new_f = self.f * factor
            new_Delta_q = self.N * self.l * self.q**2 / new_f**2
        elif param_name == 'N':
            new_N = self.N * factor
            new_Delta_q = new_N * self.l * self.q**2 / self.f**2
        elif param_name == 'l':
            new_l = self.l * factor
            new_Delta_q = self.N * new_l * self.q**2 / self.f**2
        elif param_name == 'q':
            new_q = self.q * factor
            new_Delta_q = self.N * self.l * new_q**2 / self.f**2
        else:
            raise ValueError(f"Unknown param_name: {param_name}")
        return new_Delta_q

    # ----------------------------------------------------------------------
    # Test methods
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that the required output key 'Delta_q' exists and is numeric."""
        if self.Delta_q is None:
            return (False, "Missing required output 'Delta_q'.")
        if not isinstance(self.Delta_q, (int, float)):
            return (False, f"'Delta_q' is not numeric, got {type(self.Delta_q)}.")
        if math.isnan(self.Delta_q) or math.isinf(self.Delta_q):
            return (False, f"'Delta_q' is NaN or infinite: {self.Delta_q}")
        return (True, "Delta_q is present and numeric.")

    def test_interface_givens_available(self):
        """Check that all required givens (f, N, l, q) are present."""
        missing = []
        for name, val in [('f', self.f), ('N', self.N), ('l', self.l), ('q', self.q)]:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All required givens are present.")

    def test_invariant_formula_consistency(self):
        """
        Verify that Delta_q satisfies the physical formula:
            Delta_q = N * l * q^2 / f^2
        This is a re-derivation inside the checker, not a hardcoded constant.
        """
        if any(v is None for v in [self.f, self.N, self.l, self.q, self.Delta_q]):
            return (False, "SKIP: Missing some values to check formula consistency.")
        expected = self.N * self.l * self.q**2 / self.f**2
        if math.isclose(self.Delta_q, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Delta_q = {self.Delta_q} matches formula prediction {expected}.")
        else:
            return (False, f"Delta_q = {self.Delta_q} does not match formula prediction {expected}.")

    def test_invariant_thin_lens_approximation(self):
        """
        Check that the thin lens equation 1/f = 1/q + 1/p holds approximately
        with p ≈ f (since q >> f). We can compute p from the given q and f.
        """
        if self.q is None or self.f is None:
            return (False, "SKIP: Need q and f to check thin lens.")
        if self.q <= 0 or self.f <= 0:
            return (False, "q and f must be positive for thin lens.")
        # Solve for p exactly: p = 1 / (1/f - 1/q) = f * q / (q - f)
        p_exact = self.f * self.q / (self.q - self.f)
        # For q >> f, p_exact should be close to f
        if math.isclose(p_exact, self.f, rel_tol=0.1):  # allow 10% tolerance due to q >> f approximation
            return (True, f"p ≈ {p_exact:.3f} is close to f={self.f}.")
        else:
            return (False, f"p = {p_exact:.3f} is not close to f={self.f} (q={self.q}). Check q >> f assumption.")

    def test_invariant_blur_geometry(self):
        """
        Derive Δp from Δq using derivative approximation |Δp| ≈ (f^2 / q^2) * Δq,
        then verify blur condition: l = D * |Δp| / p, with D = f/N and p ≈ f.
        This checks the core geometric relation.
        """
        if any(v is None for v in [self.f, self.N, self.l, self.q, self.Delta_q]):
            return (False, "SKIP: Missing values for blur geometry check.")
        # Compute |Δp| using small-defocus approximation
        Delta_p = (self.f**2 / self.q**2) * self.Delta_q
        # Aperture diameter D
        D = self.f / self.N
        # Image distance p (use exact, but approximation gives p ~ f)
        # For consistency, use exact thin lens formula
        p_exact = self.f * self.q / (self.q - self.f)
        # Expected blur diameter from geometry
        l_from_geometry = D * Delta_p / p_exact
        # Compare with given l
        if math.isclose(l_from_geometry, self.l, rel_tol=0.02, abs_tol=0.02):
            return (True, f"Blur geometry consistent: derived l = {l_from_geometry:.5f} matches given l = {self.l}.")
        else:
            return (False, f"Blur geometry inconsistency: derived l = {l_from_geometry:.5f}, given l = {self.l}.")

    def test_monotonicity_increases_with_l(self):
        """Increasing circle-of-confusion diameter l increases depth of field."""
        if self.Delta_q is None:
            return (False, "SKIP: No baseline Delta_q for monotonicity check.")
        try:
            perturbed = self._perturbed_Delta_q('l', 1.05)  # 5% increase
        except ValueError as e:
            return (False, f"SKIP: {e}")
        if perturbed > self.Delta_q:
            return (True, f"Increasing l by 5% increases Delta_q from {self.Delta_q:.5f} to {perturbed:.5f}.")
        else:
            return (False, f"Monotonicity fails: l increase caused Delta_q to decrease/decreased.")

    def test_monotonicity_increases_with_N(self):
        """Increasing f-stop number N increases depth of field."""
        if self.Delta_q is None:
            return (False, "SKIP: No baseline Delta_q for monotonicity check.")
        try:
            perturbed = self._perturbed_Delta_q('N', 1.05)
        except ValueError as e:
            return (False, f"SKIP: {e}")
        if perturbed > self.Delta_q:
            return (True, f"Increasing N by 5% increases Delta_q from {self.Delta_q:.5f} to {perturbed:.5f}.")
        else:
            return (False, f"Monotonicity fails: N increase caused Delta_q to decrease.")

    def test_monotonicity_increases_with_q(self):
        """Increasing object distance q increases depth of field."""
        if self.Delta_q is None:
            return (False, "SKIP: No baseline Delta_q for monotonicity check.")
        try:
            perturbed = self._perturbed_Delta_q('q', 1.05)
        except ValueError as e:
            return (False, f"SKIP: {e}")
        if perturbed > self.Delta_q:
            return (True, f"Increasing q by 5% increases Delta_q from {self.Delta_q:.5f} to {perturbed:.5f}.")
        else:
            return (False, f"Monotonicity fails: q increase caused Delta_q to decrease.")

    def test_monotonicity_decreases_with_f(self):
        """Increasing focal length f decreases depth of field."""
        if self.Delta_q is None:
            return (False, "SKIP: No baseline Delta_q for monotonicity check.")
        try:
            perturbed = self._perturbed_Delta_q('f', 1.05)
        except ValueError as e:
            return (False, f"SKIP: {e}")
        if perturbed < self.Delta_q:
            return (True, f"Increasing f by 5% decreases Delta_q from {self.Delta_q:.5f} to {perturbed:.5f}.")
        else:
            return (False, f"Monotonicity fails: f increase caused Delta_q to increase.")

    def test_modeling_small_defocus_approximation(self):
        """
        Check that the approximation |Δp| ≈ (f^2 / q^2) Δq is reasonable
        by comparing with the exact differential.
        """
        if any(v is None for v in [self.f, self.q, self.Delta_q]):
            return (False, "SKIP: Need f, q, Delta_q for this check.")
        # Exact Δp using thin lens derivative (full derivative, not approximate)
        # p = f*q/(q-f)  => dp/dq = -p^2/q^2 = -f^2/(q-f)^2
        p_exact = self.f * self.q / (self.q - self.f)
        exact_dp_dq = - (p_exact**2) / (self.q**2)
        exact_Delta_p_abs = abs(exact_dp_dq * self.Delta_q)
        # Approximate Δp using p~f
        approx_Delta_p_abs = (self.f**2 / self.q**2) * self.Delta_q
        if math.isclose(exact_Delta_p_abs, approx_Delta_p_abs, rel_tol=0.02):
            return (True, f"Small-defocus approximation error: exact |Δp|={exact_Delta_p_abs:.6f}, approx={approx_Delta_p_abs:.6f}.")
        else:
            return (False, f"Approximation error too large: exact |Δp|={exact_Delta_p_abs:.6f}, approx={approx_Delta_p_abs:.6f}.")

    def test_answer_sanity_positive(self):
        """Weak sanity: final answer (Delta_q) should be positive."""
        if self.Delta_q is None:
            return (False, "SKIP: No Delta_q to check.")
        if self.Delta_q > 0:
            return (True, f"Delta_q = {self.Delta_q} is positive.")
        else:
            return (False, f"Delta_q = {self.Delta_q} is not positive.")

    def test_modeling_explanation_steps_consistency(self):
        """
        If explanation_steps is provided, check that it contains key relations:
        thin lens, derivative, blur geometry, final expression.
        This is a soft check based on textual content.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: No explanation_steps provided.")
        steps_text = str(self.explanation_steps).lower()
        required_terms = ['thin lens', 'derivative', 'blur', 'circle of confusion', 'f-stop', 'depth of field']
        found = [term for term in required_terms if term in steps_text]
        if len(found) >= 4:  # at least 4 of 6
            return (True, f"Explanation steps contain key terms: {found}")
        else:
            missing = set(required_terms) - set(found)
            return (False, f"Explanation steps missing key terms: {missing}")

    # ----------------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Collect all bound test methods
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