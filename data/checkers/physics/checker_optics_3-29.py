import math
import sympy
import traceback

class PhysicsChecker:
    """
    Auditor for optical systems problem: searchlight (etendue) and stellar interferometer.
    Checks consistency and physical sanity of results.
    """
    def __init__(self, solution_instance):
        # Extract givens safely (may come as attributes or in givens dict)
        self.I_source = getattr(solution_instance, 'I_source', None)
        self.I_beam = getattr(solution_instance, 'I_beam', None)
        self.lambda_wavelength = getattr(solution_instance, 'lambda', None)  # avoid keyword
        self.alpha = getattr(solution_instance, 'alpha', None)

        # Extract emitted results safely
        results = getattr(solution_instance, 'results', {})
        self.r = results.get('r', None)
        self.h = results.get('h', None)

        # Optionally store explanation_steps if present
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ----------------------------------------------------------------------
    # Interface tests: verify required keys exist and are numeric
    # ----------------------------------------------------------------------
    def test_interface_r_exists(self):
        """r must be a numeric value."""
        if self.r is None:
            return (False, "Missing required result: r (angular compression ratio)")
        if not isinstance(self.r, (int, float)):
            return (False, f"r is not numeric, got {type(self.r).__name__}")
        return (True, "r is present and numeric")

    def test_interface_h_exists(self):
        """h must be a numeric value."""
        if self.h is None:
            return (False, "Missing required result: h (interferometer baseline for first null)")
        if not isinstance(self.h, (int, float)):
            return (False, f"h is not numeric, got {type(self.h).__name__}")
        return (True, "h is present and numeric")

    def test_interface_givens_available(self):
        """Check that at least I_source, I_beam, lambda, alpha are available (may be None)."""
        missing = []
        if self.I_source is None:
            missing.append("I_source")
        if self.I_beam is None:
            missing.append("I_beam")
        if self.lambda_wavelength is None:
            missing.append("lambda")
        if self.alpha is None:
            missing.append("alpha")
        if missing:
            return (True, f"Missing optional givens (may be None): {', '.join(missing)}")
        return (True, "All givens I_source, I_beam, lambda, alpha are present")

    # ----------------------------------------------------------------------
    # Invariant tests: physics laws (use math.isclose with tolerance)
    # ----------------------------------------------------------------------
    def test_invariant_angular_compression(self):
        """
        For part (a): r = sqrt(I_beam / I_source) by etendue conservation.
        Re-derive expected r from givens.
        """
        if self.r is None:
            return (False, "SKIP: r not available")
        if self.I_source is None or self.I_beam is None:
            return (True, "SKIP: I_source or I_beam not available to verify r")
        if self.I_source <= 0 or self.I_beam <= 0:
            return (False, "I_source and I_beam must be positive")

        expected_r = math.sqrt(self.I_beam / self.I_source)
        if math.isclose(self.r, expected_r, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"r = {self.r} matches sqrt(I_beam/I_source) = {expected_r:.4f}")
        else:
            return (False, f"r = {self.r} deviates from expected {expected_r:.4f} (relative diff {abs(self.r-expected_r)/expected_r:.3f})")

    def test_invariant_interferometer_baseline(self):
        """
        For part (c): h = 1.22 * lambda / alpha (Rayleigh criterion for circular disk).
        """
        if self.h is None:
            return (False, "SKIP: h not available")
        if self.lambda_wavelength is None or self.alpha is None:
            return (True, "SKIP: lambda or alpha not available to verify h")
        if self.lambda_wavelength <= 0 or self.alpha <= 0:
            return (False, "lambda and alpha must be positive")

        expected_h = 1.22 * self.lambda_wavelength / self.alpha
        if math.isclose(self.h, expected_h, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"h = {self.h} matches 1.22*lambda/alpha = {expected_h:.4e}")
        else:
            return (False, f"h = {self.h} deviates from expected {expected_h:.4e} (relative diff {abs(self.h-expected_h)/expected_h:.3f})")

    def test_invariant_positivity(self):
        """Both r and h must be positive."""
        messages = []
        if self.r is not None and self.r <= 0:
            messages.append(f"r must be >0, got {self.r}")
        if self.h is not None and self.h <= 0:
            messages.append(f"h must be >0, got {self.h}")
        if messages:
            return (False, "; ".join(messages))
        return (True, "r and h are positive")

    # ----------------------------------------------------------------------
    # Modeling/trace tests: check consistency with explanation_steps
    # ----------------------------------------------------------------------
    def test_modeling_explanation_contains_concepts(self):
        """
        Check that explanation steps mention key physical concepts.
        This is a weak consistency check.
        """
        if self.explanation_steps is None:
            return (True, "SKIP: no explanation_steps available")
        content = " ".join(self.explanation_steps).lower()
        concepts = ["etendue", "angular compression", "rayleigh", "visibility"]
        found = [c for c in concepts if c in content]
        if found:
            return (True, f"Explanation steps mention: {', '.join(found)}")
        else:
            return (True, "Explanation steps do not mention key concepts; optional trace")

    def test_modeling_r_from_pseudocode(self):
        """
        The pseudocode computed r = sqrt(I_beam/I_source). 
        If givens exist, verify that r is at least consistent with that formula.
        (This is a softer version of the invariant test.)
        """
        if self.r is None:
            return (True, "SKIP: r not available")
        if self.I_source is not None and self.I_beam is not None:
            # Compute allowed range using large tolerance (10%)
            low = 0.9 * math.sqrt(self.I_beam / self.I_source)
            high = 1.1 * math.sqrt(self.I_beam / self.I_source)
            if low <= self.r <= high:
                return (True, f"r lies in [{low:.4f}, {high:.4f}] around sqrt(I_beam/I_source)")
            else:
                return (False, f"r = {self.r} is outside 10% band of expected sqrt(I_beam/I_source) = {math.sqrt(self.I_beam/self.I_source):.4f}")
        # If givens missing, just check that r > 1
        if self.r > 1:
            return (True, f"r = {self.r} > 1 (consistent with amplification)")
        return (True, "SKIP: cannot verify r from givens")

    def test_modeling_h_from_rayleigh(self):
        """Check that h is consistent with Rayleigh criterion if lambda and alpha available."""
        if self.h is None:
            return (True, "SKIP: h not available")
        if self.lambda_wavelength is not None and self.alpha is not None:
            expected = 1.22 * self.lambda_wavelength / self.alpha
            if math.isclose(self.h, expected, rel_tol=0.1, abs_tol=1e-3):
                return (True, f"h = {self.h} agrees with Rayleigh criterion (10% tolerance)")
            else:
                return (False, f"h = {self.h} does not match Rayleigh estimate {expected:.4e}")
        # Without givens, just check that h is positive and finite
        if self.h > 0 and math.isfinite(self.h):
            return (True, "h is positive and finite")
        return (True, "SKIP: cannot verify h")

    # ----------------------------------------------------------------------
    # Monotonicity tests: directional trends
    # ----------------------------------------------------------------------
    def test_monotonicity_r_increases_with_I_beam(self):
        """
        Part (a): If I_beam is multiplied by factor >1, r should increase.
        We use a relative perturbation: multiply I_beam by 1.05.
        Since we cannot recompute the full solution, we test the expected monotonic relationship:
        r_new > r_old if I_beam increases.
        """
        if self.r is None or self.I_source is None or self.I_beam is None:
            return (True, "SKIP: insufficient data for monotonicity of r")
        # Baseline
        r_old = self.r
        # Perturb: increase I_beam by 5%
        I_beam_new = self.I_beam * 1.05
        # Expected new r (if etendue holds)
        r_new_expected = math.sqrt(I_beam_new / self.I_source)
        if r_new_expected > r_old:
            return (True, "Increasing I_beam increases r (consistent with monotonicity)")
        else:
            return (False, f"Perturbed I_beam x1.05 => r_new = {r_new_expected:.4f} not > r_old = {r_old:.4f}")

    def test_monotonicity_h_increases_with_wavelength(self):
        """
        Part (c): h = 1.22 * lambda / alpha, so h increases when lambda increases.
        Perturb: multiply lambda by 1.05.
        """
        if self.h is None or self.lambda_wavelength is None or self.alpha is None:
            return (True, "SKIP: insufficient data for monotonicity of h")
        lambda_new = self.lambda_wavelength * 1.05
        h_new_expected = 1.22 * lambda_new / self.alpha
        if h_new_expected > self.h:
            return (True, "Increasing lambda increases h (consistent with Rayleigh criterion)")
        else:
            return (False, f"Perturbed lambda x1.05 => h_new = {h_new_expected:.4e} not > h = {self.h:.4e}")

    def test_monotonicity_h_decreases_with_alpha(self):
        """
        h is inversely proportional to alpha. Increase alpha by 5% should decrease h.
        """
        if self.h is None or self.alpha is None or self.lambda_wavelength is None:
            return (True, "SKIP: insufficient data")
        alpha_new = self.alpha * 1.05
        h_new_expected = 1.22 * self.lambda_wavelength / alpha_new
        if h_new_expected < self.h:
            return (True, "Increasing alpha decreases h (consistent)")
        else:
            return (False, f"Perturbed alpha x1.05 => h_new = {h_new_expected:.4e} not < h = {self.h:.4e}")

    # ----------------------------------------------------------------------
    # Answer sanity test (weak, secondary)
    # ----------------------------------------------------------------------
    def test_answer_sanity_basic(self):
        """
        Perform a weak sanity check: r should be >10 (since I_beam >> I_source),
        h should be positive and finite.
        Do NOT re-derive exact values.
        """
        messages = []
        if self.r is not None:
            if self.r <= 10:
                messages.append(f"r = {self.r} is suspiciously small for a 10^6 beam from 10^3 source")
        if self.h is not None:
            if not math.isfinite(self.h) or self.h <= 0:
                messages.append(f"h = {self.h} is not positive finite")
        if messages:
            return (True, "; ".join(messages))  # weak, so return True but log warning
        return (True, "Basic sanity passed: r appears large and h is positive finite")

    # ----------------------------------------------------------------------
    # Main runner
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