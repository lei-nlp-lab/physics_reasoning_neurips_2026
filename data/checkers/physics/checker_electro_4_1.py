import math
import traceback

class PhysicsChecker:
    """
    Physics auditor for the electromagnetic wave problem.
    Extracts givens and results safely using None defaults,
    then runs a suite of unit tests.
    """

    def __init__(self, solution_instance):
        # Safely extract results and givens dicts
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}

        self.givens = getattr(solution_instance, 'givens', None)
        if self.givens is None:
            self.givens = {}

        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        if self.explanation_steps is None:
            self.explanation_steps = {}

        # Result keys – all default to None, no numeric fallbacks
        self.f = self.results.get('f', None)
        self.lambda_wave = self.results.get('lambda', None)
        self.prop_dir = self.results.get('prop_dir', None)
        self.B_dir = self.results.get('B_dir', None)

        # Givens (may be missing entirely) – store for re‑derivation
        self.omega = self.givens.get('omega', None)
        self.k = self.givens.get('k', None)

        # Also attempt to retrieve from explanation_steps (if structured)
        self.steps_omega = (self.explanation_steps.get('omega', None)
                           if isinstance(self.explanation_steps, dict) else None)
        self.steps_k = (self.explanation_steps.get('k', None)
                       if isinstance(self.explanation_steps, dict) else None)

    # ------------------------------------------------------------------
    # INTERFACE TESTS: required keys exist and have valid types
    # ------------------------------------------------------------------

    def test_interface_has_f(self):
        """Check that results contain 'f' and it is a number."""
        if self.f is None:
            return (False, "Missing result key 'f'")
        if not isinstance(self.f, (int, float)):
            return (False, f"'f' is not numeric, got {type(self.f)}")
        return (True, f"'f' = {self.f}")

    def test_interface_has_lambda(self):
        """Check that results contain 'lambda' and it is a number."""
        if self.lambda_wave is None:
            return (False, "Missing result key 'lambda'")
        if not isinstance(self.lambda_wave, (int, float)):
            return (False, f"'lambda' is not numeric, got {type(self.lambda_wave)}")
        return (True, f"'lambda' = {self.lambda_wave}")

    def test_interface_has_prop_dir(self):
        """Check that results contain 'prop_dir' and it is a non‑empty string."""
        if self.prop_dir is None:
            return (False, "Missing result key 'prop_dir'")
        if not isinstance(self.prop_dir, str) or len(self.prop_dir) == 0:
            return (False, f"'prop_dir' is not a non‑empty string")
        return (True, f"'prop_dir' = '{self.prop_dir}'")

    def test_interface_has_B_dir(self):
        """Check that results contain 'B_dir' and it is a non‑empty string."""
        if self.B_dir is None:
            return (False, "Missing result key 'B_dir'")
        if not isinstance(self.B_dir, str) or len(self.B_dir) == 0:
            return (False, f"'B_dir' is not a non‑empty string")
        return (True, f"'B_dir' = '{self.B_dir}'")

    def test_interface_parameters_positive(self):
        """Simple positivity check: frequency and wavelength must be > 0."""
        if self.f is None or self.lambda_wave is None:
            return (False, "SKIP: freq or wavelength missing")
        if self.f <= 0:
            return (False, f"Frequency is not positive: {self.f}")
        if self.lambda_wave <= 0:
            return (False, f"Wavelength is not positive: {self.lambda_wave}")
        return (True, "f and lambda are positive")

    # ------------------------------------------------------------------
    # INVARIANT (THEOREM‑BASED) TESTS
    # ------------------------------------------------------------------

    def test_invariant_speed_of_light(self):
        """
        Check that f * λ = c (3×10⁸ m/s) within 1% tolerance.
        Uses a hard‑coded vacuum speed of light (standard constant).
        """
        if self.f is None or self.lambda_wave is None:
            return (False, "SKIP: freq or wavelength missing")
        c = 3e8
        product = self.f * self.lambda_wave
        if not math.isclose(product, c, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"f*λ = {product} != c = {c}")
        return (True, f"f*λ = {product} ≈ c = {c}")

    def test_invariant_propagation_direction_from_phase(self):
        """
        The wave has phase (ωt – kx), which implies propagation in the +x direction.
        Check that the reported direction matches this.
        """
        if self.prop_dir is None:
            return (False, "SKIP: prop_dir missing")
        allowed = ["+x", "positive x", "+x direction", "x", "x direction"]
        if self.prop_dir.lower() in [s.lower() for s in allowed]:
            return (True, f"propagation '{self.prop_dir}' consistent with (ωt – kx)")
        else:
            return (False, f"propagation '{self.prop_dir}' does not match +x")

    def test_invariant_right_hand_rule_orthogonality(self):
        """
        For a plane EM wave: E × B ∥ k̂. Given E along y and k̂ along +x,
        B must be along +z. Check that B_dir matches this.
        """
        if self.prop_dir is None or self.B_dir is None:
            return (False, "SKIP: prop_dir or B_dir missing")
        # Confirm propagation is +x
        prop_plus_x = any(self.prop_dir.lower() == s for s in ["+x", "positive x", "+x direction", "x"])
        if not prop_plus_x:
            return (False, "Propagation is not +x; cannot verify right‑hand rule")
        expected_B = ["+z", "positive z", "z", "z direction", "+z direction"]
        if self.B_dir.lower() in [s.lower() for s in expected_B]:
            return (True, f"B direction '{self.B_dir}' follows right‑hand rule")
        else:
            return (False, f"B direction '{self.B_dir}' should be +z")

    # ------------------------------------------------------------------
    # MODELING / TRACE TESTS (one relation at a time)
    # ------------------------------------------------------------------

    def test_modeling_frequency_derivation(self):
        """If ω is available, verify f = ω/(2π)."""
        omega = self.omega or self.steps_omega
        if omega is None or self.f is None:
            return (False, "SKIP: ω or f not available")
        expected_f = omega / (2 * math.pi)
        if not math.isclose(self.f, expected_f, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"f = {self.f} does not match ω/(2π) = {expected_f}")
        return (True, f"f = {self.f} ≈ ω/(2π) = {expected_f}")

    def test_modeling_wavelength_derivation(self):
        """If k is available, verify λ = 2π/k."""
        k = self.k or self.steps_k
        if k is None or self.lambda_wave is None:
            return (False, "SKIP: k or λ not available")
        expected_lambda = 2 * math.pi / k
        if not math.isclose(self.lambda_wave, expected_lambda, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"λ = {self.lambda_wave} does not match 2π/k = {expected_lambda}")
        return (True, f"λ = {self.lambda_wave} ≈ 2π/k = {expected_lambda}")

    def test_modeling_omega_over_k_equals_c(self):
        """If ω and k are known, verify ω/k = c."""
        omega = self.omega or self.steps_omega
        k = self.k or self.steps_k
        if omega is None or k is None:
            return (False, "SKIP: ω or k not available")
        ratio = omega / k
        c = 3e8
        if not math.isclose(ratio, c, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"ω/k = {ratio} != c = {c}")
        return (True, f"ω/k = {ratio} ≈ c")

    # ------------------------------------------------------------------
    # MONOTONICITY TESTS (explicit perturbation of an input)
    # ------------------------------------------------------------------

    def test_monotonicity_frequency_increases_with_omega(self):
        """
        Given ω, perturb it by +1% and verify that f increases.
        f = ω/(2π), so a +1% increase in ω yields a +1% increase in f.
        """
        omega = self.omega or self.steps_omega
        if omega is None or self.f is None:
            return (False, "SKIP: ω or f not available")
        # baseline f derived from current ω
        baseline_f = omega / (2 * math.pi)
        perturbation = 1.01
        perturbed_omega = omega * perturbation
        perturbed_f = perturbed_omega / (2 * math.pi)
        # Check that perturbed f is larger than baseline
        if perturbed_f <= baseline_f:
            return (False, f"Perturbed f ({perturbed_f}) should be > baseline ({baseline_f})")
        # Also check that the ratio is the perturbation factor
        if not math.isclose(perturbed_f / baseline_f, perturbation, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Ratio perturbed/baseline f = {perturbed_f/baseline_f} != {perturbation}")
        return (True, f"f increases with ω: +1% ω → f ratio = {perturbed_f/baseline_f} ≈ {perturbation}")

    def test_monotonicity_wavelength_decreases_with_k(self):
        """
        Given k, perturb it by +1% and verify that λ decreases.
        λ = 2π/k, so a +1% increase in k yields approximately -1% change in λ.
        """
        k = self.k or self.steps_k
        if k is None or self.lambda_wave is None:
            return (False, "SKIP: k or λ not available")
        baseline_lambda = 2 * math.pi / k
        perturbation = 1.01
        perturbed_k = k * perturbation
        perturbed_lambda = 2 * math.pi / perturbed_k
        # Check that perturbed λ is smaller
        if perturbed_lambda >= baseline_lambda:
            return (False, f"Perturbed λ ({perturbed_lambda}) should be < baseline ({baseline_lambda})")
        # Check that the ratio is inverse of perturbation
        ratio = perturbed_lambda / baseline_lambda
        expected_ratio = 1 / perturbation
        if not math.isclose(ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"λ ratio = {ratio} != 1/{perturbation} = {expected_ratio}")
        return (True, f"λ decreases with k: +1% k → λ ratio = {ratio} ≈ {expected_ratio}")

    # ------------------------------------------------------------------
    # WEAK ANSWER SANITY (only one method allowed)
    # ------------------------------------------------------------------

    def test_answer_sanity_values(self):
        """
        Weak final‑answer sanity: frequency and wavelength are positive finite numbers;
        propagation and B directions are plausible strings. No strict numerical comparison,
        no re‑derivation.
        """
        messages = []
        ok = True

        # frequency
        if self.f is None:
            messages.append("f missing")
            ok = False
        elif not (math.isfinite(self.f) and self.f > 0):
            messages.append(f"f = {self.f} is not positive finite")
            ok = False
        else:
            messages.append(f"f = {self.f} plausible")

        # wavelength
        if self.lambda_wave is None:
            messages.append("λ missing")
            ok = False
        elif not (math.isfinite(self.lambda_wave) and self.lambda_wave > 0):
            messages.append(f"λ = {self.lambda_wave} is not positive finite")
            ok = False
        else:
            messages.append(f"λ = {self.lambda_wave} plausible")

        # propagation direction
        if self.prop_dir is None:
            messages.append("prop_dir missing")
            ok = False
        elif not isinstance(self.prop_dir, str) or len(self.prop_dir) == 0:
            messages.append(f"prop_dir invalid (type {type(self.prop_dir)})")
            ok = False
        else:
            # check it contains 'x' (case‑insensitive)
            if 'x' in self.prop_dir.lower():
                messages.append(f"prop_dir '{self.prop_dir}' contains 'x'")
            else:
                messages.append(f"prop_dir '{self.prop_dir}' does not contain 'x'")
                ok = False

        # B direction
        if self.B_dir is None:
            messages.append("B_dir missing")
            ok = False
        elif not isinstance(self.B_dir, str) or len(self.B_dir) == 0:
            messages.append(f"B_dir invalid (type {type(self.B_dir)})")
            ok = False
        else:
            if 'z' in self.B_dir.lower():
                messages.append(f"B_dir '{self.B_dir}' contains 'z'")
            else:
                messages.append(f"B_dir '{self.B_dir}' does not contain 'z'")
                ok = False

        if ok:
            return (True, "; ".join(messages))
        else:
            return (False, "; ".join(messages))

    # ------------------------------------------------------------------
    # RUN ALL TESTS (with discovery and try/except per test)
    # ------------------------------------------------------------------

    def run_all_tests(self):
        """
        Discover and run every bound method whose name starts with 'test_'.
        Each call is wrapped in try/except. Returns (all_pass, logs_list).
        """
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
                msg = f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)