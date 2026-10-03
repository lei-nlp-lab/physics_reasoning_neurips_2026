import math
import traceback

class PhysicsChecker:
    """
    Audits the solution for the satellite headlight resolution problem.
    Checks interface, physical invariants, modeling quality, monotonicity,
    and a weak final-answer sanity.
    """

    def __init__(self, solution_instance):
        # Extract results and givens safely (default to empty dict)
        self.results = getattr(solution_instance, 'results', {})
        self.givens = getattr(solution_instance, 'givens', {})

        # Expected result quantities
        self.dx = self.results.get('dx', None)
        self.s_result = self.results.get('s', None)

        # Expected given parameters (may be missing)
        self.f = self.givens.get('f', None)      # focal length (m)
        self.N = self.givens.get('N', None)      # f-number
        self.wavelength = self.givens.get('lambda', None)  # m
        self.L = self.givens.get('L', None)      # distance (m)
        self.s_given = self.givens.get('s', None)  # headlight separation (m)

    # ------------------------------------------------------------------
    # Interface checks — required keys must exist and have valid type
    # ------------------------------------------------------------------
    def test_interface_results_dx(self):
        """Check that results contain 'dx' as a finite float/int."""
        if self.dx is None:
            return False, "Missing 'dx' in results"
        if not isinstance(self.dx, (int, float)):
            return False, f"'dx' is {type(self.dx).__name__}, not numeric"
        if not math.isfinite(self.dx):
            return False, f"'dx' is not finite (value: {self.dx})"
        return True, f"dx = {self.dx} present and numeric"

    def test_interface_results_s(self):
        """Check that results contain 's' as a finite float/int."""
        if self.s_result is None:
            return False, "Missing 's' in results"
        if not isinstance(self.s_result, (int, float)):
            return False, f"'s' is {type(self.s_result).__name__}, not numeric"
        if not math.isfinite(self.s_result):
            return False, f"'s' is not finite (value: {self.s_result})"
        return True, f"s = {self.s_result} present and numeric"

    # ------------------------------------------------------------------
    # Invariant checks — theorem-based physical consistency
    # ------------------------------------------------------------------
    def test_invariant_dx_positive(self):
        """dx must be positive (physical)."""
        if self.dx is None:
            return False, "SKIP: dx not available"
        if self.dx <= 0:
            return False, f"dx must be positive, got {self.dx}"
        return True, f"dx > 0: {self.dx}"

    def test_invariant_s_positive(self):
        """s must be positive (physical)."""
        if self.s_result is None:
            return False, "SKIP: s not available"
        if self.s_result <= 0:
            return False, f"s must be positive, got {self.s_result}"
        return True, f"s > 0: {self.s_result}"

    def test_invariant_rayleigh_dx(self):
        """
        Check that dx = 1.22 * lambda * L / D, with D = f / N.
        Only if all required givens are present.
        """
        if any(x is None for x in (self.f, self.N, self.wavelength, self.L)):
            return False, "SKIP: missing one or more givens (f, N, lambda, L)"
        if self.dx is None:
            return False, "SKIP: dx not available in results"

        D = self.f / self.N
        theta_R = 1.22 * self.wavelength / D
        dx_expected = theta_R * self.L

        if not math.isclose(self.dx, dx_expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"dx from results ({self.dx}) differs from expected ({dx_expected})"
        return True, f"dx matches Rayleigh criterion: {self.dx} ≈ {dx_expected}"

    # ------------------------------------------------------------------
    # Modeling/Trace quality checks — local relations derived from physics
    # ------------------------------------------------------------------
    def test_modeling_resolution_condition(self):
        """
        If dx > s then the headlights CANNOT be resolved.
        Check that the computed dx and s satisfy this necessary condition.
        """
        if self.dx is None or self.s_result is None:
            return False, "SKIP: dx or s not available"
        if self.dx <= self.s_result:
            return False, f"dx ({self.dx}) <= s ({self.s_result}) -> would be resolvable, contradicts problem"
        return True, f"dx > s, consistent with 'cannot resolve'"

    def test_trace_aperture_diameter(self):
        """
        Verify that the aperture diameter D = f / N is positive and reasonable.
        This is a clean derived quantity that every solver should use.
        """
        if self.f is None or self.N is None:
            return False, "SKIP: missing f or N givens"
        D = self.f / self.N
        if D <= 0:
            return False, f"Aperture diameter D = {D} is not positive"
        # typical value: 25 mm = 0.025 m; allow wide range
        if not (0.001 < D < 0.5):
            return False, f"Aperture diameter D = {D} m is outside plausible range (0.001, 0.5)"
        return True, f"Aperture diameter D = {D} m is positive and plausible"

    def test_trace_angular_resolution_order(self):
        """
        Check that the angular resolution theta_R = dx / L is on the order of
        10^-5 rad (typical for diffraction-limited optical systems of this size).
        This validates the solver’s intermediate calculation without hardcoding.
        """
        if self.dx is None or self.L is None:
            return False, "SKIP: dx or L not available"
        if self.L <= 0:
            return False, f"L must be positive, got {self.L}"
        theta_R = self.dx / self.L
        # Expected range: 1e-6 to 1e-4 rad (covers typical visible-light diffraction)
        if not (1e-6 < theta_R < 1e-4):
            return False, f"Angular resolution theta_R = {theta_R} rad is outside reasonable range (1e-6, 1e-4)"
        return True, f"theta_R = {theta_R} rad is within physically plausible range"

    # ------------------------------------------------------------------
    # Monotonicity checks — directional trends under parameter change
    # ------------------------------------------------------------------
    def test_monotonicity_aperture_diameter(self):
        """
        dx ∝ 1/D.  If we increase D (by decreasing N or increasing f),
        dx should decrease.
        Perturb D by +10% and verify dx decreases.
        """
        if any(x is None for x in (self.f, self.N, self.wavelength, self.L)):
            return False, "SKIP: missing one or more givens (f, N, lambda, L)"
        if self.dx is None:
            return False, "SKIP: dx not available"

        baseline_D = self.f / self.N
        # Increase D by 10%
        perturbed_D = baseline_D * 1.10
        # Recompute dx with same lambda, L
        theta_R_pert = 1.22 * self.wavelength / perturbed_D
        dx_pert = theta_R_pert * self.L

        if dx_pert >= self.dx - 1e-12:  # allow tiny float tolerance
            return False, f"Increasing D did not decrease dx: baseline dx={self.dx}, perturbed dx={dx_pert}"
        return True, f"dx decreases when D increases (monotonic): {self.dx} -> {dx_pert}"

    # ------------------------------------------------------------------
    # Weak answer sanity — only very broad checks, NO isclose / derivation
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """
        Verify that dx and s are finite, positive, and within a plausible range
        for the problem (headlight separation ~ 1‑2 m, dx a few meters).
        """
        if self.dx is None or self.s_result is None:
            return False, "SKIP: dx or s not available"
        plausible_dx = 0.5 < self.dx < 10.0
        plausible_s = 0.5 < self.s_result < 3.0
        if not plausible_dx:
            return False, f"dx = {self.dx} not in plausible range (0.5, 10) m"
        if not plausible_s:
            return False, f"s = {self.s_result} not in plausible range (0.5, 3) m"
        return True, "dx and s are finite, positive, and within plausible ranges"

    # ------------------------------------------------------------------
    # Master runner — collect all test_ methods
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)