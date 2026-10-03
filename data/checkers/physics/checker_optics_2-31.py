import math

class PhysicsChecker:
    """
    Audits solutions for Fresnel diffraction by a circular aperture/disk.
    Checks interface completeness, physics invariants, modeling quality,
    monotonic trends, and weak answer sanity.
    """
    def __init__(self, solution_instance):
        # Extract givens and results safely
        givens = getattr(solution_instance, 'givens', None)
        results = getattr(solution_instance, 'results', None)
        self.givens = givens if isinstance(givens, dict) else {}
        self.results = results if isinstance(results, dict) else {}

        # Givens (wavelength, distance, incident intensity)
        self.lamb = self._safe_get(self.givens, 'lambda', None)
        self.L = self._safe_get(self.givens, 'L', None)
        self.I0 = self._safe_get(self.givens, 'I0', None)

        # Expected emitted results
        self.r_a = self._safe_get(self.results, 'r_a', None)
        self.r_b = self._safe_get(self.results, 'r_b', None)
        self.ratio = self._safe_get(self.results, 'ratio', None)
        self.I_disk = self._safe_get(self.results, 'I_disk', None)

    def _safe_get(self, container, key, default=None):
        if isinstance(container, dict):
            return container.get(key, default)
        return getattr(container, key, default) if container else default

    # ---------- Interface tests ----------
    def test_interface_required_keys_present(self):
        required = ['r_a', 'r_b', 'ratio', 'I_disk']
        for key in required:
            if self._safe_get(self.results, key) is None:
                return (False, f"Required key '{key}' missing in results.")
        return (True, "All required result keys present.")

    def test_interface_keys_numeric(self):
        for name, val in [('r_a', self.r_a), ('r_b', self.r_b), ('ratio', self.ratio)]:
            if val is not None:
                if not isinstance(val, (int, float)):
                    return (False, f"{name} = {val} is not numeric (type {type(val).__name__}).")
        return (True, "Interface keys have numeric types.")

    def test_interface_givens_available(self):
        if self.lamb is None or self.L is None:
            return (False, "Missing lambda or L in givens (required for physical checks).")
        return (True, "Essential givens present.")

    # ---------- Invariant (theorem-based) tests ----------
    def test_invariant_maximum_phase(self):
        """ Check that at r = r_a, sin^2(phase) = 1 (first maximum). """
        if any(v is None for v in [self.lamb, self.L, self.r_a]):
            return (False, "SKIP: lambda, L, or r_a missing.")
        k = 2 * math.pi / self.lamb
        phase = k * (self.r_a ** 2) / (4 * self.L)
        sin2 = math.sin(phase) ** 2
        if not math.isclose(sin2, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"sin^2(phase) = {sin2:.4f} at r_a, expected ≈1.")
        return (True, "Phase at r_a yields maximum intensity.")

    def test_invariant_minimum_phase(self):
        """ Check that at r = r_b, sin^2(phase) = 0 (first minimum). """
        if any(v is None for v in [self.lamb, self.L, self.r_b]):
            return (False, "SKIP: lambda, L, or r_b missing.")
        k = 2 * math.pi / self.lamb
        phase = k * (self.r_b ** 2) / (4 * self.L)
        sin2 = math.sin(phase) ** 2
        if not math.isclose(sin2, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"sin^2(phase) = {sin2:.4f} at r_b, expected ≈0.")
        return (True, "Phase at r_b yields minimum intensity.")

    def test_invariant_zone_radius_relation(self):
        """ Check Fresnel zone condition: r_a^2 = λ L, r_b^2 = 2 λ L. """
        if any(v is None for v in [self.lamb, self.L, self.r_a, self.r_b]):
            return (False, "SKIP: lambda, L, r_a, or r_b missing.")
        if not math.isclose(self.r_a ** 2, self.lamb * self.L, rel_tol=1e-2):
            return (False, f"r_a^2 = {self.r_a**2:.4f}, expected {self.lamb*self.L:.4f}.")
        if not math.isclose(self.r_b ** 2, 2 * self.lamb * self.L, rel_tol=1e-2):
            return (False, f"r_b^2 = {self.r_b**2:.4f}, expected {2*self.lamb*self.L:.4f}.")
        return (True, "Fresnel zone radius relations satisfied.")

    def test_invariant_babinet_disk_intensity(self):
        """ Check that the disk intensity equals I0 (Babinet). """
        if self.I0 is None or self.I_disk is None:
            return (False, "SKIP: I0 or I_disk missing.")
        # Accept both symbolic string and numeric value
        if isinstance(self.I_disk, str):
            if self.I_disk.strip() in ('I_\\infty', 'I_infty', 'I0'):
                return (True, f"I_disk symbolically recognized as {self.I_disk}.")
            else:
                return (False, f"I_disk string '{self.I_disk}' not recognized as I_infinity.")
        # Numeric check
        if not math.isclose(self.I_disk, self.I0, rel_tol=1e-2):
            return (False, f"I_disk = {self.I_disk}, expected I0 = {self.I0}.")
        return (True, "I_disk equals I0 (Babinet principle).")

    # ---------- Modeling / trace consistency tests ----------
    def test_modeling_ratio_is_four(self):
        """ Check that the intensity ratio I(r_a)/I_inf = 4. """
        if self.ratio is None:
            return (False, "SKIP: ratio missing.")
        if not math.isclose(self.ratio, 4.0, rel_tol=1e-2):
            return (False, f"ratio = {self.ratio}, expected 4.")
        return (True, "Intensity ratio is 4.")

    def test_modeling_zone_order(self):
        """ Check that r_b > r_a (first minimum occurs after first maximum). """
        if self.r_a is None or self.r_b is None:
            return (False, "SKIP: r_a or r_b missing.")
        if not (self.r_b > self.r_a):
            return (False, f"r_b = {self.r_b} is not greater than r_a = {self.r_a}.")
        return (True, "r_b > r_a, consistent with Fresnel zone ordering.")

    # ---------- Monotonicity / trend tests (explicit perturbation) ----------
    def test_monotonicity_phase_increases_with_r(self):
        """
        Explicitly perturb the input r (take r_a) and check that the Fresnel
        phase argument increases with r.
        """
        if any(v is None for v in [self.lamb, self.L, self.r_a]):
            return (False, "SKIP: lambda, L, or r_a missing.")
        k = 2 * math.pi / self.lamb

        # Baseline phase at r = r_a
        phase0 = k * (self.r_a ** 2) / (4 * self.L)

        # Perturb r: increase by 10%
        r_perturbed = self.r_a * 1.1
        phase1 = k * (r_perturbed ** 2) / (4 * self.L)

        if not (phase1 > phase0):
            return (False, f"Phase did not increase after +10% perturbation: "
                           f"baseline {phase0:.4f}, perturbed {phase1:.4f}.")
        return (True, f"Phase increases from {phase0:.4f} to {phase1:.4f} "
                       "when r is perturbed upward by 10%.")

    def test_monotonicity_zone_radius_with_wavelength(self):
        """
        Explicitly perturb lambda (wavelength) and check that r_a increases.
        """
        if any(v is None for v in [self.lamb, self.L, self.r_a]):
            return (False, "SKIP: lambda, L, or r_a missing.")

        # Baseline r_a
        r_a_base = self.r_a

        # Perturb lambda: increase by 10%
        new_lambda = self.lamb * 1.1
        expected_r_a_pert = math.sqrt(new_lambda * self.L)

        # From Fresnel condition, r_a should increase
        if not (expected_r_a_pert > r_a_base):
            return (False, f"Expected r_a after +10% wavelength is {expected_r_a_pert:.4f}, "
                           f"which is not > baseline {r_a_base:.4f}.")
        return (True, f"With +10% wavelength, r_a increases from {r_a_base:.4f} "
                       f"to {expected_r_a_pert:.4f} (consistent with sqrt(lambda L)).")

    # ---------- Weak answer sanity tests ----------
    def test_answer_sanity_positive_finite(self):
        """ Basic sanity: r_a, r_b, ratio are positive and finite. """
        for name, val in [('r_a', self.r_a), ('r_b', self.r_b), ('ratio', self.ratio)]:
            if val is not None:
                if not (isinstance(val, (int, float)) and val > 0 and math.isfinite(val)):
                    return (False, f"{name} = {val} is not positive and finite.")
        return (True, "Answer sanity: all numeric outputs are positive and finite.")

    # ---------- Test runner ----------
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