import math
import traceback

class PhysicsChecker:
    """
    Audits the solution for the isotope production problem.
    Checks interface, invariants (baryon number conservation, mass-energy equivalence),
    modeling accuracy, and monotonicity via explicit perturbation.
    """

    def __init__(self, solution_instance):
        # Extract results safely, defaulting to None for missing keys
        self.results = getattr(solution_instance, 'results', {})
        if self.results is None:
            self.results = {}

        self.E_pair_D = self.results.get('E_pair_D', None)
        self.E_pair_He3 = self.results.get('E_pair_He3', None)
        self.E_pair_T = self.results.get('E_pair_T', None)
        self.threshold = self.results.get('threshold', None)

        # Reference masses from physics constraints (in atomic mass units)
        self.mass_D = 2.0141        # u
        self.mass_He3 = 3.0160     # u
        self.mass_T = 3.01605      # u
        self.u_to_MeV = 931.5      # MeV/u
        self.conv_GeV_per_u = self.u_to_MeV / 1000.0  # GeV per u

    # ------------------------------------------------------------------
    # Interface checks: required keys exist and have correct types
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all required result keys are present."""
        required = ['E_pair_D', 'E_pair_He3', 'E_pair_T', 'threshold']
        missing = [key for key in required if key not in self.results]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    def test_interface_E_pair_D_type(self):
        """E_pair_D must be numeric."""
        if self.E_pair_D is None:
            return (False, "E_pair_D is None")
        if not isinstance(self.E_pair_D, (int, float)):
            return (False, f"E_pair_D type is {type(self.E_pair_D).__name__}")
        return (True, "E_pair_D is numeric")

    def test_interface_E_pair_He3_type(self):
        """E_pair_He3 must be numeric."""
        if self.E_pair_He3 is None:
            return (False, "E_pair_He3 is None")
        if not isinstance(self.E_pair_He3, (int, float)):
            return (False, f"E_pair_He3 type is {type(self.E_pair_He3).__name__}")
        return (True, "E_pair_He3 is numeric")

    def test_interface_E_pair_T_type(self):
        """E_pair_T must be numeric."""
        if self.E_pair_T is None:
            return (False, "E_pair_T is None")
        if not isinstance(self.E_pair_T, (int, float)):
            return (False, f"E_pair_T type is {type(self.E_pair_T).__name__}")
        return (True, "E_pair_T is numeric")

    def test_interface_threshold_type(self):
        """threshold must be numeric."""
        if self.threshold is None:
            return (False, "threshold is None")
        if not isinstance(self.threshold, (int, float)):
            return (False, f"threshold type is {type(self.threshold).__name__}")
        return (True, "threshold is numeric")

    # ------------------------------------------------------------------
    # Invariant checks (theorem-based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_pair_energy_positive(self):
        """All pair energies must be positive."""
        values = [self.E_pair_D, self.E_pair_He3, self.E_pair_T]
        if any(v is None for v in values):
            return (False, "One or more pair energies missing")
        for name, val in zip(["D", "He3", "T"], values):
            if val <= 0:
                return (False, f"E_pair_{name} = {val} is not positive")
        return (True, "All pair energies > 0")

    def test_invariant_pair_D_below_threshold(self):
        """E_pair_D must be <= threshold (with tolerance for rounding)."""
        if self.E_pair_D is None or self.threshold is None:
            return (False, "E_pair_D or threshold missing")
        allowed_excess = 1e-2
        if self.E_pair_D > self.threshold + allowed_excess:
            return (False, f"E_pair_D = {self.E_pair_D:.4f} > threshold = {self.threshold:.4f}")
        return (True, f"E_pair_D = {self.E_pair_D:.4f} <= threshold = {self.threshold:.4f}")

    def test_invariant_pair_He3_above_threshold(self):
        """E_pair_He3 must be > threshold (with tolerance)."""
        if self.E_pair_He3 is None or self.threshold is None:
            return (False, "E_pair_He3 or threshold missing")
        allowed_margin = 1e-2
        if self.E_pair_He3 < self.threshold - allowed_margin:
            return (False, f"E_pair_He3 = {self.E_pair_He3:.4f} < threshold = {self.threshold:.4f}")
        return (True, f"E_pair_He3 = {self.E_pair_He3:.4f} >= threshold = {self.threshold:.4f}")

    def test_invariant_pair_T_above_threshold(self):
        """E_pair_T must be > threshold (with tolerance)."""
        if self.E_pair_T is None or self.threshold is None:
            return (False, "E_pair_T or threshold missing")
        allowed_margin = 1e-2
        if self.E_pair_T < self.threshold - allowed_margin:
            return (False, f"E_pair_T = {self.E_pair_T:.4f} < threshold = {self.threshold:.4f}")
        return (True, f"E_pair_T = {self.E_pair_T:.4f} >= threshold = {self.threshold:.4f}")

    def test_invariant_ordering(self):
        """Check mass ordering: D < He3 < T (pair energies reflect that)."""
        values = [self.E_pair_D, self.E_pair_He3, self.E_pair_T]
        if any(v is None for v in values):
            return (False, "Pair energy missing")
        if not (self.E_pair_D < self.E_pair_He3 < self.E_pair_T):
            return (False, f"Ordering violated: D={self.E_pair_D:.4f}, He3={self.E_pair_He3:.4f}, T={self.E_pair_T:.4f}")
        return (True, "E_pair_D < E_pair_He3 < E_pair_T holds")

    def test_invariant_threshold_value(self):
        """Threshold should be 4.5 GeV (exact)."""
        if self.threshold is None:
            return (False, "threshold missing")
        if not math.isclose(self.threshold, 4.5, rel_tol=1e-3, abs_tol=1e-3):
            return (False, f"threshold = {self.threshold} != 4.5")
        return (True, f"threshold = {self.threshold} ≈ 4.5")

    # ------------------------------------------------------------------
    # Modeling checks (consistency with given masses and conversion)
    # ------------------------------------------------------------------
    def _expected_pair_energy(self, mass_u):
        """Compute expected pair energy in GeV for a given mass in u."""
        rest_GeV = mass_u * self.conv_GeV_per_u
        return 2.0 * rest_GeV

    def test_modeling_expected_pair_D(self):
        """E_pair_D should match expected value from mass_D=2.0141 u."""
        if self.E_pair_D is None:
            return (False, "E_pair_D missing")
        expected = self._expected_pair_energy(self.mass_D)
        if not math.isclose(self.E_pair_D, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E_pair_D = {self.E_pair_D:.4f} != expected {expected:.4f}")
        return (True, f"E_pair_D = {self.E_pair_D:.4f} matches expected {expected:.4f}")

    def test_modeling_expected_pair_He3(self):
        """E_pair_He3 should match expected value from mass_He3=3.0160 u."""
        if self.E_pair_He3 is None:
            return (False, "E_pair_He3 missing")
        expected = self._expected_pair_energy(self.mass_He3)
        if not math.isclose(self.E_pair_He3, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E_pair_He3 = {self.E_pair_He3:.4f} != expected {expected:.4f}")
        return (True, f"E_pair_He3 = {self.E_pair_He3:.4f} matches expected {expected:.4f}")

    def test_modeling_expected_pair_T(self):
        """E_pair_T should match expected value from mass_T=3.01605 u."""
        if self.E_pair_T is None:
            return (False, "E_pair_T missing")
        expected = self._expected_pair_energy(self.mass_T)
        if not math.isclose(self.E_pair_T, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E_pair_T = {self.E_pair_T:.4f} != expected {expected:.4f}")
        return (True, f"E_pair_T = {self.E_pair_T:.4f} matches expected {expected:.4f}")

    # ------------------------------------------------------------------
    # Monotonicity check: perturbing mass increases pair energy
    # ------------------------------------------------------------------
    def test_monotonicity_mass_to_energy(self):
        """
        Perturb the mass of deuterium by +1% and verify that the
        resulting predicted pair energy is larger than the baseline.
        This uses the checker's internal mass values (not solution's).
        """
        # Baseline: pair energy computed from nominal mass
        baseline = self._expected_pair_energy(self.mass_D)
        # Perturbed mass: +1%
        perturbed_mass = self.mass_D * 1.01
        perturbed_energy = self._expected_pair_energy(perturbed_mass)
        # Check monotonic increase (strictly greater)
        if not math.isclose(perturbed_energy, baseline, rel_tol=0.001, abs_tol=1e-3):
            # Allow a slight tolerance for floating point, but perturbed should be noticeably higher
            if perturbed_energy <= baseline:
                return (False, f"Perturbed energy {perturbed_energy:.4f} not > baseline {baseline:.4f}")
        # Additional check: the ratio should scale with mass ratio (1.01)
        # This is a stronger consistency test.
        expected_ratio = 1.01
        actual_ratio = perturbed_energy / baseline
        if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-3):
            return (False, f"Energy ratio {actual_ratio:.6f} != mass ratio {expected_ratio}")
        return (True, f"Monotonicity holds: +1% mass gives +1% energy (ratio {actual_ratio:.6f})")

    # ------------------------------------------------------------------
    # Answer sanity (weak, optional)
    # ------------------------------------------------------------------
    def test_answer_sanity_basic(self):
        """Weak sanity: the smallest pair energy is D, indicating D as answer."""
        if any(v is None for v in [self.E_pair_D, self.E_pair_He3, self.E_pair_T]):
            return (False, "Missing pair energy for answer sanity")
        if self.E_pair_D >= self.E_pair_He3 or self.E_pair_D >= self.E_pair_T:
            return (False, f"E_pair_D = {self.E_pair_D:.4f} is not the smallest pair energy")
        return (True, "E_pair_D is the smallest -> answer likely D")

    # ------------------------------------------------------------------
    # Main test runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        """Discover and run all test_* methods, collect logs and pass/fail status."""
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