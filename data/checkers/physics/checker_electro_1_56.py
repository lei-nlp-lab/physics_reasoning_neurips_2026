import math
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for the dielectric sphere in a parallel-plate capacitor problem.
    Audits:
      - Existence of required result keys.
      - Continuity of normal D and tangential E at the sphere surface.
      - Bound surface charge density derived from polarization.
      - Consistency of field components with the expected vector form.
      - Monotonicity of radial field and surface charge with polar angle.
      - Finite/valid numerical values.
    """

    def __init__(self, solution_instance):
        """
        Extract all givens and results safely with None defaults.
        No numeric fallbacks.
        """
        # Givens (from problem statement or solution_instance attributes)
        self.Ke = getattr(solution_instance, 'Ke', None)
        self.E0 = getattr(solution_instance, 'E0', None)
        self.theta = getattr(solution_instance, 'theta', None)
        self.epsilon0 = getattr(solution_instance, 'epsilon0', None)

        # Results (from solution_instance.results dict)
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.E_out_r = solution_instance.results.get('E_out_r', None)
            self.E_out_theta = solution_instance.results.get('E_out_theta', None)
            self.sigma_b = solution_instance.results.get('sigma_b', None)
        else:
            self.E_out_r = None
            self.E_out_theta = None
            self.sigma_b = None

    # ------------------------------------------------------------------
    # Interface tests: check required keys exist and have valid type
    # ------------------------------------------------------------------
    def test_interface_required_results(self):
        """Check that all three required result keys are present and non-None."""
        missing = []
        for key in ['E_out_r', 'E_out_theta', 'sigma_b']:
            val = getattr(self, key, None)
            if val is None:
                missing.append(key)
        if missing:
            return (False, f"Missing result(s): {', '.join(missing)}")
        return (True, "All required results present")

    def test_interface_required_givens(self):
        """Check that essential givens are provided."""
        missing = []
        for var in ['Ke', 'E0', 'theta', 'epsilon0']:
            if getattr(self, var, None) is None:
                missing.append(var)
        if missing:
            return (False, f"Missing given(s): {', '.join(missing)}")
        return (True, "All required givens present")

    # ------------------------------------------------------------------
    # Invariant tests: theorem-based physical sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_normal_D_continuity(self):
        """
        Check continuity of normal D: ε0 * E_out_r = ε * E_in_r
        with ε = Ke * ε0 and E_in_r = E0 * cosθ.
        Derived expected: E_out_r = Ke * E0 * cosθ.
        """
        if any(v is None for v in [self.Ke, self.E0, self.theta, self.epsilon0, self.E_out_r]):
            return (False, "SKIP: missing data for D continuity check")
        expected = self.Ke * self.E0 * math.cos(self.theta)
        if not math.isclose(self.E_out_r, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E_out_r mismatch: got {self.E_out_r}, expected ~{expected}")
        return (True, "Normal D continuity satisfied")

    def test_invariant_tangential_E_continuity(self):
        """
        Check continuity of tangential E: E_out_θ = E_in_θ = -E0 sinθ.
        """
        if any(v is None for v in [self.E0, self.theta, self.E_out_theta]):
            return (False, "SKIP: missing data for tangential E check")
        expected = -self.E0 * math.sin(self.theta)
        if not math.isclose(self.E_out_theta, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E_out_theta mismatch: got {self.E_out_theta}, expected ~{expected}")
        return (True, "Tangential E continuity satisfied")

    def test_invariant_sigma_b_formula(self):
        """
        Check bound surface charge density: σ_b = ε0 (Ke - 1) E0 cosθ.
        """
        if any(v is None for v in [self.epsilon0, self.Ke, self.E0, self.theta, self.sigma_b]):
            return (False, "SKIP: missing data for sigma_b check")
        expected = self.epsilon0 * (self.Ke - 1) * self.E0 * math.cos(self.theta)
        if not math.isclose(self.sigma_b, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"sigma_b mismatch: got {self.sigma_b}, expected ~{expected}")
        return (True, "Bound charge density formula holds")

    # ------------------------------------------------------------------
    # Modeling tests: additional consistency checks
    # ------------------------------------------------------------------
    def test_modeling_sigma_b_from_polarization(self):
        """
        Derive sigma_b from polarization P = ε0(Ke-1)E0 (vector along z).
        σ_b = P·r̂ = P cosθ.
        """
        if any(v is None for v in [self.epsilon0, self.Ke, self.E0, self.theta, self.sigma_b]):
            return (False, "SKIP: missing data for polarization check")
        P_mag = self.epsilon0 * (self.Ke - 1) * self.E0
        expected_sigma = P_mag * math.cos(self.theta)
        if not math.isclose(self.sigma_b, expected_sigma, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"sigma_b not consistent with polarization: got {self.sigma_b}, expected ~{expected_sigma}")
        return (True, "sigma_b consistent with P·r̂")

    def test_modeling_field_components_consistency(self):
        """
        Verify that the stored E_out_r and E_out_theta correctly combine
        into the vector form: E_out = Ke*E0*cosθ e_r - E0*sinθ e_θ.
        Compute the magnitude from the stored components and compare to the
        theoretical magnitude derived from the same expression.
        """
        if any(v is None for v in [self.Ke, self.E0, self.theta, self.E_out_r, self.E_out_theta]):
            return (False, "SKIP: missing data for field consistency check")
        # Theoretical magnitude squared
        expected_sq = (self.Ke * self.E0 * math.cos(self.theta))**2 + (self.E0 * math.sin(self.theta))**2
        # Actual magnitude squared from stored components
        actual_sq = self.E_out_r**2 + self.E_out_theta**2
        if not math.isclose(actual_sq, expected_sq, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Field magnitude mismatch: actual^2={actual_sq}, expected^2={expected_sq}")
        return (True, "Field components consistent with vector form")

    # ------------------------------------------------------------------
    # Monotonicity tests: perturb input and check directional trend
    # ------------------------------------------------------------------
    def test_monotonicity_E_out_r_with_theta(self):
        """
        E_out_r = Ke*E0*cosθ decreases as θ increases (0 < θ < π).
        Perturb θ by +5% and check that E_out_r decreases.
        """
        if any(v is None for v in [self.Ke, self.E0, self.theta, self.E_out_r]):
            return (False, "SKIP: missing data for monotonicity test")
        if self.theta <= 0 or self.theta >= math.pi:
            return (False, "SKIP: theta not in (0, π) for monotonicity test")
        # Baseline value from formula
        baseline = self.Ke * self.E0 * math.cos(self.theta)
        # First verify that the stored value matches baseline (within tolerance)
        if not math.isclose(self.E_out_r, baseline, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Baseline check failed: stored E_out_r={self.E_out_r}, expected baseline={baseline}")
        # Perturb theta by +5%
        perturbed_theta = self.theta * 1.05
        perturbed_value = self.Ke * self.E0 * math.cos(perturbed_theta)
        if perturbed_value >= baseline:
            return (False, f"E_out_r did not decrease when theta increased: baseline {baseline}, perturbed {perturbed_value}")
        return (True, "E_out_r decreases as theta increases (monotonicity OK)")

    def test_monotonicity_sigma_b_with_theta(self):
        """
        σ_b = ε0(Ke-1)E0 cosθ decreases as θ increases (0 < θ < π).
        Use +5% perturbation.
        """
        if any(v is None for v in [self.epsilon0, self.Ke, self.E0, self.theta, self.sigma_b]):
            return (False, "SKIP: missing data for sigma_b monotonicity")
        if self.theta <= 0 or self.theta >= math.pi:
            return (False, "SKIP: theta not in (0, π) for monotonicity test")
        baseline = self.epsilon0 * (self.Ke - 1) * self.E0 * math.cos(self.theta)
        if not math.isclose(self.sigma_b, baseline, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Baseline check failed: stored sigma_b={self.sigma_b}, expected baseline={baseline}")
        perturbed_theta = self.theta * 1.05
        perturbed_value = self.epsilon0 * (self.Ke - 1) * self.E0 * math.cos(perturbed_theta)
        if perturbed_value >= baseline:
            return (False, f"sigma_b did not decrease when theta increased: baseline {baseline}, perturbed {perturbed_value}")
        return (True, "sigma_b decreases as theta increases (monotonicity OK)")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one allowed – no isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_finite(self):
        """Check that all results are finite numbers (not NaN, not None)."""
        results = [self.E_out_r, self.E_out_theta, self.sigma_b]
        for i, val in enumerate(results):
            if val is None:
                return (False, f"Result at index {i} is None")
            if not isinstance(val, (int, float)):
                return (False, f"Result at index {i} is not numeric")
            if math.isnan(val) or math.isinf(val):
                return (False, f"Result at index {i} is NaN or Inf: {val}")
        return (True, "All results are finite numbers")

    # ------------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Discover all bound methods starting with 'test_'
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