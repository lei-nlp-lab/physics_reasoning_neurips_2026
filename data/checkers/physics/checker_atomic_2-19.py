import math

class PhysicsChecker:
    """
    Checker for neutron-nucleus scattering problem at 5 MeV with given phase shifts.
    Extracts givens and results, then performs unit-test‑like checks.
    """

    # Physical constants (natural units)
    mn_MeV = 939.6      # neutron mass / c^2
    hbarc_MeV_fm = 197.3

    def __init__(self, solution_instance):
        # Givens (may be missing)
        self.E = getattr(solution_instance, 'E', None)
        self.delta0_deg = getattr(solution_instance, 'delta0_deg', None)
        self.delta1_deg = getattr(solution_instance, 'delta1_deg', None)

        # Results dictionary
        self.results = getattr(solution_instance, 'results', {}) or {}

        # Expected result keys (all default to None)
        self.dSigma_dOmega_30 = self.results.get('dSigma_dOmega_30', None)
        self.dSigma_dOmega_45 = self.results.get('dSigma_dOmega_45', None)
        self.dSigma_dOmega_90 = self.results.get('dSigma_dOmega_90', None)
        self.sigma_total = self.results.get('sigma_total', None)
        self.R = self.results.get('R', None)

    # ------------------------------------------------------------------
    # Helper methods (derived from givens – no numeric fallbacks)
    # ------------------------------------------------------------------
    def _compute_k(self):
        """Return wave number k (fm⁻¹) or None if missing data."""
        if self.E is None:
            return None
        return math.sqrt(2.0 * self.mn_MeV * self.E) / self.hbarc_MeV_fm

    def _compute_sin_delta(self, which):
        """Return sin(δₗ) or None if angle missing."""
        deg = self.delta0_deg if which == 0 else self.delta1_deg
        if deg is None:
            return None
        return math.sin(math.radians(deg))

    def _compute_cos_delta(self, which):
        """Return cos(δₗ) or None if angle missing."""
        deg = self.delta0_deg if which == 0 else self.delta1_deg
        if deg is None:
            return None
        return math.cos(math.radians(deg))

    def _compute_dsigma_angle(self, theta_deg):
        """
        Compute dσ/dΩ (cm²) at a given angle using the known formula.
        Returns None if any required given is missing.
        """
        k = self._compute_k()
        sin0 = self._compute_sin_delta(0)
        sin1 = self._compute_sin_delta(1)
        cos0 = self._compute_cos_delta(0)
        cos1 = self._compute_cos_delta(1)
        if None in (k, sin0, sin1, cos0, cos1):
            return None

        delta0 = math.radians(self.delta0_deg)
        delta1 = math.radians(self.delta1_deg)

        sin2_0 = sin0 * sin0
        sin2_1 = sin1 * sin1
        cos_diff = math.cos(delta1 - delta0)

        theta = math.radians(theta_deg)
        cos_theta = math.cos(theta)

        # dσ/dΩ in fm²
        dsigma_fm2 = (sin2_0 + 9.0 * sin2_1 * cos_theta * cos_theta
                     + 6.0 * sin0 * sin1 * cos_diff * cos_theta) / (k * k)

        # Convert fm² → cm² (1 fm = 1e-13 cm)
        return dsigma_fm2 * 1e-26

    def _compute_sigma_total(self):
        """Return total cross section (cm²) or None if missing."""
        k = self._compute_k()
        sin0 = self._compute_sin_delta(0)
        sin1 = self._compute_sin_delta(1)
        if None in (k, sin0, sin1):
            return None
        sin2_0 = sin0 * sin0
        sin2_1 = sin1 * sin1
        sigma_fm2 = (4.0 * math.pi / (k * k)) * (sin2_0 + 3.0 * sin2_1)
        return sigma_fm2 * 1e-26

    def _compute_R(self):
        """Return range estimate R (fm) = 1/k, or None if k missing."""
        k = self._compute_k()
        if k is None:
            return None
        return 1.0 / k

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """All five result keys must exist and be non‑None."""
        keys = ['dSigma_dOmega_30', 'dSigma_dOmega_45', 'dSigma_dOmega_90',
                'sigma_total', 'R']
        missing = [k for k in keys if self.results.get(k) is None]
        if missing:
            return False, f"Missing result keys: {missing}"
        return True, "All required keys present"

    def test_interface_values_positive(self):
        """Cross sections and range must be positive (if present)."""
        values = {
            'dSigma_dOmega_30': self.dSigma_dOmega_30,
            'dSigma_dOmega_45': self.dSigma_dOmega_45,
            'dSigma_dOmega_90': self.dSigma_dOmega_90,
            'sigma_total': self.sigma_total,
            'R': self.R
        }
        negatives = [k for k, v in values.items() if v is not None and v <= 0]
        if negatives:
            return False, f"Non‑positive values found: {negatives}"
        return True, "All present values are positive"

    # ------------------------------------------------------------------
    # Invariant tests (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_dsigma_30_consistency(self):
        """dσ/dΩ(30°) matches theoretical formula."""
        if self.dSigma_dOmega_30 is None:
            return True, "SKIP: dSigma_dOmega_30 not available"
        expected = self._compute_dsigma_angle(30.0)
        if expected is None:
            return True, "SKIP: insufficient givens to compute expected"
        if not math.isclose(expected, self.dSigma_dOmega_30, rel_tol=0.02):
            return False, f"dσ/dΩ(30°) mismatch: expected {expected:.3e}, got {self.dSigma_dOmega_30:.3e}"
        return True, "dσ/dΩ(30°) consistent with formula"

    def test_invariant_dsigma_45_consistency(self):
        """dσ/dΩ(45°) matches theoretical formula."""
        if self.dSigma_dOmega_45 is None:
            return True, "SKIP: dSigma_dOmega_45 not available"
        expected = self._compute_dsigma_angle(45.0)
        if expected is None:
            return True, "SKIP: insufficient givens"
        if not math.isclose(expected, self.dSigma_dOmega_45, rel_tol=0.02):
            return False, f"dσ/dΩ(45°) mismatch: expected {expected:.3e}, got {self.dSigma_dOmega_45:.3e}"
        return True, "dσ/dΩ(45°) consistent with formula"

    def test_invariant_dsigma_90_consistency(self):
        """dσ/dΩ(90°) matches theoretical formula."""
        if self.dSigma_dOmega_90 is None:
            return True, "SKIP: dSigma_dOmega_90 not available"
        expected = self._compute_dsigma_angle(90.0)
        if expected is None:
            return True, "SKIP: insufficient givens"
        if not math.isclose(expected, self.dSigma_dOmega_90, rel_tol=0.02):
            return False, f"dσ/dΩ(90°) mismatch: expected {expected:.3e}, got {self.dSigma_dOmega_90:.3e}"
        return True, "dσ/dΩ(90°) consistent with formula"

    def test_invariant_sigma_total_consistency(self):
        """Total cross section matches theoretical formula."""
        if self.sigma_total is None:
            return True, "SKIP: sigma_total not available"
        expected = self._compute_sigma_total()
        if expected is None:
            return True, "SKIP: insufficient givens"
        if not math.isclose(expected, self.sigma_total, rel_tol=0.02):
            return False, f"σ mismatch: expected {expected:.3e}, got {self.sigma_total:.3e}"
        return True, "σ consistent with formula"

    def test_invariant_R_consistency(self):
        """Range estimate R ≈ 1/k."""
        if self.R is None:
            return True, "SKIP: R not available"
        expected = self._compute_R()
        if expected is None:
            return True, "SKIP: insufficient givens"
        if not math.isclose(expected, self.R, rel_tol=0.05):
            return False, f"R mismatch: expected {expected:.2f} fm, got {self.R:.2f} fm"
        return True, "R ≈ 1/k consistent"

    # ------------------------------------------------------------------
    # Modeling / trace quality tests
    # ------------------------------------------------------------------
    def test_trace_dsigma_ordered(self):
        """Differential cross section must decrease with angle: 30° > 45° > 90°."""
        vals = [self.dSigma_dOmega_30, self.dSigma_dOmega_45, self.dSigma_dOmega_90]
        if any(v is None for v in vals):
            return True, "SKIP: one or more dσ/dΩ values missing"
        if not (vals[0] > vals[1] > vals[2]):
            return False, f"Ordering broken: {vals[0]:.3e} > {vals[1]:.3e} > {vals[2]:.3e}?"
        return True, "dσ/dΩ decreases with angle: 30° > 45° > 90°"

    def test_trace_dsigma_90_s_wave_dominance(self):
        """
        At 90°, cosθ=0 → p‑wave term vanishes.  Therefore dσ/dΩ(90°) should equal
        sin²δ₀/k².  This tests the angular structure of the amplitude.
        """
        if self.dSigma_dOmega_90 is None:
            return True, "SKIP: dSigma_dOmega_90 not available"
        k = self._compute_k()
        sin0 = self._compute_sin_delta(0)
        if None in (k, sin0):
            return True, "SKIP: insufficient givens"
        s_wave_contrib = (sin0 * sin0) / (k * k) * 1e-26   # convert to cm²
        if not math.isclose(s_wave_contrib, self.dSigma_dOmega_90, rel_tol=0.02):
            return False, f"90° should be sin²δ₀/k²: expected {s_wave_contrib:.3e}, got {self.dSigma_dOmega_90:.3e}"
        return True, "dσ/dΩ(90°) equals pure s‑wave contribution"

    # ------------------------------------------------------------------
    # Monotonicity tests (must perturb an input and compare)
    # ------------------------------------------------------------------
    def test_monotonicity_angle_decrease(self):
        """Perturb angle from 30° to 31°; dσ/dΩ should decrease."""
        if self.E is None or self.delta0_deg is None or self.delta1_deg is None:
            return True, "SKIP: insufficient givens for trend test"
        d30 = self._compute_dsigma_angle(30.0)
        d31 = self._compute_dsigma_angle(31.0)
        if d30 is None or d31 is None:
            return True, "SKIP: could not compute cross sections"
        if not (d30 > d31):
            return False, f"dσ/dΩ does not decrease: {d30:.3e} (30°) vs {d31:.3e} (31°)"
        return True, "dσ/dΩ decreases with increasing angle (30° → 31°)"

    def test_monotonicity_delta1_perturbation(self):
        """
        Perturb δ₁ by +1%, recompute dσ/dΩ at 30°, and check that it increases
        (since larger δ₁ increases the p‑wave contribution, which adds to the
        forward cross section for δ₁ small).
        """
        if None in (self.E, self.delta0_deg, self.delta1_deg):
            return True, "SKIP: insufficient givens"
        # Baseline
        baseline = self._compute_dsigma_angle(30.0)
        if baseline is None:
            return True, "SKIP: could not compute baseline"

        # Perturb δ₁ by +1% (in radians)
        original_rad = math.radians(self.delta1_deg)
        perturbed_rad = original_rad * 1.01
        # Temporarily override self.delta1_deg? We'll compute directly using the helper with a parameter.
        # We need a modified _compute_dsigma_angle that allows overriding delta1.
        # Simplest: recompute using the formula but with perturbed sin/cos.
        # We'll reuse the internal logic.
        k = self._compute_k()
        sin0 = self._compute_sin_delta(0)
        cos0 = self._compute_cos_delta(0)
        if None in (k, sin0, cos0):
            return True, "SKIP: missing data"

        sin1_pert = math.sin(perturbed_rad)
        cos1_pert = math.cos(perturbed_rad)
        sin2_0 = sin0 * sin0
        sin2_1 = sin1_pert * sin1_pert
        cos_diff_pert = math.cos(perturbed_rad - math.radians(self.delta0_deg))

        theta = math.radians(30.0)
        cos_theta = math.cos(theta)

        dsigma_pert_fm2 = (sin2_0 + 9.0 * sin2_1 * cos_theta * cos_theta
                          + 6.0 * sin0 * sin1_pert * cos_diff_pert * cos_theta) / (k * k)
        dsigma_pert = dsigma_pert_fm2 * 1e-26

        if not (dsigma_pert > baseline):
            return False, f"dσ/dΩ did not increase with +1% δ₁: baseline {baseline:.3e}, perturbed {dsigma_pert:.3e}"
        return True, "dσ/dΩ(30°) increases with increasing δ₁ (+1%)"

    # ------------------------------------------------------------------
    # Weak answer sanity (only one such method)
    # ------------------------------------------------------------------
    def test_answer_sanity_physical_range(self):
        """
        Broad orders of magnitude: cross sections in 1e‑26 – 1e‑25 cm²,
        R around 2 fm.  No precise matching.
        """
        issues = []
        for name, val in [('30°', self.dSigma_dOmega_30),
                          ('45°', self.dSigma_dOmega_45),
                          ('90°', self.dSigma_dOmega_90)]:
            if val is not None and (val <= 1e-30 or val >= 1e-20):
                issues.append(f"dσ/dΩ({name}) out of plausible range: {val:.1e}")
        if self.sigma_total is not None and (self.sigma_total <= 1e-30 or self.sigma_total >= 1e-20):
            issues.append(f"σ out of plausible range: {self.sigma_total:.1e}")
        if self.R is not None and (self.R <= 0.1 or self.R >= 20):
            issues.append(f"R out of plausible range: {self.R:.1f} fm")
        if issues:
            return False, "; ".join(issues)
        return True, "All values within physically reasonable orders of magnitude"

    # ------------------------------------------------------------------
    # Test runner
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs