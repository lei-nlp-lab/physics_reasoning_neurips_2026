import math
import traceback

class PhysicsChecker:
    """
    Audits the Doppler width calculation for an optical line from an atom in a flame.
    Primary reference: Δν_D = ν₀ * sqrt(8 k T ln2 / (m c^2)).
    """

    def __init__(self, solution_instance):
        # Extract result – safe default None
        self.delta_nu_D = getattr(solution_instance, 'results', {}).get('delta_nu_D', None)

        # Extract givens (optional) – default None for all
        givens = getattr(solution_instance, 'givens', {})
        if not givens:
            givens = getattr(solution_instance, 'given', {})
        self.nu0 = givens.get('nu0', None)
        self.T   = givens.get('T',   None)
        self.m   = givens.get('m',   None)
        self.k   = givens.get('k',   None)   # Boltzmann constant (J/K)
        self.c   = givens.get('c',   None)   # speed of light (m/s)

        self.ln2 = math.log(2)

    # ------------------------------------------------------------------
    # Helper: compute expected Δν_D from givens (if available)
    # ------------------------------------------------------------------
    def _compute_expected(self, T=None, m=None, nu0=None, k=None, c=None):
        T   = T   if T   is not None else self.T
        m   = m   if m   is not None else self.m
        nu0 = nu0 if nu0 is not None else self.nu0
        k   = k   if k   is not None else self.k
        c   = c   if c   is not None else self.c
        if any(x is None for x in (T, m, nu0, k, c)):
            return None
        return nu0 * math.sqrt(8 * k * T * self.ln2 / (m * c * c))

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_delta_nu_D_exists(self):
        """Check that the computed Doppler width is present."""
        if self.delta_nu_D is None:
            return (False, "missing key 'delta_nu_D' in results")
        return (True, "delta_nu_D is present")

    def test_interface_delta_nu_D_is_finite(self):
        """Check that delta_nu_D is a finite number."""
        if self.delta_nu_D is None:
            return (False, "SKIP: delta_nu_D not available")
        if not math.isfinite(self.delta_nu_D):
            return (False, f"delta_nu_D = {self.delta_nu_D} is not finite")
        return (True, "delta_nu_D is finite")

    def test_interface_delta_nu_D_is_positive(self):
        """Check that Doppler width is positive (physical)."""
        if self.delta_nu_D is None:
            return (False, "SKIP: delta_nu_D not available")
        if self.delta_nu_D <= 0:
            return (False, f"delta_nu_D = {self.delta_nu_D} is not positive")
        return (True, "delta_nu_D > 0")

    # ------------------------------------------------------------------
    # Theorem-based invariant tests
    # ------------------------------------------------------------------
    def test_invariant_doppler_formula(self):
        """
        Check that delta_nu_D satisfies the known relation
        Δν_D = ν₀ * sqrt(8 k T ln2 / (m c^2)),
        given that all necessary input values are present.
        """
        if self.delta_nu_D is None:
            return (False, "SKIP: delta_nu_D not available")
        expected = self._compute_expected()
        if expected is None:
            return (False, "SKIP: missing one or more givens (nu0, T, m, k, c) to compute expected")
        if math.isclose(self.delta_nu_D, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Δν_D matches the Doppler formula (computed = {expected:.4e} Hz)")
        else:
            return (False, f"Δν_D = {self.delta_nu_D:.4e} Hz, expected ≈ {expected:.4e} Hz (relative diff > 1%)")

    def test_invariant_fwhm_variance(self):
        """
        Theorem: For a Gaussian line profile, FWHM = 2 * sqrt(ln2) * σ.
        The standard deviation in frequency is σ_ν = ν₀ * sqrt(kT/(mc²)).
        Therefore Δν_D should equal 2 * sqrt(ln2) * ν₀ * sqrt(kT/(mc²)).
        Check that the reported Δν_D satisfies this relation.
        """
        if self.delta_nu_D is None:
            return (False, "SKIP: delta_nu_D not available")
        if any(x is None for x in (self.nu0, self.T, self.m, self.k, self.c)):
            return (False, "SKIP: missing givens (nu0, T, m, k, c) for variance check")

        sigma_nu = self.nu0 * math.sqrt(self.k * self.T / (self.m * self.c * self.c))
        expected_fwhm = 2.0 * math.sqrt(self.ln2) * sigma_nu
        if math.isclose(self.delta_nu_D, expected_fwhm, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Δν_D ≈ 2√(ln2) σ_ν = {expected_fwhm:.4e} Hz (matches reported {self.delta_nu_D:.4e} Hz)")
        else:
            return (False, f"Δν_D = {self.delta_nu_D:.4e} Hz, but 2√(ln2) σ_ν = {expected_fwhm:.4e} Hz (diff > 1%)")

    # ------------------------------------------------------------------
    # Modeling / trace checks
    # ------------------------------------------------------------------
    def test_trace_order_of_magnitude_is_1e9(self):
        """
        Check that the Doppler width is on the order of 10^9 Hz.
        (Based on typical visible light and flame temperature.)
        """
        if self.delta_nu_D is None:
            return (False, "SKIP: delta_nu_D not available")
        try:
            log_val = math.log10(abs(self.delta_nu_D))
        except:
            return (False, "Could not compute log10 of delta_nu_D")
        if 8.5 <= log_val <= 10.5:
            return (True, f"Δν_D ≈ {self.delta_nu_D:.4e} Hz, order ≈ 10^{round(log_val)} Hz (expected ~10^9 Hz)")
        else:
            return (False, f"Δν_D order {round(log_val)} not near 9 (expected ~10^9 Hz)")

    def test_trace_formula_sign_consistency(self):
        """
        Ensure that if all givens are present, the computed formula yields positive Δν.
        """
        if any(x is None for x in (self.nu0, self.T, self.m, self.k, self.c)):
            return (False, "SKIP: missing givens for sign check")
        expected = self._compute_expected()
        if expected <= 0:
            return (False, f"Formula computed non-positive Δν = {expected}")
        return (True, "Formula yields positive Doppler width")

    # ------------------------------------------------------------------
    # Monotonicity checks
    # ------------------------------------------------------------------
    def test_monotonicity_delta_nu_D_increases_with_T(self):
        """
        Perturb temperature T upward by 10% and verify that Δν increases
        (compared to the reported answer).
        """
        if self.delta_nu_D is None:
            return (False, "SKIP: delta_nu_D not available")
        if any(x is None for x in (self.T, self.m, self.nu0, self.k, self.c)):
            return (False, "SKIP: missing givens for monotonicity with T")
        T_pert = self.T * 1.10
        baseline = self.delta_nu_D
        perturbed = self._compute_expected(T=T_pert)
        if perturbed is None:
            return (False, "SKIP: could not compute perturbed value")
        if perturbed > baseline:
            return (True, f"Increasing T from {self.T} K to {T_pert} K raises expected Δν from {baseline:.4e} to {perturbed:.4e} (consistency check passes)")
        else:
            return (False, f"Increasing T did not increase expected Δν (baseline={baseline:.4e}, perturbed={perturbed:.4e})")

    def test_monotonicity_delta_nu_D_decreases_with_m(self):
        """
        Perturb mass m upward by 10% and verify that Δν decreases
        (compared to the reported answer).
        """
        if self.delta_nu_D is None:
            return (False, "SKIP: delta_nu_D not available")
        if any(x is None for x in (self.m, self.T, self.nu0, self.k, self.c)):
            return (False, "SKIP: missing givens for monotonicity with m")
        m_pert = self.m * 1.10
        baseline = self.delta_nu_D
        perturbed = self._compute_expected(m=m_pert)
        if perturbed is None:
            return (False, "SKIP: could not compute perturbed value")
        if perturbed < baseline:
            return (True, f"Increasing m from {self.m} kg to {m_pert} kg lowers expected Δν from {baseline:.4e} to {perturbed:.4e} (consistency check passes)")
        else:
            return (False, f"Increasing m did not decrease expected Δν (baseline={baseline:.4e}, perturbed={perturbed:.4e})")

    # ------------------------------------------------------------------
    # Weak answer sanity (broad magnitude, no target values)
    # ------------------------------------------------------------------
    def test_answer_sanity_magnitude(self):
        """
        Weak check: delta_nu_D must be a positive finite number
        and lie roughly between 10^7 Hz and 10^11 Hz.
        """
        if self.delta_nu_D is None:
            return (False, "SKIP: delta_nu_D not available")
        if not math.isfinite(self.delta_nu_D):
            return (False, "delta_nu_D is not finite")
        if self.delta_nu_D <= 0:
            return (False, "delta_nu_D is not positive")
        if self.delta_nu_D < 1e7 or self.delta_nu_D > 1e11:
            return (False, f"Δν_D = {self.delta_nu_D:.4e} Hz outside expected range [1e7, 1e11] Hz")
        return (True, f"Δν_D = {self.delta_nu_D:.4e} Hz within plausible range")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        """
        Discover all bound methods starting with 'test_', execute them,
        collect results, and return (success, logs).
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
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)