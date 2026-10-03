import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens
        givens = getattr(solution_instance, 'givens', {})
        self.E = givens.get('E', None)          # kinetic energy in eV
        self.sigma_in = givens.get('sigma_in', None)  # in cm^2

        # Extract results
        results = getattr(solution_instance, 'results', {})
        self.sigma_el_min = results.get('sigma_el_min', None)
        self.sigma_el_max = results.get('sigma_el_max', None)

        # Physical constants (cgs units)
        self.m_n = 1.674927471e-24          # g
        self.m_C = 12 * 1.66053906660e-24   # g (≈1.9926e-23 g)
        self.hbar = 1.054571817e-27         # erg·s
        self.eV_to_erg = 1.602176634e-12    # 1 eV in erg

    # ------------------------------------------------------------------
    # Helper methods for internal computations
    # ------------------------------------------------------------------
    def _compute_k2(self, E_erg):
        """Return k^2 = 2 mu E / hbar^2 in cm^-2"""
        if E_erg is None or E_erg <= 0:
            return None
        mu = self.m_n * self.m_C / (self.m_n + self.m_C)
        k2 = 2.0 * mu * E_erg / (self.hbar ** 2)
        return k2

    def _compute_pi_lambda2(self, E_erg):
        """Return π/k^2 (≡ πλ^2) in cm^2"""
        k2 = self._compute_k2(E_erg)
        if k2 is None or k2 <= 0:
            return None
        return math.pi / k2

    def _compute_S_magnitude(self, pi_lambda2):
        """Return |S| from σ_in = πλ^2 (1 - |S|^2)"""
        if pi_lambda2 is None or pi_lambda2 <= 0:
            return None
        x = self.sigma_in / pi_lambda2
        if x > 1:
            return None  # unphysical
        return math.sqrt(1.0 - x)

    def _compute_unitarity_bound(self, E_erg):
        """Return 4π/k^2 in cm^2"""
        k2 = self._compute_k2(E_erg)
        if k2 is None or k2 <= 0:
            return None
        return 4.0 * math.pi / k2

    def _expected_sigma_el_max(self, E_erg):
        """Compute the theoretical maximum elastic cross section."""
        pi_lambda2 = self._compute_pi_lambda2(E_erg)
        if pi_lambda2 is None:
            return None
        abs_S = self._compute_S_magnitude(pi_lambda2)
        if abs_S is None:
            return None
        return pi_lambda2 * (1.0 + abs_S) ** 2

    def _expected_sigma_el_min(self, E_erg):
        """Compute the theoretical minimum elastic cross section."""
        pi_lambda2 = self._compute_pi_lambda2(E_erg)
        if pi_lambda2 is None:
            return None
        abs_S = self._compute_S_magnitude(pi_lambda2)
        if abs_S is None:
            return None
        return pi_lambda2 * (1.0 - abs_S) ** 2

    # ------------------------------------------------------------------
    # Interface tests: required keys exist and have valid types
    # ------------------------------------------------------------------
    def test_interface_results_present(self):
        """Check that required result keys exist."""
        if self.sigma_el_min is None:
            return (False, "sigma_el_min missing from results")
        if self.sigma_el_max is None:
            return (False, "sigma_el_max missing from results")
        return (True, "Required result keys present")

    def test_interface_givens_present(self):
        """Check that required givens are provided."""
        if self.E is None:
            return (False, "E (kinetic energy) missing from givens")
        if self.sigma_in is None:
            return (False, "sigma_in missing from givens")
        return (True, "Required givens present")

    def test_interface_values_positive(self):
        """Check that energy and inelastic cross section are positive."""
        if self.E is not None and self.E <= 0:
            return (False, f"E = {self.E} is not positive")
        if self.sigma_in is not None and self.sigma_in <= 0:
            return (False, f"sigma_in = {self.sigma_in} is not positive")
        return (True, "Energy and inelastic cross section are positive")

    # ------------------------------------------------------------------
    # Invariant tests: theorem-based physical sanity checks
    # ------------------------------------------------------------------
    def test_invariant_sigma_el_min_non_negative(self):
        """Lower bound: σ_el ≥ 0."""
        if self.sigma_el_min is None:
            return (False, "SKIP: sigma_el_min not available")
        if self.sigma_el_min < -1e-20:  # allow tiny numerical noise
            return (False, f"sigma_el_min = {self.sigma_el_min} < 0")
        return (True, "sigma_el_min is non-negative")

    def test_invariant_sigma_el_max_positive(self):
        """Upper bound must be positive."""
        if self.sigma_el_max is None:
            return (False, "SKIP: sigma_el_max not available")
        if self.sigma_el_max <= 0:
            return (False, f"sigma_el_max = {self.sigma_el_max} is not positive")
        return (True, "sigma_el_max is positive")

    def test_invariant_elastic_min_max_ordering(self):
        """σ_el_min ≤ σ_el_max must hold."""
        if self.sigma_el_min is None or self.sigma_el_max is None:
            return (False, "SKIP: one or both elastic cross sections missing")
        if self.sigma_el_min > self.sigma_el_max + 1e-20:
            return (False, f"sigma_el_min ({self.sigma_el_min}) > sigma_el_max ({self.sigma_el_max})")
        return (True, "sigma_el_min <= sigma_el_max holds")

    def test_invariant_unitarity_upper_bound(self):
        """Check that total cross section ≤ 4π/k² (s-wave unitarity bound)."""
        if self.E is None or self.sigma_in is None:
            return (False, "SKIP: required givens missing")
        if self.sigma_el_max is None:
            return (False, "SKIP: sigma_el_max not available")

        E_erg = self.E * self.eV_to_erg
        bound = self._compute_unitarity_bound(E_erg)
        if bound is None:
            return (False, "Could not compute unitarity bound")
        total_max = self.sigma_el_max + self.sigma_in
        # Allow 2% relative tolerance because of rounding in constants
        if total_max > bound * 1.02:
            return (False, f"sigma_el_max + sigma_in = {total_max} > 1.02 * 4π/k² ({bound})")
        return (True, f"Total cross section ≤ 4π/k² holds ({total_max:.3e} ≤ {bound:.3e})")

    def test_invariant_unitarity_bound_for_min(self):
        """Even with minimal elastic, total must not exceed unitarity bound (trivial)."""
        if self.E is None or self.sigma_in is None:
            return (False, "SKIP: required givens missing")
        if self.sigma_el_min is None:
            return (False, "SKIP: sigma_el_min not available")
        E_erg = self.E * self.eV_to_erg
        bound = self._compute_unitarity_bound(E_erg)
        if bound is None:
            return (False, "Could not compute unitarity bound")
        total_min = self.sigma_el_min + self.sigma_in
        if total_min > bound * 1.02:
            return (False, f"sigma_el_min + sigma_in = {total_min} > 1.02 * 4π/k² ({bound})")
        return (True, "Lower bound also satisfies unitarity")

    def test_invariant_s_wave_dominance(self):
        """Check that kR << 1 for R ~ 1e-12 cm, justifying s-wave."""
        if self.E is None:
            return (False, "SKIP: energy not available")
        E_erg = self.E * self.eV_to_erg
        k2 = self._compute_k2(E_erg)
        if k2 is None:
            return (False, "Could not compute k^2")
        k = math.sqrt(k2)  # in cm^-1
        R = 1.0e-12        # typical nuclear radius (cm)
        if k * R > 0.1:    # threshold for s-wave dominance
            return (False, f"kR = {k*R:.3e} > 0.1, s-wave may not dominate")
        return (True, f"kR = {k*R:.3e} << 1, s-wave scattering dominates")

    # ------------------------------------------------------------------
    # Modeling tests: based on explanation steps
    # ------------------------------------------------------------------
    def test_modeling_min_max_consistency(self):
        """
        Re-derive expected sigma_el_min and sigma_el_max from given
        sigma_in using the S-matrix approach (explanation steps).
        Check that the student's values match within 2%.
        """
        if self.E is None or self.sigma_in is None:
            return (False, "SKIP: required givens missing")
        if self.sigma_el_min is None or self.sigma_el_max is None:
            return (False, "SKIP: results missing")

        E_erg = self.E * self.eV_to_erg
        pi_lambda2 = self._compute_pi_lambda2(E_erg)
        if pi_lambda2 is None:
            return (False, "Could not compute πλ²")
        abs_S = self._compute_S_magnitude(pi_lambda2)
        if abs_S is None:
            return (False, "Could not compute |S| (sigma_in > πλ²?)")

        expected_min = pi_lambda2 * (1.0 - abs_S)**2
        expected_max = pi_lambda2 * (1.0 + abs_S)**2

        # Use relative tolerance 2%
        if not math.isclose(self.sigma_el_min, expected_min, rel_tol=0.02, abs_tol=1e-26):
            return (False, f"sigma_el_min {self.sigma_el_min:.3e} != expected {expected_min:.3e} (2%)")
        if not math.isclose(self.sigma_el_max, expected_max, rel_tol=0.02, abs_tol=1e-26):
            return (False, f"sigma_el_max {self.sigma_el_max:.3e} != expected {expected_max:.3e} (2%)")
        return (True, f"sigma_el_min and sigma_el_max consistent with S-matrix model")

    def test_modeling_total_cross_section_consistency(self):
        """
        Check that total cross sections derived from min and max
        satisfy the S-matrix sum rule: σ_total = 2πλ²(1 - Re[S])
        but we only verify it is less than 4π/k².
        This is a weaker check, but ensures no arithmetic mistake.
        """
        # Already covered by unitarity bound test; we skip here.
        return (True, "Covered by test_invariant_unitarity_upper_bound")

    # ------------------------------------------------------------------
    # Monotonicity tests: perturb input and compare output
    # ------------------------------------------------------------------
    def test_monotonicity_elastic_max_decreases_with_energy(self):
        """
        Perturb energy upwards by 1% and verify that the theoretical
        maximum elastic cross section decreases. This checks the
        physical monotonic behavior of the unitarity bound.
        """
        if self.E is None or self.sigma_in is None:
            return (False, "SKIP: required givens missing")

        E_erg = self.E * self.eV_to_erg
        # Baseline theoretical maximum at current energy
        max_baseline = self._expected_sigma_el_max(E_erg)
        if max_baseline is None:
            return (False, "Could not compute baseline sigma_el_max")

        # Perturbed energy: +1%
        E_perturbed = E_erg * 1.01
        max_perturbed = self._expected_sigma_el_max(E_perturbed)
        if max_perturbed is None:
            return (False, "Could not compute perturbed sigma_el_max")

        if max_perturbed >= max_baseline * 1.001:  # allow tiny numerical increase
            return (False, f"sigma_el_max increased with energy: baseline {max_baseline:.3e}, perturbed {max_perturbed:.3e}")
        return (True, f"sigma_el_max decreases with energy (baseline {max_baseline:.3e} -> perturbed {max_perturbed:.3e})")

    def test_monotonicity_unitarity_bound_Decreases_with_energy(self):
        """
        Perturb energy upwards by 1% and check that the unitarity bound
        4π/k² decreases. This is a simple algebraic monotonicity.
        """
        if self.E is None:
            return (False, "SKIP: energy not available")

        E_erg = self.E * self.eV_to_erg
        bound_baseline = self._compute_unitarity_bound(E_erg)
        if bound_baseline is None:
            return (False, "Could not compute baseline unitarity bound")

        E_perturbed = E_erg * 1.01
        bound_perturbed = self._compute_unitarity_bound(E_perturbed)
        if bound_perturbed is None:
            return (False, "Could not compute perturbed unitarity bound")

        if bound_perturbed >= bound_baseline * 1.001:
            return (False, f"4π/k² increased with energy: baseline {bound_baseline:.3e}, perturbed {bound_perturbed:.3e}")
        return (True, f"4π/k² decreases with energy (baseline {bound_baseline:.3e} -> perturbed {bound_perturbed:.3e})")

    # ------------------------------------------------------------------
    # Answer sanity test (weak only, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_reasonable_magnitude(self):
        """
        Ensure elastic cross sections have a physically plausible
        order of magnitude (10^-23 to 10^-22 cm^2 typical for nuclear).
        Use a broad range: 0 < sigma_el < 1e-15 cm^2.
        """
        if self.sigma_el_min is None or self.sigma_el_max is None:
            return (False, "SKIP: results missing")
        if self.sigma_el_min <= 0 or self.sigma_el_max <= 0:
            return (False, "Cross sections must be positive")
        # Upper bound: 4π/k² for 1 keV neutron ~ 2e-21 cm^2, so 1e-20 is safe
        max_allowed = 1e-20  # cm^2
        if self.sigma_el_min > max_allowed:
            return (False, f"sigma_el_min {self.sigma_el_min} too large (>{max_allowed})")
        if self.sigma_el_max > max_allowed:
            return (False, f"sigma_el_max {self.sigma_el_max} too large (>{max_allowed})")
        return (True, "Cross sections have reasonable magnitude")

    # ------------------------------------------------------------------
    # Runner that discovers all test_ methods
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Collect all bound methods starting with 'test_'
        test_methods = [name for name in dir(self)
                        if name.startswith('test_') and callable(getattr(self, name))]
        test_methods.sort()
        for name in test_methods:
            method = getattr(self, name)
            try:
                passed, msg = method()
            except Exception as e:
                passed = False
                msg = f"CRASHED: {traceback.format_exc()}"
                logs.append(f"FAIL: {name}: {msg}")
                all_pass = False
                continue
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)