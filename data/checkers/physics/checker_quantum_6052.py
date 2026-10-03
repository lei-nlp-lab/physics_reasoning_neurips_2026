import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Extract required results from the solution instance.
        Use None defaults for missing keys.
        Do NOT assign numeric fallback defaults for missing physics quantities.
        """
        self.solution = solution_instance
        results = getattr(solution_instance, 'results', {})
        # Emitted key mapping (from pseudocode contract)
        self.k = results.get('k', None)
        self.psi_final = results.get('psi_final', None)
        self.eta = results.get('eta', None)
        self.P1_over_P0_num = results.get('P1_over_P0_num', None)

        # Hard‑coded problem constants (derived from problem statement, not from answer)
        self.E_star = 840.0  # keV
        self.hbar_omega = 1.0  # keV
        # m c^2 for Al-27: 27 * 931.5 MeV = 2.51505e7 keV
        self.m_c2_keV = 27 * 931.5 * 1000.0  # keV
        # ħc ≈ 197.326 MeV·fm = 197326 keV·fm
        self.hbar_c = 197326.0  # keV·fm

        # Reference values that will be used in invariant tests
        self.expected_eta = self._compute_eta()
        self.expected_k = self._compute_k()

    # ------------------------------------------------------------------
    # Helper computations (in‑checker derivations, not from answer keys)
    # ------------------------------------------------------------------
    def _compute_eta(self):
        """ eta = E*^2 / (2 * m c^2 * ħω) """
        if self.m_c2_keV is None or self.E_star is None or self.hbar_omega is None:
            return None
        return self.E_star ** 2 / (2.0 * self.m_c2_keV * self.hbar_omega)

    def _compute_k(self):
        """ k = E* / (ħ c)  (fm^-1) """
        if self.E_star is None or self.hbar_c is None:
            return None
        return self.E_star / self.hbar_c

    # ------------------------------------------------------------------
    # Test methods
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all required result keys are present and have valid types."""
        if self.k is None:
            return (False, "Missing key: 'k'")
        if self.psi_final is None:
            return (False, "Missing key: 'psi_final'")
        if self.eta is None:
            return (False, "Missing key: 'eta'")
        if self.P1_over_P0_num is None:
            return (False, "Missing key: 'P1_over_P0_num'")
        # Type checks
        if not isinstance(self.psi_final, (str,)):
            return (False, "'psi_final' should be a string (expression)")
        if not isinstance(self.k, (int, float)):
            return (False, "'k' should be numeric")
        if not isinstance(self.eta, (int, float)):
            return (False, "'eta' should be numeric")
        if not isinstance(self.P1_over_P0_num, (int, float)):
            return (False, "'P1_over_P0_num' should be numeric")
        return (True, "All required keys present and valid")

    def test_invariant_eta_positive(self):
        """η must be positive."""
        if self.eta is None:
            return (False, "SKIP: 'eta' not available")
        if self.eta > 0:
            return (True, f"eta = {self.eta} > 0")
        else:
            return (False, f"eta = {self.eta} should be positive")

    def test_invariant_eta_calculation(self):
        """Check that η matches the formula η = E*^2/(2 m c^2 ħω)."""
        if self.eta is None:
            return (False, "SKIP: 'eta' not available")
        if self.expected_eta is None:
            return (False, "Cannot compute expected η; missing problem constants")
        # Use fuzzy matching (allowed for invariant test)
        if math.isclose(self.eta, self.expected_eta, rel_tol=1e-2, abs_tol=1e-4):
            return (True, f"η = {self.eta} matches expected {self.expected_eta:.6f}")
        else:
            return (False, f"η = {self.eta} deviates from expected {self.expected_eta:.6f}")

    def test_invariant_P1_over_P0_positive_and_finite(self):
        """P1/P0 must be positive and less than 1 (since P0 > P1 for small η)."""
        if self.P1_over_P0_num is None:
            return (False, "SKIP: 'P1_over_P0_num' not available")
        if self.P1_over_P0_num > 0 and self.P1_over_P0_num < 1:
            return (True, f"P1/P0 = {self.P1_over_P0_num} is in (0,1)")
        else:
            return (False, f"P1/P0 = {self.P1_over_P0_num} should be between 0 and 1")

    def test_invariant_P1_over_P0_matches_eta(self):
        """Check that P1/P0 ≈ η (they should be equal by theory)."""
        if self.eta is None or self.P1_over_P0_num is None:
            return (False, "SKIP: either 'eta' or 'P1_over_P0_num' missing")
        if math.isclose(self.eta, self.P1_over_P0_num, rel_tol=1e-2, abs_tol=1e-4):
            return (True, f"P1/P0 = {self.P1_over_P0_num} matches η = {self.eta}")
        else:
            return (False, f"P1/P0 = {self.P1_over_P0_num} differs from η = {self.eta}")

    def test_invariant_k_positive(self):
        """Recoil wave number k must be positive."""
        if self.k is None:
            return (False, "SKIP: 'k' not available")
        if self.k > 0:
            return (True, f"k = {self.k} > 0")
        else:
            return (False, f"k = {self.k} should be positive")

    def test_modeling_final_wavefunction_phase(self):
        """Check that psi_final string contains the expected phase factor."""
        if self.psi_final is None:
            return (False, "SKIP: 'psi_final' not available")
        s = str(self.psi_final)
        # Look for evidence of the recoil phase factor exp(i k x)
        has_exp = 'exp' in s.lower() or 'e^{' in s
        has_i = 'i' in s
        has_x = 'x' in s
        if has_exp and has_i and has_x:
            return (True, "Final wavefunction contains phase factor (exp, i, x)")
        else:
            return (False, f"psi_final string lacks phase factor indicators: {s[:80]}...")

    def test_modeling_k_energy_consistency(self):
        """Check that k is consistent with the recoil momentum p = E*/c.
        Use a broad tolerance because units may differ between implementations.
        Compare against expected k = E*/(ħ c) in fm^-1.
        """
        if self.k is None:
            return (False, "SKIP: 'k' not available")
        if self.expected_k is None:
            return (False, "Cannot compute expected k; missing problem constants")
        if math.isclose(self.k, self.expected_k, rel_tol=0.5, abs_tol=0.01):
            return (True, f"k = {self.k} is within order of magnitude of expected {self.expected_k:.6f}")
        else:
            return (False, f"k = {self.k} mismatches expected {self.expected_k:.6f} (consider unit conversion)")

    def test_monotonicity_eta_vs_Estar(self):
        """Verify that η scales quadratically with E*.
        Baseline: E_star = 840 keV.
        Perturb: E_star * 1.1 → η should increase by factor ~1.21.
        """
        if self.expected_eta is None:
            return (False, "Cannot compute baseline η")
        baseline_eta = self.expected_eta
        E_pert = self.E_star * 1.1
        perturbed_eta = E_pert ** 2 / (2.0 * self.m_c2_keV * self.hbar_omega)
        ratio = perturbed_eta / baseline_eta
        if math.isclose(ratio, 1.21, rel_tol=0.1):
            return (True, f"η scaling: ratio = {ratio:.3f}, expected ~1.21")
        else:
            return (False, f"η scaling: ratio = {ratio:.3f}, expected ~1.21")

    def test_answer_sanity_P1_over_P0_num_range(self):
        """Weak sanity: P1/P0 should be around 0.014 (±50%).
        This is the only answer-focused test; it does NOT rederive the value.
        """
        if self.P1_over_P0_num is None:
            return (False, "SKIP: 'P1_over_P0_num' not available")
        # Accept broad range (0.007, 0.021)
        if 0.007 < self.P1_over_P0_num < 0.021:
            return (True, f"P1/P0 = {self.P1_over_P0_num} is in plausible range (0.007-0.021)")
        else:
            return (False, f"P1/P0 = {self.P1_over_P0_num} outside plausible range")

    # ------------------------------------------------------------------
    # Test runner
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
                passed = False
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)