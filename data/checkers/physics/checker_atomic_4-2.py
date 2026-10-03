import math
import traceback

class PhysicsChecker:
    """
    Audits a solution to the relativistic meson problem.
    Checks interface completeness, physical invariants, modeling consistency,
    monotonicity under input perturbations, and a weak final-answer sanity.
    """
    def __init__(self, solution_instance):
        # Extract results safely using None defaults
        self.results = getattr(solution_instance, 'results', None)
        if not isinstance(self.results, dict):
            self.results = {}

        self.m0   = self.results.get('m0_GeV', None)
        self.KE1  = self.results.get('KE1_GeV', None)
        self.KE2  = self.results.get('KE2_GeV', None)
        self.tau  = self.results.get('tau', None)
        self.L50  = self.results.get('L50', None)

        # Extract givens safely (problem statement constants)
        self.givens = getattr(solution_instance, 'givens', None)
        if not isinstance(self.givens, dict):
            self.givens = {}

        # Store givens with None defaults so we can check for existence later
        self._rhoB1   = self.givens.get('rhoB1', None)
        self._rhoB2   = self.givens.get('rhoB2', None)
        self._v2      = self.givens.get('v2', None)
        self._x_decay = self.givens.get('x_decay', None)   # meters

        # Physical constants
        self.c = 3.0e8          # m/s
        self.conv = 300.0       # MeV/c per T·m  → convert to GeV later
        self.ln2 = 0.6931471805599453

        # Derived from givens when available (used by monotonicity tests)
        self._beta2 = None
        if self._v2 is not None:
            self._beta2 = self._v2 / self.c

    # ------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------
    def test_interface_required_keys(self):
        """All five required result keys must exist and not be None."""
        if self.m0 is None:
            return (False, "m0_GeV is missing or None")
        if self.KE1 is None:
            return (False, "KE1_GeV is missing or None")
        if self.KE2 is None:
            return (False, "KE2_GeV is missing or None")
        if self.tau is None:
            return (False, "tau is missing or None")
        if self.L50 is None:
            return (False, "L50 is missing or None")
        return (True, "All required keys present")

    def test_interface_non_negative(self):
        """All result values should be non-negative."""
        for name, val in [('m0_GeV', self.m0), ('KE1_GeV', self.KE1),
                          ('KE2_GeV', self.KE2), ('tau', self.tau),
                          ('L50', self.L50)]:
            if val is None:
                continue
            if isinstance(val, (int, float)):
                if val < -1e-20:
                    return (False, f"{name} is negative: {val}")
        return (True, "All present values are non-negative")

    # ------------------------------------------------------------
    # Invariant / theorem‑based tests (use math.isclose)
    # ------------------------------------------------------------
    def test_invariant_rest_mass_from_slow_data(self):
        """
        Use the given (ρB)₂ and v₂ to re‑derive the rest mass and compare.
        The checker must compute β₂, pc₂ = 300·(ρB)₂ (MeV), E₂ = pc₂/β₂,
        then m₀c² = sqrt(E₂² - pc₂²).  Compare to self.m0 (GeV).
        """
        if self._rhoB2 is None or self._v2 is None:
            return (False, "SKIP: rhoB2 or v2 not in givens")
        if self.m0 is None:
            return (False, "SKIP: no m0 to compare")

        beta2 = self._v2 / self.c
        if not (0 < beta2 < 1):
            return (False, f"beta2 out of range: {beta2}")

        pc2_MeV = self.conv * self._rhoB2          # in MeV
        E2_MeV = pc2_MeV / beta2                   # E = pc/β
        m0_reduced_MeV = math.sqrt(E2_MeV**2 - pc2_MeV**2)
        m0_expected_GeV = m0_reduced_MeV / 1000.0

        if not math.isclose(self.m0, m0_expected_GeV, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"m0 from slow data: expected {m0_expected_GeV:.3f} GeV, got {self.m0:.3f} GeV")
        return (True, "Rest mass consistent with slow‑meson data")

    def test_invariant_energy_momentum_initial(self):
        """
        Check that KE1 + m0 = sqrt( (300·ρB₁)² + m0² )  (all in GeV)
        This verifies E² = (pc)² + (m₀c²)².
        """
        if self._rhoB1 is None:
            return (False, "SKIP: rhoB1 not in givens")
        if self.m0 is None or self.KE1 is None:
            return (False, "SKIP: m0 or KE1 missing")

        pc1_GeV = (self.conv * self._rhoB1) / 1000.0   # convert MeV to GeV
        total_energy = math.sqrt(pc1_GeV**2 + self.m0**2)
        lhs = self.KE1 + self.m0   # should equal total energy

        if not math.isclose(lhs, total_energy, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Initial energy mismatch: KE1+m0={lhs:.3f}, sqrt(pc²+m0²)={total_energy:.3f}")
        return (True, "Initial energy‑momentum relation holds")

    def test_invariant_decay_half_life_consistency(self):
        """
        From the slow‑meson decay (50% probability after 4 m),
        compute the proper half‑life τ₀ = τ·ln2 and check that it matches
        the expected value. Actually we check tau directly:
        τ = x / (βγ c ln2).  Compare to self.tau.
        """
        if self._v2 is None or self._x_decay is None:
            return (False, "SKIP: missing givens for decay")
        if self.tau is None:
            return (False, "SKIP: tau missing")

        beta2 = self._v2 / self.c
        if not (0 < beta2 < 1):
            return (False, f"beta2 out of range: {beta2}")
        gamma2 = 1.0 / math.sqrt(1.0 - beta2**2)
        beta_gamma2 = beta2 * gamma2

        # τ = x / (βγ c ln2)
        tau_expected = self._x_decay / (beta_gamma2 * self.c * self.ln2)

        if not math.isclose(self.tau, tau_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"τ from decay: expected {tau_expected:.3e} s, got {self.tau:.3e} s")
        return (True, "Decay half‑life consistent with slow‑meson data")

    # ------------------------------------------------------------
    # Modeling / trace checks (loose, but verify logic)
    # ------------------------------------------------------------
    def test_modeling_KE2_via_mass_subtraction(self):
        """
        Check that KE2 = sqrt(pc₂² + m0²) - m0.
        """
        if self._rhoB2 is None:
            return (False, "SKIP: rhoB2 not in givens")
        if self.m0 is None or self.KE2 is None:
            return (False, "SKIP: m0 or KE2 missing")

        pc2_GeV = (self.conv * self._rhoB2) / 1000.0
        total_energy = math.sqrt(pc2_GeV**2 + self.m0**2)
        expected_KE2 = total_energy - self.m0

        if not math.isclose(self.KE2, expected_KE2, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"KE2 mismatch: expected {expected_KE2:.3f} GeV, got {self.KE2:.3f} GeV")
        return (True, "KE2 consistent with energy‑momentum relation")

    def test_modeling_L50_via_full_energy_mesons(self):
        """
        Check that L50 = β₁γ₁·c·τ·ln2 , where β₁γ₁ = p₁c/(m₀c²).
        Use givens ρB₁ and derived m₀.
        """
        if self._rhoB1 is None:
            return (False, "SKIP: rhoB1 not in givens")
        if self.m0 is None or self.tau is None or self.L50 is None:
            return (False, "SKIP: m0, tau or L50 missing")

        pc1_GeV = (self.conv * self._rhoB1) / 1000.0
        total_energy = math.sqrt(pc1_GeV**2 + self.m0**2)
        beta_gamma1 = pc1_GeV / self.m0   # since pc/E * E/m0 = pc/m0

        L50_expected = beta_gamma1 * self.c * self.tau * self.ln2

        if not math.isclose(self.L50, L50_expected, rel_tol=1e-2, abs_tol=2.0):
            return (False, f"L50 mismatch: expected {L50_expected:.1f} m, got {self.L50:.1f} m")
        return (True, "L50 consistent with fast‑meson kinematics")

    # ------------------------------------------------------------
    # Monotonicity tests (each perturbs an input and checks direction)
    # ------------------------------------------------------------
    def test_monotonicity_KE1_increases_with_rhoB1(self):
        """
        Perturb ρB₁ upward by 5%; the derived KE1 should increase.
        We re‑derive KE1 from ρB₁ and m₀ (the checker's m₀) for baseline and perturbed.
        """
        if self._rhoB1 is None:
            return (False, "SKIP: rhoB1 not in givens")
        if self.m0 is None:
            return (False, "SKIP: m0 missing")

        # Baseline KE1 from the solution's m0 and givens (independent of solution's KE1)
        pc1_GeV = (self.conv * self._rhoB1) / 1000.0
        E1     = math.sqrt(pc1_GeV**2 + self.m0**2)
        KE1_base = E1 - self.m0

        # Perturbed: increase rhoB1 by 5%
        delta = 0.05
        rhoB1_pert = self._rhoB1 * (1 + delta)
        pc1_GeV_pert = (self.conv * rhoB1_pert) / 1000.0
        E1_pert      = math.sqrt(pc1_GeV_pert**2 + self.m0**2)
        KE1_pert     = E1_pert - self.m0

        # Check monotonic direction (should increase)
        if not (KE1_pert > KE1_base):
            return (False, f"KE1 did not increase when rhoB1 increased by {delta*100}%: base={KE1_base:.4f}, perturbed={KE1_pert:.4f}")
        return (True, f"KE1 monotonic: perturbing rhoB1 +{delta*100}% gives KE1 increase {KE1_base:.4f} -> {KE1_pert:.4f}")

    def test_monotonicity_L50_increases_with_rhoB1(self):
        """
        Perturb ρB₁ upward by 5%; L50 (fast meson decay length) should increase.
        Derive L50 from ρB₁, m₀, and tau (the solution's tau).
        """
        if self._rhoB1 is None:
            return (False, "SKIP: rhoB1 not in givens")
        if self.m0 is None or self.tau is None:
            return (False, "SKIP: m0 or tau missing")

        # Baseline L50 using the provided tau (the solution's tau is used as is)
        pc1_GeV = (self.conv * self._rhoB1) / 1000.0
        beta_gamma1 = pc1_GeV / self.m0
        L50_base = beta_gamma1 * self.c * self.tau * self.ln2

        # Perturb rhoB1 by 5%
        delta = 0.05
        rhoB1_pert = self._rhoB1 * (1 + delta)
        pc1_GeV_pert = (self.conv * rhoB1_pert) / 1000.0
        beta_gamma1_pert = pc1_GeV_pert / self.m0
        L50_pert = beta_gamma1_pert * self.c * self.tau * self.ln2

        if not (L50_pert > L50_base):
            return (False, f"L50 did not increase when rhoB1 increased by {delta*100}%: base={L50_base:.2f}, perturbed={L50_pert:.2f}")
        return (True, f"L50 monotonic: perturbing rhoB1 +{delta*100}% gives L50 increase {L50_base:.2f} -> {L50_pert:.2f}")

    # ------------------------------------------------------------
    # Weak answer sanity (broad plausible ranges, no re‑derivation)
    # ------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """
        Check that all outputs lie in physically plausible ranges.
        No math.isclose, no re‑derivation.
        """
        checks = [
            ("m0_GeV", self.m0, 0.08, 0.25),
            ("KE1_GeV", self.KE1, 0.3, 1.5),
            ("KE2_GeV", self.KE2, 0.01, 0.1),
            ("tau", self.tau, 1e-9, 1e-6),
            ("L50", self.L50, 10, 60)
        ]
        for name, val, lo, hi in checks:
            if val is None:
                return (False, f"{name} is None")
            if not (lo <= val <= hi):
                return (False, f"{name} = {val} outside plausible range [{lo}, {hi}]")
        return (True, "All values within broad plausible ranges")

    # ------------------------------------------------------------
    # Master runner
    # ------------------------------------------------------------
    def run_all_tests(self):
        """Discover and run all test_* methods."""
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