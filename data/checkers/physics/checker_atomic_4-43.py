import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens safely – default to None if missing
        self.mK = getattr(solution_instance, 'mK', None)
        self.mmu = getattr(solution_instance, 'mmu', None)
        self.mnu = getattr(solution_instance, 'mnu', None)

        # Extract results dictionary safely
        self.results = getattr(solution_instance, 'results', {})
        self.p = self.results.get('p', None)
        self.T_mu = self.results.get('T_mu', None)

    # ==================== INTERFACE TESTS ====================
    def test_interface_givens_present(self):
        """Check that all required givens (mK, mmu, mnu) exist and are numeric."""
        if self.mK is None or self.mmu is None or self.mnu is None:
            return (False, "SKIP: one or more givens (mK, mmu, mnu) are missing")
        if not all(isinstance(x, (int, float)) for x in [self.mK, self.mmu, self.mnu]):
            return (False, "givens are not all numeric")
        return (True, "all givens present and numeric")

    def test_interface_results_present(self):
        """Check that required result keys 'p' and 'T_mu' exist and are numeric."""
        if self.p is None or self.T_mu is None:
            return (False, "SKIP: one or more required results (p, T_mu) missing")
        if not all(isinstance(x, (int, float)) for x in [self.p, self.T_mu]):
            return (False, "results are not all numeric")
        return (True, "required results present and numeric")

    # ==================== INVARIANT TESTS (theorem-based physical sanity) ====================
    def test_invariant_energy_conservation(self):
        """Check total energy conservation: mK ≈ sqrt(p^2 + mmu^2) + p."""
        if any(x is None for x in [self.p, self.mK, self.mmu]):
            return (False, "SKIP: missing p, mK, or mmu")
        p = self.p
        mK = self.mK
        mmu = self.mmu
        E_mu = math.sqrt(p**2 + mmu**2)
        E_tot = E_mu + p
        if math.isclose(E_tot, mK, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"energy conservation holds: sqrt(p^2+mmu^2)+p ≈ {E_tot:.3f}, mK={mK}")
        else:
            return (False, f"energy conservation violated: sqrt(p^2+mmu^2)+p={E_tot:.3f} vs mK={mK}")

    def test_invariant_muon_on_shell(self):
        """Check muon satisfies relativistic dispersion: E_mu^2 = p^2 + mmu^2."""
        if any(x is None for x in [self.p, self.mmu, self.T_mu]):
            return (False, "SKIP: missing p, mmu, or T_mu")
        E_mu = math.sqrt(self.p**2 + self.mmu**2)
        E_mu_via_T = self.T_mu + self.mmu
        if math.isclose(E_mu, E_mu_via_T, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"muon on-shell: E_mu from dispersion={E_mu:.3f}, from T_mu+mmu={E_mu_via_T:.3f}")
        else:
            return (False, f"muon on-shell violation: dispersion gives {E_mu:.3f}, but T_mu+mmu={E_mu_via_T:.3f}")

    def test_invariant_neutrino_on_shell(self):
        """Check neutrino (massless) satisfies E_nu = p, using energy conservation."""
        if any(x is None for x in [self.mK, self.mmu, self.p, self.T_mu]):
            return (False, "SKIP: missing data for neutrino test")
        # Total energy mK = E_mu + E_nu -> E_nu = mK - E_mu
        E_mu = math.sqrt(self.p**2 + self.mmu**2)
        E_nu_inferred = self.mK - E_mu
        # For massless particle, E_nu should equal p
        if math.isclose(E_nu_inferred, self.p, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"neutrino on-shell: inferred E_nu={E_nu_inferred:.3f} equals p={self.p:.3f}")
        else:
            return (False, f"neutrino on-shell violation: inferred E_nu={E_nu_inferred:.3f} != p={self.p:.3f}")

    # ==================== MODELING / TRACE TESTS (correct definitions, step consistency) ====================
    def test_modeling_momentum_formula(self):
        """Check that p derives from the standard formula: p = (mK^2 - mmu^2)/(2*mK)."""
        if any(x is None for x in [self.p, self.mK, self.mmu]):
            return (False, "SKIP: missing p, mK, or mmu")
        p_expected = (self.mK**2 - self.mmu**2) / (2 * self.mK)
        if math.isclose(self.p, p_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"p matches formula: p={self.p:.3f}, expected={p_expected:.3f}")
        else:
            return (False, f"p formula violation: p={self.p:.3f}, expected={p_expected:.3f}")

    def test_modeling_muon_kinetic_energy(self):
        """Check that muon kinetic energy is defined as sqrt(p^2+mmu^2)-mmu."""
        if any(x is None for x in [self.T_mu, self.p, self.mmu]):
            return (False, "SKIP: missing T_mu, p, or mmu")
        T_mu_expected = math.sqrt(self.p**2 + self.mmu**2) - self.mmu
        if math.isclose(self.T_mu, T_mu_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"T_mu matches definition: T_mu={self.T_mu:.3f}, expected={T_mu_expected:.3f}")
        else:
            return (False, f"T_mu definition violation: T_mu={self.T_mu:.3f}, expected={T_mu_expected:.3f}")

    def test_modeling_neutrino_kinetic_energy(self):
        """Check that neutrino kinetic energy equals p (since massless), using energy conservation."""
        if any(x is None for x in [self.mK, self.mmu, self.p]):
            return (False, "SKIP: missing mK, mmu, or p")
        E_mu = math.sqrt(self.p**2 + self.mmu**2)
        T_nu_inferred = self.mK - E_mu
        # Neutrino kinetic energy equals its total energy (massless) = p
        if math.isclose(self.p, T_nu_inferred, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"neutrino kinetic energy consistent: p={self.p:.3f}, inferred T_nu={T_nu_inferred:.3f}")
        else:
            return (False, f"neutrino kinetic energy inconsistency: p={self.p:.3f}, inferred T_nu={T_nu_inferred:.3f}")

    # ==================== MONOTONICITY TESTS (directional checks via perturbation) ====================
    def test_monotonicity_momentum_with_Kmass(self):
        """Check that increasing mK increases p (positive derivative).  Perturb mK by +1%."""
        if any(x is None for x in [self.mK, self.mmu, self.p]):
            return (False, "SKIP: missing mK, mmu, or p")
        # Baseline
        mK0 = self.mK
        mmu = self.mmu
        p0 = self.p  # baseline output
        # Perturb input mK by +1%
        mK1 = mK0 * 1.01
        # Compute perturbed output using the same physical formula
        p1 = (mK1**2 - mmu**2) / (2 * mK1)
        # Compare perturbed output to baseline
        if p1 > p0:
            return (True, f"monotonicity holds: mK +1% gives p increase from {p0:.3f} to {p1:.3f}")
        else:
            return (False, f"monotonicity violation: mK +1% gives p={p1:.3f} (expected > {p0:.3f})")

    def test_monotonicity_energy_with_muonmass(self):
        """Check that increasing mmu decreases muon kinetic energy.  Perturb mmu by +1%."""
        if any(x is None for x in [self.mK, self.mmu, self.T_mu, self.p]):
            return (False, "SKIP: missing data for monotonicity")
        # Baseline
        mK = self.mK
        mmu0 = self.mmu
        # Use stored values as baseline; also compute baseline p0 to be consistent with stored T_mu?
        # We rely on stored p and T_mu as baseline.
        p0 = self.p
        T_mu0 = self.T_mu
        # Perturb mmu by +1%
        mmu1 = mmu0 * 1.01
        # Recompute p for the perturbed system (p depends on mK and mmu)
        p1 = (mK**2 - mmu1**2) / (2 * mK)
        # Compute perturbed muon kinetic energy
        T_mu1 = math.sqrt(p1**2 + mmu1**2) - mmu1
        # Compare perturbed T_mu to baseline
        if T_mu1 < T_mu0:
            return (True, f"monotonicity holds: mmu +1% gives T_mu decrease from {T_mu0:.3f} to {T_mu1:.3f}")
        else:
            return (False, f"monotonicity violation: mmu +1% gives T_mu={T_mu1:.3f} (expected < {T_mu0:.3f})")

    # ==================== WEAK ANSWER SANITY (only one such method allowed) ====================
    def test_answer_sanity_basics(self):
        """Very basic checks: positivity, finite values, sum constraint."""
        if any(x is None for x in [self.mK, self.mmu, self.p, self.T_mu]):
            return (False, "SKIP: missing data for sanity checks")
        if self.p <= 0:
            return (False, f"p should be >0, got {self.p}")
        if self.T_mu <= 0:
            return (False, f"T_mu should be >0, got {self.T_mu}")
        # Ensure sum of kinetic energies < mK (since mass converted)
        if self.T_mu + self.p >= self.mK:
            return (False, f"T_mu+T_nu ({self.T_mu}+{self.p}) should be < mK={self.mK}")
        return (True, "basic sanity: positivity and sum constraint satisfied")

    # ==================== RUN ALL TESTS ====================
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