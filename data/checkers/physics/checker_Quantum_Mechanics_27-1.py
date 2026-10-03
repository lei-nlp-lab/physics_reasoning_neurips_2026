import math
import traceback

class PhysicsChecker:
    """
    Physics auditor for the K- + proton bound state problem.
    Checks reduced mass, Bohr radius, transition energy, probability,
    and first-order strong interaction energy shift.
    """

    # Physical constants (in natural units: MeV, fm)
    ALPHA = 1.0 / 137.0
    HBAR_C = 197.326  # MeV·fm
    M_K = 494.0       # MeV/c^2
    M_P = 938.0       # MeV/c^2
    R0 = 1.0          # fm ( = 1e-13 cm)
    V0 = 50.0         # MeV

    def __init__(self, solution_instance):
        # Extract givens if present, otherwise use defaults from problem statement
        self.m_K = getattr(solution_instance, 'm_K', self.M_K)
        self.m_p = getattr(solution_instance, 'm_p', self.M_P)
        self.alpha = getattr(solution_instance, 'alpha', self.ALPHA)
        self.hbar_c = getattr(solution_instance, 'hbar_c', self.HBAR_C)
        self.r0 = getattr(solution_instance, 'r0', self.R0)
        self.V0 = getattr(solution_instance, 'V0', self.V0)

        # Extract results dictionary safely
        self.results = getattr(solution_instance, 'results', {})
        self.dE_trans_keV = self.results.get('dE_trans_keV', None)
        self.P = self.results.get('P', None)
        self.delta_E_eV = self.results.get('delta_E_eV', None)

        # Compute derived constants for checks (re-derived in-checker)
        self.mu = self.m_K * self.m_p / (self.m_K + self.m_p)  # MeV/c^2
        self.a_mu = self.hbar_c / (self.alpha * self.mu)       # fm

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

    # ---- Interface checks ----
    def test_interface_required_keys(self):
        """Check that all required result keys are present."""
        if self.dE_trans_keV is None:
            return (False, "Missing 'dE_trans_keV'")
        if self.P is None:
            return (False, "Missing 'P'")
        if self.delta_E_eV is None:
            return (False, "Missing 'delta_E_eV'")
        return (True, "All required keys present")

    def test_interface_types(self):
        """Check that numerical values are numeric and finite."""
        for name, val in [("dE_trans_keV", self.dE_trans_keV),
                          ("P", self.P),
                          ("delta_E_eV", self.delta_E_eV)]:
            if val is None:
                return (False, f"{name} is None")
            if not isinstance(val, (int, float)):
                return (False, f"{name} is not numeric")
            if not math.isfinite(val):
                return (False, f"{name} is not finite")
        return (True, "All values are numeric and finite")

    # ---- Invariant checks (theorem-based) ----
    def test_invariant_transition_energy(self):
        """Check that transition energy matches formula: ΔE = (3/8) α^2 μ c^2 ."""
        if self.dE_trans_keV is None:
            return (False, "SKIP: dE_trans_keV not available")
        # Compute expected value in keV
        expected_MeV = (3.0/8.0) * self.alpha**2 * self.mu
        expected_keV = expected_MeV * 1000.0
        if math.isclose(self.dE_trans_keV, expected_keV, rel_tol=0.02, abs_tol=0.1):
            return (True, f"dE_trans_keV = {self.dE_trans_keV} ~ expected {expected_keV}")
        else:
            return (False, f"dE_trans_keV = {self.dE_trans_keV} but expected {expected_keV}")

    def test_invariant_probability(self):
        """Check that probability matches approximate formula: P ≈ (4/3)(r0/aμ)^3."""
        if self.P is None:
            return (False, "SKIP: P not available")
        a_mu = self.a_mu
        x = self.r0 / a_mu
        expected_P = (4.0/3.0) * x**3
        # Use relative tolerance 10% for small numbers, absolute tolerance 1e-8
        if math.isclose(self.P, expected_P, rel_tol=0.1, abs_tol=1e-8):
            return (True, f"P = {self.P} ~ expected {expected_P}")
        else:
            return (False, f"P = {self.P} but expected {expected_P}")

    def test_invariant_energy_shift(self):
        """Check that energy shift matches: ΔE = V0 * P (in eV)."""
        if self.delta_E_eV is None or self.P is None:
            return (False, "SKIP: delta_E_eV or P not available")
        expected_eV = self.V0 * self.P * 1e6  # V0 in MeV, P dimensionless -> product in MeV -> eV
        if math.isclose(self.delta_E_eV, expected_eV, rel_tol=0.1, abs_tol=1.0):
            return (True, f"delta_E_eV = {self.delta_E_eV} ~ expected {expected_eV}")
        else:
            return (False, f"delta_E_eV = {self.delta_E_eV} but expected {expected_eV}")

    # ---- Modeling checks (quality of approximation) ----
    def test_modeling_reduced_mass(self):
        """Reduced mass should be between the two constituent masses."""
        if self.m_K is None or self.m_p is None:
            return (False, "SKIP: missing masses")
        if not (min(self.m_K, self.m_p) <= self.mu <= max(self.m_K, self.m_p)):
            return (False, f"Reduced mass {self.mu} not between constituents")
        return (True, f"Reduced mass μ = {self.mu} MeV/c^2")

    def test_modeling_bohr_radius(self):
        """Bohr radius should be positive and around 77 fm."""
        if self.a_mu is None:
            return (False, "SKIP: a_mu not computed")
        if self.a_mu <= 0:
            return (False, f"a_mu = {self.a_mu} is not positive")
        # Expect around 77 fm, allow factor 2 uncertainty
        if self.a_mu < 50 or self.a_mu > 120:
            return (False, f"a_mu = {self.a_mu} out of plausible range [50,120] fm")
        return (True, f"Bohr radius a_μ = {self.a_mu} fm")

    def test_modeling_probability_small(self):
        """Probability should be small (r0 << aμ)."""
        if self.P is None:
            return (False, "SKIP: P not available")
        if self.P <= 0 or self.P >= 1e-4:
            return (False, f"P = {self.P} out of plausible range (should be ≈ 3e-6)")
        return (True, f"Probability P = {self.P} (expected ~3e-6)")

    def test_modeling_energy_shift_positive(self):
        """Energy shift should be positive (strong interaction lowers 1S state)."""
        if self.delta_E_eV is None:
            return (False, "SKIP: delta_E_eV not available")
        if self.delta_E_eV <= 0:
            return (False, f"delta_E_eV = {self.delta_E_eV} is not positive")
        return (True, f"Energy shift = {self.delta_E_eV} eV > 0")

    # ---- Monotonicity checks ----
    def test_monotonicity_transition_energy_with_mass(self):
        """Increasing K- mass should increase transition energy."""
        if self.dE_trans_keV is None:
            return (False, "SKIP: dE_trans_keV not available")
        # Perturb m_K upward by 10%
        mK_pert = self.m_K * 1.1
        mu_pert = mK_pert * self.m_p / (mK_pert + self.m_p)
        expected_pert_MeV = (3.0/8.0) * self.alpha**2 * mu_pert
        expected_pert_keV = expected_pert_MeV * 1000.0
        if expected_pert_keV > self.dE_trans_keV:
            return (True, f"Increasing m_K increases transition energy (perturbed {expected_pert_keV} > original {self.dE_trans_keV})")
        else:
            return (False, f"Monotonicity violated: perturbed {expected_pert_keV} <= original {self.dE_trans_keV}")

    def test_monotonicity_probability_with_range(self):
        """Increasing r0 should increase probability."""
        if self.P is None:
            return (False, "SKIP: P not available")
        r0_pert = self.r0 * 1.2
        x_pert = r0_pert / self.a_mu
        expected_P_pert = (4.0/3.0) * x_pert**3
        if expected_P_pert > self.P:
            return (True, f"Increasing r0 increases probability (perturbed {expected_P_pert} > original {self.P})")
        else:
            return (False, f"Monotonicity violated: perturbed {expected_P_pert} <= original {self.P}")

    # ---- Answer sanity (weak, no re-derivation) ----
    def test_answer_sanity_sign_and_range(self):
        """All results should be positive and within physically plausible ranges."""
        checks = []
        if self.dE_trans_keV is not None:
            if self.dE_trans_keV <= 0:
                checks.append(f"dE_trans_keV={self.dE_trans_keV} must be >0")
            if self.dE_trans_keV > 100:
                checks.append(f"dE_trans_keV={self.dE_trans_keV} too large (>100 keV)")
        if self.P is not None:
            if self.P <= 0:
                checks.append(f"P={self.P} must be >0")
            if self.P > 1e-3:
                checks.append(f"P={self.P} too large (>1e-3)")
        if self.delta_E_eV is not None:
            if self.delta_E_eV <= 0:
                checks.append(f"delta_E_eV={self.delta_E_eV} must be >0")
            if self.delta_E_eV > 2000:
                checks.append(f"delta_E_eV={self.delta_E_eV} too large (>2000 eV)")
        if checks:
            return (False, "; ".join(checks))
        return (True, "All results positive and within plausible range")