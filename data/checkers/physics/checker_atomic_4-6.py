import math
import sympy as sp

class PhysicsChecker:
    """
    Audits the solution for a two-part relativistic problem:
    (a) Neutron travel with time dilation
    (b) Pion decay with energy/momentum conservation
    """

    def __init__(self, solution_instance):
        # Extract results safely (expected emit keys)
        results = getattr(solution_instance, 'results', {}) or {}
        self.gamma = results.get('gamma', None)
        self.E = results.get('E', None)
        self.E_mu = results.get('E_mu', None)
        self.p_mu = results.get('p_mu', None)
        self.m_pi = results.get('m_pi', None)

        # Extract givens if available (may be in .inputs, .givens, or .results)
        # Fallback to None – never assign numeric defaults
        givens = getattr(solution_instance, 'givens', {}) or {}
        inputs = getattr(solution_instance, 'inputs', {}) or {}
        # Combine multiple possible sources, results may also hold givens
        self.d = results.get('d', givens.get('d', inputs.get('d', None)))
        self.tau0_min = results.get('tau0', givens.get('tau0', inputs.get('tau0', None)))
        self.m_n = results.get('m_n', givens.get('m_n', inputs.get('m_n', None)))  # neutron rest mass in MeV
        self.K_mu = results.get('K_mu', givens.get('K_mu', inputs.get('K_mu', None)))
        self.m_mu = results.get('m_mu', givens.get('m_mu', inputs.get('m_mu', None)))
        # Optional: neutrino mass is 0, but not needed explicitly
        # c = 1 in natural units for Q2, but for Q1 we need conversion; treat c = 1 ly/yr
        self.c = 1.0  # natural unit ly/yr (for Q1 consistency)

    # ----------------------------------------------------------------------
    # Interface tests (required result keys exist and are non‑None)
    # ----------------------------------------------------------------------
    def test_interface_gamma_exists(self):
        if self.gamma is None:
            return (False, "SKIP: gamma not provided")
        return (True, "gamma exists")

    def test_interface_E_exists(self):
        if self.E is None:
            return (False, "SKIP: E not provided")
        return (True, "E exists")

    def test_interface_E_mu_exists(self):
        if self.E_mu is None:
            return (False, "SKIP: E_mu not provided")
        return (True, "E_mu exists")

    def test_interface_p_mu_exists(self):
        if self.p_mu is None:
            return (False, "SKIP: p_mu not provided")
        return (True, "p_mu exists")

    def test_interface_m_pi_exists(self):
        if self.m_pi is None:
            return (False, "SKIP: m_pi not provided")
        return (True, "m_pi exists")

    # ----------------------------------------------------------------------
    # Invariant tests (theorem‑based physical sanity with math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_gamma_gt_one(self):
        if self.gamma is None:
            return (False, "SKIP: gamma not available")
        if not (self.gamma > 1.0):
            return (False, f"gamma = {self.gamma} is not > 1")
        return (True, f"gamma > 1 holds")

    def test_invariant_time_dilation_distance(self):
        """
        For Q1: d = gamma * beta * c * tau0  (Earth frame distance)
        beta = sqrt(1 - 1/gamma^2)
        """
        if any(v is None for v in [self.gamma, self.d, self.tau0_min]):
            return (False, "SKIP: missing givens (d, tau0, gamma) for Q1")

        # tau0 in minutes -> years
        tau0_yr = self.tau0_min / (60 * 24 * 365)
        beta = math.sqrt(1.0 - 1.0 / (self.gamma ** 2))
        expected_d = self.gamma * beta * self.c * tau0_yr
        if not math.isclose(expected_d, self.d, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"d mismatch: computed {expected_d:.2e} vs {self.d:.2e}")
        return (True, "Time dilation distance relation holds")

    def test_invariant_energy_from_gamma(self):
        """E = gamma * m_n"""
        if any(v is None for v in [self.gamma, self.E, self.m_n]):
            return (False, "SKIP: missing gamma, E, or m_n")
        expected_E = self.gamma * self.m_n
        if not math.isclose(expected_E, self.E, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E = gamma*m_n mismatch: {expected_E:.2e} vs {self.E:.2e}")
        return (True, "Relativistic energy relation holds for neutron")

    def test_invariant_muon_total_energy(self):
        """E_mu = K_mu + m_mu"""
        if any(v is None for v in [self.E_mu, self.K_mu, self.m_mu]):
            return (False, "SKIP: missing E_mu, K_mu, or m_mu")
        expected_E_mu = self.K_mu + self.m_mu
        if not math.isclose(expected_E_mu, self.E_mu, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E_mu = K_mu + m_mu mismatch: {expected_E_mu} vs {self.E_mu}")
        return (True, "Muon total energy relation holds")

    def test_invariant_muon_momentum(self):
        """p_mu = sqrt(E_mu^2 - m_mu^2)"""
        if any(v is None for v in [self.E_mu, self.p_mu, self.m_mu]):
            return (False, "SKIP: missing E_mu, p_mu, or m_mu")
        expected_p = math.sqrt(self.E_mu**2 - self.m_mu**2)
        if not math.isclose(expected_p, self.p_mu, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Muon momentum mismatch: {expected_p:.3f} vs {self.p_mu:.3f}")
        return (True, "Muon momentum from energy-momentum relation holds")

    def test_invariant_pion_mass_conservation(self):
        """m_pi = E_mu + p_mu   (since E_nu = p_mu)"""
        if any(v is None for v in [self.E_mu, self.p_mu, self.m_pi]):
            return (False, "SKIP: missing E_mu, p_mu, or m_pi")
        expected_m_pi = self.E_mu + self.p_mu
        if not math.isclose(expected_m_pi, self.m_pi, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Pion mass mismatch: {expected_m_pi:.3f} vs {self.m_pi:.3f}")
        return (True, "Energy conservation in pion decay holds")

    # ----------------------------------------------------------------------
    # Modeling / trace checks (based on explanation steps)
    # ----------------------------------------------------------------------
    def test_modeling_gamma_from_distance_time(self):
        """
        gamma should be consistent with the given d and tau0 using
        gamma = sqrt(1 + (d / (c * tau0_yr))^2)
        """
        if any(v is None for v in [self.gamma, self.d, self.tau0_min]):
            return (False, "SKIP: missing Q1 givens")
        tau0_yr = self.tau0_min / (60 * 24 * 365)
        # c = 1 ly/yr
        expected_gamma = math.sqrt(1 + (self.d / tau0_yr) ** 2)
        if not math.isclose(expected_gamma, self.gamma, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"gamma from d and tau0: expected {expected_gamma:.2e}, got {self.gamma:.2e}")
        return (True, "Gamma consistent with travel distance and proper half‑life")

    def test_modeling_neutrino_energy_is_momentum(self):
        """E_nu = p_mu (since neutrino mass=0)"""
        if self.p_mu is None:
            return (False, "SKIP: p_mu not available")
        # E_nu not emitted, but we can infer it must equal p_mu from conservation
        # This test checks that m_pi = E_mu + p_mu (already in invariant)
        # So we can just check sign and consistency
        if self.p_mu < 0:
            return (False, "p_mu is negative, unphysical")
        # Also check that m_pi - E_mu = p_mu (approximately)
        if any(v is None for v in [self.m_pi, self.E_mu]):
            return (False, "SKIP: m_pi or E_mu missing")
        if not math.isclose(self.m_pi - self.E_mu, self.p_mu, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Neutrino energy implied by difference not equal to p_mu")
        return (True, "Neutrino energy correctly taken as muon momentum")

    def test_modeling_pion_decay_momentum_conservation(self):
        """Momenta of muon and neutrino are equal and opposite (only magnitude matters)"""
        # We only have p_mu; neutrino momentum magnitude should equal p_mu.
        # The invariant already checks m_pi = E_mu + p_mu which implies conservation.
        # Additional sanity: p_mu > 0
        if self.p_mu is not None and self.p_mu <= 0:
            return (False, "Muon momentum should be positive")
        return (True, "Momentum is positive (magnitude check)")

    # ----------------------------------------------------------------------
    # Monotonicity tests (input perturbation effects)
    # ----------------------------------------------------------------------
    def test_monotonicity_gamma_increases_with_distance(self):
        """
        If distance d is increased by 1%, gamma should increase.
        We recompute gamma from the formula assuming the same tau0.
        """
        if any(v is None for v in [self.gamma, self.d, self.tau0_min]):
            return (False, "SKIP: missing Q1 data")
        tau0_yr = self.tau0_min / (60 * 24 * 365)
        base_gamma = math.sqrt(1 + (self.d / tau0_yr)**2)
        perturbed_d = self.d * 1.01
        new_gamma = math.sqrt(1 + (perturbed_d / tau0_yr)**2)
        if not (new_gamma > base_gamma):
            return (False, "Increasing distance did not increase gamma")
        return (True, "Monotonicity: gamma increases with distance")

    def test_monotonicity_gamma_decreases_with_half_life(self):
        """
        If proper half‑life tau0 is increased by 1%, gamma should decrease.
        """
        if any(v is None for v in [self.gamma, self.d, self.tau0_min]):
            return (False, "SKIP: missing Q1 data")
        tau0_yr = self.tau0_min / (60 * 24 * 365)
        base_gamma = math.sqrt(1 + (self.d / tau0_yr)**2)
        perturbed_tau0 = self.tau0_min * 1.01
        tau0_yr_new = perturbed_tau0 / (60 * 24 * 365)
        new_gamma = math.sqrt(1 + (self.d / tau0_yr_new)**2)
        if not (new_gamma < base_gamma):
            return (False, "Increasing half‑life did not decrease gamma")
        return (True, "Monotonicity: gamma decreases with longer proper half‑life")

    def test_monotonicity_pion_mass_increases_with_muon_kinetic_energy(self):
        """
        If K_mu is increased, all derived quantities (E_mu, p_mu, m_pi) increase.
        Check m_pi.
        """
        if any(v is None for v in [self.K_mu, self.m_mu, self.m_pi]):
            return (False, "SKIP: missing Q2 data")
        base_K = self.K_mu
        base_m_pi = self.m_pi
        # Simulate a 10% increase in K_mu
        new_K = base_K * 1.10
        new_E_mu = new_K + self.m_mu
        new_p_mu = math.sqrt(new_E_mu**2 - self.m_mu**2)
        new_m_pi = new_E_mu + new_p_mu
        if not (new_m_pi > base_m_pi):
            return (False, "Increasing muon kinetic energy did not increase pion mass")
        return (True, "Monotonicity: m_pi increases with K_mu")

    # ----------------------------------------------------------------------
    # Weak answer sanity test (only sign, finiteness, broad range)
    # ----------------------------------------------------------------------
    def test_answer_sanity_positive_values(self):
        """
        All final results should be positive and finite.
        """
        values = {'gamma': self.gamma, 'E': self.E,
                  'E_mu': self.E_mu, 'p_mu': self.p_mu,
                  'm_pi': self.m_pi}
        for name, val in values.items():
            if val is None:
                continue
            if not (isinstance(val, (int, float)) and math.isfinite(val)):
                return (False, f"{name} is not a finite number, got {val}")
            if val <= 0:
                return (False, f"{name} is not positive, got {val}")
        return (True, "All available values are positive and finite")

    # ----------------------------------------------------------------------
    # Central test runner
    # ----------------------------------------------------------------------
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