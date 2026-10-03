import math
import traceback
from typing import Any, Dict, List, Tuple

class PhysicsChecker:
    """
    Checker for the high-energy neutrino production problem (pion decay kinematics).
    Performs interface checks, theorem-based invariants, modeling quality checks,
    monotonicity checks, and a weak answer sanity.
    """

    # Physical constant (speed of light in m/s) – not a problem-specific given.
    C_LIGHT = 3e8

    def __init__(self, solution_instance: Any) -> None:
        # --- Extract given quantities (safely) ---
        givens = getattr(solution_instance, 'givens', {})

        self.p_pi = givens.get('p_pi', None)          # MeV/c
        self.m_pi = givens.get('m_pi', None)          # MeV
        self.m_mu = givens.get('m_mu', None)          # MeV
        self.tau_pi = givens.get('tau_pi', None)      # s
        self.c = givens.get('c', self.C_LIGHT)        # m/s (optional, default constant)

        # --- Extract results (safely) ---
        results = getattr(solution_instance, 'results', {})
        self.L = results.get('L', None)               # m
        self.theta_max_deg = results.get('theta_max_deg', None)   # degrees
        self.p_nu_min = results.get('p_nu_min', None) # MeV/c
        self.p_nu_max = results.get('p_nu_max', None) # MeV/c

        # --- Derived quantities (only if required givens are available) ---
        self.gamma = None
        self.beta = None
        self.p_star = None
        self.E_mu_CM = None
        self._compute_derived()

    def _compute_derived(self) -> None:
        """Compute gamma, beta, p_star, E_mu_CM if all needed givens are present."""
        if self.p_pi is not None and self.m_pi is not None:
            E_pi = math.sqrt(self.p_pi**2 + self.m_pi**2)
            self.gamma = E_pi / self.m_pi
            self.beta = self.p_pi / E_pi
        if self.m_pi is not None and self.m_mu is not None:
            self.p_star = (self.m_pi**2 - self.m_mu**2) / (2.0 * self.m_pi)
            self.E_mu_CM = (self.m_pi**2 + self.m_mu**2) / (2.0 * self.m_pi)

    # ------------------------------------------------------------------
    # Helper for missing givens
    # ------------------------------------------------------------------
    def _missing_ged(self) -> str:
        """Return a skip message if any required given is missing."""
        missing = []
        if self.p_pi is None:
            missing.append("p_pi")
        if self.m_pi is None:
            missing.append("m_pi")
        if self.m_mu is None:
            missing.append("m_mu")
        if self.tau_pi is None:
            missing.append("tau_pi")
        if missing:
            return "SKIP: missing givens: " + ", ".join(missing)
        return ""

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all expected result keys exist and are not None."""
        results_map = {
            'L': self.L,
            'theta_max_deg': self.theta_max_deg,
            'p_nu_min': self.p_nu_min,
            'p_nu_max': self.p_nu_max
        }
        missing_keys = [k for k, v in results_map.items() if v is None]
        if missing_keys:
            return (False, f"Missing result keys: {missing_keys}")
        for k, v in results_map.items():
            if not isinstance(v, (int, float)):
                return (False, f"Key '{k}' is not a number, got {type(v).__name__}")
        return (True, "All required result keys present and are numbers")

    def test_interface_givens_available(self) -> Tuple[bool, str]:
        """Check that all essential givens are provided."""
        missing = []
        if self.p_pi is None:
            missing.append("p_pi")
        if self.m_pi is None:
            missing.append("m_pi")
        if self.m_mu is None:
            missing.append("m_mu")
        if self.tau_pi is None:
            missing.append("tau_pi")
        if missing:
            return (False, f"Missing givens: {missing}")
        return (True, "All essential givens are present")

    # ------------------------------------------------------------------
    # Theorem-based invariants (lenient tolerance)
    # ------------------------------------------------------------------
    def test_invariant_decay_length(self) -> Tuple[bool, str]:
        """Check that L ≈ βγ c τ (time dilation)."""
        skip_msg = self._missing_ged()
        if skip_msg:
            return (False, skip_msg)
        if self.L is None or self.gamma is None or self.beta is None:
            return (False, "SKIP: missing L, gamma, or beta")
        L_expected = self.gamma * self.beta * self.c * self.tau_pi
        if math.isclose(self.L, L_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Decay length consistent: {self.L} vs {L_expected:.3e}")
        return (False, f"Decay length mismatch: {self.L} vs {L_expected:.3e}")

    def test_invariant_cm_energy_momentum(self) -> Tuple[bool, str]:
        """Energy conservation in CMS: E_mu_CM + p_star = m_pi (neutrino massless)."""
        if self.m_pi is None or self.m_mu is None:
            return (False, "SKIP: missing m_pi or m_mu")
        if self.p_star is None or self.E_mu_CM is None:
            return (False, "SKIP: p_star or E_mu_CM not computed")
        total = self.E_mu_CM + self.p_star
        if math.isclose(total, self.m_pi, rel_tol=1e-5, abs_tol=1e-2):
            return (True, "CMS energy+momentum conserved")
        return (False, f"CMS energy+momentum sum {total:.4f} != m_pi {self.m_pi}")

    def test_invariant_muon_angle(self) -> Tuple[bool, str]:
        """tan(theta_max) ≈ p_star / (γ β E_mu_CM)."""
        if self.theta_max_deg is None:
            return (False, "SKIP: missing theta_max_deg")
        if self.gamma is None or self.beta is None or self.p_star is None or self.E_mu_CM is None:
            return (False, "SKIP: missing derived quantities")
        tan_expected = self.p_star / (self.gamma * self.beta * self.E_mu_CM)
        theta_expected_rad = math.atan(tan_expected)
        theta_expected_deg = theta_expected_rad * 180.0 / math.pi
        if math.isclose(self.theta_max_deg, theta_expected_deg, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Muon angle consistent: {self.theta_max_deg} deg vs {theta_expected_deg:.6f}")
        return (False, f"Muon angle mismatch: {self.theta_max_deg} deg vs {theta_expected_deg:.6f}")

    def test_invariant_neutrino_min(self) -> Tuple[bool, str]:
        """Neutrino minimum momentum (backward emission): p_nu_min ≈ γ (1‑β) p*."""
        if self.p_nu_min is None:
            return (False, "SKIP: missing p_nu_min")
        if self.gamma is None or self.beta is None or self.p_star is None:
            return (False, "SKIP: missing gamma, beta, p_star")
        expected = self.gamma * (1.0 - self.beta) * self.p_star
        if math.isclose(self.p_nu_min, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"p_nu_min consistent: {self.p_nu_min} vs {expected:.6e}")
        return (False, f"p_nu_min mismatch: {self.p_nu_min} vs {expected:.6e}")

    def test_invariant_neutrino_max(self) -> Tuple[bool, str]:
        """Neutrino maximum momentum (forward emission): p_nu_max ≈ γ (1+β) p*."""
        if self.p_nu_max is None:
            return (False, "SKIP: missing p_nu_max")
        if self.gamma is None or self.beta is None or self.p_star is None:
            return (False, "SKIP: missing gamma, beta, p_star")
        expected = self.gamma * (1.0 + self.beta) * self.p_star
        if math.isclose(self.p_nu_max, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"p_nu_max consistent: {self.p_nu_max} vs {expected:.6e}")
        return (False, f"p_nu_max mismatch: {self.p_nu_max} vs {expected:.6e}")

    # ------------------------------------------------------------------
    # Modeling / trace checks
    # ------------------------------------------------------------------
    def test_modeling_gamma_beta_relation(self) -> Tuple[bool, str]:
        """γ β ≈ p_pi / m_pi (for relativistic pion)."""
        if self.p_pi is None or self.m_pi is None:
            return (False, "SKIP: missing p_pi or m_pi")
        if self.gamma is None or self.beta is None:
            return (False, "SKIP: gamma or beta not derived")
        lhs = self.gamma * self.beta
        rhs = self.p_pi / self.m_pi
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"γβ = {lhs:.3f} ≈ p_pi/m_pi = {rhs:.3f}")
        return (False, f"γβ mismatch: {lhs} vs {rhs}")

    def test_modeling_gamma_from_energy(self) -> Tuple[bool, str]:
        """γ = sqrt((p_pi/m_pi)^2 + 1)."""
        if self.p_pi is None or self.m_pi is None:
            return (False, "SKIP: missing p_pi or m_pi")
        if self.gamma is None:
            return (False, "SKIP: gamma not derived")
        expected = math.sqrt((self.p_pi / self.m_pi)**2 + 1)
        if math.isclose(self.gamma, expected, rel_tol=1e-5, abs_tol=1e-5):
            return (True, f"γ = {self.gamma} correct (expected {expected:.3f})")
        return (False, f"γ mismatch: {self.gamma} vs {expected}")

    def test_modeling_p_star_from_masses(self) -> Tuple[bool, str]:
        """p* = (m_pi^2 – m_mu^2)/(2 m_pi)."""
        if self.m_pi is None or self.m_mu is None:
            return (False, "SKIP: missing m_pi or m_mu")
        if self.p_star is None:
            return (False, "SKIP: p_star not derived")
        expected = (self.m_pi**2 - self.m_mu**2) / (2.0 * self.m_pi)
        if math.isclose(self.p_star, expected, rel_tol=1e-5, abs_tol=1e-2):
            return (True, f"p* = {self.p_star} correct (expected {expected:.4f})")
        return (False, f"p* mismatch: {self.p_star} vs {expected}")

    # ------------------------------------------------------------------
    # Monotonicity tests (each explicitly perturbs a direct input and compares)
    # ------------------------------------------------------------------
    def _rebuild_from_perturbed_p_pi(self, p_pi_pert: float) -> Dict[str, float]:
        """
        Compute derived quantities and outputs using a perturbed p_pi,
        keeping m_pi, m_mu, tau_pi, c unchanged (as stored in self).
        Returns dict with keys: gamma, beta, p_star (unchanged), L, theta_max_rad,
        p_nu_min, p_nu_max.
        """
        # Compute derived quantities from perturbed p_pi
        E_pi_pert = math.sqrt(p_pi_pert**2 + self.m_pi**2)
        gamma_pert = E_pi_pert / self.m_pi
        beta_pert = p_pi_pert / E_pi_pert
        # Compute outputs using the perturbed values but original masses for p_star
        p_star = (self.m_pi**2 - self.m_mu**2) / (2.0 * self.m_pi)
        E_mu_CM = (self.m_pi**2 + self.m_mu**2) / (2.0 * self.m_pi)
        L_pert = gamma_pert * beta_pert * self.c * self.tau_pi
        theta_max_rad = math.atan(p_star / (gamma_pert * beta_pert * E_mu_CM))
        p_nu_min_pert = gamma_pert * (1.0 - beta_pert) * p_star
        p_nu_max_pert = gamma_pert * (1.0 + beta_pert) * p_star
        return {
            'gamma': gamma_pert,
            'beta': beta_pert,
            'p_star': p_star,
            'L': L_pert,
            'theta_max_rad': theta_max_rad,
            'p_nu_min': p_nu_min_pert,
            'p_nu_max': p_nu_max_pert
        }

    def test_monotonicity_decay_length_increases_with_p_pi(self) -> Tuple[bool, str]:
        """Increasing p_pi → larger decay length."""
        if self.p_pi is None or self.m_pi is None or self.tau_pi is None:
            return (False, "SKIP: missing p_pi, m_pi, or tau_pi")
        # Baseline: compute L using current derived values (or directly)
        if self.gamma is None or self.beta is None:
            return (False, "SKIP: gamma/beta not derived")
        L_base = self.gamma * self.beta * self.c * self.tau_pi
        # Perturb p_pi by +1%
        p_pi_pert = self.p_pi * 1.01
        pert_data = self._rebuild_from_perturbed_p_pi(p_pi_pert)
        L_pert = pert_data['L']
        if L_pert > L_base:
            return (True, f"Decay length increases with p_pi: {L_pert:.3e} > {L_base:.3e}")
        return (False, f"Decay length does NOT increase: {L_pert:.3e} vs {L_base:.3e}")

    def test_monotonicity_theta_max_decreases_with_p_pi(self) -> Tuple[bool, str]:
        """Increasing p_pi → smaller muon maximum angle."""
        if self.p_pi is None or self.m_pi is None or self.m_mu is None:
            return (False, "SKIP: missing p_pi, m_pi, or m_mu")
        # Baseline
        if self.gamma is None or self.beta is None or self.p_star is None or self.E_mu_CM is None:
            return (False, "SKIP: derived quantities not computed")
        theta_base_rad = math.atan(self.p_star / (self.gamma * self.beta * self.E_mu_CM))
        # Perturb p_pi by +1%
        p_pi_pert = self.p_pi * 1.01
        pert_data = self._rebuild_from_perturbed_p_pi(p_pi_pert)
        theta_pert_rad = pert_data['theta_max_rad']
        if theta_pert_rad < theta_base_rad:
            return (True, f"Max muon angle decreases with p_pi: {theta_pert_rad:.6e} < {theta_base_rad:.6e} rad")
        return (False, f"Max muon angle does NOT decrease: {theta_pert_rad:.6e} vs {theta_base_rad:.6e} rad")

    def test_monotonicity_neutrino_max_increases_with_p_pi(self) -> Tuple[bool, str]:
        """Increasing p_pi → larger neutrino max momentum."""
        if self.p_pi is None or self.m_pi is None or self.m_mu is None:
            return (False, "SKIP: missing p_pi, m_pi, or m_mu")
        if self.gamma is None or self.beta is None or self.p_star is None:
            return (False, "SKIP: derived quantities not computed")
        p_max_base = self.gamma * (1.0 + self.beta) * self.p_star
        # Perturb p_pi by +1%
        p_pi_pert = self.p_pi * 1.01
        pert_data = self._rebuild_from_perturbed_p_pi(p_pi_pert)
        p_max_pert = pert_data['p_nu_max']
        if p_max_pert > p_max_base:
            return (True, f"Neutrino max increases with p_pi: {p_max_pert:.6e} > {p_max_base:.6e}")
        return (False, f"Neutrino max does NOT increase: {p_max_pert:.6e} vs {p_max_base:.6e}")

    def test_monotonicity_neutrino_min_decreases_with_p_pi(self) -> Tuple[bool, str]:
        """Increasing p_pi → smaller neutrino min momentum."""
        if self.p_pi is None or self.m_pi is None or self.m_mu is None:
            return (False, "SKIP: missing p_pi, m_pi, or m_mu")
        if self.gamma is None or self.beta is None or self.p_star is None:
            return (False, "SKIP: derived quantities not computed")
        p_min_base = self.gamma * (1.0 - self.beta) * self.p_star
        # Perturb p_pi by +1%
        p_pi_pert = self.p_pi * 1.01
        pert_data = self._rebuild_from_perturbed_p_pi(p_pi_pert)
        p_min_pert = pert_data['p_nu_min']
        if p_min_pert < p_min_base:
            return (True, f"Neutrino min decreases with p_pi: {p_min_pert:.6e} < {p_min_base:.6e}")
        return (False, f"Neutrino min does NOT decrease: {p_min_pert:.6e} vs {p_min_base:.6e}")

    # ------------------------------------------------------------------
    # Weak answer sanity (at most one test, no isclose, no re‑derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_broad_ranges(self) -> Tuple[bool, str]:
        """
        Ensure final answers are finite, positive, and within plausible orders.
        No re‑derivation of expected values.
        """
        checks = []
        # L (m) – typical value ~11 km = 1.1e4 m, plausible range 0 to 2e4 m
        if self.L is not None:
            if not (0 < self.L < 2e4):
                checks.append(f"L={self.L} outside plausible range (0, 20000) m")
        # theta_max_deg – very small angle, should be < 1 deg
        if self.theta_max_deg is not None:
            if not (0 < self.theta_max_deg < 1.0):
                checks.append(f"theta_max_deg={self.theta_max_deg} outside plausible range (0, 1) deg")
        # p_nu_min – very small, should be positive and less than 1 MeV/c
        if self.p_nu_min is not None:
            if not (0 < self.p_nu_min < 1.0):
                checks.append(f"p_nu_min={self.p_nu_min} outside plausible range (0, 1) MeV/c")
        # p_nu_max – large, should be positive and less than 1e6 (order 10^5)
        if self.p_nu_max is not None:
            if not (0 < self.p_nu_max < 1e6):
                checks.append(f"p_nu_max={self.p_nu_max} outside plausible range (0, 1e6) MeV/c")
        if checks:
            return (False, "Plausibility failures: " + "; ".join(checks))
        return (True, "All answers within broad plausible ranges")

    # ------------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover and execute all test_* methods.
        Each crash is caught and logged as a failure.
        Returns (all_pass, log_list).
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)