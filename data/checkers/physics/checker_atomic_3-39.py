import math
import traceback
from typing import Any, Dict, Optional, Tuple

class PhysicsChecker:
    """
    Auditor for the decay rate estimation of Σ⁻ → Λ e ν̄.
    Extracts necessary masses, neutron lifetime, and the computed decay rate,
    then performs:
        - interface checks (keys present, numeric)
        - invariant checks (positivity, energy conservation)
        - modeling/trace checks (phase space integral consistency, asymptotic approximation validity)
        - weak answer sanity (order of magnitude)
        - monotonicity test (perturbing m_Sigma)
    """

    def __init__(self, solution_instance: Any):
        # Helper to safely extract from object or dict
        def safe_get(obj, attr, default=None):
            if hasattr(obj, attr):
                return getattr(obj, attr)
            if isinstance(obj, dict):
                return obj.get(attr, default)
            return default

        # Extract given parameters
        self.tau_n = safe_get(solution_instance, 'tau_n', None)
        self.m_n = safe_get(solution_instance, 'm_n', None)
        self.m_p = safe_get(solution_instance, 'm_p', None)
        self.m_e = safe_get(solution_instance, 'm_e', None)
        self.m_Sigma = safe_get(solution_instance, 'm_Sigma', None)
        self.m_Lambda = safe_get(solution_instance, 'm_Lambda', None)

        # Extract result
        self.Gamma_Sigma = safe_get(solution_instance.results, 'Gamma_Sigma', None) \
            if hasattr(solution_instance, 'results') else None

        # W0 = (m_initial - m_final) / m_e
        self.W0_n = None
        self.W0_Sigma = None
        if all(v is not None for v in [self.m_n, self.m_p, self.m_e]):
            self.W0_n = (self.m_n - self.m_p) / self.m_e
        if all(v is not None for v in [self.m_Sigma, self.m_Lambda, self.m_e]):
            self.W0_Sigma = (self.m_Sigma - self.m_Lambda) / self.m_e

        # Derived neutron decay rate
        self.Gamma_n = None
        if self.tau_n is not None and self.tau_n > 0:
            self.Gamma_n = 1.0 / self.tau_n

    # ------------------------------------------------------------
    # Interface Checks
    # ------------------------------------------------------------
    def test_interface_result_present(self) -> Tuple[bool, str]:
        """Check that the required result key 'Gamma_Sigma' exists."""
        if self.Gamma_Sigma is None:
            return (False, "Results['Gamma_Sigma'] is missing.")
        return (True, "Gamma_Sigma present.")

    def test_interface_masses_provided(self) -> Tuple[bool, str]:
        """Ensure all necessary masses are present."""
        needed = {
            'm_n': self.m_n, 'm_p': self.m_p, 'm_e': self.m_e,
            'm_Sigma': self.m_Sigma, 'm_Lambda': self.m_Lambda
        }
        missing = [name for name, val in needed.items() if val is None]
        if missing:
            return (False, f"Missing masses: {missing}")
        return (True, "All masses provided.")

    def test_interface_tau_n_provided(self) -> Tuple[bool, str]:
        """Check that neutron lifetime is given."""
        if self.tau_n is None:
            return (False, "tau_n is missing.")
        return (True, "tau_n provided.")

    # ------------------------------------------------------------
    # Invariant Checks (theorem-based physics sanity)
    # ------------------------------------------------------------
    def test_invariant_decay_rate_positive(self) -> Tuple[bool, str]:
        """Decay rate must be positive."""
        if self.Gamma_Sigma is None:
            return (False, "SKIP: Gamma_Sigma missing.")
        if self.Gamma_Sigma <= 0:
            return (False, f"Gamma_Sigma = {self.Gamma_Sigma} is not positive.")
        return (True, "Gamma_Sigma > 0.")

    def test_invariant_energy_conservation_possible(self) -> Tuple[bool, str]:
        """Check that Q values are positive (decay possible)."""
        if any(v is None for v in [self.m_n, self.m_p, self.m_e, self.m_Sigma, self.m_Lambda]):
            return (False, "SKIP: masses missing.")
        Q_n = self.m_n - self.m_p - self.m_e
        Q_Sigma = self.m_Sigma - self.m_Lambda - self.m_e
        if Q_n <= 0:
            return (False, f"Neutron Q-value {Q_n} not positive.")
        if Q_Sigma <= 0:
            return (False, f"Sigma Q-value {Q_Sigma} not positive.")
        return (True, "Both Q-values positive.")

    def test_invariant_ratio_greater_than_one(self) -> Tuple[bool, str]:
        """Sigma decay should be much faster than neutron due to larger Q."""
        if self.Gamma_n is None or self.Gamma_Sigma is None:
            return (False, "SKIP: need Gamma_n and Gamma_Sigma.")
        ratio = self.Gamma_Sigma / self.Gamma_n
        if ratio <= 1:
            return (False, f"Gamma_Sigma / Gamma_n = {ratio:.2g} should be >> 1.")
        return (True, f"Ratio {ratio:.2g} >> 1.")

    # ------------------------------------------------------------
    # Modeling / Trace Checks
    # ------------------------------------------------------------
    def test_modeling_phase_space_ratio(self) -> Tuple[bool, str]:
        """
        Check that the reported Gamma_Sigma is consistent with the ratio of
        phase space integrals computed from masses.
        Uses exact integral for neutron and high-W0 approximation for Sigma.
        """
        if any(v is None for v in [self.W0_n, self.W0_Sigma, self.tau_n, self.Gamma_Sigma]):
            return (False, "SKIP: missing data for phase space check.")

        if self.W0_n <= 1:
            return (False, f"W0_n = {self.W0_n} not > 1, unphysical.")

        # Exact phase space integral for neutron
        sqrt_term = math.sqrt(self.W0_n**2 - 1)
        exact_part = (1/60) * (2*self.W0_n**4 - 9*self.W0_n**2 - 8) * sqrt_term
        log_part = 0.5 * self.W0_n * math.log(self.W0_n + sqrt_term)
        f_n = exact_part + log_part

        # High-W0 approximation for Sigma: f ≈ W0^5 / 30
        f_Sigma_approx = (self.W0_Sigma**5) / 30.0

        ratio = f_Sigma_approx / f_n
        expected_Gamma_Sigma = self.Gamma_n * ratio

        # Compare with provided value (rel_tol 2% sufficient for approximate formula)
        if not math.isclose(self.Gamma_Sigma, expected_Gamma_Sigma, rel_tol=2e-2):
            return (False, f"Gamma_Sigma = {self.Gamma_Sigma:.4e}, "
                    f"expected ~ {expected_Gamma_Sigma:.4e} from phase space ratio.")
        return (True, f"Gamma_Sigma consistent with phase space ratio (ratio={ratio:.2g}).")

    def test_modeling_asymptotic_validity(self) -> Tuple[bool, str]:
        """
        Verify that using the high-W0 asymptotic (W0^5/30) for the Sigma decay
        is a good approximation by comparing it with the exact integral.
        This test checks modelling quality: the approximation should be within 1%.
        """
        if self.W0_Sigma is None:
            return (False, "SKIP: W0_Sigma missing.")

        # Compute exact phase space integral for Sigma (same formula as neutron)
        if self.W0_Sigma <= 1:
            return (False, f"W0_Sigma = {self.W0_Sigma} not > 1, unphysical.")

        sqrt_term = math.sqrt(self.W0_Sigma**2 - 1)
        exact_part = (1/60) * (2*self.W0_Sigma**4 - 9*self.W0_Sigma**2 - 8) * sqrt_term
        log_part = 0.5 * self.W0_Sigma * math.log(self.W0_Sigma + sqrt_term)
        f_Sigma_exact = exact_part + log_part

        # Asymptotic approximation
        f_Sigma_approx = (self.W0_Sigma**5) / 30.0

        # Relative difference
        rel_diff = abs(f_Sigma_exact - f_Sigma_approx) / f_Sigma_exact
        if rel_diff > 0.01:
            return (False, f"Asymptotic approximation error {rel_diff:.4f} > 1% for Sigma.")
        return (True, f"Asymptotic error {rel_diff:.4f} <= 1%, approximation valid.")

    # ------------------------------------------------------------
    # Weak Answer Sanity (order of magnitude only)
    # ------------------------------------------------------------
    def test_answer_sanity_order(self) -> Tuple[bool, str]:
        """
        Weak check: Gamma_Sigma should be in the range 1e6 - 1e8 s⁻¹
        (given Q_Sigma ~ 80 MeV and tau_n ~ 1000 s).
        """
        if self.Gamma_Sigma is None:
            return (False, "SKIP: Gamma_Sigma missing.")
        if not (1e6 <= self.Gamma_Sigma <= 1e8):
            return (False, f"Gamma_Sigma = {self.Gamma_Sigma:.2e} outside plausible range [1e6, 1e8].")
        return (True, "Gamma_Sigma within plausible order of magnitude.")

    # ------------------------------------------------------------
    # Monotonicity (conceptual check using perturbed masses)
    # ------------------------------------------------------------
    def test_monotonicity_mass_increase(self) -> Tuple[bool, str]:
        """
        Perturb m_Sigma upward by 1%, recompute expected rate via the same
        approximate phase space ratio, and verify that the expected rate increases.
        (This tests the physical trend, not the solution's adaptation.)
        """
        if any(v is None for v in [self.m_Sigma, self.m_Lambda, self.m_e, self.tau_n]):
            return (False, "SKIP: missing data for monotonicity.")

        # Original Q_Sigma
        Q_orig = self.m_Sigma - self.m_Lambda - self.m_e
        if Q_orig <= 0:
            return (False, "Q_Sigma <= 0, cannot test.")

        # Perturb m_Sigma by +1%
        m_Sigma_pert = self.m_Sigma * 1.01
        Q_pert = m_Sigma_pert - self.m_Lambda - self.m_e
        if Q_pert <= 0:
            return (False, "Perturbed Q not positive.")

        # Use approximate phase space ratio with original neutron integral
        # Since f_Sigma ∝ W0^5 ∝ (Q_pert/m_e)^5, the ratio of rates is (Q_pert/Q_orig)^5.
        # Check that this is > 1.
        rate_ratio = (Q_pert / Q_orig) ** 5
        if rate_ratio <= 1.0:
            return (False, f"Expected rate ratio after +1% mass increase is {rate_ratio:.4f} ≤ 1.")
        return (True, f"Rate ratio {rate_ratio:.4f} > 1, monotonicity holds.")

    # ------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, list]:
        """
        Discover all bound test_* methods, execute them in a try/except,
        collect logs, and return (all_pass, logs_list).
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