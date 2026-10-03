import math
import traceback
from typing import Any, Dict, Tuple, List, Optional

class PhysicsChecker:
    """
    Audits a solution for the π⁺ → μ⁺ + ν decay problem.
    Checks the computed T_max meets kinematic constraints and derived formulas.
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract all needed quantities from the solution instance.
        Uses safe .get() or getattr with None defaults. Never assigns numeric
        fallback values.
        """
        # Results emitted by the solver (expected key: 'T_max')
        self.T_max: Optional[float] = getattr(solution_instance, 'results', {}).get('T_max', None)

        # Given masses (MeV/c²) – usually present in solution_instance.givens
        self.m_pi: Optional[float] = None
        self.m_mu: Optional[float] = None
        self.m_nu: Optional[float] = 0.0  # neutrino mass is exactly zero in problem

        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.m_pi = solution_instance.givens.get('m_pi', None)
            self.m_mu = solution_instance.givens.get('m_mu', None)
        else:
            # Some solvers store givens as direct attributes
            self.m_pi = getattr(solution_instance, 'm_pi', None)
            self.m_mu = getattr(solution_instance, 'm_mu', None)

        # Additional derived quantities (to be computed in tests if needed)
        # We'll compute them inside the relevant test methods to keep init minimal.

    def __compute_beta_mu_star(self) -> Optional[float]:
        """Return β_μ* if masses are available, else None."""
        if self.m_pi is None or self.m_mu is None:
            return None
        if self.m_pi <= 0 or self.m_mu <= 0:
            return None
        return (self.m_pi**2 - self.m_mu**2) / (self.m_pi**2 + self.m_mu**2)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_has_T_max(self) -> Tuple[bool, str]:
        """Check that the result key 'T_max' exists and is a number."""
        if self.T_max is None:
            return (False, "Missing T_max in results")
        if not isinstance(self.T_max, (int, float)):
            return (False, f"T_max has wrong type: {type(self.T_max)}")
        if math.isnan(self.T_max) or math.isinf(self.T_max):
            return (False, f"T_max is NaN or infinite: {self.T_max}")
        return (True, f"T_max = {self.T_max}")

    def test_interface_masses_available(self) -> Tuple[bool, str]:
        """Check that the required masses are present."""
        if self.m_pi is None:
            return (False, "Missing m_pi")
        if self.m_mu is None:
            return (False, "Missing m_mu")
        if self.m_pi <= 0 or self.m_mu <= 0:
            return (False, "Masses must be positive")
        return (True, f"m_pi = {self.m_pi}, m_mu = {self.m_mu}")

    # ------------------------------------------------------------------
    # Invariant / theorem-based sanity checks
    # ------------------------------------------------------------------
    def test_invariant_beta_mu_star_le_one(self) -> Tuple[bool, str]:
        """β_μ* ≤ 1 (theoretical bound from two-body decay)."""
        beta = self.__compute_beta_mu_star()
        if beta is None:
            return (False, "SKIP: cannot compute beta_mu_star")
        if beta > 1.0 + 1e-12:
            return (False, f"β_μ* = {beta} > 1, violates special relativity")
        return (True, f"β_μ* = {beta} ≤ 1")

    def test_invariant_beta_mu_star_positive(self) -> Tuple[bool, str]:
        """β_μ* must be positive because m_π > m_μ."""
        beta = self.__compute_beta_mu_star()
        if beta is None:
            return (False, "SKIP: cannot compute beta_mu_star")
        if beta <= 0:
            return (False, f"β_μ* = {beta} should be > 0 since m_π > m_μ")
        return (True, f"β_μ* = {beta} > 0")

    def test_invariant_T_max_nonnegative(self) -> Tuple[bool, str]:
        """T_max must be ≥ 0 (kinetic energy is non‑negative)."""
        if self.T_max is None:
            return (False, "SKIP: T_max not available")
        if self.T_max < 0:
            return (False, f"T_max = {self.T_max} is negative")
        return (True, f"T_max = {self.T_max} ≥ 0")

    # ------------------------------------------------------------------
    # Modeling / trace consistency checks (derived from explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_T_max_formula_numerical(self) -> Tuple[bool, str]:
        """
        Check T_max agrees with the closed‑form (m_pi - m_mu)^2/(2*m_mu).
        This is an exact derivation, so use tight tolerance.
        """
        if self.T_max is None or self.m_pi is None or self.m_mu is None:
            return (False, "SKIP: missing data for formula check")
        if self.m_mu <= 0:
            return (False, "m_mu must be positive")
        expected = (self.m_pi - self.m_mu)**2 / (2.0 * self.m_mu)
        if not math.isclose(self.T_max, expected, rel_tol=1e-5, abs_tol=1e-4):
            return (False, f"T_max computed = {self.T_max}, expected = {expected}")
        return (True, f"T_max = {self.T_max} matches formula (m_pi - m_mu)^2/(2*m_mu)")

    def test_trace_gamma_pi_consistency(self) -> Tuple[bool, str]:
        """
        Verify internal consistency: γ_pi = 1/√(1-β_μ*²) = (m_pi²+m_mu²)/(2 m_pi m_mu).
        Then T_max should equal m_pi*(γ_pi - 1).
        """
        if self.T_max is None or self.m_pi is None or self.m_mu is None:
            return (False, "SKIP: missing data for gamma check")
        beta = self.__compute_beta_mu_star()
        if beta is None:
            return (False, "SKIP: cannot compute beta_mu_star")
        # gamma_pi from beta_mu_star
        gamma_from_beta = 1.0 / math.sqrt(1.0 - beta**2)
        gamma_from_masses = (self.m_pi**2 + self.m_mu**2) / (2.0 * self.m_pi * self.m_mu)
        if not math.isclose(gamma_from_beta, gamma_from_masses, rel_tol=1e-5, abs_tol=1e-4):
            return (False, f"γ_pi inconsistent: from β = {gamma_from_beta}, from masses = {gamma_from_masses}")
        # now compute T_max from gamma
        T_max_computed = self.m_pi * (gamma_from_beta - 1.0)
        if not math.isclose(self.T_max, T_max_computed, rel_tol=1e-5, abs_tol=1e-4):
            return (False, f"T_max from gamma = {T_max_computed}, differs from given {self.T_max}")
        return (True, f"γ_pi = {gamma_from_beta} consistent, T_max derived matches")

    # ------------------------------------------------------------------
    # Monotonicity test (perturb m_mu)
    # ------------------------------------------------------------------
    def test_monotonicity_T_max_with_mmup(self) -> Tuple[bool, str]:
        """
        T_max should decrease when m_mu increases (for fixed m_pi).
        Perturb m_mu by +1% and verify the new computed T_max is smaller.
        """
        if self.m_pi is None or self.m_mu is None or self.T_max is None:
            return (False, "SKIP: missing masses or T_max")
        original_mu = self.m_mu
        original_T = self.T_max
        new_mu = original_mu * 1.01  # +1%
        new_T = (self.m_pi - new_mu)**2 / (2.0 * new_mu)
        if new_T >= original_T:
            return (False, f"Monotonicity broken: increasing m_mu by 1% gave T_max = {new_T}, original = {original_T}")
        return (True, f"T_max decreases when m_mu increases: {original_T} → {new_T}")

    # ------------------------------------------------------------------
    # Weak answer sanity check (only one, and very broad)
    # ------------------------------------------------------------------
    def test_answer_sanity_plausible_range(self) -> Tuple[bool, str]:
        """Weak check: T_max should be between 0 and 10 MeV (typical scale)."""
        if self.T_max is None:
            return (False, "SKIP: T_max not available")
        if not (0 <= self.T_max <= 10):
            return (False, f"T_max = {self.T_max} outside plausible range 0–10 MeV")
        return (True, f"T_max = {self.T_max} is within plausible range")

    # ------------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run every bound method with 'test_' prefix."""
        logs: List[str] = []
        all_pass: bool = True
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