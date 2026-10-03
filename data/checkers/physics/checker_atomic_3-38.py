import math
import traceback
from typing import Any, Dict, Optional, Tuple, List

class PhysicsChecker:
    """
    Auditor for the tau lifetime and decay length problem.
    Checks interface, invariant scaling, relativistic kinematics,
    modeling consistency, monotonicity, and a weak answer sanity.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract all needed quantities safely (default None)
        self.results = getattr(solution_instance, 'results', None)
        self.givens = getattr(solution_instance, 'givens', None)

        # Required result keys (from emit contract)
        self.tau_tau = None
        self.L = None
        if self.results is not None:
            self.tau_tau = self.results.get('tau_tau', None)
            self.L = self.results.get('L', None)

        # Given parameters (with defaults None for missing)
        self.tau_mu = None
        self.m_mu = None
        self.m_tau = None
        self.BR = None
        self.E_cm = None
        self.c = 3.0e10  # cm/s, constant
        if self.givens is not None:
            self.tau_mu = self.givens.get('tau_mu', None)
            self.m_mu = self.givens.get('m_mu', None)
            self.m_tau = self.givens.get('m_tau', None)
            self.BR = self.givens.get('BR', None)
            self.E_cm = self.givens.get('E_cm', None)

    # ------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------
    def test_interface_required_results_exist(self) -> Tuple[bool, str]:
        """Check that the required result keys 'tau_tau' and 'L' are present."""
        if self.results is None:
            return (False, "SKIP: results dict missing")
        missing = []
        if 'tau_tau' not in self.results:
            missing.append('tau_tau')
        if 'L' not in self.results:
            missing.append('L')
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required result keys present")

    def test_interface_results_dtype(self) -> Tuple[bool, str]:
        """Check that result values are numeric (float or int)."""
        for key in ['tau_tau', 'L']:
            val = self.results.get(key, None) if self.results else None
            if val is None:
                return (False, f"SKIP: {key} is None")
            if not isinstance(val, (int, float)):
                return (False, f"{key} is not numeric, got {type(val).__name__}")
        return (True, "Result values are numeric")

    # ------------------------------------------------------------
    # Invariant tests (theorem-based, use math.isclose)
    # ------------------------------------------------------------
    def test_invariant_decay_width_scaling(self) -> Tuple[bool, str]:
        """Check tau_tau = tau_mu * BR * (m_mu/m_tau)^5."""
        if self.tau_tau is None or self.tau_mu is None or self.m_mu is None or self.m_tau is None or self.BR is None:
            return (False, "SKIP: missing givens or tau_tau")
        try:
            expected_tau_tau = self.tau_mu * self.BR * (self.m_mu / self.m_tau) ** 5
        except (ZeroDivisionError, TypeError) as e:
            return (False, f"Computation error: {e}")
        if math.isclose(self.tau_tau, expected_tau_tau, rel_tol=1e-2, abs_tol=1e-20):
            return (True, f"tau_tau = {self.tau_tau:.3e} matches scaling {expected_tau_tau:.3e}")
        else:
            return (False, f"tau_tau = {self.tau_tau:.3e}, expected {expected_tau_tau:.3e} (scaling)")

    def test_invariant_relativistic_length(self) -> Tuple[bool, str]:
        """Check L = beta_gamma * c * tau_tau with beta_gamma from E_tau and m_tau."""
        if self.L is None or self.tau_tau is None or self.E_cm is None or self.m_tau is None:
            return (False, "SKIP: missing L, tau_tau, E_cm, or m_tau")
        try:
            # E_tau = E_cm/2, assume E_cm in MeV (given in GeV but converted to MeV)
            E_tau = self.E_cm / 2.0
            # Convert to MeV if E_cm was given in GeV? We'll assume both in same unit.
            # Problem statement: m_tau = 1784 MeV, E_cm = 29 GeV = 29000 MeV.
            # We must handle unit consistency. If E_cm is stored in GeV, convert.
            # We can check typical magnitude: if E_tau > 10000, assume MeV, else if ~14.5 assume GeV and convert.
            # For safety, we'll accept both and convert if needed.
            m_tau = self.m_tau  # Should be in MeV
            # Determine unit of E_tau: if it's ~14.5, it's likely GeV; if ~14500, it's MeV.
            if E_tau < 100:  # Probably in GeV
                E_tau_MeV = E_tau * 1000.0  # Convert to MeV
            else:
                E_tau_MeV = E_tau
            # Compute beta*gamma = sqrt(E^2 - m^2) / m
            beta_gamma = math.sqrt(E_tau_MeV**2 - m_tau**2) / m_tau
            expected_L = beta_gamma * self.c * self.tau_tau
        except (ZeroDivisionError, ValueError, TypeError) as e:
            return (False, f"Computation error: {e}")
        if math.isclose(self.L, expected_L, rel_tol=1e-2, abs_tol=1e-10):
            return (True, f"L = {self.L:.3e} matches relativistic formula {expected_L:.3e}")
        else:
            return (False, f"L = {self.L:.3e}, expected {expected_L:.3e} (relativistic)")

    # ------------------------------------------------------------
    # Modeling tests (check intermediate values / reasoning)
    # ------------------------------------------------------------
    def test_modeling_energy_split(self) -> Tuple[bool, str]:
        """Check that E_tau equals E_cm/2 (lab frame = CM frame)."""
        if self.E_cm is None:
            return (False, "SKIP: E_cm not given")
        # The model expects E_tau = E_cm/2. Since we don't have E_tau directly from results,
        # we can infer it from the relativistic computation. Alternatively, check consistency
        # with L computation.
        # We'll attempt to compute E_tau from L and tau_tau.
        if self.L is None or self.tau_tau is None:
            return (False, "SKIP: L or tau_tau missing")
        if self.c == 0:
            return (False, "Invalid c")
        try:
            beta_gamma = self.L / (self.c * self.tau_tau)
            # beta_gamma = sqrt(E^2 - m^2)/m => E = m * sqrt(1 + beta_gamma^2)
            E_tau_derived = self.m_tau * math.sqrt(1 + beta_gamma**2)
            # Now check that E_tau_derived is approx E_cm/2.
            # Need consistent units: assume E_cm is in same unit as m_tau (MeV).
            # If E_cm is in GeV, convert to MeV.
            E_cm_MeV = self.E_cm * 1000.0 if self.E_cm < 1000 else self.E_cm  # rough conversion
            expected_E_tau = E_cm_MeV / 2.0
            if math.isclose(E_tau_derived, expected_E_tau, rel_tol=1e-2, abs_tol=1.0):
                return (True, f"E_tau derived = {E_tau_derived:.1f} MeV matches E_cm/2 = {expected_E_tau:.1f} MeV")
            else:
                return (False, f"E_tau derived = {E_tau_derived:.1f} MeV, but expected {expected_E_tau:.1f} MeV from E_cm/2")
        except (ZeroDivisionError, ValueError, TypeError) as e:
            return (False, f"Computation error: {e}")

    def test_modeling_branching_ratio_used(self) -> Tuple[bool, str]:
        """Check that the branching ratio is used correctly in tau_tau formula."""
        if self.tau_tau is None or self.tau_mu is None or self.m_mu is None or self.m_tau is None or self.BR is None:
            return (False, "SKIP: missing data")
        try:
            # Without BR, tau_tau would be larger by factor 1/BR.
            scaling_no_BR = self.tau_mu * (self.m_mu / self.m_tau) ** 5
            # The result should include BR in numerator: tau_tau = scaling_no_BR * BR
            expected_with_BR = scaling_no_BR * self.BR
            if math.isclose(self.tau_tau, expected_with_BR, rel_tol=1e-2, abs_tol=1e-20):
                return (True, "Branching ratio correctly included")
            else:
                return (False, f"tau_tau = {self.tau_tau:.3e} does not match scaling with BR = {expected_with_BR:.3e}")
        except (ZeroDivisionError, TypeError) as e:
            return (False, f"Error: {e}")

    # ------------------------------------------------------------
    # Monotonicity tests (perturb one parameter, check trend)
    # ------------------------------------------------------------
    def test_monotonicity_lifetime_decreases_with_mass(self) -> Tuple[bool, str]:
        """If m_tau increases, tau_tau should decrease (inverse 5th power)."""
        if self.tau_tau is None or self.tau_mu is None or self.m_mu is None or self.m_tau is None or self.BR is None:
            return (False, "SKIP: missing data")
        try:
            # Perturb m_tau upward by 5%
            m_tau_perturbed = self.m_tau * 1.05
            tau_tau_perturbed = self.tau_mu * self.BR * (self.m_mu / m_tau_perturbed) ** 5
            # The perturbed lifetime should be smaller than the original
            if tau_tau_perturbed < self.tau_tau:
                return (True, "Monotonicity holds: increasing m_tau reduces tau_tau")
            else:
                return (False, "Monotonicity violated: increasing m_tau did not reduce tau_tau")
        except (ZeroDivisionError, TypeError) as e:
            return (False, f"Error: {e}")

    def test_monotonicity_length_increases_with_energy(self) -> Tuple[bool, str]:
        """If E_cm increases, L should increase (beta gamma increases)."""
        if self.L is None or self.tau_tau is None or self.E_cm is None or self.m_tau is None:
            return (False, "SKIP: missing data")
        try:
            # Compute baseline L using stored formula
            # Here we recompute baseline to avoid depending on stored L (self.L might be wrong)
            # We'll compute perturbed L assuming tau_tau unchanged (true for same tau_tau)
            # Use E_tau baseline from given E_cm/2
            E_cm_MeV_base = self.E_cm * 1000 if self.E_cm < 100 else self.E_cm
            E_tau_base = E_cm_MeV_base / 2.0
            beta_gamma_base = math.sqrt(E_tau_base**2 - self.m_tau**2) / self.m_tau
            L_base = beta_gamma_base * self.c * self.tau_tau

            # Perturb E_cm upward by 10%
            E_cm_MeV_pert = E_cm_MeV_base * 1.10
            E_tau_pert = E_cm_MeV_pert / 2.0
            beta_gamma_pert = math.sqrt(E_tau_pert**2 - self.m_tau**2) / self.m_tau
            L_pert = beta_gamma_pert * self.c * self.tau_tau

            if L_pert > L_base:
                return (True, f"Monotonicity holds: L increases from {L_base:.3e} to {L_pert:.3e} when E_cm increases by 10%")
            else:
                return (False, f"Monotonicity violated: L did not increase (base {L_base:.3e}, perturb {L_pert:.3e})")
        except (ZeroDivisionError, ValueError, TypeError) as e:
            return (False, f"Error: {e}")

    # ------------------------------------------------------------
    # Single answer sanity (weak, no math.isclose, no derivation)
    # ------------------------------------------------------------
    def test_answer_sanity_magnitudes(self) -> Tuple[bool, str]:
        """Check that tau_tau and L have physically plausible magnitudes."""
        if self.tau_tau is None or self.L is None:
            return (False, "SKIP: missing tau_tau or L")
        # tau_tau should be extremely small (much less than muon lifetime)
        if not (0.0 < self.tau_tau < 1e-10):
            return (False, f"tau_tau = {self.tau_tau:.3e} is not in plausible range (0, 1e-10 s)")
        # L should be small, sub-cm (given ~0.06 cm)
        if not (0.0 < self.L < 1.0):
            return (False, f"L = {self.L:.3e} is not in plausible range (0, 1 cm)")
        return (True, "Answer magnitudes physically plausible")

    # ------------------------------------------------------------
    # Batch runner
    # ------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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