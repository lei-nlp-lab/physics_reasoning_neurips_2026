import math
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Audits solutions to the proton spin-flip problem in a rotating magnetic field.
    Checks resonance condition, Rabi oscillation probability, and effective Rabi frequency.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract input parameters safely (default None)
        self.mu: Optional[float] = getattr(solution_instance, 'mu', None)
        self.B0: Optional[float] = getattr(solution_instance, 'B0', None)
        self.Bz: Optional[float] = getattr(solution_instance, 'Bz', None)
        self.hbar: Optional[float] = getattr(solution_instance, 'hbar', None)
        self.omega: Optional[float] = getattr(solution_instance, 'omega', None)   # driving frequency (given)
        self.t: Optional[float] = getattr(solution_instance, 't', None)

        # Extract results dict (if present) or assume results stored as attributes
        results: Optional[Dict[str, Any]] = getattr(solution_instance, 'results', None)
        if results is not None:
            self.res_omega: Optional[float] = results.get('omega', None)
            self.res_P: Optional[float] = results.get('P', None)
            self.res_Omega: Optional[float] = results.get('Omega', None)
        else:
            self.res_omega: Optional[float] = getattr(solution_instance, 'omega', None)
            self.res_P: Optional[float] = getattr(solution_instance, 'P', None)
            self.res_Omega: Optional[float] = getattr(solution_instance, 'Omega', None)

        # Helper: all numerical givens present?
        self._have_givens: bool = all(x is not None for x in [self.mu, self.B0, self.Bz, self.hbar])

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs: List[str] = []
        all_pass: bool = True
        # Discover all test_ methods
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
        return all_pass, logs

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_results_keys(self) -> Tuple[bool, str]:
        if self.res_omega is None and self.res_P is None and self.res_Omega is None:
            # Possibly no results at all – skip gracefully
            return (False, "SKIP: No results (omega, P, Omega) found")
        # Check each key individually
        if self.res_omega is None:
            return (False, "Missing result key: omega")
        if self.res_P is None:
            return (False, "Missing result key: P")
        if self.res_Omega is None:
            return (False, "Missing result key: Omega")
        return (True, "All required result keys (omega, P, Omega) present")

    def test_interface_givens_types(self) -> Tuple[bool, str]:
        if not self._have_givens:
            return (False, "SKIP: Not all given quantities (mu, B0, Bz, hbar) are available")
        for name, val in [("mu", self.mu), ("B0", self.B0), ("Bz", self.Bz), ("hbar", self.hbar)]:
            if not isinstance(val, (int, float)):
                return (False, f"Given '{name}' is not numeric: {type(val)}")
        return (True, "All givens have numeric types")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_resonance_condition(self) -> Tuple[bool, str]:
        if self.res_omega is None or not self._have_givens:
            return (False, "SKIP: Missing resonance result or givens")
        # ω_res = 2 μ Bz / ħ
        expected_omega = 2.0 * self.mu * self.Bz / self.hbar
        if math.isclose(self.res_omega, expected_omega, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Resonance omega = {self.res_omega} matches expected {expected_omega}")
        else:
            return (False, f"Resonance omega = {self.res_omega}, expected {expected_omega}")

    def test_invariant_probability_normalization(self) -> Tuple[bool, str]:
        if self.res_P is None:
            return (False, "SKIP: No probability result")
        if not (0.0 <= self.res_P <= 1.0):
            return (False, f"Probability out of [0,1]: P = {self.res_P}")
        return (True, f"Probability P = {self.res_P} in [0,1]")

    def test_invariant_probability_zero_at_t0(self) -> Tuple[bool, str]:
        if self.res_P is None:
            return (False, "SKIP: No probability result")
        if self.t is None or self.t != 0.0:
            # Only check when t=0 exactly
            return (False, "SKIP: t is not 0")
        if math.isclose(self.res_P, 0.0, abs_tol=1e-6):
            return (True, "P(t=0) = 0 as expected")
        else:
            return (False, f"P(t=0) = {self.res_P}, expected 0")

    def test_invariant_rabi_frequency_approximation_validity(self) -> Tuple[bool, str]:
        # The effective Rabi frequency Omega is an approximation valid only if B0 << Bz.
        if not self._have_givens or self.res_Omega is None:
            return (False, "SKIP: Missing givens or Omega")
        if self.B0 == 0 or self.Bz == 0:
            return (False, "SKIP: B0 or Bz zero")
        ratio = self.B0 / self.Bz
        if ratio >= 0.2:
            # Warn but not fail: the approximation may be poor
            return (True, f"Warning: B0/Bz = {ratio:.2f} not <<1; approximation may be invalid")
        return (True, "B0 << Bz condition holds for the approximation")

    # ------------------------------------------------------------------
    # Modelling / trace tests (consistency of derived results)
    # ------------------------------------------------------------------
    def test_modeling_probability_form_consistency(self) -> Tuple[bool, str]:
        """
        Check that P matches the Rabi formula P = (μ B0 / (ħ Ω))^2 sin^2(Ω t)
        when all values are present.
        """
        required = [self.mu, self.B0, self.hbar, self.res_Omega, self.res_P, self.t]
        if any(x is None for x in required):
            return (False, "SKIP: Missing required values for modelling test")
        # Compute expected P
        factor = (self.mu * self.B0) / (self.hbar * self.res_Omega)
        if factor is None:
            return (False, "Division resulted in None")
        expected_P = factor**2 * math.sin(self.res_Omega * self.t)**2
        if math.isclose(self.res_P, expected_P, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Probability P matches Rabi formula: P = {self.res_P}, expected {expected_P}")
        else:
            return (False, f"Probability P = {self.res_P} does not match Rabi formula (expected {expected_P})")

    def test_modeling_omega_vs_Omega_consistency(self) -> Tuple[bool, str]:
        """
        Check that Omega = (ħ ω / 2 + μ Bz) / ħ (the approximate formula from pseudocode).
        """
        if self.res_Omega is None or self.res_omega is None or not self._have_givens:
            return (False, "SKIP: Missing Omega, omega, or givens")
        expected_Omega = (self.hbar * self.res_omega / 2.0 + self.mu * self.Bz) / self.hbar
        if math.isclose(self.res_Omega, expected_Omega, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Omega = {self.res_Omega} matches (ħω/2 + μBz)/ħ = {expected_Omega}")
        else:
            return (False, f"Omega = {self.res_Omega}, expected {expected_Omega} from formula")

    # ------------------------------------------------------------------
    # Monotonicity tests (trend via parameter perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_probability_increases_with_B0(self) -> Tuple[bool, str]:
        """
        Increase B0 by 5%, keep all other parameters fixed, and check that
        the new probability (computed via the same formula) is larger than the reported one.
        """
        required = [self.mu, self.B0, self.hbar, self.res_Omega, self.res_P, self.t]
        if any(x is None for x in required):
            return (False, "SKIP: Missing values for monotonicity test on B0")
        # Original probability from formula
        orig_factor = (self.mu * self.B0) / (self.hbar * self.res_Omega)
        orig_P = orig_factor**2 * math.sin(self.res_Omega * self.t)**2
        # Perturb B0 upward by 5%
        B0_new = 1.05 * self.B0
        new_factor = (self.mu * B0_new) / (self.hbar * self.res_Omega)
        new_P = new_factor**2 * math.sin(self.res_Omega * self.t)**2
        # Check monotonicity (increase)
        if new_P > orig_P - 1e-9:   # allow tiny numerical noise
            return (True, f"Increasing B0 by 5% increases P from {orig_P:.4f} to {new_P:.4f}")
        else:
            return (False, f"P did not increase when B0 increased: orig {orig_P}, new {new_P}")

    def test_monotonicity_probability_periodic_with_t(self) -> Tuple[bool, str]:
        """
        Check that P(t + π/Ω) is close to P(t) (i.e., periodicity π/Ω in sin²).
        """
        required = [self.res_P, self.res_Omega, self.t]
        if any(x is None for x in required) or self.res_Omega == 0:
            return (False, "SKIP: Missing values for periodicity test")
        # Compute P at t + half period
        t_new = self.t + math.pi / self.res_Omega
        # We need to recompute P using the model (same formula)
        if not self._have_givens or self.mu is None or self.B0 is None or self.hbar is None:
            return (False, "SKIP: Incomplete givens to recompute P")
        factor = (self.mu * self.B0) / (self.hbar * self.res_Omega)
        new_P = factor**2 * math.sin(self.res_Omega * t_new)**2
        if math.isclose(self.res_P, new_P, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"P(t) ≈ P(t+π/Ω): P={self.res_P:.4f}, P(t+π/Ω)={new_P:.4f}")
        else:
            return (False, f"P not periodic: P(t)={self.res_P}, P(t+π/Ω)={new_P}")

    # ------------------------------------------------------------------
    # Answer sanity (weak and secondary)
    # ------------------------------------------------------------------
    def test_answer_sanity_broad_magnitude(self) -> Tuple[bool, str]:
        """
        Weak check: Probability should be non-negative and not huge.
        Omega should be positive.
        Resonance frequency should be positive.
        """
        if self.res_P is not None:
            if self.res_P < 0.0 or self.res_P > 1.0:
                return (False, f"P out of [0,1]: {self.res_P}")
        if self.res_Omega is not None and self.res_Omega <= 0.0:
            return (False, f"Omega is not positive: {self.res_Omega}")
        if self.res_omega is not None and self.res_omega <= 0.0:
            return (False, f"Resonance omega not positive: {self.res_omega}")
        return (True, "Basic sanity signs and range pass")