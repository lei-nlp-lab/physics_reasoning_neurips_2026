import math
import traceback
from typing import Any, List, Tuple

class PhysicsChecker:
    """
    Audits a solution instance for the given problem:
      (a) Mean time between collisions for N₂ in air at room T and 1 atm
      (b) Number density of degenerate Fermi electron gas at T=0 with p_F = m_e c
    """

    # Physical constants (SI units)
    k_B = 1.380649e-23          # Boltzmann constant (J/K)
    atm_Pa = 1.013e5            # 1 atm in Pa
    T_room = 300.0              # room temperature (K)
    d_N2 = 3.7e-10              # nitrogen molecular diameter (m)
    m_N2 = 28.0 * 1.660539e-27  # mass of N₂ molecule (kg)
    m_e = 9.109e-31             # electron mass (kg)
    c = 2.998e8                 # speed of light (m/s)
    hbar = 1.054571817e-34      # reduced Planck constant (J·s)

    def __init__(self, solution_instance: Any):
        """Extract data safely; use None for missing fields."""
        # Expected result keys from solution_instance.results
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.tau = self.results.get('tau', None)
        self.n = self.results.get('n', None)

        # Optional attributes for richer analysis
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.givens = getattr(solution_instance, 'givens', None)

    # ------------------------------------------------------------------
    # Interface checks – ensure required outputs exist and are valid
    # ------------------------------------------------------------------
    def test_interface_results_keys_exist(self) -> Tuple[bool, str]:
        """Check that 'tau' and 'n' keys are present in results."""
        if self.tau is None and 'tau' not in (self.results or {}):
            return False, "'tau' missing from results"
        if self.n is None and 'n' not in (self.results or {}):
            return False, "'n' missing from results"
        return True, "keys 'tau' and 'n' exist in results"

    def test_interface_tau_positive_finite(self) -> Tuple[bool, str]:
        """tau must be a positive finite number."""
        if self.tau is None:
            return False, "SKIP: tau not available"
        if not isinstance(self.tau, (int, float)):
            return False, f"tau is not numeric: {type(self.tau).__name__}"
        if not math.isfinite(self.tau):
            return False, f"tau is not finite: {self.tau}"
        if self.tau <= 0:
            return False, f"tau not positive: {self.tau}"
        return True, f"tau = {self.tau:.3e} s is positive and finite"

    def test_interface_n_positive_finite(self) -> Tuple[bool, str]:
        """n must be a positive finite number."""
        if self.n is None:
            return False, "SKIP: n not available"
        if not isinstance(self.n, (int, float)):
            return False, f"n is not numeric: {type(self.n).__name__}"
        if not math.isfinite(self.n):
            return False, f"n is not finite: {self.n}"
        if self.n <= 0:
            return False, f"n not positive: {self.n}"
        return True, f"n = {self.n:.3e} m⁻³ is positive and finite"

    # ------------------------------------------------------------------
    # Invariant tests – physical law consistency (use isclose with tol)
    # ------------------------------------------------------------------
    def test_invariant_tau_from_kinetic_theory(self) -> Tuple[bool, str]:
        """
        Check that τ = λ / v̄ within 1% using the ideal gas law and hard‑sphere model.
        Uses the average molecular speed (v̄ = √(8k_B T / (π m))).
        """
        if self.tau is None:
            return False, "SKIP: tau not available"

        # Compute expected tau from standard constants
        P = self.atm_Pa
        T = self.T_room
        d = self.d_N2
        m = self.m_N2

        n_air = P / (self.k_B * T)                     # ideal gas number density
        lam = 1.0 / (math.sqrt(2) * math.pi * d**2 * n_air)  # mean free path
        vbar = math.sqrt(8.0 * self.k_B * T / (math.pi * m))  # average speed
        tau_expected = lam / vbar

        if math.isclose(self.tau, tau_expected, rel_tol=1e-2, abs_tol=1e-15):
            return True, f"tau ({self.tau:.3e}) consistent with expected {tau_expected:.3e} (within 1%)"
        else:
            return False, f"tau ({self.tau:.3e}) deviates from expected {tau_expected:.3e} (rel diff={abs(self.tau/tau_expected-1):.2%})"

    def test_invariant_n_from_fermi_degeneracy(self) -> Tuple[bool, str]:
        """
        Check that n = p_F³/(3π² ħ³) where p_F = m_e c.
        """
        if self.n is None:
            return False, "SKIP: n not available"

        pF = self.m_e * self.c
        n_expected = pF**3 / (3.0 * math.pi**2 * self.hbar**3)

        if math.isclose(self.n, n_expected, rel_tol=1e-2, abs_tol=1e-10):
            return True, f"n ({self.n:.3e}) consistent with expected {n_expected:.3e} (within 1%)"
        else:
            return False, f"n ({self.n:.3e}) deviates from expected {n_expected:.3e} (rel diff={abs(self.n/n_expected-1):.2%})"

    # ------------------------------------------------------------------
    # Modeling / trace tests – verify that the explanation steps contain
    # the necessary physical relations (qualitative check)
    # ------------------------------------------------------------------
    def test_modeling_air_number_density_relation(self) -> Tuple[bool, str]:
        """
        Check that explanation steps mention the ideal gas law (P = n k_B T)
        or the inverse relation for number density.
        """
        if not self.explanation_steps:
            return False, "SKIP: no explanation_steps provided"

        steps_text = str(self.explanation_steps).lower()
        keywords = ["ideal gas", "n = p/(k t)", "n = p/(k_b t)", "n_air"]
        if any(kw in steps_text for kw in keywords):
            return True, "Ideal gas law for air density found in explanation"
        else:
            return False, "No mention of ideal gas law for air density in explanation"

    def test_modeling_fermi_number_density_relation(self) -> Tuple[bool, str]:
        """
        Check that explanation steps mention n = p_F³/(3π² ħ³) or equivalent.
        """
        if not self.explanation_steps:
            return False, "SKIP: no explanation_steps provided"

        steps_text = str(self.explanation_steps).lower()
        keywords = ["p_f^3/(3π^2 ħ^3)", "p_f^3/(3π^2 hbar^3)", "8π p_f^3/(3 h^3)",
                     "degenerate", "fermi", "n = 2 * ∫"]
        if any(kw in steps_text for kw in keywords):
            return True, "Fermi degeneracy formula for n found in explanation"
        else:
            return False, "No mention of Fermi degeneracy formula in explanation"

    # ------------------------------------------------------------------
    # Monotonicity test – perturb an input and check monotonic trend
    # ------------------------------------------------------------------
    def test_monotonicity_tau_increases_with_temperature(self) -> Tuple[bool, str]:
        """
        Verify that the mean collision time τ increases when temperature increases,
        at constant pressure.
        Baseline T = 300 K, perturbed T = 310 K.
        τ ∝ sqrt(T) / P, so higher T → larger τ.
        """
        # Compute baseline tau using the same model (independent of self.tau)
        P = self.atm_Pa
        T_base = self.T_room
        T_pert = T_base + 10.0   # increase by 10 K
        d = self.d_N2
        m = self.m_N2

        def compute_tau(T):
            n_air = P / (self.k_B * T)
            lam = 1.0 / (math.sqrt(2) * math.pi * d**2 * n_air)
            vbar = math.sqrt(8.0 * self.k_B * T / (math.pi * m))
            return lam / vbar

        tau_base = compute_tau(T_base)
        tau_pert = compute_tau(T_pert)

        if tau_pert > tau_base:
            return True, (f"tau increases from {tau_base:.3e} s (T={T_base} K) "
                          f"to {tau_pert:.3e} s (T={T_pert} K) as expected")
        else:
            return False, (f"tau should increase with T, but got tau_base={tau_base:.3e}, "
                           f"tau_pert={tau_pert:.3e}")

    # ------------------------------------------------------------------
    # Answer sanity check (weak – only order of magnitude and sign)
    # ------------------------------------------------------------------
    def test_answer_sanity_magnitude_order(self) -> Tuple[bool, str]:
        """
        Weak sanity: tau should be about 10⁻⁹ s and n about 10³⁵ m⁻³.
        Only checks order‑of‑magnitude (within a factor of 10).
        """
        messages = []
        if self.tau is not None:
            if 1e-10 <= self.tau <= 1e-8:
                messages.append(f"tau ≈ {self.tau:.1e} (expected ~4e-9)")
            else:
                return False, f"tau = {self.tau:.1e} out of plausible range [1e-10,1e-8]"
        if self.n is not None:
            if 1e35 <= self.n <= 1e36:
                messages.append(f"n ≈ {self.n:.1e} (expected ~6e35)")
            else:
                return False, f"n = {self.n:.1e} out of plausible range [1e35,1e36]"
        if not messages:
            return False, "Neither tau nor n available for sanity check"
        return True, "; ".join(messages)

    # ------------------------------------------------------------------
    # Main harness: discover and run all test_* methods
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs = []
        all_pass = True
        for attr_name in sorted(dir(self)):
            if not attr_name.startswith("test_"):
                continue
            method = getattr(self, attr_name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{attr_name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {attr_name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs