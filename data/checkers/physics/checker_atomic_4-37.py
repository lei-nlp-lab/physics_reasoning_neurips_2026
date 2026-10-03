import math
import traceback
from typing import Any, Dict, Tuple, List, Optional

class PhysicsChecker:
    """
    PhysicsChecker for Compton back-scattering problem.
    Checks interface, invariants, modeling quality, monotonicity and a weak answer sanity.
    """

    def __init__(self, solution_instance: Any):
        """
        Extract all required data from solution_instance.
        Use None for missing values – never numeric defaults.
        """
        # Try to get results dict; if absent, fallback to attributes.
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.results = solution_instance.results
        else:
            self.results = {}

        # Givens (from problem context)
        self.hnu = getattr(solution_instance, 'hnu', self.results.get('hnu', None))
        self.T_e = getattr(solution_instance, 'T_e', self.results.get('T_e', None))
        self.m_e_c2 = getattr(solution_instance, 'm_e_c2', self.results.get('m_e_c2', None))

        # Required output
        self.hnu_prime = self.results.get('hnu_prime', None)

        # Derived quantities (computed inside checker if givens present)
        self.E_e = None
        self.p_e_c = None
        if self.T_e is not None and self.m_e_c2 is not None:
            self.E_e = self.T_e + self.m_e_c2
            # Compute p_e*c = sqrt(E_e^2 - m_e_c2^2)
            try:
                self.p_e_c = math.sqrt(self.E_e**2 - self.m_e_c2**2)
            except (ValueError, TypeError):
                self.p_e_c = None  # if no real root (should not happen)

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that the results dict contains the required key."""
        if self.hnu_prime is None:
            return False, "Missing key 'hnu_prime' in results."
        return True, "Key 'hnu_prime' exists."

    def test_interface_givens_types(self) -> Tuple[bool, str]:
        """Check that givens are numeric and non-negative."""
        for name, val in [('hnu', self.hnu), ('T_e', self.T_e), ('m_e_c2', self.m_e_c2)]:
            if val is None:
                return False, f"Given '{name}' is missing."
            if not isinstance(val, (int, float)):
                return False, f"Given '{name}' is not numeric (type {type(val)})."
            if val < 0:
                return False, f"Given '{name}' is negative."
        return True, "All givens are present, numeric and non-negative."

    def test_interface_output_type(self) -> Tuple[bool, str]:
        """Check that hnu_prime is numeric and finite."""
        if self.hnu_prime is None:
            return False, "hnu_prime is missing."
        if not isinstance(self.hnu_prime, (int, float)):
            return False, f"hnu_prime is not numeric (type {type(self.hnu_prime)})."
        if not math.isfinite(self.hnu_prime):
            return False, "hnu_prime is not finite (NaN/Inf)."
        return True, "hnu_prime is numeric and finite."

    # ----------------------------------------------------------------------
    # Invariant tests (theorem-based)
    # ----------------------------------------------------------------------
    def test_invariant_relativistic_electron(self) -> Tuple[bool, str]:
        """
        Check that the given electron energy and momentum satisfy the on‑shell condition:
        E_e^2 - p_e^2 c^2 = m_e^2 c^4   (with c=1).
        """
        if self.E_e is None or self.p_e_c is None or self.m_e_c2 is None:
            return False, "Cannot compute electron invariant: missing E_e, p_e_c or m_e_c2."
        try:
            lhs = self.E_e**2 - self.p_e_c**2
            rhs = self.m_e_c2**2
        except (TypeError, ValueError):
            return False, "Error computing invariant expression."
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"Electron invariant violated: E_e^2 - p_e^2 = {lhs:.6f}, expected {rhs:.6f}."
        return True, "Electron on‑shell condition holds (within 1% tolerance)."

    def test_invariant_energy_conservation_derived_formula(self) -> Tuple[bool, str]:
        """
        Re‑derive the back‑scattered photon energy from the theoretical formula
        that follows from energy‑momentum conservation, and compare with the result.
        """
        if any(v is None for v in [self.hnu, self.E_e, self.p_e_c, self.hnu_prime]):
            return False, "Missing data for energy conservation check."
        try:
            denominator = self.E_e - self.p_e_c + 2 * self.hnu
            if denominator == 0:
                return False, "Denominator is zero – formula undefined."
            hnu_prime_derived = self.hnu * (self.E_e + self.p_e_c) / denominator
        except (TypeError, ZeroDivisionError):
            return False, "Error computing derived hnu_prime."

        if not math.isclose(hnu_prime_derived, self.hnu_prime, rel_tol=1e-2, abs_tol=1e-2):
            return False, (f"Derived hnu_prime = {hnu_prime_derived:.6f} MeV, "
                           f"but result = {self.hnu_prime:.6f} MeV (difference > 1%).")
        return True, "Result matches the energy‑momentum conservation derived formula (within 1%)."

    # ----------------------------------------------------------------------
    # Modeling quality checks
    # ----------------------------------------------------------------------
    def test_modeling_ultrarelativistic_approximation(self) -> Tuple[bool, str]:
        """
        In the ultra‑relativistic limit (E_e >> m_e_c2), we have:
          E_e - p_e_c ≈ m_e_c2^2 / (2*E_e)   and   E_e + p_e_c ≈ 2*E_e.
        Check that these approximations are reasonably accurate (within 5% relative error).
        """
        if self.E_e is None or self.p_e_c is None or self.m_e_c2 is None:
            return False, "Cannot compute approximations: missing data."
        if self.E_e == 0:
            return False, "E_e is zero – cannot evaluate approximation."
        # Exact differences
        exact_diff = self.E_e - self.p_e_c
        approx_diff = self.m_e_c2**2 / (2 * self.E_e) if self.E_e != 0 else 0
        exact_sum = self.E_e + self.p_e_c
        approx_sum = 2 * self.E_e

        success = True
        msgs = []
        # Check difference approximation
        if not math.isclose(exact_diff, approx_diff, rel_tol=5e-2, abs_tol=1e-2):
            success = False
            msgs.append(f"E_e-p_e approx off: exact {exact_diff:.6f}, approx {approx_diff:.6f} (>5%).")
        # Check sum approximation
        if not math.isclose(exact_sum, approx_sum, rel_tol=5e-2, abs_tol=1e-2):
            success = False
            msgs.append(f"E_e+p_e approx off: exact {exact_sum:.6f}, approx {approx_sum:.6f} (>5%).")
        if not success:
            return False, "; ".join(msgs)
        return True, "Ultra‑relativistic approximations hold (within 5%)."

    def test_modeling_positive_energy(self) -> Tuple[bool, str]:
        """Check that the scattered photon energy is positive."""
        if self.hnu_prime is None:
            return False, "hnu_prime missing."
        if not (self.hnu_prime > 0):
            return False, f"hnu_prime is not positive: {self.hnu_prime}."
        return True, "hnu_prime > 0."

    # ----------------------------------------------------------------------
    # Monotonicity tests
    # ----------------------------------------------------------------------
    def test_monotonicity_scattered_with_incident_energy(self) -> Tuple[bool, str]:
        """
        Perturb the incident photon energy (hnu) by +10% and verify that
        the back‑scattered photon energy increases.
        We re‑compute using the same theoretical formula.
        """
        if any(v is None for v in [self.hnu, self.E_e, self.p_e_c, self.hnu_prime]):
            return False, "Missing data for monotonicity test."
        # Baseline (already computed as self.hnu_prime)
        hnu_new = self.hnu * 1.1
        denominator = self.E_e - self.p_e_c + 2 * hnu_new
        if denominator == 0:
            return False, "Denominator zero for perturbed hnu."
        try:
            hnu_prime_new = hnu_new * (self.E_e + self.p_e_c) / denominator
        except (TypeError, ZeroDivisionError):
            return False, "Error computing perturbed hnu_prime."

        if hnu_prime_new <= self.hnu_prime:
            return False, (f"Monotonicity violated: with hnu increased 10%, "
                           f"hnu_prime decreased from {self.hnu_prime:.6f} to {hnu_prime_new:.6f}.")
        return True, "hnu_prime increases when incident photon energy increases (monotonic)."

    def test_monotonicity_scattered_with_electron_energy(self) -> Tuple[bool, str]:
        """
        Perturb the electron kinetic energy (T_e) by +10% and verify that
        the back‑scattered photon energy increases.
        Re‑compute E_e, p_e_c, and hnu_prime.
        """
        if any(v is None for v in [self.T_e, self.m_e_c2, self.hnu, self.hnu_prime]):
            return False, "Missing data for monotonicity test."
        T_e_new = self.T_e * 1.1
        E_e_new = T_e_new + self.m_e_c2
        p_e_c_new = math.sqrt(E_e_new**2 - self.m_e_c2**2)
        denominator = E_e_new - p_e_c_new + 2 * self.hnu
        if denominator == 0:
            return False, "Denominator zero for perturbed T_e."
        try:
            hnu_prime_new = self.hnu * (E_e_new + p_e_c_new) / denominator
        except (TypeError, ZeroDivisionError):
            return False, "Error computing perturbed hnu_prime."

        if hnu_prime_new <= self.hnu_prime:
            return False, (f"Monotonicity violated: with T_e increased 10%, "
                           f"hnu_prime decreased from {self.hnu_prime:.6f} to {hnu_prime_new:.6f}.")
        return True, "hnu_prime increases when electron kinetic energy increases (monotonic)."

    # ----------------------------------------------------------------------
    # Answer sanity (weak)
    # ----------------------------------------------------------------------
    def test_answer_sanity_hnu_prime_positive_and_plausible(self) -> Tuple[bool, str]:
        """
        Weak sanity: hnu_prime should be positive, not huge (e.g., < 10^5 MeV),
        and at least larger than the incident photon energy (since upscattering).
        """
        if self.hnu_prime is None or self.hnu is None:
            return False, "Missing hnu_prime or hnu for sanity check."
        if not math.isfinite(self.hnu_prime):
            return False, "hnu_prime is not finite."
        if self.hnu_prime <= 0:
            return False, "hnu_prime is not positive."
        if self.hnu_prime > 1e5:  # 100 TeV – far beyond expected
            return False, f"hnu_prime = {self.hnu_prime:.2f} MeV is implausibly large."
        # Incident photon is 2 eV = 2e-6 MeV; back‑scattered should be >> that
        if self.hnu_prime < self.hnu:
            return False, "hnu_prime is less than incident photon energy – unlikely for inverse Compton."
        return True, "hnu_prime is positive, finite, plausible magnitude (> incident energy, < 100 TeV)."

    # ----------------------------------------------------------------------
    # Test runner
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all bound methods starting with 'test_', return (all_pass, logs)."""
        logs = []
        all_pass = True
        # Get all attributes that are callable and start with "test_"
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed = False
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)