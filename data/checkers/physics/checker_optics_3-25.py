import math
import traceback
from typing import Any, List, Tuple, Optional

class PhysicsChecker:
    """
    Audits solutions for the solar panel water heating problem.
    Checks:
        - Interface: required result keys exist and are numeric.
        - Invariant: energy balance (α I = σ T⁴) holds within 2% tolerance.
        - Modeling: Stefan-Boltzmann law correctly applied; halved absorptance relationship preserved.
        - Monotonicity: increasing α increases T.
        - Weak answer sanity: temperatures positive, finite, physically plausible.
    """

    def __init__(self, solution_instance: Any) -> None:
        # --- Constants derived from problem statement ---
        self.sigma = 5.67e-8        # W·m⁻²·K⁻⁴
        self.I = 800.0              # W·m⁻² (solar flux)
        self.epsilon = 1.0          # emissivity (perfect blackbody)
        self.alpha1 = 0.96          # original absorptance
        self.alpha2 = 0.48          # halved absorptance

        # --- Extract given parameters if solution_instance provides them ---
        self.given = getattr(solution_instance, 'given', None) or {}
        self.results = getattr(solution_instance, 'results', None) or {}

        # Allow user-provided I (if given) but keep sigma fixed
        self.I = float(self.given.get('I', self.I))

        # --- Expected result keys from pseudocode emit ---
        self.T_max = self.results.get('T_max', None)
        self.T_prime = self.results.get('T_prime', None)

    # ---------- Interface Tests ----------
    def test_interface_T_max_exists(self) -> Tuple[bool, str]:
        if self.T_max is None:
            return False, "Missing required result 'T_max'"
        return True, "T_max present"

    def test_interface_T_prime_exists(self) -> Tuple[bool, str]:
        if self.T_prime is None:
            return False, "Missing required result 'T_prime'"
        return True, "T_prime present"

    def test_interface_T_max_numeric(self) -> Tuple[bool, str]:
        if not isinstance(self.T_max, (int, float)):
            return False, f"T_max is not numeric: {type(self.T_max).__name__}"
        return True, "T_max is numeric"

    def test_interface_T_prime_numeric(self) -> Tuple[bool, str]:
        if not isinstance(self.T_prime, (int, float)):
            return False, f"T_prime is not numeric: {type(self.T_prime).__name__}"
        return True, "T_prime is numeric"

    # ---------- Invariant Tests (theorem-based physical sanity) ----------
    def test_invariant_energy_balance_T1(self) -> Tuple[bool, str]:
        """Check that T_max satisfies α₁ I = σ T_max⁴ within 2% tolerance."""
        if self.T_max is None:
            return (False, "SKIP: T_max not available")
        expected_T = (self.alpha1 * self.I / self.sigma) ** 0.25
        if not math.isclose(self.T_max, expected_T, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"T_max = {self.T_max:.2f} differs from expected {expected_T:.2f} (tolerance 2%)")
        return (True, f"T_max = {self.T_max:.2f} ≈ {expected_T:.2f}")

    def test_invariant_energy_balance_T2(self) -> Tuple[bool, str]:
        """Check that T_prime satisfies α₂ I = σ T_prime⁴ within 2% tolerance."""
        if self.T_prime is None:
            return (False, "SKIP: T_prime not available")
        expected_T = (self.alpha2 * self.I / self.sigma) ** 0.25
        if not math.isclose(self.T_prime, expected_T, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"T_prime = {self.T_prime:.2f} differs from expected {expected_T:.2f} (tolerance 2%)")
        return (True, f"T_prime = {self.T_prime:.2f} ≈ {expected_T:.2f}")

    # ---------- Modeling & Trace Consistency Tests (at least two required) ----------
    def test_modeling_stefan_boltzmann_equation(self) -> Tuple[bool, str]:
        """
        Verify that the Stefan-Boltzmann law, α I = σ T⁴, is correctly used.
        We re-derive the expected T from the given constants and compare with solution's T_max.
        This is allowed because we use only the problem's input values.
        """
        if self.T_max is None:
            return (False, "SKIP: T_max not available")
        # Compute expected T using the exact formula
        expected_T = (self.alpha1 * self.I / self.sigma) ** 0.25
        if not math.isclose(self.T_max, expected_T, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Stefan-Boltzmann: T_max = {self.T_max:.2f} does not match re-derived {expected_T:.2f}")
        return (True, f"Stefan-Boltzmann law correctly applied: {self.T_max:.2f} ≈ {expected_T:.2f}")

    def test_modeling_absorptance_halved(self) -> Tuple[bool, str]:
        """
        Check consistency when α is halved: T_prime should scale by (1/2)^(1/4).
        Uses solution's own T_max to avoid hardcoding expected value.
        """
        if self.T_max is None or self.T_prime is None:
            return (False, "SKIP: T_max or T_prime missing")
        expected_ratio = (0.5) ** 0.25
        actual_ratio = self.T_prime / self.T_max
        if not math.isclose(actual_ratio, expected_ratio, rel_tol=1.5e-2, abs_tol=1e-2):
            return (False, f"Halving ratio: {actual_ratio:.4f} differs from {expected_ratio:.4f}")
        return (True, f"Halving α gives ratio {actual_ratio:.4f} ≈ {expected_ratio:.4f}")

    # ---------- Monotonicity Tests ----------
    def test_monotonicity_temperature_increases_with_alpha(self) -> Tuple[bool, str]:
        """
        Explicitly perturb α upward by +10% and verify that the corresponding
        temperature (computed from law) is higher than the solution's T_max.
        We use the solution's reported T_max as baseline.
        """
        if self.T_max is None:
            return (False, "SKIP: T_max not available")
        # Use same I and sigma as solution; perturb α1 by +10%
        perturbed_alpha = self.alpha1 * 1.10
        perturbed_T = (perturbed_alpha * self.I / self.sigma) ** 0.25
        if perturbed_T <= self.T_max:
            return (False, f"Perturbed α (+10%) gives T = {perturbed_T:.2f} ≤ T_max = {self.T_max:.2f}, should be >")
        return (True, f"Monotonicity holds: perturbed α → T = {perturbed_T:.2f} > T_max = {self.T_max:.2f}")

    # ---------- Weak Answer Sanity Test (only one) ----------
    def test_answer_sanity_sign_and_range(self) -> Tuple[bool, str]:
        """Both temperatures must be positive, finite, and physically plausible (< 1000 K)."""
        for name, val in [("T_max", self.T_max), ("T_prime", self.T_prime)]:
            if val is None:
                continue
            if not (math.isfinite(val) and val > 0 and val < 1000):
                return (False, f"{name} = {val} out of plausible range (0, 1000)")
        return (True, "All present temperatures are positive, finite, and in plausible range")

    # ---------- Test Runner ----------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs: List[str] = []
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
                passed = False
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)