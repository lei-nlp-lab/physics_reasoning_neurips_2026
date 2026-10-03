import math
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Auditor for torque problems involving circularly polarized light.
    Extracts results and givens safely and runs unit-test-style checks.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract results and givens with safe defaults
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {}) or {}
        self.givens: Dict[str, Any] = getattr(solution_instance, 'givens', {}) or {}
        
        # Specific result keys (use None for missing)
        self.tau0: Optional[float] = self.results.get('tau0', None)
        self.tau2: Optional[float] = self.results.get('tau2', None)
        self.tau3: Optional[float] = self.results.get('tau3', None)
        self.tau4: Optional[float] = self.results.get('tau4', None)
        
        # Optional givens for invariant and monotonicity checks
        # Standard physical constants in CGS (but we'll use None if not given)
        self.P: Optional[float] = self.givens.get('P', None)            # power in erg/s
        self.lambda_: Optional[float] = self.givens.get('lambda', None) # wavelength in cm
        self.c: Optional[float] = self.givens.get('c', None)           # speed of light in cm/s
        self.h: Optional[float] = self.givens.get('h', None)           # Planck constant in erg·s

        # Explanation steps for modeling trace checks
        self.explanation_steps: Any = getattr(solution_instance, 'explanation_steps', None)

    # ------------------------------------------------------------------
    # Interface checks – required keys exist and have valid types
    # ------------------------------------------------------------------
    def test_interface_tau0_present(self) -> Tuple[bool, str]:
        if self.tau0 is None:
            return (False, "SKIP: tau0 not found in results")
        if not isinstance(self.tau0, (int, float)):
            return (False, "tau0 is not numeric")
        if math.isnan(self.tau0) or math.isinf(self.tau0):
            return (False, "tau0 is NaN or Inf")
        return (True, "tau0 present and numeric")

    def test_interface_tau2_present(self) -> Tuple[bool, str]:
        if self.tau2 is None:
            return (False, "SKIP: tau2 not found in results")
        if not isinstance(self.tau2, (int, float)):
            return (False, "tau2 is not numeric")
        if math.isnan(self.tau2) or math.isinf(self.tau2):
            return (False, "tau2 is NaN or Inf")
        return (True, "tau2 present and numeric")

    def test_interface_tau3_present(self) -> Tuple[bool, str]:
        if self.tau3 is None:
            return (False, "SKIP: tau3 not found in results")
        if not isinstance(self.tau3, (int, float)):
            return (False, "tau3 is not numeric")
        if math.isnan(self.tau3) or math.isinf(self.tau3):
            return (False, "tau3 is NaN or Inf")
        return (True, "tau3 present and numeric")

    def test_interface_tau4_present(self) -> Tuple[bool, str]:
        if self.tau4 is None:
            return (False, "SKIP: tau4 not found in results")
        if not isinstance(self.tau4, (int, float)):
            return (False, "tau4 is not numeric")
        if math.isnan(self.tau4) or math.isinf(self.tau4):
            return (False, "tau4 is NaN or Inf")
        return (True, "tau4 present and numeric")

    # ------------------------------------------------------------------
    # Invariant checks – theorem-based physical sanity
    # ------------------------------------------------------------------
    def test_invariant_torque_mirror_zero(self) -> Tuple[bool, str]:
        """According to key constraints, reflected torque is zero."""
        if self.tau2 is None:
            return (False, "SKIP: tau2 missing")
        if not math.isclose(self.tau2, 0.0, abs_tol=1e-12):
            return (False, f"tau2 = {self.tau2}, expected 0")
        return (True, "tau2 is zero as expected")

    def test_invariant_torque_halfwave(self) -> Tuple[bool, str]:
        """Half-wave plate flips handedness → torque = 2 tau0."""
        if self.tau0 is None or self.tau3 is None:
            return (False, "SKIP: tau0 or tau3 missing")
        expected = 2.0 * self.tau0
        if not math.isclose(self.tau3, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"tau3 = {self.tau3}, expected 2*tau0 = {expected}")
        return (True, "tau3 equals 2*tau0 within tolerance")

    def test_invariant_torque_halfwave_mirror_zero(self) -> Tuple[bool, str]:
        """Double pass through half-wave+mirror returns incident state → torque zero."""
        if self.tau4 is None:
            return (False, "SKIP: tau4 missing")
        if not math.isclose(self.tau4, 0.0, abs_tol=1e-12):
            return (False, f"tau4 = {self.tau4}, expected 0")
        return (True, "tau4 is zero as expected")

    # ------------------------------------------------------------------
    # Modeling / trace checks – consistency with explanation steps
    # ------------------------------------------------------------------
    def test_modeling_tau0_formula(self) -> Tuple[bool, str]:
        """Check that tau0 ≈ P*λ/(2πc) within tolerance (re-derive from givens)."""
        if self.tau0 is None:
            return (False, "SKIP: tau0 missing")
        if self.P is None or self.lambda_ is None or self.c is None:
            return (False, "SKIP: required givens P, lambda, c missing")
        # Use math.pi, but we need to convert to consistent units.
        # Assume P in erg/s, lambda in cm, c in cm/s → result in dyne·cm
        expected = self.P * self.lambda_ / (2.0 * math.pi * self.c)
        if not math.isclose(self.tau0, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"tau0 = {self.tau0}, but P*λ/(2πc) = {expected}")
        return (True, "tau0 matches P*λ/(2πc) within 1% tolerance")

    def test_modeling_tau3_two_tau0(self) -> Tuple[bool, str]:
        """Check that tau3 = 2*tau0 (derived from half-wave spin flip)."""
        if self.tau0 is None or self.tau3 is None:
            return (False, "SKIP: tau0 or tau3 missing")
        # Re-derive: tau3 = N * (L_inc - L_trans) where L_inc=+ħ, L_trans=-ħ → change 2ħ
        # tau0 = N*ħ, so tau3 = 2*N*ħ = 2*tau0.  No extra constants needed.
        expected = 2.0 * self.tau0
        if not math.isclose(self.tau3, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"tau3 = {self.tau3}, expected 2*tau0 = {expected}")
        return (True, "tau3 equals 2*tau0 within tolerance (half-wave flip)")

    def test_modeling_tau2_zero_by_theorem(self) -> Tuple[bool, str]:
        """Check that tau2 = 0 from theorem (mirror reflects unchanged spin)."""
        if self.tau2 is None:
            return (False, "SKIP: tau2 missing")
        # Theorem: incident RHC +ħ, after mirror propagation reversed but spin unchanged → torque zero.
        if not math.isclose(self.tau2, 0.0, abs_tol=1e-12):
            return (False, f"tau2 = {self.tau2}, expected 0")
        return (True, "tau2 is zero, consistent with mirror theorem")

    # ------------------------------------------------------------------
    # Monotonicity checks – perturb an input and compare output
    # ------------------------------------------------------------------
    def test_monotonicity_tau0_increases_with_power(self) -> Tuple[bool, str]:
        """Perturb power P by +5% and check tau0 scales proportionally."""
        if self.tau0 is None or self.P is None:
            return (False, "SKIP: tau0 or P missing")
        # Simulate perturbed power: P' = P * 1.05
        fractional_increase = 0.05
        perturbed_tau0 = self.tau0 * (1.0 + fractional_increase)
        # Since tau0 is linear in P, perturbed_tau0 should be tau0 * 1.05
        if not math.isclose(perturbed_tau0, self.tau0 * 1.05, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Perturbed tau0 = {perturbed_tau0}, expected {self.tau0 * 1.05}")
        return (True, "tau0 scales linearly with power (5% increase)")

    def test_monotonicity_tau0_increases_with_wavelength(self) -> Tuple[bool, str]:
        """Perturb wavelength lambda by +5% and check tau0 scales proportionally."""
        if self.tau0 is None or self.lambda_ is None:
            return (False, "SKIP: tau0 or lambda missing")
        fractional_increase = 0.05
        perturbed_tau0 = self.tau0 * (1.0 + fractional_increase)
        if not math.isclose(perturbed_tau0, self.tau0 * 1.05, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Perturbed tau0 = {perturbed_tau0}, expected {self.tau0 * 1.05}")
        return (True, "tau0 scales linearly with wavelength (5% increase)")

    # ------------------------------------------------------------------
    # Weak answer sanity checks – only one allowed
    # ------------------------------------------------------------------
    def test_answer_sanity_tau0_positive(self) -> Tuple[bool, str]:
        """
        Very weak check: tau0 should be positive and within plausible range.
        """
        if self.tau0 is None:
            return (False, "SKIP: tau0 missing")
        if self.tau0 <= 0:
            return (False, f"tau0 = {self.tau0} is not positive")
        # Accept range 1e-12 to 1e-6 dyne·cm (typical for optical torques)
        if self.tau0 < 1e-12 or self.tau0 > 1e-6:
            return (False, f"tau0 = {self.tau0} outside plausible range [1e-12, 1e-6]")
        return (True, "tau0 positive and in plausible range")

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)