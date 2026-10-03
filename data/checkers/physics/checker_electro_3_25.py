import math
import traceback
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    PhysicsChecker for the two-terminal black box problem
    (series LC in parallel with R, resonance at 1 kHz).
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract all needed quantities from the solution instance safely.
        Use None as default for missing values (no numeric fallbacks).
        """
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            results = solution_instance.results
        else:
            results = {}

        # DC givens
        self.Vdc = getattr(solution_instance, 'Vdc', None) or results.get('Vdc', None)
        self.Idc = getattr(solution_instance, 'Idc', None) or results.get('Idc', None)
        # AC givens
        self.Vac = getattr(solution_instance, 'Vac', None) or results.get('Vac', None)
        self.Iac = getattr(solution_instance, 'Iac', None) or results.get('Iac', None)
        self.f0 = getattr(solution_instance, 'f0', None) or results.get('f0', None)
        self.f1 = getattr(solution_instance, 'f1', None) or results.get('f1', None)

        # Derived angular frequencies
        self.omega0 = 2.0 * math.pi * self.f0 if self.f0 is not None else None
        self.omega1 = 2.0 * math.pi * self.f1 if self.f1 is not None else None

        # Extracted results (emit keys)
        self.R = results.get('R', None)
        self.L = results.get('L', None)
        self.C = results.get('C', None)

        # Weak sanity limits
        self._R_low = 500.0
        self._R_high = 1500.0
        self._L_low = 0.5e-3
        self._L_high = 2.0e-3
        self._C_low = 10.0e-6
        self._C_high = 50.0e-6

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that R, L, C keys are present and are positive floats."""
        if self.R is None:
            return (False, "SKIP: R not provided")
        if self.L is None:
            return (False, "SKIP: L not provided")
        if self.C is None:
            return (False, "SKIP: C not provided")
        if not isinstance(self.R, (int, float)):
            return (False, f"R type invalid: {type(self.R)}")
        if not isinstance(self.L, (int, float)):
            return (False, f"L type invalid: {type(self.L)}")
        if not isinstance(self.C, (int, float)):
            return (False, f"C type invalid: {type(self.C)}")
        if self.R <= 0:
            return (False, f"R is not positive: {self.R}")
        if self.L <= 0:
            return (False, f"L is not positive: {self.L}")
        if self.C <= 0:
            return (False, f"C is not positive: {self.C}")
        return (True, "All required keys (R, L, C) present and positive.")

    # ------------------------------------------------------------------
    # Invariant / theorem checks (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_dc_resistance(self) -> Tuple[bool, str]:
        """R = Vdc / Idc (Ohm's law)."""
        if any(v is None for v in [self.R, self.Vdc, self.Idc]):
            return (False, "SKIP: missing data for DC resistance check")
        expected_R = self.Vdc / self.Idc
        if math.isclose(self.R, expected_R, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"R matches DC measurement: {self.R} ~ {expected_R:.0f}")
        return (False, f"R = {self.R} differs from Vdc/Idc = {expected_R:.0f}")

    def test_invariant_ac_impedance_magnitude(self) -> Tuple[bool, str]:
        """
        At f1 = 60 Hz, |Z| = Vac / Iac = 100 Ohm.
        Compute theoretical |Z| from L, C, R and compare.
        """
        if any(v is None for v in [self.R, self.L, self.C, self.omega1, self.Vac, self.Iac]):
            return (False, "SKIP: missing data for AC impedance check")
        X = self.omega1 * self.L - 1.0 / (self.omega1 * self.C)
        Y_sq_inv = 1.0 / (self.R * self.R) + 1.0 / (X * X)
        if Y_sq_inv <= 0:
            return (False, "Invalid admittance magnitude")
        Z_theory = 1.0 / math.sqrt(Y_sq_inv)
        Z_target = self.Vac / self.Iac
        if math.isclose(Z_theory, Z_target, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"|Z(60 Hz)| matches: {Z_theory:.2f} ~ {Z_target:.2f}")
        return (False, f"|Z(60 Hz)| = {Z_theory:.2f} differs from {Z_target:.2f}")

    def test_invariant_resonance_condition(self) -> Tuple[bool, str]:
        """omega0^2 * L * C = 1."""
        if any(v is None for v in [self.L, self.C, self.omega0]):
            return (False, "SKIP: missing data for resonance check")
        LC_omega0_sq = self.omega0 * self.omega0 * self.L * self.C
        if math.isclose(LC_omega0_sq, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Resonance condition holds: ω0²LC = {LC_omega0_sq:.5f} ≈ 1")
        return (False, f"Resonance condition fails: ω0²LC = {LC_omega0_sq:.5f}")

    # ------------------------------------------------------------------
    # Modeling / trace consistency checks (at least two)
    # ------------------------------------------------------------------
    def test_modeling_impedance_formula_consistency(self) -> Tuple[bool, str]:
        """
        Verify that the derived L and C are consistent with the
        admittance expression used in the solution steps.
        Use the DC result and the 60 Hz impedance magnitude.
        Check: 1/|Z|^2 - 1/R^2 = 1/X^2, where X = ωL - 1/(ωC).
        """
        if any(v is None for v in [self.R, self.L, self.C, self.omega1, self.Vac, self.Iac]):
            return (False, "SKIP: missing data for modeling consistency")
        X = self.omega1 * self.L - 1.0 / (self.omega1 * self.C)
        inv_Zsq = 1.0 / ((self.Vac / self.Iac) ** 2)
        inv_Rsq = 1.0 / (self.R ** 2)
        lhs = inv_Zsq - inv_Rsq
        rhs = 1.0 / (X * X)
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Modeling consistency: 1/Z² - 1/R² ≈ 1/X²  ({lhs:.6e} vs {rhs:.6e})")
        return (False, f"Modeling inconsistency: 1/Z² - 1/R² = {lhs:.6e}, 1/X² = {rhs:.6e}")

    def test_modeling_resonance_impedance_small(self) -> Tuple[bool, str]:
        """
        At a frequency very close to resonance (e.g., 999 Hz),
        the series LC branch should have nearly zero impedance,
        causing the total impedance to be very small.
        Check that |Z(f ~ f0)| < 1 Ω (much smaller than R).
        """
        if any(v is None for v in [self.R, self.L, self.C, self.f0]):
            return (False, "SKIP: missing data for resonance impedance test")
        f_near = self.f0 - 1.0  # 999 Hz
        omega_near = 2.0 * math.pi * f_near
        X_near = omega_near * self.L - 1.0 / (omega_near * self.C)
        Y_sq_near = 1.0 / (self.R**2) + 1.0 / (X_near**2)
        if Y_sq_near <= 0:
            return (False, "Invalid admittance near resonance")
        Z_near = 1.0 / math.sqrt(Y_sq_near)
        threshold = 1.0  # ohms
        if Z_near < threshold:
            return (True, f"Impedance near resonance is very small: |Z| = {Z_near:.4f} Ω < {threshold} Ω")
        return (False, f"Impedance near resonance is not small enough: |Z| = {Z_near:.4f} Ω ≥ {threshold} Ω")

    # ------------------------------------------------------------------
    # Monotonicity check (explicit perturbation of an input parameter)
    # ------------------------------------------------------------------
    def test_monotonicity_impedance_decreases_with_frequency_near_resonance(self) -> Tuple[bool, str]:
        """
        Perturb the frequency slightly upward from the given 60 Hz (f1)
        and verify that the impedance magnitude decreases (because we move
        toward resonance at 1 kHz). Baseline: impedance at f1.
        Perturbation: increase omega1 by 1% (omega1 * 1.01).
        """
        if any(v is None for v in [self.R, self.L, self.C, self.omega1]):
            return (False, "SKIP: missing data for monotonicity check")
        # baseline impedance at omega1
        X1 = self.omega1 * self.L - 1.0 / (self.omega1 * self.C)
        Y1_sq = 1.0 / (self.R**2) + 1.0 / (X1**2)
        if Y1_sq <= 0:
            return (False, "Baseline impedance computation failed")
        Z_base = 1.0 / math.sqrt(Y1_sq)

        # perturbed frequency (1% increase)
        omega_pert = self.omega1 * 1.01
        X_pert = omega_pert * self.L - 1.0 / (omega_pert * self.C)
        Y_pert_sq = 1.0 / (self.R**2) + 1.0 / (X_pert**2)
        if Y_pert_sq <= 0:
            return (False, "Perturbed impedance computation failed")
        Z_pert = 1.0 / math.sqrt(Y_pert_sq)

        # Since both frequencies are below resonance, an increase in frequency
        # brings us closer to resonance, so impedance should decrease.
        if Z_pert < Z_base:
            return (True, f"Monotonicity holds: Z(omega1*1.01) = {Z_pert:.2f} < Z(omega1) = {Z_base:.2f}")
        return (False, f"Monotonicity fails: Z(omega1*1.01) = {Z_pert:.2f} >= Z(omega1) = {Z_base:.2f}")

    # ------------------------------------------------------------------
    # Weak final answer sanity check (no isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """
        Broad magnitude checks only: R [500,1500], L [0.5,2] mH, C [10,50] µF.
        """
        if self.R is None or self.L is None or self.C is None:
            return (False, "SKIP: missing data for answer sanity")
        failures = []
        if not (self._R_low <= self.R <= self._R_high):
            failures.append(f"R={self.R} not in [{self._R_low}, {self._R_high}]")
        if not (self._L_low <= self.L <= self._L_high):
            failures.append(f"L={self.L} not in [{self._L_low}, {self._L_high}]")
        if not (self._C_low <= self.C <= self._C_high):
            failures.append(f"C={self.C} not in [{self._C_low}, {self._C_high}]")
        if failures:
            return (False, "Weak sanity failed: " + "; ".join(failures))
        return (True, "All values within plausible ranges.")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover and run all test_* methods, collect logs.
        Returns (overall_pass, list_of_log_messages).
        """
        logs = []
        all_pass = True
        for attr_name in sorted(dir(self)):
            if not attr_name.startswith("test_"):
                continue
            method = getattr(self, attr_name, None)
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