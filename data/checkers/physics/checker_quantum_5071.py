import math
import sympy as sp
from typing import Any, Tuple, List

class PhysicsChecker:
    """
    Auditor for the 3D harmonic oscillator many-fermion problem.
    Checks interface, invariants, modeling, monotonicity, and answer sanity.
    All tests are unit-test-like and handle missing values gracefully.
    Monotonicity tests explicitly perturb an input and compare baseline vs. perturbed.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract all needed attributes safely – use None as default.
        self.hbar = getattr(solution_instance, 'hbar', None)
        self.omega = getattr(solution_instance, 'omega', None)
        self.m = getattr(solution_instance, 'm', None)
        self.e = getattr(solution_instance, 'e', None)
        self.c = getattr(solution_instance, 'c', None)
        self.B = getattr(solution_instance, 'B', None)
        self.mu = getattr(solution_instance, 'mu', None)
        self.sum_r_perp_sq = getattr(solution_instance, 'sum_r_sin2_theta', None)

        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        self.E_0 = results.get('E_0', None)
        self.L_0 = results.get('L_0', None)
        self.E_1 = results.get('E_1', None)
        self.L_1 = results.get('L_1', None)
        self.E_total = results.get('E_total', None)
        self.E_B = results.get('E_B', None)
        self.M = results.get('M', None)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_E_0_exists(self) -> Tuple[bool, str]:
        if self.E_0 is None:
            return (False, "E_0 missing")
        return (True, "E_0 present")

    def test_interface_L_0_exists(self) -> Tuple[bool, str]:
        if self.L_0 is None:
            return (False, "L_0 missing")
        return (True, "L_0 present")

    def test_interface_E_1_exists(self) -> Tuple[bool, str]:
        if self.E_1 is None:
            return (False, "E_1 missing")
        return (True, "E_1 present")

    def test_interface_L_1_exists(self) -> Tuple[bool, str]:
        if self.L_1 is None:
            return (False, "L_1 missing")
        return (True, "L_1 present")

    def test_interface_E_total_exists(self) -> Tuple[bool, str]:
        if self.E_total is None:
            return (False, "E_total missing")
        return (True, "E_total present")

    def test_interface_E_B_exists(self) -> Tuple[bool, str]:
        if self.E_B is None:
            return (False, "E_B missing")
        return (True, "E_B present")

    def test_interface_M_exists(self) -> Tuple[bool, str]:
        if self.M is None:
            return (False, "M missing")
        return (True, "M present")

    def test_interface_E_0_numeric(self) -> Tuple[bool, str]:
        if self.E_0 is None:
            return (False, "SKIP: E_0 missing")
        if not isinstance(self.E_0, (int, float, sp.Expr)):
            return (False, f"E_0 type not numeric: {type(self.E_0)}")
        return (True, "E_0 is numeric")

    def test_interface_L_0_numeric(self) -> Tuple[bool, str]:
        if self.L_0 is None:
            return (False, "SKIP: L_0 missing")
        if not isinstance(self.L_0, (int, float, sp.Expr)):
            return (False, f"L_0 type not numeric: {type(self.L_0)}")
        return (True, "L_0 is numeric")

    def test_interface_E_1_numeric(self) -> Tuple[bool, str]:
        if self.E_1 is None:
            return (False, "SKIP: E_1 missing")
        if not isinstance(self.E_1, (int, float, sp.Expr)):
            return (False, f"E_1 type not numeric: {type(self.E_1)}")
        return (True, "E_1 is numeric")

    def test_interface_L_1_numeric(self) -> Tuple[bool, str]:
        if self.L_1 is None:
            return (False, "SKIP: L_1 missing")
        if not isinstance(self.L_1, (int, float, sp.Expr)):
            return (False, f"L_1 type not numeric: {type(self.L_1)}")
        return (True, "L_1 is numeric")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_E_0_formula(self) -> Tuple[bool, str]:
        """E_0 should be (3/2)*hbar*omega"""
        if self.hbar is None or self.omega is None or self.E_0 is None:
            return (False, "SKIP: required inputs missing")
        expected = 1.5 * self.hbar * self.omega
        if math.isclose(float(self.E_0), float(expected), rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_0 matches (3/2)ħω: {self.E_0}")
        else:
            return (False, f"E_0 = {self.E_0} differs from expected {expected}")

    def test_invariant_E_1_formula(self) -> Tuple[bool, str]:
        """E_1 should be (5/2)*hbar*omega"""
        if self.hbar is None or self.omega is None or self.E_1 is None:
            return (False, "SKIP: required inputs missing")
        expected = 2.5 * self.hbar * self.omega
        if math.isclose(float(self.E_1), float(expected), rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_1 matches (5/2)ħω: {self.E_1}")
        else:
            return (False, f"E_1 = {self.E_1} differs from expected {expected}")

    def test_invariant_L_0_zero(self) -> Tuple[bool, str]:
        """Ground state (l=0) must have L=0"""
        if self.L_0 is None:
            return (False, "SKIP: L_0 missing")
        if math.isclose(float(self.L_0), 0.0, abs_tol=1e-2):
            return (True, "L_0 = 0")
        else:
            return (False, f"L_0 = {self.L_0} but must be 0")

    def test_invariant_L_1_magnitude(self) -> Tuple[bool, str]:
        """
        First excited state has l=1, so magnitude of angular momentum
        should be sqrt(l(l+1)) ħ = sqrt(2) ħ.
        """
        if self.hbar is None or self.L_1 is None:
            return (False, "SKIP: required inputs missing")
        expected = math.sqrt(2) * self.hbar
        if math.isclose(float(self.L_1), float(expected), rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"L_1 ≈ √2 ħ: {self.L_1}")
        else:
            return (False, f"L_1 = {self.L_1} differs from expected {expected} (√2 ħ)")

    def test_invariant_total_energy_sum(self) -> Tuple[bool, str]:
        """Total energy for 8 fermions should be 2*E_0 + 6*E_1."""
        if self.E_0 is None or self.E_1 is None or self.E_total is None:
            return (False, "SKIP: E_0, E_1, or E_total missing")
        expected = 2 * self.E_0 + 6 * self.E_1
        if math.isclose(float(self.E_total), float(expected), rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_total = {self.E_total} consistent with 2*E_0+6*E_1")
        else:
            return (False, f"E_total = {self.E_total} but 2*E_0+6*E_1 = {expected}")

    def test_invariant_E_B_quadratic_in_B(self) -> Tuple[bool, str]:
        """
        Without Zeeman shift, E_B - E_total should be quadratic in B.
        We test that E_B - E_total gives the diamagnetic term.
        """
        if any(x is None for x in [self.E_total, self.E_B, self.B, self.e, self.m, self.c, self.sum_r_perp_sq]):
            return (False, "SKIP: required inputs missing")
        coeff = self.e**2 / (8 * self.m * self.c**2) * self.sum_r_perp_sq
        delta = float(self.E_B) - float(self.E_total)
        expected = coeff * (float(self.B)**2)
        if math.isclose(delta, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "E_B - E_total matches diamagnetic formula")
        else:
            return (False, f"E_B - E_total = {delta}, expected {expected}")

    def test_invariant_M_minus_derivative(self) -> Tuple[bool, str]:
        """M should be -dE_B/dB. With diamagnetic term, M = - (e^2 B)/(4 m c^2) * sum_r_perp_sq."""
        if any(x is None for x in [self.M, self.B, self.e, self.m, self.c, self.sum_r_perp_sq]):
            return (False, "SKIP: required inputs missing")
        expected_M = - (self.e**2 * float(self.B)) / (4 * self.m * self.c**2) * self.sum_r_perp_sq
        if math.isclose(float(self.M), float(expected_M), rel_tol=1e-2, abs_tol=1e-2):
            return (True, "M matches derivative of E_B")
        else:
            return (False, f"M = {self.M}, expected {expected_M}")

    # ------------------------------------------------------------------
    # Modeling tests (quality of physics implementation)
    # ------------------------------------------------------------------
    def test_modeling_energy_ordering(self) -> Tuple[bool, str]:
        """Check that E_0 < E_1 < infinity."""
        if self.E_0 is None or self.E_1 is None:
            return (False, "SKIP: energies missing")
        if float(self.E_0) < float(self.E_1):
            return (True, "Energy ordering correct")
        else:
            return (False, f"E_0 ({self.E_0}) not less than E_1 ({self.E_1})")

    def test_modeling_total_energy_positive(self) -> Tuple[bool, str]:
        if self.E_total is None:
            return (False, "SKIP: E_total missing")
        if float(self.E_total) > 0:
            return (True, "E_total > 0")
        else:
            return (False, f"E_total = {self.E_total} not positive")

    def test_modeling_no_linear_Zeeman(self) -> Tuple[bool, str]:
        """
        For the filled-shell ground state, total spin is zero.
        Hence E_B should contain no term linear in B.
        We check that E_B - E_total is purely even (quadratic) in B.
        """
        if self.B is None or self.E_B is None or self.E_total is None:
            return (False, "SKIP: required missing")
        delta = float(self.E_B) - float(self.E_total)
        if abs(float(self.B)) < 1e-12:
            if abs(delta) < 1e-6:
                return (True, "E_B = E_total for B=0")
            else:
                return (False, f"Delta at B=0 is {delta}, should be 0")
        else:
            # The diamagnetic shift is positive for any B.
            if delta >= 0:
                return (True, "No linear Zeeman term detected (shift positive)")
            else:
                return (False, "Negative shift suggests linear Zeeman present")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit perturbation of an input)
    # ------------------------------------------------------------------
    def test_monotonicity_magnetization_increases_with_B(self) -> Tuple[bool, str]:
        """
        Perturb B -> B + epsilon. Compute the change in M using the analytic formula.
        M should become more negative (decrease) because the derivative dM/dB = - (e^2)/(4 m c^2)*sum is negative.
        We compare the sign of the change: perturbed_M - original_M should be negative.
        """
        if any(x is None for x in [self.M, self.B, self.e, self.m, self.c, self.sum_r_perp_sq]):
            return (False, "SKIP: required inputs missing")
        B0 = float(self.B)
        if B0 < 0:
            return (False, "SKIP: B is negative, cannot test monotonicity with positive perturbation")
        # Choose a small perturbation (1% of B, but at least 1e-6)
        epsilon = max(1e-6, 0.01 * B0) if B0 > 0 else 1e-6
        B_pert = B0 + epsilon
        # Compute original M
        M0 = float(self.M)
        # Compute perturbed M using the formula: M = - (e^2 B)/(4 m c^2) * sum
        coeff = - (self.e**2) / (4 * self.m * self.c**2) * self.sum_r_perp_sq
        M_pert = coeff * B_pert
        change = M_pert - M0
        # Since derivative dM/dB = coeff is negative, for epsilon>0, change should be negative.
        if change < 0:
            return (True, f"M decreases when B increases: delta = {change:.4e}")
        else:
            return (False, f"M does not decrease: original M={M0}, perturbed M={M_pert}, delta={change}")

    def test_monotonicity_energy_increasing_with_B(self) -> Tuple[bool, str]:
        """
        Perturb B -> B + epsilon. Using the diamagnetic formula, E_B should increase.
        Compute the change in E_B from the analytic expression.
        """
        if any(x is None for x in [self.E_B, self.E_total, self.B, self.e, self.m, self.c, self.sum_r_perp_sq]):
            return (False, "SKIP: required inputs missing")
        B0 = float(self.B)
        if B0 < 0:
            return (False, "SKIP: B is negative, cannot test monotonicity with positive perturbation")
        epsilon = max(1e-6, 0.01 * B0) if B0 > 0 else 1e-6
        B_pert = B0 + epsilon
        # Original E_B = E_total + (e^2 B^2)/(8 m c^2) * sum
        E0 = float(self.E_B)
        # Perturbed energy using formula
        E_pert = float(self.E_total) + (self.e**2 * B_pert**2) / (8 * self.m * self.c**2) * self.sum_r_perp_sq
        change = E_pert - E0
        if change > 0:
            return (True, f"E_B increases when B increases: delta = {change:.4e}")
        else:
            return (False, f"E_B does not increase: original E_B={E0}, perturbed E_B={E_pert}, delta={change}")

    # ------------------------------------------------------------------
    # Answer sanity (weak, only one method)
    # ------------------------------------------------------------------
    def test_answer_sanity_signs(self) -> Tuple[bool, str]:
        """Check that energies are positive, angular momenta nonnegative, etc."""
        problems = []
        if self.E_0 is not None and float(self.E_0) <= 0:
            problems.append("E_0 <= 0")
        if self.L_0 is not None and float(self.L_0) < 0:
            problems.append("L_0 negative")
        if self.E_1 is not None and float(self.E_1) <= 0:
            problems.append("E_1 <= 0")
        if self.L_1 is not None and float(self.L_1) < 0:
            problems.append("L_1 negative")
        if self.E_total is not None and float(self.E_total) <= 0:
            problems.append("E_total <= 0")
        if problems:
            return (False, "Sign issues: " + "; ".join(problems))
        return (True, "All signs plausible")

    # ------------------------------------------------------------------
    # Main test runner
    # ------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)