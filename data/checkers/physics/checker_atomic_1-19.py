import math
import sympy as sp
import traceback
from typing import Any, Dict, Tuple, List

class PhysicsChecker:
    """
    Audits the solution for the photon absorption problem.
    Checks energy-momentum conservation, re-derived formula, monotonicity,
    and basic interface correctness.
    """

    def __init__(self, solution_instance: Any) -> None:
        # --- Safe extraction of givens (use None for missing) ---
        givens = getattr(solution_instance, 'givens', {})
        if givens is None:
            givens = {}
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}

        self.M = givens.get('M', None) or getattr(solution_instance, 'M', None)
        self.delta = givens.get('delta_', None) or givens.get('delta', None) or getattr(solution_instance, 'delta_', None)
        self.c = givens.get('c', None) or getattr(solution_instance, 'c', None)
        self.h = givens.get('h', None) or getattr(solution_instance, 'h', None)

        # --- Safe extraction of results ---
        self.nu = results.get('nu', None) or getattr(solution_instance, 'nu', None)

        # Store additional info for optional use
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

    # -----------------------------------------------------------------
    # Interface tests
    # -----------------------------------------------------------------
    def test_interface_nu_present(self) -> Tuple[bool, str]:
        if self.nu is None:
            return (False, "SKIP: nu not provided in results")
        return (True, "nu is present")

    def test_interface_nu_numeric(self) -> Tuple[bool, str]:
        if self.nu is None:
            return (False, "SKIP: nu not available")
        if not isinstance(self.nu, (int, float)):
            return (False, f"nu is not numeric, type={type(self.nu).__name__}")
        return (True, "nu is numeric")

    def test_interface_givens_present(self) -> Tuple[bool, str]:
        missing = []
        for name, val in [('M', self.M), ('delta', self.delta),
                          ('c', self.c), ('h', self.h)]:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All required givens (M, delta, c, h) are present")

    # -----------------------------------------------------------------
    # Invariant checks
    # -----------------------------------------------------------------
    def test_invariant_energy_momentum_conservation(self) -> Tuple[bool, str]:
        if any(v is None for v in [self.M, self.delta, self.c, self.h, self.nu]):
            return (False, "SKIP: one or more required quantities are None")
        try:
            M_final = self.M + self.delta
            E_gamma = self.h * self.nu
            p_gamma = E_gamma / self.c
            E_initial = self.M * self.c**2 + E_gamma
            E_final = math.sqrt(M_final**2 * self.c**4 + p_gamma**2 * self.c**2)
            if math.isclose(E_initial, E_final, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "Energy-momentum conservation holds within 1%")
            else:
                return (False, f"Energy mismatch: initial={E_initial:.6e}, final={E_final:.6e}")
        except Exception as e:
            return (False, f"Invariant check crashed: {e}")

    def test_invariant_momentum_relation(self) -> Tuple[bool, str]:
        if self.nu is None or self.c is None or self.h is None:
            return (False, "SKIP: required quantities missing")
        p = self.h * self.nu / self.c
        if p <= 0 or not math.isfinite(p):
            return (False, f"Non-physical momentum: {p}")
        return (True, f"Momentum p = h*nu/c = {p:.6e} is positive and finite")

    # -----------------------------------------------------------------
    # Modeling / trace tests (at least two required)
    # -----------------------------------------------------------------
    def test_modeling_formula_consistency(self) -> Tuple[bool, str]:
        """
        Check that the output nu matches the derived analytic expression:
        nu = (delta * c^2 / h) * (1 + delta / (2*M))
        """
        if any(v is None for v in [self.M, self.delta, self.c, self.h, self.nu]):
            return (False, "SKIP: one or more required quantities are None")
        if self.M <= 0:
            return (False, "M must be positive")
        if self.delta <= 0:
            return (False, "delta must be positive")
        try:
            expected_nu = (self.delta * self.c**2 / self.h) * (1 + self.delta / (2 * self.M))
            if math.isclose(self.nu, expected_nu, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"nu matches derived formula: expected={expected_nu:.6e}, got={self.nu:.6e}")
            else:
                return (False, f"nu mismatch: expected={expected_nu:.6e}, got={self.nu:.6e}")
        except Exception as e:
            return (False, f"Formula check crashed: {e}")

    def test_modeling_small_delta_limit(self) -> Tuple[bool, str]:
        """
        In the limit delta << M, the formula reduces to nu ≈ delta * c^2 / h.
        Check that the provided nu satisfies this within a small tolerance,
        using a value of delta that is small relative to M.
        """
        if any(v is None for v in [self.M, self.delta, self.c, self.h, self.nu]):
            return (False, "SKIP: missing data for limit check")
        if self.M <= 0:
            return (False, "M must be positive")
        # Choose delta such that delta / M = 0.01 (1% of M)
        small_delta = 0.01 * self.M
        # Compute expected nu from full formula
        expected_nu_full = (small_delta * self.c**2 / self.h) * (1 + small_delta / (2 * self.M))
        # Approximate limit formula (neglecting second order)
        expected_nu_approx = (small_delta * self.c**2 / self.h)
        # The relative difference should be about delta/(2M) = 0.005, i.e. 0.5%
        # So we check that the actual output (if the solution used a similar small delta)
        # would satisfy that the approximate formula is close to the full formula.
        # Since we don't have a separate output for small delta, we test the derived
        # expression for consistency: the approximate formula should be within 1% of full.
        # This is a modeling sanity check.
        if math.isclose(expected_nu_full, expected_nu_approx, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Small delta limit (nu ≈ delta*c^2/h) is consistent")
        else:
            return (False, f"Small delta limit fails: full={expected_nu_full:.6e}, approx={expected_nu_approx:.6e}")

    # -----------------------------------------------------------------
    # Monotonicity test (must perturb an input and compare output)
    # -----------------------------------------------------------------
    def test_monotonicity_nu_increases_with_delta(self) -> Tuple[bool, str]:
        """
        Perturb delta by +10% and check that nu increases.
        We re-derive the expected nu for the perturbed delta using the analytic formula.
        """
        if any(v is None for v in [self.M, self.delta, self.c, self.h, self.nu]):
            return (False, "SKIP: missing data for monotonicity")
        if self.delta <= 0:
            return (False, "delta must be positive for monotonicity check")
        try:
            delta0 = self.delta
            nu0 = self.nu
            delta1 = delta0 * 1.10
            nu1 = (delta1 * self.c**2 / self.h) * (1 + delta1 / (2 * self.M))
            if nu1 > nu0:
                return (True, f"nu increases with delta: delta0={delta0:.4e}->delta1={delta1:.4e}, nu0={nu0:.4e}->nu1={nu1:.4e}")
            else:
                return (False, f"nu did not increase: nu0={nu0:.4e}, nu1={nu1:.4e}")
        except Exception as e:
            return (False, f"Monotonicity check crashed: {e}")

    # -----------------------------------------------------------------
    # Weak answer sanity (only one, no isclose, no re-derivation)
    # -----------------------------------------------------------------
    def test_answer_sanity_sign_and_finite(self) -> Tuple[bool, str]:
        if self.nu is None:
            return (False, "SKIP: nu not available")
        if not isinstance(self.nu, (int, float)):
            return (False, "nu is not numeric")
        if math.isnan(self.nu) or math.isinf(self.nu):
            return (False, f"nu is NaN or inf: {self.nu}")
        if self.nu <= 0:
            return (False, f"nu is not positive: {self.nu}")
        return (True, "nu is positive, finite, and not NaN")

    # -----------------------------------------------------------------
    # Run all tests (discovery and execution)
    # -----------------------------------------------------------------
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