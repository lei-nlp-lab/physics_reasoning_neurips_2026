import sympy as sp
import math
import traceback
from typing import Any, Dict, Tuple, List, Optional

class PhysicsChecker:
    """
    Physics checker for the D0 meson decay problem.
    Tests interface completeness, invariant physical relations, modeling quality,
    monotonicity under parameter changes, and a weak final-answer sanity check.
    """

    def __init__(self, solution_instance: Any):
        # Extract givens safely – default to None if missing
        self.m = getattr(solution_instance, 'm', None)
        self.E = getattr(solution_instance, 'E', None)
        self.tau = getattr(solution_instance, 'tau', None)
        self.c = getattr(solution_instance, 'c', None)

        # Extract results
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}
        self.lambda_val = self.results.get('lambda', None)
        self.R_max = self.results.get('R_max', None)

        # Physical constant (not a fallback for missing givens)
        self.ln2 = math.log(2)

    # --------------------------------------------------------------------------
    # Interface tests
    # --------------------------------------------------------------------------
    def test_interface_has_lambda(self) -> Tuple[bool, str]:
        """Check that 'lambda' exists in results and is a positive number."""
        if self.lambda_val is None:
            return (False, "Missing 'lambda' in results")
        if not isinstance(self.lambda_val, (int, float)):
            return (False, "'lambda' is not numeric")
        if self.lambda_val <= 0:
            return (False, f"'lambda' is not positive: {self.lambda_val}")
        return (True, f"'lambda' present and positive: {self.lambda_val}")

    def test_interface_has_R_max(self) -> Tuple[bool, str]:
        """Check that 'R_max' exists in results and is a positive number."""
        if self.R_max is None:
            return (False, "Missing 'R_max' in results")
        if not isinstance(self.R_max, (int, float)):
            return (False, "'R_max' is not numeric")
        if self.R_max <= 0:
            return (False, f"'R_max' is not positive: {self.R_max}")
        return (True, f"'R_max' present and positive: {self.R_max}")

    # --------------------------------------------------------------------------
    # Invariant (theorem-based) tests
    # --------------------------------------------------------------------------
    def test_invariant_gamma_energy_mass(self) -> Tuple[bool, str]:
        if self.m is None or self.E is None:
            return (False, "SKIP: m or E not provided")
        if self.m <= 0:
            return (False, f"Invalid mass: {self.m}")
        gamma = self.E / self.m
        if not math.isclose(gamma, 10.0, rel_tol=1e-3, abs_tol=1e-3):
            return (False, f"gamma = {gamma} not close to 10")
        return (True, f"gamma = {gamma} ≈ 10")

    def test_invariant_beta_gamma_relation(self) -> Tuple[bool, str]:
        if self.m is None or self.E is None:
            return (False, "SKIP: cannot compute gamma")
        gamma = self.E / self.m
        beta_gamma_expected = math.sqrt(gamma**2 - 1)
        beta_gamma_result = self.results.get('beta_gamma', None)
        if beta_gamma_result is not None:
            if not math.isclose(beta_gamma_result, beta_gamma_expected, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"beta_gamma result {beta_gamma_result} != expected {beta_gamma_expected}")
            return (True, f"beta_gamma {beta_gamma_result} ≈ {beta_gamma_expected}")
        return (True, "beta_gamma not in results, skipping direct comparison")

    def test_invariant_lambda(self) -> Tuple[bool, str]:
        """
        Check that lambda = beta_gamma * c * tau.
        Requires c to be present (no fallback).
        """
        if self.lambda_val is None:
            return (False, "SKIP: lambda not available")
        if self.m is None or self.E is None or self.tau is None or self.c is None:
            return (False, "SKIP: insufficient givens to recompute lambda")
        gamma = self.E / self.m
        if gamma <= 1:
            return (False, f"gamma must be > 1, got {gamma}")
        beta_gamma = math.sqrt(gamma**2 - 1)
        lambda_expected = beta_gamma * self.c * self.tau
        if not math.isclose(self.lambda_val, lambda_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"lambda {self.lambda_val} != expected {lambda_expected}")
        return (True, f"lambda = {self.lambda_val} ≈ {lambda_expected}")

    def test_invariant_R_max_from_lambda(self) -> Tuple[bool, str]:
        if self.R_max is None or self.lambda_val is None:
            return (False, "SKIP: R_max or lambda missing")
        R_expected = self.lambda_val * self.ln2
        if not math.isclose(self.R_max, R_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"R_max {self.R_max} != lambda*ln2 = {R_expected}")
        return (True, f"R_max = {self.R_max} ≈ {R_expected}")

    def test_invariant_exponential_halfpoint(self) -> Tuple[bool, str]:
        if self.R_max is None or self.lambda_val is None:
            return (False, "SKIP: R_max or lambda missing")
        if self.lambda_val <= 0:
            return (False, "lambda must be positive")
        prob = math.exp(-self.R_max / self.lambda_val)
        if not math.isclose(prob, 0.5, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"exp(-R_max/lambda) = {prob} not close to 0.5")
        return (True, f"P(x > R_max) = {prob} ≈ 0.5")

    # --------------------------------------------------------------------------
    # Modeling / trace tests
    # --------------------------------------------------------------------------
    def test_modeling_lambda_positive(self) -> Tuple[bool, str]:
        if self.lambda_val is None:
            return (False, "lambda not available")
        if self.lambda_val <= 0:
            return (False, f"lambda not positive: {self.lambda_val}")
        return (True, f"lambda = {self.lambda_val} > 0")

    def test_modeling_R_max_positive(self) -> Tuple[bool, str]:
        if self.R_max is None:
            return (False, "R_max not available")
        if self.R_max <= 0:
            return (False, f"R_max not positive: {self.R_max}")
        return (True, f"R_max = {self.R_max} > 0")

    def test_modeling_gamma_greater_than_one(self) -> Tuple[bool, str]:
        if self.m is None or self.E is None:
            return (False, "SKIP: cannot compute gamma")
        gamma = self.E / self.m
        if gamma <= 1:
            return (False, f"gamma = {gamma} is not > 1")
        return (True, f"gamma = {gamma} > 1")

    # --------------------------------------------------------------------------
    # Monotonicity tests (perturb an input and compare perturbed output to baseline)
    # --------------------------------------------------------------------------
    def test_monotonicity_energy_increases_lambda(self) -> Tuple[bool, str]:
        """
        Increase energy by 10% (keeping all other givens fixed) and check that
        the predicted lambda (from physics relation) increases.
        """
        if self.m is None or self.E is None or self.tau is None or self.c is None:
            return (False, "SKIP: insufficient givens for monotonicity test")
        gamma0 = self.E / self.m
        beta_gamma0 = math.sqrt(gamma0**2 - 1)
        lambda0 = beta_gamma0 * self.c * self.tau

        # Perturb energy upward by 10%
        E_new = self.E * 1.1
        gamma_new = E_new / self.m
        beta_gamma_new = math.sqrt(gamma_new**2 - 1)
        lambda_new = beta_gamma_new * self.c * self.tau

        if lambda_new <= lambda0:
            return (False, f"Monotonicity violated: lambda with E'={E_new} = {lambda_new} <= {lambda0}")
        return (True, f"Increasing energy by 10% increases lambda from {lambda0} to {lambda_new}")

    def test_monotonicity_tau_increases_lambda(self) -> Tuple[bool, str]:
        """
        Increase proper lifetime by 10% (keeping other givens fixed) and check that
        the predicted lambda increases proportionally.
        """
        if self.m is None or self.E is None or self.tau is None or self.c is None:
            return (False, "SKIP: insufficient givens for monotonicity test")
        gamma = self.E / self.m
        beta_gamma = math.sqrt(gamma**2 - 1)
        lambda0 = beta_gamma * self.c * self.tau

        # Perturb tau upward by 10%
        tau_new = self.tau * 1.1
        lambda_new = beta_gamma * self.c * tau_new

        if lambda_new <= lambda0:
            return (False, f"Monotonicity violated: lambda with tau'={tau_new} = {lambda_new} <= {lambda0}")
        return (True, f"Increasing tau by 10% increases lambda from {lambda0} to {lambda_new}")

    # --------------------------------------------------------------------------
    # Weak final-answer sanity (only one such method allowed)
    # --------------------------------------------------------------------------
    def test_answer_sanity_R_max_realistic_range(self) -> Tuple[bool, str]:
        """
        Weak sanity: R_max must lie within a plausible range and be > 0.44 mm.
        No precise derivation – only sign, finite, and broad magnitude checks.
        """
        if self.R_max is None:
            return (False, "SKIP: R_max not available")
        # Plausible lower bound: 0.4 mm (less than the answer, but R_max must be larger)
        if self.R_max < 0.4:
            return (False, f"R_max = {self.R_max} < 0.4 mm, unrealistically small")
        if self.R_max > 5.0:
            return (False, f"R_max = {self.R_max} > 5 mm, unrealistically large")
        # Resolution 0.44 mm is the answer; it must be < R_max to observe >50% of decays.
        if self.R_max <= 0.44:
            return (False, f"R_max = {self.R_max} <= 0.44 mm, but resolution 0.44 mm would not observe >50%")
        return (True, f"R_max = {self.R_max} in plausible range (0.4–5 mm) and >0.44 mm")

    # --------------------------------------------------------------------------
    # Test runner
    # --------------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all test_* methods, execute them, collect logs.
        Returns (all_pass, logs).
        """
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
                passed, msg = False, f"{attr_name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {attr_name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)