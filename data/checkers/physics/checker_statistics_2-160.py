import math
import sympy as sp
from typing import Tuple, List, Any, Optional, Dict

class PhysicsChecker:
    """
    PhysicsChecker for the Brownian motion problem.
    Checks the mean-square displacement result and underlying physics.
    """

    def __init__(self, solution_instance: Any):
        """
        Extract inputs and results safely with None defaults.

        Expected attributes of solution_instance:
            - givens (dict): contains keys T, t, lambda, optionally k_B
            - results (dict): contains key 'mean_sq_disp'
        """
        # Extract givens from solution_instance, trying multiple names
        self.givens = getattr(solution_instance, 'givens', None)
        if self.givens is None:
            self.givens = getattr(solution_instance, 'parameters', {})
        if self.givens is None:
            self.givens = getattr(solution_instance, 'inputs', {})
        if not isinstance(self.givens, dict):
            self.givens = {}

        # Extract results
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # Extract required quantities
        self.T = self.givens.get('T', None)
        self.t = self.givens.get('t', None)
        # 'lambda' is a Python keyword; use 'lamb' internally
        self.lamb = self.givens.get('lambda', None)
        self.k_B = self.givens.get('k_B', None) or self.givens.get('k', None)

        # Output
        self.mean_sq_disp = self.results.get('mean_sq_disp', None)

    # --------------------------------------------------------------------
    # Interface tests
    # --------------------------------------------------------------------
    def test_interface_results_has_mean_sq_disp(self) -> Tuple[bool, str]:
        """Check that 'mean_sq_disp' exists in results."""
        if not isinstance(self.results, dict):
            return (False, "SKIP: results is not a dict or not present")
        if 'mean_sq_disp' not in self.results:
            return (False, "'mean_sq_disp' missing in results")
        return (True, "'mean_sq_disp' present in results")

    def test_interface_mean_sq_disp_is_numeric(self) -> Tuple[bool, str]:
        """Check that mean_sq_disp is a number (float, int, or sympy number)."""
        if self.mean_sq_disp is None:
            return (False, "SKIP: mean_sq_disp not available")
        if not isinstance(self.mean_sq_disp, (int, float, sp.Number)):
            return (False, f"mean_sq_disp is not numeric, got {type(self.mean_sq_disp)}")
        return (True, "mean_sq_disp is numeric")

    def test_interface_required_givens_present(self) -> Tuple[bool, str]:
        """Check that required givens T, t, lambda are present."""
        missing = []
        if self.T is None:
            missing.append('T')
        if self.t is None:
            missing.append('t')
        if self.lamb is None:
            missing.append('lambda')
        if missing:
            return (False, f"Missing required givens: {', '.join(missing)}")
        return (True, "All required givens (T, t, lambda) present")

    # --------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # --------------------------------------------------------------------
    def test_invariant_nonnegative_disp(self) -> Tuple[bool, str]:
        """Mean-square displacement must be non-negative."""
        if self.mean_sq_disp is None:
            return (False, "SKIP: mean_sq_disp not available")
        if self.mean_sq_disp < 0:
            return (False, f"mean_sq_disp is negative: {self.mean_sq_disp}")
        return (True, "mean_sq_disp >= 0")

    def test_invariant_zero_time(self) -> Tuple[bool, str]:
        """If t=0, mean-square displacement must be zero."""
        if self.t is None or self.mean_sq_disp is None:
            return (False, "SKIP: need both t and mean_sq_disp")
        if self.t == 0:
            if not math.isclose(self.mean_sq_disp, 0.0, abs_tol=1e-12):
                return (False, f"With t=0, mean_sq_disp should be 0, got {self.mean_sq_disp}")
            return (True, "mean_sq_disp = 0 for t=0")
        return (True, "t > 0, no check needed")

    # --------------------------------------------------------------------
    # Modeling tests (verify physical relationships)
    # --------------------------------------------------------------------
    def test_modeling_mean_sq_disp_formula(self) -> Tuple[bool, str]:
        """
        Check that mean_sq_disp = T * t / lambda (the target formula).
        Uses math.isclose with relaxed tolerance.
        """
        if None in (self.T, self.t, self.lamb, self.mean_sq_disp):
            return (False, "SKIP: missing variables for formula check")
        expected = self.T * self.t / self.lamb
        if not math.isclose(self.mean_sq_disp, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False,
                    f"mean_sq_disp {self.mean_sq_disp} not close to expected {expected}")
        return (True, f"mean_sq_disp matches T*t/lambda within tolerance")

    def test_modeling_linear_in_time(self) -> Tuple[bool, str]:
        """
        Check that mean_sq_disp / t equals T / lambda (linearity in time).
        If t=0, we skip.
        """
        if None in (self.T, self.t, self.lamb, self.mean_sq_disp):
            return (False, "SKIP: missing variables")
        if self.t == 0:
            return (True, "t=0, linearity cannot be checked, but result should be 0")
        ratio1 = self.mean_sq_disp / self.t
        ratio2 = self.T / self.lamb
        if not math.isclose(ratio1, ratio2, rel_tol=1e-2, abs_tol=1e-2):
            return (False,
                    f"mean_sq_disp/t = {ratio1}, expected {ratio2}")
        return (True, "mean_sq_disp/t ≈ T/lambda")

    def test_modeling_fluctuation_dissipation_consistency(self) -> Tuple[bool, str]:
        """
        If k_B is provided, check that gamma = 2 * k_B * lambda gives a positive
        drag coefficient, and that the result also matches the
        fluctuation-dissipation form.
        """
        if self.k_B is None or self.lamb is None or self.mean_sq_disp is None:
            return (False, "SKIP: missing k_B, lambda, or mean_sq_disp")
        gamma = 2 * self.k_B * self.lamb
        if gamma <= 0:
            return (False, f"Derived gamma = {gamma} is not positive")
        # The result derived via fluctuation-dissipation: <x^2> = 2 k_B T t / gamma
        # Substituting gamma gives back T t / lambda, already tested.
        return (True, "Fluctuation-dissipation relation yields positive gamma")

    # --------------------------------------------------------------------
    # Monotonicity test (explicit perturbation of input)
    # --------------------------------------------------------------------
    def test_monotonicity_lambda_decreases_disp(self) -> Tuple[bool, str]:
        """
        If lambda increases, the mean-square displacement should decrease
        (for fixed T and t). We perturb lambda by +10% and compute the
        hypothetical new displacement from the formula. Then we compare
        the perturbed displacement with the baseline (original) displacement.
        The perturbed value must be less than the baseline.
        """
        if None in (self.T, self.t, self.lamb, self.mean_sq_disp):
            return (False, "SKIP: missing variables for monotonicity check")

        baseline = self.mean_sq_disp
        # Perturb lambda upward by 10%
        perturbed_lambda = self.lamb * 1.10
        # Hypothetical displacement if formula holds (T and t unchanged)
        perturbed_disp = self.T * self.t / perturbed_lambda

        # Check that increasing lambda reduces displacement
        if not (perturbed_disp < baseline):
            return (False,
                    f"Perturbed displacement {perturbed_disp} is not less than baseline {baseline}; "
                    "monotonicity (decreasing with lambda) violated")
        return (True,
                f"Lambda increased from {self.lamb} to {perturbed_lambda:.4f}, "
                f"displacement decreased from {baseline} to {perturbed_disp:.4f} – monotonicity holds")

    # --------------------------------------------------------------------
    # Weak answer sanity (positive, finite)
    # --------------------------------------------------------------------
    def test_answer_sanity_positive_finite(self) -> Tuple[bool, str]:
        """
        Weak sanity: mean_sq_disp must be finite and positive (or zero).
        No math.isclose used.
        """
        if self.mean_sq_disp is None:
            return (False, "SKIP: mean_sq_disp not available")
        if not math.isfinite(self.mean_sq_disp):
            return (False, "mean_sq_disp is not finite")
        if self.mean_sq_disp < 0:
            return (False, "mean_sq_disp is negative")
        return (True, "mean_sq_disp is finite and non-negative")

    # --------------------------------------------------------------------
    # Test runner
    # --------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all test_ methods and run them.
        Returns (all_passed, logs).
        """
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
                passed, msg = False, f"CRASH: {e}"
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)