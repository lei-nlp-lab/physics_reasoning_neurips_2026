import math
import traceback
from typing import Any, Dict, List, Tuple

class PhysicsChecker:
    """
    Auditor for the free electron boundary diffusion problem.
    Extracts required quantities and performs unit-test-like checks.
    """

    def __init__(self, solution_instance: Any):
        # Extract givens with None defaults
        self.hbar = getattr(solution_instance, 'hbar', None)
        self.m = getattr(solution_instance, 'm', None)
        self.l = getattr(solution_instance, 'l', None)
        # Extract results from dict or attribute
        res = getattr(solution_instance, 'results', None)
        if res is None:
            res = getattr(solution_instance, 'result', None)  # fallback
        if res is not None:
            self.D = res.get('D', None)
            self.t_diff = res.get('t_diff', None)
            self.boundary_delay = res.get('boundary_delay', None)
            self.epsilon_threshold = res.get('epsilon_threshold', None)
        else:
            # Try direct attribute access
            self.D = getattr(solution_instance, 'D', None)
            self.t_diff = getattr(solution_instance, 't_diff', None)
            self.boundary_delay = getattr(solution_instance, 'boundary_delay', None)
            self.epsilon_threshold = getattr(solution_instance, 'epsilon_threshold', None)

    # ---------------------------------------------------------------
    # Interface tests
    # ---------------------------------------------------------------
    def test_interface_required_keys_exist(self) -> Tuple[bool, str]:
        """Check that all four required result keys are present (not None)."""
        if self.D is None:
            return (False, "Missing key 'D'")
        if self.t_diff is None:
            return (False, "Missing key 't_diff'")
        if self.boundary_delay is None:
            return (False, "Missing key 'boundary_delay'")
        if self.epsilon_threshold is None:
            return (False, "Missing key 'epsilon_threshold'")
        return (True, "All required result keys present")

    def test_interface_types_correct(self) -> Tuple[bool, str]:
        """Check that D, t_diff, epsilon_threshold are numeric and boundary_delay is string."""
        if not isinstance(self.D, (int, float)):
            return (False, f"D is not numeric: {type(self.D)}")
        if not isinstance(self.t_diff, (int, float)):
            return (False, f"t_diff is not numeric: {type(self.t_diff)}")
        if not isinstance(self.epsilon_threshold, (int, float)):
            return (False, f"epsilon_threshold is not numeric: {type(self.epsilon_threshold)}")
        if not isinstance(self.boundary_delay, str):
            return (False, f"boundary_delay is not string: {type(self.boundary_delay)}")
        return (True, "All types correct")

    # ---------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ---------------------------------------------------------------
    def test_invariant_diffusion_coefficient(self) -> Tuple[bool, str]:
        """Check D = hbar/(2m) from diffusion equation derivation."""
        if self.hbar is None or self.m is None:
            return (False, "SKIP: hbar or m not available")
        expected_D = self.hbar / (2.0 * self.m)
        if self.D is None:
            return (False, "D not available")
        if math.isclose(self.D, expected_D, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"D = {self.D} matches hbar/(2m) = {expected_D}")
        else:
            return (False, f"D = {self.D} does not match hbar/(2m) = {expected_D}")

    def test_invariant_boundary_condition(self) -> Tuple[bool, str]:
        """Verify that t_diff satisfies the characteristic diffusion time relation t_diff = l^2/D."""
        if self.l is None or self.D is None or self.t_diff is None:
            return (False, "SKIP: l, D, or t_diff missing")
        expected_t_diff = self.l**2 / self.D
        if math.isclose(self.t_diff, expected_t_diff, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"t_diff = {self.t_diff} matches l^2/D = {expected_t_diff}")
        else:
            return (False, f"t_diff = {self.t_diff} does not match l^2/D = {expected_t_diff}")

    def test_invariant_energy_threshold_scaling(self) -> Tuple[bool, str]:
        """Check epsilon_threshold = hbar^2/(2m l^2) from the condition epsilon_k t_diff/hbar ~ 1."""
        if self.hbar is None or self.m is None or self.l is None:
            return (False, "SKIP: hbar, m, or l missing")
        if self.epsilon_threshold is None:
            return (False, "epsilon_threshold not available")
        expected_eps = (self.hbar**2) / (2.0 * self.m * self.l**2)
        if math.isclose(self.epsilon_threshold, expected_eps, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"epsilon_threshold = {self.epsilon_threshold} matches hbar^2/(2ml^2) = {expected_eps}")
        else:
            return (False, f"epsilon_threshold = {self.epsilon_threshold} does not match hbar^2/(2ml^2) = {expected_eps}")

    # ---------------------------------------------------------------
    # Modeling-quality tests (based on explanation steps)
    # ---------------------------------------------------------------
    def test_modeling_t_diff_formula(self) -> Tuple[bool, str]:
        """Check that t_diff satisfies t_diff = 2 m l^2 / hbar."""
        if self.hbar is None or self.m is None or self.l is None:
            return (False, "SKIP: required givens missing")
        if self.t_diff is None:
            return (False, "t_diff not available")
        expected = 2.0 * self.m * self.l**2 / self.hbar
        if math.isclose(self.t_diff, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"t_diff = {self.t_diff} matches 2ml^2/hbar = {expected}")
        else:
            return (False, f"t_diff = {self.t_diff} does not match 2ml^2/hbar = {expected}")

    def test_modeling_epsilon_threshold_formula(self) -> Tuple[bool, str]:
        """Check that epsilon_threshold = hbar^2/(2 m l^2)."""
        if self.hbar is None or self.m is None or self.l is None:
            return (False, "SKIP: required givens missing")
        if self.epsilon_threshold is None:
            return (False, "epsilon_threshold not available")
        expected = self.hbar**2 / (2.0 * self.m * self.l**2)
        if math.isclose(self.epsilon_threshold, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"epsilon_threshold = {self.epsilon_threshold} matches hbar^2/(2ml^2) = {expected}")
        else:
            return (False, f"epsilon_threshold = {self.epsilon_threshold} does not match hbar^2/(2ml^2) = {expected}")

    def test_modeling_delay_string_content(self) -> Tuple[bool, str]:
        """Check that boundary_delay indicates a delayed (non-immediate) effect."""
        if self.boundary_delay is None:
            return (False, "boundary_delay not available")
        s = self.boundary_delay.lower()
        if "immediate" in s:
            return (False, "boundary_delay incorrectly indicates immediate effect")
        if "after" in s or "delay" in s or "time" in s:
            return (True, f"boundary_delay = '{self.boundary_delay}' indicates delayed effect")
        # Fallback: just check it's a non-empty string (weak)
        return (True, f"boundary_delay present: '{self.boundary_delay}'")

    # ---------------------------------------------------------------
    # Monotonicity tests (explicit perturbation of input)
    # ---------------------------------------------------------------
    def test_monotonicity_t_diff_increases_with_l(self) -> Tuple[bool, str]:
        """Perturb l upward and verify that t_diff (according to physics) increases."""
        if self.l is None or self.t_diff is None or self.hbar is None or self.m is None:
            return (False, "SKIP: required variables missing")
        # Baseline: use given l and t_diff
        l_base = self.l
        # Perturb l by +10%
        l_pert = l_base * 1.1
        # Compute expected t_diff for perturbed l from physics: t_diff = 2 m l^2 / hbar
        t_pert_expected = 2.0 * self.m * l_pert**2 / self.hbar
        # Compare to baseline t_diff (should be larger)
        if t_pert_expected > self.t_diff:
            return (True, f"Increasing l from {l_base:.3e} to {l_pert:.3e} increases t_diff from {self.t_diff:.3e} to {t_pert_expected:.3e}")
        else:
            return (False, f"Monotonicity violated: l increase did not lead to larger t_diff")

    def test_monotonicity_epsilon_threshold_decreases_with_l(self) -> Tuple[bool, str]:
        """Perturb l upward and verify that epsilon_threshold (according to physics) decreases."""
        if self.l is None or self.epsilon_threshold is None or self.hbar is None or self.m is None:
            return (False, "SKIP: required variables missing")
        # Baseline
        l_base = self.l
        # Perturb l by +10%
        l_pert = l_base * 1.1
        # Compute expected epsilon_threshold for perturbed l: epsilon_threshold = hbar^2/(2 m l^2)
        eps_pert_expected = self.hbar**2 / (2.0 * self.m * l_pert**2)
        # Compare to baseline epsilon_threshold (should be smaller)
        if eps_pert_expected < self.epsilon_threshold:
            return (True, f"Increasing l from {l_base:.3e} to {l_pert:.3e} decreases epsilon_threshold from {self.epsilon_threshold:.3e} to {eps_pert_expected:.3e}")
        else:
            return (False, f"Monotonicity violated: l increase did not lead to smaller epsilon_threshold")

    # ---------------------------------------------------------------
    # Weak answer sanity (only one, no math.isclose, no hardcoded targets)
    # ---------------------------------------------------------------
    def test_answer_sanity_boundary_delay(self) -> Tuple[bool, str]:
        """Very weak check: boundary_delay should be a non-empty string that does not say 'immediate'."""
        if self.boundary_delay is None:
            return (False, "boundary_delay not provided")
        if not isinstance(self.boundary_delay, str) or len(self.boundary_delay.strip()) == 0:
            return (False, "boundary_delay is empty or not a string")
        if "immediate" in self.boundary_delay.lower():
            return (False, "boundary_delay incorrectly suggests immediate effect")
        return (True, f"boundary_delay is non-empty string: '{self.boundary_delay}'")

    # ---------------------------------------------------------------
    # Main runner
    # ---------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test_ methods and run them, collecting results."""
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
                passed, msg = (False, f"{name} CRASHED: {traceback.format_exc()}")
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)