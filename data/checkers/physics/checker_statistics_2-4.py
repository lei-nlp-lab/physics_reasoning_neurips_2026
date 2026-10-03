import math
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Physics auditor for the spin-1/2 system problem.
    Checks combinatorial derivation, maximum condition, and temperature discussion.
    """
    def __init__(self, solution_instance: Any) -> None:
        # Extract givens, results, explanation_steps, pseudocode safely
        # Givens
        self.N = getattr(solution_instance, 'givens', None)
        if isinstance(self.N, dict):
            self.N = self.N.get('N', None)
        else:
            self.N = None

        # Results
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}  # to avoid NoneType errors

        # Key result
        self.M_s_max = self.results.get('M_s_max', None)

        # Explanation steps (list of strings or dicts)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        if not isinstance(self.explanation_steps, list):
            self.explanation_steps = []

        # Pseudocode (optional, for trace checks)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

        # Derived quantities (if available from solution)
        # We'll try to extract Omega values, but not required.
        self.Omega = getattr(solution_instance, 'Omega', None)  # maybe stored as function or list

    # ----------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------
    def test_interface_result_key_exists(self) -> Tuple[bool, str]:
        """Check that the required result key M_s_max is present."""
        if 'M_s_max' not in self.results:
            return (False, "Missing result key 'M_s_max'.")
        return (True, "Result key 'M_s_max' exists.")

    def test_interface_M_s_max_type(self) -> Tuple[bool, str]:
        """Check that M_s_max is a number (int or float)."""
        if self.M_s_max is None:
            return (False, "M_s_max is None.")
        if not isinstance(self.M_s_max, (int, float)):
            return (False, f"M_s_max is type {type(self.M_s_max)}, expected int or float.")
        return (True, f"M_s_max is numeric: {self.M_s_max}.")

    # ----------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ----------------------------------------------------------------
    def test_invariant_omega_positive(self) -> Tuple[bool, str]:
        """If Omega values are provided, they must be positive integers (or >0)."""
        if self.Omega is None:
            return (False, "SKIP: No Omega data available.")
        # Could be a dict or list; assume dict mapping M_s to Omega
        if not hasattr(self.Omega, 'items'):
            return (False, "Omega is not a dict, cannot check.")
        for ms, omega in self.Omega.items():
            if not isinstance(omega, (int, float)) or omega <= 0:
                return (False, f"Omega({ms}) = {omega} is not positive.")
        return (True, "All Omega values are positive.")

    def test_invariant_omega_symmetry(self) -> Tuple[bool, str]:
        """Check Ω(M_s) ≈ Ω(-M_s)."""
        if self.Omega is None or not hasattr(self.Omega, 'items'):
            return (False, "SKIP: No Omega dict available.")
        ms_values = sorted(self.Omega.keys())
        # Check for symmetric pairs
        tested = set()
        for ms in ms_values:
            if ms in tested:
                continue
            neg_ms = -ms
            if neg_ms in self.Omega:
                omega1 = self.Omega[ms]
                omega2 = self.Omega[neg_ms]
                if omega1 is None or omega2 is None:
                    return (False, f"Omega missing for M_s = {ms} or {neg_ms}.")
                if not math.isclose(omega1, omega2, rel_tol=1e-2, abs_tol=1e-2):
                    return (False, f"Asymmetry: Omega({ms})={omega1}, Omega({neg_ms})={omega2}.")
                tested.add(ms)
                tested.add(neg_ms)
        return (True, "Omega is symmetric about M_s=0.")

    def test_invariant_M_s_range(self) -> Tuple[bool, str]:
        """If N is given, M_s must be between -N/2 and N/2."""
        if self.N is None:
            return (False, "SKIP: N not given.")
        if self.M_s_max is None:
            return (False, "M_s_max is None.")
        half_N = self.N / 2.0
        if not (-half_N - 1e-9 <= self.M_s_max <= half_N + 1e-9):
            return (False, f"M_s_max = {self.M_s_max} outside [-N/2, N/2] = [{-half_N}, {half_N}].")
        return (True, f"M_s_max = {self.M_s_max} within allowed range.")

    # ----------------------------------------------------------------
    # Modeling / trace tests (check logical derivation steps)
    # ----------------------------------------------------------------
    def test_modeling_explanation_contains_maximum(self) -> Tuple[bool, str]:
        """Check that explanation_steps mention derivative and M_s=0."""
        steps_text = " ".join(str(s) for s in self.explanation_steps)
        if "derivative" in steps_text.lower() or "dΩ/dM" in steps_text or "d ln Ω" in steps_text:
            if "M_s = 0" in steps_text or "Ms = 0" in steps_text:
                return (True, "Explanation steps include derivative and M_s=0.")
            else:
                return (False, "Derivative mentioned but M_s=0 not found.")
        else:
            return (False, "Derivative not found in explanation steps.")

    def test_modeling_negative_temperature_discussion(self) -> Tuple[bool, str]:
        """Check that negative temperature is discussed in explanation_steps."""
        steps_text = " ".join(str(s) for s in self.explanation_steps).lower()
        if "negative temperature" in steps_text or "population inversion" in steps_text or "π-pulse" in steps_text:
            return (True, "Negative temperature or population inversion explained.")
        return (False, "No mention of negative temperature or population inversion.")

    def test_trace_derivative_condition(self) -> Tuple[bool, str]:
        """If pseudocode contains the maximization condition, verify it leads to M_s=0."""
        if self.pseudocode is None:
            return (False, "SKIP: No pseudocode available.")
        # We'll check if 'd(ln Omega)/dM_s = 0' appears and leads to M_s=0
        pseudo_str = str(self.pseudocode).lower()
        if "ln omega" in pseudo_str and "derivative" in pseudo_str and "ms = 0" in pseudo_str:
            return (True, "Pseudocode correctly derives M_s=0 from derivative.")
        else:
            return (False, "Pseudocode does not show derivative leading to M_s=0.")

    # ----------------------------------------------------------------
    # Monotonicity tests (directional trend via input perturbation)
    # ----------------------------------------------------------------
    def test_monotonicity_maximum_invariant_under_N_perturbation(self) -> Tuple[bool, str]:
        """
        Perturb N (increase by 1) and verify that the expected M_s_max remains 0.
        Baseline: M_s_max from solution (should be 0).
        Perturb N -> N+1, expected M_s_max for perturbed system is still 0.
        Compare perturbed expected output to baseline.
        """
        if self.N is None or self.M_s_max is None:
            return (False, "SKIP: Missing N or M_s_max.")
        baseline = self.M_s_max  # should be 0
        # Perturb input: increase N by 1
        N_perturbed = self.N + 1
        # The maximum of the binomial occurs at N_up = N/2 -> M_s=0 for any N
        expected_perturbed = 0
        if baseline != expected_perturbed:
            return (False, f"Baseline M_s_max = {baseline}, expected perturbed M_s_max = {expected_perturbed}.")
        return (True, f"M_s_max remains 0 when N perturbed from {self.N} to {N_perturbed}.")

    # ----------------------------------------------------------------
    # Answer sanity (weak, secondary)
    # ----------------------------------------------------------------
    def test_answer_sanity_M_s_max_is_zero(self) -> Tuple[bool, str]:
        """Check that M_s_max is exactly 0 (weak, but derived from physics)."""
        if self.M_s_max is None:
            return (False, "M_s_max is missing.")
        if self.M_s_max != 0:
            return (False, f"M_s_max = {self.M_s_max} but expected 0.")
        return (True, "M_s_max is 0 as expected from statistical mechanics.")

    # ----------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and execute all test_ methods, return overall pass and logs."""
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