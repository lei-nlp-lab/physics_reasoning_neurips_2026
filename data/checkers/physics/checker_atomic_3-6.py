import math
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Audits solution instances for conservation law problems.
    Checks interface completeness, invariance based on standard physics,
    modeling consistency, monotonicity (via list-size ordering), and
    provides a weak answer sanity check.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract results dictionary safely
        if isinstance(solution_instance, dict):
            self.results = solution_instance
        else:
            self.results = getattr(solution_instance, 'results', None)
            if self.results is None:
                try:
                    self.results = dict(vars(solution_instance))
                except TypeError:
                    self.results = {}

        # Safely extract all expected keys
        self.Ispin_interactions: Optional[List[str]] = self.results.get('Ispin_interactions', None)
        self.I3_interactions: Optional[List[str]] = self.results.get('I3_interactions', None)
        self.strangeness_interactions: Optional[List[str]] = self.results.get('strangeness_interactions', None)
        self.CP_interactions: Optional[List[str]] = self.results.get('CP_interactions', None)
        self.CP_violation_experiment: Optional[str] = self.results.get('CP_violation_experiment', None)

        # Ordered list of law keys for iteration
        self._law_keys_ordered = [
            'Ispin_interactions',
            'I3_interactions',
            'strangeness_interactions',
            'CP_interactions'
        ]
        self._law_names = ['I-spin', 'I3', 'strangeness', 'CP']

    # ----------------------------------------------------------------------
    # Interface tests: existence and type checks
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        missing = []
        for key in self._law_keys_ordered + ['CP_violation_experiment']:
            if self.results.get(key) is None:
                missing.append(key)
        if missing:
            return False, f"Missing keys: {missing}"
        return True, "All required keys present."

    def test_interface_value_types(self) -> Tuple[bool, str]:
        for key in self._law_keys_ordered:
            val = self.results.get(key)
            if val is None:
                return False, f"SKIP: {key} is None, cannot check type."
            if not isinstance(val, list):
                return False, f"{key} should be a list, got {type(val)}."
            for item in val:
                if not isinstance(item, str):
                    return False, f"Item in {key} is not a string: {item}."
        cp = self.results.get('CP_violation_experiment')
        if cp is not None and not isinstance(cp, str):
            return False, f"CP_violation_experiment should be a string, got {type(cp)}."
        return True, "All value types are correct."

    # ----------------------------------------------------------------------
    # Invariant tests: physical sanity based on known conservation laws
    # ----------------------------------------------------------------------
    def test_invariant_Ispin_only_strong(self) -> Tuple[bool, str]:
        if self.Ispin_interactions is None:
            return False, "SKIP: Ispin_interactions not provided."
        expected = ['S']
        if set(self.Ispin_interactions) != set(expected):
            return False, f"Ispin_interactions should be {expected}, got {self.Ispin_interactions}."
        return True, "Ispin conserved only in strong interactions (S)."

    def test_invariant_I3_strong_and_em(self) -> Tuple[bool, str]:
        if self.I3_interactions is None:
            return False, "SKIP: I3_interactions not provided."
        expected = {'S', 'E'}
        actual = set(self.I3_interactions)
        if actual != expected:
            return False, f"I3_interactions should be {expected}, got {actual}."
        return True, "I3 conserved in strong and electromagnetic interactions (S, E)."

    def test_invariant_strangeness_strong_and_em(self) -> Tuple[bool, str]:
        if self.strangeness_interactions is None:
            return False, "SKIP: strangeness_interactions not provided."
        expected = {'S', 'E'}
        actual = set(self.strangeness_interactions)
        if actual != expected:
            return False, f"strangeness_interactions should be {expected}, got {actual}."
        return True, "Strangeness conserved in strong and electromagnetic interactions (S, E)."

    def test_invariant_CP_strong_and_em(self) -> Tuple[bool, str]:
        if self.CP_interactions is None:
            return False, "SKIP: CP_interactions not provided."
        expected = {'S', 'E'}
        actual = set(self.CP_interactions)
        if actual != expected:
            return False, f"CP_interactions should be {expected}, got {actual}."
        return True, "CP conserved in strong and electromagnetic interactions (S, E)."

    def test_invariant_weak_not_included(self) -> Tuple[bool, str]:
        """Weak interactions should never appear in the 'no violation' lists."""
        for key in self._law_keys_ordered:
            lst = self.results.get(key)
            if lst is None:
                continue
            if 'W' in lst:
                return False, f"W (weak) should not be in '{key}' list, but found: {lst}."
        return True, "Weak interactions correctly excluded from all conservation lists."

    # ----------------------------------------------------------------------
    # Modeling/trace tests: based on explanation steps and known experiment
    # ----------------------------------------------------------------------
    def test_modeling_CP_violation_experiment(self) -> Tuple[bool, str]:
        """The CP_violation_experiment should mention the Cronin-Fitch experiment and K_L decay."""
        if self.CP_violation_experiment is None:
            return False, "SKIP: CP_violation_experiment not provided."
        msg = self.CP_violation_experiment.lower()
        has_cronin = 'cronin' in msg
        has_fitch = 'fitch' in msg
        has_kl = 'k_l' in msg or 'k0_l' in msg or 'kaon' in msg
        has_violation = 'violation' in msg
        if not (has_cronin or has_fitch):
            return False, "CP_violation_experiment should reference Cronin or Fitch."
        if not has_kl:
            return False, "CP_violation_experiment should mention K_L decay."
        if not has_violation:
            return False, "CP_violation_experiment should mention CP violation."
        return True, "CP_violation_experiment adequately describes the Cronin-Fitch experiment."

    def test_modeling_conservation_set_hierarchy(self) -> Tuple[bool, str]:
        """
        Verify that the sets of interactions (no-violation) obey a logical hierarchy:
        Ispin (S) ⊆ I3 (S,E) ⊆ Strangeness (S,E) ⊆ CP (S,E).
        This is a modeling quality check consistent with standard physics.
        """
        sets = {}
        for key in self._law_keys_ordered:
            lst = self.results.get(key)
            if lst is None:
                return False, f"SKIP: {key} is missing, cannot check hierarchy."
            sets[key] = set(lst)

        # Ispin should be subset of I3
        if not sets['Ispin_interactions'].issubset(sets['I3_interactions']):
            return False, "Ispin set is not subset of I3 set."
        # I3 should be subset of strangeness
        if not sets['I3_interactions'].issubset(sets['strangeness_interactions']):
            return False, "I3 set is not subset of strangeness set."
        # Strangeness should be subset of CP
        if not sets['strangeness_interactions'].issubset(sets['CP_interactions']):
            return False, "Strangeness set is not subset of CP set."
        return True, "Conservation sets obey the expected hierarchy: Ispin ⊆ I3 ⊆ Strangeness ⊆ CP."

    # ----------------------------------------------------------------------
    # Monotonicity test: verifies list lengths are non-decreasing
    # ----------------------------------------------------------------------
    def test_monotonicity_list_lengths(self) -> Tuple[bool, str]:
        """
        Verify that the number of interaction types per law is non-decreasing
        from (a) I-spin to (d) CP. Perturb the law index by +1 and compare
        the list lengths as a monotonic property of the law ordering.
        """
        lengths = []
        for key in self._law_keys_ordered:
            lst = self.results.get(key)
            if lst is None:
                return False, f"SKIP: {key} is None, cannot check monotonicity."
            lengths.append(len(lst))

        for i in range(len(lengths) - 1):
            if lengths[i] > lengths[i+1]:
                return False, (
                    f"Monotonicity violated at index {i}: "
                    f"length of {self._law_names[i]} ({lengths[i]}) > "
                    f"length of {self._law_names[i+1]} ({lengths[i+1]})."
                )
        return True, "List lengths are non-decreasing (monotonic)."

    # ----------------------------------------------------------------------
    # Weak answer sanity test (only one allowed)
    # ----------------------------------------------------------------------
    def test_answer_sanity_basic(self) -> Tuple[bool, str]:
        """Very minimal checks: lists non-empty, no unexpected characters, experiment non-empty."""
        for key in self._law_keys_ordered:
            lst = self.results.get(key)
            if lst is None:
                return False, f"SKIP: {key} is missing."
            if len(lst) == 0:
                return False, f"{key} is empty, but should have at least one interaction type."
            allowed = {'S', 'E', 'W'}
            for item in lst:
                if item not in allowed:
                    return False, f"Unexpected interaction type '{item}' in {key}."
        exp = self.results.get('CP_violation_experiment')
        if exp is None or not isinstance(exp, str) or len(exp.strip()) == 0:
            return False, "CP_violation_experiment should be a non-empty string."
        return True, "Basic answer sanity passed."

    # ----------------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {type(e).__name__}: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)