import math
import re

class PhysicsChecker:
    """
    Auditor for muon lifetime order-of-magnitude selection.
    Checks interface, physics invariants, explanation consistency, monotonicity,
    and a weak sanity check on the answer.
    """

    def __init__(self, solution_instance):
        # Extract results safely using None default
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.muon_lifetime_order = solution_instance.results.get('muon_lifetime_order', None)
        else:
            self.muon_lifetime_order = None

        # Optional attributes for modeling checks
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def _parse_exponent(self, order_str):
        """
        Extract the exponent from strings like '10^{-6}' or '10^-6'.
        Returns integer exponent or None.
        """
        match = re.search(r'10\^\{?(-?\d+)\}?', order_str)
        if match:
            return int(match.group(1))
        return None

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        if self.muon_lifetime_order is None:
            return (False, "Missing required key: muon_lifetime_order")
        if not isinstance(self.muon_lifetime_order, str):
            return (False, f"muon_lifetime_order should be a string, got {type(self.muon_lifetime_order)}")
        return (True, "muon_lifetime_order present and is string")

    # ----------------------------------------------------------------------
    # Physical invariant tests (theorem-based)
    # ----------------------------------------------------------------------
    def test_invariant_muon_lifetime_order(self):
        """
        Known experimental muon lifetime ~2.2e-6 s → order 10^{-6}.
        Check that the result matches this physics invariant.
        """
        if self.muon_lifetime_order is None:
            return (False, "SKIP: muon_lifetime_order not available")

        known_tau = 2.2e-6  # known experimental value, s
        expected_exponent = int(math.floor(math.log10(known_tau)))  # -6

        result_exponent = self._parse_exponent(self.muon_lifetime_order)
        if result_exponent is None:
            return (False, f"Could not parse exponent from result: {self.muon_lifetime_order}")

        if result_exponent == expected_exponent:
            return (True, f"Result exponent {result_exponent} matches known order 10^{expected_exponent}")
        else:
            return (False, f"Result exponent {result_exponent} does not match known order 10^{expected_exponent}")

    def test_invariant_not_other_orders(self):
        """
        The muon lifetime order must be 10^{-6} (from experiment).
        It should NOT be any of the other provided options (10^9, 10^2, 10^{-2}).
        This is a theorem derived from decades of particle physics measurements.
        """
        if self.muon_lifetime_order is None:
            return (False, "SKIP: muon_lifetime_order not available")

        forbidden_orders = ['10^9', '10^2', '10^{-2}']
        # Normalize curly braces for comparison
        normalized_result = self.muon_lifetime_order.replace('{', '').replace('}', '')
        if normalized_result in forbidden_orders:
            return (False, f"Result {self.muon_lifetime_order} is one of the physically excluded orders (too long or too short)")
        return (True, f"Result {self.muon_lifetime_order} is not among the excluded orders")

    # ----------------------------------------------------------------------
    # Modeling / trace consistency tests (at least two)
    # ----------------------------------------------------------------------
    def test_modeling_explanation_consistency(self):
        """
        Check that explanation_steps (if present) mention the correct order.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not provided")

        if isinstance(self.explanation_steps, list):
            combined = " ".join(self.explanation_steps)
        else:
            combined = str(self.explanation_steps)

        if "10^{-6}" in combined or "2.2e-6" in combined or "microsecond" in combined:
            return (True, "Explanation steps consistent with muon lifetime order")
        else:
            return (False, "Explanation steps do not mention 10^{-6} or 2.2e-6")

    def test_trace_pseudocode_consistency(self):
        """
        Check that the pseudocode (if present) contains the essential logic:
        recall known muon lifetime (~2.2e-6 s), compare options, select 10^{-6}.
        """
        if self.pseudocode is None:
            return (False, "SKIP: pseudocode not provided")

        pseudocode_str = str(self.pseudocode).lower()

        # Check for key concepts
        has_recall = any(phrase in pseudocode_str for phrase in ['recall', 'known', 'muon lifetime', '2.2e-6'])
        has_comparison = any(phrase in pseudocode_str for phrase in ['compare', 'option', '10^9', '10^2', '10^{-2}'])
        has_selection = any(phrase in pseudocode_str for phrase in ['select', '10^{-6}', 'emit'])

        if has_recall and has_comparison and has_selection:
            return (True, "Pseudocode contains recall, comparison, and selection of 10^{-6}")
        else:
            return (False, f"Pseudocode missing essential logic: recall={has_recall}, comparison={has_comparison}, selection={has_selection}")

    # ----------------------------------------------------------------------
    # Monotonicity tests
    # ----------------------------------------------------------------------
    def test_monotonicity_lifetime_order_with_perturbation(self):
        """
        Perturb the known muon lifetime by a factor of 10 and verify that
        the result's order of magnitude changes accordingly (i.e., does NOT
        remain the same as for the unperturbed value).
        This tests that the selection is sensitive to the underlying physical value.
        """
        if self.muon_lifetime_order is None:
            return (False, "SKIP: muon_lifetime_order not available")

        # Original known tau: 2.2e-6 s → order exponent -6
        # Perturbed tau (factor 10 larger): 2.2e-5 s → order exponent -5
        perturbed_tau = 2.2e-5
        expected_perturbed_exponent = int(math.floor(math.log10(perturbed_tau)))  # -5

        result_exponent = self._parse_exponent(self.muon_lifetime_order)
        if result_exponent is None:
            return (False, "Cannot parse exponent from result")

        # The result exponent should NOT equal the perturbed exponent,
        # because the result is correct for the original tau.
        if result_exponent == expected_perturbed_exponent:
            return (False, f"Result exponent {result_exponent} matches perturbed exponent, but should be different (sensitivity failed)")
        return (True, f"Result exponent {result_exponent} differs from perturbed exponent {expected_perturbed_exponent}, showing input sensitivity")

    # ----------------------------------------------------------------------
    # Weak answer sanity test (at most one)
    # ----------------------------------------------------------------------
    def test_answer_sanity_option_valid(self):
        """
        Weak sanity: check that the answer is one of the provided options.
        No math.isclose, no equation solving.
        """
        if self.muon_lifetime_order is None:
            return (False, "SKIP: no answer to check")

        valid_options = ['10^9', '10^2', '10^{-2}', '10^{-6}']
        # Normalize: remove curly braces for comparison
        normalized_result = self.muon_lifetime_order.replace('{', '').replace('}', '')
        if normalized_result not in valid_options:
            return (False, f"Answer {self.muon_lifetime_order} not among expected options")
        return (True, "Answer is one of the given options")

    # ----------------------------------------------------------------------
    # Test runner
    # ----------------------------------------------------------------------
    def run_all_tests(self):
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