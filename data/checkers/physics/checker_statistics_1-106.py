import sympy
import math
import traceback

class PhysicsChecker:
    """
    Checker for the melting point of tungsten problem.
    Validates that the solution provides a reasonable numerical melting point.
    """

    def __init__(self, solution_instance):
        """
        Extract results safely with None defaults.
        """
        # Access results dict (if present)
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}

        # Extract T_melt with safe default
        self.T_melt = results.get('T_melt', None)

        # Optionally store the whole results for interface checks
        self.results = results

        # Known physical constant (tungsten melting point in Celsius)
        self.true_melting_point = 3422.0

        # Options given in the problem (used in modeling/monotonicity tests)
        self.options = [350, 3500, 35000, 350000]

    def _closest_option(self, value):
        """Return the option closest to a given value."""
        return min(self.options, key=lambda x: abs(x - value))

    # ----------------------------------------------------------------
    # INTERFACE TESTS
    # ----------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that results dict contains key 'T_melt'."""
        if not isinstance(self.results, dict):
            return False, "results is not a dict"
        if 'T_melt' not in self.results:
            return False, "key 'T_melt' missing in results"
        return True, "key 'T_melt' present"

    def test_interface_T_melt_numeric(self):
        """Check that T_melt is a numeric value (int or float)."""
        if self.T_melt is None:
            return False, "T_melt is None"
        if not isinstance(self.T_melt, (int, float)):
            return False, f"T_melt has type {type(self.T_melt)}, expected numeric"
        return True, "T_melt is numeric"

    # ----------------------------------------------------------------
    # INVARIANT TESTS (theorem-based physical sanity)
    # ----------------------------------------------------------------
    def test_invariant_positive(self):
        """Melting point must be positive."""
        if self.T_melt is None:
            return False, "SKIP: T_melt not available"
        if self.T_melt <= 0:
            return False, f"T_melt = {self.T_melt} is not positive"
        return True, "T_melt > 0"

    def test_invariant_reasonable_magnitude(self):
        """
        Melting point of tungsten should be in the thousands of degrees Celsius.
        Broad sanity: between 1000 and 10000 °C covers the known value (~3422 °C)
        while ruling out implausible options (350, 35000, 350000).
        """
        if self.T_melt is None:
            return False, "SKIP: T_melt not available"
        if not (1000 < self.T_melt < 10000):
            return False, f"T_melt = {self.T_melt} is outside expected magnitude range (1000–10000)"
        return True, "T_melt within 1000–10000 °C"

    # ----------------------------------------------------------------
    # MODELING / TRACE TESTS (at least two required)
    # ----------------------------------------------------------------
    def test_modeling_is_option(self):
        """
        Check that the answer is exactly one of the four given options.
        The problem requires selecting among 350, 3500, 35000, 350000 °C.
        """
        if self.T_melt is None:
            return False, "SKIP: T_melt not available"
        if self.T_melt not in self.options:
            return False, f"T_melt={self.T_melt} is not one of the provided options {self.options}"
        return True, f"T_melt={self.T_melt} is a valid option"

    def test_modeling_closest_to_known(self):
        """
        The answer should be the option closest to the known true melting point of tungsten (3422°C),
        as stated in the explanation steps.
        """
        if self.T_melt is None:
            return False, "SKIP: T_melt not available"

        true_val = self.true_melting_point
        # Compute distances from each option to true_val
        distances = {opt: abs(opt - true_val) for opt in self.options}
        # The answer should have the smallest distance
        best_option = min(distances, key=distances.get)
        if self.T_melt != best_option:
            return False, f"T_melt={self.T_melt} is not the option closest to {true_val} (closest is {best_option})"
        # Also check that its distance is strictly smaller than others
        best_dist = distances[best_option]
        for opt, d in distances.items():
            if opt != best_option and d <= best_dist:
                return False, f"T_melt={self.T_melt} distance {best_dist} is not strictly smaller than option {opt} distance {d}"
        return True, f"T_melt={self.T_melt} is the closest option to known melting point {true_val}"

    # ----------------------------------------------------------------
    # MONOTONICITY TEST
    # ----------------------------------------------------------------
    def test_monotonicity_answer_vs_true_value(self):
        """
        Perturb the true melting point upward and verify the chosen answer does not decrease.
        Baseline: true value = 3422 -> answer = 3500.
        Perturbations: +1% (~3456) and +5% (~3593).
        The answer should remain 3500 (non-decreasing).
        """
        if self.T_melt is None:
            return False, "SKIP: T_melt not available"

        baseline_true = self.true_melting_point
        baseline_answer = self._closest_option(baseline_true)

        # Perturbations (positive relative changes)
        perturbations = [0.01, 0.05]  # +1%, +5%
        for p in perturbations:
            perturbed_true = baseline_true * (1 + p)
            perturbed_answer = self._closest_option(perturbed_true)
            if perturbed_answer < baseline_answer:
                msg = (f"Answer decreased when true value increased by {p*100}%: "
                       f"baseline answer {baseline_answer}, perturbed answer {perturbed_answer}")
                return False, msg
        return True, "Answer is non‑decreasing under small positive perturbations of the true melting point"

    # ----------------------------------------------------------------
    # WEAK ANSWER SANITY (only one such method allowed)
    # ----------------------------------------------------------------
    def test_answer_sanity_plausible_choice(self):
        """
        Weak final‑answer sanity: T_melt should be a reasonable integer choice.
        Only sign and order‑of‑magnitude checks; no isclose or re‑derivation.
        """
        if self.T_melt is None:
            return False, "SKIP: T_melt not available"
        # Ensure it's not an extreme outlier
        if self.T_melt < 2000 or self.T_melt > 5000:
            return False, f"T_melt = {self.T_melt} seems implausible (known value ~3422)"
        return True, "T_melt is plausible"

    # ----------------------------------------------------------------
    # RUN ALL TESTS
    # ----------------------------------------------------------------
    def run_all_tests(self):
        """
        Discover and run all test_* methods.
        Returns (success_bool, logs_list).
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)