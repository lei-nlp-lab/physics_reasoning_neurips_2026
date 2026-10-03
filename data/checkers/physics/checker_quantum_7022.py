import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Audits a solution for the problem:
    "Ratio of intensities of adjacent rotational lines in N2 spectrum = 2:1"
    based on nuclear spin statistics for identical bosons (I=1).
    """

    def __init__(self, solution_instance):
        # Safely extract inputs and results
        self.results = self._get_results(solution_instance)
        self.I = self._get_input(solution_instance, 'I', None)
        self.ratio = self.results.get('ratio', None)

    def _get_results(self, solution_instance):
        """Return the results dictionary from solution_instance."""
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            return solution_instance.results
        elif isinstance(solution_instance, dict):
            return solution_instance
        else:
            return {}

    def _get_input(self, solution_instance, key, default=None):
        """Return an input parameter, regardless of solution_instance format."""
        if hasattr(solution_instance, key):
            return getattr(solution_instance, key)
        elif isinstance(solution_instance, dict):
            return solution_instance.get(key, default)
        else:
            return default

    # ------------------------------------------------------------------
    # INTERFACE CHECKS
    # ------------------------------------------------------------------
    def test_interface_ratio_exists(self):
        if self.ratio is None:
            return (False, "SKIP: 'ratio' not found in results")
        return (True, "'ratio' key present in results")

    def test_interface_ratio_is_numeric(self):
        if self.ratio is None:
            return (False, "SKIP: 'ratio' not found")
        if not isinstance(self.ratio, (int, float)):
            return (False, f"'ratio' is not numeric, got {type(self.ratio).__name__}")
        if math.isnan(self.ratio) or math.isinf(self.ratio):
            return (False, f"'ratio' is {self.ratio}")
        return (True, "'ratio' is a finite number")

    # ------------------------------------------------------------------
    # INVARIANT CHECKS (theorem-based sanity)
    # ------------------------------------------------------------------
    def test_invariant_nuclear_spin_degeneracies(self):
        """
        Check that the number of symmetric (g_s) and antisymmetric (g_a)
        nuclear spin states for two spin-I particles satisfy:
            g_s = (2I+1)(I+1)
            g_a = (2I+1)I
        and that they sum to (2I+1)^2.
        """
        if self.I is None:
            return (False, "SKIP: I (nuclear spin) not provided")

        total = (2 * self.I + 1) ** 2
        g_s = (2 * self.I + 1) * (self.I + 1)
        g_a = (2 * self.I + 1) * self.I

        if not math.isclose(g_s + g_a, total, rel_tol=1e-2):
            msg = f"g_s + g_a = {g_s + g_a} != (2I+1)^2 = {total}"
            return (False, msg)
        return (True, f"g_s + g_a = {total} (correct)")

    def test_invariant_ratio_from_spin(self):
        """
        Verify that the ratio (even/odd) equals (I+1)/I using the spin-degeneracy formulas.
        """
        if self.I is None:
            return (False, "SKIP: I not provided")
        if self.ratio is None:
            return (False, "SKIP: ratio not in results")

        expected = (self.I + 1) / self.I
        if not math.isclose(self.ratio, expected, rel_tol=1e-2, abs_tol=1e-2):
            msg = f"ratio = {self.ratio}, expected ~{expected} from (I+1)/I"
            return (False, msg)
        return (True, f"ratio {self.ratio} matches theoretical value (I+1)/I = {expected}")

    # ------------------------------------------------------------------
    # MODELING / TRACE CHECKS
    # ------------------------------------------------------------------
    def test_modeling_total_states(self):
        """
        For I=1, there should be 9 total nuclear spin states (3x3 = 9).
        This checks that the underlying physical model respects the total state count.
        """
        if self.I is None:
            return (False, "SKIP: I not provided")
        total_states = (2 * self.I + 1) ** 2
        if total_states <= 0 or not isinstance(total_states, int):
            return (False, f"total nuclear spin states invalid: {total_states}")
        if self.I == 1 and total_states != 9:
            # For I=1 specifically, we can assert exact value because it's derived, not copied
            return (False, f"For I=1, total states should be 9, got {total_states}")
        return (True, f"Total nuclear spin states = {total_states} (for I={self.I})")

    def test_modeling_ratio_via_g_s_g_a(self):
        """
        Modeling test: The ratio should be derived as g_s/g_a.
        We re-derive g_s and g_a from I and check if the reported ratio is consistent with the
        expected ratio using those degeneracies. This is a quality test on the modeling steps.
        """
        if self.I is None:
            return (False, "SKIP: I not provided")
        if self.ratio is None:
            return (False, "SKIP: ratio not provided")

        g_s = (2 * self.I + 1) * (self.I + 1)
        g_a = (2 * self.I + 1) * self.I
        expected_ratio = g_s / g_a
        if not math.isclose(self.ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            msg = f"ratio {self.ratio} does not match expected {expected_ratio} from g_s/g_a"
            return (False, msg)
        return (True, f"ratio {self.ratio} equals g_s/g_a = {expected_ratio}")

    # ------------------------------------------------------------------
    # MONOTONICITY CHECK (trend under parameter perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_ratio_decreases_with_spin(self):
        """
        The ratio g_s/g_a = (I+1)/I is a decreasing function of I.
        If we perturb I by a small positive amount, the ratio should decrease.
        This is a physical trend that holds for any I > 0.
        """
        if self.I is None:
            return (False, "SKIP: I not provided")
        baseline = (self.I + 1) / self.I
        delta = 0.1
        new_I = self.I + delta
        new_ratio = (new_I + 1) / new_I
        if new_ratio >= baseline:
            msg = f"Ratio {new_ratio} for perturbed I={new_I} is not less than baseline {baseline}"
            return (False, msg)
        return (True, f"Ratio decreases from {baseline} to {new_ratio} upon increasing I by {delta}")

    # ------------------------------------------------------------------
    # WEAK ANSWER SANITY (no strict matching)
    # ------------------------------------------------------------------
    def test_answer_sanity_ratio_basic(self):
        """Check that the ratio is positive, finite, and in a plausible range (e.g., between 1 and 10)."""
        if self.ratio is None:
            return (False, "SKIP: ratio not in results")
        if not isinstance(self.ratio, (int, float)):
            return (False, "ratio is not numeric")
        if math.isnan(self.ratio) or math.isinf(self.ratio) or self.ratio <= 0:
            return (False, f"ratio {self.ratio} is invalid (must be positive finite)")
        if self.ratio > 10 or self.ratio < 1:
            # Extremely unlikely for any diatomic with integer spin, but still possible?
            return (True, f"ratio = {self.ratio} (outside typical range 1-10, but still possible)")
        return (True, f"ratio = {self.ratio} (positive, finite, plausible)")

    # ------------------------------------------------------------------
    # RUN ALL TESTS
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Get all bound methods starting with test_
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


# Example usage:
if __name__ == "__main__":
    # Dummy solution instance (simulating what might be passed)
    class DummySolution:
        def __init__(self):
            self.I = 1
            self.results = {'ratio': 2.0}

    checker = PhysicsChecker(DummySolution())
    success, log = checker.run_all_tests()
    for line in log:
        print(line)
    print("Overall success:", success)