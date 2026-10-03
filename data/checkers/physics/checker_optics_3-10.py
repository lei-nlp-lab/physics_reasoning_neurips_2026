import math
import traceback

class PhysicsChecker:
    """Auditor for PMT solution: checks interface, invariants, modeling, monotonicity, and final answer sanity."""

    def __init__(self, solution_instance):
        # Extract givens, results, explanation steps, and pseudocode safely.
        # Use None for missing values; never numeric defaults.
        self.givens = getattr(solution_instance, 'givens', None)
        if self.givens is None:
            self.givens = {}
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

        # Extract specific keys with None defaults.
        self.n = self.givens.get('n', None)
        self.delta = self.givens.get('delta', None)
        self.FWHM_rel = self.results.get('FWHM_rel', None)

    def run_all_tests(self):
        """Discover and run all test_* methods, collect results."""
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs

    # ----------------------------------------------------------------------
    # Interface tests: required keys exist and have correct types
    # ----------------------------------------------------------------------
    def test_interface_results_dict(self):
        if self.results is None:
            return False, "results is missing (None)"
        if not isinstance(self.results, dict):
            return False, "results is not a dict"
        return True, "results exists and is a dict"

    def test_interface_givens_dict(self):
        if self.givens is None:
            return False, "givens is missing (None)"
        if not isinstance(self.givens, dict):
            return False, "givens is not a dict"
        return True, "givens exists and is a dict"

    def test_interface_required_result_keys(self):
        if self.FWHM_rel is None:
            return False, "results['FWHM_rel'] is missing or None"
        if not isinstance(self.FWHM_rel, (int, float)):
            return False, "results['FWHM_rel'] is not numeric"
        return True, "results['FWHM_rel'] present and numeric"

    def test_interface_required_given_keys(self):
        missing = []
        if self.n is None:
            missing.append("n")
        if self.delta is None:
            missing.append("delta")
        if missing:
            return False, f"givens missing: {', '.join(missing)}"
        return True, "givens['n'] and givens['delta'] present"

    # ----------------------------------------------------------------------
    # Invariant tests: theorem-based physical sanity (use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_delta_gt_one(self):
        if self.delta is None:
            return False, "SKIP: delta not available"
        if self.delta <= 1:
            return False, f"delta <= 1 (delta={self.delta}) violates secondary emission requirement"
        return True, f"delta > 1 (delta={self.delta})"

    def test_invariant_n_positive_integer(self):
        if self.n is None:
            return False, "SKIP: n not available"
        if self.n < 1 or not isinstance(self.n, int):
            return False, f"n is not a positive integer (n={self.n})"
        return True, f"n is a positive integer (n={self.n})"

    def test_invariant_gain_greater_one(self):
        if self.n is None or self.delta is None:
            return False, "SKIP: n or delta not available"
        try:
            expected_gain = self.delta ** self.n
        except (ValueError, TypeError):
            return False, "Could not compute gain"
        if expected_gain <= 1:
            return False, f"Gain = {expected_gain} is <= 1"
        return True, f"Gain = {expected_gain} > 1"

    # ----------------------------------------------------------------------
    # Modeling / trace tests: check consistency with physics relationships
    # ----------------------------------------------------------------------
    def test_modeling_fwhm_rel_formula(self):
        """Check that FWHM_rel matches the theoretical formula from given n and delta."""
        if self.n is None or self.delta is None or self.FWHM_rel is None:
            return False, "SKIP: n, delta, or FWHM_rel not available"
        try:
            gain = self.delta ** self.n
        except (ValueError, TypeError):
            return False, "Could not compute gain from given n, delta"
        if gain <= 0:
            return False, f"Gain = {gain} is non-positive"
        # Expected FWHM_rel from explanation steps
        expected = 2.0 * math.sqrt(2.0 * math.log(2) / gain)
        rel_tol = 1e-2
        if not math.isclose(self.FWHM_rel, expected, rel_tol=rel_tol, abs_tol=1e-3):
            return False, f"FWHM_rel = {self.FWHM_rel} not close to expected {expected} (rel_tol={rel_tol})"
        return True, f"FWHM_rel {self.FWHM_rel} matches formula {expected}"

    def test_modeling_fwhm_rel_derived_from_explanation(self):
        """Verify that explanation steps mention the FWHM relation (optional check)."""
        if self.explanation_steps is None:
            return False, "SKIP: explanation_steps not provided"
        if isinstance(self.explanation_steps, str):
            text = self.explanation_steps
        elif hasattr(self.explanation_steps, '__iter__'):
            text = ' '.join(self.explanation_steps)
        else:
            return False, "explanation_steps has unexpected type"
        # Look for key patterns
        patterns = ["ΔN", "ΔN / N", "FWHM", "2√(2ln2/5^10)", "2*sqrt(2*ln(2)/G)"]
        found = any(p in text for p in patterns)
        if not found:
            return False, "Explanation steps do not contain expected FWHM relation pattern"
        return True, "Explanation steps contain FWHM relation pattern"

    # ----------------------------------------------------------------------
    # Monotonicity tests: perturb an input and check trend
    # ----------------------------------------------------------------------
    def test_monotonicity_n_decreases_fwhm(self):
        """Increasing n (number of dynodes) should decrease FWHM_rel for fixed delta."""
        if self.n is None or self.delta is None or self.FWHM_rel is None:
            return False, "SKIP: n, delta, or FWHM_rel not available"
        try:
            gain_original = self.delta ** self.n
            gain_new = self.delta ** (self.n + 1)
        except (ValueError, TypeError):
            return False, "Could not compute gains"
        if gain_original <= 0 or gain_new <= 0:
            return False, "Gain non-positive"
        expected_original = 2.0 * math.sqrt(2.0 * math.log(2) / gain_original)
        expected_new = 2.0 * math.sqrt(2.0 * math.log(2) / gain_new)
        # Expected new should be smaller
        if expected_new >= expected_original:
            return False, f"Expected FWHM_rel for n+1 ({expected_new}) not smaller than for n ({expected_original})"
        return True, f"Increasing n from {self.n} to {self.n+1} decreases expected FWHM_rel from {expected_original} to {expected_new}"

    def test_monotonicity_delta_decreases_fwhm(self):
        """Increasing delta (secondary emission factor) should decrease FWHM_rel for fixed n."""
        if self.n is None or self.delta is None or self.FWHM_rel is None:
            return False, "SKIP: n, delta, or FWHM_rel not available"
        try:
            gain_original = self.delta ** self.n
            delta_new = self.delta * 1.1  # 10% increase
            gain_new = delta_new ** self.n
        except (ValueError, TypeError):
            return False, "Could not compute gains"
        if gain_original <= 0 or gain_new <= 0:
            return False, "Gain non-positive"
        expected_original = 2.0 * math.sqrt(2.0 * math.log(2) / gain_original)
        expected_new = 2.0 * math.sqrt(2.0 * math.log(2) / gain_new)
        if expected_new >= expected_original:
            return False, f"Expected FWHM_rel for delta*1.1 ({expected_new}) not smaller than for delta ({expected_original})"
        return True, f"Increasing delta from {self.delta} to {delta_new:.2f} decreases expected FWHM_rel from {expected_original} to {expected_new}"

    # ----------------------------------------------------------------------
    # Answer sanity test (single weak sanity, no derived expected values)
    # ----------------------------------------------------------------------
    def test_answer_sanity_fwhm_rel_range_and_finite(self):
        """Weak sanity: FWHM_rel should be positive, finite, and less than 1 (50% difference is 0.5, but allow up to 1)."""
        if self.FWHM_rel is None:
            return False, "SKIP: FWHM_rel not available"
        if math.isinf(self.FWHM_rel) or math.isnan(self.FWHM_rel):
            return False, f"FWHM_rel is infinite or NaN"
        if self.FWHM_rel <= 0:
            return False, f"FWHM_rel = {self.FWHM_rel} is not positive"
        if self.FWHM_rel >= 1:
            return False, f"FWHM_rel = {self.FWHM_rel} is >= 1 (unphysical for relative resolution)"
        return True, f"FWHM_rel = {self.FWHM_rel} is finite, positive, and < 1"


if __name__ == '__main__':
    # Example usage with a dummy solution instance
    class DummySolution:
        def __init__(self):
            self.givens = {'n': 10, 'delta': 5}
            self.results = {'FWHM_rel': 2.0 * math.sqrt(2.0 * math.log(2) / (5**10))}
            self.explanation_steps = "ΔN = 2√(2ln2/5^10) ≈ ..."
            self.pseudocode = ""

    checker = PhysicsChecker(DummySolution())
    success, logs = checker.run_all_tests()
    for log in logs:
        print(log)
    print(f"\nOverall: {'PASS' if success else 'FAIL'}")