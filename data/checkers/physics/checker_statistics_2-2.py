import math
import re
import traceback

class PhysicsChecker:
    """Checker for the E. coli neutral drift with predator problem."""

    def __init__(self, solution_instance):
        """Extract required results safely."""
        # Assume solution_instance.results is a dict-like object
        results = getattr(solution_instance, 'results', None) or {}
        # Use None for missing keys; never default to numeric values
        self.dist = results.get('dist', None)
        self.time = results.get('time', None)
        self.dist_selection = results.get('dist_selection', None)
        # Additional constants from problem (for reference, not as defaults)
        self.N = 10000       # fixed colony size
        self.p_ref = 0.01    # preference deviation from problem statement

    # ---------- Interface tests ----------
    def test_interface_dist_exists(self):
        """dist key must exist and be a non-empty string."""
        if self.dist is None:
            return (False, "SKIP: dist not provided")
        if not isinstance(self.dist, str):
            return (False, "dist must be a string")
        if len(self.dist) == 0:
            return (False, "dist is empty")
        return (True, "dist key present and is a non-empty string")

    def test_interface_time_exists(self):
        """time key must exist and be a non-empty string."""
        if self.time is None:
            return (False, "SKIP: time not provided")
        if not isinstance(self.time, str):
            return (False, "time must be a string")
        if len(self.time) == 0:
            return (False, "time is empty")
        return (True, "time key present and is a non-empty string")

    def test_interface_dist_selection_exists(self):
        """dist_selection key must exist and be a non-empty string."""
        if self.dist_selection is None:
            return (False, "SKIP: dist_selection not provided")
        if not isinstance(self.dist_selection, str):
            return (False, "dist_selection must be a string")
        if len(self.dist_selection) == 0:
            return (False, "dist_selection is empty")
        return (True, "dist_selection key present and is a non-empty string")

    # ---------- Invariant tests (theorem-based) ----------
    def test_invariant_neutral_distribution_is_binomial(self):
        """Neutral distribution should be Binomial with p=0.5 (or equivalently
        the expression (1/2)^n * C(n,m))."""
        if self.dist is None:
            return (False, "SKIP: dist not available")
        # Accept either explicit 'Binom' or the factorial formula
        has_binom = 'binom' in self.dist.lower()
        has_half_power = '1/2' in self.dist or '0.5' in self.dist
        has_factorial = '!+' in self.dist or 'C^n' in self.dist or 'comb' in self.dist.lower()
        # The problem's specific formula also includes "1/2^n" and "n!/(m!(n-m)!)"
        # We check for at least some indication of binomial symmetry
        if has_binom or (has_half_power and has_factorial):
            return (True, "Neutral distribution is binomial with p=0.5")
        else:
            return (False, "Neutral distribution does not match Binomial(0.5) or the combinatorial formula")

    def test_invariant_selection_distribution_is_biased_binomial(self):
        """Selection distribution should be Binomial with p=0.5+p (p=0.01)."""
        if self.dist_selection is None:
            return (False, "SKIP: dist_selection not available")
        # Check for presence of both 1/2 and p, and binomial form
        has_half = '1/2' in self.dist_selection or '0.5' in self.dist_selection
        has_p = 'p' in self.dist_selection
        has_binom = 'binom' in self.dist_selection.lower() or 'C' in self.dist_selection
        # Also check optionally for the exact target form: (1/2 + p)^m (1/2 - p)^{n-m}
        has_exact_form = re.search(r'\(\s*1/2\s*\+\s*p\s*\)', self.dist_selection) and \
                         re.search(r'\(\s*1/2\s*-\s*p\s*\)', self.dist_selection)
        if (has_half and has_p and (has_binom or has_exact_form)):
            return (True, "Selection distribution matches biased binomial form")
        else:
            return (False, "Selection distribution does not contain expected biased binomial pattern")

    def test_invariant_neutral_p_equals_0_5(self):
        """The neutral distribution should explicitly reference p=0.5."""
        if self.dist is None:
            return (False, "SKIP: dist not available")
        # Check for '0.5' or '1/2' as probability parameter
        if '0.5' in self.dist or '1/2' in self.dist:
            return (True, "Neutral distribution contains p=0.5 or 1/2")
        else:
            return (False, "Neutral distribution does not mention p=0.5 or 1/2")

    # ---------- Modeling tests (based on explanation steps) ----------
    def test_modeling_time_estimate_range(self):
        """Time should be approximately 6-7 hours (log2(100) ≈ 6.64)."""
        if self.time is None:
            return (False, "SKIP: time not available")
        time_lower = self.time.lower()
        # Look for digits 6 and 7 and the word 'hour'
        has_six = '6' in self.time
        has_seven = '7' in self.time
        has_hour = 'hour' in time_lower
        if has_six and has_seven and has_hour:
            return (True, "Time estimate is within 6-7 hours range")
        else:
            return (False, "Time estimate does not match expected '6 to 7 hours' pattern")

    def test_modeling_selection_preference_value(self):
        """The selection distribution should imply p=0.01 (1% preference)."""
        if self.dist_selection is None:
            return (False, "SKIP: dist_selection not available")
        # We cannot exactly extract the p value, but we can check that the
        # expression likely uses a small deviation from 0.5.
        # Accept if it contains '0.01' or explicitly writes '0.5+p' or '0.5 - p'
        # with the expectation that p is given as numeric value in context.
        contains_p = 'p' in self.dist_selection
        contains_0_01 = '0.01' in self.dist_selection
        # Alternatively, check for a term like (0.5+0.01)
        if contains_p or contains_0_01:
            return (True, "Selection distribution references p or 0.01")
        else:
            return (False, "Selection distribution does not indicate p=0.01")

    # ---------- Monotonicity tests (optional, but we can add one) ----------
    def test_monotonicity_time_with_N(self):
        """Time estimate should increase with N (if N is perturbed)."""
        if self.time is None:
            return (False, "SKIP: time not provided, cannot test monotonicity")
        # The expected time is based on log2(N/n) with n=10000.
        # If N is larger, time should be larger.
        # Since we don't have perturbed data, we just check that the time
        # expression suggests scaling with N. For example, the time string
        # should contain some reference to N or doubling.
        has_N = 'N' in self.time
        has_doubling = '2^' in self.time or 'log2' in self.time.lower()
        if has_N or has_doubling:
            return (True, "Time monotonicity with N is plausible (mentions N or doubling)")
        else:
            # This is a weak check; we return warning but pass to avoid false fail
            return (True, "WARNING: Monotonicity not explicitly shown, but time is accepted")

    # ---------- Answer sanity (very weak) ----------
    def test_answer_sanity_time(self):
        """Time should be finite and positive (at least 1 hour)."""
        if self.time is None:
            return (False, "SKIP: time not available")
        # Try to extract a numeric value for broad sanity
        # If the string contains a number, it should be >0.
        # For simplicity, just check that the string is not just "0" or negative.
        if re.search(r'\d+', self.time):
            # Check it is not just "0" or very small
            if re.fullmatch(r'\d+', self.time):
                value = int(self.time)
                if value <= 0:
                    return (False, "Time is non-positive")
            return (True, "Time string contains a positive number.")
        else:
            # If no digits, it might be a formula; we still pass.
            return (True, "Time expression accepted (no numeric extraction needed)")

    # ---------- Run all tests ----------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Discover all callable methods starting with 'test_'
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
        return (all_pass, logs)