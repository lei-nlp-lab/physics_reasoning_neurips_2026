import math
import traceback
from typing import Any, List, Optional, Tuple


class PhysicsChecker:
    """
    PhysicsChecker for the twin paradox pulse counting problem.
    Audits interface, time dilation, Doppler shifts, consistency of results,
    and monotonic trends. At most one weak answer-sanity test.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract givens safely (use None defaults)
        givens = getattr(solution_instance, 'givens', {}) or {}
        results = getattr(solution_instance, 'results', {}) or {}

        # Required givens (no numeric fallbacks)
        self.t1 = givens.get('t1', None)      # traveler's proper time per leg
        self.v  = givens.get('v', None)       # relative speed
        self.c  = givens.get('c', None)       # speed of light (must be provided)

        # Derived quantities (None if not computable)
        if (self.t1 is not None and self.v is not None and self.c is not None
                and self.c != 0):
            self.beta = self.v / self.c
            if abs(self.beta) < 1.0:
                self.gamma = 1.0 / math.sqrt(1.0 - self.beta**2)
            else:
                self.gamma = None
        else:
            self.beta = None
            self.gamma = None

        # Results (expected keys) – all with None defaults
        self.N_send_traveler     = results.get('N_send_traveler', None)
        self.N_rec_out           = results.get('N_rec_out', None)
        self.N_rec_in            = results.get('N_rec_in', None)
        self.ratio_traveler      = results.get('ratio_traveler', None)
        self.N_home_out          = results.get('N_home_out', None)
        self.t2                  = results.get('t2', None)
        self.N_home_in           = results.get('N_home_in', None)
        self.t3                  = results.get('t3', None)
        self.ratio_home_received = results.get('ratio_home_received', None)
        self.ratio_home          = results.get('ratio_home', None)

    # ------------------------------------------------------------------
    # Helper: return a SKIP message if any argument is None
    # ------------------------------------------------------------------
    def _skip_if_none(self, *args: Any) -> Optional[Tuple[bool, str]]:
        for a in args:
            if a is None:
                return (False, "SKIP: missing required value")
        return None

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys_exist(self) -> Tuple[bool, str]:
        """All 10 result keys must exist and not be None."""
        keys = ['N_send_traveler', 'N_rec_out', 'N_rec_in', 'ratio_traveler',
                'N_home_out', 't2', 'N_home_in', 't3',
                'ratio_home_received', 'ratio_home']
        missing = [k for k in keys if getattr(self, k, None) is None]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required result keys present")

    def test_interface_t1_given(self) -> Tuple[bool, str]:
        if self.t1 is None:
            return (False, "SKIP: required given 't1' is missing")
        if self.t1 <= 0:
            return (False, f"t1 must be positive, got {self.t1}")
        return (True, "t1 is present and positive")

    def test_interface_c_given(self) -> Tuple[bool, str]:
        if self.c is None:
            return (False, "SKIP: required given 'c' is missing")
        if self.c <= 0:
            return (False, f"c must be positive, got {self.c}")
        return (True, "c is present and positive")

    def test_interface_beta_physical(self) -> Tuple[bool, str]:
        if self.beta is None:
            return (False, "SKIP: cannot compute beta (missing v or c)")
        if abs(self.beta) >= 1.0:
            return (False, f"beta must be < 1, got {self.beta}")
        return (True, "beta is physically allowed")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_time_dilation_total_home_time(self) -> Tuple[bool, str]:
        """Total home time = 2*gamma*t1 should equal t2 + t3."""
        skip = self._skip_if_none(self.t2, self.t3, self.t1, self.gamma)
        if skip:
            return skip
        expected_total = 2.0 * self.gamma * self.t1
        actual_total = self.t2 + self.t3
        if math.isclose(actual_total, expected_total, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"t2+t3 = {actual_total} ≈ 2γt1 = {expected_total}")
        return (False, f"t2+t3 = {actual_total}, expected {expected_total}")

    def test_invariant_doppler_receding_factor(self) -> Tuple[bool, str]:
        """N_rec_out should equal t1 * sqrt((1-beta)/(1+beta))."""
        skip = self._skip_if_none(self.N_rec_out, self.t1, self.beta)
        if skip:
            return skip
        expected = self.t1 * math.sqrt((1.0 - self.beta) / (1.0 + self.beta))
        if math.isclose(self.N_rec_out, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"N_rec_out = {self.N_rec_out} ≈ {expected}")
        return (False, f"N_rec_out = {self.N_rec_out}, expected {expected}")

    def test_invariant_doppler_approaching_factor(self) -> Tuple[bool, str]:
        """N_rec_in should equal t1 * sqrt((1+beta)/(1-beta))."""
        skip = self._skip_if_none(self.N_rec_in, self.t1, self.beta)
        if skip:
            return skip
        expected = self.t1 * math.sqrt((1.0 + self.beta) / (1.0 - self.beta))
        if math.isclose(self.N_rec_in, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"N_rec_in = {self.N_rec_in} ≈ {expected}")
        return (False, f"N_rec_in = {self.N_rec_in}, expected {expected}")

    def test_invariant_ratio_traveler_equals_gamma(self) -> Tuple[bool, str]:
        """ratio_traveler should equal gamma."""
        skip = self._skip_if_none(self.ratio_traveler, self.gamma)
        if skip:
            return skip
        if math.isclose(self.ratio_traveler, self.gamma, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"ratio_traveler = {self.ratio_traveler} ≈ γ = {self.gamma}")
        return (False, f"ratio_traveler = {self.ratio_traveler}, expected γ = {self.gamma}")

    def test_invariant_ratio_home_equals_gamma(self) -> Tuple[bool, str]:
        """ratio_home should equal gamma."""
        skip = self._skip_if_none(self.ratio_home, self.gamma)
        if skip:
            return skip
        if math.isclose(self.ratio_home, self.gamma, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"ratio_home = {self.ratio_home} ≈ γ = {self.gamma}")
        return (False, f"ratio_home = {self.ratio_home}, expected γ = {self.gamma}")

    def test_invariant_home_received_ratio_equals_one(self) -> Tuple[bool, str]:
        """ratio_home_received should be 1."""
        skip = self._skip_if_none(self.ratio_home_received)
        if skip:
            return skip
        if math.isclose(self.ratio_home_received, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"ratio_home_received = {self.ratio_home_received} ≈ 1")
        return (False, f"ratio_home_received = {self.ratio_home_received}, expected 1")

    # ------------------------------------------------------------------
    # Modeling / trace tests (based on explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_N_home_out_equals_t1(self) -> Tuple[bool, str]:
        """N_home_out should equal t1 (traveler's outgoing sent pulses)."""
        skip = self._skip_if_none(self.N_home_out, self.t1)
        if skip:
            return skip
        if math.isclose(self.N_home_out, self.t1, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"N_home_out = {self.N_home_out} ≈ t1 = {self.t1}")
        return (False, f"N_home_out = {self.N_home_out}, expected t1 = {self.t1}")

    def test_modeling_N_home_in_equals_t1(self) -> Tuple[bool, str]:
        """N_home_in should equal t1."""
        skip = self._skip_if_none(self.N_home_in, self.t1)
        if skip:
            return skip
        if math.isclose(self.N_home_in, self.t1, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"N_home_in = {self.N_home_in} ≈ t1 = {self.t1}")
        return (False, f"N_home_in = {self.N_home_in}, expected t1 = {self.t1}")

    def test_modeling_t2_t3_formulas(self) -> Tuple[bool, str]:
        """t2 = γ t1 (1+β), t3 = γ t1 (1-β)."""
        skip = self._skip_if_none(self.t2, self.t3, self.t1, self.gamma, self.beta)
        if skip:
            return skip
        expected_t2 = self.gamma * self.t1 * (1.0 + self.beta)
        expected_t3 = self.gamma * self.t1 * (1.0 - self.beta)
        ok_t2 = math.isclose(self.t2, expected_t2, rel_tol=1e-2, abs_tol=1e-2)
        ok_t3 = math.isclose(self.t3, expected_t3, rel_tol=1e-2, abs_tol=1e-2)
        if ok_t2 and ok_t3:
            return (True,
                    f"t2 = {self.t2} ≈ {expected_t2}, t3 = {self.t3} ≈ {expected_t3}")
        msg = f"t2 = {self.t2} (expected {expected_t2}), t3 = {self.t3} (expected {expected_t3})"
        return (False, msg)

    def test_modeling_N_send_traveler_equals_2t1(self) -> Tuple[bool, str]:
        """N_send_traveler should be 2*t1."""
        skip = self._skip_if_none(self.N_send_traveler, self.t1)
        if skip:
            return skip
        expected = 2.0 * self.t1
        if math.isclose(self.N_send_traveler, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"N_send_traveler = {self.N_send_traveler} ≈ 2t1 = {expected}")
        return (False, f"N_send_traveler = {self.N_send_traveler}, expected 2t1 = {expected}")

    # ------------------------------------------------------------------
    # Monotonicity tests (perturb an input and compare)
    # ------------------------------------------------------------------
    def test_monotonicity_N_rec_out_decreases_with_beta(self) -> Tuple[bool, str]:
        """When beta increases, N_rec_out should decrease."""
        skip = self._skip_if_none(self.beta, self.t1, self.N_rec_out)
        if skip:
            return skip
        # Choose perturbation that keeps beta < 1
        delta = 0.05
        if self.beta + delta >= 1.0:
            return (False, "SKIP: cannot perturb beta above 1")
        beta_high = self.beta + delta
        # Compute expected value for higher beta (using the Doppler formula)
        expected_lower = self.t1 * math.sqrt((1.0 - beta_high) / (1.0 + beta_high))
        # Baseline is self.N_rec_out (for original beta)
        if expected_lower < self.N_rec_out or math.isclose(expected_lower,
                                                           self.N_rec_out,
                                                           rel_tol=1e-2,
                                                           abs_tol=1e-2):
            return (True, f"N_rec_out decreases: baseline {self.N_rec_out} >≈ perturbed {expected_lower}")
        return (False,
                f"Expected N_rec_out to decrease, got baseline {self.N_rec_out} ≤ perturbed {expected_lower}")

    def test_monotonicity_N_rec_in_increases_with_beta(self) -> Tuple[bool, str]:
        """When beta increases, N_rec_in should increase."""
        skip = self._skip_if_none(self.beta, self.t1, self.N_rec_in)
        if skip:
            return skip
        delta = 0.05
        if self.beta + delta >= 1.0:
            return (False, "SKIP: cannot perturb beta above 1")
        beta_high = self.beta + delta
        expected_higher = self.t1 * math.sqrt((1.0 + beta_high) / (1.0 - beta_high))
        if expected_higher > self.N_rec_in or math.isclose(expected_higher,
                                                           self.N_rec_in,
                                                           rel_tol=1e-2,
                                                           abs_tol=1e-2):
            return (True, f"N_rec_in increases: baseline {self.N_rec_in} <≈ perturbed {expected_higher}")
        return (False,
                f"Expected N_rec_in to increase, got baseline {self.N_rec_in} ≥ perturbed {expected_higher}")

    # ------------------------------------------------------------------
    # Weak answer sanity (at most one test; only sign/finite checks)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """
        Sanity: all results are non‑negative, finite, and gamma‑related ratios >= 1.
        No re‑derivation of exact values.
        """
        results = [
            self.N_send_traveler, self.N_rec_out, self.N_rec_in,
            self.ratio_traveler, self.N_home_out, self.t2,
            self.N_home_in, self.t3, self.ratio_home_received, self.ratio_home
        ]
        for r in results:
            if r is None:
                continue
            if not math.isfinite(r):
                return (False, f"Non‑finite value found: {r}")
            if r < 0:
                return (False, f"Negative value found: {r}")
        # ratios (gamma) should be >= 1 in a physical twin paradox
        if self.ratio_traveler is not None and self.ratio_traveler < 1.0:
            return (False, f"ratio_traveler = {self.ratio_traveler} < 1, but expected >= 1")
        if self.ratio_home is not None and self.ratio_home < 1.0:
            return (False, f"ratio_home = {self.ratio_home} < 1, but expected >= 1")
        return (True, "All results are non‑negative, finite, and gamma‑related ratios >= 1")

    # ------------------------------------------------------------------
    # Main test runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test_* methods, run them, collect messages."""
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
                passed = False
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)