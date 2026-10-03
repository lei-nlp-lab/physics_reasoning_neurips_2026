import math
import sys
import traceback
from typing import Any, Dict, List, Tuple

class PhysicsChecker:
    """
    Audits the solution for the probability of finding a harmonic oscillator
    outside its classical region (ground state).
    """
    def __init__(self, solution_instance: Any) -> None:
        # Safe extraction from results dictionary (expected keys per contract)
        results = getattr(solution_instance, 'results', {}) or {}
        self.P_out_exact = results.get('P_out_exact', None)
        self.P_out_approx = results.get('P_out_approx', None)

        # Additional optional fields (not used in tests, but kept for potential future use)
        self.givens = getattr(solution_instance, 'givens', None)

    def _is_numeric(self, val: Any) -> bool:
        """Check if val is a finite numeric value (int or float)."""
        if isinstance(val, (int, float)):
            return math.isfinite(val)
        return False

    # ─── Interface tests ──────────────────────────────────────────
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that both required result keys exist and are numeric."""
        if self.P_out_exact is None:
            return (False, "Missing key 'P_out_exact' in results")
        if self.P_out_approx is None:
            return (False, "Missing key 'P_out_approx' in results")
        if not self._is_numeric(self.P_out_exact):
            return (False, f"'P_out_exact' is not a finite number: {self.P_out_exact}")
        if not self._is_numeric(self.P_out_approx):
            return (False, f"'P_out_approx' is not a finite number: {self.P_out_approx}")
        return (True, "Both result keys present and numeric")

    # ─── Invariant tests (theorem‑based physical sanity) ───────────
    def test_invariant_probability_bounds(self) -> Tuple[bool, str]:
        """Probability must be between 0 and 1 (inclusive of numerical tolerance)."""
        if self.P_out_exact is None or self.P_out_approx is None:
            return (False, "SKIP: required results missing")
        if not (0.0 - 1e-9 <= self.P_out_exact <= 1.0 + 1e-9):
            return (False, f"P_out_exact = {self.P_out_exact} outside [0,1]")
        if not (0.0 - 1e-9 <= self.P_out_approx <= 1.0 + 1e-9):
            return (False, f"P_out_approx = {self.P_out_approx} outside [0,1]")
        return (True, "Both probabilities lie in [0,1]")

    def test_invariant_exact_matches_erfc1(self) -> Tuple[bool, str]:
        """
        The exact result P_out_exact must equal erfc(1) to within 1% tolerance.
        (erfc(1) ≈ 0.15729920705028518)
        """
        if self.P_out_exact is None:
            return (False, "SKIP: P_out_exact not provided")
        expected = math.erfc(1.0)
        if not math.isclose(self.P_out_exact, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"P_out_exact = {self.P_out_exact}, expected ≈ {expected} (erfc(1))")
        return (True, f"P_out_exact matches erfc(1) (≈ {expected})")

    # ─── Modeling / trace quality tests ───────────────────────────
    def test_modeling_exact_vs_approx_consistency(self) -> Tuple[bool, str]:
        """
        The approximate numeric value should be close to the exact expression.
        Both should represent erfc(1) ≈ 0.1573.
        """
        if self.P_out_exact is None or self.P_out_approx is None:
            return (False, "SKIP: both results required for comparison")
        if not math.isclose(self.P_out_exact, self.P_out_approx, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Exact ({self.P_out_exact}) and approx ({self.P_out_approx}) differ by >1%")
        return (True, "Exact and approximate probabilities are consistent (relative tolerance 1%)")

    def test_modeling_normalization_holds(self) -> Tuple[bool, str]:
        """
        Verify that the probability satisfies total probability conservation:
        P_inside + P_outside = 1.
        The inside probability (|x| < classical turning point) is erf(1) ≈ 0.8427.
        """
        if self.P_out_exact is None:
            return (False, "SKIP: P_out_exact not provided for normalization check")
        # Probability inside = 1 - P_outside (by normalization)
        computed_inside = 1.0 - self.P_out_exact
        # Expected inside probability from the integral: erf(1) = (2/sqrt(pi)) ∫₀¹ exp(-u²) du
        expected_inside = math.erf(1.0)   # ≈ 0.8427
        if not math.isclose(computed_inside, expected_inside, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Inside prob {computed_inside} differs from erf(1) = {expected_inside}")
        return (True, "Normalization holds: inside prob + outside prob = 1")

    # ─── Monotonicity test (perturb turning point) ─────────────────
    def test_monotonicity_increasing_turning_point_reduces_probability(self) -> Tuple[bool, str]:
        """
        Check that if the classical turning point is increased (doubled), the probability
        outside the classical region decreases. Baseline is P_out_exact (should be erfc(1)).
        Perturbed outcome: erfc(2) ≈ 0.0047 (since probability = erfc(√α A) and doubling A makes √αA = 2).
        """
        if self.P_out_exact is None:
            return (False, "SKIP: P_out_exact not available for monotonicity baseline")
        baseline = self.P_out_exact
        # The perturbed turning point is 2 * original A → lower √αA = 2 → probability = erfc(2)
        perturbed = math.erfc(2.0)   # ≈ 0.004677
        if baseline <= perturbed:
            return (False, f"Baseline P_out = {baseline} should be > perturbed = {perturbed} when turning point is doubled")
        return (True, "Probability decreases when classical region increases (monotonic)")

    # ─── Weak answer sanity (at most one) ──────────────────────────
    def test_answer_sanity_basic_scale(self) -> Tuple[bool, str]:
        """
        Weak sanity: probability is positive, finite, and less than 1.
        No derivation or re‑computation of the target value.
        """
        if self.P_out_approx is None:
            return (False, "SKIP: P_out_approx not available")
        if not self._is_numeric(self.P_out_approx) or self.P_out_approx <= 0 or self.P_out_approx >= 1:
            return (False, f"P_out_approx = {self.P_out_approx} does not satisfy 0 < p < 1")
        if self.P_out_exact is not None:
            if not self._is_numeric(self.P_out_exact) or self.P_out_exact <= 0 or self.P_out_exact >= 1:
                return (False, f"P_out_exact = {self.P_out_exact} does not satisfy 0 < p < 1")
        return (True, "Probabilities are positive, finite, and between 0 and 1")

    # ─── Runner ────────────────────────────────────────────────────
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_* methods, collect results."""
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
                traceback.print_exc(file=sys.stderr)
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)