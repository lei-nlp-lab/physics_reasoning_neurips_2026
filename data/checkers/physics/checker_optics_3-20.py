import math
import traceback
from typing import Any, Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        # Extract givens safely
        givens = getattr(solution_instance, 'givens', {})
        if givens is None:
            givens = {}
        self.d = givens.get('d', None)   # Earth-Sun distance
        self.c = givens.get('c', None)   # speed of light

        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        self.t_min = results.get('t_min', None)   # travel time in minutes
        self.t_s = results.get('t_s', None)       # travel time in seconds (optional)

        # Store explanation steps if present (for modeling checks)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_t_min_exists(self) -> Tuple[bool, str]:
        if self.t_min is None:
            return (False, "results['t_min'] is missing (None)")
        return (True, "results['t_min'] exists")

    def test_interface_t_min_is_numeric(self) -> Tuple[bool, str]:
        if not isinstance(self.t_min, (int, float)):
            return (False, f"t_min must be a number, got {type(self.t_min).__name__}")
        if math.isnan(self.t_min):
            return (False, "t_min is NaN")
        return (True, f"t_min is a numeric value: {self.t_min}")

    def test_interface_givens_present(self) -> Tuple[bool, str]:
        # Weak check: givens are required to verify physics; warn if missing
        if self.d is None or self.c is None:
            return (False, "SKIP: givens 'd' or 'c' not available")
        return (True, f"d = {self.d}, c = {self.c} present")

    def test_interface_explanation_steps_not_empty(self) -> Tuple[bool, str]:
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps missing")
        if len(self.explanation_steps) == 0:
            return (False, "explanation_steps is empty")
        return (True, "explanation_steps provided")

    # ------------------------------------------------------------------
    # Invariant tests (physics laws, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_t_positive(self) -> Tuple[bool, str]:
        if self.t_min is None:
            return (False, "SKIP: t_min not available")
        if self.t_min > 0:
            return (True, f"t_min = {self.t_min} > 0")
        return (False, f"t_min = {self.t_min} should be positive")

    def test_invariant_d_c_t_relation(self) -> Tuple[bool, str]:
        # d = c * t  with t in seconds
        if None in (self.d, self.c, self.t_min):
            return (False, "SKIP: d, c or t_min missing")
        t_seconds = self.t_min * 60.0
        expected_d = self.c * t_seconds
        if math.isclose(self.d, expected_d, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"d ≈ c * t_min*60: {self.d} ≈ {expected_d:.3e}")
        return (False, f"d = {self.d} but c * t_min*60 = {expected_d:.3e} (t_min={self.t_min})")

    # ------------------------------------------------------------------
    # Modeling / trace tests (using explanation steps or pseudocode steps)
    # ------------------------------------------------------------------
    def test_modeling_t_s_to_t_min_conversion(self) -> Tuple[bool, str]:
        # Use t_s if available; otherwise compute from givens
        if self.t_min is None:
            return (False, "SKIP: t_min not available")
        # Compute expected t_s from givens
        if self.d is not None and self.c is not None:
            t_s_given = self.d / self.c
            # Compare with t_min converted back to seconds
            t_min_to_s = self.t_min * 60.0
            if math.isclose(t_s_given, t_min_to_s, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"t_min * 60 ≈ d/c: {t_min_to_s:.2f} ≈ {t_s_given:.2f}")
            return (False, f"t_min * 60 = {t_min_to_s:.2f} ≠ d/c = {t_s_given:.2f}")
        # If no givens, check using stored t_s if present
        if self.t_s is not None:
            expected = self.t_s / 60.0
            if math.isclose(self.t_min, expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"t_min = {self.t_min} ≈ t_s/60 = {expected:.4f}")
            return (False, f"t_min = {self.t_min} but t_s/60 = {expected:.4f}")
        return (False, "SKIP: no givens or t_s to verify conversion")

    def test_modeling_t_s_computation_from_givens(self) -> Tuple[bool, str]:
        # If t_s is provided, check that it matches d/c
        if self.t_s is None or self.d is None or self.c is None:
            return (False, "SKIP: missing t_s, d, or c")
        expected_t_s = self.d / self.c
        if math.isclose(self.t_s, expected_t_s, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"t_s = {self.t_s} ≈ d/c = {expected_t_s:.2f}")
        return (False, f"t_s = {self.t_s} ≠ d/c = {expected_t_s:.2f}")

    # ------------------------------------------------------------------
    # Monotonicity tests (perturb input and check trend)
    # ------------------------------------------------------------------
    def test_monotonicity_t_increases_with_d(self) -> Tuple[bool, str]:
        # If d increases, t should increase (linear relation)
        if None in (self.d, self.c, self.t_min):
            return (False, "SKIP: d, c or t_min missing")
        # Perturb d upward by 10%
        d_high = self.d * 1.1
        t_pred_high = d_high / (self.c * 60.0)   # minutes
        if self.t_min < t_pred_high - 1e-9:
            return (True, f"t_min = {self.t_min:.4f} < t(d+10%) = {t_pred_high:.4f}")
        return (False, f"t_min = {self.t_min:.4f} is NOT less than t(d+10%) = {t_pred_high:.4f}")

    def test_monotonicity_t_decreases_with_c(self) -> Tuple[bool, str]:
        # If c increases, t should decrease (inverse relation)
        if None in (self.d, self.c, self.t_min):
            return (False, "SKIP: d, c or t_min missing")
        # Perturb c upward by 10%
        c_high = self.c * 1.1
        t_pred_low = self.d / (c_high * 60.0)   # minutes
        if self.t_min > t_pred_low + 1e-9:
            return (True, f"t_min = {self.t_min:.4f} > t(c+10%) = {t_pred_low:.4f}")
        return (False, f"t_min = {self.t_min:.4f} is NOT greater than t(c+10%) = {t_pred_low:.4f}")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one allowed; no isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_broad_range(self) -> Tuple[bool, str]:
        if self.t_min is None:
            return (False, "SKIP: t_min not available")
        if 0 < self.t_min < 10:   # 8.33 min is plausible, but allow up to 10
            return (True, f"t_min = {self.t_min} min is within (0,10]")
        return (False, f"t_min = {self.t_min} min is outside plausible range (0,10]")

    # ------------------------------------------------------------------
    # Master runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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