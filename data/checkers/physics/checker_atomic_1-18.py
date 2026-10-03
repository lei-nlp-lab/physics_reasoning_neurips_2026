import math
import traceback

class PhysicsChecker:
    """
    Audits a solution instance for the 'Typical lifetime for an excited atom' problem.
    Checks interface, physical invariants, modeling consistency, monotonicity, and weak answer sanity.
    """

    def __init__(self, solution_instance):
        # Safely extract results and givens with None defaults
        self.results = getattr(solution_instance, 'results', {})
        self.lifetime = self.results.get('lifetime', None)

        # Try to extract the list of given choices (if provided)
        self.given_choices = getattr(solution_instance, 'given_choices', None)
        if self.given_choices is None:
            self.given_choices = getattr(solution_instance, 'givens', None)

        # Fallback: use the known set from the problem description
        if self.given_choices is None:
            self.given_choices = [1e-1, 1e-8, 1e-13, 1e-23]

    # ------------------------------------------------------------------
    # Interface tests: ensure required result keys exist and have valid types
    # ------------------------------------------------------------------
    def test_interface_lifetime_exists(self):
        """Check that the 'lifetime' key is present in results."""
        if self.lifetime is None:
            return (False, "lifetime not found in results")
        return (True, "lifetime key present")

    def test_interface_lifetime_is_numeric(self):
        """Check that the lifetime value is numeric."""
        if self.lifetime is None:
            return (False, "SKIP: lifetime missing")
        if not isinstance(self.lifetime, (int, float)):
            return (False, f"lifetime is not numeric, got {type(self.lifetime)}")
        return (True, "lifetime is numeric")

    # ------------------------------------------------------------------
    # Invariant tests: physical theorem-based sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_typical_allowed_transition(self):
        """Check that the selected lifetime matches the known ~10^{-8} s scale."""
        if self.lifetime is None:
            return (False, "SKIP: lifetime missing")
        expected = 1e-8
        if math.isclose(self.lifetime, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"lifetime {self.lifetime} matches typical allowed transition ~1e-8 s")
        else:
            return (False, f"lifetime {self.lifetime} is not near typical allowed transition {expected} s")

    def test_invariant_not_other_extreme(self):
        """Verify the lifetime is not one of the clearly wrong extremes (1e-1, 1e-13, 1e-23)."""
        if self.lifetime is None:
            return (False, "SKIP: lifetime missing")
        # The other choices are orders of magnitude away
        extremes = [1e-1, 1e-13, 1e-23]
        if math.isclose(self.lifetime, extremes[0], rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"lifetime {self.lifetime} is the 0.1 s extreme, not typical")
        if math.isclose(self.lifetime, extremes[1], rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"lifetime {self.lifetime} is the 1e-13 s extreme, not typical")
        if math.isclose(self.lifetime, extremes[2], rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"lifetime {self.lifetime} is the 1e-23 s extreme, not typical")
        return (True, "lifetime is not one of the clearly wrong extreme values")

    # ------------------------------------------------------------------
    # Modeling tests: checks based on the selection process and constraints
    # ------------------------------------------------------------------
    def test_modeling_choice_from_given_set(self):
        """Verify that the selected lifetime is one of the original given choices."""
        if self.lifetime is None:
            return (False, "SKIP: lifetime missing")
        # self.given_choices is guaranteed to be a list here
        if self.lifetime in self.given_choices:
            return (True, f"lifetime {self.lifetime} is one of the given choices")
        else:
            return (False, f"lifetime {self.lifetime} is not in the given choices list {self.given_choices}")

    def test_modeling_selected_is_typical(self):
        """Check that the selected lifetime is the only one that lies in the typical range (~1e-9 to 1e-7 s)."""
        if self.lifetime is None:
            return (False, "SKIP: lifetime missing")
        typical_min = 1e-9
        typical_max = 1e-7
        if typical_min <= self.lifetime <= typical_max:
            return (True, f"lifetime {self.lifetime} lies within [1e-9, 1e-7] typical range")
        else:
            return (False, f"lifetime {self.lifetime} is outside the typical allowed-transition range")

    # ------------------------------------------------------------------
    # Monotonicity test: demonstrate a trend by perturbing an input
    # ------------------------------------------------------------------
    def test_monotonicity_selected_is_not_extreme(self):
        """
        Verify that the selected lifetime is not the smallest nor the largest
        among the given choices. This shows monotonicity in the selection: 
        the correct answer is intermediate, not at the extremes.
        We perturb the list by hypothetically adding a value closer to the typical,
        but here we simply check ordering relative to the original list.
        If we had a continuous input, we would perturb it; here the "input"
        is the set of choices, and we check monotonicity by comparing to extremes.
        """
        if self.lifetime is None:
            return (False, "SKIP: lifetime missing")
        if not self.given_choices or len(self.given_choices) < 2:
            return (False, "SKIP: insufficient given choices to perform ordering test")
        min_choice = min(self.given_choices)
        max_choice = max(self.given_choices)
        # The selected lifetime must be strictly between the min and max
        # (it should not be the extreme values)
        if math.isclose(self.lifetime, min_choice, rel_tol=1e-12) or math.isclose(self.lifetime, max_choice, rel_tol=1e-12):
            return (False, f"lifetime {self.lifetime} is an extreme value (min or max) in the choices; expected intermediate")
        return (True, f"lifetime {self.lifetime} is intermediate between {min_choice} and {max_choice}")

    # ------------------------------------------------------------------
    # Answer sanity test (weak, no math.isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_positive_and_finite(self):
        """Weak sanity: the lifetime must be a positive finite number."""
        if self.lifetime is None:
            return (False, "SKIP: lifetime missing")
        if self.lifetime <= 0:
            return (False, f"lifetime must be positive, got {self.lifetime}")
        if not math.isfinite(self.lifetime):
            return (False, f"lifetime must be finite, got {self.lifetime}")
        return (True, "lifetime is positive and finite")

    # ------------------------------------------------------------------
    # Main test discovery and execution
    # ------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)