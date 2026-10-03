import math
import itertools

class PhysicsChecker:
    """
    Checker for the orthogonalized drunk probability problem.
    """
    def __init__(self, solution_instance):
        # Extract results safely, defaulting to None for missing keys
        self.favorable = getattr(solution_instance, 'results', {}).get('favorable', None)
        self.probability = getattr(solution_instance, 'results', {}).get('probability', None)
        # Also store the solution instance for potential future use
        self.solution = solution_instance

    # ---------------------------------------------------------------
    # Interface checks
    # ---------------------------------------------------------------
    def test_interface_favorable_type(self):
        """Check that 'favorable' is an integer."""
        if self.favorable is None:
            return (False, "SKIP: 'favorable' not provided")
        if not isinstance(self.favorable, int):
            return (False, f"'favorable' should be int, got {type(self.favorable)}")
        return (True, "favorable is an integer")

    def test_interface_probability_type(self):
        """Check that 'probability' is a float (or convertible to float)."""
        if self.probability is None:
            return (False, "SKIP: 'probability' not provided")
        try:
            float(self.probability)
        except (TypeError, ValueError):
            return (False, f"'probability' is not a valid number: {self.probability}")
        return (True, "probability is numeric")

    def test_interface_favorable_nonnegative(self):
        """Check that 'favorable' is >= 0."""
        if self.favorable is None:
            return (False, "SKIP: 'favorable' not provided")
        if self.favorable < 0:
            return (False, f"favorable is negative: {self.favorable}")
        return (True, "favorable is nonnegative")

    def test_interface_probability_within_unit_interval(self):
        """Probability must be between 0 and 1 inclusive."""
        if self.probability is None:
            return (False, "SKIP: 'probability' not provided")
        p = float(self.probability)
        if not (0.0 <= p <= 1.0):
            return (False, f"probability {p} is outside [0,1]")
        return (True, "probability in [0,1]")

    # ---------------------------------------------------------------
    # Invariant / theorem-based checks
    # ---------------------------------------------------------------
    def test_invariant_total_sequences(self):
        """Check that favorable <= 64 (total sequences)."""
        if self.favorable is None:
            return (False, "SKIP: 'favorable' not provided")
        if self.favorable > 64:
            return (False, f"favorable ({self.favorable}) exceeds 64")
        return (True, "favorable ≤ 64")

    def test_invariant_probability_derived_from_favorable(self):
        """Check that probability equals favorable / 64."""
        if self.favorable is None or self.probability is None:
            return (False, "SKIP: missing favorable or probability")
        expected = self.favorable / 64.0
        if not math.isclose(float(self.probability), expected, rel_tol=1e-12):
            return (False, f"probability {self.probability} != favorable/64 = {expected}")
        return (True, "probability = favorable / 64")

    # ---------------------------------------------------------------
    # Modeling / re-derivation checks
    # ---------------------------------------------------------------
    def test_modeling_enumeration(self):
        """
        Re-derive the probability by enumerating all 4^3 step sequences
        and counting those within distance ≤ 2L. Compare with user's probability
        using math.isclose.
        """
        directions = [(1,0), (-1,0), (0,1), (0,-1)]  # east, west, north, south
        total = 0
        favorable = 0
        for steps in itertools.product(directions, repeat=3):
            total += 1
            x = sum(d[0] for d in steps)
            y = sum(d[1] for d in steps)
            dist_sq = x*x + y*y
            if dist_sq <= 4:
                favorable += 1
        expected_prob = favorable / total

        if self.probability is None:
            return (False, "SKIP: user did not provide probability")
        user_prob = float(self.probability)
        if math.isclose(user_prob, expected_prob, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"user probability {user_prob} matches re-derived {expected_prob}")
        else:
            return (False, f"user probability {user_prob} differs from re-derived {expected_prob}")

    def test_modeling_multinomial_derivation(self):
        """
        Re-derive by enumerating (N,S,E,W) quadruples summing to 3 and using
        multinomial coefficients. Compare with user's probability.
        """
        factorial = [1, 1, 2, 6]  # 0! to 3!
        total_fav = 0
        for N in range(4):
            for S in range(4-N):
                for E in range(4-N-S):
                    W = 3 - N - S - E
                    if W < 0:
                        continue
                    dx = E - W
                    dy = N - S
                    if dx*dx + dy*dy <= 4:
                        coeff = factorial[3] // (factorial[N]*factorial[S]*factorial[E]*factorial[W])
                        total_fav += coeff
        expected_prob = total_fav / 64.0

        if self.probability is None:
            return (False, "SKIP: user did not provide probability")
        user_prob = float(self.probability)
        if math.isclose(user_prob, expected_prob, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"user probability matches multinomial derivation {expected_prob}")
        else:
            return (False, f"user probability {user_prob} != multinomial derivation {expected_prob}")

    # ---------------------------------------------------------------
    # Monotonicity checks (explicit perturbation)
    # ---------------------------------------------------------------
    def test_monotonicity_radius(self):
        """
        Check monotonicity with respect to the radius threshold.
        Baseline: probability for radius=2L (threshold squared = 4).
        Perturb to smaller radius (1.5L => threshold squared = 2.25) and larger radius (2.5L => 6.25).
        Verify: prob(smaller) < prob(baseline) < prob(larger).
        The probabilities are re-derived inside the checker.
        """
        # Helper to compute probability for a given squared-radius threshold
        def prob_for_threshold(threshold_sq):
            directions = [(1,0), (-1,0), (0,1), (0,-1)]
            fav = 0
            for steps in itertools.product(directions, repeat=3):
                x = sum(d[0] for d in steps)
                y = sum(d[1] for d in steps)
                if x*x + y*y <= threshold_sq:
                    fav += 1
            return fav / 64.0

        p_baseline = prob_for_threshold(4.0)      # radius = 2L
        p_smaller = prob_for_threshold(2.25)      # radius = 1.5L
        p_larger  = prob_for_threshold(6.25)      # radius = 2.5L

        if not (p_smaller < p_baseline < p_larger):
            return (False, f"Monotonicity violated: p(1.5)={p_smaller}, p(2)={p_baseline}, p(2.5)={p_larger}")
        return (True, f"Monotonic good: {p_smaller} < {p_baseline} < {p_larger}")

    # ---------------------------------------------------------------
    # Weak final-answer sanity (only one such method)
    # ---------------------------------------------------------------
    def test_answer_sanity_not_extreme(self):
        """
        Verify that the probability is not trivially 0 or 1.
        This is a weak sanity check only.
        """
        if self.probability is None:
            return (False, "SKIP: probability not provided")
        p = float(self.probability)
        if abs(p) < 1e-12 or abs(1 - p) < 1e-12:
            return (False, f"probability is extreme: {p}")
        return (True, "probability is not 0 or 1")

    # ---------------------------------------------------------------
    # Run all tests
    # ---------------------------------------------------------------
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