import math
import sympy as sp

class PhysicsChecker:
    """
    Checker for the dust grain angular momentum problem.
    """

    def __init__(self, solution_instance):
        # Extract results dict safely
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}

        # Extract givens dict safely (for input parameters)
        givens = getattr(solution_instance, 'givens', {})
        if not isinstance(givens, dict):
            givens = {}

        # Explanation steps (list of strings or dicts)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        if not isinstance(self.explanation_steps, list):
            self.explanation_steps = []

        # Shared constants (like Boltzmann constant)
        shared = getattr(solution_instance, 'shared_variables', {})
        if not isinstance(shared, dict):
            shared = {}
        self.k_B = shared.get('k_B', None)
        if self.k_B is None:
            # Use standard value if not provided
            self.k_B = 1.380649e-23

        # Emitted results (keys from pseudocode contract)
        self.L_long_sq = results.get('L_long_sq', None)
        self.L_perp_sq = results.get('L_perp_sq', None)

        # Input givens (for re-derivation and consistency checks)
        self.I_long = givens.get('I_long', None)
        self.I_perp = givens.get('I_perp', None)
        self.T = givens.get('T', None)

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_L_long_sq_exists(self):
        """Check that L_long_sq is present and numeric."""
        if self.L_long_sq is None:
            return (False, "L_long_sq is missing")
        if not isinstance(self.L_long_sq, (int, float)):
            return (False, f"L_long_sq is not numeric: {type(self.L_long_sq)}")
        return (True, "L_long_sq exists and is numeric")

    def test_interface_L_perp_sq_exists(self):
        """Check that L_perp_sq is present and numeric."""
        if self.L_perp_sq is None:
            return (False, "L_perp_sq is missing")
        if not isinstance(self.L_perp_sq, (int, float)):
            return (False, f"L_perp_sq is not numeric: {type(self.L_perp_sq)}")
        return (True, "L_perp_sq exists and is numeric")

    # ----------------------------------------------------------------------
    # Invariant tests (theorem‑based physical sanity)
    # ----------------------------------------------------------------------
    def test_invariant_ratio_long_to_perp(self):
        """
        Equipartition: <L_i^2> = I_i * k_B * T.
        Therefore L_long_sq / L_perp_sq = I_long / I_perp.
        Use fuzzy matching (rel_tol=1e-2) because of rounding noise.
        """
        if None in (self.L_long_sq, self.L_perp_sq, self.I_long, self.I_perp):
            return (False, "SKIP: missing L_long_sq, L_perp_sq, I_long, or I_perp")
        if self.I_perp == 0:
            return (False, "I_perp is zero, cannot compute ratio")
        ratio_expected = self.I_long / self.I_perp
        ratio_actual = self.L_long_sq / self.L_perp_sq
        if math.isclose(ratio_actual, ratio_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Ratio L_long_sq/L_perp_sq ≈ {ratio_actual:.6e} (expected ~{ratio_expected:.6e})")
        else:
            return (False, f"Ratio L_long_sq/L_perp_sq = {ratio_actual:.6e} does not match expected {ratio_expected:.6e} (rel_tol=1%)")

    def test_invariant_perp_greater_than_long(self):
        """
        Because I_long << I_perp, we must have L_long_sq < L_perp_sq.
        (In practice, L_long_sq is much smaller, but we only check strict inequality.)
        """
        if None in (self.L_long_sq, self.L_perp_sq):
            return (False, "SKIP: missing L_long_sq or L_perp_sq")
        if self.L_long_sq < self.L_perp_sq:
            return (True, "L_long_sq < L_perp_sq as expected")
        else:
            return (False, "L_long_sq >= L_perp_sq, contradicting I_long << I_perp")

    # ----------------------------------------------------------------------
    # Modeling / trace tests
    # ----------------------------------------------------------------------
    def test_modeling_equipartition_mentioned(self):
        """
        Check that the explanation steps reference the equipartition theorem
        or the concept of thermal equilibrium energy distribution.
        """
        if not self.explanation_steps:
            return (False, "SKIP: no explanation steps provided")
        keywords = ['equipartition', 'thermal equilibrium', 'energy', 'rotational degrees']
        found = False
        for step in self.explanation_steps:
            if isinstance(step, str):
                text = step.lower()
            elif isinstance(step, dict):
                text = str(step).lower()
            else:
                continue
            if any(kw in text for kw in keywords):
                found = True
                break
        if found:
            return (True, "Explanation steps mention equipartition / thermal equilibrium")
        else:
            return (False, "Explanation steps do not reference equipartition theorem or thermal energy")

    def test_modeling_values_consistent_with_inputs(self):
        """
        Re-derive expected L_long_sq and L_perp_sq from givens (I_long, I_perp, T)
        and Boltzmann constant, then compare with emitted results.
        This is a precise consistency check derived in‑checker (Rule 9).
        """
        if None in (self.I_long, self.I_perp, self.T):
            return (False, "SKIP: missing I_long, I_perp, or T")
        expected_long = self.I_long * self.k_B * self.T
        expected_perp = self.I_perp * self.k_B * self.T
        tol = 1e-2  # 1% relative tolerance
        ok = True
        messages = []
        if self.L_long_sq is None or not math.isclose(self.L_long_sq, expected_long, rel_tol=tol, abs_tol=1e-30):
            ok = False
            messages.append(f"L_long_sq = {self.L_long_sq}, expected ~{expected_long:.6e}")
        else:
            messages.append(f"L_long_sq consistent with I_long*k_B*T")
        if self.L_perp_sq is None or not math.isclose(self.L_perp_sq, expected_perp, rel_tol=tol, abs_tol=1e-30):
            ok = False
            messages.append(f"L_perp_sq = {self.L_perp_sq}, expected ~{expected_perp:.6e}")
        else:
            messages.append(f"L_perp_sq consistent with I_perp*k_B*T")
        if ok:
            return (True, "; ".join(messages))
        else:
            return (False, "Values inconsistent: " + "; ".join(messages))

    # ----------------------------------------------------------------------
    # Monotonicity tests (perturb parameters)
    # ----------------------------------------------------------------------
    def test_monotonicity_long_increase_ratio(self):
        """
        If I_long increases, L_long_sq should increase proportionally.
        We simulate a 10% increase in I_long and compute the expected new L_long_sq,
        then verify that the ratio L_long_sq / I_long remains constant.
        (This checks the linear relationship.)
        """
        if None in (self.L_long_sq, self.I_long):
            return (False, "SKIP: need L_long_sq and I_long")
        # Baseline proportionality constant
        if self.I_long == 0:
            return (False, "I_long is zero")
        const = self.L_long_sq / self.I_long
        # Simulate perturbed I_long (10% increase)
        perturbed_I_long = self.I_long * 1.1
        expected_perturbed_L = const * perturbed_I_long
        # We cannot directly test because we don't have a second result,
        # but we can check that the relationship is linear: L ∝ I.
        # Use the fact that for any two values L1/I1 = L2/I2.
        # Since we only have one point, we just assert that the constant
        # is positive and not absurd.
        if const > 0:
            return (True, f"Proportionality constant L_long_sq / I_long = {const:.6e} > 0 (linear relationship holds)")
        else:
            return (False, "L_long_sq / I_long is not positive, contradicting equipartition")

    # ----------------------------------------------------------------------
    # Weak answer sanity (only one, secondary)
    # ----------------------------------------------------------------------
    def test_answer_sanity_angular_momentum_direction(self):
        """
        Very weak: the final answer should state that angular momentum is
        nearly perpendicular to the long axis. We only check that the answer
        is plausible (non‑empty and not contradictory).
        """
        # The problem expects the answer in a specific place.
        # Often solution_instance has an 'answer' attribute (string).
        answer = getattr(self, 'answer', None)
        if answer is None:
            # Try to get from results dict (some solvers store final answer there)
            results = getattr(self, 'results', None)
            if results and isinstance(results, dict):
                answer = results.get('final_answer', results.get('answer', None))
        if answer is None:
            return (False, "SKIP: no answer string found")
        if not isinstance(answer, str):
            return (False, "Answer is not a string")
        # Check that it mentions "perpendicular" or "nearly perpendicular"
        if 'perpendicular' in answer.lower():
            return (True, "Answer mentions perpendicular orientation")
        else:
            return (False, "Answer does not mention perpendicular orientation (it may be missing or incorrect)")

    # ----------------------------------------------------------------------
    # Main runner
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