import math
import traceback

class PhysicsChecker:
    """
    A robust physics checker for the quantum measurement problem.

    Extracts results from the solution instance and runs multiple small,
    isolated tests focused on invariants, modeling correctness, and minimal
    answer sanity. Uses safe defaults and never crashes.
    """

    def __init__(self, solution_instance):
        # Safely extract the results dictionary (if present)
        self.results = getattr(solution_instance, 'results', {})
        # Extract each required key with None default
        self.state_after_A = self.results.get('state_after_A', None)
        self.P_b1 = self.results.get('P_b1', None)
        self.P_b2 = self.results.get('P_b2', None)
        self.P_a1 = self.results.get('P_a1', None)

        # Pre‑compute the transformation coefficients given in the problem
        # (these are physical givens, not derived from the solution)
        self.c1 = 3/5   # <φ1|ψ1>
        self.c2 = 4/5   # <φ2|ψ1>
        self.expected_P_b1 = self.c1 * self.c1   # 9/25
        self.expected_P_b2 = self.c2 * self.c2   # 16/25

    # ───────── interface checks ─────────
    def test_interface_required_keys(self):
        """Check that all four required keys are present and not None."""
        missing = []
        if self.state_after_A is None:
            missing.append('state_after_A')
        if self.P_b1 is None:
            missing.append('P_b1')
        if self.P_b2 is None:
            missing.append('P_b2')
        if self.P_a1 is None:
            missing.append('P_a1')
        if missing:
            return (False, f"Missing keys: {', '.join(missing)}")
        return (True, "All required keys present")

    def test_interface_state_after_A_type(self):
        """Check that state_after_A is a string."""
        if self.state_after_A is None:
            return (False, "SKIP: state_after_A not available")
        if not isinstance(self.state_after_A, str):
            return (False, f"Expected string, got {type(self.state_after_A).__name__}")
        return (True, "state_after_A is a string")

    # ───────── invariant / theorem sanity checks ─────────
    def test_invariant_born_normalization(self):
        """P_b1 + P_b2 should be ≈ 1 (Born rule completeness)."""
        if self.P_b1 is None or self.P_b2 is None:
            return (False, "SKIP: P_b1 or P_b2 missing")
        if not math.isclose(self.P_b1 + self.P_b2, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"P_b1 + P_b2 = {self.P_b1 + self.P_b2} ≠ 1")
        return (True, f"P_b1 + P_b2 = 1 (with tolerance)")

    def test_invariant_probability_range(self):
        """All probabilities must lie in [0,1]."""
        probs = {'P_b1': self.P_b1, 'P_b2': self.P_b2, 'P_a1': self.P_a1}
        for name, val in probs.items():
            if val is None:
                continue
            if val < 0 or val > 1:
                return (False, f"{name} = {val} outside [0,1]")
            if math.isnan(val):
                return (False, f"{name} is NaN")
        return (True, "All available probabilities in [0,1] and not NaN")

    def test_invariant_transformation_unitary(self):
        """
        Verify that the transformation matrix between bases is orthogonal.
        This is a property of the problem statement, not the solution, but
        it serves as a sanity check on the problem setup.
        """
        # matrix M such that psi = M * phi
        # M = [[3/5, 4/5], [4/5, -3/5]]
        # Check columns are orthonormal
        col1 = [3/5, 4/5]
        col2 = [4/5, -3/5]
        dot = col1[0]*col2[0] + col1[1]*col2[1]
        norm1 = col1[0]**2 + col1[1]**2
        norm2 = col2[0]**2 + col2[1]**2
        if not (math.isclose(dot, 0.0, rel_tol=1e-2, abs_tol=1e-2) and
                math.isclose(norm1, 1.0, rel_tol=1e-2, abs_tol=1e-2) and
                math.isclose(norm2, 1.0, rel_tol=1e-2, abs_tol=1e-2)):
            return (False, "Transformation matrix is not orthogonal")
        return (True, "Transformation matrix is orthogonal")

    # ───────── modeling / trace checks ─────────
    def test_modeling_state_after_A(self):
        """Check that state after measuring a1 is correctly identified."""
        if self.state_after_A is None:
            return (False, "SKIP: state_after_A not available")
        if self.state_after_A != 'psi1':
            return (False, f"Expected 'psi1', got '{self.state_after_A}'")
        return (True, "state_after_A = psi1")

    def test_modeling_P_b1(self):
        """P_b1 should be (3/5)^2 = 9/25."""
        if self.P_b1 is None:
            return (False, "SKIP: P_b1 not available")
        if not math.isclose(self.P_b1, self.expected_P_b1, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"P_b1 = {self.P_b1} ≠ {self.expected_P_b1}")
        return (True, f"P_b1 ≈ {self.expected_P_b1}")

    def test_modeling_P_b2(self):
        """P_b2 should be (4/5)^2 = 16/25."""
        if self.P_b2 is None:
            return (False, "SKIP: P_b2 not available")
        if not math.isclose(self.P_b2, self.expected_P_b2, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"P_b2 = {self.P_b2} ≠ {self.expected_P_b2}")
        return (True, f"P_b2 ≈ {self.expected_P_b2}")

    def test_modeling_law_total_probability(self):
        """
        Check that the final probability P_a1 follows the law of total
        probability: P(a1) = P(b1)*|⟨ψ1|φ1⟩|^2 + P(b2)*|⟨ψ1|φ2⟩|^2.
        The conditional probabilities are the same squared overlaps.
        """
        if any(v is None for v in (self.P_b1, self.P_b2, self.P_a1)):
            return (False, "SKIP: one of P_b1, P_b2, P_a1 missing")
        # conditional probabilities
        cond1 = self.expected_P_b1   # |⟨ψ1|φ1⟩|^2 = (3/5)^2
        cond2 = self.expected_P_b2   # |⟨ψ1|φ2⟩|^2 = (4/5)^2
        expected_P_a1 = self.P_b1 * cond1 + self.P_b2 * cond2
        # Expected from problem: 337/625 = 0.5392
        if not math.isclose(self.P_a1, expected_P_a1, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"P_a1 = {self.P_a1} ≠ {expected_P_a1} (via law of total prob.)")
        return (True, f"P_a1 ≈ {expected_P_a1} from total probability")

    # ───────── monotonicity tests (input perturbation) ─────────
    def test_monotonicity_born_overlap(self):
        """
        Perturb the input overlap magnitude |<φ1|ψ1>| (given as 3/5) and verify
        that the probability P_b1 increases monotonically with it.
        Baseline = self.P_b1 (solution's value). Perturb c1 by +Δ, recompute
        P_b1_perturbed = (c1+Δ)^2, check it's larger.
        """
        if self.P_b1 is None:
            return (False, "SKIP: P_b1 not available")
        # Baseline is the solution's P_b1
        baseline = self.P_b1
        # Perturb the overlap coefficient c1 by +1% relative (but absolute delta small)
        delta = 0.01 * self.c1   # relative 1% perturbation
        c1_perturbed = self.c1 + delta
        perturbed_P_b1 = c1_perturbed * c1_perturbed
        # The perturbed probability should be larger than baseline
        if perturbed_P_b1 <= baseline:
            return (False, f"Perturbed P_b1 = {perturbed_P_b1} should be > baseline {baseline}")
        return (True, f"P_b1 increases with |<φ1|ψ1>| (baseline {baseline}, perturbed {perturbed_P_b1})")

    # ───────── weak answer sanity (exactly one) ─────────
    def test_answer_sanity_finite(self):
        """
        Very weak check: P_a1 is a finite, non‑NaN number.
        (We already check [0,1] in invariants, but this is the designated
        answer sanity method.)
        """
        if self.P_a1 is None:
            return (False, "SKIP: P_a1 not available")
        if not math.isfinite(self.P_a1):
            return (False, f"P_a1 is not finite: {self.P_a1}")
        return (True, "P_a1 is finite")

    # ───────── test runner ─────────
    def run_all_tests(self):
        """Discover and run all test_* methods, catch exceptions, return results."""
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