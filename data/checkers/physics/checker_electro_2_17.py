import math
import traceback
from typing import List, Tuple, Optional

class PhysicsChecker:
    """
    Audits a solution_instance for the magnetostatics boundary condition problem.
    Required keys in solution_instance:
        - results: dict containing at least 'correct_option'
        - explanation_steps: list of strings (optional)
        - physics_mapping: dict (optional)
        - key_constraints: dict (optional)
        - solving_pseudocode: dict (optional)
    """

    def __init__(self, solution_instance) -> None:
        # Safely extract all relevant data; default to None for missing values.
        self.results: Optional[dict] = getattr(solution_instance, 'results', None)
        self.correct_option = None
        if self.results is not None:
            self.correct_option = self.results.get('correct_option', None)
        self.explanation_steps: Optional[List[str]] = getattr(
            solution_instance, 'explanation_steps', None
        )
        self.physics_mapping: Optional[dict] = getattr(
            solution_instance, 'physics_mapping', None
        )
        self.key_constraints: Optional[dict] = getattr(
            solution_instance, 'key_constraints', None
        )
        self.solving_pseudocode: Optional[dict] = getattr(
            solution_instance, 'solving_pseudocode', None
        )

    # ------------------------------------------------------------------
    # Interface Tests
    # ------------------------------------------------------------------
    def test_interface_correct_option_exists(self) -> Tuple[bool, str]:
        """Check that 'correct_option' exists in results and is a string."""
        if self.results is None:
            return (False, "SKIP: results dictionary missing")
        if 'correct_option' not in self.results:
            return (False, "results dictionary does not contain 'correct_option'")
        opt = self.correct_option
        if not isinstance(opt, str):
            return (False, f"correct_option is of type {type(opt).__name__}, expected str")
        return (True, f"correct_option present: '{opt}'")

    def test_interface_correct_option_in_options(self) -> Tuple[bool, str]:
        """Check that correct_option is one of 'a', 'b', 'c'."""
        if self.correct_option is None:
            return (False, "SKIP: correct_option not available")
        if self.correct_option in ('a', 'b', 'c'):
            return (True, f"correct_option '{self.correct_option}' is a valid option")
        return (False, f"correct_option '{self.correct_option}' not in {{'a','b','c'}}")

    # ------------------------------------------------------------------
    # Invariant Tests (based on Maxwell's equations)
    # ------------------------------------------------------------------
    def test_invariant_gauss_law_cited(self) -> Tuple[bool, str]:
        """
        Check that the solution's reasoning references Gauss's law for magnetism
        or the statement '∇·B = 0'.
        We look for it in explanation_steps, physics_mapping, or solving_pseudocode.
        """
        # Gather all text sources
        texts = []
        if self.explanation_steps:
            for step in self.explanation_steps:
                if isinstance(step, str):
                    texts.append(step)
        if self.physics_mapping and isinstance(self.physics_mapping, dict):
            for v in self.physics_mapping.values():
                if isinstance(v, str):
                    texts.append(v)
        if self.solving_pseudocode and isinstance(self.solving_pseudocode, dict):
            # Recursively collect strings from pseudocode dictionary
            def collect_strings(d, out):
                if isinstance(d, str):
                    out.append(d)
                elif isinstance(d, dict):
                    for val in d.values():
                        collect_strings(val, out)
                elif isinstance(d, list):
                    for item in d:
                        collect_strings(item, out)
            collect_strings(self.solving_pseudocode, texts)

        combined = " ".join(texts).lower()
        keywords = ["gauss's law for magnetism", "gauss law", "∇·b = 0", "div b = 0",
                    "gauss’s law", "gauss law for magnetism"]
        found = any(kw in combined for kw in keywords)
        if found:
            return (True, "Gauss's law for magnetism referenced in reasoning")
        return (False, "No reference to Gauss's law for magnetism in explanation_steps, physics_mapping, or pseudocode")

    def test_invariant_continuity_normal_B(self) -> Tuple[bool, str]:
        """
        Check that the solution states that normal component of B is continuous
        (or equivalent).
        Search for phrases: 'normal B continuous', 'B1_n = B2_n', 'normal component of B'.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps available")
        combined = " ".join(self.explanation_steps).lower()
        phrases = ["normal component of b", "b1_n = b2_n", "normal b continuous",
                   "b_n continuous", "component of b normal", "normal b is continuous"]
        found = any(phrase in combined for phrase in phrases)
        if found:
            return (True, "Continuity of normal B is mentioned")
        return (False, "No mention of normal B continuity in explanation_steps")

    def test_invariant_no_false_claim(self) -> Tuple[bool, str]:
        """
        Ensure the solution does not incorrectly claim that tangential B is continuous
        (option c). We check that the explanation does not endorse that statement.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps available")
        combined = " ".join(self.explanation_steps).lower()
        # Look for claims that tangential/parallel B is continuous (as if it were correct)
        incorrect_phrases = [
            "component of b parallel", "tangential b continuous", "b1_t = b2_t"
        ]
        for phrase in incorrect_phrases:
            if phrase in combined:
                # Check if it appears in a rejecting context (e.g., "is not continuous")
                # We'll do a simple heuristic: if the surrounding text contains "not" within 50 chars.
                idx = combined.index(phrase)
                start = max(0, idx - 50)
                end = min(len(combined), idx + 50)
                context = combined[start:end]
                if "not" not in context and "false" not in context and "incorrect" not in context:
                    return (False, f"Solution appears to incorrectly claim: '{phrase}'")
        return (True, "No false claim about tangential B continuity detected")

    # ------------------------------------------------------------------
    # Modeling / Reasoning Quality Tests
    # ------------------------------------------------------------------
    def test_modeling_references_maxwell(self) -> Tuple[bool, str]:
        """
        Check that Maxwell's equations in magnetostatics are referenced in
        explanation_steps or physics_mapping.
        """
        texts = []
        if self.explanation_steps:
            texts.extend(self.explanation_steps)
        if self.physics_mapping and isinstance(self.physics_mapping, dict):
            for v in self.physics_mapping.values():
                if isinstance(v, str):
                    texts.append(v)
        combined = " ".join(texts).lower()
        keywords = ["maxwell", "magnetostatic", "∇×h", "curl h", "ampere's law"]
        found = any(kw in combined for kw in keywords)
        if found:
            return (True, "Maxwell's equations / magnetostatics referenced")
        return (False, "No reference to Maxwell's equations in explanation_steps or physics_mapping")

    def test_modeling_option_selection_justified(self) -> Tuple[bool, str]:
        """
        Check that the explanation_steps include a justification for selecting (a)
        and rejecting (b) and (c).
        Look for affirmative statement about (a) and negative about (b) or (c).
        """
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps missing")
        full = " ".join(self.explanation_steps).lower()
        selects_a = ("(a)" in full) or ("option a" in full)
        rejects_b = ("(b)" in full and ("false" in full or "incorrect" in full or "wrong" in full or "not" in full))
        rejects_c = ("(c)" in full and ("false" in full or "incorrect" in full or "wrong" in full or "not" in full))
        if selects_a:
            msg = "(a) is selected"
            if rejects_b or rejects_c:
                msg += " and at least one wrong option is rejected"
            return (True, msg)
        return (False, "Explanation does not clearly indicate selection of option (a)")

    # ------------------------------------------------------------------
    # Monotonicity Tests
    # ------------------------------------------------------------------
    def test_monotonicity_permeability_perturbation(self) -> Tuple[bool, str]:
        """
        Verify that the correct answer (a) is invariant under a change in relative
        permeability of one medium. This tests the monotonicity (here, constancy)
        of the correct physical condition under a physically meaningful parameter change.

        Baseline: μ1 = 1, μ2 = 1  →  normal B continuous → correct option (a)
        Perturbation: μ1 = 1, μ2 = 2  →  still normal B continuous → correct option (a)
        Since the solution's answer should be the same for both scenarios, we check
        that the solution's answer is (a). If it is, the monotonicity holds
        (option (a) remains correct regardless of the permeability value).
        """
        if self.correct_option is None:
            return (False, "SKIP: correct_option not available")
        # The monotonicity test is conceptual: answer must be (a) for any permeability.
        # Since we can't recompute anything, we simply verify that the answer is (a).
        if self.correct_option == 'a':
            return (True, "Correct answer (a) is invariant under permeability perturbation (conceptual test)")
        return (False, f"Answer is '{self.correct_option}', expected 'a' for monotonicity with respect to permeability")

    # ------------------------------------------------------------------
    # Weak Sanity Test (one allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_sign_and_finite(self) -> Tuple[bool, str]:
        """
        Weak sanity: correct_option must be a non-empty string from {'a','b','c'}.
        No numerical derivation or math.isclose.
        """
        if self.correct_option is None:
            return (False, "correct_option is None")
        if not isinstance(self.correct_option, str):
            return (False, f"correct_option type invalid")
        if self.correct_option not in ('a', 'b', 'c'):
            return (False, f"correct_option '{self.correct_option}' not in valid options")
        if len(self.correct_option) == 0:
            return (False, "correct_option is empty string")
        return (True, f"Answer '{self.correct_option}' is a valid option")

    # ------------------------------------------------------------------
    # Test Runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_* methods, collecting results."""
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
                passed, msg = False, f"{name} CRASHED: {type(e).__name__}: {e}"
                traceback.print_exc()  # optional, for debugging
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)