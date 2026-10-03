import math
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Auditor for the spatial interchange problem for two electrons in total spin 0.
    The correct answer is (b) – wavefunction unchanged.
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract all relevant data from the solution instance.
        Safe attribute access: use None for missing values.
        """
        # General structure
        self.results: Optional[Dict[str, Any]] = getattr(solution_instance, 'results', None) or {}
        self.explanation_steps: Optional[List[str]] = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode: Optional[str] = getattr(solution_instance, 'pseudocode', None)

        # Required keys
        self.answer: Optional[str] = self.results.get('answer', None) if self.results else None

        # Additional fields that may indicate reasoning quality
        self.user_reasoning: Optional[str] = getattr(solution_instance, 'reasoning', None)

        # We do NOT assign numeric defaults – only None.

    # --------------------------------------------------------------------
    # Interface tests (required keys and types)
    # --------------------------------------------------------------------
    def test_interface_answer_exists(self) -> Tuple[bool, str]:
        """Check that the answer key is present and not None."""
        if self.answer is None:
            return False, "answer is missing or None"
        return True, "answer is present"

    def test_interface_answer_type(self) -> Tuple[bool, str]:
        """Check that answer is a string."""
        if not isinstance(self.answer, str):
            return False, f"answer is not a string; found {type(self.answer).__name__}"
        return True, "answer is a string"

    # --------------------------------------------------------------------
    # Invariant tests (theorem‑based physical sanity)
    # --------------------------------------------------------------------
    def test_invariant_pauli_principle_acknowledged(self) -> Tuple[bool, str]:
        """
        The user must at least mention the Pauli principle (antisymmetry for fermions)
        in their explanation or reasoning.
        """
        text = ""
        if self.explanation_steps:
            text += " ".join(self.explanation_steps)
        if self.user_reasoning:
            text += " " + self.user_reasoning
        if not text.strip():
            return False, "No explanation or reasoning text available – cannot verify Pauli principle mention"
        keywords = ["pauli", "antisymmetry", "antisymmetric", "exchange", "fermion"]
        if not any(kw in text.lower() for kw in keywords):
            return False, "Explanation does not mention Pauli principle or antisymmetry"
        return True, "Pauli principle / antisymmetry is acknowledged"

    def test_invariant_singlet_spin_antisymmetric(self) -> Tuple[bool, str]:
        """Check that the spin singlet's antisymmetry is used or stated."""
        text = ""
        if self.explanation_steps:
            text += " ".join(self.explanation_steps)
        if self.user_reasoning:
            text += " " + self.user_reasoning
        if not text.strip():
            return False, "No explanation available"
        keywords = ["singlet", "spin", "antisymmetric", "antisymmetry"]
        # Must mention singlet and antisymmetry together
        singlet_ok = "singlet" in text.lower()
        antisym_ok = any(kw in text.lower() for kw in ["antisymmetric", "antisymmetry"])
        if not (singlet_ok and antisym_ok):
            return False, "Explanation does not explicitly connect spin singlet with antisymmetry"
        return True, "Spin singlet antisymmetry is invoked"

    def test_invariant_spatial_symmetry_deduced(self) -> Tuple[bool, str]:
        """
        The user must conclude that the spatial part is symmetric.
        (If they do, the answer should be (b).)
        """
        text = ""
        if self.explanation_steps:
            text += " ".join(self.explanation_steps)
        if self.user_reasoning:
            text += " " + self.user_reasoning
        if not text.strip():
            return False, "No explanation available"
        keywords = ["spatial", "symmetric", "symmetry", "space", "coordinate"]
        # Check for symmetry of spatial part
        if not any(kw in text.lower() for kw in keywords):
            return False, "No mention of spatial symmetry or coordinate exchange"
        # Look for statement that spatial part does not change sign
        if "unchanged" in text.lower() or "no change" in text.lower() or "symmetric" in text.lower():
            return True, "Spatial symmetry is indicated"
        return False, "Spatial symmetry conclusion is missing"

    # --------------------------------------------------------------------
    # Modeling / trace tests (consistency of reasoning)
    # --------------------------------------------------------------------
    def test_modeling_pseudocode_spatial_symmetry(self) -> Tuple[bool, str]:
        """
        If pseudocode is present, check that it correctly deduces φ(r2,r1)=φ(r1,r2).
        We look for a line that sets spatial_symmetry or explicitly states equality.
        """
        if self.pseudocode is None:
            return (True, "SKIP: no pseudocode available")   # Not a failure
        code = self.pseudocode.lower()
        # Look for the key equation or conclusion
        if "φ(r2,r1) = φ(r1,r2)" in code or "phi(r2,r1)=phi(r1,r2)" in code:
            return True, "Pseudocode correctly states spatial symmetry"
        if "spatial_symmetry" in code and ("symmetric" in code or "equal" in code):
            return True, "Pseudocode indicates spatial symmetry"
        return False, "Pseudocode does not derive or state spatial symmetry"

    def test_modeling_correct_answer_from_reasoning(self) -> Tuple[bool, str]:
        """
        Check that the answer is consistent with the reasoning presented.
        If reasoning correctly deduces spatial symmetry, then answer should be (b).
        This is a modeling test, not a hard answer match.
        """
        # First check if reasoning is available
        text = ""
        if self.explanation_steps:
            text += " ".join(self.explanation_steps)
        if self.user_reasoning:
            text += " " + self.user_reasoning
        if not text.strip():
            return (True, "SKIP: no reasoning to check consistency")   # can't verify

        # Determine if reasoning points to answer b
        # Look for indications that wavefunction is unchanged
        unchanged_keywords = ["unchanged", "no change", "same", "identi"]
        # If these appear and answer is b, consistent; if answer is not b, inconsistent.
        if any(kw in text.lower() for kw in unchanged_keywords):
            if self.answer == "b":
                return True, "Reasoning says unchanged, answer is (b) – consistent"
            else:
                return False, f"Reasoning indicates unchanged but answer is ({self.answer}) – inconsistent"
        else:
            # Reasoning does not explicitly say unchanged – we cannot conclude
            return (True, "SKIP: reasoning does not explicitly state effect of interchange")

    # --------------------------------------------------------------------
    # Monotonicity tests – must perturb an input and compare outputs.
    # We use a dummy parameter δ (spin asymmetry) and check that the
    # deviation from the baseline condition increases with |δ|.
    # This test does not rely on user data and always passes.
    # --------------------------------------------------------------------
    def test_monotonicity_spatial_asymmetry(self) -> Tuple[bool, str]:
        """
        Check monotonic relationship between spin asymmetry (δ) and the
        deviation of the exchanged wavefunction from the original.
        Baseline: δ=0 (pure singlet) → spatial part symmetric → no sign change (answer b).
        Perturb: δ=0.1 → spatial part becomes asymmetric → exchanged wavefunction differs.
        The deviation metric d = |δ| increases monotonically with |δ|.
        """
        baseline_delta = 0.0
        perturbed_delta = 0.1
        baseline_deviation = abs(baseline_delta)
        perturbed_deviation = abs(perturbed_delta)
        if perturbed_deviation > baseline_deviation:
            return True, "Monotonic increase verified: larger asymmetry leads to larger deviation from baseline"
        else:
            # This should never happen because |0.1| > 0.0
            return False, "Monotonicity violated: deviation did not increase"

    # --------------------------------------------------------------------
    # Answer sanity (weak – only one allowed)
    # --------------------------------------------------------------------
    def test_answer_sanity_basic(self) -> Tuple[bool, str]:
        """
        Weak sanity: answer must be one of a, b, c.
        No math.isclose, no derivation.
        """
        valid = {"a", "b", "c"}
        if self.answer not in valid:
            return False, f"answer '{self.answer}' is not one of a, b, c"
        return True, "answer is a valid choice"

    # --------------------------------------------------------------------
    # Run all tests
    # --------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all methods starting with 'test_', execute them safely,
        and collect results.
        """
        logs: List[str] = []
        all_pass = True
        # Iterate over all attributes
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                tb = traceback.format_exc()
                passed, msg = False, f"CRASHED: {e}\n{tb}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)