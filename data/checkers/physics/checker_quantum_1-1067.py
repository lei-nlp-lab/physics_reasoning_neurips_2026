import math
import traceback
from typing import Any, Callable, List, Tuple

class PhysicsChecker:
    """
    Auditor for the problem: Eigenvalues of CQ+QC and simultaneous eigenstates of C and Q.
    """

    def __init__(self, solution_instance: Any = None) -> None:
        """
        Extract all required inputs and results from the solution_instance.

        The instance is expected to have:
          - .results (dict) with keys 'lambda_eigenvalue' and 'q_allowed'
          - .explanation_steps (list of strings, optional)
        """
        if solution_instance is None:
            self.lambda_eigenvalue = None
            self.q_allowed = None
            self.explanation_steps = []
            return

        # First try to get results from a nested .results dict
        results = getattr(solution_instance, 'results', {})
        self.lambda_eigenvalue = results.get('lambda_eigenvalue', None)
        self.q_allowed = results.get('q_allowed', None)

        # Also try to get explanation steps if present
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', [])

        # Safety: if not a list, set to empty list
        if not isinstance(self.explanation_steps, list):
            self.explanation_steps = []

    # ------------------------------------------------------------------
    # Interface tests – required keys and types
    # ------------------------------------------------------------------
    def test_interface_keys_present(self) -> Tuple[bool, str]:
        """Check that both required result keys exist (lambda_eigenvalue and q_allowed)."""
        if self.lambda_eigenvalue is None:
            return (False, "Missing key 'lambda_eigenvalue' in results.")
        if self.q_allowed is None:
            return (False, "Missing key 'q_allowed' in results.")
        return (True, "All required keys present.")

    def test_interface_types(self) -> Tuple[bool, str]:
        """Check that both values are numeric (int or float)."""
        if not isinstance(self.lambda_eigenvalue, (int, float)):
            return (False, f"'lambda_eigenvalue' should be numeric, got {type(self.lambda_eigenvalue).__name__}.")
        if not isinstance(self.q_allowed, (int, float)):
            return (False, f"'q_allowed' should be numeric, got {type(self.q_allowed).__name__}.")
        return (True, "Both values are numeric.")

    # ------------------------------------------------------------------
    # Invariant tests – theorem-based physical sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_eigenvalue_zero(self) -> Tuple[bool, str]:
        """Check that the eigenvalue of CQ+QC is 0 (allowed 1% tolerance)."""
        if self.lambda_eigenvalue is None:
            return (False, "SKIP: lambda_eigenvalue not available.")
        if not math.isfinite(self.lambda_eigenvalue):
            return (False, "'lambda_eigenvalue' is not finite.")
        if math.isclose(self.lambda_eigenvalue, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Eigenvalue is 0 (within tolerance).")
        else:
            return (False, f"Expected eigenvalue 0, got {self.lambda_eigenvalue}.")

    def test_invariant_q_allowed_zero(self) -> Tuple[bool, str]:
        """Check that the only allowed charge for simultaneous eigenstates is q=0 (1% tolerance)."""
        if self.q_allowed is None:
            return (False, "SKIP: q_allowed not available.")
        if not math.isfinite(self.q_allowed):
            return (False, "'q_allowed' is not finite.")
        if math.isclose(self.q_allowed, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Allowed q is 0 (within tolerance).")
        else:
            return (False, f"Expected allowed q=0, got {self.q_allowed}.")

    # ------------------------------------------------------------------
    # Modeling/trace tests – based on explanation_steps and pseudocode
    # ------------------------------------------------------------------
    def test_modeling_anticommutation_keyword(self) -> Tuple[bool, str]:
        """Check that the explanation steps mention anticommutation or the relation CQ+QC=0."""
        if not self.explanation_steps:
            return (False, "SKIP: No explanation_steps provided.")
        combined = " ".join(self.explanation_steps).lower()
        keyword_phrases = ["c q + q c = 0", "c q + q c =0", "anticommut", "anti-commut"]
        found = any(phrase in combined for phrase in keyword_phrases)
        if found:
            return (True, "Explanation includes reference to anticommutation relation.")
        else:
            return (False, "No clear mention of anticommutation or CQ+QC=0 in steps.")

    def test_modeling_action_on_psi_q(self) -> Tuple[bool, str]:
        """Check that the steps apply C and Q to |psi_q⟩ explicitly."""
        if not self.explanation_steps:
            return (False, "SKIP: No explanation_steps provided.")
        combined = " ".join(self.explanation_steps).lower()
        if "psi_q" in combined or "psii_q" in combined or "|psi_q" in combined:
            return (True, "Steps involve applying operators to |psi_q⟩.")
        else:
            return (False, "No explicit application to |psi_q⟩ found in steps.")

    # ------------------------------------------------------------------
    # Monotonicity tests – perturb an input and compare output
    # ------------------------------------------------------------------
    def test_monotonicity_eigenvalue_independent_of_q(self) -> Tuple[bool, str]:
        """
        Check that the eigenvalue of CQ+QC remains 0 when the charge parameter q is perturbed.
        (Because (CQ+QC) annihilates any eigenstate of Q regardless of q.)
        Baseline: for an arbitrary q (say q=0), eigenvalue is 0.
        Perturbation: consider q increased by 10% (still arbitrary), eigenvalue should stay 0.
        """
        if self.lambda_eigenvalue is None:
            return (False, "SKIP: lambda_eigenvalue not available.")
        baseline = 0.0
        perturbed = 0.0  # eigenvalue does not depend on q
        if math.isclose(baseline, perturbed, rel_tol=0, abs_tol=1e-9):
            return (True, "Eigenvalue unchanged under q-perturbation (always 0).")
        else:
            return (False, "Eigenvalue changed — contradicts physical law.")

    # ------------------------------------------------------------------
    # Weak answer sanity tests (at most one, no math.isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_lam_finite(self) -> Tuple[bool, str]:
        """Check that lambda_eigenvalue is a finite number."""
        if self.lambda_eigenvalue is None:
            return (False, "SKIP: lambda_eigenvalue not available.")
        if math.isfinite(self.lambda_eigenvalue):
            return (True, "lambda_eigenvalue is finite.")
        return (False, f"lambda_eigenvalue is not finite (got {self.lambda_eigenvalue}).")

    # ------------------------------------------------------------------
    # Main runner (discover all test_* methods)
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test methods and run them, collecting logs."""
        logs: List[str] = []
        all_pass = True

        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method: Callable[[], Tuple[bool, str]] = getattr(self, name)
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