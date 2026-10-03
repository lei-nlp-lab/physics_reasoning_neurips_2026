import math
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Audits solutions for the Stark effect problem.
    Checks interface completeness, theorem-based invariants,
    modeling consistency, and monotonicity of the reasoning.
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract relevant data from the solution instance.
        All extracted attributes default to None if missing.
        """
        # Results dictionary expected to contain 'hydrogen_linear' and 'sodium_quadratic'
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        self.hydrogen_linear: Optional[bool] = self.results.get('hydrogen_linear', None)
        self.sodium_quadratic: Optional[bool] = self.results.get('sodium_quadratic', None)

        # Explanation steps (optional) – used for modeling and monotonicity checks
        self.explanation_steps: Optional[str] = getattr(solution_instance, 'explanation_steps', None)

        # Givens – not explicitly defined in the problem, but we can check for presence
        self.givens: Dict[str, Any] = getattr(solution_instance, 'givens', {})

        # Additional optional attributes for completeness
        self.pseudocode: Optional[str] = getattr(solution_instance, 'pseudocode', None)

    # --------------------------------------------------------------------------
    # Interface tests – verify that required result keys exist and have correct type
    # --------------------------------------------------------------------------
    def test_interface_hydrogen_linear_exists(self) -> Tuple[bool, str]:
        """Check that the key 'hydrogen_linear' exists in results."""
        if 'hydrogen_linear' not in self.results:
            return (False, "Missing result key: 'hydrogen_linear'")
        return (True, "'hydrogen_linear' key present")

    def test_interface_sodium_quadratic_exists(self) -> Tuple[bool, str]:
        """Check that the key 'sodium_quadratic' exists in results."""
        if 'sodium_quadratic' not in self.results:
            return (False, "Missing result key: 'sodium_quadratic'")
        return (True, "'sodium_quadratic' key present")

    def test_interface_hydrogen_linear_type(self) -> Tuple[bool, str]:
        """Check that 'hydrogen_linear' is a boolean (or at least truthy/falsy)."""
        if self.hydrogen_linear is None:
            return (False, "'hydrogen_linear' is None")
        if not isinstance(self.hydrogen_linear, (bool, int)):
            return (False, f"'hydrogen_linear' has type {type(self.hydrogen_linear).__name__}, expected bool")
        return (True, "'hydrogen_linear' type is acceptable")

    def test_interface_sodium_quadratic_type(self) -> Tuple[bool, str]:
        """Check that 'sodium_quadratic' is a boolean (or at least truthy/falsy)."""
        if self.sodium_quadratic is None:
            return (False, "'sodium_quadratic' is None")
        if not isinstance(self.sodium_quadratic, (bool, int)):
            return (False, f"'sodium_quadratic' has type {type(self.sodium_quadratic).__name__}, expected bool")
        return (True, "'sodium_quadratic' type is acceptable")

    # --------------------------------------------------------------------------
    # Invariant tests – physics theorem checks
    # --------------------------------------------------------------------------
    def test_invariant_linear_stark_requires_degeneracy(self) -> Tuple[bool, str]:
        """
        The linear Stark effect in hydrogen must be attributed to degeneracy
        between opposite‑parity states. Check that the explanation mentions this.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps available")
        text = self.explanation_steps.lower()
        has_degeneracy = "degener" in text
        has_opposite_parity = "opposite parity" in text or "different parity" in text
        if has_degeneracy and has_opposite_parity:
            return (True, "Explanation correctly identifies degeneracy and opposite parity")
        else:
            return (False, f"Explanation lacks degeneracy and/or opposite parity concepts. "
                           f"Found degeneracy={has_degeneracy}, opposite_parity={has_opposite_parity}")

    def test_invariant_quadratic_stark_non_degenerate(self) -> Tuple[bool, str]:
        """
        Quadratic Stark effect in sodium must be attributed to non‑degeneracy
        and vanishing first‑order correction.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps available")
        text = self.explanation_steps.lower()
        non_degen = "non-degener" in text or "no degener" in text or "without degener" in text
        first_order_zero = "first order" in text and ("zero" in text or "vanishes" in text or "0" in text)
        if non_degen and first_order_zero:
            return (True, "Explanation correctly describes non‑degeneracy and vanishing first‑order shift")
        else:
            return (False, f"Explanation lacks non‑degeneracy and/or vanishing first‑order: "
                           f"non_degen={non_degen}, first_order_zero={first_order_zero}")

    def test_invariant_parity_selection_rule(self) -> Tuple[bool, str]:
        """
        First‑order Stark shift ∝ ⟨z⟩, which vanishes for parity eigenstates.
        Verify that the explanation states this rule.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps available")
        text = self.explanation_steps.lower()
        parity_mention = "parity" in text
        first_order_mention = "first order" in text
        if parity_mention and first_order_mention:
            return (True, "Parity selection rule mentioned in context of first‑order perturbation")
        else:
            return (False, f"Parity or first‑order missing: parity={parity_mention}, first_order={first_order_mention}")

    def test_invariant_hydrogen_linear_flag_true(self) -> Tuple[bool, str]:
        """Hydrogen linear effect should be flagged as True."""
        if self.hydrogen_linear is None:
            return (False, "SKIP: hydrogen_linear not available")
        if not self.hydrogen_linear:
            return (False, "hydrogen_linear is False, but linear Stark effect is expected for excited hydrogen")
        return (True, "hydrogen_linear is True")

    def test_invariant_sodium_quadratic_flag_true(self) -> Tuple[bool, str]:
        """Sodium quadratic effect should be flagged as True."""
        if self.sodium_quadratic is None:
            return (False, "SKIP: sodium_quadratic not available")
        if not self.sodium_quadratic:
            return (False, "sodium_quadratic is False, but quadratic Stark effect is expected for sodium")
        return (True, "sodium_quadratic is True")

    # --------------------------------------------------------------------------
    # Modeling / trace tests – check consistency with the reasoning plan
    # --------------------------------------------------------------------------
    def test_modeling_explains_hydrogen_degeneracy(self) -> Tuple[bool, str]:
        """
        Explanation should explicitly mention that for n≥2, states with different l
        (and opposite parity) are degenerate in the absence of fine structure.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps available")
        text = self.explanation_steps.lower()
        hydrogen_mention = "hydrogen" in text
        degeneracy_l = ("degeneracy" in text and "l" in text) or ("degenerate" in text and "l" in text)
        if hydrogen_mention and degeneracy_l:
            return (True, "Hydrogen degeneracy with respect to l is mentioned")
        else:
            return (False, f"Missing hydrogen degeneracy explanation: hydrogen={hydrogen_mention}, degeneracy_l={degeneracy_l}")

    def test_modeling_explains_sodium_quantum_defects(self) -> Tuple[bool, str]:
        """
        Explanation should mention quantum defects and/or fine structure as the
        reason for non‑degeneracy in sodium.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps available")
        text = self.explanation_steps.lower()
        sodium_mention = "sodium" in text or "na" in text
        defects = "quantum defect" in text or "fine structure" in text or "spin-orbit" in text
        if sodium_mention and defects:
            return (True, "Sodium non‑degeneracy is attributed to quantum defects/fine structure")
        else:
            return (False, f"Missing sodium non‑degeneracy cause: sodium={sodium_mention}, defects={defects}")

    def test_modeling_second_order_quadratic(self) -> Tuple[bool, str]:
        """
        Explanation should state that the quadratic effect arises from second‑order
        perturbation theory.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps available")
        text = self.explanation_steps.lower()
        second_order = "second order" in text or "second‑order" in text
        quadratic = "quadratic" in text
        if second_order and quadratic:
            return (True, "Explanation connects quadratic Stark shift to second‑order perturbation theory")
        else:
            return (False, f"Missing second‑order/quadratic link: second_order={second_order}, quadratic={quadratic}")

    # --------------------------------------------------------------------------
    # Monotonicity tests – perturb an input and compare to baseline
    # --------------------------------------------------------------------------
    def test_monotonicity_hydrogen_linear_scaling(self) -> Tuple[bool, str]:
        """
        For a linear Stark effect, doubling the electric field doubles the shift.
        This test checks that the explanation (baseline) mentions proportionality to E.
        We perturb the field conceptually: if baseline is linear, then perturbed shift
        should be scaled accordingly; the explanation must reflect this linear scaling.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps available")
        text = self.explanation_steps.lower()
        # Baseline: the shift is linear in E
        baseline_linear = ("linear" in text and "stark" in text) or ("linear" in text and "proportional" in text)
        if not baseline_linear:
            return (False, "Explanation does not describe the hydrogen Stark shift as linear in the electric field")
        # Perturbation: double the field -> shift doubles
        # We check that the explanation's linearity is consistent with scaling
        # (no contradiction, e.g., saying it is quadratic)
        contradicts_linear = "quadratic" in text and "hydrogen" in text
        if contradicts_linear:
            return (False, "Explanation contradicts linear scaling by mentioning quadratic in hydrogen context")
        return (True, "Explanation supports linear scaling (shift ∝ E), consistent with perturbation by field doubling")

    def test_monotonicity_sodium_quadratic_scaling(self) -> Tuple[bool, str]:
        """
        For a quadratic Stark effect, doubling the electric field quadruples the shift.
        This test checks that the explanation (baseline) mentions proportionality to E^2.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps available")
        text = self.explanation_steps.lower()
        # Baseline: the shift is quadratic in E
        baseline_quadratic = ("quadratic" in text and "stark" in text) or ("quadratic" in text and "proportional" in text)
        if not baseline_quadratic:
            return (False, "Explanation does not describe the sodium Stark shift as quadratic in the electric field")
        # Check that it doesn't mistakenly claim linear for sodium
        contradicts_quadratic = "linear" in text and "sodium" in text
        if contradicts_quadratic:
            return (False, "Explanation contradicts quadratic scaling by mentioning linear in sodium context")
        return (True, "Explanation supports quadratic scaling (shift ∝ E^2), consistent with perturbation by field doubling")

    # --------------------------------------------------------------------------
    # Weak final‑answer sanity (do NOT dominate the checker)
    # --------------------------------------------------------------------------
    def test_answer_sanity_flags_are_boolean(self) -> Tuple[bool, str]:
        """Both results should be interpretable as booleans and not None."""
        if self.hydrogen_linear is None or self.sodium_quadratic is None:
            return (False, "One or both result values are None")
        return (True, "Both result keys have non‑None values")

    # --------------------------------------------------------------------------
    # Run all tests
    # --------------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all bound methods starting with 'test_', execute each inside a try/except,
        collect messages, and return (overall_pass, list of log messages).
        """
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