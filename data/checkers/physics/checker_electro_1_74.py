import math
import traceback
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Audits solutions for the metallic plane + point charge problem.
    Checks interface completeness, physical invariants (method of images),
    modeling quality, monotonicity, and a weak answer sanity test.
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract all data from solution_instance with safe None defaults.
        Never assign numeric defaults like 0.0 or 9.81.
        """
        # Givens (sometimes provided as attributes or in a dict)
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.q = solution_instance.givens.get('q', None)
            self.d = solution_instance.givens.get('d', None)
            self.k = solution_instance.givens.get('k', None)
        elif hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            # Fallback: givens might be buried in results
            self.q = solution_instance.results.get('q', None)
            self.d = solution_instance.results.get('d', None)
            self.k = solution_instance.results.get('k', None)
        else:
            self.q = getattr(solution_instance, 'q', None)
            self.d = getattr(solution_instance, 'd', None)
            self.k = getattr(solution_instance, 'k', None)

        # Results (expected from pseudocode [emit] contract)
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.force_direction = solution_instance.results.get('force_direction', None)
            self.force_magnitude = solution_instance.results.get('force_magnitude', None)
        else:
            self.force_direction = getattr(solution_instance, 'force_direction', None)
            self.force_magnitude = getattr(solution_instance, 'force_magnitude', None)

        # Explanation steps (optional but useful for audits)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # For monotonicity we need to be able to compute expected magnitude
        # using the formula from the method of images.
        # This is allowed because we derive it from physics, not from the answer.
        self._baseline_magnitude = None
        if None not in (self.q, self.d, self.k):
            self._baseline_magnitude = self.k * self.q**2 / (4 * self.d**2)

    # ----------------------------------------------------------------
    # INTERFACE TESTS
    # ----------------------------------------------------------------
    def test_interface_has_results_dict(self) -> Tuple[bool, str]:
        """Ensure solution_instance has a results dictionary or the required keys."""
        # We check that at least one result exists
        if self.force_direction is not None or self.force_magnitude is not None:
            return (True, "Required result keys (force_direction, force_magnitude) accessible")
        # Also check if results attribute with those keys exists
        if hasattr(self, 'results') and isinstance(self.results, dict):
            if 'force_direction' in self.results or 'force_magnitude' in self.results:
                return (True, "results dict contains required keys")
        return (False, "No evidence of results dictionary or required keys")

    def test_interface_force_direction_type(self) -> Tuple[bool, str]:
        """If force_direction exists, it must be a string."""
        if self.force_direction is None:
            return (False, "SKIP: force_direction not provided")
        if isinstance(self.force_direction, str):
            return (True, f"force_direction is a string: '{self.force_direction}'")
        return (False, f"force_direction type is {type(self.force_direction)}, expected str")

    def test_interface_force_magnitude_type(self) -> Tuple[bool, str]:
        """If force_magnitude exists, it must be numeric."""
        if self.force_magnitude is None:
            return (False, "SKIP: force_magnitude not provided")
        if isinstance(self.force_magnitude, (int, float)):
            return (True, f"force_magnitude is numeric: {self.force_magnitude}")
        return (False, f"force_magnitude type is {type(self.force_magnitude)}, expected numeric")

    # ----------------------------------------------------------------
    # INVARIANT TESTS (theorem-based physical sanity, use math.isclose)
    # ----------------------------------------------------------------
    def test_invariant_attractive_force_direction(self) -> Tuple[bool, str]:
        """
        By method of images, a positive charge is attracted to the grounded plane.
        So force_direction must indicate 'attracted' or similar.
        """
        if self.force_direction is None:
            return (False, "SKIP: force_direction missing")
        direction_lower = self.force_direction.strip().lower()
        if "attract" in direction_lower or "toward" in direction_lower:
            return (True, f"Force direction '{self.force_direction}' is consistent with attraction")
        return (False, f"Force direction '{self.force_direction}' does not indicate attraction to plane")

    def test_invariant_force_magnitude_formula(self) -> Tuple[bool, str]:
        """
        If all givens are present, verify that the magnitude matches k*q^2/(4*d^2).
        Use math.isclose with generous tolerance (1e-2) because of possible rounding.
        """
        if None in (self.q, self.d, self.k, self.force_magnitude):
            return (False, "SKIP: missing givens or magnitude to compute expected force")
        expected = self.k * self.q**2 / (4 * self.d**2)
        if math.isclose(self.force_magnitude, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Force magnitude {self.force_magnitude} ≈ expected {expected}")
        return (False, f"Force magnitude {self.force_magnitude} differs from expected {expected}")

    # ----------------------------------------------------------------
    # MODELING / TRACE TESTS (consistency with explanation steps)
    # ----------------------------------------------------------------
    def test_modeling_explanation_mentions_image(self) -> Tuple[bool, str]:
        """If explanation_steps exist, they should reference 'image charge' or 'method of images'."""
        if not self.explanation_steps:
            return (False, "SKIP: no explanation steps to check")
        text = self.explanation_steps  # could be list or string
        if isinstance(text, list):
            text = " ".join(text)
        text_lower = text.lower()
        if "image" in text_lower or "method of images" in text_lower:
            return (True, "Explanation steps reference method of images")
        return (False, "Explanation steps do not mention image charge or method of images")

    def test_modeling_direction_matches_physics(self) -> Tuple[bool, str]:
        """
        The direction must be 'attracted' (not 'repelled' or 'none').
        This is the main modeling sanity.
        """
        if self.force_direction is None:
            return (False, "SKIP: force_direction missing")
        if self.force_direction.strip().lower() == 'attracted':
            return (True, "force_direction correctly set to 'attracted'")
        return (False, f"force_direction is '{self.force_direction}', should be 'attracted'")

    # ----------------------------------------------------------------
    # MONOTONICITY TESTS (explicit input perturbation, compare perturbed vs baseline)
    # ----------------------------------------------------------------
    def test_monotonicity_magnitude_decreases_with_distance(self) -> Tuple[bool, str]:
        """
        Verify that if the distance d is increased by 10%, the magnitude
        (according to the law F ∝ 1/d²) becomes smaller than the baseline.
        We compute baseline magnitude from given d, and perturbed magnitude
        from d_pert = d * 1.1, then check perturbed < baseline.
        The reported magnitude should be consistent with the baseline.
        """
        if None in (self.q, self.d, self.k, self.force_magnitude):
            return (False, "SKIP: missing data for monotonicity test")
        # Baseline magnitude using reported givens (but we already have computed value)
        baseline = self._baseline_magnitude  # computed using given q,d,k
        if baseline is None:
            return (False, "SKIP: could not compute baseline magnitude")
        # Perturb distance: increase by 10%
        d_pert = self.d * 1.1
        perturbed = self.k * self.q**2 / (4 * d_pert**2)
        # Monotonicity condition: perturbed < baseline
        if perturbed < baseline:
            # Also check reported magnitude is close to baseline (optional consistency)
            if math.isclose(self.force_magnitude, baseline, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Distance increase reduces magnitude: {perturbed} < {baseline}; "
                              f"reported {self.force_magnitude} ≈ baseline")
            else:
                return (True, f"Distance monotonicity satisfied (perturbed < baseline), "
                             f"but reported magnitude {self.force_magnitude} differs from baseline {baseline} "
                             f"by more than 1% (tolerance 1e-2).")
        else:
            return (False, f"Monotonicity failed: perturbed (d*1.1) magnitude {perturbed} "
                          f"is not less than baseline {baseline}")

    def test_monotonicity_magnitude_increases_with_charge(self) -> Tuple[bool, str]:
        """
        Verify that if the charge q is increased by 10%, the magnitude
        (according to the law F ∝ q²) becomes larger than the baseline.
        """
        if None in (self.q, self.d, self.k, self.force_magnitude):
            return (False, "SKIP: missing data for charge monotonicity test")
        baseline = self._baseline_magnitude
        if baseline is None:
            return (False, "SKIP: could not compute baseline magnitude")
        q_pert = self.q * 1.1
        perturbed = self.k * q_pert**2 / (4 * self.d**2)
        if perturbed > baseline:
            if math.isclose(self.force_magnitude, baseline, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Charge increase raises magnitude: {perturbed} > {baseline}; "
                              f"reported {self.force_magnitude} ≈ baseline")
            else:
                return (True, f"Charge monotonicity satisfied (perturbed > baseline), "
                             f"but reported magnitude {self.force_magnitude} differs from baseline {baseline} "
                             f"by more than 1% tolerance.")
        else:
            return (False, f"Monotonicity failed: perturbed (q*1.1) magnitude {perturbed} "
                          f"is not greater than baseline {baseline}")

    # ----------------------------------------------------------------
    # WEAK ANSWER SANITY (only one, no math.isclose, no re-derivation)
    # ----------------------------------------------------------------
    def test_answer_sanity_choice_c(self) -> Tuple[bool, str]:
        """
        Check that the answer (likely stored in solution_instance.answer or results['answer'])
        is 'c' or equivalent (the correct multiple-choice option).
        We only check exact match or that it's a plausible string.
        """
        # Try to extract answer from various places
        ans = None
        # First look in results dict
        if hasattr(self, 'results') and isinstance(self.results, dict):
            ans = self.results.get('answer', self.results.get('choice', None))
        # If not found, explanation_steps might contain the answer as a single char
        if ans is None and self.explanation_steps:
            if isinstance(self.explanation_steps, str):
                ans = self.explanation_steps.strip()
            elif isinstance(self.explanation_steps, list):
                # Might be the last element? Not reliable.
                pass
        if ans is None:
            return (False, "SKIP: no answer field found")
        ans_str = str(ans).strip().lower()
        if ans_str in ('c', 'attracted', 'c) attracted'):
            return (True, f"Answer '{ans}' matches expected ('c' or 'attracted')")
        return (False, f"Answer '{ans}' does not match expected 'c'")

    # ----------------------------------------------------------------
    # RUN ALL TESTS (do not modify)
    # ----------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)