import math
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Auditor for a solution estimating the s-wave scattering threshold temperature
    for hydrogen gas atoms. Checks interface completeness, invariant physics relationships,
    modeling consistency from explanation steps, monotonicity, and a weak final answer sanity.
    """

    def __init__(self, solution_instance: Any) -> None:
        # --- Extract results dictionary safely ---
        self.results: Optional[Dict] = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.T0 = None
            self.T_estimate = None
        else:
            self.T0 = self.results.get('T0', None)
            self.T_estimate = self.results.get('T_estimate', None)

        # --- Extract other relevant attributes (explanation, pseudocode) ---
        self.explanation_steps: Optional[Any] = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode: Optional[str] = getattr(solution_instance, 'pseudocode', None)
        self.pseudocode_plan: Optional[str] = getattr(solution_instance, 'pseudocode_plan', None)

        # --- Extract physical constants from solution_instance (with None defaults) ---
        self.hbar = self._get_constant(solution_instance, 'hbar')
        self.kB = self._get_constant(solution_instance, 'kB')
        self.m_H = self._get_constant(solution_instance, 'm_H')
        self.R = self._get_constant(solution_instance, 'R')

    # ----------------------------------------------------------------------
    # Helper: extract a numeric constant from solution_instance
    # ----------------------------------------------------------------------
    def _get_constant(self, inst: Any, name: str) -> Optional[float]:
        # try direct attribute
        val = getattr(inst, name, None)
        if val is not None:
            return val
        # try givens dict
        givens = getattr(inst, 'givens', None)
        if isinstance(givens, dict):
            val = givens.get(name, None)
            if val is not None:
                return val
        return None

    # ----------------------------------------------------------------------
    # INTERFACE TESTS
    # ----------------------------------------------------------------------
    def test_interface_results_keys(self) -> Tuple[bool, str]:
        """Check that the results dictionary contains the required keys."""
        if self.results is None:
            return False, "results dictionary is missing"
        missing = []
        for key in ('T0', 'T_estimate'):
            if key not in self.results:
                missing.append(key)
        if missing:
            return False, f"results missing key(s): {missing}"
        return True, "all required result keys present"

    def test_interface_results_value_types(self) -> Tuple[bool, str]:
        """Ensure T0 and T_estimate are finite numbers."""
        if self.T0 is None:
            return False, "T0 is None"
        if self.T_estimate is None:
            return False, "T_estimate is None"
        for name, val in [('T0', self.T0), ('T_estimate', self.T_estimate)]:
            if not isinstance(val, (int, float)):
                return False, f"{name} has invalid type: {type(val)}"
            if not math.isfinite(val):
                return False, f"{name} is not finite"
        return True, "T0 and T_estimate are finite numbers"

    # ----------------------------------------------------------------------
    # INVARIANT TESTS (physics theorem checks, use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_T0_formula(self) -> Tuple[bool, str]:
        """Verify that T0 matches the theoretical formula T0 = ħ²/(2μ R² kB), μ = m_H/2."""
        if any(v is None for v in (self.hbar, self.kB, self.m_H, self.R, self.T0)):
            return False, "SKIP: missing one or more constants (hbar, kB, m_H, R, T0)"
        mu = self.m_H / 2.0
        expected_T0 = self.hbar**2 / (2.0 * mu * self.R**2 * self.kB)
        if math.isclose(self.T0, expected_T0, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"T0 matches formula within 1%: T0={self.T0:.4e}, expected ~{expected_T0:.4e}"
        else:
            return False, f"T0 deviates from formula: got {self.T0:.4e}, expected ~{expected_T0:.4e}"

    def test_invariant_T_estimate_less_than_T0(self) -> Tuple[bool, str]:
        """Check the fundamental inequality: T_estimate < T0."""
        if self.T0 is None or self.T_estimate is None:
            return False, "SKIP: T0 or T_estimate missing"
        if self.T_estimate < self.T0:
            return True, f"T_estimate={self.T_estimate:.4e} < T0={self.T0:.4e}"
        else:
            return False, f"T_estimate ({self.T_estimate:.4e}) is not less than T0 ({self.T0:.4e})"

    def test_invariant_positive_values(self) -> Tuple[bool, str]:
        """Ensure both temperatures are positive."""
        if self.T0 is None or self.T_estimate is None:
            return False, "SKIP: temperatures missing"
        if self.T0 <= 0:
            return False, f"T0 is not positive: {self.T0}"
        if self.T_estimate <= 0:
            return False, f"T_estimate is not positive: {self.T_estimate}"
        return True, "both T0 and T_estimate are positive"

    # ----------------------------------------------------------------------
    # MODELING / TRACE TESTS (based on explanation_steps and pseudocode)
    # ----------------------------------------------------------------------
    def test_trace_explanation_contains_condition(self) -> Tuple[bool, str]:
        """Verify that the explanation or pseudocode mentions the s‑wave condition (kR << 1)."""
        if self.explanation_steps is None and self.pseudocode is None and self.pseudocode_plan is None:
            return False, "SKIP: no explanation_steps or pseudocode available"
        # combine all text into one string for search
        text_parts = []
        if self.explanation_steps:
            if isinstance(self.explanation_steps, list):
                text_parts.extend(self.explanation_steps)
            else:
                text_parts.append(str(self.explanation_steps))
        if self.pseudocode:
            text_parts.append(self.pseudocode)
        if self.pseudocode_plan:
            text_parts.append(self.pseudocode_plan)
        full_text = " ".join(text_parts).lower()
        # Check for common representations: "kr", "ka", "s-wave"
        if "kr" in full_text or "ka" in full_text or "s-wave" in full_text:
            return True, "explanation/pseudocode contains s‑wave condition (kR or ka)"
        else:
            return False, "explanation/pseudocode does not mention the s‑wave condition (kR << 1)"

    def test_trace_T0_T_estimate_ratio(self) -> Tuple[bool, str]:
        """Check that the solution uses the conservative estimate T_estimate ≈ T0/3, as in pseudocode."""
        if self.T0 is None or self.T_estimate is None:
            return False, "SKIP: T0 or T_estimate missing"
        expected_est = self.T0 / 3.0
        if math.isclose(self.T_estimate, expected_est, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"T_estimate ≈ T0/3: T0/3={expected_est:.4f}, T_estimate={self.T_estimate:.4f}"
        else:
            return False, f"T_estimate does not match T0/3: T0/3={expected_est:.4f}, T_estimate={self.T_estimate:.4f}"

    # ----------------------------------------------------------------------
    # MONOTONICITY TEST
    # ----------------------------------------------------------------------
    def test_monotonicity_T0_decreases_with_R(self) -> Tuple[bool, str]:
        """
        Perturb the interaction range R upward by 10% and verify that the
        theoretical T0 (computed from the same formula) decreases.
        This checks the inherent monotonicity T0 ∝ 1/R².
        """
        if any(v is None for v in (self.hbar, self.kB, self.m_H, self.R, self.T0)):
            return False, "SKIP: missing constants or T0"
        mu = self.m_H / 2.0
        original_T0 = self.T0
        # Perturb R upward by 10%
        factor = 1.1
        perturbed_R = self.R * factor
        # Compute expected perturbed T0 using the same physics formula
        perturbed_T0 = self.hbar**2 / (2.0 * mu * perturbed_R**2 * self.kB)
        # Check monotonicity: perturbed_T0 must be smaller
        if perturbed_T0 < original_T0:
            return True, (
                f"Perturbing R by +{(factor-1)*100:.0f}% decreases T0 "
                f"from {original_T0:.4e} to {perturbed_T0:.4e} (as expected)"
            )
        else:
            return False, (
                f"Monotonicity violation: perturbed_T0={perturbed_T0:.4e} "
                f">= original_T0={original_T0:.4e}"
            )

    # ----------------------------------------------------------------------
    # WEAK ANSWER SANITY TEST (only one, no math.isclose, no re‑derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """
        Weak final‑answer sanity: check positivity, finiteness, inequality,
        and that T_estimate is of order 1 K (≤ 10 K).
        """
        if self.T0 is None or self.T_estimate is None:
            return False, "SKIP: temperatures missing"
        if not (self.T0 > 0 and self.T_estimate > 0):
            return False, "non‑positive temperature(s) detected"
        if not (self.T_estimate < self.T0):
            return False, "T_estimate is not less than T0"
        if self.T_estimate > 10:   # very weak bound
            return False, f"T_estimate={self.T_estimate} exceeds 10 K, unlikely"
        # ensure not NaN or infinite (already checked types)
        return True, "answer passes weak sanity: positive, finite, T_estimate<T0, and O(1 K)"

    # ----------------------------------------------------------------------
    # RUN ALL TESTS
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs: List[str] = []
        all_pass = True
        # iterate over all bound methods starting with "test_"
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
        return all_pass, logs