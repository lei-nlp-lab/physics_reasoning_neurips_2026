import math
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Audits a solution_instance for the He³ cooling problem.
    Checks focus on:
        - Interface: required result key 'T_min' exists and is numeric.
        - Invariant: if gamma is provided, T_min must equal ln2/gamma.
        - Modeling: explanation steps/equations should mention Clausius-Clapeyron and entropy difference.
        - Monotonicity: if gamma is given, T_min decreases when gamma increases.
        - Weak answer sanity: T_min is in plausible range.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract results (primary)
        results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        self.T_min: Optional[float] = results.get('T_min', None)

        # Extract givens (secondary)
        givens: Dict[str, Any] = getattr(solution_instance, 'givens', {})
        self.gamma: Optional[float] = getattr(solution_instance, 'gamma', None)
        if self.gamma is None:
            self.gamma = givens.get('gamma', None)
        if self.gamma is None:
            self.gamma = results.get('gamma', None)  # fallback

        # Explanation steps (optional)
        self.explanation_steps: Optional[str] = getattr(solution_instance, 'explanation_steps', None)
        if self.explanation_steps is None:
            self.explanation_steps = results.get('explanation_steps', None)

        # Pseudocode (optional, for trace consistency)
        self.pseudocode: Optional[str] = getattr(solution_instance, 'pseudocode', None)
        if self.pseudocode is None:
            self.pseudocode = results.get('pseudocode', None)

        # R constant – not needed for T_min check, but used for sign analysis
        self.R: Optional[float] = getattr(solution_instance, 'R', None)
        if self.R is None:
            self.R = givens.get('R', None)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_T_min_exists(self) -> Tuple[bool, str]:
        """Check that T_min is present and is a finite number."""
        if self.T_min is None:
            return (False, "T_min is None")
        if not isinstance(self.T_min, (int, float)):
            return (False, f"T_min has bad type: {type(self.T_min)}")
        if math.isnan(self.T_min) or math.isinf(self.T_min):
            return (False, f"T_min is NaN or inf: {self.T_min}")
        return (True, "T_min exists and is finite")

    # ------------------------------------------------------------------
    # Invariant tests (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_T_min_positive(self) -> Tuple[bool, str]:
        """T_min must be positive."""
        if self.T_min is None:
            return (False, "SKIP: T_min not available")
        if self.T_min <= 0:
            return (False, f"T_min = {self.T_min} is not positive")
        return (True, f"T_min = {self.T_min} > 0")

    def test_invariant_T_min_exact(self) -> Tuple[bool, str]:
        """
        If gamma is given, T_min must equal ln2/gamma.
        Use math.isclose with tolerances suitable for rounding noise.
        """
        if self.gamma is None or self.gamma <= 0:
            return (False, "SKIP: gamma not available or non-positive")
        if self.T_min is None:
            return (False, "SKIP: T_min not available")
        expected = math.log(2) / self.gamma
        if math.isclose(self.T_min, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"T_min = {self.T_min} ≈ ln2/gamma = {expected}")
        else:
            return (False, f"T_min = {self.T_min}, expected ln2/gamma = {expected}")

    # ------------------------------------------------------------------
    # Modeling / trace tests
    # ------------------------------------------------------------------
    def test_modeling_entropy_expression(self) -> Tuple[bool, str]:
        """Check that the explanation or pseudocode mentions the entropy expressions."""
        text = (self.explanation_steps or "") + (self.pseudocode or "")
        if not text:
            return (False, "SKIP: no explanation_steps or pseudocode available")
        has_SL = ("S_L" in text or "S_L = " in text or "γRT" in text)
        has_SS = ("S_S" in text or "R ln 2" in text or "R*ln(2)" in text)
        if has_SL and has_SS:
            return (True, "Entropy expressions found in explanation/pseudocode")
        else:
            missing = []
            if not has_SL:
                missing.append("S_L expression")
            if not has_SS:
                missing.append("S_S expression")
            return (False, f"Missing: {', '.join(missing)}")

    def test_modeling_clausius_clapeyron(self) -> Tuple[bool, str]:
        """Check that the Clausius-Clapeyron relation appears in the solution description."""
        text = (self.explanation_steps or "") + (self.pseudocode or "")
        if not text:
            return (False, "SKIP: no explanation_steps or pseudocode available")
        keywords = ["Clausius", "Clapeyron", "dP/dT", "dP/dT =", "ΔS/ΔV", "deltaS/deltaV"]
        found = [kw for kw in keywords if kw in text]
        if found:
            return (True, f"Clausius-Clapeyron mentioned via: {', '.join(found)}")
        else:
            return (False, "Clausius-Clapeyron equation not found in explanation/pseudocode")

    # ------------------------------------------------------------------
    # Monotonicity tests (must perturb input and compare perturbed output to baseline)
    # ------------------------------------------------------------------
    def test_monotonicity_T_min_inverse_gamma(self) -> Tuple[bool, str]:
        """
        If gamma is provided, check that increasing gamma decreases T_min.
        Baseline: T_min0 = ln2/gamma.
        Perturb: gamma' = 1.1 * gamma, then T_min' = ln2/gamma'.
        Assert T_min' < T_min0 (monotonic decreasing with gamma).
        """
        if self.gamma is None or self.gamma <= 0:
            return (False, "SKIP: gamma not available or non-positive")
        if self.T_min is None:
            return (False, "SKIP: T_min not available")
        # Use the given gamma to compute baseline T_min (expected)
        baseline_Tmin = math.log(2) / self.gamma
        # Perturb gamma upward
        gamma_perturbed = self.gamma * 1.1
        perturbed_Tmin = math.log(2) / gamma_perturbed
        if perturbed_Tmin < baseline_Tmin:
            return (True, f"Perturbing gamma by +10% reduces T_min: {baseline_Tmin} -> {perturbed_Tmin}")
        else:
            return (False, f"Monotonicity violated: perturbed T_min = {perturbed_Tmin} >= baseline {baseline_Tmin}")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_broad_range(self) -> Tuple[bool, str]:
        """
        Weak sanity: T_min should be in a physically plausible range.
        The problem states gamma ~ 4.6 K^{-1}, so ln2/4.6 ≈ 0.15 K.
        Allow a factor of 10 around that.
        """
        if self.T_min is None:
            return (False, "SKIP: T_min not available")
        # Expected about 0.15 K, so range 0.015 K to 1.5 K
        if 0.0 < self.T_min < 1.5:
            return (True, f"T_min = {self.T_min} K is within plausible range")
        else:
            return (False, f"T_min = {self.T_min} K is outside plausible range (0.015–1.5 K)")

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all bound methods starting with 'test_', execute them in try/except,
        and return (all_pass, logs).
        """
        logs: List[str] = []
        all_pass = True
        method_names = sorted([name for name in dir(self) if name.startswith("test_") and callable(getattr(self, name))])
        for name in method_names:
            method = getattr(self, name)
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)