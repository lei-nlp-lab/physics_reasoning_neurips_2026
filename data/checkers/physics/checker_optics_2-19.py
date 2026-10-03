import math
import traceback
from typing import Any, Tuple

class PhysicsChecker:
    """
    Audits a physics solution for a problem involving flatness and parallelism
    measurement of a glass window.  Focuses on the parallelism check using
    Fizeau interferometry.
    """

    def __init__(self, solution_instance: Any):
        # ------------------------------------------------------------------
        # Extract givens (expected keys: 'lambda', 'n', 'theta_spec', 'D_window')
        # Use .get() on a dict, or getattr on object if necessary.
        # ------------------------------------------------------------------
        givens = getattr(solution_instance, 'givens', {})
        self.given_lambda = givens.get('lambda', None)
        self.given_n = givens.get('n', None)
        self.given_theta_spec = givens.get('theta_spec', None)  # in radians
        self.given_D_window = givens.get('D_window', None)

        # ------------------------------------------------------------------
        # Extract results (expected keys: 'S_min' from pseudocode emit)
        # ------------------------------------------------------------------
        results = getattr(solution_instance, 'results', {})
        self.S_min = results.get('S_min', None)

        # ------------------------------------------------------------------
        # Extract explanation steps (used in modelling tests)
        # ------------------------------------------------------------------
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # ------------------------------------------------------------------
        # Pre‑compute expected S_min and the product invariant for use in tests,
        # but ONLY when all required givens are available.
        # ------------------------------------------------------------------
        self._expected_S_min = None
        self._invariant_product = None
        if (self.given_lambda is not None and
            self.given_n is not None and
            self.given_theta_spec is not None):
            lam = self.given_lambda
            n = self.given_n
            theta = self.given_theta_spec
            self._expected_S_min = lam / (2 * n * theta)
            self._invariant_product = lam / (2 * n * theta) * theta * 2 * n  # simplifies to lam

    # ------------------------------------------------------------------
    # Helper: safe numeric extraction (to avoid crashes)
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_float(val: Any) -> float | None:
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # REQUIRED: run_all_tests – discover all test_ methods and run them
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, list]:
        logs = []
        all_pass = True
        # Collect all callable methods whose names start with 'test_'
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name, None)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {e}"
                traceback.print_exc()
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

    # ==================================================================
    # INTERFACE TEST: required result keys are present and of correct type
    # ==================================================================

    def test_interface_results_key_S_min(self) -> Tuple[bool, str]:
        """Check that 'S_min' exists in results and is numeric."""
        if self.S_min is None:
            return (False, "Missing result: 'S_min' not found in solution.results")
        s = self._safe_float(self.S_min)
        if s is None:
            return (False, f"'S_min' is not numeric: {self.S_min}")
        return (True, f"'S_min' is present and numeric: {s}")

    # ==================================================================
    # INVARIANT TESTS (theorem‑based physical laws)
    # ==================================================================

    def test_invariant_fringe_spacing_formula(self) -> Tuple[bool, str]:
        """
        Verify that S_min = lambda / (2 * n * theta_spec) within tolerance.
        This is the core Fizeau fringe spacing relation.
        """
        if self._expected_S_min is None:
            return (False, "SKIP: missing required givens (lambda, n, theta_spec) to compute expected S_min")
        if self.S_min is None:
            return (False, "SKIP: S_min not available in results")
        s = self._safe_float(self.S_min)
        if s is None:
            return (False, f"S_min is not numeric: {self.S_min}")
        exp = self._expected_S_min
        # Use math.isclose with 1% relative tolerance (lenient for rounding)
        if math.isclose(s, exp, rel_tol=1e-2, abs_tol=1e-6):
            return (True, f"S_min = {s} matches expected {exp} (λ/(2nθ))")
        else:
            return (False, f"S_min = {s} differs from expected {exp} (λ/(2nθ))")

    def test_invariant_product_formula(self) -> Tuple[bool, str]:
        """
        Verify the algebraic identity: (S_min * theta_spec * 2 * n) == lambda.
        This is a direct consequence of the fringe spacing formula.
        """
        if self._invariant_product is None:
            return (False, "SKIP: missing required givens to compute invariant product")
        if self.S_min is None:
            return (False, "SKIP: S_min not available in results")
        s = self._safe_float(self.S_min)
        if s is None:
            return (False, f"S_min is not numeric: {self.S_min}")
        # compute product from solution's S_min, givens
        lam = self.given_lambda
        n = self.given_n
        theta = self.given_theta_spec
        product = s * theta * 2 * n
        if math.isclose(product, lam, rel_tol=1e-2, abs_tol=1e-6):
            return (True, f"Product S_min*θ*2n = {product} matches λ = {lam}")
        else:
            return (False, f"Product S_min*θ*2n = {product} does not match λ = {lam}")

    # ==================================================================
    # MODELING / TRACE QUALITY TESTS (at least two)
    # ==================================================================

    def test_modeling_formula_includes_refractive_index(self) -> Tuple[bool, str]:
        """
        Check that the explanation steps (or pseudocode) mention the refractive
        index n in the fringe‑spacing formula.  This ensures the model is correct.
        """
        steps = self.explanation_steps
        if steps is None:
            return (False, "SKIP: no explanation_steps provided")
        # Convert to a single string if it's a list
        if isinstance(steps, (list, tuple)):
            text = " ".join(str(step) for step in steps)
        else:
            text = str(steps)

        # Look for phrases that indicate n is used: e.g. "λ/(2nθ)", "2nθ", "refractive index"
        keywords = ["λ/(2nθ)", r"2n\theta", "refractive index", "n = 1.500"]
        found = any(kw in text for kw in keywords)
        if found:
            return (True, "Explanation steps correctly reference refractive index in fringe‑spacing formula")
        else:
            return (False, "Explanation steps do not mention n in the fringe‑spacing formula (expected λ/(2nθ))")

    def test_modeling_computes_fringe_spacing(self) -> Tuple[bool, str]:
        """
        Check that the explanation steps include a calculation of the fringe spacing
        S_min = λ/(2nθ).  This verifies that the modeling addresses the required metric.
        """
        steps = self.explanation_steps
        if steps is None:
            return (False, "SKIP: no explanation_steps provided")
        # Convert to a single string if it's a list
        if isinstance(steps, (list, tuple)):
            text = " ".join(str(step) for step in steps)
        else:
            text = str(steps)

        # Look for numeric results like "0.75 cm" or "S >= 0.75 cm" or explicit calculation
        key_phrases = ["0.75 cm", "S ≥ 0.75", "S_min", "fringe spacing"]
        found = any(phrase in text for phrase in key_phrases)
        if found:
            return (True, "Explanation steps contain a computed fringe spacing value (or related expression)")
        else:
            return (False, "Explanation steps do not show the computation of the fringe spacing threshold")

    # ==================================================================
    # MONOTONICITY TESTS: explicit perturbation of an input parameter
    # ==================================================================

    def test_monotonicity_S_min_with_lambda(self) -> Tuple[bool, str]:
        """
        Test monotonicity: if λ increases, S_min (computed from the physical formula)
        increases proportionally.
        """
        if self._expected_S_min is None:
            return (False, "SKIP: cannot compute baseline S_min (missing givens)")
        lam = self.given_lambda
        n = self.given_n
        theta = self.given_theta_spec
        S_baseline = self._expected_S_min
        S_perturbed = (1.1 * lam) / (2 * n * theta)
        if S_perturbed > S_baseline:
            return (True, f"Monotonicity holds: λ↑ → S↑ (baseline {S_baseline:.2e}, perturbed {S_perturbed:.2e})")
        else:
            return (False, f"Monotonicity violated: λ↑ should increase S, but computed {S_perturbed} ≤ {S_baseline}")

    def test_monotonicity_S_min_with_theta(self) -> Tuple[bool, str]:
        """
        Test monotonicity: if θ_spec increases, S_min decreases.
        """
        if self._expected_S_min is None:
            return (False, "SKIP: cannot compute baseline S_min (missing givens)")
        lam = self.given_lambda
        n = self.given_n
        theta = self.given_theta_spec
        S_baseline = self._expected_S_min
        S_perturbed = lam / (2 * n * (1.1 * theta))
        if S_perturbed < S_baseline:
            return (True, f"Monotonicity holds: θ↑ → S↓ (baseline {S_baseline:.2e}, perturbed {S_perturbed:.2e})")
        else:
            return (False, f"Monotonicity violated: θ↑ should decrease S, but computed {S_perturbed} ≥ {S_baseline}")

    # ==================================================================
    # ANSWER SANITY (weak; only one such method allowed, minimal)
    # ==================================================================

    def test_answer_sanity_S_min_positive_and_finite(self) -> Tuple[bool, str]:
        """
        Very weak sanity: S_min must be positive and finite.
        Do NOT re‑derive expected value or use math.isclose.
        """
        if self.S_min is None:
            return (False, "SKIP: S_min not available")
        s = self._safe_float(self.S_min)
        if s is None:
            return (False, f"S_min is not numeric: {self.S_min}")
        if math.isnan(s) or math.isinf(s):
            return (False, f"S_min is NaN or Inf: {s}")
        if s <= 0:
            return (False, f"S_min is non‑positive: {s}")
        # Also check that S_min is not absurdly large (e.g., > 10 cm)
        # The window diameter is 2 cm, so fringes must fit; but we avoid derivation.
        # Instead just a rough sanity: S_min < 1 m.
        if s > 1.0:
            return (False, f"S_min = {s} m is unreasonably large for an optical test")
        return (True, f"S_min is positive, finite, and of plausible magnitude: {s}")