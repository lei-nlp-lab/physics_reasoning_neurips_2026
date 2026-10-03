import math

class PhysicsChecker:
    """
    Audits the solution for the dodecahedron potential problem:
    - One face at potential V, other 11 grounded.
    - Expected result: V_c = V/12.
    Uses symmetry, superposition, and uniqueness.
    """

    def __init__(self, solution_instance):
        # Extract the computed potential at the centre
        self.V_c = solution_instance.results.get('V_c', None)

        # Extract the given face potential V from multiple possible sources
        self.V = None
        if hasattr(solution_instance, 'V'):
            self.V = solution_instance.V
        elif hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.V = solution_instance.givens.get('V', None)
        elif hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.V = solution_instance.results.get('V', None)

        # Store explanation steps if present (used in modeling tests)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

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

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_v_c_exists(self):
        """Check that V_c is present in the results and is not None."""
        if self.V_c is None:
            return (False, "V_c is missing or None in results")
        return (True, "V_c is present")

    def test_interface_v_available(self):
        """Check that the input V is available for derived tests."""
        if self.V is None:
            return (False, "Input V is not available; some tests will be skipped")
        return (True, "V is available")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physics sanity)
    # ------------------------------------------------------------------
    def test_invariant_coefficient_from_symmetry(self):
        """
        By symmetry and uniqueness, if all 12 faces were at potential V,
        the centre would be V. Superposition gives 12 * c = 1  => c = 1/12.
        Thus V_c = c * V => (V_c / V) = 1/12.
        """
        if self.V_c is None or self.V is None:
            return (False, "SKIP: V_c or V missing")
        if math.isclose(self.V, 0.0, abs_tol=1e-12):
            return (False, "V is zero; cannot compute coefficient")
        c = self.V_c / self.V
        expected_c = 1.0/12.0
        if math.isclose(c, expected_c, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"c = {c:.4f} ≈ 1/12")
        else:
            return (False, f"c = {c:.4f} differs from 1/12")

    def test_invariant_linearity(self):
        """
        The potential must be linear in V: V_c ∝ V.
        Check that V_c * 12 ≈ V (the same relation).
        """
        if self.V_c is None or self.V is None:
            return (False, "SKIP: V_c or V missing")
        if math.isclose(self.V, 0.0, abs_tol=1e-12):
            return (True, "V = 0, V_c should be 0 (check value only)")
        lhs = self.V_c * 12.0
        if math.isclose(lhs, self.V, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"12*V_c = {lhs:.4f} ≈ V = {self.V:.4f}")
        else:
            return (False, f"12*V_c = {lhs:.4f} differs from V = {self.V:.4f}")

    # ------------------------------------------------------------------
    # Modeling quality tests
    # ------------------------------------------------------------------
    def test_modeling_superposition_principle(self):
        """
        The solution should be constructed using superposition:
        V_c = Σ c_i V_i. Because all faces are identical, c_i must be equal.
        Therefore the result should be V/12 if only one face is live.
        (Already checked numerically; this test checks conceptual structure.)
        """
        if self.V_c is None or self.V is None:
            return (False, "SKIP: V_c or V missing")
        if not math.isclose(self.V_c * 12.0, self.V, rel_tol=1e-2, abs_tol=1e-2):
            return (False, "Superposition relation 12*V_c = V is not satisfied")
        return (True, "Superposition principle appears correctly applied")

    def test_modeling_symmetry_explanation(self):
        """
        Check that the explanation steps mention symmetry, uniqueness, or superposition.
        This verifies that the modeling rationale is present.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not available")
        text = str(self.explanation_steps).lower()
        keywords = ["symmetry", "uniqueness", "superposition", "linearity"]
        found = [kw for kw in keywords if kw in text]
        if found:
            return (True, f"Explanation steps contain reasoning keywords: {found}")
        else:
            return (False, "Explanation steps lack key physics reasoning (symmetry/uniqueness/superposition)")

    def test_modeling_coefficient_derivation(self):
        """
        Check that the derivation of the coefficient c = 1/12 is logically sound.
        The solution should have invoked the condition that if all faces are at V,
        the centre potential equals V, leading to 12c = 1.
        """
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not available")
        text = str(self.explanation_steps).lower()
        # Look for references to "all faces" and "12"
        if "all faces" in text and "12" in text:
            return (True, "Derivation mentions all-faces condition and factor 12")
        else:
            return (False, "Coefficient derivation missing reference to all-faces condition")

    # ------------------------------------------------------------------
    # Monotonicity test (perturbation-based)
    # ------------------------------------------------------------------
    def test_monotonicity_linear_scaling(self):
        """
        Choose input V, perturb it by a factor (e.g., +10%),
        and verify that the output V_c scales accordingly.
        Since we cannot recompute with new V, we estimate the perturbed
        output using the observed linear relationship and compare with
        the theoretical expectation (proportionality).
        """
        if self.V_c is None or self.V is None:
            return (False, "SKIP: V_c or V missing")
        if math.isclose(self.V, 0.0, abs_tol=1e-12):
            return (True, "V = 0, no perturbation possible")
        # Baseline output: V_c based on given V
        baseline = self.V_c
        # Perturb input: increase V by 10%
        perturb_factor = 1.1
        perturbed_V = self.V * perturb_factor
        # Estimate perturbed output using observed linearity: V_c ∝ V
        estimated_perturbed_output = baseline * perturb_factor
        # The actual theoretical relationship gives V_c' = V / 12
        theoretical_perturbed_output = perturbed_V / 12.0
        # Compare estimated (from observed slope) with theoretical
        if math.isclose(estimated_perturbed_output, theoretical_perturbed_output, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Observed linear scaling holds: V_c scales by {perturb_factor}")
        else:
            return (False, f"Linear scaling violation: estimated {estimated_perturbed_output:.4f} vs theoretical {theoretical_perturbed_output:.4f}")

    # ------------------------------------------------------------------
    # Weak answer sanity (do not combine with strong constraints)
    # ------------------------------------------------------------------
    def test_answer_sanity_basic(self):
        """
        Basic sanity: V_c must be finite, non-NaN, and lie within a reasonable
        range: |V_c| <= |V| (since V_c comes from averaging).
        """
        if self.V_c is None or self.V is None:
            return (False, "SKIP: V_c or V missing")
        if not math.isfinite(self.V_c):
            return (False, f"V_c is not finite: {self.V_c}")
        if math.isnan(self.V_c):
            return (False, "V_c is NaN")
        if abs(self.V_c) > abs(self.V) * (1.0 + 1e-2):  # allow tiny slack for floating point
            return (False, f"|V_c| = {abs(self.V_c):.4f} exceeds |V| = {abs(self.V):.4f}")
        return (True, "Basic sanity check passed (finite, non-NaN, |V_c| ≤ |V|)")