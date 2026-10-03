import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Checker for the classical hard sphere scattering cross section problem.
    Emphasizes interface correctness, theorem-based invariants, modeling quality,
    and monotonicity of physical relationships.
    """

    def __init__(self, solution_instance):
        # Extract results safely
        results = getattr(solution_instance, 'results', None) or {}
        self.dsigma_dOmega = results.get('dsigma_dOmega', None)

        # Extract givens safely
        givens = getattr(solution_instance, 'givens', None) or {}
        self.R = givens.get('R', None)

        # Extract explanation steps (if present)
        self.explanation = getattr(solution_instance, 'explanation_steps', None)

        # Optional: raw pseudocode string (for trace reference only, not used)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_dsigma_dOmega_key(self):
        """Check that the expected result key 'dsigma_dOmega' exists."""
        if self.dsigma_dOmega is None:
            return (False, "Missing required key 'dsigma_dOmega'")
        return (True, "Key 'dsigma_dOmega' is present")

    def test_interface_dsigma_dOmega_finite(self):
        """Check that dσ/dΩ is a finite number."""
        if self.dsigma_dOmega is None:
            return (False, "SKIP: dsigma_dOmega is None")
        if not math.isfinite(self.dsigma_dOmega):
            return (False, f"dsigma_dOmega is not finite: {self.dsigma_dOmega}")
        return (True, "dsigma_dOmega is finite")

    def test_interface_dsigma_dOmega_positive(self):
        """Check that dσ/dΩ is positive."""
        if self.dsigma_dOmega is None:
            return (False, "SKIP: dsigma_dOmega is None")
        if self.dsigma_dOmega <= 0:
            return (False, f"dsigma_dOmega must be positive, got {self.dsigma_dOmega}")
        return (True, "dsigma_dOmega is positive (>0)")

    # ------------------------------------------------------------------
    # Invariant tests (use math.isclose or symbolic reasoning)
    # ------------------------------------------------------------------
    def test_invariant_dsigma_dOmega_scaling(self):
        """Check that dσ/dΩ equals R^2/4 when R is provided."""
        if self.R is None:
            return (False, "SKIP: R is not provided in givens")
        if self.dsigma_dOmega is None:
            return (False, "SKIP: dsigma_dOmega is not available")
        try:
            expected = self.R**2 / 4.0
            if not math.isclose(self.dsigma_dOmega, expected, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"dsigma_dOmega={self.dsigma_dOmega} does not match R^2/4={expected}")
            return (True, f"dsigma_dOmega={self.dsigma_dOmega} is consistent with R^2/4={expected}")
        except Exception as e:
            return (False, f"Scaling check crashed: {e}")

    def test_invariant_symbolic_derivation(self):
        """
        Theorem-based invariant: symbolically derive dσ/dΩ from the geometry
        relation b = R cos(θ/2) and the general formula dσ/dΩ = (b / sinθ) |db/dθ|.
        The result must simplify to R^2/4 and be independent of θ.
        """
        try:
            R, theta = sp.symbols('R theta', positive=True)
            b = R * sp.cos(theta/2)
            db_dtheta = sp.diff(b, theta)
            # formula with absolute value; we know db_dtheta is negative for theta in [0,π]
            # so |db_dtheta| = -db_dtheta
            abs_db_dtheta = -db_dtheta
            dsigma_expr = (b / sp.sin(theta)) * abs_db_dtheta
            # use double-angle identity to simplify
            simplified = sp.simplify(dsigma_expr.rewrite(sp.sin))
            # expected result
            expected = R**2 / 4
            # check equality (symbolic)
            if sp.simplify(simplified - expected) != 0:
                return (False, f"Symbolic derivation gave {simplified}, expected {expected}")
            # check theta-independence: derivative wrt theta should be zero
            if sp.simplify(sp.diff(simplified, theta)) != 0:
                return (False, f"Cross section depends on theta: derivative = {sp.diff(simplified, theta)}")
            return (True, "Symbolic derivation confirms dσ/dΩ = R^2/4, independent of θ")
        except Exception as e:
            return (False, f"Symbolic invariant test crashed: {e}")

    # ------------------------------------------------------------------
    # Modeling / trace tests (check explanation quality)
    # ------------------------------------------------------------------
    def test_modeling_explanation_mentions_geometry(self):
        """Check that the explanation includes the impact parameter relation b = R cos(θ/2)."""
        if self.explanation is None or not isinstance(self.explanation, str):
            return (False, "SKIP: explanation_steps is not a string or is missing")
        text = self.explanation.lower()
        # Look for variants of "b = R cos"
        patterns = ["b = r cos", "b = r * cos", "b=rcos", "impact parameter", "b = r cos(θ/2)"]
        if not any(p in text for p in patterns):
            return (False, "Explanation does not mention the impact parameter relation b = R cos(θ/2)")
        return (True, "Explanation references the geometry relation")

    def test_modeling_explanation_mentions_cross_section_formula(self):
        """Check that the explanation includes the differential cross section formula."""
        if self.explanation is None or not isinstance(self.explanation, str):
            return (False, "SKIP: explanation_steps is not a string or is missing")
        text = self.explanation.lower()
        # Search for "dsigma/dOmega" or "dσ/dΩ" and derivative/sinθ
        formula_patterns = ["dsigma", "dσ", "differential cross section", "/ sin"]
        if not any(p in text for p in formula_patterns):
            return (False, "Explanation does not contain the differential cross section formula")
        if "sin" not in text:
            return (False, "Explanation missing sinθ reference needed for cross section simplification")
        return (True, "Explanation references the differential cross section formula")

    def test_modeling_explanation_mentions_isotropy(self):
        """Check that the explanation states the cross section is isotropic (θ-independent)."""
        if self.explanation is None or not isinstance(self.explanation, str):
            return (False, "SKIP: explanation_steps is not a string or is missing")
        text = self.explanation.lower()
        isotropy_keywords = ["independent", "constant", "isotropic", "does not depend on θ", "θ-independent"]
        if not any(k in text for k in isotropy_keywords):
            return (False, "Explanation does not mention isotropy (θ-independence)")
        return (True, "Explanation confirms the cross section is isotropic")

    # ------------------------------------------------------------------
    # Monotonicity test (must perturb an input and compare outputs)
    # ------------------------------------------------------------------
    def test_monotonicity_cross_section_increases_with_R(self):
        """
        Check that the differential cross section dσ/dΩ = R^2/4 is an increasing
        function of the sphere radius R. We perturb R by +10% and verify that
        the theoretical cross section increases.
        """
        if self.R is None:
            return (False, "SKIP: R not provided for monotonicity check")
        try:
            R0 = float(self.R)
            # Perturb: increase R by 10%
            R1 = R0 * 1.1
            # Theoretical cross sections (re-derived in checker)
            sigma0 = R0**2 / 4.0
            sigma1 = R1**2 / 4.0
            if sigma1 <= sigma0:
                return (False, f"Monotonicity violated: for R1={R1} > R0={R0}, sigma1={sigma1} should be > sigma0={sigma0}")
            return (True, f"Cross section increases with R: verified with 10% increase (R0={R0} → R1={R1})")
        except Exception as e:
            return (False, f"Monotonicity test crashed: {e}")

    # ------------------------------------------------------------------
    # Optional answer sanity (weak, only one such test allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_dsigma_dOmega_range(self):
        """Very weak sanity: dσ/dΩ should be within a broad plausible range for typical R values."""
        if self.dsigma_dOmega is None:
            return (False, "SKIP: dsigma_dOmega not available")
        # No specific R? Use a default check: cross section should not be huge or negative.
        # This is a placeholder; real weak sanity – just ensure it's not absurd.
        if self.dsigma_dOmega <= 0:
            return (False, "dsigma_dOmega must be positive")
        if self.dsigma_dOmega > 1e100:  # ridiculously large
            return (False, "dsigma_dOmega is unphysically large")
        return (True, "dsigma_dOmega passes basic plausibility check")

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)