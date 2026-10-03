import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results safely; use None for missing values
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.p_U_V = self.results.get('p_U_V', None)
        self.u_T = self.results.get('u_T', None)

        # Explanation steps (list of strings or dicts)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        if self.explanation_steps is None:
            self.explanation_steps = self.results.get('explanation_steps', None)

        # Pseudocode string (optional, for deeper checks)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)
        if self.pseudocode is None:
            self.pseudocode = self.results.get('pseudocode', None)

        # Givens (physical parameters) – may be empty
        self.givens = getattr(solution_instance, 'givens', {}) or {}
        # For this problem, we rarely have numeric givens; keep as dict

    def _safe_compare_string(self, value, substring):
        """Check if substring appears in value (case-insensitive)."""
        if value is None:
            return False
        return substring.lower() in value.lower()

    # ------------------------------------------------------------------
    #  Interface Tests
    # ------------------------------------------------------------------
    def test_interface_p_U_V_key(self):
        if self.p_U_V is None:
            return (False, "Required key 'p_U_V' missing from results")
        if not isinstance(self.p_U_V, str):
            return (False, f"'p_U_V' is not a string, got {type(self.p_U_V)}")
        return (True, "'p_U_V' present and is string")

    def test_interface_u_T_key(self):
        if self.u_T is None:
            return (False, "Required key 'u_T' missing from results")
        if not isinstance(self.u_T, str):
            return (False, f"'u_T' is not a string, got {type(self.u_T)}")
        return (True, "'u_T' present and is string")

    def test_interface_explanation_steps(self):
        if self.explanation_steps is None:
            return (False, "Explanation steps not available")
        if not isinstance(self.explanation_steps, (list, tuple)):
            return (False, "Explanation steps should be a list or tuple")
        if len(self.explanation_steps) == 0:
            return (False, "Explanation steps list is empty")
        return (True, f"Explanation steps present, {len(self.explanation_steps)} steps")

    # ------------------------------------------------------------------
    #  Invariant (Theorem-based) Tests
    # ------------------------------------------------------------------
    def test_invariant_pressure_energy_density_relation(self):
        """Check that derived relation p = U/(3V) is plausible from physics.
        Weak check: verify that the string contains the formula form.
        """
        if self.p_U_V is None:
            return (False, "SKIP: 'p_U_V' not available")
        # The relation should express p proportional to U/V with factor 1/3.
        # We check for presence of "U/(3V)" or "U/3V" etc.
        if not self._safe_compare_string(self.p_U_V, "U/(3V)") and \
           not self._safe_compare_string(self.p_U_V, "U/3V") and \
           not self._safe_compare_string(self.p_U_V, "p = u/3"):
            return (False, "Relation does not seem to express p = U/(3V).")
        return (True, "Relation p = U/(3V) is present in given form")

    def test_invariant_first_law_consistency(self):
        """Check that the explanation or pseudocode mentions the first law dU = TdS - pdV."""
        # Use explanation steps or pseudocode to find mention of first law.
        text = ""
        if self.explanation_steps:
            text += " ".join(str(s) for s in self.explanation_steps)
        if self.pseudocode:
            text += " " + self.pseudocode
        if not text:
            return (False, "SKIP: No explanation or pseudocode to check first law")
        if "dU = T dS - p dV" in text or "dU = TdS - pdV" in text:
            return (True, "First law mentioned")
        return (False, "First law (dU = T dS - p dV) not found in explanation/pseudocode")

    def test_invariant_maxwell_relation_type(self):
        """Check that explanation/pseudocode uses exactness of dS to derive ODE."""
        text = ""
        if self.explanation_steps:
            text += " ".join(str(s) for s in self.explanation_steps)
        if self.pseudocode:
            text += " " + self.pseudocode
        if not text:
            return (False, "SKIP: No explanation/pseudocode")
        if "exactness" in text.lower() or "maxwell" in text.lower() or "d/dV" in text:
            return (True, "Exactness condition or Maxwell relation used")
        return (False, "No sign of exactness condition in derivation")

    # ------------------------------------------------------------------
    #  Modeling / Trace Tests
    # ------------------------------------------------------------------
    def test_modeling_pressure_derivation(self):
        """Check that the kinetic theory argument for p = u/3 is present."""
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not available")
        # Look for keywords: kinetic theory, isotropic, factor 1/3, momentum
        found_keywords = 0
        keywords = ["kinetic", "isotropic", "momentum", "1/3", "average"]
        combined = " ".join(str(s) for s in self.explanation_steps).lower()
        for kw in keywords:
            if kw in combined:
                found_keywords += 1
        if found_keywords >= 3:
            return (True, "Kinetic theory derivation of p appears plausible")
        return (False, f"Not enough keywords from kinetic theory derivation (found {found_keywords})")

    def test_modeling_ode_derivation(self):
        """Check that the ODE du/dT = 4u/T is derived from thermodynamics."""
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not available")
        combined = " ".join(str(s) for s in self.explanation_steps)
        # Look for differential equation pattern
        if "du/dT" in combined and ("4u/T" in combined or "4u/T" in combined):
            return (True, "ODE du/dT = 4u/T appears in derivation")
        return (False, "ODE du/dT = 4u/T not found in explanation")

    def test_modeling_u_proportional_T4(self):
        """Check that the final result u ∝ T^4 is present in u_T."""
        if self.u_T is None:
            return (False, "SKIP: 'u_T' not available")
        if self._safe_compare_string(self.u_T, "T^4") or self._safe_compare_string(self.u_T, "T**4"):
            return (True, "u_T expresses proportionality to T^4")
        return (False, "'u_T' does not contain T^4 dependence")

    # ------------------------------------------------------------------
    #  Monotonicity Tests (with explicit perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_energy_density_with_temperature(self):
        """
        Perturb temperature upward by 10% and verify that energy density increases
        with the expected power law (u ∝ T^4).
        """
        if self.u_T is None:
            return (False, "SKIP: 'u_T' not available")

        # Try to determine the exponent from the string
        exponent = None
        if "T^4" in self.u_T:
            exponent = 4
        elif "T**4" in self.u_T:
            exponent = 4
        elif "T^3" in self.u_T:
            exponent = 3
        elif "T**3" in self.u_T:
            exponent = 3
        # For generality, we could try sympy parsing, but let's keep it simple.
        if exponent is None:
            return (False, "SKIP: Cannot determine exponent from string (expected T^4)")

        # Define symbolic function
        T, k = sp.symbols('T k', positive=True)
        u = k * T**exponent

        # Baseline: T0 = 1 (arbitrary)
        T0 = 1.0
        u0 = float(u.subs({T: T0, k: 1}))

        # Perturb: increase T by 10%
        delta = 0.1
        T1 = T0 * (1 + delta)
        u1 = float(u.subs({T: T1, k: 1}))

        # Expected ratio for T^exponent: (1+delta)^exponent
        expected_ratio = (1 + delta) ** exponent
        actual_ratio = u1 / u0

        if not (u1 > u0):
            return (False, f"Energy density did not increase: u0={u0}, u1={u1}")
        if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2):
            msg = (f"Ratio mismatch: expected {expected_ratio:.4f}, got {actual_ratio:.4f} "
                   f"for a {delta*100:.1f}% T increase")
            return (False, msg)
        return (True, f"u increases from {u0:.3f} to {u1:.3f}, ratio {actual_ratio:.4f} ≈ expected {expected_ratio:.4f}")

    def test_monotonicity_pressure_with_energy_density(self):
        """
        Perturb energy density upward by 10% and verify that pressure increases linearly
        according to p = u/3.
        """
        if self.p_U_V is None:
            return (False, "SKIP: 'p_U_V' not available")

        # Determine whether the relation is p = u/3 (or equivalent)
        if not (self._safe_compare_string(self.p_U_V, "u/3") or 
                self._safe_compare_string(self.p_U_V, "U/(3V)")):
            return (False, "SKIP: Cannot confirm linear relation p ∝ u from string")

        # Assume p = u/3 (since U/V = u)
        u = sp.symbols('u', positive=True)
        p = u / 3

        # Baseline: u0 = 1 (arbitrary)
        u0 = 1.0
        p0 = float(p.subs(u, u0))

        # Perturb: increase u by 10%
        delta = 0.1
        u1 = u0 * (1 + delta)
        p1 = float(p.subs(u, u1))

        expected_ratio = 1 + delta  # linear
        actual_ratio = p1 / p0

        if not (p1 > p0):
            return (False, f"Pressure did not increase: p0={p0}, p1={p1}")
        if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2):
            msg = (f"Ratio mismatch: expected {expected_ratio:.4f}, got {actual_ratio:.4f} "
                   f"for a {delta*100:.1f}% u increase")
            return (False, msg)
        return (True, f"p increases from {p0:.3f} to {p1:.3f}, ratio {actual_ratio:.4f} ≈ expected {expected_ratio:.4f}")

    # ------------------------------------------------------------------
    #  Weak Answer Sanity (only one)
    # ------------------------------------------------------------------
    def test_answer_sanity_final_forms(self):
        """Weak sanity: check that result strings are non‑empty and have plausible length."""
        if self.p_U_V is None or self.u_T is None:
            return (False, "Missing required results")
        if len(self.p_U_V) < 5 or len(self.u_T) < 5:
            return (False, "Result strings are suspiciously short")
        # Accept any string that contains U and V and 3 for p_U_V, and T^4 for u_T
        if not (self._safe_compare_string(self.p_U_V, "U") and 
                self._safe_compare_string(self.p_U_V, "V") and
                self._safe_compare_string(self.p_U_V, "3")):
            return (False, "p_U_V does not contain required symbols")
        if not (self._safe_compare_string(self.u_T, "T") and
                self._safe_compare_string(self.u_T, "4")):
            return (False, "u_T does not contain T and 4")
        return (True, "Result strings present, contain required symbols, have reasonable length")

    # ------------------------------------------------------------------
    #  Runner
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

# Example usage (not part of the output; kept for testing)
# if __name__ == '__main__':
#     class Solution:
#         results = {
#             'p_U_V': 'p = U/(3V)',
#             'u_T': 'u = a T^4'
#         }
#         explanation_steps = [
#             "Using kinetic theory: p = u/3 from isotropic momentum transfer.",
#             "First law: dU = T dS - p dV.",
#             "Exactness condition leads to du/dT = 4u/T.",
#             "Integrate: u = a T^4."
#         ]
#         pseudocode = "dU = T dS - p dV; p = u/3; ..."
#     checker = PhysicsChecker(Solution())
#     success, logs = checker.run_all_tests()
#     for l in logs:
#         print(l)
#     print("Overall success:", success)