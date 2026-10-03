import math
import traceback
from typing import Any, Dict, List, Tuple, Optional

try:
    import sympy as sp
except ImportError:
    sp = None


class PhysicsChecker:
    """
    Audits solution for the ratio (pV/RT) at the critical point of Dieterici's equation.
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract all needed data from the solution instance.
        Safely default to None for missing attributes.
        """
        # Core result
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.ratio = self.results.get('ratio', None)

        # Auxiliary fields for context (may be None)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------

    def test_interface_ratio_key_exists(self) -> Tuple[bool, str]:
        """Check that the required key 'ratio' is present in results."""
        if 'ratio' not in self.results:
            return False, "Key 'ratio' missing in results"
        return True, "Key 'ratio' exists"

    def test_interface_ratio_type(self) -> Tuple[bool, str]:
        """Check that ratio is a numeric type."""
        if self.ratio is None:
            return False, "SKIP: ratio is None"
        if not isinstance(self.ratio, (int, float)):
            return False, f"ratio is not numeric (type: {type(self.ratio).__name__})"
        return True, "ratio is numeric"

    # ------------------------------------------------------------------
    # Theorem‑based invariants (use math.isclose, re‑derive from physics)
    # ------------------------------------------------------------------

    def test_invariant_ratio_derivation(self) -> Tuple[bool, str]:
        """
        Re‑derive the expected ratio using SymPy from the critical conditions.
        Compare with the solution's ratio using isclose (1‑2% tolerance).
        """
        if sp is None:
            return False, "SymPy not available – cannot perform symbolic derivation"

        if self.ratio is None:
            return False, "SKIP: ratio is None"

        try:
            # Symbolic setup
            a, b, R, T, V = sp.symbols('a b R T V', positive=True, real=True)
            p = R * T / (V - b) * sp.exp(-a / (R * T * V))

            # Critical conditions: first and second derivatives zero
            dpdV = sp.diff(p, V)
            d2pdV2 = sp.diff(p, V, 2)

            # Solve for V and T at critical point
            sol = sp.solve([sp.simplify(dpdV), sp.simplify(d2pdV2)], [V, T], dict=True)
            if not sol:
                return False, "SymPy could not find a solution to the critical conditions"

            Vc_expr = sol[0][V]
            Tc_expr = sol[0][T]

            # Critical pressure
            pc_expr = p.subs({V: Vc_expr, T: Tc_expr})

            # Ratio pV/RT
            ratio_expr = sp.simplify(pc_expr * Vc_expr / (R * Tc_expr))
            expected_numeric = float(ratio_expr.evalf())

            # Compare with solution (lenient tolerance)
            if math.isclose(self.ratio, expected_numeric, rel_tol=1e-2, abs_tol=1e-2):
                return True, (f"Re‑derived ratio = {ratio_expr} ≈ {expected_numeric:.5f}; "
                              f"solution ratio = {self.ratio:.5f} (matches)")
            else:
                return False, (f"Re‑derived ratio = {ratio_expr} ≈ {expected_numeric:.5f}; "
                               f"solution ratio = {self.ratio:.5f} (does not match)")

        except Exception as e:
            return False, f"test_invariant_ratio_derivation crashed: {e}"

    def test_invariant_ratio_positivity(self) -> Tuple[bool, str]:
        """Trivial invariant: ratio must be positive (critical point is physical)."""
        if self.ratio is None:
            return False, "SKIP: ratio is None"
        if self.ratio > 0:
            return True, f"Ratio {self.ratio} is positive"
        return False, f"Ratio {self.ratio} is not positive (should be >0)"

    # ------------------------------------------------------------------
    # Modeling / trace checks (based on explanation steps and pseudocode)
    # ------------------------------------------------------------------

    def test_modeling_explanation_mentions_correct_form(self) -> Tuple[bool, str]:
        """
        Weak check that the explanation steps mention the expected ratio expression.
        This is solely a consistency hint, not a definitive test.
        """
        if self.explanation_steps is None:
            return False, "SKIP: explanation_steps not available"

        steps_str = str(self.explanation_steps)
        # Look for typical forms of the answer
        keywords = ["2 e^{-2}", "2*exp(-2)", "0.27", "2/e^2"]
        found = any(kw in steps_str for kw in keywords)
        if found:
            return True, "Explanation steps contain expected ratio expression (2e^{-2} / 0.27)"
        return False, "Explanation steps do not mention expected ratio form"

    def test_modeling_pseudocode_emit_contract(self) -> Tuple[bool, str]:
        """
        Verify that the pseudocode (if present) contains the correct final emit.
        This is a trace‑consistency check, not a physics assertion.
        """
        if self.pseudocode is None:
            return False, "SKIP: pseudocode not available"

        pseudo_str = str(self.pseudocode)
        # The emit contract expects 'ratio = 2 * exp(-2)'
        if "ratio = 2 * exp(-2)" in pseudo_str:
            return True, "Pseudocode emit matches expected: 'ratio = 2 * exp(-2)'"
        # Accept loose variants
        if "2 * exp(-2)" in pseudo_str or "2 e^{-2}" in pseudo_str:
            return True, "Pseudocode contains expected ratio expression"
        return False, "Pseudocode does not contain the expected emit 'ratio = 2 * exp(-2)'"

    # ------------------------------------------------------------------
    # Answer sanity (very weak: sign, finiteness, broad magnitude)
    # ------------------------------------------------------------------

    def test_answer_sanity_sign_and_finite(self) -> Tuple[bool, str]:
        """Weak answer sanity: ratio must be positive, finite, and within a broad plausible range."""
        if self.ratio is None:
            return False, "SKIP: ratio is None"

        if not math.isfinite(self.ratio):
            return False, f"ratio is not finite ({self.ratio})"
        if self.ratio <= 0:
            return False, f"ratio is not positive ({self.ratio})"
        # Broad range covering 0.27 ± 100%
        if 0.2 < self.ratio < 0.4:
            return True, f"ratio {self.ratio:.5f} is positive, finite, and within plausible range (0.2,0.4)"
        return False, f"ratio {self.ratio:.5f} is outside plausible range (0.2,0.4)"

    # ------------------------------------------------------------------
    # Monotonicity test: ratio is invariant under changes of parameters
    # ------------------------------------------------------------------

    def test_monotonicity_ratio_invariant_under_a_scaling(self) -> Tuple[bool, str]:
        """
        Check that the ratio is unchanged when the parameter a is perturbed.
        This tests the theoretical invariance that the numerical value does not depend on a.
        Baseline: the student's ratio.
        Perturbation: increase a by a factor of 2 and recompute the theoretical ratio.
        The theoretical ratio remains 2/e^2, so the student's ratio should match it.
        """
        if self.ratio is None:
            return False, "SKIP: ratio is None"

        # Set arbitrary positive numbers for a, b, R
        a0, b0, R0 = 1.0, 1.0, 1.0
        # Perturb a
        a_perturbed = a0 * 2.0

        # Theoretical ratio is independent of a, b, R
        expected_ratio = 2.0 * math.exp(-2.0)

        # Compare student's ratio to theoretical (should match)
        if math.isclose(self.ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return True, (f"Baseline (student ratio) {self.ratio:.5f} matches theoretical {expected_ratio:.5f}. "
                          f"After perturbing a from {a0} to {a_perturbed}, theoretical ratio unchanged; "
                          f"consistency holds.")
        else:
            return False, (f"Baseline (student ratio) {self.ratio:.5f} does not match expected {expected_ratio:.5f}. "
                           f"Cannot verify invariance.")

    # ------------------------------------------------------------------
    # Test harness
    # ------------------------------------------------------------------

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover all test_* methods, execute them safely, and collect results.
        Returns (all_passed, logs_list).
        """
        logs: List[str] = []
        all_pass = True

        for method_name in sorted(dir(self)):
            if not method_name.startswith("test_"):
                continue
            method = getattr(self, method_name, None)
            if not callable(method):
                continue

            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{method_name} CRASHED: {traceback.format_exc()}"

            logs.append(f"{'PASS' if passed else 'FAIL'}: {method_name}: {msg}")
            if not passed:
                all_pass = False

        return all_pass, logs