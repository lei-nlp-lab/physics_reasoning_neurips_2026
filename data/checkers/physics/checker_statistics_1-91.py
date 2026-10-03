import math
import sympy as sp
from typing import Any, Dict, List, Tuple, Optional, Callable
import re
import traceback

class PhysicsChecker:
    """
    Audits a solution_instance for the derivation:
        (∂Q_emitted/∂p)_T = T (∂V/∂T)_p
    Uses Maxwell relations from Gibbs free energy.
    """

    def __init__(self, solution_instance: Any):
        # Extract results dictionary safely
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # Explanation steps (list of strings)
        self.explanation_steps: Optional[List[str]] = getattr(solution_instance, 'explanation_steps', None)
        if self.explanation_steps is None:
            self.explanation_steps = []

        # Parse the main result as a sympy expression
        raw = self.results.get('dQ_emitted_dp_T', None)
        self.dQ_expr: Optional[sp.Expr] = None
        if raw is not None:
            try:
                self.dQ_expr = sp.sympify(raw)
            except (sp.SympifyError, TypeError):
                self.dQ_expr = None

        # Symbols used in symbolic reasoning (not extracted from solution)
        self.T = sp.Symbol('T', real=True, positive=True)
        self.p = sp.Symbol('p', real=True, positive=True)
        # Volume as function of temperature and pressure
        self.V = sp.Function('V')(self.T, self.p)
        self.S = sp.Function('S')(self.T, self.p)

    # ================================================================
    # Interface tests
    # ================================================================
    def test_interface_required_key_present(self) -> Tuple[bool, str]:
        if 'dQ_emitted_dp_T' not in self.results:
            return (False, "Missing required key 'dQ_emitted_dp_T' in results")
        return (True, "Key 'dQ_emitted_dp_T' exists")

    def test_interface_result_is_expression(self) -> Tuple[bool, str]:
        if self.dQ_expr is None:
            return (False, "Result 'dQ_emitted_dp_T' is not a valid sympy expression")
        return (True, "Result is a valid symbolic expression")

    # ================================================================
    # Invariant tests (theorem‑based sanity, use math.isclose if numeric)
    # ================================================================
    def test_invariant_maxwell_relation_referenced(self) -> Tuple[bool, str]:
        if not self.explanation_steps:
            return (False, "No explanation_steps available")
        combined = " ".join(self.explanation_steps).lower()
        patterns = [
            r"\\(\\partial s / \\partial p\\)",
            r"\(∂s/∂p\)_t",
            r"ds/dp",
            r"maxwell",
        ]
        found = any(re.search(pat, combined) for pat in patterns)
        if found:
            return (True, "Maxwell relation or its derivative referenced")
        return (False, "No reference to Maxwell relation")

    def test_invariant_heat_definition_referenced(self) -> Tuple[bool, str]:
        if not self.explanation_steps:
            return (False, "No explanation_steps available")
        combined = " ".join(self.explanation_steps).lower()
        patterns = [
            r"dq_emitted\s*=\s*-t\s*ds",
            r"heat emitted.*= -t ds",
            r"dQ_emitted = -T \\, dS",
        ]
        found = any(re.search(pat, combined) for pat in patterns)
        if found:
            return (True, "Heat emitted definition referenced")
        return (False, "Definition dQ_emitted = -T dS not found")

    def test_invariant_result_derivable_from_maxwell(self) -> Tuple[bool, str]:
        """
        Symbolic check: if student’s result equals T * (∂V/∂T)_p.
        We test equality via substitution of a test function.
        """
        if self.dQ_expr is None:
            return (False, "Cannot perform symbolic check: expression is None")
        # Build expected expression
        dV_dT = sp.Derivative(self.V, self.T)
        expected = self.T * dV_dT
        # Try direct symbolic simplification
        diff = sp.simplify(self.dQ_expr - expected)
        if diff == 0:
            return (True, "Expression equals T * dV/dT symbolically")
        # Fallback: substitute a test function V = T^2 / p
        test_V = self.T**2 / self.p
        dV_dT_test = sp.diff(test_V, self.T)  # 2*T/p
        # Replace Derivative(V,T) with the actual derivative in student expression
        student_subst = self.dQ_expr.subs(sp.Derivative(self.V, self.T), dV_dT_test)
        # Also replace V itself? Not needed because derivative substitution suffices.
        expected_test = self.T * dV_dT_test  # T * (2T/p) = 2T^2/p
        if sp.simplify(student_subst - expected_test) == 0:
            return (True, "Expression consistent with T*dV/dT under test function V=T^2/p")
        return (False, "Expression does not match expected form")

    # ================================================================
    # Modeling / trace checks
    # ================================================================
    def test_modeling_step_flow(self) -> Tuple[bool, str]:
        if not self.explanation_steps:
            return (False, "No explanation_steps available")
        combined = " ".join(self.explanation_steps).lower()
        ideas = [
            r"dq_emitted\s*=\s*-t\s*ds",
            r"\(∂s/∂p\)_t",
            r"\(∂v/∂t\)_p",
            r"maxwell",
            r"gibbs",
            r"reversible",
            r"constant temperature",
        ]
        count = sum(1 for pat in ideas if re.search(pat, combined))
        if count >= 3:
            return (True, f"Explanation covers {count}/7 key thermodynamic ideas (≥3 required)")
        return (False, f"Only {count} key ideas found; need at least 3")

    def test_modeling_pseudocode_consistency(self) -> Tuple[bool, str]:
        pseudocode = self.results.get('pseudocode', None)
        if pseudocode is None:
            if self.explanation_steps:
                combined = " ".join(self.explanation_steps).lower()
                if "results['dq_emitted_dp_t']" in combined:
                    return (True, "Emit pattern found in explanation steps")
                return (False, "No explicit emit in explanation")
            return (False, "No pseudocode or explanation steps available")
        if isinstance(pseudocode, list):
            pseudocode = " ".join(pseudocode)
        elif not isinstance(pseudocode, str):
            return (False, "Pseudocode format not recognized")
        if "results['dq_emitted_dp_t']" in pseudocode.lower():
            return (True, "Expected emit key found in pseudocode")
        return (False, "Emit key 'dQ_emitted_dp_T' missing from pseudocode")

    # ================================================================
    # Monotonicity test (explicit perturbation)
    # ================================================================
    def test_monotonicity_with_pressure(self) -> Tuple[bool, str]:
        """
        Verify monotonic behavior under perturbation of pressure p.
        Use a concrete test function V = T^2 / p, then the correct result is
        R = T * (∂V/∂T)_p = 2T^2/p.
        For fixed T = 1 (chosen for simplicity), we compare baseline p=1 and perturbed p=1.1.
        The result should decrease when pressure increases.
        """
        if self.dQ_expr is None:
            return (False, "SKIP: dQ_expr is None, cannot evaluate")
        # Choose a concrete test function for V
        test_V = self.T**2 / self.p
        dV_dT_test = sp.diff(test_V, self.T)  # 2T/p
        # Substitute the actual derivative in student's expression
        # First replace symbolic Derivative(V,T) with the concrete derivative
        student_func = self.dQ_expr.subs(sp.Derivative(self.V, self.T), dV_dT_test)
        # Also ensure T and p are symbols we can substitute
        # Evaluate baseline: T=1, p=1
        baseline_val = float(student_func.subs({self.T: 1, self.p: 1}).evalf())
        # Evaluate perturbed: T=1, p=1.1
        perturbed_val = float(student_func.subs({self.T: 1, self.p: 1.1}).evalf())
        # Correct behavior: R ∝ 1/p, so increasing p decreases R
        if perturbed_val < baseline_val:
            return (True, f"Monotonicity correct: perturbed ({perturbed_val}) < baseline ({baseline_val})")
        elif math.isclose(perturbed_val, baseline_val, rel_tol=1e-6):
            return (False, "No change under perturbation – not monotonic")
        else:
            return (False, f"Wrong monotonicity: increase in p increased result (baseline={baseline_val}, perturbed={perturbed_val})")

    # ================================================================
    # Weak answer sanity (no heavy computation, no math.isclose)
    # ================================================================
    def test_answer_sanity_basic(self) -> Tuple[bool, str]:
        if 'dQ_emitted_dp_T' not in self.results:
            return (False, "No result present")
        raw = self.results['dQ_emitted_dp_T']
        if raw is None:
            return (False, "Result is None")
        raw_str = str(raw).lower()
        # Contains T
        if 't' not in raw_str and 'T' not in raw_str:
            return (False, "Result does not contain temperature variable T")
        # Contains some derivative indicator
        has_deriv = any(term in raw_str for term in ['d/d', '∂v/∂t', 'dv/dt', 'derivative'])
        if not has_deriv:
            if 'Derivative' not in repr(raw):
                return (False, "No derivative visible (string or symbolic)")
        if len(raw_str) > 500:
            return (False, "Result string suspiciously long")
        return (True, "Basic sanity: result contains T and derivative indicator")

    # ================================================================
    # Main runner
    # ================================================================
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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
                passed, msg = False, f"CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)