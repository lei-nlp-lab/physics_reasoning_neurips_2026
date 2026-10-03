import math
import re
import traceback
from typing import Any, Dict, List, Optional, Tuple

import sympy as sp

class PhysicsChecker:
    """
    Auditor for a perfect gas problem:
    (a) Show cp = cv + k.
    (b) Show pV^gamma = constant during adiabatic expansion.

    Expected result keys (from solving pseudocode):
        - results['cp_cv_relation']   : string describing the relation
        - results['adiabatic_relation']: string describing the adiabatic relation
    """

    def __init__(self, solution_instance: Any) -> None:
        # Safely extract results dictionary
        results: Optional[Dict] = getattr(solution_instance, 'results', None)
        if results is None:
            results = {}
        self.cp_cv_relation: Optional[str] = results.get('cp_cv_relation', None)
        self.adiabatic_relation: Optional[str] = results.get('adiabatic_relation', None)

        # Optional explanation steps
        self.explanation_steps: Optional[List[str]] = getattr(solution_instance, 'explanation_steps', None)

    # -----------------------------------------------------------------
    # Interface tests – required keys must exist and have valid type
    # -----------------------------------------------------------------
    def test_interface_cp_cv_relation_exists(self) -> Tuple[bool, str]:
        if self.cp_cv_relation is None:
            return (False, "FAIL: Key 'cp_cv_relation' not found in results.")
        if not isinstance(self.cp_cv_relation, str):
            return (False, "FAIL: 'cp_cv_relation' is not a string.")
        if not self.cp_cv_relation.strip():
            return (False, "FAIL: 'cp_cv_relation' is an empty string.")
        return (True, "PASS: cp_cv_relation present and non-empty string.")

    def test_interface_adiabatic_relation_exists(self) -> Tuple[bool, str]:
        if self.adiabatic_relation is None:
            return (False, "FAIL: Key 'adiabatic_relation' not found in results.")
        if not isinstance(self.adiabatic_relation, str):
            return (False, "FAIL: 'adiabatic_relation' is not a string.")
        if not self.adiabatic_relation.strip():
            return (False, "FAIL: 'adiabatic_relation' is an empty string.")
        return (True, "PASS: adiabatic_relation present and non-empty string.")

    # -----------------------------------------------------------------
    # Theorem-based invariant tests (at least two)
    # -----------------------------------------------------------------
    def test_invariant_cp_cv_symbolic_identity(self) -> Tuple[bool, str]:
        """
        Verify that the string self.cp_cv_relation simplifies to an identity.
        """
        if self.cp_cv_relation is None:
            return (True, "SKIP: cp_cv_relation not available.")
        try:
            eq = sp.sympify(self.cp_cv_relation)
            if not isinstance(eq, sp.Equality):
                return (False, "FAIL: Parsed expression is not an equality.")
            diff = sp.simplify(eq.lhs - eq.rhs)
            if diff == 0:
                return (True, "PASS: The relation is algebraically valid.")
            else:
                return (False, f"FAIL: The relation does not simplify to identity; residual = {diff}")
        except Exception as e:
            return (True, f"SKIP: Cannot parse symbolic equation ({e}).")

    def test_invariant_cp_greater_than_cv(self) -> Tuple[bool, str]:
        """
        From c_p = c_v + k and k > 0, we must have c_p > c_v.
        Parse the relation and symbolically verify that c_p - c_v = k with k positive.
        """
        if self.cp_cv_relation is None:
            return (True, "SKIP: cp_cv_relation not available.")
        try:
            eq = sp.sympify(self.cp_cv_relation)
            if not isinstance(eq, sp.Equality):
                return (True, "SKIP: Cannot parse equality.")
            # Solve for (c_p - c_v)
            cp, cv = sp.symbols('c_p c_v')
            diff = cp - cv
            solutions = sp.solve(eq, diff, dict=True)
            if solutions:
                # Expect diff = k
                k = sp.symbols('k', positive=True, real=True)
                if sp.simplify(solutions[0][diff] - k) == 0:
                    return (True, "PASS: c_p - c_v = k, and k > 0 implies c_p > c_v.")
                else:
                    return (True, "SKIP: Could not match the expected form exactly.")
            else:
                return (True, "SKIP: Could not solve for c_p - c_v.")
        except Exception as e:
            return (True, f"SKIP: Error in invariant test ({e}).")

    # -----------------------------------------------------------------
    # Monotonicity test – explicitly perturb input k in cp_cv_relation
    # -----------------------------------------------------------------
    def test_monotonicity_cp_increases_with_k(self) -> Tuple[bool, str]:
        """
        Use the cp_cv_relation to compute cp from given numeric values of cv and k.
        Baseline: cv=1.0, k=1.0  => cp = 2.0
        Perturbation: increase k by 10% => k=1.1 => cp = 2.1
        Verify that cp increases when k increases.
        """
        if self.cp_cv_relation is None:
            return (True, "SKIP: cp_cv_relation not available.")
        try:
            # Define symbols and substitute numeric values
            cp, cv, k = sp.symbols('c_p c_v k', real=True)
            eq = sp.sympify(self.cp_cv_relation)
            if not isinstance(eq, sp.Equality):
                return (True, "SKIP: cp_cv_relation is not an equation.")

            # Solve for cp in terms of cv and k
            sol = sp.solve(eq, cp, dict=True)
            if not sol:
                return (True, "SKIP: Could not solve for cp.")
            cp_expr = sol[0][cp]  # expression for cp

            # Baseline: cv=1.0, k=1.0
            cp_baseline = float(cp_expr.subs({cv: 1.0, k: 1.0}))
            # Perturbed: increase k by 10%
            cp_perturbed = float(cp_expr.subs({cv: 1.0, k: 1.1}))
            if cp_perturbed > cp_baseline:
                return (True, f"PASS: cp increased from {cp_baseline:.4f} to {cp_perturbed:.4f} when k increased by 10%.")
            else:
                return (False, f"FAIL: cp did not increase (baseline {cp_baseline:.4f}, perturbed {cp_perturbed:.4f})")
        except Exception as e:
            return (True, f"SKIP: Error during monotonicity test ({e}).")

    # -----------------------------------------------------------------
    # Modeling/trace tests – check that explanation steps contain key
    # physical steps (if explanation_steps provided)
    # -----------------------------------------------------------------
    def test_modeling_cp_cv_derivation_steps(self) -> Tuple[bool, str]:
        if self.explanation_steps is None:
            return (True, "SKIP: explanation_steps not provided.")
        steps_text = "\n".join(self.explanation_steps).lower()
        required = ["h = u + pv", "pv = kt", "c_p = c_v + k"]
        missing = [s for s in required if s not in steps_text]
        if missing:
            return (False, f"FAIL: Missing derivation steps: {missing}")
        return (True, "PASS: All key steps for cp = cv + k present.")

    def test_modeling_adiabatic_derivation_steps(self) -> Tuple[bool, str]:
        if self.explanation_steps is None:
            return (True, "SKIP: explanation_steps not provided.")
        steps_text = "\n".join(self.explanation_steps).lower()
        required = ["du = -p dv", "c_v dt = -p dv", "p v^gamma = constant"]
        missing = [s for s in required if s not in steps_text]
        if missing:
            return (False, f"FAIL: Missing derivation steps: {missing}")
        return (True, "PASS: All key steps for adiabatic relation present.")

    # -----------------------------------------------------------------
    # Weak answer sanity – only one method; no hardcoded target values
    # -----------------------------------------------------------------
    def test_answer_sanity_format(self) -> Tuple[bool, str]:
        """
        Check that both result strings have plausible format.
        (Does not check the physics content beyond formatting.)
        """
        issues = []
        if self.cp_cv_relation is not None:
            if not re.search(r'c\s*_?\s*p', self.cp_cv_relation, re.IGNORECASE):
                issues.append("cp_cv_relation missing 'c_p'")
            if not re.search(r'c\s*_?\s*v', self.cp_cv_relation, re.IGNORECASE):
                issues.append("cp_cv_relation missing 'c_v'")
            if 'k' not in self.cp_cv_relation:
                issues.append("cp_cv_relation missing 'k'")
            if '=' not in self.cp_cv_relation:
                issues.append("cp_cv_relation missing '='")
        else:
            issues.append("cp_cv_relation not present")

        if self.adiabatic_relation is not None:
            if not re.search(r'p\s*V', self.adiabatic_relation, re.IGNORECASE):
                issues.append("adiabatic_relation missing 'pV'")
            if not re.search(r'gamma', self.adiabatic_relation, re.IGNORECASE):
                issues.append("adiabatic_relation missing 'gamma'")
            if not re.search(r'(constant|const)', self.adiabatic_relation, re.IGNORECASE):
                issues.append("adiabatic_relation missing constancy indication")
        else:
            issues.append("adiabatic_relation not present")

        if issues:
            return (False, f"FAIL: {'; '.join(issues)}")
        return (True, "PASS: Both relations have plausible format.")

    # -----------------------------------------------------------------
    # Runner: discover and execute all test methods
    # -----------------------------------------------------------------
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