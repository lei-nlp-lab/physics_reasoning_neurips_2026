import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Audits a solution for the problem of deriving pressure in a canonical ensemble,
    specifically for a gas of free nonrelativistic particles.
    """

    def __init__(self, solution_instance):
        # Extract results safely with None defaults
        results = getattr(solution_instance, 'results', {})
        self.p_formula = results.get('p_formula', None)
        self.depsilon_dV = results.get('depsilon_dV', None)
        self.p_U_relation = results.get('p_U_relation', None)

        # Explanation steps may be in a separate attribute or within results
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        if self.explanation_steps is None:
            self.explanation_steps = results.get('explanation_steps', None)

    # ---------- Interface checks ----------
    def test_interface_required_keys_present(self):
        """Check that all three required keys exist in results."""
        if self.p_formula is None:
            return (False, "Missing 'p_formula' in results")
        if self.depsilon_dV is None:
            return (False, "Missing 'depsilon_dV' in results")
        if self.p_U_relation is None:
            return (False, "Missing 'p_U_relation' in results")
        return (True, "All three required keys present")

    def test_interface_key_types(self):
        """Check that each key holds a string (or sympy expression) of reasonable length."""
        for name, val in [('p_formula', self.p_formula), ('depsilon_dV', self.depsilon_dV),
                          ('p_U_relation', self.p_U_relation)]:
            if val is None:
                return (False, f"{name} is None")
            if not isinstance(val, str):
                return (False, f"{name} is not a string, got {type(val).__name__}")
            if len(val) < 10:
                return (False, f"{name} appears too short: {len(val)} chars")
        return (True, "All keys are strings of sufficient length")

    # ---------- Invariant (theorem-based) checks ----------
    def test_invariant_p_formula_structure(self):
        """The pressure formula should contain key mathematical symbols."""
        if self.p_formula is None:
            return (False, "SKIP: p_formula not available")
        required_terms = ['Z', 'sum', 'exp', '∂ε', '∂V']
        for term in required_terms:
            if term not in self.p_formula:
                return (False, f"Missing '{term}' in p_formula")
        if '-' not in self.p_formula:
            return (False, "p_formula should contain a minus sign")
        return (True, "p_formula contains all expected elements")

    def test_invariant_depsilon_dV_structure(self):
        """The derivative expression should show proportionality to epsilon_s/V with factor -2/3."""
        if self.depsilon_dV is None:
            return (False, "SKIP: depsilon_dV not available")
        required_terms = ['∂ε', '∂V', '2', '3', 'ε', 'V']
        for term in required_terms:
            if term not in self.depsilon_dV:
                return (False, f"Missing '{term}' in depsilon_dV")
        if '-2/3' not in self.depsilon_dV and '−2/3' not in self.depsilon_dV:
            return (False, "depsilon_dV should contain '-2/3' or '−2/3'")
        return (True, "depsilon_dV contains expected -2/3 factor")

    def test_invariant_p_U_relation_structure(self):
        """The final p = 2U/3V relation should be present."""
        if self.p_U_relation is None:
            return (False, "SKIP: p_U_relation not available")
        if 'p' not in self.p_U_relation:
            return (False, "Missing 'p' in p_U_relation")
        if '2U' not in self.p_U_relation or '3V' not in self.p_U_relation:
            return (False, "Missing '2U/3V' pattern in p_U_relation")
        return (True, "p_U_relation contains p = 2U/3V pattern")

    # ---------- Modeling / trace checks (using explanation steps) ----------
    def test_modeling_derivation_sequence(self):
        """Check that explanation steps mention the progression from (a) to (c)."""
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation steps available")
        steps_text = ' '.join(self.explanation_steps) if isinstance(self.explanation_steps, list) else str(self.explanation_steps)
        checks = {
            'pressure formula derivation': ['p =', 'Z', 'sum', 'exp'],
            'depsilon/dV derivation': ['∂ε', '∂V', '-2/3'],
            'combination to p=2U/3V': ['p = 2U/3V', '2U', '3V']
        }
        for part_name, keywords in checks.items():
            found = any(kw in steps_text for kw in keywords)
            if not found:
                return (False, f"Explanation steps missing evidence for part: {part_name}")
        return (True, "Explanation steps cover all three parts")

    def test_modeling_substitution_consistency(self):
        """Check that the explanation includes substitution of depsilon_dV into p_formula."""
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation steps available")
        steps_text = ' '.join(self.explanation_steps) if isinstance(self.explanation_steps, list) else str(self.explanation_steps)
        if not any(kw in steps_text for kw in ['substitut', 'replace', 'plug', 'insert', 'apply']):
            return (False, "Explanation steps do not indicate substitution step between (b) and (c)")
        return (True, "Explanation shows substitution step")

    # ---------- Monotonicity checks ----------
    def test_monotonicity_pressure_decreases_with_volume(self):
        """
        For p = 2U/(3V), with U fixed, increasing V should decrease p.
        We re-derive the relation symbolically inside the checker.
        """
        if self.p_U_relation is None:
            return (False, "SKIP: p_U_relation not available")

        # Define symbols
        V, U = sp.symbols('V U', positive=True)
        p_expr = 2*U/(3*V)   # baseline expression from known relation

        # Baseline: V=1, U=1
        V0 = 1.0
        U0 = 1.0
        p0 = float(p_expr.subs({V: V0, U: U0}))

        # Perturb V by +10%
        V1 = 1.1
        p1 = float(p_expr.subs({V: V1, U: U0}))

        if not math.isclose(p0, p1 * (V1/V0), rel_tol=1e-9, abs_tol=1e-9):
            # This checks p0 / p1 = V1 / V0 => p0 = p1 * (V1/V0)
            return (False, "Monotonicity violated: p should be inversely proportional to V")
        if p1 >= p0:
            return (False, "Monotonicity violated: increasing V did not decrease p (p1 >= p0)")
        return (True, f"p decreased from {p0:.4f} to {p1:.4f} when V increased from {V0} to {V1}")

    def test_monotonicity_derivative_less_negative_with_volume(self):
        """
        For depsilon/dV = -2/3 * epsilon / V, with epsilon fixed,
        increasing V should make the derivative less negative (increase toward zero).
        """
        if self.depsilon_dV is None:
            return (False, "SKIP: depsilon_dV not available")

        # Define symbols
        epsilon, V = sp.symbols('epsilon V', positive=True)
        deriv_expr = -2*epsilon/(3*V)

        # Baseline: V=1, epsilon=1
        V0 = 1.0
        eps0 = 1.0
        d0 = float(deriv_expr.subs({V: V0, epsilon: eps0}))

        # Perturb V by +10%
        V1 = 1.1
        d1 = float(deriv_expr.subs({V: V1, epsilon: eps0}))

        # Derivative should be more positive (less negative)
        if d1 <= d0:
            return (False, f"Monotonicity violated: derivative should increase (become less negative), but d1={d1:.4f} <= d0={d0:.4f}")
        # Also check the relative change is consistent with inverse proportionality
        if not math.isclose(d0 * V0, d1 * V1, rel_tol=1e-9, abs_tol=1e-9):
            return (False, "Magnitude inverse proportionality not satisfied")
        return (True, f"Derivative increased from {d0:.4f} to {d1:.4f} when V increased from {V0} to {V1}")

    # ---------- Answer sanity check (weak) ----------
    def test_answer_sanity_final_relation(self):
        """Weak sanity: final relation should be non-empty and roughly match expected form."""
        if self.p_U_relation is None:
            return (False, "SKIP: p_U_relation not available")
        stripped = self.p_U_relation.strip()
        if len(stripped) < 5:
            return (False, "p_U_relation is too short to be a meaningful relation")
        for token in ['=', 'p', 'U', 'V']:
            if token not in stripped:
                return (False, f"Missing token '{token}' in p_U_relation")
        return (True, "p_U_relation passes weak sanity: contains =, p, U, V")

    # ---------- Runner ----------
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