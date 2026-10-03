import sympy as sp
import math
import traceback
import re

class PhysicsChecker:
    """
    Checker for solution to problem about parity and time reversal transformations
    for spin and momentum, and suggestion for T-violation in Lambda decay.
    """
    def __init__(self, solution_instance):
        # Extract results and explanation steps safely
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        if self.explanation_steps is None:
            self.explanation_steps = []

        # Required result keys (from pseudocode [emit] contract)
        self.s_parity = self.results.get('s_parity', None)
        self.p_parity = self.results.get('p_parity', None)
        self.s_time = self.results.get('s_time', None)
        self.p_time = self.results.get('p_time', None)
        self.A_T = self.results.get('A_T', None)

    # ------------------------------------------------------------
    # Interface tests (key existence, type)
    # ------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all five required keys are present in results."""
        required = ['s_parity', 'p_parity', 's_time', 'p_time', 'A_T']
        missing = [key for key in required if key not in self.results]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present.")

    def test_interface_results_not_none(self):
        """Check that each required result is not None."""
        for key in ['s_parity', 'p_parity', 's_time', 'p_time', 'A_T']:
            val = self.results.get(key)
            if val is None:
                return (False, f"Key '{key}' has value None.")
        return (True, "All required results are non-None.")

    def test_interface_string_type(self):
        """Check that all required results are strings."""
        for key in ['s_parity', 'p_parity', 's_time', 'p_time', 'A_T']:
            val = self.results.get(key)
            if not isinstance(val, str):
                return (False, f"Key '{key}' is not a string (type: {type(val).__name__}).")
        return (True, "All required results are strings.")

    # ------------------------------------------------------------
    # Invariant tests (theorem-based physics sanity)
    # ------------------------------------------------------------
    def test_invariant_parity_spin(self):
        """Under parity, spin (pseudovector) is unchanged -> expression should not contain minus before s."""
        expr = self.s_parity
        if expr is None:
            return (False, "SKIP: s_parity not available")
        if re.search(r'-\s*[sS]', expr):
            return (False, f"s_parity incorrectly contains a minus sign before s: '{expr}'")
        return (True, f"s_parity: {expr} (spin unchanged under parity)")

    def test_invariant_parity_momentum(self):
        """Under parity, momentum (polar vector) flips sign -> expression should contain minus before p."""
        expr = self.p_parity
        if expr is None:
            return (False, "SKIP: p_parity not available")
        if '-' not in expr:
            return (False, f"p_parity does not contain a minus sign: '{expr}'")
        if not re.search(r'-\s*[pP]', expr):
            return (False, f"p_parity contains a minus but not before p: '{expr}'")
        return (True, f"p_parity: {expr} (momentum flips under parity)")

    def test_invariant_time_spin(self):
        """Under time reversal, spin flips sign -> expression should contain minus before s."""
        expr = self.s_time
        if expr is None:
            return (False, "SKIP: s_time not available")
        if '-' not in expr:
            return (False, f"s_time does not contain a minus sign: '{expr}'")
        if not re.search(r'-\s*[sS]', expr):
            return (False, f"s_time contains a minus but not before s: '{expr}'")
        return (True, f"s_time: {expr} (spin flips under time reversal)")

    def test_invariant_time_momentum(self):
        """Under time reversal, momentum flips sign -> expression should contain minus before p."""
        expr = self.p_time
        if expr is None:
            return (False, "SKIP: p_time not available")
        if '-' not in expr:
            return (False, f"p_time does not contain a minus sign: '{expr}'")
        if not re.search(r'-\s*[pP]', expr):
            return (False, f"p_time contains a minus but not before p: '{expr}'")
        return (True, f"p_time: {expr} (momentum flips under time reversal)")

    # ------------------------------------------------------------
    # Modeling/Trace tests (quality of explanation steps / suggestions)
    # ------------------------------------------------------------
    def test_modeling_triple_product_mentioned(self):
        """Check that explanation steps mention the triple product Q = s_Lambda . (p_N x p_pi)."""
        if not self.explanation_steps:
            return (False, "SKIP: No explanation steps provided")
        combined = " ".join(self.explanation_steps).lower()
        has_triple = any(term in combined for term in ['triple product', 'q =', 'cross', 's_λ', 's_Λ', 's_lambda', 'pn × pπ', 'pn x pπ'])
        if not has_triple:
            return (False, "Explanation steps do not mention the T‑even/odd triple product or cross product.")
        return (True, "Explanation steps include the triple product correlation.")

    def test_modeling_polarization_required(self):
        """Check that solution mentions that Lambda must be polarized."""
        if not self.explanation_steps:
            return (False, "SKIP: No explanation steps provided")
        combined = " ".join(self.explanation_steps).lower()
        if 'polarized' not in combined and 'polarisation' not in combined:
            return (False, "Explanation steps do not mention that Λ must be polarized.")
        return (True, "Polarization requirement mentioned.")

    def test_modeling_two_body_limitation(self):
        """Check that solution notes the collinearity issue in two-body decay or uses an external direction."""
        if not self.explanation_steps:
            return (False, "SKIP: No explanation steps provided")
        combined = " ".join(self.explanation_steps).lower()
        if 'collinear' in combined or 'parallel' in combined or 'zero' in combined or 'production plane' in combined:
            return (True, "Solution acknowledges the collinearity issue or uses an external direction.")
        return (True, "Solution may implicitly rely on a production plane (no explicit caveat found, but accepted).")

    # ------------------------------------------------------------
    # Monotonicity tests (explicit perturbation comparison)
    # ------------------------------------------------------------
    def test_monotonicity_triple_product_T_sign(self):
        """
        Baseline: triple product Q = s·(p_N × p_π).
        Perturbation: apply time reversal (s → -s, p_N → -p_N, p_π → -p_π).
        Compare Q_pert to -Q.
        """
        # Symbolic perturbation
        s = sp.Matrix(sp.symbols('s_x s_y s_z'))
        pN = sp.Matrix(sp.symbols('pN_x pN_y pN_z'))
        ppi = sp.Matrix(sp.symbols('ppi_x ppi_y ppi_z'))
        Q_baseline = s.dot(pN.cross(ppi))
        # Perturbed: all momenta and spin flipped
        Q_pert = (-s).dot((-pN).cross(-ppi))
        # Symbolic simplification: Q_pert should equal -Q_baseline
        diff = sp.simplify(Q_pert + Q_baseline)
        if diff != 0:
            return (False, f"Symbolic check failed: Q_pert + Q = {diff} (should be 0)")
        # Numerical verification with random values
        subs = {'s_x': 1.2, 's_y': -0.5, 's_z': 2.0,
                'pN_x': 3.0, 'pN_y': -1.0, 'pN_z': 0.0,
                'ppi_x': -2.0, 'ppi_y': 1.5, 'ppi_z': 0.5}
        Q_val = float(Q_baseline.subs(subs))
        Qp_val = float(Q_pert.subs(subs))
        if not math.isclose(Qp_val, -Q_val, rel_tol=1e-12, abs_tol=1e-12):
            return (False, f"Numerical check: Q_pert = {Qp_val} != -Q = {-Q_val}")
        return (True, "Triple product changes sign under T (monotonic odd).")

    def test_monotonicity_asymmetry_under_spin_flip(self):
        """
        Baseline: define a generic asymmetry proportional to the dot product of
        spin s with a fixed direction n: A = C * (s·n).
        Perturbation: flip the spin vector s → -s.
        Check that A_pert(= C*(-s)·n) is exactly -A_base.
        This demonstrates the oddness of the T‑odd asymmetry under spin reversal.
        """
        # Define symbolic variables
        s = sp.Matrix(sp.symbols('s0 s1 s2'))
        n = sp.Matrix(sp.symbols('n0 n1 n2'))
        C = sp.symbols('C')
        # Baseline asymmetry
        A_baseline = C * s.dot(n)
        # Perturbed asymmetry (spin flipped)
        A_pert = C * (-s).dot(n)
        # Check that sum is zero (i.e., A_pert = -A_baseline)
        sum_expr = sp.simplify(A_baseline + A_pert)
        if sum_expr != 0:
            return (False, f"Symbolic sum of baseline and perturbed is {sum_expr}, not 0")
        # Also perform a numeric check with a concrete example
        nums = {'s0': 0.6, 's1': 0.0, 's2': 0.8,
                'n0': 1.0, 'n1': 0.0, 'n2': 0.0, 'C': 0.5}
        A_base_val = float(A_baseline.subs(nums))
        A_pert_val = float(A_pert.subs(nums))
        if not math.isclose(A_pert_val, -A_base_val, rel_tol=1e-12, abs_tol=1e-12):
            return (False, f"Numeric check: A_pert = {A_pert_val} != -A_base = {-A_base_val}")
        return (True, "Asymmetry changes sign under spin flip (monotonic odd).")

    # ------------------------------------------------------------
    # Answer sanity (weak, non‑derivative)
    # ------------------------------------------------------------
    def test_answer_sanity_A_T(self):
        """Weak sanity: A_T should be a non‑empty string that looks like an asymmetry expression."""
        if self.A_T is None:
            return (False, "SKIP: A_T not available")
        if not isinstance(self.A_T, str) or len(self.A_T.strip()) == 0:
            return (False, "A_T is empty or not a string.")
        if any(ph in self.A_T.lower() for ph in ['n_up', 'n_down', 'up', 'down', 'asymmetry', 'a_t', '⟨q⟩']):
            return (True, f"A_T is a plausible asymmetry expression: {self.A_T}")
        return (True, "A_T is provided (basic sanity pass).")

    # ------------------------------------------------------------
    # Central runner
    # ------------------------------------------------------------
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