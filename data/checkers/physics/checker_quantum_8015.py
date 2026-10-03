import math
import re
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results and givens safely
        self.results = getattr(solution_instance, 'results', None) or {}
        self.givens = getattr(solution_instance, 'givens', None) or {}
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Laue equation strings from results
        self.laue_a = self.results.get('laue_a', None)
        self.laue_b = self.results.get('laue_b', None)
        self.laue_c = self.results.get('laue_c', None)

        # Optional numeric quantities
        self.a = self.givens.get('a', None)
        self.b = self.givens.get('b', None)
        self.c = self.givens.get('c', None)
        self.k_i = self.givens.get('k_i', None)
        self.k_f = self.givens.get('k_f', None)
        # Also try to get Δk if provided
        self.delta_k = self.givens.get('delta_k', None) or self.results.get('delta_k', None)

    # --------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------
    def _is_laue_shape(self, s):
        if not isinstance(s, str):
            return False
        # Pattern: vector_op vector = 2π lX
        pat = r'^[\w\s\-]+[\*·\.\s]?[\w\s=]+2π\s*l[123]\s*$'
        return re.match(pat, s.strip()) is not None

    # --------------------------------------------------------------
    # Interface tests
    # --------------------------------------------------------------
    def test_interface_laue_keys_present(self):
        if self.laue_a is None:
            return False, "Missing results['laue_a']"
        if self.laue_b is None:
            return False, "Missing results['laue_b']"
        if self.laue_c is None:
            return False, "Missing results['laue_c']"
        return True, "All three Laue keys present"

    def test_interface_laue_type_string(self):
        for key, val in [('laue_a', self.laue_a), ('laue_b', self.laue_b), ('laue_c', self.laue_c)]:
            if val is not None and not isinstance(val, str):
                return False, f"results['{key}'] is not a string"
        return True, "All Laue values are strings if present"

    # --------------------------------------------------------------
    # Invariant tests (physical sanity, use math.isclose where numbers exist)
    # --------------------------------------------------------------
    def test_invariant_laue_pattern(self):
        if self.laue_a is None or self.laue_b is None or self.laue_c is None:
            return False, "Cannot check pattern: missing Laue values"
        if not self._is_laue_shape(self.laue_a):
            return False, f"laue_a does not look like a Laue equation: {self.laue_a}"
        if not self._is_laue_shape(self.laue_b):
            return False, f"laue_b does not look like a Laue equation: {self.laue_b}"
        if not self._is_laue_shape(self.laue_c):
            return False, f"laue_c does not look like a Laue equation: {self.laue_c}"
        return True, "All three equations have correct pattern"

    def test_invariant_elastic_scattering(self):
        """If k_i and k_f are numeric, verify |k_i| = |k_f|."""
        if self.k_i is None or self.k_f is None:
            return (False, "SKIP: k_i or k_f not provided")
        try:
            ki = sp.Matrix(self.k_i) if hasattr(self.k_i, '__iter__') else sp.sympify(self.k_i)
            kf = sp.Matrix(self.k_f) if hasattr(self.k_f, '__iter__') else sp.sympify(self.k_f)
            ki_norm = float(ki.norm())
            kf_norm = float(kf.norm())
        except Exception:
            return (False, "SKIP: cannot parse k_i/k_f")
        if not math.isclose(ki_norm, kf_norm, rel_tol=1e-2):
            return False, f"Elastic scattering violated: |k_i|={ki_norm}, |k_f|={kf_norm}"
        return True, "Elastic scattering condition holds"

    # --------------------------------------------------------------
    # Modeling / trace consistency tests
    # --------------------------------------------------------------
    def test_modeling_explanation_steps_contains_factorization(self):
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps")
        txt = " ".join(self.explanation_steps).lower()
        if "factor" not in txt and "form factor" not in txt and "lattice sum" not in txt:
            return False, "Explanation steps do not mention amplitude factorization"
        return True, "Explanation steps mention Born amplitude factorization"

    def test_modeling_explanation_steps_mentions_laue(self):
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps")
        txt = " ".join(self.explanation_steps).lower()
        if "laue" not in txt and "bragg" not in txt:
            return False, "Explanation steps do not mention Laue or Bragg"
        return True, "Explanation steps reference Laue/Bragg condition"

    def test_modeling_explanation_steps_mentions_integers(self):
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps")
        txt = " ".join(self.explanation_steps).lower()
        if "integer" not in txt and "l1" not in txt:
            return False, "Explanation steps do not mention integer indices"
        return True, "Integer indices appear in explanation steps"

    # --------------------------------------------------------------
    # Monotonicity test: perturb lattice vector a and see the dot product scaling
    # --------------------------------------------------------------
    def test_monotonicity_laue_a_scaling(self):
        """If a and delta_k are provided numerically, scale a by factor 1.5 and verify the dot product scales accordingly."""
        if self.a is None or self.delta_k is None:
            return (False, "SKIP: missing numeric a or delta_k")
        try:
            a_vec = sp.Matrix(self.a) if hasattr(self.a, '__iter__') else sp.sympify(self.a)
            dk_vec = sp.Matrix(self.delta_k) if hasattr(self.delta_k, '__iter__') else sp.sympify(self.delta_k)
        except:
            return (False, "SKIP: cannot parse a or delta_k")
        # Baseline dot product
        try:
            dot_base = float(a_vec.dot(dk_vec))
        except:
            return (False, "SKIP: dot product evaluation failed")
        # Perturb a by factor 1.5 (scale all components)
        a_pert = a_vec * 1.5
        dot_pert = float(a_pert.dot(dk_vec))
        # The dot product should also scale by 1.5
        expected_dot = dot_base * 1.5
        if not math.isclose(dot_pert, expected_dot, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"Scaling a by 1.5 changed dot product from {dot_base} to {dot_pert}, expected {expected_dot}"
        return True, "Laue condition dot product scales linearly with lattice vector a"

    # --------------------------------------------------------------
    # Weak answer sanity (at most ONE such test)
    # --------------------------------------------------------------
    def test_answer_sanity_laue_not_empty(self):
        """Check that Laue strings are non-empty and contain no obvious contradictions."""
        for key, val in [('laue_a', self.laue_a), ('laue_b', self.laue_b), ('laue_c', self.laue_c)]:
            if val is None or len(val.strip()) == 0:
                return False, f"results['{key}'] is empty"
            if "contradiction" in val.lower():
                return False, f"results['{key}'] contains contradictory language"
        return True, "All Laue strings are non-empty and plausible"

    # --------------------------------------------------------------
    # Main runner
    # --------------------------------------------------------------
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