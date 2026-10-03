import re
import math
import sympy as sp
from collections import defaultdict

class PhysicsChecker:
    """
    Audits student solutions for the semiempirical binding energy formula and its explanation.
    Covers interface, invariants, modeling/trace, monotonicity, and one weak answer sanity.
    """

    def __init__(self, solution_instance):
        # Extract formula from pseudocode [emit] contract
        self.formula = solution_instance.results.get('formula', None)
        # Explanation may come from different fields
        explanation = solution_instance.results.get('explanation', None)
        if explanation is None:
            explanation = getattr(solution_instance, 'explanation', None)
        self.explanation = explanation if explanation else ""

        # Normalize formula for easier pattern matching
        self.clean_formula = self._clean_formula(self.formula) if self.formula else ""

        # Symbolic infrastructure – not strictly needed for numeric checks but kept for interface
        self.a_v, self.a_s, self.a_c, self.a_a, self.a_p, self.delta = sp.symbols('a_v a_s a_c a_a a_p delta', positive=True)
        self.A, self.Z = sp.symbols('A Z', positive=True)

    def _clean_formula(self, s):
        """Remove LaTeX braces and underscores to get a plainer string."""
        s = s.replace('{', '').replace('}', '')
        s = s.replace('_', '')
        s = s.replace('\\', '')
        return s

    def run_all_tests(self):
        """Discover and execute all test_ methods, collect results."""
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
                import traceback
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

    # ------------------------------------------------------------------
    # Interface tests – check required keys and structure
    # ------------------------------------------------------------------
    def test_interface_formula_exists(self):
        if self.formula is None:
            return (False, "Missing required key 'formula' in results.")
        return (True, "Formula key present.")

    def test_interface_formula_contains_volume_term(self):
        if self.formula is None:
            return (False, "SKIP: formula missing")
        pattern = r'a_v\s*\*?\s*A'
        if re.search(pattern, self.clean_formula):
            return (True, "Volume term (a_v A) found.")
        return (False, f"Volume term not found in formula: {self.formula}")

    def test_interface_formula_contains_surface_term(self):
        if self.formula is None:
            return (False, "SKIP: formula missing")
        pattern = r'\-a_s\s*\*?\s*A\^?\{?2/3\}?'
        if re.search(pattern, self.clean_formula):
            return (True, "Surface term (-a_s A^(2/3)) found.")
        return (False, f"Surface term not found: {self.formula}")

    def test_interface_formula_contains_coulomb_term(self):
        if self.formula is None:
            return (False, "SKIP: formula missing")
        pattern = r'\-a_[ce]\s*\*?\s*Z\^?\{?2\}?\s*\*?\s*A\^?\{\-?1/3\}?'
        if re.search(pattern, self.clean_formula):
            return (True, "Coulomb term (-a_c Z^2 A^{-1/3}) found.")
        return (False, f"Coulomb term not found: {self.formula}")

    def test_interface_formula_contains_asymmetry_term(self):
        if self.formula is None:
            return (False, "SKIP: formula missing")
        # Broad check: contains a_a and (A/2 - Z)^2 or (A-2Z)^2
        if re.search(r'a_a.*\([A].*[Z]\).*\^[^0-9]', self.clean_formula):
            return (True, "Asymmetry term found.")
        if 'a_a' in self.clean_formula and ('(A/2 - Z)^2' in self.clean_formula or '(A-2Z)^2' in self.clean_formula):
            return (True, "Asymmetry term found.")
        if 'a_a' in self.clean_formula and 'A^{-1}' in self.clean_formula:
            return (True, "Asymmetry term (with A^{-1}) found.")
        return (False, f"Asymmetry term not clearly found: {self.formula}")

    def test_interface_formula_contains_pairing_term(self):
        if self.formula is None:
            return (False, "SKIP: formula missing")
        if re.search(r'a_p', self.clean_formula):
            return (True, "Pairing term (a_p) present.")
        if 'delta' in self.clean_formula and 'A^{-1/2}' in self.clean_formula:
            return (True, "Pairing term (delta*A^{-1/2}) found.")
        return (False, f"Pairing term not found: {self.formula}")

    # ------------------------------------------------------------------
    # Invariant tests – physical sanity based on explanation
    # ------------------------------------------------------------------
    def test_invariant_surface_significant_small_A(self):
        if not self.explanation:
            return (False, "SKIP: no explanation provided")
        exp_lower = self.explanation.lower()
        keywords = ['surface', 'small a', 'light nuclei', 'low a']
        if any(kw in exp_lower for kw in keywords):
            return (True, "Explanation mentions surface term significance for small A.")
        return (False, "Explanation does not discuss surface term for small A.")

    def test_invariant_coulomb_significant_large_A(self):
        if not self.explanation:
            return (False, "SKIP: no explanation provided")
        exp_lower = self.explanation.lower()
        keywords = ['coulomb', 'large a', 'heavy nuclei', 'high a']
        if any(kw in exp_lower for kw in keywords):
            return (True, "Explanation mentions Coulomb term significance for large A.")
        return (False, "Explanation does not discuss Coulomb term for large A.")

    def test_invariant_peak_near_Fe56(self):
        if not self.explanation:
            return (False, "SKIP: no explanation provided")
        exp_lower = self.explanation.lower()
        if '56' in exp_lower or 'peak' in exp_lower or 'maximum' in exp_lower or 'iron' in exp_lower:
            return (True, "Explanation mentions the peak near A=56 (iron).")
        return (False, "Explanation does not reference the binding energy peak near Fe-56.")

    # ------------------------------------------------------------------
    # Modeling / trace tests – check explanation covers all terms and trend
    # ------------------------------------------------------------------
    def test_modeling_explanation_mentions_five_terms(self):
        if not self.explanation:
            return (False, "SKIP: no explanation provided")
        exp_lower = self.explanation.lower()
        terms = ['volume', 'surface', 'coulomb', 'asymmetry', 'pairing']
        found = [t for t in terms if t in exp_lower]
        if len(found) >= 4:
            return (True, f"Explanation mentions {len(found)} of the five terms: {found}.")
        return (False, f"Explanation mentions only {len(found)} of the five terms ({found}). Need at least 4.")

    def test_modeling_explanation_trend(self):
        if not self.explanation:
            return (False, "SKIP: no explanation provided")
        exp_lower = self.explanation.lower()
        increase_words = ['increase', 'rise', 'grow']
        decrease_words = ['decrease', 'fall', 'drop', 'decline', 'go down']
        has_increase = any(w in exp_lower for w in increase_words)
        has_decrease = any(w in exp_lower for w in decrease_words)
        if has_increase and has_decrease:
            return (True, "Explanation describes both the initial increase and later decrease of B/A with A.")
        elif has_increase:
            return (False, "Explanation only mentions increase, not the eventual decrease.")
        elif has_decrease:
            return (False, "Explanation only mentions decrease, not the initial increase.")
        else:
            return (False, "Explanation does not clearly describe the trend of B/A with A.")

    # ------------------------------------------------------------------
    # Monotonicity tests – explicit perturbation of A and check of B/A change
    # ------------------------------------------------------------------
    def _stable_Z(self, A):
        """Approximate Z for a stable nucleus with mass number A."""
        return A / (2.0 + 0.015 * A ** (2.0 / 3.0))

    def _binding_per_nucleon(self, A, Z=None):
        """
        Compute binding energy per nucleon (MeV) using typical coefficients.
        Pairing term omitted for simplicity; the trend is robust.
        """
        if Z is None:
            Z = self._stable_Z(A)
        a_v = 15.75
        a_s = 17.8
        a_c = 0.711
        a_a = 23.7
        term_v = a_v
        term_s = -a_s * A ** (-1.0 / 3.0)
        term_c = -a_c * (Z ** 2) * A ** (-4.0 / 3.0)
        term_a = -a_a * (0.5 - Z / A) ** 2
        return term_v + term_s + term_c + term_a

    def test_monotonicity_surface_dominated_increase(self):
        """
        For small A, perturbing A upward (e.g., from 20 to 21) should increase B/A
        because the surface term's negative contribution decreases faster than the Coulomb term increases.
        This tests the monotonic increase in the light nucleus region.
        """
        if self.formula is None:
            return (False, "SKIP: formula not provided, cannot infer physics context.")
        try:
            baseline_A = 20.0
            delta_A = 1.0  # 5% perturbation
            perturbed_A = baseline_A + delta_A
            eps0 = self._binding_per_nucleon(baseline_A)
            eps1 = self._binding_per_nucleon(perturbed_A)
            if eps1 > eps0:
                return (True, f"ε({perturbed_A}) = {eps1:.4f} > ε({baseline_A}) = {eps0:.4f}, "
                              f"monotonic increase when perturbing A upward by {delta_A}.")
            else:
                return (False, f"Expected monotonic increase, but ε({perturbed_A}) = {eps1:.4f} <= ε({baseline_A}) = {eps0:.4f}.")
        except Exception as e:
            return (False, f"Error during computation: {e}")

    def test_monotonicity_coulomb_dominated_decrease(self):
        """
        For large A, perturbing A upward (e.g., from 200 to 201) should decrease B/A
        because the Coulomb term becomes more negative and dominates.
        This tests the monotonic decrease in the heavy nucleus region.
        """
        if self.formula is None:
            return (False, "SKIP: formula not provided, cannot infer physics context.")
        try:
            baseline_A = 200.0
            delta_A = 1.0  # 0.5% perturbation
            perturbed_A = baseline_A + delta_A
            eps0 = self._binding_per_nucleon(baseline_A)
            eps1 = self._binding_per_nucleon(perturbed_A)
            if eps1 < eps0:
                return (True, f"ε({perturbed_A}) = {eps1:.4f} < ε({baseline_A}) = {eps0:.4f}, "
                              f"monotonic decrease when perturbing A upward by {delta_A}.")
            else:
                return (False, f"Expected monotonic decrease, but ε({perturbed_A}) = {eps1:.4f} >= ε({baseline_A}) = {eps0:.4f}.")
        except Exception as e:
            return (False, f"Error during computation: {e}")

    # ------------------------------------------------------------------
    # Answer sanity – weak check only
    # ------------------------------------------------------------------
    def test_answer_sanity_formula_not_trivial(self):
        if self.formula is None:
            return (False, "SKIP: formula missing")
        if len(self.clean_formula) < 30:
            return (False, f"Formula seems too short to be correct: {self.formula}")
        return (True, "Formula length is reasonable.")