import re
import traceback
from typing import Any, List, Tuple

class PhysicsChecker:
    """
    Audits a solution for the two-electron system in a three‑level potential.
    Checks antisymmetry, Slater determinant form, matrix element structure,
    and linearity in the interaction (monotonicity).
    """

    def __init__(self, solution_instance: Any):
        # Extract results safely; default to None if missing
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None) or []
        self.pseudocode = getattr(solution_instance, 'pseudocode', None) or {}

        # Safely retrieve keys
        self.psi12 = self.results.get('psi12', None)
        self.psi13 = self.results.get('psi13', None)
        self.psi23 = self.results.get('psi23', None)
        self.M = self.results.get('M', None)

    # ------------------------------------------------------------------
    # Helper: clean LaTeX string for pattern matching
    # ------------------------------------------------------------------
    def _clean(self, s: str) -> str:
        """Remove whitespace, double backslashes, and $ signs for matching."""
        s = re.sub(r'\s+', '', s)
        s = s.replace('\\\\', '').replace('$', '')
        return s

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all four required keys exist and are non‑None."""
        missing = []
        for key in ['psi12', 'psi13', 'psi23', 'M']:
            if self.results.get(key) is None:
                missing.append(key)
        if missing:
            return False, f"Missing result keys: {missing}"
        return True, "All required keys present"

    # ------------------------------------------------------------------
    # Invariant checks
    # ------------------------------------------------------------------
    def _check_antisymmetry_string(self, wf_str: str, i: int, j: int) -> Tuple[bool, str]:
        """Weak structural antisymmetry check."""
        if not wf_str or not isinstance(wf_str, str):
            return False, "Wavefunction is not a string"
        cleaned = self._clean(wf_str)
        # Normalise \frac{1}{\sqrt{2}} to 1/sqrt(2)
        cleaned = cleaned.replace(r'\frac{1}{\sqrt{2}}', '1/sqrt(2)')
        pat1 = rf'psi_{i}\(r1\)psi_{j}\(r2\)-psi_{j}\(r1\)psi_{i}\(r2\)'
        pat2 = rf'psi_{i}\(r2\)psi_{j}\(r1\)-psi_{j}\(r2\)psi_{i}\(r1\)'
        if re.search(pat1, cleaned) or re.search(pat2, cleaned):
            return True, f"Wavefunction ψ_{i}{j} shows antisymmetric structure"
        else:
            return False, f"Wavefunction ψ_{i}{j} lacks visible antisymmetric pattern"

    def test_invariant_wavefunction_antisymmetry_psi12(self) -> Tuple[bool, str]:
        if self.psi12 is None:
            return (False, "SKIP: psi12 not available")
        return self._check_antisymmetry_string(self.psi12, 1, 2)

    def test_invariant_wavefunction_antisymmetry_psi13(self) -> Tuple[bool, str]:
        if self.psi13 is None:
            return (False, "SKIP: psi13 not available")
        return self._check_antisymmetry_string(self.psi13, 1, 3)

    def test_invariant_wavefunction_antisymmetry_psi23(self) -> Tuple[bool, str]:
        if self.psi23 is None:
            return (False, "SKIP: psi23 not available")
        return self._check_antisymmetry_string(self.psi23, 2, 3)

    def test_invariant_matrix_element_direct_exchange(self) -> Tuple[bool, str]:
        """Check that M is written as direct minus exchange with correct ordering."""
        if self.M is None or not isinstance(self.M, str):
            return (False, "SKIP: M not available or not a string")
        cleaned = self._clean(self.M)
        # Allow V' or Vprime or V'
        direct = r'<psi_3\(r1\)psi_1\(r2\)\|[Vv]\'?[^|]*\|psi_2\(r1\)psi_1\(r2\)>'
        exchange = r'<psi_1\(r1\)psi_3\(r2\)\|[Vv]\'?[^|]*\|psi_2\(r1\)psi_1\(r2\)>'
        pattern = direct + r'-' + exchange
        if re.search(pattern, cleaned):
            return True, "Matrix element shows correct direct–exchange structure"
        else:
            return False, "Matrix element does not match expected direct–exchange form"

    # ------------------------------------------------------------------
    # Modeling / trace checks
    # ------------------------------------------------------------------
    def test_modeling_slater_determinant_features(self) -> Tuple[bool, str]:
        """Check each wavefunction has 1/√2, minus sign, and both r1,r2."""
        for name, wf in [('psi12', self.psi12), ('psi13', self.psi13), ('psi23', self.psi23)]:
            if wf is None:
                continue
            if not isinstance(wf, str):
                return False, f"{name} is not a string"
            cleaned = self._clean(wf)
            # Check normalisation factor (allow 1/√2 or 1/sqrt(2) or \frac{1}{\sqrt{2}})
            norm = r'(1/√2|1/sqrt\(2\)|\\frac\{1\}\{\\sqrt\{2\}\})'
            if not re.search(norm, cleaned):
                return False, f"{name} missing normalisation factor 1/√2"
            if '-' not in cleaned:
                return False, f"{name} missing minus sign (antisymmetry)"
            if 'r1' not in cleaned or 'r2' not in cleaned:
                return False, f"{name} does not contain both r1 and r2"
        return True, "All wavefunctions contain basic Slater determinant features"

    def test_trace_explanation_mentions_direct_exchange(self) -> Tuple[bool, str]:
        """Check explanation steps or pseudocode mention direct–exchange or Slater."""
        steps_text = ' '.join(str(s) for s in self.explanation_steps).lower()
        pseudo_text = str(self.pseudocode).lower()
        combined = steps_text + ' ' + pseudo_text
        keywords = ['direct', 'exchange', 'antisymmetric', 'slater']
        found = [kw for kw in keywords if kw in combined]
        if len(found) >= 2:
            return True, f"Tracing references key concepts: {found}"
        else:
            return False, "Explanation does not clearly reference direct–exchange or Slater determinant"

    # ------------------------------------------------------------------
    # Monotonicity check: linear scaling with interaction strength
    # ------------------------------------------------------------------
    def test_monotonicity_matrix_element_linear_in_interaction_strength(self) -> Tuple[bool, str]:
        """
        Verify that the matrix element M scales linearly with the interaction
        strength λ.  We treat the interaction operator V' as having strength λ
        (i.e., λ V').  The baseline (λ = 1) should give each term exactly one V'
        with no quadratic or higher powers.  We check this by counting the
        number of V' tokens in each bra‑ket term.  If the expression is linear,
        scaling λ by a factor c would multiply M by c.  This test confirms that
        the expression supports such linear scaling.
        """
        if self.M is None or not isinstance(self.M, str):
            return (False, "SKIP: M not available or not a string")

        cleaned = self._clean(self.M)

        # Define the V' token patterns (allow LaTeX variants)
        vprime_pattern = r"V'|V\\prime|V_{\\text{prime}}|V_{\\text{int}}"

        # Count total V' occurrences in the whole expression
        total_vprime = len(re.findall(vprime_pattern, cleaned))
        if total_vprime == 0:
            return False, "No interaction operator V' found in M"

        # Check there is no exponent or square on V' (e.g., V'^2)
        if re.search(r"V'.*?\^\{?2\}?", cleaned) or re.search(r"Vprime.*?\^\{?2\}?", cleaned):
            return False, "Interaction appears with exponent 2, nonlinear in V'"

        # Extract bra‑ket terms (direct and exchange) separated by the outer minus.
        # The expression should have exactly two bra‑kets.
        braket_pattern_angle = r'<[^>]*\|[^|]*\|[^>]*>'
        braket_pairs = re.findall(braket_pattern_angle, cleaned)

        # Also try Unicode angle brackets if standard ones not found
        if len(braket_pairs) < 2:
            braket_pattern_unicode = r'⟨[^⟩]*\|[^|]*\|[^⟩]*⟩'
            braket_pairs = re.findall(braket_pattern_unicode, cleaned)

        if len(braket_pairs) < 2:
            # Fallback: just check that there are two occurrences of '|' inside each term
            # Split on outer '-' (careful: minus sign not inside bra‑ket)
            # Use simple split on '-' that is not inside angle brackets
            # This is more fragile; we'll rely on the bra‑ket extraction.
            pass

        for bk in braket_pairs:
            cnt = len(re.findall(vprime_pattern, bk))
            if cnt != 1:
                return False, f"Found bra‑ket with {cnt} V' operators (expected exactly 1)"

        return True, ("Matrix element is linear in V': each bra‑ket contains exactly one V' token, "
                      "supporting scaling by an interaction strength λ (monotonicity)")

    # ------------------------------------------------------------------
    # Weak final‑answer sanity (do NOT re‑derive, just check non‑emptiness)
    # ------------------------------------------------------------------
    def test_answer_sanity_outputs_nonempty(self) -> Tuple[bool, str]:
        """Weak check: all four result values are non‑empty strings."""
        for name, val in [('psi12', self.psi12), ('psi13', self.psi13),
                          ('psi23', self.psi23), ('M', self.M)]:
            if val is None or (isinstance(val, str) and not val.strip()):
                return False, f"{name} is empty or None"
            if not isinstance(val, str):
                return False, f"{name} is not a string"
        return True, "All outputs are non‑empty strings"

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)


# Example usage (for testing):
if __name__ == "__main__":
    # Dummy solution instance
    class DummySolution:
        results = {
            'psi12': r'$\psi_{12} = \frac{1}{\sqrt{2}} (\psi_1(r_1) \psi_2(r_2) - \psi_1(r_2) \psi_2(r_1))$',
            'psi13': r'$\psi_{13} = \frac{1}{\sqrt{2}} (\psi_1(r_1) \psi_3(r_2) - \psi_1(r_2) \psi_3(r_1))$',
            'psi23': r'$\psi_{23} = \frac{1}{\sqrt{2}} (\psi_2(r_1) \psi_3(r_2) - \psi_2(r_2) \psi_3(r_1))$',
            'M': r'$\langle \psi_{13} | \delta H | \psi_{12} \rangle = \langle \psi_3(r_1) \psi_1(r_2) | V\'(r_1, r_2) | \psi_2(r_1) \psi_1(r_2) \rangle - \langle \psi_1(r_1) \psi_3(r_2) | V\'(r_1, r_2) | \psi_2(r_1) \psi_1(r_2) \rangle$'
        }
        explanation_steps = [
            "The wave functions for a fermion system are antisymmetric as Slater determinants...",
            "Using the direct-exchange formula: <ψ_ab|V'|ψ_cd> = <ψ_aψ_b|V'|ψ_cψ_d> - <ψ_aψ_b|V'|ψ_dψ_c>"
        ]
        pseudocode = {"sub2": {"steps": ["Insert antisymmetric wavefunctions..."]}}

    checker = PhysicsChecker(DummySolution())
    success, logs = checker.run_all_tests()
    for log in logs:
        print(log)
    print(f"\nOverall success: {success}")