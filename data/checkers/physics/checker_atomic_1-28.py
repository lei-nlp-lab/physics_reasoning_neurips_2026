import math
import sympy
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for fine structure multiple-choice problem.
    """
    def __init__(self, solution_instance):
        # Extract results and explanation safely
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        self.fine_structure_cause = results.get('fine_structure_cause', None)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_key_exists(self):
        """Check that the key 'fine_structure_cause' exists in results."""
        if self.fine_structure_cause is None:
            return (False, "Missing key 'fine_structure_cause' in results")
        return (True, "Key 'fine_structure_cause' is present")

    def test_interface_valid_option(self):
        """Check that the value is one of the allowed options: (a), (b), (c)."""
        if self.fine_structure_cause not in ['(a)', '(b)', '(c)']:
            return (False, f"Invalid option value: {self.fine_structure_cause!r}")
        return (True, f"Option is valid: {self.fine_structure_cause}")

    # ------------------------------------------------------------------
    # Invariant tests (physics mapping based)
    # ------------------------------------------------------------------
    def test_invariant_fine_structure_cause(self):
        """
        According to the physics mapping, fine structure arises from 
        electron spin-orbit coupling (option (a)).
        """
        if self.fine_structure_cause is None:
            return (False, "SKIP: fine_structure_cause unavailable")
        if self.fine_structure_cause != '(a)':
            return (False, f"Fine structure cause should be '(a)' but got {self.fine_structure_cause!r}")
        return (True, "Fine structure cause matches physics mapping (option (a))")

    def test_invariant_not_hyperfine(self):
        """
        Fine structure splitting is much larger than hyperfine; 
        nuclear spin (c) is not the correct cause.
        """
        if self.fine_structure_cause is None:
            return (False, "SKIP: fine_structure_cause unavailable")
        if self.fine_structure_cause == '(c)':
            return (False, "Nuclear spin causes hyperfine structure, not fine structure")
        return (True, "Option (c) is correctly excluded")

    # ------------------------------------------------------------------
    # Modeling / trace tests (explanation and pseudocode consistency)
    # ------------------------------------------------------------------
    def test_modeling_explanation_mentions_spin_orbit(self):
        """
        Check that explanation_steps mention spin‑orbit coupling 
        (a characteristic of correct modeling).
        """
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not available")
        text = str(self.explanation_steps).lower()
        keywords = ['spin-orbit', 'spin orbit', 'spin‐orbit']
        found = any(kw in text for kw in keywords)
        if not found:
            return (False, "Explanation does not mention spin-orbit coupling")
        return (True, "Explanation mentions spin-orbit coupling")

    def test_trace_pseudocode_emit_consistency(self):
        """
        Check that the pseudocode's [emit] contract matches the actual result.
        """
        if self.pseudocode is None:
            return (False, "SKIP: pseudocode not available")
        text = str(self.pseudocode)
        if "results['fine_structure_cause']" in text:
            # Parse expected emit value (simple check)
            # Typically it is '(a)'
            expected = '(a)'  # from contract
            if self.fine_structure_cause != expected:
                return (False, f"Pseudocode emit expects {expected!r} but results have {self.fine_structure_cause!r}")
            return (True, f"Pseudocode emit contract matches result: {expected!r}")
        return (False, "Pseudocode does not contain emit key 'fine_structure_cause'")

    # ------------------------------------------------------------------
    # Monotonicity test (based on physics mapping of spin-orbit energy)
    # ------------------------------------------------------------------
    def test_monotonicity_spin_orbit_energy_increases_with_j(self):
        """
        Perturb total angular momentum j and verify that the spin‑orbit 
        energy shift ΔE = (ħ²/2)[j(j+1)-l(l+1)-s(s+1)] increases.

        For a fixed p‑state (l=1, s=1/2), baseline j1 = l+s = 1.5.
        Perturb j to j2 = 1.51 and compare ΔE.
        """
        # Physical constants (ħ = 1 for simplicity, irrelevant for monotonicity)
        hbar2 = 1.0          # ħ² units
        l = 1.0
        s = 0.5
        # Baseline j (parallel alignment)
        j1 = l + s
        # Perturb j to a slightly larger value (still allowed for a p‑state)
        j2 = 1.51
        # Energy shift function (scaled by 1/ħ² for simplicity, but monotonicity unaffected)
        def delta_E(j):
            return 0.5 * hbar2 * (j*(j+1) - l*(l+1) - s*(s+1))
        E1 = delta_E(j1)
        E2 = delta_E(j2)
        if not (E2 > E1):
            return (False, f"Energy shift did not increase: E(j1={j1})={E1}, E(j2={j2})={E2}")
        return (True, f"ΔE increases from {E1} to {E2} when j increases from {j1} to {j2}")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one)
    # ------------------------------------------------------------------
    def test_answer_sanity_valid_format(self):
        """
        Weak sanity: the answer must be a string starting with '(' and ending with ')'.
        """
        if self.fine_structure_cause is None:
            return (False, "SKIP: fine_structure_cause unavailable")
        val = self.fine_structure_cause
        if not isinstance(val, str):
            return (False, f"Answer is not a string: {type(val).__name__}")
        if not (val.startswith('(') and val.endswith(')')):
            return (False, f"Answer '{val}' does not have parentheses format")
        return (True, "Answer format is valid")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        """
        Discover and run all test_* methods. Returns (success, logs).
        """
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
                passed = False
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)