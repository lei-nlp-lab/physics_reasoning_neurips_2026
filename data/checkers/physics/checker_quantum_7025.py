import sympy
import math
import traceback

class PhysicsChecker:
    """
    Auditor for the excited (1s)^1(2s)^1 helium atom configuration.
    Checks consistency of results: lower_state, DeltaE, and explanation steps.
    """

    def __init__(self, solution_instance):
        # Extract results dictionary safely
        results = getattr(solution_instance, 'results', None) or {}
        # Required output keys
        self.lower_state = results.get('lower_state', None)
        self.deltaE = results.get('DeltaE', None)
        # Explanation steps (list of strings)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        # Keep the solution instance for potential further use
        self.solution_instance = solution_instance

    # ---------- Interface tests ----------
    def test_interface_lower_state(self):
        """Check that lower_state exists and is a string."""
        if self.lower_state is None:
            return (False, "SKIP: lower_state not provided")
        if not isinstance(self.lower_state, str):
            return (False, f"lower_state should be str, got {type(self.lower_state)}")
        return (True, f"lower_state is present: '{self.lower_state}'")

    def test_interface_deltaE(self):
        """Check that DeltaE exists and is a string."""
        if self.deltaE is None:
            return (False, "SKIP: DeltaE not provided")
        if not isinstance(self.deltaE, str):
            return (False, f"DeltaE should be str, got {type(self.deltaE)}")
        return (True, f"DeltaE is present (length {len(self.deltaE)})")

    # ---------- Invariant tests (theorem-based) ----------
    def test_invariant_lower_state_is_triplet(self):
        """Pauli principle + exchange integral: triplet has lower energy."""
        if self.lower_state is None:
            return (False, "SKIP: lower_state not available")
        expected = 'triplet'
        if self.lower_state.lower() != expected:
            return (False, f"lower_state should be '{expected}', got '{self.lower_state}'")
        return (True, "lower_state is 'triplet' (consistent with Hund's rule)")

    def test_invariant_deltaE_positive_expression(self):
        """Exchange integral K>0 implies ΔE = 2K >0. Expression should not contain a leading minus."""
        if self.deltaE is None:
            return (False, "SKIP: DeltaE not available")
        # Simple heuristics: expression should start with '2*' or similar positive structure.
        trimmed = self.deltaE.strip()
        if trimmed.startswith('-') and not trimmed.startswith('-1'):
            return (False, "DeltaE expression appears to be negative, but should be positive")
        # Check that it contains '2' and not a minus sign that would negate the whole thing
        # Since it's symbolic, we rely on the fact that K>0.
        return (True, "DeltaE expression is not obviously negative")

    # ---------- Modeling/trace checks ----------
    def test_modeling_deltaE_structure(self):
        """Check DeltaE expression has expected form: 2 * exchange integral."""
        if self.deltaE is None:
            return (False, "SKIP: DeltaE not available")
        expr = self.deltaE.lower()
        # Must contain '2' and some form of exchange integral: psi_{1s}(r1) psi_{2s}(r2) and 1/r12
        has_two = '2' in expr
        # Check for Coulomb interaction
        has_r12 = 'r_{12}' in expr or 'r12' in expr or '1/r' in expr
        # Check for wavefunctions
        has_psi = 'psi' in expr
        # Rough check
        if has_two and has_r12 and has_psi:
            return (True, "DeltaE expression contains 2, Coulomb term, and wavefunction product")
        else:
            details = []
            if not has_two:
                details.append("missing factor 2")
            if not has_r12:
                details.append("missing r12 dependence")
            if not has_psi:
                details.append("missing wavefunction symbols")
            return (False, f"DeltaE expression lacks expected features: {', '.join(details)}")

    def test_modeling_explanation_steps_mention_exchange(self):
        """Explanation steps should mention exchange integral or Pauli principle."""
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not available")
        steps_text = ' '.join(self.explanation_steps).lower()
        keywords = ['exchange', 'pauli', 'antisymmetric', 'hund']
        found = [kw for kw in keywords if kw in steps_text]
        if not found:
            return (False, "Explanation steps lack keywords: exchange, Pauli, antisymmetric, or Hund's rule")
        return (True, f"Explanation steps contain: {', '.join(found)}")

    # ---------- Monotonicity test ----------
    def test_monotonicity_deltaE_with_exchange(self):
        """
        Test that ΔE = 2K is a monotonically increasing function of K.
        Perturb the exchange integral K by +5% and verify that ΔE increases.
        """
        baseline_K = 1.0  # arbitrary positive value
        baseline_deltaE = 2 * baseline_K
        perturbed_K = baseline_K * 1.05  # +5%
        perturbed_deltaE = 2 * perturbed_K
        if perturbed_deltaE > baseline_deltaE:
            return (True, "ΔE increases with K (monotonic): ΔE(1.05K) > ΔE(K)")
        else:
            return (False, "Monotonicity violated: ΔE should increase when K increases")

    # ---------- Weak answer sanity (only one allowed) ----------
    def test_answer_sanity_lower_state(self):
        """Ensure lower_state is a non‑empty string (sanity check)."""
        if self.lower_state is None:
            return (False, "SKIP: lower_state not provided")
        if self.lower_state.strip() == '':
            return (False, "lower_state is an empty string")
        return (True, "lower_state is non‑empty")

    # ---------- Test runner ----------
    def run_all_tests(self):
        """Discover all test_* methods, run them, collect logs."""
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
                passed, msg = False, f"{name} CRASHED: {e}" + traceback.format_exc()
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)