import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Audits the student's solution for the Thevenin equivalent circuit problem.
    Checks consistency with Thévenin's theorem: epsilon0, r, and I_sc.
    """
    def __init__(self, solution_instance):
        # Extract results safely; default None for missing keys
        results = getattr(solution_instance, 'results', {})
        self.epsilon0 = results.get('epsilon0', None)
        self.r = results.get('r', None)
        self.I_sc = results.get('I_sc', None)

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that epsilon0, r, and I_sc are present and numeric."""
        if self.epsilon0 is None:
            return (False, "Missing key: epsilon0")
        if self.r is None:
            return (False, "Missing key: r")
        if self.I_sc is None:
            return (False, "Missing key: I_sc")

        if not isinstance(self.epsilon0, (int, float)):
            return (False, f"epsilon0 is not numeric: {type(self.epsilon0)}")
        if not isinstance(self.r, (int, float)):
            return (False, f"r is not numeric: {type(self.r)}")
        if not isinstance(self.I_sc, (int, float)):
            return (False, f"I_sc is not numeric: {type(self.I_sc)}")

        return (True, "All required keys present and numeric")

    # ------------------------------------------------------------------
    # Invariant / physical sanity checks
    # ------------------------------------------------------------------
    def test_invariant_epsilon0_positive(self):
        if self.epsilon0 is None:
            return (False, "SKIP: epsilon0 not available")
        if self.epsilon0 <= 0:
            return (False, f"epsilon0 must be positive, got {self.epsilon0}")
        return (True, "epsilon0 > 0")

    def test_invariant_r_positive(self):
        if self.r is None:
            return (False, "SKIP: r not available")
        if self.r <= 0:
            return (False, f"r must be positive, got {self.r}")
        return (True, "r > 0")

    def test_invariant_I_sc_positive(self):
        if self.I_sc is None:
            return (False, "SKIP: I_sc not available")
        if self.I_sc <= 0:
            return (False, f"I_sc must be positive, got {self.I_sc}")
        return (True, "I_sc > 0")

    def test_invariant_short_circuit_relation(self):
        """
        Check that I_sc ≈ ε₀ / r using math.isclose (fuzzy for SFT noise).
        """
        if self.epsilon0 is None or self.r is None or self.I_sc is None:
            return (False, "SKIP: one or more required values missing")
        if self.r == 0:
            return (False, "r is zero, cannot compute I_sc")

        expected = self.epsilon0 / self.r
        if math.isclose(self.I_sc, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"I_sc ≈ ε₀/r: {self.I_sc} ≈ {expected}")
        else:
            return (False, f"I_sc = {self.I_sc} does not match ε₀/r = {expected}")

    def test_invariant_open_circuit_emf(self):
        """
        Check that ε₀ is the open-circuit voltage.
        If the solution also stores V_oc, verify consistency.
        Otherwise, just ensure ε₀ is not negative (already checked).
        """
        # No direct V_oc in results; this test is weak but valid.
        if self.epsilon0 is None:
            return (False, "SKIP: epsilon0 not available")
        if self.epsilon0 < 0:
            return (False, "epsilon0 should be non-negative")
        return (True, "epsilon0 is non-negative (consistent with open-circuit voltage)")

    # ------------------------------------------------------------------
    # Modeling / trace checks
    # ------------------------------------------------------------------
    def test_modeling_from_Pseudocode_consistency(self):
        """
        Verify that r is derived correctly from slope if V_1 and I_1 are present.
        This test uses the pseudocode's intermediate data if available.
        """
        # We try to get V_1 and I_1 from the solution instance (they may not exist)
        v1 = getattr(self, 'V_1', None)
        i1 = getattr(self, 'I_1', None)
        # If not directly stored, try to get from results or attributes
        if v1 is None:
            v1 = getattr(self, 'results', {}).get('V_1', None)
        if i1 is None:
            i1 = getattr(self, 'results', {}).get('I_1', None)

        # If the data is present, we can cross-check r
        if (v1 is not None and i1 is not None and self.epsilon0 is not None
                and self.r is not None and i1 != 0):
            expected_r = (self.epsilon0 - v1) / i1
            if math.isclose(self.r, expected_r, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"r from V-I point matches: {self.r} ≈ {expected_r}")
            else:
                return (False, f"r = {self.r} does not match derived r = {expected_r} from V_1={v1}, I_1={i1}")

        # If data not available, skip
        return (False, "SKIP: V_1 or I_1 not available for modeling check")

    def test_modeling_thevenin_linearity(self):
        """
        Check that the V-I characteristic is linear: V = ε₀ - I r.
        Since we don't have data points, we check that ε₀ > 0, r > 0, and I_sc positive,
        which are necessary for linear passive model.
        Already covered, so this is a placeholder.
        """
        # This test is redundant but included for completeness.
        # We can check sign consistency.
        if self.epsilon0 is not None and self.r is not None and self.I_sc is not None:
            if self.epsilon0 > 0 and self.r > 0 and self.I_sc > 0:
                return (True, "Signs consistent with linear Thevenin model")
        return (False, "SKIP: missing values")

    # ------------------------------------------------------------------
    # Monotonicity checks
    # ------------------------------------------------------------------
    def test_monotonicity_I_sc_increases_with_epsilon0(self):
        """
        Verify that increasing ε₀ increases I_sc (positive derivative).
        Perturb ε₀ by +1% and compare I_sc.
        """
        if self.epsilon0 is None or self.r is None or self.I_sc is None:
            return (False, "SKIP: missing required values")
        if self.r == 0:
            return (False, "r is zero, cannot evaluate monotonicity")

        epsilon0_pert = self.epsilon0 * 1.01
        I_sc_pert = epsilon0_pert / self.r
        if I_sc_pert > self.I_sc:
            return (True, f"I_sc increases with ε₀: {self.I_sc} -> {I_sc_pert}")
        else:
            return (False, f"Monotonicity violated: epsilon0 increased but I_sc did not")

    def test_monotonicity_I_sc_decreases_with_r(self):
        """
        Verify that increasing r decreases I_sc (negative derivative).
        Perturb r by +1% and compare I_sc.
        """
        if self.epsilon0 is None or self.r is None or self.I_sc is None:
            return (False, "SKIP: missing required values")
        if self.r == 0:
            return (False, "r is zero, cannot evaluate monotonicity")

        r_pert = self.r * 1.01
        I_sc_pert = self.epsilon0 / r_pert
        if I_sc_pert < self.I_sc:
            return (True, f"I_sc decreases with r: {self.I_sc} -> {I_sc_pert}")
        else:
            return (False, f"Monotonicity violated: r increased but I_sc did not")

    # ------------------------------------------------------------------
    # Weak final-answer sanity (single test, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_reasonable_magnitudes(self):
        """
        Weak check: values should be plausible for typical Thevenin circuit.
        No exact matching, just broad magnitude order.
        """
        if self.epsilon0 is None or self.r is None or self.I_sc is None:
            return (False, "SKIP: missing values")

        # Allow very wide ranges; break only if unphysical
        if not (0.1 < self.epsilon0 < 100):
            return (False, f"epsilon0 = {self.epsilon0} seems unrealistic")
        if not (0.1 < self.r < 100):
            return (False, f"r = {self.r} seems unrealistic")
        if not (0.01 < self.I_sc < 10):
            return (False, f"I_sc = {self.I_sc} seems unrealistic")

        return (True, "Magnitudes within plausible ranges")

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)