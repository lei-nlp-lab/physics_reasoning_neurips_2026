import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for the nuclear fission problem:
    Express Q in terms of binding energy per nucleon epsilon(A) and mass number A,
    then estimate Q for symmetric fission of A0=240.
    """

    def __init__(self, solution_instance):
        # Safely extract givens and results; use None for missing values.
        # Expected keys: results['Q'], results['A_half']
        # Givens might include A0, eps_240, eps_120 (from pseudocode or explanation)
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.givens = getattr(solution_instance, 'givens', {}) or {}

        # Results
        self.Q = self.results.get('Q', None)
        self.A_half = self.results.get('A_half', None)

        # Givens or derived variables (may be in results or givens)
        self.A0 = self.givens.get('A0', None) or self.results.get('A0', None)
        self.eps_120 = self.givens.get('eps_120', None) or self.results.get('eps_120', None)
        self.eps_240 = self.givens.get('eps_240', None) or self.results.get('eps_240', None)

        # If A_half is missing but A0 present, we can compute it
        if self.A_half is None and self.A0 is not None:
            self.A_half = self.A0 / 2.0

    # ---------- Interface tests ----------
    def test_interface_keys_exists(self):
        """Check that required result keys are present."""
        if 'Q' not in self.results:
            return (False, "Missing key 'Q' in results")
        if 'A_half' not in self.results:
            return (False, "Missing key 'A_half' in results")
        return (True, "Required keys 'Q' and 'A_half' present")

    def test_interface_Q_type(self):
        """Check Q is numeric (int or float)."""
        if self.Q is None:
            return (False, "Q is None")
        if not isinstance(self.Q, (int, float)):
            return (False, f"Q has invalid type {type(self.Q).__name__}")
        return (True, f"Q is numeric: {self.Q}")

    def test_interface_A_half_type(self):
        """Check A_half is numeric (int or float)."""
        if self.A_half is None:
            return (False, "A_half is None")
        if not isinstance(self.A_half, (int, float)):
            return (False, f"A_half has invalid type {type(self.A_half).__name__}")
        return (True, f"A_half is numeric: {self.A_half}")

    # ---------- Invariant tests (physical constraints) ----------
    def test_invariant_symmetric_fission_A_half(self):
        """Check that A_half equals A0/2 for symmetric fission."""
        if self.A0 is None:
            return (False, "SKIP: A0 not available")
        if self.A_half is None:
            return (False, "SKIP: A_half not available")
        # Use exact equality because A_half is defined as A0/2
        expected = self.A0 / 2.0
        if self.A_half != expected:
            return (False, f"A_half = {self.A_half}, expected {expected}")
        return (True, f"A_half = {self.A_half} equals A0/2")

    def test_invariant_Q_positive(self):
        """Check that Q > 0 (exothermic reaction)."""
        if self.Q is None:
            return (False, "SKIP: Q not available")
        if self.Q <= 0:
            return (False, f"Q = {self.Q} is not positive")
        return (True, f"Q = {self.Q} > 0")

    def test_invariant_A_half_positive(self):
        """Check that A_half > 0."""
        if self.A_half is None:
            return (False, "SKIP: A_half not available")
        if self.A_half <= 0:
            return (False, f"A_half = {self.A_half} is not positive")
        return (True, f"A_half = {self.A_half} > 0")

    # ---------- Modeling / trace tests (derivation consistency) ----------
    def test_modeling_Q_formula(self):
        """
        Check that Q matches A0*(eps_120 - eps_240) from the derivation.
        Use math.isclose because epsilon values are approximate.
        """
        if None in (self.Q, self.A0, self.eps_120, self.eps_240):
            return (False, "SKIP: missing data for Q formula check")
        Q_expected = self.A0 * (self.eps_120 - self.eps_240)
        if not math.isclose(self.Q, Q_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Q = {self.Q} deviates from expected {Q_expected}")
        return (True, f"Q = {self.Q} matches A0*(eps_120-eps_240) = {Q_expected}")

    def test_modeling_A_half_from_A0(self):
        """Optional: check A_half derived from A0 if not directly given."""
        if self.A0 is None or self.A_half is None:
            return (False, "SKIP: missing data for A_half derivation")
        expected = self.A0 / 2.0
        if not math.isclose(self.A_half, expected, rel_tol=1e-2):
            return (False, f"A_half = {self.A_half}, expected {expected} from A0")
        return (True, f"A_half = {self.A_half} consistent with A0/2")

    # ---------- Monotonicity tests (trend checks via perturbation) ----------
    def test_monotonicity_eps_120_increases_Q(self):
        """
        Increase eps_120 slightly and check that Q increases.
        Perturbation: derivative dQ/d(eps_120) = A0 > 0.
        """
        if None in (self.Q, self.A0, self.eps_120, self.eps_240):
            return (False, "SKIP: missing data for monotonicity test")
        # Simulate a small increase in eps_120
        delta = 0.5  # small compared to typical values ~8.5
        Q_perturbed = self.A0 * ((self.eps_120 + delta) - self.eps_240)
        if Q_perturbed <= self.Q:
            return (False, f"Q does not increase when eps_120 increases: baseline Q={self.Q}, perturbed Q={Q_perturbed}")
        return (True, f"Increasing eps_120 increases Q (baseline {self.Q} -> {Q_perturbed})")

    def test_monotonicity_eps_240_decreases_Q(self):
        """
        Increase eps_240 slightly and check that Q decreases.
        Derivative dQ/d(eps_240) = -A0 < 0.
        """
        if None in (self.Q, self.A0, self.eps_120, self.eps_240):
            return (False, "SKIP: missing data for monotonicity test")
        delta = 0.5
        Q_perturbed = self.A0 * (self.eps_120 - (self.eps_240 + delta))
        if Q_perturbed >= self.Q:
            return (False, f"Q does not decrease when eps_240 increases: baseline Q={self.Q}, perturbed Q={Q_perturbed}")
        return (True, f"Increasing eps_240 decreases Q (baseline {self.Q} -> {Q_perturbed})")

    # ---------- Answer sanity (weak, broad constraints only) ----------
    def test_answer_sanity_Q_range(self):
        """
        Weak sanity: Q should be a reasonable energy for fission of a heavy nucleus.
        Typical values are tens to hundreds of MeV.
        """
        if self.Q is None:
            return (False, "SKIP: Q not available")
        # Do not use math.isclose here; just a broad magnitude check.
        if not (50 < self.Q < 500):  # MeV, arbitrary range
            return (False, f"Q = {self.Q} is outside expected feasible range (50-500 MeV)")
        return (True, f"Q = {self.Q} is within reasonable range")

    # ---------- Run all tests ----------
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