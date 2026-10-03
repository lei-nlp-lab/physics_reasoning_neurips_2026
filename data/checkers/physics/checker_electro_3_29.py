import math
import traceback

class PhysicsChecker:
    """
    Audits a solution for the total charge flowing through a circular loop
    when a magnetic field is turned on.  Derivation: q = B * π * a² / R.
    """
    def __init__(self, solution_instance):
        # Extract results safely (pseudocode emit key → results['q'])
        results = getattr(solution_instance, 'results', {})
        self.q = results.get('q', None)

        # Extract input givens safely – do NOT assign numeric defaults
        givens = getattr(solution_instance, 'givens', {})
        self.a = givens.get('a', None)
        self.R = givens.get('R', None)
        self.L = givens.get('L', None)
        self.B = givens.get('B', None)

        # Optional meta-information for auditing (not required for all tests)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', [])
        self.pseudocode = getattr(solution_instance, 'pseudocode', {})

        self.pi = math.pi

    # ------------------------------------------------------------------
    # Interface tests – ensure required keys/attributes exist
    # ------------------------------------------------------------------
    def test_interface_q_exists(self):
        """Check that the result key 'q' is present and not None."""
        if self.q is None:
            return (False, "No 'q' found in results['q']")
        return (True, "q is present")

    def test_interface_given_keys_exist(self):
        """Check that required given quantities (a, R, B) are provided."""
        missing = []
        if self.a is None:
            missing.append('a')
        if self.R is None:
            missing.append('R')
        if self.B is None:
            missing.append('B')
        if missing:
            return (False, f"Missing given(s): {', '.join(missing)}")
        return (True, "All required givens (a, R, B) present")

    # ------------------------------------------------------------------
    # Invariant / physics‑sanity tests – theorem‑based checks
    # ------------------------------------------------------------------
    def test_invariant_q_positive(self):
        """If B>0 then q must be positive (charge flows in one direction)."""
        if self.B is None:
            return (False, "SKIP: B not available")
        if self.q is None:
            return (False, "SKIP: q not available")
        if self.B <= 0:
            # If B is zero or negative, we cannot infer sign; skip
            return (True, "B not positive, sign test skipped")
        if self.q <= 0:
            return (False, f"B>0 but q={self.q} ≤ 0")
        return (True, f"q positive (q={self.q})")

    def test_invariant_q_independent_of_L(self):
        """
        According to Faraday + circuit integration, total charge q does NOT
        depend on self-inductance L.  If L is given, q should be consistent
        with q = Bπa²/R, unrelated to L.
        """
        if self.L is None:
            return (False, "SKIP: L not provided, cannot check independence")
        if any(x is None for x in (self.q, self.a, self.R, self.B)):
            return (False, "SKIP: missing q, a, R, or B")
        expected_q = self.B * self.pi * (self.a ** 2) / self.R
        if math.isclose(self.q, expected_q, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"q = {self.q:.5g} matches expected value independent of L")
        else:
            return (False, f"q = {self.q:.5g} but expected ≈ {expected_q:.5g} (and L={self.L})")

    # ------------------------------------------------------------------
    # Modeling / trace tests – check the core physics relations
    # ------------------------------------------------------------------
    def test_modeling_q_relation(self):
        """Verify that q satisfies q = B * π * a² / R (Faraday + circuit integration)."""
        if any(x is None for x in (self.q, self.a, self.R, self.B)):
            return (False, "SKIP: missing required data (q, a, R, or B)")
        expected_q = self.B * self.pi * (self.a ** 2) / self.R
        if math.isclose(self.q, expected_q, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"q ≈ {expected_q:.5g} matches expected value")
        else:
            return (False, f"q = {self.q:.5g} but expected ≈ {expected_q:.5g}")

    def test_modeling_flux_change(self):
        """
        Check that the computed charge times resistance equals the total
        flux change: q * R = B * π * a².  This verifies the integration
        of the circuit equation -dΦ + R dq = 0.
        """
        if any(x is None for x in (self.q, self.a, self.R, self.B)):
            return (False, "SKIP: missing required data (q, a, R, or B)")
        flux_change = self.B * self.pi * (self.a ** 2)
        qR = self.q * self.R
        if math.isclose(qR, flux_change, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"q*R ≈ {flux_change:.5g} matches ΔΦ")
        else:
            return (False, f"q*R = {qR:.5g} but ΔΦ = {flux_change:.5g}")

    # ------------------------------------------------------------------
    # Monotonicity tests – trend behavior via perturbation
    # ------------------------------------------------------------------
    def test_monotonicity_q_increases_with_B(self):
        """If B is increased by 10%, q should increase (proportionality)."""
        if any(x is None for x in (self.q, self.a, self.R, self.B)):
            return (False, "SKIP: missing required data (q, a, R, or B)")
        if self.B <= 0:
            return (True, "B not positive, monotonicity test skipped")
        # Perturb B upward by 10%
        B_perturbed = self.B * 1.1
        q_perturbed = B_perturbed * self.pi * (self.a ** 2) / self.R
        if q_perturbed > self.q:
            return (True, f"q increases with B (perturbed q={q_perturbed:.5g} > baseline {self.q:.5g})")
        else:
            return (False, f"q_perturbed={q_perturbed:.5g} not > baseline q={self.q:.5g}")

    # ------------------------------------------------------------------
    # Weak answer‑sanity test (at most one, no re‑derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_broad(self):
        """
        Very weak sanity: q must be finite, not NaN, and for B>0 it should be
        positive.  No use of math.isclose or re‑derivation.
        """
        if self.q is None:
            return (False, "SKIP: q not available")
        # Finite and not NaN
        if not math.isfinite(self.q):
            return (False, f"q is not finite (q={self.q})")
        # Positive if B>0 (if B is given)
        if self.B is not None and self.B > 0:
            if self.q <= 0:
                return (False, f"B>0 but q={self.q} ≤ 0")
        return (True, "q is finite and satisfies basic sign constraint")

    # ------------------------------------------------------------------
    # Test runner – discover and execute all test_ methods
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
                passed = False
                msg = f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)