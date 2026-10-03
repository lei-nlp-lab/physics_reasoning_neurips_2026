import math
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for the particle decay problem.
    Audits mass uncertainty (Γ) and isospin (I_X) based on decay widths and symmetry.
    """

    def __init__(self, solution_instance):
        # Givens: partial decay rates gamma1, gamma2 (may not be provided)
        self.gamma1 = getattr(solution_instance, 'gamma1', None)
        self.gamma2 = getattr(solution_instance, 'gamma2', None)

        # Results extracted from solution_instance.results dict
        results = getattr(solution_instance, 'results', {})
        self.dm = results.get('dm', None)          # computed mass uncertainty
        self.I_X = results.get('I_X', None)        # computed isospin

    # ------------------------------------------------------------------
    # Interface tests: required keys and types
    # ------------------------------------------------------------------
    def test_interface_dm_exists(self):
        """Result key 'dm' must be present and be a number."""
        if self.dm is None:
            return (False, "dm not found in results")
        if not isinstance(self.dm, (int, float)):
            return (False, f"dm must be numeric, got {type(self.dm).__name__}")
        return (True, "dm key present and numeric")

    def test_interface_I_X_exists(self):
        """Result key 'I_X' must be present and be an integer."""
        if self.I_X is None:
            return (False, "I_X not found in results")
        if not isinstance(self.I_X, int):
            return (False, f"I_X must be integer, got {type(self.I_X).__name__}")
        return (True, "I_X key present and integer")

    def test_interface_total_width_positive(self):
        """If gamma1 and gamma2 are given, total width must be > 0."""
        if self.gamma1 is None or self.gamma2 is None:
            return (False, "SKIP: gamma1 or gamma2 not available")
        if self.gamma1 + self.gamma2 <= 0:
            return (False, f"Total width γ1+γ2 = {self.gamma1+self.gamma2} not positive")
        return (True, "Total width is positive")

    # ------------------------------------------------------------------
    # Invariant / theorem-based tests
    # ------------------------------------------------------------------
    def test_invariant_width_relation(self):
        """
        Energy-time uncertainty: Δm = Γ (ℏ=1).
        Since Γ = γ1+γ2, we require dm ≈ γ1+γ2 within tolerance.
        """
        if self.gamma1 is None or self.gamma2 is None or self.dm is None:
            return (False, "SKIP: missing gamma1, gamma2, or dm")
        total_width = self.gamma1 + self.gamma2
        if math.isclose(self.dm, total_width, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"dm ≈ total width: {self.dm} ≈ {total_width}")
        else:
            return (False, f"dm={self.dm} not close to total width={total_width}")

    def test_invariant_isospin_possible_values(self):
        """
        Isospin of X must be one of {0,1,2} for a two-pion final state.
        (The problem forces I=2, but this test checks that I_X lies in the allowed set.)
        """
        if self.I_X is None:
            return (False, "SKIP: I_X not available")
        if self.I_X in (0, 1, 2):
            return (True, f"I_X={self.I_X} is a possible isospin for a two-pion state")
        else:
            return (False, f"I_X={self.I_X} is not allowed (must be 0,1,2)")

    def test_invariant_symmetry_constraint(self):
        """
        For π⁺π⁺ final state (identical bosons), the wavefunction must be symmetric.
        Only I=0 and I=2 have symmetric wavefunctions. Combined with I_z=+2, only I=2 is possible.
        Therefore I_X must be 2.
        """
        if self.I_X is None:
            return (False, "SKIP: I_X not available")
        # Symmetry argument: I=1 is antisymmetric, incompatible with identical bosons.
        # I=0 has I_z=0, incompatible with final-state I_z=+2.
        # Hence the only consistent value is 2.
        if self.I_X == 2:
            return (True, "I_X=2 is the only symmetric, I_z-compatible value")
        else:
            return (False, f"I_X={self.I_X} violates symmetry or I_z conservation")

    # ------------------------------------------------------------------
    # Modeling / trace checks
    # ------------------------------------------------------------------
    def test_modeling_width_derivation(self):
        """
        Check that dm = gamma1+gamma2 (ℏ=1) using fuzzy comparison.
        This verifies the modeling of mass uncertainty as total decay width.
        """
        if self.gamma1 is None or self.gamma2 is None or self.dm is None:
            return (False, "SKIP: missing data for width modeling")
        # Re-derive expected dm from scratch: dm = gamma1+gamma2
        expected_dm = self.gamma1 + self.gamma2
        if math.isclose(self.dm, expected_dm, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Modeling correct: dm = {expected_dm}")
        else:
            return (False, f"dm={self.dm} does not match expected {expected_dm}")

    def test_modeling_isospin_symmetry_chain(self):
        """
        Verify that the final state π⁺π⁺ forces I=2 through:
        - I_z = +2 → I ≥ 2
        - Symmetric wavefunction → I=0 or I=2
        - Intersection → I=2
        """
        if self.I_X is None:
            return (False, "SKIP: I_X not available")
        # The reasoning is qualitative: we can only check that the result is 2.
        if self.I_X == 2:
            return (True, "Isospin modeling yields I_X=2")
        else:
            return (False, f"I_X={self.I_X} not 2")

    # ------------------------------------------------------------------
    # Monotonicity tests: explicitly perturb one input and compare
    # perturbed output (computed from physics) to baseline output.
    # ------------------------------------------------------------------
    def test_monotonicity_width_increases_with_gamma1(self):
        """
        Verify that increasing gamma1 leads to an increase in total width
        (and hence dm). Since dm = γ1+γ2, we test that the relation is monotonic.
        """
        if self.gamma1 is None or self.gamma2 is None:
            return (False, "SKIP: gamma1 or gamma2 missing")
        # baseline total width
        baseline = self.gamma1 + self.gamma2
        # perturb gamma1 by +10%
        gamma1_pert = self.gamma1 * 1.1
        perturbed_output = gamma1_pert + self.gamma2
        if perturbed_output > baseline:
            return (True, f"Increasing γ1 from {self.gamma1} to {gamma1_pert} raises total width from {baseline} to {perturbed_output}")
        else:
            return (False, f"Monotonicity violated: perturbed output {perturbed_output} <= baseline {baseline}")

    def test_monotonicity_width_increases_with_gamma2(self):
        """
        Verify that increasing gamma2 leads to an increase in total width.
        """
        if self.gamma1 is None or self.gamma2 is None:
            return (False, "SKIP: gamma1 or gamma2 missing")
        baseline = self.gamma1 + self.gamma2
        gamma2_pert = self.gamma2 * 1.1
        perturbed_output = self.gamma1 + gamma2_pert
        if perturbed_output > baseline:
            return (True, f"Increasing γ2 from {self.gamma2} to {gamma2_pert} raises total width from {baseline} to {perturbed_output}")
        else:
            return (False, f"Monotonicity violated: perturbed output {perturbed_output} <= baseline {baseline}")

    # ------------------------------------------------------------------
    # Answer sanity (weak, must not re-derive or use math.isclose)
    # ------------------------------------------------------------------
    def test_answer_sanity_finite_and_positive(self):
        """Check that dm is finite and positive, and I_X is the integer 2."""
        if self.dm is None or self.I_X is None:
            return (False, "SKIP: missing dm or I_X")
        ok = True
        msgs = []
        if not math.isfinite(self.dm) or self.dm <= 0:
            ok = False
            msgs.append(f"dm={self.dm} not finite positive")
        if not isinstance(self.I_X, int) or self.I_X != 2:
            ok = False
            msgs.append(f"I_X={self.I_X} is not 2")
        if ok:
            return (True, "dm finite positive, I_X=2")
        else:
            return (False, "; ".join(msgs))

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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)