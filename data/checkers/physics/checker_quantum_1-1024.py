import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Audits a solution for the infinite square well with a repulsive delta potential.
    Checks interface completeness, physical invariants, modeling consistency, and monotonicity.
    """

    def __init__(self, solution_instance):
        # Extract results safely, default None for missing keys.
        results = getattr(solution_instance, 'results', {})
        self.E_inf = results.get('E_inf', None)
        self.E_0 = results.get('E_0', None)

        # Also extract any given constants (optional, for completeness)
        givens = getattr(solution_instance, 'givens', {})
        self.L = givens.get('L', None)
        self.m = givens.get('m', None)
        self.hbar = givens.get('hbar', None)

    # ------------------------------------------------------------
    # Interface tests: check that required keys exist and are numeric
    # ------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that E_inf and E_0 are present and numeric."""
        if self.E_inf is None:
            return (False, "E_inf is missing from results")
        if self.E_0 is None:
            return (False, "E_0 is missing from results")
        if not isinstance(self.E_inf, (int, float)):
            return (False, f"E_inf type {type(self.E_inf)} is not numeric")
        if not isinstance(self.E_0, (int, float)):
            return (False, f"E_0 type {type(self.E_0)} is not numeric")
        return (True, "E_inf and E_0 present and numeric")

    def test_interface_E0_positive(self):
        """E_0 must be positive."""
        if self.E_0 is None:
            return (False, "SKIP: E_0 not available")
        if self.E_0 <= 0:
            return (False, f"E_0 = {self.E_0} is not positive")
        return (True, f"E_0 = {self.E_0} is positive")

    # ------------------------------------------------------------
    # Invariant tests: theorem-based physical sanity checks
    # ------------------------------------------------------------
    def test_invariant_energy_ratio(self):
        """
        In the limit λ → ∞, the ground state energy should become 4E_0.
        Use fuzzy matching (math.isclose) because rounding may occur.
        """
        if self.E_inf is None or self.E_0 is None:
            return (False, "SKIP: E_inf or E_0 not available")
        expected = 4.0 * self.E_0
        if math.isclose(self.E_inf, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_inf = {self.E_inf} ≈ 4*E_0 = {expected}")
        else:
            return (False, f"E_inf = {self.E_inf} differs significantly from 4*E_0 = {expected}")

    def test_invariant_energy_increase(self):
        """Repulsive delta increases the ground state energy."""
        if self.E_inf is None or self.E_0 is None:
            return (False, "SKIP: E_inf or E_0 not available")
        if self.E_inf >= self.E_0:
            return (True, f"E_inf = {self.E_inf} ≥ E_0 = {self.E_0}")
        else:
            return (False, f"Energy decreased: E_inf = {self.E_inf} < E_0 = {self.E_0}")

    # ------------------------------------------------------------
    # Modeling tests: check consistency with known physical constraints
    # ------------------------------------------------------------
    def test_modeling_ground_state_bounds(self):
        """
        The perturbed ground state energy lies between E_0 and 4E_0
        for any finite λ, and equals 4E_0 as λ→∞.
        """
        if self.E_inf is None or self.E_0 is None:
            return (False, "SKIP: E_inf or E_0 not available")
        if self.E_0 <= self.E_inf <= 4.0 * self.E_0:
            return (True, f"E_inf = {self.E_inf} lies within [E_0, 4E_0]")
        else:
            return (False, f"E_inf = {self.E_inf} outside expected range [E_0, 4E_0]")

    def test_modeling_node_condition(self):
        """
        As λ→∞, the wavefunction must have a node at x = L/2.
        The lowest such state is the n=2 state of the unperturbed well,
        giving E = 4E_0.
        """
        # We cannot directly check the wavefunction, but we can verify
        # that the energy matches the n=2 state if E_0 is known.
        if self.E_inf is None or self.E_0 is None:
            return (False, "SKIP: E_inf or E_0 not available")
        # The n=2 state energy is exactly 4E_0.
        expected = 4.0 * self.E_0
        if math.isclose(self.E_inf, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_inf = {self.E_inf} consistent with n=2 state")
        else:
            return (False, f"E_inf = {self.E_inf} not near 4E_0; node condition may not hold")

    # ------------------------------------------------------------
    # Monotonicity test: energy increases with repulsive strength λ
    # ------------------------------------------------------------
    def test_monotonicity_energy_increases_with_lambda(self):
        """
        For a repulsive delta, energy is a monotonically increasing function of λ.
        We cannot test the full function, but we can verify that E_inf (λ→∞)
        is not smaller than the unperturbed E_0.
        """
        if self.E_inf is None or self.E_0 is None:
            return (False, "SKIP: E_inf or E_0 not available")
        if self.E_inf >= self.E_0:
            return (True, "E_inf ≥ E_0, consistent with monotonic increase as λ→∞")
        else:
            return (False, "E_inf < E_0, violating monotonic increase")

    # ------------------------------------------------------------
    # Answer sanity test (weak, secondary)
    # ------------------------------------------------------------
    def test_answer_sanity_finite(self):
        """Check that E_inf is finite and not NaN."""
        if self.E_inf is None:
            return (False, "SKIP: E_inf not available")
        if math.isfinite(self.E_inf):
            return (True, f"E_inf = {self.E_inf} is finite")
        else:
            return (False, f"E_inf is not finite (value: {self.E_inf})")

    # ------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------
    def run_all_tests(self):
        """Discover all test_ methods, execute them, collect logs."""
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