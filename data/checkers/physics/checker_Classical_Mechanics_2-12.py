import math
import sympy

class PhysicsChecker:
    """
    Audits a solution instance for the problem:
    "A particle moves in a circular orbit in a force field F(r) = -k/r^2.
     Show that if k suddenly decreases to half its original value,
     the particle's orbit becomes parabolic."

    The checker verifies:
    - Interface: expected result keys exist.
    - Invariants: final total energy is zero, eccentricity is 1 (within tolerances).
    - Modeling quality: symbolic derivation confirms E_f=0,
      numerical derivation matches if givens are present, monotonicity holds.
    - Weak sanity: epsilon_prime is finite and positive.
    """

    def __init__(self, solution_instance):
        # Safely extract results and givens
        if solution_instance is None:
            self.results = {}
            self.m = None
            self.k = None
            self.r0 = None
        else:
            self.results = getattr(solution_instance, 'results', {})
            # Attempt to obtain givens from solution_instance.givens or as attributes
            self.m = None
            self.k = None
            self.r0 = None
            if hasattr(solution_instance, 'givens'):
                givens = solution_instance.givens
                self.m = givens.get('m', None)
                self.k = givens.get('k', None)
                self.r0 = givens.get('r0', None)
            else:
                self.m = getattr(solution_instance, 'm', None)
                self.k = getattr(solution_instance, 'k', None)
                self.r0 = getattr(solution_instance, 'r0', None)

        # Required result keys (from pseudocode [emit] contract)
        self.E_f = self.results.get('E_f', None)
        self.epsilon_prime = self.results.get('epsilon_prime', None)

    # ------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------
    def test_interface_E_f_exists(self):
        if self.E_f is None:
            return (False, "E_f not found in results")
        return (True, f"E_f = {self.E_f}")

    def test_interface_epsilon_prime_exists(self):
        if self.epsilon_prime is None:
            return (False, "epsilon_prime not found in results")
        return (True, f"epsilon_prime = {self.epsilon_prime}")

    # ------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------
    def test_invariant_E_f_zero(self):
        if self.E_f is None:
            return (False, "SKIP: E_f not available")
        tol = 1e-2
        if math.isclose(self.E_f, 0.0, rel_tol=tol, abs_tol=tol):
            return (True, f"E_f = {self.E_f} is close to 0")
        else:
            return (False, f"E_f = {self.E_f} is not close to 0 (tolerance {tol})")

    def test_invariant_epsilon_prime_one(self):
        if self.epsilon_prime is None:
            return (False, "SKIP: epsilon_prime not available")
        tol = 1e-2
        if math.isclose(self.epsilon_prime, 1.0, rel_tol=tol, abs_tol=tol):
            return (True, f"epsilon_prime = {self.epsilon_prime} is close to 1")
        else:
            return (False, f"epsilon_prime = {self.epsilon_prime} is not close to 1")

    # ------------------------------------------------------------
    # Modeling / trace tests (quality of physics reasoning)
    # ------------------------------------------------------------
    def test_modeling_symbolic_derivation(self):
        """
        Symbolically verify that E_f = 0 given the circular orbit condition.
        This test does not depend on the solution's numerical values.
        """
        m, k, r0 = sympy.symbols('m k r0', positive=True)
        v0_sq = k / (m * r0)                     # circular orbit condition
        E_f_sym = (sympy.Rational(1, 2) * m * v0_sq - (k / 2) / r0).simplify()
        if sympy.simplify(E_f_sym) == 0:
            return (True, "Symbolic derivation confirms E_f = 0")
        else:
            return (False, f"Symbolic derivation gave {E_f_sym}")

    def test_modeling_energy_derivation_numerical(self):
        """
        If givens are available, recompute E_f from the derived formula
        and compare with the solution's reported value.
        """
        if None in (self.m, self.k, self.r0):
            return (False, "SKIP: missing givens m, k, r0")
        if self.E_f is None:
            return (False, "SKIP: E_f not available")
        v0_sq = self.k / (self.m * self.r0)
        expected_E_f = 0.5 * self.m * v0_sq - (self.k / 2) / self.r0
        tol = 1e-2
        if math.isclose(self.E_f, expected_E_f, rel_tol=tol, abs_tol=tol):
            return (True, f"E_f = {self.E_f} matches derived {expected_E_f}")
        else:
            return (False, f"E_f = {self.E_f} != derived {expected_E_f} (tolerance {tol})")

    def test_modeling_eccentricity_from_energy(self):
        """
        For an inverse-square force, E=0 implies eccentricity=1.
        Verify that if E_f is close to 0, epsilon_prime is close to 1.
        """
        if self.E_f is None or self.epsilon_prime is None:
            return (False, "SKIP: E_f or epsilon_prime not available")
        tol = 1e-2
        if math.isclose(self.E_f, 0.0, rel_tol=tol, abs_tol=tol):
            if math.isclose(self.epsilon_prime, 1.0, rel_tol=tol, abs_tol=tol):
                return (True, "E_f zero and epsilon_prime one are consistent")
            else:
                return (False, f"E_f≈0 but epsilon_prime={self.epsilon_prime} not ≈1")
        else:
            return (False, "E_f is not zero; eccentricity consistency check skipped")

    def test_monotonicity_E_f_decreases_with_increasing_k_new(self):
        """
        Monotonicity check: if the new force constant k_new is increased
        above half the original value, the final energy E_f should decrease
        (become more negative). Perturb k_new by +1% from the half value,
        recompute E_f using the unchanged v0, and compare with baseline E_f=0.
        """
        # Need givens to compute v0 and baseline
        if None in (self.m, self.k, self.r0):
            return (False, "SKIP: missing givens m, k, r0")
        if self.E_f is None:
            return (False, "SKIP: E_f not available")
        # Baseline: k_new = k/2
        v0_sq = self.k / (self.m * self.r0)
        baseline_E_f = 0.5 * self.m * v0_sq - (self.k / 2) / self.r0  # should be 0
        # Perturb: increase k_new by 1% (relative to half)
        k_new_pert = (self.k / 2) * 1.01   # +1%
        E_f_pert = 0.5 * self.m * v0_sq - k_new_pert / self.r0
        # Since k_new_pert > k/2, E_f_pert should be < baseline_E_f
        if E_f_pert < baseline_E_f - 1e-12:
            return (True, f"E_f decreased from baseline {baseline_E_f} to {E_f_pert} when k_new increased by 1%")
        else:
            return (False, f"Monotonicity failed: baseline_E_f={baseline_E_f}, perturbed_E_f={E_f_pert}")

    # ------------------------------------------------------------
    # Weak answer sanity test (only one, secondary)
    # ------------------------------------------------------------
    def test_answer_sanity_epsilon_finite_positive(self):
        if self.epsilon_prime is None:
            return (False, "epsilon_prime not available")
        if math.isfinite(self.epsilon_prime) and self.epsilon_prime > 0:
            return (True, f"epsilon_prime = {self.epsilon_prime} is finite and positive")
        else:
            return (False, f"epsilon_prime invalid: {self.epsilon_prime}")

    # ------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------
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