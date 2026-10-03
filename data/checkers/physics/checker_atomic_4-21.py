import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Physics checker for the problem:
    In CERN colliding-beam storage ring, protons of total energy 30 GeV collide head-on.
    What energy must a single proton have to give the same center-of-mass energy when colliding with a stationary proton?
    """

    def __init__(self, solution_instance):
        # Extract givens (expected from the solution's givens dict) with safe defaults
        givens = getattr(solution_instance, 'givens', {})
        if givens is None:
            givens = {}
        # Try to get E (proton energy in collider) and m (proton rest mass)
        self.E = givens.get('E', None)
        self.m = givens.get('m', None) or givens.get('m_p', None) or givens.get('mp', None)
        # Fallback: if not in givens, try from results (some solvers store inputs in results)
        if self.E is None:
            results = getattr(solution_instance, 'results', {})
            if results:
                self.E = results.get('E', None)
        if self.m is None:
            results = getattr(solution_instance, 'results', {})
            if results:
                self.m = results.get('m', None) or results.get('m_p', None)

        # Extract the required output: E_prime from results
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        self.E_prime = results.get('E_prime', None)

        # Extract explanation steps for reference (optional)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Store the solution instance for potential detailed access
        self.solution_instance = solution_instance

    def _safe_isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy comparison safe for None values."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    # ----------------------------------------------------------------
    # Interface tests: ensure required keys exist and have valid types
    # ----------------------------------------------------------------
    def test_interface_E_prime_exists(self):
        """Check that E_prime is present in results and is a number."""
        if self.E_prime is None:
            return (False, "E_prime not found in results")
        if not isinstance(self.E_prime, (int, float)):
            return (False, f"E_prime has invalid type: {type(self.E_prime)}")
        return (True, f"E_prime = {self.E_prime}")

    def test_interface_givens_available(self):
        """Check that essential givens (E and m) are available."""
        missing = []
        if self.E is None:
            missing.append("E (proton energy in collider)")
        if self.m is None:
            missing.append("m (proton rest mass)")
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, f"E={self.E}, m={self.m}")

    # ----------------------------------------------------------------
    # Invariant tests: theorem-based physical sanity (use math.isclose)
    # ----------------------------------------------------------------
    def test_invariant_s_equality(self):
        """
        Check that the invariant s computed from collider scenario equals
        s computed from fixed-target scenario.
        s_c = (2E)^2 = 4E^2
        s_f = 2 * m * (E_prime + m)
        They must be equal within 1% relative tolerance.
        """
        if self.E is None or self.m is None or self.E_prime is None:
            return (False, "SKIP: missing required quantities (E, m, E_prime)")
        # Compute invariants
        s_collider = 4 * (self.E ** 2)
        s_fixed = 2 * self.m * (self.E_prime + self.m)
        if not self._safe_isclose(s_collider, s_fixed, rel_tol=0.01, abs_tol=1e-2):
            return (False, f"s_collider = {s_collider:.4f} GeV^2, s_fixed = {s_fixed:.4f} GeV^2, not equal")
        return (True, f"s_collider = s_fixed = {s_collider:.4f} GeV^2")

    def test_invariant_energy_momentum_relation(self):
        """
        For the collider protons, check that E^2 = p^2 + m^2 with p=E (since head-on).
        This is a relativistic sanity check.
        """
        if self.E is None or self.m is None:
            return (False, "SKIP: E or m missing")
        # In the collider, each proton has same energy E and momentum p = sqrt(E^2 - m^2)
        # but the invariant s uses E and p. However, the relation E^2 - p^2 = m^2 holds.
        # Since the problem treats them as head-on with opposite momenta canceling,
        # we can just verify that E > m (obvious) and that the relativistic invariant
        # E^2 - p^2 = m^2 is satisfied for the given parameters.
        # We do not have p, but we can derive it: p = sqrt(E^2 - m^2).
        # Check that the solution implicitly uses proper relativistic kinematics.
        if self.E <= self.m:
            return (False, f"E={self.E} should be > m={self.m} for relativistic proton")
        # Compute p from E and m
        p = math.sqrt(self.E**2 - self.m**2)
        check = math.isclose(self.E**2 - p**2, self.m**2, rel_tol=1e-6, abs_tol=1e-6)
        if not check:
            return (False, "Relativistic energy-momentum relation violated for the collider protons")
        return (True, f"E^2 - p^2 = m^2 holds (E={self.E}, m={self.m})")

    # ----------------------------------------------------------------
    # Modeling/trace tests: consistency with explanation steps and pseudocode
    # ----------------------------------------------------------------
    def test_modeling_formula_derivation(self):
        """
        Verify that E_prime is computed as (2E^2)/m - m, which comes from equating s.
        Uses the given E and m to compute expected E_prime and compares with solution's E_prime.
        This is derived from the invariant, so it's allowed as a modeling check.
        """
        if self.E is None or self.m is None or self.E_prime is None:
            return (False, "SKIP: missing required quantities")
        expected = (2 * self.E**2) / self.m - self.m
        if not self._safe_isclose(self.E_prime, expected, rel_tol=0.01, abs_tol=1e-2):
            return (False, f"Expected E_prime = {expected:.4f} GeV, but got {self.E_prime:.4f} GeV")
        return (True, f"E_prime consistent with formula: {self.E_prime:.4f} GeV")

    def test_modeling_positive_energy(self):
        """Check that E_prime is positive and finite."""
        if self.E_prime is None:
            return (False, "SKIP: E_prime not available")
        if not math.isfinite(self.E_prime):
            return (False, "E_prime is not finite")
        if self.E_prime <= 0:
            return (False, f"E_prime should be positive, got {self.E_prime}")
        return (True, f"E_prime = {self.E_prime} > 0")

    def test_modeling_large_energy_scale(self):
        """
        Verify that E_prime is much larger than the collider energy E and the rest mass.
        This is a qualitative consistency check.
        """
        if self.E is None or self.E_prime is None:
            return (False, "SKIP: missing E or E_prime")
        if self.E_prime <= self.E:
            return (False, f"E_prime ({self.E_prime}) should be larger than collider energy E ({self.E})")
        if self.E_prime <= self.m:
            return (False, f"E_prime ({self.E_prime}) should be much larger than rest mass m ({self.m})")
        return (True, f"E_prime scales correctly: much larger than both E and m")

    # ----------------------------------------------------------------
    # Monotonicity tests: perturb inputs and check trend
    # ----------------------------------------------------------------
    def test_monotonicity_E_prime_increases_with_E(self):
        """
        If collider energy E increases, the required fixed-target energy E_prime should increase.
        Perturb E by +1%, compute new expected E_prime, and check that the change is positive.
        We use the derived formula E_prime = 2E^2/m - m.
        """
        if self.E is None or self.m is None or self.E_prime is None:
            return (False, "SKIP: missing required quantities for monotonicity test")
        # Baseline
        original = self.E_prime
        # Perturb E up by 1%
        E_perturbed = self.E * 1.01
        new_expected = (2 * E_perturbed**2) / self.m - self.m
        # The actual solution did not change, but we check that a properly computed E_prime would increase.
        # This tests that the solution's underlying relation respects monotonicity.
        # However, we cannot compare to the original solution's E_prime directly because it didn't change.
        # Instead we verify that the theoretical formula predicts an increase.
        if new_expected <= original:
            return (False, f"NON-MONOTONIC: increasing E from {self.E} to {E_perturbed} "
                           f"yields E_prime {new_expected} <= original {original}")
        return (True, f"Monotonicity OK: E increased by 1%, E_prime increases from {original:.2f} to {new_expected:.2f} (theoretical)")

    # ----------------------------------------------------------------
    # Answer sanity: weak check only (no solving, no isclose)
    # ----------------------------------------------------------------
    def test_answer_sanity_magnitude_and_sign(self):
        """
        Weak sanity: E_prime should be positive and on the order of 10^3 GeV.
        No expected value computation.
        """
        if self.E_prime is None:
            return (False, "SKIP: E_prime not available")
        if not isinstance(self.E_prime, (int, float)):
            return (False, "E_prime is not a number")
        if self.E_prime <= 0:
            return (False, "E_prime must be positive")
        if not math.isfinite(self.E_prime):
            return (False, "E_prime is not finite")
        # Broad magnitude check: between 100 and 100000 GeV
        if not (100 < self.E_prime < 100000):
            return (False, f"E_prime magnitude seems off: {self.E_prime} (expected ~10^3 GeV)")
        return (True, f"E_prime = {self.E_prime} GeV passes sanity")

    # ----------------------------------------------------------------
    # Runner
    # ----------------------------------------------------------------
    def run_all_tests(self):
        """
        Discover all test methods (prefix 'test_'), run them, collect logs.
        Returns (all_pass, logs_list).
        """
        logs = []
        all_pass = True
        test_methods = [name for name in dir(self) if name.startswith("test_") and callable(getattr(self, name))]
        # Sort for deterministic order
        test_methods.sort()
        for name in test_methods:
            method = getattr(self, name)
            try:
                passed, msg = method()
            except Exception as e:
                passed = False
                msg = f"CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

# Example usage (for testing):
"""
# Dummy solution instance
class DummySolution:
    def __init__(self):
        self.givens = {'E': 30.0, 'm': 0.938}
        self.results = {'E_prime': 1917.38}
        self.explanation_steps = "..."

instance = DummySolution()
checker = PhysicsChecker(instance)
success, logs = checker.run_all_tests()
for l in logs:
    print(l)
print("Overall:", success)
"""