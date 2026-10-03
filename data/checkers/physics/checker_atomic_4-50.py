import math
import sympy as sp

class PhysicsChecker:
    """
    PhysicsChecker for neutral meson decay problem.
    Audits: beta, gamma, rest energy, and photon energy consistency.
    """

    def __init__(self, solution_instance):
        # Extract inputs from solution_instance safely
        # Try to get givens from .givens dict or direct attributes
        givens = getattr(solution_instance, 'givens', {})
        self.E_f = givens.get('E_f', None) or getattr(solution_instance, 'E_f', None)
        self.E_b = givens.get('E_b', None) or getattr(solution_instance, 'E_b', None)

        # Extract results
        results = getattr(solution_instance, 'results', {})
        self.beta = results.get('beta', None)
        self.gamma = results.get('gamma', None)
        self.E0 = results.get('E0', None)
        self.M_rest = results.get('M_rest', None)

        # Expected constants from problem (not defaults, only for internal checks)
        # We do NOT use them as fallback – only for comparison when givens missing.
        self._GIVEN_E_F = 96.0
        self._GIVEN_E_B = 48.0

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

    # ========== Interface tests ==========
    def test_interface_required_keys(self):
        """Check that all required result keys are present and not None."""
        required = ['beta', 'gamma', 'E0', 'M_rest']
        missing = [k for k in required if getattr(self, k, None) is None]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required keys present")

    # ========== Invariant tests (theorem-based) ==========
    def test_invariant_gamma_from_beta(self):
        """Check gamma = 1/sqrt(1-beta^2)."""
        if self.beta is None or self.gamma is None:
            return (False, "SKIP: beta or gamma missing")
        expected = 1.0 / math.sqrt(1 - self.beta**2)
        if not math.isclose(self.gamma, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"gamma = {self.gamma}, expected ~ {expected}")
        return (True, "gamma consistent with beta")

    def test_invariant_rest_energy_from_E0(self):
        """Check M_rest*c^2 = 2*E0."""
        if self.E0 is None or self.M_rest is None:
            return (False, "SKIP: E0 or M_rest missing")
        # M_rest is in MeV/c^2, E0 in MeV -> M_rest = 2*E0 / c^2? Actually E0 already in MeV,
        # c=1 units, so M_rest = 2*E0.
        expected_M = 2.0 * self.E0
        if not math.isclose(self.M_rest, expected_M, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"M_rest = {self.M_rest}, expected ~ {expected_M}")
        return (True, "M_rest consistent with 2*E0")

    def test_invariant_forward_backward_ratio(self):
        """Check E_f/E_b = (1+beta)/(1-beta) using external givens if available."""
        E_f = self.E_f
        E_b = self.E_b
        beta = self.beta
        if E_f is None or E_b is None:
            # Skip if external givens not provided – cannot check without them
            return (False, "SKIP: external E_f, E_b not available")
        if beta is None:
            return (False, "SKIP: beta missing")
        ratio_given = E_f / E_b
        ratio_expected = (1 + beta) / (1 - beta)
        if not math.isclose(ratio_given, ratio_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E_f/E_b = {ratio_given}, expected ~ {ratio_expected} from beta")
        return (True, "E_f/E_b consistent with beta")

    def test_invariant_E0_from_geometric_mean(self):
        """Check E0 = sqrt(E_f * E_b) using external givens."""
        E_f = self.E_f
        E_b = self.E_b
        E0 = self.E0
        if E_f is None or E_b is None:
            return (False, "SKIP: external E_f, E_b not available")
        if E0 is None:
            return (False, "SKIP: E0 missing")
        expected_E0 = math.sqrt(E_f * E_b)
        if not math.isclose(E0, expected_E0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E0 = {E0}, expected ~ {expected_E0}")
        return (True, "E0 consistent with sqrt(E_f*E_b)")

    # ========== Modeling/trace tests ==========
    def test_modeling_beta_from_energy_momentum(self):
        """Check beta = (E_f - E_b)/(E_f + E_b) from explanation step."""
        E_f = self.E_f
        E_b = self.E_b
        beta = self.beta
        if E_f is None or E_b is None:
            return (False, "SKIP: external E_f, E_b not available")
        if beta is None:
            return (False, "SKIP: beta missing")
        total_E = E_f + E_b
        total_p = E_f - E_b  # assuming forward = +, backward = -
        expected_beta = total_p / total_E
        if not math.isclose(beta, expected_beta, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"beta = {beta}, expected ~ {expected_beta}")
        return (True, "beta consistent with (E_f - E_b)/(E_f + E_b)")

    def test_modeling_lab_energy_sum(self):
        """Check that E_f + E_b equals gamma*E0*(1+beta) + gamma*E0*(1-beta) = 2*gamma*E0."""
        # This is a derived invariant from lorentz transformation.
        if any(x is None for x in [self.E0, self.gamma, self.beta]):
            return (False, "SKIP: missing E0, gamma, or beta")
        # Could also recompute from E_f, E_b if available.
        # But we can also just check internal consistency: total lab energy = 2*gamma*E0.
        total_lab_exp = 2 * self.gamma * self.E0
        # If external E_f, E_b available, compare.
        if self.E_f is not None and self.E_b is not None:
            total_ext = self.E_f + self.E_b
            if not math.isclose(total_ext, total_lab_exp, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"E_f+E_b={total_ext}, 2*gamma*E0={total_lab_exp}")
        # If external not available, just check positivity and finite.
        if not (math.isfinite(total_lab_exp) and total_lab_exp > 0):
            return (False, f"non-positive total lab energy: {total_lab_exp}")
        return (True, "total lab energy consistent")

    # ========== Monotonicity tests ==========
    def test_monotonicity_gamma_increases_with_beta(self):
        """Check that gamma increases as beta increases (local derivative)."""
        if self.beta is None or self.gamma is None:
            return (False, "SKIP: beta or gamma missing")
        # Perturb beta+
        beta_plus = self.beta * 1.01  # +1%
        if beta_plus >= 1.0:
            return (False, "SKIP: beta too close to 1 for perturbation")
        gamma_plus = 1.0 / math.sqrt(1 - beta_plus**2)
        if not (gamma_plus > self.gamma):
            return (False, "gamma does not increase with beta (perturbation)")
        return (True, "gamma monotonic with beta (local)")

    # ========== Weak answer sanity ==========
    def test_answer_sanity_basic_ranges(self):
        """Weak sanity: beta in (0,1), gamma>1, E0>0, M_rest>0, finite."""
        checks = []
        if self.beta is not None:
            if not (0 < self.beta < 1):
                checks.append(f"beta out of (0,1): {self.beta}")
        if self.gamma is not None:
            if not self.gamma > 1:
                checks.append(f"gamma not >1: {self.gamma}")
        if self.E0 is not None:
            if not self.E0 > 0:
                checks.append(f"E0 not positive: {self.E0}")
        if self.M_rest is not None:
            if not self.M_rest > 0:
                checks.append(f"M_rest not positive: {self.M_rest}")
        # Also check not inf
        for attr in ['beta', 'gamma', 'E0', 'M_rest']:
            val = getattr(self, attr, None)
            if val is not None and not math.isfinite(val):
                checks.append(f"{attr} non-finite: {val}")
        if checks:
            return (False, "; ".join(checks))
        return (True, "Basic ranges pass")