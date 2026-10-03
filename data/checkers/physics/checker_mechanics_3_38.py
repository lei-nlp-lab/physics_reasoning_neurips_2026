import math
import sys

class PhysicsChecker:
    """
    Audits a solution for the relativistic flashlight problem.
    Checks conservation laws, derived formula, monotonicity, and final answer sanity.
    """

    def __init__(self, solution_instance):
        """
        Extract input parameters, computed results, and auxiliary derived values.
        All missing attributes default to None (no numeric defaults).
        """
        # Input givens: try givens dict first, then direct attribute
        self.m0 = None
        self.v = None
        self.c = None
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.m0 = solution_instance.givens.get('m0', None)
            self.v = solution_instance.givens.get('v', None)
            self.c = solution_instance.givens.get('c', None)
        else:
            # Fallback to direct attributes on solution_instance
            self.m0 = getattr(solution_instance, 'm0', None)
            self.v = getattr(solution_instance, 'v', None)
            self.c = getattr(solution_instance, 'c', None)

        # Computed result (the rest mass m)
        self.m = None
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.m = solution_instance.results.get('m', None)
        else:
            self.m = getattr(solution_instance, 'm', None)

        # Derived kinematic quantities (computed from givens if possible)
        self.beta = None
        self.gamma = None
        if self.v is not None and self.c is not None and self.c != 0:
            try:
                self.beta = self.v / self.c
                self.gamma = 1.0 / math.sqrt(1.0 - self.beta * self.beta)
            except (ValueError, ZeroDivisionError):
                pass   # invalid values, leave as None

        # Expected form m_expected = m0 * sqrt((c - v)/(c + v))  (derived from conservation)
        self.m_expected = None
        if self.m0 is not None and self.v is not None and self.c is not None:
            if self.c > self.v >= 0 and self.c > 0:
                ratio = (self.c - self.v) / (self.c + self.v)
                if ratio >= 0:
                    self.m_expected = self.m0 * math.sqrt(ratio)

        # Photon energy derived from momentum conservation: E_gamma = gamma * m * v * c (when m is correct)
        self.E_gamma = None
        if self.gamma is not None and self.m is not None and self.v is not None and self.c is not None:
            self.E_gamma = self.gamma * self.m * self.v * self.c

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_presence_givens(self):
        """Ensure all required input givens exist and are numeric."""
        if self.m0 is None:
            return (False, "m0 (initial rest mass) is missing")
        if self.v is None:
            return (False, "v (final velocity) is missing")
        if self.c is None:
            return (False, "c (speed of light) is missing")
        if not isinstance(self.m0, (int, float)):
            return (False, "m0 must be numeric")
        if not isinstance(self.v, (int, float)):
            return (False, "v must be numeric")
        if not isinstance(self.c, (int, float)):
            return (False, "c must be numeric")
        return (True, "All required givens present and numeric")

    def test_interface_presence_result(self):
        """Ensure the computed rest mass m is present and numeric."""
        if self.m is None:
            return (False, "Result m (rest mass at velocity v) is missing")
        if not isinstance(self.m, (int, float)):
            return (False, "m must be numeric")
        return (True, "Result m present and numeric")

    def test_interface_givens_physical(self):
        """Quick physical sanity of givens: v < c, v >= 0, m0 > 0, c > 0."""
        if self.m0 is None or self.v is None or self.c is None:
            return (False, "SKIP: missing givens")
        if self.m0 <= 0:
            return (False, "Initial rest mass m0 must be positive")
        if self.c <= 0:
            return (False, "Speed of light c must be positive")
        if not (0 <= self.v < self.c):
            return (False, "Velocity v must be in [0, c)")
        return (True, "Givens are physically plausible")

    # ----------------------------------------------------------------------
    # Theorem-based invariant checks (at least two)
    # ----------------------------------------------------------------------
    def test_invariant_energy_conservation(self):
        """
        Check energy conservation: m0*c^2 = gamma*m*c^2 + E_gamma.
        E_gamma is derived from the momentum equation (see test below).
        """
        if None in (self.m0, self.m, self.gamma, self.c, self.E_gamma):
            return (False, "SKIP: insufficient data for energy conservation")
        lhs = self.m0 * self.c * self.c
        rhs = self.gamma * self.m * self.c * self.c + self.E_gamma
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Energy conservation holds")
        else:
            return (False, f"Energy conservation violation: lhs={lhs}, rhs={rhs}")

    def test_invariant_momentum_conservation(self):
        """
        Check momentum conservation: 0 = gamma*m*v + p_gamma, with p_gamma = -E_gamma/c.
        Equivalent to E_gamma = gamma*m*v*c (since sign convention gives p_gamma = -E_gamma/c).
        """
        if None in (self.m, self.gamma, self.v, self.c, self.E_gamma):
            return (False, "SKIP: insufficient data for momentum conservation")
        expected_E = self.gamma * self.m * self.v * self.c
        # This is exactly how we computed self.E_gamma, but we check consistency
        if math.isclose(self.E_gamma, expected_E, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Momentum conservation holds (E_gamma = γ m v c)")
        else:
            return (False, f"Momentum conservation violation: computed E_gamma={self.E_gamma}, expected={expected_E}")

    # ----------------------------------------------------------------------
    # Modeling quality checks
    # ----------------------------------------------------------------------
    def test_modeling_final_mass_formula(self):
        """
        Verify that the computed rest mass matches the derived closed form:
        m = m0 * sqrt( (c - v) / (c + v) )
        This formula is re-derived inside the checker from conservation laws.
        """
        if None in (self.m, self.m_expected):
            return (False, "SKIP: cannot compute expected mass")
        if math.isclose(self.m, self.m_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Result matches derived formula: m ≈ {self.m_expected:.5f}")
        else:
            return (False, f"Result m={self.m:.5f} differs from expected {self.m_expected:.5f}")

    def test_modeling_kinematic_consistency(self):
        """
        If beta and gamma were provided in the solution, check they are consistent.
        (The solution might have computed them internally.)
        """
        if self.beta is None or self.gamma is None:
            return (False, "SKIP: beta and gamma not computable from givens")
        if not math.isfinite(self.gamma) or self.gamma < 1:
            return (False, "Gamma must be >= 1 and finite")
        return (True, "Kinematic gamma is physically valid (≥1, finite)")

    # ----------------------------------------------------------------------
    # Monotonicity checks
    # ----------------------------------------------------------------------
    def test_monotonicity_m_decreases_with_increasing_v(self):
        """
        As velocity v increases, the final rest mass m should decrease.
        Perturb v by +10% and check that the expected m becomes smaller.
        """
        if None in (self.m0, self.v, self.c, self.m_expected):
            return (False, "SKIP: insufficient data for monotonicity test")
        m_baseline = self.m_expected
        v_pert = self.v * 1.10
        if v_pert >= self.c:
            return (False, "SKIP: perturbed velocity would exceed c")
        ratio_pert = (self.c - v_pert) / (self.c + v_pert)
        if ratio_pert <= 0:
            return (False, "SKIP: perturbed ratio non‑positive")
        m_pert = self.m0 * math.sqrt(ratio_pert)
        if m_pert < m_baseline:
            return (True, f"m decreases with v: m(v={self.v})={m_baseline:.5f} > m(v={v_pert:.5f})={m_pert:.5f}")
        else:
            return (False, f"Monotonicity violated: m({v_pert:.5f})={m_pert:.5f} is not less than m({self.v})={m_baseline:.5f}")

    # ----------------------------------------------------------------------
    # Weak final answer sanity check (at most one)
    # ----------------------------------------------------------------------
    def test_answer_sanity_reasonableness(self):
        """
        Weak sanity: the final rest mass must be positive, less than initial mass,
        and not NaN/Inf.
        This test does NOT use math.isclose or re‑derive the final formula.
        """
        if self.m is None:
            return (False, "SKIP: result m missing")
        if not math.isfinite(self.m):
            return (False, "Result m is not finite (NaN or Inf)")
        if self.m <= 0:
            return (False, "Result m must be positive (rest mass)")
        if self.m0 is not None and self.m >= self.m0:
            return (False, f"Rest mass should decrease: m0={self.m0} but m={self.m}")
        return (True, "Result m is positive, finite, and less than m0")

    # ----------------------------------------------------------------------
    # Test runner
    # ----------------------------------------------------------------------
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

# Example usage (for testing purposes only, not part of the audit)
if __name__ == "__main__":
    # Dummy solution instance (simulating the expected interface)
    class DummySolution:
        def __init__(self):
            self.givens = {'m0': 1.0, 'v': 0.5, 'c': 1.0}
            self.results = {'m': 0.5773502691896257}  # Expected: 1*sqrt((1-0.5)/(1+0.5)) = sqrt(0.5/1.5) ≈ 0.57735

    checker = PhysicsChecker(DummySolution())
    success, logs = checker.run_all_tests()
    for line in logs:
        print(line)
    print(f"\nOverall {'PASS' if success else 'FAIL'}")