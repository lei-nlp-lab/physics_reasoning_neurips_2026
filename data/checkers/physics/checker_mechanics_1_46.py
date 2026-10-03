import math
import sympy as sp
from typing import Any, Dict, List, Tuple, Optional

class PhysicsChecker:
    """
    Audits a solution for the repulsive inverse-square scattering problem.
    Checks interface completeness, physical invariants, modeling consistency,
    and monotonicity trends.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Safely extract givens – use None if not available.
        self.m = getattr(solution_instance, 'm', None)
        self.V0 = getattr(solution_instance, 'V0', None)
        self.b = getattr(solution_instance, 'b', None)
        self.k = getattr(solution_instance, 'k', None)

        # Safely extract results – use None if not available.
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        self.r_min = results.get('r_min', None)
        self.theta_relation = results.get('theta_relation', None)
        self.d_sigma_d_Omega = results.get('d_sigma_d_Omega', None)

        # Derived quantities (computed from givens when possible).
        self.E = None
        self.L = None
        self.a = None
        if self.m is not None and self.V0 is not None:
            self.E = 0.5 * self.m * self.V0 ** 2
        if self.m is not None and self.V0 is not None and self.b is not None:
            self.L = self.m * self.V0 * self.b
        if self.m is not None and self.V0 is not None and self.k is not None:
            self.a = self.k / (self.m * self.V0 ** 2)

    def _check_none(self, *args: Any) -> bool:
        """Returns True if any argument is None."""
        return any(arg is None for arg in args)

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all required result keys are present."""
        keys = ['r_min', 'theta_relation', 'd_sigma_d_Omega']
        missing = [k for k in keys if k not in self.__dict__ or self.__dict__.get(k) is None]
        if missing:
            return False, f"Missing result keys: {missing}"
        return True, "All required keys present."

    def test_interface_givens_types(self) -> Tuple[bool, str]:
        """Check that givens have plausible numeric types."""
        for name in ['m', 'V0', 'b', 'k']:
            val = getattr(self, name)
            if val is None:
                return False, f"Missing given: {name}"
            if not isinstance(val, (int, float)):
                return False, f"Given '{name}' is not numeric (type: {type(val).__name__})"
        return True, "All givens present and numeric."

    # ----------------------------------------------------------------------
    # Invariant tests (use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_energy_at_closest_approach(self) -> Tuple[bool, str]:
        """
        Check that at r=r_min the total energy equals
        k/r_min + L^2/(2 m r_min^2).
        """
        if self._check_none(self.m, self.V0, self.b, self.k, self.r_min, self.E, self.L):
            return False, "SKIP: missing givens or results"

        # L = m*V0*b
        L = self.m * self.V0 * self.b
        E = 0.5 * self.m * self.V0 ** 2
        r = self.r_min

        # Energy balance: E = U_eff(r_min) = k/r_min + L^2/(2m r_min^2)
        U_eff = self.k / r + (L ** 2) / (2 * self.m * r ** 2)

        if not math.isclose(E, U_eff, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"Energy mismatch: E={E}, U_eff={U_eff}, diff={abs(E-U_eff)}"
        return True, "Energy conserved at closest approach."

    def test_invariant_angular_momentum_scaling(self) -> Tuple[bool, str]:
        """
        Check that L = m*V0*b is consistent with the orbit integral
        (indirectly via derived quantities). At closest approach, V_perp * m * r_min = L.
        We don't have V_perp directly, but we can verify that the quadratic
        that determines r_min is satisfied.
        """
        if self._check_none(self.m, self.V0, self.b, self.k, self.r_min, self.a):
            return False, "SKIP: missing data"

        # Quadratic: r_min^2 - 2*a*r_min - b^2 = 0
        quad = self.r_min ** 2 - 2 * self.a * self.r_min - self.b ** 2
        if not math.isclose(quad, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"r_min does not satisfy quadratic: quad={quad}"
        return True, "Angular momentum conservation (via quadratic) is satisfied."

    # ----------------------------------------------------------------------
    # Modeling / trace tests
    # ----------------------------------------------------------------------
    def test_trace_closest_approach_formula(self) -> Tuple[bool, str]:
        """Check that r_min = a + sqrt(a^2 + b^2)."""
        if self._check_none(self.r_min, self.a, self.b):
            return False, "SKIP: missing data"
        expected = self.a + math.sqrt(self.a ** 2 + self.b ** 2)
        if not math.isclose(self.r_min, expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"r_min {self.r_min} != expected {expected}"
        return True, "Closest approach formula matches."

    def test_trace_theta_relation_content(self) -> Tuple[bool, str]:
        """
        Check that theta_relation string contains the correct functional form.
        We do not recompute theta numerically; we just verify the string matches.
        """
        if self.theta_relation is None:
            return False, "SKIP: theta_relation not available"
        expected = "cot(theta) = m*V0^2*b/k"
        # Allow minor whitespace differences.
        if expected.replace(" ", "").lower() not in self.theta_relation.replace(" ", "").lower():
            return False, f"theta_relation does not match expected. Got: {self.theta_relation}"
        return True, "Theta relation string is correct."

    def test_trace_differential_cross_section_positive(self) -> Tuple[bool, str]:
        """
        Differential cross section must be positive and finite.
        """
        if self.d_sigma_d_Omega is None:
            return False, "SKIP: d_sigma_d_Omega not available"
        if not isinstance(self.d_sigma_d_Omega, (int, float)):
            return False, "d_sigma_d_Omega is not numeric"
        if self.d_sigma_d_Omega < 0:
            return False, "d_sigma_d_Omega is negative"
        if math.isnan(self.d_sigma_d_Omega) or math.isinf(self.d_sigma_d_Omega):
            return False, "d_sigma_d_Omega is NaN or infinite"
        return True, "d_sigma_d_Omega is positive and finite."

    def test_modeling_quadratic_root(self) -> Tuple[bool, str]:
        """
        Verify that r_min satisfies the original energy equation:
        E = k/r_min + L^2/(2*m*r_min^2) exactly (to numerical tolerance).
        This is the same as invariant but uses derived E and L.
        """
        if self._check_none(self.m, self.V0, self.b, self.k, self.r_min, self.E, self.L):
            return False, "SKIP: missing data"
        E_eff = self.k / self.r_min + (self.L ** 2) / (2 * self.m * self.r_min ** 2)
        if not math.isclose(self.E, E_eff, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"Effective potential energy mismatch: E={self.E}, E_eff={E_eff}"
        return True, "Quadratic root satisfies energy conservation."

    # ----------------------------------------------------------------------
    # Monotonicity tests
    # ----------------------------------------------------------------------
    def test_monotonicity_r_min_with_b(self) -> Tuple[bool, str]:
        """
        The distance of closest approach should increase as impact parameter b
        increases (for fixed m, V0, k). Perturb b to +5% and check that new r_min
        is larger.
        """
        if self._check_none(self.r_min, self.m, self.V0, self.b, self.k, self.a):
            return False, "SKIP: missing data for monotonicity"
        # Increase b by 5%
        b_pert = self.b * 1.05
        expected_r_min_pert = self.a + math.sqrt(self.a ** 2 + b_pert ** 2)
        # We don't have r_min_pert from solution, so just check trend: it should be larger.
        if expected_r_min_pert <= self.r_min:
            return False, "Expected r_min to increase with b, but it did not."
        return True, "Monotonicity: r_min increases with b (confirmed by expected formula)."

    def test_monotonicity_r_min_with_k(self) -> Tuple[bool, str]:
        """
        For fixed m, V0, b, increasing k (repulsion strength) should increase r_min.
        Perturb k to +10%.
        """
        if self._check_none(self.r_min, self.m, self.V0, self.b, self.k):
            return False, "SKIP: missing data for monotonicity"
        k_pert = self.k * 1.1
        a_pert = k_pert / (self.m * self.V0 ** 2)
        expected_r_min_pert = a_pert + math.sqrt(a_pert ** 2 + self.b ** 2)
        if expected_r_min_pert <= self.r_min:
            return False, "Expected r_min to increase with k, but it did not."
        return True, "Monotonicity: r_min increases with k (confirmed by expected formula)."

    # ----------------------------------------------------------------------
    # Optional weak answer sanity (non-dominant)
    # ----------------------------------------------------------------------
    def test_answer_sanity_r_min_sign(self) -> Tuple[bool, str]:
        """
        r_min must be positive and larger than b (since repulsive).
        """
        if self.r_min is None:
            return False, "SKIP: r_min not available"
        if not isinstance(self.r_min, (int, float)):
            return False, "r_min not numeric"
        if self.r_min <= 0:
            return False, "r_min is not positive"
        if self.b is not None and self.r_min <= self.b:
            return False, "r_min should be > b for repulsive force (if b>0)"
        return True, "r_min is positive and plausible."

    # ----------------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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