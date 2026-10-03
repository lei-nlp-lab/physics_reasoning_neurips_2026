import math
from typing import Tuple, List, Any

class PhysicsChecker:
    """
    Audits solution for 'Man in accelerating elevator' problem.
    Covers interface, invariant, modeling/trace, monotonicity, and weak answer sanity.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract given variables (safe defaults: None)
        self.w = getattr(solution_instance, 'w', None)
        self.a = getattr(solution_instance, 'a', None)
        self.g = getattr(solution_instance, 'g', None)
        self.V = getattr(solution_instance, 'V', None)
        self.v = getattr(solution_instance, 'v', None)

        # Extract result dictionary
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        self.N = results.get('N', None)
        self.P = results.get('P', None)

    # -----------------------------------------------------------------
    # Interface checks
    # -----------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that results contains both 'N' and 'P'."""
        if self.N is None:
            return (False, "Missing key 'N' in results")
        if self.P is None:
            return (False, "Missing key 'P' in results")
        return (True, "Both 'N' and 'P' are present")

    def test_interface_givens_defined(self) -> Tuple[bool, str]:
        """All required givens (w, a, g, V, v) must be provided (not None)."""
        missing = []
        for name, val in [('w', self.w), ('a', self.a), ('g', self.g),
                          ('V', self.V), ('v', self.v)]:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All givens are defined")

    # -----------------------------------------------------------------
    # Invariant checks (theorem-based physical relations)
    # -----------------------------------------------------------------
    def test_invariant_newton_law(self) -> Tuple[bool, str]:
        """Newton's second law: N - w = (w/g) * a ."""
        if any(v is None for v in (self.w, self.a, self.g, self.N)):
            return (False, "SKIP: one or more required values missing")
        if not all(isinstance(v, (int, float)) for v in (self.w, self.a, self.g, self.N)):
            return (False, "SKIP: non-numeric values encountered")
        mass = self.w / self.g
        expected_net_force = mass * self.a
        actual_net_force = self.N - self.w
        if math.isclose(actual_net_force, expected_net_force, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"N - w = {actual_net_force} = (w/g)*a = {expected_net_force}")
        else:
            return (False, f"N - w = {actual_net_force}, expected {expected_net_force}")

    def test_invariant_power_force_velocity(self) -> Tuple[bool, str]:
        """Power = N * (V + v) ."""
        if any(v is None for v in (self.N, self.V, self.v, self.P)):
            return (False, "SKIP: required values missing")
        if not all(isinstance(v, (int, float)) for v in (self.N, self.V, self.v, self.P)):
            return (False, "SKIP: non-numeric values encountered")
        speed = self.V + self.v
        expected = self.N * speed
        if math.isclose(self.P, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"P = {self.P} matches N*(V+v) = {expected}")
        else:
            return (False, f"P = {self.P}, expected {expected}")

    # -----------------------------------------------------------------
    # Modeling/trace checks (local relations from explanation steps)
    # -----------------------------------------------------------------
    def test_modeling_man_absolute_acceleration(self) -> Tuple[bool, str]:
        """
        The man's absolute acceleration equals the elevator's acceleration a
        (since relative speed constant). Then N - w = m*a.
        Check that result N is consistent with this reasoning.
        """
        if any(v is None for v in (self.w, self.a, self.g, self.N)):
            return (False, "SKIP: values missing")
        if not all(isinstance(v, (int, float)) for v in (self.w, self.a, self.g, self.N)):
            return (False, "SKIP: non-numeric values")
        # Derive N from the step: N = w + (w/g)*a
        computed_N = self.w + (self.w / self.g) * self.a
        if math.isclose(self.N, computed_N, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Modeling: N = w + (w/g)*a gives {computed_N}, matches stored N = {self.N}")
        else:
            return (False, f"Modeling inconsistency: stored N = {self.N}, but step yields {computed_N}")

    def test_modeling_power_from_step(self) -> Tuple[bool, str]:
        """
        Power output is computed as N * absolute velocity of man = N*(V+v).
        Verify that P matches this derived quantity.
        """
        if any(v is None for v in (self.N, self.V, self.v, self.P)):
            return (False, "SKIP: values missing")
        if not all(isinstance(v, (int, float)) for v in (self.N, self.V, self.v, self.P)):
            return (False, "SKIP: non-numeric values")
        derived_P = self.N * (self.V + self.v)
        if math.isclose(self.P, derived_P, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Modeling: P = N*(V+v) = {derived_P}, matches stored P = {self.P}")
        else:
            return (False, f"Modeling inconsistency: stored P = {self.P}, but step yields {derived_P}")

    # -----------------------------------------------------------------
    # Monotonicity checks (explicit input perturbation)
    # -----------------------------------------------------------------
    def test_monotonicity_apparent_weight_increases_with_a(self) -> Tuple[bool, str]:
        """If upward acceleration a increases, apparent weight N increases."""
        if any(v is None for v in (self.w, self.a, self.g)):
            return (False, "SKIP: missing required givens")
        if not all(isinstance(v, (int, float)) for v in (self.w, self.a, self.g)):
            return (False, "SKIP: non-numeric values")
        delta = 0.05 * abs(self.a)  # 5% increase
        a_new = self.a + delta
        N_base = self.w * (1.0 + self.a / self.g)
        N_new   = self.w * (1.0 + a_new / self.g)
        if N_new > N_base:
            return (True, f"Monotonicity: a from {self.a} to {a_new}: N {N_base} → {N_new}")
        else:
            return (False, f"Monotonicity violated: N did not increase with a")

    def test_monotonicity_power_increases_with_v(self) -> Tuple[bool, str]:
        """If relative speed v increases, power P increases."""
        if any(v is None for v in (self.w, self.a, self.g, self.V, self.v)):
            return (False, "SKIP: missing required givens")
        if not all(isinstance(v, (int, float)) for v in (self.w, self.a, self.g, self.V, self.v)):
            return (False, "SKIP: non-numeric values")
        delta = 0.05 * abs(self.v)
        v_new = self.v + delta
        factor = self.w * (1.0 + self.a / self.g)
        P_base = factor * (self.V + self.v)
        P_new  = factor * (self.V + v_new)
        if P_new > P_base:
            return (True, f"Monotonicity: v from {self.v} to {v_new}: P {P_base} → {P_new}")
        else:
            return (False, f"Monotonicity violated: P did not increase with v")

    def test_monotonicity_power_increases_with_V(self) -> Tuple[bool, str]:
        """If elevator speed V increases, power P increases."""
        if any(v is None for v in (self.w, self.a, self.g, self.V, self.v)):
            return (False, "SKIP: missing required givens")
        if not all(isinstance(v, (int, float)) for v in (self.w, self.a, self.g, self.V, self.v)):
            return (False, "SKIP: non-numeric values")
        delta = 0.05 * abs(self.V)
        V_new = self.V + delta
        factor = self.w * (1.0 + self.a / self.g)
        P_base = factor * (self.V + self.v)
        P_new  = factor * (V_new + self.v)
        if P_new > P_base:
            return (True, f"Monotonicity: V from {self.V} to {V_new}: P {P_base} → {P_new}")
        else:
            return (False, f"Monotonicity violated: P did not increase with V")

    # -----------------------------------------------------------------
    # Weak answer sanity (at most one)
    # -----------------------------------------------------------------
    def test_answer_sanity_finite_nonnegative(self) -> Tuple[bool, str]:
        """
        N and P must be finite real numbers. Positive only if a > -g, which we do not enforce,
        but at least check finiteness and non-NaN.
        """
        if self.N is None or self.P is None:
            return (False, "SKIP: results missing")
        if not (isinstance(self.N, (int, float)) and isinstance(self.P, (int, float))):
            return (False, "SKIP: non-numeric results")
        if not (math.isfinite(self.N) and math.isfinite(self.P)):
            return (False, f"Non-finite values: N={self.N}, P={self.P}")
        # Simple range: absolute values not astronomically large (arbitrary sanity)
        max_sensible = 1e12
        if abs(self.N) > max_sensible or abs(self.P) > max_sensible:
            return (False, f"Extreme magnitude detected: N={self.N}, P={self.P}")
        return (True, "N and P are finite and not extreme")

    # -----------------------------------------------------------------
    # Main test runner
    # -----------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs = []
        all_pass = True
        # Discover all test_ methods
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