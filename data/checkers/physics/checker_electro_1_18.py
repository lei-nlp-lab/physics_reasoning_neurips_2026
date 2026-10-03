import math
import sys
import traceback

class PhysicsChecker:
    """
    Auditor for the current in a spherical capacitor with nonlinear conductivity σ = K E.
    """
    def __init__(self, solution_instance):
        # Extract givens (assumed to be attributes of solution_instance)
        self.a = getattr(solution_instance, 'a', None)
        self.b = getattr(solution_instance, 'b', None)
        self.K = getattr(solution_instance, 'K', None)
        self.V = getattr(solution_instance, 'V', None)

        # Extract result I: first try attribute, then results dict
        self.I = getattr(solution_instance, 'I', None)
        if self.I is None and hasattr(solution_instance, 'results'):
            self.I = solution_instance.results.get('I', None)

        # Do NOT assign numeric defaults – let tests fail explicitly.

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

    # ===== Interface tests =====
    def test_interface_required_keys(self):
        """Check that all required givens and result I are present with valid types."""
        missing = []
        for name, val in [("a", self.a), ("b", self.b), ("K", self.K), ("V", self.V), ("I", self.I)]:
            if val is None:
                missing.append(name)
            elif not isinstance(val, (int, float)):
                missing.append(f"{name} (non-numeric type)")
        if missing:
            return (False, f"Missing or invalid keys: {', '.join(missing)}")
        return (True, "All required keys present and numeric.")

    def test_interface_geometry_constraints(self):
        """Check b > a > 0 and K > 0, V >= 0 (weak sanity)."""
        if any(v is None for v in [self.a, self.b, self.K, self.V]):
            return (False, "SKIP: missing geometry/conductivity/voltage")
        if not (self.a > 0):
            return (False, f"a must be positive, got {self.a}")
        if not (self.b > self.a):
            return (False, f"b must be > a, got a={self.a}, b={self.b}")
        if not (self.K > 0):
            return (False, f"K must be positive, got {self.K}")
        if self.V < 0:
            return (False, f"V should be non-negative (convention), got {self.V}")
        return (True, "Geometry and conductivity constraints satisfied.")

    # ===== Invariant tests (theorem-based) =====
    def test_invariant_current_continuity(self):
        """
        Check the derived integral relation: V == sqrt(I/(4π K)) * ln(b/a)
        This is a direct consequence of steady current and Ohm's law.
        """
        if any(v is None for v in [self.a, self.b, self.K, self.V, self.I]):
            return (False, "SKIP: missing data for continuity check.")
        if self.I == 0:
            if math.isclose(self.V, 0.0, abs_tol=1e-12):
                return (True, "I=0 with V=0 is consistent.")
            else:
                return (False, "I=0 but V != 0 violates continuity.")
        try:
            expected_V = math.sqrt(self.I / (4.0 * math.pi * self.K)) * math.log(self.b / self.a)
        except (ValueError, ZeroDivisionError) as e:
            return (False, f"Numerical error in continuity check: {e}")
        if math.isclose(expected_V, self.V, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Derived V ({expected_V:.5g}) matches given V ({self.V:.5g}).")
        else:
            return (False, f"Derived V ({expected_V:.5g}) differs from given V ({self.V:.5g}).")

    def test_invariant_positivity_of_current(self):
        """Current must be positive (or zero only if V=0)."""
        if self.I is None:
            return (False, "SKIP: I not available")
        if self.V is None:
            return (False, "SKIP: V not available for context")
        if self.V > 0 and self.I <= 0:
            return (False, f"Positive V ({self.V}) should give positive I, got {self.I}")
        if self.V == 0 and not math.isclose(self.I, 0.0, abs_tol=1e-12):
            return (False, f"Zero V should give zero I, got {self.I}")
        return (True, "Current sign consistent with voltage.")

    # ===== Modeling / trace tests =====
    def test_modeling_formula_recalculation(self):
        """
        Recompute I from givens using the derived formula and compare to the result.
        This is a direct modeling check, acceptable because it is re-derived inside the checker.
        """
        if any(v is None for v in [self.a, self.b, self.K, self.V, self.I]):
            return (False, "SKIP: missing data for formula check")
        if self.b <= self.a or self.a <= 0 or self.K <= 0:
            return (False, "Invalid geometry or conductivity for formula")
        try:
            denom = (math.log(self.b / self.a)) ** 2
            if denom == 0:
                return (False, "Denominator ln(b/a)^2 is zero (b==a)")
            expected_I = 4.0 * math.pi * self.K * (self.V ** 2) / denom
        except (ValueError, ZeroDivisionError) as e:
            return (False, f"Numerical error in formula: {e}")
        if math.isclose(expected_I, self.I, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Recalculated I ({expected_I:.5g}) matches result I ({self.I:.5g}).")
        else:
            return (False, f"Recalculated I ({expected_I:.5g}) differs from result I ({self.I:.5g}).")

    def test_modeling_geometry_scale_invariance(self):
        """
        The current depends only on the ratio b/a, not on absolute sizes.
        If we scale both a and b by the same factor (keeping ratio constant),
        the formula yields the same I. We test this by computing the expected
        I for a scaled geometry (a' = 2a, b' = 2b) using the same K and V,
        and comparing to the original I.
        """
        if any(v is None for v in [self.a, self.b, self.K, self.V, self.I]):
            return (False, "SKIP: missing data for scale invariance test")
        if self.a <= 0 or self.b <= self.a:
            return (False, "Invalid geometry")
        scale_factor = 2.0
        a_scaled = self.a * scale_factor
        b_scaled = self.b * scale_factor
        try:
            denom_scaled = (math.log(b_scaled / a_scaled)) ** 2
            # Since ratio unchanged, denom_scaled == denom_original
            # But we compute anyway to avoid any rounding issues
            expected_I_scaled = 4.0 * math.pi * self.K * (self.V ** 2) / denom_scaled
        except (ValueError, ZeroDivisionError) as e:
            return (False, f"Numerical error: {e}")
        if math.isclose(expected_I_scaled, self.I, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Scale invariance holds: I for scaled geometry ({expected_I_scaled:.5g}) ≈ original I ({self.I:.5g}).")
        else:
            return (False, f"Scale invariance violated: expected I for scaled geometry ({expected_I_scaled:.5g}) ≠ original I ({self.I:.5g}).")

    # ===== Monotonicity tests =====
    def test_monotonicity_current_with_voltage(self):
        """
        Current scales as V^2. Perturb V by +10% and check that the scaling holds.
        We use the original givens to compute the new expected I and compare with
        the scaling factor derived from the original I.
        """
        if any(v is None for v in [self.a, self.b, self.K, self.V, self.I]):
            return (False, "SKIP: missing data for monotonicity test")
        if self.V <= 0:
            return (False, "SKIP: V must be positive to test scaling")
        V_new = 1.1 * self.V
        try:
            denom = (math.log(self.b / self.a)) ** 2
            expected_I_new = 4.0 * math.pi * self.K * (V_new ** 2) / denom
        except (ValueError, ZeroDivisionError) as e:
            return (False, f"Numerical error: {e}")
        scaling_from_original = expected_I_new / self.I
        if math.isclose(scaling_from_original, 1.21, rel_tol=1e-2):
            return (True, f"Scaling with V^2 holds: I_new/I = {scaling_from_original:.5g} ≈ 1.21.")
        else:
            return (False, f"Scaling deviation: I_new/I = {scaling_from_original:.5g}, expected 1.21.")

    def test_monotonicity_current_with_b(self):
        """
        Current decreases as ln(b/a) increases. Increase b by 10% and check that I decreases.
        We compare the ratio of new I to original I.
        """
        if any(v is None for v in [self.a, self.b, self.K, self.V, self.I]):
            return (False, "SKIP: missing data for monotonicity test")
        if self.b <= self.a:
            return (False, "b must be > a")
        b_new = 1.1 * self.b
        try:
            denom_orig = (math.log(self.b / self.a)) ** 2
            denom_new = (math.log(b_new / self.a)) ** 2
            expected_I_new = 4.0 * math.pi * self.K * (self.V ** 2) / denom_new
        except (ValueError, ZeroDivisionError) as e:
            return (False, f"Numerical error: {e}")
        if expected_I_new < self.I:
            return (True, f"Increasing b reduces I: I_new={expected_I_new:.5g} < I={self.I:.5g}.")
        else:
            return (False, f"Monotonicity violation: I_new={expected_I_new:.5g} >= I={self.I:.5g} for increased b.")

    # ===== Weak answer sanity (only one allowed) =====
    def test_answer_sanity_broad_magnitude(self):
        """
        Weak sanity: check that I is finite and between some extreme bounds.
        Do NOT re-derive exact value.
        """
        if self.I is None:
            return (False, "SKIP: I not available")
        if not math.isfinite(self.I):
            return (False, f"I is not finite (got {self.I})")
        # Broad magnitude bound: not unreasonably large
        if abs(self.I) > 1e100:
            return (False, f"I is implausibly large: {self.I}")
        return (True, f"Current is finite and within reasonable magnitude bounds.")

if __name__ == "__main__":
    # Dummy test to demonstrate usage
    class DummySolution:
        def __init__(self):
            self.a = 0.05
            self.b = 0.10
            self.K = 2.0
            self.V = 10.0
            self.I = 4.0 * math.pi * 2.0 * (10.0**2) / (math.log(0.10/0.05)**2)
            self.results = {'I': self.I}
    checker = PhysicsChecker(DummySolution())
    success, logs = checker.run_all_tests()
    for log in logs:
        print(log)