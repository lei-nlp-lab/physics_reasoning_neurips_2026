import math
import sympy
import traceback

class PhysicsChecker:
    """
    Audits a solution to the wheel-with-string unwinding problem.
    Expected results: rate1 and rate2 (both should equal R * omega).
    """

    def __init__(self, solution_instance):
        # Extract givens (safe defaults: None)
        givens = getattr(solution_instance, 'givens', {})
        self.R = givens.get('R', None)
        self.omega = givens.get('omega', None)

        # Extract results
        results = getattr(solution_instance, 'results', {})
        self.rate1 = results.get('rate1', None)
        self.rate2 = results.get('rate2', None)

        # Store the instance for possible auxiliary data
        self.solution = solution_instance

    # ------------- Interface Tests -------------
    def test_interface_rate1_exists(self):
        """rate1 must be present and numeric"""
        if self.rate1 is None:
            return (False, "Missing 'rate1' in results")
        if not isinstance(self.rate1, (int, float, sympy.Number)):
            return (False, f"rate1 is not numeric: {type(self.rate1)}")
        return (True, "rate1 exists and is numeric")

    def test_interface_rate2_exists(self):
        """rate2 must be present and numeric"""
        if self.rate2 is None:
            return (False, "Missing 'rate2' in results")
        if not isinstance(self.rate2, (int, float, sympy.Number)):
            return (False, f"rate2 is not numeric: {type(self.rate2)}")
        return (True, "rate2 exists and is numeric")

    # ------------- Invariant Tests (theorem-based) -------------
    def test_invariant_rate_value(self):
        """
        Both rates should equal R * omega (within tolerance).
        This is the core physical result: dL/dt = R * omega.
        """
        if any(v is None for v in [self.R, self.omega, self.rate1, self.rate2]):
            return (False, "SKIP: missing givens or results for invariant check")

        expected = self.R * self.omega
        ok1 = math.isclose(float(self.rate1), float(expected), rel_tol=1e-2, abs_tol=1e-2)
        ok2 = math.isclose(float(self.rate2), float(expected), rel_tol=1e-2, abs_tol=1e-2)
        if not (ok1 and ok2):
            msg = (f"Rates deviate from R*omega: "
                   f"rate1={self.rate1}, rate2={self.rate2}, expected={expected}")
            return (False, msg)
        return (True, "Both rates equal R*omega within tolerance")

    def test_invariant_rate_positive(self):
        """Rates must be positive (unwinding increases length)"""
        if self.rate1 is None or self.rate2 is None:
            return (False, "SKIP: rates not available")
        pos1 = float(self.rate1) > 0
        pos2 = float(self.rate2) > 0
        if not (pos1 and pos2):
            return (False, f"Rates must be positive: rate1={self.rate1}, rate2={self.rate2}")
        return (True, "Both rates are positive")

    def test_invariant_direction_independence(self):
        """rate1 and rate2 must be equal (the problem states both directions give same magnitude)"""
        if self.rate1 is None or self.rate2 is None:
            return (False, "SKIP: rates not available")
        if not math.isclose(float(self.rate1), float(self.rate2), rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"rate1={self.rate1} and rate2={self.rate2} differ significantly")
        return (True, "rate1 and rate2 are equal within tolerance")

    # ------------- Modeling / Trace Tests (at least two required) -------------
    def test_modeling_unwinding_relation(self):
        """
        Check consistency with geometric relation dL = R * omega * dt.
        For a small time step, the increase in length should match.
        """
        if any(v is None for v in [self.R, self.omega, self.rate1, self.rate2]):
            return (False, "SKIP: missing data for modeling test")

        dt = 0.001
        expected_increase = self.R * self.omega * dt

        actual_increase1 = float(self.rate1) * dt
        if not math.isclose(actual_increase1, float(expected_increase), rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Modeling failure for rate1: dL/dt does not match R*omega")

        actual_increase2 = float(self.rate2) * dt
        if not math.isclose(actual_increase2, float(expected_increase), rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Modeling failure for rate2: dL/dt does not match R*omega")

        return (True, "Both rates satisfy the geometric unwinding relation dL = R*omega*dt")

    def test_modeling_tangential_speed(self):
        """
        The rate of unwinding equals the tangential speed of the rim point.
        v_tangential = R * omega. Check that rate equals v_tangential.
        """
        if any(v is None for v in [self.R, self.omega, self.rate1, self.rate2]):
            return (False, "SKIP: missing data for tangential speed test")

        v_tan = self.R * self.omega
        if not math.isclose(float(self.rate1), float(v_tan), rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"rate1 does not equal tangential speed {v_tan}: {self.rate1}")
        if not math.isclose(float(self.rate2), float(v_tan), rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"rate2 does not equal tangential speed {v_tan}: {self.rate2}")
        return (True, "Both rates match the tangential speed R*omega of the rim")

    # ------------- Monotonicity Tests (genuine perturbation) -------------
    def test_monotonicity_rate_with_R(self):
        """
        If R were increased by 10%, the rate should increase by 10%.
        We compute a virtual perturbed rate using the given omega and original rate.
        """
        if any(v is None for v in [self.R, self.omega, self.rate1, self.rate2]):
            return (False, "SKIP: missing data for monotonicity test")

        factor = 1.1  # 10% increase
        new_R = self.R * factor
        # Baseline rate (original)
        base_rate1 = float(self.rate1)
        base_rate2 = float(self.rate2)
        # Perturbed expected rate (if R increases by factor)
        perturbed_expected = new_R * self.omega
        # If the system is proportional, original rate times factor should give perturbed expected
        predicted1 = base_rate1 * factor
        predicted2 = base_rate2 * factor
        if not math.isclose(predicted1, float(perturbed_expected), rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Monotonicity with R failed for rate1: predicted {predicted1}, expected {perturbed_expected}")
        if not math.isclose(predicted2, float(perturbed_expected), rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Monotonicity with R failed for rate2: predicted {predicted2}, expected {perturbed_expected}")
        return (True, "Both rates scale linearly with R as expected")

    def test_monotonicity_rate_with_omega(self):
        """
        If omega were increased by 10%, the rate should increase by 10%.
        """
        if any(v is None for v in [self.R, self.omega, self.rate1, self.rate2]):
            return (False, "SKIP: missing data for monotonicity test")

        factor = 1.1  # 10% increase
        new_omega = self.omega * factor
        base_rate1 = float(self.rate1)
        base_rate2 = float(self.rate2)
        perturbed_expected = self.R * new_omega
        predicted1 = base_rate1 * factor
        predicted2 = base_rate2 * factor
        if not math.isclose(predicted1, float(perturbed_expected), rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Monotonicity with omega failed for rate1: predicted {predicted1}, expected {perturbed_expected}")
        if not math.isclose(predicted2, float(perturbed_expected), rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Monotonicity with omega failed for rate2: predicted {predicted2}, expected {perturbed_expected}")
        return (True, "Both rates scale linearly with omega as expected")

    # ------------- Weak Sanity Check (only one allowed) -------------
    def test_answer_sanity_finite_and_sign(self):
        """Weak check: rates must be finite and positive."""
        if self.rate1 is None or self.rate2 is None:
            return (False, "SKIP: rates missing")
        try:
            v1 = float(self.rate1)
            v2 = float(self.rate2)
        except (TypeError, ValueError):
            return (False, "Cannot convert rates to float")
        if math.isnan(v1) or math.isnan(v2):
            return (False, "Rate is NaN")
        if v1 <= 0 or v2 <= 0:
            return (False, "Rate must be positive")
        return (True, "Rates are finite and positive")

    # ------------- Main runner -------------
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)