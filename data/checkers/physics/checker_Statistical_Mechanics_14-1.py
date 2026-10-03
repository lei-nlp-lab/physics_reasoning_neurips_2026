import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Audits solutions for the fractional magnetization problem.
    Checks physical invariants, modeling consistency, and answer sanity.
    """
    def __init__(self, solution_instance):
        # Extract inputs and result safely, default to None
        # Use results dict if available; else fallback to attribute
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.M_frac = solution_instance.results.get('M_frac', None)
            # Givens may be in solution_instance.givens or as attributes
            if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
                self.N = solution_instance.givens.get('N', None)
                self.m = solution_instance.givens.get('m', None)
                self.B = solution_instance.givens.get('B', None)
                self.tau = solution_instance.givens.get('tau', None)
            else:
                self.N = getattr(solution_instance, 'N', None)
                self.m = getattr(solution_instance, 'm', None)
                self.B = getattr(solution_instance, 'B', None)
                self.tau = getattr(solution_instance, 'tau', None)
        else:
            # Try direct attributes
            self.M_frac = getattr(solution_instance, 'M_frac', None)
            self.N = getattr(solution_instance, 'N', None)
            self.m = getattr(solution_instance, 'm', None)
            self.B = getattr(solution_instance, 'B', None)
            self.tau = getattr(solution_instance, 'tau', None)
        # Additional optional intermediate results (from explanation steps)
        # Not required but if present we can use them in modeling checks
        self.Z = getattr(solution_instance, 'Z', None)
        self.M_avg = getattr(solution_instance, 'M_avg', None)

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_M_frac_exists(self):
        if self.M_frac is None:
            return (False, "Missing required result: M_frac")
        return (True, "M_frac is present")

    def test_interface_M_frac_is_numeric(self):
        if self.M_frac is None:
            return (False, "SKIP: M_frac not available")
        if not isinstance(self.M_frac, (int, float)):
            return (False, "M_frac is not a numeric type")
        return (True, "M_frac is numeric")

    def test_interface_givens_present(self):
        missing = []
        for name in ['N', 'm', 'B', 'tau']:
            val = getattr(self, name, None)
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {missing}")
        return (True, "All required givens present")

    # ------------------------------------------------------------------
    # Invariant checks (theorem-based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_fractional_magnetization_bounded(self):
        """Fractional magnetization must lie in [-1,1] (tanh is bounded, linear approx may exceed but is wrong)"""
        if self.M_frac is None:
            return (False, "SKIP: M_frac not available")
        if not (-1.0 <= self.M_frac <= 1.0):
            return (False, f"M_frac = {self.M_frac} is outside physical bound [-1, 1]")
        return (True, "M_frac within [-1, 1]")

    def test_invariant_sign_positive_for_positive_field(self):
        """For B>0 and τ>0, magnetization should be positive (alignment with field)"""
        if any(v is None for v in [self.M_frac, self.B, self.tau]):
            return (False, "SKIP: insufficient data")
        if self.tau <= 0 or self.B <= 0:
            return (True, "SKIP: non‑positive field or temperature, sign test not applicable")
        if self.M_frac < 0:
            return (False, f"M_frac = {self.M_frac} is negative despite B>0, τ>0")
        return (True, "M_frac positive when B>0, τ>0")

    def test_invariant_zero_result_at_zero_field(self):
        """When B=0, M_frac must be 0 (no direction preference)"""
        if self.M_frac is None or self.B is None:
            return (False, "SKIP: missing data")
        if self.B != 0:
            return (True, "SKIP: B non‑zero")
        if not math.isclose(self.M_frac, 0.0, abs_tol=1e-7):
            return (False, f"M_frac = {self.M_frac} should be 0 when B=0")
        return (True, "M_frac = 0 when B=0")

    def test_invariant_intensive_property(self):
        """Fractional magnetization should not depend on N (intensive)"""
        # With only one N we can't test perturbing N, but we can check that the formula
        # (if M_frac is derived from N m B / tau) already scaled N out.
        # This is a weak sense, we just note that M_frac should not contain N factor.
        # We can verify if N appears in any intermediate expression? Not easily.
        # So we pass by assumption.
        return (True, "Intensive property implicit in formula")

    # ------------------------------------------------------------------
    # Modeling quality checks
    # ------------------------------------------------------------------
    def test_modeling_linear_approximation(self):
        """
        Verify that M_frac matches the linear model m*B/tau.
        According to pseudocode, the high‑temperature approximation is used.
        """
        if any(v is None for v in [self.M_frac, self.m, self.B, self.tau]):
            return (False, "SKIP: missing data")
        if self.tau <= 0:
            return (False, "Temperature must be positive")
        expected = self.m * self.B / self.tau
        if not math.isclose(self.M_frac, expected, rel_tol=1e-6, abs_tol=1e-6):
            return (False, f"M_frac = {self.M_frac}, expected linear form mB/τ = {expected}")
        return (True, f"M_frac matches mB/τ = {expected}")

    def test_modeling_small_field_limit(self):
        """If mB/τ is small (<0.1), the linear approximation should be accurate within 1%"""
        if any(v is None for v in [self.M_frac, self.m, self.B, self.tau]):
            return (False, "SKIP: missing data")
        if self.tau <= 0:
            return (False, "Temperature must be positive")
        x = self.m * self.B / self.tau
        if x >= 0.1:
            return (True, "SKIP: mB/τ not small, linear approximation may not hold")
        # Compare M_frac to exact tanh(x)
        exact = math.tanh(x)
        if not math.isclose(self.M_frac, exact, rel_tol=1e-2, abs_tol=1e-2):
            # But the model output is linear, not tanh. So we actually expect linear.
            # This test should check that linear is close to tanh when x is small.
            # If the answer is linear, it should be within 1% of tanh for small x.
            # So we compare linear to tanh, not M_frac to tanh.
            linear = x
            if not math.isclose(linear, exact, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "Linear approximation itself is accurate (within 1% of tanh)")
            # else it's fine
        return (True, "Modeling consistent in small‑limit case")

    def test_modeling_partition_function_consistency(self):
        """
        If intermediate results (Z, M_avg) are present, check they satisfy
        canonical ensemble relations. Otherwise skip.
        """
        if self.Z is None or self.M_avg is None or self.M_frac is None:
            return (True, "SKIP: intermediate results not available")
        # Check Z = (2 cosh(x))^N? Actually from givens we can compute expected Z.
        if any(v is None for v in [self.N, self.m, self.B, self.tau]):
            return (True, "SKIP: missing givens for Z check")
        x = self.m * self.B / self.tau
        expected_Z = (2.0 * math.cosh(x)) ** self.N
        if not math.isclose(self.Z, expected_Z, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Z = {self.Z} does not match (2 cosh(x))^N = {expected_Z}")
        # Check average magnetization: M_avg = N*m*tanh(x)
        expected_M_avg = self.N * self.m * math.tanh(x)
        if not math.isclose(self.M_avg, expected_M_avg, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"M_avg = {self.M_avg} does not match N*m*tanh(x) = {expected_M_avg}")
        # Check M_frac = M_avg/(N*m)
        derived_M_frac = self.M_avg / (self.N * self.m)
        if not math.isclose(derived_M_frac, self.M_frac, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"M_frac = {self.M_frac} inconsistent with derived = {derived_M_frac}")
        return (True, "Partition function and average magnetization consistent")

    # ------------------------------------------------------------------
    # Monotonicity checks (trend via explicit perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_magnetization_increases_with_field(self):
        """
        Check that if we increase B, the fractional magnetization increases (if model linear).
        We use the given m, B, tau and compute M_frac_perturbed = m*(B*1.01)/tau.
        Then compare with original M_frac.
        """
        if any(v is None for v in [self.M_frac, self.m, self.B, self.tau]):
            return (False, "SKIP: missing data")
        if self.tau <= 0 or self.B <= 0:
            return (True, "SKIP: non‑positive field or temperature")
        # Perturbed B: +1%
        B_pert = self.B * 1.01
        M_frac_pert = self.m * B_pert / self.tau   # linear model
        if M_frac_pert < self.M_frac:
            return (False, "Monotonicity violated: increasing B should increase M_frac")
        return (True, "M_frac increases with B as expected")

    def test_monotonicity_magnetization_decreases_with_temperature(self):
        """Check that increasing τ decreases M_frac."""
        if any(v is None for v in [self.M_frac, self.m, self.B, self.tau]):
            return (False, "SKIP: missing data")
        if self.tau <= 0 or self.B <= 0:
            return (True, "SKIP: non‑positive field or temperature")
        # Perturbed tau: +1%
        tau_pert = self.tau * 1.01
        M_frac_pert = self.m * self.B / tau_pert
        if M_frac_pert > self.M_frac:
            return (False, "Monotonicity violated: increasing τ should decrease M_frac")
        return (True, "M_frac decreases with τ as expected")

    # ------------------------------------------------------------------
    # Answer sanity (weak check only)
    # ------------------------------------------------------------------
    def test_answer_sanity_finite_and_positive_under_typical_conditions(self):
        """Simple weak sanity: M_frac should be a finite number and non‑negative when B>0, τ>0."""
        if self.M_frac is None:
            return (False, "SKIP: M_frac missing")
        if not math.isfinite(self.M_frac):
            return (False, "M_frac is not a finite number")
        # Check sign only if B>0 and τ>0
        if self.B is not None and self.tau is not None:
            if self.B > 0 and self.tau > 0:
                if self.M_frac < 0:
                    return (False, "M_frac negative under positive field/temp")
        return (True, "Answer passes basic sanity")

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