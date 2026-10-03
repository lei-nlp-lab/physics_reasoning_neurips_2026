import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for the problem:
    V(x) = c*x/(x^2 + a^2), equilibrium and escape conditions.
    Robust attribute access with None defaults.
    """

    def __init__(self, solution_instance):
        # Safe extraction of givens
        self.m = getattr(solution_instance, 'm', None)
        self.c = getattr(solution_instance, 'c', None)
        self.a = getattr(solution_instance, 'a', None)
        self.v = getattr(solution_instance, 'v', None)

        # Safe extraction of results (keys defined by [emit] contract)
        results = getattr(solution_instance, 'results', {}) or {}
        self.T = results.get('T', None)
        self.v_osc_thresh = results.get('v_osc_thresh', None)
        self.v_esc_left_thresh = results.get('v_esc_left_thresh', None)
        self.v_esc_right_thresh = results.get('v_esc_right_thresh', None)

        # Explanation steps for reference (optional)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all required keys exist and are numeric."""
        keys = ['T', 'v_osc_thresh', 'v_esc_left_thresh', 'v_esc_right_thresh']
        vals = [self.T, self.v_osc_thresh, self.v_esc_left_thresh, self.v_esc_right_thresh]
        missing = [k for k, v in zip(keys, vals) if v is None]
        if missing:
            return (False, f"Missing result keys: {missing}")
        # Check that they are finite numbers
        for k, v in zip(keys, vals):
            if not isinstance(v, (int, float)):
                return (False, f"Key {k} is not a number (type={type(v).__name__})")
            if math.isnan(v) or math.isinf(v):
                return (False, f"Key {k} is NaN or Inf")
        return (True, "All required result keys present and numeric.")

    def test_interface_givens_positive(self):
        """Check that m, c, a are positive numbers; v can be any real."""
        if self.m is None:
            return (False, "SKIP: m is missing")
        if self.c is None:
            return (False, "SKIP: c is missing")
        if self.a is None:
            return (False, "SKIP: a is missing")
        for name, val in [('m', self.m), ('c', self.c), ('a', self.a)]:
            if not (isinstance(val, (int, float)) and val > 0):
                return (False, f"Given {name} must be positive (got {val})")
        # v can be any real; OK if missing
        return (True, "All givens positive (or v not required).")

    # ------------------------------------------------------------------
    # Invariant / theorem-based checks
    # ------------------------------------------------------------------
    def test_invariant_threshold_order(self):
        """Check ordering: v_esc_right > v_osc_thresh == v_esc_left_thresh."""
        if any(v is None for v in [self.v_osc_thresh, self.v_esc_left_thresh, self.v_esc_right_thresh]):
            return (False, "SKIP: threshold values missing")
        # v_osc_thresh equals v_esc_left_thresh exactly
        if not math.isclose(self.v_osc_thresh, self.v_esc_left_thresh, rel_tol=1e-12, abs_tol=1e-12):
            return (False, "v_osc_thresh != v_esc_left_thresh")
        # v_esc_right > v_osc_thresh (since sqrt(2) > 1)
        if not (self.v_esc_right_thresh > self.v_osc_thresh):
            return (False, "v_esc_right_thresh should be greater than v_osc_thresh")
        return (True, "Threshold ordering correct.")

    def test_invariant_period_formula(self):
        """Compute expected T from givens and compare with results['T']."""
        if any(v is None for v in [self.m, self.c, self.a, self.T]):
            return (False, "SKIP: missing givens or T")
        expected_T = 2 * math.pi * math.sqrt(2 * self.m * self.a**3 / self.c)
        if math.isclose(self.T, expected_T, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Period matches formula: T = {self.T} vs expected {expected_T}")
        else:
            return (False, f"Period mismatch: got {self.T}, expected {expected_T}")

    def test_invariant_threshold_formulas(self):
        """
        Compute expected thresholds from givens and compare with results.
        """
        if any(v is None for v in [self.m, self.c, self.a, self.v_osc_thresh, self.v_esc_left_thresh, self.v_esc_right_thresh]):
            return (False, "SKIP: missing givens or thresholds")
        # common sqrt(c/(m*a))
        expected_common = math.sqrt(self.c / (self.m * self.a))
        expected_right = math.sqrt(2 * self.c / (self.m * self.a))
        # Compare with rel_tol 1e-2
        ok = True
        msg = ""
        if not math.isclose(self.v_osc_thresh, expected_common, rel_tol=1e-2, abs_tol=1e-2):
            ok = False
            msg += f"v_osc_thresh mismatch: got {self.v_osc_thresh}, expected {expected_common}. "
        if not math.isclose(self.v_esc_left_thresh, expected_common, rel_tol=1e-2, abs_tol=1e-2):
            ok = False
            msg += f"v_esc_left_thresh mismatch: got {self.v_esc_left_thresh}, expected {expected_common}. "
        if not math.isclose(self.v_esc_right_thresh, expected_right, rel_tol=1e-2, abs_tol=1e-2):
            ok = False
            msg += f"v_esc_right_thresh mismatch: got {self.v_esc_right_thresh}, expected {expected_right}. "
        if ok:
            return (True, "Threshold formulas match derived values.")
        else:
            return (False, msg)

    # ------------------------------------------------------------------
    # Modeling / trace consistency checks (at least two required)
    # ------------------------------------------------------------------
    def test_modeling_second_derivative_sign(self):
        """
        Check that the second derivative of V at x=-a is positive (stable).
        Re-derive from givens: V''(-a) = c/(2a^3) > 0.
        """
        if any(v is None for v in [self.c, self.a]):
            return (False, "SKIP: c or a missing")
        expected_Vpp = self.c / (2 * self.a**3)
        if expected_Vpp <= 0:
            return (False, "Calculated V''(-a) is not positive; stability fails.")
        # Also check that the period T derived from this matches if T is available.
        if self.m is not None and self.T is not None:
            omega_sq = expected_Vpp / self.m
            expected_T = 2 * math.pi / math.sqrt(omega_sq)
            if not math.isclose(self.T, expected_T, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"Period T from second derivative does not match: got {self.T}, expected {expected_T}")
        return (True, f"V''(-a) = {expected_Vpp} > 0, stable equilibrium confirmed.")

    def test_modeling_threshold_ratio(self):
        """
        Check the ratio v_esc_right_thresh / v_osc_thresh ≈ sqrt(2).
        This follows from energy conservation: left escape requires KE = c/(2a),
        right escape requires KE = c/a. So the square of the ratio is 2.
        """
        if any(v is None for v in [self.v_osc_thresh, self.v_esc_right_thresh]):
            return (False, "SKIP: missing thresholds")
        if self.v_osc_thresh == 0:
            return (False, "v_osc_thresh is zero, cannot compute ratio")
        ratio = self.v_esc_right_thresh / self.v_osc_thresh
        expected_ratio = math.sqrt(2)
        if math.isclose(ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Ratio = {ratio} matches sqrt(2) = {expected_ratio}")
        else:
            return (False, f"Ratio = {ratio} deviates from sqrt(2) = {expected_ratio}")

    # ------------------------------------------------------------------
    # Monotonicity test (with explicit perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_threshold_with_c(self):
        """
        Check that increasing c (by a small factor) increases the thresholds.
        Use the derived formulas to compute perturbed values.
        """
        if any(v is None for v in [self.c, self.m, self.a, self.v_osc_thresh, self.v_esc_right_thresh]):
            return (False, "SKIP: missing givens or thresholds")
        # Baseline thresholds
        base_osc = self.v_osc_thresh
        base_right = self.v_esc_right_thresh

        # Perturb: increase c by 5%
        c_pert = self.c * 1.05
        expected_osc_pert = math.sqrt(c_pert / (self.m * self.a))
        expected_right_pert = math.sqrt(2 * c_pert / (self.m * self.a))

        # Check that perturbed thresholds are larger than baseline
        if not (expected_osc_pert > base_osc):
            return (False, "Increasing c should increase v_osc_thresh (monotonic).")
        if not (expected_right_pert > base_right):
            return (False, "Increasing c should increase v_esc_right_thresh (monotonic).")
        return (True, f"c increased by 5%: both thresholds increase as expected.")

    def test_monotonicity_threshold_with_a(self):
        """
        Check that increasing a decreases thresholds (since they scale as 1/sqrt(a)).
        Use the derived formulas to compute perturbed values.
        """
        if any(v is None for v in [self.c, self.m, self.a, self.v_osc_thresh, self.v_esc_right_thresh]):
            return (False, "SKIP: missing givens or thresholds")
        base_osc = self.v_osc_thresh
        base_right = self.v_esc_right_thresh

        # Perturb: increase a by 5%
        a_pert = self.a * 1.05
        expected_osc_pert = math.sqrt(self.c / (self.m * a_pert))
        expected_right_pert = math.sqrt(2 * self.c / (self.m * a_pert))

        # Check that perturbed thresholds are smaller than baseline
        if not (expected_osc_pert < base_osc):
            return (False, "Increasing a should decrease v_osc_thresh (monotonic).")
        if not (expected_right_pert < base_right):
            return (False, "Increasing a should decrease v_esc_right_thresh (monotonic).")
        return (True, f"a increased by 5%: both thresholds decrease as expected.")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_positive_and_finite(self):
        """Weak sanity: all results are positive and finite."""
        if any(v is None for v in [self.T, self.v_osc_thresh, self.v_esc_left_thresh, self.v_esc_right_thresh]):
            return (False, "SKIP: missing results")
        for name, val in [('T', self.T), ('v_osc_thresh', self.v_osc_thresh),
                          ('v_esc_left_thresh', self.v_esc_left_thresh),
                          ('v_esc_right_thresh', self.v_esc_right_thresh)]:
            if not (val > 0 and math.isfinite(val)):
                return (False, f"{name} is not positive and finite: {val}")
        return (True, "All results positive and finite.")

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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)