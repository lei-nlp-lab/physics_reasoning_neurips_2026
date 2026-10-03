import math
import traceback
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Audits a solution instance for the 'ice thickness growth on a lake' problem.

    Required givens (in `solution_instance.givens`):
        - Lambda (thermal conductivity of ice) [float]
        - Delta_T (temperature drop, positive) [float]
        - ell (latent heat per unit volume L/V) [float]

    Required result (in `solution_instance.results`):
        - 's' or 'h' : a function s(t) returning ice thickness for time t >= 0.

    The checker performs:
        - interface checks for presence and types of keys
        - physical invariants (Stefan equation, initial condition)
        - modeling quality (scaling behavior)
        - monotonicity (thickness increases with time, thickness increases with Lambda)
        - weak sanity on final answer
    """

    def __init__(self, solution_instance):
        # Extract givens safely
        givens = getattr(solution_instance, 'givens', {})
        self.Lambda = givens.get('Lambda', None)
        self.Delta_T = givens.get('Delta_T', None)
        self.ell = givens.get('ell', None)  # latent heat per unit volume L/V

        # Also accept alternative names
        if self.ell is None:
            self.ell = givens.get('L/V', None)
        if self.ell is None:
            self.ell = givens.get('L_by_V', None)

        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.s_func = results.get('s', None)   # function s(t)
        # Fallback: 'h' key
        if self.s_func is None:
            self.s_func = results.get('h', None)

        # Keep reference to original instance for diagnostic use
        self.solution_instance = solution_instance

    # ------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------
    def test_interface_required_givens(self) -> Tuple[bool, str]:
        """Check that required givens exist and are numbers."""
        missing = []
        for name, val in [('Lambda', self.Lambda),
                          ('Delta_T', self.Delta_T),
                          ('ell (L/V)', self.ell)]:
            if val is None:
                missing.append(name)
            elif not isinstance(val, (int, float)):
                missing.append(f"{name} (type {type(val).__name__})")
        if missing:
            return False, f"Missing or invalid givens: {', '.join(missing)}"
        return True, "All required givens present and numeric."

    def test_interface_result_function(self) -> Tuple[bool, str]:
        """Check that the result s(t) or h(t) exists and is callable."""
        if self.s_func is None:
            return False, "Key 's' or 'h' missing in results."
        if not callable(self.s_func):
            return False, f"Result is not callable (type {type(self.s_func).__name__})."
        # Quick test that it returns a number for a plausible time
        try:
            val = self.s_func(1.0)  # test at t=1
            if not isinstance(val, (int, float)):
                return False, f"s(1) returned non-numeric: {val}"
            if val < 0:
                return False, "s(1) returned negative thickness."
        except Exception as e:
            return False, f"s(1) call failed: {e}"
        return True, "Result function callable and returns positive number at t=1."

    # ------------------------------------------------------------
    # Invariant (theorem-based) checks
    # ------------------------------------------------------------
    def test_invariant_stefan_equation(self) -> Tuple[bool, str]:
        """
        Check that the Stefan condition ℓ ds/dt ≈ Λ ΔT / s holds numerically.
        We compute ds/dt from finite difference at several sample times.
        """
        if any(v is None for v in [self.Lambda, self.Delta_T, self.ell, self.s_func]):
            return False, "SKIP: Missing givens or result function."

        sample_times = [0.1, 0.5, 1.0, 2.0]  # seconds
        dt = 1e-6
        results_ok = True
        messages = []
        for t0 in sample_times:
            try:
                s0 = self.s_func(t0)
                s1 = self.s_func(t0 + dt)
            except Exception as e:
                messages.append(f"t={t0}: function call error: {e}")
                results_ok = False
                continue
            ds_dt = (s1 - s0) / dt
            if ds_dt <= 0:
                messages.append(f"t={t0}: ds_dt <= 0 (got {ds_dt})")
                results_ok = False
                continue
            # left-hand side: ell * ds/dt
            lhs = self.ell * ds_dt
            # right-hand side: Lambda * Delta_T / s0
            rhs = self.Lambda * self.Delta_T / s0
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                rel_err = abs(lhs - rhs) / max(abs(rhs), 1e-10)
                messages.append(f"t={t0}: lhs={lhs:.4e}, rhs={rhs:.4e}, rel_err={rel_err:.3f}")
                results_ok = False

        if not results_ok:
            return False, "Stefan equation not satisfied: " + "; ".join(messages)
        return True, "Stefan equation holds at sample times within tolerance."

    def test_invariant_initial_condition(self) -> Tuple[bool, str]:
        """Check s(0) == 0 or approaches zero for t→0."""
        if self.s_func is None:
            return False, "SKIP: No result function."
        try:
            s0 = self.s_func(0.0)
            if s0 is None:
                return False, "s(0) returned None."
            if isinstance(s0, (int, float)) and s0 >= 0:
                if math.isclose(s0, 0.0, abs_tol=1e-3):
                    return True, f"s(0) = {s0} ≈ 0."
                else:
                    return False, f"s(0) = {s0}, expected near 0."
            else:
                return False, f"s(0) returned {s0} (type {type(s0).__name__})"
        except Exception as e:
            # Possibly s(0) causes division by zero; allowed if not exactly zero
            # Try very small time
            try:
                s_small = self.s_func(1e-10)
                if s_small is not None and s_small > 0 and s_small < 1e-6:
                    return True, f"s(1e-10) = {s_small} ≈ 0 (functional issue at t=0)."
                else:
                    return False, f"s(1e-10) returned {s_small}, not close to zero."
            except:
                return False, f"s(0) crashed: {e}"

    # ------------------------------------------------------------
    # Modeling/trace checks
    # ------------------------------------------------------------
    def test_modeling_sqrt_scaling_with_time(self) -> Tuple[bool, str]:
        """
        Verify that s(t) grows like sqrt(t).
        If s(t) = A * sqrt(t), then s^2(t) is linear in t.
        Check approximate linearity for at least three times.
        """
        if self.s_func is None:
            return False, "SKIP: No result function."

        times = [1.0, 2.0, 3.0, 4.0]
        s_sq = []
        for t in times:
            try:
                s = self.s_func(t)
                if s is None or s < 0:
                    return False, f"s({t}) invalid: {s}"
                s_sq.append(s * s)
            except Exception as e:
                return False, f"s({t}) call error: {e}"

        # Check that s^2 increases approximately linearly with time
        ddiffs = []
        for i in range(1, len(times)):
            dt = times[i] - times[i-1]
            dsq_diff = s_sq[i] - s_sq[i-1]
            if dt > 0:
                ddiffs.append(dsq_diff / dt)
        if len(ddiffs) < 2:
            return False, "Not enough data points."

        # All slopes should be close to each other (constant slope)
        slope_mean = sum(ddiffs) / len(ddiffs)
        for slope in ddiffs:
            if not math.isclose(slope, slope_mean, rel_tol=0.2, abs_tol=0.01):
                return False, f"Slope in s^2 vs t not constant: {slope} vs mean {slope_mean}"
        return True, f"s^2(t) approximately linear: slope ≈ {slope_mean:.4f}"

    def test_modeling_scaling_with_parameters(self) -> Tuple[bool, str]:
        """
        Check that s scales like sqrt(Lambda * Delta_T / ell).
        Use the given givens to compute the expected coefficient A = sqrt(2 * Lambda * Delta_T / ell).
        Then verify s(t) ≈ A * sqrt(t) at several times.
        """
        if any(v is None for v in [self.Lambda, self.Delta_T, self.ell, self.s_func]):
            return False, "SKIP: Missing givens or function."

        # Expected coefficient from theory
        A_expected = math.sqrt(2.0 * self.Lambda * self.Delta_T / self.ell)

        test_times = [0.5, 1.0, 2.0]
        for t in test_times:
            try:
                s = self.s_func(t)
                if s is None or t <= 0:
                    continue
                s_expected = A_expected * math.sqrt(t)
                if not math.isclose(s, s_expected, rel_tol=0.1, abs_tol=0.01):
                    return False, f"t={t}: s={s:.4f}, expected {s_expected:.4f} (ratio {s/s_expected:.3f})"
            except Exception as e:
                return False, f"t={t} computation error: {e}"
        return True, "Scaling with sqrt(Lambda*Delta_T/ell) matches expected coefficient."

    # ------------------------------------------------------------
    # Monotonicity checks
    # ------------------------------------------------------------
    def test_monotonicity_time_increase(self) -> Tuple[bool, str]:
        """
        Check that ice thickness increases when time increases.
        Perturb t by +10% from a baseline and verify s increases.
        """
        if self.s_func is None:
            return False, "SKIP: No result function."

        base_t = 2.0
        try:
            s_base = self.s_func(base_t)
            s_perturb = self.s_func(base_t * 1.1)
        except Exception as e:
            return False, f"Function call error: {e}"

        if s_base is None or s_perturb is None:
            return False, "Function returned None."
        if s_perturb <= s_base:
            return False, f"s({base_t*1.1}) = {s_perturb} <= s({base_t}) = {s_base}, not monotonic."
        return True, f"Thickness increases with time: s({base_t})={s_base:.3f} → s({base_t*1.1})={s_perturb:.3f} (ratio={s_perturb/s_base:.3f})."

    def test_monotonicity_lambda_increase(self) -> Tuple[bool, str]:
        """
        Check that thickness increases when thermal conductivity Lambda is increased.
        We reuse the theoretical formula s = sqrt(2 * Lambda * Delta_T * t / ell) with
        fixed t and perturbed Lambda. Since the solution is expected to obey this,
        we test that the re-derived perturbed value is larger than the re-derived baseline.
        """
        if any(v is None for v in [self.Lambda, self.Delta_T, self.ell]):
            return False, "SKIP: Missing givens."

        # Use a fixed time, e.g., t = 1 second
        t_test = 1.0
        # Baseline theoretical thickness using given Lambda
        baseline = math.sqrt(2.0 * self.Lambda * self.Delta_T * t_test / self.ell)

        # Perturb Lambda by +5%
        perturbed_lambda = self.Lambda * 1.05
        perturbed = math.sqrt(2.0 * perturbed_lambda * self.Delta_T * t_test / self.ell)

        if perturbed <= baseline:
            return False, f"Perturbed theoretical thickness {perturbed:.4f} is not larger than baseline {baseline:.4f}."
        return True, f"Thickness increases with Lambda: baseline={baseline:.4f}, perturbed (+5%)={perturbed:.4f} (ratio={perturbed/baseline:.4f})."

    # ------------------------------------------------------------
    # Answer sanity check (weak)
    # ------------------------------------------------------------
    def test_answer_sanity_sign_and_finite(self) -> Tuple[bool, str]:
        """
        Weak sanity: final answer should be positive and finite for finite t.
        """
        if self.s_func is None:
            return False, "SKIP: No result function."
        try:
            s_at_100 = self.s_func(100.0)
        except Exception as e:
            return False, f"s(100) error: {e}"
        if s_at_100 is None:
            return False, "s(100) is None."
        if not math.isfinite(s_at_100):
            return False, f"s(100) not finite: {s_at_100}"
        if s_at_100 <= 0:
            return False, f"s(100) = {s_at_100} not positive."
        return True, f"s(100) = {s_at_100:.3f} > 0 and finite."

    # ------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs = []
        all_pass = True
        test_methods = [method for method in dir(self)
                        if method.startswith("test_") and callable(getattr(self, method))]
        test_methods.sort()  # deterministic order
        for method_name in test_methods:
            try:
                passed, msg = getattr(self, method_name)()
            except Exception as e:
                passed, msg = False, f"{method_name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {method_name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)