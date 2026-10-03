import math
import traceback
from typing import Tuple, List, Any

class PhysicsChecker:
    """
    Audits a solution for the entropy change problem:
    body of constant Cp in contact with reservoir at Tf.
    """
    def __init__(self, solution_instance: Any):
        # Safely extract givens and results
        givens = getattr(solution_instance, 'givens', {})
        results = getattr(solution_instance, 'results', {})
        self.Ti = givens.get('Ti', None)
        self.Tf = givens.get('Tf', None)
        self.Cp = givens.get('Cp', None)

        self.dS_total = results.get('dS_total', None)
        self.x = results.get('x', None)

        # Derived quantities (for checks only)
        self.dS_body = None
        self.dS_res = None
        if all(v is not None for v in (self.Cp, self.Ti, self.Tf)):
            if self.Tf > 0 and self.Ti > 0:
                self.dS_body = self.Cp * math.log(self.Tf / self.Ti)
                self.dS_res = -self.Cp * (self.Tf - self.Ti) / self.Tf

    # -----------------------------------------------------------------
    # Interface tests
    # -----------------------------------------------------------------
    def test_interface_required_results_exist(self) -> Tuple[bool, str]:
        """Check that dS_total and x are present in results."""
        if self.dS_total is None:
            return (False, "Missing required result 'dS_total'")
        if self.x is None:
            return (False, "Missing required result 'x'")
        return (True, "dS_total and x are present")

    def test_interface_results_are_numeric(self) -> Tuple[bool, str]:
        """Check that dS_total and x are numeric (float/int) and finite."""
        for name, val in [('dS_total', self.dS_total), ('x', self.x)]:
            if val is None:
                return (False, f"{name} is None – cannot check type")
            if not isinstance(val, (int, float)):
                return (False, f"{name} must be numeric, got {type(val).__name__}")
            if not math.isfinite(val):
                return (False, f"{name} is not finite")
        return (True, "dS_total and x are numeric and finite")

    def test_interface_givens_present_if_needed(self) -> Tuple[bool, str]:
        """At least Cp, Ti, Tf should be present for a meaningful solution."""
        if self.Cp is None:
            return (False, "Given 'Cp' is missing")
        if self.Ti is None:
            return (False, "Given 'Ti' is missing")
        if self.Tf is None:
            return (False, "Given 'Tf' is missing")
        return (True, "Required givens (Cp, Ti, Tf) are present")

    # -----------------------------------------------------------------
    # Invariant tests (theorem-based sanity)
    # -----------------------------------------------------------------
    def test_invariant_x_range(self) -> Tuple[bool, str]:
        """Check |x| < 1 from problem constraint."""
        if self.x is None:
            return (False, "SKIP: x not available")
        if abs(self.x) >= 1 - 1e-9:  # allow numerical tolerance
            return (False, f"|x| should be < 1, got |x| = {abs(self.x):.4f} >= 1")
        return (True, f"|x| = {abs(self.x):.4f} < 1")

    def test_invariant_dS_total_positivity(self) -> Tuple[bool, str]:
        """
        The total entropy change must be > 0 for any Tf != Ti.
        (Equilibrium implies Tf != Ti, so we expect strict positivity.)
        """
        if self.dS_total is None:
            return (False, "SKIP: dS_total not available")
        # Allow small negative due to numerics, but should be positive
        if self.dS_total <= -1e-9:
            return (False, f"dS_total should be > 0, got {self.dS_total:.6e}")
        return (True, f"dS_total = {self.dS_total:.6e} is positive (or zero within tolerance)")

    def test_invariant_dS_total_self_consistent(self) -> Tuple[bool, str]:
        """
        If we can compute dS_body and dS_res from givens, verify that
        dS_total (from results) equals dS_body + dS_res within tolerance.
        """
        if self.dS_body is None or self.dS_res is None:
            return (False, "SKIP: cannot compute dS_body and dS_res from givens")
        if self.dS_total is None:
            return (False, "SKIP: dS_total not available")
        expected_sum = self.dS_body + self.dS_res
        if not math.isclose(self.dS_total, expected_sum, rel_tol=1e-2):
            return (False, f"dS_total ({self.dS_total:.6e}) != dS_body+dS_res ({expected_sum:.6e})")
        return (True, "dS_total equals dS_body + dS_res")

    # -----------------------------------------------------------------
    # Modeling / trace tests (based on explanation steps)
    # -----------------------------------------------------------------
    def test_modeling_x_definition(self) -> Tuple[bool, str]:
        """Check that x = (Tf - Ti)/Tf."""
        if any(v is None for v in (self.Ti, self.Tf, self.x)):
            return (False, "SKIP: missing Ti, Tf, or x")
        expected_x = (self.Tf - self.Ti) / self.Tf
        if not math.isclose(self.x, expected_x, rel_tol=1e-2):
            return (False, f"x = {self.x:.4f}, expected (Tf-Ti)/Tf = {expected_x:.4f}")
        return (True, f"x matches (Tf-Ti)/Tf")

    def test_modeling_dS_total_expression(self) -> Tuple[bool, str]:
        """
        Verify that dS_total agrees with the derived formula:
        Cp * (ln(Tf/Ti) - (Tf-Ti)/Tf)
        """
        if any(v is None for v in (self.Cp, self.Ti, self.Tf, self.dS_total)):
            return (False, "SKIP: missing required quantities")
        expected = self.Cp * (math.log(self.Tf / self.Ti) - (self.Tf - self.Ti) / self.Tf)
        if not math.isclose(self.dS_total, expected, rel_tol=1e-2):
            return (False, f"dS_total = {self.dS_total:.6e}, expected formula gives {expected:.6e}")
        return (True, "dS_total matches derived expression")

    def test_modeling_dS_body_expression(self) -> Tuple[bool, str]:
        """If dS_body can be derived from givens, check consistency with total."""
        if self.dS_body is None or self.dS_total is None:
            return (False, "SKIP: cannot compute dS_body or dS_total missing")
        # dS_body = Cp*ln(Tf/Ti).  The total includes this plus dS_res.
        # We can check that dS_total - dS_body equals -Cp*(Tf-Ti)/Tf
        if any(v is None for v in (self.Cp, self.Ti, self.Tf)):
            return (False, "SKIP: missing givens")
        expected_res = -self.Cp * (self.Tf - self.Ti) / self.Tf
        if not math.isclose(self.dS_total - self.dS_body, expected_res, rel_tol=1e-2):
            return (False, "dS_total - dS_body does not match -Cp*(Tf-Ti)/Tf")
        return (True, "dS_body and dS_total relationship holds")

    # -----------------------------------------------------------------
    # Monotonicity tests (perturb inputs and compare)
    # -----------------------------------------------------------------
    def test_monotonicity_dS_total_scales_with_Cp(self) -> Tuple[bool, str]:
        """
        Since ΔS_total = Cp * f(Ti,Tf), scaling Cp by factor k should scale ΔS_total by k.
        Perturb Cp by +10% and check that dS_total scales proportionally.
        """
        if self.dS_total is None or self.Cp is None:
            return (False, "SKIP: dS_total or Cp not available")
        if math.isclose(self.Cp, 0.0, abs_tol=1e-12):
            return (False, "SKIP: Cp is zero, cannot scale")
        if any(v is None for v in (self.Ti, self.Tf)):
            return (False, "SKIP: Ti or Tf missing")

        scale = 1.1  # 10% increase
        Cp_new = self.Cp * scale
        dS_body_new = Cp_new * math.log(self.Tf / self.Ti)
        dS_res_new = -Cp_new * (self.Tf - self.Ti) / self.Tf
        dS_total_new = dS_body_new + dS_res_new

        # The new total should be scale times the old total (within numerical tolerance)
        if not math.isclose(dS_total_new / self.dS_total, scale, rel_tol=1e-2):
            return (False,
                    f"dS_total not scaling linearly with Cp: "
                    f"expected ratio={scale:.4f}, got ratio={dS_total_new/self.dS_total:.4f}")
        return (True, f"dS_total scales with Cp: ratio = {dS_total_new/self.dS_total:.4f} ≈ {scale}")

    def test_monotonicity_dS_total_increases_with_temp_diff(self) -> Tuple[bool, str]:
        """
        Perturb Tf to make |Tf-Ti| larger, and check that dS_total increases.
        For Tf > Ti, we increase Tf by a small positive factor.
        For Tf < Ti, we decrease Tf by a factor (making the negative difference larger).
        """
        if any(v is None for v in (self.Ti, self.Tf, self.dS_total)):
            return (False, "SKIP: missing Ti, Tf, or dS_total")
        if math.isclose(self.Ti, 0.0, abs_tol=1e-12) or math.isclose(self.Tf, 0.0, abs_tol=1e-12):
            return (False, "SKIP: zero temperature encountered")
        # Determine direction: increase absolute temperature difference
        if self.Tf > self.Ti:
            # Increase Tf => larger positive difference
            Tf_new = self.Tf * 1.01  # 1% increase
        else:
            # Decrease Tf => larger negative difference (more negative Tf-Ti)
            # But we must keep Tf positive and |x|<1. Decrease by 1% (if Tf < Ti, decrease makes it smaller)
            # Actually if Tf < Ti, decreasing Tf makes |Tf-Ti| larger.
            Tf_new = self.Tf * 0.99  # 1% decrease
        # Check that Tf_new > 0 and |(Tf_new - Ti)/Tf_new| < 1 (should hold)
        if Tf_new <= 0:
            return (False, "Perturbed Tf became non-positive")

        # Compute new dS_total using derived formula
        dS_total_new = self.Cp * (math.log(Tf_new / self.Ti) - (Tf_new - self.Ti) / Tf_new)
        # This should be larger than baseline (since positive and difference increased)
        if dS_total_new <= self.dS_total * (1 - 1e-9):
            return (False,
                    f"dS_total did not increase when temperature difference increased: "
                    f"baseline {self.dS_total:.6e} -> perturbed {dS_total_new:.6e}")
        return (True,
                f"dS_total increases with |Tf-Ti|: baseline {self.dS_total:.6e} -> perturbed {dS_total_new:.6e}")

    # -----------------------------------------------------------------
    # Answer sanity (weak, at most one)
    # -----------------------------------------------------------------
    def test_answer_sanity_finite_and_reasonable(self) -> Tuple[bool, str]:
        """
        Weak sanity: dS_total and x must be finite, not NaN, and x within (-1,1).
        No re-derivation or comparison to expected formula.
        """
        for name, val in [('dS_total', self.dS_total), ('x', self.x)]:
            if val is None:
                return (False, f"SKIP: {name} not present")
            if not isinstance(val, (int, float)):
                return (False, f"{name} is not numeric (type {type(val).__name__})")
            if not math.isfinite(val):
                return (False, f"{name} is not finite")
            if name == 'x' and (abs(val) >= 1 - 1e-12):
                return (False, f"x = {val:.4f} is outside required range |x| < 1")
        # Check that dS_total is not extremely large in magnitude (bad numerical)
        if abs(self.dS_total) > 1e100:
            return (False, "dS_total is unreasonably large (>1e100)")
        return (True, "dS_total and x are finite, x in range, dS_total not huge")

    # -----------------------------------------------------------------
    # Run all tests
    # -----------------------------------------------------------------
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
                passed, msg = (False, f"{name} CRASHED: {e}\n{traceback.format_exc()}")
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)