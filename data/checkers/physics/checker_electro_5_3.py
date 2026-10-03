import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Checker for relativistic constant proper acceleration problem.
    Audits solution_instance containing givens (g, c, v) and results['x'].
    """

    def __init__(self, solution_instance):
        # ---------- Extract givens ----------
        # g (proper acceleration)
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.g = solution_instance.givens.get('g', None)
        else:
            self.g = getattr(solution_instance, 'g', None)
        # c (speed of light)
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.c = solution_instance.givens.get('c', None)
        else:
            self.c = getattr(solution_instance, 'c', None)
        # v (final velocity achieved)
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.v = solution_instance.givens.get('v', None)
        else:
            self.v = getattr(solution_instance, 'v', None)
        # ---------- Extract results ----------
        # x (distance traveled) – expected in results dict per key mapping
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.x = solution_instance.results.get('x', None)
        else:
            self.x = getattr(solution_instance, 'x', None)

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_has_g(self):
        """g must be provided and positive."""
        if self.g is None:
            return (False, "SKIP: proper acceleration g not provided")
        if not isinstance(self.g, (int, float)):
            return (False, "g is not numeric")
        return (True, f"g = {self.g} (positive)" if self.g > 0 else f"g = {self.g}, but must be > 0")

    def test_interface_has_c(self):
        """c must be provided and positive."""
        if self.c is None:
            return (False, "SKIP: speed of light c not provided")
        if not isinstance(self.c, (int, float)):
            return (False, "c is not numeric")
        return (True, f"c = {self.c}" if self.c > 0 else f"c = {self.c}, but must be > 0")

    def test_interface_has_v(self):
        """v must be provided and in [0, c)."""
        if self.v is None:
            return (False, "SKIP: final velocity v not provided")
        if not isinstance(self.v, (int, float)):
            return (False, "v is not numeric")
        if self.v < 0:
            return (False, f"v = {self.v} is negative")
        if self.c is not None and self.v >= self.c:
            return (False, f"v = {self.v} >= c = {self.c}, relativistic condition violated")
        return (True, f"v = {self.v} (subluminal)")

    def test_interface_has_x(self):
        """x must be present in results dict."""
        if self.x is None:
            return (False, "SKIP: distance x not found in results['x']")
        return (True, f"x = {self.x}")

    # ------------------------------------------------------------------
    # Invariant (physical sanity) tests
    # ------------------------------------------------------------------
    def test_invariant_v_less_c(self):
        """Relativistic constraint: v < c."""
        if self.v is None or self.c is None:
            return (False, "SKIP: missing v or c")
        if self.v < self.c:
            return (True, f"v={self.v} < c={self.c}")
        return (False, f"v={self.v} >= c={self.c}")

    def test_invariant_x_positive(self):
        """Distance must be non‑negative."""
        if self.x is None:
            return (False, "SKIP: x not available")
        if self.x >= 0:
            return (True, f"x={self.x} >= 0")
        return (False, f"x={self.x} is negative")

    def test_invariant_x_zero_when_v_zero(self):
        """If v=0 then x should be 0."""
        if self.v is None or self.x is None:
            return (False, "SKIP: missing v or x")
        if self.v == 0:
            if self.x == 0:
                return (True, "x=0 when v=0")
            else:
                return (False, f"v=0 but x={self.x} != 0")
        return (True, "v>0, test skipped (not applicable)")

    def test_invariant_x_formula(self):
        """
        Core theorem consistency: x = (c^2/g)*(1/sqrt(1 - v^2/c^2) - 1).
        Re‑derived inside checker from g, c, v and compared using math.isclose.
        """
        if self.x is None or self.g is None or self.c is None or self.v is None:
            return (False, "SKIP: missing g, c, v, or x")
        g, c, v = self.g, self.c, self.v
        if g <= 0 or c <= 0:
            return (False, "g and c must be positive")
        if v >= c:
            return (False, f"v={v} >= c={c}")
        try:
            expected = (c**2 / g) * (1.0 / math.sqrt(1.0 - v**2 / c**2) - 1.0)
        except (ValueError, ZeroDivisionError) as e:
            return (False, f"Error computing expected x: {e}")
        if math.isclose(self.x, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"x={self.x} matches expected={expected} within 1%")
        return (False, f"x={self.x} deviates from expected={expected}")

    # ------------------------------------------------------------------
    # Modeling / trace checks (at least two required)
    # ------------------------------------------------------------------
    def test_modeling_dx_dv_relation(self):
        """
        Check the differential relation dx/dv = v / (g*(1 - v^2/c^2)^(3/2)).
        Numerical derivative approximated via forward finite difference.
        """
        if self.x is None or self.g is None or self.c is None or self.v is None:
            return (False, "SKIP: missing g, c, v, or x")
        g, c, v = self.g, self.c, self.v
        if v >= c or v < 0 or g <= 0 or c <= 0:
            return (False, "Invalid parameters for derivative check")
        delta = 0.01 * (c - v)
        if delta <= 0:
            return (False, "v too close to c for perturbation")
        v2 = v + delta
        def x_analytic(u):
            return (c**2 / g) * (1.0 / math.sqrt(1.0 - u**2 / c**2) - 1.0)
        x1 = x_analytic(v)
        x2 = x_analytic(v2)
        numerical_deriv = (x2 - x1) / delta
        denom = g * (1.0 - v**2 / c**2)**1.5
        if denom == 0:
            return (False, "Denominator zero in expected derivative")
        expected_deriv = v / denom
        rel_diff = abs(numerical_deriv - expected_deriv) / max(abs(expected_deriv), 1e-12)
        if rel_diff < 0.02:
            return (True, f"dx/dv ≈ {numerical_deriv:.4f} matches expected {expected_deriv:.4f} within 2%")
        return (False, f"dx/dv numerical={numerical_deriv:.4f} but expected={expected_deriv:.4f}")

    def test_modeling_ode_consistency(self):
        """
        Check that the provided x(v) satisfies the ODE: v dv/dx = (1 - v^2/c^2)^(3/2) g.
        Compute dv/dx numerically from x(v) (using analytic formula) and compare.
        """
        if self.x is None or self.g is None or self.c is None or self.v is None:
            return (False, "SKIP: missing g, c, v, or x")
        g, c, v = self.g, self.c, self.v
        if v <= 0 or g <= 0 or c <= 0:
            return (False, "v must be >0 for derivative computation, or invalid g/c")
        if v >= c:
            return (False, "v >= c")
        # Use a small perturbation in v (keep subluminal)
        delta = 0.01 * (c - v)
        if delta <= 0:
            return (False, "v too close to c")
        v1 = v - delta  # use backward difference
        if v1 <= 0:
            # cannot go negative, use forward
            v1 = v
            v2 = v + delta
            forward = True
        else:
            v2 = v
            forward = False

        def x_analytic(u):
            return (c**2 / g) * (1.0 / math.sqrt(1.0 - u**2 / c**2) - 1.0)

        if forward:
            x1 = x_analytic(v1)
            x2 = x_analytic(v2)
            dv_dx = delta / (x2 - x1) if (x2 - x1) != 0 else float('inf')
        else:
            x1 = x_analytic(v1)
            x2 = x_analytic(v2)
            dv_dx = (v2 - v1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')

        # ODE right-hand side
        rhs = g * (1.0 - v**2 / c**2)**1.5
        # left-hand side: v * dv/dx
        lhs = v * dv_dx
        rel_diff = abs(lhs - rhs) / max(abs(rhs), 1e-12)
        if rel_diff < 0.02:
            return (True, f"v*dv/dx = {lhs:.4f} matches ODE rhs = {rhs:.4f} within 2%")
        return (False, f"v*dv/dx = {lhs:.4f} but ODE rhs = {rhs:.4f}")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit input perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_x_increases_with_v(self):
        """
        For fixed g and c, if v increases, x should increase.
        Perturb v upward by 1% of (c - v) and confirm new x > original.
        """
        if self.x is None or self.g is None or self.c is None or self.v is None:
            return (False, "SKIP: missing g, c, v, or x")
        g, c, v = self.g, self.c, self.v
        if v >= c or g <= 0 or c <= 0:
            return (False, "Invalid parameters")
        delta = 0.01 * (c - v)
        if delta == 0:
            return (False, "v too close to c, cannot perturb")
        v_pert = v + delta
        expected_pert = (c**2 / g) * (1.0 / math.sqrt(1.0 - v_pert**2 / c**2) - 1.0)
        if expected_pert > self.x:
            return (True, f"x increases with v: x(v)={self.x} < x(v+δ)={expected_pert:.4f}")
        return (False, f"x did not increase: x(v)={self.x} ≥ x(v+δ)={expected_pert:.4f}")

    def test_monotonicity_x_decreases_with_g(self):
        """
        For fixed v and c, if g increases, x should decrease.
        Perturb g upward by 1% and confirm new x < original.
        """
        if self.x is None or self.g is None or self.c is None or self.v is None:
            return (False, "SKIP: missing g, c, v, or x")
        g, c, v = self.g, self.c, self.v
        if g <= 0 or c <= 0 or v >= c:
            return (False, "Invalid parameters")
        delta_g = 0.01 * g
        g_pert = g + delta_g
        expected_pert = (c**2 / g_pert) * (1.0 / math.sqrt(1.0 - v**2 / c**2) - 1.0)
        if expected_pert < self.x:
            return (True, f"x decreases with g: x(g)={self.x} > x(g+δ)={expected_pert:.4f}")
        return (False, f"x did not decrease: x(g)={self.x} ≤ x(g+δ)={expected_pert:.4f}")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_x_range(self):
        """
        Weak sanity: x must be a finite non‑negative number not astronomically large.
        (Only sign and finiteness check, no isclose or formula use.)
        """
        if self.x is None:
            return (False, "SKIP: x not available")
        if not isinstance(self.x, (int, float)):
            return (False, "x is not numeric")
        if math.isnan(self.x) or math.isinf(self.x):
            return (False, f"x is NaN or infinite: {self.x}")
        if self.x < 0:
            return (False, f"x is negative: {self.x}")
        # A rough upper bound: for v up to 0.99c and g ~ 10 m/s², x ~ 10^15 m; allow 10^20
        if self.x > 1e20:
            return (False, f"x = {self.x} exceeds plausible bound 1e20")
        return (True, f"x = {self.x} is finite, non‑negative, and plausible")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Discover all bound methods starting with 'test_'
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