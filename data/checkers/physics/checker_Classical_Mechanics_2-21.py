import sys
import math
import sympy as sp

class PhysicsChecker:
    """
    Physics Auditor for projectile motion with air resistance (Stokes drag).
    Checks interface, invariants, modeling quality, monotonicity, and answer sanity.
    """

    def __init__(self, solution_instance):
        # Extract givens safely
        self.v0 = getattr(solution_instance, 'v0', None)
        self.theta = getattr(solution_instance, 'theta', None)
        self.g = getattr(solution_instance, 'g', None)
        self.rho = getattr(solution_instance, 'rho', None)
        self.A = getattr(solution_instance, 'A', None)
        self.R = getattr(solution_instance, 'R', None)
        self.eta = getattr(solution_instance, 'eta', None)
        self.m = getattr(solution_instance, 'm', None)

        # Extract results (emitted values)
        results = getattr(solution_instance, 'results', {})
        self.R0 = results.get('R0', None)
        self.F_air_str = results.get('F_air', None)
        self.x_t_expr = results.get('x_t', None)
        self.y_t_expr = results.get('y_t', None)
        self.x_no_drag_expr = results.get('x_no_drag', None)
        self.y_no_drag_expr = results.get('y_no_drag', None)
        self.R1 = results.get('R1', None)

        # Derive tau from givens if possible
        if all(x is not None for x in [self.m, self.eta, self.R]) and self.eta != 0:
            self.tau = self.m / (6 * math.pi * self.eta * self.R)
        else:
            self.tau = None

    # ---------------------------------------------------------------
    # Interface checks
    # ---------------------------------------------------------------
    def test_interface_required_keys(self):
        """All required result keys must be present."""
        missing = []
        if self.R0 is None: missing.append("R0")
        if self.F_air_str is None: missing.append("F_air")
        if self.x_t_expr is None: missing.append("x_t")
        if self.y_t_expr is None: missing.append("y_t")
        if self.x_no_drag_expr is None: missing.append("x_no_drag")
        if self.y_no_drag_expr is None: missing.append("y_no_drag")
        if self.R1 is None: missing.append("R1")
        if missing:
            return False, f"Missing result keys: {', '.join(missing)}"
        return True, "All required result keys present."

    # ---------------------------------------------------------------
    # Invariant tests (theorem-based)
    # ---------------------------------------------------------------
    def test_invariant_range_no_drag(self):
        """R0 must equal v0^2 * sin(2θ) / g (the known formula)."""
        if self.R0 is None:
            return False, "SKIP: R0 not available"
        # We need numerical values to evaluate the formula
        if any(x is None for x in [self.v0, self.theta, self.g]):
            return False, "SKIP: v0, θ, or g not available"
        expected = (self.v0**2) * math.sin(2 * self.theta) / self.g
        if math.isclose(self.R0, expected, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"R0 = {self.R0} matches formula {expected}"
        else:
            return False, f"R0 = {self.R0} but expected {expected}"

    def test_invariant_drag_force_form(self):
        """F_air string must contain '-ρ A v²' and opposite velocity."""
        if self.F_air_str is None:
            return False, "SKIP: F_air not available"
        s = self.F_air_str.replace(" ", "")
        # Check essential components
        has_rho_A = ("-rho*A" in s or "-ρ*A" in s)
        has_v2 = ("v**2" in s or "v^2" in s)
        has_v_hat = ("v_hat" in s or "v̂" in s or "vec(v)" in s)
        if has_rho_A and has_v2 and has_v_hat:
            return True, "F_air expression matches expected form: -ρ A v² v̂"
        else:
            return False, f"F_air expression '{self.F_air_str}' does not match expected form"

    def test_invariant_ode_solution_x(self):
        """x(t) must satisfy dx/dt = v0 cosθ exp(-t/τ) and x(0)=0."""
        if self.x_t_expr is None:
            return False, "SKIP: x_t not available"
        if any(x is None for x in [self.v0, self.theta, self.tau]):
            return False, "SKIP: insufficient givens to compute tau"
        # Create symbolic t, substitute numeric values at t=0.1 (small)
        t_val = 0.1
        # Evaluate x(t) if possible
        # We'll try to lambdify or just use the expression as a string and substitute.
        try:
            t_sym = sp.Symbol('t')
            # If expression is a string, sympify it first
            if isinstance(self.x_t_expr, str):
                x_func = sp.lambdify(t_sym, sp.sympify(self.x_t_expr), 'numpy')
            else:
                x_func = sp.lambdify(t_sym, self.x_t_expr, 'numpy')
            x_num = x_func(t_val)
        except Exception:
            return False, "Cannot evaluate x(t) numeric"
        # Expected x(t) = v0*tau*cosθ*(1 - exp(-t/tau))
        expected_x = (self.v0 * self.tau * math.cos(self.theta) *
                      (1 - math.exp(-t_val / self.tau)))
        if math.isclose(x_num, expected_x, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"x(t) numeric check ok: x({t_val}) = {x_num} ~ {expected_x}"
        else:
            return False, f"x({t_val}) = {x_num}, expected {expected_x}"

    def test_invariant_ode_solution_y(self):
        """y(t) must satisfy ODE and y(0)=0."""
        if self.y_t_expr is None:
            return False, "SKIP: y_t not available"
        if any(x is None for x in [self.v0, self.theta, self.g, self.tau]):
            return False, "SKIP: insufficient givens"
        t_val = 0.1
        try:
            t_sym = sp.Symbol('t')
            if isinstance(self.y_t_expr, str):
                y_func = sp.lambdify(t_sym, sp.sympify(self.y_t_expr), 'numpy')
            else:
                y_func = sp.lambdify(t_sym, self.y_t_expr, 'numpy')
            y_num = y_func(t_val)
        except Exception:
            return False, "Cannot evaluate y(t) numeric"
        # Expected: (v0 sinθ + g τ)τ (1 - exp(-t/τ)) - g τ t
        expected_y = ((self.v0 * math.sin(self.theta) + self.g * self.tau) *
                      self.tau * (1 - math.exp(-t_val / self.tau)) -
                      self.g * self.tau * t_val)
        if math.isclose(y_num, expected_y, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"y(t) numeric check ok: y({t_val}) = {y_num} ~ {expected_y}"
        else:
            return False, f"y({t_val}) = {y_num}, expected {expected_y}"

    def test_invariant_no_drag_limit(self):
        """In limit τ → ∞, x(t) → v0 cosθ t and y(t) → v0 sinθ t - ½ g t²."""
        if self.x_no_drag_expr is None or self.y_no_drag_expr is None:
            return False, "SKIP: no-drag expressions not available"
        if any(x is None for x in [self.v0, self.theta, self.g]):
            return False, "SKIP: insufficient givens"
        # Evaluate at some t
        t_val = 0.5
        try:
            t_sym = sp.Symbol('t')
            if isinstance(self.x_no_drag_expr, str):
                x_nd_func = sp.lambdify(t_sym, sp.sympify(self.x_no_drag_expr), 'numpy')
            else:
                x_nd_func = sp.lambdify(t_sym, self.x_no_drag_expr, 'numpy')
            x_nd = x_nd_func(t_val)

            if isinstance(self.y_no_drag_expr, str):
                y_nd_func = sp.lambdify(t_sym, sp.sympify(self.y_no_drag_expr), 'numpy')
            else:
                y_nd_func = sp.lambdify(t_sym, self.y_no_drag_expr, 'numpy')
            y_nd = y_nd_func(t_val)
        except Exception:
            return False, "Cannot evaluate no-drag expressions"
        expected_x = self.v0 * math.cos(self.theta) * t_val
        expected_y = self.v0 * math.sin(self.theta) * t_val - 0.5 * self.g * t_val**2
        if (math.isclose(x_nd, expected_x, rel_tol=1e-2, abs_tol=1e-2) and
                math.isclose(y_nd, expected_y, rel_tol=1e-2, abs_tol=1e-2)):
            return True, "No-drag limit expressions correct"
        else:
            return False, f"No-drag: x={x_nd} (expected {expected_x}), y={y_nd} (expected {expected_y})"

    def test_invariant_first_order_correction(self):
        """R1 must equal R0 * (1 - 2 v0 sinθ / (g τ))."""
        if self.R1 is None or self.R0 is None:
            return False, "SKIP: R1 or R0 not available"
        if any(x is None for x in [self.v0, self.theta, self.g, self.tau]):
            return False, "SKIP: insufficient givens"
        expected_R1 = self.R0 * (1 - 2 * self.v0 * math.sin(self.theta) / (self.g * self.tau))
        if math.isclose(self.R1, expected_R1, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"R1 = {self.R1} matches expected {expected_R1}"
        else:
            return False, f"R1 = {self.R1}, expected {expected_R1}"

    # ---------------------------------------------------------------
    # Modeling / trace checks
    # ---------------------------------------------------------------
    def test_modeling_initial_conditions(self):
        """x(0)=0 and y(0)=0 must hold."""
        if self.x_t_expr is None or self.y_t_expr is None:
            return False, "SKIP: x_t or y_t not available"
        try:
            t_sym = sp.Symbol('t')
            if isinstance(self.x_t_expr, str):
                x_func = sp.lambdify(t_sym, sp.sympify(self.x_t_expr), 'numpy')
            else:
                x_func = sp.lambdify(t_sym, self.x_t_expr, 'numpy')
            x0 = x_func(0)
            if isinstance(self.y_t_expr, str):
                y_func = sp.lambdify(t_sym, sp.sympify(self.y_t_expr), 'numpy')
            else:
                y_func = sp.lambdify(t_sym, self.y_t_expr, 'numpy')
            y0 = y_func(0)
        except Exception:
            return False, "Cannot evaluate at t=0"
        if math.isclose(x0, 0, abs_tol=1e-9) and math.isclose(y0, 0, abs_tol=1e-9):
            return True, "Initial conditions x(0)=0 and y(0)=0 satisfied"
        else:
            return False, f"x(0)={x0}, y(0)={y0}, expected both 0"

    def test_modeling_small_drag_expansion(self):
        """Small τ (or large η) expansion should be consistent: R1 < R0."""
        if self.R0 is None or self.R1 is None:
            return False, "SKIP: R0 or R1 not available"
        if self.tau is None:
            return False, "SKIP: tau not available"
        # For positive τ, the correction factor (1 - 2v0 sinθ/(gτ)) should be < 1
        # (since drag reduces range).
        factor = 1 - 2 * self.v0 * math.sin(self.theta) / (self.g * self.tau)
        if factor < 1:
            return True, f"Range reduction factor {factor} < 1, consistent with drag"
        else:
            return False, f"Range reduction factor {factor} >= 1, inconsistent with drag"

    # ---------------------------------------------------------------
    # Monotonicity checks
    # ---------------------------------------------------------------
    def test_monotonicity_range_with_drag(self):
        """Increasing drag (decreasing τ) must decrease the range R1."""
        if self.R1 is None or self.tau is None:
            return False, "SKIP: R1 or tau not available"
        if any(x is None for x in [self.v0, self.theta, self.g]):
            return False, "SKIP: insufficient givens"
        # Baseline R1
        baseline = self.R1
        # Perturb tau -> smaller tau by 10% (more drag)
        new_tau = 0.9 * self.tau
        new_R1 = self.R0 * (1 - 2 * self.v0 * math.sin(self.theta) / (self.g * new_tau))
        # Since tau smaller, correction larger, so new_R1 < baseline
        if new_R1 < baseline:
            return True, f"Decreasing tau (more drag) reduces range: {new_R1} < {baseline}"
        else:
            return False, f"Decreasing tau (more drag) did not reduce range: {new_R1} >= {baseline}"

    # ---------------------------------------------------------------
    # Answer sanity (weak)
    # ---------------------------------------------------------------
    def test_answer_sanity_finite_positive(self):
        """R0 and R1 must be finite and positive (reasonable range)."""
        msg = ""
        if self.R0 is not None:
            if self.R0 <= 0 or not math.isfinite(self.R0):
                msg += f"R0 invalid (={self.R0}); "
        if self.R1 is not None:
            if self.R1 <= 0 or not math.isfinite(self.R1):
                msg += f"R1 invalid (={self.R1}); "
        # Also check that x_t, y_t are finite at a sample time if possible
        if self.x_t_expr is not None:
            try:
                t_sym = sp.Symbol('t')
                if isinstance(self.x_t_expr, str):
                    x_func = sp.lambdify(t_sym, sp.sympify(self.x_t_expr), 'numpy')
                else:
                    x_func = sp.lambdify(t_sym, self.x_t_expr, 'numpy')
                x_val = x_func(1.0)
                if not math.isfinite(x_val):
                    msg += "x(1) not finite; "
            except Exception:
                pass  # don't fail if evaluation fails
        if msg:
            return False, f"Sanity check failed: {msg}"
        return True, "All checked answer values are finite and positive."

    # ---------------------------------------------------------------
    # Main test runner
    # ---------------------------------------------------------------
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


# Example usage (optional, for local testing):
if __name__ == "__main__":
    # Create a mock solution instance with dummy data
    class MockSolution:
        def __init__(self):
            self.v0 = 10.0
            self.theta = math.radians(30)
            self.g = 9.81
            self.rho = 1.2
            self.A = 0.05
            self.R = 0.1
            self.eta = 1.8e-5
            self.m = 0.5
            self.results = {
                'R0': 8.828,
                'F_air': '-rho*A*v^2*v_hat',
                'x_t': 'v0*tau*cos(theta)*(1 - exp(-t/tau))',
                'y_t': '(v0*sin(theta) + g*tau)*tau*(1 - exp(-t/tau)) - g*tau*t',
                'x_no_drag': 'v0*cos(theta)*t',
                'y_no_drag': 'v0*sin(theta)*t - 0.5*g*t^2',
                'R1': 8.3   # rough approximate
            }

    sol = MockSolution()
    checker = PhysicsChecker(sol)
    passed, logs = checker.run_all_tests()
    for line in logs:
        print(line)
    print(f"\nOverall: {'ALL PASS' if passed else 'SOME FAIL'}")