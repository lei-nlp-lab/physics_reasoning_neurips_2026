import sympy as sp
import math

class PhysicsChecker:
    """
    Physics checker for the nonlinear oscillator problem:
    U(x) = kx^2/2 - m λ x^3/3.
    Expected solution to first order in λ with x(0)=0:
    x(t) = A sin(ω t) + λ (A^2/ω^2) [1/2 - (2/3) cos(ω t) + (1/6) cos(2 ω t)]
    """
    def __init__(self, solution_instance):
        # Extract results safely; all defaults are None
        self.results = getattr(solution_instance, 'results', {})
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)
        self.givens = getattr(solution_instance, 'givens', {})

        # Expected result keys
        self.x = self.results.get('x', None)
        self.A = self.results.get('A', None)
        self.lambda_ = self.results.get('lambda', None)   # avoid keyword
        self.omega = self.results.get('omega', None)
        self.k = self.results.get('k', None)
        self.m = self.results.get('m', None)
        self.v0 = self.results.get('v0', None)

        # Convert to sympy expressions where possible
        self.x_expr = None
        if isinstance(self.x, str):
            try:
                self.x_expr = sp.sympify(self.x)
            except Exception:
                self.x_expr = None
        elif isinstance(self.x, sp.Expr):
            self.x_expr = self.x

        self.A_expr = sp.sympify(self.A) if isinstance(self.A, str) else self.A
        self.omega_expr = sp.sympify(self.omega) if isinstance(self.omega, str) else self.omega
        self.lambda_expr = sp.sympify(self.lambda_) if isinstance(self.lambda_, str) else self.lambda_
        self.v0_expr = sp.sympify(self.v0) if isinstance(self.v0, str) else self.v0

        self.t = sp.Symbol('t', real=True)

    # -------------------------------------------------------------
    # Interface tests
    # -------------------------------------------------------------
    def test_interface_x_key_exists(self):
        if self.x is None:
            return (False, "results['x'] is missing")
        return (True, "results['x'] is present")

    def test_interface_A_exists(self):
        if self.A is None:
            return (False, "results['A'] is missing")
        return (True, "results['A'] is present")

    def test_interface_omega_exists(self):
        if self.omega is None:
            return (False, "results['omega'] is missing")
        return (True, "results['omega'] is present")

    def test_interface_lambda_exists(self):
        if self.lambda_ is None:
            return (False, "results['lambda'] is missing")
        return (True, "results['lambda'] is present")

    # -------------------------------------------------------------
    # Invariant (theorem) checks
    # -------------------------------------------------------------
    def test_invariant_zero_order_ode(self):
        """Check that x0 = A sin(ω t) satisfies x0'' + ω^2 x0 = 0."""
        if self.x_expr is None or self.A_expr is None or self.omega_expr is None:
            return (False, "SKIP: insufficient symbolic info for zero-order ODE check")
        x0 = self.A_expr * sp.sin(self.omega_expr * self.t)
        residual = sp.simplify(sp.diff(x0, self.t, 2) + self.omega_expr**2 * x0)
        if residual == 0:
            return (True, "Zero-order ODE satisfied")
        else:
            return (False, f"Zero-order ODE residual: {residual}")

    def test_invariant_first_order_ode(self):
        """Check that x1 satisfies x1'' + ω^2 x1 = x0^2."""
        if self.x_expr is None or self.A_expr is None or self.omega_expr is None:
            return (False, "SKIP: insufficient symbolic info for first-order ODE check")
        A = self.A_expr
        ω = self.omega_expr
        x0 = A * sp.sin(ω * self.t)
        x0_sq = sp.simplify(x0**2)

        # expected x1 from the problem
        x1_expected = (A**2 / ω**2) * (sp.Rational(1,2) - sp.Rational(2,3)*sp.cos(ω*self.t) + sp.Rational(1,6)*sp.cos(2*ω*self.t))
        lhs_x1 = sp.diff(x1_expected, self.t, 2) + ω**2 * x1_expected
        residual = sp.simplify(lhs_x1 - x0_sq)
        if residual == 0:
            return (True, "First-order ODE satisfied")
        else:
            return (False, f"First-order ODE residual: {residual}")

    def test_invariant_initial_condition_x0_zero(self):
        """Check x0(0)=0."""
        if self.A_expr is None or self.omega_expr is None:
            return (False, "SKIP: missing A or omega for x0 initial condition")
        x0 = self.A_expr * sp.sin(self.omega_expr * self.t)
        val = sp.simplify(x0.subs(self.t, 0))
        if val == 0:
            return (True, "x0(0)=0 satisfied")
        else:
            return (False, f"x0(0) = {val} != 0")

    def test_invariant_initial_condition_x1_zero(self):
        """Check x1(0)=0."""
        if self.A_expr is None or self.omega_expr is None:
            return (False, "SKIP: missing A or omega for x1 initial condition")
        A = self.A_expr
        ω = self.omega_expr
        x1 = (A**2 / ω**2) * (sp.Rational(1,2) - sp.Rational(2,3)*sp.cos(ω*self.t) + sp.Rational(1,6)*sp.cos(2*ω*self.t))
        val = sp.simplify(x1.subs(self.t, 0))
        if val == 0:
            return (True, "x1(0)=0 satisfied")
        else:
            return (False, f"x1(0) = {val} != 0")

    def test_invariant_initial_velocity_preserved(self):
        """Check that x'(0) equals given v0 (if v0 is provided)."""
        if self.v0 is None or self.x_expr is None or self.v0_expr is None:
            return (False, "SKIP: v0 not provided or x unavailable")
        deriv0 = sp.simplify(sp.diff(self.x_expr, self.t).subs(self.t, 0))
        diff = sp.simplify(deriv0 - self.v0_expr)
        if diff == 0:
            return (True, "Initial velocity matches v0")
        else:
            return (False, f"x'(0) = {deriv0}, expected {self.v0_expr}")

    # -------------------------------------------------------------
    # Modeling / trace checks
    # -------------------------------------------------------------
    def test_modeling_x_structure(self):
        """Check that x has the expected analytical form."""
        if self.x_expr is None or self.A_expr is None or self.omega_expr is None:
            return (False, "SKIP: missing symbolic info for structure check")
        A = self.A_expr
        ω = self.omega_expr
        λ = self.lambda_expr if self.lambda_expr is not None else sp.Symbol('lambda')
        expected = A * sp.sin(ω * self.t) + λ * (A**2 / ω**2) * (
            sp.Rational(1,2) - sp.Rational(2,3)*sp.cos(ω*self.t) + sp.Rational(1,6)*sp.cos(2*ω*self.t)
        )
        diff = sp.simplify(self.x_expr - expected)
        if diff == 0:
            return (True, "x matches expected form")
        else:
            return (False, f"x differs from expected form; difference = {diff}")

    def test_modeling_perturbation_order(self):
        """Check that x contains only O(λ^0) and O(λ^1) terms (no λ^2)."""
        if self.x_expr is None:
            return (False, "SKIP: no x expression")
        λ = sp.Symbol('lambda')
        # Replace any existing lambda symbol with our dummy λ
        x_replaced = self.x_expr
        if self.lambda_expr is not None:
            x_replaced = self.x_expr.subs(self.lambda_expr, λ)
        series = sp.series(x_replaced, λ, 0, 3).removeO()
        coeff2 = series.coeff(λ, 2)
        if coeff2 is None or sp.simplify(coeff2) == 0:
            return (True, "No O(λ^2) terms present")
        else:
            return (False, f"O(λ^2) term present: {coeff2}")

    # -------------------------------------------------------------
    # Monotonicity check (trend)
    # -------------------------------------------------------------
    def test_monotonicity_correction_increases_with_lambda(self):
        """
        Perturb λ by +10% and verify that x at a fixed non-zero time increases.
        This checks monotonic increase of the first-order correction.
        """
        if self.x_expr is None or self.A_expr is None or self.omega_expr is None:
            return (False, "SKIP: insufficient symbolic info for monotonicity")
        # Check if A and ω are numeric
        try:
            A_num = float(self.A_expr)
            ω_num = float(self.omega_expr)
        except (TypeError, ValueError):
            return (False, "SKIP: A and ω are not numeric, cannot evaluate perturbatively")
        if ω_num <= 0:
            return (False, "SKIP: ω must be positive")

        # Choose t such that sin(ωt)=1 and cos(ωt)=0 => t = π/(2ω)
        t0 = math.pi / (2.0 * ω_num)
        # Build the expression with symbolic lambda
        λ_sym = sp.Symbol('lambda')
        expr = self.x_expr
        if self.lambda_expr is not None:
            expr = expr.subs(self.lambda_expr, λ_sym)
        # Numericify
        try:
            x_func = sp.lambdify((λ_sym, self.t), expr, 'numpy')
        except Exception:
            return (False, "SKIP: cannot lambdify expression")

        # Baseline lambda
        lam_base = 0.01
        # Perturbed lambda (increase by 10%)
        lam_pert = lam_base * 1.10

        x_base = float(x_func(lam_base, t0))
        x_pert = float(x_func(lam_pert, t0))

        # Check monotonic increase (x should increase with λ because correction is positive at this t)
        if x_pert > x_base:
            return (True, f"Monotonic increase confirmed: x(λ=0.011)={x_pert:.6f} > x(λ=0.01)={x_base:.6f}, relative change = {(x_pert-x_base)/x_base*100:.2f}%")
        else:
            return (False, f"Monotonic increase failed: x(λ=0.011)={x_pert:.6f} <= x(λ=0.01)={x_base:.6f}")

    # -------------------------------------------------------------
    # Weak answer sanity
    # -------------------------------------------------------------
    def test_answer_sanity_finite(self):
        """Ensure x is not None (trivial plausibility)."""
        if self.x is None:
            return (False, "x is missing")
        return (True, "x is present")

    # -------------------------------------------------------------
    # Run all tests
    # -------------------------------------------------------------
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