import math
import traceback
from sympy import sympify, Symbol, sin, simplify, diff

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution_instance = solution_instance
        # Safe extraction of results (None defaults)
        self.results = getattr(solution_instance, 'results', {})
        self.lambda_general = self.results.get('lambda_general', None)
        self.mono_possible = self.results.get('mono_possible', None)

        # Safe extraction of givens (may be missing)
        self.givens = getattr(solution_instance, 'givens', {})
        self.N = self.givens.get('N', None) if isinstance(self.givens, dict) else None
        self.theta = self.givens.get('theta', None) if isinstance(self.givens, dict) else None
        self.lambda1 = self.givens.get('lambda1', None) if isinstance(self.givens, dict) else None

        # Attempt to compute sin(theta) if both exist
        self.sin_theta = None
        if self.theta is not None:
            try:
                self.sin_theta = math.sin(self.theta)
            except Exception:
                self.sin_theta = None

        # Helper symbols for symbolic checks
        self.N_sym = Symbol('N', positive=True)
        self.theta_sym = Symbol('theta', real=True)
        self.m_sym = Symbol('m', integer=True, positive=True)

        # Parse lambda_general into sympy expression if possible
        self._lambda_expr = None
        if self.lambda_general is not None:
            try:
                self._lambda_expr = sympify(str(self.lambda_general))
            except Exception:
                self._lambda_expr = None

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that both expected result keys are present."""
        if self.lambda_general is None and self.mono_possible is None:
            return (False, "Both 'lambda_general' and 'mono_possible' are missing")
        if self.lambda_general is None:
            return (False, "'lambda_general' is missing")
        if self.mono_possible is None:
            return (False, "'mono_possible' is missing")
        return (True, "Both required keys are present")

    def test_interface_key_types(self):
        """Check that lambda_general is a string/sympifyable and mono_possible is bool."""
        if self.lambda_general is None:
            return (False, "SKIP: lambda_general is missing")
        if self.mono_possible is None:
            return (False, "SKIP: mono_possible is missing")

        # lambda_general should be a string or numeric or sympy expression
        if not isinstance(self.lambda_general, (str, int, float)):
            if not hasattr(self.lambda_general, 'free_symbols'):  # sympy expression
                return (False, "lambda_general has unexpected type")
        if not isinstance(self.mono_possible, bool):
            return (False, f"mono_possible should be bool, got {type(self.mono_possible).__name__}")
        return (True, "Key types are acceptable")

    # ------------------------------------------------------------------
    # Invariant checks (theorem-based)
    # ------------------------------------------------------------------
    def test_invariant_littrow_holds_symbolically(self):
        """Verify that the symbolic expression satisfies m*λ = 2 sin(θ)/N."""
        if self._lambda_expr is None:
            return (False, "SKIP: Cannot parse lambda_general as sympy expression")
        left = self.m_sym * self._lambda_expr
        right = 2 * sin(self.theta_sym) / self.N_sym
        if simplify(left - right) == 0:
            return (True, "Symbolic Littrow condition holds: m*λ = 2 sinθ / N")
        return (False, "Symbolically, m*λ does not equal 2 sinθ / N")

    def test_invariant_littrow_numerically(self):
        """Numerical check of Littrow condition using supplied N and θ."""
        if self._lambda_expr is None or self.N is None or self.theta is None:
            return (False, "SKIP: Missing N, θ, or parseable lambda_general")
        for m_test in [1, 2, 3]:
            try:
                subs_dict = {self.N_sym: self.N, self.theta_sym: self.theta, self.m_sym: m_test}
                expr_val = float(self._lambda_expr.subs(subs_dict))
            except Exception:
                continue
            # Re-derive the Littrow value from physics: λ = 2 sinθ / (m N)
            phys_val = 2 * math.sin(self.theta) / (m_test * self.N)
            if math.isclose(expr_val, phys_val, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Numerical Littrow condition holds for m={m_test}: {expr_val} ≈ {phys_val}")
        return (False, "Numerical Littrow condition failed for m=1,2,3 (error > 1%)")

    def test_invariant_mono_possible(self):
        """Check that mono_possible is False (both wavelengths diffract together)."""
        if self.mono_possible is None:
            return (False, "SKIP: mono_possible not available")
        if not isinstance(self.mono_possible, bool):
            return (False, f"mono_possible should be bool, got {type(self.mono_possible).__name__}")
        if self.mono_possible:
            return (False, "mono_possible should be False, but it is True")
        return (True, "mono_possible is False as expected – both orders diffract simultaneously")

    # ------------------------------------------------------------------
    # Modeling quality checks (trace consistency)
    # ------------------------------------------------------------------
    def test_modeling_lambda_contains_required_symbols(self):
        """Check that the expression explicitly contains N, θ, and m."""
        if self._lambda_expr is None:
            return (False, "SKIP: Cannot parse lambda_general")
        syms = self._lambda_expr.free_symbols
        if self.N_sym not in syms:
            return (False, "Expression does not contain N")
        if self.theta_sym not in syms:
            return (False, "Expression does not contain θ")
        if self.m_sym not in syms:
            return (False, "Expression does not contain order m")
        return (True, "Expression includes N, θ, m as expected")

    def test_modeling_lambda_contains_sin_theta(self):
        """Specifically check that sin(θ) appears in the expression (not just θ)."""
        if self._lambda_expr is None:
            return (False, "SKIP: Cannot parse lambda_general")
        if self._lambda_expr.has(sin(self.theta_sym)):
            return (True, "sin(θ) appears in lambda expression")
        return (False, "sin(θ) not directly found in lambda expression")

    # ------------------------------------------------------------------
    # Monotonicity checks (trend analysis with explicit perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_lambda_decreases_with_m(self):
        """Compute λ for m=1 and m=2; verify λ(2) < λ(1)."""
        if self._lambda_expr is None:
            return (False, "SKIP: Cannot parse lambda_general")
        if self.N is None or self.theta is None:
            return (False, "SKIP: Missing N or θ for numerical perturbation")
        try:
            subs1 = {self.N_sym: self.N, self.theta_sym: self.theta, self.m_sym: 1}
            lambda1 = float(self._lambda_expr.subs(subs1))
            subs2 = {self.N_sym: self.N, self.theta_sym: self.theta, self.m_sym: 2}
            lambda2 = float(self._lambda_expr.subs(subs2))
            if lambda2 < lambda1:
                return (True, f"λ decreases with m: λ(1)={lambda1}, λ(2)={lambda2}")
            else:
                return (False, f"λ does not decrease: λ(1)={lambda1}, λ(2)={lambda2}")
        except Exception as e:
            return (False, f"Perturbation test crashed: {e}")

    def test_monotonicity_lambda_increases_with_sin_theta(self):
        """Perturb sinθ by +10% and verify λ increases (using fixed m=1)."""
        if self._lambda_expr is None:
            return (False, "SKIP: Cannot parse lambda_general")
        if self.N is None or self.theta is None:
            return (False, "SKIP: Missing N or θ for numerical perturbation")
        try:
            # Baseline
            subs_base = {self.N_sym: self.N, self.theta_sym: self.theta, self.m_sym: 1}
            lambda_base = float(self._lambda_expr.subs(subs_base))
            # Perturb theta to increase sinθ by ~10% (small angle approximation not needed)
            theta_new = math.asin(1.1 * math.sin(self.theta))
            if math.isnan(theta_new):  # sinθ may exceed 1
                theta_new = math.pi/2  # cap to 90°
            subs_pert = {self.N_sym: self.N, self.theta_sym: theta_new, self.m_sym: 1}
            lambda_pert = float(self._lambda_expr.subs(subs_pert))
            if lambda_pert > lambda_base:
                return (True, f"λ increases with sinθ: base={lambda_base}, perturbed={lambda_pert}")
            else:
                return (False, f"λ does not increase: base={lambda_base}, perturbed={lambda_pert}")
        except Exception as e:
            return (False, f"Perturbation test crashed: {e}")

    # ------------------------------------------------------------------
    # Answer sanity (weak sanity only)
    # ------------------------------------------------------------------
    def test_answer_sanity_lambda_nonempty(self):
        """Weak check: lambda_general is not None and not an empty string."""
        if self.lambda_general is None:
            return (False, "lambda_general is None")
        if isinstance(self.lambda_general, str) and self.lambda_general.strip() == "":
            return (False, "lambda_general is an empty string")
        return (True, "lambda_general is non-empty")

    # ------------------------------------------------------------------
    # Master runner
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
                passed = False
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)