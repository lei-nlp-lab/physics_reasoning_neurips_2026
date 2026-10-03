import sympy as sp
import math

class PhysicsChecker:
    """
    Physics auditor for the magnetic spin system problem.
    Checks thermodynamic consistency of free energy derivatives, convexity,
    and basic structure of piecewise and polynomial expressions.
    """

    def __init__(self, solution_instance):
        # Extract results and givens safely
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.givens = getattr(solution_instance, 'givens', {}) or {}

        # Physical constants (with None default)
        self.mu = self.givens.get('mu', getattr(solution_instance, 'mu', None))
        self.lam = self.givens.get('lambda', getattr(solution_instance, 'lambda', None))

        # Expected result keys
        self.B_piecewise = self.results.get('B_piecewise', None)
        self.d2F_dM2 = self.results.get('d2F_dM2', None)
        self.instability_example = self.results.get('instability_example', None)

        # For numeric evaluation of piecewise B
        self._B_func = None
        if self.B_piecewise is not None:
            if callable(self.B_piecewise):
                self._B_func = self.B_piecewise
            else:
                # Attempt to convert sympy Piecewise or dict to callable
                try:
                    if hasattr(self.B_piecewise, 'subs'):
                        # SymPy expression, lambdify
                        M = sp.symbols('M')
                        self._B_func = sp.lambdify(M, self.B_piecewise, 'numpy')
                    elif isinstance(self.B_piecewise, dict):
                        # Fallback: do not attempt conversion, skip advanced structure
                        self._B_func = None
                    else:
                        self._B_func = None
                except:
                    self._B_func = None

    # -------------------------------------------------------
    # Helper: evaluate B_piecewise at a given magnetization value
    # -------------------------------------------------------
    def _eval_B(self, M_val):
        if self._B_func is not None:
            try:
                return self._B_func(M_val)
            except:
                return None
        return None

    # -------------------------------------------------------
    # Helper: evaluate d2F_dM2 expression at a given M
    # -------------------------------------------------------
    def _eval_d2F(self, M_val):
        if self.d2F_dM2 is None:
            return None
        if callable(self.d2F_dM2):
            try:
                return self.d2F_dM2(M_val)
            except:
                return None
        # SymPy expression or string
        try:
            M_sym = sp.symbols('M')
            if isinstance(self.d2F_dM2, str):
                expr = sp.sympify(self.d2F_dM2)
            else:
                expr = self.d2F_dM2
            subs = expr.subs(M_sym, M_val)
            if subs.is_number:
                return float(subs)
            # If expression contains other symbols, try numeric evaluation
            return None
        except:
            return None

    # -------------------------------------------------------
    # Pre‑evaluation guard
    # -------------------------------------------------------
    def _has_mu_lambda(self):
        return self.mu is not None and self.lam is not None

    # =======================================================
    # INTERFACE TESTS
    # =======================================================
    def test_interface_B_piecewise_key(self):
        """Check that B_piecewise result exists."""
        if self.B_piecewise is None:
            return (False, "B_piecewise is None")
        return (True, "B_piecewise exists")

    def test_interface_B_piecewise_callable(self):
        """Check that B_piecewise is callable (or evaluable)."""
        if self._B_func is None and self.B_piecewise is not None:
            # If it's a dict or expression that we could not convert, warn but don't fail
            return (True, "B_piecewise not callable but present (ignoring advanced structure)")
        if self._B_func is None:
            return (False, "B_piecewise cannot be evaluated")
        # Try a simple evaluation
        test = self._eval_B(0.0)
        if test is None:
            return (False, "B_piecewise evaluation at M=0 returned None")
        return (True, f"B_piecewise callable, evaluation at 0 returned {test}")

    def test_interface_d2F_dM2_key(self):
        """Check that d2F_dM2 result exists."""
        if self.d2F_dM2 is None:
            return (False, "d2F_dM2 is None")
        return (True, "d2F_dM2 exists")

    def test_interface_d2F_dM2_evaluable(self):
        """Check that d2F_dM2 can be evaluated at some M."""
        val = self._eval_d2F(0.0)
        if val is None:
            return (False, "Cannot evaluate d2F_dM2 at M=0")
        return (True, f"d2F_dM2 at M=0 gives {val}")

    def test_interface_instability_example_key(self):
        """Check that instability_example exists and is a non‑empty string."""
        if self.instability_example is None:
            return (False, "instability_example is None")
        if not isinstance(self.instability_example, str) or len(self.instability_example.strip()) == 0:
            return (False, "instability_example is not a non‑empty string")
        return (True, "instability_example present and is a non‑empty string")

    # =======================================================
    # INVARIANT TESTS (theorem‑based)
    # =======================================================
    def test_invariant_convexity_polynomial_center(self):
        """
        For the polynomial free energy F = λ[(M/μ)^4 - (M/μ)^2],
        the second derivative is d2F/dM2 = (2λ/μ^2)(6M^2/μ^2 - 1).
        At M=0 it should be negative: -2λ/μ^2.
        """
        if self.d2F_dM2 is None or not self._has_mu_lambda():
            return (False, "SKIP: missing d2F_dM2 or mu/lambda")
        expected_center = -2.0 * self.lam / (self.mu**2)
        computed_center = self._eval_d2F(0.0)
        if computed_center is None:
            return (False, "Cannot evaluate d2F_dM2 at M=0")
        if not math.isclose(computed_center, expected_center, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"d2F_dM2 at M=0: expected {expected_center}, got {computed_center}")
        return (True, f"d2F_dM2 at M=0 matches negative value ({computed_center})")

    def test_invariant_convexity_polynomial_edge(self):
        """
        At M = μ (beyond instability region), d2F/dM2 should be positive.
        d2F/dM2 at M = μ: (2λ/μ^2)(6 - 1) = 10 λ / μ^2
        """
        if self.d2F_dM2 is None or not self._has_mu_lambda():
            return (False, "SKIP: missing d2F_dM2 or mu/lambda")
        expected_edge = 10.0 * self.lam / (self.mu**2)
        computed_edge = self._eval_d2F(self.mu)
        if computed_edge is None:
            return (False, "Cannot evaluate d2F_dM2 at M=μ")
        if not math.isclose(computed_edge, expected_edge, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"d2F_dM2 at M=μ: expected {expected_edge}, got {computed_edge}")
        return (True, f"d2F_dM2 at M=μ matches positive value ({computed_edge})")

    def test_invariant_derivative_piecewise_center(self):
        """
        In the center region |M/μ| ≤ 1/2, F=0 so dF/dM=0, hence B = M.
        Test for a sample M = 0.1μ.
        """
        if self.B_piecewise is None or not self._has_mu_lambda():
            return (False, "SKIP: B_piecewise or mu/lambda missing")
        M_test = 0.1 * self.mu
        B_val = self._eval_B(M_test)
        if B_val is None:
            return (False, "Cannot evaluate B at M=0.1μ")
        expected = M_test
        if not math.isclose(B_val, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"B(0.1μ) should be {expected}, got {B_val}")
        return (True, f"B(0.1μ) equals M ({B_val})")

    def test_invariant_derivative_piecewise_positive_linearity(self):
        """
        For M in (μ/2, μ], B - M should be proportional to (M/μ - 1/2).
        Test linearity by comparing two points.
        """
        if self.B_piecewise is None or not self._has_mu_lambda():
            return (False, "SKIP: B_piecewise or mu/lambda missing")
        M1 = 0.6 * self.mu
        M2 = 0.8 * self.mu
        B1 = self._eval_B(M1)
        B2 = self._eval_B(M2)
        if B1 is None or B2 is None:
            return (False, "Cannot evaluate B at sample points")
        x1 = M1 / self.mu
        x2 = M2 / self.mu
        # Test that (B - M) / (x - 0.5) is constant for x>0.5
        ratio1 = (B1 - M1) / (x1 - 0.5) if (x1 - 0.5) != 0 else None
        ratio2 = (B2 - M2) / (x2 - 0.5) if (x2 - 0.5) != 0 else None
        if ratio1 is None or ratio2 is None:
            return (False, "Division by zero in linearity test (x=0.5)")
        if not math.isclose(ratio1, ratio2, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"B - M not proportional to (x-0.5): ratios {ratio1} and {ratio2}")
        return (True, f"B - M linearly proportional to (x-0.5) with consistent ratio ~{ratio1:.3f}")

    def test_invariant_continuity_at_half(self):
        """
        B should be continuous at |M|/μ = 0.5.
        Compare B at M = 0.5μ - epsilon and 0.5μ + epsilon.
        """
        if self.B_piecewise is None or not self._has_mu_lambda():
            return (False, "SKIP: missing B_piecewise or mu/lambda")
        eps = 1e-3 * self.mu
        M_left = 0.5 * self.mu - eps
        M_right = 0.5 * self.mu + eps
        B_left = self._eval_B(M_left)
        B_right = self._eval_B(M_right)
        if B_left is None or B_right is None:
            return (False, "Cannot evaluate B near 0.5μ")
        if not math.isclose(B_left, B_right, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"B discontinuous at 0.5μ: left {B_left}, right {B_right}")
        return (True, f"B continuous at 0.5μ: both sides approximately {B_left}")

    # =======================================================
    # MODELING / TRACE TESTS
    # =======================================================
    def test_modeling_second_derivative_expression(self):
        """
        Check that the d2F_dM2 expression has the correct functional form:
        should be λ * (12 M^2 / μ^4 - 2 / μ^2 ).
        Evaluate at two independent M and compare ratios.
        """
        if self.d2F_dM2 is None or not self._has_mu_lambda():
            return (False, "SKIP: missing d2F_dM2 or mu/lambda")
        M1 = 0.3 * self.mu
        M2 = 0.7 * self.mu
        val1 = self._eval_d2F(M1)
        val2 = self._eval_d2F(M2)
        if val1 is None or val2 is None:
            return (False, "Cannot evaluate d2F_dM2 at sample points")
        # Expected ratio: (6*(M2/μ)^2 - 1) / (6*(M1/μ)^2 - 1)
        x1 = M1 / self.mu
        x2 = M2 / self.mu
        expected_ratio = (6*x2*x2 - 1) / (6*x1*x1 - 1)
        actual_ratio = val2 / val1
        if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Ratio of second derivatives {actual_ratio} does not match expected {expected_ratio}")
        return (True, f"Second derivative ratio matches expected form ({actual_ratio})")

    def test_modeling_B_piecewise_structure(self):
        """
        Check that B_piecewise has (or can be interpreted as) a piecewise function
        with at least two distinct regions.  We test by evaluating at three points:
        one in center, one in positive wing, one in negative wing.
        """
        if self.B_piecewise is None or not self._has_mu_lambda():
            return (False, "SKIP: missing B_piecewise or mu/lambda")
        # Pick: M/μ = 0 (center), 0.75 (positive), -0.75 (negative)
        B0 = self._eval_B(0.0)
        Bpos = self._eval_B(0.75 * self.mu)
        Bneg = self._eval_B(-0.75 * self.mu)
        if None in (B0, Bpos, Bneg):
            return (False, "Cannot evaluate B at sample points")
        # In the center, B = M, so B0 should be near 0.
        if abs(B0) > 1e-6 * self.mu:
            return (False, f"At M=0, B should be 0, got {B0}")
        # Positive wing: B - M should be positive (since derivative is positive)
        if not (Bpos - 0.75*self.mu > 0):
            return (False, f"At M=0.75μ, B - M should be positive, got {Bpos - 0.75*self.mu}")
        # Negative wing: B - M should be negative (since derivative is negative for negative M)
        if not (Bneg - (-0.75*self.mu) < 0):
            return (False, f"At M=-0.75μ, B - M should be negative, got {Bneg - (-0.75*self.mu)}")
        return (True, "B has proper piecewise structure with distinct regions")

    # =======================================================
    # MONOTONICITY TESTS (input perturbation)
    # =======================================================
    def test_monotonicity_convexity_increases_with_M_squared(self):
        """
        For the polynomial free energy, d2F/dM2 should increase when |M| increases.
        We perturb M from a small value to a larger value and check that d2F/dM2 increases.
        """
        if self.d2F_dM2 is None or not self._has_mu_lambda():
            return (False, "SKIP: missing d2F_dM2 or mu/lambda")
        M_small = 0.2 * self.mu
        M_large = 0.4 * self.mu
        val_small = self._eval_d2F(M_small)
        val_large = self._eval_d2F(M_large)
        if val_small is None or val_large is None:
            return (False, "Cannot evaluate d2F_dM2 at sample M")
        # Since expression is 2λ/μ^2*(6(M/μ)^2 - 1), it increases with |M|.
        if val_large <= val_small:
            return (False, f"d2F/dM2 did not increase: at M={M_small:.4f} -> {val_small:.4f}, at M={M_large:.4f} -> {val_large:.4f}")
        return (True, f"d2F/dM2 increased from {val_small:.4f} to {val_large:.4f}")

    def test_monotonicity_B_increases_with_M_in_wing(self):
        """
        For the piecewise function in the positive wing (M > μ/2), B should increase with M.
        Perturb M upward and check that B increases.
        """
        if self.B_piecewise is None or not self._has_mu_lambda():
            return (False, "SKIP: missing B_piecewise or mu/lambda")
        M1 = 0.6 * self.mu
        M2 = 0.7 * self.mu
        B1 = self._eval_B(M1)
        B2 = self._eval_B(M2)
        if B1 is None or B2 is None:
            return (False, "Cannot evaluate B at sample points")
        if B2 <= B1:
            return (False, f"B did not increase with M in wing: B({M1:.4f})={B1:.4f}, B({M2:.4f})={B2:.4f}")
        return (True, f"B increased from {B1:.4f} to {B2:.4f} as M increased")

    # =======================================================
    # WEAK ANSWER SANITY (at most one)
    # =======================================================
    def test_answer_sanity_finite_and_real(self):
        """
        Weak sanity check: B at a representative point should be finite and real.
        No hardcoded target values.
        """
        if self.B_piecewise is None or not self._has_mu_lambda():
            return (False, "SKIP: missing B_piecewise or mu/lambda")
        test_points = [0.0, 0.3*self.mu, 0.8*self.mu, -0.8*self.mu]
        for M_test in test_points:
            B_val = self._eval_B(M_test)
            if B_val is None:
                return (False, f"B evaluation at M={M_test} returned None")
            if not math.isfinite(B_val):
                return (False, f"B at M={M_test} is not finite: {B_val}")
        return (True, "B values are finite and real at sampled points")

    # =======================================================
    # RUN ALL TESTS
    # =======================================================
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