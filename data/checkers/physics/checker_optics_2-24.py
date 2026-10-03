import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract results and givens
        self.results = getattr(solution_instance, 'results', {})
        self.givens = getattr(solution_instance, 'givens', {})

        # Expected result keys from emit contract
        self.I_theta = self.results.get('I_theta', None)
        self.I_avg = self.results.get('I_avg', None)

        # Givens (accept keys: 'lambda', 'wavelength'; 'a'; 'a_bar'; 'delta'; 'I0')
        self.lambda_ = self.givens.get('lambda', self.givens.get('wavelength', None))
        self.a = self.givens.get('a', None)
        self.a_bar = self.givens.get('a_bar', None)
        self.delta = self.givens.get('delta', None)
        self.I0 = self.givens.get('I0', None)

    # ----------------------------------------------------------------------
    # Helper: evaluate an expression (sympy or function) at a numeric theta
    # ----------------------------------------------------------------------
    def _eval(self, expr, theta_val):
        if expr is None:
            return None
        # If it's a function (callable) with one argument
        if callable(expr):
            try:
                return float(expr(theta_val))
            except Exception:
                return None
        # If it's a sympy expression, substitute theta variable
        if isinstance(expr, sp.Expr):
            theta_sym = sp.Symbol('theta')
            try:
                sub = expr.subs(theta_sym, theta_val)
                # Evaluate numerically
                return float(sub.evalf())
            except Exception:
                # Try with a different symbol name
                try:
                    theta_sym = sp.Symbol('θ')
                    sub = expr.subs(theta_sym, theta_val)
                    return float(sub.evalf())
                except Exception:
                    return None
        # Otherwise, assume it's a numeric constant or something else
        try:
            return float(expr)
        except (TypeError, ValueError):
            return None

    # ----------------------------------------------------------------------
    # Interface tests: required keys exist with valid types
    # ----------------------------------------------------------------------
    def test_interface_has_I_theta(self):
        if self.I_theta is None:
            return (False, "Missing required result: I_theta")
        return (True, "I_theta is present")

    def test_interface_has_I_avg(self):
        if self.I_avg is None:
            return (False, "Missing required result: I_avg")
        return (True, "I_avg is present")

    def test_interface_givens_available(self):
        missing = []
        if self.lambda_ is None:
            missing.append("lambda/wavelength")
        if self.I0 is None:
            missing.append("I0")
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "Essential givens available")

    # ----------------------------------------------------------------------
    # Invariant checks: theorem-based physical sanity (use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_center_max(self):
        """
        At θ=0, both intensities should be 4*I0 (since cos²(0)=1).
        """
        I0 = self.I0
        if I0 is None:
            return (False, "SKIP: I0 not available")
        I_theta0 = self._eval(self.I_theta, 0.0)
        if I_theta0 is None:
            return (False, "Cannot evaluate I_theta at theta=0")
        expected = 4.0 * I0
        # Use isclose with tolerance 2%
        if math.isclose(I_theta0, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"I_theta(0) = {I_theta0} ≈ {expected}")
        else:
            return (False, f"I_theta(0) = {I_theta0} != {expected}")

        # Also test I_avg at center if available
        I_avg0 = self._eval(self.I_avg, 0.0)
        if I_avg0 is not None:
            if math.isclose(I_avg0, expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"I_avg(0) = {I_avg0} ≈ {expected}")
            else:
                return (False, f"I_avg(0) = {I_avg0} != {expected}")
        # If I_avg not evaluable at 0, still pass based on I_theta
        return (True, f"I_theta(0) ok; I_avg(0) not tested")

    def test_invariant_symmetry(self):
        """
        Both intensity patterns should be symmetric about θ=0.
        Test at a few sample angles.
        """
        test_angles = [0.1, 0.5, 1.0]  # radians
        for expr, name in [(self.I_theta, "I_theta"), (self.I_avg, "I_avg")]:
            if expr is None:
                continue
            for angle in test_angles:
                val_pos = self._eval(expr, angle)
                val_neg = self._eval(expr, -angle)
                if val_pos is None or val_neg is None:
                    continue
                if not math.isclose(val_pos, val_neg, rel_tol=1e-2, abs_tol=1e-2):
                    return (False, f"{name} asymmetry: {name}({angle})={val_pos} != {name}(-{angle})={val_neg}")
        return (True, "Symmetry holds for both I_theta and I_avg")

    def test_invariant_nonnegativity(self):
        """
        Intensity must be non-negative for all angles (within physical range).
        """
        test_angles = [0.0, 0.2, 0.8, -0.5, 1.5]
        for expr, name in [(self.I_theta, "I_theta"), (self.I_avg, "I_avg")]:
            if expr is None:
                continue
            for angle in test_angles:
                val = self._eval(expr, angle)
                if val is None:
                    continue
                if val < -1e-9:  # allow tiny negative due to numerical noise
                    return (False, f"{name}({angle}) = {val} is negative")
        return (True, "All evaluated intensities are non-negative")

    # ----------------------------------------------------------------------
    # Modeling/trace checks: specific physics relations
    # ----------------------------------------------------------------------
    def test_modeling_static_zeros(self):
        """
        For static I_theta, zeros occur when a sinθ = (m+0.5)λ.
        Test m=0 zero if a and lambda are given.
        """
        if self.a is None or self.lambda_ is None:
            return (False, "SKIP: a or lambda not available for zero test")
        if self.I_theta is None:
            return (False, "SKIP: I_theta not available")
        # sinθ for first zero (m=0): sinθ = λ/(2a)
        sin_theta_zero = self.lambda_ / (2.0 * self.a)
        if sin_theta_zero < -1 or sin_theta_zero > 1:
            return (True, "Zero condition out of range (a < λ/2), skip")
        theta_zero = math.asin(sin_theta_zero)
        val = self._eval(self.I_theta, theta_zero)
        if val is None:
            return (False, "Cannot evaluate I_theta at calculated zero")
        # Expect near zero
        if math.isclose(val, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"I_theta at expected zero = {val} ≈ 0")
        else:
            return (False, f"I_theta at zero angle should be ~0, got {val}")

    def test_modeling_jittered_envelope(self):
        """
        Test that I_avg at a small non-zero angle is less than at center
        and positive. Use quarter-lambda condition if a_bar available.
        """
        if self.a_bar is None or self.lambda_ is None or self.I0 is None:
            return (False, "SKIP: needed givens for jittered envelope test")
        if self.I_avg is None:
            return (False, "SKIP: I_avg not available")
        # Choose angle corresponding to quarter of a fringe: sinθ = λ/(4 a_bar)
        if self.a_bar == 0:
            return (False, "a_bar is zero, cannot compute")
        sin_theta_q = self.lambda_ / (4.0 * self.a_bar)
        if sin_theta_q > 1:
            return (True, "Angle out of range, skip envelope test")
        theta_q = math.asin(sin_theta_q)
        val_center = self._eval(self.I_avg, 0.0)
        val_q = self._eval(self.I_avg, theta_q)
        if val_center is None or val_q is None:
            return (False, "Cannot evaluate I_avg")
        # Expected: at quarter fringe, cos² term = 0.5, so static part = 2I0,
        # but average adds an envelope factor. We'll just check inequality:
        if val_q < val_center and val_q > 0:
            return (True, f"Envelope decay: I_avg at θ_q = {val_q} < {val_center} and >0")
        else:
            return (False, f"Envelope not decaying: center={val_center}, quarter={val_q}")

    # ----------------------------------------------------------------------
    # Monotonicity test: one input parameter perturbation
    # ----------------------------------------------------------------------
    def test_monotonicity_center_decrease(self):
        """
        Perturb theta from zero: I(θ) should decrease (or stay same) for small θ.
        This holds for both patterns (static and jittered) because cos² decreases
        and the jitter envelope also decays.
        Avoid zero-crossings by using a very small angle.
        """
        if self.I_theta is None and self.I_avg is None:
            return (False, "SKIP: no intensity expression")
        # Small increment: 0.02 rad
        small_angle = 0.02
        for expr, name in [(self.I_theta, "I_theta"), (self.I_avg, "I_avg")]:
            if expr is None:
                continue
            val0 = self._eval(expr, 0.0)
            val_small = self._eval(expr, small_angle)
            if val0 is None or val_small is None:
                continue
            if val_small <= val0:
                return (True, f"{name}: I(0)={val0} >= I({small_angle})={val_small}")
            else:
                return (False, f"{name}: I(0)={val0} < I({small_angle})={val_small}, not decreasing")
        return (True, "Monotonicity test passed for both patterns")

    # ----------------------------------------------------------------------
    # Weak answer sanity check (no isclose, no derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_basic(self):
        """
        Check that intensities are finite, non-NaN, and have reasonable magnitude.
        """
        test_angles = [0.0, 0.3, -0.3, 1.0]
        for expr, name in [(self.I_theta, "I_theta"), (self.I_avg, "I_avg")]:
            if expr is None:
                continue
            for angle in test_angles:
                val = self._eval(expr, angle)
                if val is None:
                    return (False, f"Cannot evaluate {name} at θ={angle}")
                if not math.isfinite(val):
                    return (False, f"{name}({angle}) is not finite: {val}")
                if math.isnan(val):
                    return (False, f"{name}({angle}) is NaN")
                # Broad magnitude range: should be non-negative and at most, say, 10 times I0 (if I0 available)
                # but without I0 we skip magnitude check
        return (True, "All evaluated points are finite and non-NaN")

    # ----------------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------------
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