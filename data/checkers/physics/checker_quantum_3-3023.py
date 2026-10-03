import math
import sympy as sp

class PhysicsChecker:
    """
    Physics checker for electron spin precession in a uniform magnetic field.
    Verifies that the provided solution satisfies the physical constraints from
    Ehrenfest's theorem: probability conservation, correct expectation value
    relations, and correct functional forms.
    """

    def __init__(self, solution_instance):
        # Safely extract results from solution_instance, which may be a dict or an object.
        if isinstance(solution_instance, dict):
            results = solution_instance.get('results', solution_instance)
        else:
            results_attr = getattr(solution_instance, 'results', None)
            results = results_attr if isinstance(results_attr, dict) else {}

        self.results = results

        # Use None for any missing key – never default to a numeric constant.
        self.P_x_up = self.results.get('P_x_up', None)
        self.P_x_down = self.results.get('P_x_down', None)
        self.P_y_up = self.results.get('P_y_up', None)
        self.P_y_down = self.results.get('P_y_down', None)
        self.P_z_up = self.results.get('P_z_up', None)
        self.P_z_down = self.results.get('P_z_down', None)

        # Symbols that may appear in the expressions
        self._omega, self._t = sp.symbols('omega t', positive=True)

    # -------------------------------------------------------------------------
    # Helper
    # -------------------------------------------------------------------------
    def _to_expr(self, val):
        """Convert a value to a sympy expression safely."""
        if isinstance(val, sp.Basic):
            return val
        if isinstance(val, str):
            # Try to parse as a sympy expression; assume well‑formed input.
            return sp.sympify(val)
        if isinstance(val, (int, float)):
            return sp.nsimplify(val)
        # Last resort: use sympify (may fail for unsupported types)
        return sp.sympify(val)

    def _eval_prob(self, name, omega_val=1.0, t_val=0.5):
        """Evaluate probability expression numerically at given omega and t."""
        expr = self._to_expr(getattr(self, name))
        subs_dict = {self._omega: omega_val, self._t: t_val}
        return float(expr.subs(subs_dict).evalf())

    # -------------------------------------------------------------------------
    # Interface checks
    # -------------------------------------------------------------------------
    def test_interface_required_keys(self):
        """All six probability keys must be present and not None."""
        keys = ['P_x_up', 'P_x_down', 'P_y_up', 'P_y_down', 'P_z_up', 'P_z_down']
        missing = [k for k in keys if getattr(self, k) is None]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    def test_interface_types(self):
        """Basic type check: all values must be convertible to sympy expressions."""
        for name in ['P_x_up', 'P_x_down', 'P_y_up', 'P_y_down', 'P_z_up', 'P_z_down']:
            val = getattr(self, name)
            if val is None:
                return (False, f"SKIP: {name} is None")
            try:
                self._to_expr(val)
            except Exception as e:
                return (False, f"{name} cannot be converted to sympy: {e}")
        return (True, "All values convertible to sympy expressions")

    # -------------------------------------------------------------------------
    # Invariant checks (theorem‑based physical sanity)
    # -------------------------------------------------------------------------
    def test_invariant_x_prob_sum(self):
        """P_x_up + P_x_down must equal 1."""
        if any(x is None for x in [self.P_x_up, self.P_x_down]):
            return (False, "SKIP: missing P_x_up or P_x_down")
        s = sp.simplify(self._to_expr(self.P_x_up) + self._to_expr(self.P_x_down) - 1)
        if s == 0 or (s.is_number and abs(float(s)) < 1e-12):
            return (True, "P_x_up + P_x_down = 1 holds")
        return (False, f"P_x_up + P_x_down != 1, leftover = {s}")

    def test_invariant_y_prob_sum(self):
        """P_y_up + P_y_down must equal 1."""
        if any(x is None for x in [self.P_y_up, self.P_y_down]):
            return (False, "SKIP: missing P_y_up or P_y_down")
        s = sp.simplify(self._to_expr(self.P_y_up) + self._to_expr(self.P_y_down) - 1)
        if s == 0 or (s.is_number and abs(float(s)) < 1e-12):
            return (True, "P_y_up + P_y_down = 1 holds")
        return (False, f"P_y_up + P_y_down != 1, leftover = {s}")

    def test_invariant_z_prob_sum(self):
        """P_z_up + P_z_down must equal 1."""
        if any(x is None for x in [self.P_z_up, self.P_z_down]):
            return (False, "SKIP: missing P_z_up or P_z_down")
        s = sp.simplify(self._to_expr(self.P_z_up) + self._to_expr(self.P_z_down) - 1)
        if s == 0 or (s.is_number and abs(float(s)) < 1e-12):
            return (True, "P_z_up + P_z_down = 1 holds")
        return (False, f"P_z_up + P_z_down != 1, leftover = {s}")

    def test_invariant_z_expectation_zero(self):
        """Since <S_z> = 0, P_z_up must equal P_z_down = 1/2."""
        if any(x is None for x in [self.P_z_up, self.P_z_down]):
            return (False, "SKIP: missing P_z_up or P_z_down")
        diff = sp.simplify(self._to_expr(self.P_z_up) - self._to_expr(self.P_z_down))
        if diff == 0 or (diff.is_number and abs(float(diff)) < 1e-12):
            return (True, "P_z_up = P_z_down (expectation zero)")
        return (False, f"P_z_up != P_z_down, difference = {diff}")

    def test_invariant_x_y_conservation(self):
        """
        Ehrenfest conservation: (2P_x_up - 1)^2 + (2P_y_up - 1)^2 = 1
        because 2P_x_up - 1 = cos(ωt) and 2P_y_up - 1 = sin(ωt).
        """
        if any(x is None for x in [self.P_x_up, self.P_y_up]):
            return (False, "SKIP: missing P_x_up or P_y_up")
        expr = (2*self._to_expr(self.P_x_up) - 1)**2 + (2*self._to_expr(self.P_y_up) - 1)**2
        simplified = sp.simplify(expr)
        if simplified == 1:
            return (True, "Conservation holds exactly")
        if simplified.is_number and abs(float(simplified) - 1.0) < 1e-9:
            return (True, "Conservation approximately holds (numerical tolerance)")
        return (False, f"Conservation violated, expression = {simplified}")

    # -------------------------------------------------------------------------
    # Modeling / trace checks (quality of the functional form)
    # -------------------------------------------------------------------------
    def test_modeling_x_up_form(self):
        """P_x_up should be cos²(ωt/2) or equivalently (1+cos(ωt))/2."""
        if self.P_x_up is None:
            return (False, "SKIP: missing P_x_up")
        expr = self._to_expr(self.P_x_up)
        target1 = (1 + sp.cos(self._omega * self._t)) / 2
        if sp.simplify(expr - target1) == 0:
            return (True, "P_x_up matches (1+cos(ωt))/2")
        target2 = sp.cos(self._omega * self._t / 2)**2
        if sp.simplify(expr - target2) == 0:
            return (True, "P_x_up matches cos²(ωt/2)")
        return (False, f"P_x_up does not match expected form; expression = {expr}")

    def test_modeling_x_down_form(self):
        """P_x_down should be sin²(ωt/2) or equivalently (1-cos(ωt))/2."""
        if self.P_x_down is None:
            return (False, "SKIP: missing P_x_down")
        expr = self._to_expr(self.P_x_down)
        target1 = (1 - sp.cos(self._omega * self._t)) / 2
        if sp.simplify(expr - target1) == 0:
            return (True, "P_x_down matches (1-cos(ωt))/2")
        target2 = sp.sin(self._omega * self._t / 2)**2
        if sp.simplify(expr - target2) == 0:
            return (True, "P_x_down matches sin²(ωt/2)")
        return (False, f"P_x_down does not match expected form; expression = {expr}")

    def test_modeling_y_up_form(self):
        """P_y_up should be cos²(ωt/2 - π/4) or equivalently (1+sin(ωt))/2."""
        if self.P_y_up is None:
            return (False, "SKIP: missing P_y_up")
        expr = self._to_expr(self.P_y_up)
        target1 = (1 + sp.sin(self._omega * self._t)) / 2
        if sp.simplify(expr - target1) == 0:
            return (True, "P_y_up matches (1+sin(ωt))/2")
        target2 = sp.cos(self._omega * self._t / 2 - sp.pi / 4)**2
        if sp.simplify(expr - target2) == 0:
            return (True, "P_y_up matches cos²(ωt/2 - π/4)")
        return (False, f"P_y_up does not match expected form; expression = {expr}")

    def test_modeling_y_down_form(self):
        """P_y_down should be sin²(ωt/2 - π/4) or equivalently (1-sin(ωt))/2."""
        if self.P_y_down is None:
            return (False, "SKIP: missing P_y_down")
        expr = self._to_expr(self.P_y_down)
        target1 = (1 - sp.sin(self._omega * self._t)) / 2
        if sp.simplify(expr - target1) == 0:
            return (True, "P_y_down matches (1-sin(ωt))/2")
        target2 = sp.sin(self._omega * self._t / 2 - sp.pi / 4)**2
        if sp.simplify(expr - target2) == 0:
            return (True, "P_y_down matches sin²(ωt/2 - π/4)")
        return (False, f"P_y_down does not match expected form; expression = {expr}")

    def test_modeling_z_up_form(self):
        """P_z_up must be exactly 1/2 (since ⟨S_z⟩=0)."""
        if self.P_z_up is None:
            return (False, "SKIP: missing P_z_up")
        expr = self._to_expr(self.P_z_up)
        if expr == sp.Rational(1, 2) or (expr.is_number and abs(float(expr) - 0.5) < 1e-12):
            return (True, "P_z_up = 1/2")
        return (False, f"P_z_up != 1/2; got {expr}")

    def test_modeling_z_down_form(self):
        """P_z_down must be exactly 1/2 (since ⟨S_z⟩=0)."""
        if self.P_z_down is None:
            return (False, "SKIP: missing P_z_down")
        expr = self._to_expr(self.P_z_down)
        if expr == sp.Rational(1, 2) or (expr.is_number and abs(float(expr) - 0.5) < 1e-12):
            return (True, "P_z_down = 1/2")
        return (False, f"P_z_down != 1/2; got {expr}")

    # -------------------------------------------------------------------------
    # Monotonicity tests
    # -------------------------------------------------------------------------
    def test_monotonicity_x_and_y_trends(self):
        """
        Verify the monotonic behaviour of probabilities with increasing ωt in the
        region 0 < ωt < π/2.  At ωt = 0.5 rad, perturb to ωt = 0.6 rad.
        Expected:
          - P_x_up decreases (cos²(θ/2) decreasing on (0,π))
          - P_x_down increases (sin²(θ/2) increasing)
          - P_y_up increases (1+sin(θ) increasing on (0,π/2))
          - P_z_up stays at 0.5 (constant)
        """
        # If any required probability is missing, skip
        req = ['P_x_up', 'P_x_down', 'P_y_up', 'P_z_up']
        for name in req:
            if getattr(self, name) is None:
                return (False, f"SKIP: missing {name}")

        # Evaluate at ω=1, t=0.5 (θ=0.5) and at t=0.6 (θ=0.6)
        try:
            base = {n: self._eval_prob(n, omega_val=1.0, t_val=0.5) for n in req}
            perturbed = {n: self._eval_prob(n, omega_val=1.0, t_val=0.6) for n in req}
        except Exception as e:
            return (False, f"Numerical evaluation failed: {e}")

        # Check monotonicity with tolerance
        tol = 1e-9
        # P_x_up: should decrease
        if not (perturbed['P_x_up'] < base['P_x_up'] + tol):
            return (False, f"P_x_up did not decrease: base={base['P_x_up']}, perturbed={perturbed['P_x_up']}")
        # P_x_down: should increase
        if not (perturbed['P_x_down'] > base['P_x_down'] - tol):
            return (False, f"P_x_down did not increase: base={base['P_x_down']}, perturbed={perturbed['P_x_down']}")
        # P_y_up: should increase (since sin increasing on (0,π/2))
        if not (perturbed['P_y_up'] > base['P_y_up'] - tol):
            return (False, f"P_y_up did not increase: base={base['P_y_up']}, perturbed={perturbed['P_y_up']}")
        # P_z_up: should remain 0.5 (within numerical error)
        if abs(perturbed['P_z_up'] - base['P_z_up']) > 1e-12:
            return (False, f"P_z_up changed: base={base['P_z_up']}, perturbed={perturbed['P_z_up']}")

        return (True, "All monotonicity trends hold for perturbation ωt from 0.5 to 0.6 rad")

    # -------------------------------------------------------------------------
    # Answer sanity (weak: only check that probabilities lie in [0,1] for a
    # sample point; no hardcoded expected values).
    # -------------------------------------------------------------------------
    def test_answer_sanity_range(self):
        """
        Evaluate all probabilities at a representative point (e.g. ωt = 1 rad)
        and verify they are between 0 and 1 inclusive.
        """
        names = ['P_x_up', 'P_x_down', 'P_y_up', 'P_y_down', 'P_z_up', 'P_z_down']
        vals = [getattr(self, n) for n in names]
        if any(v is None for v in vals):
            return (False, "SKIP: one or more probabilities missing")

        subs = {self._omega: 1.0, self._t: 1.0}  # ωt = 1 rad
        for name, val in zip(names, vals):
            try:
                num_val = float(self._to_expr(val).subs(subs).evalf())
            except Exception as e:
                return (False, f"Cannot evaluate {name} numerically: {e}")
            if num_val < -1e-9 or num_val > 1 + 1e-9:
                return (False, f"{name} = {num_val} outside [0,1]")
        return (True, "All probabilities in [0,1] at sample point ωt = 1 rad")

    # -------------------------------------------------------------------------
    # Test runner
    # -------------------------------------------------------------------------
    def run_all_tests(self):
        """Discover and run every test_* method. Returns (all_pass, logs)."""
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