import sympy as sp
import math

class PhysicsChecker:
    """
    Audits a solution for the potential inside and outside a sphere with a point charge
    at the origin and boundary potential V0*cos(theta).
    """

    def __init__(self, solution_instance):
        # Extract results safely using None defaults
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            results = solution_instance.results
            self.Phi_in = results.get('Phi_in', None)
            self.Phi_out = results.get('Phi_out', None)
        else:
            self.Phi_in = getattr(solution_instance, 'Phi_in', None)
            self.Phi_out = getattr(solution_instance, 'Phi_out', None)

        # Extract givens; if missing, set to None
        self.q = getattr(solution_instance, 'q', None)
        self.R = getattr(solution_instance, 'R', None)
        self.V0 = getattr(solution_instance, 'V0', None)
        self.epsilon0 = getattr(solution_instance, 'epsilon0', None)

        # SymPy symbols for symbolic checks
        self.r, self.theta = sp.symbols('r theta', real=True, positive=True)
        self.q_sym, self.R_sym = sp.symbols('q R', real=True, positive=True)
        self.V0_sym, self.eps0_sym = sp.symbols('V0 epsilon0', real=True)
        self.pi = sp.pi

        # Convert potential expressions to SymPy if they are strings
        if isinstance(self.Phi_in, str):
            try:
                self.Phi_in_expr = sp.sympify(self.Phi_in)
            except:
                self.Phi_in_expr = None
        else:
            self.Phi_in_expr = self.Phi_in

        if isinstance(self.Phi_out, str):
            try:
                self.Phi_out_expr = sp.sympify(self.Phi_out)
            except:
                self.Phi_out_expr = None
        else:
            self.Phi_out_expr = self.Phi_out

        # Numeric values for checks (try to convert)
        self._numeric_q = self._to_float(self.q)
        self._numeric_R = self._to_float(self.R)
        self._numeric_V0 = self._to_float(self.V0)
        self._numeric_eps0 = self._to_float(self.epsilon0)

    def _to_float(self, val):
        if val is None:
            return None
        try:
            return float(val)
        except:
            return None

    # ------------------------------------------------------------------
    # Interface tests: check that required result keys exist and are usable
    # ------------------------------------------------------------------
    def test_interface_Phi_in_exists(self):
        if self.Phi_in is None:
            return (False, "Phi_in is missing or None")
        if self.Phi_in_expr is None:
            return (False, "Phi_in could not be converted to SymPy expression")
        return (True, "Phi_in provided and convertible to SymPy")

    def test_interface_Phi_out_exists(self):
        if self.Phi_out is None:
            return (False, "Phi_out is missing or None")
        if self.Phi_out_expr is None:
            return (False, "Phi_out could not be converted to SymPy expression")
        return (True, "Phi_out provided and convertible to SymPy")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_point_charge_term_inside(self):
        """Inside potential must contain term q/(4π ε0 r)."""
        if self.Phi_in_expr is None:
            return (False, "SKIP: Phi_in not available")
        expr = sp.expand(self.Phi_in_expr)
        coeff = sp.collect(expr, 1/self.r, evaluate=False).get(1/self.r, 0)
        expected_coeff = self.q_sym/(4*self.pi*self.eps0_sym)
        diff = sp.simplify(coeff - expected_coeff)
        if diff != 0:
            return (False, f"Point charge term coefficient mismatch: {diff}")
        return (True, "Inside contains correct point charge term")

    def test_invariant_constant_term_cancels_at_boundary(self):
        """At r=R, the constant (l=0) part of Phi_in should be zero because boundary is pure cosθ."""
        if self.Phi_in_expr is None:
            return (False, "SKIP: Phi_in not available")
        expr_at_R = self.Phi_in_expr.subs(self.r, self.R_sym)
        val_at_pi2 = sp.simplify(expr_at_R.subs(self.theta, sp.pi/2))
        if val_at_pi2 != 0:
            # Try numeric fallback
            if self._numeric_q is not None and self._numeric_R is not None and self._numeric_eps0 is not None:
                try:
                    numeric_val = float(val_at_pi2.subs({self.q_sym: self._numeric_q,
                                                         self.R_sym: self._numeric_R,
                                                         self.eps0_sym: self._numeric_eps0}))
                    if math.isclose(numeric_val, 0, rel_tol=1e-2, abs_tol=1e-2):
                        return (True, "Constant term vanishes at boundary (numerical check)")
                except:
                    pass
            return (False, f"Constant term at boundary is {val_at_pi2}, expected 0")
        return (True, "Constant part of Phi_in vanishes at r=R (l=0 condition satisfied)")

    def test_invariant_Phi_out_vanishes_at_infinity(self):
        """Outside potential must go to 0 as r -> ∞."""
        if self.Phi_out_expr is None:
            return (False, "SKIP: Phi_out not available")
        limit_val = sp.limit(self.Phi_out_expr, self.r, sp.oo)
        if limit_val == 0:
            return (True, "Outside potential vanishes at infinity")
        # If limit involves symbols, it should still be 0 for any finite constants.
        # Check if the expression has no positive powers of r.
        # Only negative powers guarantee vanishing.
        if sp.simplify(limit_val) != 0:
            return (False, f"Phi_out does not vanish at infinity, limit {limit_val}")
        return (True, "Outside potential vanishes at infinity")

    def test_invariant_inside_no_negative_powers_except_point_charge(self):
        """Inside potential (excluding point charge) must have only non-negative powers of r."""
        if self.Phi_in_expr is None:
            return (False, "SKIP: Phi_in not available")
        q_term = self.q_sym / (4 * self.pi * self.eps0_sym * self.r)
        rest = sp.expand(self.Phi_in_expr - q_term)
        terms = sp.Add.make_args(rest)
        for term in terms:
            if term.has(1/self.r):
                return (False, f"Inside potential (excl. point charge) has term with 1/r: {term}")
        return (True, "Inside potential (excl. point charge) has only non-negative powers")

    def test_invariant_outside_only_negative_powers(self):
        """Outside potential must have only negative powers of r (no constant or positive powers)."""
        if self.Phi_out_expr is None:
            return (False, "SKIP: Phi_out not available")
        terms = sp.Add.make_args(sp.expand(self.Phi_out_expr))
        for term in terms:
            indep, r_part = term.as_independent(self.r, as_Add=False)
            if r_part != 1 and r_part.has(self.r):
                # r_part is something like r^2 or 1/r^2.
                if r_part.is_Pow:
                    base, exp = r_part.as_base_exp()
                    if base == self.r and exp.is_positive:
                        return (False, f"Outside potential has positive power of r: {term}")
                else:
                    if r_part == self.r:
                        return (False, f"Outside potential has linear term in r: {term}")
            # Constant term not allowed
            if r_part == 1 and not term.has(1/self.r):
                return (False, f"Outside potential has constant term: {term}")
        return (True, "Outside potential contains only negative powers of r")

    # ------------------------------------------------------------------
    # Modeling / trace tests (based on explanation_steps)
    # ------------------------------------------------------------------
    def test_modeling_angular_linearity(self):
        """Both potentials should be linear in cosθ (only l=0 and l=1 modes)."""
        if self.Phi_in_expr is None and self.Phi_out_expr is None:
            return (False, "SKIP: no potentials available")
        for name, expr in [("Phi_in", self.Phi_in_expr), ("Phi_out", self.Phi_out_expr)]:
            if expr is None:
                continue
            # Check that expression is exactly of form A + B*cosθ (in) or B*cosθ (out)
            cos = sp.cos(self.theta)
            # Substitute cosθ with a dummy to check polynomial degree
            try:
                # Expand and then replace cosθ with variable x, then expand: should have degree <=1.
                x = sp.symbols('x')
                poly = sp.simplify(expr.subs(cos, x))
                poly_expr = sp.expand(poly)
                # Ensure it's polynomial in x and degree <=1
                if poly_expr.has(x**2) or poly_expr.has(x**3) or (poly_expr.has(1/x) and not expr.has(1/x)):
                    return (False, f"{name} has non-linear angular dependence (powers of cosθ)")
                # Also check that no other angular functions (sinθ, etc.) appear
                sin = sp.sin(self.theta)
                if expr.has(sin) and not expr.has(sp.sin(2*self.theta)):  # only check sin; cos is fine
                    # But sin might appear via cos^2? Actually if only cos, fine.
                    # We'll just check that there is no sin separately.
                    if expr.has(sp.sin(self.theta)):
                        return (False, f"{name} contains sinθ term")
            except:
                return (False, f"Cannot verify angular linearity for {name}")
        return (True, "Potentials are linear in cosθ (only l=0,1 modes)")

    def test_modeling_boundary_matching(self):
        """Phi_in and Phi_out must be equal on the surface r=R for all θ."""
        if self.Phi_in_expr is None or self.Phi_out_expr is None:
            return (False, "SKIP: potentials missing")
        expr_in_at_R = self.Phi_in_expr.subs(self.r, self.R_sym)
        expr_out_at_R = self.Phi_out_expr.subs(self.r, self.R_sym)
        diff = sp.simplify(expr_in_at_R - expr_out_at_R)
        if diff == 0:
            return (True, "Boundary condition matches exactly")
        # Numerical fallback
        if self._numeric_R is not None and self._numeric_V0 is not None and self._numeric_eps0 is not None and self._numeric_q is not None:
            try:
                test_theta = sp.pi/4
                diff_num = float(diff.subs({self.R_sym: self._numeric_R, self.V0_sym: self._numeric_V0,
                                            self.eps0_sym: self._numeric_eps0, self.q_sym: self._numeric_q,
                                            self.theta: test_theta}))
                if math.isclose(diff_num, 0, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, "Boundary condition matches within tolerance (numerical)")
            except:
                pass
        return (False, f"Boundary condition mismatch, difference = {diff}")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_Phi_in_increases_with_V0(self):
        """Increasing V0 should increase the amplitude of the cosθ term inside."""
        if self.Phi_in_expr is None:
            return (False, "SKIP: Phi_in not available")
        # Choose a test point: r = R/2, θ=0
        if self._numeric_R is None:
            return (False, "SKIP: no numeric R available to evaluate")
        test_r = self._numeric_R / 2.0
        test_theta_val = 0.0  # cos = 1
        # Perturb V0: baseline value 1.0 (if no given V0), else use given V0
        V0_base = 1.0 if self._numeric_V0 is None else self._numeric_V0
        perturbations = [1.0, 1.1]  # 0% and +10%
        values = []
        for factor in perturbations:
            V0_pert = V0_base * factor
            # Evaluate Phi_in at this V0
            try:
                Phi_in_num = float(self.Phi_in_expr.subs({
                    self.r: test_r,
                    self.theta: test_theta_val,
                    self.R_sym: self._numeric_R,
                    self.V0_sym: V0_pert,
                    self.q_sym: self._numeric_q if self._numeric_q is not None else 0,
                    self.eps0_sym: self._numeric_eps0 if self._numeric_eps0 is not None else 1.0
                }))
                values.append(Phi_in_num)
            except:
                return (False, "Could not evaluate Phi_in numerically for monotonicity test")
        if len(values) != 2:
            return (False, "Could not get two values for comparison")
        if values[1] <= values[0]:
            return (False, f"Phi_in did not increase with V0: baseline={values[0]}, perturbed={values[1]}")
        return (True, f"Phi_in increases with V0: baseline={values[0]}, +10% V0 gives {values[1]}")

    def test_monotonicity_Phi_out_increases_with_V0(self):
        """Increasing V0 should increase the magnitude of Phi_out (for fixed r>R, cosθ>0)."""
        if self.Phi_out_expr is None:
            return (False, "SKIP: Phi_out not available")
        if self._numeric_R is None:
            return (False, "SKIP: no numeric R available")
        test_r = 2.0 * self._numeric_R  # outside
        test_theta_val = 0.0  # cos=1
        V0_base = 1.0 if self._numeric_V0 is None else self._numeric_V0
        perturbations = [1.0, 1.1]
        values = []
        for factor in perturbations:
            V0_pert = V0_base * factor
            try:
                Phi_out_num = float(self.Phi_out_expr.subs({
                    self.r: test_r,
                    self.theta: test_theta_val,
                    self.R_sym: self._numeric_R,
                    self.V0_sym: V0_pert,
                    self.q_sym: self._numeric_q if self._numeric_q is not None else 0,
                    self.eps0_sym: self._numeric_eps0 if self._numeric_eps0 is not None else 1.0
                }))
                values.append(Phi_out_num)
            except:
                return (False, "Could not evaluate Phi_out numerically for monotonicity test")
        if len(values) != 2:
            return (False, "Could not get two values")
        if values[1] <= values[0]:
            return (False, f"Phi_out did not increase with V0: baseline={values[0]}, perturbed={values[1]}")
        return (True, f"Phi_out increases with V0: baseline={values[0]}, +10% V0 gives {values[1]}")

    # ------------------------------------------------------------------
    # Weak answer sanity test (only sign/order-of-magnitude)
    # ------------------------------------------------------------------
    def test_answer_sanity_potentials_finite_and_nonzero(self):
        """Both potentials should be finite and non-zero at a sample point (sanity check)."""
        # We only check that we can evaluate them and get a finite number (not NaN, not infinite).
        # Also check sign? Not required, but we can check that they are not trivial zero.
        if self.Phi_in_expr is None or self.Phi_out_expr is None:
            return (False, "SKIP: one or both potentials missing")
        if self._numeric_R is None:
            return (False, "SKIP: no numeric R available to define sample point")
        # Sample points
        test_r_in = self._numeric_R / 2.0
        test_r_out = 2.0 * self._numeric_R
        test_theta_val = sp.pi/4  # cos=~0.707
        # Use default values for missing givens
        q_val = 1.0 if self._numeric_q is None else self._numeric_q
        V0_val = 1.0 if self._numeric_V0 is None else self._numeric_V0
        eps0_val = 1.0 if self._numeric_eps0 is None else self._numeric_eps0
        try:
            phi_in_val = float(self.Phi_in_expr.subs({self.r: test_r_in, self.theta: test_theta_val,
                                                       self.R_sym: self._numeric_R,
                                                       self.V0_sym: V0_val, self.q_sym: q_val,
                                                       self.eps0_sym: eps0_val}))
            phi_out_val = float(self.Phi_out_expr.subs({self.r: test_r_out, self.theta: test_theta_val,
                                                         self.R_sym: self._numeric_R,
                                                         self.V0_sym: V0_val, self.q_sym: q_val,
                                                         self.eps0_sym: eps0_val}))
        except Exception as e:
            return (False, f"Cannot evaluate potentials numerically: {e}")
        if not math.isfinite(phi_in_val):
            return (False, f"Phi_in is not finite at sample point: {phi_in_val}")
        if not math.isfinite(phi_out_val):
            return (False, f"Phi_out is not finite at sample point: {phi_out_val}")
        # Also check they are non-zero (very permissive: magnitude > 0)
        if abs(phi_in_val) < 1e-10 and abs(phi_out_val) < 1e-10:
            return (False, "Both potentials are essentially zero at sample points, suspicious")
        return (True, f"Potentials evaluate to finite values: Phi_in={phi_in_val}, Phi_out={phi_out_val}")

    # ------------------------------------------------------------------
    # Run all tests
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)