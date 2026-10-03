import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Givens: safe None defaults, no numeric fallbacks
        self.R = None
        self.sigma0 = None
        self.epsilon0 = None
        self.theta = None
        self.r = None
        if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
            self.R = solution_instance.givens.get('R', None)
            self.sigma0 = solution_instance.givens.get('sigma0', None)
            self.epsilon0 = solution_instance.givens.get('epsilon0', None)
            self.theta = solution_instance.givens.get('theta', None)
            self.r = solution_instance.givens.get('r', None)

        # Results from solution_instance (could be attribute or dict)
        self.V_out = None
        self.V_in = None
        self.E_out_r = None
        self.E_out_theta = None
        self.E_in = None
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.V_out = solution_instance.results.get('V_out', None)
            self.V_in = solution_instance.results.get('V_in', None)
            self.E_out_r = solution_instance.results.get('E_out_r', None)
            self.E_out_theta = solution_instance.results.get('E_out_theta', None)
            self.E_in = solution_instance.results.get('E_in', None)
        elif hasattr(solution_instance, 'V_out') and hasattr(solution_instance, 'V_in'):
            # Fallback if results not separate
            self.V_out = getattr(solution_instance, 'V_out', None)
            self.V_in = getattr(solution_instance, 'V_in', None)
            self.E_out_r = getattr(solution_instance, 'E_out_r', None)
            self.E_out_theta = getattr(solution_instance, 'E_out_theta', None)
            self.E_in = getattr(solution_instance, 'E_in', None)

        # Symbols for internal use
        self.sym_r = sp.Symbol('r', real=True, positive=True)
        self.sym_theta = sp.Symbol('theta', real=True)
        self.sym_R = sp.Symbol('R', real=True, positive=True)
        self.sym_sigma0 = sp.Symbol('sigma0', real=True)
        self.sym_eps0 = sp.Symbol('epsilon0', real=True, positive=True)
        self.costheta = sp.cos(self.sym_theta)
        self.sintheta = sp.sin(self.sym_theta)

        # Re-derived expected forms from theorem constraints (not copied from answers)
        # Only l=1 contributes, continuity and field jump yield:
        self.expected_V_in = (self.sym_sigma0/(3*self.sym_eps0)) * self.sym_r * self.costheta
        self.expected_V_out = (self.sym_sigma0 * self.sym_R**3 / (3*self.sym_eps0)) * self.costheta / self.sym_r**2
        self.expected_E_out_r = (2*self.sym_sigma0*self.sym_R**3*self.costheta)/(3*self.sym_eps0*self.sym_r**3)
        self.expected_E_out_theta = (self.sym_sigma0*self.sym_R**3*self.sintheta)/(3*self.sym_eps0*self.sym_r**3)
        self.expected_E_in_z = -self.sym_sigma0/(3*self.sym_eps0)  # component along z

    def _to_sympy(self, expr):
        """Safely convert to sympy expression; return None on failure."""
        if expr is None:
            return None
        if isinstance(expr, sp.Basic):
            return expr
        try:
            return sp.sympify(expr)
        except:
            return None

    def _substitute_numeric(self, expr, values):
        """Evaluate expression numerically using dict of values (symbol_name->number)."""
        if expr is None:
            return None
        sexpr = self._to_sympy(expr)
        if sexpr is None:
            return None
        subs_dict = {}
        for sym in sexpr.free_symbols:
            sname = str(sym)
            # Map standard names to values dict; handle cos/sin as functions
            if sname in values:
                subs_dict[sym] = values[sname]
            elif sname == 'costheta':
                subs_dict[sym] = math.cos(values.get('theta', 0))
            elif sname == 'sintheta':
                subs_dict[sym] = math.sin(values.get('theta', 0))
            elif sname in ('epsilon0', 'eps0', 'epsilon_0'):
                subs_dict[sym] = values.get('epsilon0', 1.0)
            elif sname in ('sigma0', 'sigma_0'):
                subs_dict[sym] = values.get('sigma0', 1.0)
            elif sname == 'R':
                subs_dict[sym] = values.get('R', 1.0)
            elif sname == 'r':
                subs_dict[sym] = values.get('r', 1.0)
            elif sname == 'theta':
                subs_dict[sym] = values.get('theta', 1.0)
            else:
                # Unknown symbol; cannot substitute
                return None
        try:
            return float(sexpr.subs(subs_dict).evalf())
        except:
            return None

    # --------------------------------------------------------------
    # Interface Tests
    # --------------------------------------------------------------
    def test_interface_required_keys_present(self):
        """Check that all five required result keys exist and are not None."""
        keys = ['V_out', 'V_in', 'E_out_r', 'E_out_theta', 'E_in']
        missing = [k for k in keys if getattr(self, k, None) is None]
        if missing:
            return (False, f"Missing results: {missing}")
        return (True, "All required keys present")

    def test_interface_types_convertible(self):
        """Check that each result can be parsed as a sympy expression."""
        for name in ['V_out', 'V_in', 'E_out_r', 'E_out_theta', 'E_in']:
            val = getattr(self, name, None)
            if val is None:
                continue
            if self._to_sympy(val) is None:
                return (False, f"{name} is not a valid symbolic expression")
        return (True, "All results are valid sympy expressions")

    # --------------------------------------------------------------
    # Invariant Tests (theorem-based physical sanity)
    # --------------------------------------------------------------
    def test_invariant_potential_continuity_at_R(self):
        """Check V_in(R,θ) = V_out(R,θ) (potential continuous across shell)."""
        if self.V_in is None or self.V_out is None:
            return (False, "SKIP: V_in or V_out not available")
        V_in = self._to_sympy(self.V_in)
        V_out = self._to_sympy(self.V_out)
        if V_in is None or V_out is None:
            return (False, "SKIP: cannot convert potentials")
        diff = sp.simplify(V_in.subs(self.sym_r, self.sym_R) - V_out.subs(self.sym_r, self.sym_R))
        # Numerically test
        test_cases = [
            {'R':1.0, 'sigma0':1.0, 'epsilon0':1.0, 'theta':0.3},
            {'R':2.0, 'sigma0':3.0, 'epsilon0':8.854e-12, 'theta':1.2}
        ]
        for vals in test_cases:
            num = self._substitute_numeric(diff, vals)
            if num is None:
                continue
            if not math.isclose(num, 0.0, rel_tol=1e-5, abs_tol=1e-5):
                return (False, f"Continuity fails: diff = {num}")
        return (True, "Potential continuous at r=R")

    def test_invariant_field_discontinuity(self):
        """Check (∂V_out/∂r - ∂V_in/∂r)|_{r=R} = -σ(θ)/ε₀."""
        if self.V_in is None or self.V_out is None:
            return (False, "SKIP: V_in or V_out not available")
        V_in = self._to_sympy(self.V_in)
        V_out = self._to_sympy(self.V_out)
        if V_in is None or V_out is None:
            return (False, "SKIP: cannot convert potentials")
        dV_in_dr = sp.diff(V_in, self.sym_r).subs(self.sym_r, self.sym_R)
        dV_out_dr = sp.diff(V_out, self.sym_r).subs(self.sym_r, self.sym_R)
        disc = sp.simplify(dV_out_dr - dV_in_dr)
        expected = -self.sym_sigma0 * self.costheta / self.sym_eps0
        test_cases = [
            {'R':1.0, 'sigma0':1.0, 'epsilon0':1.0, 'theta':0.5},
            {'R':1.5, 'sigma0':2.0, 'epsilon0':8.854e-12, 'theta':1.0}
        ]
        for vals in test_cases:
            num_disc = self._substitute_numeric(disc, vals)
            num_exp = self._substitute_numeric(expected, vals)
            if num_disc is None or num_exp is None:
                continue
            if not math.isclose(num_disc, num_exp, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"Discontinuity mismatch: {num_disc} vs {num_exp}")
        return (True, "Field discontinuity condition satisfied")

    def test_invariant_laplace_inside(self):
        """Check ∇²V_in = 0 for r < R."""
        if self.V_in is None:
            return (False, "SKIP: V_in not available")
        V_in = self._to_sympy(self.V_in)
        if V_in is None:
            return (False, "SKIP: cannot convert V_in")
        # Spherical Laplacian with azimuthal symmetry
        term_r = (1/self.sym_r**2) * sp.diff(self.sym_r**2 * sp.diff(V_in, self.sym_r), self.sym_r)
        term_theta_numer = (1/(self.sym_r**2 * sp.sin(self.sym_theta))) * sp.diff(sp.sin(self.sym_theta) * sp.diff(V_in, self.sym_theta), self.sym_theta)
        laplacian = sp.simplify(term_r + term_theta_numer)
        test_cases = [
            {'r':0.5, 'theta':0.7, 'R':1.0, 'sigma0':1.0, 'epsilon0':1.0},
            {'r':1.2, 'theta':0.3, 'R':2.0, 'sigma0':2.0, 'epsilon0':8.854e-12}
        ]
        for vals in test_cases:
            num = self._substitute_numeric(laplacian, vals)
            if num is None:
                continue
            if not math.isclose(num, 0.0, rel_tol=1e-5, abs_tol=1e-5):
                return (False, f"Laplacian inside not zero: {num}")
        return (True, "Laplace equation satisfied inside")

    def test_invariant_laplace_outside(self):
        """Check ∇²V_out = 0 for r > R."""
        if self.V_out is None:
            return (False, "SKIP: V_out not available")
        V_out = self._to_sympy(self.V_out)
        if V_out is None:
            return (False, "SKIP: cannot convert V_out")
        term_r = (1/self.sym_r**2) * sp.diff(self.sym_r**2 * sp.diff(V_out, self.sym_r), self.sym_r)
        term_theta = (1/(self.sym_r**2 * sp.sin(self.sym_theta))) * sp.diff(sp.sin(self.sym_theta) * sp.diff(V_out, self.sym_theta), self.sym_theta)
        laplacian = sp.simplify(term_r + term_theta)
        test_cases = [
            {'r':2.0, 'theta':0.5, 'R':1.0, 'sigma0':1.0, 'epsilon0':1.0},
            {'r':3.0, 'theta':1.0, 'R':2.0, 'sigma0':2.0, 'epsilon0':8.854e-12}
        ]
        for vals in test_cases:
            num = self._substitute_numeric(laplacian, vals)
            if num is None:
                continue
            if not math.isclose(num, 0.0, rel_tol=1e-5, abs_tol=1e-5):
                return (False, f"Laplacian outside not zero: {num}")
        return (True, "Laplace equation satisfied outside")

    def test_invariant_finiteness_at_origin(self):
        """Check V_in finite as r→0 (should be 0)."""
        if self.V_in is None:
            return (False, "SKIP: V_in not available")
        V_in = self._to_sympy(self.V_in)
        if V_in is None:
            return (False, "SKIP: cannot convert V_in")
        limit = sp.limit(V_in, self.sym_r, 0, dir='+')
        try:
            num = float(limit.evalf())
            if not math.isfinite(num):
                return (False, f"V_in diverges at origin: limit = {num}")
            if not math.isclose(num, 0.0, abs_tol=1e-5):
                return (False, f"V_in not zero at origin: {num}")
        except Exception as e:
            return (False, f"Limit evaluation error: {e}")
        return (True, "V_in finite at origin")

    def test_invariant_vanishing_at_infinity(self):
        """Check V_out → 0 as r→∞."""
        if self.V_out is None:
            return (False, "SKIP: V_out not available")
        V_out = self._to_sympy(self.V_out)
        if V_out is None:
            return (False, "SKIP: cannot convert V_out")
        limit = sp.limit(V_out, self.sym_r, sp.oo)
        try:
            num = float(limit.evalf())
            if not math.isclose(num, 0.0, abs_tol=1e-5):
                return (False, f"V_out does not vanish at infinity: {num}")
        except Exception as e:
            return (False, f"Limit evaluation error: {e}")
        return (True, "V_out vanishes at infinity")

    # --------------------------------------------------------------
    # Modeling / Trace Quality Tests
    # --------------------------------------------------------------
    def test_modeling_V_in_linear_r(self):
        """Check that V_in ∝ r (l=1 inside solution)."""
        if self.V_in is None:
            return (False, "SKIP: V_in not available")
        sexpr = self._to_sympy(self.V_in)
        if sexpr is None:
            return (False, "SKIP: cannot convert V_in")
        ratio = sp.simplify(sexpr / self.sym_r)
        if self.sym_r in ratio.free_symbols:
            return (False, "V_in is not linear in r")
        return (True, "V_in ∝ r")

    def test_modeling_V_out_quadratic_decay(self):
        """Check that V_out ∝ 1/r^2."""
        if self.V_out is None:
            return (False, "SKIP: V_out not available")
        sexpr = self._to_sympy(self.V_out)
        if sexpr is None:
            return (False, "SKIP: cannot convert V_out")
        prod = sp.simplify(sexpr * self.sym_r**2)
        if self.sym_r in prod.free_symbols:
            return (False, "V_out does not decay as 1/r^2")
        return (True, "V_out ∝ 1/r^2")

    def test_modeling_angular_dependence_cos_theta(self):
        """Check that both potentials are proportional to cosθ (only l=1)."""
        for name, expr in [('V_out', self.V_out), ('V_in', self.V_in)]:
            if expr is None:
                return (False, f"SKIP: {name} not available")
            sexpr = self._to_sympy(expr)
            if sexpr is None:
                return (False, f"SKIP: cannot convert {name}")
            # Check that after dividing by cosθ, no theta dependence remains
            ratio = sp.simplify(sexpr / self.costheta)
            if self.sym_theta in ratio.free_symbols:
                return (False, f"{name} has angular dependence beyond cosθ")
        return (True, "Both potentials ∝ cosθ")

    def test_modeling_E_out_r_sign(self):
        """Check that E_out_r has the same sign as costheta (physical: field points away from positive σ)."""
        if self.E_out_r is None:
            return (False, "SKIP: E_out_r not available")
        sexpr = self._to_sympy(self.E_out_r)
        if sexpr is None:
            return (False, "SKIP: cannot convert E_out_r")
        # For fixed positive sigma0, R, epsilon0, and positive r>R, sign(sexpr) should equal sign(costheta)
        # Numerically test a few points
        test_cases = [
            {'r':2.0, 'theta':0.3, 'R':1.0, 'sigma0':1.0, 'epsilon0':1.0},  # cos>0
            {'r':2.0, 'theta':2.0, 'R':1.0, 'sigma0':1.0, 'epsilon0':1.0}   # cos<0
        ]
        for vals in test_cases:
            num = self._substitute_numeric(sexpr, vals)
            if num is None:
                continue
            cos_val = math.cos(vals['theta'])
            if (num >= 0) != (cos_val >= 0):
                return (False, f"E_out_r sign inconsistent: E_r={num}, cosθ={cos_val}")
        return (True, "E_out_r has correct sign relative to cosθ")

    def test_modeling_E_in_constant_z(self):
        """Check that E_in is uniform and points along -e_z (if expressed as scalar magnitude, check constancy)."""
        if self.E_in is None:
            return (False, "SKIP: E_in not available")
        sexpr = self._to_sympy(self.E_in)
        if sexpr is None:
            return (False, "SKIP: cannot convert E_in")
        # If E_in is a vector/expression, check that it's independent of r and θ
        # We'll just check numeric constancy over different points
        test_cases = [
            {'r':0.5, 'theta':0.3, 'R':1.0, 'sigma0':1.0, 'epsilon0':1.0},
            {'r':0.8, 'theta':1.5, 'R':1.0, 'sigma0':1.0, 'epsilon0':1.0},
            {'r':0.2, 'theta':2.5, 'R':1.0, 'sigma0':2.0, 'epsilon0':2.0}
        ]
        values = []
        for vals in test_cases:
            num = self._substitute_numeric(sexpr, vals)
            if num is None:
                continue
            values.append(num)
        if len(values) < 2:
            return (False, "Insufficient numeric evaluations for E_in constant test")
        # Check that all values are close
        for v in values[1:]:
            if not math.isclose(v, values[0], rel_tol=1e-5, abs_tol=1e-5):
                return (False, f"E_in varies with (r,θ): {values}")
        return (True, "E_in is constant (uniform)")

    # --------------------------------------------------------------
    # Monotonicity Tests
    # --------------------------------------------------------------
    def test_monotonicity_V_in_increases_with_r(self):
        """For fixed θ with cosθ>0, V_in should increase linearly with r."""
        if self.V_in is None:
            return (False, "SKIP: V_in not available")
        sexpr = self._to_sympy(self.V_in)
        if sexpr is None:
            return (False, "SKIP: cannot convert V_in")
        # Baseline at some r0, perturb r by +10%
        r0 = 0.5
        delta = 0.05  # 10%
        baseline = self._substitute_numeric(sexpr, {'r':r0, 'theta':0.3, 'R':1.0, 'sigma0':1.0, 'epsilon0':1.0})
        perturbed = self._substitute_numeric(sexpr, {'r':r0+delta, 'theta':0.3, 'R':1.0, 'sigma0':1.0, 'epsilon0':1.0})
        if baseline is None or perturbed is None:
            return (False, "SKIP: could not evaluate numerically")
        if perturbed <= baseline:
            return (False, f"V_in did not increase with r: baseline={baseline}, perturbed={perturbed}")
        return (True, "V_in increases with r (for cosθ>0)")

    def test_monotonicity_V_out_decreases_with_r(self):
        """For fixed θ with cosθ>0, V_out should decrease as r increases (1/r^2)."""
        if self.V_out is None:
            return (False, "SKIP: V_out not available")
        sexpr = self._to_sympy(self.V_out)
        if sexpr is None:
            return (False, "SKIP: cannot convert V_out")
        r0 = 2.0
        delta = 0.2  # +10%
        baseline = self._substitute_numeric(sexpr, {'r':r0, 'theta':0.3, 'R':1.0, 'sigma0':1.0, 'epsilon0':1.0})
        perturbed = self._substitute_numeric(sexpr, {'r':r0+delta, 'theta':0.3, 'R':1.0, 'sigma0':1.0, 'epsilon0':1.0})
        if baseline is None or perturbed is None:
            return (False, "SKIP: could not evaluate numerically")
        if perturbed >= baseline:
            return (False, f"V_out did not decrease with r: baseline={baseline}, perturbed={perturbed}")
        return (True, "V_out decreases with r")

    def test_monotonicity_E_out_r_decays_with_r(self):
        """Check that magnitude of E_out_r decreases as r increases."""
        if self.E_out_r is None:
            return (False, "SKIP: E_out_r not available")
        sexpr = self._to_sympy(self.E_out_r)
        if sexpr is None:
            return (False, "SKIP: cannot convert E_out_r")
        r0 = 2.0
        delta = 0.2
        # Use cosθ>0 so E_out_r positive
        baseline = self._substitute_numeric(sexpr, {'r':r0, 'theta':0.3, 'R':1.0, 'sigma0':1.0, 'epsilon0':1.0})
        perturbed = self._substitute_numeric(sexpr, {'r':r0+delta, 'theta':0.3, 'R':1.0, 'sigma0':1.0, 'epsilon0':1.0})
        if baseline is None or perturbed is None:
            return (False, "SKIP: could not evaluate numerically")
        if abs(perturbed) > abs(baseline):
            return (False, f"E_out_r magnitude increased with r: abs(base)={abs(baseline)}, abs(pert)={abs(perturbed)}")
        return (True, "E_out_r magnitude decreases with r")

    # --------------------------------------------------------------
    # Weak Answer Sanity (at most one such method)
    # --------------------------------------------------------------
    def test_answer_sanity_sign_of_E_in(self):
        """Weak sanity: E_in should point downward (negative z-direction) if σ₀>0."""
        if self.E_in is None:
            return (False, "SKIP: E_in not available")
        # If E_in is expressed as a magnitude or vector, check that its numeric value for positive sigma0 is negative.
        # This is a sign check only – no re-derivation of exact formula.
        sexpr = self._to_sympy(self.E_in)
        if sexpr is None:
            return (False, "SKIP: cannot convert E_in")
        test_vals = {'r':0.5, 'theta':0.3, 'R':1.0, 'sigma0':1.0, 'epsilon0':1.0}
        num = self._substitute_numeric(sexpr, test_vals)
        if num is None:
            return (False, "SKIP: could not evaluate numerically")
        if num >= 0:
            return (False, f"E_in should be negative for sigma0>0, got {num}")
        # Also test with negative sigma0
        test_vals['sigma0'] = -1.0
        num2 = self._substitute_numeric(sexpr, test_vals)
        if num2 is not None and num2 <= 0:
            return (False, f"E_in should be positive for sigma0<0, got {num2}")
        return (True, "E_in sign consistent: opposite to sigma0")

    # --------------------------------------------------------------
    # Mandatory run_all_tests
    # --------------------------------------------------------------
    def run_all_tests(self):
        """Discover all methods starting with 'test_', execute each in try/except, collect results."""
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
                traceback.print_exc()
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)