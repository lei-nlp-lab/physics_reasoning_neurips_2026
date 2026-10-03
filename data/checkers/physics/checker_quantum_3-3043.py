import math
import sympy as sp

class PhysicsChecker:
    """
    Auditor for a spin-1/2 particle in a uniform magnetic field along the y-axis.
    Focuses on theorem-based invariants, modeling consistency, and monotonicity.
    """

    def __init__(self, solution_instance):
        # Safely extract expectations (can be sympy expressions or floats)
        self.psi = getattr(solution_instance, 'psi', None)       # list or sympy matrix
        self.psi_ket = getattr(solution_instance, 'psi_ket', None) # string or symbolic
        self.exp_sx = getattr(solution_instance, 'exp_sx', None)
        self.exp_sy = getattr(solution_instance, 'exp_sy', None)
        self.exp_sz = getattr(solution_instance, 'exp_sz', None)

        # Try to get numerical givens from solution_instance if they exist
        self.hbar = getattr(solution_instance, 'hbar', None)
        self.gamma = getattr(solution_instance, 'gamma', None)
        self.B = getattr(solution_instance, 'B', None)
        self.t = getattr(solution_instance, 't', None)

    def _numeric_eval(self, expr, subs_dict=None):
        """Try to evaluate an expression to a float. If symbolic and subs_dict provided, substitute first."""
        try:
            if subs_dict:
                expr = expr.subs(subs_dict)
            return float(sp.N(expr))
        except Exception:
            return None

    # --------------------------------------------------------------------
    # Interface checks
    # --------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that at least the expectation values are present."""
        if self.exp_sx is None or self.exp_sy is None or self.exp_sz is None:
            return (False, "Missing one or more expectation results (exp_sx, exp_sy, exp_sz)")
        return (True, "All required expectation keys present")

    def test_interface_psi_initial_present(self):
        """Check that psi or psi_ket is provided."""
        if self.psi is None and self.psi_ket is None:
            return (False, "Neither psi nor psi_ket provided")
        return (True, "State representation available")

    # --------------------------------------------------------------------
    # Invariant tests (theorem-based, use math.isclose or sympy)
    # --------------------------------------------------------------------
    def test_invariant_normalization(self):
        """|psi|^2 must be 1 (unitarity of time evolution)."""
        if self.psi is None:
            return (False, "SKIP: psi not provided")
        try:
            # psi could be a list [c1, c2] or a sympy Matrix
            if hasattr(self.psi, 'shape'):  # sympy Matrix
                norm_sq = (self.psi.H * self.psi).simplify()
                if norm_sq == 1:
                    return (True, "Normalization holds symbolically")
                else:
                    return (False, f"Symbolic norm^2 = {norm_sq} ≠ 1")
            else:  # assume list
                a, b = self.psi[0], self.psi[1]
                if callable(getattr(a, 'is_Float', None)):  # numeric
                    norm_sq = abs(a)**2 + abs(b)**2
                    if math.isclose(norm_sq, 1.0, rel_tol=1e-2):
                        return (True, f"Numeric norm^2 = {norm_sq:.4f} ≈ 1")
                    else:
                        return (False, f"Numeric norm^2 = {norm_sq:.4f} ≠ 1")
                else:  # symbolic but no shape
                    return (False, "SKIP: cannot determine normalization (symbolic unstructured)")
        except Exception as e:
            return (False, f"Normalization check crashed: {e}")

    def test_invariant_sy_zero(self):
        """⟨s_y⟩ must be zero because [s_y, H]=0 and initial state has ⟨s_y⟩=0."""
        if self.exp_sy is None:
            return (False, "SKIP: exp_sy missing")
        try:
            val = sp.N(self.exp_sy) if hasattr(self.exp_sy, 'subs') else self.exp_sy
            if isinstance(val, sp.Number):
                if math.isclose(float(val), 0.0, rel_tol=1e-2):
                    return (True, f"⟨s_y⟩ = {float(val):.4f} ≈ 0")
                else:
                    return (False, f"⟨s_y⟩ = {float(val):.4f} ≠ 0")
            else:
                simp = sp.simplify(self.exp_sy)
                if simp == 0:
                    return (True, "⟨s_y⟩ = 0 symbolically")
                else:
                    return (False, f"⟨s_y⟩ does not simplify to zero: {simp}")
        except Exception as e:
            return (False, f"⟨s_y⟩ check crashed: {e}")

    def test_invariant_total_spin_magnitude(self):
        """Check that (2/ħ)²(⟨s_x⟩² + ⟨s_z⟩²) = 1 when ⟨s_y⟩=0."""
        if self.exp_sx is None or self.exp_sz is None or self.hbar is None:
            return (False, "SKIP: missing exp_sx, exp_sz or hbar")
        if self.exp_sy is None or (hasattr(self.exp_sy, 'subs') and sp.N(self.exp_sy) != 0):
            return (False, "SKIP: ⟨s_y⟩ is not zero, total spin not conserved in this simple form")
        try:
            hbar = sp.N(self.hbar) if hasattr(self.hbar, 'subs') else self.hbar
            sx = self.exp_sx
            sz = self.exp_sz
            lhs = (2.0/hbar)**2 * (sx**2 + sz**2)
            lhs_s = sp.simplify(lhs)
            if lhs_s.is_Number:
                if math.isclose(float(lhs_s), 1.0, rel_tol=1e-2):
                    return (True, f"(2/ħ)²(⟨s_x⟩²+⟨s_z⟩²) = {float(lhs_s):.4f} ≈ 1")
                else:
                    return (False, f"(2/ħ)²(⟨s_x⟩²+⟨s_z⟩²) = {float(lhs_s):.4f} ≠ 1")
            else:
                if lhs_s == 1:
                    return (True, "Invariant holds symbolically")
                else:
                    return (False, f"Invariant expression = {lhs_s} (not symbolically 1)")
        except Exception as e:
            return (False, f"Total spin magnitude check crashed: {e}")

    # --------------------------------------------------------------------
    # Modeling / trace tests (use isclose)
    # --------------------------------------------------------------------
    def test_modeling_psi_initial_value(self):
        """At t=0, |ψ⟩ must be |+⟩ = (1,0)."""
        if self.psi is None:
            return (False, "SKIP: psi missing")
        try:
            psi_t = self.psi
            if hasattr(psi_t, 'subs') and hasattr(psi_t, 'free_symbols'):
                t_sym = sp.Symbol('t')
                psi0 = psi_t.subs(t_sym, 0)
                if hasattr(psi0, '__getitem__'):
                    a = sp.N(psi0[0])
                    b = sp.N(psi0[1]) if len(psi0)>1 else 0
                    if math.isclose(float(a), 1.0, abs_tol=1e-2) and math.isclose(float(b), 0.0, abs_tol=1e-2):
                        return (True, "ψ(t=0) ≈ (1,0)")
                    else:
                        return (False, f"ψ(t=0) = ({float(a):.4f}, {float(b):.4f})")
            return (False, "SKIP: cannot evaluate psi at t=0")
        except Exception as e:
            return (False, f"Modeling psi initial crashed: {e}")

    def test_modeling_sx_zero_at_tzero(self):
        """⟨s_x⟩ should be 0 at t=0 because |+⟩ has zero x-component."""
        if self.exp_sx is None:
            return (False, "SKIP: exp_sx missing")
        try:
            sx = self.exp_sx
            if hasattr(sx, 'subs'):
                t_sym = sp.Symbol('t')
                sx0 = sp.N(sx.subs(t_sym, 0))
            else:
                sx0 = sp.N(sx)
            if math.isclose(float(sx0), 0.0, abs_tol=1e-2):
                return (True, f"⟨s_x⟩(0) = {float(sx0):.4f} ≈ 0")
            else:
                return (False, f"⟨s_x⟩(0) = {float(sx0):.4f} ≠ 0")
        except Exception as e:
            return (False, f"Modeling sx initial crashed: {e}")

    def test_modeling_sz_max_at_tzero(self):
        """⟨s_z⟩ at t=0 should be ħ/2."""
        if self.exp_sz is None or self.hbar is None:
            return (False, "SKIP: missing exp_sz or hbar")
        try:
            hbar_s = sp.N(self.hbar) if hasattr(self.hbar, 'subs') else self.hbar
            sz = self.exp_sz
            if hasattr(sz, 'subs'):
                t_sym = sp.Symbol('t')
                sz0 = sp.N(sz.subs(t_sym, 0))
            else:
                sz0 = sp.N(sz)
            target = hbar_s / 2
            if math.isclose(float(sz0), float(target), rel_tol=1e-2):
                return (True, f"⟨s_z⟩(0) ≈ ħ/2 = {float(target):.4f}")
            else:
                return (False, f"⟨s_z⟩(0) = {float(sz0):.4f} ≠ ħ/2 = {float(target):.4f}")
        except Exception as e:
            return (False, f"Modeling sz initial crashed: {e}")

    def test_trace_expectation_forms(self):
        """Check that expectation expressions match the expected mathematical forms:
        ⟨s_x⟩ ∝ sin, ⟨s_z⟩ ∝ cos, ⟨s_y⟩ = 0."""
        if self.exp_sx is None or self.exp_sz is None:
            return (False, "SKIP: missing exp_sx or exp_sz")
        try:
            sx = sp.sympify(self.exp_sx) if not isinstance(self.exp_sx, sp.Basic) else self.exp_sx
            sz = sp.sympify(self.exp_sz) if not isinstance(self.exp_sz, sp.Basic) else self.exp_sz
            has_sin_x = sx.has(sp.sin)
            has_cos_z = sz.has(sp.cos)
            if has_sin_x and has_cos_z:
                sy = sp.sympify(self.exp_sy) if self.exp_sy is not None else sp.Integer(0)
                if sp.simplify(sy) == 0:
                    return (True, "Expectations have correct trigonometric forms: ⟨s_x⟩ ∝ sin, ⟨s_z⟩ ∝ cos, ⟨s_y⟩=0")
                else:
                    return (False, "⟨s_y⟩ is not zero as expected")
            else:
                return (False, "⟨s_x⟩ or ⟨s_z⟩ do not contain expected trig functions")
        except Exception as e:
            return (False, f"Trace expectation forms crashed: {e}")

    # --------------------------------------------------------------------
    # Monotonicity tests
    # --------------------------------------------------------------------
    def test_monotonicity_sz_decrease_at_origin(self):
        """
        Explicitly perturb the time input from t=0 to a small positive t.
        Baseline: ⟨s_z⟩(0) = ħ/2.
        Perturbed: ⟨s_z⟩(ε). For small ε>0, ⟨s_z⟩ should be smaller (since cos decreases).
        """
        if self.exp_sz is None or self.hbar is None:
            return (False, "SKIP: missing exp_sz or hbar")
        try:
            sz = sp.sympify(self.exp_sz) if not isinstance(self.exp_sz, sp.Basic) else self.exp_sz
            hbar_s = sp.sympify(self.hbar) if not isinstance(self.hbar, sp.Basic) else self.hbar

            # Define placeholder values for constants if they are symbolic
            subs_map = {}
            for sym in sz.free_symbols:
                if sym.name == 't':
                    continue
                # Use value 1 for any other symbols (gamma, B, hbar)
                subs_map[sym] = 1.0
            if hbar_s in subs_map:
                hbar_val = 1.0
            else:
                hbar_val = float(sp.N(hbar_s))

            # Baseline at t=0
            t_sym = sp.Symbol('t')
            sz_base = sz.subs(t_sym, 0).subs(subs_map)
            sz_base_val = float(sp.N(sz_base))

            # Perturbed: t = 0.1 (small positive)
            epsilon = 0.1
            sz_pert = sz.subs(t_sym, epsilon).subs(subs_map)
            sz_pert_val = float(sp.N(sz_pert))

            # Since ⟨s_z⟩ = (hbar/2) cos(gamma B t), it decreases from maximum at t=0.
            if sz_pert_val < sz_base_val - 1e-12:
                return (True, f"Perturb t from 0 to {epsilon}: ⟨s_z⟩ goes from {sz_base_val:.6f} to {sz_pert_val:.6f} (decrease)")
            else:
                return (False, f"Perturb t from 0 to {epsilon}: ⟨s_z⟩ = {sz_pert_val:.6f} not less than baseline {sz_base_val:.6f}")
        except Exception as e:
            return (False, f"Monotonicity sz crashed: {e}")

    def test_monotonicity_sx_negative_small_t(self):
        """
        Baseline: ⟨s_x⟩(0) = 0.
        Perturb to a small positive t: ⟨s_x⟩ should become negative (since -sin(gamma B t) < 0 for small positive argument).
        """
        if self.exp_sx is None:
            return (False, "SKIP: missing exp_sx")
        try:
            sx = sp.sympify(self.exp_sx) if not isinstance(self.exp_sx, sp.Basic) else self.exp_sx

            # Build substitution map for symbolic constants (excluding t)
            subs_map = {}
            for sym in sx.free_symbols:
                if sym.name == 't':
                    continue
                subs_map[sym] = 1.0

            # Baseline at t=0
            t_sym = sp.Symbol('t')
            sx_base = sx.subs(t_sym, 0).subs(subs_map)
            sx_base_val = float(sp.N(sx_base))

            # Perturbed: t = 0.1
            epsilon = 0.1
            sx_pert = sx.subs(t_sym, epsilon).subs(subs_map)
            sx_pert_val = float(sp.N(sx_pert))

            # Expect ⟨s_x⟩(ε) < 0 for ε>0 (since -ħ/2 sin(γ B t) is negative for small positive argument)
            if sx_pert_val < sx_base_val - 1e-12:
                return (True, f"Perturb t from 0 to {epsilon}: ⟨s_x⟩ goes from {sx_base_val:.6f} to {sx_pert_val:.6f} (negative)")
            else:
                return (False, f"Perturb t from 0 to {epsilon}: ⟨s_x⟩ = {sx_pert_val:.6f} not less than baseline {sx_base_val:.6f}")
        except Exception as e:
            return (False, f"Monotonicity sx crashed: {e}")

    # --------------------------------------------------------------------
    # Optional weak answer sanity (must avoid re-deriving final answers)
    # --------------------------------------------------------------------
    def test_answer_sanity_signs_and_range(self):
        """
        Very weak check: all expectation values must be between -ħ/2 and ħ/2,
        and ⟨s_y⟩ should be zero.
        (We do not compute expected values from scratch.)
        """
        if self.hbar is None:
            return (False, "SKIP: hbar missing")
        try:
            hbar_val = float(sp.N(sp.sympify(self.hbar)))
            limit = hbar_val / 2.0
            for name, expr in [('exp_sx', self.exp_sx), ('exp_sy', self.exp_sy), ('exp_sz', self.exp_sz)]:
                if expr is None:
                    continue
                # Try to evaluate numerically; if fails, skip
                val = self._numeric_eval(expr)
                if val is None:
                    continue
                if val < -limit - 1e-9 or val > limit + 1e-9:
                    return (False, f"{name} = {val:.4f} out of range [{-limit:.4f}, {limit:.4f}]")
                if name == 'exp_sy' and abs(val) > 1e-9:
                    return (False, f"exp_sy = {val:.4e} is not zero")
            return (True, "All expectation values within physical range and ⟨s_y⟩ ≈ 0")
        except Exception as e:
            return (False, f"Answer sanity crashed: {e}")

    # --------------------------------------------------------------------
    # Runner
    # --------------------------------------------------------------------
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