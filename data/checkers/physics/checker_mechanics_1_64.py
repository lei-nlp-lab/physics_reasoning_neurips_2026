import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    """
    Robust auditor for the Interstellar Bullets problem.
    Extracts solution_instance results and performs unit-test-like checks.
    All tests are theorem/modeling based; no hardcoded target answer comparisons.
    """

    def __init__(self, solution_instance):
        # Safe attribute access: default to None
        self.results = getattr(solution_instance, 'results', {})
        self.givens = getattr(solution_instance, 'givens', {})

        # Emitted results (strings or symbolic expressions)
        self.F_inside = self.results.get('F_inside', None)
        self.V_inside = self.results.get('V_inside', None)
        self.r_plus_expr = self.results.get('r_plus', None)
        self.r_minus_expr = self.results.get('r_minus', None)
        self.dtheta_dr = self.results.get('dtheta_dr', None)
        self.orbit_eq = self.results.get('orbit_eq', None)

        # Givens (numerical or symbolic)
        self.G = self.givens.get('G', None)
        self.M = self.givens.get('M', None)
        self.m = self.givens.get('m', None)
        self.R = self.givens.get('R', None)

        # Common symbols
        self.r, self.R_sym = sp.symbols('r R', positive=True)
        self.G_sym, self.M_sym, self.m_sym = sp.symbols('G M m', positive=True)
        self.theta = sp.symbols('theta', real=True)

    def _parse_expr(self, expr_str):
        """Safely parse a string into a sympy expression."""
        if expr_str is None:
            return None
        try:
            return sp.sympify(expr_str)
        except (sp.SympifyError, TypeError):
            return None

    def _missing_msg(self, key):
        return f"SKIP: Required result '{key}' is None"

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_keys(self) -> Tuple[bool, str]:
        """Check that all expected result keys are present and non‑empty."""
        expected = {
            'F_inside': self.F_inside,
            'V_inside': self.V_inside,
            'r_plus': self.r_plus_expr,
            'r_minus': self.r_minus_expr,
            'dtheta_dr': self.dtheta_dr,
            'orbit_eq': self.orbit_eq,
        }
        for key, val in expected.items():
            if val is None:
                return False, f"Missing key: {key}"
            if isinstance(val, str) and len(val.strip()) == 0:
                return False, f"Empty string for key: {key}"
        return True, "All required keys present and non‑empty"

    # ------------------------------------------------------------------
    # Invariant checks (physics truths)
    # ------------------------------------------------------------------
    def test_invariant_potential_continuity(self) -> Tuple[bool, str]:
        """V(r) must be continuous at r = R."""
        if self.V_inside is None:
            return False, self._missing_msg("V_inside")
        V_inside = self._parse_expr(self.V_inside)
        if V_inside is None:
            return False, "Cannot parse V_inside"
        # V_outside at r=R: -G*M*m/R
        V_outside = -self.G_sym * self.M_sym * self.m_sym / self.R_sym
        # Evaluate V_inside at r=R
        try:
            V_inside_at_R = V_inside.subs({self.r: self.R_sym}).simplify()
        except Exception:
            return False, "Symbolic substitution failed"
        diff = sp.simplify(V_inside_at_R - V_outside)
        if diff == 0:
            return True, "Potential is continuous at r=R (symbolic)"
        # Numeric fallback if G,M,m,R are given
        if all(v is not None for v in [self.G, self.M, self.m, self.R]):
            try:
                num_diff = float(diff.subs({
                    self.G_sym: self.G,
                    self.M_sym: self.M,
                    self.m_sym: self.m,
                    self.R_sym: self.R
                }).evalf())
            except Exception:
                return False, "Numeric evaluation failed"
            if math.isclose(num_diff, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                return True, "Potential is continuous at r=R (numeric)"
            else:
                return False, f"Potential discontinuity: diff={num_diff:.4f}"
        return False, "Cannot verify continuity (no numeric fallback, symbolic not exact zero)"

    def test_invariant_force_sign(self) -> Tuple[bool, str]:
        """Force inside must be centripetal (negative radial)."""
        if self.F_inside is None:
            return False, self._missing_msg("F_inside")
        F_expr = self._parse_expr(self.F_inside)
        if F_expr is None:
            return False, "Cannot parse F_inside"
        # Expected form: -G*M*m*r/R^3
        expected = -self.G_sym * self.M_sym * self.m_sym * self.r / self.R_sym**3
        diff = sp.simplify(F_expr - expected)
        if diff == 0:
            return True, "F_inside matches shell theorem form"
        # numeric fallback
        if all(v is not None for v in [self.G, self.M, self.m, self.R]):
            test_r = 0.5 * self.R
            subs = {self.G_sym: self.G, self.M_sym: self.M,
                    self.m_sym: self.m, self.R_sym: self.R, self.r: test_r}
            try:
                F_num = float(F_expr.subs(subs).evalf())
                F_expected = float(expected.subs(subs).evalf())
            except Exception:
                return False, "Numeric evaluation failed"
            if math.isclose(F_num, F_expected, rel_tol=1e-2, abs_tol=1e-2):
                return True, "F_inside numeric match"
            else:
                return False, f"Numeric mismatch: got {F_num:.4f}, expected {F_expected:.4f}"
        return False, "Cannot verify force (no numeric fallback, symbolic not exact)"

    def test_invariant_turning_equation(self) -> Tuple[bool, str]:
        """The turning points must satisfy 32u^2 - 16u + 1 = 0, u = (r/R)^2.
           This equation is derived from E = V_eff inside the cloud, not from any target answer.
        """
        if self.r_plus_expr is None or self.r_minus_expr is None:
            return False, self._missing_msg("r_plus or r_minus")
        rp = self._parse_expr(self.r_plus_expr)
        rm = self._parse_expr(self.r_minus_expr)
        if rp is None or rm is None:
            return False, "Cannot parse turning point expressions"
        u = sp.symbols('u')
        quad = 32*u**2 - 16*u + 1
        u_plus = sp.simplify((rp / self.R_sym)**2)
        u_minus = sp.simplify((rm / self.R_sym)**2)
        for name, u_val in [("r_plus", u_plus), ("r_minus", u_minus)]:
            val = sp.simplify(quad.subs(u, u_val))
            if val == 0:
                continue
            # numeric fallback if R is numeric
            if self.R is not None:
                try:
                    num_val = float(val.subs(self.R_sym, self.R).evalf())
                except Exception:
                    return False, f"Cannot evaluate quad at {name}"
                if math.isclose(num_val, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                    continue
                return False, f"{name} does not satisfy turning point quadratic: residual={num_val:.4f}"
            else:
                return False, f"{name} not verified (no numeric fallback, symbolic not zero)"
        return True, "Both turning points satisfy the quadratic equation"

    # ------------------------------------------------------------------
    # Modeling checks (follow reasoning steps, no hardcoded constants)
    # ------------------------------------------------------------------
    def test_modeling_effective_potential(self) -> Tuple[bool, str]:
        """Check that at turning points, E = V_eff (inside region) using given L and E."""
        if any(v is None for v in [self.r_plus_expr, self.r_minus_expr, self.V_inside]):
            return False, "Missing required data for effective potential check"
        rp = self._parse_expr(self.r_plus_expr)
        rm = self._parse_expr(self.r_minus_expr)
        V_inside = self._parse_expr(self.V_inside)
        if None in (rp, rm, V_inside):
            return False, "Parsing failed"
        # Problem given E = -5*G*M*m/(4*R) and L^2 = m^2*G*M*R/32
        E = -5 * self.G_sym * self.M_sym * self.m_sym / (4 * self.R_sym)
        L_sq_over_2m = self.m_sym * self.G_sym * self.M_sym * self.R_sym / 64
        # V_eff_inside = V_inside + L^2/(2*m*r^2)
        V_eff = V_inside + L_sq_over_2m / (self.r**2)
        for name, r_val in [("r_plus", rp), ("r_minus", rm)]:
            eq = sp.simplify(E - V_eff.subs(self.r, r_val))
            if eq == 0:
                continue
            # numeric fallback
            if all(v is not None for v in [self.G, self.M, self.m, self.R]):
                subs = {self.G_sym: self.G, self.M_sym: self.M,
                        self.m_sym: self.m, self.R_sym: self.R}
                try:
                    num_eq = float(eq.subs(subs).evalf())
                except Exception:
                    return False, f"Cannot evaluate eq at {name}"
                if math.isclose(num_eq, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                    continue
                return False, f"E = V_eff fails at {name}: residual {num_eq:.4f}"
            else:
                return False, f"Cannot verify E = V_eff at {name} (no numeric fallback)"
        return True, "Effective potential equation satisfied at both turning points"

    def test_modeling_dtheta_derivation(self) -> Tuple[bool, str]:
        """Check that dtheta/dr expression is consistent with energy and angular momentum conservation."""
        if self.dtheta_dr is None or self.V_inside is None:
            return False, self._missing_msg("dtheta_dr or V_inside")
        dtheta = self._parse_expr(self.dtheta_dr)
        V_inside = self._parse_expr(self.V_inside)
        if None in (dtheta, V_inside):
            return False, "Parsing failed"
        # Reconstruct derived (dtheta/dr)^2 from conservation
        L_sq = self.m_sym**2 * self.G_sym * self.M_sym * self.R_sym / 32
        E = -5 * self.G_sym * self.M_sym * self.m_sym / (4 * self.R_sym)
        V_eff = V_inside + L_sq / (2 * self.m_sym * self.r**2)
        denom = self.r**4 * 2 * self.m_sym * (E - V_eff)
        rhs_sq = L_sq / denom  # (dtheta/dr)^2 expected
        lhs_sq = dtheta**2
        diff = sp.simplify(lhs_sq - rhs_sq)
        if diff == 0:
            return True, "dtheta/dr matches derived form symbolically"
        # numeric fallback
        if all(v is not None for v in [self.G, self.M, self.m, self.R]):
            test_r = 0.7 * self.R  # interior point
            subs = {self.G_sym: self.G, self.M_sym: self.M,
                    self.m_sym: self.m, self.R_sym: self.R, self.r: test_r}
            try:
                lhs_num = float(lhs_sq.subs(subs).evalf())
                rhs_num = float(rhs_sq.subs(subs).evalf())
            except Exception:
                return False, "Numeric evaluation failed"
            if math.isclose(lhs_num, rhs_num, rel_tol=1e-2, abs_tol=1e-2):
                return True, "dtheta/dr matches numerically"
            else:
                return False, f"Numeric mismatch: lhs^2={lhs_num:.4f}, rhs^2={rhs_num:.4f}"
        return False, "Cannot verify dtheta/dr (symbolic not exact, no numeric fallback)"

    def test_modeling_orbit_equation_differential(self) -> Tuple[bool, str]:
        """Check that orbit_eq satisfies the differential equation dtheta_dr.
           That is, differentiate orbit_eq and compare to dtheta/dr expression.
        """
        if self.orbit_eq is None or self.dtheta_dr is None:
            return False, self._missing_msg("orbit_eq or dtheta_dr")
        orbit = self._parse_expr(self.orbit_eq)
        dtheta = self._parse_expr(self.dtheta_dr)
        if orbit is None or dtheta is None:
            return False, "Parsing failed"
        # Expect orbit_eq = Equality: (r/R)^2 = f(theta)
        if not isinstance(orbit, sp.Eq):
            return False, "orbit_eq must be a sympy Equality"
        # Differentiate orbit implicitly: d/dtheta of LHS and RHS
        lhs = orbit.lhs   # e.g., (r/R)^2
        rhs = orbit.rhs   # expression in theta
        # Differentiate wrt theta
        try:
            dlhs_dtheta = sp.diff(lhs, self.theta)  # this will give (2*r/R^2)*dr/dtheta
            drhs_dtheta = sp.diff(rhs, self.theta)
        except Exception:
            return False, "Differentiation failed"
        # Solve for dr/dtheta from orbit_eq: dlhs_dtheta = drhs_dtheta
        # We need to isolate dr/dtheta
        # Let's set variable dr_dtheta = sp.symbols('dr_dtheta')
        # Express dlhs_dtheta in terms of dr_dtheta: dlhs_dtheta = (2*r/R^2)*dr_dtheta
        dr_dtheta = sp.symbols('dr_dtheta')
        # We'll symbolically solve: (2*r/R^2)*dr_dtheta - drhs_dtheta = 0
        eq_diff = sp.Eq((2*lhs**(1/2)/self.R_sym)*dr_dtheta, drhs_dtheta)  # since r/R = sqrt(lhs)
        sol = sp.solve(eq_diff, dr_dtheta)
        if not sol:
            return False, "Cannot solve for dr/dtheta"
        derived_dr_dtheta = sol[0]
        # Then dtheta/dr = 1 / derived_dr_dtheta
        derived_dtheta_dr = 1 / derived_dr_dtheta
        # Compare to given dtheta expression (which is dtheta/dr)
        diff = sp.simplify(derived_dtheta_dr - dtheta)
        if diff == 0:
            return True, "Orbit equation satisfies the differential equation"
        # numeric fallback
        if self.R is not None:
            test_theta = 0.5  # rad
            subs = {self.R_sym: self.R, self.r: self.R * 0.7, self.theta: test_theta}
            # But we need r consistent with orbit_eq; get r from orbit_eq
            try:
                r_from_orbit = sp.solve(orbit, self.r)
                if not r_from_orbit:
                    raise ValueError("No solution for r")
                r_val = r_from_orbit[0].subs({self.theta: test_theta, self.R_sym: self.R})
                if r_val.is_number:
                    r_num = float(r_val)
                else:
                    r_num = float(r_val.evalf())
            except Exception:
                return False, "Cannot extract r from orbit_eq"
            subs = {self.R_sym: self.R, self.r: r_num, self.theta: test_theta}
            try:
                derived_num = float(derived_dtheta_dr.subs(subs).evalf())
                given_num = float(dtheta.subs(subs).evalf())
            except Exception:
                return False, "Numeric evaluation failed"
            if math.isclose(derived_num, given_num, rel_tol=1e-2, abs_tol=1e-2):
                return True, "Orbit equation differential holds numerically"
            else:
                return False, f"Derivative mismatch: derived={derived_num:.4f}, given={given_num:.4f}"
        return False, "Cannot verify orbit equation (symbolic not exact, no numeric fallback)"

    # ------------------------------------------------------------------
    # Monotonicity check (must perturb input and compare)
    # ------------------------------------------------------------------
    def test_monotonicity_dtheta_increasing_intensity(self) -> Tuple[bool, str]:
        """As r increases from r_minus to r_plus, dtheta/dr should remain positive.
           Perturb: increase r by 10% and check that dtheta/dr stays positive.
        """
        if self.dtheta_dr is None or self.r_plus_expr is None or self.r_minus_expr is None:
            return False, self._missing_msg("dtheta_dr, r_plus, r_minus")
        dtheta = self._parse_expr(self.dtheta_dr)
        rp = self._parse_expr(self.r_plus_expr)
        rm = self._parse_expr(self.r_minus_expr)
        if None in (dtheta, rp, rm):
            return False, "Parsing failed"
        # We need numeric values for R to compute actual r_minus and r_plus
        if self.R is None:
            return False, "R not provided, cannot evaluate r_minus and r_plus numerically"
        try:
            r_minus_num = float(rm.subs(self.R_sym, self.R).evalf())
            r_plus_num = float(rp.subs(self.R_sym, self.R).evalf())
        except Exception:
            return False, "Cannot compute numeric turning points"
        # Choose two test r values inside the interval
        r_test1 = r_minus_num + 0.1 * (r_plus_num - r_minus_num)  # near min
        r_test2 = r_test1 * 1.1  # perturb up by 10%
        subs1 = {self.r: r_test1, self.R_sym: self.R}
        subs2 = {self.r: r_test2, self.R_sym: self.R}
        try:
            val1 = float(dtheta.subs(subs1).evalf())
            val2 = float(dtheta.subs(subs2).evalf())
        except Exception:
            return False, "Cannot evaluate dtheta at test points"
        if val1 <= 0 or val2 <= 0:
            return False, f"dtheta/dr not positive: at r={r_test1:.4f}: {val1:.4f}, at r={r_test2:.4f}: {val2:.4f}"
        # Check that both values are finite
        if not (math.isfinite(val1) and math.isfinite(val2)):
            return False, "dtheta/dr not finite"
        return True, "dtheta/dr positive and finite at two interior points (perturbed by 10%)"

    # ------------------------------------------------------------------
    # Weak answer sanity (only one, no target values)
    # ------------------------------------------------------------------
    def test_answer_sanity_turning_points_inside_cloud(self) -> Tuple[bool, str]:
        """Check that both turning points are smaller than R (bullet stays inside the cloud)."""
        if self.r_plus_expr is None or self.r_minus_expr is None:
            return False, self._missing_msg("r_plus or r_minus")
        rp = self._parse_expr(self.r_plus_expr)
        rm = self._parse_expr(self.r_minus_expr)
        if None in (rp, rm):
            return False, "Parsing failed"
        if self.R is None:
            return False, "R not given, cannot verify turning point ranges"
        try:
            r_plus_num = float(rp.subs(self.R_sym, self.R).evalf())
            r_minus_num = float(rm.subs(self.R_sym, self.R).evalf())
        except Exception:
            return False, "Cannot evaluate turning points numerically"
        if r_plus_num <= self.R and r_minus_num >= 0:
            return True, f"Both turning points lie inside: r_minus={r_minus_num:.4f}, r_plus={r_plus_num:.4f} <= R={self.R}"
        else:
            return False, f"Turning points not inside: r_minus={r_minus_num:.4f}, r_plus={r_plus_num:.4f}, R={self.R}"

    # ------------------------------------------------------------------
    # Test runner
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