import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for the problem: Determine potential energy U(r) for orbit rθ = constant.
    """
    def __init__(self, solution_instance):
        # Safe extraction from results dictionary
        self.results = getattr(solution_instance, 'results', {})
        self.U = self.results.get('U', None)
        
        # Sympy symbols for checking
        self.m = sp.symbols('m', positive=True, real=True)
        self.h = sp.symbols('h', positive=True, real=True)
        self.r = sp.symbols('r', positive=True, real=True)
        
        # Convert U to sympy expression if it is a string
        if isinstance(self.U, str):
            try:
                self.U_expr = sp.sympify(self.U)
            except (sp.SympifyError, TypeError):
                self.U_expr = None
        else:
            self.U_expr = self.U  # assume it's already sympy or numeric

    # ---- Interface Tests ----
    def test_interface_has_U(self):
        if self.U is None:
            return (False, "SKIP: U is missing from results")
        return (True, "U present in results")

    def test_interface_U_type(self):
        if self.U is None:
            return (False, "SKIP: U is missing")
        if self.U_expr is None:
            return (False, "U could not be parsed as a symbolic expression")
        return (True, f"U is a symbolic expression: {self.U_expr}")

    # ---- Invariant Tests (theorem-based physical sanity) ----
    def test_invariant_force_from_potential(self):
        """Check that F = -dU/dr equals -m h^2 / r^3."""
        if self.U_expr is None:
            return (False, "SKIP: symbolic U not available")
        try:
            F_from_U = -sp.diff(self.U_expr, self.r)
            expected_F = -self.m * self.h**2 / self.r**3
            diff = sp.simplify(F_from_U - expected_F)
            if diff == 0:
                return (True, "Force derived from U matches Binet's formula: -m h^2 / r^3")
            else:
                return (False, f"Force mismatch: derived F = {F_from_U}, expected {expected_F}, diff = {diff}")
        except Exception as e:
            return (False, f"Error in force check: {e}")

    def test_invariant_potential_sign(self):
        """Check that U is negative for positive m, h, r (attractive force)."""
        if self.U_expr is None:
            return (False, "SKIP: symbolic U not available")
        try:
            subs = {self.m: 1, self.h: 1, self.r: 2}
            val = self.U_expr.subs(subs).evalf()
            if val < 0:
                return (True, f"U evaluated at (m=1, h=1, r=2) is negative: {val}")
            else:
                return (False, f"U evaluated at (m=1, h=1, r=2) is not negative: {val}")
        except Exception as e:
            return (False, f"Error in sign check: {e}")

    def test_invariant_potential_zero_at_infinity(self):
        """Check that limit as r->infinity of U is 0 (constant set to zero)."""
        if self.U_expr is None:
            return (False, "SKIP: symbolic U not available")
        try:
            limit_val = sp.limit(self.U_expr, self.r, sp.oo)
            if limit_val == 0:
                return (True, "Limit of U as r->infinity is 0")
            else:
                return (False, f"Limit of U as r->infinity is {limit_val}, expected 0")
        except Exception as e:
            return (False, f"Error in limit check: {e}")

    # ---- Modeling/Trace Tests ----
    def test_modeling_binet_consistency(self):
        """
        From derived U, compute force, then check that Binet's formula is consistent
        with the orbit rθ = constant (i.e., u = 1/r satisfies d²u/dθ² + u = -F/(m h² u²)).
        """
        if self.U_expr is None:
            return (False, "SKIP: symbolic U not available")
        try:
            # Define u = 1/r
            u = 1 / self.r
            # Force from potential
            F = -sp.diff(self.U_expr, self.r)
            # Right-hand side of Binet's formula: -F/(m h² u²)
            rhs = -F / (self.m * self.h**2 * u**2)
            # Simplify rhs; it should equal u because left side = u (since d²u/dθ² = 0 for u linear in θ)
            simplified = sp.simplify(rhs)
            # Check if simplified equals u
            if sp.simplify(simplified - u) == 0:
                return (True, "Binet's formula consistency: -F/(m h^2 u^2) simplifies to u")
            else:
                return (False, f"Binet consistency failed: rhs = {simplified}, expected u = {u}")
        except Exception as e:
            return (False, f"Error in modeling test: {e}")

    def test_modeling_scaling_invariance(self):
        """
        Check that U scales as 1/r^2: U(λr) = λ^{-2} U(r).
        This verifies the homogeneous degree -2 in r.
        """
        if self.U_expr is None:
            return (False, "SKIP: symbolic U not available")
        try:
            lam = sp.symbols('lam', positive=True, real=True)
            # Substitute r -> lam*r
            U_scaled = self.U_expr.subs(self.r, lam * self.r)
            # Ratio U_scaled / self.U_expr (assuming U_expr nonzero)
            ratio = sp.simplify(U_scaled / self.U_expr)
            expected = lam**(-2)
            if sp.simplify(ratio - expected) == 0:
                return (True, f"U(λr)/U(r) simplifies to λ^(-2): {ratio}")
            else:
                return (False, f"Scaling property fails: ratio = {ratio}, expected {expected}")
        except Exception as e:
            return (False, f"Error in scaling test: {e}")

    # ---- Monotonicity Test (explicit perturbation) ----
    def test_monotonicity_U_increases_with_r(self):
        """
        Perturb r upward and verify that U increases (becomes less negative).
        """
        if self.U_expr is None:
            return (False, "SKIP: symbolic U not available")
        try:
            # Use representative numerical values
            m0, h0 = 1.0, 1.0
            r0 = 2.0
            subs_base = {self.m: m0, self.h: h0, self.r: r0}
            U0 = float(self.U_expr.subs(subs_base).evalf())
            # Perturb r upward by 5%
            r1 = r0 * 1.05
            subs_pert = {self.m: m0, self.h: h0, self.r: r1}
            U1 = float(self.U_expr.subs(subs_pert).evalf())
            # For attractive potential, U increases (higher, less negative) as r increases
            if U1 > U0:
                return (True, f"U increased from {U0} at r={r0} to {U1} at r={r1}, as expected for attractive force")
            else:
                return (False, f"U did not increase: U0={U0}, U1={U1}")
        except Exception as e:
            return (False, f"Error in monotonicity check: {e}")

    # ---- Weak Answer Sanity (at most one, secondary) ----
    def test_answer_sanity_sign_and_magnitude(self):
        """
        Weak numeric sanity: check that U evaluates to a finite, negative value
        of plausible magnitude for typical parameters.
        """
        if self.U_expr is None:
            return (False, "SKIP: symbolic U not available")
        try:
            # Evaluate at representative values
            subs = {self.m: 1, self.h: 1, self.r: 2}
            val = float(self.U_expr.subs(subs).evalf())
            # Check it is finite
            if math.isfinite(val):
                # Check it is negative (attractive potential)
                if val < 0:
                    # Order-of-magnitude: for m=1, h=1, r=2, -m h^2/(2r^2) = -0.125
                    # Accept any value between -10 and 0 as plausible
                    if val > -10:
                        return (True, f"U evaluated at (1,1,2) = {val}: negative, finite, plausible magnitude")
                    else:
                        return (False, f"U evaluated at (1,1,2) = {val}: magnitude too large (plausible range -10 to 0)")
                else:
                    return (False, f"U evaluated at (1,1,2) = {val}: not negative")
            else:
                return (False, f"U evaluated at (1,1,2) is not finite")
        except Exception as e:
            return (False, f"Error in answer sanity: {e}")

    # ---- Run All Tests ----
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)