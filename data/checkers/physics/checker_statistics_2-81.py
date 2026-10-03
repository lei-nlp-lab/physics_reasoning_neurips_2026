import math
import sympy as sp
import traceback
from typing import Any

class PhysicsChecker:
    """
    Audits a solution for the problem:
    Graphite low-temperature specific heat – Debye theory adaptation.
    Focus: presence of quadratic dispersion (flexural mode) leading to C_V ∝ T.
    The solution may provide C_V, area, sound speeds, and possibly the density of states.
    """

    def __init__(self, solution_instance: Any):
        # Extract all relevant attributes safely; never assign numeric defaults for missing data.
        self.C_V = getattr(solution_instance, 'C_V', None)
        # If results dict exists (from pseudocode emit), try it
        if self.C_V is None and hasattr(solution_instance, 'results'):
            self.C_V = solution_instance.results.get('C_V', None)
        # Also try to extract other potentially useful quantities
        self.T = getattr(solution_instance, 'T', None)
        self.area = getattr(solution_instance, 'A', None)
        self.c_para = getattr(solution_instance, 'c_para', None)
        self.c_perp = getattr(solution_instance, 'c_perp', None)
        self.g_omega = getattr(solution_instance, 'g_omega', None)   # density of states (symbolic or numeric)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Convenience: if T is inside givens (solutions often have input dict)
        self.givens = getattr(solution_instance, 'givens', {})
        if self.T is None:
            self.T = self.givens.get('T', None)
        if self.area is None:
            self.area = self.givens.get('A', self.givens.get('L^2', None))
        if self.c_para is None:
            self.c_para = self.givens.get('c_para', self.givens.get('c_parallel', None))
        if self.c_perp is None:
            self.c_perp = self.givens.get('c_perp', self.givens.get('c_perp', None))
        # For scaling tests we need baseline T; if still missing, use a fictitious one.
        self.baseline_T = self.T if self.T is not None else 1.0   # only for monotonicity tests

    # -----------------------------------------------------------------
    # Interface checks
    # -----------------------------------------------------------------
    def test_interface_C_V_present(self):
        if self.C_V is None:
            return False, "C_V missing from solution"
        return True, "C_V is present"

    def test_interface_C_V_type(self):
        if self.C_V is None:
            return False, "SKIP: C_V not available"
        if not (isinstance(self.C_V, (int, float, sp.Expr, str))):
            return False, f"C_V type {type(self.C_V)} not numeric or sympy"
        return True, f"C_V type is valid ({type(self.C_V).__name__})"

    # -----------------------------------------------------------------
    # Physical invariants
    # -----------------------------------------------------------------
    def test_invariant_C_V_positive(self):
        """Specific heat must be positive for T>0."""
        if self.C_V is None:
            return False, "SKIP: C_V not available"
        try:
            val = self._numeric_C_V()
            if val <= 0:
                return False, f"C_V = {val} is not positive"
            return True, f"C_V = {val} > 0"
        except Exception as e:
            return False, f"Cannot evaluate C_V positivity: {e}"

    def test_invariant_low_T_limit_zero(self):
        """At T=0, C_V should be zero."""
        if self.C_V is None or self.baseline_T is None:
            return False, "SKIP: C_V or T not available"
        try:
            expr = self._sympy_C_V()
            # substitute T -> 0
            limit = sp.limit(expr, sp.Symbol('T'), 0)
            if limit is sp.nan or limit == 0:
                return True, f"C_V(T->0) = {limit}"
            else:
                return False, f"C_V(T->0) = {limit}, expected 0"
        except Exception as e:
            return False, f"Limit test failed: {e}"

    def test_invariant_C_V_increases_with_T(self):
        """C_V should be an increasing function of T (for T>0, up to saturation)."""
        if self.C_V is None or self.baseline_T is None:
            return False, "SKIP: C_V or T not available"
        try:
            expr = self._sympy_C_V()
            # derivative w.r.t. T
            deriv = sp.diff(expr, sp.Symbol('T'))
            # evaluate at a small positive T (e.g., 1 K)
            subs = {sp.Symbol('T'): 1}
            # also substitute area, sound speeds if they are symbols
            if isinstance(self.area, (int, float)):
                subs[sp.Symbol('A')] = self.area
            if isinstance(self.c_para, (int, float)):
                subs[sp.Symbol('c_para')] = self.c_para
            if isinstance(self.c_perp, (int, float)):
                subs[sp.Symbol('c_perp')] = self.c_perp
            val = float(deriv.subs(subs))
            if val > 0:
                return True, f"dC_V/dT at T=1 K = {val} > 0"
            else:
                return False, f"dC_V/dT = {val} <= 0"
        except Exception as e:
            return False, f"Derivative test failed: {e}"

    # -----------------------------------------------------------------
    # Modeling / trace checks
    # -----------------------------------------------------------------
    def test_modeling_low_T_scaling_is_physical(self):
        """
        At very low T, C_V should scale as T (from flexural mode) or T^2 (from linear branches).
        Our test: if we can extract the power of T from the symbolic expression, verify it is 1 or 2.
        """
        if self.C_V is None:
            return False, "SKIP: C_V not available"
        try:
            expr = self._sympy_C_V()
            # assume expression is of form constant * T^n; extract power
            # handle rational powers
            t = sp.Symbol('T')
            poly = sp.expand(expr)
            # get all monomials in T
            monoms = poly.as_coefficients_dict() if poly.is_Add else {poly: 1}
            # find the power of T with highest coefficient (dominant at low T)
            # but simpler: compute limit of C_V/T as T->0: if limit finite -> exponent 1
            limit_ratio = sp.limit(expr / t, t, 0)
            if limit_ratio != sp.nan and limit_ratio != sp.oo:
                # exponent 1
                return True, "C_V ∝ T at low T (flexural mode present)"
            else:
                # check exponent 2
                limit_ratio2 = sp.limit(expr / t**2, t, 0)
                if limit_ratio2 != sp.nan and limit_ratio2 != sp.oo:
                    return True, "C_V ∝ T^2 at low T (linear modes only)"
                else:
                    return False, f"C_V scaling not T or T^2 as T->0: limit C_V/T = {limit_ratio}"
        except Exception as e:
            return False, f"Scaling test error: {e}"

    def test_modeling_flexural_mode_constant_DOS(self):
        """
        If the solution provides a density of states g(omega), check that it has a constant term,
        which is the signature of a quadratic dispersion (flexural mode).
        """
        if self.g_omega is None:
            # try to extract from explanation_steps (not reliable, skip)
            return False, "SKIP: g(omega) not provided"
        try:
            g = self._sympy_expression(self.g_omega)
            # Test for constant term: differentiate, see if derivative is zero at omega=0
            omega = sp.Symbol('omega')
            deriv = sp.diff(g, omega)
            at_zero = sp.limit(deriv, omega, 0)
            if at_zero == 0:
                return True, "g(omega) has constant term (constant as omega->0)"
            else:
                return False, f"g(omega) not constant at low omega, derivative = {at_zero}"
        except Exception as e:
            return False, f"DOS constant test error: {e}"

    # -----------------------------------------------------------------
    # Monotonicity via perturbation
    # -----------------------------------------------------------------
    def test_monotonicity_C_V_with_T(self):
        """Perturb T by +10% and check that C_V increases."""
        if self.C_V is None or self.baseline_T is None:
            return False, "SKIP: C_V or T not available"
        try:
            expr = self._sympy_C_V()
            # Build substitution dict for symbolic parameters
            subs = {}
            # use the supplied T as baseline, but may be symbolic; we want numeric
            T0 = 1.0   # use a typical low T (e.g., 1 K)
            subs[sp.Symbol('T')] = T0
            # fill area and sound speeds if they are numerical
            if isinstance(self.area, (int, float)):
                subs[sp.Symbol('A')] = self.area
            if isinstance(self.c_para, (int, float)):
                subs[sp.Symbol('c_para')] = self.c_para
            if isinstance(self.c_perp, (int, float)):
                subs[sp.Symbol('c_perp')] = self.c_perp
            # Baseline
            base = float(expr.subs(subs))
            # Perturb T by +10%
            subs[sp.Symbol('T')] = T0 * 1.1
            perturbed = float(expr.subs(subs))
            if perturbed > base:
                return True, f"C_V increased from {base:.4g} to {perturbed:.4g} when T increased 10%"
            else:
                return False, f"C_V did not increase: base={base:.4g}, perturbed={perturbed:.4g}"
        except Exception as e:
            return False, f"Monotonicity test error: {e}"

    # -----------------------------------------------------------------
    # Weak answer sanity (one method only)
    # -----------------------------------------------------------------
    def test_answer_sanity_C_V_finite(self):
        """C_V should be finite and not NaN."""
        if self.C_V is None:
            return False, "SKIP: C_V not available"
        try:
            val = self._numeric_C_V()
            if math.isfinite(val):
                return True, f"C_V = {val} is finite"
            else:
                return False, f"C_V = {val} is not finite"
        except Exception as e:
            return False, f"C_V evaluation error: {e}"

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------
    def _sympy_expression(self, obj):
        """Convert object to sympy expression, if possible."""
        if isinstance(obj, sp.Expr):
            return obj
        if isinstance(obj, str):
            return sp.parse_expr(obj)
        if isinstance(obj, (int, float)):
            return sp.nsimplify(obj)
        raise ValueError(f"Cannot convert to sympy: {obj}")

    def _numeric_C_V(self) -> float:
        """Evaluate C_V as a float, if possible."""
        if isinstance(self.C_V, (int, float)):
            return float(self.C_V)
        # sympy expression
        try:
            expr = self._sympy_expression(self.C_V)
            # Try to evaluate by substituting symbols with placeholder numbers if missing
            subs = {}
            # Only substitute if we have numerical values
            if self.T is not None:
                subs[sp.Symbol('T')] = self.T if isinstance(self.T, (int, float)) else 1.0
            if self.area is not None and isinstance(self.area, (int, float)):
                subs[sp.Symbol('A')] = self.area
            if self.c_para is not None and isinstance(self.c_para, (int, float)):
                subs[sp.Symbol('c_para')] = self.c_para
            if self.c_perp is not None and isinstance(self.c_perp, (int, float)):
                subs[sp.Symbol('c_perp')] = self.c_perp
            # For missing symbols, try to substitute 1 (might be wrong but for positivity it's ok)
            all_symbols = expr.free_symbols
            for s in all_symbols:
                if s not in subs:
                    subs[s] = 1.0   # fallback (only for positivity check)
            return float(expr.subs(subs))
        except Exception:
            # Last attempt: try direct float conversion
            if isinstance(self.C_V, str):
                # Try sympy again
                try:
                    return float(self._sympy_expression(self.C_V).evalf())
                except Exception:
                    pass
            raise ValueError("Cannot evaluate C_V numerically")

    def _sympy_C_V(self) -> sp.Expr:
        """Return C_V as a sympy expression."""
        return self._sympy_expression(self.C_V)

    # -----------------------------------------------------------------
    # Main test runner
    # -----------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # get all method names that start with test_
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