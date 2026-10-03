import math
import sympy as sp
from typing import Any, Dict, List, Tuple, Optional, Callable

class PhysicsChecker:
    """
    Audits the solution for: U(r) from orbit r = a e^{-bθ} using Binet's formula.
    """
    def __init__(self, solution_instance: Any) -> None:
        # Extract results and givens safely
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        self.givens: Dict[str, Any] = getattr(solution_instance, 'givens', {})
        self.explanation_steps: Optional[str] = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode: Optional[str] = getattr(solution_instance, 'pseudocode', None)

        # Result key: 'U'
        self.U_raw: Any = self.results.get('U', None)

        # Attempt to extract parameters a, b, h from givens or results
        self.a = self.givens.get('a', None)
        self.b = self.givens.get('b', None)
        self.h = self.givens.get('h', None)   # angular momentum

        # If U_raw is a string, try to parse as sympy expression
        self.U_expr: Optional[sp.Expr] = None
        self.U_func: Optional[Callable[[float], float]] = None
        if self.U_raw is not None:
            try:
                if isinstance(self.U_raw, str):
                    # Assume r is variable, h and b are parameters
                    r, h, b = sp.symbols('r h b', positive=True)
                    parsed = sp.parse_expr(self.U_raw)
                    # Substitute known h,b if available
                    subs_dict = {}
                    if self.h is not None:
                        subs_dict[h] = self.h
                    if self.b is not None:
                        subs_dict[b] = self.b
                    if subs_dict:
                        parsed = parsed.subs(subs_dict)
                    self.U_expr = parsed
                    # Create numerical function for r only
                    self.U_func = sp.lambdify(r, parsed, 'numpy')
                elif isinstance(self.U_raw, sp.Expr):
                    self.U_expr = self.U_raw
                    # extract free symbols; substitute known parameters
                    free = self.U_expr.free_symbols
                    r_var = sp.Symbol('r', positive=True)
                    subs = {}
                    if self.h is not None and sp.Symbol('h') in free:
                        subs[sp.Symbol('h')] = self.h
                    if self.b is not None and sp.Symbol('b') in free:
                        subs[sp.Symbol('b')] = self.b
                    if subs:
                        self.U_expr = self.U_expr.subs(subs)
                    self.U_func = sp.lambdify(r_var, self.U_expr, 'numpy')
                else:
                    # assume it's a numeric constant? unlikely
                    self.U_func = lambda r_val: float(self.U_raw)
            except Exception:
                self.U_func = None

        # Define symbol for r for symbolic manipulations
        self.r = sp.Symbol('r', positive=True)

    # ---------- Interface Tests ----------
    def test_interface_U_present(self) -> Tuple[bool, str]:
        if self.U_raw is None:
            return (False, "Missing key 'U' in results")
        return (True, f"U present as {type(self.U_raw).__name__}")

    def test_interface_U_parsable(self) -> Tuple[bool, str]:
        if self.U_func is None:
            return (False, "U expression could not be parsed or evaluated")
        return (True, "U expression parsed successfully, func created")

    # ---------- Invariant Tests ----------
    def test_invariant_r2_times_U_constant(self) -> Tuple[bool, str]:
        """U(r) * r^2 should be independent of r (constant w.r.t. r)."""
        if self.U_expr is None:
            return (False, "SKIP: U expression not available")
        try:
            expr = self.U_expr * self.r**2
            deriv = sp.diff(expr, self.r)
            simplified = sp.simplify(deriv)
            if simplified == 0:
                return (True, "U(r)*r^2 is independent of r (derivative = 0)")
            else:
                if self.U_func is None:
                    return (False, "Derivative non-zero; need numerical fallback, but func missing")
                r1, r2 = 1.0, 2.0
                try:
                    v1 = self.U_func(r1) * r1**2
                    v2 = self.U_func(r2) * r2**2
                    if math.isfinite(v1) and math.isfinite(v2) and math.isclose(v1, v2, rel_tol=1e-2, abs_tol=1e-2):
                        return (True, f"U(r)*r^2 numerically constant: {v1:.4e} ~ {v2:.4e}")
                    else:
                        return (False, f"U(r)*r^2 not constant: {v1:.4e} vs {v2:.4e}")
                except Exception as e:
                    return (False, f"Numerical check failed: {e}")
        except Exception as e:
            return (False, f"Error checking constancy: {e}")

    def test_invariant_sign_negative(self) -> Tuple[bool, str]:
        """U(r) should be negative for positive r, h, b."""
        if self.U_func is None:
            return (False, "SKIP: U func not available")
        try:
            r_test = 1.0
            val = self.U_func(r_test)
            if val is None or not math.isfinite(val):
                return (False, f"U(r) at r=1 is invalid: {val}")
            if val < 0:
                return (True, f"U(1.0) = {val:.4f} < 0 (attractive potential)")
            else:
                return (False, f"U(1.0) = {val:.4f} >= 0, expected negative for attractive central force")
        except Exception as e:
            return (False, f"Error evaluating sign: {e}")

    def test_invariant_b_dependence_sign(self) -> Tuple[bool, str]:
        """For fixed r, increasing b should make U more negative (decrease)."""
        if self.U_expr is None:
            return (False, "SKIP: no U expression")
        b_sym = sp.Symbol('b', positive=True)
        try:
            free = self.U_expr.free_symbols
            if b_sym not in free:
                return (True, "SKIP: U does not depend on b")
            r_val = 1.0
            b1, b2 = 0.5, 1.0
            try:
                U1 = self.U_expr.subs({b_sym: b1, self.r: r_val}).evalf()
                U2 = self.U_expr.subs({b_sym: b2, self.r: r_val}).evalf()
                if U1 is None or U2 is None:
                    return (False, "Cannot evaluate U for different b")
                if U2 < U1:
                    return (True, f"Increasing b makes U more negative: U(b=0.5)={U1:.4e}, U(b=1.0)={U2:.4e}")
                else:
                    return (False, f"U does not become more negative with b: {U1:.4e} vs {U2:.4e}")
            except Exception as e:
                return (False, f"Numerical b-dependence check failed: {e}")
        except Exception as e:
            return (False, f"Error in b-dependence test: {e}")

    # ---------- Modeling / Trace Tests ----------
    def test_modeling_Binet_form(self) -> Tuple[bool, str]:
        """Numerically verify that -dU/dr * r^3 is constant ( = h^2 (b^2+1) )."""
        if self.U_func is None or self.U_expr is None:
            return (False, "SKIP: U not available for Binet test")
        try:
            dU_dr = sp.diff(self.U_expr, self.r)
            simplified = sp.simplify(dU_dr)
            r1, r2 = 1.0, 2.0
            b_val = self.b if self.b is not None else 1.0
            dU_func = sp.lambdify(self.r, -simplified * self.r**3, 'numpy')
            try:
                f1 = dU_func(r1)
                f2 = dU_func(r2)
                if not (math.isfinite(f1) and math.isfinite(f2)):
                    return (False, f"Force-derived quantity not finite: {f1}, {f2}")
                if math.isclose(f1, f2, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, f"-dU/dr * r^3 constant: {f1:.4e} ~ {f2:.4e}")
                else:
                    return (False, f"-dU/dr * r^3 not constant: {f1:.4e} vs {f2:.4e}")
            except Exception as e:
                return (False, f"Numerical Binet test error: {e}")
        except Exception as e:
            return (False, f"Error in Binet symbolic: {e}")

    def test_modeling_U_independent_of_a(self) -> Tuple[bool, str]:
        """Check that U expression does not depend on the parameter a (orbit amplitude)."""
        if self.U_expr is None:
            return (False, "SKIP: no U expression")
        a_sym = sp.Symbol('a', positive=True)
        if a_sym in self.U_expr.free_symbols:
            return (False, "U depends on parameter 'a'; should be independent of orbit amplitude")
        return (True, "U expression does not contain parameter 'a'")

    def test_modeling_U_no_theta_dependence(self) -> Tuple[bool, str]:
        """Check that U depends only on r (and possibly constants) and not on θ."""
        if self.U_expr is None:
            return (False, "SKIP: no U expression")
        theta_sym = sp.Symbol('theta', real=True)
        theta_greek = sp.Symbol('θ', real=True)
        free = self.U_expr.free_symbols
        if theta_sym in free or theta_greek in free:
            return (False, "U depends on angular variable; should be central potential, function of r only")
        return (True, "U expression contains no angular dependence (central potential)")

    # ---------- Monotonicity Tests ----------
    def test_monotonicity_U_increases_with_r(self) -> Tuple[bool, str]:
        """Perturb r by +10% and check that U increases (becomes less negative)."""
        if self.U_func is None:
            return (False, "SKIP: U func not available")
        try:
            r_baseline = 1.0
            perturbation = 0.1  # +10%
            r_perturbed = r_baseline * (1 + perturbation)
            u_baseline = self.U_func(r_baseline)
            u_perturbed = self.U_func(r_perturbed)
            if u_baseline is None or u_perturbed is None:
                return (False, "Cannot evaluate U at baseline or perturbed r")
            if not math.isfinite(u_baseline) or not math.isfinite(u_perturbed):
                return (False, f"U values not finite: baseline={u_baseline}, perturbed={u_perturbed}")
            # Expect U to increase (less negative) for larger r
            if u_perturbed > u_baseline:
                return (True, f"U increases by {((u_perturbed - u_baseline)/abs(u_baseline))*100:.1f}% when r increases by 10%")
            else:
                return (False, f"U decreased: baseline={u_baseline:.4e}, perturbed={u_perturbed:.4e} (should be >)")
        except Exception as e:
            return (False, f"Monotonicity error: {e}")

    def test_monotonicity_U_decreases_with_b(self) -> Tuple[bool, str]:
        """Perturb b by +10% and check that U becomes more negative (decreases)."""
        if self.U_expr is None:
            return (False, "SKIP: no U expression for b perturbation")
        b_sym = sp.Symbol('b', positive=True)
        free = self.U_expr.free_symbols
        if b_sym not in free:
            return (True, "SKIP: U does not depend on b")
        try:
            r_val = 1.0
            b_baseline = self.b if self.b is not None else 1.0
            perturbation = 0.1  # +10%
            b_perturbed = b_baseline * (1 + perturbation)
            # Evaluate U at baseline b and perturbed b using substitution
            U_baseline = float(self.U_expr.subs({b_sym: b_baseline, self.r: r_val}).evalf())
            U_perturbed = float(self.U_expr.subs({b_sym: b_perturbed, self.r: r_val}).evalf())
            if not math.isfinite(U_baseline) or not math.isfinite(U_perturbed):
                return (False, f"U values not finite: baseline={U_baseline}, perturbed={U_perturbed}")
            # U should become more negative (decrease) when b increases
            if U_perturbed < U_baseline:
                return (True, f"U decreases by {((U_perturbed - U_baseline)/abs(U_baseline))*100:.1f}% when b increases by 10%")
            else:
                return (False, f"U increased: baseline={U_baseline:.4e}, perturbed={U_perturbed:.4e} (should be <)")
        except Exception as e:
            return (False, f"b monotonicity error: {e}")

    # ---------- Weak Answer Sanity ----------
    def test_answer_sanity_form(self) -> Tuple[bool, str]:
        """Weak check: U should tend to 0 as r → ∞ (since constant at infinity set to 0)."""
        if self.U_func is None:
            return (False, "SKIP: no U function")
        try:
            large_r = 1e6
            val_large = self.U_func(large_r)
            if math.isfinite(val_large) and abs(val_large) < 1e-3:
                return (True, f"U tends to 0 as r→∞: U(1e6)={val_large:.4e}")
            else:
                return (False, f"U does not vanish at infinity: U(1e6)={val_large:.4e}")
        except Exception as e:
            return (False, f"Answer sanity error: {e}")

    # ---------- Test Runner ----------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_ methods. Return (all_pass, logs)."""
        logs: List[str] = []
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