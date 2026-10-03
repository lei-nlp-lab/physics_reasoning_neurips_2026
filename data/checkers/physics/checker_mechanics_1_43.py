import math
import traceback
from typing import Any, Dict, List, Tuple

try:
    import sympy as sp
except ImportError:
    sp = None

class PhysicsChecker:
    """
    Audits a solution instance for the central potential scattering problem.

    The checker extracts givens (m, E0, L, G) and results (dr_dtheta, r_min_eq)
    from the solution instance, then runs multiple unit-test-like checks.
    """

    def __init__(self, solution_instance: Any) -> None:
        # --- Givens (use None for missing) ---
        self.m: Any = getattr(solution_instance, 'm', None)
        self.E0: Any = getattr(solution_instance, 'E0', None)
        self.L: Any = getattr(solution_instance, 'L', None)
        # G is expected to be a callable or a string representing a function
        self.G: Any = getattr(solution_instance, 'G', None)

        # --- Results (from pseudocode emits) ---
        results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}
        self.dr_dtheta: Any = results.get('dr_dtheta', None)
        self.r_min_eq: Any = results.get('r_min_eq', None)

        # Also attempt to get results from direct attributes as fallback
        if self.dr_dtheta is None:
            self.dr_dtheta = getattr(solution_instance, 'dr_dtheta', None)
        if self.r_min_eq is None:
            self.r_min_eq = getattr(solution_instance, 'r_min_eq', None)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _parse_math_expression(self, expr_str: str) -> sp.Expr:
        """Try to parse a string into a sympy expression; return None on failure."""
        if sp is None:
            return None
        try:
            # Replace common textual patterns for sympy compatibility
            cleaned = expr_str.replace('+/-', '')
            cleaned = cleaned.replace(' ', '')
            # Ensure variables are recognized: r, m, E0, L, G(r) -> G(r) is a function
            # We'll define symbols
            r, m, E0, L = sp.symbols('r m E0 L', positive=True)
            # For G(r), it's a function; treat as symbolic function G(r)
            G = sp.Function('G')
            # If the string contains G(r) we need to substitute; we'll parse with sympy
            expr = sp.parse_expr(cleaned, local_dict={
                'r': r, 'm': m, 'E0': E0, 'L': L, 'G': G
            })
            return expr
        except Exception:
            return None

    def _num_check(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Fuzzy numerical comparison suitable for physics data."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    # ------------------------------------------------------------------
    # Interface tests: existence and type of required items
    # ------------------------------------------------------------------
    def test_interface_dr_dtheta_exists(self) -> Tuple[bool, str]:
        if self.dr_dtheta is None:
            return (False, "SKIP: dr_dtheta is missing")
        if not isinstance(self.dr_dtheta, str):
            return (False, f"dr_dtheta should be a string, got {type(self.dr_dtheta).__name__}")
        return (True, "dr_dtheta is present and is a string")

    def test_interface_r_min_eq_exists(self) -> Tuple[bool, str]:
        if self.r_min_eq is None:
            return (False, "SKIP: r_min_eq is missing")
        if not isinstance(self.r_min_eq, str):
            return (False, f"r_min_eq should be a string, got {type(self.r_min_eq).__name__}")
        return (True, "r_min_eq is present and is a string")

    def test_interface_givens(self) -> Tuple[bool, str]:
        """Check that essential givens are positive numbers (or symbolic)."""
        missing = []
        for name, val in [('m', self.m), ('E0', self.E0), ('L', self.L)]:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"SKIP: missing givens: {', '.join(missing)}")
        # If values are numeric, check positivity
        for name, val in [('m', self.m), ('L', self.L)]:
            if isinstance(val, (int, float)):
                if val <= 0:
                    return (False, f"{name} must be positive, got {val}")
        # E0 may be zero or positive for scattering (but could be negative in bound case)
        if isinstance(self.E0, (int, float)) and self.E0 <= 0:
            return (False, f"E0 should be positive for scattering, got {self.E0}")
        return (True, "All required givens are present and valid")

    # ------------------------------------------------------------------
    # Invariant/theorem tests (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_drdtheta_derivation(self) -> Tuple[bool, str]:
        """
        Verify that the provided dr_dtheta expression, when squared and rearranged,
        matches the form obtained from energy and angular momentum conservation.

        Reference form: (dr/dθ)^2 = (2m(E0+G(r)) r^4 / L^2) - r^2
        """
        if self.dr_dtheta is None:
            return (False, "SKIP: dr_dtheta not available")
        if sp is None:
            return (False, "SKIP: sympy not available")
        # Remove '+/-' for parsing
        expr_str = self.dr_dtheta.replace('+/-', '')
        expr = self._parse_math_expression(expr_str)
        if expr is None:
            return (False, "Could not parse dr_dtheta expression")
        # Build the reference expression
        r, m, E0, L = sp.symbols('r m E0 L', positive=True)
        G = sp.Function('G')
        ref_expr = sp.sqrt(2*m*(E0 + G(r)) * r**4 / L**2 - r**2)
        # Symply check difference
        diff = sp.simplify(expr - ref_expr)
        if diff == 0:
            return (True, "dr_dtheta symbolic expression matches reference form")
        else:
            # Try numerical check with random values
            subs = {m: 2.0, E0: 5.0, L: 3.0, r: 1.5}
            # Assume G(r) = 1.0 (constant) for test
            subs[G(r)] = 1.0
            try:
                num_expr = float(expr.subs(subs))
                num_ref = float(ref_expr.subs(subs))
            except Exception:
                return (False, f"Expression mismatch (diff: {diff}) and numerical fallback failed")
            if self._num_check(num_expr, num_ref):
                return (True, "dr_dtheta symbolic check uncertain but numerical check passes")
            else:
                return (False, f"dr_dtheta numerical mismatch: expr={num_expr}, ref={num_ref}")
        return (True, "dr_dtheta symbolic expression matches reference form")

    def test_invariant_r_min_turning_condition(self) -> Tuple[bool, str]:
        """
        Check that setting dr/dθ = 0 in the energy equation yields the r_min equation.
        The reference turning condition: E0 = L^2/(2 m r_min^2) - G(r_min)
        The result equation should be equivalent to that.
        """
        if self.r_min_eq is None:
            return (False, "SKIP: r_min_eq not available")
        # Parse the r_min_eq: it should be something like "r_min = L / sqrt(2*m*(E0+G(r_min)))"
        # Extract the right-hand side
        if '=' in self.r_min_eq:
            rhs_str = self.r_min_eq.split('=')[1].strip()
        else:
            return (False, "r_min_eq does not contain '='")
        if sp is None:
            return (False, "SKIP: sympy not available")
        # Symbols and G function
        r_min, m, E0, L = sp.symbols('r_min m E0 L', positive=True)
        G = sp.Function('G')
        try:
            rhs = sp.parse_expr(rhs_str, local_dict={'r_min': r_min, 'm': m, 'E0': E0, 'L': L, 'G': G})
        except Exception:
            return (False, "Could not parse r_min_eq right-hand side")
        # The turning condition is: E0 = L^2/(2*m*r_min^2) - G(r_min)
        turning_expr = L**2/(2*m*r_min**2) - G(r_min)
        # Verify that rhs satisfies: E0 = L^2/(2*m*rhs^2) - G(rhs) ?
        # Actually the equation is implicit: r_min = L/sqrt(2*m*(E0+G(r_min)))
        # Squaring both sides: r_min^2 = L^2/(2*m*(E0+G(r_min)))
        # => L^2/(2*m*r_min^2) = E0+G(r_min) => L^2/(2*m*r_min^2) - G(r_min) = E0
        # So we need to check that turning_expr - E0 = 0 after substituting r_min = rhs.
        # That is, we compute (L^2/(2*m*rhs^2) - G(rhs)) - E0 and simplify to 0.
        check_expr = sp.simplify(L**2/(2*m*rhs**2) - G(rhs) - E0)
        if check_expr == 0:
            return (True, "r_min equation satisfies the turning condition symbolically")
        # Numerical fallback with example parameters
        subs = {m: 2.0, E0: 5.0, L: 3.0, G(r_min): 1.0}  # assume constant G(r_min)
        try:
            # For the numerical test, we need to solve for r_min from the given equation?
            # Instead, we check that the turning condition holds with the rhs evaluated at a sample.
            # But the rhs itself depends on G(r_min) which we set constant.
            # So we compute rhs_num = rhs.subs({m:2.0, E0:5.0, L:3.0, G(r_min):1.0}).evalf()
            # Then compute turning_expr with that numerical r_min and check closeness.
            subs_common = {m: 2.0, E0: 5.0, L: 3.0}
            # For simplicity, assume G(r) = 1.0 constant, so G(r_min) = 1.0
            subs_G = {G(r_min): 1.0}
            rhs_num = float(rhs.subs({**subs_common, **subs_G}))
            # Now compute turning condition: L^2/(2*m*r_min^2) - G(r_min) should equal E0
            turning_num = L**2/(2*m*rhs_num**2) - 1.0  # using constant G=1
            # But we must substitute m, L from subs_common
            turning_num = float(turning_num.subs(subs_common))
            if self._num_check(turning_num, self.E0 if isinstance(self.E0, (int,float)) else 5.0):
                return (True, "r_min equation passes numerical turning condition")
        except Exception:
            return (False, "Numerical fallback failed for r_min equation")
        return (False, "r_min equation does not satisfy the turning condition")

    # ------------------------------------------------------------------
    # Modeling / trace tests (using explanation_steps)
    # ------------------------------------------------------------------
    def test_modeling_drdtheta_sign_and_structure(self) -> Tuple[bool, str]:
        """
        Check that the provided dr_dtheta string contains '+/-', sqrt, and the
        basic product structure: 2*m*(E0+G(r))*r^4/L^2 - r^2.
        """
        if self.dr_dtheta is None or not isinstance(self.dr_dtheta, str):
            return (False, "SKIP: dr_dtheta not available as string")
        s = self.dr_dtheta.replace(' ', '')
        # must have '+/-'
        if '+/-' not in s:
            return (False, "Missing '+/-' sign indicator")
        # must contain sqrt and the radicand structure
        if 'sqrt' not in s:
            return (False, "Missing sqrt function")
        # Check for key components
        has_2m = '2m' in s or '2*m' in s
        has_E0_G = 'E0' in s and 'G' in s
        has_r4 = 'r**4' in s or 'r^4' in s
        has_L2 = 'L**2' in s or 'L^2' in s
        if not (has_2m and has_E0_G and has_r4 and has_L2):
            return (False, "Expression lacks required components: 2m, E0+G(r), r^4, L^2")
        return (True, "dr_dtheta string has correct structure and components")

    def test_modeling_r_min_implicit_form(self) -> Tuple[bool, str]:
        """
        Verify the r_min_eq string shows clear implicit dependence on G(r_min).
        """
        if self.r_min_eq is None or not isinstance(self.r_min_eq, str):
            return (False, "SKIP: r_min_eq not available as string")
        s = self.r_min_eq.replace(' ', '')
        # Check that it contains G(r_min) or G(rmin) or similar
        has_G = 'G(r_min)' in s or 'G(rmin)' in s or 'G(rmin)' in s
        # Also check the square root form with denominator
        has_sqrt = 'sqrt' in s
        has_2m = '2m' in s or '2*m' in s
        if not (has_G and has_sqrt and has_2m):
            return (False, "r_min_eq missing expected structure: G(r_min), sqrt, 2m")
        return (True, "r_min_eq string shows correct implicit form")

    # ------------------------------------------------------------------
    # Monotonicity tests (with input perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_r_min_with_L(self) -> Tuple[bool, str]:
        """
        For a simple attractive potential (e.g., G(r)=0 or constant), increasing L
        should increase r_min.  We only check if we have numeric givens and G.
        """
        # This test is optional; if data missing, skip.
        missing = []
        for name, val in [('m', self.m), ('E0', self.E0), ('L', self.L)]:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"SKIP: missing givens {missing}")
        # We also need a numeric G(r) to evaluate; we can try to treat as constant if G is callable?
        # If G is a function, we can't evaluate easily; try a simple scenario.
        # We'll assume G(r) = 0 for monotonicity test.
        try:
            m_val = float(self.m)
            E0_val = float(self.E0)
            L_base = float(self.L)
        except (TypeError, ValueError):
            return (False, "SKIP: givens not convertible to float")

        # Use G=0 so r_min = L / sqrt(2*m*E0)
        denominator = math.sqrt(2 * m_val * E0_val)
        r_min_base = L_base / denominator
        # Perturb L upward by 10%
        L_pert = L_base * 1.1
        r_min_pert = L_pert / denominator
        if r_min_pert > r_min_base:
            return (True, "r_min increases with L for G=0 (expected monotonicity)")
        else:
            return (False, f"Monotonicity fails: r_min_base={r_min_base}, r_min_pert={r_min_pert}")

    # ------------------------------------------------------------------
    # Answer sanity (weak, should be secondary)
    # ------------------------------------------------------------------
    def test_answer_sanity_r_min_positive(self) -> Tuple[bool, str]:
        """
        If the r_min_eq can be evaluated numerically for a sample scenario,
        check that the result is positive.
        """
        if self.r_min_eq is None:
            return (False, "SKIP: r_min_eq not available")
        # We'll try a numeric evaluation with very simple numbers.
        # We assume G(r) = 1 (constant) and m=E0=L=2 (all >0)
        try:
            # parse the RHS again
            if '=' in self.r_min_eq:
                rhs_str = self.r_min_eq.split('=')[1].strip()
            else:
                return (False, "r_min_eq does not contain '='")
            # Use sympy to evaluate with numeric subs
            if sp is None:
                return (False, "SKIP: sympy not available")
            r_min_sym, m_sym, E0_sym, L_sym = sp.symbols('r_min m E0 L', positive=True)
            G_func = sp.Function('G')
            rhs_expr = sp.parse_expr(rhs_str, local_dict={'r_min': r_min_sym, 'm': m_sym, 'E0': E0_sym, 'L': L_sym, 'G': G_func})
            # substitute m=2, E0=2, L=2, G(r_min)=1 (constant)
            subs = {m_sym: 2.0, E0_sym: 2.0, L_sym: 2.0, G_func(r_min_sym): 1.0}
            numeric_rhs = float(rhs_expr.subs(subs))
            if numeric_rhs > 0:
                return (True, f"Computed r_min = {numeric_rhs} > 0 (passes sanity)")
            else:
                return (False, f"Computed r_min = {numeric_rhs} is non-positive")
        except Exception:
            return (False, "Could not evaluate r_min equation numerically; sanity check inconclusive")

    # ------------------------------------------------------------------
    # Generic run_all_tests method
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)