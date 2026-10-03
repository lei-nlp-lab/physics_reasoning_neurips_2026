import math
import sympy as sp
from typing import Tuple, List, Dict, Any, Optional

class PhysicsChecker:
    """
    PhysicsChecker for the Gibbs free energy problem: Cp from G(p,T).
    Audits interface, thermodynamic invariants, modeling quality, and answer sanity.
    """

    def __init__(self, solution_instance: Any):
        """
        Extract all required data safely from the solution instance.
        Use None defaults for missing attributes; never use numeric defaults.
        """
        # Results dict holding emitted values
        results = getattr(solution_instance, 'results', None)
        if results is None:
            results = {}
        self.results = results
        self.Cp = results.get('Cp', None)

        # Givens and domain parameters (may be absent)
        self.p = getattr(solution_instance, 'p', None)
        self.T = getattr(solution_instance, 'T', None)
        self.R = getattr(solution_instance, 'R', None)
        self.a = getattr(solution_instance, 'a', None)

        # Try to get G expression if provided (string or sympy)
        raw_G = getattr(solution_instance, 'G', None)
        if raw_G is not None:
            if isinstance(raw_G, str):
                try:
                    self.G = sp.sympify(raw_G)
                except Exception:
                    self.G = None
            else:
                self.G = raw_G
        else:
            self.G = None

        # Explanation steps (for reference, not used heavily)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Build default symbolic quantities if givens are missing
        # These are used only when we need to re-derive; they are NOT assigned to self
        self._p_sym, self._T_sym, self._R_sym, self._a_sym = sp.symbols('p T R a', positive=True)

    # ------------------------------------------------------------------
    # Helper: Build G expression or derive it from scratch
    # ------------------------------------------------------------------
    def _get_G_expr(self) -> sp.Expr:
        """
        Return a sympy expression for G(p,T). Use self.G if available,
        otherwise construct from domain symbols.
        """
        if self.G is not None:
            return self.G
        # Fallback: construct from expression
        return self._R_sym * self._T_sym * sp.log(self._a_sym * self._p_sym / (self._R_sym * self._T_sym)**(sp.Rational(5,2)))

    def _get_numeric_values(self, p: Optional[float] = None,
                                 T: Optional[float] = None,
                                 R: Optional[float] = None,
                                 a: Optional[float] = None) -> Dict[sp.Symbol, float]:
        """
        Provide a numeric substitution dict for symbolic expressions.
        Uses given values if available, else sensible defaults (1 for constants, 300 for T, 1e5 for p).
        """
        p_val = p if p is not None else (self.p if self.p is not None else 1e5)
        T_val = T if T is not None else (self.T if self.T is not None else 300.0)
        R_val = R if R is not None else (self.R if self.R is not None else 8.314)
        a_val = a if a is not None else (self.a if self.a is not None else 1.0)
        return {self._p_sym: p_val, self._T_sym: T_val,
                self._R_sym: R_val, self._a_sym: a_val}

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_results_Cp(self) -> Tuple[bool, str]:
        """Check that the results dict contains 'Cp'."""
        if 'Cp' not in self.results:
            return (False, "Missing 'Cp' key in results")
        return (True, "'Cp' key present")

    def test_interface_Cp_non_none(self) -> Tuple[bool, str]:
        """Check that Cp value is not None."""
        if self.Cp is None:
            return (False, "Cp is None")
        return (True, "Cp is not None")

    # ------------------------------------------------------------------
    # Invariant checks (thermodynamic definitions, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_entropy_definition(self) -> Tuple[bool, str]:
        """
        Verify: S = - (∂G/∂T)_p  (not directly tested, but used as intermediate)
        Actually we test that the derived Cp via the entropy route matches directly.
        """
        # If Cp is missing, skip
        if self.Cp is None:
            return (False, "SKIP: Cp not available")
        # Build G expression
        try:
            G_expr = self._get_G_expr()
            # Compute S = -dG/dT
            S_expr = -sp.diff(G_expr, self._T_sym)
            # Compute Cp = T * dS/dT
            Cp_expr = sp.simplify(self._T_sym * sp.diff(S_expr, self._T_sym))
            # Evaluate numerically
            subs = self._get_numeric_values()
            Cp_num = float(Cp_expr.subs(subs).evalf())
            Cp_given = float(self.Cp) if not isinstance(self.Cp, sp.Basic) else float(self.Cp.evalf())
            if math.isclose(Cp_num, Cp_given, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Calculated Cp ({Cp_num}) matches provided Cp ({Cp_given})")
            else:
                return (False, f"Calculated Cp ({Cp_num}) deviates from provided Cp ({Cp_given})")
        except Exception as e:
            return (False, f"Error in symbolic derivation: {e}")

    def test_invariant_enthalpy_relation(self) -> Tuple[bool, str]:
        """
        Optional test: H = G + T S, then Cp = dH/dT.
        This duplicates the above, but checks intermediate consistency.
        """
        # This is a modeling-quality test, but we include it here as invariant.
        if self.Cp is None:
            return (False, "SKIP: Cp not available")
        try:
            G_expr = self._get_G_expr()
            S_expr = -sp.diff(G_expr, self._T_sym)
            H_expr = sp.simplify(G_expr + self._T_sym * S_expr)
            Cp_expr = sp.simplify(sp.diff(H_expr, self._T_sym))
            subs = self._get_numeric_values()
            Cp_num = float(Cp_expr.subs(subs).evalf())
            Cp_given = float(self.Cp) if not isinstance(self.Cp, sp.Basic) else float(self.Cp.evalf())
            if math.isclose(Cp_num, Cp_given, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Derived Cp via H matches provided Cp")
            else:
                return (False, f"Derived Cp via H ({Cp_num}) != provided Cp ({Cp_given})")
        except Exception as e:
            return (False, f"Error in H derivation: {e}")

    # ------------------------------------------------------------------
    # Modeling-quality / trace consistency checks
    # ------------------------------------------------------------------
    def test_modeling_Cp_symbolic_expression(self) -> Tuple[bool, str]:
        """
        Check that the symbolic expression derived from G simplifies to (5/2)R.
        This is a pure symbolic test; does not use math.isclose.
        """
        try:
            G_expr = self._get_G_expr()
            S_expr = -sp.diff(G_expr, self._T_sym)
            Cp_sym = sp.simplify(self._T_sym * sp.diff(S_expr, self._T_sym))
            # Compare to (5/2)R symbolically
            target = sp.Rational(5,2) * self._R_sym
            diff = sp.simplify(Cp_sym - target)
            if diff == 0:
                return (True, "Symbolic Cp exactly equals (5/2)R")
            else:
                return (False, f"Symbolic Cp simplifies to {Cp_sym}, not (5/2)R")
        except Exception as e:
            return (False, f"Symbolic derivation failed: {e}")

    def test_modeling_Cp_independent_of_p(self) -> Tuple[bool, str]:
        """
        Verify that Cp does not depend on p (by perturbing p and re-evaluating symbolically).
        """
        try:
            G_expr = self._get_G_expr()
            S_expr = -sp.diff(G_expr, self._T_sym)
            Cp_expr = sp.simplify(self._T_sym * sp.diff(S_expr, self._T_sym))
            # Compute symbolic derivative w.r.t p
            dCp_dp = sp.diff(Cp_expr, self._p_sym)
            if sp.simplify(dCp_dp) == 0:
                return (True, "Cp is independent of p (symbolic derivative zero)")
            else:
                return (False, f"dCp/dp = {dCp_dp}, not zero")
        except Exception as e:
            return (False, f"Error in p-independence check: {e}")

    # ------------------------------------------------------------------
    # Monotonicity test (perturb T, see if Cp stays constant)
    # ------------------------------------------------------------------
    def test_monotonicity_Cp_constant_with_T(self) -> Tuple[bool, str]:
        """
        Cp should be independent of T. Perturb T by +10% and check that derived Cp does not change.
        """
        if self.Cp is None:
            return (False, "SKIP: Cp not available")
        try:
            G_expr = self._get_G_expr()
            S_expr = -sp.diff(G_expr, self._T_sym)
            Cp_expr = sp.simplify(self._T_sym * sp.diff(S_expr, self._T_sym))
            # Baseline numeric
            subs_base = self._get_numeric_values()
            Cp_base = float(Cp_expr.subs(subs_base).evalf())
            # Perturb T: increase by 10%
            subs_pert = self._get_numeric_values(T=subs_base[self._T_sym] * 1.1)
            Cp_pert = float(Cp_expr.subs(subs_pert).evalf())
            if math.isclose(Cp_base, Cp_pert, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Cp unchanged under +10% T perturbation (base={Cp_base}, pert={Cp_pert})")
            else:
                return (False, f"Cp changed under T perturbation: base={Cp_base}, pert={Cp_pert}")
        except Exception as e:
            return (False, f"Error in monotonicity test: {e}")

    # ------------------------------------------------------------------
    # Weak answer sanity (no ref derivation, no isclose)
    # ------------------------------------------------------------------
    def test_answer_sanity_Cp(self) -> Tuple[bool, str]:
        """
        Weak check: Cp should be positive and finite.
        """
        if self.Cp is None:
            return (False, "SKIP: Cp not available")
        try:
            Cp_val = float(self.Cp) if not isinstance(self.Cp, sp.Basic) else float(self.Cp.evalf())
        except Exception:
            return (False, "Cannot convert Cp to float")
        if Cp_val <= 0:
            return (False, f"Cp is not positive: {Cp_val}")
        if math.isinf(Cp_val) or math.isnan(Cp_val):
            return (False, f"Cp is inf or NaN: {Cp_val}")
        return (True, f"Cp is positive and finite ({Cp_val})")

    # ------------------------------------------------------------------
    # Test runner (discover all test_ methods)
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all bound test_ methods, collect logs.
        Returns (all_pass, logs_list).
        """
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
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)