import sympy as sp
import math
from typing import Any, Dict, Tuple, List

class PhysicsChecker:
    """
    Audits a solution instance for the Van der Waals equation problem.
    Tests focus on:
      - interface completeness
      - physical invariants (positivity, critical relations)
      - modeling consistency (derivation of a and b from critical conditions)
      - monotonicity (how a and b change with critical variables)
      - weak answer sanity
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract inputs and results safely with None defaults.
        Assumes solution_instance has .results dict and possibly .givens dict.
        """
        # Results
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        self.vdw_eq: str | None = self.results.get('vdw_eq', None)
        self.a: float | None = self.results.get('a', None)
        self.b: float | None = self.results.get('b', None)

        # Givens (may be missing)
        self.Tc: float | None = None
        self.Vc: float | None = None
        self.pc: float | None = None
        self.R: float | None = None
        self.n: float | None = None

        # Try to extract from solution_instance properties
        for attr in ['Tc', 'Vc', 'pc', 'R', 'n']:
            raw = getattr(solution_instance, attr, None)
            if raw is not None:
                setattr(self, attr, raw)

        # Also try .givens dictionary if present
        givens = getattr(solution_instance, 'givens', {})
        if isinstance(givens, dict):
            for key in ['Tc', 'Vc', 'pc', 'R', 'n']:
                if key in givens and getattr(self, key) is None:
                    setattr(self, key, givens[key])

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_vdw_eq_present(self) -> Tuple[bool, str]:
        """Check that the Van der Waals equation string is provided."""
        if self.vdw_eq is None:
            return (False, "Missing 'vdw_eq' in results")
        return (True, "vdw_eq present")

    def test_interface_vdw_eq_type(self) -> Tuple[bool, str]:
        """Check that vdw_eq is a non-empty string."""
        if not isinstance(self.vdw_eq, str) or not self.vdw_eq.strip():
            return (False, "vdw_eq must be a non-empty string")
        return (True, "vdw_eq is a string")

    def test_interface_a_present(self) -> Tuple[bool, str]:
        """Check that constant 'a' is provided."""
        if self.a is None:
            return (False, "Missing 'a' in results")
        return (True, "a present")

    def test_interface_b_present(self) -> Tuple[bool, str]:
        """Check that constant 'b' is provided."""
        if self.b is None:
            return (False, "Missing 'b' in results")
        return (True, "b present")

    def test_interface_a_numeric(self) -> Tuple[bool, str]:
        """Check that a is numeric (float or int)."""
        if not isinstance(self.a, (int, float)):
            return (False, f"a must be numeric, got {type(self.a).__name__}")
        return (True, "a is numeric")

    def test_interface_b_numeric(self) -> Tuple[bool, str]:
        """Check that b is numeric."""
        if not isinstance(self.b, (int, float)):
            return (False, f"b must be numeric, got {type(self.b).__name__}")
        return (True, "b is numeric")

    # ------------------------------------------------------------------
    # Physical invariant tests
    # ------------------------------------------------------------------
    def test_invariant_a_positive(self) -> Tuple[bool, str]:
        """a must be positive (intermolecular attraction constant)."""
        if self.a is None:
            return (False, "SKIP: a not available")
        if self.a <= 0:
            return (False, f"a must be positive, got {self.a}")
        return (True, f"a = {self.a} > 0")

    def test_invariant_b_positive(self) -> Tuple[bool, str]:
        """b must be positive (excluded volume)."""
        if self.b is None:
            return (False, "SKIP: b not available")
        if self.b <= 0:
            return (False, f"b must be positive, got {self.b}")
        return (True, f"b = {self.b} > 0")

    def test_invariant_b_less_than_molar_volume(self) -> Tuple[bool, str]:
        """If Vc is given, check that b = Vc/3 from critical relations."""
        if self.b is None or self.Vc is None:
            return (False, "SKIP: need both b and Vc")
        expected_b = self.Vc / 3.0
        if not math.isclose(self.b, expected_b, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"b={self.b} does not match Vc/3={expected_b}")
        return (True, f"b = Vc/3 = {expected_b}")

    # ------------------------------------------------------------------
    # Modeling (derivation) tests
    # ------------------------------------------------------------------
    def test_modeling_a_expression_from_critical(self) -> Tuple[bool, str]:
        """
        Derive a from critical conditions using sympy and compare with given a.
        Uses critical data if available; otherwise skip.
        """
        if self.a is None or self.Vc is None or self.pc is None:
            return (False, "SKIP: need a, Vc, and pc")
        # Expected: a = 3 * pc * Vc^2
        expected_a = 3.0 * self.pc * self.Vc**2
        if not math.isclose(self.a, expected_a, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"a = {self.a} does not match 3*pc*Vc^2 = {expected_a}")
        return (True, f"a = {expected_a} from 3*pc*Vc^2")

    def test_modeling_b_expression_from_critical(self) -> Tuple[bool, str]:
        """Derive b from critical conditions: b = Vc/3."""
        if self.b is None or self.Vc is None:
            return (False, "SKIP: need b and Vc")
        expected_b = self.Vc / 3.0
        if not math.isclose(self.b, expected_b, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"b = {self.b} does not match Vc/3 = {expected_b}")
        return (True, f"b = Vc/3 = {expected_b}")

    def test_modeling_critical_conditions_symbolic(self) -> Tuple[bool, str]:
        """
        Use symbolic algebra to verify that a,b satisfy the critical point equations
        (∂p/∂V=0, ∂²p/∂V²=0) and the pc definition. If critical data are missing, skip.
        """
        # Need numeric values for comparison
        if self.a is None or self.b is None or self.Vc is None or self.pc is None or self.R is None:
            return (False, "SKIP: need a, b, Vc, pc, R for full symbolic check")

        # Symbolic variables
        R, Tc, Vc, pc, a, b, V = sp.symbols('R Tc Vc pc a b V', real=True, positive=True)
        # Van der Waals pressure for 1 mole
        p = R*Tc/(V - b) - a/V**2

        # First derivative
        dp_dV = sp.diff(p, V)
        # Second derivative
        d2p_dV2 = sp.diff(dp_dV, V)

        # Substitute critical point: V = Vc
        dp_dV_at_crit = dp_dV.subs({V: Vc, Tc: self.Tc if self.Tc else sp.Symbol('Tc')})
        d2p_dV2_at_crit = d2p_dV2.subs({V: Vc, Tc: self.Tc if self.Tc else sp.Symbol('Tc')})
        p_at_crit = p.subs({V: Vc, Tc: self.Tc if self.Tc else sp.Symbol('Tc')})

        # Set equations = 0 and pc = given expression
        eq1 = sp.Eq(dp_dV_at_crit, 0)
        eq2 = sp.Eq(d2p_dV2_at_crit, 0)
        eq3 = sp.Eq(p_at_crit, pc)

        # Solve for a, b, and R (or treat R as known)
        sol = sp.solve([eq1, eq2, eq3], [a, b, R], dict=True)
        if not sol:
            return (False, "Symbolic solve returned no solution")

        # Expect single solution
        sol = sol[0]
        a_expr = sol[a]
        b_expr = sol[b]

        # Evaluate numerically using given data
        # Need numeric values: Vc, pc, Tc
        if self.Tc is None:
            return (False, "SKIP: Tc not provided for numeric evaluation")
        # Evaluate expressions as floats
        try:
            a_numeric = float(a_expr.subs({Vc: self.Vc, pc: self.pc, Tc: self.Tc}))
            b_numeric = float(b_expr.subs({Vc: self.Vc}))
        except Exception:
            return (False, "Numerical evaluation of symbolic expressions failed")

        # Compare with given a,b
        if not (math.isclose(self.a, a_numeric, rel_tol=1e-2, abs_tol=1e-2) and
                math.isclose(self.b, b_numeric, rel_tol=1e-2, abs_tol=1e-2)):
            return (False, f"Derived a={a_numeric}, b={b_numeric} do not match given a={self.a}, b={self.b}")
        return (True, f"Symbolic derivation confirms a={a_numeric}, b={b_numeric}")

    # ------------------------------------------------------------------
    # Monotonicity tests
    # ------------------------------------------------------------------
    def test_monotonicity_a_increases_with_Vc(self) -> Tuple[bool, str]:
        """
        Perturb Vc upward by 10% and check that a (computed from formula) increases.
        This tests the functional relationship a = 3 * pc * Vc^2.
        """
        if self.a is None or self.Vc is None or self.pc is None:
            return (False, "SKIP: need a, Vc, and pc")
        # Baseline formula: a = 3 * pc * Vc^2
        base_a = 3.0 * self.pc * self.Vc**2
        if not math.isclose(self.a, base_a, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Given a={self.a} does not match formula a={base_a}")

        # Perturb Vc by +10%
        Vc_new = self.Vc * 1.1
        a_new = 3.0 * self.pc * Vc_new**2
        # monotonicity: a_new should be greater than base_a
        if a_new > base_a:
            return (True, f"a increases with Vc: base={base_a}, perturbed={a_new}")
        else:
            return (False, f"a did not increase: base={base_a}, perturbed={a_new}")

    def test_monotonicity_b_increases_with_Vc(self) -> Tuple[bool, str]:
        """
        Perturb Vc upward by 10% and check that b (computed from formula) increases.
        This tests the functional relationship b = Vc / 3.
        """
        if self.b is None or self.Vc is None:
            return (False, "SKIP: need b and Vc")
        base_b = self.Vc / 3.0
        if not math.isclose(self.b, base_b, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Given b={self.b} does not match formula b={base_b}")

        # Perturb Vc by +10%
        Vc_new = self.Vc * 1.1
        b_new = Vc_new / 3.0
        if b_new > base_b:
            return (True, f"b increases with Vc: base={base_b}, perturbed={b_new}")
        else:
            return (False, f"b did not increase: base={base_b}, perturbed={b_new}")

    def test_monotonicity_a_increases_with_pc(self) -> Tuple[bool, str]:
        """
        Perturb pc upward by 10% and check that a (computed from formula) increases.
        This tests the functional relationship a = 3 * pc * Vc^2.
        """
        if self.a is None or self.Vc is None or self.pc is None:
            return (False, "SKIP: need a, Vc, and pc")
        base_a = 3.0 * self.pc * self.Vc**2
        if not math.isclose(self.a, base_a, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Given a={self.a} does not match formula a={base_a}")

        # Perturb pc by +10%
        pc_new = self.pc * 1.1
        a_new = 3.0 * pc_new * self.Vc**2
        if a_new > base_a:
            return (True, f"a increases with pc: base={base_a}, perturbed={a_new}")
        else:
            return (False, f"a did not increase: base={base_a}, perturbed={a_new}")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self) -> Tuple[bool, str]:
        """
        Weak check: a and b should be finite, positive, and of reasonable magnitude
        (not astronomically large or negative). This is not a precision test.
        """
        if self.a is None or self.b is None:
            return (False, "SKIP: a or b missing")
        # Finite check
        if not (math.isfinite(self.a) and math.isfinite(self.b)):
            return (False, "a or b is not finite")
        # Positive
        if self.a <= 0 or self.b <= 0:
            return (False, f"a={self.a} or b={self.b} not positive")
        # Order-of-magnitude sanity: typical a ~ 0.1-10, b ~ 10-5 to 10-3
        # But this is problem-dependent; just ensure no extreme exponent
        if abs(self.a) > 1e10 or abs(self.b) > 1e3:
            return (False, f"Unrealistic magnitude: a={self.a}, b={self.b}")
        return (True, "a and b are finite, positive, and plausible")

    # ------------------------------------------------------------------
    # Main test runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_ methods, return overall pass/fail and logs."""
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