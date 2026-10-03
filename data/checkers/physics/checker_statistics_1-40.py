import math
import sympy as sp
from typing import Any, Tuple, List

class PhysicsChecker:
    """
    Audits a solution for a heat pump problem with Carnot cycle and heat loss.
    """

    def __init__(self, solution_instance: Any) -> None:
        self.results = getattr(solution_instance, 'results', {})
        self.givens = getattr(solution_instance, 'givens', {})

        # Extract numeric givens (temperatures may be in Celsius or Kelvin)
        self.T1_ = self.givens.get('T1', None)
        self.T2_ = self.givens.get('T2', None)
        self.A = self.givens.get('A', None)
        self.dW_dt = self.givens.get('dW/dt', None)

        # Convert to Kelvin if given in Celsius (assume Celsius if <= 100)
        if self.T1_ is not None and self.T1_ <= 100:
            self.T1_K = self.T1_ + 273.15
        else:
            self.T1_K = self.T1_
        if self.T2_ is not None and self.T2_ <= 100:
            self.T2_K = self.T2_ + 273.15
        else:
            self.T2_K = self.T2_

        # Extract results safely
        self.dQm_dt_max = self.results.get('dQm_dt_max', None)
        self.gain = self.results.get('gain', None)
        self.gain_num = self.results.get('gain_num', None)
        self.T2_eq = self.results.get('T2_eq', None)

        # Convert string expressions to sympy for analysis
        self._convert_to_sympy()

        # Define symbols for use in tests
        self.sym_A = sp.symbols('A', positive=True)
        self.sym_dW_dt = sp.symbols('dW_dt', positive=True)
        self.sym_T1 = sp.symbols('T1', positive=True)
        self.sym_T2 = sp.symbols('T2', positive=True)

    def _convert_to_sympy(self) -> None:
        """Convert string expressions to sympy objects where possible."""
        for attr in ['dQm_dt_max', 'gain', 'T2_eq']:
            val = getattr(self, attr, None)
            if isinstance(val, str):
                try:
                    setattr(self, attr, sp.parse_expr(val))
                except Exception:
                    pass  # keep as string if parsing fails

    # ---------- Interface Tests ----------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all required result keys are present."""
        required = ['dQm_dt_max', 'gain', 'gain_num', 'T2_eq']
        missing = [k for k in required if k not in self.results]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required result keys present.")

    def test_interface_valid_types(self) -> Tuple[bool, str]:
        """Check that results have valid types (sympy expressions or numeric)."""
        issues = []
        for key, val in self.results.items():
            if val is None:
                issues.append(f"'{key}' is None")
            elif key == 'gain_num':
                if not isinstance(val, (int, float)):
                    issues.append(f"'{key}' should be numeric, got {type(val)}")
            else:
                if not isinstance(val, (sp.Expr, str)):
                    issues.append(f"'{key}' should be sympy expression or string, got {type(val)}")
        if issues:
            return (False, "; ".join(issues))
        return (True, "All result types valid.")

    # ---------- Invariant Tests (theorem-based) ----------
    def test_invariant_carnot_cop(self) -> Tuple[bool, str]:
        """Check that dQm_dt_max / dW_dt = T2/(T2-T1) using given values."""
        if self.dQm_dt_max is None or self.dW_dt is None:
            return (False, "SKIP: dQm_dt_max or dW_dt not available for numeric check.")
        if self.T1_K is None or self.T2_K is None:
            return (False, "SKIP: Temperatures not available for numeric check.")

        if isinstance(self.dQm_dt_max, (int, float)) and isinstance(self.dW_dt, (int, float)):
            ratio = self.dQm_dt_max / self.dW_dt
            expected = self.T2_K / (self.T2_K - self.T1_K)
            if math.isclose(ratio, expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"COP matches: {ratio:.4f} ≈ {expected:.4f}")
            else:
                return (False, f"COP mismatch: {ratio:.4f} vs {expected:.4f}")
        else:
            try:
                expr = sp.simplify(self.dQm_dt_max - (self.T2_K/(self.T2_K - self.T1_K))*self.dW_dt)
                if expr == 0:
                    return (True, "Symbolic COP equality holds.")
                else:
                    return (False, f"Symbolic COP equality failed: difference = {expr}")
            except Exception as e:
                return (False, f"Symbolic check error: {e}")

    def test_invariant_equilibrium(self) -> Tuple[bool, str]:
        """Check that equilibrium equation A*(T2 - T1) = (T2/(T2-T1))*dW_dt holds symbolically."""
        if self.T2_eq is None or self.A is None or self.dW_dt is None:
            return (False, "SKIP: T2_eq, A, or dW_dt not available for equilibrium check.")
        T2_sym = self.T2_eq
        if not isinstance(T2_sym, sp.Expr):
            if isinstance(T2_sym, str):
                try:
                    T2_sym = sp.parse_expr(T2_sym)
                except Exception:
                    return (False, "Cannot parse T2_eq as sympy expression.")
            else:
                return (False, "T2_eq is not a sympy expression.")

        T1_val = self.T1_K if self.T1_K is not None else sp.symbols('T1')
        try:
            equilibrium = self.A * (T2_sym - T1_val) - (T2_sym / (T2_sym - T1_val)) * self.dW_dt
            simplified = sp.simplify(equilibrium)
            if simplified == 0:
                return (True, "Equilibrium equation holds symbolically.")
            else:
                return (False, f"Equilibrium equation not zero: {simplified}")
        except Exception as e:
            return (False, f"Symbolic equilibrium check failed: {e}")

    # ---------- Modeling / Trace Tests ----------
    def test_modeling_gain_expression(self) -> Tuple[bool, str]:
        """Check that gain equals T2/(T2-T1) symbolically or numerically."""
        if self.gain is None:
            return (False, "SKIP: gain expression not available.")
        if self.T1_K is None or self.T2_K is None:
            return (False, "SKIP: Temperatures not available for numeric check.")
        if isinstance(self.gain, (int, float)):
            expected = self.T2_K / (self.T2_K - self.T1_K)
            if math.isclose(self.gain, expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"gain numeric matches: {self.gain:.4f} ≈ {expected:.4f}")
            else:
                return (False, f"gain numeric mismatch: {self.gain:.4f} vs {expected:.4f}")
        else:
            return (False, "gain is not numeric; cannot verify.")

    def test_modeling_gain_numeric(self) -> Tuple[bool, str]:
        """Check that gain_num is close to the theoretical value from given temperatures."""
        if self.gain_num is None:
            return (False, "SKIP: gain_num not available.")
        if self.T1_K is None or self.T2_K is None:
            return (False, "SKIP: Temperatures not available.")
        expected = self.T2_K / (self.T2_K - self.T1_K)
        if math.isclose(self.gain_num, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"gain_num {self.gain_num:.4f} ≈ {expected:.4f}")
        else:
            return (False, f"gain_num {self.gain_num:.4f} vs expected {expected:.4f}")

    def test_trace_T2_formula_structure(self) -> Tuple[bool, str]:
        """Check that T2_eq expression has the expected quadratic form (positive root)."""
        if self.T2_eq is None or self.A is None or self.dW_dt is None:
            return (False, "SKIP: T2_eq, A, or dW_dt not available.")
        T2_expr = self.T2_eq
        if isinstance(T2_expr, str):
            try:
                T2_expr = sp.parse_expr(T2_expr)
            except Exception:
                return (False, "Cannot parse T2_eq as expression.")
        if not isinstance(T2_expr, sp.Expr):
            return (False, "T2_eq is not a sympy expression.")
        has_sqrt = any(isinstance(arg, sp.Pow) and arg.exp == sp.Rational(1, 2)
                       for arg in sp.preorder_traversal(T2_expr))
        if not has_sqrt:
            return (False, "T2_eq does not contain a square root term as expected.")
        has_division = any(isinstance(arg, sp.Mul) and self.sym_A in arg.args
                          for arg in sp.preorder_traversal(T2_expr))
        if not has_division:
            return (False, "T2_eq does not involve division by A.")
        return (True, "T2_eq structural checks passed (sqrt and division by A present).")

    # ---------- Monotonicity Tests ----------
    def test_monotonicity_T2_increases_with_power(self) -> Tuple[bool, str]:
        """
        Explicitly perturb dW_dt (+10%) and check that T2_eq increases.
        Uses numerical evaluation of T2_eq expression; defaults if parameters missing.
        """
        # Need T2_eq (as expression), A, dW_dt; use defaults if not provided
        if self.T2_eq is None:
            return (False, "SKIP: T2_eq not available.")
        T2_expr = self.T2_eq
        if isinstance(T2_expr, str):
            try:
                T2_expr = sp.parse_expr(T2_expr)
            except Exception:
                return (False, "Cannot parse T2_eq as expression.")
        if not isinstance(T2_expr, sp.Expr):
            return (False, "T2_eq is not a sympy expression.")

        # Define default parameters if missing
        T1_val = self.T1_K if self.T1_K is not None else 275.15
        A_val = self.A if self.A is not None else 0.1
        dW_dt_val = self.dW_dt if self.dW_dt is not None else 100.0

        # Substitute numeric values into T2_expr
        # First replace symbolic T1, A, dW_dt with actual values
        subs_dict = {}
        if self.T1_K is None:  # use default T1 (275.15) but only if symbolic T1 present
            # Check if expression contains the symbol T1
            if self.sym_T1 in T2_expr.free_symbols:
                subs_dict[self.sym_T1] = T1_val
        else:
            subs_dict[self.sym_T1] = self.T1_K
        if self.A is None:
            if self.sym_A in T2_expr.free_symbols:
                subs_dict[self.sym_A] = A_val
        else:
            subs_dict[self.sym_A] = self.A
        if self.dW_dt is None:
            if self.sym_dW_dt in T2_expr.free_symbols:
                subs_dict[self.sym_dW_dt] = dW_dt_val
        else:
            subs_dict[self.sym_dW_dt] = self.dW_dt

        # If there are still free symbols, we cannot evaluate numerically
        expr_subbed = T2_expr.subs(subs_dict)
        if len(expr_subbed.free_symbols) > 0:
            return (False, f"Cannot fully evaluate T2_eq; free symbols: {expr_subbed.free_symbols}")

        try:
            T2_baseline = float(expr_subbed.evalf())
        except Exception as e:
            return (False, f"Failed to evaluate T2 baseline: {e}")

        # Perturb dW_dt by +10%
        dW_dt_pert = dW_dt_val * 1.1
        # Update subs_dict for dW_dt
        # Note: we need to use the same symbolic key as before
        # If self.dW_dt was None, we used self.sym_dW_dt; if not, we used self.sym_dW_dt as well? Actually we used self.sym_dW_dt for substitution. So update that key.
        if self.dW_dt is None:
            # We substituted self.sym_dW_dt with dW_dt_val; now replace with perturbed
            subs_dict[self.sym_dW_dt] = dW_dt_pert
        else:
            # Original substitution used self.sym_dW_dt as well? Yes, we used self.sym_dW_dt. So update.
            subs_dict[self.sym_dW_dt] = dW_dt_pert  # this will override

        expr_perturbed = T2_expr.subs(subs_dict)
        try:
            T2_perturbed = float(expr_perturbed.evalf())
        except Exception as e:
            return (False, f"Failed to evaluate T2 perturbed: {e}")

        if T2_perturbed > T2_baseline + 1e-6:
            return (True, f"T2 increases with dW_dt: baseline {T2_baseline:.4f} -> perturbed {T2_perturbed:.4f}")
        else:
            return (False, f"T2 did not increase (baseline {T2_baseline:.4f}, perturbed {T2_perturbed:.4f})")

    # ---------- Answer Sanity (Weak) ----------
    def test_answer_sanity_gain_num(self) -> Tuple[bool, str]:
        """Weak sanity: gain_num is positive and finite."""
        if self.gain_num is None:
            return (False, "SKIP: gain_num not available.")
        if not isinstance(self.gain_num, (int, float)):
            return (False, "gain_num is not numeric.")
        if math.isfinite(self.gain_num) and self.gain_num > 0:
            return (True, f"gain_num is positive and finite ({self.gain_num}).")
        else:
            return (False, f"gain_num is not positive/finite: {self.gain_num}")

    # ---------- Test Runner ----------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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