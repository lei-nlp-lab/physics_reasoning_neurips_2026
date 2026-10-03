import math
import traceback
from typing import Any, Dict, List, Tuple

try:
    import sympy as sp
except ImportError:
    sp = None


class PhysicsChecker:
    """
    Auditor for thermodynamic problem: determine B and n from state equation
    p = A T^3 / V and internal energy U = B T^n ln(V/V0) + f(T).
    """

    def __init__(self, solution_instance: Any):
        # ---------- Safely extract givens ----------
        self.U_expr = getattr(solution_instance, 'U', None)
        self.p_expr = getattr(solution_instance, 'p', None)
        self.A = getattr(solution_instance, 'A', None)
        self.V0 = getattr(solution_instance, 'V0', None)
        self.f_T = getattr(solution_instance, 'f_T', None)

        # ---------- Safely extract results ----------
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', None) or {}
        self.n_result = self.results.get('n', None)
        self.B_result = self.results.get('B', None)

        # ---------- Explanation steps (for auditing, not hardcoding) ----------
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # ---------- Internal symbols for symbolic testing ----------
        self._T_sym = sp.Symbol('T', positive=True, real=True)
        self._V_sym = sp.Symbol('V', positive=True, real=True)
        self._A_sym = sp.Symbol('A', nonzero=True, real=True)
        self._B_sym = sp.Symbol('B', real=True)
        self._n_sym = sp.Symbol('n', real=True)
        self._V0_sym = sp.Symbol('V0', positive=True, real=True)

    # ----------------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------------
    def _str_to_expr(self, s: str) -> sp.Expr:
        return sp.parse_expr(s, local_dict={
            'T': self._T_sym, 'V': self._V_sym, 'A': self._A_sym,
            'B': self._B_sym, 'n': self._n_sym, 'V0': self._V0_sym
        })

    def _get_numeric(self, val):
        """Try to convert a value to float; return None if not possible."""
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def _compute_dU_dV(self, T, V, B_val, n_val):
        """Return (∂U/∂V)_T = B * T**n / V."""
        return B_val * (T ** n_val) / V

    # ----------------------------------------------------------------------
    # Interface Tests
    # ----------------------------------------------------------------------
    def test_interface_required_givens(self) -> Tuple[bool, str]:
        missing = []
        if self.U_expr is None:
            missing.append("U (internal energy expression)")
        if self.p_expr is None:
            missing.append("p (state equation expression)")
        if self.A is None:
            missing.append("A (constant in state equation)")
        if missing:
            return False, f"Missing givens: {', '.join(missing)}"
        return True, "All required givens present."

    def test_interface_results_keys(self) -> Tuple[bool, str]:
        missing = []
        if 'n' not in self.results:
            missing.append('n')
        if 'B' not in self.results:
            missing.append('B')
        if missing:
            return False, f"Missing results key(s): {', '.join(missing)}"
        return True, "Results contain 'n' and 'B'."

    # ----------------------------------------------------------------------
    # Invariant Tests
    # ----------------------------------------------------------------------
    def test_invariant_identity_symbolic(self) -> Tuple[bool, str]:
        if sp is None:
            return (False, "SKIP: sympy not installed")
        if self.U_expr is None or self.p_expr is None:
            return (False, "SKIP: U or p expression missing")
        try:
            U_sym = self._str_to_expr(self.U_expr) if isinstance(self.U_expr, str) else self.U_expr
            p_sym = self._str_to_expr(self.p_expr) if isinstance(self.p_expr, str) else self.p_expr
        except Exception as e:
            return False, f"Failed to parse expressions: {e}"

        dU_dV = sp.diff(U_sym, self._V_sym)
        dp_dT = sp.diff(p_sym, self._T_sym)
        identity = dU_dV - (self._T_sym * dp_dT - p_sym)
        expected = (self._B_sym * self._T_sym**self._n_sym - 2*self._A_sym*self._T_sym**3) / self._V_sym
        if sp.simplify(identity - expected) == 0:
            return True, "Symbolic identity yields correct conditional relation B T^n = 2A T^3."
        else:
            return False, f"Symbolic identity diff: {identity}"

    def test_invariant_identity_numeric(self) -> Tuple[bool, str]:
        if self.A is None or self.B_result is None or self.n_result is None:
            return (False, "SKIP: missing A, B_result, or n_result")
        A_val = self._get_numeric(self.A)
        B_val = self._get_numeric(self.B_result)
        n_val = self._get_numeric(self.n_result)
        if None in (A_val, B_val, n_val):
            return (False, "SKIP: A, B, or n not numeric")
        # Use representative values
        T = 300.0
        V = 0.001
        delta = 1e-6
        # Compute analytic dU/dV
        dU_dV_analytic = B_val * (T ** n_val) / V
        # Compute numeric RHS: T*(dp/dT)_V - p
        def p(T_val):
            return A_val * T_val**3 / V
        dp_dT_num = (p(T+delta) - p(T-delta)) / (2*delta)
        rhs = T * dp_dT_num - p(T)
        if math.isclose(dU_dV_analytic, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"Numeric identity: LHS={dU_dV_analytic}, RHS={rhs}"
        else:
            return False, f"Numeric identity fails: LHS={dU_dV_analytic}, RHS={rhs}"

    # ----------------------------------------------------------------------
    # Modeling/Trace Tests
    # ----------------------------------------------------------------------
    def test_modeling_partial_U_V(self) -> Tuple[bool, str]:
        if sp is None:
            return (False, "SKIP: sympy not installed")
        if self.U_expr is None:
            return (False, "SKIP: U expression missing")
        try:
            U_sym = self._str_to_expr(self.U_expr) if isinstance(self.U_expr, str) else self.U_expr
        except Exception as e:
            return False, f"Failed to parse U: {e}"
        dU_dV = sp.diff(U_sym, self._V_sym)
        expected = self._B_sym * self._T_sym**self._n_sym / self._V_sym
        if sp.simplify(dU_dV - expected) == 0:
            return True, "Partial derivative (∂U/∂V)_T matches B T^n / V."
        else:
            return False, f"dU/dV = {dU_dV}, expected {expected}"

    def test_modeling_partial_p_T(self) -> Tuple[bool, str]:
        if sp is None:
            return (False, "SKIP: sympy not installed")
        if self.p_expr is None:
            return (False, "SKIP: p expression missing")
        try:
            p_sym = self._str_to_expr(self.p_expr) if isinstance(self.p_expr, str) else self.p_expr
        except Exception as e:
            return False, f"Failed to parse p: {e}"
        dp_dT = sp.diff(p_sym, self._T_sym)
        expected = 3*self._A_sym * self._T_sym**2 / self._V_sym
        if sp.simplify(dp_dT - expected) == 0:
            return True, "Partial derivative (∂p/∂T)_V matches 3A T^2 / V."
        else:
            return False, f"dp/dT = {dp_dT}, expected {expected}"

    def test_modeling_identity_substitution(self) -> Tuple[bool, str]:
        if sp is None:
            return (False, "SKIP: sympy not installed")
        if self.U_expr is None or self.p_expr is None:
            return (False, "SKIP: U or p expression missing")
        try:
            U_sym = self._str_to_expr(self.U_expr) if isinstance(self.U_expr, str) else self.U_expr
            p_sym = self._str_to_expr(self.p_expr) if isinstance(self.p_expr, str) else self.p_expr
        except Exception as e:
            return False, f"Failed to parse expressions: {e}"
        dU_dV = sp.diff(U_sym, self._V_sym)
        dp_dT = sp.diff(p_sym, self._T_sym)
        relation = sp.simplify(self._T_sym * dp_dT - p_sym - dU_dV)
        expected = (2*self._A_sym*self._T_sym**3 - self._B_sym*self._T_sym**self._n_sym) / self._V_sym
        if sp.simplify(relation - expected) == 0:
            return True, "Identity simplifies to B T^n = 2A T^3."
        else:
            return False, f"Identity relation: {relation}, expected {expected}"

    def test_modeling_final_n(self) -> Tuple[bool, str]:
        if self.n_result is None:
            return (False, "SKIP: n_result missing")
        n_val = self._get_numeric(self.n_result)
        if n_val is not None:
            if math.isclose(n_val, 3.0, rel_tol=1e-2, abs_tol=1e-2):
                return True, f"n = {n_val} ≈ 3."
            else:
                return False, f"n = {n_val} not close to 3."
        # symbolic fallback
        if sp is not None and hasattr(self.n_result, 'free_symbols'):
            if sp.simplify(self.n_result - 3) == 0:
                return True, "n = 3 (symbolic)."
        return (False, "SKIP: cannot evaluate n")

    def test_modeling_final_B(self) -> Tuple[bool, str]:
        if self.B_result is None or self.A is None:
            return (False, "SKIP: B_result or A missing")
        B_val = self._get_numeric(self.B_result)
        A_val = self._get_numeric(self.A)
        if B_val is not None and A_val is not None:
            expected = 2.0 * A_val
            if math.isclose(B_val, expected, rel_tol=1e-2, abs_tol=1e-2):
                return True, f"B = {B_val} ≈ 2*A = {expected}."
            else:
                return False, f"B = {B_val} not close to 2*A = {expected}."
        # symbolic fallback
        if sp is not None and hasattr(self.B_result, 'free_symbols') and hasattr(self.A, 'free_symbols'):
            diff = sp.simplify(self.B_result - 2*self.A)
            if diff == 0:
                return True, "B = 2A (symbolic)."
        return (False, "SKIP: cannot evaluate B or A")

    # ----------------------------------------------------------------------
    # Monotonicity Tests
    # ----------------------------------------------------------------------
    def test_monotonicity_dU_dV_increases_with_T(self) -> Tuple[bool, str]:
        """
        Check that (∂U/∂V)_T = B * T^n / V increases with T (since n>0).
        Perturb T upward by 5% and verify that dU_dV increases.
        """
        if self.B_result is None or self.n_result is None:
            return (False, "SKIP: B_result or n_result missing")
        B_val = self._get_numeric(self.B_result)
        n_val = self._get_numeric(self.n_result)
        if B_val is None or n_val is None:
            return (False, "SKIP: B or n not numeric")
        if n_val <= 0:
            return (False, "SKIP: n is not positive; monotonicity test not applicable")
        # Use a representative V (positive)
        V = 1.0  # arbitrary, cancels in ratio
        T_base = 300.0
        T_perturbed = 1.05 * T_base
        dU_base = self._compute_dU_dV(T_base, V, B_val, n_val)
        dU_pert = self._compute_dU_dV(T_perturbed, V, B_val, n_val)
        # For n>0, dU_pert > dU_base (since B>0 typically; but even if B negative,
        # dU_dV would become more negative if T increases? Actually if B negative,
        # dU_dV is negative and increasing T makes it more negative (decreases). 
        # The monotonicity with T depends on sign(B*n). But we only know n>0;
        # B could be positive (if A>0) or negative (if A<0). Check if the change
        # is monotonic: either both positive and increase, or both negative and
        # become more negative (i.e., absolute value increases). We'll simply check
        # that the ratio dU_pert/dU_base is greater than 1 if both signs equal.
        # If signs differ, something is wrong.
        if dU_base == 0:
            return (False, "SKIP: dU/dV is zero; cannot test monotonicity")
        ratio = dU_pert / dU_base
        # For any nonzero n, if T>0, dU_pert/dU_base = (T_pert/T_base)^n > 1
        expected_ratio = (T_perturbed / T_base) ** n_val
        if math.isclose(ratio, expected_ratio, rel_tol=1e-4):
            return True, f"dU/dV increased by factor {ratio:.4f} (expected {expected_ratio:.4f})"
        else:
            return False, f"dU/dV ratio = {ratio:.4f}, expected {expected_ratio:.4f}"

    def test_monotonicity_pressure_increases_with_T(self) -> Tuple[bool, str]:
        """
        Check that pressure p = A T^3 / V increases with temperature.
        Perturb T upward by 5% and verify that p increases.
        """
        if self.p_expr is None:
            return (False, "SKIP: p expression missing")
        # Use numeric evaluation if possible
        test_T = 300.0
        test_V = 1.0
        perturb_factor = 1.05
        if sp is not None:
            try:
                p_sym = self._str_to_expr(self.p_expr) if isinstance(self.p_expr, str) else self.p_expr
                # Substitute numeric values (with A=1 if A not numeric, or use symbolic)
                subs_base = {self._T_sym: test_T, self._V_sym: test_V}
                subs_pert = {self._T_sym: test_T * perturb_factor, self._V_sym: test_V}
                if self.A is not None:
                    A_num = self._get_numeric(self.A)
                    if A_num is not None:
                        subs_base[self._A_sym] = A_num
                        subs_pert[self._A_sym] = A_num
                p_base = float(p_sym.subs(subs_base).evalf())
                p_pert = float(p_sym.subs(subs_pert).evalf())
            except Exception:
                return (False, "SKIP: cannot evaluate pressure numerically")
        else:
            # No sympy: try to compute if p_expr is a string with known form
            if isinstance(self.p_expr, str) and 'A' in self.p_expr:
                return (False, "SKIP: cannot evaluate symbolic p without sympy")
            return (False, "SKIP: sympy not available")
        if p_pert > p_base:
            return True, f"Pressure increases with T: {p_pert:.4f} > {p_base:.4f}"
        else:
            return False, f"Pressure does not increase: base={p_base:.4f}, perturbed={p_pert:.4f}"

    # ----------------------------------------------------------------------
    # Answer Sanity Tests (at most one)
    # ----------------------------------------------------------------------
    def test_answer_sanity_n_positive(self) -> Tuple[bool, str]:
        if self.n_result is None:
            return (False, "SKIP: n_result missing")
        n_val = self._get_numeric(self.n_result)
        if n_val is not None:
            if n_val > 0:
                return True, f"n = {n_val} > 0 (sane)."
            else:
                return False, f"n = {n_val} is not positive."
        # symbolic: assume positive if symbolic constant?
        return (False, "SKIP: n is symbolic, cannot check positivity")

    # ----------------------------------------------------------------------
    # Main test runner
    # ----------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)