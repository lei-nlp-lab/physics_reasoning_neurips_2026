import math
import sympy as sp
from typing import Any, Dict, Tuple, List

class PhysicsChecker:
    """
    Robust auditor for the elastic cylinder thermodynamics problem.
    Checks interface presence, theorem-based invariants, modeling quality,
    monotonicity (with explicit input perturbation), and one weak answer sanity.
    """

    def __init__(self, solution_instance: Any):
        # Safely extract the results dict
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}

        # Required outputs
        self.Q = results.get('Q', None)
        self.dTdL_S = results.get('dTdL_S', None)
        self.CL_depends_on_L = results.get('CL_depends_on_L', None)

        # Givens (try results first, then direct attribute fallback)
        self.a = results.get('a', None) or getattr(solution_instance, 'a', None)
        self.T = results.get('T', None) or getattr(solution_instance, 'T', None)
        self.L0 = results.get('L0', None) or getattr(solution_instance, 'L0', None)
        self.alpha0 = results.get('alpha0', None) or getattr(solution_instance, 'alpha0', None)
        self.CL = results.get('CL', None) or getattr(solution_instance, 'CL', None)
        self.L = results.get('L', None) or getattr(solution_instance, 'L', None)

        # Explanation steps (for trace consistency, not used heavily)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ------------------------------------------------------------------
    # Helper: safe conversion to float / sympy
    # ------------------------------------------------------------------
    def _num_val(self, x: Any, default: float = None) -> float:
        """Convert to float; return default on failure."""
        if x is None:
            return default
        try:
            return float(x)
        except (ValueError, TypeError):
            return default

    def _parse_sym(self, expr: Any):
        """Parse to sympy expression or return None."""
        if expr is None:
            return None
        if isinstance(expr, (int, float)):
            return sp.nsimplify(expr)
        try:
            return sp.sympify(expr)
        except (sp.SympifyError, TypeError):
            return None

    # ===================== INTERFACE TESTS =====================
    def test_interface_has_Q(self) -> Tuple[bool, str]:
        if self.Q is None:
            return (False, "results['Q'] is missing")
        return (True, "results['Q'] present")

    def test_interface_has_dTdL_S(self) -> Tuple[bool, str]:
        if self.dTdL_S is None:
            return (False, "results['dTdL_S'] is missing")
        return (True, "results['dTdL_S'] present")

    def test_interface_has_CL_depends_on_L(self) -> Tuple[bool, str]:
        if self.CL_depends_on_L is None:
            return (False, "results['CL_depends_on_L'] is missing")
        return (True, "results['CL_depends_on_L'] present")

    # ===================== INVARIANT TESTS (Physics‑based) =====================
    def test_invariant_Q_derivation(self) -> Tuple[bool, str]:
        """Check Q = -a T L0 (1 + 5/2 T α0) using numeric test."""
        if any(v is None for v in [self.a, self.T, self.L0, self.alpha0]):
            return (False, "SKIP: missing a, T, L0, or α0")
        a = self._num_val(self.a)
        T = self._num_val(self.T)
        L0 = self._num_val(self.L0)
        alpha0 = self._num_val(self.alpha0)
        if None in (a, T, L0, alpha0):
            return (False, "SKIP: non‑numeric givens")
        expected = -a * T * L0 * (1 + 2.5 * T * alpha0)
        Q_val = self._num_val(self.Q)
        if Q_val is None:
            return (False, "SKIP: results['Q'] not numeric")
        if math.isclose(Q_val, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Q ≈ {expected} (within tolerance)")
        else:
            return (False, f"Q = {Q_val}, expected {expected}")

    def test_invariant_dTdL_S_derivation(self) -> Tuple[bool, str]:
        """
        Check (∂T/∂L)_S = (T/CL) (∂F/∂T)_L.
        Uses numeric evaluation with sample values.
        """
        needed = [self.a, self.T, self.L0, self.alpha0, self.CL, self.L]
        if any(v is None for v in needed):
            return (False, "SKIP: missing a, T, L0, α0, CL, or L")
        a = self._num_val(self.a)
        T = self._num_val(self.T)
        L0 = self._num_val(self.L0)
        alpha0 = self._num_val(self.alpha0)
        CL = self._num_val(self.CL)
        L = self._num_val(self.L)
        if None in (a, T, L0, alpha0, CL, L):
            return (False, "SKIP: non‑numeric givens")

        # (dF/dT)_L = a (L/L0 - L0²/L²) + a T α0 (-L/L0 + 2 L0²/L²)
        dFdT = (a * (L / L0 - L0 ** 2 / L ** 2) +
                a * T * alpha0 * (-L / L0 + 2 * L0 ** 2 / L ** 2))
        expected = (T / CL) * dFdT
        dTdL_val = self._num_val(self.dTdL_S)
        if dTdL_val is None:
            return (False, "SKIP: results['dTdL_S'] not numeric")
        if math.isclose(dTdL_val, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"(∂T/∂L)_S ≈ {expected} (within tolerance)")
        else:
            return (False, f"(∂T/∂L)_S = {dTdL_val}, expected {expected}")

    def test_invariant_CL_depends_on_L(self) -> Tuple[bool, str]:
        """Check that CL_depends_on_L is True as per physics."""
        if self.CL_depends_on_L is None:
            return (False, "results['CL_depends_on_L'] missing")
        if isinstance(self.CL_depends_on_L, bool):
            if self.CL_depends_on_L:
                return (True, "CL depends on L (True)")
            else:
                return (False, "CL should depend on L, but result says False")
        try:
            val = bool(self.CL_depends_on_L)
            if val:
                return (True, "CL_depends_on_L truthy")
            else:
                return (False, "CL_depends_on_L falsy, expected True")
        except (ValueError, TypeError):
            return (False, "Cannot interpret CL_depends_on_L as boolean")

    # ===================== MODELING / TRACE CONSISTENCY =====================
    def test_trace_Q_expression(self) -> Tuple[bool, str]:
        """Symbolic check that Q matches -a T L0 (1 + 5/2 T α0)."""
        if self.Q is None:
            return (False, "SKIP: no Q expression")
        Q_sym = self._parse_sym(self.Q)
        if Q_sym is None:
            return (False, "SKIP: Q not parseable to sympy")
        a, T, L0, a0 = sp.symbols('a T L0 alpha0')
        expected = -a * T * L0 * (1 + sp.Rational(5, 2) * T * a0)
        diff = sp.simplify(Q_sym - expected)
        if diff == 0:
            return (True, "Q symbolically matches expected expression")
        else:
            return (False, f"Symbolic difference: {diff}")

    def test_trace_dTdL_S_expression(self) -> Tuple[bool, str]:
        """Symbolic check that dTdL_S matches the derived formula."""
        if self.dTdL_S is None:
            return (False, "SKIP: no dTdL_S expression")
        expr = self._parse_sym(self.dTdL_S)
        if expr is None:
            return (False, "SKIP: dTdL_S not parseable to sympy")
        a, T, L0, a0, CL, L = sp.symbols('a T L0 alpha0 CL L')
        expected = (a * T * L0 ** 2) / (CL * L ** 2) * (
            -1 + L ** 3 / L0 ** 3 + T * (2 + L ** 3 / L0 ** 3) * a0
        )
        diff = sp.simplify(expr - expected)
        if diff == 0:
            return (True, "dTdL_S symbolically matches expected expression")
        else:
            return (False, f"Symbolic difference: {diff}")

    # ===================== MONOTONICITY TESTS =====================
    def test_monotonicity_Q_vs_T(self) -> Tuple[bool, str]:
        """
        Q becomes more negative as T increases (assuming α0 > 0).
        Perturb T by +10% and verify that Q decreases (becomes more negative).
        """
        if any(v is None for v in [self.a, self.T, self.L0, self.alpha0]):
            return (False, "SKIP: missing a, T, L0, α0")
        a = self._num_val(self.a)
        T0 = self._num_val(self.T)
        L0 = self._num_val(self.L0)
        alpha0 = self._num_val(self.alpha0)
        if None in (a, T0, L0, alpha0):
            return (False, "SKIP: non‑numeric givens")
        # Baseline Q
        Q_base = -a * T0 * L0 * (1 + 2.5 * T0 * alpha0)
        # Increase T by 10%
        T_pert = T0 * 1.1
        Q_pert = -a * T_pert * L0 * (1 + 2.5 * T_pert * alpha0)
        # Since Q is negative, a more negative value is smaller.
        if Q_pert < Q_base:
            return (True, f"Q becomes more negative from {Q_base} to {Q_pert} when T +10%")
        else:
            return (False, f"Monotonicity violated: Q_base={Q_base}, Q_pert={Q_pert}")

    def test_monotonicity_dTdL_S_vs_L(self) -> Tuple[bool, str]:
        """
        For a given set, (∂T/∂L)_S increases with L (because the dominant term L^3/L0^3 grows).
        Perturb L by +10% and verify that the computed dTdL_S increases.
        """
        needed = [self.a, self.T, self.L0, self.alpha0, self.CL, self.L]
        if any(v is None for v in needed):
            return (False, "SKIP: missing a, T, L0, α0, CL, or L")
        a = self._num_val(self.a)
        T = self._num_val(self.T)
        L0 = self._num_val(self.L0)
        alpha0 = self._num_val(self.alpha0)
        CL = self._num_val(self.CL)
        L_base = self._num_val(self.L)
        if None in (a, T, L0, alpha0, CL, L_base):
            return (False, "SKIP: non‑numeric givens")

        def dTdL_formula(L_val: float) -> float:
            return (a * T * L0 ** 2) / (CL * L_val ** 2) * (
                -1 + L_val ** 3 / L0 ** 3 + T * (2 + L_val ** 3 / L0 ** 3) * alpha0
            )

        base_val = dTdL_formula(L_base)
        L_pert = L_base * 1.1
        pert_val = dTdL_formula(L_pert)
        # Expect base_val < pert_val because the function is increasing for large L
        if pert_val > base_val:
            return (True, f"dTdL_S increases from {base_val} to {pert_val} when L +10%")
        else:
            return (False, f"Monotonicity violated: base={base_val}, perturbed={pert_val}")

    # ===================== WEAK ANSWER SANITY (only one method) =====================
    def test_answer_sanity_basic_checks(self) -> Tuple[bool, str]:
        """
        Weak sanity: all three results should be present and have plausible types.
        (Only one answer_sanity method per requirement.)
        """
        messages = []
        ok = True
        if self.Q is None:
            messages.append("missing Q")
            ok = False
        else:
            q_val = self._num_val(self.Q)
            if q_val is not None and not math.isfinite(q_val):
                messages.append("Q non‑finite")
                ok = False
        if self.dTdL_S is None:
            messages.append("missing dTdL_S")
            ok = False
        else:
            d_val = self._num_val(self.dTdL_S)
            if d_val is not None and not math.isfinite(d_val):
                messages.append("dTdL_S non‑finite")
                ok = False
        if self.CL_depends_on_L is None:
            messages.append("missing CL_depends_on_L")
            ok = False
        elif not isinstance(self.CL_depends_on_L, bool):
            messages.append("CL_depends_on_L not bool")
            ok = False
        if not messages:
            messages.append("all present and plausible")
        return (ok, "; ".join(messages))

    # ===================== RUN ALL TESTS =====================
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and execute every test_* method, collect results."""
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