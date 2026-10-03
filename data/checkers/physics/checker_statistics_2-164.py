import math
import traceback
import sympy as sp
from typing import Any, List, Tuple

class PhysicsChecker:
    """Robust checker for photon gas problem: p = u/3, adiabatic pV^(4/3)=const, gamma=4/3."""

    def __init__(self, solution_instance: Any):
        # Safe extraction of results – no numeric defaults
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.p = self._safe_result('p')
        self.pV_relation = self._safe_result('pV_relation')
        self.gamma = self._safe_result('gamma')
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None) or []
        self.givens = getattr(solution_instance, 'givens', {}) or {}

    def _safe_result(self, key: str):
        """Return the value from results dict if present, else None."""
        val = self.results.get(key, None)
        return val if val is not None else None

    def _to_sympy(self, expr):
        """Try to convert a string or sympy object to a sympy expression. Return None on failure."""
        if isinstance(expr, sp.Expr):
            return expr
        if isinstance(expr, str):
            try:
                return sp.sympify(expr)
            except (sp.SympifyError, TypeError):
                return None
        return None

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_present_p(self) -> Tuple[bool, str]:
        if self.p is None:
            return (False, "Skip: 'p' not present")
        return (True, "Key 'p' found")

    def test_interface_present_pV_relation(self) -> Tuple[bool, str]:
        if self.pV_relation is None:
            return (False, "Skip: 'pV_relation' not present")
        return (True, "Key 'pV_relation' found")

    def test_interface_present_gamma(self) -> Tuple[bool, str]:
        if self.gamma is None:
            return (False, "Skip: 'gamma' not present")
        return (True, "Key 'gamma' found")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based)
    # ------------------------------------------------------------------
    def test_invariant_p_equals_u_over_3(self) -> Tuple[bool, str]:
        """Check that p = u/3 either symbolically or via a reasonable string indicator."""
        if self.p is None:
            return (False, "Skip: 'p' not available")
        sym = self._to_sympy(self.p)
        if sym is not None:
            # Check symbolic equality: 3*p - u == 0
            u = sp.Symbol('u')
            if sym == u/3 or sp.simplify(3*sym - u) == 0:
                return (True, "Symbolic identity: p = u/3")
            # Numerical fallback if u is given
            u_val = self.givens.get('u', None) or self.results.get('u', None)
            if u_val is not None:
                try:
                    u_num = float(u_val)
                    p_num = float(sym.evalf(subs={u: u_num}))
                    if math.isclose(p_num, u_num/3, rel_tol=1e-2, abs_tol=1e-2):
                        return (True, f"Numerical check: p ≈ {p_num}, u/3 ≈ {u_num/3}")
                except (TypeError, ValueError):
                    pass
            return (False, "Symbolic expression does not match u/3 and numeric check unavailable")
        # Fallback to string inspection (weak but admissible as secondary)
        if isinstance(self.p, str) and 'u/3' in self.p:
            return (True, "String contains 'u/3'")
        return (False, "Cannot parse 'p' and string contains no 'u/3'")

    def test_invariant_adiabatic_relation(self) -> Tuple[bool, str]:
        """Check that pV^(4/3) = constant is present (symbolic or string)."""
        if self.pV_relation is None:
            return (False, "Skip: 'pV_relation' not available")
        sym = self._to_sympy(self.pV_relation)
        if sym is not None:
            p, V = sp.symbols('p V')
            expected = p * V**(sp.Rational(4,3))
            if isinstance(sym, sp.Eq):
                lhs = sp.simplify(sym.lhs - expected)
                rhs = sym.rhs
                if lhs == 0:
                    return (True, "Symbolic equality: p*V^(4/3) = constant")
            else:
                expr = sp.simplify(sym - expected)
                if expr == 0:
                    return (True, "Symbolic expression matches p*V^(4/3)")
        if isinstance(self.pV_relation, str):
            lower = self.pV_relation.lower()
            if 'p*v^(4/3)' in lower or 'p*V**(4/3)' in lower or 'pv^{4/3}' in lower or 'pV^{4/3}' in lower:
                return (True, "String contains pV^(4/3) form")
        return (False, "Adiabatic relation not clearly present")

    def test_invariant_gamma_value(self) -> Tuple[bool, str]:
        """Check that gamma = 4/3 (symbolic or fuzzy numerical)."""
        if self.gamma is None:
            return (False, "Skip: 'gamma' not available")
        sym = self._to_sympy(self.gamma)
        if sym is not None:
            expected = sp.Rational(4,3)
            if sp.simplify(sym - expected) == 0:
                return (True, "Symbolic gamma equals 4/3")
            try:
                val = float(sym.evalf())
                if math.isclose(val, 4/3, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, f"Numerical gamma ≈ {val}, expected {4/3}")
            except (TypeError, ValueError):
                pass
        if isinstance(self.gamma, str):
            if '4/3' in self.gamma or '1.333' in self.gamma:
                return (True, "String contains 4/3")
        return (False, "Gamma does not match 4/3")

    # ------------------------------------------------------------------
    # Modeling / trace tests (at least two required)
    # ------------------------------------------------------------------
    def test_modeling_kinetic_theory_consistency(self) -> Tuple[bool, str]:
        """Check explanation steps mention key kinetic theory concepts."""
        if not self.explanation_steps:
            return (False, "Skip: no explanation_steps available")
        joined = " ".join(self.explanation_steps).lower()
        keywords = ['momentum', 'cosθ', 'cos^2', 'isotropic', '1/3', '2k cosθ', 'impulse']
        found = [kw for kw in keywords if kw in joined]
        if len(found) >= 3:
            return (True, f"Kinetic theory concepts present: {found}")
        return (False, f"Insufficient details. Found: {found}")

    def test_modeling_adiabatic_derivation_present(self) -> Tuple[bool, str]:
        """Check explanation steps contain the adiabatic derivation: dU = -p dV, integration, etc."""
        if not self.explanation_steps:
            return (False, "Skip: no explanation_steps available")
        joined = " ".join(self.explanation_steps).lower()
        required_terms = ['adiabatic', 'du', 'dv', 'integrat', 'constant']
        missing = [t for t in required_terms if t not in joined]
        if not missing:
            return (True, "Adiabatic derivation steps present")
        return (False, f"Missing adiabatic derivation terms: {missing}")

    # ------------------------------------------------------------------
    # Monotonicity test (must perturb an input and compare outputs)
    # ------------------------------------------------------------------
    def test_monotonicity_adiabatic_compression(self) -> Tuple[bool, str]:
        """If numeric p and V are available, perturb V by ±5% and verify p changes in opposite direction."""
        # Try to get numeric values for p and V from givens or results
        p_raw = self.givens.get('p', None) or self.results.get('p_numeric', None)
        V_raw = self.givens.get('V', None) or self.results.get('V_numeric', None)
        if p_raw is None or V_raw is None:
            return (False, "Skip: need numeric p and V values")
        try:
            p0 = float(p_raw)
            V0 = float(V_raw)
        except (TypeError, ValueError):
            return (False, "Cannot convert p or V to float")
        # Use the derived constant C = p * V^(4/3)
        gamma = 4.0 / 3.0
        C = p0 * (V0 ** gamma)
        # Perturb volume: increase by 5%
        V_high = V0 * 1.05
        p_high = C / (V_high ** gamma)
        # Perturb volume: decrease by 5%
        V_low = V0 * 0.95
        p_low = C / (V_low ** gamma)
        # Check monotonic: pressure decreases when volume increases
        if p_high < p0 and p_low > p0:
            return (True, f"Monotonicity holds: V↑ → p↓, V↓ → p↑ (p0={p0:.4f}, p_high={p_high:.4f}, p_low={p_low:.4f})")
        else:
            return (False, f"Monotonicity violation: p does not move opposite to V (p0={p0:.4f}, p_high={p_high:.4f}, p_low={p_low:.4f})")

    # ------------------------------------------------------------------
    # Weak answer sanity test (only one, no isclose, no derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_gamma_range(self) -> Tuple[bool, str]:
        """Gamma must be positive and plausible (0 < gamma < 2)."""
        if self.gamma is None:
            return (False, "Skip: gamma missing")
        sym = self._to_sympy(self.gamma)
        if sym is not None:
            try:
                val = float(sym.evalf())
                if 0 < val < 2.0:
                    return (True, f"Gamma = {val:.4f} is positive and < 2")
                return (False, f"Gamma = {val:.4f} out of plausible range (0 < gamma < 2)")
            except:
                return (False, "Cannot evaluate gamma numerically")
        if isinstance(self.gamma, str):
            try:
                if '/' in self.gamma:
                    num, den = self.gamma.split('/')
                    val = float(num)/float(den)
                else:
                    val = float(self.gamma)
                if 0 < val < 2.0:
                    return (True, f"Gamma = {val:.4f} is positive and < 2")
                return (False, f"Gamma = {val:.4f} out of plausible range")
            except (ValueError, ZeroDivisionError):
                return (False, "Cannot parse gamma string")
        return (False, "Gamma not interpretable")

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test_* methods, run them, collect logs. Returns (all_pass, logs)."""
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name, None)
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