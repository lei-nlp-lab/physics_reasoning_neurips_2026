import math
import sympy as sp
import traceback
from typing import Tuple, List, Optional, Any

class PhysicsChecker:
    """
    Physics checker for heat conduction problem: analogy, heat equation,
    steady-state radial temperature, and heat loss rate.
    """

    def __init__(self, solution_instance: Any):
        # Extract results dict safely
        results = getattr(solution_instance, 'results', None) or {}
        self.analogy = results.get('analogy', None)
        self.heat_eq = results.get('heat_eq', None)
        self.T_r = results.get('T_r', None)
        self.heat_loss = results.get('heat_loss', None)

        # Extract explanation steps safely
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None) or []

        # Extract givens safely (numeric parameters)
        givens = getattr(solution_instance, 'givens', None)
        if givens is None:
            givens = {}

        self.r1 = givens.get('r1', None)
        self.r2 = givens.get('r2', None)
        self.L = givens.get('L', None)
        self.k = givens.get('k', None)
        self.T1 = givens.get('T1', None)
        self.T2 = givens.get('T2', None)
        self.c = givens.get('c', None)
        self.rho = givens.get('rho', None)

        # Sympy symbols for parsing
        self._r, self._r1, self._r2 = sp.symbols('r r1 r2')
        self._T1, self._T2 = sp.symbols('T1 T2')

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #

    def _sympify_T_r(self) -> Optional[sp.Expr]:
        """Try to parse self.T_r into a sympy expression. Returns None on failure."""
        if not self.T_r:
            return None
        try:
            # Remove 'T(r) = ' or similar prefix, keep only expression part
            expr_str = self.T_r.strip()
            if '=' in expr_str:
                expr_str = expr_str.split('=')[-1].strip()
            # Replace 'ln' with 'log' and handle formatting
            expr_str = expr_str.replace('ln', 'log')
            # Replace variable names in string with sympy equivalent
            local_dict = {
                'r': self._r, 'r1': self._r1, 'r2': self._r2,
                'T1': self._T1, 'T2': self._T2
            }
            expr = sp.sympify(expr_str, locals=local_dict)
            return expr
        except Exception:
            return None

    def _sympify_heat_loss(self) -> Optional[sp.Expr]:
        """Try to parse self.heat_loss into a sympy expression. Returns None on failure."""
        if not self.heat_loss:
            return None
        try:
            expr_str = self.heat_loss.strip()
            if '=' in expr_str:
                expr_str = expr_str.split('=')[-1].strip()
            # Replace common symbols: pi, ln, etc.
            expr_str = expr_str.replace('π', 'pi').replace('ln', 'log')
            # This string may have multiplication as simple juxtaposition; we need to insert *
            # We'll attempt sympify directly; it may fail.
            local_dict = {
                'r': self._r, 'r1': self._r1, 'r2': self._r2,
                'T1': self._T1, 'T2': self._T2, 'k': sp.Symbol('k'),
                'L': sp.Symbol('L'), 'pi': sp.pi
            }
            expr = sp.sympify(expr_str, locals=local_dict)
            return expr
        except Exception:
            return None

    def _has_numeric_givens(self) -> bool:
        """Check if we have all numeric inputs for pipe problem."""
        return all(v is not None for v in [self.r1, self.r2, self.L, self.k, self.T1, self.T2])

    # ------------------------------------------------------------------ #
    # Interface tests
    # ------------------------------------------------------------------ #

    def test_interface_keys_present(self) -> Tuple[bool, str]:
        """Check that all required result keys are present."""
        expected_keys = ['analogy', 'heat_eq', 'T_r', 'heat_loss']
        missing = [k for k in expected_keys if getattr(self, k) is None]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    def test_interface_analogy_format(self) -> Tuple[bool, str]:
        """Check that analogy string contains expected mapping symbols."""
        if self.analogy is None:
            return (False, "SKIP: analogy not available")
        parts = ['H<->J', 'k<->sigma', 'T<->V', 'crhoT<->rho']
        missing = [p for p in parts if p not in self.analogy]
        if missing:
            return (False, f"Analogy missing parts: {missing}")
        return (True, "Analogy contains all expected mappings")

    def test_interface_heat_eq_format(self) -> Tuple[bool, str]:
        """Check that heat_eq string contains dT/dt or similar."""
        if self.heat_eq is None:
            return (False, "SKIP: heat_eq not available")
        if 'dT/dt' in self.heat_eq or '∂T/∂t' in self.heat_eq:
            return (True, "heat_eq contains time derivative")
        return (False, "heat_eq does not contain time derivative")

    def test_interface_T_r_format(self) -> Tuple[bool, str]:
        """Check that T_r string contains ln and expected variables."""
        if self.T_r is None:
            return (False, "SKIP: T_r not available")
        needed = ['ln', 'T1', 'T2', 'r1', 'r2']
        missing = [s for s in needed if s not in self.T_r]
        if missing:
            return (False, f"T_r missing parts: {missing}")
        return (True, "T_r contains enough expected terms")

    def test_interface_heat_loss_format(self) -> Tuple[bool, str]:
        """Check that heat_loss string contains ln and pi and L."""
        if self.heat_loss is None:
            return (False, "SKIP: heat_loss not available")
        needed = ['ln', 'π', 'L', 'k', 'T1', 'T2']
        missing = [s for s in needed if s not in self.heat_loss]
        if missing:
            return (False, f"heat_loss missing parts: {missing}")
        return (True, "heat_loss contains enough expected terms")

    # ------------------------------------------------------------------ #
    # Invariant tests (theorem-based)
    # ------------------------------------------------------------------ #

    def test_invariant_steady_state_laplace(self) -> Tuple[bool, str]:
        """
        Check that the symbolic T(r) expression satisfies radial Laplace equation:
        (1/r) d/dr (r dT/dr) = 0.
        """
        expr = self._sympify_T_r()
        if expr is None:
            return (False, "SKIP: cannot parse T_r expression")
        deriv = sp.diff(expr, self._r)
        radial_part = self._r * deriv
        d_radial = sp.diff(radial_part, self._r)
        simplified = sp.simplify(d_radial)
        if simplified == 0:
            return (True, "T(r) satisfies (1/r) d/dr(r dT/dr)=0")
        else:
            return (False, f"T(r) does not satisfy equation: {simplified} != 0")

    def test_invariant_boundary_conditions(self) -> Tuple[bool, str]:
        """Check that T(r1)=T1 and T(r2)=T2 symbolically."""
        expr = self._sympify_T_r()
        if expr is None:
            return (False, "SKIP: cannot parse T_r expression")
        T_at_r1 = sp.simplify(expr.subs({self._r: self._r1}))
        T_at_r2 = sp.simplify(expr.subs({self._r: self._r2}))
        if T_at_r1 == self._T1 and T_at_r2 == self._T2:
            return (True, "Boundary conditions satisfied symbolically")
        else:
            return (False, f"BC failed: T(r1)={T_at_r1}, T(r2)={T_at_r2}")

    def test_invariant_fourier_law_heat_loss(self) -> Tuple[bool, str]:
        """
        Check that from T_r we can derive the heat loss expression q = 2πkL (T1-T2)/ln(r2/r1).
        Compare with heat_loss string if parseable.
        """
        T_expr = self._sympify_T_r()
        q_expr = self._sympify_heat_loss()
        if T_expr is None or q_expr is None:
            return (False, "SKIP: cannot parse T_r or heat_loss expression")
        dT_dr = sp.diff(T_expr, self._r)
        k_sym, L_sym = sp.Symbol('k'), sp.Symbol('L')
        derived_q = -k_sym * dT_dr * 2 * sp.pi * self._r * L_sym
        derived_q_simpl = sp.simplify(derived_q)
        derived_q_at_r1 = sp.simplify(derived_q_simpl.subs({self._r: self._r1}))
        diff = sp.simplify(derived_q_at_r1 - q_expr)
        if diff == 0:
            return (True, "Heat loss derived from Fourier's law matches user expression")
        else:
            return (False, f"Mismatch: derived={derived_q_at_r1}, user={q_expr}, diff={diff}")

    def test_invariant_analogy_continuity(self) -> Tuple[bool, str]:
        """Check that explanation steps mention continuity/energy conservation."""
        if not self.explanation_steps:
            return (False, "SKIP: no explanation_steps available")
        steps_text = ' '.join(self.explanation_steps).lower()
        keywords = ['continuity', '∇·j', 'divergence', 'conservation']
        matches = [kw for kw in keywords if kw in steps_text]
        if matches:
            return (True, f"Explanation steps mention continuity: {matches}")
        if 'energy conservation' in steps_text or 'steady state' in steps_text:
            return (True, "Explanation steps mention energy conservation")
        return (False, "Explanation steps do not reference continuity or conservation")

    # ------------------------------------------------------------------ #
    # Modeling / trace tests
    # ------------------------------------------------------------------ #

    def test_modeling_laplace_mentioned(self) -> Tuple[bool, str]:
        """Check that explanation steps mention Laplace's equation."""
        if not self.explanation_steps:
            return (False, "SKIP: no explanation_steps available")
        steps_text = ' '.join(self.explanation_steps).lower()
        if 'laplace' in steps_text:
            return (True, "Laplace's equation mentioned")
        return (False, "Laplace's equation not mentioned")

    def test_modeling_integral_form(self) -> Tuple[bool, str]:
        """Check that explanation steps mention integration (C ln r + D pattern)."""
        if not self.explanation_steps:
            return (False, "SKIP: no explanation_steps available")
        steps_text = ' '.join(self.explanation_steps).lower()
        if 'integrate' in steps_text or 'c ln r' in steps_text or 'ln(r)' in steps_text:
            return (True, "Integration is mentioned")
        return (False, "Integration pattern not found in steps")

    def test_modeling_symmetric_form_T_r(self) -> Tuple[bool, str]:
        """Check that T_r expression is given in symmetric form (1/ln(r1/r2)[...])"""
        if self.T_r is None:
            return (False, "SKIP: T_r not available")
        if 'ln(r1/r2)' in self.T_r or 'ln(r_1/r_2)' in self.T_r or '\\ln\\frac{r_1}{r_2}' in self.T_r:
            return (True, "T_r uses symmetric ln(r1/r2) form")
        if 'r1/r2' in self.T_r and 'ln' in self.T_r:
            return (True, "T_r appears to have symmetric form")
        return (False, "T_r does not show symmetric ln(r1/r2) form")

    # ------------------------------------------------------------------ #
    # Monotonicity tests (each must perturb an input and compare output)
    # ------------------------------------------------------------------ #

    def test_monotonicity_q_increases_with_T1_increase(self) -> Tuple[bool, str]:
        """
        Numerically test that if T1 increases by 1%, q increases.
        Requires numeric givens with T1 > T2.
        """
        if not self._has_numeric_givens():
            return (False, "SKIP: numeric givens not available")
        if self.T1 <= self.T2:
            return (False, "SKIP: T1 <= T2, monotonicity sign may reverse")
        q_expr = self._sympify_heat_loss()
        if q_expr is None:
            return (False, "SKIP: cannot parse heat_loss")
        subs_base = {sp.Symbol('k'): self.k, sp.Symbol('L'): self.L,
                     self._r1: self.r1, self._r2: self.r2,
                     self._T1: self.T1, self._T2: self.T2}
        try:
            q_base = float(q_expr.evalf(subs=subs_base))
        except Exception:
            return (False, "SKIP: numeric evaluation failed")
        # Perturb T1 up by 1%
        T1_pert = self.T1 * 1.01
        subs_pert = subs_base.copy()
        subs_pert[self._T1] = T1_pert
        try:
            q_pert = float(q_expr.evalf(subs=subs_pert))
        except Exception:
            return (False, "SKIP: perturbed evaluation failed")
        if q_pert > q_base:
            return (True, f"q increases when T1 increases: base={q_base}, perturbed={q_pert}")
        else:
            return (False, f"q did not increase: base={q_base}, perturbed={q_pert}")

    def test_monotonicity_q_decreases_with_r2_increase(self) -> Tuple[bool, str]:
        """
        Numerically test that if r2 increases by 1%, q decreases.
        Requires numeric givens with r2 > r1.
        """
        if not self._has_numeric_givens():
            return (False, "SKIP: numeric givens not available")
        if self.r2 <= self.r1:
            return (False, "SKIP: r2 <= r1, not physical")
        q_expr = self._sympify_heat_loss()
        if q_expr is None:
            return (False, "SKIP: cannot parse heat_loss")
        subs_base = {sp.Symbol('k'): self.k, sp.Symbol('L'): self.L,
                     self._r1: self.r1, self._r2: self.r2,
                     self._T1: self.T1, self._T2: self.T2}
        try:
            q_base = float(q_expr.evalf(subs=subs_base))
        except Exception:
            return (False, "SKIP: numeric evaluation failed")
        # Perturb r2 up by 1%
        r2_pert = self.r2 * 1.01
        subs_pert = subs_base.copy()
        subs_pert[self._r2] = r2_pert
        try:
            q_pert = float(q_expr.evalf(subs=subs_pert))
        except Exception:
            return (False, "SKIP: perturbed evaluation failed")
        if q_pert < q_base:
            return (True, f"q decreases when r2 increases: base={q_base}, perturbed={q_pert}")
        else:
            return (False, f"q did not decrease: base={q_base}, perturbed={q_pert}")

    def test_monotonicity_T_r_increases_with_T1(self) -> Tuple[bool, str]:
        """
        Numerically test that if T1 increases by 1%, the temperature T at a fixed
        interior radius increases as well (should be monotonic).
        Requires numeric givens with T1 > T2.
        """
        if not self._has_numeric_givens():
            return (False, "SKIP: numeric givens not available")
        if self.T1 <= self.T2:
            return (False, "SKIP: T1 <= T2, monotonicity sign may reverse")
        T_expr = self._sympify_T_r()
        if T_expr is None:
            return (False, "SKIP: cannot parse T_r expression")
        # Choose a fixed interior radius (midpoint)
        r_mid = 0.5 * (self.r1 + self.r2)
        # Baseline substitution (with original T1)
        subs_base = {self._r: r_mid, self._r1: self.r1, self._r2: self.r2,
                     self._T1: self.T1, self._T2: self.T2}
        try:
            T_base = float(T_expr.evalf(subs=subs_base))
        except Exception:
            return (False, "SKIP: numeric evaluation of baseline T failed")
        # Perturb T1 up by 1%
        T1_pert = self.T1 * 1.01
        subs_pert = subs_base.copy()
        subs_pert[self._T1] = T1_pert
        try:
            T_pert = float(T_expr.evalf(subs=subs_pert))
        except Exception:
            return (False, "SKIP: numeric evaluation of perturbed T failed")
        if T_pert > T_base:
            return (True, f"T increases when T1 increases: base T(r_mid)={T_base}, perturbed T(r_mid)={T_pert}")
        else:
            return (False, f"T did not increase: base T(r_mid)={T_base}, perturbed T(r_mid)={T_pert}")

    # ------------------------------------------------------------------ #
    # Answer sanity (weak, only one allowed)
    # ------------------------------------------------------------------ #

    def test_answer_sanity_heat_loss_non_negative(self) -> Tuple[bool, str]:
        """Weak check: heat_loss expression should not start with a minus sign after equals."""
        if self.heat_loss is None:
            return (False, "SKIP: heat_loss not available")
        expr_clean = self.heat_loss.split('=')[-1].strip()
        if expr_clean.startswith('-'):
            return (False, "heat_loss expression starts with minus")
        # Also check that it contains (T1 - T2) (implies positive for typical T1 > T2)
        if '(T1 - T2)' in self.heat_loss or '(T_1 - T_2)' in self.heat_loss:
            return (True, "heat_loss expression has (T1-T2) numerator, likely positive")
        return (True, "heat_loss expression does not obviously indicate negative value")

    # ------------------------------------------------------------------ #
    # Main runner
    # ------------------------------------------------------------------ #

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