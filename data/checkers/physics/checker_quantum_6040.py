import sympy as sp
import math
import re
from typing import Any, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Physics auditor for the neutral kaon oscillation problem.
    Checks the expression for P0(t) - \bar{P}_0(t) obtained from 
    time evolution of the strangeness eigenstates.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Safely extract the result dictionary
        self.results: Dict[str, Any] = getattr(solution_instance, 'results', {})
        # Store the expression string only; never assign numeric defaults
        self.delta_P: Optional[str] = self.results.get('delta_P', None)

        # Symbolic variables for later derivation
        self.gamma_L, self.gamma_S = sp.symbols('gamma_L gamma_S', real=True, positive=True)
        self.m_L, self.m_S = sp.symbols('m_L m_S', real=True, positive=True)
        self.c, self.hbar, self.t = sp.symbols('c hbar t', real=True, positive=True)
        # Derived quantities
        self.omega_L = self.m_L * self.c**2 / self.hbar
        self.omega_S = self.m_S * self.c**2 / self.hbar

        # Expected expression derived from the physics mapping
        self.expected_expr = sp.exp(-(self.gamma_L + self.gamma_S) * self.t / 2) * \
                             sp.cos((self.m_L - self.m_S) * self.c**2 * self.t / self.hbar)

    def _parse_expr(self, expr_str: str) -> Optional[sp.Expr]:
        """Try to parse a string into a sympy expression.
        Handles common LaTeX and plain text representations.
        Returns None if parsing fails.
        """
        if not isinstance(expr_str, str):
            return None
        s = expr_str.strip().replace('$', '').replace('\\', '')

        # Replace LaTeX commands with plain names
        s = re.sub(r'\bcos\b', 'cos', s)
        s = re.sub(r'\bsin\b', 'sin', s)
        s = re.sub(r'\bexp\b', 'exp', s)
        s = s.replace('gamma_L', 'gamma_L').replace('gamma_S', 'gamma_S')
        s = s.replace('hbar', 'hbar').replace('c', 'c')
        # Remove curly braces
        s = s.replace('{', '').replace('}', '')
        # Convert e^{...} to exp(...)
        s = re.sub(r'e\^([^\s,)]+)', r'exp(\1)', s)
        # Also handle e^{...} with braces already removed
        s = re.sub(r'exp\(([^)]*)\)', r'exp(\1)', s)

        try:
            expr = sp.sympify(s)
            return expr
        except (sp.SympifyError, TypeError, AttributeError):
            return None

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover and run all test_ methods, returning overall success and log."""
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
        return all_pass, logs

    # ------------------------------------------------------------------
    # INTERFACE TESTS
    # ------------------------------------------------------------------
    def test_interface_delta_P_exists(self) -> Tuple[bool, str]:
        """Check that the required key 'delta_P' exists in results and is a string."""
        if self.delta_P is None:
            return (False, "Missing 'delta_P' key in results.")
        if not isinstance(self.delta_P, str):
            return (False, f"delta_P is not a string (type: {type(self.delta_P)}).")
        return (True, "delta_P exists and is a string.")

    # ------------------------------------------------------------------
    # INVARIANT BASED TESTS (theorem‑grounded, use math.isclose or symbolic equality)
    # ------------------------------------------------------------------
    def test_invariant_initial_condition(self) -> Tuple[bool, str]:
        """At t=0, the probability difference must be 1 (initial |K0>)."""
        if self.delta_P is None:
            return (False, "SKIP: delta_P not available.")
        expr = self._parse_expr(self.delta_P)
        if expr is None:
            return (False, "Cannot parse delta_P expression.")
        try:
            val = sp.N(expr.subs({self.t: 0}))
        except Exception as e:
            return (False, f"Substitution failed: {e}")
        if math.isclose(float(val), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            return (True, "delta_P(t=0) = 1 as required by initial condition.")
        else:
            return (False, f"delta_P(t=0) = {val}, expected 1.")

    def test_invariant_symmetry_gamma_exchange(self) -> Tuple[bool, str]:
        """The expression must be symmetric under exchange of gamma_L and gamma_S."""
        if self.delta_P is None:
            return (False, "SKIP: delta_P not available.")
        expr = self._parse_expr(self.delta_P)
        if expr is None:
            return (False, "Cannot parse delta_P expression.")
        swapped = expr.subs({self.gamma_L: self.gamma_S, self.gamma_S: self.gamma_L})
        diff = sp.simplify(expr - swapped)
        if diff == 0:
            return (True, "Expression is symmetric under gamma_L <-> gamma_S.")
        else:
            return (False, f"Asymmetry detected: difference = {diff}")

    def test_invariant_delta_P_real(self) -> Tuple[bool, str]:
        """The probability difference must be purely real."""
        if self.delta_P is None:
            return (False, "SKIP: delta_P not available.")
        expr = self._parse_expr(self.delta_P)
        if expr is None:
            return (False, "Cannot parse delta_P expression.")
        imag_part = sp.im(expr)
        if imag_part == 0:
            return (True, "Expression is real.")
        else:
            return (False, f"Expression has non‑zero imaginary part: {imag_part}")

    # ------------------------------------------------------------------
    # MODELING CONSISTENCY TESTS (trace verification)
    # ------------------------------------------------------------------
    def test_modeling_structure_vs_physics(self) -> Tuple[bool, str]:
        """Confirm that the submitted expression matches the expected formula derived from amplitudes."""
        if self.delta_P is None:
            return (False, "SKIP: delta_P not available.")
        expr = self._parse_expr(self.delta_P)
        if expr is None:
            return (False, "Cannot parse delta_P expression.")
        diff = sp.simplify(expr - self.expected_expr)
        if diff == 0:
            return (True, "Expression matches the derived physical formula.")
        else:
            return (False, f"Expression differs from expected: difference = {diff}")

    def test_modeling_amplitude_form(self) -> Tuple[bool, str]:
        """Check that the expression contains a cosine term and an exponential factor (interference form)."""
        if self.delta_P is None:
            return (False, "SKIP: delta_P not available.")
        expr = self._parse_expr(self.delta_P)
        if expr is None:
            return (False, "Cannot parse delta_P expression.")
        s = self.delta_P.lower()
        has_cos = 'cos' in s or 'cos(' in s
        has_exp = 'exp' in s or 'e^' in s or ('e' in s and '^' in s)
        if not (has_cos and has_exp):
            return (False, "Missing required exponential or cosine component in the expression.")
        return (True, "Expression has the correct functional building blocks.")

    # ------------------------------------------------------------------
    # MONOTONICITY TEST (perturb input and compare baseline vs perturbed)
    # ------------------------------------------------------------------
    def test_monotonicity_small_t_decrease(self) -> Tuple[bool, str]:
        """For very small t, the exponential decay dominates; delta_P should decrease from its initial value."""
        if self.delta_P is None:
            return (False, "SKIP: delta_P not available.")
        expr = self._parse_expr(self.delta_P)
        if expr is None:
            return (False, "Cannot parse delta_P expression.")
        # Baseline: t = 0 -> delta_P = 1
        baseline = 1.0
        # Perturbation: t = epsilon (small positive)
        eps = 0.01
        try:
            # Use symbolic substitution with numeric values for gamma and masses
            # Choose typical values: gamma_L=1, gamma_S=100 (short-lived), m_L-m_S=3.5e-12 eV/c^2? but any positive works.
            # We just need the sign of change; we can evaluate numerically.
            subs_dict = {self.gamma_L: 1.0, self.gamma_S: 100.0, self.m_L: 1.0, self.m_S: 0.5, self.c: 1.0, self.hbar: 1.0}
            t0 = 0.0
            t1 = eps
            delta_P_0 = float(expr.subs({**subs_dict, self.t: t0}))
            delta_P_1 = float(expr.subs({**subs_dict, self.t: t1}))
        except Exception as e:
            return (False, f"Numeric evaluation failed: {e}")
        # Expect decreasing: delta_P_1 < delta_P_0
        if delta_P_1 < delta_P_0:
            return (True, f"For small t perturbation (t={eps}), delta_P decreases from {delta_P_0:.6f} to {delta_P_1:.6f}.")
        else:
            return (False, f"Expected decrease but got: baseline={delta_P_0:.6f}, perturbed={delta_P_1:.6f}")

    # ------------------------------------------------------------------
    # WEAK ANSWER SANITY (only one minimal check allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_variable_presence(self) -> Tuple[bool, str]:
        """Weak sanity: check that the string contains all the key variables and constants."""
        if self.delta_P is None:
            return (False, "SKIP: delta_P not available.")
        s = self.delta_P
        required = ['gamma_L', 'gamma_S', 'm_L', 'm_S', 'c', 'hbar']
        missing = []
        for var in required:
            pattern = var.replace('_', r'[_ ]?')
            if not re.search(pattern, s, re.IGNORECASE):
                missing.append(var)
        if not re.search(r'cos|Cos', s, re.IGNORECASE):
            missing.append('cos')
        if not re.search(r'exp|e\^', s, re.IGNORECASE):
            missing.append('exp')
        if 't' not in s.replace(' ', ''):
            missing.append('t')
        if missing:
            return (False, f"Missing essential components: {missing}")
        return (True, "All necessary variables and operators are present in the expression.")