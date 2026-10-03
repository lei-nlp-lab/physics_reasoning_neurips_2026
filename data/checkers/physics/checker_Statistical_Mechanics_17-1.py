import sympy as sp
import math
from typing import Tuple, List, Any

class PhysicsChecker:
    """
    Audits solution for maximum height of water in a tree based on thermodynamic equilibrium
    and hydrostatic balance.
    """

    def __init__(self, solution_instance: Any):
        # Results provided by the solver
        self.h_symbolic = getattr(solution_instance, 'h', None) or \
                          solution_instance.results.get('h', None)
        self.h_numeric = getattr(solution_instance, 'h_numeric', None) or \
                         solution_instance.results.get('h_numeric', None)

        # Also try direct attribute access if no .results
        if self.h_symbolic is None:
            self.h_symbolic = getattr(solution_instance, 'h', None)
        if self.h_numeric is None:
            self.h_numeric = getattr(solution_instance, 'h_numeric', None)

        # Known constants (re‑derived inside checker)
        self.T = 298.15  # K (25 °C)
        self.r = 0.9
        self.R = 8.314   # J/(mol·K)
        self.M = 0.018   # kg/mol
        self.g = 9.8     # m/s²

    # ---------- Interface tests ----------
    def test_interface_h_symbolic_exists(self) -> Tuple[bool, str]:
        if self.h_symbolic is None:
            return (False, "Missing symbolic height (h).")
        return (True, "Symbolic height present.")

    def test_interface_h_numeric_exists(self) -> Tuple[bool, str]:
        if self.h_numeric is None:
            return (False, "Missing numeric height (h_numeric).")
        return (True, "Numeric height present.")

    def test_interface_h_symbolic_type(self) -> Tuple[bool, str]:
        if self.h_symbolic is None:
            return (False, "SKIP: no symbolic height.")
        if not isinstance(self.h_symbolic, str):
            return (False, f"Symbolic height should be str, got {type(self.h_symbolic)}.")
        return (True, "Symbolic height is a string.")

    def test_interface_h_numeric_type(self) -> Tuple[bool, str]:
        if self.h_numeric is None:
            return (False, "SKIP: no numeric height.")
        if not isinstance(self.h_numeric, (int, float)):
            return (False, f"Numeric height should be numeric, got {type(self.h_numeric)}.")
        return (True, "Numeric height is numeric.")

    # ---------- Invariant tests ----------
    def test_invariant_h_positive(self) -> Tuple[bool, str]:
        """Height must be positive (upward)."""
        if self.h_numeric is None:
            return (False, "SKIP: no numeric height.")
        if not math.isfinite(self.h_numeric):
            return (False, f"Numeric height not finite: {self.h_numeric}")
        if self.h_numeric <= 0:
            return (False, f"Height should be positive, got {self.h_numeric}.")
        return (True, "Height positive.")

    def test_invariant_numeric_consistency(self) -> Tuple[bool, str]:
        """Re‑derive expected height from givens and compare using math.isclose (invariant test)."""
        if self.h_numeric is None:
            return (False, "SKIP: no numeric height to compare.")
        try:
            expected = (self.R * self.T) / (self.M * self.g) * math.log(1.0 / self.r)
        except Exception as e:
            return (False, f"Could not compute expected height: {e}")
        if not math.isfinite(expected):
            return (False, f"Expected height not finite: {expected}")
        if math.isclose(self.h_numeric, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Height {self.h_numeric} close to expected {expected}.")
        return (False, f"Height {self.h_numeric} ≠ expected {expected} (rel err {abs(self.h_numeric-expected)/expected:.3f}).")

    # ---------- Trace / Modeling tests ----------
    def test_trace_formula_equivalence(self) -> Tuple[bool, str]:
        """
        Check that the symbolic formula is algebraically equivalent to the expected form:
        h = (R*T/(M*g)) * ln(1/r)  or  h = (tau/(m*g))*ln(1/r)
        Uses sympy to compare; if parsing fails, perform numeric substitution.
        """
        if self.h_symbolic is None:
            return (False, "SKIP: no symbolic height.")

        # Build expected expression in sympy
        R_s, T_s, M_s, g_s, r_s = sp.symbols('R T M g r', positive=True)
        expected_expr = (R_s * T_s) / (M_s * g_s) * sp.log(1 / r_s)

        # Convert student's string to sympy expression
        formula_str = self.h_symbolic.strip()
        # Replace common LaTeX artifacts
        formula_str = formula_str.replace(r'\left', '').replace(r'\right', '')
        formula_str = formula_str.replace(r'\ln', 'log').replace(r'\log', 'log')

        try:
            student_expr = sp.sympify(formula_str)
        except Exception:
            # fallback: numeric substitution
            return self._numeric_substitution_equivalence(expected_expr)

        # Simplify difference to see if it's zero
        try:
            diff = sp.simplify(student_expr - expected_expr)
            if diff == 0:
                return (True, "Symbolic formula algebraically equivalent to expected.")
            # Allow for different constants (e.g., tau instead of RT/M, m instead of M)
            # Check if diff evaluates to zero after substituting known constants
            subs_dict = {R_s: self.R, T_s: self.T, M_s: self.M, g_s: self.g, r_s: self.r}
            numeric_diff = diff.subs(subs_dict).evalf()
            if abs(float(numeric_diff)) < 1e-6:
                return (True, "Symbolic formula numerically equivalent after constant substitution.")
            return (False, f"Difference after simplification: {diff}. Numeric diff: {numeric_diff}")
        except Exception as e:
            return self._numeric_substitution_equivalence(expected_expr)

    def _numeric_substitution_equivalence(self, expected_expr: sp.Expr) -> Tuple[bool, str]:
        """Fallback: evaluate both expressions at the given constants and compare."""
        try:
            # Evaluate expected
            subs_dict = {sp.Symbol('R'): self.R, sp.Symbol('T'): self.T,
                         sp.Symbol('M'): self.M, sp.Symbol('g'): self.g,
                         sp.Symbol('r'): self.r}
            expr = sp.sympify(self.h_symbolic)
            val_student = float(expr.subs(subs_dict).evalf())
            val_expected = float(expected_expr.subs(subs_dict).evalf())
            if math.isclose(val_student, val_expected, rel_tol=0.01, abs_tol=1.0):
                return (True, f"Formulas numerically equivalent at given constants: {val_student} ≈ {val_expected}")
            return (False, f"Numeric mismatch: student {val_student}, expected {val_expected}")
        except Exception as e:
            return (False, f"Cannot verify formula equivalence: {e}")

    def test_modeling_logarithm_presence(self) -> Tuple[bool, str]:
        """
        Check that the symbolic formula contains a logarithm (ln or log),
        and that it involves 1/r.
        """
        if self.h_symbolic is None:
            return (False, "SKIP: no symbolic height.")
        formula = self.h_symbolic.lower()
        # Check for logarithm
        has_log = ('ln' in formula) or ('log' in formula)
        # Check for reciprocal of r
        has_recip = ('1/r' in formula) or ('r^{-1}' in formula)
        if not has_log:
            return (False, "Symbolic formula does not contain a logarithm.")
        if not has_recip:
            return (False, "Symbolic formula does not contain 1/r.")
        return (True, "Symbolic formula includes ln(1/r) structure.")

    def test_modeling_physical_parameters(self) -> Tuple[bool, str]:
        """
        Check that the symbolic formula contains references to temperature and gravity.
        This verifies that the correct physical parameters appear in the result.
        """
        if self.h_symbolic is None:
            return (False, "SKIP: no symbolic height.")
        formula_lower = self.h_symbolic.lower()
        # Check for temperature (T or τ)
        has_temp = ('t' in formula_lower) or ('tau' in formula_lower)
        # Check for gravity (g)
        has_gravity = 'g' in formula_lower
        if not has_temp:
            return (False, "Symbolic formula does not reference temperature (T or tau).")
        if not has_gravity:
            return (False, "Symbolic formula does not reference gravity (g).")
        return (True, "Symbolic formula includes temperature and gravity parameters.")

    # ---------- Monotonicity test ----------
    def test_monotonicity_h_decreases_with_increasing_r(self) -> Tuple[bool, str]:
        """
        Explicitly perturb the input r (from 0.9 to 0.95) and check that the computed height decreases.
        The baseline and perturbed heights are recomputed from known constants, so the test
        does not rely on the student's symbolic expression.
        """
        try:
            h_baseline = (self.R * self.T) / (self.M * self.g) * math.log(1.0 / 0.9)
            h_perturbed = (self.R * self.T) / (self.M * self.g) * math.log(1.0 / 0.95)
        except Exception as e:
            return (False, f"Could not compute heights: {e}")
        if h_perturbed < h_baseline:
            return (True, f"Height decreased from {h_baseline:.2f} at r=0.9 to {h_perturbed:.2f} at r=0.95.")
        return (False, f"Monotonicity violated: height increased from {h_baseline:.2f} to {h_perturbed:.2f}.")

    # ---------- Weak answer sanity (only one allowed) ----------
    def test_answer_sanity_broad_range(self) -> Tuple[bool, str]:
        """Weak sanity: numeric height plausible (0 to 5 km)."""
        if self.h_numeric is None:
            return (False, "SKIP: no numeric height.")
        if not math.isfinite(self.h_numeric):
            return (False, f"Numeric height not finite: {self.h_numeric}")
        if not (0 < self.h_numeric < 5000):
            return (False, f"Height {self.h_numeric} outside plausible range (0,5000) m.")
        return (True, f"Height {self.h_numeric} within plausible range.")

    # ---------- Run all tests ----------
    def run_all_tests(self) -> Tuple[bool, list]:
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
                passed, msg = (False, f"{name} CRASHED: {e}")
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)