import re
import math
import traceback
from typing import Any, Tuple, List
import sympy as sp

class PhysicsChecker:
    """
    Audits the physical correctness and consistency of the solution for the
    magnetic field problem defined in the problem statement.
    Includes interface, invariant, modeling/trace, and monotonicity tests.
    """

    def __init__(self, solution_instance: Any) -> None:
        """
        Extract relevant data from the solution instance safely.
        All missing fields default to None to avoid silent numerical defaults.
        """
        # Extract the results dictionary
        results = getattr(solution_instance, 'results', None)
        if results is None:
            results = {}

        # Extract the four expected result keys (symbolic strings)
        self.field_line_eq = results.get('field_line_eq', None)
        self.flux_per_length = results.get('flux_per_length', None)
        self.phi = results.get('phi', None)
        self.E = results.get('E', None)

        # Optional givens (may be used in monotonicity or modeling checks)
        self.B0 = getattr(solution_instance, 'B0', None)
        self.a = getattr(solution_instance, 'a', None)
        self.v = getattr(solution_instance, 'v', None)
        self.R = getattr(solution_instance, 'R', None)
        self.dB0_dt = getattr(solution_instance, 'dB0_dt', None)
        # Optional test point coordinates for phi/E
        self.x = getattr(solution_instance, 'x', None)
        self.y = getattr(solution_instance, 'y', None)

    # ------------------------------------------------------------------
    # Helper: safely convert a symbolic expression string to a sympy expression
    # ------------------------------------------------------------------
    def _parse_expr(self, expr_str: str) -> sp.Expr:
        """Convert a string like 'B0*R^2/(2*a)' to a sympy expression.
        Replace common LaTeX-style notation: ^ -> **, and handle \frac.
        """
        if expr_str is None:
            return None
        # Replace LaTeX fractions: \frac{num}{den} -> (num)/(den)
        expr_str = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', expr_str)
        # Replace ^ with **
        expr_str = expr_str.replace('^', '**')
        # Replace subscript: B_0 -> B0
        expr_str = re.sub(r'([A-Za-z])_(\d+)', r'\1\2', expr_str)
        # Replace \hat{z} or similar with nothing (ignore unit vectors)
        expr_str = re.sub(r'\\hat\{z\}', '', expr_str)
        expr_str = re.sub(r'\s+', '', expr_str)
        # If expression contains \\dot, handle as derivative symbol
        expr_str = expr_str.replace(r'\dot{B}_0(t)', 'dB0_dt')
        expr_str = expr_str.replace('dB0/dt', 'dB0_dt')
        # Replace other common patterns
        expr_str = expr_str.replace(r'\mathbf{E}', 'E')
        try:
            return sp.sympify(expr_str)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Interface Tests
    # ------------------------------------------------------------------
    def test_interface_field_line_eq(self) -> Tuple[bool, str]:
        """Check that field_line_eq is a non-empty string."""
        if self.field_line_eq is None:
            return (False, "field_line_eq is missing (None)")
        if not isinstance(self.field_line_eq, str) or len(self.field_line_eq) == 0:
            return (False, "field_line_eq is not a non-empty string")
        return (True, "field_line_eq present and is a string")

    def test_interface_flux_per_length(self) -> Tuple[bool, str]:
        """Check that flux_per_length is a non-empty string."""
        if self.flux_per_length is None:
            return (False, "flux_per_length is missing (None)")
        if not isinstance(self.flux_per_length, str) or len(self.flux_per_length) == 0:
            return (False, "flux_per_length is not a non-empty string")
        return (True, "flux_per_length present and is a string")

    def test_interface_phi(self) -> Tuple[bool, str]:
        """Check that phi is a non-empty string."""
        if self.phi is None:
            return (False, "phi is missing (None)")
        if not isinstance(self.phi, str) or len(self.phi) == 0:
            return (False, "phi is not a non-empty string")
        return (True, "phi present and is a string")

    def test_interface_E(self) -> Tuple[bool, str]:
        """Check that E is a non-empty string."""
        if self.E is None:
            return (False, "E is missing (None)")
        if not isinstance(self.E, str) or len(self.E) == 0:
            return (False, "E is not a non-empty string")
        return (True, "E present and is a string")

    # ------------------------------------------------------------------
    # Invariant Tests
    # ------------------------------------------------------------------
    def test_invariant_field_line_eq_form(self) -> Tuple[bool, str]:
        """Field line equation must be of the form 'xy = const'."""
        if self.field_line_eq is None:
            return (False, "Skip: field_line_eq not available")
        pattern = r'xy\s*=\s*const(ant)?'
        if re.search(pattern, self.field_line_eq, re.IGNORECASE):
            return (True, "Field line equation matches xy = const")
        else:
            return (False, f"Field line equation '{self.field_line_eq}' does not match 'xy = const'")

    def test_invariant_flux_per_length_form(self) -> Tuple[bool, str]:
        """Flux per length must contain B0, R^2, and 2a."""
        if self.flux_per_length is None:
            return (False, "Skip: flux_per_length not available")
        s = self.flux_per_length
        has_B0 = 'B0' in s or 'B_0' in s
        has_R2 = 'R^2' in s
        has_2a = '2a' in s
        has_div = '/' in s or '\\frac' in s
        if has_B0 and has_R2 and (has_2a or has_div):
            return (True, "Flux per length contains B0, R^2, and 2a")
        else:
            return (False, f"Flux per length '{s}' does not match expected form")

    def test_invariant_phi_form(self) -> Tuple[bool, str]:
        """Potential phi must contain a minus sign, B0, v, xy, and a."""
        if self.phi is None:
            return (False, "Skip: phi not available")
        s = self.phi
        has_minus = '-' in s
        has_B0 = 'B0' in s or 'B_0' in s
        has_v = 'v' in s
        has_xy = 'xy' in s
        has_a = 'a' in s
        if has_minus and has_B0 and has_v and has_xy and has_a:
            return (True, "phi contains minus sign, B0, v, xy, a")
        else:
            return (False, f"phi '{s}' is missing required elements")

    def test_invariant_E_form(self) -> Tuple[bool, str]:
        """Induced E must contain dB0/dt, xy, a, and z direction."""
        if self.E is None:
            return (False, "Skip: E not available")
        s = self.E
        has_dB0_dt = 'dB0/dt' in s or r'\dot{B}_0' in s
        has_xy = 'xy' in s
        has_a = 'a' in s
        has_z = 'z' in s.lower() or '\\hat{z}' in s or 'e_z' in s
        if has_dB0_dt and has_xy and has_a and has_z:
            return (True, "E contains dB0/dt, xy, a, and z direction")
        else:
            return (False, f"E '{s}' is missing required elements")

    # ------------------------------------------------------------------
    # Modeling / Trace Tests
    # ------------------------------------------------------------------
    def test_modeling_flux_numerical(self) -> Tuple[bool, str]:
        """
        If B0, a, R are provided as numbers, compute flux = B0*R^2/(2*a)
        and compare with the stored flux_per_length expression evaluated
        at the same values.
        """
        if self.flux_per_length is None:
            return (False, "Skip: flux_per_length not available")
        if None in (self.B0, self.a, self.R):
            return (False, "Skip: B0, a, or R not available for numerical check")
        try:
            # Compute expected flux
            expected = (self.B0 * self.R**2) / (2 * self.a)
            # Parse stored expression
            expr = self._parse_expr(self.flux_per_length)
            if expr is None:
                return (False, "Could not parse flux_per_length expression")
            # Substitute numerical values
            subs_dict = {'B0': self.B0, 'a': self.a, 'R': self.R}
            stored_val = expr.subs(subs_dict).evalf()
            if math.isclose(float(stored_val), expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Flux numerical check passed: expected {expected:.4g}, stored gives {float(stored_val):.4g}")
            else:
                return (False, f"Flux numerical mismatch: expected {expected:.4g}, stored gives {float(stored_val):.4g}")
        except Exception as e:
            return (False, f"Flux numerical check crashed: {e}")

    def test_modeling_phi_numerical(self) -> Tuple[bool, str]:
        """
        If B0, v, a, x, y are provided, compute phi = -B0*v*x*y/a and
        compare with the stored phi expression evaluated at the same values.
        """
        if self.phi is None:
            return (False, "Skip: phi not available")
        if None in (self.B0, self.v, self.a, self.x, self.y):
            return (False, "Skip: B0, v, a, x, or y not available for numerical check")
        try:
            expected = - (self.B0 * self.v * self.x * self.y) / self.a
            expr = self._parse_expr(self.phi)
            if expr is None:
                return (False, "Could not parse phi expression")
            subs_dict = {'B0': self.B0, 'v': self.v, 'a': self.a, 'x': self.x, 'y': self.y}
            stored_val = expr.subs(subs_dict).evalf()
            if math.isclose(float(stored_val), expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"Phi numerical check passed: expected {expected:.4g}, stored gives {float(stored_val):.4g}")
            else:
                return (False, f"Phi numerical mismatch: expected {expected:.4g}, stored gives {float(stored_val):.4g}")
        except Exception as e:
            return (False, f"Phi numerical check crashed: {e}")

    def test_modeling_E_numerical(self) -> Tuple[bool, str]:
        """
        If dB0_dt, a, x, y are provided, compute E_z = - (dB0_dt)*x*y/a
        and compare with the stored E expression evaluated at the same values.
        """
        if self.E is None:
            return (False, "Skip: E not available")
        if None in (self.dB0_dt, self.a, self.x, self.y):
            return (False, "Skip: dB0_dt, a, x, or y not available for numerical check")
        try:
            expected = - (self.dB0_dt * self.x * self.y) / self.a
            expr = self._parse_expr(self.E)
            if expr is None:
                return (False, "Could not parse E expression")
            # The expression may contain 'z_hat' and 'dB0/dt' symbol
            subs_dict = {'dB0_dt': self.dB0_dt, 'a': self.a, 'x': self.x, 'y': self.y}
            stored_val = expr.subs(subs_dict).evalf()
            if math.isclose(float(stored_val), expected, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"E numerical check passed: expected {expected:.4g}, stored gives {float(stored_val):.4g}")
            else:
                return (False, f"E numerical mismatch: expected {expected:.4g}, stored gives {float(stored_val):.4g}")
        except Exception as e:
            return (False, f"E numerical check crashed: {e}")

    # ------------------------------------------------------------------
    # Monotonicity Tests
    # ------------------------------------------------------------------
    def test_monotonicity_flux_increases_with_R(self) -> Tuple[bool, str]:
        """
        If B0, a, R are provided, compute flux at R and at 1.1*R (perturbed).
        Verify that flux increases.
        """
        if None in (self.B0, self.a, self.R):
            return (False, "Skip: B0, a, or R not available for monotonicity check")
        try:
            flux_base = (self.B0 * self.R**2) / (2 * self.a)
            R_perturbed = 1.1 * self.R
            flux_perturbed = (self.B0 * R_perturbed**2) / (2 * self.a)
            if flux_perturbed > flux_base:
                return (True, f"Flux increases with R: base {flux_base:.4g}, perturbed {flux_perturbed:.4g}")
            else:
                return (False, f"Flux does not increase: base {flux_base:.4g}, perturbed {flux_perturbed:.4g}")
        except Exception as e:
            return (False, f"Monotonicity flux test crashed: {e}")

    def test_monotonicity_phi_becomes_more_negative_with_xy(self) -> Tuple[bool, str]:
        """
        If B0, v, a, x, y are provided, compute phi at (x,y) and at (1.1*x, y).
        Since phi = -B0*v*x*y/a, increasing x should make phi more negative (decrease).
        """
        if None in (self.B0, self.v, self.a, self.x, self.y):
            return (False, "Skip: B0, v, a, x, or y not available for monotonicity check")
        try:
            phi_base = - (self.B0 * self.v * self.x * self.y) / self.a
            x_perturbed = 1.1 * self.x
            phi_perturbed = - (self.B0 * self.v * x_perturbed * self.y) / self.a
            if phi_perturbed < phi_base:
                return (True, f"Phi becomes more negative: base {phi_base:.4g}, perturbed {phi_perturbed:.4g}")
            else:
                return (False, f"Phi does not become more negative: base {phi_base:.4g}, perturbed {phi_perturbed:.4g}")
        except Exception as e:
            return (False, f"Monotonicity phi test crashed: {e}")

    # ------------------------------------------------------------------
    # Weak Answer Sanity (at most one)
    # ------------------------------------------------------------------
    def test_answer_sanity_phi_negative_sign(self) -> Tuple[bool, str]:
        """
        Very weak sanity: the phi expression should contain a minus sign,
        consistent with a negative potential for positive B0, v, x, y.
        (No numerical re-derivation, just pattern check.)
        """
        if self.phi is None:
            return (False, "Skip: phi not available")
        if '-' in self.phi:
            return (True, "phi contains a minus sign (as expected)")
        else:
            return (False, "phi does not contain a minus sign")

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Discover and run all test_* methods, collect results.
        Returns (all_pass, list_of_logs).
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)