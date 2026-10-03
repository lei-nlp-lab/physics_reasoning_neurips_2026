import sympy as sp
import math
import traceback
import re

class PhysicsChecker:
    """
    Auditor for the Thomas-Fermi model derivation.
    Checks that the three result equations (p_F_sq, rho, pde) are present,
    have the correct algebraic form, and satisfy consistency conditions.
    """

    def __init__(self, solution_instance):
        # Extract results dictionary safely
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            try:
                self.results = solution_instance.results
            except AttributeError:
                self.results = None

        # Extract expected result keys with None defaults
        self.p_F_sq = self.results.get('p_F_sq', None) if self.results else None
        self.rho = self.results.get('rho', None) if self.results else None
        self.pde = self.results.get('pde', None) if self.results else None

        # Explanation steps (optional, for context)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ----------------------------------------------------------------------
    # Helper: try to parse a simple equation string into a sympy expression
    # for the right-hand side. Assumes form "X = expression"
    # ----------------------------------------------------------------------
    @staticmethod
    def _extract_rhs(equation_str):
        """Extract the right-hand side of a simple equation (e.g., 'p_F^2 = 2me φ')."""
        if '=' not in equation_str:
            return equation_str
        parts = equation_str.split('=')
        if len(parts) == 2:
            return parts[1].strip()
        else:
            return equation_str

    @staticmethod
    def _parse_expression(expr_str, subs_dict=None):
        """
        Convert a simple LaTeX-like expression to a sympy expression.
        Handles common notations: φ, phi, ^, *, /, constants.
        """
        s = expr_str.replace('{', '').replace('}', '')
        s = s.replace('\\phi', 'phi')
        s = s.replace('φ', 'phi')
        s = s.replace('π', 'pi')
        s = s.replace('\\pi', 'pi')
        s = s.replace('\\hbar', 'hbar')
        s = s.replace('\\hbar', 'hbar')
        s = s.replace('^', '**')
        s = s.replace(' ', '')
        # Replace numbers/constants: e, m, hbar, pi, phi, etc.
        # Convert to sympy symbolic variable accordingly
        try:
            expr = sp.sympify(s, locals={'e': sp.symbols('e'), 'm': sp.symbols('m'),
                                         'hbar': sp.symbols('hbar'), 'pi': sp.pi,
                                         'phi': sp.symbols('phi')})
            if subs_dict:
                expr = expr.subs(subs_dict)
            return expr
        except Exception:
            return None

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        if self.results is None:
            return (False, "SKIP: No results dictionary found.")
        missing = []
        if self.p_F_sq is None:
            missing.append('p_F_sq')
        else:
            if not isinstance(self.p_F_sq, str):
                return (False, "p_F_sq is not a string.")
        if self.rho is None:
            missing.append('rho')
        else:
            if not isinstance(self.rho, str):
                return (False, "rho is not a string.")
        if self.pde is None:
            missing.append('pde')
        else:
            if not isinstance(self.pde, str):
                return (False, "pde is not a string.")
        if missing:
            return (False, f"Missing result keys: {', '.join(missing)}")
        return (True, "All required result keys present and are strings.")

    # ----------------------------------------------------------------------
    # Invariant checks
    # ----------------------------------------------------------------------
    def test_invariant_fermi_energy_equation(self):
        if self.p_F_sq is None:
            return (False, "SKIP: p_F_sq not available.")
        s = self.p_F_sq.replace(" ", "").lower()
        if "2me" not in s:
            return (False, "p_F_sq does not contain '2me' (expected 2 m e φ).")
        if "φ" not in s and "phi" not in s:
            return (False, "p_F_sq does not contain φ or phi.")
        return (True, "p_F_sq contains the expected Fermi energy relation.")

    def test_invariant_rho_form(self):
        if self.rho is None:
            return (False, "SKIP: rho not available.")
        s = self.rho.replace(" ", "").replace("\\", "").lower()
        if "e/(3π^2ħ^3)" not in s and "e/(3pi^2hbar^3)" not in s:
            return (False, "rho does not contain the expected prefactor e/(3π^2 ħ^3).")
        if "φ^{3/2}" not in s and "phi^{3/2}" not in s and "^{3/2}" not in s:
            # also check for (2meφ)^{3/2}
            if "2me" not in s or ")^{3/2}" not in s:
                return (False, "rho does not contain the (2me φ)^{3/2} factor.")
        return (True, "rho has the correct functional dependence on φ.")

    def test_invariant_pde_form(self):
        if self.pde is None:
            return (False, "SKIP: pde not available.")
        s = self.pde.replace(" ", "").replace("\\", "").lower()
        if "1/r" not in s and "d^2/dr^2" not in s:
            return (False, "PDE does not contain a spherical Laplacian (1/r d^2/dr^2 (r φ)).")
        if "4e/(3πħ^3)" not in s and "4e/(3pi hbar^3)" not in s:
            return (False, "PDE does not contain the factor 4e/(3π ħ^3).")
        if "2me" not in s or "^{3/2}" not in s:
            return (False, "PDE does not contain (2me φ)^{3/2}.")
        return (True, "PDE has the expected spherical symmetric form.")

    # ----------------------------------------------------------------------
    # Modeling/trace checks
    # ----------------------------------------------------------------------
    def test_modeling_rho_power_three_halves(self):
        if self.rho is None:
            return (False, "SKIP: rho not available.")
        if "^{3/2}" not in self.rho and "^{1.5}" not in self.rho:
            return (False, "rho does not show the required exponent 3/2 on φ.")
        return (True, "rho has exponent 3/2 on the potential.")

    def test_modeling_spherical_symmetry_in_pde(self):
        if self.pde is None:
            return (False, "SKIP: pde not available.")
        s = self.pde.replace(" ", "").lower()
        if "1/r" not in s or "d^2/dr^2" not in s:
            return (False, "PDE does not contain the spherical Laplacian operator.")
        return (True, "PDE is correctly written in spherical coordinates.")

    def test_modeling_rho_proportional_to_phi_three_halves(self):
        """Check that rho expression is indeed proportional to φ^{3/2}, not something else."""
        if self.rho is None:
            return (False, "SKIP: rho not available.")
        # Extract RHS and check proportionality
        rhs = self._extract_rhs(self.rho)
        expr = self._parse_expression(rhs)
        if expr is None:
            return (False, "Could not parse rho expression.")
        phi = sp.symbols('phi')
        # See if expr can be written as C * phi**(3/2)
        try:
            # isolate φ
            factor = sp.simplify(expr / phi**(sp.Rational(3,2)))
            # factor should not depend on φ symbolically
            if factor.has(phi):
                return (False, "rho expression is not proportional to φ^{3/2}.")
            return (True, "rho ∝ φ^{3/2} is satisfied.")
        except Exception:
            return (False, "Could not verify proportionality.")

    # ----------------------------------------------------------------------
    # Monotonicity check: verify p_F_sq increases with φ
    # ----------------------------------------------------------------------
    def test_monotonicity_p_F_sq_increases_with_phi(self):
        """
        Choose φ as input, perturb it by +5%, and check that p_F_sq ( = 2me φ ) increases.
        Uses numeric evaluation with dummy values e=1, m=1.
        """
        if self.p_F_sq is None:
            return (False, "SKIP: p_F_sq not available.")
        # Assume p_F_sq string is something like "p_F^2 = 2me φ(x)"
        rhs = self._extract_rhs(self.p_F_sq)
        # Replace symbols with numeric placeholders: e=1, m=1, φ variable.
        s = rhs.replace('φ', 'phi').replace('\\phi', 'phi')
        # Try to evaluate using sympy
        phi = sp.symbols('phi')
        try:
            expr = sp.sympify(s, locals={'e': 1, 'm': 1, 'hbar': 1, 'pi': sp.pi,
                                         'phi': phi, 'hbar': 1})
        except Exception:
            return (False, "Could not parse p_F_sq numeric expression.")
        # Define baseline φ = 1.0
        base = phi.subs(phi, 1.0)
        # Perturbed φ = 1.05 (+5%)
        perturbed = phi.subs(phi, 1.05)
        try:
            f_base = float(expr.subs(phi, base))
            f_pert = float(expr.subs(phi, perturbed))
        except Exception:
            return (False, "Numeric evaluation failed.")
        if f_pert > f_base:
            return (True, f"p_F_sq increases when φ increases: {f_base:.3f} -> {f_pert:.3f}.")
        else:
            return (False, f"p_F_sq does not increase: {f_base:.3f} -> {f_pert:.3f}.")

    # ----------------------------------------------------------------------
    # Weak answer sanity (sign / basic structure)
    # ----------------------------------------------------------------------
    def test_answer_sanity_signs(self):
        if self.rho is not None:
            s = self.rho.replace(" ", "").replace("-", "")
            if '=' in s:
                s = s.split('=')[-1].strip()
            if s.startswith("-"):
                return (False, "rho has a leading minus sign, but should be positive.")
        if self.pde is not None:
            # Check RHS for unexpected minus
            s = self.pde.replace(" ", "")
            if '=' in s:
                rhs = s.split('=')[-1].strip()
                if rhs.startswith("-"):
                    return (False, "PDE RHS starts with a minus sign (expected positive).")
        return (True, "Signs appear consistent.")

    # ----------------------------------------------------------------------
    # Main test runner
    # ----------------------------------------------------------------------
    def run_all_tests(self):
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