import math
import sympy as sp
import traceback
from typing import Any, Dict, Optional

class PhysicsChecker:
    """
    Checker for the quantum scattering problem:
    low-energy electron scattering by a neutral atom (kR << 1).
    Audits emitted results: psi_asymptotic, diff_cross_section, sigma_max.
    """

    def __init__(self, solution_instance: Any):
        """
        Extract required data from solution_instance.
        Use None defaults for missing attributes/keys.
        """
        # Givens (optional, for numeric checks)
        self.k = getattr(solution_instance, 'k', None)
        self.R = getattr(solution_instance, 'R', None)
        # Also try results dict
        if self.k is None and hasattr(solution_instance, 'results'):
            self.k = solution_instance.results.get('k', None)
        if self.R is None and hasattr(solution_instance, 'results'):
            self.R = solution_instance.results.get('R', None)

        # Emitted results (according to pseudocode contract)
        self.psi_asymptotic = None
        self.diff_cross_section = None
        self.sigma_max = None

        if hasattr(solution_instance, 'results'):
            self.psi_asymptotic = solution_instance.results.get('psi_asymptotic', None)
            self.diff_cross_section = solution_instance.results.get('diff_cross_section', None)
            self.sigma_max = solution_instance.results.get('sigma_max', None)
        else:
            # fallback to direct attributes
            self.psi_asymptotic = getattr(solution_instance, 'psi_asymptotic', None)
            self.diff_cross_section = getattr(solution_instance, 'diff_cross_section', None)
            self.sigma_max = getattr(solution_instance, 'sigma_max', None)

        # For symbolic checks, store sympy expressions if possible
        self.psi_asymptotic_expr = self._parse_expr(self.psi_asymptotic)
        self.diff_cross_section_expr = self._parse_expr(self.diff_cross_section)
        self.sigma_max_expr = self._parse_expr(self.sigma_max)

    def _parse_expr(self, value: Any) -> Optional[sp.Expr]:
        """Try to convert a string or sympy expression to sympy expression."""
        if value is None:
            return None
        if isinstance(value, sp.Expr):
            return value
        if isinstance(value, str):
            try:
                return sp.sympify(value)
            except (sp.SympifyError, TypeError):
                return None
        return None

    def _has_symbolic_variables(self, expr: sp.Expr, symbols: set) -> bool:
        """Check if all required symbols are present in expression."""
        if expr is None:
            return False
        free_symbols = set(expr.free_symbols)
        return symbols.issubset(free_symbols)

    # ---------- Interface Tests ----------
    def test_interface_psi_asymptotic_present(self):
        if self.psi_asymptotic is None:
            return (False, "psi_asymptotic is missing")
        if not isinstance(self.psi_asymptotic, (str, sp.Expr)):
            return (False, "psi_asymptotic is not a string or sympy expression")
        return (True, "psi_asymptotic present")

    def test_interface_diff_cross_section_present(self):
        if self.diff_cross_section is None:
            return (False, "diff_cross_section is missing")
        if not isinstance(self.diff_cross_section, (str, sp.Expr)):
            return (False, "diff_cross_section is not a string or sympy expression")
        return (True, "diff_cross_section present")

    def test_interface_sigma_max_present(self):
        if self.sigma_max is None:
            return (False, "sigma_max is missing")
        if not isinstance(self.sigma_max, (str, sp.Expr)):
            return (False, "sigma_max is not a string or sympy expression")
        return (True, "sigma_max present")

    def test_interface_kR_product_small(self):
        if self.k is None or self.R is None:
            return (False, "SKIP: k or R not available")
        if not isinstance(self.k, (int, float)) or not isinstance(self.R, (int, float)):
            return (False, "SKIP: k or R not numeric")
        kR = self.k * self.R
        if kR >= 1:
            return (False, f"kR = {kR} is not << 1 (kR < 1 expected)")
        if kR <= 0:
            return (False, f"kR = {kR} is not positive")
        return (True, f"kR = {kR} is small (<1)")

    # ---------- Invariant / Modeling Tests (Symbolic) ----------
    def test_invariant_psi_asymptotic_form(self):
        """
        Check that psi_asymptotic is of the form sin(kr+delta)/(kr)
        or an equivalent expression.
        """
        expr = self.psi_asymptotic_expr
        if expr is None:
            return (False, "SKIP: cannot parse psi_asymptotic to sympy")
        k, delta, r = sp.symbols('k delta r', real=True, positive=True)
        target = sp.sin(k*r + delta) / (k*r)
        try:
            diff = sp.simplify(expr - target)
            if diff == 0:
                return (True, "psi_asymptotic matches sin(kr+delta)/(kr)")
            # Check if numerator has sin and denominator has k*r structure
            numerator = sp.simplify(expr * (k*r))
            if numerator.has(sp.sin):
                # Allow different variable names; check that expression contains sin and division by k*r
                return (True, "psi_asymptotic appears to have sin/(kr) structure")
            else:
                return (False, f"psi_asymptotic does not match expected form: {expr}")
        except Exception as e:
            return (False, f"Error checking psi_asymptotic form: {e}")

    def test_invariant_diff_cross_section_form(self):
        """
        Check that diff_cross_section equals sin^2(delta)/k^2.
        """
        expr = self.diff_cross_section_expr
        if expr is None:
            return (False, "SKIP: cannot parse diff_cross_section to sympy")
        k, delta = sp.symbols('k delta', real=True, positive=True)
        target = sp.sin(delta)**2 / k**2
        try:
            diff = sp.simplify(expr - target)
            if diff == 0:
                return (True, "diff_cross_section matches sin^2(delta)/k^2")
            # Alternative: check functional dependence
            if expr.has(sp.sin(sp.Symbol('delta'))) or expr.has(sp.Symbol('delta')):
                # Check for 1/k^2 factor
                factor_1k2 = sp.simplify(expr * k**2)
                if factor_1k2.has(sp.sin(sp.Symbol('delta'))**2):
                    return (True, "diff_cross_section appears to have sin^2/k^2 structure")
                else:
                    return (False, f"diff_cross_section does not have sin^2/k^2: {factor_1k2}")
            else:
                return (False, f"diff_cross_section missing sin(delta) or 1/k^2: {expr}")
        except Exception as e:
            return (False, f"Error checking diff_cross_section form: {e}")

    def test_invariant_sigma_max_form(self):
        """
        Check that sigma_max is either 4π/k² or 4πR² (or both).
        """
        expr = self.sigma_max_expr
        if expr is None:
            return (False, "SKIP: cannot parse sigma_max to sympy")
        k, R = sp.symbols('k R', real=True, positive=True)
        pi = sp.pi
        form1 = 4*pi/k**2
        form2 = 4*pi*R**2
        try:
            if sp.simplify(expr - form1) == 0:
                return (True, "sigma_max matches 4π/k²")
            if sp.simplify(expr - form2) == 0:
                return (True, "sigma_max matches 4πR²")
            # Check if expression contains 4π times something with k or R
            if expr.has(pi):
                factor = sp.simplify(expr / (4*pi))
                if factor.has(sp.Pow(k, -2)) or factor.has(R**2):
                    return (True, "sigma_max is proportional to 4π and has 1/k² or R² factor")
                else:
                    return (False, f"sigma_max does not match expected forms: {expr}")
            return (False, f"sigma_max missing factor π: {expr}")
        except Exception as e:
            return (False, f"Error checking sigma_max form: {e}")

    def test_modeling_sigma_max_unitarity(self):
        """
        If k is numeric, verify that sigma_max (if numeric) is ≤ 4π/k².
        """
        if self.k is None or not isinstance(self.k, (int, float)):
            return (False, "SKIP: k not available as numeric")
        if self.sigma_max_expr is None:
            return (False, "SKIP: sigma_max not parseable")
        k_val = self.k
        subs_dict = {sp.Symbol('k'): k_val}
        if self.R is not None:
            subs_dict[sp.Symbol('R')] = self.R
        else:
            subs_dict[sp.Symbol('R')] = 1.0  # placeholder (but will not affect if sigma_max is 4π/k²)
        try:
            sigma_num = float(self.sigma_max_expr.subs(subs_dict).evalf())
            bound = 4*math.pi / (k_val**2)
            if sigma_num <= bound + 1e-9:
                return (True, f"sigma_max = {sigma_num:.3f} ≤ 4π/k² = {bound:.3f}")
            else:
                return (False, f"sigma_max = {sigma_num:.3f} exceeds unitarity bound {bound:.3f}")
        except Exception as e:
            return (False, f"Cannot numerically evaluate sigma_max: {e}")

    def test_modeling_diff_cross_section_angular_independence(self):
        """
        Check that diff_cross_section does not depend on angle (s-wave dominance).
        """
        expr = self.diff_cross_section_expr
        if expr is None:
            return (False, "SKIP: cannot parse diff_cross_section")
        theta = sp.Symbol('theta', real=True)
        if theta in expr.free_symbols:
            return (False, "diff_cross_section depends on theta unexpectedly (should be isotropic for s-wave)")
        return (True, "diff_cross_section independent of angle (s-wave)")

    # ---------- Monotonicity Tests ----------
    def _is_sigma_max_form_1_k2(self):
        """Helper: check if sigma_max expression is exactly 4π/k² (or proportional to 1/k²)."""
        expr = self.sigma_max_expr
        if expr is None:
            return False
        k = sp.Symbol('k', real=True, positive=True)
        pi = sp.pi
        form1 = 4*pi/k**2
        try:
            if sp.simplify(expr - form1) == 0:
                return True
            # More permissive: check that expr/ (4π) simplifies to 1/k²
            factor = sp.simplify(expr / (4*pi))
            if sp.simplify(factor * k**2) == 1:
                return True
        except:
            pass
        return False

    def _is_sigma_max_form_R2(self):
        """Helper: check if sigma_max expression is exactly 4πR²."""
        expr = self.sigma_max_expr
        if expr is None:
            return False
        R = sp.Symbol('R', real=True, positive=True)
        pi = sp.pi
        form2 = 4*pi*R**2
        try:
            if sp.simplify(expr - form2) == 0:
                return True
        except:
            pass
        return False

    def test_monotonicity_sigma_max_decreases_with_k(self):
        """
        If sigma_max is of the form 4π/k², then increasing k should decrease sigma_max.
        Perturb k by +10% and check sigma_max decreases (relative change > 0).
        """
        if not self._is_sigma_max_form_1_k2():
            return (False, "SKIP: sigma_max not of form 4π/k², cannot test monotonicity w.r.t. k")

        if self.k is None or not isinstance(self.k, (int, float)):
            return (False, "SKIP: need numeric k for perturbation")
        expr = self.sigma_max_expr
        k_sym = sp.Symbol('k', real=True, positive=True)
        baseline = float(expr.subs({k_sym: self.k}).evalf())
        perturbed_k = self.k * 1.1
        perturbed = float(expr.subs({k_sym: perturbed_k}).evalf())
        # sigma_max should decrease
        if perturbed < baseline - 1e-12:  # allow tiny rounding
            return (True, f"sigma_max decreased from {baseline:.4f} to {perturbed:.4f} when k increased by 10%")
        elif math.isclose(perturbed, baseline, rel_tol=1e-2):
            return (False, f"sigma_max did not change significantly (from {baseline:.4f} to {perturbed:.4f})")
        else:
            return (False, f"sigma_max increased (from {baseline:.4f} to {perturbed:.4f}) when k increased")

    def test_monotonicity_sigma_max_increases_with_R(self):
        """
        If sigma_max is of the form 4πR², then increasing R should increase sigma_max.
        Perturb R by +10% and check sigma_max increases.
        """
        if not self._is_sigma_max_form_R2():
            return (False, "SKIP: sigma_max not of form 4πR², cannot test monotonicity w.r.t. R")

        if self.R is None or not isinstance(self.R, (int, float)):
            return (False, "SKIP: need numeric R for perturbation")
        expr = self.sigma_max_expr
        R_sym = sp.Symbol('R', real=True, positive=True)
        baseline = float(expr.subs({R_sym: self.R}).evalf())
        perturbed_R = self.R * 1.1
        perturbed = float(expr.subs({R_sym: perturbed_R}).evalf())
        # sigma_max should increase
        if perturbed > baseline + 1e-12:
            return (True, f"sigma_max increased from {baseline:.4f} to {perturbed:.4f} when R increased by 10%")
        elif math.isclose(perturbed, baseline, rel_tol=1e-2):
            return (False, f"sigma_max did not change significantly (from {baseline:.4f} to {perturbed:.4f})")
        else:
            return (False, f"sigma_max decreased (from {baseline:.4f} to {perturbed:.4f}) when R increased")

    # ---------- Answer Sanity (weak) ----------
    def test_answer_sanity_sigma_max_positive(self):
        """
        Weak sanity: sigma_max should be positive for positive k,R.
        """
        expr = self.sigma_max_expr
        if expr is None:
            return (False, "SKIP: sigma_max not parseable")
        # Try to evaluate with typical positive numbers
        subs = {}
        if sp.Symbol('k') in expr.free_symbols:
            subs[sp.Symbol('k')] = 0.1
        if sp.Symbol('R') in expr.free_symbols:
            subs[sp.Symbol('R')] = 1.0
        try:
            val = float(expr.subs(subs).evalf())
            if val > 0:
                return (True, f"sigma_max numerically positive: {val}")
            else:
                return (False, f"sigma_max not positive: {val}")
        except:
            # If substitution fails, check string for negative sign
            expr_str = str(expr)
            if expr_str.startswith('-'):
                return (False, "sigma_max expression appears negative")
            return (True, "sigma_max expression looks non-negative")
        return (False, "sigma_max expression not recognized")

    # ---------- Main Runner ----------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Discover test methods
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

# Example usage (for testing, not part of checker)
if __name__ == "__main__":
    class Solution:
        def __init__(self):
            self.k = 0.1
            self.R = 1.0
            self.results = {
                'psi_asymptotic': 'sin(k*r + delta)/(k*r)',
                'diff_cross_section': 'sin(delta)**2/k**2',
                'sigma_max': '4*pi*R**2'
            }

    sol = Solution()
    checker = PhysicsChecker(sol)
    success, logs = checker.run_all_tests()
    for log in logs:
        print(log)
    print(f"\nAll tests passed: {success}")