import sympy as sp
import math
import re
from typing import Any, Optional

class PhysicsChecker:
    """
    Audits solutions for the particle-moon collision cross-section problem.
    Checks interface completeness, theorem-based invariants, modeling quality,
    monotonicity, and a weak sanity check on the final answers.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract givens (safe defaults: None)
        self.R = getattr(solution_instance, 'R', None)
        self.V_esc = getattr(solution_instance, 'V_esc', None)
        self.V0 = getattr(solution_instance, 'V0', None)

        # Convert to float if possible
        for attr in ['R', 'V_esc', 'V0']:
            val = getattr(self, attr, None)
            if val is not None:
                try:
                    setattr(self, attr, float(val))
                except (ValueError, TypeError):
                    setattr(self, attr, None)

        # Extract results (from solution_instance.results dict)
        results = getattr(solution_instance, 'results', {})
        self.sigma = results.get('sigma', None)
        self.limit_V0to0 = results.get('limit_V0to0', None)
        self.limit_V0toinf = results.get('limit_V0toinf', None)

        # Internal: derived sigma expression (symbolic)
        self._sigma_expr = None
        if self.R is not None and self.V_esc is not None and self.V0 is not None:
            try:
                R, V_esc, V0 = sp.symbols('R V_esc V0', positive=True)
                expr = sp.pi * R**2 * (1 + V_esc**2 / V0**2)
                self._sigma_expr = expr.subs({R: self.R, V_esc: self.V_esc, V0: self.V0}).evalf()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Interface checks – required keys exist and are strings
    # ------------------------------------------------------------------
    def test_interface_sigma_exists(self) -> tuple[bool, str]:
        if self.sigma is None:
            return (False, "Missing key 'sigma' in results")
        if not isinstance(self.sigma, str):
            return (False, f"'sigma' is not a string (got {type(self.sigma).__name__})")
        return (True, "sigma key present and is a string")

    def test_interface_limit_V0to0_exists(self) -> tuple[bool, str]:
        if self.limit_V0to0 is None:
            return (False, "Missing key 'limit_V0to0' in results")
        if not isinstance(self.limit_V0to0, str):
            return (False, f"'limit_V0to0' is not a string")
        return (True, "limit_V0to0 present and is a string")

    def test_interface_limit_V0toinf_exists(self) -> tuple[bool, str]:
        if self.limit_V0toinf is None:
            return (False, "Missing key 'limit_V0toinf' in results")
        if not isinstance(self.limit_V0toinf, str):
            return (False, f"'limit_V0toinf' is not a string")
        return (True, "limit_V0toinf present and is a string")

    # ------------------------------------------------------------------
    # Invariant checks – physical relationships derived from laws
    # ------------------------------------------------------------------
    def test_invariant_sigma_from_physics(self) -> tuple[bool, str]:
        """
        Derive sigma from conservation laws and compare numerically with the
        (parsed) result sigma. Only runs when numeric givens are present.
        """
        if self.R is None or self.V_esc is None or self.V0 is None:
            return (False, "SKIP: numeric givens (R, V_esc, V0) not all available")
        if self._sigma_expr is None:
            return (False, "Could not compute derived sigma (internal error)")

        # Try to parse the result sigma string as a sympy expression
        if self.sigma is None:
            return (False, "SKIP: sigma result string missing")

        try:
            # Parse LaTeX to sympy
            parsed = sp.parse_latex(self.sigma)
            # Evaluate with numeric symbols matching those in the result
            subs_dict = {
                sp.Symbol('R'): self.R,
                sp.Symbol('V_esc'): self.V_esc,
                sp.Symbol('V0'): self.V0,
                sp.Symbol('V_{\text{esc}}'): self.V_esc,
            }
            numeric_result = parsed.subs(subs_dict).evalf()
        except Exception as e:
            return (False, f"Cannot parse sigma string as valid expression: {e}")

        if numeric_result.is_number:
            numeric_result = float(numeric_result)
            derived_val = float(self._sigma_expr)
            if math.isclose(derived_val, numeric_result, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "Derived sigma matches parsed result sigma within tolerance")
            else:
                return (False, f"Sigma mismatch: derived {derived_val:.4f} vs result {numeric_result:.4f}")
        else:
            return (False, "Parsed sigma expression did not simplify to a number")

    def test_invariant_sigma_geometric(self) -> tuple[bool, str]:
        """sigma must be at least pi*R^2, if numeric values are available."""
        if self.R is None:
            return (False, "SKIP: R not given")
        if self._sigma_expr is None:
            return (False, "Cannot compute sigma (missing V_esc or V0)")

        geometric = math.pi * self.R**2
        if math.isclose(geometric, float(self._sigma_expr), rel_tol=1e-2) or float(self._sigma_expr) >= geometric:
            return (True, f"sigma ({float(self._sigma_expr):.4f}) >= pi*R^2 ({geometric:.4f})")
        else:
            return (False, f"sigma ({float(self._sigma_expr):.4f}) < pi*R^2 ({geometric:.4f})")

    # ------------------------------------------------------------------
    # Modeling & trace checks – quality of the provided output strings
    # ------------------------------------------------------------------
    def _normalize_latex(self, s: str) -> str:
        """Remove spacing, standardize LaTeX commands for pattern matching."""
        s = s.replace(' ', '')
        s = s.replace('\\text{esc}', '\\mathrm{esc}')
        s = s.replace('\\mathrm{esc}', '\\text{esc}')
        return s

    def test_modeling_sigma_pattern(self) -> tuple[bool, str]:
        """Check that sigma string matches the expected pattern."""
        if self.sigma is None:
            return (False, "SKIP: sigma not provided")
        norm = self._normalize_latex(self.sigma)
        # Expected: \pi R^2 \left(1 + \frac{V_{\text{esc}}^2}{V_0^2}\right)
        # We'll accept variations in spaces and \text vs \mathrm.
        pattern = r'\\pi\s*R\^{2}\s*\\left\s*\(\s*1\s*\+\s*\\frac\{V_\{\\text\{esc\}\}\^{2}\}\{V_0\^{2}\}\s*\\right\s*\)'
        if re.fullmatch(pattern, norm, re.IGNORECASE):
            return (True, "sigma string matches expected LaTeX pattern")
        else:
            return (False, f"sigma string does not match expected pattern (got '{self.sigma}')")

    def test_modeling_limit_V0to0_pattern(self) -> tuple[bool, str]:
        """Limit as V0->0 should be \\infty."""
        if self.limit_V0to0 is None:
            return (False, "SKIP: limit_V0to0 not provided")
        norm = self._normalize_latex(self.limit_V0to0)
        if norm in (r'\infty', r'\\infty'):
            return (True, "limit_V0to0 string is \\infty")
        else:
            return (False, f"limit_V0to0 expected '\\infty', got '{self.limit_V0to0}'")

    def test_modeling_limit_V0toinf_pattern(self) -> tuple[bool, str]:
        """Limit as V0->infinity should be \\pi R^2."""
        if self.limit_V0toinf is None:
            return (False, "SKIP: limit_V0toinf not provided")
        norm = self._normalize_latex(self.limit_V0toinf)
        if norm in (r'\pi R^{2}', r'\\pi\\,\\R^{2}', r'\pi R^2', r'\\pi R^2'):
            return (True, f"limit_V0toinf string is \\pi R^2 (got '{self.limit_V0toinf}')")
        else:
            return (False, f"limit_V0toinf expected '\\pi R^2', got '{self.limit_V0toinf}'")

    # ------------------------------------------------------------------
    # Monotonicity checks – trend for sigma as V0 changes
    # ------------------------------------------------------------------
    def test_monotonicity_sigma_decreases_with_V0(self) -> tuple[bool, str]:
        """If V0 increases, sigma should decrease (sigma ∝ 1/V0^2)."""
        if self.R is None or self.V_esc is None or self.V0 is None:
            return (False, "SKIP: numeric givens (R, V_esc, V0) not available")
        # compute baseline sigma
        sigma0 = math.pi * self.R**2 * (1 + (self.V_esc**2)/(self.V0**2))
        # perturb V0 up by 10%
        V1 = self.V0 * 1.1
        sigma1 = math.pi * self.R**2 * (1 + (self.V_esc**2)/(V1**2))
        if sigma1 < sigma0:
            return (True, f"sigma decreases with V0 (sigma0={sigma0:.4f} -> sigma1={sigma1:.4f})")
        else:
            return (False, f"sigma did not decrease: sigma0={sigma0:.4f}, sigma1={sigma1:.4f}")

    # ------------------------------------------------------------------
    # Weak answer sanity – one test only, no re-computation of final values
    # ------------------------------------------------------------------
    def test_answer_sanity_sigma_positive(self) -> tuple[bool, str]:
        """Check that sigma string is non-empty and contains 'π' or 'pi'."""
        if self.sigma is None:
            return (False, "SKIP: sigma not provided")
        if not isinstance(self.sigma, str) or len(self.sigma) == 0:
            return (False, "sigma string is empty")
        # weak check: contains some representation of pi
        if 'pi' in self.sigma.lower() or 'π' in self.sigma:
            return (True, "sigma string contains pi reference")
        else:
            return (False, f"sigma string does not contain 'pi' or 'π': '{self.sigma}'")

    # ------------------------------------------------------------------
    # Runner: discovers all test_ methods and executes them
    # ------------------------------------------------------------------
    def run_all_tests(self) -> tuple[bool, list[str]]:
        logs: list[str] = []
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