import math
import traceback
from typing import Any, Dict, List, Tuple

import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance: Any) -> None:
        # Extract results dict safely
        results = getattr(solution_instance, 'results', None) or {}
        if not isinstance(results, dict):
            results = {}
        
        self.G_E_asymptotic = results.get('G_E_asymptotic', None)
        self.r = results.get('r', None)
        self.R = results.get('R', None)
        self.eta = results.get('eta', None)
        
        # For numeric substitution we define symbols (they are likely used in the expressions)
        self._symbols = {
            'm': sp.Symbol('m', positive=True, real=True),
            'V0': sp.Symbol('V0', real=True),
            'a': sp.Symbol('a', positive=True, real=True),
            'hbar': sp.Symbol('hbar', positive=True, real=True),
            'k': sp.Symbol('k', positive=True, real=True),
            'x': sp.Symbol('x', real=True),
        }

    def _sympify(self, expr: str) -> sp.Expr | None:
        """Try to convert a string to a sympy expression using common substitutions.
        Returns None if parsing fails."""
        try:
            # Replace common LaTeX forms
            expr_clean = expr.replace('\\', '').replace(' ', '')
            return sp.sympify(expr_clean, locals=self._symbols)
        except Exception:
            return None

    def _numeric_sub(self, expr: sp.Expr, vals: Dict[str, float]) -> sp.Expr:
        """Substitute numeric values for symbols and evaluate."""
        subs_dict = {self._symbols[sn]: val for sn, val in vals.items()}
        return expr.subs(subs_dict).evalf()

    # ---------- Interface tests ----------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all four emission keys exist in results."""
        missing = []
        for key in ['G_E_asymptotic', 'r', 'R', 'eta']:
            if getattr(self, key, None) is None:
                missing.append(key)
        if missing:
            return (False, f"Missing keys: {', '.join(missing)}")
        return (True, "All required keys present")

    def test_interface_G_E_asymptotic_type(self) -> Tuple[bool, str]:
        if self.G_E_asymptotic is None:
            return (False, "SKIP: G_E_asymptotic not provided")
        if not isinstance(self.G_E_asymptotic, str):
            return (False, "G_E_asymptotic should be a string")
        return (True, "G_E_asymptotic is string")

    def test_interface_r_type(self) -> Tuple[bool, str]:
        if self.r is None:
            return (False, "SKIP: r not provided")
        if not isinstance(self.r, str):
            return (False, "r should be a string")
        return (True, "r is string")

    def test_interface_R_type(self) -> Tuple[bool, str]:
        if self.R is None:
            return (False, "SKIP: R not provided")
        if not isinstance(self.R, str):
            return (False, "R should be a string")
        return (True, "R is string")

    def test_interface_eta_type(self) -> Tuple[bool, str]:
        if self.eta is None:
            return (False, "SKIP: eta not provided")
        if not isinstance(self.eta, str):
            return (False, "eta should be a string")
        return (True, "eta is string")

    # ---------- Invariant tests (theorem-based) ----------
    def test_invariant_outgoing_wave(self) -> Tuple[bool, str]:
        """Check that the asymptotic Green's function has the form e^{i k |x|}."""
        if self.G_E_asymptotic is None:
            return (False, "SKIP: G_E_asymptotic not provided")
        expr = self._sympify(self.G_E_asymptotic)
        if expr is None:
            return (False, "Cannot parse G_E_asymptotic as sympy expression")
        # check that it contains exp(I*k*|x|) or similar
        # We'll look for a factor e^{i k |x|} by substituting a random numeric x>0
        vals = {'m': 1, 'hbar': 1, 'k': 2}
        x_pos = 2.3
        subs_pos = {**vals, 'x': x_pos}
        # The expression should be proportional to exp(I*k*x) for x>0
        # We'll check that the ratio with exp(I*k*x) is constant (i.e., the coefficient)
        # For x negative, it should be exp(-I*k*x) = exp(I*k*|x|)
        expr_num_pos = self._numeric_sub(expr, {**subs_pos})
        # If the expression contains the correct exponential, we can check the absolute phase
        # Simpler: check that the string contains 'e^{i k |x|}' or 'e^{i k |x|}'
        if 'e^{i k |x|}' in self.G_E_asymptotic or 'e^{i k |x|}' in self.G_E_asymptotic:
            return (True, "Green's function has outgoing wave form")
        # Fallback: check that for x>0, the expression can be written as -i m/(hbar^2 k) * e^{I k x}
        # We'll evaluate the coefficient
        try:
            coeff = expr / sp.exp(sp.I * self._symbols['k'] * self._symbols['x'])
            coeff_val = coeff.subs({self._symbols['x']: x_pos, **vals}).evalf()
            # The coefficient should be constant (-i) for x>0? Actually -i m/(hbar^2 k) * e^{i k |x|} = -i m/(hbar^2 k) * e^{i k x}
            # So coefficient = -i m/(hbar^2 k)
            expected_coeff = -sp.I * self._symbols['m'] / (self._symbols['hbar']**2 * self._symbols['k'])
            expected_val = expected_coeff.subs(vals).evalf()
            if math.isclose(float(sp.Abs(coeff_val - expected_val)), 0.0, rel_tol=1e-4, abs_tol=1e-4):
                return (True, "Outgoing wave coefficient verified for x>0")
            else:
                return (False, f"Outgoing wave coefficient mismatch: expected {expected_val}, got {coeff_val}")
        except Exception:
            return (False, "Could not verify outgoing wave factor")

    def test_invariant_r_imaginary(self) -> Tuple[bool, str]:
        """Check that reflection amplitude r is purely imaginary (has factor -i)."""
        if self.r is None:
            return (False, "SKIP: r not provided")
        expr = self._sympify(self.r)
        if expr is None:
            return (False, "Cannot parse r as sympy expression")
        # Check that the real part is zero when evaluated with random numeric values
        vals = {'m': 1, 'V0': 0.5, 'a': 1, 'hbar': 1, 'k': 2}
        try:
            r_val = expr.subs({self._symbols[s]: v for s, v in vals.items()}).evalf()
            real_part = float(sp.re(r_val))
            if math.isclose(real_part, 0.0, abs_tol=1e-10):
                return (True, "r is purely imaginary")
            else:
                return (False, f"r has nonzero real part {real_part}")
        except Exception as e:
            return (False, f"Could not evaluate r: {e}")

    def test_invariant_R_nonnegativity(self) -> Tuple[bool, str]:
        """Check that R expression is manifestly non-negative (squared modulus)."""
        if self.R is None:
            return (False, "SKIP: R not provided")
        expr = self._sympify(self.R)
        if expr is None:
            return (False, "Cannot parse R as sympy expression")
        # Check that R is a squared quantity (contains a square factor)
        # We'll check that the expression is non-negative for random positive parameters.
        vals = {'m':2.5, 'V0':1.2, 'a':0.7, 'hbar':1.05, 'k':3.1}
        try:
            R_val = float(expr.subs({self._symbols[s]: v for s, v in vals.items()}).evalf())
            if R_val >= 0:
                return (True, "R is non-negative for sample values")
            else:
                return (False, f"R is negative: {R_val}")
        except Exception as e:
            return (False, f"Could not evaluate R: {e}")

    # ---------- Modeling / trace tests ----------
    def test_modeling_consistency_r_R(self) -> Tuple[bool, str]:
        """Check that R = |r|^2 holds symbolically (or numerically)."""
        if self.r is None or self.R is None:
            return (False, "SKIP: r or R not provided")
        r_expr = self._sympify(self.r)
        R_expr = self._sympify(self.R)
        if r_expr is None or R_expr is None:
            return (False, "Could not parse r or R")
        # define a test numeric substitution
        vals = {'m': 1.2, 'V0': 0.8, 'a': 0.5, 'hbar': 1.0, 'k': 1.5}
        try:
            r_num = complex(r_expr.subs({self._symbols[s]: v for s, v in vals.items()}).evalf())
            R_num = float(R_expr.subs({self._symbols[s]: v for s, v in vals.items()}).evalf())
            expected_R = abs(r_num)**2
            if math.isclose(expected_R, R_num, rel_tol=1e-6, abs_tol=1e-6):
                return (True, "R = |r|^2 holds numerically")
            else:
                return (False, f"R = {R_num} but |r|^2 = {expected_R}")
        except Exception as e:
            return (False, f"Could not evaluate consistency: {e}")

    def test_modeling_eta_validity(self) -> Tuple[bool, str]:
        """Check that eta is proportional to |V0| m a/(hbar^2 k)."""
        if self.eta is None:
            return (False, "SKIP: eta not provided")
        expr = self._sympify(self.eta)
        if expr is None:
            return (False, "Cannot parse eta as sympy expression")
        # The expected form: |V0| m a/(hbar^2 k). But absolute value might be written as abs(V0)
        # We'll check that the expression contains factors V0, m, a, 1/(hbar^2 k)
        # Use symbolic simplification: define expected = abs(V0)*m*a/(hbar^2*k)
        # But abs is tricky; we can check by expansion or numeric test.
        # Numerically: for positive V0, eta should equal V0*m*a/(hbar^2*k)
        vals = {'m': 1.5, 'V0': 2.0, 'a': 1.2, 'hbar': 1.0, 'k': 0.8}
        try:
            eta_val = float(expr.subs({self._symbols[s]: v for s, v in vals.items()}).evalf())
            expected = abs(vals['V0']) * vals['m'] * vals['a'] / (vals['hbar']**2 * vals['k'])
            if math.isclose(eta_val, expected, rel_tol=1e-6, abs_tol=1e-6):
                return (True, "eta formula matches expected")
            else:
                return (False, f"eta = {eta_val}, expected = {expected}")
        except Exception as e:
            return (False, f"Could not evaluate eta: {e}")

    # ---------- Monotonicity tests (explicit perturbation) ----------
    def test_monotonicity_R_vs_V0(self) -> Tuple[bool, str]:
        """R increases with |V0| (squared). For fixed other parameters, increasing V0 should increase R."""
        if self.R is None:
            return (False, "SKIP: R not provided")
        expr = self._sympify(self.R)
        if expr is None:
            return (False, "Cannot parse R")
        # baseline V0=1, perturbed V0=1.5
        base_vals = {'m': 1, 'a': 1, 'hbar': 1, 'k': 2, 'V0': 1.0}
        perturb_vals = base_vals.copy()
        perturb_vals['V0'] = 1.5
        try:
            R_base = float(expr.subs({self._symbols[s]: v for s, v in base_vals.items()}).evalf())
            R_perturb = float(expr.subs({self._symbols[s]: v for s, v in perturb_vals.items()}).evalf())
            if R_perturb > R_base:
                return (True, "R increases with V0 (as V0^2)")
            else:
                return (False, f"R_base={R_base}, R_perturb={R_perturb}, expected increase")
        except Exception as e:
            return (False, f"Could not evaluate monotonicity: {e}")

    def test_monotonicity_R_vs_k(self) -> Tuple[bool, str]:
        """R decreases as k increases (since denominator has k^4)."""
        if self.R is None:
            return (False, "SKIP: R not provided")
        expr = self._sympify(self.R)
        if expr is None:
            return (False, "Cannot parse R")
        # baseline k=2, perturbed k=3
        base_vals = {'m': 1, 'a': 1, 'hbar': 1, 'V0': 1, 'k': 2.0}
        perturb_vals = base_vals.copy()
        perturb_vals['k'] = 3.0
        try:
            R_base = float(expr.subs({self._symbols[s]: v for s, v in base_vals.items()}).evalf())
            R_perturb = float(expr.subs({self._symbols[s]: v for s, v in perturb_vals.items()}).evalf())
            if R_perturb < R_base:
                return (True, "R decreases with increasing k (k^4 in denominator)")
            else:
                return (False, f"R_base={R_base}, R_perturb={R_perturb}, expected decrease")
        except Exception as e:
            return (False, f"Could not evaluate monotonicity: {e}")

    def test_monotonicity_R_vs_a(self) -> Tuple[bool, str]:
        """R depends on sin^2(ka); for small a, sin(ka)~ka so R ~ (m^2 V0^2 a^2)/(hbar^4 k^2). For large a, oscillates. 
        We'll test that for ka small (<pi/2) R increases with a."""
        if self.R is None:
            return (False, "SKIP: R not provided")
        expr = self._sympify(self.R)
        if expr is None:
            return (False, "Cannot parse R")
        # choose k=1, a small: 0.5 -> ka=0.5 < pi/2, sin^2 increases with a
        base_vals = {'m': 1, 'hbar': 1, 'V0': 1, 'k': 1.0, 'a': 0.5}
        perturb_vals = base_vals.copy()
        perturb_vals['a'] = 1.0
        try:
            R_base = float(expr.subs({self._symbols[s]: v for s, v in base_vals.items()}).evalf())
            R_perturb = float(expr.subs({self._symbols[s]: v for s, v in perturb_vals.items()}).evalf())
            if R_perturb > R_base:
                return (True, "R increases with a for ka < pi/2")
            else:
                return (False, f"R_base={R_base}, R_perturb={R_perturb}, expected increase")
        except Exception as e:
            return (False, f"Could not evaluate monotonicity: {e}")

    # ---------- Weak answer sanity (at most one) ----------
    def test_answer_sanity_R_finite(self) -> Tuple[bool, str]:
        """Check that R is finite and physically plausible (0 ≤ R ≤ 1)."""
        if self.R is None:
            return (False, "SKIP: R not provided")
        expr = self._sympify(self.R)
        if expr is None:
            return (False, "Cannot parse R")
        # Test with moderate parameters
        vals = {'m': 2, 'a': 0.5, 'hbar': 1, 'V0': 0.3, 'k': 1.5}
        try:
            R_val = float(expr.subs({self._symbols[s]: v for s, v in vals.items()}).evalf())
        except Exception:
            return (False, "Could not evaluate R")
        if R_val < 0:
            return (False, f"Negative R = {R_val}")
        if R_val > 1:
            return (False, f"R > 1, unphysical: {R_val}")
        return (True, f"R = {R_val}, in [0,1]")

    # ---------- Main runner ----------
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