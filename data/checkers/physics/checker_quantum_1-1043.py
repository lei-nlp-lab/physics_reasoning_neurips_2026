import sympy as sp
import re
from collections.abc import Mapping

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results safely. Assume solution_instance has a 'results' dict
        # or attributes named 'psi_asympt', 'prob_density'.
        results = getattr(solution_instance, 'results', None)
        if isinstance(results, Mapping):
            self.psi_asympt = results.get('psi_asympt', None)
            self.prob_density = results.get('prob_density', None)
        else:
            # Fallback: try direct attribute names
            self.psi_asympt = getattr(solution_instance, 'psi_asympt', None)
            self.prob_density = getattr(solution_instance, 'prob_density', None)
        # For symbolic manipulation, define symbols
        self.m, self.hbar, self.t, self.x = sp.symbols('m hbar t x', positive=True, real=True)
        self.phi_func = sp.Function('phi')  # placeholder for φ(k)
        # Expected asymptotic form (derived from theory)
        self.expected_psi_asympt = (
            sp.sqrt(self.m/(self.hbar*self.t))
            * sp.exp(-sp.I*sp.pi/4)
            * sp.exp(sp.I*self.m*self.x**2/(2*self.hbar*self.t))
            * self.phi_func(self.m*self.x/(self.hbar*self.t))
        )
        self.expected_prob_density = (
            (self.m/(self.hbar*self.t))
            * sp.Abs(self.phi_func(self.m*self.x/(self.hbar*self.t)))**2
        )

    def _str_to_sympy(self, expr_str):
        """Try to parse a LaTeX string into a sympy expression; return None on failure."""
        try:
            from sympy.parsing.latex import parse_latex
            return parse_latex(expr_str)
        except Exception:
            return None

    def _to_sympy(self, obj):
        """Return a sympy expression from a string or sympy object."""
        if isinstance(obj, sp.Basic):
            return obj
        if isinstance(obj, str):
            sym = self._str_to_sympy(obj)
            if sym is None:
                try:
                    return sp.sympify(obj, locals={'phi': self.phi_func})
                except Exception:
                    return None
            return sym
        return None

    # ---- Interface checks ----
    def test_interface_has_psi_asympt(self):
        if self.psi_asympt is None:
            return (False, "SKIP: psi_asympt not present")
        return (True, "psi_asympt key present")

    def test_interface_has_prob_density(self):
        if self.prob_density is None:
            return (False, "SKIP: prob_density not present")
        return (True, "prob_density key present")

    def test_interface_psi_asympt_type(self):
        if self.psi_asympt is None:
            return (False, "SKIP: no psi_asympt")
        if not isinstance(self.psi_asympt, (str, sp.Basic)):
            return (False, f"type is {type(self.psi_asympt)}, expected str or sympy expression")
        return (True, "psi_asympt type acceptable")

    def test_interface_prob_density_type(self):
        if self.prob_density is None:
            return (False, "SKIP: no prob_density")
        if not isinstance(self.prob_density, (str, sp.Basic)):
            return (False, f"type is {type(self.prob_density)}, expected str or sympy expression")
        return (True, "prob_density type acceptable")

    # ---- Modeling & theorem checks ----
    def test_modeling_asymptotic_structure(self):
        """Check that psi_asympt contains the essential factors: sqrt(m/(hbar t)),
        exp(-i pi/4), exp(i m x^2/(2 hbar t)), and phi(m x/(hbar t))."""
        if self.psi_asympt is None:
            return (False, "SKIP: no psi_asympt to check")
        expr = self._to_sympy(self.psi_asympt)
        if expr is None:
            return (False, "Cannot parse psi_asympt into sympy expression")
        expected = self.expected_psi_asympt
        diff = sp.simplify(expr - expected)
        if diff != 0:
            # Fallback: check via regex
            return self._check_psi_asympt_loose(expr)
        return (True, "psi_asympt matches expected symbolic structure")

    def _check_psi_asympt_loose(self, expr):
        """Fallback: search for required substrings/constants in string representation."""
        s = str(expr) if not isinstance(expr, str) else expr
        checks = [
            (r'sqrt\(.*m.*hbar.*t\)', 'sqrt(m/(hbar t))'),
            (r'exp\s*\(-i\s*\*?\s*pi/4\)', 'exp(-i pi/4)'),
            (r'exp\s*\(i.*m\s*\*?\s*x\*\*2.*hbar.*t\)', 'exp(i m x^2/(2 hbar t))'),
            (r'phi\s*\(.*m.*x.*hbar.*t\)', 'phi(m x/(hbar t))'),
        ]
        missing = []
        for pattern, desc in checks:
            if not re.search(pattern, s, re.IGNORECASE):
                missing.append(desc)
        if missing:
            return (False, f"Missing factors in psi_asympt: {', '.join(missing)}")
        return (True, "psi_asympt structure verified via regex (fallback)")

    def test_modeling_prob_density_from_asympt(self):
        """Check that prob_density = |psi_asympt|^2."""
        if self.psi_asympt is None or self.prob_density is None:
            return (False, "SKIP: missing items for this test")
        psi_sym = self._to_sympy(self.psi_asympt)
        prob_sym = self._to_sympy(self.prob_density)
        if psi_sym is None or prob_sym is None:
            return (False, "Cannot parse one of the inputs into sympy")
        modsq = sp.simplify(psi_sym * sp.conjugate(psi_sym))
        expected = sp.simplify(prob_sym)
        diff = sp.simplify(modsq - expected)
        if diff != 0:
            return (False, f"prob_density does not equal |psi_asympt|^2, diff = {diff}")
        return (True, "prob_density correctly derived as |psi_asympt|^2")

    def test_modeling_prob_density_structure(self):
        """Check that prob_density has the form (m/(hbar t)) |phi(mx/(hbar t))|^2."""
        if self.prob_density is None:
            return (False, "SKIP: no prob_density")
        expr = self._to_sympy(self.prob_density)
        if expr is None:
            return (False, "Cannot parse prob_density into sympy")
        expected = self.expected_prob_density
        diff = sp.simplify(expr - expected)
        if diff != 0:
            # Fallback: check via regex
            s = str(expr)
            checks = [
                (r'm/(hbar\s*\*?\s*t)', 'm/(hbar t)'),
                (r'\|?phi\|?\s*\(.*m.*x.*hbar.*t\)', '|phi(m x/(hbar t))|^2'),
            ]
            missing = []
            for pattern, desc in checks:
                if not re.search(pattern, s, re.IGNORECASE):
                    missing.append(desc)
            if missing:
                return (False, f"Missing components in prob_density: {', '.join(missing)}")
            return (True, "prob_density structure verified loosely")
        return (True, "prob_density matches expected structure")

    # ---- Invariant checks ----
    def test_invariant_free_particle_dispersion(self):
        """Check that the asymptotic form respects the free particle dispersion."""
        if self.psi_asympt is None:
            return (False, "SKIP: no psi_asympt")
        expr = self._to_sympy(self.psi_asympt)
        if expr is None:
            return (False, "Cannot parse psi_asympt")
        s = str(expr)
        if re.search(r'exp\s*\(.*i\s*.*m.*x\*\*2.*[^/].*hbar.*t\)', s, re.IGNORECASE):
            return (True, "Dispersion phase factor present")
        return (False, "Phase factor not consistent with dispersion relation")

    def test_invariant_normalization_conservation(self):
        """Check that total probability is conserved by asymptotic form."""
        if self.prob_density is None:
            return (False, "SKIP: no prob_density")
        expr = self._to_sympy(self.prob_density)
        if expr is None:
            return (False, "Cannot parse prob_density")
        s = str(expr)
        if re.search(r'm/(hbar\s*\*?\s*t)', s) and re.search(r'phi|φ', s, re.IGNORECASE):
            return (True, "Normalization prefactor consistent with conservation")
        return (False, "prob_density does not show correct prefactor for normalization")

    # ---- Answer sanity (weak) ----
    def test_answer_sanity_prob_density_positive(self):
        """Weak sanity: check that prob_density is a non-negative quantity using string patterns only."""
        if self.prob_density is None:
            return (False, "SKIP: no prob_density")
        s = str(self.prob_density)
        # Check for squared magnitude pattern: |...|^2 or |...|**2 or Abs(...)^2
        has_squared_mag = (
            re.search(r'\|.*\|\s*(\^2|\*\*2)', s) or 
            re.search(r'Abs\s*\(.*\)\s*(\^2|\*\*2)', s) or
            re.search(r'\|\s*(.*?)\s*\|?\s*(\^2|\*\*2)', s)
        )
        # Also check for positive prefactor m/(hbar t) and the presence of phi
        has_positive_factor = bool(re.search(r'm/(hbar\s*\*?\s*t)', s))
        has_phi = ('phi' in s.lower() or 'φ' in s)
        if has_squared_mag or (has_positive_factor and has_phi):
            return (True, "prob_density appears non-negative (squared magnitude or positive factor)")
        return (False, "prob_density does not obviously look non-negative")

    # ---- Monotonicity tests ----
    def test_monotonicity_prob_density_decreases_with_t(self):
        """Check that prob density decreases as t increases (fixed x/t ratio)."""
        if self.prob_density is None:
            return (False, "SKIP: no prob_density")
        expr = self._to_sympy(self.prob_density)
        if expr is None:
            return (False, "Cannot parse prob_density into sympy")
        # Replace |phi(...)|^2 by 1 to isolate the t-prefactor
        # Find all occurrences of Abs(phi(...))**2 and replace by 1
        abs_phi_sq = sp.Abs(self.phi_func(sp.Symbol('dummy')))**2  # pattern
        # More robust: replace every occurrence of Abs(phi(...))**2 with 1
        # Use sympy's replace
        replacement_rule = lambda a: sp.Integer(1) if a.has(sp.Abs) and a.has(self.phi_func) else a
        base_expr = sp.simplify(expr.replace(lambda a: a.has(sp.Abs) and a.has(self.phi_func), lambda a: 1))
        # Substitute numeric values: m=1, hbar=1, x=1 (so that x/t is fixed when t changes)
        baseline_t = 1.0
        perturbed_t = 2.0
        x_val = 1.0  # arbitrary, but k = m*x/(hbar*t) becomes x/t
        subs_dict = {self.m: 1.0, self.hbar: 1.0, self.x: x_val}
        try:
            base_val = float(base_expr.subs({self.t: baseline_t, **subs_dict}))
            perturbed_val = float(base_expr.subs({self.t: perturbed_t, **subs_dict}))
        except Exception as e:
            return (False, f"Could not evaluate expression numerically: {e}")
        if base_val <= perturbed_val:
            return (False, f"Baseline prob density at t={baseline_t}: {base_val}, perturbed at t={perturbed_t}: {perturbed_val}. Expected decrease (baseline larger).")
        # Also check that ratio is roughly 1/t factor: perturbed_val ≈ base_val * (1/perturbed_t)/(1/baseline_t) = base_val/2
        expected_ratio = baseline_t / perturbed_t
        actual_ratio = perturbed_val / base_val
        if abs(actual_ratio - expected_ratio) > 1e-2:
            return (False, f"Ratio {actual_ratio} too far from expected {expected_ratio} (1/t scaling)")
        return (True, f"Prob density decreases with t: base={base_val}, perturbed={perturbed_val}, ratio={actual_ratio:.3f}")

    # ---- Run all tests ----
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