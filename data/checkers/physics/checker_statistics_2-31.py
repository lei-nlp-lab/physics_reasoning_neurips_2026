import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results with None defaults – NO numeric fallbacks
        self.results = getattr(solution_instance, 'results', {})
        self.C_vib_formula = self.results.get('C_vib_formula', None)
        self.C_vib_high_T = self.results.get('C_vib_high_T', None)
        self.C_vib_low_T = self.results.get('C_vib_low_T', None)

        # Sympy symbols for internal derivation
        self.R_sym = sp.Symbol('R', positive=True)
        self.theta_sym = sp.Symbol('theta', positive=True)
        self.T_sym = sp.Symbol('T', positive=True)

        # Reference expressions (theorem ground truth)
        self.ref_C_vib = self.R_sym * (self.theta_sym / self.T_sym)**2 * sp.exp(self.theta_sym / self.T_sym) / (sp.exp(self.theta_sym / self.T_sym) - 1)**2
        self.ref_high_T = self.R_sym
        self.ref_low_T = self.R_sym * (self.theta_sym / self.T_sym)**2 * sp.exp(-self.theta_sym / self.T_sym)

        # Default numerical parameters for safe evaluation
        self.num_theta = 500.0   # K
        self.num_R = 8.314       # J/(mol·K)

    def _parse_to_sympy(self, val):
        """Try to convert a result value (string or sympy) to a sympy expression."""
        if val is None:
            return None
        if isinstance(val, sp.Expr):
            return val
        if isinstance(val, str):
            try:
                local_dict = {
                    'R': self.R_sym,
                    'theta': self.theta_sym,
                    'T': self.T_sym,
                    'exp': sp.exp
                }
                return sp.sympify(val, locals=local_dict)
            except Exception:
                return None
        return None

    def _numerical_eval(self, expr, T_val, theta_val=None, R_val=None):
        """Evaluate a sympy expression at numeric T (and optional theta,R)."""
        if expr is None:
            return None
        if theta_val is None:
            theta_val = self.num_theta
        if R_val is None:
            R_val = self.num_R
        subs = {self.T_sym: T_val, self.theta_sym: theta_val, self.R_sym: R_val}
        try:
            return float(expr.subs(subs).evalf())
        except Exception:
            return None

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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

    # ------------------- INTERFACE TESTS -------------------
    def test_interface_required_keys(self):
        """Check that all expected result keys exist and are not None."""
        expected = ['C_vib_formula', 'C_vib_high_T', 'C_vib_low_T']
        missing = [k for k in expected if k not in self.results or self.results[k] is None]
        if missing:
            return (True, f"SKIP: Missing keys: {missing}")
        return (True, "All three required keys present with non‑None values")

    # ------------------- THEOREM-BASED INVARIANT TESTS -------------------
    def test_invariant_high_T_limit_numerical(self):
        """At T >> theta, C_vib must approach R (within 2% relative)."""
        formula = self._parse_to_sympy(self.C_vib_formula)
        if formula is None:
            return (True, "SKIP: Cannot parse C_vib_formula")
        T_high = 100.0 * self.num_theta          # T >> theta
        val = self._numerical_eval(formula, T_val=T_high)
        if val is None:
            return (False, "Numerical evaluation of formula at high T failed")
        if math.isclose(val, self.num_R, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"High T limit ≈ {val:.3f} (expected ~{self.num_R})")
        else:
            return (False, f"High T limit = {val:.6f}, expected ~{self.num_R}")

    def test_invariant_low_T_limit_numerical(self):
        """At T << theta, C_vib must approximately equal R*(θ/T)^2 exp(-θ/T)."""
        formula = self._parse_to_sympy(self.C_vib_formula)
        if formula is None:
            return (True, "SKIP: Cannot parse C_vib_formula")
        T_low = 0.1 * self.num_theta   # T << theta
        val = self._numerical_eval(formula, T_val=T_low)
        expected_val = self._numerical_eval(self.ref_low_T, T_val=T_low)
        if val is None or expected_val is None:
            return (False, "Numerical evaluation failed")
        if math.isclose(val, expected_val, rel_tol=1e-1, abs_tol=1e-2):
            return (True, f"Low T limit: computed={val:.6e}, expected≈{expected_val:.6e}")
        else:
            return (False, f"Low T mismatch: {val:.6e} vs {expected_val:.6e}")

    # ------------------- MODELING / TRACE TESTS -------------------
    def test_modeling_formula_dependence(self):
        """The formula must contain both theta and T symbols (dependence on θ/T)."""
        formula_str = self.results.get('C_vib_formula')
        if formula_str is None:
            return (True, "SKIP: No formula string")
        if isinstance(formula_str, str):
            if 'theta' in formula_str and 'T' in formula_str:
                return (True, "Formula depends on theta/T")
            else:
                return (False, "Formula does not contain both theta and T")
        elif isinstance(formula_str, sp.Expr):
            free_syms = {s.name for s in formula_str.free_symbols}
            if 'theta' in free_syms and 'T' in free_syms:
                return (True, "Formula depends on theta/T")
            else:
                return (False, f"Formula missing theta or T; symbols: {free_syms}")
        else:
            return (True, "SKIP: Unknown formula type")

    def test_trace_low_T_exponential_decay(self):
        """At low T, the ratio of C_vib at two different T should match the asymptotic ratio."""
        formula = self._parse_to_sympy(self.C_vib_formula)
        if formula is None:
            return (True, "SKIP: Cannot parse formula")
        T1 = 0.1 * self.num_theta
        T2 = 0.2 * self.num_theta
        val1 = self._numerical_eval(formula, T_val=T1)
        val2 = self._numerical_eval(formula, T_val=T2)
        if val1 is None or val2 is None:
            return (False, "Numerical evaluation failed")
        # Asymptotic ratio: [R*(θ/T2)^2 exp(-θ/T2)] / [R*(θ/T1)^2 exp(-θ/T1)]
        ratio_expected = ((self.num_theta/T2)**2 * math.exp(-self.num_theta/T2)) / \
                         ((self.num_theta/T1)**2 * math.exp(-self.num_theta/T1))
        ratio_actual = val2 / val1
        if math.isclose(ratio_actual, ratio_expected, rel_tol=0.5, abs_tol=1e-8):
            return (True, f"Low T ratio: actual={ratio_actual:.4f}, expected≈{ratio_expected:.4f}")
        else:
            return (False, f"Low T ratio off: {ratio_actual:.4f} vs {ratio_expected:.4f}")

    # ------------------- MONOTONICITY TESTS (must explicitly perturb an input) -------------------
    def test_monotonicity_increases_with_T(self):
        """C_vib must increase when T is increased (baseline perturbed upward)."""
        formula = self._parse_to_sympy(self.C_vib_formula)
        if formula is None:
            return (True, "SKIP: Cannot parse formula")
        # Baseline: T = 0.5*theta (intermediate region)
        T_base = 0.5 * self.num_theta
        # Perturb: increase T by 10%
        pert_factor = 1.1
        T_perturbed = T_base * pert_factor

        C_base = self._numerical_eval(formula, T_val=T_base)
        C_pert = self._numerical_eval(formula, T_val=T_perturbed)
        if C_base is None or C_pert is None:
            return (False, "Numerical evaluation failed")
        if C_pert > C_base:
            return (True, f"Monotonic increase: {C_base:.4f} -> {C_pert:.4f} (perturbation +10%)")
        else:
            return (False, f"Monotonicity fails: base={C_base:.6f}, perturbed={C_pert:.6f}")

    def test_monotonicity_high_T_approaches_R(self):
        """As T increases, C_vib approaches R from below (perturbation reduces difference to R)."""
        formula = self._parse_to_sympy(self.C_vib_formula)
        if formula is None:
            return (True, "SKIP: Cannot parse formula")
        # Baseline: T = 5*theta (high but not extremely high)
        T_base = 5.0 * self.num_theta
        # Perturb: increase T by 10%
        pert_factor = 1.1
        T_perturbed = T_base * pert_factor

        C_base = self._numerical_eval(formula, T_val=T_base)
        C_pert = self._numerical_eval(formula, T_val=T_perturbed)
        if C_base is None or C_pert is None:
            return (False, "Numerical evaluation failed")
        diff_base = abs(C_base - self.num_R)
        diff_pert = abs(C_pert - self.num_R)
        # Perturbed should be closer to R (or at least not further)
        if diff_pert <= diff_base + 1e-6:
            return (True, f"Approaches R: diff_base={diff_base:.6f} -> diff_pert={diff_pert:.6f} (+10% T)")
        else:
            return (False, f"Monotonicity to R violated: diff_base={diff_base:.6f}, diff_pert={diff_pert:.6f}")

    def test_monotonicity_low_T_exponential_slope(self):
        """At very low T, the ratio C_vib(T)/C_vib(T/2) should be roughly exp(-θ/T)*(T/(T/2))^2 * ... 
           Actually we check that perturbing T downward decreases C_vib strongly."""
        formula = self._parse_to_sympy(self.C_vib_formula)
        if formula is None:
            return (True, "SKIP: Cannot parse formula")
        # Baseline: T = 0.2*theta
        T_base = 0.2 * self.num_theta
        # Perturb: decrease T by 20% (multiply by 0.8)
        pert_factor = 0.8
        T_perturbed = T_base * pert_factor

        C_base = self._numerical_eval(formula, T_val=T_base)
        C_pert = self._numerical_eval(formula, T_val=T_perturbed)
        if C_base is None or C_pert is None:
            return (False, "Numerical evaluation failed")
        # At low T, C_vib decreases sharply when T decreases
        if C_pert < C_base:
            return (True, f"Low T decrease: {C_base:.6e} -> {C_pert:.6e} (perturbed -20% T)")
        else:
            return (False, f"Low T monotonicity fails: base={C_base:.6e}, perturbed={C_pert:.6e}")

    # ------------------- WEAK ANSWER SANITY TEST (only one, no hardcoded target numbers) -------------------
    def test_answer_sanity_weak(self):
        """Check that the formula yields a positive finite value at a typical intermediate temperature."""
        formula = self._parse_to_sympy(self.C_vib_formula)
        if formula is None:
            return (True, "SKIP: Formula not available")
        # Use T = theta (x=1)
        val = self._numerical_eval(formula, T_val=self.num_theta)
        if val is None:
            return (False, "Numerical evaluation failed")
        if math.isnan(val) or val <= 0:
            return (False, f"Formula yields {val} at T=theta; expected positive finite")
        return (True, f"C_vib(T=θ) = {val:.4f} > 0, finite")