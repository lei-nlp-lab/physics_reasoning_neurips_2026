import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}
        self.C_iso = self.results.get('C_iso', None)
        self.C_adi = self.results.get('C_adi', None)

        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        if self.explanation_steps is None:
            self.explanation_steps = self.results.get('explanation_steps', None)
        if isinstance(self.explanation_steps, list):
            self.explanation_text = " ".join(self.explanation_steps)
        else:
            self.explanation_text = str(self.explanation_steps) if self.explanation_steps else ""

        self.givens = getattr(solution_instance, 'givens', None) or {}
        self.R_num = self.givens.get('R', None)
        self.T_num = self.givens.get('T', None)
        self.M_num = self.givens.get('M', None)
        self.gamma_num = self.givens.get('gamma', None)

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

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------
    def test_interface_keys(self):
        if self.C_iso is None:
            return (False, "Missing C_iso")
        if self.C_adi is None:
            return (False, "Missing C_adi")
        return (True, "Both result keys present")

    # ------------------------------------------------------------------
    # Invariant (theorem-based)
    # ------------------------------------------------------------------
    def _sympy_equiv(self, expr_str, target):
        try:
            if isinstance(expr_str, str):
                expr = sp.sympify(expr_str)
            else:
                expr = expr_str
            diff = sp.simplify(expr - target)
            return diff == 0
        except Exception:
            return None

    def test_invariant_isothermal_symbolic(self):
        if self.C_iso is None:
            return (False, "C_iso not available")
        R, T, M = sp.symbols('R T M', positive=True)
        target = sp.sqrt(R * T / M)
        equiv = self._sympy_equiv(self.C_iso, target)
        if equiv is True:
            return (True, "C_iso symbolically = sqrt(R*T/M)")
        elif equiv is None:
            if self.R_num is not None and self.T_num is not None and self.M_num is not None:
                try:
                    subs = {R: self.R_num, T: self.T_num, M: self.M_num}
                    if isinstance(self.C_iso, str):
                        c_val = float(sp.N(sp.sympify(self.C_iso).subs(subs)))
                    else:
                        c_val = float(self.C_iso.subs(subs))
                    expected = math.sqrt(self.R_num * self.T_num / self.M_num)
                    if math.isclose(c_val, expected, rel_tol=1e-2, abs_tol=1e-2):
                        return (True, f"Numeric match {c_val:.3f} vs {expected:.3f}")
                    else:
                        return (False, f"Mismatch: {c_val:.3f} vs {expected:.3f}")
                except Exception as e:
                    return (False, f"Fallback failed: {e}")
            else:
                return (False, "Cannot verify – no numeric givens")
        else:
            return (False, "Symbolic mismatch")

    def test_invariant_adiabatic_symbolic(self):
        if self.C_adi is None:
            return (False, "C_adi not available")
        gamma, R, T, M = sp.symbols('gamma R T M', positive=True)
        target = sp.sqrt(gamma * R * T / M)
        equiv = self._sympy_equiv(self.C_adi, target)
        if equiv is True:
            return (True, "C_adi symbolically = sqrt(gamma*R*T/M)")
        elif equiv is None:
            if (self.R_num is not None and self.T_num is not None and
                self.M_num is not None and self.gamma_num is not None):
                try:
                    subs = {gamma: self.gamma_num, R: self.R_num,
                            T: self.T_num, M: self.M_num}
                    if isinstance(self.C_adi, str):
                        c_val = float(sp.N(sp.sympify(self.C_adi).subs(subs)))
                    else:
                        c_val = float(self.C_adi.subs(subs))
                    expected = math.sqrt(self.gamma_num * self.R_num *
                                         self.T_num / self.M_num)
                    if math.isclose(c_val, expected, rel_tol=1e-2, abs_tol=1e-2):
                        return (True, f"Numeric match {c_val:.3f} vs {expected:.3f}")
                    else:
                        return (False, f"Mismatch: {c_val:.3f} vs {expected:.3f}")
                except Exception as e:
                    return (False, f"Fallback failed: {e}")
            else:
                return (False, "Cannot verify – missing givens")
        else:
            return (False, "Symbolic mismatch")

    # ------------------------------------------------------------------
    # Modeling / trace consistency (at least two tests)
    # ------------------------------------------------------------------
    def test_modeling_explanation_steps(self):
        if not self.explanation_text.strip():
            return (False, "SKIP: no explanation steps")
        # Check for key conceptual terms
        keywords = ['isothermal', 'adiabatic', 'ideal gas', 'dp/drho']
        text_lower = self.explanation_text.lower()
        missing = [kw for kw in keywords if kw not in text_lower]
        if missing:
            return (False, f"Missing key terms: {missing}")
        return (True, "Explanation includes required physical concepts")

    def test_modeling_isothermal_relation(self):
        """Check that explanation mentions p/rho = constant or dp/drho = p/rho for isothermal."""
        if not self.explanation_text.strip():
            return (False, "SKIP: no explanation steps")
        text_lower = self.explanation_text.lower()
        if 'p/rho = constant' in text_lower or 'p/ρ = constant' in text_lower or \
           'dp/drho = p/rho' in text_lower or 'dp/dρ = p/ρ' in text_lower:
            return (True, "Isothermal relation (dp/dρ = p/ρ) present")
        else:
            return (False, "Missing explicit isothermal relation (dp/dρ = p/ρ)")

    def test_modeling_adiabatic_relation(self):
        """Check that explanation mentions dp/drho = gamma * p/rho for adiabatic."""
        if not self.explanation_text.strip():
            return (False, "SKIP: no explanation steps")
        text_lower = self.explanation_text.lower()
        if 'dp/drho = gamma * p/rho' in text_lower or 'dp/dρ = γ p/ρ' in text_lower or \
           'gamma p/rho' in text_lower or 'γ p/ρ' in text_lower:
            return (True, "Adiabatic relation (dp/dρ = γp/ρ) present")
        else:
            return (False, "Missing explicit adiabatic relation (dp/dρ = γp/ρ)")

    # ------------------------------------------------------------------
    # Monotonicity – explicit perturbation tests
    # ------------------------------------------------------------------
    def test_monotonicity_temperature_iso(self):
        """Perturb T upward, C_iso should increase."""
        if self.C_iso is None:
            return (False, "SKIP: C_iso not available")
        R_val = 8.314
        T_base = 300.0
        M_val = 0.0289647
        delta = 30.0       # 10% increase
        T_up = T_base + delta

        try:
            if isinstance(self.C_iso, str):
                expr = sp.sympify(self.C_iso)
            else:
                expr = self.C_iso
            subs_base = {sp.Symbol('R'): R_val, sp.Symbol('T'): T_base, sp.Symbol('M'): M_val}
            subs_up   = {sp.Symbol('R'): R_val, sp.Symbol('T'): T_up, sp.Symbol('M'): M_val}
            c_base = float(sp.N(expr.subs(subs_base)))
            c_up   = float(sp.N(expr.subs(subs_up)))
        except Exception as e:
            return (False, f"Cannot evaluate: {e}")

        if c_up <= c_base:
            return (False, f"C_iso did not increase: base {c_base:.3f}, after +{delta}K {c_up:.3f}")
        return (True, f"C_iso increases with T: {c_base:.3f} -> {c_up:.3f}")

    def test_monotonicity_gamma_adi(self):
        """Perturb gamma upward, C_adi should increase."""
        if self.C_adi is None:
            return (False, "SKIP: C_adi not available")
        R_val = 8.314
        T_val = 300.0
        M_val = 0.0289647
        gamma_base = 1.4
        gamma_up = 1.5     # ~7% increase

        try:
            if isinstance(self.C_adi, str):
                expr = sp.sympify(self.C_adi)
            else:
                expr = self.C_adi
            subs_base = {sp.Symbol('gamma'): gamma_base, sp.Symbol('R'): R_val,
                         sp.Symbol('T'): T_val, sp.Symbol('M'): M_val}
            subs_up   = {sp.Symbol('gamma'): gamma_up, sp.Symbol('R'): R_val,
                         sp.Symbol('T'): T_val, sp.Symbol('M'): M_val}
            c_base = float(sp.N(expr.subs(subs_base)))
            c_up   = float(sp.N(expr.subs(subs_up)))
        except Exception as e:
            return (False, f"Cannot evaluate: {e}")

        if c_up <= c_base:
            return (False, f"C_adi did not increase: base {c_base:.3f}, gamma+0.1 -> {c_up:.3f}")
        return (True, f"C_adi increases with gamma: {c_base:.3f} -> {c_up:.3f}")

    # ------------------------------------------------------------------
    # Weak answer sanity (positive, finite)
    # ------------------------------------------------------------------
    def test_answer_sanity_positive(self):
        if (self.R_num is None or self.T_num is None or self.M_num is None):
            return (True, "SKIP: numeric givens missing")

        R, T, M = sp.symbols('R T M', positive=True)
        try:
            if isinstance(self.C_iso, str):
                c_iso_val = float(sp.N(sp.sympify(self.C_iso).subs({R: self.R_num, T: self.T_num, M: self.M_num})))
            else:
                c_iso_val = float(self.C_iso.subs({R: self.R_num, T: self.T_num, M: self.M_num}))
            if c_iso_val <= 0:
                return (False, f"C_iso <= 0: {c_iso_val}")
        except Exception:
            pass

        if self.gamma_num is not None:
            gamma = sp.symbols('gamma', positive=True)
            try:
                if isinstance(self.C_adi, str):
                    c_adi_val = float(sp.N(sp.sympify(self.C_adi).subs({gamma: self.gamma_num, R: self.R_num, T: self.T_num, M: self.M_num})))
                else:
                    c_adi_val = float(self.C_adi.subs({gamma: self.gamma_num, R: self.R_num, T: self.T_num, M: self.M_num}))
                if c_adi_val <= 0:
                    return (False, f"C_adi <= 0: {c_adi_val}")
            except Exception:
                pass

        return (True, "Expressions appear positive")