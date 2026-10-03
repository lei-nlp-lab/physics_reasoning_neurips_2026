import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Thermodynamics checker for the problem:
    (a) Derive cp - cv = [p + (∂U/∂V)_T] (∂V/∂T)_p
    (b) Compute cp - cv for a van der Waals gas and show limit V→∞ gives ideal gas result.
    """

    def __init__(self, solution_instance):
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}

        # Result keys expected from solving pseudocode; use None defaults
        self.cp_cv_general = self.results.get('cp_cv_general', None)
        self.cp_cv_vdw = self.results.get('cp_cv_vdw', None)
        self.cp_cv_limit = self.results.get('cp_cv_limit', None)

        # Symbolic variables for internal re-derivation
        self.T, self.p, self.V, self.R, self.a, self.b = sp.symbols('T p V R a b')
        self.U = sp.Function('U')(self.T, self.V)

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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs

    # ========== Interface tests ==========

    def test_interface_general_result_exists(self):
        if self.cp_cv_general is None:
            return False, "SKIP: cp_cv_general is missing (None)."
        if not isinstance(self.cp_cv_general, (str, sp.Basic)):
            return False, f"cp_cv_general has type {type(self.cp_cv_general).__name__}, expected str/sympy."
        return True, "cp_cv_general present and of valid type."

    def test_interface_vdw_result_exists(self):
        if self.cp_cv_vdw is None:
            return False, "SKIP: cp_cv_vdw is missing (None)."
        if not isinstance(self.cp_cv_vdw, (str, sp.Basic)):
            return False, f"cp_cv_vdw has type {type(self.cp_cv_vdw).__name__}, expected str/sympy."
        return True, "cp_cv_vdw present and of valid type."

    def test_interface_limit_result_exists(self):
        if self.cp_cv_limit is None:
            return False, "SKIP: cp_cv_limit is missing (None)."
        if not isinstance(self.cp_cv_limit, (str, sp.Basic, int, float)):
            return False, f"cp_cv_limit has type {type(self.cp_cv_limit).__name__}, expected string/sympy/number."
        return True, "cp_cv_limit present and of valid type."

    # ========== Invariant / theorem-level tests ==========

    def test_invariant_general_form_structure(self):
        if self.cp_cv_general is None:
            return False, "SKIP: cp_cv_general not available."
        expr = self.cp_cv_general
        if isinstance(expr, str):
            # Check for required symbols
            required = ['p', 'U', 'V', 'T']
            missing = [s for s in required if s not in expr]
            if missing:
                return False, f"General expression missing symbols: {missing}"
            # Check product-like structure (two factors, one in brackets)
            if '(' not in expr or ')' not in expr:
                return False, "General expression does not contain factorised form."
            return True, "General expression contains required symbols and structure."
        else:
            # sympy expression: check free symbols
            free = expr.free_symbols
            expected = {sp.Symbol('p'), sp.Symbol('U'), sp.Symbol('V'), sp.Symbol('T')}
            if not expected.issubset(free):
                missing = expected - free
                return False, f"General expression missing symbols: {missing}"
            return True, "General expression contains required symbols."

    def test_invariant_vdw_positive_denominator(self):
        if self.cp_cv_vdw is None:
            return False, "SKIP: cp_cv_vdw not available."
        # Use typical numeric values
        R_val, a_val, b_val, T_val, V_val = 8.314, 1.4, 0.037, 300.0, 0.1
        cp_val = self._evaluate_vdw(self.cp_cv_vdw, R_val, a_val, b_val, T_val, V_val)
        if cp_val is None:
            return False, "Evaluation failed."
        if not (cp_val > R_val):
            return False, f"cp‑cv = {cp_val:.4f} is not greater than R ({R_val})."
        return True, f"cp‑cv = {cp_val:.4f} > R, as expected for finite V."

    # ========== Modeling / trace tests ==========

    def test_modeling_vdw_derivation_sym(self):
        """
        Re-derive cp_cv for van der Waals inside the checker and compare with stored.
        """
        if self.cp_cv_vdw is None:
            return False, "SKIP: cp_cv_vdw not available."

        # (∂p/∂T)_V from van der Waals
        p_expr = self.R * self.T / (self.V - self.b) - self.a / self.V**2
        dp_dT_V = sp.diff(p_expr, self.T).simplify()  # R/(V-b)

        # Given: p + dU/dV = T * dp/dT  →  dU/dV = T * dp/dT - p
        dU_dV_T = self.T * dp_dT_V - p_expr

        # Implicit derivative (∂V/∂T)_p from EOS
        f = (p_expr + self.a / self.V**2) * (self.V - self.b) - self.R * self.T
        df_dT = sp.diff(f, self.T)
        df_dV = sp.diff(f, self.V)
        dV_dT_p = -df_dT / df_dV

        # General cp - cv = T * (dp/dT)_V * (dV/dT)_p
        cp_cv_derived = sp.simplify(self.T * dp_dT_V * dV_dT_p)

        # Compare with stored expression
        stored = self.cp_cv_vdw
        if isinstance(stored, str):
            try:
                stored_expr = sp.parse_expr(stored)
            except Exception:
                return False, "Could not parse stored string."
        else:
            stored_expr = stored

        diff = sp.simplify(cp_cv_derived - stored_expr)
        if diff == 0:
            return True, "Derived cp_cv_vdw matches stored symbolically."
        # Fallback numeric comparison
        subst = {self.R: 8.314, self.a: 1.4, self.b: 0.037, self.T: 300, self.V: 0.1}
        try:
            val1 = cp_cv_derived.subs(subst).evalf()
            val2 = stored_expr.subs(subst).evalf()
            if abs(val1 - val2) < 1e-8:
                return True, "Derived and stored expressions agree numerically (diff < 1e-8)."
            else:
                return False, f"Derived={val1:.6f}, stored={val2:.6f}, diff={val1-val2:.6e}"
        except Exception as e:
            return False, f"Numeric fallback failed: {e}"

    def test_modeling_limit_convergence(self):
        """
        Take symbolic limit V→∞ of cp_cv_vdw and compare with stored limit.
        """
        if self.cp_cv_vdw is None or self.cp_cv_limit is None:
            return False, "SKIP: one of required results missing."

        # Convert vdw expression to sympy
        if isinstance(self.cp_cv_vdw, str):
            try:
                expr = sp.parse_expr(self.cp_cv_vdw)
            except Exception:
                return False, "Could not parse vdw string."
        else:
            expr = self.cp_cv_vdw

        limit_val = sp.limit(expr, self.V, sp.oo)

        # Convert stored limit
        if isinstance(self.cp_cv_limit, str):
            try:
                expected = sp.parse_expr(self.cp_cv_limit)
            except Exception:
                expected = sp.symbols(self.cp_cv_limit)
        else:
            expected = sp.sympify(self.cp_cv_limit)

        diff = sp.simplify(limit_val - expected)
        if diff == 0:
            return True, "Limit V→∞ matches stored limit symbolically."
        # Numeric fallback with large V
        subst = {self.R: 8.314, self.a: 1.4, self.b: 0.037, self.T: 300, self.V: 1000}
        try:
            val = expr.subs(subst).evalf()
            expected_num = expected.subs({self.R: 8.314}).evalf()
            if math.isclose(val, expected_num, rel_tol=1e-3):
                return True, f"Large V value {val:.4f} ≈ expected {expected_num:.4f}."
            else:
                return False, f"Limit gave {val:.4f}, expected ~{expected_num:.4f}."
        except Exception:
            return False, "Could not verify limit numerically."

    # ========== Monotonicity tests ==========

    def test_monotonicity_cp_cv_decreases_with_volume(self):
        """
        Perturb V by +10%: cp_cv should decrease.
        """
        if self.cp_cv_vdw is None:
            return False, "SKIP: cp_cv_vdw not available."

        R_num, a_num, b_num, T_num = 8.314, 1.4, 0.037, 300.0
        V_base = 0.2
        V_pert = V_base * 1.10  # +10%

        try:
            cp_base = self._evaluate_vdw(self.cp_cv_vdw, R_num, a_num, b_num, T_num, V_base)
            cp_pert = self._evaluate_vdw(self.cp_cv_vdw, R_num, a_num, b_num, T_num, V_pert)
        except Exception as e:
            return False, f"Evaluation failed: {e}"

        if cp_base is None or cp_pert is None:
            return False, "Evaluation returned None."

        if not (cp_base > R_num and cp_pert > R_num):
            return False, "cp_cv is not greater than R in both cases."
        if cp_base <= cp_pert:
            return False, f"cp_cv did not decrease: baseline={cp_base:.4f}, perturbed={cp_pert:.4f}."
        return True, f"cp_cv decreased from {cp_base:.4f} to {cp_pert:.4f} after +10% volume."

    # ========== Weak answer sanity (at most one method) ==========

    def test_answer_sanity_weak(self):
        """
        Single weak sanity check: general expression non‑trivial, vdw expression non‑trivial,
        limit is positive.
        """
        any_fail = False
        msgs = []

        # 1) General expression present and not tiny
        if self.cp_cv_general is None:
            msgs.append("cp_cv_general missing.")
            any_fail = True
        else:
            s = str(self.cp_cv_general)
            if len(s) < 10:
                msgs.append("General expression string too short.")
                any_fail = True
            else:
                msgs.append("General expression present.")

        # 2) vdw expression present and not tiny
        if self.cp_cv_vdw is None:
            msgs.append("cp_cv_vdw missing.")
            any_fail = True
        else:
            s = str(self.cp_cv_vdw)
            if len(s) < 10:
                msgs.append("vdW expression string too short.")
                any_fail = True
            else:
                msgs.append("vdW expression present.")

        # 3) Limit is positive (or symbolic 'R')
        if self.cp_cv_limit is None:
            msgs.append("cp_cv_limit missing.")
            any_fail = True
        else:
            try:
                val = float(self.cp_cv_limit)
                if val <= 0:
                    msgs.append("Limit is not positive.")
                    any_fail = True
                else:
                    msgs.append("Limit is positive number.")
            except (ValueError, TypeError):
                s = str(self.cp_cv_limit).strip()
                if s == 'R':
                    msgs.append("Limit is symbol R (positive).")
                else:
                    msgs.append(f"Limit is not a clear positive: {s}")

        return (not any_fail), "; ".join(msgs)

    # ========== Helper ==========

    def _evaluate_vdw(self, expr, R_val, a_val, b_val, T_val, V_val):
        """Try to evaluate the given cp_cv_vdw expression (string or sympy) numerically."""
        if isinstance(expr, str):
            try:
                parsed = sp.parse_expr(expr)
            except Exception:
                return None
        else:
            parsed = expr

        subst = {self.R: R_val, self.a: a_val, self.b: b_val, self.T: T_val, self.V: V_val}
        try:
            res = parsed.subs(subst).evalf()
            return float(res)
        except Exception:
            return None