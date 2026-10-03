import math
import traceback
import sympy as sp

class PhysicsChecker:
    """
    Checks a solution for the three-body gravitational problem.
    """
    def __init__(self, solution_instance):
        givens = getattr(solution_instance, 'givens', {}) or {}
        self.m1 = givens.get('m1', None)
        self.m2 = givens.get('m2', None)
        self.m3 = givens.get('m3', None)
        self.d = givens.get('d', None)
        self.G = givens.get('G', None)

        results = getattr(solution_instance, 'results', {}) or {}
        self.eq_motion = results.get('eq_motion', None)
        self.omega = results.get('omega', None)

        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Build symbolic substitution dictionary for later evaluation
        self.symbol_vals = {}
        if self.m1 is not None:
            self.symbol_vals[sp.Symbol('m1')] = self.m1
        if self.m2 is not None:
            self.symbol_vals[sp.Symbol('m2')] = self.m2
        if self.m3 is not None:
            self.symbol_vals[sp.Symbol('m3')] = self.m3
        if self.d is not None:
            self.symbol_vals[sp.Symbol('d')] = self.d
        if self.G is not None:
            self.symbol_vals[sp.Symbol('G')] = self.G

    def _eval_omega_numeric(self, subs_override=None):
        """Evaluate the omega expression numerically with optional override substitutions."""
        if self.omega is None:
            return None
        # Convert to sympy expression if string
        expr = self.omega
        if isinstance(expr, str):
            try:
                expr = sp.sympify(expr)
            except Exception:
                return None
        if not isinstance(expr, sp.Basic):
            try:
                return float(expr)
            except (TypeError, ValueError):
                return None
        # Build substitution dictionary
        subs = self.symbol_vals.copy()
        if subs_override:
            subs.update(subs_override)
        try:
            num_val = expr.subs(subs).evalf()
            return float(num_val)
        except Exception:
            return None

    # -------- Interface tests ----------
    def test_interface_required_keys(self):
        if self.eq_motion is None or self.omega is None:
            return False, "Missing one or both required keys ('eq_motion','omega')"
        return True, "Both required keys present"

    def test_interface_eq_motion_is_string(self):
        if not isinstance(self.eq_motion, str):
            return False, f"eq_motion should be a string, got {type(self.eq_motion)}"
        return True, "eq_motion is a string"

    def test_interface_omega_is_evaluable(self):
        num = self._eval_omega_numeric()
        if num is None:
            return False, "omega cannot be evaluated to a number"
        return True, f"omega evaluates to {num}"

    # -------- Theorem-based invariant tests ----------
    def test_invariant_omega_squared_times_d_cubed(self):
        """Check ω² d³ = G M numerically (using isclose)."""
        omega_num = self._eval_omega_numeric()
        if omega_num is None:
            return False, "SKIP: omega not evaluable"
        if None in (self.m1, self.m2, self.m3, self.d, self.G):
            return False, "SKIP: givens (m1,m2,m3,d,G) not all provided"
        M = self.m1 + self.m2 + self.m3
        lhs = omega_num**2 * self.d**3
        rhs = self.G * M
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return True, f"ω²d³ = {lhs} ≈ G M = {rhs}"
        else:
            return False, f"ω²d³ = {lhs}, G M = {rhs}, difference too large"

    def test_invariant_omega_symbolic_form(self):
        """
        Symbolic check: omega expression must equal sqrt(G*(m1+m2+m3)/d**3).
        Handles cases where total mass is named M in the expression.
        """
        if self.omega is None:
            return False, "SKIP: no omega expression available"
        try:
            expr = self.omega
            if isinstance(expr, str):
                expr = sp.sympify(expr)
            # Define expected symbolic form (using generic symbols)
            G_s, m1_s, m2_s, m3_s, d_s = sp.symbols('G_s m1_s m2_s m3_s d_s')
            expected = sp.sqrt(G_s*(m1_s + m2_s + m3_s)/d_s**3)
            # Try to match by substituting variable names
            # First try to rename variables in the expression to the generic ones
            # This is fragile; instead we check if (omega**2 * d**3 / G) equals (m1+m2+m3)
            # by using the actual symbolic expression itself.
            # We'll extract the set of free symbols:
            free_syms = expr.free_symbols
            # If the expression is exactly sqrt(G*(m1+m2+m3)/d**3) with the same symbols,
            # then the difference should be zero.
            diff = sp.simplify(expr - expected)
            if diff == 0:
                return True, "Omega expression exactly matches sqrt(G*(m1+m2+m3)/d**3)"
            # If not, maybe it uses a symbol M for total mass.
            # Try substituting M = m1+m2+m3
            M_sym = [s for s in free_syms if str(s).upper() == 'M']
            if M_sym:
                M_s = M_sym[0]
                expr_sub = expr.subs(M_s, m1_s + m2_s + m3_s)
                diff2 = sp.simplify(expr_sub - expected)
                if diff2 == 0:
                    return True, "Omega expression matches after substituting M = m1+m2+m3"
            # Another try: check if omega^2 * d^3 / G equals total mass symbolically
            # (without assuming specific variable names)
            # Define a dummy total mass variable
            T = sp.Symbol('T')
            lhs_expr = expr**2 * d_s**3 / G_s
            # Substitute any existing mass variables as T? Hard.
            # Fallback: compute the expression numerically with random values to see if holds.
            # For random positive numbers, the numeric ratio should be 1.
            import random
            for _ in range(5):
                rG = random.uniform(0.1, 10)
                rm1 = random.uniform(0.1, 10)
                rm2 = random.uniform(0.1, 10)
                rm3 = random.uniform(0.1, 10)
                rd = random.uniform(0.1, 10)
                subs_test = {sp.Symbol('G'): rG, sp.Symbol('m1'): rm1, sp.Symbol('m2'): rm2,
                             sp.Symbol('m3'): rm3, sp.Symbol('d'): rd}
                # Also include symbol names like 'm_1'? Unlikely.
                num_expr = expr.subs(subs_test).evalf()
                if num_expr is None:
                    continue
                lhs_num = float(num_expr**2 * rd**3 / rG)
                rhs_num = rm1+rm2+rm3
                if not math.isclose(lhs_num, rhs_num, rel_tol=1e-2):
                    return False, f"Numeric test failed: ratio = {lhs_num/rhs_num} (should be 1)"
            return True, "Omega expression verified numerically to match sqrt(G*(m1+m2+m3)/d**3)"
        except Exception as e:
            return False, f"Symbolic/numeric invariant check crashed: {e}"

    # -------- Modeling / trace tests ----------
    def test_trace_eq_motion_contains_sum(self):
        if self.eq_motion is None:
            return False, "SKIP: eq_motion not provided"
        s = self.eq_motion
        checks = ['sum', 'G', 'm_', 'r_ij']
        missing = [c for c in checks if c not in s]
        if missing:
            return False, f"eq_motion missing terms: {missing}"
        return True, "eq_motion contains expected sum, G, m_, r_ij"

    def test_trace_explanation_uses_center_of_mass(self):
        if not self.explanation_steps:
            return False, "SKIP: no explanation_steps available"
        steps_text = ""
        if isinstance(self.explanation_steps, list):
            steps_text = " ".join(str(s) for s in self.explanation_steps)
        elif isinstance(self.explanation_steps, str):
            steps_text = self.explanation_steps
        else:
            return False, "SKIP: explanation_steps format unknown"
        keywords = ['center of mass', 'sum m_i r_i', '∑ m_i r_i']
        found = any(k in steps_text for k in keywords)
        if found:
            return True, "Explanation mentions center of mass condition"
        else:
            return False, "Explanation does not mention center of mass condition"

    def test_trace_stability_discussion(self):
        if not self.explanation_steps:
            return False, "SKIP: no explanation_steps"
        steps_text = ""
        if isinstance(self.explanation_steps, list):
            steps_text = " ".join(str(s) for s in self.explanation_steps)
        elif isinstance(self.explanation_steps, str):
            steps_text = self.explanation_steps
        else:
            return False, "SKIP: explanation_steps format unknown"
        keywords = ['stable', 'unstable', 'transverse', 'longitudinal']
        found = any(k in steps_text for k in keywords)
        if found:
            return True, "Explanation discusses stability directions"
        else:
            return False, "Explanation missing stability discussion"

    # -------- Proper monotonicity tests (explicit perturbation) ----------
    def test_monotonicity_omega_increases_with_M(self):
        """
        Perturb total mass M by increasing m1 by 10% and check that omega increases.
        Requires m1, d, G, and a symbolic omega expression that can be evaluated.
        """
        if self.omega is None:
            return False, "SKIP: no omega expression available"
        if self.d is None or self.G is None or self.m1 is None:
            return False, "SKIP: missing required givens (m1, d, G)"
        # Evaluate baseline omega
        baseline = self._eval_omega_numeric()
        if baseline is None:
            return False, "SKIP: baseline omega not evaluable"

        # Perturb m1 by 10% (increase)
        perturbed_m1 = self.m1 * 1.1
        subs_override = {sp.Symbol('m1'): perturbed_m1}
        perturbed = self._eval_omega_numeric(subs_override=subs_override)
        if perturbed is None:
            return False, "SKIP: perturbed omega not evaluable"

        if perturbed > baseline:
            return True, f"baseline ω = {baseline}, after +10% m1 → ω = {perturbed}, increased"
        else:
            return False, f"baseline ω = {baseline}, after +10% m1 → ω = {perturbed}, did not increase"

    def test_monotonicity_omega_decreases_with_d(self):
        """
        Perturb distance d by increasing 10% and check that omega decreases.
        """
        if self.omega is None:
            return False, "SKIP: no omega expression available"
        if self.d is None or self.G is None or self.m1 is None:
            return False, "SKIP: missing required givens (d, G, m1)"
        baseline = self._eval_omega_numeric()
        if baseline is None:
            return False, "SKIP: baseline omega not evaluable"

        perturbed_d = self.d * 1.1
        subs_override = {sp.Symbol('d'): perturbed_d}
        perturbed = self._eval_omega_numeric(subs_override=subs_override)
        if perturbed is None:
            return False, "SKIP: perturbed omega not evaluable"

        if perturbed < baseline:
            return True, f"baseline ω = {baseline}, after +10% d → ω = {perturbed}, decreased"
        else:
            return False, f"baseline ω = {baseline}, after +10% d → ω = {perturbed}, did not decrease"

    # -------- Weak answer sanity (no isclose, no re-derivation) ----------
    def test_answer_sanity_omega_positive(self):
        omega_num = self._eval_omega_numeric()
        if omega_num is None:
            return False, "SKIP: omega not evaluable"
        if omega_num > 0:
            return True, f"omega = {omega_num} > 0"
        else:
            return False, f"omega = {omega_num} is not positive"

    # -------- Runner ----------
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