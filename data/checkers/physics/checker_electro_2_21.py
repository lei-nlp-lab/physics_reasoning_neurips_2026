import math
import sympy as sp

class PhysicsChecker:
    """
    Auditor for rotating polarized dielectric cylinder problem.
    Checks interface, invariants, modeling, monotonicity (with explicit perturbation),
    and one weak answer sanity.
    """

    def __init__(self, solution_instance):
        # Extract givens safely (default None)
        self.R = getattr(solution_instance, 'R', None)
        self.a = getattr(solution_instance, 'a', None)
        self.epsilon0 = getattr(solution_instance, 'epsilon0', None)
        self.mu0 = getattr(solution_instance, 'mu0', None)
        self.w = getattr(solution_instance, 'w', None)
        self.c = getattr(solution_instance, 'c', None)

        # Extract results dict
        results = getattr(solution_instance, 'results', {}) or {}
        self.E_inside = results.get('E_inside', None)
        self.E_outside = results.get('E_outside', None)
        self.B_inside = results.get('B_inside', None)
        self.B_outside = results.get('B_outside', None)
        self.U_E = results.get('U_E', None)
        self.U_B = results.get('U_B', None)
        self.U_total = results.get('U_total', None)

        # Optional explanation steps (for context)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # --------------------------------------------------
    # Numeric evaluation helpers
    # --------------------------------------------------
    def _is_numeric(self, val):
        try:
            float(val)
            return True
        except:
            return False

    def _eval_numeric(self, expr, subs_dict=None):
        """Try to evaluate expression numerically, with optional substitution."""
        if expr is None:
            return None
        if self._is_numeric(expr):
            return float(expr)
        if subs_dict is None:
            subs_dict = {}
        # Build substitution for known givens
        subs = {}
        for name, sym in [('R', sp.Symbol('R')), ('a', sp.Symbol('a')),
                          ('epsilon0', sp.Symbol('epsilon0')), ('mu0', sp.Symbol('mu0')),
                          ('w', sp.Symbol('w')), ('c', sp.Symbol('c'))]:
            val = getattr(self, name, None)
            if val is not None:
                try:
                    subs[sym] = float(val)
                except:
                    subs[sym] = val
        subs.update(subs_dict)
        try:
            return float(expr.subs(subs))
        except:
            return None

    def _get_field_numeric(self, field, r_val):
        """Evaluate field at a given r value."""
        subs = {sp.Symbol('r'): r_val}
        return self._eval_numeric(field, subs)

    # --------------------------------------------------
    # Interface tests
    # --------------------------------------------------
    def test_interface_required_results(self):
        required = ['E_inside', 'E_outside', 'B_inside', 'B_outside',
                    'U_E', 'U_B', 'U_total']
        missing = [k for k in required if getattr(self, k, None) is None]
        if missing:
            return (False, f"Missing results: {missing}")
        return (True, "All required results present.")

    def test_interface_givens_positive(self):
        givens = {'R': self.R, 'a': self.a, 'epsilon0': self.epsilon0,
                  'mu0': self.mu0, 'w': self.w, 'c': self.c}
        for name, val in givens.items():
            if val is None:
                return (False, f"Given {name} is missing.")
            try:
                if float(val) <= 0:
                    return (False, f"Given {name} not positive ({val}).")
            except:
                return (False, f"Given {name} cannot convert to number.")
        return (True, "All givens positive.")

    # --------------------------------------------------
    # Invariant tests (theorem-based, use isclose)
    # --------------------------------------------------
    def test_invariant_E_inside_sign(self):
        if self.E_inside is None:
            return (False, "SKIP: E_inside missing.")
        try:
            r_test = float(self.R) / 2.0
        except:
            return (False, "SKIP: cannot compute r.")
        E = self._get_field_numeric(self.E_inside, r_test)
        if E is None:
            return (False, "SKIP: cannot evaluate E_inside.")
        if E > 0:
            return (False, f"E_inside at r=R/2 = {E} > 0, expected negative (inward).")
        return (True, f"E_inside({r_test}) = {E} < 0.")

    def test_invariant_E_inside_linear(self):
        if self.E_inside is None:
            return (False, "SKIP: E_inside missing.")
        try:
            r1 = float(self.R) / 4.0
            r2 = float(self.R) / 2.0
        except:
            return (False, "SKIP: cannot compute r values.")
        E1 = self._get_field_numeric(self.E_inside, r1)
        E2 = self._get_field_numeric(self.E_inside, r2)
        if E1 is None or E2 is None:
            return (False, "SKIP: cannot evaluate E_inside.")
        # |E| should be proportional to r
        ratio = abs(E2)/abs(E1) if abs(E1) > 1e-15 else None
        expected_r = r2/r1
        if ratio is None or not math.isclose(ratio, expected_r, rel_tol=1e-2):
            return (False, f"E not linear: |E| ratio = {ratio}, expected {expected_r}.")
        return (True, "E_inside is linear in r.")

    def test_invariant_E_outside_zero(self):
        if self.E_outside is None:
            return (False, "SKIP: E_outside missing.")
        try:
            r_test = 2.0 * float(self.R)
        except:
            return (False, "SKIP: cannot compute r.")
        E = self._get_field_numeric(self.E_outside, r_test)
        if E is None:
            return (False, "SKIP: cannot evaluate E_outside.")
        if not math.isclose(E, 0.0, abs_tol=1e-3):
            return (False, f"E_outside({r_test}) = {E}, expected 0.")
        return (True, "E_outside is zero.")

    def test_invariant_B_inside_sign(self):
        if self.B_inside is None:
            return (False, "SKIP: B_inside missing.")
        try:
            r_test = float(self.R) / 2.0
        except:
            return (False, "SKIP: cannot compute r.")
        B = self._get_field_numeric(self.B_inside, r_test)
        if B is None:
            return (False, "SKIP: cannot evaluate B_inside.")
        if B < 0:
            return (False, f"B_inside({r_test}) = {B} < 0, expected positive (z-direction).")
        return (True, f"B_inside({r_test}) = {B} > 0.")

    def test_invariant_B_inside_quadratic(self):
        if self.B_inside is None:
            return (False, "SKIP: B_inside missing.")
        try:
            r1 = float(self.R) / 4.0
            r2 = float(self.R) / 2.0
        except:
            return (False, "SKIP: cannot compute r values.")
        B1 = self._get_field_numeric(self.B_inside, r1)
        B2 = self._get_field_numeric(self.B_inside, r2)
        if B1 is None or B2 is None:
            return (False, "SKIP: cannot evaluate B_inside.")
        # B ∝ r^2
        ratio = B2/B1 if abs(B1) > 1e-15 else None
        expected = (r2/r1)**2
        if ratio is None or not math.isclose(ratio, expected, rel_tol=1e-2):
            return (False, f"B not quadratic: ratio = {ratio}, expected {expected}.")
        return (True, "B_inside is quadratic in r.")

    def test_invariant_B_outside_zero(self):
        if self.B_outside is None:
            return (False, "SKIP: B_outside missing.")
        try:
            r_test = 2.0 * float(self.R)
        except:
            return (False, "SKIP: cannot compute r.")
        B = self._get_field_numeric(self.B_outside, r_test)
        if B is None:
            return (False, "SKIP: cannot evaluate B_outside.")
        if not math.isclose(B, 0.0, abs_tol=1e-3):
            return (False, f"B_outside({r_test}) = {B}, expected 0.")
        return (True, "B_outside is zero.")

    # --------------------------------------------------
    # Modeling / trace tests (based on explanation steps, re-derive exact formulas)
    # --------------------------------------------------
    def test_modeling_U_E_formula(self):
        if self.U_E is None:
            return (False, "SKIP: U_E missing.")
        if any(v is None for v in [self.a, self.R, self.epsilon0]):
            return (False, "SKIP: missing required givens for formula check.")
        try:
            a_num = float(self.a)
            R_num = float(self.R)
            e0_num = float(self.epsilon0)
        except:
            return (False, "SKIP: cannot convert givens to numbers.")
        expected = math.pi * a_num**2 * R_num**4 / (4.0 * e0_num)
        ue = self._eval_numeric(self.U_E)
        if ue is None:
            return (False, "SKIP: cannot evaluate U_E numerically.")
        if not math.isclose(ue, expected, rel_tol=1e-2):
            return (False, f"U_E = {ue}, expected {expected}.")
        return (True, "U_E matches formula.")

    def test_modeling_U_B_formula(self):
        if self.U_B is None:
            return (False, "SKIP: U_B missing.")
        if any(v is None for v in [self.a, self.mu0, self.w, self.R]):
            return (False, "SKIP: missing required givens.")
        try:
            a_num = float(self.a)
            mu0_num = float(self.mu0)
            w_num = float(self.w)
            R_num = float(self.R)
        except:
            return (False, "SKIP: cannot convert givens.")
        expected = math.pi * mu0_num * a_num**2 * w_num**2 * R_num**6 / 6.0
        ub = self._eval_numeric(self.U_B)
        if ub is None:
            return (False, "SKIP: cannot evaluate U_B.")
        if not math.isclose(ub, expected, rel_tol=1e-2):
            return (False, f"U_B = {ub}, expected {expected}.")
        return (True, "U_B matches formula.")

    def test_modeling_U_total_consistency(self):
        if any(x is None for x in [self.U_total, self.U_E, self.U_B]):
            return (False, "SKIP: missing energy results.")
        ut = self._eval_numeric(self.U_total)
        ue = self._eval_numeric(self.U_E)
        ub = self._eval_numeric(self.U_B)
        if any(x is None for x in [ut, ue, ub]):
            return (False, "SKIP: cannot evaluate energies.")
        if not math.isclose(ut, ue + ub, rel_tol=1e-2):
            return (False, f"U_total = {ut}, but U_E+U_B = {ue + ub}.")
        return (True, "U_total = U_E + U_B.")

    # --------------------------------------------------
    # Monotonicity tests (explicit perturbation of input, compare output)
    # --------------------------------------------------
    def test_monotonicity_U_E_increases_with_R(self):
        """
        Increase R by 10% and check that U_E (evaluated from its expression)
        increases accordingly.
        """
        if self.U_E is None:
            return (False, "SKIP: U_E missing.")
        if self.R is None:
            return (False, "SKIP: R missing.")
        try:
            R0 = float(self.R)
        except:
            return (False, "SKIP: R not numeric.")
        # Baseline U_E (using current givens)
        U0 = self._eval_numeric(self.U_E)
        if U0 is None:
            return (False, "SKIP: cannot evaluate baseline U_E.")
        # New R (10% increase)
        R1 = 1.1 * R0
        # Evaluate U_E with new R (substitute R1 into expression)
        subs_new = {sp.Symbol('R'): R1}
        U1 = self._eval_numeric(self.U_E, subs_new)
        if U1 is None:
            return (False, "SKIP: cannot evaluate U_E with perturbed R.")
        if U1 <= U0:
            return (False, f"U_E did not increase when R increased from {R0} to {R1}: baseline {U0}, perturbed {U1}.")
        return (True, f"U_E increased from {U0} to {U1} for R increase.")

    def test_monotonicity_U_B_increases_with_w(self):
        """
        Increase ω by 10% and check that U_B (evaluated from its expression)
        increases (quadratically).
        """
        if self.U_B is None:
            return (False, "SKIP: U_B missing.")
        if self.w is None:
            return (False, "SKIP: ω missing.")
        try:
            w0 = float(self.w)
        except:
            return (False, "SKIP: ω not numeric.")
        U0 = self._eval_numeric(self.U_B)
        if U0 is None:
            return (False, "SKIP: cannot evaluate baseline U_B.")
        w1 = 1.1 * w0
        subs_new = {sp.Symbol('w'): w1}
        U1 = self._eval_numeric(self.U_B, subs_new)
        if U1 is None:
            return (False, "SKIP: cannot evaluate U_B with perturbed ω.")
        if U1 <= U0:
            return (False, f"U_B did not increase when ω increased from {w0} to {w1}: baseline {U0}, perturbed {U1}.")
        return (True, f"U_B increased from {U0} to {U1} for ω increase.")

    def test_monotonicity_U_total_increases_with_w(self):
        """
        Increase ω by 10% and check that total energy U_total increases
        (since U_E constant, U_B increases).
        """
        if self.U_total is None:
            return (False, "SKIP: U_total missing.")
        if self.w is None:
            return (False, "SKIP: ω missing.")
        try:
            w0 = float(self.w)
        except:
            return (False, "SKIP: ω not numeric.")
        U0 = self._eval_numeric(self.U_total)
        if U0 is None:
            return (False, "SKIP: cannot evaluate baseline U_total.")
        w1 = 1.1 * w0
        subs_new = {sp.Symbol('w'): w1}
        U1 = self._eval_numeric(self.U_total, subs_new)
        if U1 is None:
            return (False, "SKIP: cannot evaluate U_total with perturbed ω.")
        if U1 <= U0:
            return (False, f"U_total did not increase when ω increased from {w0} to {w1}: baseline {U0}, perturbed {U1}.")
        return (True, f"U_total increased from {U0} to {U1} for ω increase.")

    # --------------------------------------------------
    # One weak answer sanity (no isclose, no re-derive formulas for final answer)
    # --------------------------------------------------
    def test_answer_sanity_energies_nonnegative(self):
        """All energies should be non-negative (numeric values)."""
        for name in ['U_E', 'U_B', 'U_total']:
            val = getattr(self, name, None)
            if val is None:
                continue
            num = self._eval_numeric(val)
            if num is not None and num < 0:
                return (False, f"{name} is negative ({num}).")
        return (True, "All available energies are non-negative.")

    # --------------------------------------------------
    # Main runner
    # --------------------------------------------------
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