import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Checker for cylindrical capacitor with partial dielectric insertion.
    Verifies interface, invariants, modeling, monotonicity, and weak answer sanity.
    """
    def __init__(self, solution_instance):
        # --- Givens (with None defaults) ---
        # Extract from solution_instance or its attributes
        self.a = getattr(solution_instance, 'a', None)
        self.b = getattr(solution_instance, 'b', None)
        self.L = getattr(solution_instance, 'L', None)
        self.Q = getattr(solution_instance, 'Q', None)          # for Q1, Q2
        self.V = getattr(solution_instance, 'V', None)          # for Q3
        self.epsilon = getattr(solution_instance, 'epsilon', None)  # absolute permittivity of dielectric
        self.epsilon0 = getattr(solution_instance, 'epsilon0', None) # vacuum permittivity
        self.x = getattr(solution_instance, 'x', None)           # length pulled out (vacuum part)

        # Possibly givens in a dict
        givens = getattr(solution_instance, 'givens', None) or {}
        if self.a is None:
            self.a = givens.get('a', None)
        if self.b is None:
            self.b = givens.get('b', None)
        if self.L is None:
            self.L = givens.get('L', None)
        if self.Q is None:
            self.Q = givens.get('Q', None)
        if self.V is None:
            self.V = givens.get('V', None)
        if self.epsilon is None:
            self.epsilon = givens.get('epsilon', None)
        if self.epsilon0 is None:
            self.epsilon0 = givens.get('epsilon0', None)
        if self.x is None:
            self.x = givens.get('x', None)

        # --- Results (with None defaults) ---
        results = getattr(solution_instance, 'results', None) or {}
        self.E = results.get('E', None)       # symbolic or numeric expression for E(r)
        self.C = results.get('C', None)       # capacitance (full insertion)
        self.F = results.get('F', None)       # external force to hold dielectric

        # Additional derived quantities for testing (evaluated later if possible)
        self._C_total_expr = None  # will be built if givens allow
        self._dC_dx_expr = None

    # ----------------------------------------------------------------------
    # Helper: check if a value is a sympy expression (symbolic)
    # ----------------------------------------------------------------------
    def _is_sympy(self, val):
        return isinstance(val, sp.Basic)

    # ----------------------------------------------------------------------
    # Helper: safely evaluate a sympy expression to float if all symbols are numeric
    # ----------------------------------------------------------------------
    def _eval_num(self, expr, subs_dict=None):
        """Try to evaluate a sympy expression numerically. Return None if fails."""
        if not self._is_sympy(expr):
            # if already numeric, return as float or int
            try:
                return float(expr)
            except:
                return None
        subs = {}
        # Collect free symbols
        free = expr.free_symbols
        # Map known givens
        mapping = {'a': self.a, 'b': self.b, 'L': self.L, 'Q': self.Q,
                   'V': self.V, 'epsilon': self.epsilon, 'epsilon0': self.epsilon0,
                   'x': self.x}
        for sym in free:
            name = str(sym)
            if name in mapping and mapping[name] is not None:
                subs[sym] = mapping[name]
        if subs_dict:
            subs.update(subs_dict)
        # check if all symbols have substitution
        missing = free - set(subs.keys())
        if missing:
            return None
        try:
            val = expr.subs(subs)
            if val.is_number:
                return float(val)
            else:
                return None
        except:
            return None

    # ======================================================================
    # Interface tests
    # ======================================================================
    def test_interface_E_exists(self):
        if self.E is None:
            return (False, "E is missing (None)")
        return (True, "E field result present")

    def test_interface_C_exists(self):
        if self.C is None:
            return (False, "C is missing (None)")
        return (True, "Capacitance result present")

    def test_interface_F_exists(self):
        if self.F is None:
            return (False, "F is missing (None)")
        return (True, "Force result present")

    def test_interface_givens_basic(self):
        missing = []
        for name in ['a', 'b', 'L', 'epsilon', 'epsilon0']:
            if getattr(self, name) is None:
                missing.append(name)
        if missing:
            return (False, f"Missing needed givens: {', '.join(missing)}")
        # Q may be needed for E and C sanity; V for force
        return (True, "Core givens (a,b,L,epsilon,epsilon0) are present")

    # ======================================================================
    # Invariant tests (theorem-based physical sanity)
    # ======================================================================
    def test_invariant_E_radial_dependence(self):
        """Check that E(r) is proportional to 1/r, i.e., E*r constant."""
        if self.E is None:
            return (False, "SKIP: E is missing")
        if self.a is None or self.b is None:
            return (False, "SKIP: need radii a and b for numeric check")

        # Try to evaluate E at two different radii (within a<r<b)
        r1 = self.a * 1.2  # some point inside
        r2 = self.b * 0.8
        if r1 >= self.b or r2 <= self.a:
            return (False, "Cannot choose valid radii; check a,b values")

        # Build a substitution dict for r
        # If E is symbolic, substitute numeric r
        subs1 = {sp.Symbol('r'): r1}
        subs2 = {sp.Symbol('r'): r2}
        e1 = self._eval_num(self.E, subs1)
        e2 = self._eval_num(self.E, subs2)
        if e1 is None or e2 is None:
            return (False, "Cannot evaluate E numerically; maybe missing Q or epsilon")
        # Check E1 * r1 ≈ E2 * r2
        lhs = e1 * r1
        rhs = e2 * r2
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E*r constant: E(r1)*r1={lhs:.3e}, E(r2)*r2={rhs:.3e}")
        else:
            return (False, f"E*r not constant: E(r1)*r1={lhs:.3e}, E(r2)*r2={rhs:.3e}")

    def test_invariant_C_log_dependence(self):
        """Check that C is proportional to L/ln(b/a)."""
        if self.C is None:
            return (False, "SKIP: C missing")
        if self.L is None or self.a is None or self.b is None:
            return (False, "SKIP: missing L, a, or b")
        if self.L == 0 or self.b <= self.a:
            return (False, "Invalid geometry")

        # Expected form: C = 2*pi*epsilon*L / ln(b/a)  (full insertion, part b)
        # Compute numeric value from formula if epsilon numeric
        if self.epsilon is None:
            return (False, "SKIP: epsilon missing to evaluate numeric C")

        # Build expected expression
        expected = 2 * sp.pi * self.epsilon * self.L / sp.log(self.b / self.a)
        expected_num = float(expected.evalf()) if expected.is_number else None
        if expected_num is None:
            # try to use numeric substitutes manually
            try:
                import math as m
                expected_num = 2 * m.pi * float(self.epsilon) * float(self.L) / m.log(float(self.b / self.a))
            except:
                return (False, "Cannot evaluate expected capacitance numerically")
        # Get actual C numeric
        actual = self._eval_num(self.C) if self._is_sympy(self.C) else float(self.C)
        if actual is None:
            return (False, "Cannot evaluate actual C numerically")
        # Compare
        if math.isclose(actual, expected_num, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"C matches expected formula: {actual:.3e} vs {expected_num:.3e}")
        else:
            return (False, f"C mismatch: actual {actual:.3e}, expected {expected_num:.3e}")

    def test_invariant_force_from_capacitance_derivative(self):
        """For partial insertion, check F = 1/2 * V^2 * dC_total/dx."""
        if self.F is None or self.V is None:
            return (False, "SKIP: missing F or V")
        if self.a is None or self.b is None or self.L is None:
            return (False, "SKIP: missing geometry")
        if self.epsilon is None or self.epsilon0 is None or self.x is None:
            return (False, "SKIP: missing epsilon, epsilon0, or x")

        # Build total capacitance expression: C_total = 2pi*eps0*x/ln(b/a) + 2pi*eps*(L-x)/ln(b/a)
        # consistent with explanation (x = pulled out length, vacuum part)
        ln_term = sp.log(self.b / self.a)
        C_total = (2*sp.pi*self.epsilon0*self.x / ln_term +
                   2*sp.pi*self.epsilon*(self.L - self.x) / ln_term)
        # derivative w.r.t. x
        dC_dx = sp.diff(C_total, self.x) if self._is_sympy(self.x) else sp.diff(C_total, sp.Symbol('x'))
        # Expected force: F_ext = -F_elec = - 1/2 V^2 dC/dx  (since F_elec = dU/dx = 1/2 V^2 dC/dx, and F_elec is toward capacitor)
        # But note: pseudocode says F_ext = pi*eps0*V^2*(1 - eps/eps0)/ln(b/a). This matches -1/2 V^2 dC/dx.
        expected_F = - sp.Rational(1,2) * self.V**2 * dC_dx
        # Numerically evaluate
        expected_num = self._eval_num(expected_F)
        actual_num = self._eval_num(self.F)
        if expected_num is None or actual_num is None:
            return (False, "Cannot evaluate numeric values for F comparison")
        if math.isclose(actual_num, expected_num, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"F matches dU/dx: actual {actual_num:.3e}, expected from derivative {expected_num:.3e}")
        else:
            return (False, f"F mismatch: actual {actual_num:.3e}, expected from derivative {expected_num:.3e}")

    # ======================================================================
    # Modeling/trace tests
    # ======================================================================
    def test_modeling_partial_capacitance_expression(self):
        """Check that C_total expression is sum of two parts (vacuum + dielectric) correctly placed."""
        if self.C is None:
            # If self.C is for full insertion, we might not have partial info. We'll skip if x not present.
            if self.x is None:
                return (False, "SKIP: no partial insertion data")
            else:
                return (False, "C is missing; cannot check partial capacitance")
        if self.x is None:
            return (False, "SKIP: x missing")
        # Build expected C_total: 2π ε0 x / ln(b/a) + 2π ε (L-x) / ln(b/a)
        ln_term = sp.log(self.b / self.a)
        C_total_expr = (2*sp.pi*self.epsilon0*self.x/ln_term +
                        2*sp.pi*self.epsilon*(self.L - self.x)/ln_term)
        # If self.C is for full insertion, it should be 2π ε L/ln(b/a), which is a special case.
        # That's fine; we can still check that the partial formula is consistent.
        # However, the problem expects self.C to be the partial capacitance? Usually part (b) is full C.
        # But we'll check that at x=0, C_total equals full C. We cannot assume x=0, so just check structure.
        # Use symbolic equality if both are sympy
        if self._is_sympy(self.C) and self._is_sympy(C_total_expr):
            # We can check if they are structurally similar, but not exact because different variables.
            pass  # not a strict test
        # Instead, if we have numeric values for all, check numeric identity
        actual_num = self._eval_num(self.C)
        expected_num = self._eval_num(C_total_expr)
        if actual_num is not None and expected_num is not None:
            if math.isclose(actual_num, expected_num, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"C_total matches expected form: {actual_num:.3e}")
            else:
                return (False, f"C_total numeric mismatch: actual {actual_num:.3e}, expected {expected_num:.3e}")
        else:
            # Symbolic check: check that C_total expression contains epsilon0, epsilon, ln(b/a)
            # This is weak but better than nothing.
            expr_str = sp.srepr(C_total_expr) if self._is_sympy(C_total_expr) else str(C_total_expr)
            if 'epsilon0' in expr_str and 'epsilon' in expr_str and 'log' in expr_str:
                return (True, "Partial C expression includes correct pieces (symbolic)")
            else:
                return (False, "Partial C expression does not look correct symbolically")

    def test_modeling_force_sign_and_direction(self):
        """Check sign of F: if epsilon > epsilon0, F (external) should be positive (outward)."""
        if self.F is None:
            return (False, "SKIP: F missing")
        if self.epsilon is None or self.epsilon0 is None:
            return (False, "SKIP: need epsilon and epsilon0")
        # Compare epsilon and epsilon0 (dielectric constant usually > epsilon0)
        # If either is symbolic, we can try numeric evaluation
        eps_num = None
        if self._is_sympy(self.epsilon):
            eps_num = self._eval_num(self.epsilon)
        else:
            try:
                eps_num = float(self.epsilon)
            except:
                pass
        eps0_num = None
        if self._is_sympy(self.epsilon0):
            eps0_num = self._eval_num(self.epsilon0)
        else:
            try:
                eps0_num = float(self.epsilon0)
            except:
                pass
        if eps_num is None or eps0_num is None:
            return (False, "Cannot determine numeric epsilon or epsilon0")
        if eps_num <= eps0_num:
            # In typical dielectrics, epsilon > epsilon0; if not, force sign flips.
            # Still check consistency with sign of F.
            pass  # We'll just check the formula sign relation.
        # Evaluate F numerically
        F_num = self._eval_num(self.F)
        if F_num is None:
            if self._is_sympy(self.F):
                # Maybe we can infer sign from expression: pi epsilon0 V^2 (1 - epsilon/epsilon0)/ln(b/a)
                # Since (1 - epsilon/epsilon0) < 0, F is negative? Wait: Explanation says F<0 attractive, so external force is opposite.
                # In pseudocode they gave F_ext = pi epsilon0 V^2 (1 - epsilon/epsilon0)/ln(b/a) which is negative if epsilon>epsilon0.
                # But then they said "outward, pulling dielectric out" which would be positive? Inconsistency.
                # Let's rely on the explanation: "F will tend to decrease x, i.e., F is attractive. Then to hold the dielectric ... a force must be applied with magnitude F and a direction away from the capacitor."
                # So the electrostatic force on dielectric is attractive (negative if outward direction positive). External force is opposite: outward (positive).
                # In pseudocode they defined F_ext as the external force. According to their subans: F = pi epsilon0 V^2/ln(b/a) (1 - epsilon/epsilon0) (outward).
                # For epsilon>epsilon0, (1 - epsilon/epsilon0) is negative, so F_ext negative? That would be inward. That contradicts "outward".
                # Let's check the subans: "F = pi epsilon0 V^2/ln(b/a) (1 - epsilon/epsilon0) (outward, pulling dielectric out)"
                # But (1 - epsilon/epsilon0) is negative => F negative => inward. That's a sign error in pseudocode.
                # The correct external force should be positive outward: F_ext = pi epsilon0 V^2 (epsilon/epsilon0 - 1)/ln(b/a) = pi V^2 (epsilon - epsilon0)/ln(b/a).
                # But explanation says "F = 1/2 V^2 dC/dx = pi epsilon0 V^2/ln(b/a) (1 - epsilon/epsilon0)". That is the electrostatic force on dielectric? Let's check:
                # dC/dx = 2pi (epsilon0 - epsilon)/ln(b/a) => 1/2 V^2 dC/dx = pi V^2 (epsilon0 - epsilon)/ln(b/a) = pi epsilon0 V^2 (1 - epsilon/epsilon0)/ln(b/a). That's negative (attractive). So F_elec = that.
                # Since F_elec is negative (attractive), the external force to hold is positive (outward) with same magnitude? Or equal magnitude? Actually to hold, external force must cancel F_elec: F_ext = -F_elec = pi V^2 (epsilon - epsilon0)/ln(b/a).
                # The pseudocode says F = ... (1 - epsilon/epsilon0) and says "outward". That is wrong sign.
                # Our test should check that the sign convention is consistent with the definition used. We'll detect the sign error? But we are auditing the solution; if its sign is wrong, it should fail.
                # However, we must be careful: the problem ask "Find the force necessary to hold the dielectric in this position. In which direction must the force be applied?" According to explanation: "force must be applied with magnitude F and a direction away from the capacitor." So external force is outward. So if F_ext is negative, that would be inward. So the student's answer should have positive outward force.
                # The target answer given in "TARGET ANSWERS" is: "F = \\frac{\\pi \\varepsilon_0 V^2}{\\ln(\\frac{b}{a})} \\left( 1 - \\frac{\\varepsilon}{\\varepsilon_0} \\right)". That expression is negative if epsilon>epsilon0. So the target answer given is negative, which would correspond to electrostatic force, not external. The problem statement for target answer might be ambiguous. Let's read: "F = ..." with no subscript. Probably they mean the electrostatic force on dielectric (which is attractive/negative). And then they say "find the force necessary to hold" – that would be opposite. The explanation says "F < 0, hence F will tend to decrease x, i.e., F is attractive. Then to hold ... a force must be applied with magnitude F and a direction away from the capacitor." That means the holding force has magnitude |F| and direction outward. So the correct answer for the holding force should be positive outward with same magnitude? Actually they say "magnitude F and direction away", so they reuse the same variable F for magnitude, but the vector direction is outward. So the sign convention is that F (magnitude) is positive, and direction is outward. The expression they gave for F is negative, so they are inconsistent. But many textbooks give electrostatic force as negative, and then say external force is opposite. So we need to be lenient: we will test that the magnitude matches something, and that the sign is consistent with the convention used (if they define F_elec with sign, that's fine; if they define F_hold, that's fine). Our test should check that the given F expression, when evaluated numeric, matches the expected magnitude in some way.
                # Let's compute expected magnitude: |F| = pi V^2 (epsilon - epsilon0)/ln(b/a)
                expected_mag = sp.pi * self.V**2 * (self.epsilon - self.epsilon0) / sp.log(self.b/self.a)
                expected_mag_num = self._eval_num(expected_mag)
                if expected_mag_num is None:
                    return (False, "Cannot compute expected force magnitude")
                # Actual magnitude
                F_num = self._eval_num(self.F)
                if F_num is None:
                    return (False, "Cannot evaluate actual F")
                if math.isclose(abs(F_num), expected_mag_num, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, f"Force magnitude matches expected: |F|={abs(F_num):.3e} == {expected_mag_num:.3e}")
                else:
                    return (False, f"Force magnitude mismatch: |F|={abs(F_num):.3e}, expected={expected_mag_num:.3e}")
            else:
                return (False, "Cannot evaluate F numerically")
        # If we have numeric, just check magnitude against expected formula.
        expected_mag = sp.pi * self.V**2 * (self.epsilon - self.epsilon0) / sp.log(self.b/self.a)
        expected_mag_num = self._eval_num(expected_mag)
        if expected_mag_num is None:
            return (False, "Cannot compute expected force magnitude from formula")
        if math.isclose(abs(F_num), expected_mag_num, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Force magnitude matches expected: |F|={abs(F_num):.3e} == {expected_mag_num:.3e}")
        else:
            return (False, f"Force magnitude mismatch: |F|={abs(F_num):.3e}, expected={expected_mag_num:.3e}")

    # ======================================================================
    # Monotonicity tests
    # ======================================================================
    def test_monotonicity_force_increases_with_V(self):
        """Check that increasing V increases the magnitude of F."""
        if self.F is None or self.V is None:
            return (False, "SKIP: missing F or V")
        # Get baseline magnitude
        F0 = self._eval_num(self.F)
        if F0 is None:
            return (False, "Cannot evaluate baseline F")
        # Perturb V: increase by 10%
        V_new = self.V * 1.1 if not self._is_sympy(self.V) else sp.Symbol('V') * 1.1
        # But we need to recompute F_new. We cannot call solution again. Instead, we can use the expected formula: F ∝ V^2.
        # So predict F_new = F0 * (V_new/V0)^2.
        V0_num = float(self.V) if not self._is_sympy(self.V) else self._eval_num(self.V)
        if V0_num is None:
            return (False, "Cannot evaluate baseline V")
        V_new_num = V0_num * 1.1
        predicted_F_new = F0 * (V_new_num / V0_num)**2
        # Now we need to see if the student's formula gives that scaling. If F is symbolic, we can substitute V_new and evaluate.
        if self._is_sympy(self.F):
            # substitute V with V_new
            subs = {sp.Symbol('V'): V_new_num}
            F_new_num = self._eval_num(self.F, subs)
            if F_new_num is None:
                return (False, "Cannot evaluate F for perturbed V")
            if math.isclose(F_new_num, predicted_F_new, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"F scales as V^2: baseline {F0:.3e}, predicted {predicted_F_new:.3e}, actual {F_new_num:.3e}")
            else:
                return (False, f"F does not scale as V^2: predicted {predicted_F_new:.3e}, actual {F_new_num:.3e}")
        else:
            # If numeric constant, we can't perturb, so skip
            return (False, "SKIP: F is numeric constant, cannot test monotonicity by substitution")

    def test_monotonicity_capacitance_decreases_with_x(self):
        """Check that increasing x (vacuum length) decreases total capacitance."""
        if self.C is None or self.x is None:
            return (False, "SKIP: missing C or x")
        # We need total capacitance expression; if C is for full insertion, it's constant, so skip
        # Assume C is the partial capacitance (or we can compute C_total from formula)
        # Build C_total expression
        ln_term = sp.log(self.b / self.a)
        C_total = (2*sp.pi*self.epsilon0*self.x/ln_term +
                   2*sp.pi*self.epsilon*(self.L - self.x)/ln_term)
        # Try to evaluate at x and at x+δ (δ=0.1* L maybe)
        delta = (self.L - self.x) * 0.1 if self._is_sympy(self.L) else (float(self.L) - float(self.x)) * 0.1
        x1 = self.x
        x2 = self.x + delta if not self._is_sympy(self.x) else sp.Symbol('x') + delta
        # Evaluate C at x1 and x2
        C1 = self._eval_num(C_total, {sp.Symbol('x'): x1} if self._is_sympy(x1) else {})
        C2 = self._eval_num(C_total, {sp.Symbol('x'): x2} if self._is_sympy(x2) else {})
        if C1 is None or C2 is None:
            return (False, "Cannot evaluate C at two x values")
        if C2 < C1:
            return (True, f"As x increases (dielectric pulled out), C decreases: C({x1})={C1:.3e} -> C({x2})={C2:.3e}")
        else:
            return (False, f"C does not decrease with x: C({x1})={C1:.3e}, C({x2})={C2:.3e}")

    # ======================================================================
    # Answer sanity (weak)
    # ======================================================================
    def test_answer_sanity_general(self):
        """Weak sanity: all results should be finite and non-negative (where applicable)."""
        failures = []
        # E magnitude should be positive
        if self.E is not None:
            e_val = self._eval_num(self.E, {sp.Symbol('r'): (self.a+self.b)/2}) if self._is_sympy(self.E) else float(self.E)
            if e_val is not None:
                if e_val <= 0:
                    failures.append("E magnitude not positive (maybe sign convention; but magnitude should be >0)")
            else:
                failures.append("Cannot evaluate E magnitude numerically")
        # C should be positive
        if self.C is not None:
            c_val = self._eval_num(self.C)
            if c_val is not None:
                if c_val <= 0:
                    failures.append("C is not positive")
            else:
                failures.append("Cannot evaluate C numerically")
        # F: if we assume external force outward, it should be positive; but many solutions give negative electrostatic force.
        # So we only check that F is not NaN.
        if self.F is not None:
            f_val = self._eval_num(self.F)
            if f_val is not None:
                if math.isnan(f_val):
                    failures.append("F is NaN")
                elif math.isinf(f_val):
                    failures.append("F is infinite")
            else:
                failures.append("Cannot evaluate F numerically")
        if failures:
            return (False, "; ".join(failures))
        return (True, "All sanity checks passed")

    # ======================================================================
    # Run all tests
    # ======================================================================
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
        return (all_pass, logs)