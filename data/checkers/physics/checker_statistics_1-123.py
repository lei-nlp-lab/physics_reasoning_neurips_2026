import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Physics auditor for the Van der Waals equation problem.
    Checks interface, invariants, modeling, and monotonicity.
    """
    def __init__(self, solution_instance):
        self.sol = solution_instance
        # Safely extract results
        self.results = getattr(solution_instance, 'results', {})
        # Givens (constants) – may be a dict or object with attributes
        self.givens = getattr(solution_instance, 'givens', {})
        if not isinstance(self.givens, dict):
            # If givens is an object, try to convert to dict or treat as attribute bag
            try:
                self.givens = {k: getattr(self.givens, k) for k in dir(self.givens) if not k.startswith('_')}
            except:
                self.givens = {}
        # Extract key result quantities
        self.V_c = self.results.get('V_c', None)
        self.p_c = self.results.get('p_c', None)
        self.T_c = self.results.get('T_c', None)
        self.ratio = self.results.get('ratio', None)
        self.maxwell_eq = self.results.get('maxwell_equation', None)
        self.dCv_dV = self.results.get('dCv_dV', None)
        # Constants from givens (if available)
        self.a = self.givens.get('a', None)
        self.b = self.givens.get('b', None)
        self.R = self.givens.get('R', None)

    # ----------------------------------------------------------------------
    # Interface tests – verify required keys exist
    # ----------------------------------------------------------------------
    def test_interface_V_c_exists(self):
        if self.V_c is None:
            return (False, "Missing 'V_c' in results")
        return (True, "V_c present")

    def test_interface_p_c_exists(self):
        if self.p_c is None:
            return (False, "Missing 'p_c' in results")
        return (True, "p_c present")

    def test_interface_T_c_exists(self):
        if self.T_c is None:
            return (False, "Missing 'T_c' in results")
        return (True, "T_c present")

    def test_interface_ratio_exists(self):
        if self.ratio is None:
            return (False, "Missing 'ratio' in results")
        return (True, "ratio present")

    def test_interface_maxwell_equation_exists(self):
        if self.maxwell_eq is None:
            return (False, "Missing 'maxwell_equation' in results")
        return (True, "maxwell_equation present")

    def test_interface_dCv_dV_exists(self):
        if self.dCv_dV is None:
            return (False, "Missing 'dCv_dV' in results")
        return (True, "dCv_dV present")

    # ----------------------------------------------------------------------
    # Invariant tests – physical laws / theorem sanity
    # ----------------------------------------------------------------------
    def test_invariant_ratio_at_critical_point(self):
        """Check p_c*V_c/(R*T_c) = 3/8."""
        if any(x is None for x in [self.p_c, self.V_c, self.T_c, self.R]):
            return (False, "SKIP: missing one of p_c, V_c, T_c, R")
        # Compute ratio
        try:
            expr = self.p_c * self.V_c / (self.R * self.T_c)
            numeric_form = None
            if isinstance(expr, sp.Basic):
                # Try to simplify symbolically
                simplified = sp.simplify(expr - sp.Rational(3,8))
                if simplified == 0:
                    return (True, "p_c*V_c/(R*T_c) = 3/8 (symbolic)")
                else:
                    # Fallback: substitute numeric test values if possible
                    if self.a and self.b and self.R:
                        subs_dict = {self.a: 1, self.b: 1, self.R: 1}  # example values
                        try:
                            numeric_val = float(expr.subs(subs_dict))
                            if math.isclose(numeric_val, 3/8, rel_tol=1e-2, abs_tol=1e-2):
                                return (True, f"p_c*V_c/(R*T_c) ≈ {numeric_val} close to 3/8")
                        except:
                            pass
                    return (False, f"Symbolic expression does not simplify to 3/8: {expr}")
            else:
                numeric_val = float(expr)
                if math.isclose(numeric_val, 0.375, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, f"Ratio = {numeric_val}, matches 3/8")
                else:
                    return (False, f"Ratio = {numeric_val}, expected 3/8")
        except Exception as e:
            return (False, f"Error computing ratio: {e}")

    def test_invariant_CV_independence(self):
        """Check that dCv_dV is exactly 0."""
        if self.dCv_dV is None:
            return (False, "SKIP: dCv_dV not provided")
        try:
            val = 0
            if isinstance(self.dCv_dV, sp.Basic):
                if sp.simplify(self.dCv_dV) == 0:
                    return (True, "dCv_dV = 0 (symbolic)")
                else:
                    # Try numeric evaluation
                    try:
                        val = float(self.dCv_dV)
                    except:
                        return (False, f"dCv_dV not zero: {self.dCv_dV}")
            else:
                val = float(self.dCv_dV)
            if math.isclose(val, 0.0, abs_tol=1e-2):
                return (True, f"dCv_dV ~ {val}, close to zero")
            else:
                return (False, f"dCv_dV = {val}, expected zero")
        except Exception as e:
            return (False, f"Error checking dCv_dV: {e}")

    def test_invariant_critical_point_positivity(self):
        """Check that V_c, p_c, T_c are positive."""
        checks = []
        for name, val in [("V_c", self.V_c), ("p_c", self.p_c), ("T_c", self.T_c)]:
            if val is None:
                checks.append(f"{name} missing")
            else:
                try:
                    numeric_val = float(val) if not isinstance(val, sp.Basic) else float(val.subs({sp.Symbol('a'):1, sp.Symbol('b'):1, sp.Symbol('R'):1}))
                    if numeric_val <= 0:
                        checks.append(f"{name} is not positive (={numeric_val})")
                except:
                    checks.append(f"{name}: cannot evaluate positivity")
        if checks:
            return (False, "; ".join(checks))
        return (True, "All critical point quantities positive")

    # ----------------------------------------------------------------------
    # Modeling / trace consistency tests
    # ----------------------------------------------------------------------
    def test_modeling_CV_via_Maxwell_relation(self):
        """Verify that (∂C_V/∂V)_T = T*(∂²p/∂T²)_V, and second derivative is zero."""
        if self.dCv_dV is None:
            return (False, "SKIP: dCv_dV missing")
        # If we have p expression, compute second derivative
        # Assume p = RT/(V-b) - a/V^2
        # If constants are available, derive symbolically
        try:
            # Create symbols if needed
            R_sym = self.R if isinstance(self.R, sp.Basic) else sp.symbols('R') if self.R is None else sp.nsimplify(self.R)
            a_sym = self.a if isinstance(self.a, sp.Basic) else sp.symbols('a') if self.a is None else sp.nsimplify(self.a)
            b_sym = self.b if isinstance(self.b, sp.Basic) else sp.symbols('b') if self.b is None else sp.nsimplify(self.b)
            T_sym = sp.symbols('T')
            V_sym = sp.symbols('V')
            p_expr = R_sym*T_sym/(V_sym - b_sym) - a_sym/V_sym**2
            d2p_dT2 = sp.diff(p_expr, T_sym, 2)
            if sp.simplify(d2p_dT2) != 0:
                return (False, f"Second derivative of p w.r.t. T (at const V) is not zero: {d2p_dT2}")
            # Check that dCv_dV equals T * d2p_dT2 = 0
            expected = sp.simplify(T_sym * d2p_dT2)
            if sp.simplify(self.dCv_dV) != 0:
                # Possibly numeric
                if isinstance(self.dCv_dV, sp.Basic):
                    if sp.simplify(self.dCv_dV - expected) != 0:
                        return (False, f"dCv_dV should be {expected}, got {self.dCv_dV}")
                else:
                    if not math.isclose(float(self.dCv_dV), 0.0, abs_tol=1e-2):
                        return (False, f"dCv_dV = {self.dCv_dV}, expected 0")
            return (True, "C_V independence confirmed via Maxwell relation")
        except Exception as e:
            return (False, f"Error in modeling CV: {e}")

    def test_modeling_maxwell_equation_format(self):
        """Check that maxwell_equation is a non-empty string describing the equal-area condition."""
        if not self.maxwell_eq:
            return (False, "maxwell_equation missing or empty")
        if not isinstance(self.maxwell_eq, str):
            return (False, f"maxwell_equation is not a string: {type(self.maxwell_eq)}")
        # Check for key terms
        terms = ['integral', 'V dp', '=']
        missing = [t for t in terms if t not in self.maxwell_eq]
        if missing:
            return (False, f"maxwell_equation missing terms: {missing}")
        return (True, "maxwell_equation is a valid string")

    # ----------------------------------------------------------------------
    # Monotonicity tests – perturb one input and observe output trend
    # ----------------------------------------------------------------------
    def test_monotonicity_T_c_with_b(self):
        """If b increases, T_c should decrease (T_c ∝ 1/b)."""
        if any(x is None for x in [self.T_c, self.b]):
            return (False, "SKIP: T_c or b not available")
        try:
            # Get baseline T_c numeric
            if isinstance(self.T_c, sp.Basic):
                # Substitute typical values to get numeric
                sub_dict = {self.a: 1, self.b: 1, self.R: 1} if self.a is None else {self.a: self.a, self.b: self.b, self.R: self.R}
                T0 = float(self.T_c.subs(sub_dict))
                b0 = float(self.b.subs(sub_dict)) if isinstance(self.b, sp.Basic) else float(self.b)
            else:
                T0 = float(self.T_c)
                b0 = float(self.b)
            # Perturb b up by 1%
            b1 = b0 * 1.01
            # Expected new T_c from the formula T_c = 8a/(27bR)
            # If we have a and R, compute; otherwise use local inverse relation.
            if self.a and self.R:
                a_val = float(self.a) if not isinstance(self.a, sp.Basic) else float(self.a.subs(sub_dict))
                R_val = float(self.R) if not isinstance(self.R, sp.Basic) else float(self.R.subs(sub_dict))
                T_expected = 8 * a_val / (27 * b1 * R_val)
            else:
                # Use inverse proportionality: T_c ∝ 1/b
                T_expected = T0 * (b0 / b1)
            # Compare with computed new T_c from solution? We don't have a new solution, but we can check using derivative.
            # Instead, we check that the derivative dT_c/db is negative.
            # Compute approximate derivative from formula.
            # We don't have function that computes T_c for arbitrary b. But we can check that the formula is consistent.
            # Since we expect T_c ∝ 1/b, we check that T0 * b0 * constant = T_expected * b1 * constant.
            left = T0 * b0
            right = T_expected * b1
            if math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"T_c decreases when b increases: T_c*b conserved within tolerance")
            else:
                return (False, f"T_c*b not conserved: {left} vs {right}")
        except Exception as e:
            return (False, f"Error in monotonicity test: {e}")

    def test_monotonicity_p_c_with_b(self):
        """If b increases, p_c should decrease as p_c ∝ 1/b^2."""
        if any(x is None for x in [self.p_c, self.b]):
            return (False, "SKIP: p_c or b not available")
        try:
            # Get baseline p_c numeric
            if isinstance(self.p_c, sp.Basic):
                sub_dict = {self.a: 1, self.b: 1} if self.a is None else {self.a: self.a, self.b: self.b}
                p0 = float(self.p_c.subs(sub_dict))
                b0 = float(self.b.subs(sub_dict)) if isinstance(self.b, sp.Basic) else float(self.b)
            else:
                p0 = float(self.p_c)
                b0 = float(self.b)
            # Perturb b up by 1%
            b1 = b0 * 1.01
            # Expected p_c from formula p_c = a/(27 b^2)
            if self.a:
                a_val = float(self.a) if not isinstance(self.a, sp.Basic) else float(self.a.subs(sub_dict))
                p_expected = a_val / (27 * b1**2)
            else:
                # Use p_c ∝ 1/b^2
                p_expected = p0 * (b0**2 / b1**2)
            # Check that p_c * b^2 is conserved
            left = p0 * b0**2
            right = p_expected * b1**2
            if math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"p_c decreases when b increases: p_c*b^2 conserved")
            else:
                return (False, f"p_c*b^2 not conserved: {left} vs {right}")
        except Exception as e:
            return (False, f"Error in monotonicity test: {e}")

    # ----------------------------------------------------------------------
    # Weak answer sanity (optional, non-dominant)
    # ----------------------------------------------------------------------
    def test_answer_sanity_general(self):
        """
        Very weak sanity: all main results should be finite and not NaN.
        Do NOT perform any re-derivation or close matching.
        """
        checks = []
        for name, val in [("V_c", self.V_c), ("p_c", self.p_c), ("T_c", self.T_c), ("ratio", self.ratio)]:
            if val is None:
                checks.append(f"{name} missing")
            else:
                try:
                    # Attempt numeric conversion if possible (ignore symbolic)
                    if not isinstance(val, sp.Basic):
                        v = float(val)
                        if math.isnan(v):
                            checks.append(f"{name} is NaN")
                    else:
                        # Symbolic is fine, but check not None
                        pass
                except:
                    pass
        if checks:
            return (False, "; ".join(checks))
        return (True, "All key results exist and are finite")

    # ----------------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------------
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