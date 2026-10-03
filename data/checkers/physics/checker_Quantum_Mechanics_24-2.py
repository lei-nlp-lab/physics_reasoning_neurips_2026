import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Physics checker for the two-flavor neutrino oscillation problem.
    Checks consistency of results with the physics of maximal mixing,
    equal amplitudes, and time evolution in the rest frame.
    """

    def __init__(self, solution_instance):
        # Safely extract results
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        self.P_ee = results.get('P_ee', None)
        self.t_conv = results.get('t_conv', None)
        self.P_1 = results.get('P_1', None)

        # Safely extract givens (may be numeric or symbolic)
        givens = getattr(solution_instance, 'givens', {})
        if givens is None:
            givens = {}
        self.hbar = givens.get('hbar', None)
        self.c = givens.get('c', None)
        self.m1 = givens.get('m1', None)
        self.m2 = givens.get('m2', None)
        self.t = givens.get('t', None)          # time parameter, may be variable

        # For symbolic work, define default symbols if givens are not provided
        # but we will only use them when needed; no numeric fallback defaults.
        self._t_sym = sp.symbols('t', positive=True)
        self._hbar_sym = sp.symbols('hbar', positive=True)
        self._c_sym = sp.symbols('c', positive=True)
        self._m1_sym = sp.symbols('m1', positive=True)
        self._m2_sym = sp.symbols('m2', positive=True)
        # self._deltaE_sym = (self._m2_sym - self._m1_sym) * self._c_sym**2  # not needed explicitly

    # ------------------------------------------------------------------
    # Interface tests: check that required result keys exist
    # ------------------------------------------------------------------
    def test_interface_P_ee_not_none(self):
        if self.P_ee is None:
            return (False, "P_ee is missing")
        return (True, "P_ee is present")

    def test_interface_t_conv_not_none(self):
        if self.t_conv is None:
            return (False, "t_conv is missing")
        return (True, "t_conv is present")

    def test_interface_P_1_not_none(self):
        if self.P_1 is None:
            return (False, "P_1 is missing")
        return (True, "P_1 is present")

    # ------------------------------------------------------------------
    # Invariant tests: theorem‐based physical sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_P_1_constant(self):
        """
        Check that the probability to measure mass m1 is 1/2 (constant).
        This is a fundamental prediction of the model.
        """
        if self.P_1 is None:
            return (False, "SKIP: P_1 not available")
        # Try to interpret as a number first
        try:
            p1_val = float(self.P_1)
        except (TypeError, ValueError):
            # Possibly a symbolic expression; evaluate with default symbols
            try:
                # If the expression does not contain free variables, sympy can evaluate numerically
                p1_val = float(sp.N(self.P_1))
            except:
                return (False, "SKIP: P_1 cannot be converted to a number")
        if math.isclose(p1_val, 0.5, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"P_1 = {p1_val} is close to 0.5")
        else:
            return (False, f"P_1 = {p1_val} is not 0.5 (±1e-2)")

    def test_invariant_t_conv_positive(self):
        """
        Check that the conversion time is positive (if numeric).
        """
        if self.t_conv is None:
            return (False, "SKIP: t_conv not available")
        try:
            t_val = float(self.t_conv)
        except (TypeError, ValueError):
            # Symbolic expression: check if it's trivially positive? Skip.
            return (False, "SKIP: t_conv is symbolic, cannot assert positivity")
        if t_val > 0:
            return (True, f"t_conv = {t_val} > 0")
        else:
            return (False, f"t_conv = {t_val} must be positive")

    def test_invariant_P_ee_range(self):
        """
        Check that P_ee is in [0,1] (if numeric).
        """
        if self.P_ee is None:
            return (False, "SKIP: P_ee not available")
        try:
            p_val = float(self.P_ee)
        except (TypeError, ValueError):
            # Could be symbolic; we can check that it's a cos^2 which implies range [0,1]
            # But we won't assume; just skip.
            return (False, "SKIP: P_ee is symbolic, cannot check numeric range")
        if 0 <= p_val <= 1:
            return (True, f"P_ee = {p_val} is between 0 and 1")
        else:
            return (False, f"P_ee = {p_val} is outside [0,1]")

    # ------------------------------------------------------------------
    # Modeling tests: check that the formulas (if provided) match physics
    # ------------------------------------------------------------------
    def test_modeling_P_ee_form(self):
        """
        Check that P_ee has the form cos^2(ΔE t/(2ħ)) by substituting
        numeric values for the parameters and comparing to the predicted form.
        Uses fuzzy matching.
        """
        if self.P_ee is None:
            return (False, "SKIP: P_ee not available")
        # We need at least some givens to construct the expected expression.
        # If givens are missing, we can't check properly.
        if any(g is None for g in [self.hbar, self.c, self.m1, self.m2]):
            return (False, "SKIP: missing givens (hbar, c, m1, m2) needed to check P_ee form")
        # Convert givens to numbers for a concrete test
        try:
            hbar_val = float(self.hbar)
            c_val = float(self.c)
            m1_val = float(self.m1)
            m2_val = float(self.m2)
        except (TypeError, ValueError):
            # If any givens are symbolic, we cannot evaluate numerically for this test.
            return (False, "SKIP: givens are symbolic, cannot evaluate numeric form")
        # Use a test time (1 second for concreteness) – we only need relative comparison.
        # We also need to handle the case where t is given; if not, assume t=1.
        if self.t is not None:
            try:
                t_val = float(self.t)
            except (TypeError, ValueError):
                t_val = 1.0
        else:
            t_val = 1.0

        # Both P_ee and the expected expression should be numeric after substitution.
        # We need to extract a value from P_ee; it may be a symbolic expression.
        # We'll try to evaluate it with numeric givens and t.
        # If P_ee is already a string, convert to sympy.
        try:
            p_expr = sp.sympify(self.P_ee)
        except:
            return (False, "P_ee could not be parsed as a sympy expression")
        # Substitute known numeric parameters into P_ee expression.
        # We also need to substitute t if it is a symbol.
        # We'll create a substitution dictionary.
        subs_dict = {}
        # Identify free symbols in p_expr
        free_syms = p_expr.free_symbols
        # Map known symbols
        for sym in free_syms:
            name = str(sym)
            if name == 'hbar':
                subs_dict[sym] = hbar_val
            elif name == 'c':
                subs_dict[sym] = c_val
            elif name == 'm1':
                subs_dict[sym] = m1_val
            elif name == 'm2':
                subs_dict[sym] = m2_val
            elif name == 't':
                subs_dict[sym] = t_val
            # else: leave unsolved, will cause error
        try:
            p_num = float(sp.N(p_expr.subs(subs_dict)))
        except Exception as e:
            return (False, f"Failed to evaluate P_ee numerically: {e}")

        # Now compute expected P_ee: cos^2( (m2-m1)*c^2 * t / (2*hbar) )
        deltaE = (m2_val - m1_val) * c_val**2
        expected_val = math.cos(deltaE * t_val / (2 * hbar_val))**2

        if math.isclose(p_num, expected_val, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"P_ee ≈ {p_num} matches expected {expected_val}")
        else:
            return (False, f"P_ee = {p_num}, expected {expected_val}")

    def test_modeling_t_conv_form(self):
        """
        Check that t_conv = πħ / ( (m2-m1)*c^2 ) by substituting numeric values.
        """
        if self.t_conv is None:
            return (False, "SKIP: t_conv not available")
        if any(g is None for g in [self.hbar, self.c, self.m1, self.m2]):
            return (False, "SKIP: missing givens (hbar, c, m1, m2)")
        try:
            hbar_val = float(self.hbar)
            c_val = float(self.c)
            m1_val = float(self.m1)
            m2_val = float(self.m2)
        except (TypeError, ValueError):
            return (False, "SKIP: givens are symbolic")
        # Evaluate t_conv from result
        try:
            t_expr = sp.sympify(self.t_conv)
        except:
            return (False, "t_conv could not be parsed")
        subs_dict = {}
        for sym in t_expr.free_symbols:
            name = str(sym)
            if name == 'hbar':
                subs_dict[sym] = hbar_val
            elif name == 'c':
                subs_dict[sym] = c_val
            elif name == 'm1':
                subs_dict[sym] = m1_val
            elif name == 'm2':
                subs_dict[sym] = m2_val
        try:
            t_num = float(sp.N(t_expr.subs(subs_dict)))
        except:
            return (False, "Failed to evaluate t_conv numerically")
        # Expected:
        deltaE = (m2_val - m1_val) * c_val**2
        expected_t = math.pi * hbar_val / deltaE
        if math.isclose(t_num, expected_t, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"t_conv ≈ {t_num} matches expected {expected_t}")
        else:
            return (False, f"t_conv = {t_num}, expected {expected_t}")

    # ------------------------------------------------------------------
    # Monotonicity tests: directional checks on physical parameters
    # ------------------------------------------------------------------
    def test_monotonicity_t_conv_decreases_with_deltaE(self):
        """
        If we have numeric givens and a computed t_conv, verify that
        increasing m2 (or decreasing m1) leads to a smaller t_conv.
        We'll test by perturbing m2 upward by 10% and recomputing.
        """
        if self.t_conv is None:
            return (False, "SKIP: t_conv not available")
        if any(g is None for g in [self.hbar, self.c, self.m1, self.m2]):
            return (False, "SKIP: missing givens (hbar, c, m1, m2)")
        try:
            hbar_val = float(self.hbar)
            c_val = float(self.c)
            m1_val = float(self.m1)
            m2_val = float(self.m2)
        except (TypeError, ValueError):
            return (False, "SKIP: givens are symbolic")
        # Get the baseline t_conv numeric value
        try:
            t_expr = sp.sympify(self.t_conv)
            subs_dict = {}
            for sym in t_expr.free_symbols:
                name = str(sym)
                if name == 'hbar':
                    subs_dict[sym] = hbar_val
                elif name == 'c':
                    subs_dict[sym] = c_val
                elif name == 'm1':
                    subs_dict[sym] = m1_val
                elif name == 'm2':
                    subs_dict[sym] = m2_val
            baseline = float(sp.N(t_expr.subs(subs_dict)))
        except:
            return (False, "Failed to compute baseline t_conv")
        # Perturb m2 upward by 10%
        m2_new = m2_val * 1.1
        try:
            subs_dict[str(sp.symbols('m2'))] = m2_new
            perturbed = float(sp.N(t_expr.subs(subs_dict)))
        except:
            return (False, "Failed to compute perturbed t_conv")
        if perturbed < baseline:
            return (True, f"t_conv decreases when m2 increases (baseline={baseline}, perturbed={perturbed})")
        else:
            return (False, f"t_conv did not decrease when m2 increased (baseline={baseline}, perturbed={perturbed})")

    def test_monotonicity_P_ee_oscillation_deltaE(self):
        """
        Check that the oscillation (as seen through P_ee at fixed t) 
        is sensitive to ΔE: if ΔE increases, the phase changes.
        We'll compute P_ee at a fixed t (e.g., t=1) with baseline and perturbed ΔE.
        """
        if self.P_ee is None:
            return (False, "SKIP: P_ee not available")
        if any(g is None for g in [self.hbar, self.c, self.m1, self.m2]):
            return (False, "SKIP: missing givens (hbar, c, m1, m2)")
        try:
            hbar_val = float(self.hbar)
            c_val = float(self.c)
            m1_val = float(self.m1)
            m2_val = float(self.m2)
        except (TypeError, ValueError):
            return (False, "SKIP: givens are symbolic")
        # Use a fixed t; if not given, assume 1.
        if self.t is not None:
            try:
                t_val = float(self.t)
            except (TypeError, ValueError):
                t_val = 1.0
        else:
            t_val = 1.0

        # Evaluate baseline P_ee
        try:
            p_expr = sp.sympify(self.P_ee)
            subs_dict = {}
            for sym in p_expr.free_symbols:
                name = str(sym)
                if name == 'hbar':
                    subs_dict[sym] = hbar_val
                elif name == 'c':
                    subs_dict[sym] = c_val
                elif name == 'm1':
                    subs_dict[sym] = m1_val
                elif name == 'm2':
                    subs_dict[sym] = m2_val
                elif name == 't':
                    subs_dict[sym] = t_val
            baseline = float(sp.N(p_expr.subs(subs_dict)))
        except:
            return (False, "Failed to compute baseline P_ee")
        # Increase ΔE by increasing m2 by 10%
        m2_new = m2_val * 1.1
        try:
            subs_dict[str(sp.symbols('m2'))] = m2_new
            perturbed = float(sp.N(p_expr.subs(subs_dict)))
        except:
            return (False, "Failed to compute perturbed P_ee")
        # The P_ee should change (not be identical) because phase changed.
        if not math.isclose(baseline, perturbed, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"P_ee changes when ΔE changes (baseline={baseline}, perturbed={perturbed})")
        else:
            return (False, f"P_ee did not change sufficiently (baseline={baseline}, perturbed={perturbed})")

    # ------------------------------------------------------------------
    # (Optional) weak answer sanity: only sign/range checks, no re-derivation
    # ------------------------------------------------------------------
    def test_answer_sanity_P_1_constant(self):
        """
        Very weak sanity: P_1 should be a constant (independent of time).
        We only check that it is not NaN and not negative.
        """
        if self.P_1 is None:
            return (False, "SKIP: P_1 not available")
        try:
            p1_val = float(self.P_1)
        except (TypeError, ValueError):
            # symbolic: assume it is a constant expression; check that it has no free t symbol.
            try:
                expr = sp.sympify(self.P_1)
                if sp.symbols('t') in expr.free_symbols:
                    return (False, "P_1 appears to depend on t, should be constant")
                # also check that it is non-negative
                if any(s.is_negative for s in expr.free_symbols):
                    return (False, "P_1 may be negative")
                return (True, "P_1 is an expression with no t dependence")
            except:
                return (False, "Cannot interpret P_1")
        if math.isnan(p1_val):
            return (False, "P_1 is NaN")
        if p1_val < 0:
            return (False, "P_1 is negative")
        return (True, f"P_1 = {p1_val} is non‑negative and finite")

    # ------------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------------
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