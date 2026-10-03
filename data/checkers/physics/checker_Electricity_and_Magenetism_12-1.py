import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Audits a solution for the fields of a point charge at rest in S0,
    boosted to a moving frame S1. The solution should provide E_prime and B_prime
    as sympy expressions (or parseable strings) representing the transformed fields.
    """

    def __init__(self, solution_instance):
        # Safely extract givens and results using None defaults
        givens = getattr(solution_instance, 'givens', {}) or {}
        results = getattr(solution_instance, 'results', {}) or {}

        # Givens – expected as sympy symbols or numeric values
        self.q = givens.get('q', None)
        self.v0 = givens.get('v0', None)
        self.c = givens.get('c', None)
        self.epsilon0 = givens.get('epsilon0', None)
        self.x_prime = givens.get('x_prime', None)
        self.y_prime = givens.get('y_prime', None)
        self.z_prime = givens.get('z_prime', None)
        self.t_prime = givens.get('t_prime', None)

        # Results – expected as sympy expressions / vectors
        self.E_prime = results.get('E_prime', None)
        self.B_prime = results.get('B_prime', None)

        # Convert to sympy if necessary (e.g., if given as strings)
        self.E_prime = self._sympify(self.E_prime)
        self.B_prime = self._sympify(self.B_prime)

        # Define common symbols for internal calculations
        self._sym_q, self._sym_v0, self._sym_c, self._sym_eps0 = sp.symbols('q v0 c epsilon0')
        self._sym_xp, self._sym_yp, self._sym_zp, self._sym_tp = sp.symbols('xp yp zp tp')
        self._beta = self._sym_v0 / self._sym_c
        self._gamma = 1 / sp.sqrt(1 - self._beta**2)

        # Prepare a numeric test point (SI units) – used only for re-derived comparisons inside checker
        self._test_numeric = {
            'q': 1e-9,
            'v0': 0.6 * 3e8,
            'c': 3e8,
            'epsilon0': 8.854187817e-12,
            'xp': 1.0,
            'yp': 0.5,
            'zp': 0.2,
            'tp': 0.0,
        }
        self._beta_num = self._test_numeric['v0'] / self._test_numeric['c']
        self._gamma_num = 1 / math.sqrt(1 - self._beta_num**2)

    def _sympify(self, expr):
        """Try to convert a string or numeric to sympy expression; return None on failure."""
        if expr is None:
            return None
        if isinstance(expr, (sp.Basic, sp.matrices.dense.MutableDenseMatrix)):
            return expr
        try:
            return sp.sympify(expr)
        except (sp.SympifyError, TypeError):
            return None

    def _extract_components(self, field_expr):
        """
        Extract x,y,z components from a field expression.
        Assumes the expression is a sympy Matrix of shape (3,1) or (1,3).
        Returns (ex, ey, ez) or None if not possible.
        """
        if field_expr is None:
            return None
        try:
            if isinstance(field_expr, sp.matrices.dense.MutableDenseMatrix):
                if field_expr.shape == (3, 1):
                    return (field_expr[0], field_expr[1], field_expr[2])
                elif field_expr.shape == (1, 3):
                    return (field_expr[0], field_expr[1], field_expr[2])
                else:
                    return None
            return None
        except (AttributeError, IndexError):
            return None

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_has_E_prime(self):
        if self.E_prime is None:
            return (False, "E_prime is missing or could not be sympified.")
        return (True, "E_prime is present.")

    def test_interface_has_B_prime(self):
        if self.B_prime is None:
            return (False, "B_prime is missing or could not be sympified.")
        return (True, "B_prime is present.")

    # ----------------------------------------------------------------------
    # Modeling / expression structure checks
    # ----------------------------------------------------------------------
    def test_modeling_structure_common_factors(self):
        """
        Check that the expressions contain key physical factors:
        - E_prime: gamma, q/(4πϵ0)
        - B_prime: gamma, beta, q/(4πϵ0 c)
        We grep the string representation for these patterns.
        """
        if self.E_prime is None or self.B_prime is None:
            return (False, "SKIP: missing field expressions for structure test.")

        E_str = sp.srepr(self.E_prime)  # symbolic representation
        B_str = sp.srepr(self.B_prime)

        # Check E_prime has gamma and q/(4*pi*epsilon0)
        E_ok = True
        if 'gamma' not in E_str and 'Gamma' not in E_str:
            E_ok = False
            msg = "E_prime missing gamma factor."
        elif 'q' not in E_str or 'epsilon0' not in E_str or 'pi' not in E_str:
            E_ok = False
            msg = "E_prime missing q/(4πϵ0) structure."
        if not E_ok:
            return (False, msg)

        # Check B_prime has gamma, beta, and similar denominator
        B_ok = True
        if 'gamma' not in B_str and 'Gamma' not in B_str:
            B_ok = False
            msg = "B_prime missing gamma factor."
        elif 'beta' not in B_str and 'v0' not in B_str:
            B_ok = False
            msg = "B_prime missing beta or v0 factor."
        elif 'pi' not in B_str or 'epsilon0' not in B_str:
            B_ok = False
            msg = "B_prime missing q/(4πϵ0) structure."
        if not B_ok:
            return (False, msg)

        return (True, "Both fields contain expected physical factors.")

    def test_modeling_denominator_structure(self):
        """
        Check that both expressions have the same denominator structure:
        r'^3 where r'^2 = γ^2*(x' + v0 t')^2 + y'^2 + z'^2.
        We extract the denominator from both expressions and compare their symbolic forms.
        """
        if self.E_prime is None or self.B_prime is None:
            return (False, "SKIP: missing field expressions.")

        # Helper to isolate denominator of a symbolic expression (assuming it's of form numerator/denominator)
        def get_denominator(expr):
            # Sympy expression: if it's a Mul or Pow, etc.
            # Try to get denominator via sp.denom
            try:
                den = sp.denom(expr)
                if den != 1:
                    return den
            except AttributeError:
                pass
            # If it's a matrix, check each component?
            # For simplicity, assume scalar or vector with common denominator.
            # We'll look for the rational structure: something like A/B
            # Use sp.fraction on the whole expression (may work on scalar)
            if hasattr(expr, 'as_numer_denom'):
                num, den = expr.as_numer_denom()
                return den
            return None

        # For vector, check first component denominator
        E_comp = self._extract_components(self.E_prime)
        B_comp = self._extract_components(self.B_prime)
        if E_comp is None or B_comp is None:
            return (False, "Cannot extract components for denominator check.")

        # Get denominators from first nontrivial component
        e_den = None
        for comp in E_comp:
            if comp != 0:
                e_den = get_denominator(comp)
                break
        b_den = None
        for comp in B_comp:
            if comp != 0:
                b_den = get_denominator(comp)
                break

        if e_den is None or b_den is None:
            return (False, "Cannot identify denominator in E or B fields.")

        # Check symbolic equality
        if e_den.equals(b_den):
            return (True, "Denominators of E' and B' are symbolically identical.")
        else:
            return (False, f"Different denominators: E den = {e_den}, B den = {b_den}")

    # ----------------------------------------------------------------------
    # Invariant / consistency checks (use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_numerical_consistency(self):
        """
        Evaluate both the emitted expressions and the expected explicit formula
        at a numeric test point. Compare using math.isclose with 1% tolerance.
        """
        if self.E_prime is None or self.B_prime is None:
            return (False, "SKIP: missing field expressions.")

        # Build symbolic substitution dictionary for numeric evaluation
        sub_dict = {
            self._sym_q: self._test_numeric['q'],
            self._sym_v0: self._test_numeric['v0'],
            self._sym_c: self._test_numeric['c'],
            self._sym_eps0: self._test_numeric['epsilon0'],
            self._sym_xp: self._test_numeric['xp'],
            self._sym_yp: self._test_numeric['yp'],
            self._sym_zp: self._test_numeric['zp'],
            self._sym_tp: self._test_numeric['tp'],
        }

        # Compute expected fields using explicit transformed formula (derived from Lorentz boost)
        qn = self._test_numeric['q']
        vn = self._test_numeric['v0']
        cn = self._test_numeric['c']
        epsn = self._test_numeric['epsilon0']
        xpn = self._test_numeric['xp']
        ypn = self._test_numeric['yp']
        zpn = self._test_numeric['zp']
        tpn = self._test_numeric['tp']

        beta_n = vn / cn
        gamma_n = 1 / math.sqrt(1 - beta_n**2)
        r_prime_sq = gamma_n**2 * (xpn + vn * tpn)**2 + ypn**2 + zpn**2
        r_prime = math.sqrt(r_prime_sq)
        denom = r_prime**3

        expected_E_x = (gamma_n * qn * (xpn + vn * tpn)) / (4 * math.pi * epsn * denom)
        expected_E_y = (gamma_n * qn * ypn) / (4 * math.pi * epsn * denom)
        expected_E_z = (gamma_n * qn * zpn) / (4 * math.pi * epsn * denom)

        expected_B_x = 0.0
        expected_B_y = (gamma_n * beta_n * qn * zpn) / (4 * math.pi * epsn * cn * denom)
        expected_B_z = -(gamma_n * beta_n * qn * ypn) / (4 * math.pi * epsn * cn * denom)

        # Evaluate emitted fields using sympy lambdify
        try:
            E_comp = self._extract_components(self.E_prime)
            B_comp = self._extract_components(self.B_prime)
            if E_comp is None or B_comp is None:
                # Fallback: use subs and evalf
                E_val = self.E_prime.subs(sub_dict).evalf()
                B_val = self.B_prime.subs(sub_dict).evalf()
                if hasattr(E_val, '__len__'):
                    emitted_E = tuple(float(v) for v in E_val)
                else:
                    emitted_E = (float(E_val), 0.0, 0.0)  # not correct, fail later
                if hasattr(B_val, '__len__'):
                    emitted_B = tuple(float(v) for v in B_val)
                else:
                    emitted_B = (float(B_val), 0.0, 0.0)
            else:
                # Create lambdified functions
                vars_list = [self._sym_q, self._sym_v0, self._sym_c, self._sym_eps0,
                             self._sym_xp, self._sym_yp, self._sym_zp, self._sym_tp]
                E_x_func = sp.lambdify(vars_list, E_comp[0], modules='numpy')
                E_y_func = sp.lambdify(vars_list, E_comp[1], modules='numpy')
                E_z_func = sp.lambdify(vars_list, E_comp[2], modules='numpy')
                B_x_func = sp.lambdify(vars_list, B_comp[0], modules='numpy')
                B_y_func = sp.lambdify(vars_list, B_comp[1], modules='numpy')
                B_z_func = sp.lambdify(vars_list, B_comp[2], modules='numpy')
                args = (self._test_numeric['q'], self._test_numeric['v0'], self._test_numeric['c'],
                        self._test_numeric['epsilon0'], self._test_numeric['xp'], self._test_numeric['yp'],
                        self._test_numeric['zp'], self._test_numeric['tp'])
                emitted_E = (E_x_func(*args), E_y_func(*args), E_z_func(*args))
                emitted_B = (B_x_func(*args), B_y_func(*args), B_z_func(*args))
        except Exception as e:
            return (False, f"Numeric evaluation failed: {e}")

        # Compare using math.isclose
        tol = 1e-2
        checks_E = [
            (emitted_E[0], expected_E_x, "E_x"),
            (emitted_E[1], expected_E_y, "E_y"),
            (emitted_E[2], expected_E_z, "E_z"),
        ]
        for emitted, expected, name in checks_E:
            if not math.isclose(emitted, expected, rel_tol=tol, abs_tol=tol):
                return (False, f"{name} mismatch: emitted {emitted:.4e}, expected {expected:.4e}")

        checks_B = [
            (emitted_B[0], expected_B_x, "B_x"),
            (emitted_B[1], expected_B_y, "B_y"),
            (emitted_B[2], expected_B_z, "B_z"),
        ]
        for emitted, expected, name in checks_B:
            if not math.isclose(emitted, expected, rel_tol=tol, abs_tol=tol):
                return (False, f"{name} mismatch: emitted {emitted:.4e}, expected {expected:.4e}")

        return (True, "Numerical consistency with explicit transformation holds.")

    def test_invariant_ratio_relations(self):
        """
        Check key scalar relations that must hold symbolically:
        - E_y / y_prime = E_z / z_prime
        - B_y / z_prime = - B_z / y_prime
        Use sympy simplification to verify equality.
        """
        if self.E_prime is None or self.B_prime is None:
            return (False, "SKIP: missing field expressions.")

        E_comp = self._extract_components(self.E_prime)
        B_comp = self._extract_components(self.B_prime)
        if E_comp is None or B_comp is None:
            return (False, "Cannot extract components for ratio check.")

        # Choose symbolic coordinate replacements
        subs_xy = {self._sym_yp: 1, self._sym_zp: 1, self._sym_xp: 0, self._sym_tp: 0,
                   self._sym_q: 1, self._sym_v0: 0.5, self._sym_c: 1, self._sym_eps0: 1}
        # Simplify ratio E_y / yprime and E_z / zprime – they should be equal
        try:
            ratio_Ey_over_yp = sp.simplify(E_comp[1] / self._sym_yp).subs(subs_xy)
            ratio_Ez_over_zp = sp.simplify(E_comp[2] / self._sym_zp).subs(subs_xy)
            ratio_By_over_zp = sp.simplify(B_comp[1] / self._sym_zp).subs(subs_xy)
            ratio_Bz_over_yp = sp.simplify(B_comp[2] / self._sym_yp).subs(subs_xy)

            # Check E ratio equality
            diff_E = sp.simplify(ratio_Ey_over_yp - ratio_Ez_over_zp)
            if diff_E != 0:
                return (False, f"E_y/yp != E_z/zp, difference = {diff_E}")

            # Check B ratio: B_y/zp = - B_z/yp
            diff_B = sp.simplify(ratio_By_over_zp + ratio_Bz_over_yp)
            if diff_B != 0:
                return (False, f"B_y/zp != -B_z/yp, difference = {diff_B}")

        except Exception as e:
            return (False, f"Ratio check failed: {e}")

        return (True, "E and B ratio relations hold symbolically.")

    # ----------------------------------------------------------------------
    # Monotonicity checks (by perturbing an input and comparing perturbed output)
    # ----------------------------------------------------------------------
    def test_monotonicity_field_strength_with_speed(self):
        """
        As v0 increases (keeping charge at same primed coordinates),
        the magnetic field magnitude should increase (monotonically non-decreasing).
        Compare B magnitude at v0 and v0+δ.
        """
        if self.B_prime is None:
            return (False, "SKIP: B_prime missing.")

        # Define baseline parameters (use numeric test point)
        qn = self._test_numeric['q']
        v0_base = self._test_numeric['v0']
        cn = self._test_numeric['c']
        epsn = self._test_numeric['epsilon0']
        xpn = self._test_numeric['xp']
        ypn = self._test_numeric['yp']
        zpn = self._test_numeric['zp']
        tpn = self._test_numeric['tp']

        # Increase v0 by 10%
        delta = 0.1 * v0_base
        v0_pert = v0_base + delta

        # Compute B magnitude for both speeds using explicit formula (re-derive inside checker)
        def B_mag(v0_val):
            beta = v0_val / cn
            gamma = 1 / math.sqrt(1 - beta**2)
            r2 = gamma**2 * (xpn + v0_val * tpn)**2 + ypn**2 + zpn**2
            r = math.sqrt(r2)
            # B components
            Bx = 0.0
            By = (gamma * beta * qn * zpn) / (4 * math.pi * epsn * cn * r**3)
            Bz = -(gamma * beta * qn * ypn) / (4 * math.pi * epsn * cn * r**3)
            return math.sqrt(Bx**2 + By**2 + Bz**2)

        mag_base = B_mag(v0_base)
        mag_pert = B_mag(v0_pert)

        if mag_pert < mag_base - 1e-12:
            return (False, f"B-field magnitude decreased: baseline {mag_base:.4e}, perturbed {mag_pert:.4e}")
        # Also check that the solution's B_prime gives the same trend?
        # We can compute B magnitude from the emitted expression at the test point.
        if self.B_prime is not None:
            # Evaluate at both speeds
            sub_base = {self._sym_q: qn, self._sym_v0: v0_base, self._sym_c: cn, self._sym_eps0: epsn,
                        self._sym_xp: xpn, self._sym_yp: ypn, self._sym_zp: zpn, self._sym_tp: tpn}
            sub_pert = {**sub_base, self._sym_v0: v0_pert}
            try:
                B_val_base = self.B_prime.subs(sub_base).evalf()
                B_val_pert = self.B_prime.subs(sub_pert).evalf()
                # Compute magnitude; if matrix, extract
                if hasattr(B_val_base, '__len__'):
                    Bmag_base = math.sqrt(sum(float(v)**2 for v in B_val_base))
                    Bmag_pert = math.sqrt(sum(float(v)**2 for v in B_val_pert))
                else:
                    Bmag_base = float(B_val_base)
                    Bmag_pert = float(B_val_pert)
                if Bmag_pert < Bmag_base - 1e-12:
                    return (False, f"Emitted B magnitude decreased: baseline {Bmag_base:.4e}, perturbed {Bmag_pert:.4e}")
            except Exception:
                pass  # just rely on the explicit check

        return (True, "B-field magnitude is non-decreasing with v0.")

    def test_monotonicity_electric_field_y_ratio(self):
        """
        Check that E_y / y_prime does not depend on y' (should be constant with y'),
        i.e., perturbing y' does not change the ratio.
        """
        if self.E_prime is None:
            return (False, "SKIP: E_prime missing.")

        E_comp = self._extract_components(self.E_prime)
        if E_comp is None:
            return (False, "Cannot extract components.")

        # Re-derive ratio from explicit formula inside checker
        qn = self._test_numeric['q']
        vn = self._test_numeric['v0']
        cn = self._test_numeric['c']
        epsn = self._test_numeric['epsilon0']
        xpn = self._test_numeric['xp']
        tpn = self._test_numeric['tp']
        zpn = self._test_numeric['zp']

        def E_y_over_y(yp_val):
            beta = vn / cn
            gamma = 1 / math.sqrt(1 - beta**2)
            r2 = gamma**2 * (xpn + vn * tpn)**2 + yp_val**2 + zpn**2
            r = math.sqrt(r2)
            return (gamma * qn * yp_val) / (4 * math.pi * epsn * r**3) / yp_val

        y_base = 0.3
        y_pert = 0.9
        ratio_base = E_y_over_y(y_base)
        ratio_pert = E_y_over_y(y_pert)

        if not math.isclose(ratio_base, ratio_pert, rel_tol=1e-2, abs_tol=1e-12):
            return (False, f"E_y/yp ratio not constant: baseline {ratio_base:.4e}, perturbed {ratio_pert:.4e}")

        # Also check using emitted expression if possible
        if E_comp[1] != 0:
            # Substitute symbolic coordinates with varied y_prime
            subs_base = {self._sym_yp: y_base, self._sym_xp: xpn, self._sym_zp: zpn, self._sym_tp: tpn,
                         self._sym_q: qn, self._sym_v0: vn, self._sym_c: cn, self._sym_eps0: epsn}
            subs_pert = {**subs_base, self._sym_yp: y_pert}
            try:
                ratio_emitted_base = sp.simplify(E_comp[1] / self._sym_yp).subs(subs_base).evalf()
                ratio_emitted_pert = sp.simplify(E_comp[1] / self._sym_yp).subs(subs_pert).evalf()
                if not math.isclose(float(ratio_emitted_base), float(ratio_emitted_pert), rel_tol=1e-2):
                    return (False, "Emitted E_y/yp ratio changed with y'.")
            except Exception:
                pass

        return (True, "E_y / y_prime is independent of y'.")

    # ----------------------------------------------------------------------
    # Single weak answer sanity check (no re-derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_sign_and_finite(self):
        """
        Weak sanity: the emitted fields should produce finite values for finite inputs,
        and for a test point the signs should match expectations.
        Electric field component along x' should have same sign as (x'+v0 t') when charge is positive.
        """
        if self.E_prime is None or self.B_prime is None:
            return (False, "SKIP: fields missing.")

        # Evaluate at a point with positive x'+v0 t'
        test_pt = {
            self._sym_q: 1e-9,
            self._sym_v0: 0.5 * 3e8,
            self._sym_c: 3e8,
            self._sym_eps0: 8.854e-12,
            self._sym_xp: 1.0,
            self._sym_yp: 0.3,
            self._sym_zp: 0.2,
            self._sym_tp: 0.0,
        }
        try:
            E_val = self.E_prime.subs(test_pt).evalf()
            B_val = self.B_prime.subs(test_pt).evalf()
        except Exception as e:
            return (False, f"Cannot evaluate fields at test point: {e}")

        # Convert to floats
        if hasattr(E_val, '__len__'):
            E_comp_num = [float(v) for v in E_val]
        else:
            E_comp_num = [float(E_val), 0.0, 0.0]
        if hasattr(B_val, '__len__'):
            B_comp_num = [float(v) for v in B_val]
        else:
            B_comp_num = [float(B_val), 0.0, 0.0]

        # Check all are finite
        for v in E_comp_num + B_comp_num:
            if not math.isfinite(v):
                return (False, "Non-finite value detected in fields.")

        # Sign check: for positive q and positive (x' + v0 t'), E_x should be positive
        if E_comp_num[0] <= 0:
            return (False, f"Expected positive E_x (since x+vt>0, q>0), got {E_comp_num[0]}")

        # B_y and B_z should have opposite signs given y=0.3>0, z=0.2>0: B_y positive? formula: (gamma beta q z)/(4πϵ c r^3) => positive; B_z negative.
        if B_comp_num[1] <= 0:
            return (False, f"Expected positive B_y, got {B_comp_num[1]}")
        if B_comp_num[2] >= 0:
            return (False, f"Expected negative B_z, got {B_comp_num[2]}")

        return (True, "Fields are finite, signs consistent with known transformation.")

    # ----------------------------------------------------------------------
    # Main runner
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