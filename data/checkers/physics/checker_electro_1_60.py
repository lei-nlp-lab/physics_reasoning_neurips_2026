import math
import traceback
import sympy as sp

class PhysicsChecker:
    """
    Robust Physics Checker for the spherical capacitor problem with inhomogeneous dielectric.
    """

    def __init__(self, solution_instance):
        # Extract results safely; default None
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # Givens: assumed to be either in solution_instance.givens dict or as attributes
        self.givens = getattr(solution_instance, 'givens', {})
        if not isinstance(self.givens, dict):
            self.givens = {}

        # Extract each result key
        self.D_r = self.results.get('D_r', None)
        self.V = self.results.get('V', None)
        self.C = self.results.get('C', None)
        self.rho_p = self.results.get('rho_p', None)
        self.sigma_a = self.results.get('sigma_a', None)
        self.sigma_b = self.results.get('sigma_b', None)

        # Extract givens with defaults None
        self.Q = self.givens.get('Q', None)
        self.a = self.givens.get('a', None)
        self.b = self.givens.get('b', None)
        self.epsilon0 = self.givens.get('epsilon0', None)
        self.K = self.givens.get('K', None)

        # Additional parameters that may be needed
        self.r = self.givens.get('r', None)  # possibly not given

        # Symbols for sympy
        self.sym_Q = sp.Symbol('Q')
        self.sym_a = sp.Symbol('a')
        self.sym_b = sp.Symbol('b')
        self.sym_eps0 = sp.Symbol('epsilon0')
        self.sym_K = sp.Symbol('K')
        self.sym_r = sp.Symbol('r')
        self.sym_pi = sp.pi

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _get_numeric_context(self):
        """
        Return a dict of numeric values for Q, a, b, epsilon0, K.
        If any is None, return None (skip test).
        """
        if None in (self.Q, self.a, self.b, self.epsilon0, self.K):
            return None
        try:
            Q_val = float(self.Q)
            a_val = float(self.a)
            b_val = float(self.b)
            eps0_val = float(self.epsilon0)
            K_val = float(self.K)
        except (TypeError, ValueError):
            return None
        return {
            'Q': Q_val,
            'a': a_val,
            'b': b_val,
            'epsilon0': eps0_val,
            'K': K_val
        }

    def _eval_expr_string(self, expr_str, numeric_vals, var_val_dict=None):
        """
        Evaluate a sympy expression string with given numeric values.
        var_val_dict: additional substitutions (e.g., r value).
        Returns float or None on error.
        """
        try:
            expr = sp.parse_expr(expr_str)
            sub_dict = {
                self.sym_Q: numeric_vals['Q'],
                self.sym_a: numeric_vals['a'],
                self.sym_b: numeric_vals['b'],
                self.sym_eps0: numeric_vals['epsilon0'],
                self.sym_K: numeric_vals['K'],
                self.sym_pi: math.pi
            }
            if var_val_dict:
                for sym_sym, val in var_val_dict.items():
                    if isinstance(sym_sym, sp.Symbol):
                        sub_dict[sym_sym] = val
                    else:
                        # allow string key matching symbol name
                        if sym_sym == 'r':
                            sub_dict[self.sym_r] = val
            result = expr.subs(sub_dict).evalf()
            return float(result)
        except Exception:
            return None

    def _get_result_value(self, key, numeric_vals, var_val_dict=None):
        """
        Get numeric value of a result field (string or numeric) from self.results.
        """
        val = self.results.get(key, None)
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            return self._eval_expr_string(val, numeric_vals, var_val_dict)
        # if it's sympy expression, substitute numeric
        try:
            sub_dict = {
                self.sym_Q: numeric_vals['Q'],
                self.sym_a: numeric_vals['a'],
                self.sym_b: numeric_vals['b'],
                self.sym_eps0: numeric_vals['epsilon0'],
                self.sym_K: numeric_vals['K'],
                self.sym_pi: math.pi
            }
            if var_val_dict:
                for sym, v in var_val_dict.items():
                    if sym == 'r':
                        sub_dict[self.sym_r] = v
            return float(val.subs(sub_dict).evalf())
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all expected result keys exist and are not None."""
        required = ['D_r', 'V', 'C', 'rho_p', 'sigma_a', 'sigma_b']
        missing = [k for k in required if self.results.get(k, None) is None]
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required result keys present.")

    def test_interface_givens_types(self):
        """Check that givens are numbers or strings convertible to numbers."""
        num = self._get_numeric_context()
        if num is None:
            return (False, "Cannot parse givens Q, a, b, epsilon0, K as numbers.")
        # Check positivity of radii and epsilon0
        if num['a'] <= 0 or num['b'] <= 0:
            return (False, f"Radii must be positive: a={num['a']}, b={num['b']}.")
        if num['epsilon0'] <= 0:
            return (False, f"epsilon0 must be positive: {num['epsilon0']}.")
        return (True, "Givens are valid positive numbers.")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based sanity)
    # ------------------------------------------------------------------
    def test_invariant_gauss_displacement(self):
        """
        Gauss's law: D(r) * 4π r^2 = Q (free charge).
        Choose r = (a+b)/2.
        """
        num = self._get_numeric_context()
        if num is None:
            return (False, "SKIP: numeric context missing.")
        r0 = (num['a'] + num['b']) / 2.0
        var_vals = {'r': r0}
        D_r_val = self._get_result_value('D_r', num, var_vals)
        if D_r_val is None:
            return (False, "Cannot evaluate D_r numerically.")
        expected_D = num['Q'] / (4.0 * math.pi * r0 * r0)
        if math.isclose(D_r_val, expected_D, rel_tol=1e-2, abs_tol=1e-12):
            return (True, f"D_r={D_r_val} matches Gauss law (expected {expected_D}).")
        else:
            return (False, f"D_r={D_r_val} differs from {expected_D} by {abs(D_r_val-expected_D)}.")

    def test_invariant_voltage_from_field(self):
        """
        V should equal ∫_a^b E·dr, with E = D/ε.
        Numerically integrate using known formula: V = Q/(4π ε0)*(1/a - 1/b + K ln(b/a)).
        Compare to result V.
        """
        num = self._get_numeric_context()
        if num is None:
            return (False, "SKIP: numeric context missing.")
        V_val = self._get_result_value('V', num)
        if V_val is None:
            return (False, "Cannot evaluate V numerically.")
        # Expected V from formula
        expected_V = (num['Q'] / (4.0 * math.pi * num['epsilon0'])) * \
                     (1.0/num['a'] - 1.0/num['b'] + num['K'] * math.log(num['b']/num['a']))
        if math.isclose(V_val, expected_V, rel_tol=1e-2, abs_tol=1e-12):
            return (True, f"V={V_val} matches formula (expected {expected_V}).")
        else:
            return (False, f"V={V_val} differs from {expected_V} by {abs(V_val-expected_V)}.")

    def test_invariant_capacitance_from_Q_V(self):
        """C = Q/V."""
        num = self._get_numeric_context()
        if num is None:
            return (False, "SKIP: numeric context missing.")
        C_val = self._get_result_value('C', num)
        V_val = self._get_result_value('V', num)
        if C_val is None or V_val is None:
            return (False, "Cannot evaluate C or V numerically.")
        expected_C = num['Q'] / V_val
        if math.isclose(C_val, expected_C, rel_tol=1e-2, abs_tol=1e-12):
            return (True, f"C={C_val} matches Q/V (expected {expected_C}).")
        else:
            return (False, f"C={C_val} differs from Q/V {expected_C} by {abs(C_val-expected_C)}.")

    def test_invariant_capacitance_formula(self):
        """
        C should equal 4π ε0 a b / ((b-a) + a b K ln(b/a)).
        """
        num = self._get_numeric_context()
        if num is None:
            return (False, "SKIP: numeric context missing.")
        C_val = self._get_result_value('C', num)
        if C_val is None:
            return (False, "Cannot evaluate C numerically.")
        expected_C = (4 * math.pi * num['epsilon0'] * num['a'] * num['b']) / \
                     ((num['b'] - num['a']) + num['a'] * num['b'] * num['K'] * math.log(num['b']/num['a']))
        if math.isclose(C_val, expected_C, rel_tol=1e-2, abs_tol=1e-12):
            return (True, f"C={C_val} matches closed-form (expected {expected_C}).")
        else:
            return (False, f"C={C_val} differs from closed-form {expected_C} by {abs(C_val-expected_C)}.")

    # ------------------------------------------------------------------
    # Modeling / trace consistency tests
    # ------------------------------------------------------------------
    def test_modeling_rho_p_from_divergence(self):
        """
        Volume polarization charge density: ρ_p = -∇·P.
        Using P_r = -KQ/(4π r) (from D and relation).
        Compute ∇·P in spherical coordinates: (1/r^2) ∂(r^2 P_r)/∂r.
        Expected ρ_p = KQ/(4π r^2).
        """
        num = self._get_numeric_context()
        if num is None:
            return (False, "SKIP: numeric context missing.")
        r0 = (num['a'] + num['b']) / 2.0
        var_vals = {'r': r0}
        rho_val = self._get_result_value('rho_p', num, var_vals)
        if rho_val is None:
            return (False, "Cannot evaluate rho_p numerically.")
        # Expected rho_p = K*Q/(4π r^2)
        expected_rho = (num['K'] * num['Q']) / (4.0 * math.pi * r0 * r0)
        if math.isclose(rho_val, expected_rho, rel_tol=1e-2, abs_tol=1e-12):
            return (True, f"rho_p={rho_val} matches expected (expected {expected_rho}).")
        else:
            return (False, f"rho_p={rho_val} differs from {expected_rho} by {abs(rho_val-expected_rho)}.")

    def test_modeling_sigma_a_sign_and_magnitude(self):
        """
        Surface polarization charge density at r=a: σ_a = P·n̂ where n̂ = -ȓ (outward from dielectric).
        P_r(a) = -KQ/(4π a). So σ_a = -(-KQ/(4π a)) = KQ/(4π a).
        """
        num = self._get_numeric_context()
        if num is None:
            return (False, "SKIP: numeric context missing.")
        sigma_a_val = self._get_result_value('sigma_a', num)
        if sigma_a_val is None:
            return (False, "Cannot evaluate sigma_a numerically.")
        expected_sigma = (num['K'] * num['Q']) / (4.0 * math.pi * num['a'])
        if math.isclose(sigma_a_val, expected_sigma, rel_tol=1e-2, abs_tol=1e-12):
            return (True, f"sigma_a={sigma_a_val} matches expected (expected {expected_sigma}).")
        else:
            return (False, f"sigma_a={sigma_a_val} differs from {expected_sigma} by {abs(sigma_a_val-expected_sigma)}.")

    def test_modeling_sigma_b_sign_and_magnitude(self):
        """
        Surface polarization charge density at r=b: σ_b = P·n̂ where n̂ = +ȓ (outward from dielectric).
        P_r(b) = -KQ/(4π b). So σ_b = -KQ/(4π b).
        """
        num = self._get_numeric_context()
        if num is None:
            return (False, "SKIP: numeric context missing.")
        sigma_b_val = self._get_result_value('sigma_b', num)
        if sigma_b_val is None:
            return (False, "Cannot evaluate sigma_b numerically.")
        expected_sigma = - (num['K'] * num['Q']) / (4.0 * math.pi * num['b'])
        if math.isclose(sigma_b_val, expected_sigma, rel_tol=1e-2, abs_tol=1e-12):
            return (True, f"sigma_b={sigma_b_val} matches expected (expected {expected_sigma}).")
        else:
            return (False, f"sigma_b={sigma_b_val} differs from {expected_sigma} by {abs(sigma_b_val-expected_sigma)}.")

    # ------------------------------------------------------------------
    # Monotonicity tests (trend checks)
    # ------------------------------------------------------------------
    def test_monotonicity_capacitance_decreases_with_K(self):
        """
        For fixed a,b,ε0, increasing K should decrease C (denominator increases).
        Use two values K1=K, K2=K+0.1 and compare C.
        """
        num = self._get_numeric_context()
        if num is None:
            return (False, "SKIP: numeric context missing.")
        if not hasattr(self, 'C') or self.C is None:
            return (False, "SKIP: C not available.")
        # Evaluate C for two values of K
        K_orig = num['K']
        # compute C1 with original K
        C1_val = self._get_result_value('C', num)
        if C1_val is None:
            return (False, "Cannot evaluate C for original K.")
        # perturb K upward
        num2 = num.copy()
        num2['K'] = K_orig + 0.1 * abs(K_orig) + 0.01  # ensure positive change
        C2_val = self._get_result_value('C', num2)
        if C2_val is None:
            return (False, "Cannot evaluate C for perturbed K.")
        if C2_val < C1_val - 1e-9:
            return (True, f"C decreases from {C1_val} to {C2_val} when K increases.")
        else:
            return (False, f"C did not decrease: K increased, C went from {C1_val} to {C2_val}.")

    def test_monotonicity_displacement_increases_with_Q(self):
        """
        D_r should be proportional to Q.
        If we double Q, D_r should double.
        """
        num = self._get_numeric_context()
        if num is None:
            return (False, "SKIP: numeric context missing.")
        r0 = (num['a'] + num['b']) / 2.0
        var_vals = {'r': r0}
        D_orig = self._get_result_value('D_r', num, var_vals)
        if D_orig is None:
            return (False, "Cannot evaluate D_r for original Q.")
        # double Q
        num2 = num.copy()
        num2['Q'] = 2.0 * num['Q']
        D_double = self._get_result_value('D_r', num2, var_vals)
        if D_double is None:
            return (False, "Cannot evaluate D_r for doubled Q.")
        if math.isclose(D_double, 2.0 * D_orig, rel_tol=1e-2, abs_tol=1e-12):
            return (True, f"D_r doubled proportionally: original {D_orig}, doubled {D_double}.")
        else:
            return (False, f"D_r did not double: original {D_orig}, doubled {D_double}, expected {2*D_orig}.")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """
        Weak checks: signs and positivity for typical positive Q.
        Since Q is placed on inner sphere, and with default sign (positive assumed),
        we expect: C > 0, V > 0, sigma_a > 0, sigma_b < 0, rho_p > 0.
        If any of these signs deviate, it's suspicious but not final.
        """
        num = self._get_numeric_context()
        if num is None:
            return (False, "SKIP: numeric context missing.")
        # Get numeric values for the quantities
        C_val = self._get_result_value('C', num)
        V_val = self._get_result_value('V', num)
        sigma_a_val = self._get_result_value('sigma_a', num)
        sigma_b_val = self._get_result_value('sigma_b', num)
        rho_val = self._get_result_value('rho_p', num, {'r': (num['a']+num['b'])/2.0})
        errors = []
        if C_val is not None and C_val <= 0:
            errors.append(f"C={C_val} is not positive.")
        if V_val is not None and V_val <= 0:
            errors.append(f"V={V_val} is not positive.")
        if sigma_a_val is not None and sigma_a_val < 0:
            errors.append(f"sigma_a={sigma_a_val} is negative (expected positive for Q>0).")
        if sigma_b_val is not None and sigma_b_val > 0:
            errors.append(f"sigma_b={sigma_b_val} is positive (expected negative for Q>0).")
        if rho_val is not None and rho_val < 0:
            errors.append(f"rho_p={rho_val} is negative (expected positive).")
        if errors:
            return (False, "Weak sanity failed: " + "; ".join(errors))
        return (True, "Weak sanity passed: signs and positivity plausible.")

    # ------------------------------------------------------------------
    # Main runner
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
                passed = False
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)