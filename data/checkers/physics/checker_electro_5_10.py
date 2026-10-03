import math
import traceback
import sympy as sp

class PhysicsChecker:
    """
    PhysicsChecker for the infinitely long conducting wire problem.
    Audits relativistic electrodynamics: Lorentz transformation of fields,
    charge densities, velocity addition, and consistency relations.
    """

    # Symbolic parameters for internal derivation (not from solution)
    _symbolic_params = {
        'mu0': sp.symbols('mu0', positive=True),
        'epsilon0': sp.symbols('epsilon0', positive=True),
        'c': sp.symbols('c', positive=True),
        'r0': sp.symbols('r0', positive=True),
        'i': sp.symbols('i', real=True),
        'U': sp.symbols('U', positive=True),
        'v': sp.symbols('v', positive=True),
        'r': sp.symbols('r', nonnegative=True)
    }

    # Numeric test values for evaluation (SI units)
    _numeric_params = {
        'mu0': 4 * math.pi * 1e-7,
        'epsilon0': 8.854187817e-12,
        'c': 299792458,
        'r0': 0.001,      # 1 mm radius
        'i': 10,          # 10 A
        'U': 0.6 * 299792458,
        'v': 0.2 * 299792458,
        'r': 0.0005       # inside wire
    }

    def __init__(self, solution_instance):
        # Safe extraction – NEVER use numeric defaults for missing values
        self.results = getattr(solution_instance, 'results', {})
        self.givens = getattr(solution_instance, 'givens', {})
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

        # Required result keys (from [emit] contract)
        self.E_prime = self.results.get('E_prime', None)
        self.B_prime = self.results.get('B_prime', None)
        self.rho_net = self.results.get('rho_net', None)
        self.v_e_prime = self.results.get('v_e_prime', None)
        self.v_i_prime = self.results.get('v_i_prime', None)
        self.rho_e_prime = self.results.get('rho_e_prime', None)
        self.rho_i_prime = self.results.get('rho_i_prime', None)

        # Try to parse symbolic expressions if stored as strings
        self._symbolic = {}

    # ------------------------------------------------------------------
    # Helper methods (not tests)
    # ------------------------------------------------------------------
    def _make_symbolic(self, expr, context=None):
        """Try to convert a string or symbolic expression to a SymPy expression."""
        if expr is None:
            return None
        if isinstance(expr, sp.Basic):
            return expr
        if isinstance(expr, str):
            try:
                # Replace common LaTeX constructs for sympify
                s = expr.replace('\\gamma', 'gamma')
                s = s.replace('\\mu_0', 'mu0')
                s = s.replace('\\varepsilon_0', 'epsilon0')
                s = s.replace('\\pi', 'pi')
                s = s.replace('\\mathbf{e}_', 'e')  # remove vectors
                s = s.replace('{', '')
                s = s.replace('}', '')
                s = s.replace('\\text', '')
                # Environment sympathetic
                local_context = {'gamma': sp.symbols('gamma', positive=True)}
                local_context.update(self._symbolic_params)
                if context:
                    local_context.update(context)
                return sp.sympify(s, locals=local_context)
            except Exception:
                return None
        return None

    def _evaluate_numeric(self, expr_sym, numeric_params=None):
        """Evaluate a SymPy expression with numeric values."""
        if expr_sym is None:
            return None
        if numeric_params is None:
            numeric_params = self._numeric_params
        subs_dict = {sp.Symbol(k): v for k, v in numeric_params.items()}
        try:
            val = float(expr_sym.subs(subs_dict))
            return val
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all required result keys exist."""
        required_keys = [
            'E_prime', 'B_prime', 'rho_net',
            'v_e_prime', 'v_i_prime',
            'rho_e_prime', 'rho_i_prime'
        ]
        missing = [k for k in required_keys if k not in self.results]
        if missing:
            msg = f"Missing keys: {missing}"
            return (False, msg)
        return (True, "All required keys present")

    def test_interface_keys_not_none(self):
        """Check that all result values are not None."""
        if any(getattr(self, k) is None for k in ['E_prime', 'B_prime', 'rho_net',
                                                     'v_e_prime', 'v_i_prime',
                                                     'rho_e_prime', 'rho_i_prime']):
            msg = "One or more result values are None"
            return (False, msg)
        return (True, "All result values non‑None")

    # ------------------------------------------------------------------
    # Invariant tests (theorem‑based, using math.isclose where numeric)
    # ------------------------------------------------------------------
    def test_invariant_lorentz_fields_inside(self):
        """Check that inside wire E' = -γ v B_S and B' = γ B_S."""
        # Build expected expressions
        p = self._symbolic_params
        gamma = 1 / sp.sqrt(1 - p['v']**2 / p['c']**2)
        B_S_inside = p['mu0'] * p['i'] * p['r'] / (2 * sp.pi * p['r0']**2)
        E_expected = -gamma * p['v'] * B_S_inside
        B_expected = gamma * B_S_inside

        # Try to get symbolic forms from results
        E_prime_sym = self._make_symbolic(self.E_prime)
        B_prime_sym = self._make_symbolic(self.B_prime)

        # We need to extract the inside case; assume piecewise structure
        if E_prime_sym is None or B_prime_sym is None:
            return (False, "SKIP: cannot parse E_prime or B_prime symbolically")

        # For inside, we expect the first case (r < r0) to match.
        # Try to extract piecewise parts
        inside_E = None
        inside_B = None
        if isinstance(E_prime_sym, sp.Piecewise):
            # Assume first piece corresponds to r < r0
            inside_E = E_prime_sym.args[0][0]
        else:
            inside_E = E_prime_sym
        if isinstance(B_prime_sym, sp.Piecewise):
            inside_B = B_prime_sym.args[0][0]
        else:
            inside_B = B_prime_sym

        if inside_E is None or inside_B is None:
            return (False, "Cannot extract inside expressions")

        # Simplify difference
        diff_E = sp.simplify(inside_E - E_expected)
        diff_B = sp.simplify(inside_B - B_expected)

        # Evaluate numerically
        diff_E_num = self._evaluate_numeric(diff_E)
        diff_B_num = self._evaluate_numeric(diff_B)

        if diff_E_num is None or diff_B_num is None:
            # fallback: check symbolically that difference is zero (should hold)
            try:
                zero_E = sp.simplify(diff_E) == 0
                zero_B = sp.simplify(diff_B) == 0
                if zero_E and zero_B:
                    return (True, "E' and B' inside match Lorentz transform symbolically")
                else:
                    return (False, "E' or B' inside symbolic mismatch")
            except Exception:
                return (False, "Cannot verify inside fields symbolically")
        else:
            if math.isclose(diff_E_num, 0, rel_tol=1e-2, abs_tol=1e-2) and \
               math.isclose(diff_B_num, 0, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "E' and B' inside match Lorentz transform numerically")
            else:
                return (False, f"E' inside diff = {diff_E_num}, B' inside diff = {diff_B_num}")

    def test_invariant_gauss_law_inside(self):
        """Check that inside the wire ρ_net = ε0 ∇·E'."""
        if self.rho_net is None:
            return (False, "SKIP: rho_net not available")
        if self.E_prime is None:
            return (False, "SKIP: E_prime not available")

        p = self._symbolic_params
        gamma = 1 / sp.sqrt(1 - p['v']**2 / p['c']**2)

        # Expected ρ_net from Gauss law: -γ v i / (π r0^2 c^2)
        rho_expected = -gamma * p['v'] * p['i'] / (sp.pi * p['r0']**2 * p['c']**2)

        # Parse stored ρ_net
        rho_sym = self._make_symbolic(self.rho_net)
        if rho_sym is None:
            return (False, "SKIP: cannot parse rho_net symbolically")

        diff = sp.simplify(rho_sym - rho_expected)
        diff_num = self._evaluate_numeric(diff)
        if diff_num is None:
            try:
                if sp.simplify(diff) == 0:
                    return (True, "rho_net matches Gauss law symbolically")
                else:
                    return (False, "rho_net symbolic mismatch")
            except Exception:
                return (False, "Cannot verify rho_net")
        else:
            if math.isclose(diff_num, 0, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "rho_net matches Gauss law numerically")
            else:
                return (False, f"rho_net diff = {diff_num}")

    def test_invariant_velocity_addition_electrons(self):
        """Check that v_e' follows relativistic velocity addition formula."""
        if self.v_e_prime is None:
            return (False, "SKIP: v_e_prime not available")

        p = self._symbolic_params
        # v_e (in S) = -U
        v_e_S = -p['U']
        # v_e' from formula: (v_e_S - v) / (1 - v_e_S * v / c^2) = (-U - v)/(1 + U v / c^2)
        expected_v_e_prime = (v_e_S - p['v']) / (1 - v_e_S * p['v'] / p['c']**2)
        expected_v_e_prime = sp.simplify(expected_v_e_prime)

        v_sym = self._make_symbolic(self.v_e_prime)
        if v_sym is None:
            return (False, "SKIP: cannot parse v_e_prime symbolically")

        diff = sp.simplify(v_sym - expected_v_e_prime)
        diff_num = self._evaluate_numeric(diff)
        if diff_num is None:
            try:
                if sp.simplify(diff) == 0:
                    return (True, "v_e_prime velocity addition holds symbolically")
                else:
                    return (False, "v_e_prime symbolic mismatch")
            except Exception:
                return (False, "Cannot verify v_e_prime")
        else:
            if math.isclose(diff_num, 0, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "v_e_prime velocity addition holds numerically")
            else:
                return (False, f"v_e_prime diff = {diff_num}")

    def test_invariant_velocity_addition_ions(self):
        """Check that v_i' = -v (ions at rest in S)."""
        if self.v_i_prime is None:
            return (False, "SKIP: v_i_prime not available")

        p = self._symbolic_params
        expected_v_i_prime = -p['v']

        v_sym = self._make_symbolic(self.v_i_prime)
        if v_sym is None:
            return (False, "SKIP: cannot parse v_i_prime symbolically")

        diff = sp.simplify(v_sym - expected_v_i_prime)
        diff_num = self._evaluate_numeric(diff)
        if diff_num is None:
            try:
                if sp.simplify(diff) == 0:
                    return (True, "v_i_prime = -v holds symbolically")
                else:
                    return (False, "v_i_prime symbolic mismatch")
            except Exception:
                return (False, "Cannot verify v_i_prime")
        else:
            if math.isclose(diff_num, 0, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "v_i_prime = -v holds numerically")
            else:
                return (False, f"v_i_prime diff = {diff_num}")

    def test_invariant_charge_density_sum(self):
        """Check that ρ_e' + ρ_i' = ρ_net."""
        if any(getattr(self, k) is None for k in ['rho_e_prime', 'rho_i_prime', 'rho_net']):
            return (False, "SKIP: one or more charge densities missing")

        rho_e_sym = self._make_symbolic(self.rho_e_prime)
        rho_i_sym = self._make_symbolic(self.rho_i_prime)
        rho_net_sym = self._make_symbolic(self.rho_net)

        if None in (rho_e_sym, rho_i_sym, rho_net_sym):
            return (False, "SKIP: cannot parse charge densities symbolically")

        sum_rho = sp.simplify(rho_e_sym + rho_i_sym)
        diff = sp.simplify(sum_rho - rho_net_sym)

        diff_num = self._evaluate_numeric(diff)
        if diff_num is None:
            try:
                if sp.simplify(diff) == 0:
                    return (True, "ρ_e' + ρ_i' = ρ_net holds symbolically")
                else:
                    return (False, "ρ_e' + ρ_i' ≠ ρ_net symbolically")
            except Exception:
                return (False, "Cannot verify charge density sum")
        else:
            if math.isclose(diff_num, 0, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "ρ_e' + ρ_i' = ρ_net holds numerically")
            else:
                return (False, f"ρ_e' + ρ_i' - ρ_net = {diff_num}")

    # ------------------------------------------------------------------
    # Modeling/trace tests
    # ------------------------------------------------------------------
    def test_modeling_field_signs(self):
        """Check that E'_r < 0 and B'_phi > 0 for r > 0 (consistent signs)."""
        # We'll evaluate at our numeric test point inside the wire.
        if self.E_prime is None or self.B_prime is None:
            return (False, "SKIP: fields not available")

        # Try to evaluate E' and B' numerically at a point
        E_sym = self._make_symbolic(self.E_prime)
        B_sym = self._make_symbolic(self.B_prime)

        if E_sym is None or B_sym is None:
            return (False, "SKIP: cannot parse fields symbolically")

        # For the inside point, evaluate
        E_val = self._evaluate_numeric(E_sym)
        B_val = self._evaluate_numeric(B_sym)

        if E_val is None or B_val is None:
            return (False, "SKIP: cannot evaluate fields numerically")

        if E_val < 0 and B_val > 0:
            return (True, "Signs correct: E'_r<0, B'_phi>0")
        else:
            return (False, f"Signs wrong: E'_r={E_val}, B'_phi={B_val}")

    def test_modeling_rho_net_sign(self):
        """Check that net charge density ρ' is negative (since v>0, i>0)."""
        if self.rho_net is None:
            return (False, "SKIP: rho_net not available")

        rho_sym = self._make_symbolic(self.rho_net)
        if rho_sym is None:
            return (False, "SKIP: cannot parse rho_net")

        rho_val = self._evaluate_numeric(rho_sym)
        if rho_val is None:
            return (False, "SKIP: cannot evaluate rho_net numerically")

        if rho_val < 0:
            return (True, "rho_net is negative as expected")
        else:
            return (False, f"rho_net should be negative, got {rho_val}")

    def test_modeling_velocity_signs(self):
        """Check that both velocities are negative (moving in -x direction)."""
        for name, val in [('v_e_prime', self.v_e_prime), ('v_i_prime', self.v_i_prime)]:
            if val is None:
                return (False, f"SKIP: {name} not available")
            v_sym = self._make_symbolic(val)
            if v_sym is None:
                return (False, f"SKIP: cannot parse {name}")
            v_num = self._evaluate_numeric(v_sym)
            if v_num is None:
                return (False, f"SKIP: cannot evaluate {name}")
            if v_num >= 0:
                return (False, f"{name} should be negative, got {v_num}")
        return (True, "Both velocities negative")

    def test_modeling_charge_density_signs(self):
        """Check electron density negative, ion density positive."""
        for name, val in [('rho_e_prime', self.rho_e_prime), ('rho_i_prime', self.rho_i_prime)]:
            if val is None:
                return (False, f"SKIP: {name} not available")
            rho_sym = self._make_symbolic(val)
            if rho_sym is None:
                return (False, f"SKIP: cannot parse {name}")
            rho_num = self._evaluate_numeric(rho_sym)
            if rho_num is None:
                return (False, f"SKIP: cannot evaluate {name}")
            if name == 'rho_e_prime' and rho_num >= 0:
                return (False, f"rho_e_prime should be negative, got {rho_num}")
            if name == 'rho_i_prime' and rho_num <= 0:
                return (False, f"rho_i_prime should be positive, got {rho_num}")
        return (True, "Charge densities signs correct")

    # ------------------------------------------------------------------
    # Monotonicity tests (perturb a parameter and check expected trend)
    # ------------------------------------------------------------------
    def test_monotonicity_E_prime_with_v(self):
        """If v increases, |E'| inside should increase (E' ∝ γ v)."""
        if self.E_prime is None:
            return (False, "SKIP: E_prime not available")

        E_sym = self._make_symbolic(self.E_prime)
        if E_sym is None:
            return (False, "SKIP: cannot parse E_prime")

        # Baseline v
        v0 = self._numeric_params['v']
        gamma0 = 1 / math.sqrt(1 - v0**2 / self._numeric_params['c']**2)
        # Expected E' at baseline: E0
        E0 = self._evaluate_numeric(E_sym)
        if E0 is None:
            return (False, "SKIP: cannot evaluate baseline E'")

        # Perturb v upward by 10%
        v1 = v0 * 1.1
        params1 = self._numeric_params.copy()
        params1['v'] = v1
        E1 = self._evaluate_numeric(E_sym, params1)
        if E1 is None:
            return (False, "SKIP: cannot evaluate perturbed E'")

        # Since E' negative, magnitude should increase (more negative)
        if abs(E1) > abs(E0):
            return (True, "|E'| increases with v as expected")
        else:
            return (False, f"|E'| did not increase: baseline {E0}, perturbed {E1}")

    def test_monotonicity_rho_net_with_v(self):
        """If v increases, |ρ_net| should increase (ρ_net ∝ γv)."""
        if self.rho_net is None:
            return (False, "SKIP: rho_net not available")

        rho_sym = self._make_symbolic(self.rho_net)
        if rho_sym is None:
            return (False, "SKIP: cannot parse rho_net")

        rho0 = self._evaluate_numeric(rho_sym)
        if rho0 is None:
            return (False, "SKIP: cannot evaluate baseline rho_net")

        v1 = self._numeric_params['v'] * 1.1
        params1 = self._numeric_params.copy()
        params1['v'] = v1
        rho1 = self._evaluate_numeric(rho_sym, params1)
        if rho1 is None:
            return (False, "SKIP: cannot evaluate perturbed rho_net")

        if abs(rho1) > abs(rho0):
            return (True, "|ρ_net| increases with v as expected")
        else:
            return (False, f"|ρ_net| did not increase: baseline {rho0}, perturbed {rho1}")

    # ------------------------------------------------------------------
    # Weak answer sanity test (only one, no math.isclose, no re‑derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_sign_consistency(self):
        """
        Weak sanity: all quantities have the physically expected sign and are finite.
        No numeric comparison against reference answer.
        """
        checks = []
        # Check E' sign (we already have a modeling test, but this is a global sanity)
        # We'll just try to evaluate at a sample point and check sign
        for name, expr in [('E_prime', self.E_prime), ('B_prime', self.B_prime),
                          ('rho_net', self.rho_net), ('v_e_prime', self.v_e_prime),
                          ('v_i_prime', self.v_i_prime), ('rho_e_prime', self.rho_e_prime),
                          ('rho_i_prime', self.rho_i_prime)]:
            if expr is None:
                checks.append((name, 'missing'))
                continue
            sym = self._make_symbolic(expr)
            if sym is None:
                checks.append((name, 'cannot parse'))
                continue
            val = self._evaluate_numeric(sym)
            if val is None:
                checks.append((name, 'cannot evaluate'))
                continue
            if math.isfinite(val):
                checks.append((name, f'finite'))
            else:
                checks.append((name, 'not finite'))
        failures = [c[0] for c in checks if 'cannot' in c[1] or 'not finite' in c[1]]
        if failures:
            return (False, f"Sanity issues: {failures}")
        return (True, "All quantities are finite and parseable")

    # ------------------------------------------------------------------
    # Test runner
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)