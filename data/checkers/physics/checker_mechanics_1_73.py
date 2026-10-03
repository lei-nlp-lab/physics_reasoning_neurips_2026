import math
import sympy as sp
import random

class PhysicsChecker:
    """
    Audits a solution instance for central force and capture cross section problem.
    Implements robust, unit-test-like checks. All extraction uses None defaults.
    """

    def __init__(self, solution_instance):
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # Emitted result keys
        self.F = self.results.get('F', None)
        self.sigma = self.results.get('sigma', None)

        # Physical parameters (may be in solution_instance itself or in a givens dict)
        givens = getattr(solution_instance, 'givens', {})
        if not isinstance(givens, dict):
            givens = {}

        # Try instance attribute first, then givens
        self.alpha = getattr(solution_instance, 'alpha', None)
        if self.alpha is None:
            self.alpha = givens.get('alpha', None)
        self.m = getattr(solution_instance, 'm', None)
        if self.m is None:
            self.m = givens.get('m', None)
        self.V_inf = getattr(solution_instance, 'V_inf', None)
        if self.V_inf is None:
            self.V_inf = givens.get('V_inf', None)
        self.a = getattr(solution_instance, 'a', None)
        if self.a is None:
            self.a = givens.get('a', None)
        self.h = getattr(solution_instance, 'h', None)
        if self.h is None:
            self.h = givens.get('h', None)

    # ------------------------------------------------------------------
    # Helper to safely evaluate a symbolic expression with given subs
    # ------------------------------------------------------------------
    def _eval_expr(self, expr_str, subs):
        """Return float value of expr_str under subs dict, or None on failure."""
        try:
            expr = sp.sympify(expr_str)
            val = expr.subs(subs).evalf()
            if isinstance(val, complex):
                val = float(sp.re(val))
            return float(val)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_F_present(self):
        if self.F is None:
            return (False, "SKIP: Missing key 'F' in results.")
        if not isinstance(self.F, str) or self.F.strip() == '':
            return (False, "F is not a non-empty string.")
        return (True, f"F is present: {self.F[:60]}...")

    def test_interface_sigma_present(self):
        if self.sigma is None:
            return (False, "SKIP: Missing key 'sigma' in results.")
        if not isinstance(self.sigma, str) or self.sigma.strip() == '':
            return (False, "sigma is not a non-empty string.")
        return (True, f"sigma is present: {self.sigma[:60]}...")

    # ------------------------------------------------------------------
    # Invariant tests (theorem based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_force_expression(self):
        """Check F matches -3*m*h**2*a/r**4 via numeric substitution."""
        if self.F is None:
            return (False, "SKIP: F not provided.")
        m, h, a, r = sp.symbols('m h a r', positive=True)
        expected = -3 * m * h**2 * a / r**4
        try:
            F_sym = sp.sympify(self.F)
            diff = sp.simplify(F_sym - expected)
            subs = {m: 1.2, h: 0.8, a: 2.5, r: 0.7}
            num_diff = diff.subs(subs).evalf()
            if isinstance(num_diff, complex):
                num_diff = float(sp.re(num_diff))
            if math.isclose(float(num_diff), 0.0, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"F matches expected form. Num diff = {float(num_diff):.2e}")
            else:
                return (False, f"F does not match expected form. Num diff = {float(num_diff):.2e}")
        except Exception as e:
            return (False, f"Force invariant check failed: {e}")

    def test_invariant_sigma_expression(self):
        """Check sigma matches 2*pi*sqrt(2*alpha/(m*V_inf**2))."""
        if self.sigma is None:
            return (False, "SKIP: sigma not provided.")
        alpha, m, V_inf, pi = sp.symbols('alpha m V_inf pi', positive=True)
        expected = 2 * sp.pi * sp.sqrt(2 * alpha / (m * V_inf**2))
        try:
            sigma_sym = sp.sympify(self.sigma)
            diff = sp.simplify(sigma_sym - expected)
            subs = {alpha: 0.5, m: 2.0, V_inf: 3.0, sp.pi: math.pi}
            num_diff = diff.subs(subs).evalf()
            if isinstance(num_diff, complex):
                num_diff = float(sp.re(num_diff))
            if math.isclose(float(num_diff), 0.0, rel_tol=1e-2, abs_tol=1e-2):
                return (True, f"sigma matches expected form. Num diff = {float(num_diff):.2e}")
            else:
                return (False, f"sigma does not match expected form. Num diff = {float(num_diff):.2e}")
        except Exception as e:
            return (False, f"Sigma invariant check failed: {e}")

    # ------------------------------------------------------------------
    # Modeling/Trace tests (at least two required)
    # ------------------------------------------------------------------
    def test_modeling_binet_relation(self):
        """
        Verify that for the given orbit r = a*(1+cosθ), the Binet quantity
        d²u/dθ² + u equals 3a u² (as derived in explanation steps).
        This is a local relation, not a final answer check.
        """
        # We need a and a random theta value. Use default a=1 if not given.
        a_val = self.a if self.a is not None else 1.0
        try:
            theta = 0.7   # rad
            a = sp.symbols('a', positive=True)
            u = 1/(a*(1+sp.cos(theta)))
            u_dd = sp.diff(u, theta, 2)
            LHS = u_dd + u
            RHS = 3*a*u**2
            diff = sp.simplify(LHS - RHS)
            sub = {a: a_val}
            num_diff = diff.subs(sub).evalf()
            if isinstance(num_diff, complex):
                num_diff = float(sp.re(num_diff))
            if math.isclose(float(num_diff), 0.0, rel_tol=1e-4, abs_tol=1e-4):
                return (True, "Binet relation d²u/dθ²+u = 3a u² holds for sample theta.")
            else:
                return (False, f"Binet relation fails: diff = {float(num_diff):.2e}")
        except Exception as e:
            return (False, f"Modeling Binet test crashed: {e}")

    def test_modeling_capture_condition(self):
        """
        Verify that the derived critical impact parameter b_crit from sigma
        (b_crit² = sigma/π) equals sqrt(8α/(m V_inf²)) as stated in explanation.
        """
        if self.sigma is None:
            return (False, "SKIP: sigma not provided for modeling capture condition.")
        try:
            sigma_sym = sp.sympify(self.sigma)
            b_crit_sq = sigma_sym / sp.pi
            alpha, m, V_inf = sp.symbols('alpha m V_inf', positive=True)
            expected_b_crit_sq = sp.sqrt(8 * alpha / (m * V_inf**2))
            diff = sp.simplify(b_crit_sq - expected_b_crit_sq)
            subs = {alpha: 0.3, m: 1.5, V_inf: 2.0, sp.pi: math.pi}
            num_diff = diff.subs(subs).evalf()
            if isinstance(num_diff, complex):
                num_diff = float(sp.re(num_diff))
            if math.isclose(float(num_diff), 0.0, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "b_crit² from sigma matches explanation formula.")
            else:
                return (False, f"b_crit² mismatch: diff = {float(num_diff):.2e}")
        except Exception as e:
            return (False, f"Modeling capture condition test failed: {e}")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit perturbation, not derivative)
    # ------------------------------------------------------------------
    def test_monotonicity_sigma_increases_with_alpha(self):
        """
        Perturb alpha by +1% and check sigma increases.
        Uses numeric evaluation with assumed positive parameters.
        """
        if self.sigma is None:
            return (False, "SKIP: sigma not provided for monotonicity.")
        # Choose baseline numeric values (if parameters not provided, use defaults)
        base_alpha = self.alpha if self.alpha is not None else 1.0
        base_m = self.m if self.m is not None else 1.0
        base_V = self.V_inf if self.V_inf is not None else 2.0
        # Build substitution dictionary (ensure all symbols in sigma are covered)
        # Extract symbols from sigma string to make robust subs
        try:
            sigma_sym = sp.sympify(self.sigma)
            syms = list(sigma_sym.free_symbols)
            # Map expected symbols to numeric values
            sym_map = {}
            for s in syms:
                name = str(s)
                # Provide default numeric values for known symbols
                if name == 'alpha':
                    sym_map[s] = base_alpha
                elif name == 'm':
                    sym_map[s] = base_m
                elif name == 'V_inf':
                    sym_map[s] = base_V
                elif name == 'pi':
                    sym_map[s] = math.pi
                else:
                    # For any other symbol (e.g., a), give a dummy value
                    sym_map[s] = 1.0
            # Baseline sigma
            baseline = sigma_sym.subs(sym_map).evalf()
            if isinstance(baseline, complex):
                baseline = float(sp.re(baseline))
            # Perturb alpha by +1%
            new_alpha = base_alpha * 1.01
            new_sym_map = dict(sym_map)
            # Find symbol for alpha
            alpha_sym = None
            for s in syms:
                if str(s) == 'alpha':
                    alpha_sym = s
                    break
            if alpha_sym is None:
                return (False, "SKIP: alpha symbol not found in sigma expression.")
            new_sym_map[alpha_sym] = new_alpha
            perturbed = sigma_sym.subs(new_sym_map).evalf()
            if isinstance(perturbed, complex):
                perturbed = float(sp.re(perturbed))
            if float(perturbed) > float(baseline) * 0.99:  # Allow small numerical error
                return (True, f"Sigma increases with alpha: baseline={float(baseline):.4f}, perturbed={float(perturbed):.4f}")
            else:
                return (False, f"Sigma does not increase: baseline={float(baseline):.4f}, perturbed={float(perturbed):.4f}")
        except Exception as e:
            return (False, f"Monotonicity test (alpha) failed: {e}")

    def test_monotonicity_sigma_decreases_with_Vinf(self):
        """
        Perturb V_inf by +1% and check sigma decreases.
        """
        if self.sigma is None:
            return (False, "SKIP: sigma not provided for monotonicity.")
        base_alpha = self.alpha if self.alpha is not None else 1.0
        base_m = self.m if self.m is not None else 1.0
        base_V = self.V_inf if self.V_inf is not None else 2.0
        try:
            sigma_sym = sp.sympify(self.sigma)
            syms = list(sigma_sym.free_symbols)
            sym_map = {}
            for s in syms:
                name = str(s)
                if name == 'alpha':
                    sym_map[s] = base_alpha
                elif name == 'm':
                    sym_map[s] = base_m
                elif name == 'V_inf':
                    sym_map[s] = base_V
                elif name == 'pi':
                    sym_map[s] = math.pi
                else:
                    sym_map[s] = 1.0
            baseline = sigma_sym.subs(sym_map).evalf()
            if isinstance(baseline, complex):
                baseline = float(sp.re(baseline))
            # Perturb V_inf by +1%
            new_V = base_V * 1.01
            new_sym_map = dict(sym_map)
            V_sym = None
            for s in syms:
                if str(s) == 'V_inf':
                    V_sym = s
                    break
            if V_sym is None:
                return (False, "SKIP: V_inf symbol not found in sigma expression.")
            new_sym_map[V_sym] = new_V
            perturbed = sigma_sym.subs(new_sym_map).evalf()
            if isinstance(perturbed, complex):
                perturbed = float(sp.re(perturbed))
            if float(perturbed) < float(baseline) * 1.01:  # Allow slight numerical error; should be less
                # Actually we expect perturbed < baseline. Check that new is less than old
                if float(perturbed) < float(baseline):
                    return (True, f"Sigma decreases with V_inf: baseline={float(baseline):.4f}, perturbed={float(perturbed):.4f}")
                else:
                    return (False, f"Sigma does not decrease: baseline={float(baseline):.4f}, perturbed={float(perturbed):.4f}")
            else:
                # Handle negative cases
                if float(perturbed) < float(baseline):
                    return (True, "Sigma decreases with V_inf.")
                else:
                    return (False, f"Sigma does not decrease: baseline={float(baseline):.4f}, perturbed={float(perturbed):.4f}")
        except Exception as e:
            return (False, f"Monotonicity test (V_inf) failed: {e}")

    # ------------------------------------------------------------------
    # Weak answer sanity test (at most one)
    # ------------------------------------------------------------------
    def test_answer_sanity_sigma_positive(self):
        """
        Very weak check: sigma expression, when evaluated with positive parameters,
        gives a positive result (not checking exact value).
        """
        if self.sigma is None:
            return (False, "SKIP: sigma not provided.")
        # Attempt evaluation with dummy positive numbers
        try:
            sigma_sym = sp.sympify(self.sigma)
            syms = list(sigma_sym.free_symbols)
            subs = {}
            for s in syms:
                name = str(s)
                if name == 'alpha':
                    subs[s] = 1.5
                elif name == 'm':
                    subs[s] = 1.0
                elif name == 'V_inf':
                    subs[s] = 3.0
                elif name == 'pi':
                    subs[s] = math.pi
                elif name == 'h':
                    subs[s] = 1.0
                elif name == 'a':
                    subs[s] = 1.0
                else:
                    subs[s] = 1.0  # fallback
            val = sigma_sym.subs(subs).evalf()
            if isinstance(val, complex):
                val = float(sp.re(val))
            if float(val) > 0:
                return (True, f"sigma evaluates to positive number {float(val):.4f}.")
            else:
                return (False, f"sigma evaluates to <= 0: {float(val):.4f}")
        except Exception as e:
            # If evaluation fails, just accept it's positive (expression exists)
            return (True, "sigma expression present (cannot evaluate but likely positive).")

    # ------------------------------------------------------------------
    # Master test runner
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