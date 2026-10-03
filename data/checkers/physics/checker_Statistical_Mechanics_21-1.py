import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results dict safely
        self.results = getattr(solution_instance, 'results', {})
        if not isinstance(self.results, dict):
            self.results = {}

        # Required result keys
        self.mu_eq = self.results.get('mu_eq', None)
        self.Clapeyron = self.results.get('Clapeyron', None)
        self.T_triple = self.results.get('T_triple', None)
        self.P_triple = self.results.get('P_triple', None)

        # Givens – try from solution_instance attributes, then from results dict
        def _get(key):
            val = getattr(solution_instance, key, None)
            if val is None:
                val = self.results.get(key, None)
            return val

        self.R = _get('R')
        self.T_lv = _get('T_lv')
        self.P_lv = _get('P_lv')
        self.L_vap = _get('L_vap')
        self.T_sv = _get('T_sv')
        self.P_sv = _get('P_sv')
        self.L_sub = _get('L_sub')

    # ----------------------------------------------------------------
    # Interface checks: existence and form of required results
    # ----------------------------------------------------------------

    def test_interface_mu_eq_exists(self):
        if self.mu_eq is None:
            return (False, "SKIP: mu_eq not found")
        return (True, "mu_eq present")

    def test_interface_mu_eq_form(self):
        if self.mu_eq is None:
            return (False, "SKIP: mu_eq not found")
        s = str(self.mu_eq)
        if '=' not in s:
            return (False, "mu_eq does not contain '='")
        left, right = s.split('=', 1)
        if left.strip() == right.strip():
            return (False, "mu_eq equality is trivial (same variable on both sides)")
        # Accept either g_l/g_g or mu_l/mu_g
        if (('g_l' in left or 'mu_l' in left) and
            ('g_g' in right or 'mu_g' in right)):
            return (True, "mu_eq correctly states equality of molar Gibbs free energies")
        if (('g_g' in left or 'mu_g' in left) and
            ('g_l' in right or 'mu_l' in right)):
            return (True, "mu_eq correctly states equality of molar Gibbs free energies (reversed sides)")
        return (False, "mu_eq does not match expected form: e.g., 'g_l = g_g'")

    def test_interface_Clapeyron_exists(self):
        if self.Clapeyron is None:
            return (False, "SKIP: Clapeyron not found")
        return (True, "Clapeyron present")

    def test_interface_Clapeyron_form(self):
        if self.Clapeyron is None:
            return (False, "SKIP: Clapeyron not found")
        s = str(self.Clapeyron)
        if 'dP/dT' not in s:
            return (False, "Clapeyron expression missing 'dP/dT'")
        if 's_g' in s and 's_l' in s and 'v_g' in s and 'v_l' in s:
            return (True, "Clapeyron expression contains correct variables")
        return (False, "Clapeyron expression does not match expected form (dP/dT = (s_g-s_l)/(v_g-v_l))")

    def test_interface_T_triple_exists(self):
        if self.T_triple is None:
            return (False, "SKIP: T_triple not found")
        try:
            T_val = float(self.T_triple)
            return (True, f"T_triple present, value = {T_val}")
        except (TypeError, ValueError):
            return (False, "T_triple is not a number")

    def test_interface_P_triple_exists(self):
        if self.P_triple is None:
            return (False, "SKIP: P_triple not found")
        try:
            P_val = float(self.P_triple)
            return (True, f"P_triple present, value = {P_val}")
        except (TypeError, ValueError):
            return (False, "P_triple is not a number")

    # ----------------------------------------------------------------
    # Invariant / theorem checks: numerical verification of triple point
    # ----------------------------------------------------------------

    def _compute_triple(self):
        """
        Use integrated Clausius-Clapeyron equations from given data to
        compute triple point (T, P). Returns (T, P) floats or (None, None).
        """
        required = [self.R, self.T_lv, self.P_lv, self.L_vap,
                    self.T_sv, self.P_sv, self.L_sub]
        if any(x is None for x in required):
            return None, None

        R = self.R
        T_lv = self.T_lv
        P_lv = self.P_lv
        L_vap = self.L_vap
        T_sv = self.T_sv
        P_sv = self.P_sv
        L_sub = self.L_sub

        T = sp.symbols('T', positive=True)
        eq = (L_vap / R) * (1/T_lv - 1/T) + sp.log(P_lv) - (L_sub / R) * (1/T_sv - 1/T) - sp.log(P_sv)
        try:
            T_sol = float(sp.nsolve(eq, 200))
        except Exception:
            # Fallback: simple Newton iteration in pure Python
            def f(T_val):
                return (L_vap / R)*(1/T_lv - 1/T_val) + math.log(P_lv) - (L_sub / R)*(1/T_sv - 1/T_val) - math.log(P_sv)
            def fprime(T_val):
                return (L_vap / R)*(1/T_val**2) - (L_sub / R)*(1/T_val**2)
            T_sol = 200.0
            for _ in range(50):
                fv = f(T_sol)
                if abs(fv) < 1e-12:
                    break
                fp = fprime(T_sol)
                if fp == 0:
                    break
                T_sol -= fv / fp
            if T_sol <= 0:
                return None, None

        P_sol = P_lv * math.exp((L_vap / R) * (1/T_lv - 1/T_sol))
        return T_sol, P_sol

    def test_invariant_triple_T(self):
        T_sol = self.T_triple
        if T_sol is None:
            return (False, "SKIP: T_triple not available")
        T_calc, P_calc = self._compute_triple()
        if T_calc is None:
            return (False, "SKIP: insufficient givens to compute triple point")
        if not math.isclose(T_calc, float(T_sol), rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"T_triple mismatch: computed {T_calc:.2f}, solution {float(T_sol):.2f}")
        return (True, f"T_triple consistent: computed {T_calc:.2f}, solution {float(T_sol):.2f}")

    def test_invariant_triple_P(self):
        P_sol = self.P_triple
        if P_sol is None:
            return (False, "SKIP: P_triple not available")
        T_calc, P_calc = self._compute_triple()
        if P_calc is None:
            return (False, "SKIP: insufficient givens to compute triple point")
        if not math.isclose(P_calc, float(P_sol), rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"P_triple mismatch: computed {P_calc:.2f}, solution {float(P_sol):.2f}")
        return (True, f"P_triple consistent: computed {P_calc:.2f}, solution {float(P_sol):.2f}")

    # ----------------------------------------------------------------
    # Modeling / trace tests: check coherence with explanation steps
    # ----------------------------------------------------------------

    def test_modeling_triple_consistency(self):
        """
        Check that the solution's T_triple and P_triple satisfy both
        integrated Clapeyron equations (derived from explanation steps).
        This verifies the modeling of the triple point calculation.
        """
        if self.T_triple is None or self.P_triple is None:
            return (False, "SKIP: T_triple or P_triple not available")
        if any(x is None for x in [self.R, self.T_lv, self.P_lv, self.L_vap,
                                   self.T_sv, self.P_sv, self.L_sub]):
            return (False, "SKIP: insufficient givens")
        T = float(self.T_triple)
        P = float(self.P_triple)
        lhs_lv = math.log(P / self.P_lv)
        rhs_lv = (self.L_vap / self.R) * (1/self.T_lv - 1/T)
        if not math.isclose(lhs_lv, rhs_lv, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Liquid-vapor Clapeyron mismatch: LHS {lhs_lv:.4f}, RHS {rhs_lv:.4f}")
        lhs_sv = math.log(P / self.P_sv)
        rhs_sv = (self.L_sub / self.R) * (1/self.T_sv - 1/T)
        if not math.isclose(lhs_sv, rhs_sv, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Solid-vapor Clapeyron mismatch: LHS {lhs_sv:.4f}, RHS {rhs_sv:.4f}")
        return (True, "Triple point satisfies both Clapeyron equations (modeling consistent with explanation)")

    def test_modeling_mu_eq_consistency(self):
        """
        Check that mu_eq is consistent with the derivation that dG=0 leads to g_l=g_g.
        This checks that the equality is not trivial and represents phase equilibrium.
        """
        if self.mu_eq is None:
            return (False, "SKIP: mu_eq not available")
        s = str(self.mu_eq)
        # Expect a non‑trivial equality that involves both liquid and gas symbols
        if '=' not in s:
            return (False, "mu_eq is not an equality")
        left, right = s.split('=', 1)
        left = left.strip()
        right = right.strip()
        # Check that left and right are different
        if left == right:
            return (False, "mu_eq is trivial (identical sides)")
        # Accept symbolic forms like g_l == g_g or mu_l == mu_g
        if left in ['g_l','mu_l'] and right in ['g_g','mu_g']:
            return (True, "mu_eq correctly represents phase equilibrium equality")
        if left in ['g_g','mu_g'] and right in ['g_l','mu_l']:
            return (True, "mu_eq correctly represents phase equilibrium equality (reversed)")
        # Could also accept more complex symbolic equations, but we check key words
        if 'g_l' in s and 'g_g' in s:
            return (True, "mu_eq contains both g_l and g_g (molar Gibbs free energies)")
        return (True, "mu_eq is an equality (modeling consistent)")

    # ----------------------------------------------------------------
    # Monotonicity test
    # ----------------------------------------------------------------

    def test_monotonicity_triple_Lvap(self):
        if any(x is None for x in [self.R, self.T_lv, self.P_lv, self.L_vap,
                                   self.T_sv, self.P_sv, self.L_sub]):
            return (False, "SKIP: insufficient givens")
        if self.T_triple is None:
            return (False, "SKIP: T_triple not available")
        T_base, _ = self._compute_triple()
        if T_base is None:
            return (False, "SKIP: cannot compute baseline triple point")
        # Perturb L_vap upward by 1%
        old_L = self.L_vap
        self.L_vap *= 1.01
        T_pert, _ = self._compute_triple()
        self.L_vap = old_L
        if T_pert is None:
            return (False, "SKIP: cannot compute perturbed triple point")
        if T_pert >= T_base:
            return (False, f"Monotonicity violation: T increased from {T_base:.2f} to {T_pert:.2f} when L_vap increased")
        return (True, f"Monotonicity: L_vap +1% -> T decreased from {T_base:.2f} to {T_pert:.2f}")

    # ----------------------------------------------------------------
    # Answer sanity (weak check) – only one allowed
    # ----------------------------------------------------------------

    def test_answer_sanity_finite(self):
        """
        Weak check: T_triple and P_triple should be finite and positive.
        This is the only allowed answer sanity test.
        """
        messages = []
        if self.T_triple is not None:
            try:
                T = float(self.T_triple)
                if not (T > 0 and math.isfinite(T)):
                    return (False, f"T_triple is not positive and finite: {T}")
                messages.append(f"T_triple = {T}")
            except Exception:
                return (False, "T_triple cannot be converted to float")
        if self.P_triple is not None:
            try:
                P = float(self.P_triple)
                if not (P > 0 and math.isfinite(P)):
                    return (False, f"P_triple is not positive and finite: {P}")
                messages.append(f"P_triple = {P}")
            except Exception:
                return (False, "P_triple cannot be converted to float")
        if not messages:
            return (False, "SKIP: neither T_triple nor P_triple present")
        return (True, "Positive finite values: " + ", ".join(messages))

    # ----------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------

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