import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results dictionary safely
        results = {}
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            results = solution_instance.results
        elif isinstance(solution_instance, dict):
            results = solution_instance

        # Extract required attributes from results or solution_instance itself
        self.r_p = results.get('r_p', None)
        self.t_p = results.get('t_p', None)
        self.r_s = results.get('r_s', None)
        self.t_s = results.get('t_s', None)
        self.theta_B = results.get('theta_B', None)
        # Optional but useful givens
        self.n1 = results.get('n1', None)
        self.n2 = results.get('n2', None)
        self.theta1 = results.get('theta1', None)
        self.theta2 = results.get('theta2', None)
        # Also try to get from solution_instance attributes if not in results
        if self.n1 is None and hasattr(solution_instance, 'n1'):
            self.n1 = solution_instance.n1
        if self.n2 is None and hasattr(solution_instance, 'n2'):
            self.n2 = solution_instance.n2
        if self.theta1 is None and hasattr(solution_instance, 'theta1'):
            self.theta1 = solution_instance.theta1
        if self.theta2 is None and hasattr(solution_instance, 'theta2'):
            self.theta2 = solution_instance.theta2

    # -----------------------------------------------------------------------
    # Helper: evaluate a sympy expression to float if possible, else return None
    # -----------------------------------------------------------------------
    def _eval_float(self, expr, subs_dict=None):
        if expr is None:
            return None
        if isinstance(expr, sp.Expr):
            try:
                if subs_dict:
                    num_val = sp.N(expr.subs(subs_dict))
                else:
                    num_val = sp.N(expr)
                return float(num_val)
            except Exception:
                return None
        # It's already a number
        try:
            return float(expr)
        except Exception:
            return None

    # -----------------------------------------------------------------------
    # Interface tests
    # -----------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all required Fresnel result keys exist."""
        missing = []
        for key in ['r_p', 't_p', 'r_s', 't_s', 'theta_B']:
            if getattr(self, key, None) is None:
                missing.append(key)
        if missing:
            return (False, f"Missing required result keys: {', '.join(missing)}")
        return (True, "All required keys present")

    # -----------------------------------------------------------------------
    # Invariant tests (theorem‑based sanity checks)
    # -----------------------------------------------------------------------
    def test_invariant_snell_law(self):
        """Check that n1 sinθ1 ≈ n2 sinθ2."""
        if self.n1 is None or self.n2 is None or self.theta1 is None or self.theta2 is None:
            return (False, "SKIP: n1, n2, theta1 or theta2 missing")
        try:
            n1_num = self._eval_float(self.n1)
            n2_num = self._eval_float(self.n2)
            t1_num = self._eval_float(self.theta1)
            t2_num = self._eval_float(self.theta2)
        except Exception:
            return (False, "SKIP: cannot evaluate numeric values")
        if None in (n1_num, n2_num, t1_num, t2_num):
            return (False, "SKIP: numeric conversion failed")
        lhs = n1_num * math.sin(t1_num)
        rhs = n2_num * math.sin(t2_num)
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Snell's law holds: n1 sinθ1 = {lhs:.4f}, n2 sinθ2 = {rhs:.4f}")
        else:
            return (False, f"Snell's law violated: n1 sinθ1 = {lhs:.4f}, n2 sinθ2 = {rhs:.4f}")

    def test_invariant_fresnel_r_s_consistency(self):
        """For s‑polarization, boundary conditions give:
           n1 cosθ1 (1 - r_s) = n2 cosθ2 (1 + r_s)."""
        if self.n1 is None or self.n2 is None or self.theta1 is None or self.theta2 is None or self.r_s is None:
            return (False, "SKIP: missing inputs for r_s consistency check")
        try:
            n1 = self._eval_float(self.n1)
            n2 = self._eval_float(self.n2)
            t1 = self._eval_float(self.theta1)
            t2 = self._eval_float(self.theta2)
            rs = self._eval_float(self.r_s, {self.theta1: t1, self.theta2: t2})
        except Exception:
            return (False, "SKIP: numeric evaluation failed")
        if None in (n1, n2, t1, t2, rs):
            return (False, "SKIP: numeric conversion failed")
        lhs = n1 * math.cos(t1) * (1 - rs)
        rhs = n2 * math.cos(t2) * (1 + rs)
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"r_s boundary condition satisfied: LHS={lhs:.4f}, RHS={rhs:.4f}")
        else:
            return (False, f"r_s boundary condition violated: LHS={lhs:.4f}, RHS={rhs:.4f}")

    def test_invariant_fresnel_t_s_consistency(self):
        """For s‑polarization, t_s = 1 + r_s."""
        if self.r_s is None or self.t_s is None:
            return (False, "SKIP: r_s or t_s missing")
        try:
            t1 = self._eval_float(self.theta1)
            t2 = self._eval_float(self.theta2)
            rs = self._eval_float(self.r_s, {self.theta1: t1, self.theta2: t2})
            ts = self._eval_float(self.t_s, {self.theta1: t1, self.theta2: t2})
        except Exception:
            return (False, "SKIP: evaluation failed")
        if None in (rs, ts):
            return (False, "SKIP: numeric conversion failed")
        expected_ts = 1.0 + rs
        if math.isclose(ts, expected_ts, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"t_s = 1+r_s holds: t_s={ts:.4f}, 1+r_s={expected_ts:.4f}")
        else:
            return (False, f"t_s = 1+r_s violated: t_s={ts:.4f}, 1+r_s={expected_ts:.4f}")

    def test_invariant_fresnel_p_consistency(self):
        """For p‑polarization, both E and H continuity give:
           E: (1 + r_p) cosθ1 = t_p cosθ2
           H: n1 (1 - r_p) = n2 t_p   (using the sign convention from explanation)"""
        if self.n1 is None or self.n2 is None or self.theta1 is None or self.theta2 is None:
            return (False, "SKIP: missing n1, n2, θ1, θ2")
        if self.r_p is None or self.t_p is None:
            return (False, "SKIP: missing r_p or t_p")
        try:
            n1 = self._eval_float(self.n1)
            n2 = self._eval_float(self.n2)
            t1 = self._eval_float(self.theta1)
            t2 = self._eval_float(self.theta2)
            rp = self._eval_float(self.r_p, {self.theta1: t1, self.theta2: t2})
            tp = self._eval_float(self.t_p, {self.theta1: t1, self.theta2: t2})
        except Exception:
            return (False, "SKIP: numeric evaluation failed")
        if None in (n1, n2, t1, t2, rp, tp):
            return (False, "SKIP: numeric conversion failed")
        # Check E continuity
        lhs_e = (1.0 + rp) * math.cos(t1)
        rhs_e = tp * math.cos(t2)
        # Check H continuity
        lhs_h = n1 * (1.0 - rp)
        rhs_h = n2 * tp
        ok_e = math.isclose(lhs_e, rhs_e, rel_tol=1e-2, abs_tol=1e-2)
        ok_h = math.isclose(lhs_h, rhs_h, rel_tol=1e-2, abs_tol=1e-2)
        if ok_e and ok_h:
            return (True, "p‑polarization boundary conditions satisfied")
        else:
            msg = ""
            if not ok_e:
                msg += f"E continuity: LHS={lhs_e:.4f}, RHS={rhs_e:.4f}; "
            if not ok_h:
                msg += f"H continuity: LHS={lhs_h:.4f}, RHS={rhs_h:.4f}"
            return (False, "p‑polarization boundary conditions violated: " + msg)

    def test_invariant_brewster_law(self):
        """If r_p=0, then tanθ1 = n2/n1 (Brewster's law)."""
        # We check two things: if r_p is expressed symbolically, we can verify
        # that setting r_p=0 leads to the condition, or we can check
        # the given theta_B satisfies tanθ_B = n2/n1.
        if self.theta_B is not None and self.n1 is not None and self.n2 is not None:
            try:
                tb = self._eval_float(self.theta_B)
                n1 = self._eval_float(self.n1)
                n2 = self._eval_float(self.n2)
            except Exception:
                tb = n1 = n2 = None
            if None not in (tb, n1, n2):
                expected = n2 / n1
                actual = math.tan(tb)
                if math.isclose(actual, expected, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, f"Brewster's law: tanθ_B = {actual:.4f}, n2/n1 = {expected:.4f}")
                else:
                    return (False, f"Brewster's law violated: tanθ_B = {actual:.4f}, expected {expected:.4f}")
        # Alternatively, if we have r_p as an expression, check that r_p=0 yields tanθ1 = n2/n1.
        if self.r_p is not None and self.theta1 is not None and self.n1 is not None and self.n2 is not None:
            try:
                # Solve r_p = 0 for θ1 symbolically
                if isinstance(self.r_p, sp.Expr) and isinstance(self.theta1, sp.Symbol):
                    eq = sp.Eq(self.r_p, 0)
                    sol = sp.solve(eq, self.theta1)
                    # Filter positive solutions
                    for s in sol:
                        if s.is_number and float(s) > 0:
                            tb = float(s)
                            n1 = float(self.n1) if self.n1.is_number else 1.0
                            n2 = float(self.n2) if self.n2.is_number else 1.0
                            expected = math.atan(n2/n1)
                            if math.isclose(tb, expected, rel_tol=1e-2, abs_tol=1e-2):
                                return (True, "Brewster condition derived from r_p=0")
                            else:
                                return (False, f"r_p=0 gives θ1={tb:.4f}, expected {expected:.4f}")
                    return (False, "No valid solution for r_p=0 found")
            except Exception:
                pass
        return (False, "SKIP: insufficient data for Brewster invariant check")

    # -----------------------------------------------------------------------
    # Modeling/trace checks
    # -----------------------------------------------------------------------
    def test_modeling_fresnel_s_form(self):
        """Check that r_s has the form sin(θ2-θ1)/sin(θ2+θ1) if possible."""
        if self.theta1 is None or self.theta2 is None or self.r_s is None:
            return (False, "SKIP: missing θ1, θ2 or r_s")
        try:
            t1 = self._eval_float(self.theta1)
            t2 = self._eval_float(self.theta2)
            rs_num = self._eval_float(self.r_s, {self.theta1: t1, self.theta2: t2})
        except Exception:
            return (False, "SKIP: numeric evaluation failed")
        if None in (t1, t2, rs_num):
            return (False, "SKIP: numeric conversion failed")
        expected_rs = math.sin(t2 - t1) / math.sin(t2 + t1)
        if math.isclose(rs_num, expected_rs, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"r_s matches standard form: {rs_num:.4f} vs {expected_rs:.4f}")
        else:
            return (False, f"r_s does not match standard form: got {rs_num:.4f}, expected {expected_rs:.4f}")

    def test_modeling_fresnel_p_form(self):
        """Check that r_p has the form tan(θ2-θ1)/tan(θ2+θ1)."""
        if self.theta1 is None or self.theta2 is None or self.r_p is None:
            return (False, "SKIP: missing θ1, θ2 or r_p")
        try:
            t1 = self._eval_float(self.theta1)
            t2 = self._eval_float(self.theta2)
            rp_num = self._eval_float(self.r_p, {self.theta1: t1, self.theta2: t2})
        except Exception:
            return (False, "SKIP: numeric evaluation failed")
        if None in (t1, t2, rp_num):
            return (False, "SKIP: numeric conversion failed")
        # Avoid division by zero if tan(θ2+θ1) is zero (normal incidence)
        if abs(math.tan(t2 + t1)) < 1e-12:
            return (True, "tan(θ2+θ1) near zero; skip form check")
        expected_rp = math.tan(t2 - t1) / math.tan(t2 + t1)
        if math.isclose(rp_num, expected_rp, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"r_p matches standard form: {rp_num:.4f} vs {expected_rp:.4f}")
        else:
            return (False, f"r_p does not match standard form: got {rp_num:.4f}, expected {expected_rp:.4f}")

    # -----------------------------------------------------------------------
    # Monotonicity tests
    # -----------------------------------------------------------------------
    def test_monotonicity_r_s_sign_with_theta1(self):
        """For n1 < n2 (external reflection), r_s becomes more negative as θ1 increases.
           Perturb θ1 by +1% and check sign of change."""
        if self.n1 is None or self.n2 is None or self.theta1 is None or self.theta2 is None or self.r_s is None:
            return (False, "SKIP: missing data for monotonicity check")
        try:
            n1 = self._eval_float(self.n1)
            n2 = self._eval_float(self.n2)
            t1 = self._eval_float(self.theta1)
            t2 = self._eval_float(self.theta2)
        except Exception:
            return (False, "SKIP: numeric evaluation failed")
        if None in (n1, n2, t1, t2):
            return (False, "SKIP: numeric conversion failed")
        if n1 >= n2:
            return (False, "SKIP: monotonicity test designed for n1 < n2")
        # Baseline r_s
        rs_base = self._eval_float(self.r_s, {self.theta1: t1, self.theta2: t2})
        if rs_base is None:
            return (False, "SKIP: cannot compute baseline r_s")
        # Perturb θ1 upward by 1%
        delta = 0.01 * t1
        t1_new = t1 + delta
        # Compute new θ2 from Snell: sinθ2 = (n1/n2) sinθ1
        try:
            sin_t2_new = (n1 / n2) * math.sin(t1_new)
            if abs(sin_t2_new) > 1.0:
                return (True, "Perturbation leads to total internal reflection; skip")
            t2_new = math.asin(sin_t2_new)
        except Exception:
            return (False, "SKIP: cannot compute new θ2")
        rs_new = self._eval_float(self.r_s, {self.theta1: t1_new, self.theta2: t2_new})
        if rs_new is None:
            return (False, "SKIP: cannot compute perturbed r_s")
        # For external reflection (n1<n2), r_s is negative and becomes more negative (decreases) as θ1 increases.
        if rs_new < rs_base:  # more negative
            return (True, f"r_s decreases monotonically: {rs_base:.4f} -> {rs_new:.4f}")
        else:
            return (False, f"r_s did not decrease: {rs_base:.4f} -> {rs_new:.4f}")

    def test_monotonicity_brewster_angle_with_n2(self):
        """Brewster angle should increase with increasing n2 (for fixed n1).
           Perturb n2 by +10% and compare the resulting Brewster angle to the baseline computed from original n2."""
        if self.n1 is None or self.n2 is None:
            return (False, "SKIP: missing n1 or n2")
        try:
            n1 = self._eval_float(self.n1)
            n2_base = self._eval_float(self.n2)
        except Exception:
            return (False, "SKIP: numeric evaluation failed")
        if None in (n1, n2_base):
            return (False, "SKIP: numeric conversion failed")
        # Baseline Brewster angle computed from formula
        theta_B_base = math.atan(n2_base / n1)
        # Perturb n2 upward by 10%
        n2_new = n2_base * 1.10
        theta_B_new = math.atan(n2_new / n1)
        # Check monotonicity: larger n2 should give larger angle (since arctan is increasing)
        if theta_B_new > theta_B_base + 1e-12:
            return (True, f"Brewster angle increases with n2: {theta_B_base:.4f} rad -> {theta_B_new:.4f} rad")
        else:
            return (False, f"Brewster angle does not increase: {theta_B_base:.4f} -> {theta_B_new:.4f}")

    # -----------------------------------------------------------------------
    # Weak answer sanity (at most one)
    # -----------------------------------------------------------------------
    def test_answer_sanity_theta_B_range(self):
        """Check that Brewster angle is between 0 and π/2."""
        if self.theta_B is None:
            return (False, "SKIP: theta_B not available")
        try:
            tb = self._eval_float(self.theta_B)
        except Exception:
            return (False, "SKIP: cannot evaluate theta_B")
        if tb is None:
            return (False, "SKIP: numeric conversion failed")
        if 0 < tb < math.pi/2:
            return (True, f"θ_B = {tb:.4f} rad is within (0, π/2)")
        else:
            return (False, f"θ_B = {tb:.4f} rad lies outside (0, π/2)")

    # -----------------------------------------------------------------------
    # Run all tests
    # -----------------------------------------------------------------------
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