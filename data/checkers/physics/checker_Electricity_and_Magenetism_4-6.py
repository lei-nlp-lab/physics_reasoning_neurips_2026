import sympy as sp
import math
import sys

class PhysicsChecker:
    """
    Auditor for dipole electric field problem.
    Checks consistency with theoretical formulas and constraints.
    """

    def __init__(self, solution_instance):
        # Extract results dictionary safely
        self.results = getattr(solution_instance, 'results', {})
        # Extract given constants (may be None)
        self.p = getattr(solution_instance, 'p', None)
        self.q = getattr(solution_instance, 'q', None)
        self.epsilon0 = getattr(solution_instance, 'epsilon0', None)
        self.g = getattr(solution_instance, 'g', None)
        self.r = getattr(solution_instance, 'r', None)
        self.theta = getattr(solution_instance, 'theta', None)

        # Sympy symbols for free variables
        self._p_sym = sp.Symbol('p', positive=True)
        self._q_sym = sp.Symbol('q', positive=True)
        self._eps0_sym = sp.Symbol('epsilon0', positive=True)
        self._g_sym = sp.Symbol('g', positive=True)
        self._r_sym = sp.Symbol('r', positive=True)
        self._theta_sym = sp.Symbol('theta', real=True)

        self._pi = sp.pi

    def _sympify(self, value):
        """Safely convert a value to a Sympy expression, returning None on failure."""
        if value is None:
            return None
        try:
            if isinstance(value, sp.Expr):
                return value
            return sp.sympify(value)
        except Exception:
            return None

    def _numerical_check(self, expected_expr, actual_expr, atol=1e-2, rel_tol=1e-2):
        """Fallback numerical check using safe defaults."""
        const = {
            self._p_sym: self.p if self.p is not None else 1.0,
            self._q_sym: self.q if self.q is not None else 1.0,
            self._eps0_sym: self.epsilon0 if self.epsilon0 is not None else 8.85e-12,
            self._g_sym: self.g if self.g is not None else 9.81,
            self._r_sym: self.r if self.r is not None else 0.1,
            self._theta_sym: self.theta if self.theta is not None else 0.5,
        }
        try:
            exp_val = float(expected_expr.subs(const).evalf())
            act_val = float(actual_expr.subs(const).evalf())
            return math.isclose(exp_val, act_val, rel_tol=rel_tol, abs_tol=atol)
        except Exception:
            return False

    # ---------- Interface checks ----------
    def test_interface_E_r_exists(self):
        E_r = self.results.get('E_r', None)
        if E_r is None:
            return False, "E_r key missing or None"
        return True, "E_r exists"

    def test_interface_E_theta_exists(self):
        E_th = self.results.get('E_theta', None)
        if E_th is None:
            return False, "E_theta key missing or None"
        return True, "E_theta exists"

    def test_interface_F_r_exists(self):
        F_r = self.results.get('F_r', None)
        if F_r is None:
            return False, "F_r key missing or None"
        return True, "F_r exists"

    def test_interface_F_theta_exists(self):
        F_th = self.results.get('F_theta', None)
        if F_th is None:
            return False, "F_theta key missing or None"
        return True, "F_theta exists"

    def test_interface_m_exists(self):
        m = self.results.get('m', None)
        if m is None:
            return False, "m key missing or None"
        return True, "m exists"

    # ---------- Invariant checks (theorem-based) ----------
    def test_invariant_E_field_from_gradient(self):
        """Verify that E = -∇V_dip."""
        E_r_expr = self._sympify(self.results.get('E_r', None))
        E_theta_expr = self._sympify(self.results.get('E_theta', None))
        if E_r_expr is None or E_theta_expr is None:
            return False, "E_r or E_theta not available for gradient check"

        # Expected V and gradient
        V = (1/(4*self._pi*self._eps0_sym)) * (self._p_sym * sp.cos(self._theta_sym) / self._r_sym**2)
        grad_V_r = sp.diff(V, self._r_sym)
        grad_V_theta = (1/self._r_sym) * sp.diff(V, self._theta_sym)
        expected_E_r = -grad_V_r
        expected_E_theta = -grad_V_theta

        diff_r = sp.simplify(E_r_expr - expected_E_r)
        diff_theta = sp.simplify(E_theta_expr - expected_E_theta)

        if diff_r == 0 and diff_theta == 0:
            return True, "E field matches gradient symbolically"
        if self._numerical_check(expected_E_r, E_r_expr) and self._numerical_check(expected_E_theta, E_theta_expr):
            return True, "E field matches gradient numerically (rel_tol=1e-2)"
        return False, f"E field does not match gradient. Component differences: r={diff_r}, θ={diff_theta}"

    def test_invariant_force_ratio(self):
        """Check that F_r/(2cosθ) = F_θ/sinθ (common factor = m g)."""
        F_r_expr = self._sympify(self.results.get('F_r', None))
        F_theta_expr = self._sympify(self.results.get('F_theta', None))
        if F_r_expr is None or F_theta_expr is None:
            return False, "F_r or F_theta not available for ratio check"

        ratio1 = sp.simplify(F_r_expr / (2*sp.cos(self._theta_sym)))
        ratio2 = sp.simplify(F_theta_expr / sp.sin(self._theta_sym))
        diff = sp.simplify(ratio1 - ratio2)
        if diff == 0:
            return True, "Force ratios are symbolically equal"

        # Numerical fallback (avoid division by zero)
        safe_theta = 0.5
        subs = {self._theta_sym: safe_theta}
        const = {self._p_sym: 1.0, self._q_sym: 1.0, self._eps0_sym: 1.0, self._g_sym: 1.0, self._r_sym: 1.0}
        try:
            r1_val = float(ratio1.subs({**const, self._theta_sym: safe_theta}).evalf())
            r2_val = float(ratio2.subs({**const, self._theta_sym: safe_theta}).evalf())
            if math.isclose(r1_val, r2_val, rel_tol=1e-2):
                return True, "Force ratios numerically equal at θ=0.5 rad"
        except Exception:
            pass
        return False, f"Force ratios differ. Ratio1={ratio1}, Ratio2={ratio2}"

    # ---------- Modeling checks ----------
    def test_modeling_E_field_components(self):
        """Check that E_r and E_θ have correct functional forms."""
        E_r_expr = self._sympify(self.results.get('E_r', None))
        E_theta_expr = self._sympify(self.results.get('E_theta', None))
        if E_r_expr is None or E_theta_expr is None:
            return False, "E_r or E_theta not available"

        expected_E_r = self._p_sym * sp.cos(self._theta_sym) / (2*self._pi*self._eps0_sym * self._r_sym**3)
        expected_E_theta = self._p_sym * sp.sin(self._theta_sym) / (4*self._pi*self._eps0_sym * self._r_sym**3)

        diff_r = sp.simplify(E_r_expr - expected_E_r)
        diff_th = sp.simplify(E_theta_expr - expected_E_theta)
        if diff_r == 0 and diff_th == 0:
            return True, "E field components have symbolic correct forms"
        if self._numerical_check(expected_E_r, E_r_expr) and self._numerical_check(expected_E_theta, E_theta_expr):
            return True, "E field components numerically match expected forms"
        return False, f"E field components deviation: r={diff_r}, θ={diff_th}"

    def test_modeling_F_components(self):
        """Check that F_r = q*E_r, F_θ = q*E_θ."""
        E_r_expr = self._sympify(self.results.get('E_r', None))
        E_theta_expr = self._sympify(self.results.get('E_theta', None))
        F_r_expr = self._sympify(self.results.get('F_r', None))
        F_theta_expr = self._sympify(self.results.get('F_theta', None))
        if any(x is None for x in [E_r_expr, E_theta_expr, F_r_expr, F_theta_expr]):
            return False, "One or more field/force components missing"

        expected_F_r = self._q_sym * E_r_expr
        expected_F_theta = self._q_sym * E_theta_expr
        diff_r = sp.simplify(F_r_expr - expected_F_r)
        diff_th = sp.simplify(F_theta_expr - expected_F_theta)
        if diff_r == 0 and diff_th == 0:
            return True, "F = qE holds symbolically"
        if self._numerical_check(expected_F_r, F_r_expr) and self._numerical_check(expected_F_theta, F_theta_expr):
            return True, "F = qE holds numerically"
        return False, f"F = qE violated: r diff={diff_r}, θ diff={diff_th}"

    def test_modeling_mass_consistency(self):
        """Check mass expression and consistency with force components."""
        m_expr = self._sympify(self.results.get('m', None))
        F_r_expr = self._sympify(self.results.get('F_r', None))
        F_theta_expr = self._sympify(self.results.get('F_theta', None))
        if any(x is None for x in [m_expr, F_r_expr, F_theta_expr]):
            return False, "m, F_r, or F_theta missing"

        expected_m = self._q_sym * self._p_sym / (4*self._pi*self._eps0_sym * self._g_sym * self._r_sym**3)
        diff_m = sp.simplify(m_expr - expected_m)
        if diff_m != 0:
            return False, f"Mass expression differs from expected: difference {diff_m}"

        # Check that force components imply same mass
        m_from_Fr = sp.simplify(F_r_expr / (2*self._g_sym*sp.cos(self._theta_sym)))
        m_from_Ftheta = sp.simplify(F_theta_expr / (self._g_sym*sp.sin(self._theta_sym)))
        diff1 = sp.simplify(m_from_Fr - expected_m)
        diff2 = sp.simplify(m_from_Ftheta - expected_m)
        if diff1 == 0 and diff2 == 0:
            return True, "Mass matches expected and consistent with force components"
        return False, f"Mass matches expected but force ratio inconsistency: Fr/2gcosθ differs by {diff1}, Fθ/gsinθ differs by {diff2}"

    # ---------- Monotonicity checks ----------
    def test_monotonicity_E_r_with_theta(self):
        """E_r ∝ cosθ, decreasing with θ in (0, π). Baseline at θ=0.1, perturb to θ=0.6."""
        E_r_expr = self._sympify(self.results.get('E_r', None))
        if E_r_expr is None:
            return False, "E_r not available"
        const = {self._p_sym: 1.0, self._eps0_sym: 8.85e-12, self._r_sym: 1.0}
        try:
            baseline = float(E_r_expr.subs({**const, self._theta_sym: 0.1}).evalf())
            perturbed = float(E_r_expr.subs({**const, self._theta_sym: 0.6}).evalf())
            if baseline > perturbed:
                return True, f"E_r decreases with θ: baseline {baseline} > perturbed {perturbed}"
            return False, f"E_r did not decrease: baseline {baseline} <= perturbed {perturbed}"
        except Exception as e:
            return False, f"Monotonicity test failed: {e}"

    def test_monotonicity_E_theta_with_theta(self):
        """E_θ ∝ sinθ, increasing with θ in (0, π/2). Baseline at θ=0.2, perturb to θ=0.8."""
        E_th_expr = self._sympify(self.results.get('E_theta', None))
        if E_th_expr is None:
            return False, "E_theta not available"
        const = {self._p_sym: 1.0, self._eps0_sym: 8.85e-12, self._r_sym: 1.0}
        try:
            baseline = float(E_th_expr.subs({**const, self._theta_sym: 0.2}).evalf())
            perturbed = float(E_th_expr.subs({**const, self._theta_sym: 0.8}).evalf())
            if baseline < perturbed:
                return True, f"E_θ increases with θ: baseline {baseline} < perturbed {perturbed}"
            return False, f"E_θ did not increase: baseline {baseline} >= perturbed {perturbed}"
        except Exception as e:
            return False, f"Monotonicity test failed: {e}"

    def test_monotonicity_mass_with_distance(self):
        """Mass m ∝ 1/r³, so increasing r decreases m. Baseline r=0.5, perturb to r=1.0."""
        m_expr = self._sympify(self.results.get('m', None))
        if m_expr is None:
            return False, "m not available"
        const = {self._p_sym: 1.0, self._q_sym: 1.0, self._eps0_sym: 8.85e-12, self._g_sym: 9.81, self._theta_sym: 0.5}
        try:
            baseline = float(m_expr.subs({**const, self._r_sym: 0.5}).evalf())
            perturbed = float(m_expr.subs({**const, self._r_sym: 1.0}).evalf())
            if baseline > perturbed:
                return True, f"Mass decreases with r: baseline {baseline} > perturbed {perturbed}"
            return False, f"Mass did not decrease: baseline {baseline} <= perturbed {perturbed}"
        except Exception as e:
            return False, f"Monotonicity test failed: {e}"

    # ---------- Weak answer sanity (only one allowed) ----------
    def test_answer_sanity_simple(self):
        """
        Basic sanity: m expression should be positive for positive constants.
        No hardcoded target values, no calculation of expected values.
        """
        m_expr = self._sympify(self.results.get('m', None))
        if m_expr is None:
            return False, "m not available"
        try:
            # Evaluate at arbitrary positive numeric values (no constants from problem)
            val = float(m_expr.subs({
                self._p_sym: 1.0, self._q_sym: 1.0, self._eps0_sym: 1.0,
                self._g_sym: 1.0, self._r_sym: 1.0, self._theta_sym: 0.5
            }).evalf())
            if val > 0:
                return True, f"Mass expression yields positive value {val} for positive constants"
            return False, f"Mass expression yields non-positive value {val}"
        except Exception as e:
            return False, f"Cannot evaluate mass sign: {e}"

    # ---------- Main runner ----------
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