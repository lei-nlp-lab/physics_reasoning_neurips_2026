import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for a problem with two parts:
    (a) constraint force magnitude for a particle on a smooth vertical paraboloid,
    (b) Lagrangian in rotating spherical coordinates, velocity‑dependent potential U,
        and centrifugal/Coriolis force components.
    """

    def __init__(self, solution_instance):
        # Extract givens (all with None defaults)
        self.m = getattr(solution_instance, 'm', None)
        self.g = getattr(solution_instance, 'g', None)
        self.a = getattr(solution_instance, 'a', None)
        self.omega = getattr(solution_instance, 'omega', None)
        self.r = getattr(solution_instance, 'r', None)
        self.theta = getattr(solution_instance, 'theta', None)
        self.phi = getattr(solution_instance, 'phi', None)
        self.r_dot = getattr(solution_instance, 'r_dot', None)
        self.theta_dot = getattr(solution_instance, 'theta_dot', None)
        self.phi_dot = getattr(solution_instance, 'phi_dot', None)

        # Extract results from solution_instance.results (dict)
        results = getattr(solution_instance, 'results', {})
        self.N_mag = results.get('N_mag', None)
        self.L_rot = results.get('L_rot', None)
        self.U = results.get('U', None)
        self.F_r = results.get('F_r', None)
        self.F_phi = results.get('F_phi', None)

    # ----------------------------------------------------------------------
    # Helper: safely parse a string or sympy expression to a sympy expression
    # ----------------------------------------------------------------------
    def _parse_expr(self, expr_input):
        """Convert a string (or sympy expression) to a sympy expression.
        Returns None on failure."""
        if expr_input is None:
            return None
        if isinstance(expr_input, sp.Expr):
            return expr_input
        if isinstance(expr_input, str):
            try:
                # Define known symbols for parsing
                m, g, a, omega, r, theta, phi, r_dot, theta_dot, phi_dot, V = \
                    sp.symbols('m g a omega r theta phi r_dot theta_dot phi_dot V')
                return sp.sympify(expr_input, locals={
                    'm': m, 'g': g, 'a': a, 'omega': omega,
                    'r': r, 'theta': theta, 'phi': phi,
                    'r_dot': r_dot, 'theta_dot': theta_dot, 'phi_dot': phi_dot,
                    'sin': sp.sin, 'cos': sp.cos,
                    'V': sp.Function('V')(r)
                })
            except:
                return None
        return None

    # ----------------------------------------------------------------------
    # Helper: evaluate a sympy expression with given numeric substitutions
    # ----------------------------------------------------------------------
    def _evaluate_expr(self, expr, subs_dict):
        """Substitute numeric values into a sympy expression and return float result.
        Returns None if substitution fails."""
        try:
            val = expr.subs(subs_dict).evalf()
            if val.is_number:
                return float(val)
            else:
                return None
        except:
            return None

    # ----------------------------------------------------------------------
    # Test: interface – required keys are present and not None
    # ----------------------------------------------------------------------
    def test_interface_has_results(self):
        """Check all required result keys exist."""
        required = ['N_mag', 'L_rot', 'U', 'F_r', 'F_phi']
        for key in required:
            if getattr(self, key, None) is None:
                return (False, f"SKIP: Missing required result: {key}")
        return (True, "All required results present")

    # ----------------------------------------------------------------------
    # Test: invariant – constraint force exponent (part a)
    # Theorem: N_mag ∝ (1+4r^2/a^2)^{-3/2}
    # ----------------------------------------------------------------------
    def test_invariant_constraint_force_exponent(self):
        """Check that the logarithmic derivative of N_mag w.r.t. r gives -3/2 * (8r/a^2)/(1+4r^2/a^2)."""
        if self.N_mag is None:
            return (False, "SKIP: Missing N_mag")
        if self.a is None or self.r is None:
            return (False, "SKIP: Missing a or r for evaluation")
        expr = self._parse_expr(self.N_mag)
        if expr is None:
            return (False, "Failed to parse N_mag")
        r_sym = sp.symbols('r')
        a_sym = sp.symbols('a')
        try:
            # Substitute the symbols into the expression
            expr_sub = expr.subs({sp.Symbol('r'): r_sym, sp.Symbol('a'): a_sym})
            # Compute d(log|f|)/dr
            log_expr = sp.log(sp.Abs(expr_sub))
            dlog_dr = sp.diff(log_expr, r_sym)
            # Expected derivative from the theorem
            expected_dlog = -sp.Rational(3,2) * (8*r_sym / a_sym**2) / (1 + 4*r_sym**2 / a_sym**2)
            diff = sp.simplify(dlog_dr - expected_dlog)
            if diff == 0:
                return (True, "Exponent -3/2 satisfied symbolically")
            # Numeric fallback
            for test_r, test_a in [(1.0,2.0), (2.0,4.0), (0.5,1.0)]:
                dlog_num = float(dlog_dr.subs({r_sym:test_r, a_sym:test_a}))
                exp_num = float(expected_dlog.subs({r_sym:test_r, a_sym:test_a}))
                if not math.isclose(dlog_num, exp_num, rel_tol=1e-2):
                    return (False, f"Exponent mismatch at r={test_r}, a={test_a}: {dlog_num} vs {exp_num}")
            return (True, "Exponent -3/2 confirmed numerically")
        except Exception as e:
            return (False, f"Error during exponent check: {str(e)}")

    # ----------------------------------------------------------------------
    # Test: invariant – Lagrangian reduces to fixed-frame when ω=0 (part b)
    # Theorem: L_rot(ω=0) = L_fixed = 0.5*m*(r_dot^2 + r^2*θ_dot^2 + r^2*sin^2θ*φ_dot^2) - V(r)
    # ----------------------------------------------------------------------
    def test_invariant_omega_zero_reduces_to_fixed_frame(self):
        """Check that setting ω=0 in L_rot yields the fixed-frame Lagrangian."""
        if self.L_rot is None:
            return (False, "SKIP: Missing L_rot")
        expr = self._parse_expr(self.L_rot)
        if expr is None:
            return (False, "SKIP: Failed to parse L_rot")
        m_sym, r_sym, theta_sym, r_dot_sym, theta_dot_sym, phi_dot_sym, omega_sym, V_sym = \
            sp.symbols('m r theta r_dot theta_dot phi_dot omega V')
        # Build fixed-frame Lagrangian (with general V(r))
        L_fixed = sp.Rational(1,2)*m_sym*(r_dot_sym**2 + r_sym**2*theta_dot_sym**2 +
                         r_sym**2*sp.sin(theta_sym)**2*phi_dot_sym**2) - sp.Function('V')(r_sym)
        try:
            sub_map = {
                sp.Symbol('m'): m_sym,
                sp.Symbol('r'): r_sym,
                sp.Symbol('theta'): theta_sym,
                sp.Symbol('r_dot'): r_dot_sym,
                sp.Symbol('theta_dot'): theta_dot_sym,
                sp.Symbol('phi_dot'): phi_dot_sym,
                sp.Symbol('omega'): omega_sym,
                sp.Function('V'): sp.Function('V')(r_sym)
            }
            L_rot_mapped = expr.subs(sub_map)
            L_rot_zero = L_rot_mapped.subs({omega_sym: 0})
            diff = sp.simplify(L_rot_zero - L_fixed)
            if diff == 0:
                return (True, "L_rot(ω=0) matches fixed-frame Lagrangian")
            # Numeric check
            test_vals = {m_sym:1.0, r_sym:0.5, theta_sym:0.8,
                         r_dot_sym:1.2, theta_dot_sym:-0.3, phi_dot_sym:2.1,
                         omega_sym:0.0}
            # Replace V(r) with a numeric constant for evaluation
            V_val = 3.0
            test_vals[sp.Function('V')(r_sym)] = V_val
            L_rot_zero_num = float(L_rot_zero.subs(test_vals).evalf())
            L_fixed_num = float(L_fixed.subs(test_vals).evalf())
            if math.isclose(L_rot_zero_num, L_fixed_num, rel_tol=1e-2):
                return (True, "L_rot(ω=0) matches fixed-frame numerically")
            else:
                return (False, f"Numeric mismatch: L_rot(ω=0)={L_rot_zero_num}, L_fixed={L_fixed_num}")
        except Exception as e:
            return (False, f"Error during ω→0 check: {str(e)}")

    # ----------------------------------------------------------------------
    # Test: modeling – Lagrangian L_rot has correct form
    # L_rot = 0.5*m*(r_dot^2 + r^2*theta_dot^2 + r^2*sin^2(theta)*(phi_dot+omega)^2) - V(r)
    # ----------------------------------------------------------------------
    def test_modeling_lagrangian_rotating(self):
        """Check L_rot expression symbolically matches the expected form."""
        if self.L_rot is None:
            return (False, "SKIP: L_rot not provided")
        expr = self._parse_expr(self.L_rot)
        if expr is None:
            return (False, "Failed to parse L_rot")
        m_sym, r_sym, theta_sym, r_dot_sym, theta_dot_sym, phi_dot_sym, omega_sym, V_sym = \
            sp.symbols('m r theta r_dot theta_dot phi_dot omega V')
        expected = sp.Rational(1,2)*m_sym*(r_dot_sym**2 + r_sym**2*theta_dot_sym**2 +
                             r_sym**2*sp.sin(theta_sym)**2*(phi_dot_sym+omega_sym)**2) - sp.Function('V')(r_sym)
        try:
            sub_map = {
                sp.Symbol('m'): m_sym,
                sp.Symbol('r'): r_sym,
                sp.Symbol('theta'): theta_sym,
                sp.Symbol('r_dot'): r_dot_sym,
                sp.Symbol('theta_dot'): theta_dot_sym,
                sp.Symbol('phi_dot'): phi_dot_sym,
                sp.Symbol('omega'): omega_sym,
                sp.Function('V'): sp.Function('V')(r_sym)
            }
            expr_mapped = expr.subs(sub_map)
            diff = sp.simplify(expr_mapped - expected)
            if diff == 0:
                return (True, "L_rot matches expected symbolic form")
            # Numeric test
            test_vals = {m_sym:2.0, r_sym:0.5, theta_sym:0.8,
                         r_dot_sym:1.2, theta_dot_sym:-0.3, phi_dot_sym:2.1,
                         omega_sym:0.4}
            V_val = 3.0
            test_vals[sp.Function('V')(r_sym)] = V_val
            expr_num = float(expr_mapped.subs(test_vals).evalf())
            exp_num = float(expected.subs(test_vals).evalf())
            if math.isclose(expr_num, exp_num, rel_tol=1e-2):
                return (True, "L_rot matches expected form (numerical check)")
            else:
                return (False, f"Numeric mismatch: {expr_num} vs {exp_num}")
        except Exception as e:
            return (False, f"Error during L_rot check: {str(e)}")

    # ----------------------------------------------------------------------
    # Test: modeling – velocity-dependent potential U derived from L_rot
    # U = -0.5*m*r^2*sin^2(theta)*(2*omega*phi_dot + omega^2)
    # ----------------------------------------------------------------------
    def test_modeling_velocity_dependent_potential(self):
        """Check that U expression matches the difference L_fixed - L_rot."""
        if self.U is None or self.L_rot is None:
            return (False, "SKIP: U or L_rot not provided")
        U_expr = self._parse_expr(self.U)
        L_rot_expr = self._parse_expr(self.L_rot)
        if U_expr is None or L_rot_expr is None:
            return (False, "Failed to parse expressions")
        m_sym, r_sym, theta_sym, r_dot_sym, theta_dot_sym, phi_dot_sym, omega_sym, V_sym = \
            sp.symbols('m r theta r_dot theta_dot phi_dot omega V')
        L_fixed = sp.Rational(1,2)*m_sym*(r_dot_sym**2 + r_sym**2*theta_dot_sym**2 +
                          r_sym**2*sp.sin(theta_sym)**2*phi_dot_sym**2) - sp.Function('V')(r_sym)
        sub_map = {
            sp.Symbol('m'): m_sym,
            sp.Symbol('r'): r_sym,
            sp.Symbol('theta'): theta_sym,
            sp.Symbol('r_dot'): r_dot_sym,
            sp.Symbol('theta_dot'): theta_dot_sym,
            sp.Symbol('phi_dot'): phi_dot_sym,
            sp.Symbol('omega'): omega_sym,
            sp.Function('V'): sp.Function('V')(r_sym)
        }
        try:
            L_rot_mapped = L_rot_expr.subs(sub_map)
            expected_U = sp.simplify(L_fixed - L_rot_mapped)
            U_expr_mapped = U_expr.subs(sub_map)
            diff = sp.simplify(expected_U - U_expr_mapped)
            if diff == 0:
                return (True, "U matches expected expression")
            # Numeric test
            test_vals = {m_sym:1.0, r_sym:0.3, theta_sym:1.2,
                         r_dot_sym:0.5, theta_dot_sym:-0.4, phi_dot_sym:1.5,
                         omega_sym:0.6}
            V_val = 2.0
            test_vals[sp.Function('V')(r_sym)] = V_val
            exp_U_num = float(expected_U.subs(test_vals).evalf())
            given_U_num = float(U_expr_mapped.subs(test_vals).evalf())
            if math.isclose(exp_U_num, given_U_num, rel_tol=1e-2):
                return (True, "U matches expected form (numerical check)")
            else:
                return (False, f"Numeric mismatch: expected U={exp_U_num}, given U={given_U_num}")
        except Exception as e:
            return (False, f"Error during U check: {str(e)}")

    # ----------------------------------------------------------------------
    # Test: modeling – centrifugal/Coriolis force components from U
    # F_r = -∂U/∂r  ;  F_phi = Q_phi/(r sinθ) where Q_phi = d/dt(∂U/∂φ̇)
    # ----------------------------------------------------------------------
    def test_modeling_force_components(self):
        """Check that F_r and F_phi are correctly derived from U."""
        if self.U is None or self.F_r is None or self.F_phi is None:
            return (False, "SKIP: Missing U, F_r, or F_phi")
        U_expr = self._parse_expr(self.U)
        F_r_expr = self._parse_expr(self.F_r)
        F_phi_expr = self._parse_expr(self.F_phi)
        if U_expr is None or F_r_expr is None or F_phi_expr is None:
            return (False, "Failed to parse expressions")
        # Define symbols
        m, r, theta, r_dot, theta_dot, phi_dot, omega = \
            sp.symbols('m r theta r_dot theta_dot phi_dot omega')
        # U sympy expression
        U_sym = -sp.Rational(1,2)*m*r**2*sp.sin(theta)**2*(2*omega*phi_dot + omega**2)
        expected_F_r = -sp.diff(U_sym, r)
        # Q_phi = d/dt(∂U/∂φ̇)
        expected_Q_phi = -2*m*omega*r*r_dot*sp.sin(theta)**2 - 2*m*omega*r**2*sp.sin(theta)*sp.cos(theta)*theta_dot
        expected_F_phi = expected_Q_phi / (r * sp.sin(theta))
        sub_map = {
            sp.Symbol('m'): m,
            sp.Symbol('r'): r,
            sp.Symbol('theta'): theta,
            sp.Symbol('r_dot'): r_dot,
            sp.Symbol('theta_dot'): theta_dot,
            sp.Symbol('phi_dot'): phi_dot,
            sp.Symbol('omega'): omega
        }
        try:
            F_r_given_mapped = F_r_expr.subs(sub_map)
            F_phi_given_mapped = F_phi_expr.subs(sub_map)
            diff_r = sp.simplify(F_r_given_mapped - expected_F_r)
            diff_phi = sp.simplify(F_phi_given_mapped - expected_F_phi)
            if diff_r == 0 and diff_phi == 0:
                return (True, "F_r and F_phi correctly derived from U")
            # Numeric test
            test_vals = {m:2.0, r:1.2, theta:0.7, r_dot:0.5, theta_dot:-0.2, phi_dot:1.8, omega:0.3}
            fr_num = float(F_r_given_mapped.subs(test_vals).evalf())
            fr_exp = float(expected_F_r.subs(test_vals).evalf())
            fphi_num = float(F_phi_given_mapped.subs(test_vals).evalf())
            fphi_exp = float(expected_F_phi.subs(test_vals).evalf())
            if (math.isclose(fr_num, fr_exp, rel_tol=1e-2) and
                math.isclose(fphi_num, fphi_exp, rel_tol=1e-2)):
                return (True, "F_r and F_phi correct numerically")
            else:
                msg = f"F_r: {fr_num} vs {fr_exp}; F_phi: {fphi_num} vs {fphi_exp}"
                return (False, f"Force components mismatch: {msg}")
        except Exception as e:
            return (False, f"Error during force check: {str(e)}")

    # ----------------------------------------------------------------------
    # Test: monotonicity – centrifugal component grows with omega
    # ----------------------------------------------------------------------
    def test_monotonicity_centrifugal_with_omega(self):
        """Perturb ω by +10% and verify that F_r increases (monotonic increase)."""
        if self.F_r is None:
            return (False, "SKIP: Missing F_r")
        fr_expr = self._parse_expr(self.F_r)
        if fr_expr is None:
            return (False, "Failed to parse F_r")
        # Use symbolic variables and fixed numeric values (no self.* defaults)
        m_sym, r_sym, theta_sym, phi_dot_sym, omega_sym = sp.symbols('m r theta phi_dot omega')
        sub_forward = {
            sp.Symbol('m'): m_sym,
            sp.Symbol('r'): r_sym,
            sp.Symbol('theta'): theta_sym,
            sp.Symbol('phi_dot'): phi_dot_sym,
            sp.Symbol('omega'): omega_sym
        }
        try:
            fr_sym = fr_expr.subs(sub_forward)
            # Fixed test values (independent of self.*)
            test_vals = {
                m_sym: 1.0,
                r_sym: 1.0,
                theta_sym: 0.5,
                phi_dot_sym: 2.0,
                omega_sym: 0.5  # baseline
            }
            base_val = float(fr_sym.subs(test_vals).evalf())
            # Perturb omega by +10%
            perturbed_vals = test_vals.copy()
            perturbed_vals[omega_sym] = test_vals[omega_sym] * 1.1
            perturbed_val = float(fr_sym.subs(perturbed_vals).evalf())
            if perturbed_val > base_val:
                return (True, f"F_r increases with ω: {base_val} -> {perturbed_val}")
            else:
                return (False, f"Monotonicity failed: F_r did not increase (baseline {base_val}, perturbed {perturbed_val})")
        except Exception as e:
            return (False, f"Monotonicity check error: {str(e)}")

    # ----------------------------------------------------------------------
    # Test: answer sanity (weak) – N_mag positive and finite (weak sanity)
    # ----------------------------------------------------------------------
    def test_answer_sanity_N_mag_sign_and_finite(self):
        """Check that N_mag expression yields positive finite values (weak check)."""
        if self.N_mag is None:
            return (False, "SKIP: Missing N_mag")
        expr = self._parse_expr(self.N_mag)
        if expr is None:
            return (False, "Failed to parse N_mag")
        # Use typical values if not provided
        m_val = self.m if self.m is not None else 1.0
        g_val = self.g if self.g is not None else 9.81
        a_val = self.a if self.a is not None else 2.0
        r_val = self.r if self.r is not None else 1.0
        try:
            m_sym, g_sym, a_sym, r_sym = sp.symbols('m g a r')
            expr_sub = expr.subs({
                sp.Symbol('m'): m_sym,
                sp.Symbol('g'): g_sym,
                sp.Symbol('a'): a_sym,
                sp.Symbol('r'): r_sym
            })
            val = float(expr_sub.subs({
                m_sym: m_val, g_sym: g_val, a_sym: a_val, r_sym: r_val
            }).evalf())
            if math.isfinite(val) and val > 0:
                return (True, f"N_mag positive and finite: {val}")
            else:
                return (False, f"N_mag not positive/finite: {val}")
        except Exception as e:
            return (False, f"Evaluation error: {str(e)}")

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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)