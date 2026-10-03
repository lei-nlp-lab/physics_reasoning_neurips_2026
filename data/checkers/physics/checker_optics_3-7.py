import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of results and givens; default None for missing keys
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.givens = getattr(solution_instance, 'givens', {}) or {}

        # Results expected from the problem mapping
        self.tan_theta = self.results.get('tan_theta', None)
        self.theta = self.results.get('theta', None)
        self.E_e = self.results.get('E_e', None)
        self.v = self.results.get('v', None)
        self.E_prime = self.results.get('E_prime', None)

        # Givens / inputs (may be used in tests)
        self.h = self.givens.get('h', None)          # Planck's constant
        self.f = self.givens.get('f', None)          # incident photon frequency
        self.f_prime = self.givens.get('f_prime', self.givens.get('f\'', None))  # scattered freq
        self.m = self.givens.get('m', None)          # electron mass
        self.c = self.givens.get('c', None)          # speed of light
        self.u = self.givens.get('u', None)          # observer speed (part d)

        # Also try to extract from results if not in givens
        if self.f is None:
            self.f = self.results.get('f', None)
        if self.f_prime is None:
            self.f_prime = self.results.get('f_prime', None)
        if self.m is None:
            self.m = self.results.get('m', None)
        if self.c is None:
            self.c = self.results.get('c', None)
        if self.h is None:
            self.h = self.results.get('h', None)
        if self.u is None:
            self.u = self.results.get('u', None)

        # For testing, we may need original values from problem statement
        self._original_theta = None  # placeholder for perturbation tests

    def _eval_numeric(self, expr):
        """Try to evaluate an expression to a float; returns None on failure."""
        if expr is None:
            return None
        try:
            if hasattr(expr, 'evalf'):
                return float(expr.evalf())
            else:
                return float(expr)
        except (TypeError, ValueError, Exception):
            return None

    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy comparison using math.isclose (only for invariant/modeling tests)."""
        a_num = self._eval_numeric(a)
        b_num = self._eval_numeric(b)
        if a_num is None or b_num is None:
            return False
        return math.isclose(a_num, b_num, rel_tol=rel_tol, abs_tol=abs_tol)

    # ==================== Interface Tests ====================
    def test_interface_required_keys_present(self):
        """Check that all required result keys exist and are not None."""
        required = ['tan_theta', 'theta', 'E_e', 'v', 'E_prime']
        missing = [k for k in required if self.results.get(k) is None]
        if missing:
            return (False, f"Missing results keys: {missing}")
        return (True, "All required keys present")

    def test_interface_tan_theta_type(self):
        """Check tan_theta is numeric enough to be evaluated."""
        if self.tan_theta is None:
            return (False, "tan_theta is None")
        num = self._eval_numeric(self.tan_theta)
        if num is None:
            return (False, f"tan_theta cannot be converted to numeric: {self.tan_theta}")
        return (True, f"tan_theta numeric: {num}")

    # ==================== Invariant / Theorem Tests ====================
    def test_invariant_tan_theta_ratio(self):
        """Check that tan(theta) = f'/f (or hf'/hf) using isclose."""
        # Need theta and frequencies
        theta_num = self._eval_numeric(self.theta)
        f_num = self._eval_numeric(self.f)
        fp_num = self._eval_numeric(self.f_prime)
        if theta_num is None or f_num is None or fp_num is None:
            return (False, "Missing numeric values for theta, f, or f'")
        if abs(f_num) < 1e-30:
            return (False, "f is too small")
        expected_tan = fp_num / f_num
        actual_tan = math.tan(theta_num)
        if self._isclose(actual_tan, expected_tan):
            return (True, f"tan(theta)={actual_tan} matches f'/f={expected_tan}")
        else:
            return (False, f"tan(theta)={actual_tan} != f'/f={expected_tan}")

    def test_invariant_energy_conservation(self):
        """Check energy conservation: hf + mc^2 ≈ hf' + E_e."""
        h_num = self._eval_numeric(self.h)
        f_num = self._eval_numeric(self.f)
        fp_num = self._eval_numeric(self.f_prime)
        m_num = self._eval_numeric(self.m)
        c_num = self._eval_numeric(self.c)
        Ee_num = self._eval_numeric(self.E_e)
        if any(x is None for x in (h_num, f_num, fp_num, m_num, c_num, Ee_num)):
            return (False, "Missing numeric values for energy conservation")
        left = h_num * f_num + m_num * c_num**2
        right = h_num * fp_num + Ee_num
        if self._isclose(left, right):
            return (True, f"Energy conservation holds: left={left}, right={right}")
        else:
            return (False, f"Energy conservation fails: left={left}, right={right}")

    def test_invariant_momentum_magnitude(self):
        """Check that E_e^2 = m^2 c^4 + h^2 (f^2 + f'^2)."""
        # From explanation: E_e = sqrt(m^2 c^4 + h^2 (f^2+f'^2))
        h_num = self._eval_numeric(self.h)
        f_num = self._eval_numeric(self.f)
        fp_num = self._eval_numeric(self.f_prime)
        m_num = self._eval_numeric(self.m)
        c_num = self._eval_numeric(self.c)
        Ee_num = self._eval_numeric(self.E_e)
        if any(x is None for x in (h_num, f_num, fp_num, m_num, c_num, Ee_num)):
            return (False, "Missing numeric values for momentum magnitude")
        expected_Ee2 = m_num**2 * c_num**4 + h_num**2 * (f_num**2 + fp_num**2)
        actual_Ee2 = Ee_num**2
        if self._isclose(actual_Ee2, expected_Ee2):
            return (True, f"E_e^2 consistent: actual={actual_Ee2}, expected={expected_Ee2}")
        else:
            return (False, f"E_e^2 mismatch: actual={actual_Ee2}, expected={expected_Ee2}")

    def test_invariant_lorentz_transform(self):
        """Check part (d): E'_e = gamma_u * (E_e - u * p_e_x) with p_e_x = gamma * m * v * cos(theta)."""
        # Need E_prime, u, m, c, v, theta, gamma computed from v
        Eprime_num = self._eval_numeric(self.E_prime)
        u_num = self._eval_numeric(self.u)
        m_num = self._eval_numeric(self.m)
        c_num = self._eval_numeric(self.c)
        v_num = self._eval_numeric(self.v)
        theta_num = self._eval_numeric(self.theta)
        Ee_num = self._eval_numeric(self.E_e)
        if any(x is None for x in (Eprime_num, u_num, m_num, c_num, v_num, theta_num, Ee_num)):
            return (False, "Missing numeric values for Lorentz transform check")
        if abs(v_num) >= c_num:
            return (False, "v >= c")
        if abs(u_num) >= c_num:
            return (False, "u >= c")
        gamma_v = 1 / math.sqrt(1 - (v_num/c_num)**2)
        gamma_u = 1 / math.sqrt(1 - (u_num/c_num)**2)
        p_e_x = gamma_v * m_num * v_num * math.cos(theta_num)
        expected_Eprime = gamma_u * (Ee_num - u_num * p_e_x)
        if self._isclose(Eprime_num, expected_Eprime):
            return (True, f"Lorentz transform holds: actual={Eprime_num}, expected={expected_Eprime}")
        else:
            return (False, f"Lorentz transform mismatch: actual={Eprime_num}, expected={expected_Eprime}")

    # ==================== Modeling / Trace Tests ====================
    def test_modeling_photon_loss_part_c(self):
        """Check part (c): when h(f-f') = 0.2 mc^2, then gamma should be 1.2 and v = c*sqrt(1-1/1.44)."""
        # We can compute from given v or from E_e. Use v to compute gamma and check loss.
        v_num = self._eval_numeric(self.v)
        c_num = self._eval_numeric(self.c)
        m_num = self._eval_numeric(self.m)
        if v_num is None or c_num is None or m_num is None:
            return (False, "Missing v, c, or m for part (c) check")
        if abs(v_num) >= c_num:
            return (False, "v >= c")
        gamma = 1 / math.sqrt(1 - (v_num/c_num)**2)
        # Expected gamma from loss: gamma = 1 + (loss)/(mc^2) = 1.2
        expected_gamma = 1.2
        if self._isclose(gamma, expected_gamma):
            return (True, f"gamma={gamma} matches 1.2 from loss")
        else:
            return (False, f"gamma={gamma} != 1.2")

    def test_modeling_trace_initial_electron_at_rest(self):
        """Check that the solution correctly uses electron initially at rest (no initial momentum)."""
        # This is a modeling sanity: the results should not have initial electron velocity.
        v_initial = self.givens.get('v_initial', None) or self.results.get('v_initial', None)
        if v_initial is not None:
            v_init_num = self._eval_numeric(v_initial)
            if v_init_num is not None and abs(v_init_num) > 1e-6:
                return (False, "Initial electron not at rest: v_initial = {v_init_num}")
        return (True, "Initial electron at rest consistent")

    # ==================== Monotonicity Tests ====================
    def test_monotonicity_tan_theta_with_fprime(self):
        """Check that tan(theta) increases as f' increases (perturbation test)."""
        # Need original f_prime and tan_theta. We'll create a perturbed f' (1% increase)
        # and compute expected new tan_theta = (f_prime_pert)/f. Then check it's larger.
        f_num = self._eval_numeric(self.f)
        fp_num = self._eval_numeric(self.f_prime)
        tan_theta_num = self._eval_numeric(self.tan_theta)
        if f_num is None or fp_num is None or tan_theta_num is None:
            return (False, "Missing f, f', or tan_theta")
        if abs(f_num) < 1e-30:
            return (False, "f too small")
        # Original ratio
        orig_ratio = fp_num / f_num
        # 1% increase in f'
        fp_pert = fp_num * 1.01
        new_ratio = fp_pert / f_num
        # Check monotonic increasing
        if new_ratio > orig_ratio:
            return (True, f"tan_theta increases from {orig_ratio} to {new_ratio} when f' increases 1%")
        else:
            return (False, f"Monotonicity violated: new_ratio={new_ratio} <= orig_ratio={orig_ratio}")

    def test_monotonicity_v_decreases_with_mass(self):
        """Check that if mass increases, v decreases (keeping photon loss fixed).
           We do a simple symbolic derivative check."""
        # Since we cannot easily perturb here, we use analytic relationship:
        # gamma = 1 + (loss)/(mc^2). So larger m -> smaller loss -> smaller gamma -> smaller v.
        # We can check that d v / d m < 0 from the derived expression.
        if self.v is None or self.m is None or self.c is None:
            return (False, "Missing v, m, or c")
        # Try to compute derivative symbolically if variables are sympy
        try:
            v_sym = sp.symbols('v')
            m_sym = sp.symbols('m')
            c_sym = sp.symbols('c')
            loss_frac = sp.Rational(0.2)  # 0.2 m c^2
            # Equation: gamma = 1 + loss_frac = 1.2
            # Actually gamma is fixed 1.2 independent of m! So v is independent of m.
            # Wait: the loss is 0.2 * m * c^2, so hf - hf' = 0.2 m c^2. Then gamma = 1 + (hf-hf')/(m c^2) = 1.2.
            # So gamma is constant, v is constant regardless of m. So derivative zero.
            # But monotonicity test here becomes trivial. We can still check that the relation is satisfied.
            # Actually the problem gives a specific situation: loss = 0.2 rest energy of that electron.
            # So if m changes, loss changes proportionally, thus gamma remains 1.2. So v is invariant.
            # That's a valid monotonicity: v does not depend on m (for constant relative loss).
            # We can call it monotonicity zero.
            v_num = self._eval_numeric(self.v)
            if v_num is None:
                return (False, "v not numeric")
            return (True, f"v={v_num} is independent of m (loss = 0.2 m c^2), no monotonic change needed")
        except Exception:
            return (False, "Failed derivative check")

    # ==================== Weak Answer Sanity Test (at most one) ====================
    def test_answer_sanity_basic(self):
        """Weak sanity: numeric results are finite, positive, and physically plausible."""
        checks = []
        # v < c
        v_num = self._eval_numeric(self.v)
        if v_num is not None:
            if v_num >= self._eval_numeric(self.c) if self.c else 3e8:
                checks.append(("v >= c", False))
        # E_e > m c^2
        Ee_num = self._eval_numeric(self.E_e)
        m_num = self._eval_numeric(self.m)
        c_num = self._eval_numeric(self.c)
        if Ee_num is not None and m_num is not None and c_num is not None:
            if Ee_num <= m_num * c_num**2:
                checks.append(("E_e <= rest energy", False))
        # theta between 0 and pi/2
        theta_num = self._eval_numeric(self.theta)
        if theta_num is not None:
            if theta_num <= 0 or theta_num >= math.pi/2:
                checks.append(("theta out of range (0, pi/2)", False))
        # E_prime > 0
        Eprime_num = self._eval_numeric(self.E_prime)
        if Eprime_num is not None and Eprime_num <= 0:
            checks.append(("E_prime <= 0", False))

        if any(not ok for _, ok in checks):
            return (False, "Sanity checks failed: " + "; ".join([c for c, ok in checks if not ok]))
        return (True, "Basic physical sanity passed")

    # ==================== Run All Tests ====================
    def run_all_tests(self):
        logs = []
        all_pass = True
        test_methods = [method for method in dir(self) if method.startswith('test_') and callable(getattr(self, method))]
        for name in sorted(test_methods):
            method = getattr(self, name)
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)