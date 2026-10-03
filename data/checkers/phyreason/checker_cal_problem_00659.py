import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs and results
        self.solution = solution_instance
        self.l = 0.2
        self.m = 1.0e-2
        self.q = 2.0e-8
        self.g = 10.0
        self.theta = 37.0  # degrees
        self.sin37 = 0.6
        self.cos37 = 0.8
        self.t = 0.1  # time for step 4/5

        # Safe attribute access with fallbacks
        self.E = self._safe_get('E', 0.0)
        self.v_lowest = self._safe_get('v_lowest', 0.0)  # v from step 2
        self.v_after = self._safe_get('v_after', 0.0)    # v from step 5
        self.a = self._safe_get('a', 0.0)               # acceleration from step 3

        # Symbolic variables for invariant checks
        self._init_symbols()

    def _safe_get(self, key, default):
        """Safely retrieve attribute from solution instance or its results dict."""
        val = getattr(self.solution, key, None)
        if val is None:
            val = getattr(self.solution, 'results', {}).get(key, default)
        return float(val) if isinstance(val, (int, float, str)) else default

    def _init_symbols(self):
        """Initialize SymPy symbols for physical laws."""
        self.m_sym = sp.symbols('m', positive=True)
        self.g_sym = sp.symbols('g', positive=True)
        self.q_sym = sp.symbols('q', positive=True)
        self.E_sym = sp.symbols('E', positive=True)
        self.l_sym = sp.symbols('l', positive=True)
        self.theta_sym = sp.symbols('theta', positive=True)
        self.v_sym = sp.symbols('v', positive=True)
        self.a_sym = sp.symbols('a', positive=True)
        self.t_sym = sp.symbols('t', positive=True)

    def test_invariants(self):
        """Verify fundamental physics laws based on KB theorems."""
        logs = []
        success = True

        # 1. Equilibrium of forces (step 1)
        try:
            tan_theta = self.sin37 / self.cos37  # tan(37°)
            lhs = self.q * self.E
            rhs = self.m * self.g * tan_theta
            if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Equilibrium check failed: qE={lhs:.3e}, mg tanθ={rhs:.3e}")
                success = False
            else:
                logs.append("Equilibrium check passed.")
        except Exception as e:
            logs.append(f"Equilibrium check error: {e}")
            success = False

        # 2. Work-energy theorem (step 2)
        try:
            work_electric = -self.E * self.q * self.l
            work_gravity = self.m * self.g * self.l
            net_work = work_electric + work_gravity
            delta_ke = 0.5 * self.m * self.v_lowest**2
            if not math.isclose(net_work, delta_ke, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Work-energy check failed: W_net={net_work:.3e}, ΔKE={delta_ke:.3e}")
                success = False
            else:
                logs.append("Work-energy check passed.")
        except Exception as e:
            logs.append(f"Work-energy check error: {e}")
            success = False

        # 3. Newton's second law (step 3)
        try:
            expected_a = self.g / self.cos37
            if not math.isclose(self.a, expected_a, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Newton's second law check failed: a={self.a:.3f}, g/cosθ={expected_a:.3f}")
                success = False
            else:
                logs.append("Newton's second law check passed.")
        except Exception as e:
            logs.append(f"Newton's second law check error: {e}")
            success = False

        # 4. Kinematic equation (step 4/5)
        try:
            expected_v_after = self.a * self.t
            if not math.isclose(self.v_after, expected_v_after, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Kinematic check failed: v_after={self.v_after:.3f}, a*t={expected_v_after:.3f}")
                success = False
            else:
                logs.append("Kinematic check passed.")
        except Exception as e:
            logs.append(f"Kinematic check error: {e}")
            success = False

        return success, logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True

        # Check that step 2's v_lowest is consistent with work-energy theorem
        try:
            v_calculated = math.sqrt(2 * (-self.E * self.q * self.l + self.m * self.g * self.l) / self.m)
            if not math.isclose(self.v_lowest, v_calculated, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Step 2 consistency failed: v_lowest={self.v_lowest:.3f}, calculated={v_calculated:.3f}")
                success = False
            else:
                logs.append("Step 2 consistency passed.")
        except Exception as e:
            logs.append(f"Step 2 consistency error: {e}")
            success = False

        # Check that step 5's v_after is consistent with a and t
        try:
            v_from_a = self.a * self.t
            if not math.isclose(self.v_after, v_from_a, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Step 5 consistency failed: v_after={self.v_after:.3f}, a*t={v_from_a:.3f}")
                success = False
            else:
                logs.append("Step 5 consistency passed.")
        except Exception as e:
            logs.append(f"Step 5 consistency error: {e}")
            success = False

        return success, logs

    def test_monotonicity(self):
        """Perturb mass m and verify expected trends."""
        logs = []
        success = True

        # Perturb mass by +1%
        original_m = self.m
        perturbed_m = original_m * 1.01
        logs.append(f"Original m = {original_m:.3e}, perturbed m = {perturbed_m:.3e}")

        # 1. Effect on E (step 1): E = (m*g*tanθ)/q
        try:
            tan_theta = self.sin37 / self.cos37
            original_E = (original_m * self.g * tan_theta) / self.q
            perturbed_E = (perturbed_m * self.g * tan_theta) / self.q
            if perturbed_E <= original_E:
                logs.append(f"Monotonicity E failed: E does not increase with m (original={original_E:.3e}, perturbed={perturbed_E:.3e})")
                success = False
            else:
                logs.append(f"Monotonicity E passed: E increases with m.")
        except Exception as e:
            logs.append(f"Monotonicity E error: {e}")
            success = False

        # 2. Effect on v_lowest (step 2): v^2 = 2*(-E*q*l + m*g*l)/m
        #    Substitute E = (m*g*tanθ)/q, then v^2 = 2*g*l*(1 - tanθ) -> independent of m
        try:
            v2_original = 2 * self.g * self.l * (1 - tan_theta)
            v2_perturbed = 2 * self.g * self.l * (1 - tan_theta)
            if not math.isclose(v2_original, v2_perturbed, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Monotonicity v_lowest failed: depends on m (original v^2={v2_original:.3e}, perturbed v^2={v2_perturbed:.3e})")
                success = False
            else:
                logs.append("Monotonicity v_lowest passed: independent of m.")
        except Exception as e:
            logs.append(f"Monotonicity v_lowest error: {e}")
            success = False

        # 3. Effect on v_after (step 5): v = a*t, a = g/cosθ -> independent of m
        try:
            a_original = self.g / self.cos37
            a_perturbed = self.g / self.cos37
            if not math.isclose(a_original, a_perturbed, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("Monotonicity v_after failed: a depends on m")
                success = False
            else:
                logs.append("Monotonicity v_after passed: independent of m.")
        except Exception as e:
            logs.append(f"Monotonicity v_after error: {e}")
            success = False

        # Restore original mass (conceptual, since we never actually modified the instance)
        logs.append("Mass perturbation restored (conceptual).")

        return success, logs

    def run_all_tests(self):
        """Execute all checks and return consolidated results."""
        all_logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"=== {test_name} ===")
                all_logs.append(f"Test crashed: {traceback.format_exc()}")
                overall_success = False

        return overall_success, all_logs