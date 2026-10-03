import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access
        self.inputs = getattr(solution_instance, 'inputs', {})
        self.results = getattr(solution_instance, 'results', {})

        # Extract inputs with defaults
        self.m = self.inputs.get('m', 1.0)
        self.R = self.inputs.get('R', 1.0)
        self.g = self.inputs.get('g', 9.8)
        self.mu = self.inputs.get('mu_k', 0.5)

        # Extract results with defaults
        self.x = self.results.get('x', 0.0)
        self.y = self.results.get('y', 0.0)
        self.E_kB = self.results.get('E_kB', 0.0)
        self.F_1 = self.results.get('F_1', 0.0)
        self.W_f = self.results.get('W_f', 0.0)
        self.v_C = self.results.get('v_C', 0.0)
        self.F_centripetal = self.results.get('F_centripetal', 0.0)
        self.theta = self.results.get('theta', 0.0)  # could be radians or degrees
        self.W = self.results.get('W', 0.0)
        self.s = self.results.get('s', 0.0)
        self.F_2 = self.results.get('F_2', 0.0)

    def test_invariants(self):
        logs = []
        # 1. Centripetal force at B should be 7mg
        expected_F_cent = 7 * self.m * self.g
        if not math.isclose(self.F_centripetal, expected_F_cent, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Invariant failed: F_centripetal = {self.F_centripetal}, expected {expected_F_cent}")

        # 2. Velocity at C should be sqrt(gR)
        expected_v_C = math.sqrt(self.g * self.R)
        if not math.isclose(self.v_C, expected_v_C, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Invariant failed: v_C = {self.v_C}, expected {expected_v_C}")

        # 3. Projectile motion consistency
        # Time to fall from height 2R
        t = math.sqrt(4 * self.R / self.g)
        expected_x = self.v_C * t  # should be 2R
        if not math.isclose(self.x, expected_x, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Invariant failed: x = {self.x}, expected {expected_x}")
        expected_y = 2 * self.R
        if not math.isclose(self.y, expected_y, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Invariant failed: y = {self.y}, expected {expected_y}")
        expected_s = math.sqrt(self.x**2 + self.y**2)
        if not math.isclose(self.s, expected_s, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Invariant failed: s = {self.s}, expected {expected_s}")
        if self.x != 0:
            expected_theta_rad = math.atan(self.y / self.x)
            # Check both radians and degrees (45° = π/4 rad)
            if not (math.isclose(self.theta, expected_theta_rad, rel_tol=1e-2, abs_tol=1e-2) or
                    math.isclose(self.theta, math.degrees(expected_theta_rad), rel_tol=1e-2, abs_tol=1e-2)):
                logs.append(f"Invariant failed: theta = {self.theta}, expected {expected_theta_rad} rad or {math.degrees(expected_theta_rad)} deg")
        else:
            logs.append("Invariant warning: x is zero, skipping theta check")

        # 4. Work-energy theorem: W - |W_f| = E_kB (assuming W_f stored as positive magnitude)
        if not math.isclose(self.W - self.W_f, self.E_kB, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Invariant failed: W - W_f = {self.W - self.W_f}, E_kB = {self.E_kB}")

        # 5. Friction work: W_f = μ * m * g * 3R
        expected_W_f = self.mu * self.m * self.g * 3 * self.R
        if not math.isclose(self.W_f, expected_W_f, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Invariant failed: W_f = {self.W_f}, expected {expected_W_f}")

        # 6. Spring work: W = 5mgR
        expected_W = 5 * self.m * self.g * self.R
        if not math.isclose(self.W, expected_W, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Invariant failed: W = {self.W}, expected {expected_W}")

        # 7. Kinetic energy at B: E_kB = 3.5mgR
        expected_E_kB = 3.5 * self.m * self.g * self.R
        if not math.isclose(self.E_kB, expected_E_kB, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Invariant failed: E_kB = {self.E_kB}, expected {expected_E_kB}")

        return len(logs) == 0, logs

    def test_trace_consistency(self):
        logs = []
        # 1. Check v_B from E_kB and from F_centripetal
        if self.m > 0:
            v_B_from_E = math.sqrt(2 * self.E_kB / self.m)
            v_B_from_F = math.sqrt(self.F_centripetal * self.R / self.m)
            if not math.isclose(v_B_from_E, v_B_from_F, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Trace inconsistency: v_B from E_kB = {v_B_from_E}, from F_centripetal = {v_B_from_F}")
        else:
            logs.append("Trace warning: m is zero, skipping v_B consistency")

        # 2. Projectile time from y and g
        if self.y > 0:
            t_from_y = math.sqrt(2 * self.y / self.g)
            x_from_v_C_t = self.v_C * t_from_y
            if not math.isclose(self.x, x_from_v_C_t, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Trace inconsistency: x from v_C and t = {x_from_v_C_t}, stored x = {self.x}")
        else:
            logs.append("Trace warning: y is non-positive, skipping projectile time check")

        # 3. s and theta from x and y
        s_calc = math.sqrt(self.x**2 + self.y**2)
        if not math.isclose(self.s, s_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Trace inconsistency: s computed = {s_calc}, stored s = {self.s}")
        if self.x != 0:
            theta_calc_rad = math.atan(self.y / self.x)
            # Allow both radians and degrees
            if not (math.isclose(self.theta, theta_calc_rad, rel_tol=1e-2, abs_tol=1e-2) or
                    math.isclose(self.theta, math.degrees(theta_calc_rad), rel_tol=1e-2, abs_tol=1e-2)):
                logs.append(f"Trace inconsistency: theta computed = {theta_calc_rad} rad, stored theta = {self.theta}")
        else:
            logs.append("Trace warning: x is zero, skipping theta consistency")

        return len(logs) == 0, logs

    def test_monotonicity(self):
        logs = []
        # Helper to compute expected values for given inputs
        def compute_expected(m, R, g, mu):
            F_cent = 7 * m * g
            v_C = math.sqrt(g * R)
            t = math.sqrt(4 * R / g)
            x = v_C * t
            y = 2 * R
            s = math.sqrt(x**2 + y**2)
            theta = math.atan(y / x)  # radians
            W_f = mu * m * g * 3 * R
            E_kB = 3.5 * m * g * R
            W = 5 * m * g * R
            return {
                'F_cent': F_cent, 's': s, 'W': W,
                'x': x, 'y': y, 'theta': theta,
                'W_f': W_f, 'E_kB': E_kB, 'v_C': v_C
            }

        # Original expected values
        orig_exp = compute_expected(self.m, self.R, self.g, self.mu)
        # Perturb mass by +10%
        perturbed_m = self.m * 1.1
        pert_exp = compute_expected(perturbed_m, self.R, self.g, self.mu)

        # Check monotonic trends
        if not pert_exp['F_cent'] > orig_exp['F_cent']:
            logs.append("Monotonicity fail: F_centripetal does not increase with m")
        if not pert_exp['W'] > orig_exp['W']:
            logs.append("Monotonicity fail: W does not increase with m")
        if not pert_exp['W_f'] > orig_exp['W_f']:
            logs.append("Monotonicity fail: W_f does not increase with m")
        if not pert_exp['E_kB'] > orig_exp['E_kB']:
            logs.append("Monotonicity fail: E_kB does not increase with m")
        if not math.isclose(pert_exp['s'], orig_exp['s'], rel_tol=1e-2, abs_tol=1e-2):
            logs.append("Monotonicity fail: s changes with m (should be independent)")
        # v_C, x, y, theta should be independent of m
        if not math.isclose(pert_exp['v_C'], orig_exp['v_C'], rel_tol=1e-2, abs_tol=1e-2):
            logs.append("Monotonicity fail: v_C changes with m (should be independent)")
        if not math.isclose(pert_exp['x'], orig_exp['x'], rel_tol=1e-2, abs_tol=1e-2):
            logs.append("Monotonicity fail: x changes with m (should be independent)")
        if not math.isclose(pert_exp['y'], orig_exp['y'], rel_tol=1e-2, abs_tol=1e-2):
            logs.append("Monotonicity fail: y changes with m (should be independent)")
        if not math.isclose(pert_exp['theta'], orig_exp['theta'], rel_tol=1e-2, abs_tol=1e-2):
            logs.append("Monotonicity fail: theta changes with m (should be independent)")

        return len(logs) == 0, logs

    def run_all_tests(self):
        all_logs = []
        success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            try:
                test_success, logs = test_func()
                if not test_success:
                    success = False
                    all_logs.extend([f"{test_name}: {log}" for log in logs])
                else:
                    all_logs.append(f"{test_name}: passed")
            except Exception as e:
                success = False
                all_logs.append(f"{test_name}: error - {traceback.format_exc()}")

        return success, all_logs