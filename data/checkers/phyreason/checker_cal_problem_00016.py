import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract inputs with safe defaults
        self.m = getattr(solution_instance, 'm', 110.0)
        self.s_AB = 12.0  # m
        self.s_BC = 20.0  # m
        self.theta_deg = 15.0  # degrees
        self.sin_theta = 0.26  # given
        self.v_A = 0.0
        self.v_B = 8.0  # m/s
        self.total_time = 5.0  # s
        self.g = 10.0  # assumed, as it matches the given numbers

        # Extract results safely
        self.a_AB = self._safe_get(solution_instance, 'a_1')
        self.t_AB = self._safe_get(solution_instance, 't')
        self.v_C = self._safe_get(solution_instance, 'v_C')
        self.f = self._safe_get(solution_instance, 'f')

        # Derived quantities
        self.t_BC = self.total_time - self.t_AB if self.t_AB is not None else None
        self.a_BC = (self.v_C - self.v_B) / self.t_BC if (self.v_C is not None and self.t_BC is not None and self.t_BC > 0) else None

    def _safe_get(self, obj, key):
        """Safely retrieve attribute or result dict entry."""
        val = getattr(obj, key, None)
        if val is None and hasattr(obj, 'results'):
            val = obj.results.get(key, None)
        return float(val) if val is not None else None

    def test_invariants(self):
        logs = []
        try:
            # 1. Kinematic invariant for AB: v_B^2 = 2 * a_AB * s_AB
            if self.a_AB is not None:
                left = self.v_B ** 2
                right = 2 * self.a_AB * self.s_AB
                if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: AB kinematic v_B^2 = {left} != 2*a_AB*s_AB = {right}")
                else:
                    logs.append(f"PASS: AB kinematic v_B^2 = 2*a_AB*s_AB")

            # 2. Kinematic invariant for AB: v_B = a_AB * t_AB
            if self.a_AB is not None and self.t_AB is not None:
                left = self.v_B
                right = self.a_AB * self.t_AB
                if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: AB kinematic v_B = {left} != a_AB*t_AB = {right}")
                else:
                    logs.append(f"PASS: AB kinematic v_B = a_AB*t_AB")

            # 3. Kinematic invariant for BC: s_BC = (v_B + v_C)/2 * t_BC
            if self.v_C is not None and self.t_BC is not None:
                left = self.s_BC
                right = (self.v_B + self.v_C) / 2 * self.t_BC
                if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: BC kinematic s_BC = {left} != (v_B+v_C)/2 * t_BC = {right}")
                else:
                    logs.append(f"PASS: BC kinematic s_BC = (v_B+v_C)/2 * t_BC")

            # 4. Kinematic invariant for BC: v_C = v_B + a_BC * t_BC
            if self.a_BC is not None and self.t_BC is not None and self.v_C is not None:
                left = self.v_C
                right = self.v_B + self.a_BC * self.t_BC
                if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: BC kinematic v_C = {left} != v_B + a_BC*t_BC = {right}")
                else:
                    logs.append(f"PASS: BC kinematic v_C = v_B + a_BC*t_BC")

            # 5. Newton's second law on BC: m*g*sinθ - f = m * a_BC
            if self.f is not None and self.a_BC is not None:
                left = self.m * self.g * self.sin_theta - self.f
                right = self.m * self.a_BC
                if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Newton's law on BC: m*g*sinθ - f = {left} != m*a_BC = {right}")
                else:
                    logs.append(f"PASS: Newton's law on BC: m*g*sinθ - f = m*a_BC")

        except Exception as e:
            logs.append(f"ERROR in test_invariants: {e}")
        return logs

    def test_trace_consistency(self):
        logs = []
        try:
            # Check t_AB from v_B and a_AB
            if self.a_AB is not None:
                t_AB_calc = self.v_B / self.a_AB
                if self.t_AB is not None and not math.isclose(self.t_AB, t_AB_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: t_AB from v_B/a_AB = {t_AB_calc} != given t_AB = {self.t_AB}")
                else:
                    logs.append(f"PASS: t_AB consistent with v_B and a_AB")

            # Check a_BC from kinematics vs Newton
            if self.a_BC is not None and self.f is not None:
                a_BC_newton = (self.m * self.g * self.sin_theta - self.f) / self.m
                if not math.isclose(self.a_BC, a_BC_newton, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: a_BC from kinematics = {self.a_BC} != a_BC from Newton = {a_BC_newton}")
                else:
                    logs.append(f"PASS: a_BC consistent between kinematics and Newton")

            # Check v_C from average velocity
            if self.v_C is not None and self.t_BC is not None:
                v_C_avg = 2 * self.s_BC / self.t_BC - self.v_B
                if not math.isclose(self.v_C, v_C_avg, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: v_C from average velocity = {v_C_avg} != given v_C = {self.v_C}")
                else:
                    logs.append(f"PASS: v_C consistent with average velocity formula")

        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {e}")
        return logs

    def test_monotonicity(self):
        logs = []
        original_m = self.m
        try:
            # Perturb mass by +10%
            perturbed_m = original_m * 1.10
            # Recalculate f using Newton's law, assuming a_BC unchanged
            if self.a_BC is not None:
                f_new = perturbed_m * (self.g * self.sin_theta - self.a_BC)
                expected_ratio = perturbed_m / original_m
                actual_ratio = f_new / self.f if self.f != 0 else 1.0
                if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: f does not scale linearly with mass. Expected ratio {expected_ratio}, got {actual_ratio}")
                else:
                    logs.append(f"PASS: f scales linearly with mass (monotonicity)")
        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {e}")
        finally:
            # Restore original mass (not strictly needed as we didn't modify instance)
            self.m = original_m
        return logs

    def run_all_tests(self):
        all_logs = []
        try:
            all_logs.append("=== test_invariants ===")
            all_logs.extend(self.test_invariants())
            all_logs.append("=== test_trace_consistency ===")
            all_logs.extend(self.test_trace_consistency())
            all_logs.append("=== test_monotonicity ===")
            all_logs.extend(self.test_monotonicity())
            # Determine overall success: no lines starting with "FAIL:" or "ERROR:"
            success = not any(line.startswith("FAIL:") or line.startswith("ERROR:") for line in all_logs)
        except Exception as e:
            all_logs.append(f"CRITICAL ERROR in run_all_tests: {e}")
            success = False
        return success, all_logs