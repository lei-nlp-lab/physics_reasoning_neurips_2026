import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', 2.0))
        self.L = getattr(solution_instance, 'L', solution_instance.results.get('L', 2.5))
        self.theta = getattr(solution_instance, 'theta', solution_instance.results.get('theta', 37.0))
        self.mu1 = getattr(solution_instance, 'mu1', solution_instance.results.get('mu1', 0.65))
        self.mu2 = getattr(solution_instance, 'mu2', solution_instance.results.get('mu2', 0.2))
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', 10.0))
        self.sin37 = getattr(solution_instance, 'sin37', solution_instance.results.get('sin37', 0.6))
        self.cos37 = getattr(solution_instance, 'cos37', solution_instance.results.get('cos37', 0.8))
        self.h = getattr(solution_instance, 'h', solution_instance.results.get('h', 0.8))
        self.s = getattr(solution_instance, 's', solution_instance.results.get('s', 1.6))

        # Safe extraction of results
        self.v_1 = getattr(solution_instance, 'v_1', solution_instance.results.get('v_1', None))
        self.v_2 = getattr(solution_instance, 'v_2', solution_instance.results.get('v_2', None))
        self.t = getattr(solution_instance, 't', solution_instance.results.get('t', None))
        self.a = getattr(solution_instance, 'a', solution_instance.results.get('a', None))
        self.x_1 = getattr(solution_instance, 'x_1', solution_instance.results.get('x_1', None))
        self.Q = getattr(solution_instance, 'Q', solution_instance.results.get('Q', None))
        self.v_C = getattr(solution_instance, 'v_C', solution_instance.results.get('v_C', None))
        self.x_2 = getattr(solution_instance, 'x_2', solution_instance.results.get('x_2', None))
        self.t_1 = getattr(solution_instance, 't_1', solution_instance.results.get('t_1', None))

    def test_invariants(self):
        logs = []
        success = True

        # 1. Work-energy theorem on incline: m*g*L*sinθ - μ1*m*g*cosθ*L = 0.5*m*v_1^2
        if self.v_1 is not None:
            expected_KE = 0.5 * self.m * self.v_1**2
            work_done = self.m * self.g * self.L * (self.sin37 - self.mu1 * self.cos37)
            if not math.isclose(expected_KE, work_done, rel_tol=1e-2, abs_tol=1e-2):
                success = False
                logs.append(f"Work-energy theorem on incline failed: KE={expected_KE:.3f} J, work={work_done:.3f} J")
        else:
            success = False
            logs.append("Missing v_1 for work-energy check.")

        # 2. Newton's second law on conveyor belt: μ2*m*g = m*a → a = μ2*g
        if self.a is not None:
            expected_a = self.mu2 * self.g
            if not math.isclose(self.a, expected_a, rel_tol=1e-2, abs_tol=1e-2):
                success = False
                logs.append(f"Newton's second law on belt failed: a={self.a:.3f} m/s², expected {expected_a:.3f} m/s²")
        else:
            success = False
            logs.append("Missing a for Newton's second law check.")

        # 3. Projectile motion: t = sqrt(2h/g)
        if self.t is not None:
            expected_t = math.sqrt(2 * self.h / self.g)
            if not math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2):
                success = False
                logs.append(f"Projectile time failed: t={self.t:.3f} s, expected {expected_t:.3f} s")
        else:
            success = False
            logs.append("Missing t for projectile check.")

        # 4. Horizontal velocity at C: v_C = s / t
        if self.v_C is not None and self.t is not None:
            expected_v_C = self.s / self.t
            if not math.isclose(self.v_C, expected_v_C, rel_tol=1e-2, abs_tol=1e-2):
                success = False
                logs.append(f"Horizontal velocity at C failed: v_C={self.v_C:.3f} m/s, expected {expected_v_C:.3f} m/s")
        elif self.v_C is None:
            success = False
            logs.append("Missing v_C for horizontal velocity check.")

        # 5. Conveyor belt speed equals exit speed: v_2 = v_C
        if self.v_2 is not None and self.v_C is not None:
            if not math.isclose(self.v_2, self.v_C, rel_tol=1e-2, abs_tol=1e-2):
                success = False
                logs.append(f"Belt/exit speed mismatch: v_2={self.v_2:.3f} m/s, v_C={self.v_C:.3f} m/s")
        elif self.v_2 is None:
            success = False
            logs.append("Missing v_2 for speed comparison.")

        return success, logs

    def test_trace_consistency(self):
        logs = []
        success = True

        # Check kinematics on conveyor belt: t_1 = (v_2 - v_1) / a
        if all(val is not None for val in [self.v_1, self.v_2, self.a, self.t_1]):
            expected_t_1 = (self.v_2 - self.v_1) / self.a
            if not math.isclose(self.t_1, expected_t_1, rel_tol=1e-2, abs_tol=1e-2):
                success = False
                logs.append(f"Time t_1 inconsistent: {self.t_1:.3f} s, expected {expected_t_1:.3f} s")
        else:
            success = False
            logs.append("Missing values for t_1 check.")

        # Check x_1 = (v_1 + v_2) * t_1 / 2
        if all(val is not None for val in [self.v_1, self.v_2, self.t_1, self.x_1]):
            expected_x_1 = (self.v_1 + self.v_2) * self.t_1 / 2
            if not math.isclose(self.x_1, expected_x_1, rel_tol=1e-2, abs_tol=1e-2):
                success = False
                logs.append(f"Displacement x_1 inconsistent: {self.x_1:.3f} m, expected {expected_x_1:.3f} m")
        else:
            success = False
            logs.append("Missing values for x_1 check.")

        # Check x_2 = v_2 * t_1
        if all(val is not None for val in [self.v_2, self.t_1, self.x_2]):
            expected_x_2 = self.v_2 * self.t_1
            if not math.isclose(self.x_2, expected_x_2, rel_tol=1e-2, abs_tol=1e-2):
                success = False
                logs.append(f"Belt displacement x_2 inconsistent: {self.x_2:.3f} m, expected {expected_x_2:.3f} m")
        else:
            success = False
            logs.append("Missing values for x_2 check.")

        # Check heat generated: Q = μ2 * m * g * (x_2 - x_1)
        if all(val is not None for val in [self.mu2, self.m, self.g, self.x_2, self.x_1, self.Q]):
            expected_Q = self.mu2 * self.m * self.g * (self.x_2 - self.x_1)
            if not math.isclose(self.Q, expected_Q, rel_tol=1e-2, abs_tol=1e-2):
                success = False
                logs.append(f"Heat Q inconsistent: {self.Q:.3f} J, expected {expected_Q:.3f} J")
        else:
            success = False
            logs.append("Missing values for Q check.")

        return success, logs

    def test_monotonicity(self):
        logs = []
        success = True

        # Perturb mass m by +5% and verify Q scales linearly
        original_m = self.m
        perturbed_m = original_m * 1.05

        # Recompute v_1 (independent of m): v_1 = sqrt(2*g*L*(sinθ - μ1*cosθ))
        v_1 = math.sqrt(2 * self.g * self.L * (self.sin37 - self.mu1 * self.cos37))
        # v_2 is from projectile motion and is independent of m
        v_2 = self.v_2 if self.v_2 is not None else self.s / math.sqrt(2 * self.h / self.g)
        a = self.mu2 * self.g
        t_1 = (v_2 - v_1) / a
        x_1 = (v_1 + v_2) * t_1 / 2
        x_2 = v_2 * t_1
        Q_perturbed = self.mu2 * perturbed_m * self.g * (x_2 - x_1)

        if self.Q is not None:
            expected_ratio = perturbed_m / original_m
            actual_ratio = Q_perturbed / self.Q
            if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                success = False
                logs.append(f"Monotonicity test failed: Q should scale linearly with m. Ratio = {actual_ratio:.3f}, expected {expected_ratio:.3f}")
        else:
            success = False
            logs.append("Missing Q for monotonicity check.")

        return success, logs

    def run_all_tests(self):
        all_success = True
        all_logs = []

        tests = [
            ("invariants", self.test_invariants),
            ("trace_consistency", self.test_trace_consistency),
            ("monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            try:
                test_success, test_logs = test_func()
                if not test_success:
                    all_success = False
                    all_logs.append(f"[{test_name}] failed:")
                    all_logs.extend([f"  - {log}" for log in test_logs])
                else:
                    all_logs.append(f"[{test_name}] passed.")
            except Exception as e:
                all_success = False
                all_logs.append(f"[{test_name}] crashed: {str(e)}")
                all_logs.append(traceback.format_exc())

        return all_success, all_logs