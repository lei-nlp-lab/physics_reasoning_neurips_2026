import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for inputs
        self.m = getattr(solution_instance, 'm', 50.0)
        self.mu = getattr(solution_instance, 'mu', 0.1)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.F = getattr(solution_instance, 'F', 200.0)
        self.L1 = getattr(solution_instance, 'L1', 8.0)
        self.L2 = getattr(solution_instance, 'L2', 2.0)

        # Safe attribute access for results
        self.results = getattr(solution_instance, 'results', {})
        self.a_1 = self.results.get('a_1', 0.0)
        self.F_net = self.results.get('F_net', 0.0)
        self.t = self.results.get('t', 0.0)  # assume single numeric value
        self.F_net2 = self.results.get('F_net2', 0.0)
        self.a_2 = self.results.get('a_2', 0.0)
        self.v = self.results.get('v', 0.0)
        self.x_2 = self.results.get('x_2', 0.0)
        self.x_total = self.results.get('x_total', 0.0)
        self.x_1 = self.results.get('x_1', 0.0)

        # Expected values from physics
        self.expected_F_net = self.F - self.mu * self.m * self.g
        self.expected_a_1 = self.expected_F_net / self.m
        self.expected_a_2 = self.mu * self.g
        self.expected_F_net2 = self.mu * self.m * self.g  # magnitude

    def test_invariants(self):
        logs = []
        success = True

        # Check Newton's second law during thrust
        if not math.isclose(self.F_net, self.expected_F_net, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"F_net mismatch: expected {self.expected_F_net:.2f}, got {self.F_net:.2f}")
            success = False
        if not math.isclose(self.a_1, self.expected_a_1, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"a_1 mismatch: expected {self.expected_a_1:.2f}, got {self.a_1:.2f}")
            success = False

        # Check Newton's second law after thrust
        if not math.isclose(self.F_net2, self.expected_F_net2, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"F_net2 mismatch: expected {self.expected_F_net2:.2f}, got {self.F_net2:.2f}")
            success = False
        if not math.isclose(self.a_2, self.expected_a_2, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"a_2 mismatch: expected {self.expected_a_2:.2f}, got {self.a_2:.2f}")
            success = False

        # Energy/work consistency: (F - μmg)*x_1 = μmg*x_2
        lhs = (self.F - self.mu * self.m * self.g) * self.x_1
        rhs = self.mu * self.m * self.g * self.x_2
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Work-energy mismatch: (F - μmg)*x_1 = {lhs:.2f}, μmg*x_2 = {rhs:.2f}")
            success = False

        return success, logs

    def test_trace_consistency(self):
        logs = []
        success = True

        # Kinematics: x_1 = 0.5 * a_1 * t^2
        expected_x_1 = 0.5 * self.a_1 * self.t**2
        if not math.isclose(self.x_1, expected_x_1, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"x_1 mismatch: expected {expected_x_1:.2f}, got {self.x_1:.2f}")
            success = False

        # v = a_1 * t
        expected_v = self.a_1 * self.t
        if not math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"v mismatch: expected {expected_v:.2f}, got {self.v:.2f}")
            success = False

        # x_2 = v^2 / (2 * a_2)
        if abs(self.a_2) > 1e-12:
            expected_x_2 = self.v**2 / (2 * self.a_2)
            if not math.isclose(self.x_2, expected_x_2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"x_2 mismatch: expected {expected_x_2:.2f}, got {self.x_2:.2f}")
                success = False
        else:
            logs.append("Warning: a_2 too small for x_2 check")

        # x_total = x_1 + x_2
        expected_x_total = self.x_1 + self.x_2
        if not math.isclose(self.x_total, expected_x_total, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"x_total mismatch: expected {expected_x_total:.2f}, got {self.x_total:.2f}")
            success = False

        # Success region: L1-L2 <= x_total <= L1
        lower_bound = self.L1 - self.L2
        if not (lower_bound - 1e-2 <= self.x_total <= self.L1 + 1e-2):
            logs.append(f"x_total = {self.x_total:.2f} not in [{lower_bound:.2f}, {self.L1:.2f}]")
            success = False

        # t within [1, 2√3/3]
        t_min = 1.0
        t_max = 2.0 * math.sqrt(3) / 3.0  # ≈ 1.1547
        if not (t_min - 1e-2 <= self.t <= t_max + 1e-2):
            logs.append(f"t = {self.t:.2f} not in [{t_min:.2f}, {t_max:.2f}]")
            success = False

        return success, logs

    def test_monotonicity(self):
        logs = []
        success = True

        # Perturb μ by +10%
        mu_perturbed = self.mu * 1.1

        # Recompute accelerations
        a_1_perturbed = (self.F - mu_perturbed * self.m * self.g) / self.m
        a_2_perturbed = mu_perturbed * self.g

        # Check trends
        if a_1_perturbed >= self.a_1:
            logs.append(f"a_1 should decrease when μ increases: {self.a_1:.2f} -> {a_1_perturbed:.2f}")
            success = False
        if a_2_perturbed <= self.a_2:
            logs.append(f"a_2 should increase when μ increases: {self.a_2:.2f} -> {a_2_perturbed:.2f}")
            success = False

        return success, logs

    def run_all_tests(self):
        all_logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            try:
                test_success, logs = test_func()
                if not test_success:
                    overall_success = False
                    all_logs.append(f"{test_name} failed:")
                    all_logs.extend(logs)
                else:
                    all_logs.append(f"{test_name} passed.")
            except Exception as e:
                overall_success = False
                all_logs.append(f"{test_name} error: {e}")
                all_logs.append(traceback.format_exc())

        return overall_success, all_logs