import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access using getattr and .get()
        self.R = 0.40
        self.m = 0.04
        self.q = -1e-4
        self.E = 1e3
        self.mu = 0.05
        self.g = 10.0

        # Extract results from solution_instance
        self.v_A = getattr(solution_instance, 'v_A', 
                          getattr(solution_instance, 'results', {}).get('v_A', None))
        self.x = getattr(solution_instance, 'x', 
                        getattr(solution_instance, 'results', {}).get('x', None))
        self.v_min = getattr(solution_instance, 'v_min', 
                            getattr(solution_instance, 'results', {}).get('v_min', None))
        self.v_x_expr = getattr(solution_instance, 'v_x', 
                               getattr(solution_instance, 'results', {}).get('v_x', None))
        self.v_y_expr = getattr(solution_instance, 'v_y', 
                               getattr(solution_instance, 'results', {}).get('v_y', None))
        self.v_expr = getattr(solution_instance, 'v', 
                             getattr(solution_instance, 'results', {}).get('v', None))

        # Store original parameters for restoration
        self.original_params = {
            'm': self.m,
            'q': self.q,
            'mu': self.mu,
            'E': self.E,
            'R': self.R,
            'g': self.g
        }

    def test_invariants(self):
        logs = []
        try:
            # 1. Check v_A from circular motion at point A: m*v_A^2/R = m*g
            if self.v_A is not None:
                expected_v_A = math.sqrt(self.g * self.R)
                if math.isclose(self.v_A, expected_v_A, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"✓ v_A invariant passed: {self.v_A} ≈ sqrt(g*R) = {expected_v_A}")
                else:
                    logs.append(f"✗ v_A invariant failed: {self.v_A} != sqrt(g*R) = {expected_v_A}")
            else:
                logs.append("⚠ v_A not provided, skipping invariant check")

            # 2. Check work-energy theorem from release to point A
            if self.v_A is not None and self.x is not None:
                # Net work = change in KE
                # Forces: electric (qE), friction (mu*m*g), gravity (mg*2R)
                # Work by electric: -qE*x (since q negative and displacement opposite to E)
                # Work by friction: -mu*m*g*x
                # Work by gravity: -m*g*2R
                net_work = -self.q * self.E * self.x - self.mu * self.m * self.g * self.x - 2 * self.m * self.g * self.R
                delta_ke = 0.5 * self.m * self.v_A**2
                if math.isclose(net_work, delta_ke, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"✓ Work-energy invariant passed: net_work = {net_work} ≈ ΔKE = {delta_ke}")
                else:
                    logs.append(f"✗ Work-energy invariant failed: net_work = {net_work} != ΔKE = {delta_ke}")
            else:
                logs.append("⚠ v_A or x not provided, skipping work-energy check")

            # 3. Check minimum speed consistency (if v_min provided)
            if self.v_min is not None and self.v_A is not None:
                # Analytical minimum from projectile motion:
                # v_x = v_A + (qE/m)*t, v_y = g*t
                a_x = self.q * self.E / self.m  # = -2.5
                a_y = self.g  # = 10
                t_min = -self.v_A * a_x / (a_x**2 + a_y**2)  # from derivative of v^2
                v_min_sq = (self.v_A + a_x * t_min)**2 + (a_y * t_min)**2
                v_min_calc = math.sqrt(v_min_sq)
                if math.isclose(self.v_min, v_min_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"✓ v_min invariant passed: {self.v_min} ≈ calculated {v_min_calc}")
                else:
                    logs.append(f"✗ v_min invariant failed: {self.v_min} != calculated {v_min_calc}")
            else:
                logs.append("⚠ v_min or v_A not provided, skipping minimum speed check")

        except Exception as e:
            logs.append(f"✗ Error in test_invariants: {traceback.format_exc()}")
        return logs

    def test_trace_consistency(self):
        logs = []
        try:
            # Cross-check intermediate results with final answers
            if self.v_A is not None and self.x is not None and self.v_min is not None:
                # Check that v_min < v_A (since speed decreases after A)
                if self.v_min < self.v_A:
                    logs.append(f"✓ v_min ({self.v_min}) < v_A ({self.v_A}) as expected")
                else:
                    logs.append(f"✗ v_min ({self.v_min}) >= v_A ({self.v_A}), unexpected")

                # Verify that the work-energy equation yields the same x when solved
                # -qE*x - mu*m*g*x - 2*m*g*R = 0.5*m*v_A^2
                x_calc = (0.5 * self.m * self.v_A**2 + 2 * self.m * self.g * self.R) / (-self.q * self.E - self.mu * self.m * self.g)
                if math.isclose(self.x, x_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"✓ x trace consistency passed: {self.x} ≈ calculated {x_calc}")
                else:
                    logs.append(f"✗ x trace consistency failed: {self.x} != calculated {x_calc}")

            # Check symbolic expressions if provided
            if self.v_x_expr is not None and isinstance(self.v_x_expr, str):
                # Expected: "2 - 2.5*t"
                expected = "2 - 2.5*t"
                if self.v_x_expr.strip() == expected:
                    logs.append("✓ v_x expression matches expected")
                else:
                    logs.append(f"✗ v_x expression mismatch: got {self.v_x_expr}, expected {expected}")

            if self.v_y_expr is not None and isinstance(self.v_y_expr, str):
                # Expected: "10*t"
                expected = "10*t"
                if self.v_y_expr.strip() == expected:
                    logs.append("✓ v_y expression matches expected")
                else:
                    logs.append(f"✗ v_y expression mismatch: got {self.v_y_expr}, expected {expected}")

            if self.v_expr is not None and isinstance(self.v_expr, str):
                # Expected: "sqrt(106.25*t**2 - 10*t + 4)" or similar
                # Use sympy to simplify and compare
                t = sp.symbols('t')
                expr = sp.sympify(self.v_expr)
                expected_expr = sp.sqrt(106.25*t**2 - 10*t + 4)
                if sp.simplify(expr - expected_expr) == 0:
                    logs.append("✓ v(t) expression matches expected")
                else:
                    logs.append(f"✗ v(t) expression mismatch: got {self.v_expr}")

        except Exception as e:
            logs.append(f"✗ Error in test_trace_consistency: {traceback.format_exc()}")
        return logs

    def test_monotonicity(self):
        logs = []
        try:
            # Perturb mass m by +1% and check trends
            original_m = self.m
            perturbed_m = original_m * 1.01

            # v_A should remain unchanged (independent of m)
            v_A_original = math.sqrt(self.g * self.R)
            v_A_perturbed = math.sqrt(self.g * self.R)
            if math.isclose(v_A_original, v_A_perturbed, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("✓ v_A independent of mass perturbation")

            # Compute x for perturbed mass using work-energy
            numerator = 0.5 * perturbed_m * v_A_perturbed**2 + 2 * perturbed_m * self.g * self.R
            denominator = -self.q * self.E - self.mu * perturbed_m * self.g
            x_perturbed = numerator / denominator

            # Original x
            if self.x is not None:
                # Expect x to increase with m (since denominator decreases slightly)
                if x_perturbed > self.x:
                    logs.append(f"✓ Monotonicity: x increases with m (perturbed x = {x_perturbed:.3f} > original {self.x:.3f})")
                else:
                    logs.append(f"✗ Monotonicity: x does not increase with m (perturbed x = {x_perturbed:.3f} <= original {self.x:.3f})")
            else:
                logs.append("⚠ x not provided, skipping monotonicity check for x")

            # Restore original m (though not strictly needed as we use local variable)
            self.m = original_m

        except Exception as e:
            logs.append(f"✗ Error in test_monotonicity: {traceback.format_exc()}")
            # Ensure restoration
            self.m = self.original_params['m']
        return logs

    def run_all_tests(self):
        all_logs = []
        success = True

        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            logs = test_method()
            if any(log.startswith("✗") for log in logs):
                success = False
            all_logs.append(f"--- {test_name} ---")
            all_logs.extend(logs)

        return success, all_logs