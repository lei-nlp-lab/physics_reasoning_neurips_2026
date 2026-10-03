import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs (with defaults)
        self.m = 2.0
        self.F = 20.0
        self.g = 10.0
        self.x = 10.0           # distance in first phase
        self.t1 = 2.0           # time for first phase (to compute a)
        self.t2 = 3.0           # time for step_2 (to compute v) – from steps analysis
        self.sin_theta = 0.6
        self.cos_theta = 0.8

        # Safe extraction of results from solution_instance
        self.a = getattr(solution_instance, 'a', solution_instance.results.get('a', 0.0))
        self.v = getattr(solution_instance, 'v', solution_instance.results.get('v', 0.0))
        self.mu = getattr(solution_instance, 'mu', solution_instance.results.get('mu', 0.0))
        self.a_prime = getattr(solution_instance, 'a_prime', solution_instance.results.get('a\'', 0.0))  # note: key is 'a'' in mapping
        self.x1 = getattr(solution_instance, 'x1', solution_instance.results.get('x₁', 0.0))

        # For symbolic checks
        self.symbols = sp.symbols('m F g x t1 t2 sin_theta cos_theta a v mu a_prime x1')
        self.equations = {}

    def _isclose(self, a, b):
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

    def test_invariants(self):
        logs = []
        try:
            # 1. Check kinematic relation for first phase: a = 2*x / t1^2
            expected_a = 2 * self.x / (self.t1 ** 2)
            if not self._isclose(self.a, expected_a):
                logs.append(f"Invariant (step_1) failed: a={self.a}, expected {expected_a}")
            else:
                logs.append("Step 1 (a = 2x/t1^2) consistent.")

            # 2. Check kinematic relation for step_2: v = a * t2
            expected_v = self.a * self.t2
            if not self._isclose(self.v, expected_v):
                logs.append(f"Invariant (step_2) failed: v={self.v}, expected {expected_v}")
            else:
                logs.append("Step 2 (v = a*t2) consistent.")

            # 3. Check Newton's second law during first phase:
            #    F*cosθ - μ*(m*g - F*sinθ) = m*a
            lhs = self.F * self.cos_theta - self.mu * (self.m * self.g - self.F * self.sin_theta)
            rhs = self.m * self.a
            if not self._isclose(lhs, rhs):
                logs.append(f"Newton's 2nd law (first phase) failed: lhs={lhs}, rhs={rhs}")
            else:
                logs.append("Newton's 2nd law (first phase) consistent.")

            # 4. Check deceleration after force removal: a' = -μ*g
            expected_a_prime = -self.mu * self.g
            if not self._isclose(self.a_prime, expected_a_prime):
                logs.append(f"Deceleration invariant failed: a'={self.a_prime}, expected {expected_a_prime}")
            else:
                logs.append("Step 4 (a' = -μ*g) consistent.")

            # 5. Check kinematic relation for stopping distance: 0 - v^2 = 2*a'*x1
            #    => x1 = -v^2 / (2*a')
            if abs(self.a_prime) > 1e-12:
                expected_x1 = -self.v**2 / (2 * self.a_prime)
                if not self._isclose(self.x1, expected_x1):
                    logs.append(f"Stopping distance invariant failed: x1={self.x1}, expected {expected_x1}")
                else:
                    logs.append("Step 6 (x1 = -v^2/(2a')) consistent.")
            else:
                logs.append("Warning: a' is zero, cannot compute expected x1.")

        except Exception as e:
            logs.append(f"Error in test_invariants: {e}\n{traceback.format_exc()}")

        return logs

    def test_trace_consistency(self):
        logs = []
        try:
            # Use symbolic equations to check consistency across steps
            m, F, g, x, t1, t2, sin_theta, cos_theta, a, v, mu, a_prime, x1 = self.symbols

            eq1 = sp.Eq(a, 2*x / t1**2)                     # step_1
            eq2 = sp.Eq(v, a * t2)                          # step_2
            eq3 = sp.Eq(F*cos_theta - mu*(m*g - F*sin_theta), m*a)  # step_3
            eq4 = sp.Eq(a_prime, -mu * g)                    # step_4
            eq5 = sp.Eq(x1, -v**2 / (2*a_prime))             # step_6

            # Substitute known numeric values
            subs_dict = {
                m: self.m, F: self.F, g: self.g,
                x: self.x, t1: self.t1, t2: self.t2,
                sin_theta: self.sin_theta, cos_theta: self.cos_theta,
                a: self.a, v: self.v, mu: self.mu,
                a_prime: self.a_prime, x1: self.x1
            }

            for eq, desc in [(eq1, "step_1"), (eq2, "step_2"), (eq3, "step_3"), (eq4, "step_4"), (eq5, "step_6")]:
                lhs_val = eq.lhs.subs(subs_dict)
                rhs_val = eq.rhs.subs(subs_dict)
                if not self._isclose(float(lhs_val), float(rhs_val)):
                    logs.append(f"Trace consistency failed for {desc}: lhs={float(lhs_val):.3f}, rhs={float(rhs_val):.3f}")
                else:
                    logs.append(f"Trace consistency passed for {desc}.")

        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {e}\n{traceback.format_exc()}")

        return logs

    def test_monotonicity(self):
        logs = []
        original_m = self.m
        try:
            # Perturb mass by +5%
            perturbed_m = original_m * 1.05

            # Recompute expected values using the steps (formulas from the problem)
            # Step 1: a = 2*x / t1^2  (independent of m)
            a_perturbed = 2 * self.x / (self.t1 ** 2)

            # Step 2: v = a * t2  (independent of m)
            v_perturbed = a_perturbed * self.t2

            # Step 3: μ = (F*cosθ - m*a) / (m*g - F*sinθ)
            numerator = self.F * self.cos_theta - perturbed_m * a_perturbed
            denominator = perturbed_m * self.g - self.F * self.sin_theta
            mu_perturbed = numerator / denominator if denominator != 0 else 0.0

            # Step 4: a' = -μ*g
            a_prime_perturbed = -mu_perturbed * self.g

            # Step 6: x1 = -v^2 / (2*a')
            if a_prime_perturbed != 0:
                x1_perturbed = -v_perturbed**2 / (2 * a_prime_perturbed)
            else:
                x1_perturbed = 0.0

            # Check trends:
            # If m increases, μ should decrease, a' should increase (become less negative), x1 should increase.
            if mu_perturbed > self.mu:
                logs.append(f"Monotonicity warning: μ increased (from {self.mu:.3f} to {mu_perturbed:.3f}) with increased mass.")
            else:
                logs.append(f"Monotonicity: μ decreased as expected (from {self.mu:.3f} to {mu_perturbed:.3f}).")

            if a_prime_perturbed > self.a_prime:  # less negative means larger value
                logs.append(f"Monotonicity: a' increased (became less negative) as expected (from {self.a_prime:.3f} to {a_prime_perturbed:.3f}).")
            else:
                logs.append(f"Monotonicity warning: a' did not increase (from {self.a_prime:.3f} to {a_prime_perturbed:.3f}).")

            if x1_perturbed > self.x1:
                logs.append(f"Monotonicity: x1 increased as expected (from {self.x1:.3f} to {x1_perturbed:.3f}).")
            else:
                logs.append(f"Monotonicity warning: x1 did not increase (from {self.x1:.3f} to {x1_perturbed:.3f}).")

        except Exception as e:
            logs.append(f"Error in test_monotonicity: {e}\n{traceback.format_exc()}")
        finally:
            # Restore original mass (though it's not stored in the instance)
            self.m = original_m

        return logs

    def run_all_tests(self):
        all_logs = []
        success = True

        # Run each test and collect logs
        for test_name, test_method in [("Invariants", self.test_invariants),
                                       ("Trace Consistency", self.test_trace_consistency),
                                       ("Monotonicity", self.test_monotonicity)]:
            all_logs.append(f"=== {test_name} ===")
            logs = test_method()
            all_logs.extend(logs)
            # Check for any errors or failures in logs
            if any("failed" in log.lower() or "error" in log.lower() for log in logs):
                success = False

        return success, all_logs