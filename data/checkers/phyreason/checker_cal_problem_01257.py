import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Robust initialization with safe attribute access.
        Inputs and results are extracted from the solution instance.
        """
        # Extract inputs safely
        self.m1 = getattr(solution_instance, 'm1', 0.1)
        self.m2 = getattr(solution_instance, 'm2', 0.4)
        self.d = getattr(solution_instance, 'd', 0.15)
        self.mu = getattr(solution_instance, 'mu', 0.2)
        self.g = getattr(solution_instance, 'g', 10.0)

        # Extract results safely from results dictionary
        results = getattr(solution_instance, 'results', {})
        self.F_m = results.get('F_m', 0.0)
        self.v_1 = results.get('v_1', 0.0)
        self.a_m = results.get('a_m', 0.0)
        self.a_2 = results.get('a_2', 0.0)
        self.f_2 = results.get('f_2', 0.0)
        self.x_2 = results.get('x_2', 0.0)
        self.f_1 = results.get('f_1', 0.0)
        self.s = results.get('s', 0.0)
        self.x_1 = results.get('x_1', 0.0)

        # For symbolic checks
        self.symbols = {}
        self.equations = []

    def is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical matching."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """
        Verify physical laws based on KB and steps analysis.
        Returns: (bool, log_message)
        """
        logs = []
        success = True

        # Step 1: f_1 = μ*(m1 + m2)*g
        f1_expected = self.mu * (self.m1 + self.m2) * self.g
        if not self.is_close(self.f_1, f1_expected):
            logs.append(f"Step 1 invariant failed: f_1 = {self.f_1}, expected {f1_expected}")
            success = False
        else:
            logs.append(f"Step 1 OK: f_1 = {self.f_1} N")

        # Step 2: f_2 = μ*m2*g
        f2_expected = self.mu * self.m2 * self.g
        if not self.is_close(self.f_2, f2_expected):
            logs.append(f"Step 2 invariant failed: f_2 = {self.f_2}, expected {f2_expected}")
            success = False
        else:
            logs.append(f"Step 2 OK: f_2 = {self.f_2} N")

        # Step 3: a_m = f_2 / m2 = μ*g
        a_m_expected = self.mu * self.g
        if not self.is_close(self.a_m, a_m_expected):
            logs.append(f"Step 3 invariant failed: a_m = {self.a_m}, expected {a_m_expected}")
            success = False
        else:
            logs.append(f"Step 3 OK: a_m = {self.a_m} m/s²")

        # Step 4: Newton's 2nd Law for system (cardboard + object) when moving together
        # F_m - f_1 = (m1 + m2) * a_m
        net_force = self.F_m - self.f_1
        lhs = net_force
        rhs = (self.m1 + self.m2) * self.a_m
        if not self.is_close(lhs, rhs):
            logs.append(f"Step 4 invariant failed: F_m - f_1 = {lhs}, (m1+m2)*a_m = {rhs}")
            success = False
        else:
            logs.append(f"Step 4 OK: Newton's 2nd Law holds for system")

        # Step 5: F_m = 2*μ*g*(m1 + m2)
        F_m_expected = 2 * self.mu * self.g * (self.m1 + self.m2)
        if not self.is_close(self.F_m, F_m_expected):
            logs.append(f"Step 5 invariant failed: F_m = {self.F_m}, expected {F_m_expected}")
            success = False
        else:
            logs.append(f"Step 5 OK: F_m = {self.F_m} N")

        # Step 6: a_2 = μ*g (acceleration of object due to kinetic friction)
        a2_expected = self.mu * self.g
        if not self.is_close(self.a_2, a2_expected):
            logs.append(f"Step 6 invariant failed: a_2 = {self.a_2}, expected {a2_expected}")
            success = False
        else:
            logs.append(f"Step 6 OK: a_2 = {self.a_2} m/s²")

        # Step 7 & 8: Kinematics for uniformly accelerated motion
        # From v_1 = a_2 * t and x_1 = 0.5 * a_2 * t^2, we get v_1^2 = 2 * a_2 * x_1
        v1_sq_from_kinematics = 2 * self.a_2 * self.x_1
        if not self.is_close(self.v_1**2, v1_sq_from_kinematics):
            logs.append(f"Step 7-8 kinematics invariant failed: v_1^2 = {self.v_1**2}, 2*a_2*x_1 = {v1_sq_from_kinematics}")
            success = False
        else:
            logs.append(f"Step 7-8 OK: kinematics consistent")

        # Step 9: Kinematics for uniformly decelerated motion
        # x_2 = v_1^2 / (2 * a_2)
        x2_expected = self.v_1**2 / (2 * self.a_2)
        if not self.is_close(self.x_2, x2_expected):
            logs.append(f"Step 9 invariant failed: x_2 = {self.x_2}, expected v_1^2/(2a_2) = {x2_expected}")
            success = False
        else:
            logs.append(f"Step 9 OK: x_2 = {self.x_2} m")

        # Step 10: Total distance s = x_1 + x_2
        s_expected = self.x_1 + self.x_2
        if not self.is_close(self.s, s_expected):
            logs.append(f"Step 10 invariant failed: s = {self.s}, expected x_1+x_2 = {s_expected}")
            success = False
        else:
            logs.append(f"Step 10 OK: s = {self.s} m")

        # Energy check: Work done by friction on object equals change in kinetic energy
        # Initial KE = 0, final KE = 0.5 * m2 * v_1^2 (at moment of leaving cardboard)
        # Work by friction on object during sliding: W = f_2 * x_1 (positive, since friction accelerates object)
        # This should equal KE gain: 0.5 * m2 * v_1^2
        work_friction = self.f_2 * self.x_1
        ke_gain = 0.5 * self.m2 * self.v_1**2
        if not self.is_close(work_friction, ke_gain):
            logs.append(f"Energy-work invariant warning: f_2*x_1 = {work_friction}, KE gain = {ke_gain} (within tolerance?)")

        # Check that object falls off: total distance s > table edge distance d
        if self.s <= self.d:
            logs.append(f"Fall-off condition invariant failed: s = {self.s} m ≤ d = {self.d} m, object should remain on table")
            success = False
        else:
            logs.append(f"Fall-off condition OK: s = {self.s} m > d = {self.d} m, object will fall")

        return success, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answer and inputs.
        Returns: (bool, log_message)
        """
        logs = []
        success = True

        # Check that a_m equals a_2 (both should be μ*g)
        if not self.is_close(self.a_m, self.a_2):
            logs.append(f"Trace consistency failed: a_m = {self.a_m}, a_2 = {self.a_2} (should both be μ*g)")
            success = False
        else:
            logs.append(f"Trace consistency OK: a_m = a_2 = {self.a_m} m/s²")

        # Check that F_m from step 5 matches the condition F > F_m for slipping
        # The threshold force is F_m, so for slipping we need F > F_m.
        # We can't check F directly, but we can verify that F_m is computed correctly.
        logs.append(f"Threshold force F_m = {self.F_m} N, condition for slipping is F > {self.F_m} N")

        # Check that total distance s matches target answer 0.18 m
        if not self.is_close(self.s, 0.18):
            logs.append(f"Trace consistency failed: s = {self.s} m, target answer expects 0.18 m")
            success = False
        else:
            logs.append(f"Trace consistency OK: s = {self.s} m matches target")

        # Check that f_1 matches target answer 1 N
        if not self.is_close(self.f_1, 1.0):
            logs.append(f"Trace consistency failed: f_1 = {self.f_1} N, target answer expects 1 N")
            success = False
        else:
            logs.append(f"Trace consistency OK: f_1 = {self.f_1} N matches target")

        return success, logs

    def test_monotonicity(self):
        """
        Perturb an input (mass m2) and verify that output changes sensibly.
        Returns: (bool, log_message)
        """
        logs = []
        success = True

        try:
            # Perturb m2 by +10%
            perturbed_m2 = self.m2 * 1.10
            original_m2 = self.m2

            # Recalculate key quantities using formulas from steps
            # Step 1: f_1 = μ*(m1 + m2)*g
            f1_perturbed = self.mu * (self.m1 + perturbed_m2) * self.g
            # Step 2: f_2 = μ*m2*g
            f2_perturbed = self.mu * perturbed_m2 * self.g
            # Step 3: a_m = μ*g (unchanged)
            a_m_perturbed = self.mu * self.g
            # Step 5: F_m = 2*μ*g*(m1 + m2)
            F_m_perturbed = 2 * self.mu * self.g * (self.m1 + perturbed_m2)
            # Step 6: a_2 = μ*g (unchanged)
            a2_perturbed = self.mu * self.g

            # For kinematics, assume same time t? Not directly available.
            # Instead, we can check that the threshold force F_m increases with m2.
            if F_m_perturbed <= self.F_m:
                logs.append(f"Monotonicity failed: F_m does not increase when m2 increases")
                success = False
            else:
                logs.append(f"Monotonicity OK: F_m increases with m2 (from {self.F_m} N to {F_m_perturbed} N)")

            # Also check that f_1 increases with m2
            if f1_perturbed <= self.f_1:
                logs.append(f"Monotonicity warning: f_1 does not increase when m2 increases")
            else:
                logs.append(f"Monotonicity OK: f_1 increases with m2")

            # Restore original m2 (conceptually)
            logs.append("Monotonicity test completed with perturbation restored.")

        except Exception as e:
            logs.append(f"Monotonicity test error: {str(e)}")
            success = False

        return success, logs

    def run_all_tests(self):
        """
        Run all audit tests and return overall success and logs.
        """
        all_logs = ["=== Physics Auditor Report ==="]
        overall_success = True

        # Test invariants
        success, logs = self.test_invariants()
        all_logs.append("\n--- Invariants Test ---")
        all_logs.extend(logs)
        if not success:
            overall_success = False

        # Test trace consistency
        success, logs = self.test_trace_consistency()
        all_logs.append("\n--- Trace Consistency Test ---")
        all_logs.extend(logs)
        if not success:
            overall_success = False

        # Test monotonicity
        success, logs = self.test_monotonicity()
        all_logs.append("\n--- Monotonicity Test ---")
        all_logs.extend(logs)
        if not success:
            overall_success = False

        all_logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, all_logs