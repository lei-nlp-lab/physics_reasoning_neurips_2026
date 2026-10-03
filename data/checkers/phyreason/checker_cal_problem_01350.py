import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        # Safe extraction of inputs (with defaults from problem statement)
        self.h1 = getattr(solution_instance, 'h1', 1.8)
        self.h2 = getattr(solution_instance, 'h2', 4.0)
        self.x1 = getattr(solution_instance, 'x1', 4.8)
        self.x2 = getattr(solution_instance, 'x2', 8.0)
        self.M = getattr(solution_instance, 'M', 10.0)
        self.m = getattr(solution_instance, 'm', 2.0)
        self.g = getattr(solution_instance, 'g', 10.0)

        # Results dictionary (safe access)
        self.results = getattr(solution_instance, 'results', {})
        self.t = self.results.get('t', 0.0)
        self.v_min = self.results.get('v_min', 0.0)
        self.v_C = self.results.get('v_C', 0.0)
        self.L = self.results.get('L', 0.0)
        self.F_T = self.results.get('F_T', 0.0)
        self.F = self.results.get('F', 0.0)

        # Pre-compute expected values for invariant checks
        self.expected_t = math.sqrt(2 * self.h1 / self.g)                 # step 1
        self.expected_v_min = self.x1 / self.expected_t                   # step 2 & 3
        self.expected_v_C = math.sqrt(2 * self.g * self.h2)               # step 4
        self.expected_L = (self.h2**2 + self.x2**2) / (2 * self.h2)       # step 6
        self.expected_F = (self.M + self.m) * (self.g + self.expected_v_C**2 / self.expected_L)  # step 5 & 7

    def test_invariants(self):
        """
        Verify fundamental physics laws: projectile motion, energy conservation,
        and Newton's second law.
        """
        logs = []
        success = True

        # 1. Projectile motion: time of flight from height h1
        if not math.isclose(self.t, self.expected_t, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Invariant (projectile time): expected t={self.expected_t:.3f}, got {self.t:.3f}")
            success = False

        # 2. Projectile motion: horizontal velocity v_min = x1 / t
        if not math.isclose(self.v_min, self.expected_v_min, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Invariant (horizontal velocity): expected v_min={self.expected_v_min:.3f}, got {self.v_min:.3f}")
            success = False

        # 3. Conservation of mechanical energy: v_C = sqrt(2 g h2)
        if not math.isclose(self.v_C, self.expected_v_C, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Invariant (energy conservation): expected v_C={self.expected_v_C:.3f}, got {self.v_C:.3f}")
            success = False

        # 4. Geometry: vine length L from (L - h2)^2 + x2^2 = L^2
        if not math.isclose(self.L, self.expected_L, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Invariant (geometry): expected L={self.expected_L:.3f}, got {self.L:.3f}")
            success = False

        # 5. Newton's second law (circular motion): tension F_T (or F) at lowest point
        if not math.isclose(self.F, self.expected_F, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Invariant (Newton's second law): expected F={self.expected_F:.3f}, got {self.F:.3f}")
            success = False

        return success, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with the final answer.
        """
        logs = []
        success = True

        # 1. Check step 1 & 2: x1 = v_min * t
        if not math.isclose(self.x1, self.v_min * self.t, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Trace consistency (x1 = v_min * t): {self.x1} vs {self.v_min * self.t}")
            success = False

        # 2. Check step 4: energy conservation (M+m)g h2 = 0.5 (M+m) v_C^2
        energy_lhs = (self.M + self.m) * self.g * self.h2
        energy_rhs = 0.5 * (self.M + self.m) * self.v_C**2
        if not math.isclose(energy_lhs, energy_rhs, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Trace consistency (energy): {energy_lhs} vs {energy_rhs}")
            success = False

        # 3. Check step 5: Newton's second law at lowest point
        if self.L > 0:
            centripetal_force = (self.M + self.m) * self.v_C**2 / self.L
            net_force = self.F_T - (self.M + self.m) * self.g
            if not math.isclose(net_force, centripetal_force, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Trace consistency (circular motion): net force {net_force} vs centripetal {centripetal_force}")
                success = False

        # 4. Check step 6: geometry equation
        lhs = (self.L - self.h2)**2 + self.x2**2
        rhs = self.L**2
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Trace consistency (geometry): {lhs} vs {rhs}")
            success = False

        # 5. Check step 7: final force equals tension
        if not math.isclose(self.F, self.F_T, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Trace consistency (final force): F={self.F} vs F_T={self.F_T}")
            success = False

        return success, logs

    def test_monotonicity(self):
        """
        Perturb mass M and verify that the final force F changes in the expected direction.
        """
        logs = []
        success = True

        # Helper to compute force F from given mass M
        def compute_force(M_val, m_val, g_val, h2_val, x2_val):
            L_val = (h2_val**2 + x2_val**2) / (2 * h2_val)
            v_C_val = math.sqrt(2 * g_val * h2_val)
            return (M_val + m_val) * (g_val + v_C_val**2 / L_val)

        # Original force
        F_orig = compute_force(self.M, self.m, self.g, self.h2, self.x2)

        # Perturb M by +10%
        M_perturbed = self.M * 1.10
        F_perturbed = compute_force(M_perturbed, self.m, self.g, self.h2, self.x2)

        # Expect F to increase because F ∝ (M+m)
        if not (F_perturbed > F_orig):
            logs.append(f"Monotonicity failed: F did not increase when M increased by 10%.")
            logs.append(f"  Original M={self.M:.3f}, F={F_orig:.3f}")
            logs.append(f"  Perturbed M={M_perturbed:.3f}, F={F_perturbed:.3f}")
            success = False
        else:
            logs.append(f"Monotonicity passed: F increased from {F_orig:.3f} to {F_perturbed:.3f}")

        return success, logs

    def run_all_tests(self):
        """
        Execute all tests and return overall success flag and log messages.
        """
        all_logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]

        for test_name, test_func in tests:
            try:
                test_success, logs = test_func()
                all_logs.append(f"--- {test_name} ---")
                all_logs.extend(logs)
                all_logs.append(f"Result: {'PASS' if test_success else 'FAIL'}")
                if not test_success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"Error in {test_name}: {traceback.format_exc()}")
                overall_success = False

        return overall_success, all_logs