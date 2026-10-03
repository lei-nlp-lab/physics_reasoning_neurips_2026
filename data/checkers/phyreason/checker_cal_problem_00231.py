import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Uses getattr to avoid AttributeError in sandbox environments.
        """
        # Input parameters
        self.L = getattr(solution_instance, 'L', None)
        self.d = getattr(solution_instance, 'd', None)
        self.v0 = getattr(solution_instance, 'v0', None)
        self.m = getattr(solution_instance, 'm', None)
        self.q = getattr(solution_instance, 'q', None)
        self.k = getattr(solution_instance, 'k', None)   # drag constant
        self.R = getattr(solution_instance, 'R', None)   # original particle radius

        # Results dictionary (if provided)
        self.results = getattr(solution_instance, 'results', {})

        # Final results (with fallback to results dict)
        self.U1 = getattr(solution_instance, 'U_1', self.results.get('U_1', None))
        self.U2 = getattr(solution_instance, 'U_2', self.results.get('U_2', None))
        self.percentage = getattr(solution_instance, 'percentage', self.results.get('percentage', None))

        # Intermediate results (if available)
        self.t = getattr(solution_instance, 't', self.results.get('t', None))
        self.a = getattr(solution_instance, 'a', self.results.get('a', None))
        self.E = getattr(solution_instance, 'E', self.results.get('E', None))
        self.v_max = getattr(solution_instance, 'v_max', self.results.get('v_max', None))
        self.f = getattr(solution_instance, 'f', self.results.get('f', None))
        self.q_prime = getattr(solution_instance, 'q_prime', self.results.get('q\'', None))  # charge of 2.5 μm particle
        self.d_prime = getattr(solution_instance, 'd_prime', self.results.get('d\'', None))  # vertical displacement for 2.5 μm
        self.f_prime = getattr(solution_instance, 'f_prime', self.results.get('f\'', None))  # drag force on 2.5 μm particle

    def test_invariants(self):
        """Verify physical laws and derived formulas."""
        logs = []

        # 1. Check U1 formula from kinematics and Newton's second law
        if all(v is not None for v in [self.L, self.d, self.v0, self.m, self.q, self.U1]):
            try:
                expected_U1 = 2 * self.d**2 * self.m * self.v0**2 / (self.q * self.L**2)
                if math.isclose(self.U1, expected_U1, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("PASS: U1 satisfies kinematics and Newton's second law.")
                else:
                    logs.append(f"FAIL: U1 mismatch. Expected {expected_U1:.4g}, got {self.U1:.4g}.")
            except Exception as e:
                logs.append(f"ERROR in U1 invariant check: {e}")
        else:
            logs.append("SKIP: Missing parameters for U1 check.")

        # 2. Check U2 formula from drag force balance
        if all(v is not None for v in [self.k, self.R, self.d, self.q, self.L, self.v0, self.U2]):
            try:
                expected_U2 = self.k * self.R * self.d**2 * self.v0 / (self.q * self.L)
                if math.isclose(self.U2, expected_U2, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("PASS: U2 satisfies drag force balance.")
                else:
                    logs.append(f"FAIL: U2 mismatch. Expected {expected_U2:.4g}, got {self.U2:.4g}.")
            except Exception as e:
                logs.append(f"ERROR in U2 invariant check: {e}")
        else:
            logs.append("SKIP: Missing parameters for U2 check.")

        # 3. Check percentage = 25%
        if self.percentage is not None:
            if math.isclose(self.percentage, 25.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("PASS: Percentage is 25% as derived.")
            else:
                logs.append(f"FAIL: Percentage is {self.percentage:.4g}%, expected 25%.")
        else:
            logs.append("SKIP: Missing percentage value.")

        return logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []

        # 1. U1 consistency: t = L/v0, a from d = 0.5 a t^2, E from a = qE/m, U1 = E*d
        if all(v is not None for v in [self.L, self.v0, self.d, self.m, self.q, self.U1]):
            try:
                t = self.L / self.v0
                a = 2 * self.d / t**2
                E = a * self.m / self.q
                U1_calc = E * self.d
                if math.isclose(self.U1, U1_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("PASS: U1 consistent with intermediate kinematics and electric field.")
                else:
                    logs.append(f"FAIL: U1 trace mismatch. Computed {U1_calc:.4g}, got {self.U1:.4g}.")
            except Exception as e:
                logs.append(f"ERROR in U1 trace check: {e}")
        else:
            logs.append("SKIP: Missing parameters for U1 trace.")

        # 2. U2 consistency: t = L/v0, v_max = d/t, f_drag = k R v_max, U2 = f_drag * d / q
        if all(v is not None for v in [self.L, self.v0, self.d, self.k, self.R, self.q, self.U2]):
            try:
                t = self.L / self.v0
                v_max = self.d / t
                f_drag = self.k * self.R * v_max
                U2_calc = f_drag * self.d / self.q
                if math.isclose(self.U2, U2_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("PASS: U2 consistent with intermediate drag force and velocity.")
                else:
                    logs.append(f"FAIL: U2 trace mismatch. Computed {U2_calc:.4g}, got {self.U2:.4g}.")
            except Exception as e:
                logs.append(f"ERROR in U2 trace check: {e}")
        else:
            logs.append("SKIP: Missing parameters for U2 trace.")

        # 3. d' = d/4 consistency
        if self.d_prime is not None and self.d is not None:
            try:
                if math.isclose(self.d_prime, self.d / 4, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("PASS: d' = d/4 as required.")
                else:
                    logs.append(f"FAIL: d' mismatch. Expected {self.d/4:.4g}, got {self.d_prime:.4g}.")
            except Exception as e:
                logs.append(f"ERROR in d' trace check: {e}")
        else:
            logs.append("SKIP: Missing d or d' for trace check.")

        return logs

    def test_monotonicity(self):
        """Perturb inputs and verify expected trends."""
        logs = []

        # 1. U1 ∝ m, v0^2; inversely ∝ q, L^2
        if all(v is not None for v in [self.L, self.d, self.v0, self.m, self.q, self.U1]):
            try:
                m_new = self.m * 1.1  # +10%
                U1_new = 2 * self.d**2 * m_new * self.v0**2 / (self.q * self.L**2)
                if math.isclose(U1_new, self.U1 * 1.1, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("PASS: U1 increases proportionally with mass (monotonicity).")
                else:
                    logs.append("FAIL: U1 does not show expected monotonicity with mass.")
            except Exception as e:
                logs.append(f"ERROR in U1 monotonicity check: {e}")
        else:
            logs.append("SKIP: Missing parameters for U1 monotonicity.")

        # 2. U2 ∝ k, R, d^2, v0; inversely ∝ q, L
        if all(v is not None for v in [self.k, self.R, self.d, self.v0, self.q, self.L, self.U2]):
            try:
                k_new = self.k * 1.1  # +10%
                U2_new = k_new * self.R * self.d**2 * self.v0 / (self.q * self.L)
                if math.isclose(U2_new, self.U2 * 1.1, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("PASS: U2 increases proportionally with drag constant k (monotonicity).")
                else:
                    logs.append("FAIL: U2 does not show expected monotonicity with k.")
            except Exception as e:
                logs.append(f"ERROR in U2 monotonicity check: {e}")
        else:
            logs.append("SKIP: Missing parameters for U2 monotonicity.")

        # 3. Percentage remains 25% under parameter changes (algebraic invariance)
        if all(v is not None for v in [self.L, self.d, self.v0, self.k, self.R, self.q]):
            try:
                v0_new = self.v0 * 1.1  # perturb v0
                # Recompute U2 for new v0 (assuming U2 is set accordingly)
                U2_new = self.k * self.R * self.d**2 * v0_new / (self.q * self.L)
                # For 2.5 μm particle: q' = q/16, radius = R/4 (from step12)
                q_prime = self.q / 16
                f_prime = q_prime * U2_new / self.d
                v_max_prime = (4 * f_prime) / (self.k * self.R)  # because f' = k*(R/4)*v_max'
                t_new = self.L / v0_new
                d_prime_new = v_max_prime * t_new
                percentage_new = (d_prime_new / self.d) * 100
                if math.isclose(percentage_new, 25.0, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("PASS: Percentage remains 25% when v0 is perturbed.")
                else:
                    logs.append(f"FAIL: Percentage changes to {percentage_new:.4g}% when v0 is increased.")
            except Exception as e:
                logs.append(f"ERROR in percentage monotonicity check: {e}")
        else:
            logs.append("SKIP: Missing parameters for percentage monotonicity.")

        return logs

    def run_all_tests(self):
        """Execute all checks and return success flag and logs."""
        all_logs = ["=== PhysicsChecker Start ==="]
        success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            try:
                logs = test_func()
                all_logs.append(f"--- {test_name} ---")
                all_logs.extend(logs)
                # Check for any failures in this test
                for log in logs:
                    if log.startswith("FAIL") or log.startswith("ERROR"):
                        success = False
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {traceback.format_exc()}")
                success = False

        all_logs.append(f"=== PhysicsChecker End -> Success: {success} ===")
        return success, all_logs