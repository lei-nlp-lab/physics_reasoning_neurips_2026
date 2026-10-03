import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs and results using getattr and .get()
        self.v_A = 5.0                     # given constant velocity of A
        self.Deltax = 8.36                 # initial separation (B behind A)

        # Dictionary to hold results from solution_instance
        self.results = {}
        # List of expected keys (from the problem's key mapping)
        expected_keys = [
            'Deltax_m', 't\'', 'x_3', 'x_A', 'x_A\'', 'x_A1', 'Deltat', 't_B',
            'Deltax\'', 'x_n', 'x_2', 'x_B1', 't_1', 'x_B\'', 'x_B', 'x_1'
        ]
        
        # Try to get results from solution_instance.results dictionary first
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.results = solution_instance.results
        else:
            # Fallback: try to get each key as an attribute
            for key in expected_keys:
                self.results[key] = getattr(solution_instance, key, None)

        # Extract specific quantities with safe defaults
        self.x1 = self.results.get('x_1', 0.0)
        self.x2 = self.results.get('x_2', 0.0)
        self.x3 = self.results.get('x_3', 0.0)
        self.xn = self.results.get('x_n', 0.0)
        self.xA = self.results.get('x_A', 0.0)        # displacement of A at t1
        self.xB = self.results.get('x_B', 0.0)        # displacement of B at t1
        self.Deltax_m = self.results.get('Deltax_m', 0.0)
        self.t1 = self.results.get('t_1', 0.0)        # time when distance is maximized
        self.Deltat = self.results.get('Deltat', 0.0) # time for B to reach v_A
        self.tB = self.results.get('t_B', 0.0)        # total catch-up time
        self.xA1 = self.results.get('x_A1', 0.0)      # displacement of A at 10 s
        self.xB1 = self.results.get('x_B1', 0.0)      # displacement of B at 10 s
        self.Deltax_prime = self.results.get('Deltax\'', 0.0) # gap after 10 s
        self.t_prime = self.results.get('t\'', 0.0)   # additional time after 10 s

        # Acceleration inferred from step_1 and step_3
        self.a = 2.0  # m/s²

    def test_invariants(self):
        logs = []
        success = True

        # 1. Check displacement in first second: x1 = 0.5 * a * (1 s)^2
        if self.x1 is not None:
            expected_x1 = 0.5 * self.a * 1.0**2
            if not math.isclose(self.x1, expected_x1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: x1 = {self.x1}, expected {expected_x1}")
                success = False
            else:
                logs.append(f"OK: x1 = {self.x1}")

        # 2. Check Δt = v_A / a
        if self.Deltat is not None:
            expected_Deltat = self.v_A / self.a
            if not math.isclose(self.Deltat, expected_Deltat, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Δt = {self.Deltat}, expected {expected_Deltat}")
                success = False
            else:
                logs.append(f"OK: Δt = {self.Deltat}")

        # 3. Check displacement of A at t1: xA = v_A * t1
        if self.t1 is not None and self.xA is not None:
            expected_xA = self.v_A * self.t1
            if not math.isclose(self.xA, expected_xA, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: xA = {self.xA}, expected {expected_xA}")
                success = False
            else:
                logs.append(f"OK: xA = {self.xA}")

        # 4. Check displacement of B at t1 using step_9 formula
        if self.xB is not None:
            # Sum of displacements for first 4 seconds: 1+2+3+4 = 10 m
            sum_first4 = 10.0
            # Extra displacement in the following 0.5 s: 2*a*ΔT*0.5 + 0.5*a*(0.5)^2
            extra = 2 * self.a * 1.0 * 0.5 + 0.5 * self.a * (0.5**2)
            expected_xB = sum_first4 + extra
            if not math.isclose(self.xB, expected_xB, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: xB = {self.xB}, expected {expected_xB}")
                success = False
            else:
                logs.append(f"OK: xB = {self.xB}")

        # 5. Check maximum distance: Δx_m = Δx + xA - xB
        if all(v is not None for v in [self.Deltax_m, self.xA, self.xB]):
            expected_Deltax_m = self.Deltax + self.xA - self.xB
            if not math.isclose(self.Deltax_m, expected_Deltax_m, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Δx_m = {self.Deltax_m}, expected {expected_Deltax_m}")
                success = False
            else:
                logs.append(f"OK: Δx_m = {self.Deltax_m}")

        # 6. Check displacements at 10 seconds
        if self.xA1 is not None:
            expected_xA1 = self.v_A * 10.0
            if not math.isclose(self.xA1, expected_xA1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: xA1 = {self.xA1}, expected {expected_xA1}")
                success = False
            else:
                logs.append(f"OK: xA1 = {self.xA1}")

        if self.xB1 is not None:
            expected_xB1 = 10.0 * 11.0 / 2.0  # sum of 1 to 10
            if not math.isclose(self.xB1, expected_xB1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: xB1 = {self.xB1}, expected {expected_xB1}")
                success = False
            else:
                logs.append(f"OK: xB1 = {self.xB1}")

        # 7. Check gap after 10 s: Δx' = Δx + xA1 - xB1
        if all(v is not None for v in [self.Deltax_prime, self.xA1, self.xB1]):
            expected_Deltax_prime = self.Deltax + self.xA1 - self.xB1
            if not math.isclose(self.Deltax_prime, expected_Deltax_prime, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Δx' = {self.Deltax_prime}, expected {expected_Deltax_prime}")
                success = False
            else:
                logs.append(f"OK: Δx' = {self.Deltax_prime}")

        # 8. Check t' satisfies: t'^2 + 5 t' = Δx'  (from step_18)
        if self.t_prime is not None and self.Deltax_prime is not None:
            lhs = self.t_prime**2 + 5.0 * self.t_prime
            if not math.isclose(lhs, self.Deltax_prime, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: t' equation: t'^2 + 5 t' = {lhs}, Δx' = {self.Deltax_prime}")
                success = False
            else:
                logs.append(f"OK: t' satisfies equation")

        # 9. Check total catch‑up time: t_B = 10 + t'
        if self.tB is not None and self.t_prime is not None:
            expected_tB = 10.0 + self.t_prime
            if not math.isclose(self.tB, expected_tB, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: t_B = {self.tB}, expected {expected_tB}")
                success = False
            else:
                logs.append(f"OK: t_B = {self.tB}")

        return success, logs

    def test_trace_consistency(self):
        logs = []
        success = True

        # Cross‑check that intermediate results lead to final answers
        # 1. Maximum distance
        if all(v is not None for v in [self.t1, self.xA, self.xB, self.Deltax_m]):
            computed_Deltax_m = self.Deltax + self.xA - self.xB
            if not math.isclose(self.Deltax_m, computed_Deltax_m, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Trace mismatch: Δx_m = {self.Deltax_m}, recomputed = {computed_Deltax_m}")
                success = False
            else:
                logs.append(f"Trace OK for Δx_m.")

        # 2. Catch‑up part
        if all(v is not None for v in [self.xA1, self.xB1, self.Deltax_prime, self.t_prime, self.tB]):
            # Δx' consistency
            computed_Deltax_prime = self.Deltax + self.xA1 - self.xB1
            if not math.isclose(self.Deltax_prime, computed_Deltax_prime, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Trace mismatch: Δx' = {self.Deltax_prime}, recomputed = {computed_Deltax_prime}")
                success = False
            else:
                logs.append(f"Trace OK for Δx'.")

            # t' from quadratic equation
            D = 25.0 + 4.0 * self.Deltax_prime
            if D < 0:
                logs.append(f"Trace error: negative discriminant for t' equation.")
                success = False
            else:
                t_prime_sol = (-5.0 + math.sqrt(D)) / 2.0
                if not math.isclose(self.t_prime, t_prime_sol, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Trace mismatch: t' = {self.t_prime}, quadratic root = {t_prime_sol}")
                    success = False
                else:
                    logs.append(f"Trace OK for t'.")

            # t_B consistency
            computed_tB = 10.0 + self.t_prime
            if not math.isclose(self.tB, computed_tB, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Trace mismatch: t_B = {self.tB}, recomputed = {computed_tB}")
                success = False
            else:
                logs.append(f"Trace OK for t_B.")

        return success, logs

    def test_monotonicity(self):
        logs = []
        success = True

        # Perturb initial separation Δx by +1%
        delta = self.Deltax * 0.01
        perturbed_Deltax = self.Deltax + delta

        # Assume other quantities (t1, xA, xB, xA1, xB1) remain unchanged
        # because they are determined by kinematics independent of Δx.

        # 1. Maximum distance
        if self.xA is not None and self.xB is not None:
            new_Deltax_m = perturbed_Deltax + self.xA - self.xB
            if not new_Deltax_m > self.Deltax_m:
                logs.append(f"Monotonicity failed: Δx ↗ but Δx_m did not ↗")
                success = False
            else:
                logs.append(f"Monotonicity OK: Δx_m increases with Δx.")

        # 2. Gap after 10 s
        if self.xA1 is not None and self.xB1 is not None:
            new_Deltax_prime = perturbed_Deltax + self.xA1 - self.xB1
            if not new_Deltax_prime > self.Deltax_prime:
                logs.append(f"Monotonicity failed: Δx ↗ but Δx' did not ↗")
                success = False
            else:
                logs.append(f"Monotonicity OK: Δx' increases with Δx.")

            # 3. Additional time t'
            D_new = 25.0 + 4.0 * new_Deltax_prime
            new_t_prime = (-5.0 + math.sqrt(D_new)) / 2.0
            if not new_t_prime > self.t_prime:
                logs.append(f"Monotonicity failed: Δx ↗ but t' did not ↗")
                success = False
            else:
                logs.append(f"Monotonicity OK: t' increases with Δx.")

            # 4. Total catch‑up time t_B
            new_t_B = 10.0 + new_t_prime
            if not new_t_B > self.tB:
                logs.append(f"Monotonicity failed: Δx ↗ but t_B did not ↗")
                success = False
            else:
                logs.append(f"Monotonicity OK: t_B increases with Δx.")

        return success, logs

    def run_all_tests(self):
        all_logs = []
        overall_success = True

        tests = [
            ("test_invariants", self.test_invariants),
            ("test_trace_consistency", self.test_trace_consistency),
            ("test_monotonicity", self.test_monotonicity)
        ]

        for name, test_func in tests:
            try:
                all_logs.append(f"--- Running {name} ---")
                success, logs = test_func()
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR in {name}: {traceback.format_exc()}")
                overall_success = False

        return overall_success, all_logs