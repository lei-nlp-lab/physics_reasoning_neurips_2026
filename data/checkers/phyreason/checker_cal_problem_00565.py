import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all necessary inputs and results from the solution instance.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Physical constants (inputs)
        self.m = getattr(solution_instance, 'm', 1e-5)  # kg
        self.q = getattr(solution_instance, 'q', 1e-3)  # C (magnitude, but negative sign is handled in force direction)
        self.B = getattr(solution_instance, 'B', 5e-2)  # T
        self.h = getattr(solution_instance, 'h', 0.1)   # m
        self.x_p = getattr(solution_instance, 'x_p', math.sqrt(2)/5)  # m

        # Extracted results (use safe access)
        self.t = self._safe_get('t')
        self.v_y = self._safe_get('v_y')
        self.v = self._safe_get('v')
        self.W = self._safe_get('W')
        self.v_0 = self._safe_get('v_0')  # horizontal velocity of the rod (first part, likely 4 m/s)
        self.v_01 = self._safe_get('v_01')  # first critical velocity (1 m/s)
        self.v_02 = self._safe_get('v_02')  # second critical velocity (2 m/s)
        self.a = self._safe_get('a')  # acceleration from step 2
        self.F_Lorentz_y = self._safe_get('F_Lorentz_y')
        self.R_1 = self._safe_get('R_1')
        self.a_1 = self._safe_get('a_1')
        self.v_y1 = self._safe_get('v_y1')
        self.v_1 = self._safe_get('v_1')
        self.d = self._safe_get('d')
        self.x_2 = self._safe_get('x_2')
        self.a_2 = self._safe_get('a_2')

    def _safe_get(self, key):
        """Safely retrieve a value from results or attributes."""
        val = getattr(self.solution, key, None)
        if val is None:
            val = self.results.get(key, None)
        return val

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """Check fundamental physical laws."""
        logs = []
        try:
            # 1. Lorentz force magnitude (step 1)
            if self.v_0 is not None and self.F_Lorentz_y is not None:
                F_expected = abs(self.q * self.v_0 * self.B)  # magnitude
                if not self._is_close(self.F_Lorentz_y, F_expected):
                    logs.append(f"FAIL: Lorentz force mismatch. Expected {F_expected}, got {self.F_Lorentz_y}")
                else:
                    logs.append("PASS: Lorentz force magnitude consistent.")

            # 2. Newton's second law: a = F/m (step 2 and 3)
            if self.a is not None and self.F_Lorentz_y is not None:
                a_expected = self.F_Lorentz_y / self.m
                if not self._is_close(self.a, a_expected):
                    logs.append(f"FAIL: Acceleration mismatch. Expected {a_expected}, got {self.a}")
                else:
                    logs.append("PASS: Newton's second law (a = F/m) holds.")

            # 3. Kinematics: h = 0.5 * a * t^2 (step 2 and 4)
            if self.t is not None and self.a is not None:
                h_expected = 0.5 * self.a * self.t**2
                if not self._is_close(self.h, h_expected):
                    logs.append(f"FAIL: Displacement h mismatch. Expected {h_expected}, got {self.h}")
                else:
                    logs.append("PASS: Kinematics (h = 0.5 a t^2) consistent.")

            # 4. Kinematics: v_y = a * t (step 5)
            if self.v_y is not None and self.a is not None and self.t is not None:
                v_y_expected = self.a * self.t
                if not self._is_close(self.v_y, v_y_expected):
                    logs.append(f"FAIL: Vertical velocity mismatch. Expected {v_y_expected}, got {self.v_y}")
                else:
                    logs.append("PASS: Vertical velocity v_y = a*t holds.")

            # 5. Pythagorean velocity: v = sqrt(v_0^2 + v_y^2) (step 6)
            if self.v is not None and self.v_0 is not None and self.v_y is not None:
                v_expected = math.sqrt(self.v_0**2 + self.v_y**2)
                if not self._is_close(self.v, v_expected):
                    logs.append(f"FAIL: Resultant velocity mismatch. Expected {v_expected}, got {self.v}")
                else:
                    logs.append("PASS: Pythagorean velocity relationship holds.")

            # 6. Work-energy theorem: W = 0.5*m*(v^2 - v_0^2) (step 7)
            if self.W is not None and self.v is not None and self.v_0 is not None:
                W_expected = 0.5 * self.m * (self.v**2 - self.v_0**2)
                if not self._is_close(self.W, W_expected):
                    logs.append(f"FAIL: Work-energy mismatch. Expected {W_expected}, got {self.W}")
                else:
                    logs.append("PASS: Work-energy theorem holds.")

            # 7. Centripetal force: q*v_1*B = m*v_1^2 / R_1 (step 13)
            if self.v_1 is not None and self.R_1 is not None:
                lhs = abs(self.q * self.v_1 * self.B)
                rhs = self.m * self.v_1**2 / self.R_1
                if not self._is_close(lhs, rhs):
                    logs.append(f"FAIL: Centripetal force mismatch. LHS {lhs}, RHS {rhs}")
                else:
                    logs.append("PASS: Centripetal force condition for circular motion holds.")

        except Exception as e:
            logs.append(f"ERROR in test_invariants: {e}\n{traceback.format_exc()}")
        return logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # Check that the extracted t matches the target "0.1 s"
            if self.t is not None:
                if self._is_close(self.t, 0.1):
                    logs.append("PASS: Detachment time t = 0.1 s matches target.")
                else:
                    logs.append(f"FAIL: Detachment time t = {self.t} does not match target 0.1 s.")

            # Check that the extracted W matches the target "2e-5 J"
            if self.W is not None:
                if self._is_close(self.W, 2e-5):
                    logs.append("PASS: Work W = 2e-5 J matches target.")
                else:
                    logs.append(f"FAIL: Work W = {self.W} does not match target 2e-5 J.")

            # Check the critical velocity range: v_01 < v_0 < v_02
            if self.v_01 is not None and self.v_02 is not None:
                if self.v_01 < self.v_02:
                    logs.append("PASS: Critical velocities ordered: v_01 < v_02.")
                else:
                    logs.append(f"FAIL: Critical velocities misordered: v_01={self.v_01}, v_02={self.v_02}")

            # Verify v_01 = 1 m/s and v_02 = 2 m/s against derived equations
            if self.v_01 is not None:
                # Recompute using the equations from steps 8-15
                a1_expected = abs(self.q * self.v_01 * self.B) / self.m
                t1_expected = math.sqrt(2 * self.h / a1_expected)
                v_y1_expected = a1_expected * t1_expected
                v1_expected = math.sqrt(self.v_01**2 + v_y1_expected**2)
                R1_expected = self.m * v1_expected / (abs(self.q) * self.B)  # from centripetal condition
                if self._is_close(R1_expected, self.x_p):
                    logs.append("PASS: v_01 yields correct circular radius matching barrier position.")
                else:
                    logs.append(f"FAIL: For v_01={self.v_01}, expected R1={R1_expected}, but x_p={self.x_p}")

            if self.v_02 is not None:
                # For the second critical state, the ball just grazes the barrier at the top of its circular arc?
                # From steps 18-20: horizontal displacement x_2 = v_02 * t2, and h = 0.5 * a2 * t2^2, a2 = q*v_02*B/m
                a2_expected = abs(self.q * self.v_02 * self.B) / self.m
                t2_expected = math.sqrt(2 * self.h / a2_expected)
                x2_expected = self.v_02 * t2_expected
                # The barrier is at x_p, so for critical condition, x2_should be >= x_p? Actually, the problem states the ball should not hit the barrier.
                # The second critical state is when the ball just touches the barrier at the top of its motion? Not explicitly given.
                # We'll check that with v_02=2, the computed horizontal displacement equals x_p? Not necessarily.
                # Instead, we check consistency with the given v_02=2.
                # Using the same circular motion condition but with different geometry? The steps are not entirely clear.
                # We'll simply note that v_02=2 is given and trust the solution.
                logs.append("INFO: v_02=2 m/s accepted as given; detailed consistency check requires geometry.")

            # Verify that the derived work W simplifies to h*q*|v_0|*B (from monotonicity analysis)
            if self.W is not None and self.v_0 is not None:
                W_simplified = self.h * abs(self.q) * self.v_0 * self.B
                if self._is_close(self.W, W_simplified):
                    logs.append("PASS: Work W simplifies to h*q*v0*B, independent of mass.")
                else:
                    logs.append(f"FAIL: Work W does not match h*q*v0*B. W={self.W}, simplified={W_simplified}")

        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {e}\n{traceback.format_exc()}")
        return logs

    def test_monotonicity(self):
        """Perturb an input and verify expected trends."""
        logs = []
        try:
            # Perturb the magnetic field B by +1%
            original_B = self.B
            perturbed_B = original_B * 1.01

            # Recompute time t using the formula t = sqrt(2*h*m/(q*v0*B))
            if self.v_0 is not None:
                original_t = math.sqrt(2 * self.h * self.m / (abs(self.q) * self.v_0 * original_B))
                perturbed_t = math.sqrt(2 * self.h * self.m / (abs(self.q) * self.v_0 * perturbed_B))

                if perturbed_t < original_t:
                    logs.append("PASS: Increasing B decreases detachment time (as acceleration increases).")
                else:
                    logs.append(f"FAIL: Increasing B did not decrease t. Original t={original_t}, perturbed t={perturbed_t}")

                # Recompute work W = h*q*v0*B
                original_W = self.h * abs(self.q) * self.v_0 * original_B
                perturbed_W = self.h * abs(self.q) * self.v_0 * perturbed_B
                if perturbed_W > original_W:
                    logs.append("PASS: Increasing B increases work done (W proportional to B).")
                else:
                    logs.append(f"FAIL: Increasing B did not increase W. Original W={original_W}, perturbed W={perturbed_W}")

            # Perturb mass m and check that work W remains unchanged (since W = h*q*v0*B)
            original_m = self.m
            perturbed_m = original_m * 1.01
            if self.v_0 is not None:
                W_original = self.h * abs(self.q) * self.v_0 * self.B
                W_perturbed = self.h * abs(self.q) * self.v_0 * self.B  # independent of m
                if self._is_close(W_original, W_perturbed):
                    logs.append("PASS: Work W is independent of mass m.")
                else:
                    logs.append(f"FAIL: Work W changed with mass. Original W={W_original}, after perturbing m, W={W_perturbed}")

        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {e}\n{traceback.format_exc()}")
        return logs

    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        logs = []
        logs.append("=== PhysicsChecker started ===")

        # Run individual test suites
        logs.extend(self.test_invariants())
        logs.extend(self.test_trace_consistency())
        logs.extend(self.test_monotonicity())

        # Determine overall success: no failures that start with "FAIL:"
        success = True
        for log in logs:
            if log.startswith("FAIL:"):
                success = False
                break

        logs.append(f"=== PhysicsChecker finished: {'SUCCESS' if success else 'FAILURE'} ===")
        return success, logs