import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Use getattr with fallback to results dictionary.
        """
        self.solution = solution_instance
        self.logs = []

        # Input parameters (with given values as defaults for safety, but should be overridden)
        self.v1 = self._safe_get('v1', 80.0)          # takeoff speed (m/s)
        self.t1 = self._safe_get('t1', 1.5)           # time to reach v1 (s)
        self.n = self._safe_get('n', 100)             # number of turns
        self.l = self._safe_get('l', 1.0)             # perimeter per turn (m)
        self.m = self._safe_get('m', 5.0)             # mass of mover+coil (kg)
        self.M = self._safe_get('M', 10.0)            # mass of aircraft (kg)
        self.R0 = self._safe_get('R0', 9.5)           # fixed resistor (Ω)
        self.B = self._safe_get('B', 0.1)             # magnetic field (T)

        # Result quantities (from steps)
        self.I = self._safe_get('I', None)            # constant current (A)
        self.a = self._safe_get('a', None)            # acceleration during launch (m/s²)
        self.F1 = self._safe_get('F_1', None)         # Ampere force during launch (N)
        self.R = self._safe_get('R', None)            # coil resistance (Ω)
        self.a_prime = self._safe_get('a\'', None)    # acceleration during retraction (m/s²)
        self.t2 = self._safe_get('t_2', None)         # time when retraction starts? (s)
        self.s = self._safe_get('s', None)            # displacement 0 to t2 (m)
        self.I_prime = self._safe_get('I\'', None)    # induced current (A)
        self.F1_prime = self._safe_get('F_1\'', None) # Ampere force during retraction (N)
        self.Deltaq = self._safe_get('Deltaq', None)  # change in charge (C)
        self.E = self._safe_get('E', None)            # induced EMF (V)
        self.t3 = self._safe_get('t_3', None)         # final time (s)

        # Store original inputs for monotonicity test
        self.original_inputs = {
            'v1': self.v1, 't1': self.t1, 'n': self.n, 'l': self.l,
            'm': self.m, 'M': self.M, 'R0': self.R0, 'B': self.B
        }

    def _safe_get(self, key, default=None):
        """
        Safely retrieve attribute from solution instance.
        Priority: instance attribute -> results dictionary -> default.
        """
        # Some keys might have special characters, so we try to get as is
        val = getattr(self.solution, key, None)
        if val is None:
            # Try results dictionary if it exists
            if hasattr(self.solution, 'results') and isinstance(self.solution.results, dict):
                val = self.solution.results.get(key, default)
            else:
                val = default
        return val

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """
        Verify physical invariants based on the steps analysis.
        Returns (success, logs).
        """
        logs = []
        success = True

        try:
            # 1. Check acceleration during launch: a = v1 / t1
            a_expected = self.v1 / self.t1
            if self.a is not None:
                if not self._is_close(self.a, a_expected):
                    logs.append(f"FAIL: a = {self.a}, expected {a_expected} from v1/t1")
                    success = False
                else:
                    logs.append(f"OK: a = v1/t1 = {a_expected}")

            # 2. Check Ampere force during launch: F1 = n*B*I*l = (M+m)*a
            # First compute expected F1 from Newton's second law
            F1_newton = (self.M + self.m) * a_expected
            # Then from magnetic force
            if self.I is not None:
                F1_magnetic = self.n * self.B * self.I * self.l
                if not self._is_close(F1_newton, F1_magnetic):
                    logs.append(f"FAIL: F1 from Newton ({F1_newton}) != from magnetic ({F1_magnetic})")
                    success = False
                else:
                    logs.append(f"OK: F1 consistency (Newton = magnetic = {F1_newton})")

            # 3. Check the condition: n^2 * B^2 * l^2 / (R0 + R) = 10
            if self.R is not None:
                lhs = self.n**2 * self.B**2 * self.l**2 / (self.R0 + self.R)
                if not self._is_close(lhs, 10.0):
                    logs.append(f"FAIL: n^2 B^2 l^2/(R0+R) = {lhs}, expected 10")
                    success = False
                else:
                    logs.append(f"OK: n^2 B^2 l^2/(R0+R) = 10")

            # 4. Check acceleration during retraction: a' = 800 / m (from step 9)
            a_prime_expected = 800.0 / self.m
            if self.a_prime is not None:
                if not self._is_close(self.a_prime, a_prime_expected):
                    logs.append(f"FAIL: a' = {self.a_prime}, expected 800/m = {a_prime_expected}")
                    success = False
                else:
                    logs.append(f"OK: a' = 800/m = {a_prime_expected}")

            # 5. Check t2: t2 - t1 = v1 / a'
            if self.t2 is not None and self.a_prime is not None:
                t2_expected = self.t1 + self.v1 / self.a_prime
                if not self._is_close(self.t2, t2_expected):
                    logs.append(f"FAIL: t2 = {self.t2}, expected t1 + v1/a' = {t2_expected}")
                    success = False
                else:
                    logs.append(f"OK: t2 = t1 + v1/a' = {t2_expected}")

            # 6. Check displacement s: s = 0.5 * v1 * t2
            if self.s is not None and self.t2 is not None:
                s_expected = 0.5 * self.v1 * self.t2
                if not self._is_close(self.s, s_expected):
                    logs.append(f"FAIL: s = {self.s}, expected 0.5*v1*t2 = {s_expected}")
                    success = False
                else:
                    logs.append(f"OK: s = 0.5*v1*t2 = {s_expected}")

            # 7. Check the quadratic for dt = t3 - t2: dt^2 + dt - (t2-t1)*t2 = 0
            if self.t3 is not None and self.t2 is not None:
                dt = self.t3 - self.t2
                t1 = self.t1
                t2 = self.t2
                residual = dt**2 + dt - (t2 - t1) * t2
                if not self._is_close(residual, 0.0, abs_tol=1e-2):
                    logs.append(f"FAIL: dt^2 + dt - (t2-t1)*t2 = {residual}, expected 0")
                    success = False
                else:
                    logs.append(f"OK: dt = t3-t2 satisfies quadratic (dt={dt})")

        except Exception as e:
            logs.append(f"ERROR in test_invariants: {e}")
            success = False

        self.logs.extend(logs)
        return success, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        logs = []
        success = True

        try:
            # Target answers: I=80A, R=0.5 Ω, t3 = (sqrt(5)+3)/2 s
            target_I = 80.0
            target_R = 0.5
            target_t3 = (math.sqrt(5) + 3) / 2

            # Check I
            if self.I is not None:
                if not self._is_close(self.I, target_I):
                    logs.append(f"FAIL: I = {self.I}, target {target_I} A")
                    success = False
                else:
                    logs.append(f"OK: I matches target {target_I} A")

            # Check R
            if self.R is not None:
                if not self._is_close(self.R, target_R):
                    logs.append(f"FAIL: R = {self.R}, target {target_R} Ω")
                    success = False
                else:
                    logs.append(f"OK: R matches target {target_R} Ω")

            # Check t3
            if self.t3 is not None:
                if not self._is_close(self.t3, target_t3):
                    logs.append(f"FAIL: t3 = {self.t3}, target {target_t3} s")
                    success = False
                else:
                    logs.append(f"OK: t3 matches target {target_t3} s")

            # Additional trace: compute I from formula and compare
            I_computed = (self.m + self.M) * self.v1 / (self.n * self.l * self.B * self.t1)
            if self.I is not None and not self._is_close(self.I, I_computed):
                logs.append(f"FAIL: I from solution {self.I} != computed from formula {I_computed}")
                success = False
            else:
                logs.append(f"OK: I formula yields {I_computed}")

            # Compute R from condition
            R_computed = self.n**2 * self.B**2 * self.l**2 / 10.0 - self.R0
            if self.R is not None and not self._is_close(self.R, R_computed):
                logs.append(f"FAIL: R from solution {self.R} != computed from condition {R_computed}")
                success = False
            else:
                logs.append(f"OK: R condition yields {R_computed}")

        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {e}")
            success = False

        self.logs.extend(logs)
        return success, logs

    def test_monotonicity(self):
        """
        Perturb mass m and check trends.
        """
        logs = []
        success = True

        original_m = self.m
        perturb_factor = 1.01  # increase by 1%
        self.m = original_m * perturb_factor

        try:
            # Recompute quantities based on perturbed m
            # I = (m+M)*v1/(n*l*B*t1)
            I_new = (self.m + self.M) * self.v1 / (self.n * self.l * self.B * self.t1)
            # R is independent of m (from condition n^2 B^2 l^2/(R0+R)=10)
            R_new = self.n**2 * self.B**2 * self.l**2 / 10.0 - self.R0
            # a' = 800 / m
            a_prime_new = 800.0 / self.m
            # t2 = t1 + v1 / a'
            t2_new = self.t1 + self.v1 / a_prime_new
            # s = 0.5 * v1 * t2
            s_new = 0.5 * self.v1 * t2_new
            # dt from quadratic: dt^2 + dt - (t2-t1)*t2 = 0
            # Solve dt = (-1 + sqrt(1 + 4*(t2-t1)*t2)) / 2   (positive root)
            discriminant = 1 + 4 * (t2_new - self.t1) * t2_new
            if discriminant >= 0:
                dt_new = (-1 + math.sqrt(discriminant)) / 2
                t3_new = t2_new + dt_new
            else:
                logs.append("FAIL: discriminant negative in monotonicity test")
                success = False
                dt_new = None
                t3_new = None

            # Check trends:
            # As m increases, I should increase (since I ∝ m+M)
            if I_new <= self.I:
                logs.append(f"FAIL: I did not increase when m increased. I_old={self.I}, I_new={I_new}")
                success = False
            else:
                logs.append(f"OK: I increases with m (from {self.I} to {I_new})")

            # R should remain constant
            if not self._is_close(R_new, self.R):
                logs.append(f"FAIL: R changed with m. R_old={self.R}, R_new={R_new}")
                success = False
            else:
                logs.append(f"OK: R constant (R={R_new})")

            # a' should decrease (since a' = 800/m)
            if a_prime_new >= self.a_prime:
                logs.append(f"FAIL: a' did not decrease when m increased. a'_old={self.a_prime}, a'_new={a_prime_new}")
                success = False
            else:
                logs.append(f"OK: a' decreases with m (from {self.a_prime} to {a_prime_new})")

            # t2 should increase (since t2 = t1 + v1/a', and a' decreases)
            if t2_new <= self.t2:
                logs.append(f"FAIL: t2 did not increase when m increased. t2_old={self.t2}, t2_new={t2_new}")
                success = False
            else:
                logs.append(f"OK: t2 increases with m (from {self.t2} to {t2_new})")

            # s should increase (since s = 0.5*v1*t2)
            if s_new <= self.s:
                logs.append(f"FAIL: s did not increase when m increased. s_old={self.s}, s_new={s_new}")
                success = False
            else:
                logs.append(f"OK: s increases with m (from {self.s} to {s_new})")

            # t3? We can check if t3 increases or decreases. Compute original dt and new dt.
            if dt_new is not None and self.t3 is not None:
                # Original dt = t3 - t2
                dt_original = self.t3 - self.t2
                if dt_new <= dt_original:
                    logs.append(f"INFO: dt decreased (from {dt_original} to {dt_new}) when m increased")
                else:
                    logs.append(f"INFO: dt increased (from {dt_original} to {dt_new}) when m increased")
                # Overall t3 trend?
                if t3_new <= self.t3:
                    logs.append(f"INFO: t3 decreased (from {self.t3} to {t3_new}) when m increased")
                else:
                    logs.append(f"INFO: t3 increased (from {self.t3} to {t3_new}) when m increased")

        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {e}")
            success = False

        # Restore original mass
        self.m = original_m

        self.logs.extend(logs)
        return success, logs

    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for name, test_func in tests:
            all_logs.append(f"=== {name} Test ===")
            success, logs = test_func()
            all_logs.extend(logs)
            if not success:
                overall_success = False
            all_logs.append("")

        return overall_success, all_logs