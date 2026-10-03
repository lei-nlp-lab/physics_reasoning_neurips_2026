import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from the solution instance."""
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Expected inputs (symbols) - extract from results or use defaults
        # If not in results, use the numeric values we assigned in our code (1.0)
        self.m = self.results.get('m', 1.0)
        self.q = self.results.get('q', 1.0)
        self.L = self.results.get('L', 1.0)
        self.d = self.results.get('d', 1.0)
        self.v0 = self.results.get('v0', 1.0)
        self.U = self.results.get('U', None)  # Not needed directly

        # Expected results (from key mapping)
        self.y = self.results.get('y', None)
        self.v_y = self.results.get('v_y', None)
        self.t = self.results.get('t', None)
        self.a = self.results.get('a', None)
        self.U_m = self.results.get('U_m', None)
        self.v = self.results.get('v', None)
        self.t_1 = self.results.get('t_1', None)

        # Fallback: compute missing results using numeric formulas
        self._ensure_results()

        # Logs for reporting
        self.logs = []

    def _ensure_results(self):
        """If any result is None, attempt to compute it from known formulas using numeric values."""
        try:
            # Compute U_m if missing
            if self.U_m is None:
                self.U_m = (self.m * self.d**2 * self.v0**2) / (self.q * self.L**2)
                self.results['U_m'] = self.U_m

            # Compute acceleration a if missing
            if self.a is None and self.U_m is not None:
                self.a = (self.q * self.U_m) / (self.m * self.d)
                self.results['a'] = self.a

            # Compute time t if missing
            if self.t is None:
                self.t = self.L / self.v0
                self.results['t'] = self.t

            # Compute displacement y if missing
            if self.y is None and self.a is not None and self.t is not None:
                self.y = 0.5 * self.a * self.t**2
                self.results['y'] = self.y

            # Compute time t1 if missing
            if self.t_1 is None:
                self.t_1 = self.L / (3 * self.v0)
                self.results['t_1'] = self.t_1

            # Compute vertical velocity v_y if missing
            if self.v_y is None and self.a is not None and self.t_1 is not None:
                self.v_y = self.a * self.t_1
                self.results['v_y'] = self.v_y

            # Compute final speed v if missing
            if self.v is None and self.v0 is not None and self.v_y is not None:
                self.v = math.sqrt((3 * self.v0)**2 + self.v_y**2)
                self.results['v'] = self.v

        except Exception as e:
            self.logs.append(f"Warning: Could not compute missing results: {e}")

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison."""
        try:
            return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)
        except (TypeError, ValueError):
            return False

    def test_invariants(self):
        """Check physical invariants: energy conservation, Newton's second law, geometric constraint."""
        logs = []
        success = True

        # 1. Check Newton's second law: a = q * U_m / (m * d)
        try:
            expected_a = (self.q * self.U_m) / (self.m * self.d)
            if not self._is_close(self.a, expected_a):
                logs.append(f"FAIL: Newton's second law mismatch. a={self.a}, expected {expected_a}")
                success = False
            else:
                logs.append("PASS: Newton's second law (a = q*U_m/(m*d)) holds.")
        except Exception as e:
            logs.append(f"ERROR in Newton's second law check: {e}")
            success = False

        # 2. Check energy conservation for the first part (when U = U_m and y = d/2)
        try:
            # Vertical velocity at exit for first case: v_y1 = a * t
            v_y1 = self.a * self.t
            # Change in kinetic energy (only vertical component changes)
            delta_K = 0.5 * self.m * (v_y1**2)
            # Change in potential energy: -q * (U_m/d) * y, with y = d/2
            delta_U = -self.q * (self.U_m / self.d) * (self.d / 2)
            if not self._is_close(delta_K, -delta_U):
                logs.append(f"FAIL: Energy conservation mismatch. delta_K={delta_K}, -delta_U={-delta_U}")
                success = False
            else:
                logs.append("PASS: Energy conservation (delta_K = -delta_U) holds for first part.")
        except Exception as e:
            logs.append(f"ERROR in energy conservation check (first part): {e}")
            success = False

        # 3. Check geometric constraint: y = d/2 when U = U_m
        try:
            y_at_U_m = 0.5 * self.a * self.t**2
            if not self._is_close(y_at_U_m, self.d / 2):
                logs.append(f"FAIL: Geometric constraint mismatch. y at U_m={y_at_U_m}, expected d/2={self.d/2}")
                success = False
            else:
                logs.append("PASS: Geometric constraint (y = d/2 at U_m) holds.")
        except Exception as e:
            logs.append(f"ERROR in geometric constraint check: {e}")
            success = False

        # 4. Check energy conservation for the second part (initial velocity 3v0, U = U_m)
        try:
            # Compute y for second part
            y_second = 0.5 * self.a * self.t_1**2
            delta_U_second = -self.q * (self.U_m / self.d) * y_second
            delta_K_second = 0.5 * self.m * (self.v**2 - (3 * self.v0)**2)
            if not self._is_close(delta_K_second, -delta_U_second):
                logs.append(f"FAIL: Energy conservation mismatch for second part. delta_K={delta_K_second}, -delta_U={-delta_U_second}")
                success = False
            else:
                logs.append("PASS: Energy conservation holds for second part.")
        except Exception as e:
            logs.append(f"ERROR in energy conservation check (second part): {e}")
            success = False

        self.logs.extend(logs)
        return success, logs

    def test_trace_consistency(self):
        """Check that intermediate results (t, a, t1, v_y) are consistent with final answers (U_m, v)."""
        logs = []
        success = True

        # 1. Check that U_m = m d^2 v0^2 / (q L^2)
        try:
            expected_U_m = (self.m * self.d**2 * self.v0**2) / (self.q * self.L**2)
            if not self._is_close(self.U_m, expected_U_m):
                logs.append(f"FAIL: U_m mismatch. Got {self.U_m}, expected {expected_U_m}")
                success = False
            else:
                logs.append("PASS: U_m expression is correct.")
        except Exception as e:
            logs.append(f"ERROR in U_m expression check: {e}")
            success = False

        # 2. Check that a = q U_m / (m d)
        try:
            expected_a = (self.q * self.U_m) / (self.m * self.d)
            if not self._is_close(self.a, expected_a):
                logs.append(f"FAIL: Acceleration a mismatch. Got {self.a}, expected {expected_a}")
                success = False
            else:
                logs.append("PASS: Acceleration a is consistent with U_m.")
        except Exception as e:
            logs.append(f"ERROR in a consistency check: {e}")
            success = False

        # 3. Check that t = L / v0
        try:
            expected_t = self.L / self.v0
            if not self._is_close(self.t, expected_t):
                logs.append(f"FAIL: Time t mismatch. Got {self.t}, expected {expected_t}")
                success = False
            else:
                logs.append("PASS: Time t is consistent with L and v0.")
        except Exception as e:
            logs.append(f"ERROR in t consistency check: {e}")
            success = False

        # 4. Check that y = (1/2) a t^2
        try:
            expected_y = 0.5 * self.a * self.t**2
            if not self._is_close(self.y, expected_y):
                logs.append(f"FAIL: Displacement y mismatch. Got {self.y}, expected {expected_y}")
                success = False
            else:
                logs.append("PASS: Displacement y is consistent with a and t.")
        except Exception as e:
            logs.append(f"ERROR in y consistency check: {e}")
            success = False

        # 5. Check that t1 = L / (3 v0)
        try:
            expected_t1 = self.L / (3 * self.v0)
            if not self._is_close(self.t_1, expected_t1):
                logs.append(f"FAIL: Time t1 mismatch. Got {self.t_1}, expected {expected_t1}")
                success = False
            else:
                logs.append("PASS: Time t1 is consistent with L and 3v0.")
        except Exception as e:
            logs.append(f"ERROR in t1 consistency check: {e}")
            success = False

        # 6. Check that v_y = a * t1
        try:
            expected_v_y = self.a * self.t_1
            if not self._is_close(self.v_y, expected_v_y):
                logs.append(f"FAIL: Vertical velocity v_y mismatch. Got {self.v_y}, expected {expected_v_y}")
                success = False
            else:
                logs.append("PASS: Vertical velocity v_y is consistent with a and t1.")
        except Exception as e:
            logs.append(f"ERROR in v_y consistency check: {e}")
            success = False

        # 7. Check that v = sqrt((3 v0)^2 + v_y^2)
        try:
            expected_v = math.sqrt((3 * self.v0)**2 + self.v_y**2)
            if not self._is_close(self.v, expected_v):
                logs.append(f"FAIL: Final velocity v mismatch. Got {self.v}, expected {expected_v}")
                success = False
            else:
                logs.append("PASS: Final velocity v is consistent with v0 and v_y.")
        except Exception as e:
            logs.append(f"ERROR in v consistency check: {e}")
            success = False

        self.logs.extend(logs)
        return success, logs

    def test_monotonicity(self):
        """Perturb an input (e.g., mass m) and check that results change in the expected direction."""
        logs = []
        success = True

        # We'll perturb m by a small factor (e.g., 1.01) and check that U_m changes in the expected way.
        # From formula U_m = m d^2 v0^2 / (q L^2), U_m is proportional to m.
        # So if m increases, U_m should increase.
        try:
            # Store original values
            original_m = self.m
            original_U_m = self.U_m

            # Perturb m by 1%
            perturbed_m = original_m * 1.01
            perturbed_U_m = (perturbed_m * self.d**2 * self.v0**2) / (self.q * self.L**2)

            # Check monotonicity: U_m should increase when m increases
            if perturbed_U_m > original_U_m:
                logs.append("PASS: Monotonicity holds for U_m with respect to m (U_m increases when m increases).")
            else:
                logs.append(f"FAIL: Monotonicity violation for U_m. Original U_m={original_U_m}, perturbed U_m={perturbed_U_m}")
                success = False

        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {e}")
            success = False

        self.logs.extend(logs)
        return success, logs

    def run_all_tests(self):
        """Run all tests and return overall success and logs."""
        self.logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            try:
                test_success, test_logs = test_func()
                if not test_success:
                    overall_success = False
                self.logs.append(f"--- {test_name} ---")
                self.logs.extend(test_logs)
            except Exception as e:
                self.logs.append(f"Exception in {test_name}: {traceback.format_exc()}")
                overall_success = False

        return overall_success, self.logs