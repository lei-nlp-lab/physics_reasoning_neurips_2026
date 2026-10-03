import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the checker with safe attribute access.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        self.logs = []

        # Input parameters (with defaults if missing)
        self.m = getattr(solution_instance, 'm', 0.05)
        self.L = getattr(solution_instance, 'L', 0.1)
        self.s = getattr(solution_instance, 's', 2.0)
        self.mu = getattr(solution_instance, 'mu', 0.25)
        self.h = getattr(solution_instance, 'h', 5.0)
        self.g = getattr(solution_instance, 'g', 10.0)

        # Results (with safe extraction)
        self.v_1 = self._get_result('v_1')
        self.v_1_prime = self._get_result('v_1\'') or self._get_result('v_1p') or self._get_result('v_1_prime')
        self.v_2 = self._get_result('v_2')
        self.T = self._get_result('T')
        self.v = self._get_result('v')
        self.s_prime = self._get_result('s\'') or self._get_result('s_prime')
        self.n = self._get_result('n')

    def _get_result(self, key):
        """Safe extraction from results dict, with fallback to 0.0."""
        return self.results.get(key, 0.0)

    def _log(self, message, success):
        """Helper to log test results."""
        self.logs.append((message, success))
        return success

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical comparison."""
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        # If symbolic, evaluate numerically
        try:
            a_val = float(a) if not isinstance(a, (int, float)) else a
            b_val = float(b) if not isinstance(b, (int, float)) else b
            return math.isclose(a_val, b_val, rel_tol=rel_tol, abs_tol=abs_tol)
        except:
            return False

    def test_invariants(self):
        """Check fundamental physical laws."""
        success = True

        # 1. Conservation of energy for slider B sliding down (with friction)
        # Equation: mgh = 0.5*m*v_1^2 + μ*m*g*(s/2)
        left = self.m * self.g * self.h
        right = 0.5 * self.m * self.v_1**2 + self.mu * self.m * self.g * (self.s / 2)
        if not self._is_close(left, right):
            success = self._log(
                f"Energy conservation for initial slide failed: {left} vs {right}",
                False
            )
        else:
            self._log("Energy conservation for initial slide passed", True)

        # 2. Elastic collision: conservation of momentum and kinetic energy
        # Momentum: m*v_1 = m*v_1' + m*v_2  => v_1 = v_1' + v_2
        momentum_eq = self.v_1 - (self.v_1_prime + self.v_2)
        if not self._is_close(momentum_eq, 0):
            success = self._log(
                f"Momentum conservation in collision failed: v_1 - (v_1' + v_2) = {momentum_eq}",
                False
            )
        else:
            self._log("Momentum conservation in collision passed", True)

        # Kinetic energy: 0.5*m*v_1^2 = 0.5*m*v_1'^2 + 0.5*m*v_2^2
        ke_left = 0.5 * self.m * self.v_1**2
        ke_right = 0.5 * self.m * self.v_1_prime**2 + 0.5 * self.m * self.v_2**2
        if not self._is_close(ke_left, ke_right):
            success = self._log(
                f"Kinetic energy conservation in collision failed: {ke_left} vs {ke_right}",
                False
            )
        else:
            self._log("Kinetic energy conservation in collision passed", True)

        # 3. Newton's second law for ball A after collision (centripetal force)
        # T - m*g = m * v_2^2 / L
        if self.v_2 != 0 and self.T != 0:
            left_force = self.T - self.m * self.g
            right_force = self.m * self.v_2**2 / self.L
            if not self._is_close(left_force, right_force):
                success = self._log(
                    f"Newton's second law (centripetal) failed: T - mg = {left_force}, mv²/L = {right_force}",
                    False
                )
            else:
                self._log("Newton's second law (centripetal) passed", True)

        # 4. Conservation of energy for circular motion condition (step 10)
        # 0.5*m*v^2 = 2*m*g*L + 0.5*m*v_0^2, with v_0 = sqrt(g*L) from step 9
        v_0 = math.sqrt(self.g * self.L)
        left_energy = 0.5 * self.m * self.v**2
        right_energy = 2 * self.m * self.g * self.L + 0.5 * self.m * v_0**2
        if not self._is_close(left_energy, right_energy):
            success = self._log(
                f"Energy conservation for circular motion failed: {left_energy} vs {right_energy}",
                False
            )
        else:
            self._log("Energy conservation for circular motion passed", True)

        return success

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        success = True

        # Check v_1 from step_1 (√95)
        expected_v1 = math.sqrt(95)
        if not self._is_close(self.v_1, expected_v1):
            success = self._log(
                f"v_1 consistency failed: expected √95 ≈ {expected_v1}, got {self.v_1}",
                False
            )
        else:
            self._log("v_1 (√95) consistent", True)

        # Check v_1' from step_4 (0)
        if not self._is_close(self.v_1_prime, 0):
            success = self._log(
                f"v_1' consistency failed: expected 0, got {self.v_1_prime}",
                False
            )
        else:
            self._log("v_1' (0) consistent", True)

        # Check v_2 from step_5 (√95)
        if not self._is_close(self.v_2, expected_v1):
            success = self._log(
                f"v_2 consistency failed: expected √95 ≈ {expected_v1}, got {self.v_2}",
                False
            )
        else:
            self._log("v_2 (√95) consistent", True)

        # Check T from step_7 (48 N)
        expected_T = 48.0
        if not self._is_close(self.T, expected_T):
            success = self._log(
                f"T consistency failed: expected 48 N, got {self.T}",
                False
            )
        else:
            self._log("T (48 N) consistent", True)

        # Check v from step_10 (√5)
        expected_v = math.sqrt(5)
        if not self._is_close(self.v, expected_v):
            success = self._log(
                f"v (circular threshold) consistency failed: expected √5 ≈ {expected_v}, got {self.v}",
                False
            )
        else:
            self._log("v (√5) consistent", True)

        # Check s' from step_12 (19 m)
        expected_s_prime = 19.0
        if not self._is_close(self.s_prime, expected_s_prime):
            success = self._log(
                f"s' consistency failed: expected 19 m, got {self.s_prime}",
                False
            )
        else:
            self._log("s' (19 m) consistent", True)

        # Check n from step_13 (10)
        expected_n = 10.0
        # Allow either exact 10 or the computed formula value 9.5
        formula_n = (self.s_prime - self.s) / self.s + 1
        if not (self._is_close(self.n, expected_n) or self._is_close(self.n, formula_n)):
            success = self._log(
                f"n consistency failed: expected 10 or formula value {formula_n}, got {self.n}",
                False
            )
        else:
            self._log("n (10 or formula) consistent", True)

        # Verify n formula: n = (s' - s)/s + 1
        # This is the theoretical formula, but n might be rounded
        computed_n = (self.s_prime - self.s) / self.s + 1
        # Allow either exact match or integer rounding
        if not (self._is_close(self.n, computed_n) or self._is_close(self.n, round(computed_n))):
            success = self._log(
                f"n formula consistency failed: (s'-s)/s + 1 = {computed_n}, but n = {self.n}",
                False
            )
        else:
            self._log("n formula consistent", True)

        return success

    def test_monotonicity(self):
        """
        Perturb an input (height h) and check that outputs change in expected direction.
        Then restore original h.
        """
        success = True
        original_h = self.h

        # Perturb h by +10%
        perturbed_h = original_h * 1.10
        self.h = perturbed_h

        # Recompute v_1 using the equation from step_1
        # mgh = 0.5*m*v_1^2 + μ*m*g*(s/2)
        # => v_1_new = sqrt(2*(g*h - μ*g*(s/2)))
        try:
            v_1_new = math.sqrt(2 * (self.g * perturbed_h - self.mu * self.g * (self.s / 2)))
            # v_1 should increase with h
            if v_1_new <= self.v_1:
                success = self._log(
                    f"Monotonicity: v_1 did not increase when h increased (from {self.v_1} to {v_1_new})",
                    False
                )
            else:
                self._log("Monotonicity: v_1 increases with h (as expected)", True)

            # Recompute v_2 from elastic collision (assuming v_1' still 0 and v_2 = v_1)
            v_2_new = v_1_new  # from elastic collision with equal masses and v_1' = 0
            # v_2 should also increase
            if v_2_new <= self.v_2:
                success = self._log(
                    f"Monotonicity: v_2 did not increase when h increased (from {self.v_2} to {v_2_new})",
                    False
                )
            else:
                self._log("Monotonicity: v_2 increases with h (as expected)", True)

            # Recompute T from centripetal force: T = m*g + m*v_2^2/L
            T_new = self.m * self.g + self.m * v_2_new**2 / self.L
            if T_new <= self.T:
                success = self._log(
                    f"Monotonicity: T did not increase when h increased (from {self.T} to {T_new})",
                    False
                )
            else:
                self._log("Monotonicity: T increases with h (as expected)", True)

            # Recompute s' from energy: mgh = 0.5*m*v^2 + μ*m*g*s'
            # v is fixed at √5 (minimum for circular motion) so:
            v_min = math.sqrt(5)  # from step_10
            s_prime_new = (self.g * perturbed_h - 0.5 * v_min**2) / (self.mu * self.g)
            if s_prime_new <= self.s_prime:
                success = self._log(
                    f"Monotonicity: s' did not increase when h increased (from {self.s_prime} to {s_prime_new})",
                    False
                )
            else:
                self._log("Monotonicity: s' increases with h (as expected)", True)

            # Recompute n using the same formula as in the solution
            # Use formula n = (s' - s)/s + 1
            n_new = (s_prime_new - self.s) / self.s + 1
            if n_new <= self.n:
                success = self._log(
                    f"Monotonicity: n did not increase when h increased (from {self.n} to {n_new})",
                    False
                )
            else:
                self._log("Monotonicity: n increases with h (as expected)", True)

        except Exception as e:
            success = self._log(f"Monotonicity test raised exception: {e}", False)

        finally:
            # Restore original h
            self.h = original_h

        return success

    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs.clear()
        success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]

        for test_name, test_func in tests:
            try:
                if not test_func():
                    success = False
            except Exception as e:
                self.logs.append((f"{test_name} raised exception: {traceback.format_exc()}", False))
                success = False

        return success, self.logs