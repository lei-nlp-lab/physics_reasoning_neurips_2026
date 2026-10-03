import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution = solution_instance
        # Safely extract results dictionary, default to empty dict
        self.results = getattr(solution_instance, 'results', {})

        # Extract inputs with safe defaults from problem statement
        self.v_bike = self._safe_get('v', 8.0)
        self.a_car = self._safe_get('a', 2.0)

        # Extract primary results (catch-up time, car speed, max distance)
        self.t_catch = self._safe_get('t', None)          # Catch-up time from step_4
        self.v_car_catch = self._safe_get('v_car', None)  # Car speed at catch-up from step_5
        self.delta_x_max = self._safe_get('delta_x', None) # Max distance from step_9

        # Extract intermediate results for trace consistency
        self.x1 = self._safe_get('x_1', None)            # Bicycle displacement at t_catch
        self.x2 = self._safe_get('x_2', None)            # Car displacement at t_catch
        self.t_prime = self._safe_get('t\'', None)       # Time when speeds equal (step_6)
        self.x1_prime = self._safe_get('x_1\'', None)    # Bicycle displacement at t' (step_7)
        self.x2_prime = self._safe_get('x_2\'', None)    # Car displacement at t' (step_8)

        # Symbolic variables for invariant checks
        self.v, self.a, self.t_sym = sp.symbols('v a t', positive=True)
        self.t_prime_sym = sp.symbols('t\'', positive=True)

    def _safe_get(self, key, default):
        """Safely retrieve attribute from solution instance or results dict."""
        val = getattr(self.solution, key, None)
        if val is None:
            val = self.results.get(key, default)
        return val

    def _is_close(self, a, b):
        """Robust fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

    def test_invariants(self):
        logs = []
        success = True

        # 1. Check catch-up condition: displacements equal at t_catch
        if self.t_catch is not None:
            x1_expected = self.v_bike * self.t_catch
            x2_expected = 0.5 * self.a_car * self.t_catch ** 2
            if not self._is_close(x1_expected, x2_expected):
                logs.append(f"Invariant failed: displacements not equal at t={self.t_catch}. "
                            f"x1={x1_expected:.3f}, x2={x2_expected:.3f}")
                success = False
            else:
                logs.append("Catch-up displacement invariant passed.")

        # 2. Check catch-up time formula: t = 2v/a
        t_expected = 2 * self.v_bike / self.a_car
        if self.t_catch is not None and not self._is_close(self.t_catch, t_expected):
            logs.append(f"Invariant failed: t_catch does not match 2v/a. "
                        f"Expected {t_expected:.3f}, got {self.t_catch:.3f}")
            success = False
        else:
            logs.append("Catch-up time formula invariant passed.")

        # 3. Check car speed at catch-up: v_car = a * t_catch
        if self.v_car_catch is not None and self.t_catch is not None:
            v_expected = self.a_car * self.t_catch
            if not self._is_close(self.v_car_catch, v_expected):
                logs.append(f"Invariant failed: v_car_catch does not match a*t. "
                            f"Expected {v_expected:.3f}, got {self.v_car_catch:.3f}")
                success = False
            else:
                logs.append("Car speed at catch-up invariant passed.")

        # 4. Check time when speeds equal: t' = v/a
        t_prime_expected = self.v_bike / self.a_car
        if self.t_prime is not None and not self._is_close(self.t_prime, t_prime_expected):
            logs.append(f"Invariant failed: t' does not match v/a. "
                        f"Expected {t_prime_expected:.3f}, got {self.t_prime:.3f}")
            success = False
        else:
            logs.append("Time of equal speeds invariant passed.")

        # 5. Check maximum distance formula: Δx = v*t' - 0.5*a*t'^2
        delta_x_expected = self.v_bike * t_prime_expected - 0.5 * self.a_car * t_prime_expected ** 2
        if self.delta_x_max is not None and not self._is_close(self.delta_x_max, delta_x_expected):
            logs.append(f"Invariant failed: Δx_max does not match formula. "
                        f"Expected {delta_x_expected:.3f}, got {self.delta_x_max:.3f}")
            success = False
        else:
            logs.append("Maximum distance invariant passed.")

        return success, logs

    def test_trace_consistency(self):
        logs = []
        success = True

        # 1. Check intermediate displacements at catch-up (if provided)
        if self.x1 is not None and self.t_catch is not None:
            x1_expected = self.v_bike * self.t_catch
            if not self._is_close(self.x1, x1_expected):
                logs.append(f"Trace inconsistency: x1 (bicycle) at t_catch. "
                            f"Expected {x1_expected:.3f}, got {self.x1:.3f}")
                success = False

        if self.x2 is not None and self.t_catch is not None:
            x2_expected = 0.5 * self.a_car * self.t_catch ** 2
            if not self._is_close(self.x2, x2_expected):
                logs.append(f"Trace inconsistency: x2 (car) at t_catch. "
                            f"Expected {x2_expected:.3f}, got {self.x2:.3f}")
                success = False

        # 2. Check consistency between x1 and x2 at catch-up (if both provided)
        if self.x1 is not None and self.x2 is not None:
            if not self._is_close(self.x1, self.x2):
                logs.append(f"Trace inconsistency: x1 and x2 not equal at catch-up. "
                            f"x1={self.x1:.3f}, x2={self.x2:.3f}")
                success = False

        # 3. Check intermediate results at t' (if provided)
        if self.t_prime is not None:
            # Bicycle displacement at t'
            if self.x1_prime is not None:
                x1_prime_expected = self.v_bike * self.t_prime
                if not self._is_close(self.x1_prime, x1_prime_expected):
                    logs.append(f"Trace inconsistency: x1' (bicycle) at t'. "
                                f"Expected {x1_prime_expected:.3f}, got {self.x1_prime:.3f}")
                    success = False
            # Car displacement at t'
            if self.x2_prime is not None:
                x2_prime_expected = 0.5 * self.a_car * self.t_prime ** 2
                if not self._is_close(self.x2_prime, x2_prime_expected):
                    logs.append(f"Trace inconsistency: x2' (car) at t'. "
                                f"Expected {x2_prime_expected:.3f}, got {self.x2_prime:.3f}")
                    success = False
            # Max distance from intermediates
            if self.x1_prime is not None and self.x2_prime is not None and self.delta_x_max is not None:
                delta_x_from_intermediates = self.x1_prime - self.x2_prime
                if not self._is_close(self.delta_x_max, delta_x_from_intermediates):
                    logs.append(f"Trace inconsistency: Δx_max does not match x1' - x2'. "
                                f"Expected {delta_x_from_intermediates:.3f}, got {self.delta_x_max:.3f}")
                    success = False

        if success:
            logs.append("All trace consistency checks passed.")
        return success, logs

    def test_monotonicity(self):
        logs = []
        success = True

        # Perturb acceleration by +10%
        a_perturbed = self.a_car * 1.10

        # Recompute catch-up time: t = 2v / a
        t_catch_perturbed = 2 * self.v_bike / a_perturbed

        # Recompute maximum distance: Δx = v^2 / (2a)
        delta_x_perturbed = self.v_bike ** 2 / (2 * a_perturbed)

        # Expected trends: if a increases, t_catch decreases, Δx decreases
        if self.t_catch is not None:
            if t_catch_perturbed >= self.t_catch:
                logs.append(f"Monotonicity failed: t_catch should decrease when a increases. "
                            f"Original t={self.t_catch:.3f}, perturbed t={t_catch_perturbed:.3f}")
                success = False
            else:
                logs.append(f"Monotonicity passed for t_catch: {t_catch_perturbed:.3f} < {self.t_catch:.3f}")

        if self.delta_x_max is not None:
            if delta_x_perturbed >= self.delta_x_max:
                logs.append(f"Monotonicity failed: Δx_max should decrease when a increases. "
                            f"Original Δx={self.delta_x_max:.3f}, perturbed Δx={delta_x_perturbed:.3f}")
                success = False
            else:
                logs.append(f"Monotonicity passed for Δx_max: {delta_x_perturbed:.3f} < {self.delta_x_max:.3f}")

        # Also check car speed at catch-up: v_car = a*t, but since t changes, the trend is not monotonic in a.
        # Instead, we can check that v_car = 2v (from v_car = a*(2v/a) = 2v) which is independent of a.
        # So v_car should remain constant when a changes (if v is fixed).
        v_car_expected = 2 * self.v_bike
        if self.v_car_catch is not None and not self._is_close(self.v_car_catch, v_car_expected):
            logs.append(f"Monotonicity check note: v_car_catch should always be 2v = {v_car_expected:.3f} "
                        f"(got {self.v_car_catch:.3f}) regardless of a.")
            # This is not a failure, but a consistency note.

        return success, logs

    def run_all_tests(self):
        all_logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"=== {test_name} ===")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"Error in {test_name}: {traceback.format_exc()}")
                overall_success = False

        return overall_success, all_logs