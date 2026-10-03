import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely initialize by extracting inputs and results from the solution instance.
        Use getattr and .get() to avoid AttributeError in sandbox.
        """
        self.solution = solution_instance

        # Extract input parameters with safe defaults
        self.v0 = getattr(solution_instance, 'v0', 23.0)
        self.v1 = getattr(solution_instance, 'v1', 25.0)
        self.x0 = getattr(solution_instance, 'x0', 28.0)
        self.t = getattr(solution_instance, 't', 3.5)
        self.dt = getattr(solution_instance, 'dt', 0.5)
        self.truck_stop_dist = getattr(solution_instance, 'truck_stop_dist', 125.0)

        # Extract results with safe defaults
        results = getattr(solution_instance, 'results', {})
        # Note: The code stores 'a_prime' and 't_prime', not "a'" and "t'"
        self.t_prime = results.get('t_prime', 0.0)
        self.v2 = results.get('v_2', 0.0)
        self.x1 = results.get('x_1', 0.0)
        self.a_truck = results.get('a', 0.0)  # This is stored as positive magnitude
        self.x2 = results.get('x_2', 0.0)
        self.a_car = results.get('a_prime', 0.0)  # Key is 'a_prime', not "a'"
        self.delta_x = results.get('Deltax', 0.0)

        # Precompute derived quantities for checking
        self.total_time_before_braking = self.t + self.dt  # 4.0 s

    def test_invariants(self):
        """Verify key physical invariants based on kinematics theorems."""
        logs = []
        try:
            # 1. Truck stopping condition: v1^2 = 2 * a_truck * truck_stop_dist
            # a_truck is stored as positive magnitude
            expected_a_truck_mag = self.v1**2 / (2 * self.truck_stop_dist)
            if not math.isclose(self.a_truck, expected_a_truck_mag, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Truck deceleration invariant. a_truck = {self.a_truck:.3f}, expected {expected_a_truck_mag:.3f}")
            else:
                logs.append(f"PASS: Truck deceleration magnitude correct.")

            # 2. Relative motion condition for critical avoidance
            # a_truck and a_car are positive magnitudes
            if abs(self.delta_x) > 1e-9:
                expected_a_car = self.a_truck + (self.v0 - self.v2)**2 / (2 * self.delta_x)
                if not math.isclose(self.a_car, expected_a_car, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Critical avoidance condition. a_car = {self.a_car:.3f}, expected {expected_a_car:.3f}")
                else:
                    logs.append(f"PASS: Critical avoidance condition satisfied.")
            else:
                logs.append("SKIP: delta_x is zero, cannot check avoidance condition.")

            # 3. Speed equality time t' from step 6: v0 - a_car*t' = v2 - a_truck*t'
            # Using magnitudes: v0 - a_car*t' = v2 - a_truck*t'
            if self.a_car != self.a_truck:
                t_prime_calc = (self.v0 - self.v2) / (self.a_car - self.a_truck)
                if not math.isclose(self.t_prime, t_prime_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: t' from speed equality. t' = {self.t_prime:.3f}, expected {t_prime_calc:.3f}")
                else:
                    logs.append(f"PASS: t' consistent with speed equality.")
            else:
                logs.append("SKIP: a_car == a_truck, infinite t'.")

        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
        return logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with formulas from steps."""
        logs = []
        try:
            # Step 1: Truck deceleration magnitude (positive)
            a_truck_calc = self.v1**2 / (2 * self.truck_stop_dist)
            if not math.isclose(self.a_truck, a_truck_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 1 truck a. a = {self.a_truck:.3f}, expected {a_truck_calc:.3f}")

            # Step 2: x1 = v1*(t+dt) - 0.5*a_truck*(t+dt)^2 (using positive a_truck)
            x1_calc = self.v1 * self.total_time_before_braking - 0.5 * self.a_truck * self.total_time_before_braking**2
            if not math.isclose(self.x1, x1_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 2 x1. x1 = {self.x1:.3f}, expected {x1_calc:.3f}")

            # Step 3: x2 = v0*(t+dt)
            x2_calc = self.v0 * self.total_time_before_braking
            if not math.isclose(self.x2, x2_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 3 x2. x2 = {self.x2:.3f}, expected {x2_calc:.3f}")

            # Step 4: delta_x = x0 + x1 - x2
            delta_x_calc = self.x0 + self.x1 - self.x2
            if not math.isclose(self.delta_x, delta_x_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 4 delta_x. delta_x = {self.delta_x:.3f}, expected {delta_x_calc:.3f}")

            # Step 5: v2 = v1 - a_truck*(t+dt) (using positive a_truck)
            v2_calc = self.v1 - self.a_truck * self.total_time_before_braking
            if not math.isclose(self.v2, v2_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Step 5 v2. v2 = {self.v2:.3f}, expected {v2_calc:.3f}")

            # If all pass
            if all('FAIL' not in log for log in logs[-5:] if 'FAIL' in log):
                logs.append("PASS: All intermediate steps consistent.")

        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
        return logs

    def test_monotonicity(self):
        """
        Perturb an input (initial distance x0) and check that required a_car changes in expected direction.
        We'll temporarily increase x0, which should decrease the required deceleration a_car.
        """
        logs = []
        original_x0 = self.x0
        try:
            # Perturb: increase x0 by 10%
            perturbed_x0 = original_x0 * 1.10

            # Recompute delta_x with perturbed x0 (keeping other intermediates fixed)
            perturbed_delta_x = perturbed_x0 + self.x1 - self.x2

            # Compute required a_car for perturbed case using derived formula
            if perturbed_delta_x > 0:
                perturbed_a_car = self.a_truck + (self.v0 - self.v2)**2 / (2 * perturbed_delta_x)

                # Check monotonicity: if x0 increases, delta_x increases, so a_car should decrease
                if perturbed_a_car < self.a_car:
                    logs.append(f"PASS: Monotonicity holds. Increased x0 leads to smaller a_car ({perturbed_a_car:.3f} < {self.a_car:.3f}).")
                else:
                    logs.append(f"FAIL: Monotonicity violated. Increased x0 leads to a_car = {perturbed_a_car:.3f}, not less than {self.a_car:.3f}.")
            else:
                logs.append("SKIP: Perturbed delta_x non-positive, cannot compute a_car.")

        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
        finally:
            # Restore original x0 (though self.x0 is just a local copy)
            self.x0 = original_x0
        return logs

    def run_all_tests(self):
        """Run all tests and return success flag and logs."""
        logs = []
        logs.append("=== Physics Checker Started ===")

        # Run tests
        logs.extend(self.test_invariants())
        logs.extend(self.test_trace_consistency())
        logs.extend(self.test_monotonicity())

        # Determine overall success: no FAIL or ERROR entries
        success = True
        for log in logs:
            if log.startswith("FAIL") or "ERROR" in log:
                success = False
                break

        logs.append(f"=== Physics Checker Finished: {'SUCCESS' if success else 'FAILED'} ===")
        return success, logs