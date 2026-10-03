import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract all required quantities with safe fallbacks
        self.a = getattr(solution_instance, 'a', self.results.get('a', 0.0))
        self.v_bike = getattr(solution_instance, 'v_bike', self.results.get('v_bike', 0.0))
        self.t_max = getattr(solution_instance, 't', self.results.get('t', 0.0))
        self.x_bike_at_tmax = getattr(solution_instance, 'x_bike', self.results.get('x_bike', 0.0))
        self.x_car_at_tmax = getattr(solution_instance, 'x_car', self.results.get('x_car', 0.0))
        self.Delta_x = getattr(solution_instance, 'Deltax', self.results.get('Deltax', 0.0))
        self.t_catch = getattr(solution_instance, 't_11', self.results.get('t_11', 0.0))
        self.v_car_at_catch = getattr(solution_instance, 'v_car1', self.results.get('v_car1', 0.0))
        
        # Fixed initial separation from problem statement
        self.x0 = 8.0

    def test_invariants(self):
        logs = []
        try:
            # 1. Check car acceleration from x=2t^2
            if not math.isclose(self.a, 4.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Acceleration a should be 4 m/s², got {self.a}")
            else:
                logs.append("PASS: a = 4 m/s²")

            # 2. Check bicycle velocity from x=6t
            if not math.isclose(self.v_bike, 6.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Bicycle velocity should be 6 m/s, got {self.v_bike}")
            else:
                logs.append("PASS: v_bike = 6 m/s")

            # 3. At max‑distance time, car velocity must equal bicycle velocity
            v_car_at_tmax = self.a * self.t_max
            if not math.isclose(v_car_at_tmax, self.v_bike, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: At t_max, v_car = {v_car_at_tmax} ≠ v_bike = {self.v_bike}")
            else:
                logs.append("PASS: v_car(t_max) = v_bike")

            # 4. Verify positions at t_max
            if not math.isclose(self.x_bike_at_tmax, 9.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: x_bike(t_max) should be 9 m, got {self.x_bike_at_tmax}")
            else:
                logs.append("PASS: x_bike(t_max) = 9 m")

            if not math.isclose(self.x_car_at_tmax, 4.5, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: x_car(t_max) should be 4.5 m, got {self.x_car_at_tmax}")
            else:
                logs.append("PASS: x_car(t_max) = 4.5 m")

            # 5. Maximum distance formula: Δx = x0 + v_bike t - ½ a t²
            expected_Delta_x = self.x0 + self.v_bike * self.t_max - 0.5 * self.a * self.t_max**2
            if not math.isclose(self.Delta_x, expected_Delta_x, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Δx should be {expected_Delta_x} m, got {self.Delta_x}")
            else:
                logs.append("PASS: Δx matches kinematic formula")

            # 6. Catch‑up time
            if not math.isclose(self.t_catch, 4.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Catch‑up time should be 4 s, got {self.t_catch}")
            else:
                logs.append("PASS: t_catch = 4 s")

            # 7. Positions equal at catch‑up time
            x_car_catch = 0.5 * self.a * self.t_catch**2
            x_bike_catch = self.x0 + self.v_bike * self.t_catch
            if not math.isclose(x_car_catch, x_bike_catch, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: At t_catch, x_car = {x_car_catch} ≠ x_bike = {x_bike_catch}")
            else:
                logs.append("PASS: Positions equal at catch‑up")

            # 8. Car velocity at catch‑up
            v_car_catch_computed = self.a * self.t_catch
            if not math.isclose(self.v_car_at_catch, v_car_catch_computed, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_car1 should be {v_car_catch_computed} m/s, got {self.v_car_at_catch}")
            else:
                logs.append("PASS: v_car1 = a * t_catch")

        except Exception as e:
            logs.append(f"ERROR in test_invariants: {e}\n{traceback.format_exc()}")
        return logs

    def test_trace_consistency(self):
        logs = []
        try:
            # x_bike(t_max) = v_bike * t_max
            computed_x_bike = self.v_bike * self.t_max
            if not math.isclose(self.x_bike_at_tmax, computed_x_bike, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: x_bike should be {computed_x_bike} m, got {self.x_bike_at_tmax}")
            else:
                logs.append("PASS: x_bike consistent with v_bike and t_max")

            # x_car(t_max) = ½ a t_max²
            computed_x_car = 0.5 * self.a * self.t_max**2
            if not math.isclose(self.x_car_at_tmax, computed_x_car, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: x_car should be {computed_x_car} m, got {self.x_car_at_tmax}")
            else:
                logs.append("PASS: x_car consistent with a and t_max")

            # Δx = x0 + x_bike - x_car
            computed_Delta_x = self.x0 + self.x_bike_at_tmax - self.x_car_at_tmax
            if not math.isclose(self.Delta_x, computed_Delta_x, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Δx should be {computed_Delta_x} m, got {self.Delta_x}")
            else:
                logs.append("PASS: Δx consistent with x0, x_bike, x_car")

            # v_car1 = a * t_catch
            computed_v_car1 = self.a * self.t_catch
            if not math.isclose(self.v_car_at_catch, computed_v_car1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_car1 should be {computed_v_car1} m/s, got {self.v_car_at_catch}")
            else:
                logs.append("PASS: v_car1 consistent with a and t_catch")

        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {e}\n{traceback.format_exc()}")
        return logs

    def test_monotonicity(self):
        logs = []
        try:
            # Perturb bicycle velocity by +10%
            factor = 1.1
            v_bike_new = self.v_bike * factor
            a = self.a
            x0 = self.x0

            # 1. New t_max = v_bike_new / a
            t_max_new = v_bike_new / a
            if not t_max_new > self.t_max:
                logs.append(f"FAIL: t_max should increase when v_bike increases. Was {self.t_max}, now {t_max_new}")
            else:
                logs.append("PASS: t_max increases with v_bike")

            # 2. New D_max = x0 + v_bike_new²/(2a)
            D_max_new = x0 + v_bike_new**2 / (2 * a)
            if not D_max_new > self.Delta_x:
                logs.append(f"FAIL: D_max should increase when v_bike increases. Was {self.Delta_x}, now {D_max_new}")
            else:
                logs.append("PASS: D_max increases with v_bike")

            # 3. New catch‑up time: solve ½ a t² = x0 + v_bike_new t
            disc = v_bike_new**2 + 2 * a * x0
            if disc < 0:
                logs.append("ERROR: Negative discriminant in catch‑up equation")
            else:
                t_catch_new = (v_bike_new + math.sqrt(disc)) / a
                if not t_catch_new > self.t_catch:
                    logs.append(f"FAIL: t_catch should increase when v_bike increases. Was {self.t_catch}, now {t_catch_new}")
                else:
                    logs.append("PASS: t_catch increases with v_bike")

                # 4. New car velocity at catch‑up
                v_catch_new = a * t_catch_new
                if not v_catch_new > self.v_car_at_catch:
                    logs.append(f"FAIL: v_catch should increase when v_bike increases. Was {self.v_car_at_catch}, now {v_catch_new}")
                else:
                    logs.append("PASS: v_catch increases with v_bike")

        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {e}\n{traceback.format_exc()}")
        return logs

    def run_all_tests(self):
        all_logs = ["=== PhysicsChecker starting ==="]
        
        all_logs.append("--- test_invariants ---")
        all_logs.extend(self.test_invariants())
        
        all_logs.append("--- test_trace_consistency ---")
        all_logs.extend(self.test_trace_consistency())
        
        all_logs.append("--- test_monotonicity ---")
        all_logs.extend(self.test_monotonicity())
        
        # Determine overall success
        success = True
        for log in all_logs:
            if log.startswith("FAIL") or log.startswith("ERROR"):
                success = False
                break
        
        all_logs.append(f"=== Overall success: {success} ===")
        return success, all_logs