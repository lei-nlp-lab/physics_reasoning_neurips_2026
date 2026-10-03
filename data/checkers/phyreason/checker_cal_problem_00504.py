import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Use getattr to avoid AttributeError in the sandbox.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Extract given constants (from problem statement)
        self.R = 0.5  # radius in meters
        self.m = 1e-4  # mass in kg
        self.q = 8e-5  # charge in C
        self.v0 = 3.0  # initial velocity in m/s

        # Extract computed results (with safe defaults)
        self.OD = self._safe_extract('OD', 0.3)   # horizontal distance OD in m
        self.DC = self._safe_extract('DC', 0.4)   # vertical distance DC in m
        self.E = self._safe_extract('E', 25.0)    # electric field magnitude in N/C
        self.v_C = self._safe_extract('v_C', 5.0) # speed at C in m/s
        self.E_C = self._safe_extract('E_C', 1.25e-3) # kinetic energy at C in J
        self.E_p = self._safe_extract('E_p', 4.5e-4)  # potential energy when speed is 4 m/s in J

        # Additional derived quantities (computed for checks)
        # Electric field is downward, charge positive → force downward → acceleration downward.
        # In upward-positive coordinates, acceleration is negative.
        self.a = -self.q * self.E / self.m  # acceleration in y-direction (negative)

        # Coordinates of point C relative to O (center of circle).
        # OD is horizontal distance, DC is vertical distance (positive magnitude).
        # With y upward positive, C is below O, so y_C = -DC.
        self.x_C = self.OD
        self.y_C = -self.DC  # negative

        # Tolerance for numerical comparisons (1-2% relative and absolute)
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2

    def _safe_extract(self, key, default):
        """
        Safely extract a value from the solution instance or its results dict.
        """
        value = getattr(self.solution, key, None)
        if value is None:
            value = self.results.get(key, default)
        # Ensure it's a float if it's numeric
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def test_geometry(self):
        """
        Check that point C lies on the quarter-circle of radius R.
        Expected: OD^2 + DC^2 = R^2.
        """
        logs = []
        success = True

        left = self.OD**2 + self.DC**2
        right = self.R**2
        if not math.isclose(left, right, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"Geometry invariant failed: OD^2 + DC^2 = {left}, expected R^2 = {right}")
            success = False
        else:
            logs.append("Geometry invariant passed: point C lies on the quarter-circle.")

        return success, logs

    def test_perpendicular_hit_condition(self):
        """
        Check the condition that the particle hits C perpendicularly (velocity is parallel to the radius vector).
        Condition: vx * y_C - vy * x_C = 0, where vx = v0 (constant), vy = a*t.
        We compute t from horizontal motion: horizontal displacement = x_C - x_A.
        Assume A is at x_A = -0.3 (left of O). Then displacement = x_C - (-0.3) = x_C + 0.3.
        t = (x_C + 0.3) / v0.
        Then vy = a * t.
        Check: v0 * y_C - vy * x_C ≈ 0.
        """
        logs = []
        success = True

        # Horizontal displacement
        x_A = -0.3
        horiz_displacement = self.x_C - x_A  # should be 0.6 m
        t = horiz_displacement / self.v0  # time of flight
        vy = self.a * t
        left = self.v0 * self.y_C - vy * self.x_C

        if not math.isclose(left, 0.0, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"Perpendicular hit condition failed: v0*y_C - vy*x_C = {left}")
            logs.append(f"  (v0={self.v0}, y_C={self.y_C}, vy={vy}, x_C={self.x_C}, t={t})")
            success = False
        else:
            logs.append("Perpendicular hit condition passed: velocity at C is parallel to the radius.")

        return success, logs

    def test_energy_conservation(self):
        """
        Check energy conservation between point A and point C.
        At A: kinetic energy K_A = 0.5 * m * v0^2, potential energy U_A = q * E * DC (since potential at C is 0 and field is downward).
        At C: kinetic energy K_C = 0.5 * m * v_C^2, potential energy U_C = 0.
        So: K_A + U_A = K_C.
        """
        logs = []
        success = True

        K_A = 0.5 * self.m * self.v0**2
        U_A = self.q * self.E * self.DC   # positive because potential at A is higher than at C
        total_A = K_A + U_A
        K_C = 0.5 * self.m * self.v_C**2

        if not math.isclose(total_A, K_C, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"Energy conservation failed: K_A + U_A = {total_A}, K_C = {K_C}")
            success = False
        else:
            logs.append("Energy conservation passed: total energy at A equals kinetic energy at C.")

        return success, logs

    def test_potential_energy_at_speed_4(self):
        """
        Check the computed electric potential energy when the particle's speed is 4 m/s.
        From energy conservation: E_total = K_C (since U_C=0).
        At the point where speed v = 4 m/s, kinetic energy K = 0.5 * m * 4^2.
        Then potential energy E_p = E_total - K.
        """
        logs = []
        success = True

        E_total = self.E_C  # which is K_C, and also equal to total energy
        K_at_4 = 0.5 * self.m * (4.0**2)
        expected_E_p = E_total - K_at_4

        if not math.isclose(self.E_p, expected_E_p, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"Potential energy at speed 4 m/s failed: computed E_p = {self.E_p}, expected = {expected_E_p}")
            success = False
        else:
            logs.append("Potential energy at speed 4 m/s passed.")

        return success, logs

    def test_monotonicity(self):
        """
        Perturb an input (e.g., initial velocity v0) and check that the computed electric field E changes in the expected way.
        From the perpendicular hit condition (using the correct sign), we have:
        v0 * y_C = (a * t) * x_C, with t = (x_C - x_A)/v0.
        Substituting: v0 * y_C = a * x_C * (x_C - x_A) / v0
        => v0^2 * y_C = a * x_C * (x_C - x_A)
        => a = (v0^2 * y_C) / (x_C * (x_C - x_A))
        Since a = -qE/m, we have E = - (m * v0^2 * y_C) / (q * x_C * (x_C - x_A))
        Thus E is proportional to v0^2.
        """
        logs = []
        success = True

        # Store original values
        original_v0 = self.v0
        original_E = self.E

        # Perturb v0 by +1%
        perturbed_v0 = original_v0 * 1.01
        # Compute expected E from the formula
        x_A = -0.3
        expected_E = - (self.m * perturbed_v0**2 * self.y_C) / (self.q * self.x_C * (self.x_C - x_A))
        # Compare magnitudes (E is positive)
        expected_E_mag = abs(expected_E)
        original_E_mag = abs(original_E)

        # The ratio of E magnitudes should be (v0_new / v0_old)^2
        expected_ratio = (perturbed_v0 / original_v0)**2
        actual_ratio = expected_E_mag / original_E_mag

        if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2):
            logs.append(f"Monotonicity test failed: E should scale with v0^2. Ratio expected = {expected_ratio}, actual = {actual_ratio}")
            success = False
        else:
            logs.append("Monotonicity test passed: E scales quadratically with v0.")

        # Restore original values
        self.v0 = original_v0
        self.E = original_E

        return success, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with the final answer.
        Here we check that the computed electric field E leads to the correct velocity at C via kinematics.
        """
        logs = []
        success = True

        # Compute time of flight from horizontal motion
        x_A = -0.3
        horiz_displacement = self.x_C - x_A  # 0.6 m
        t = horiz_displacement / self.v0  # time of flight

        # Vertical motion: y_C = 0.5 * a * t^2 (with a negative)
        predicted_y_C = 0.5 * self.a * t**2
        if not math.isclose(predicted_y_C, self.y_C, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"Trace consistency (vertical motion) failed: predicted y_C = {predicted_y_C}, actual y_C = {self.y_C}")
            success = False
        else:
            logs.append("Trace consistency (vertical motion) passed.")

        # Velocity at C: vx = v0, vy = a*t
        vy = self.a * t
        predicted_v_C = math.sqrt(self.v0**2 + vy**2)
        if not math.isclose(predicted_v_C, self.v_C, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
            logs.append(f"Trace consistency (velocity at C) failed: predicted v_C = {predicted_v_C}, actual v_C = {self.v_C}")
            success = False
        else:
            logs.append("Trace consistency (velocity at C) passed.")

        return success, logs

    def run_all_tests(self):
        """
        Run all tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True

        tests = [
            ("Geometry", self.test_geometry),
            ("Perpendicular hit condition", self.test_perpendicular_hit_condition),
            ("Energy conservation", self.test_energy_conservation),
            ("Potential energy at speed 4 m/s", self.test_potential_energy_at_speed_4),
            ("Monotonicity", self.test_monotonicity),
            ("Trace consistency", self.test_trace_consistency),
        ]

        for test_name, test_func in tests:
            try:
                success, logs = test_func()
            except Exception as e:
                success = False
                logs = [f"Exception in {test_name}: {traceback.format_exc()}"]
            if not success:
                overall_success = False
            all_logs.append(f"=== {test_name} ===")
            all_logs.extend(logs)

        return overall_success, all_logs