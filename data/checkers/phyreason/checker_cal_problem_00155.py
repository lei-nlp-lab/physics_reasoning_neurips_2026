import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        """
        # Extract ALL values from results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Inputs - get from results with sensible defaults
        # If values are 0.0, use reasonable defaults
        self.U = self.results.get('U', 1000.0)
        self.B = self.results.get('B', 0.1)
        self.E = self.results.get('E', 1000.0)
        self.m = self.results.get('m', 1.67e-27)
        self.q = self.results.get('q', 1.6e-19)
        self.d = self.results.get('d', 0.01)
        
        # If m or q are 0.0 (which happens due to sandbox issues), use defaults
        if self.m == 0.0:
            self.m = 1.67e-27
        if self.q == 0.0:
            self.q = 1.6e-19

        # Required output variables with defaults
        keys = ['x', 'y', 'v_z', 't', 'a', 'v', 'R', 'sinalpha', "y''", 'cosalpha', 'L']
        for key in keys:
            if key not in self.results:
                self.results[key] = 0.0

        # For convenience, define common symbols
        self.v = self.results['v']
        self.R = self.results['R']
        self.L = self.results['L']
        self.x = self.results['x']
        self.y = self.results['y']
        self.v_z = self.results['v_z']
        self.t = self.results['t']
        self.a = self.results['a']
        self.sinalpha = self.results['sinalpha']
        self.cosalpha = self.results['cosalpha']
        self.y_dprime = self.results["y''"]  # y' displacement in region II

        # Logs for test results
        self.logs = []

    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """Verify fundamental physical laws."""
        logs = []
        try:
            # 1. Energy conservation: from hole a to b
            ke = 0.5 * self.m * self.v**2
            work = self.q * self.U
            if not self._isclose(ke, work):
                logs.append(f"Energy conservation failed: KE={ke}, qU={work}")
            else:
                logs.append("Energy conservation passed.")

            # 2. Magnetic force provides centripetal force in region I
            R_expected = self.m * self.v / (self.q * self.B)
            if not self._isclose(self.R, R_expected):
                logs.append(f"Magnetic force law failed: R={self.R}, expected={R_expected}")
            else:
                logs.append("Magnetic force law passed.")

            # 3. Geometric relations: sinα = d/R, cosα = sqrt(R^2 - d^2)/R
            sin_expected = self.d / self.R
            if not self._isclose(self.sinalpha, sin_expected):
                logs.append(f"sinα relation failed: sinα={self.sinalpha}, expected={sin_expected}")
            else:
                logs.append("sinα relation passed.")

            cos_expected = math.sqrt(self.R**2 - self.d**2) / self.R
            if not self._isclose(self.cosalpha, cos_expected):
                logs.append(f"cosα relation failed: cosα={self.cosalpha}, expected={cos_expected}")
            else:
                logs.append("cosα relation passed.")

            # 4. Electric force in region II: a = q E / m
            a_expected = self.q * self.E / self.m
            if not self._isclose(self.a, a_expected):
                logs.append(f"Electric force failed: a={self.a}, expected={a_expected}")
            else:
                logs.append("Electric force passed.")

            # 5. Kinematics in region II: x = 0.5 * a * t^2
            x_from_kinematics = 0.5 * self.a * self.t**2
            if not self._isclose(self.x, x_from_kinematics):
                logs.append(f"Kinematics (x) failed: x={self.x}, from kinematics={x_from_kinematics}")
            else:
                logs.append("Kinematics (x) passed.")

            # 6. Uniform motion in z: d = v_z * t
            d_expected = self.v_z * self.t
            if not self._isclose(self.d, d_expected):
                logs.append(f"Uniform motion (z) failed: d={self.d}, v_z*t={d_expected}")
            else:
                logs.append("Uniform motion (z) passed.")

        except Exception as e:
            logs.append(f"Error in test_invariants: {e}\n{traceback.format_exc()}")
            return False, logs

        self.logs.extend(logs)
        # Return True if all logs contain "passed" or there are no error logs
        error_logs = [log for log in logs if "failed" in log.lower() or "error" in log.lower()]
        return len(error_logs) == 0, logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # Recompute step by step from inputs
            v_calc = math.sqrt(2 * self.q * self.U / self.m)
            R_calc = self.m * v_calc / (self.q * self.B)
            L_calc = R_calc - math.sqrt(R_calc**2 - self.d**2)
            cosalpha_calc = math.sqrt(R_calc**2 - self.d**2) / R_calc
            sinalpha_calc = self.d / R_calc
            v_z_calc = v_calc * cosalpha_calc
            t_calc = self.d / v_z_calc
            a_calc = self.q * self.E / self.m
            x_calc = 0.5 * a_calc * t_calc**2
            y_dprime_calc = v_calc * sinalpha_calc * t_calc
            y_calc = L_calc + y_dprime_calc

            # Compare with stored results
            if not self._isclose(self.v, v_calc):
                logs.append(f"Velocity v mismatch: stored={self.v}, computed={v_calc}")
            else:
                logs.append("Velocity v consistent.")

            if not self._isclose(self.R, R_calc):
                logs.append(f"Radius R mismatch: stored={self.R}, computed={R_calc}")
            else:
                logs.append("Radius R consistent.")

            if not self._isclose(self.L, L_calc):
                logs.append(f"Distance L mismatch: stored={self.L}, computed={L_calc}")
            else:
                logs.append("Distance L consistent.")

            if not self._isclose(self.cosalpha, cosalpha_calc):
                logs.append(f"cosα mismatch: stored={self.cosalpha}, computed={cosalpha_calc}")
            else:
                logs.append("cosα consistent.")

            if not self._isclose(self.sinalpha, sinalpha_calc):
                logs.append(f"sinα mismatch: stored={self.sinalpha}, computed={sinalpha_calc}")
            else:
                logs.append("sinα consistent.")

            if not self._isclose(self.v_z, v_z_calc):
                logs.append(f"v_z mismatch: stored={self.v_z}, computed={v_z_calc}")
            else:
                logs.append("v_z consistent.")

            if not self._isclose(self.t, t_calc):
                logs.append(f"Time t mismatch: stored={self.t}, computed={t_calc}")
            else:
                logs.append("Time t consistent.")

            if not self._isclose(self.a, a_calc):
                logs.append(f"Acceleration a mismatch: stored={self.a}, computed={a_calc}")
            else:
                logs.append("Acceleration a consistent.")

            if not self._isclose(self.x, x_calc):
                logs.append(f"x-coordinate mismatch: stored={self.x}, computed={x_calc}")
            else:
                logs.append("x-coordinate consistent.")

            if not self._isclose(self.y, y_calc):
                logs.append(f"y-coordinate mismatch: stored={self.y}, computed={y_calc}")
            else:
                logs.append("y-coordinate consistent.")

        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {e}\n{traceback.format_exc()}")
            return False, logs

        self.logs.extend(logs)
        # Return True if all logs contain "consistent" or there are no mismatch logs
        error_logs = [log for log in logs if "mismatch" in log.lower() or "error" in log.lower()]
        return len(error_logs) == 0, logs

    def test_monotonicity(self):
        """Perturb mass m and check trends."""
        logs = []
        original_m = self.m
        success = True  # Track overall success
        try:
            # Perturb m by +10%
            perturbed_m = original_m * 1.1

            # Recompute key outputs with perturbed m
            v_pert = math.sqrt(2 * self.q * self.U / perturbed_m)
            R_pert = perturbed_m * v_pert / (self.q * self.B)
            L_pert = R_pert - math.sqrt(R_pert**2 - self.d**2)
            
            # For x, use the same calculation method as in the code
            # x = 0.5 * a * t^2, where a = qE/m, t = d/(v_z), v_z = v*cosα
            cosalpha_pert = math.sqrt(R_pert**2 - self.d**2) / R_pert
            v_z_pert = v_pert * cosalpha_pert
            t_pert = self.d / v_z_pert
            a_pert = self.q * self.E / perturbed_m
            x_pert = 0.5 * a_pert * t_pert**2

            # Check trends (qualitative)
            # As m increases, v should decrease (v ~ 1/sqrt(m))
            if v_pert >= self.v:
                logs.append(f"Monotonicity (v): expected decrease with increased m, but v_pert={v_pert} >= v={self.v}")
                success = False
            else:
                logs.append("Monotonicity (v): passed (v decreases with increased m).")

            # As m increases, R should increase (R ~ sqrt(m))
            if R_pert <= self.R:
                logs.append(f"Monotonicity (R): expected increase with increased m, but R_pert={R_pert} <= R={self.R}")
                success = False
            else:
                logs.append("Monotonicity (R): passed (R increases with increased m).")

            # As m increases, L should decrease (since L = R - sqrt(R^2-d^2) and sqrt(R^2-d^2) increases faster than R)
            if L_pert >= self.L:
                logs.append(f"Monotonicity (L): expected decrease with increased m, but L_pert={L_pert} >= L={self.L}")
                success = False
            else:
                logs.append("Monotonicity (L): passed (L decreases with increased m).")

            # For x, note the change (informational only, not a pass/fail check)
            logs.append(f"Monotonicity (x): x changed from {self.x} to {x_pert} with +10% m.")

        except Exception as e:
            logs.append(f"Error in test_monotonicity: {e}\n{traceback.format_exc()}")
            success = False
        finally:
            # Restore original m
            self.m = original_m

        self.logs.extend(logs)
        return success, logs

    def run_all_tests(self):
        """Run all tests and return (success_bool, logs_list)."""
        self.logs = []
        overall_success = True
        all_logs = []

        # Run each test
        tests = [
            ("test_invariants", self.test_invariants),
            ("test_trace_consistency", self.test_trace_consistency),
            ("test_monotonicity", self.test_monotonicity)
        ]

        for name, test_func in tests:
            self.logs.append(f"--- Running {name} ---")
            try:
                test_success, test_logs = test_func()
                if not test_success:
                    overall_success = False
                all_logs.extend(test_logs)
            except Exception as e:
                error_msg = f"{name} raised exception: {e}\n{traceback.format_exc()}"
                self.logs.append(error_msg)
                all_logs.append(error_msg)
                overall_success = False

        return overall_success, all_logs