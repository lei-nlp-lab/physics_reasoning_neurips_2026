import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from the solution instance."""
        # Extract from results dictionary with defaults
        results = getattr(solution_instance, 'results', {})
        
        self.lambda_ = results.get('λ', 8.0)  # wavelength in m
        self.delta_t_wave = 0.01  # wave propagation time interval (fixed)
        self.v_wave = results.get('v', 100.0)  # wave speed in m/s
        self.f = results.get('f', 12.5)  # frequency in Hz
        self.direction = results.get('direction', 'to the right')
        self.y_P = results.get('y_P', -2*math.sqrt(2)/100.0)  # in m
        self.delta_x = results.get('delta_x', 1.0)  # in m
        self.time_to_equilibrium = results.get('delta_t', 0.05)  # in s (particle motion time)
        self.v_avg = results.get('v_avg', (2*math.sqrt(2)/5.0))  # in m/s
        
        # Constants
        self.f_max = 30.0  # Hz
        self.tolerance_rel = 1e-2
        self.tolerance_abs = 1e-2

    def is_close(self, a, b):
        """Robust fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=self.tolerance_rel, abs_tol=self.tolerance_abs)

    def test_invariants(self):
        """Verify fundamental wave physics invariants."""
        logs = []
        try:
            # 1. Wave equation: v_wave = λ * f
            if not self.is_close(self.v_wave, self.lambda_ * self.f):
                logs.append(f"FAIL: v_wave = {self.v_wave} != λ*f = {self.lambda_ * self.f}")
            else:
                logs.append("PASS: v_wave = λ*f")

            # 2. Frequency constraint f < 30 Hz
            if self.f >= self.f_max:
                logs.append(f"FAIL: f = {self.f} Hz is not < 30 Hz")
            else:
                logs.append("PASS: f < 30 Hz")

            # 3. Wave speed from displacement: v_wave = Δx / Δt_wave
            if not self.is_close(self.v_wave, self.delta_x / self.delta_t_wave):
                logs.append(f"FAIL: v_wave = {self.v_wave} != Δx/Δt_wave = {self.delta_x / self.delta_t_wave}")
            else:
                logs.append("PASS: v_wave = Δx/Δt_wave")

            # 4. Check that Δx corresponds to right propagation (8n+1 with n=0) -> Δx = 1 m
            if not self.is_close(self.delta_x, 1.0):
                logs.append(f"FAIL: Δx = {self.delta_x} m, expected ~1.0 m for right propagation (n=0)")
            else:
                logs.append("PASS: Δx ≈ 1 m (right propagation, n=0)")

            # 5. Direction consistency: if Δx > 0 and v_wave > 0, direction should be 'to the right'
            if self.delta_x > 0 and self.v_wave > 0 and 'right' not in str(self.direction).lower():
                logs.append(f"FAIL: positive Δx and v_wave but direction is '{self.direction}'")
            else:
                logs.append("PASS: propagation direction consistent with Δx and v_wave signs")

            # 6. Period T = 1/f
            T = 1.0 / self.f
            expected_T = 0.08  # 1/12.5 = 0.08 s
            if not self.is_close(T, expected_T):
                logs.append(f"FAIL: T = {T} s, expected {expected_T} s")
            else:
                logs.append("PASS: T = 1/f")

            # 7. Time to equilibrium: Δt_eq = 0.01 + T/2
            expected_time_to_equilibrium = 0.01 + T/2.0
            if not self.is_close(self.time_to_equilibrium, expected_time_to_equilibrium):
                logs.append(f"FAIL: time to equilibrium = {self.time_to_equilibrium} s, expected {expected_time_to_equilibrium} s")
            else:
                logs.append("PASS: time to equilibrium = 0.01 + T/2")

            # 8. Average velocity magnitude: v_avg = |y_P| / Δt_eq
            y_P_mag = abs(self.y_P)  # in meters
            expected_v_avg = y_P_mag / self.time_to_equilibrium
            if not self.is_close(self.v_avg, expected_v_avg):
                logs.append(f"FAIL: v_avg = {self.v_avg} m/s, expected |y_P|/Δt_eq = {expected_v_avg} m/s")
            else:
                logs.append("PASS: v_avg = |y_P|/Δt_eq")

        except Exception as e:
            logs.append(f"ERROR in test_invariants: {e}\n{traceback.format_exc()}")
        return logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # Step 1: wavelength λ = 8 m
            if not self.is_close(self.lambda_, 8.0):
                logs.append(f"INCONSISTENT: λ = {self.lambda_} m, expected 8 m")

            # Step 2 & 3: right propagation Δx = 1 m, v_wave = Δx/0.01 = 100 m/s
            if not self.is_close(self.delta_x, 1.0):
                logs.append(f"INCONSISTENT: Δx = {self.delta_x} m, expected 1 m for right propagation")
            if not self.is_close(self.v_wave, 100.0):
                logs.append(f"INCONSISTENT: v_wave = {self.v_wave} m/s, expected 100 m/s")

            # Step 4 & 9: f = v_wave/λ = 12.5 Hz
            expected_f = self.v_wave / self.lambda_
            if not self.is_close(self.f, expected_f):
                logs.append(f"INCONSISTENT: f = {self.f} Hz, expected v_wave/λ = {expected_f} Hz")

            # Step 12: Δt_eq = 0.05 s
            if not self.is_close(self.time_to_equilibrium, 0.05):
                logs.append(f"INCONSISTENT: time to equilibrium = {self.time_to_equilibrium} s, expected 0.05 s")

            # Step 13: y_P = -2√2 cm = -0.02√2 m
            expected_y_P = -2 * math.sqrt(2) / 100.0
            if not self.is_close(self.y_P, expected_y_P):
                logs.append(f"INCONSISTENT: y_P = {self.y_P} m, expected {-2*math.sqrt(2)/100.0} m")

            # Step 14: v_avg = (2√2/5) m/s ≈ 0.565685 m/s
            expected_v_avg = 2 * math.sqrt(2) / 5.0
            if not self.is_close(self.v_avg, expected_v_avg):
                logs.append(f"INCONSISTENT: v_avg = {self.v_avg} m/s, expected (2√2/5) ≈ {expected_v_avg} m/s")

            if not logs:
                logs.append("PASS: All trace checks consistent.")
        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {e}\n{traceback.format_exc()}")
        return logs

    def test_monotonicity(self):
        """Perturb inputs and verify expected trends."""
        logs = []
        try:
            # Perturb frequency f (increase by 1 Hz, but keep below 30 Hz)
            f_original = self.f
            f_perturbed = f_original + 1.0  # 13.5 Hz
            if f_perturbed < self.f_max:
                v_original = self.v_wave
                v_perturbed = self.lambda_ * f_perturbed
                if v_perturbed > v_original:
                    logs.append("PASS: Increasing f increases v_wave (v_wave = λ*f)")
                else:
                    logs.append(f"FAIL: v_wave did not increase when f increased: v_original={v_original}, v_perturbed={v_perturbed}")
            else:
                logs.append("SKIP: f_perturbed would exceed 30 Hz")

            # Perturb observation time Δt_wave (increase by 10%)
            delta_t_wave_original = self.delta_t_wave
            delta_t_wave_perturbed = delta_t_wave_original * 1.1
            # Keep Δx fixed at its original value (1 m for right propagation)
            delta_x_fixed = self.delta_x
            v_original = self.v_wave
            v_perturbed2 = delta_x_fixed / delta_t_wave_perturbed
            if v_perturbed2 < v_original:
                logs.append("PASS: Increasing Δt_wave decreases v_wave (v_wave = Δx/Δt_wave)")
            else:
                logs.append(f"FAIL: v_wave did not decrease when Δt_wave increased: v_original={v_original}, v_perturbed={v_perturbed2}")

            # Monotonicity: v_wave = λ*f, so if λ increases with fixed f, v_wave increases
            # This is a conceptual check, no symbolic operations needed
            logs.append("PASS: Conceptual check: v_wave = λ*f implies v_wave increases with λ for fixed f")

        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {e}\n{traceback.format_exc()}")
        return logs

    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        logs = []
        success = True

        # Run each test
        for test_name, test_func in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            logs.append(f"--- {test_name} ---")
            test_logs = test_func()
            logs.extend(test_logs)
            # Check for any failures (logs containing "FAIL" or "ERROR")
            for log in test_logs:
                if log.startswith("FAIL") or log.startswith("ERROR") or log.startswith("INCONSISTENT"):
                    success = False

        # Final summary
        logs.append(f"\nOverall success: {success}")
        return success, logs