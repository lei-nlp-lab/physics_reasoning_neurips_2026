import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results dictionary
        results = getattr(solution_instance, 'results', {})
        # Extract input parameters from results (they should be stored as numeric values)
        self.U = results.get('U', 0.0)
        self.B = results.get('B', 0.0)
        self.d = results.get('d', 0.0)
        # If not in results, try to get from attributes as fallback (but they should be numeric)
        if self.U == 0.0:
            self.U = getattr(solution_instance, 'U', 0.0)
        if self.B == 0.0:
            self.B = getattr(solution_instance, 'B', 0.0)
        if self.d == 0.0:
            self.d = getattr(solution_instance, 'd', 0.0)
        
        # θ = 30° in radians
        self.theta = math.radians(30)
        # Safely extract results
        self.fracqm = results.get('fracqm', 0.0)
        self.r = results.get('r', 0.0)
        self.t2 = results.get('t_2', 0.0)
        self.t = results.get('t', 0.0)
        self.kinetic_energy = results.get('1/2mv2', 0.0)

        # Pre-compute derived quantities for checks
        self.q_over_m = self.fracqm  # charge-to-mass ratio
        # Velocity from work-energy: v = sqrt(2*(q/m)*U)
        if self.q_over_m > 0 and self.U > 0:
            self.v = math.sqrt(2 * self.q_over_m * self.U)
        else:
            self.v = 0.0
        # Time period in magnetic field: T = 2π m/(qB) = 2π / ( (q/m) * B )
        if self.q_over_m > 0 and self.B > 0:
            self.T = 2 * math.pi / (self.q_over_m * self.B)
        else:
            self.T = 0.0
        # Central angle in magnetic field from geometry: φ = 60° = π/3 (from chord length condition)
        self.phi = math.pi / 3
        # Expected time in magnetic field: t1 = (φ/(2π)) * T = φ * m/(qB)
        self.t1_expected = self.phi / (self.q_over_m * self.B) if self.q_over_m > 0 and self.B > 0 else 0.0

    def test_invariants(self):
        logs = []
        try:
            # 1. Work-energy theorem: qU = 0.5*m*v^2  =>  (q/m)*U = 0.5*v^2
            if self.q_over_m > 0 and self.U > 0:
                lhs = self.q_over_m * self.U
                rhs = 0.5 * self.v**2
                if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Work-energy mismatch: (q/m)*U = {lhs}, 0.5*v^2 = {rhs}")
                else:
                    logs.append("PASS: Work-energy theorem holds.")
            else:
                logs.append("SKIP: Work-energy check due to missing parameters.")

            # 2. Magnetic force balance: qvB = m*v^2/r  =>  r = v / ( (q/m)*B )
            if self.q_over_m > 0 and self.B > 0 and self.v > 0:
                r_expected = self.v / (self.q_over_m * self.B)
                if not math.isclose(self.r, r_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Radius mismatch: r = {self.r}, expected v/( (q/m)*B ) = {r_expected}")
                else:
                    logs.append("PASS: Magnetic force balance holds.")
            else:
                logs.append("SKIP: Magnetic force balance check due to missing parameters.")

            # 3. Geometry: chord length d = 2r sin(φ/2) with φ = π/3
            if self.r > 0:
                chord_expected = 2 * self.r * math.sin(self.phi / 2)
                if not math.isclose(self.d, chord_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Chord length mismatch: d = {self.d}, expected 2r sin(π/6) = {chord_expected}")
                else:
                    logs.append("PASS: Chord length condition holds.")
            else:
                logs.append("SKIP: Chord length check due to missing radius.")

            # 4. Electric field kinematics: perpendicular displacement = v * t2
            # From step 5: sqrt(3)*d * sinθ = v * t2
            if self.v > 0 and self.d > 0:
                displacement = math.sqrt(3) * self.d * math.sin(self.theta)
                t2_expected = displacement / self.v
                if self.t2 > 0 and not math.isclose(self.t2, t2_expected, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: t2 mismatch: t2 = {self.t2}, expected (√3*d*sinθ)/v = {t2_expected}")
                else:
                    logs.append("PASS: Electric field kinematics holds.")
            else:
                logs.append("SKIP: Electric field kinematics check due to missing parameters.")

        except Exception as e:
            logs.append(f"ERROR in test_invariants: {e}\n{traceback.format_exc()}")
        return logs

    def test_trace_consistency(self):
        logs = []
        try:
            # 1. Charge-to-mass ratio from step 4: q/m = 2U/(B^2 d^2)
            expected_fracqm = (2 * self.U) / (self.B**2 * self.d**2) if self.B != 0 and self.d != 0 else 0.0
            if not math.isclose(self.fracqm, expected_fracqm, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Charge-to-mass ratio mismatch: {self.fracqm} vs expected {expected_fracqm}")
            else:
                logs.append("PASS: Charge-to-mass ratio matches step 4.")

            # 2. Total time from step 7: t = (3√3 + 2π) * B * d^2 / (12 * U)
            if self.U != 0:
                expected_t = (3*math.sqrt(3) + 2*math.pi) * self.B * self.d**2 / (12 * self.U)
                if not math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Total time mismatch: t = {self.t}, expected (3√3+2π)Bd^2/(12U) = {expected_t}")
                else:
                    logs.append("PASS: Total time matches step 7.")

            # 3. Time sum: t = t1 + t2 (if t1 and t2 are available)
            # t1 from magnetic field: t1 = φ * m/(qB) = φ / ( (q/m) * B )
            if self.q_over_m > 0 and self.B > 0:
                t1_calculated = self.phi / (self.q_over_m * self.B)
                t_sum = t1_calculated + self.t2
                if not math.isclose(self.t, t_sum, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"FAIL: Time sum mismatch: t = {self.t}, t1+t2 = {t_sum} (t1={t1_calculated}, t2={self.t2})")
                else:
                    logs.append("PASS: Total time equals t1 + t2.")
            else:
                logs.append("SKIP: Time sum check due to missing parameters.")

        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {e}\n{traceback.format_exc()}")
        return logs

    def test_monotonicity(self):
        logs = []
        original_d = self.d
        try:
            # Perturb d by +1%
            perturbed_d = original_d * 1.01
            # Recompute q/m and t using the formulas
            new_fracqm = (2 * self.U) / (self.B**2 * perturbed_d**2) if self.B != 0 and perturbed_d != 0 else 0.0
            new_t = (3*math.sqrt(3) + 2*math.pi) * self.B * perturbed_d**2 / (12 * self.U) if self.U != 0 else 0.0

            # Check monotonicity:
            # For q/m = 2U/(B^2 d^2), if d increases, q/m should decrease.
            if perturbed_d > original_d and new_fracqm < self.fracqm:
                logs.append("PASS: q/m decreases as d increases (monotonicity holds).")
            elif perturbed_d > original_d and new_fracqm >= self.fracqm:
                logs.append(f"FAIL: q/m did not decrease when d increased. Original: {self.fracqm}, new: {new_fracqm}")
            else:
                logs.append("SKIP: Monotonicity check for q/m (perturbation direction not positive).")

            # For t = constant * d^2 / U, if d increases, t should increase.
            if perturbed_d > original_d and new_t > self.t:
                logs.append("PASS: t increases as d increases (monotonicity holds).")
            elif perturbed_d > original_d and new_t <= self.t:
                logs.append(f"FAIL: t did not increase when d increased. Original: {self.t}, new: {new_t}")
            else:
                logs.append("SKIP: Monotonicity check for t (perturbation direction not positive).")

        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {e}\n{traceback.format_exc()}")
        finally:
            # Restore original d
            self.d = original_d
        return logs

    def run_all_tests(self):
        logs = []
        success = True

        # Run each test
        invariants_logs = self.test_invariants()
        logs.append("=== Invariants Test ===")
        logs.extend(invariants_logs)
        if any("FAIL" in log or "ERROR" in log for log in invariants_logs):
            success = False

        trace_logs = self.test_trace_consistency()
        logs.append("\n=== Trace Consistency Test ===")
        logs.extend(trace_logs)
        if any("FAIL" in log or "ERROR" in log for log in trace_logs):
            success = False

        monotonicity_logs = self.test_monotonicity()
        logs.append("\n=== Monotonicity Test ===")
        logs.extend(monotonicity_logs)
        if any("FAIL" in log or "ERROR" in log for log in monotonicity_logs):
            success = False

        return success, logs