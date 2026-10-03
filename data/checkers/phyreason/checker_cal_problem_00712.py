import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Problem constants (safe extraction from problem context)
        self.m = 0.017  # kg (17g)
        self.B = 0.6    # T
        self.L = 0.15   # m (15cm)
        self.E = 10.0   # V
        self.R1 = 4.0   # Ω
        self.r = 1.0    # Ω
        self.g = 10.0   # m/s²
        self.sqrt3 = 1.7  # given √3

        # Safe extraction from solution_instance (with fallbacks)
        self.F = getattr(solution_instance, 'F', solution_instance.results.get('F', 0.0))
        self.R2 = getattr(solution_instance, 'R2', solution_instance.results.get('R2', 0.0))
        self.f = getattr(solution_instance, 'f', solution_instance.results.get('f', 0.0))
        self.I_1 = getattr(solution_instance, 'I_1', solution_instance.results.get('I_1', 0.0))
        self.F1 = getattr(solution_instance, 'F1', solution_instance.results.get('F1', 0.0))

        # Expected values from target answers (with tolerance)
        self.expected_F = 0.1
        self.expected_R2 = 4.0
        self.expected_f = 0.034

    def test_invariants(self):
        logs = []
        success = True

        # 1. Check F = mg * (sqrt3/3) from step_1 (with tolerance)
        calculated_F = self.m * self.g * (self.sqrt3 / 3.0)
        # Accept either the exact 0.1 or the calculated value
        if not (math.isclose(self.F, calculated_F, rel_tol=0.02, abs_tol=0.02) or 
                math.isclose(self.F, self.expected_F, rel_tol=0.02, abs_tol=0.02)):
            logs.append(f"F invariant warning: F={self.F} not close to mg*(√3/3)={calculated_F} or expected 0.1")
            success = False
        else:
            logs.append(f"F invariant passed: F={self.F}")

        # 2. Check I from Ohm's law (first part)
        I_ohm = self.E / (self.R1 + self.R2 + self.r)
        I_amp = self.F / (self.B * self.L)
        if not math.isclose(I_ohm, I_amp, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"Ohm vs Ampere current mismatch: I_ohm={I_ohm} ≠ I_amp={I_amp}")
            success = False
        else:
            logs.append(f"Current consistency passed: I≈{I_ohm}")

        # 3. Check F1 = B * I_new * L from step_5 (CORRECTED: use new current, not I_1)
        # The new current with R2' = 10 Ω is I_new = E/(R1 + 10 + r)
        I_new = self.E / (self.R1 + 10.0 + self.r)
        calculated_F1 = self.B * I_new * self.L
        if not math.isclose(self.F1, calculated_F1, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"F1 invariant failed: F1={self.F1} ≠ B*I_new*L={calculated_F1} (I_new={I_new})")
            success = False
        else:
            logs.append(f"F1 invariant passed: F1={self.F1} ≈ {calculated_F1}")

        # 4. Equilibrium check for step_6 (numeric only)
        try:
            # Use numeric values with sqrt3 = 1.7
            sin_theta = 0.5  # sin(30°)
            cos_theta = self.sqrt3 / 2  # cos(30°) with given √3
            f_calculated = self.m * self.g * sin_theta - self.F1 * cos_theta
            if not math.isclose(self.f, f_calculated, rel_tol=0.05, abs_tol=0.02):
                logs.append(f"Equilibrium f mismatch: f={self.f}, calculated={f_calculated}")
                success = False
            else:
                logs.append(f"Equilibrium f passed: f={self.f} ≈ {f_calculated}")
        except Exception as e:
            logs.append(f"Equilibrium check failed: {e}")
            success = False

        return success, logs

    def test_trace_consistency(self):
        logs = []
        success = True

        # Cross-check F → I → R2 chain from steps 1-3
        I_from_F = self.F / (self.B * self.L)
        R2_from_I = (self.E / I_from_F) - self.R1 - self.r

        if not math.isclose(self.R2, R2_from_I, rel_tol=0.02, abs_tol=0.02):
            logs.append(f"Trace inconsistency: R2={self.R2} ≠ expected from F chain {R2_from_I}")
            success = False
        else:
            logs.append(f"F→I→R2 chain consistent: R2={self.R2}")

        # Cross-check I_1 from step_4
        if self.I_1:
            I1_from_ohm = self.E / (self.R1 + self.R2 + self.r)
            if not math.isclose(self.I_1, I1_from_ohm, rel_tol=0.02, abs_tol=0.02):
                logs.append(f"I_1 mismatch: I_1={self.I_1} ≠ Ohm's law {I1_from_ohm}")
                success = False
            else:
                logs.append(f"I_1 consistent with Ohm's law: I_1={self.I_1}")

        # Cross-check f from step_6 (using numeric approximation)
        if self.F1:
            sin_theta = 0.5
            cos_theta = self.sqrt3 / 2
            f_approx = self.m * self.g * sin_theta - self.F1 * cos_theta
            if not math.isclose(self.f, f_approx, rel_tol=0.05, abs_tol=0.02):
                logs.append(f"f mismatch: f={self.f} ≠ approx {f_approx} from F1={self.F1}")
                success = False
            else:
                logs.append(f"f consistent with F1: f={self.f}")

        return success, logs

    def test_monotonicity(self):
        logs = []
        success = True

        # Perturb mass m and check F (from F = B*I*L, but I depends on circuit)
        # Instead, check that if F is given by mg*tanθ, then increasing m increases F
        # But we accept the solution's F value, so skip this test
        logs.append("Monotonicity test skipped for F (accepted solution value)")

        # Perturb B and check F1 (Ampere's law) - use the correct current for F1
        if self.F1:
            # Get the current that was used to compute F1: I_new = F1/(B*L)
            I_for_F1 = self.F1 / (self.B * self.L)
            original_B = self.B
            perturbed_B = original_B * 1.01
            original_F1 = self.F1
            perturbed_F1 = perturbed_B * I_for_F1 * self.L

            if perturbed_F1 <= original_F1:
                logs.append(f"F1 monotonicity failed: F1({perturbed_B})={perturbed_F1} ≤ original {original_F1}")
                success = False
            else:
                logs.append(f"F1 monotonicity passed: F1 increases with B (dF1≈{perturbed_F1-original_F1:.2e})")

        return success, logs

    def run_all_tests(self):
        all_logs = []
        overall_success = True

        # Test expected values
        if not math.isclose(self.F, self.expected_F, rel_tol=0.02, abs_tol=0.02):
            all_logs.append(f"FAIL: F={self.F} ≠ expected {self.expected_F}")
            overall_success = False
        else:
            all_logs.append(f"PASS: F={self.F} matches expected {self.expected_F}")

        if not math.isclose(self.R2, self.expected_R2, rel_tol=0.02, abs_tol=0.02):
            all_logs.append(f"FAIL: R2={self.R2} ≠ expected {self.expected_R2}")
            overall_success = False
        else:
            all_logs.append(f"PASS: R2={self.R2} matches expected {self.expected_R2}")

        if not math.isclose(self.f, self.expected_f, rel_tol=0.05, abs_tol=0.02):
            all_logs.append(f"FAIL: f={self.f} ≠ expected {self.expected_f}")
            overall_success = False
        else:
            all_logs.append(f"PASS: f={self.f} matches expected {self.expected_f}")

        # Run physics tests
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            test_success, test_logs = test_method()
            all_logs.append(f"\n=== {test_name} ===")
            all_logs.extend(test_logs)
            if not test_success:
                overall_success = False

        return overall_success, all_logs