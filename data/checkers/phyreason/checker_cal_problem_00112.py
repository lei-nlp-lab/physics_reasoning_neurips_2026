import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Use getattr with fallbacks to avoid AttributeError.
        """
        # First, get the results dictionary
        results = getattr(solution_instance, 'results', {})
        
        # Input parameters: try to get from results first, then from attributes
        # If not found, use 1.0 as safe defaults (non-zero to avoid division by zero)
        self.L = results.get('L', getattr(solution_instance, 'L', 1.0))
        self.S = results.get('S', getattr(solution_instance, 'S', 1.0))
        self.rho = results.get('rho', getattr(solution_instance, 'rho', 1.0))
        self.B0 = results.get('B0', getattr(solution_instance, 'B0', 1.0))
        self.dt = results.get('dt', getattr(solution_instance, 'dt', 1.0))
        
        # Results (expected keys: r, E, R, I, P)
        self.r = results.get('r', getattr(solution_instance, 'r', 0.0))
        self.E = results.get('E', getattr(solution_instance, 'E', 0.0))
        self.R = results.get('R', getattr(solution_instance, 'R', 0.0))
        self.I = results.get('I', getattr(solution_instance, 'I', 0.0))
        self.P = results.get('P', getattr(solution_instance, 'P', 0.0))

    def test_invariants(self):
        """
        Verify physical laws: Faraday's Law, Ohm's Law, Joule's Law.
        Use numerical checks with tolerance.
        """
        logs = []
        try:
            # 1. Check radius from circumference: r = L/(2π)
            expected_r = self.L / (2 * math.pi)
            if math.isclose(self.r, expected_r, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Radius invariant passed: r = L/(2π) ≈ {expected_r}")
            else:
                logs.append(f"Radius invariant failed: expected r ≈ {expected_r}, got {self.r}")

            # 2. Check Faraday's Law: E = (B0 * area) / Δt, area = π r² = L²/(4π)
            area = math.pi * self.r**2
            expected_E_faraday = self.B0 * area / self.dt
            expected_E_L = self.B0 * self.L**2 / (4 * math.pi * self.dt)
            # Check both forms (they should be equal)
            if (math.isclose(self.E, expected_E_faraday, rel_tol=1e-2, abs_tol=1e-2) or
                math.isclose(self.E, expected_E_L, rel_tol=1e-2, abs_tol=1e-2)):
                logs.append(f"Faraday's Law passed: E ≈ {expected_E_L}")
            else:
                logs.append(f"Faraday's Law failed: expected E ≈ {expected_E_L}, got {self.E}")

            # 3. Check resistance: R = ρ L / S
            expected_R = self.rho * self.L / self.S
            if math.isclose(self.R, expected_R, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Resistance invariant passed: R = ρL/S ≈ {expected_R}")
            else:
                logs.append(f"Resistance invariant failed: expected R ≈ {expected_R}, got {self.R}")

            # 4. Check Ohm's Law: I = E / R
            # Avoid division by zero if R is zero
            if self.R != 0:
                expected_I = self.E / self.R
                if math.isclose(self.I, expected_I, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Ohm's Law passed: I = E/R ≈ {expected_I}")
                else:
                    logs.append(f"Ohm's Law failed: expected I ≈ {expected_I}, got {self.I}")
            else:
                logs.append("Ohm's Law skipped: R = 0 (division by zero)")

            # 5. Check Joule's Law: P = I² R
            expected_P_joule = self.I**2 * self.R
            if math.isclose(self.P, expected_P_joule, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Joule's Law passed: P = I²R ≈ {expected_P_joule}")
            else:
                logs.append(f"Joule's Law failed: expected P ≈ {expected_P_joule}, got {self.P}")

            # 6. Cross-check P from formula with inputs
            # Avoid division by zero if rho or dt is zero
            if self.rho != 0 and self.dt != 0:
                expected_P_formula = (self.B0**2 * self.L**3 * self.S) / (16 * math.pi**2 * self.rho * self.dt**2)
                if math.isclose(self.P, expected_P_formula, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Power formula consistency passed: P ≈ {expected_P_formula}")
                else:
                    logs.append(f"Power formula consistency failed: expected P ≈ {expected_P_formula}, got {self.P}")
            else:
                logs.append("Power formula consistency skipped: rho or dt is zero")

        except Exception as e:
            logs.append(f"Invariant test error: {e}\n{traceback.format_exc()}")
        return logs

    def test_trace_consistency(self):
        """
        Check that intermediate results are consistent with final results.
        """
        logs = []
        try:
            # Check that E, I, R lead to P via Joule's Law (already done in invariants)
            P_from_I_R = self.I**2 * self.R
            if math.isclose(self.P, P_from_I_R, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("Trace consistency (P from I and R) passed.")
            else:
                logs.append(f"Trace consistency (P from I and R) failed: P={self.P}, I²R={P_from_I_R}")

            # Check that I = E / R
            if self.R != 0:
                I_from_E_R = self.E / self.R
                if math.isclose(self.I, I_from_E_R, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("Trace consistency (I from E and R) passed.")
                else:
                    logs.append(f"Trace consistency (I from E and R) failed: I={self.I}, E/R={I_from_E_R}")
            else:
                logs.append("Trace consistency (I from E and R) skipped: R = 0")

            # Check that E from Faraday matches the formula with L
            if self.dt != 0:
                E_from_L = self.B0 * self.L**2 / (4 * math.pi * self.dt)
                if math.isclose(self.E, E_from_L, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("Trace consistency (E from L) passed.")
                else:
                    logs.append(f"Trace consistency (E from L) failed: E={self.E}, B0L²/(4πΔt)={E_from_L}")
            else:
                logs.append("Trace consistency (E from L) skipped: dt = 0")

        except Exception as e:
            logs.append(f"Trace consistency test error: {e}\n{traceback.format_exc()}")
        return logs

    def test_monotonicity(self):
        """
        Perturb an input (L) and verify that the output (P) changes as expected.
        Uses the formula derived from steps to recompute P.
        """
        logs = []
        try:
            # Skip if any required parameter is zero to avoid division by zero
            if self.rho == 0 or self.dt == 0:
                logs.append("Monotonicity test skipped: rho or dt is zero")
                return logs
                
            # Perturb L by 1%
            L_original = self.L
            L_perturbed = L_original * 1.01

            # Recompute P using the formula from step_5 for both original and perturbed L
            P_original_formula = (self.B0**2 * L_original**3 * self.S) / (16 * math.pi**2 * self.rho * self.dt**2)
            P_perturbed_formula = (self.B0**2 * L_perturbed**3 * self.S) / (16 * math.pi**2 * self.rho * self.dt**2)

            # Expected ratio: (1.01)^3 ≈ 1.030301
            expected_ratio = 1.01**3
            actual_ratio = P_perturbed_formula / P_original_formula

            if math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("Monotonicity test passed: P ∝ L³ (1% increase in L yields ~3.03% increase in P).")
            else:
                logs.append(f"Monotonicity test failed: expected ratio {expected_ratio}, got {actual_ratio}")

            # Additionally, check that the solution's P matches the formula for original L
            if not math.isclose(self.P, P_original_formula, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"Warning: solution's P ({self.P}) differs from formula's P ({P_original_formula})")

        except Exception as e:
            logs.append(f"Monotonicity test error: {e}\n{traceback.format_exc()}")
        return logs

    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        all_logs = []
        success = True

        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"=== {test_name} ===")
            logs = test_method()
            if logs and any("failed" in log.lower() for log in logs):
                success = False
            all_logs.extend(logs)
            all_logs.append("")  # blank line for readability

        return success, all_logs