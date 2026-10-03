import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the PhysicsChecker with safe extraction of results.
        """
        self.solution = solution_instance
        # Constants from the problem statement
        self.E = 12.0               # EMF of power supply (V)
        self.U_A = 10.0             # Rated voltage of lamp A (V)
        self.P_A = 10.0             # Rated power of lamp A (W)
        self.U_B = 9.0              # Rated voltage of lamp B (V)
        self.R_rheostat_max = 180.0 # Maximum resistance of rheostat (Ω)

        # Helper to safely extract values from solution_instance
        def get_val(key, default=0.0):
            if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
                return solution_instance.results.get(key, default)
            else:
                return getattr(solution_instance, key, default)

        # Extract results with safe fallbacks
        self.R_A = get_val('R_A', 0.0)   # Resistance of lamp A (Ω)
        self.r = get_val('r', 0.0)       # Internal resistance of power supply (Ω)
        self.I_1 = get_val('I_1', 0.0)   # Circuit current when C at E (A)
        self.P = get_val('P', 0.0)       # Rated power of lamp B (W)
        self.I = get_val('I', 0.0)       # Main circuit current when C at F (A)
        self.I_B = get_val('I_B', 0.0)   # Current (either rheostat or lamp B) (A)

    def test_invariants(self):
        """
        Test fundamental physical laws: Ohm's Law, power formulas, and Kirchhoff's Law.
        """
        logs = []

        # 1. Ohm's Law and power formula for lamp A
        expected_R_A = self.U_A**2 / self.P_A
        if math.isclose(self.R_A, expected_R_A, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"OK: R_A = {self.R_A} Ω ≈ U_A²/P_A = {expected_R_A} Ω")
        else:
            logs.append(f"FAIL: R_A = {self.R_A} Ω but expected {expected_R_A} Ω (U_A²/P_A)")

        expected_I_1 = self.P_A / self.U_A
        if math.isclose(self.I_1, expected_I_1, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"OK: I_1 = {self.I_1} A ≈ P_A/U_A = {expected_I_1} A")
        else:
            logs.append(f"FAIL: I_1 = {self.I_1} A but expected {expected_I_1} A (P_A/U_A)")

        # 2. Ohm's law for closed circuit when C at E
        expected_E = self.U_A + self.I_1 * self.r
        if math.isclose(self.E, expected_E, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"OK: E = {self.E} V ≈ U_A + I_1·r = {expected_E} V")
        else:
            logs.append(f"FAIL: E = {self.E} V but U_A + I_1·r = {expected_E} V")

        # 3. Kirchhoff's Current Law and Ohm's Law when C at F
        I_rheostat = self.U_B / self.R_rheostat_max  # current through rheostat
        I_B_lamp = self.P / self.U_B                 # current through lamp B
        expected_I_total = I_rheostat + I_B_lamp     # KCL: total current into node

        if math.isclose(self.I, expected_I_total, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"OK: I = {self.I} A ≈ I_rheostat + I_B_lamp = {expected_I_total} A")
        else:
            logs.append(f"FAIL: I = {self.I} A but expected {expected_I_total} A (I_rheostat + I_B_lamp)")

        # 4. Ohm's law for closed circuit when C at F
        expected_E_from_F = self.U_B + expected_I_total * (self.R_A + self.r)
        if math.isclose(self.E, expected_E_from_F, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"OK: E = {self.E} V ≈ U_B + I·(R_A+r) = {expected_E_from_F} V")
        else:
            logs.append(f"FAIL: E = {self.E} V but U_B + I·(R_A+r) = {expected_E_from_F} V")

        # 5. Identify what I_B represents (rheostat or lamp B current)
        if math.isclose(self.I_B, I_rheostat, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"NOTE: I_B = {self.I_B} A matches rheostat current {I_rheostat} A")
        elif math.isclose(self.I_B, I_B_lamp, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"NOTE: I_B = {self.I_B} A matches lamp B current {I_B_lamp} A")
        else:
            logs.append(f"WARNING: I_B = {self.I_B} A does not match rheostat ({I_rheostat} A) nor lamp B ({I_B_lamp} A)")

        return logs

    def test_trace_consistency(self):
        """
        Verify that intermediate results lead consistently to the final answer.
        """
        logs = []

        # Recompute internal resistance from first part
        if self.I_1 != 0:
            expected_r = (self.E - self.U_A) / self.I_1
            if math.isclose(self.r, expected_r, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"OK: r = {self.r} Ω matches (E - U_A)/I_1 = {expected_r} Ω")
            else:
                logs.append(f"FAIL: r = {self.r} Ω but (E - U_A)/I_1 = {expected_r} Ω")
        else:
            logs.append("ERROR: I_1 is zero, cannot compute expected r")

        # Recompute P from the second part using extracted R_A and r
        if self.R_A + self.r != 0:
            I_rheostat = self.U_B / self.R_rheostat_max
            computed_P = 9.0 * ((self.E - self.U_B) / (self.R_A + self.r) - I_rheostat)
            if math.isclose(self.P, computed_P, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"OK: P = {self.P} W matches recomputed value {computed_P} W")
            else:
                logs.append(f"FAIL: P = {self.P} W but recomputed gives {computed_P} W")
        else:
            logs.append("ERROR: R_A + r is zero, cannot recompute P")

        return logs

    def test_monotonicity(self):
        """
        Perturb an input (rheostat max resistance) and check the trend in output P.
        """
        logs = []
        original_R = self.R_rheostat_max
        perturbed_R = original_R * 1.1  # +10% increase

        # Compute new rheostat current and resulting P
        I_rheostat_perturbed = self.U_B / perturbed_R
        if self.R_A + self.r != 0:
            P_perturbed = 9.0 * ((self.E - self.U_B) / (self.R_A + self.r) - I_rheostat_perturbed)

            # Expected trend: increasing R_rheostat decreases rheostat current,
            # so to maintain circuit equation, P must increase.
            if P_perturbed > self.P:
                logs.append(f"OK: Increasing R_rheostat from {original_R} Ω to {perturbed_R} Ω increases P from {self.P} W to {P_perturbed:.3f} W")
            else:
                logs.append(f"FAIL: Increasing R_rheostat did not increase P (original {self.P} W, perturbed {P_perturbed:.3f} W)")
        else:
            logs.append("ERROR: R_A + r is zero, cannot compute perturbed P")

        return logs

    def run_all_tests(self):
        """
        Execute all tests and return a summary.
        """
        all_logs = []
        all_ok = True

        tests = [self.test_invariants, self.test_trace_consistency, self.test_monotonicity]
        for test in tests:
            try:
                logs = test()
                all_logs.extend(logs)
                for log in logs:
                    if log.startswith(("FAIL", "ERROR")):
                        all_ok = False
            except Exception as e:
                error_msg = f"Exception in {test.__name__}: {str(e)}\n{traceback.format_exc()}"
                all_logs.append(error_msg)
                all_ok = False

        return all_ok, all_logs