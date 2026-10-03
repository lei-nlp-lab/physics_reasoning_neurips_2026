import math
import sympy as sp

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all required inputs and results from the solution instance.
        """
        self.logs = []
        self.solution_instance = solution_instance

        # Extract inputs with safe defaults from problem context
        self.E = getattr(solution_instance, 'E', 12.0)  # V
        self.r = getattr(solution_instance, 'r', 0.5)   # Ω
        self.R1 = getattr(solution_instance, 'R1', 5.5) # Ω
        self.R2 = getattr(solution_instance, 'R2', 12.0) # Ω
        self.C = getattr(solution_instance, 'C', 2.0e-4) # F (deduced from steps)

        # Extract results with safe fallback to results dictionary
        self.I = getattr(solution_instance, 'I', 
                        getattr(solution_instance, 'results', {}).get('I', 0.0))
        self.U2 = getattr(solution_instance, 'U_2', 
                         getattr(solution_instance, 'results', {}).get('U_2', 0.0))
        self.Q = getattr(solution_instance, 'Q', 
                        getattr(solution_instance, 'results', {}).get('Q', 0.0))

        # Target values from steps analysis (for trace consistency)
        self.I_target = 0.67  # A
        self.U2_target = 8.0  # V
        self.Q_target = 1.6e-3  # C

    def _is_close(self, a, b):
        """Robust fuzzy comparison with 2% tolerance."""
        return math.isclose(a, b, rel_tol=0.02, abs_tol=0.02)

    def test_invariants(self):
        """Check fundamental physical laws: Ohm's Law and charge conservation."""
        logs = []
        success = True

        # 1. Ohm's Law for entire circuit
        try:
            R_total = self.R1 + self.R2 + self.r
            I_expected = self.E / R_total
            if not self._is_close(self.I, I_expected):
                logs.append(f"FAIL: Ohm's Law (circuit) violated. I={self.I:.3f}, expected {I_expected:.3f}")
                success = False
            else:
                logs.append(f"PASS: Ohm's Law (circuit) satisfied. I={self.I:.3f} A")
        except Exception as e:
            logs.append(f"ERROR in circuit Ohm's Law: {str(e)}")
            success = False

        # 2. Ohm's Law for R2
        try:
            U2_expected = self.I * self.R2
            if not self._is_close(self.U2, U2_expected):
                logs.append(f"FAIL: Ohm's Law (R2) violated. U2={self.U2:.3f}, expected {U2_expected:.3f}")
                success = False
            else:
                logs.append(f"PASS: Ohm's Law (R2) satisfied. U2={self.U2:.3f} V")
        except Exception as e:
            logs.append(f"ERROR in R2 Ohm's Law: {str(e)}")
            success = False

        # 3. Capacitor charge relation Q = C * U2
        try:
            Q_expected = self.C * self.U2
            if not self._is_close(self.Q, Q_expected):
                logs.append(f"FAIL: Q = C*U2 violated. Q={self.Q:.2e}, expected {Q_expected:.2e}")
                success = False
            else:
                logs.append(f"PASS: Capacitance equation satisfied. Q={self.Q:.2e} C")
        except Exception as e:
            logs.append(f"ERROR in capacitor equation: {str(e)}")
            success = False

        # 4. Power conservation (optional check)
        try:
            P_supply = self.E * self.I
            P_R1 = self.I**2 * self.R1
            P_R2 = self.I**2 * self.R2
            P_r = self.I**2 * self.r
            P_total_dissipated = P_R1 + P_R2 + P_r
            if not self._is_close(P_supply, P_total_dissipated):
                logs.append(f"WARNING: Power conservation off. Supply={P_supply:.3f}W, Dissipated={P_total_dissipated:.3f}W")
            else:
                logs.append(f"PASS: Power conservation verified.")
        except Exception as e:
            logs.append(f"NOTE: Power check skipped: {str(e)}")

        self.logs.extend(logs)
        return success

    def test_trace_consistency(self):
        """Check if intermediate results match the steps analysis."""
        logs = []
        success = True

        # Compare with step-by-step values
        checks = [
            ("Current I", self.I, self.I_target, "A"),
            ("Voltage U2", self.U2, self.U2_target, "V"),
            ("Charge Q", self.Q, self.Q_target, "C")
        ]

        for name, actual, target, unit in checks:
            if self._is_close(actual, target):
                logs.append(f"PASS: {name} matches target. {actual:.3f}{unit} ≈ {target:.3f}{unit}")
            else:
                logs.append(f"FAIL: {name} mismatch. {actual:.3f}{unit} vs target {target:.3f}{unit}")
                success = False

        self.logs.extend(logs)
        return success

    def test_monotonicity(self):
        """
        Perturb E (increase by 10%) and verify physical trends.
        Restore original values after test.
        """
        logs = []
        success = True

        # Save original state
        E_original = self.E
        I_original = self.I
        U2_original = self.U2
        Q_original = self.Q

        try:
            # Perturbation
            self.E *= 1.10  # +10%

            # Recalculate using physical formulas (not solution instance)
            R_total = self.R1 + self.R2 + self.r
            I_new = self.E / R_total
            U2_new = I_new * self.R2
            Q_new = self.C * U2_new

            # Check monotonic trends
            if not (I_new > I_original and U2_new > U2_original and Q_new > Q_original):
                logs.append("FAIL: Increasing E should increase I, U2, and Q.")
                success = False
            else:
                logs.append("PASS: Monotonic trend correct (+E → +I → +U2 → +Q)")

            # Check proportionality (Ohm's Law)
            I_ratio = I_new / I_original
            E_ratio = self.E / E_original
            if not self._is_close(I_ratio, E_ratio):
                logs.append(f"WARNING: I not proportional to E. I_ratio={I_ratio:.3f}, E_ratio={E_ratio:.3f}")
            else:
                logs.append(f"PASS: I ∝ E verified (ratio={I_ratio:.3f})")

        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {str(e)}")
            success = False
        finally:
            # Restore original values
            self.E = E_original
            self.I = I_original
            self.U2 = U2_original
            self.Q = Q_original

        self.logs.extend(logs)
        return success

    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = ["=== Physics Auditor Report ==="]
        overall_success = True

        tests = [
            ("Invariant Tests", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            self.logs.append(f"\n--- {test_name} ---")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self.logs.append(f"ERROR: Test crashed - {str(e)}")
                overall_success = False

        self.logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, self.logs

# Optional: Example usage pattern (commented out)
# if __name__ == "__main__":
#     class MockSolution:
#         def __init__(self):
#             self.E = 12.0
#             self.r = 0.5
#             self.R1 = 5.5
#             self.R2 = 12.0
#             self.C = 2.0e-4
#             self.results = {
#                 'I': 0.6667,
#                 'U_2': 8.0004,
#                 'Q': 1.60008e-3
#             }
#     
#     mock_solution = MockSolution()
#     checker = PhysicsChecker(mock_solution)
#     success, logs = checker.run_all_tests()
#     for log in logs:
#         print(log)