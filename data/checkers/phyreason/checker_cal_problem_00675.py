import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs with defaults from problem statement
        self.R = getattr(solution_instance, 'R', 10.0)
        self.r = getattr(solution_instance, 'r', 2.0)
        self.U = getattr(solution_instance, 'U', 44.0)
        self.U_M = getattr(solution_instance, 'U_M', 24.0)

        # Safely extract results from solution_instance.results
        results = getattr(solution_instance, 'results', {})
        self.I = results.get('I', 0.0)
        self.U_R = results.get('U_R', 0.0)
        self.P_M = results.get('P_M', 0.0)
        self.P_heat = results.get('P_heat', 0.0)
        self.P_output = results.get('P_output', 0.0)

        # Expected intermediate values from steps analysis
        self.expected = {
            'U_R': 20.0,
            'I': 2.0,
            'P_M': 48.0,
            'P_heat': 8.0,
            'P_output': 40.0
        }

        self.logs = []

    def _log(self, msg, success):
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {msg}")

    def _isclose(self, a, b):
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

    def test_invariants(self):
        """Check physical laws (Ohm's law, power conservation, series voltage)."""
        success = True

        # 1. Series voltage: U = U_R + U_M
        if not self._isclose(self.U, self.U_R + self.U_M):
            self._log(f"Voltage series mismatch: U={self.U}, U_R+U_M={self.U_R+self.U_M}", False)
            success = False
        else:
            self._log("Series voltage U = U_R + U_M holds.", True)

        # 2. Ohm's law for fixed resistor: U_R = I * R
        if not self._isclose(self.U_R, self.I * self.R):
            self._log(f"Ohm's law for R failed: U_R={self.U_R}, I*R={self.I*self.R}", False)
            success = False
        else:
            self._log("Ohm's law for fixed resistor U_R = I * R holds.", True)

        # 3. Motor power: P_M = U_M * I
        if not self._isclose(self.P_M, self.U_M * self.I):
            self._log(f"Motor power mismatch: P_M={self.P_M}, U_M*I={self.U_M*self.I}", False)
            success = False
        else:
            self._log("Motor power P_M = U_M * I holds.", True)

        # 4. Heat power: P_heat = I^2 * r
        if not self._isclose(self.P_heat, self.I**2 * self.r):
            self._log(f"Heat power mismatch: P_heat={self.P_heat}, I^2*r={self.I**2*self.r}", False)
            success = False
        else:
            self._log("Heat power P_heat = I^2 * r holds.", True)

        # 5. Output power: P_output = P_M - P_heat
        if not self._isclose(self.P_output, self.P_M - self.P_heat):
            self._log(f"Output power mismatch: P_output={self.P_output}, P_M-P_heat={self.P_M-self.P_heat}", False)
            success = False
        else:
            self._log("Output power P_output = P_M - P_heat holds.", True)

        return success

    def test_trace_consistency(self):
        """Check that intermediate results match expected values from steps analysis."""
        success = True
        for key, expected_val in self.expected.items():
            actual_val = getattr(self, key, None)
            if actual_val is None or not self._isclose(actual_val, expected_val):
                self._log(f"Trace mismatch for {key}: expected {expected_val}, got {actual_val}", False)
                success = False
            else:
                self._log(f"Trace {key} = {actual_val} matches expected {expected_val}.", True)
        return success

    def test_monotonicity(self):
        """Perturb R by +10%, check that currents and powers decrease (U and U_M fixed)."""
        original_R = self.R
        perturbed_R = original_R * 1.10  # 10% increase

        # Recalculate using formulas from steps analysis
        U_R_perturbed = self.U - self.U_M
        I_perturbed = U_R_perturbed / perturbed_R
        P_M_perturbed = self.U_M * I_perturbed
        P_heat_perturbed = I_perturbed**2 * self.r
        P_output_perturbed = P_M_perturbed - P_heat_perturbed

        # Check trends: if R increases, I decreases, then all powers decrease
        trend_ok = (
            I_perturbed < self.I and
            P_M_perturbed < self.P_M and
            P_heat_perturbed < self.P_heat and
            P_output_perturbed < self.P_output
        )

        if not trend_ok:
            self._log(
                f"Monotonicity failed: R↑ from {original_R} to {perturbed_R} did not cause I↓, P_M↓, P_heat↓, P_output↓. "
                f"Got I={I_perturbed:.3f} (was {self.I:.3f}), P_M={P_M_perturbed:.3f} (was {self.P_M:.3f})",
                False
            )
            success = False
        else:
            self._log(
                f"Monotonicity passed: R↑ by 10% caused I↓, P_M↓, P_heat↓, P_output↓ as expected.",
                True
            )
            success = True

        # Restore original R
        self.R = original_R
        return success

    def run_all_tests(self):
        """Execute all checks and return (overall_success, logs)."""
        self.logs.clear()
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for name, test_func in tests:
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self._log(f"{name} test crashed: {e}\n{traceback.format_exc()}", False)
                overall_success = False

        return overall_success, self.logs