import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safe initialization: extract inputs and results without direct attribute access.
        """
        self.solution = solution_instance
        # Extract inputs safely with defaults (convert to base SI units: W, V)
        self.P = getattr(solution_instance, 'P', 0.0)  # in W (200 kW = 200,000 W)
        self.U1 = getattr(solution_instance, 'U1', 0.0)  # in V (11 kV = 11,000 V)
        self.U2 = getattr(solution_instance, 'U2', 0.0)  # in V (220 kV = 220,000 V)
        # If not directly available, try to get from results or compute from given
        if self.P == 0.0:
            self.P = solution_instance.results.get('P', 200000.0)
        if self.U1 == 0.0:
            self.U1 = solution_instance.results.get('U1', 11000.0)
        if self.U2 == 0.0:
            self.U2 = solution_instance.results.get('U2', 220000.0)

        # Extract results safely
        self.n_ratio = solution_instance.results.get('fracn_1n_2', 0.0)  # turns ratio n1/n2
        self.I = solution_instance.results.get('I', 0.0)  # transmission current in A

        # Convert if stored as string (e.g., "1/20") or fraction
        if isinstance(self.n_ratio, str):
            if '/' in self.n_ratio:
                num, den = map(float, self.n_ratio.split('/'))
                self.n_ratio = num / den
            else:
                try:
                    self.n_ratio = float(self.n_ratio)
                except ValueError:
                    self.n_ratio = 0.0
        if isinstance(self.I, str):
            self.I = float(self.I.replace('A', '').strip()) if 'A' in self.I else float(self.I)

        # Expected values for comparison (from steps analysis)
        self.expected_ratio = 1.0 / 20.0  # 1:20 -> 1/20 = 0.05
        self.expected_I = 0.90  # A

        self.logs = []

    def _log(self, msg, is_error=False):
        """Helper to log messages."""
        self.logs.append(('ERROR' if is_error else 'INFO', msg))

    def test_invariants(self):
        """
        Verify fundamental physical invariants:
        1. Transformer voltage ratio equals turns ratio (for ideal transformer).
        2. Power conservation: P_primary = P_secondary (since ideal).
        3. Current relation: I_secondary = P / U2.
        """
        try:
            # 1. Voltage ratio vs. turns ratio
            voltage_ratio = self.U1 / self.U2
            if not math.isclose(self.n_ratio, voltage_ratio, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Invariant failed: n1/n2 = {self.n_ratio} != U1/U2 = {voltage_ratio}", is_error=True)
                return False

            # 2. Power conservation (input power = output power, both 200 kW)
            # The problem states the transformer is ideal, so no loss.
            # The power at secondary (transmission side) should equal the plant output power.
            P_secondary = self.U2 * self.I  # P = U * I
            if not math.isclose(self.P, P_secondary, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Power conservation failed: P_plant = {self.P} W, P_transmission = {P_secondary} W", is_error=True)
                return False

            # 3. Current from power and voltage (secondary side)
            I_from_power = self.P / self.U2
            if not math.isclose(self.I, I_from_power, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Current relation failed: I = {self.I} A, expected P/U2 = {I_from_power} A", is_error=True)
                return False

            self._log("All invariants passed.")
            return True
        except Exception as e:
            self._log(f"Exception in test_invariants: {e}", is_error=True)
            return False

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with the final answer and expected steps.
        """
        try:
            # Step 1: Turns ratio from voltage ratio
            computed_ratio = self.U1 / self.U2
            if not math.isclose(self.n_ratio, computed_ratio, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Trace inconsistency in step 1: n1/n2 = {self.n_ratio}, computed U1/U2 = {computed_ratio}", is_error=True)
                return False

            # Step 2: Transmission current from power and secondary voltage
            computed_I = self.P / self.U2
            if not math.isclose(self.I, computed_I, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Trace inconsistency in step 2: I = {self.I} A, computed P/U2 = {computed_I} A", is_error=True)
                return False

            # Compare with expected values (from steps analysis)
            if not math.isclose(self.n_ratio, self.expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Turns ratio mismatch: {self.n_ratio} vs expected {self.expected_ratio}", is_error=True)
                return False
            if not math.isclose(self.I, self.expected_I, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Current mismatch: {self.I} A vs expected {self.expected_I} A", is_error=True)
                return False

            self._log("Trace consistency passed.")
            return True
        except Exception as e:
            self._log(f"Exception in test_trace_consistency: {e}", is_error=True)
            return False

    def test_monotonicity(self):
        """
        Perturb an input (e.g., U1) and verify the trends match KB expectations.
        We must restore the original values after perturbation.
        """
        original_U1 = self.U1
        original_n_ratio = self.n_ratio
        original_I = self.I

        try:
            # Perturb U1 by +10% (increase)
            perturb_factor = 1.10
            self.U1 *= perturb_factor
            # Recompute expected changes using formulas (symbolic/numerical fallback)
            new_ratio_expected = self.U1 / self.U2  # n1/n2 should increase proportionally with U1
            new_I_expected = self.P / self.U2       # I should remain unchanged because U2 and P are fixed

            # According to KB: if primary voltage V_p (U1) increases, secondary voltage V_s (U2) is fixed,
            # so turns ratio n1/n2 increases proportionally.
            # And the transmission current I = P / U2 remains constant.

            # We don't have a direct way to recompute the results from the solution instance,
            # so we use the formulas to check the trend.
            # The expected trend: n_ratio increases, I stays the same.

            # Check that the new ratio is greater than the old one (within tolerance)
            if not math.isclose(new_ratio_expected, original_n_ratio * perturb_factor, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Monotonicity failed: new expected ratio {new_ratio_expected} not {original_n_ratio * perturb_factor}", is_error=True)
                return False

            # Check that I remains the same (since U2 and P unchanged)
            if not math.isclose(new_I_expected, original_I, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Monotonicity failed: current changed from {original_I} to {new_I_expected}", is_error=True)
                return False

            # Restore original values
            self.U1 = original_U1
            self.n_ratio = original_n_ratio
            self.I = original_I

            self._log("Monotonicity test passed (U1 perturbation).")
            return True
        except Exception as e:
            # Restore on exception
            self.U1 = original_U1
            self.n_ratio = original_n_ratio
            self.I = original_I
            self._log(f"Exception in test_monotonicity: {e}", is_error=True)
            return False

    def run_all_tests(self):
        """
        Execute all tests and return (success_bool, logs_list).
        """
        self.logs = []
        success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]

        for test_name, test_func in tests:
            self._log(f"Running {test_name}...")
            try:
                if not test_func():
                    success = False
                    self._log(f"{test_name} failed.", is_error=True)
            except Exception as e:
                success = False
                self._log(f"{test_name} crashed: {e}", is_error=True)

        return success, self.logs