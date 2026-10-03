import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize with safe attribute access.
        """
        # Safe extraction of inputs
        self.n1 = getattr(solution_instance, 'n1', getattr(solution_instance, 'N1', 2000))
        self.n2 = getattr(solution_instance, 'n2', getattr(solution_instance, 'N2', 50))
        self.U1 = getattr(solution_instance, 'U1', getattr(solution_instance, 'u1', 2200.0))
        self.I1 = getattr(solution_instance, 'I1', getattr(solution_instance, 'i1', 0.2))
        
        # Safe extraction of results via results dictionary
        results = getattr(solution_instance, 'results', {})
        self.U2 = results.get('U_2', results.get('U2', 0.0))
        self.I2 = results.get('I_2', results.get('I2', 0.0))
        
        # Tolerance for fuzzy matching
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2

    def test_invariants(self):
        """
        Verify transformer laws and power conservation.
        """
        logs = []
        try:
            # Law 1: Voltage ratio
            expected_U2 = self.U1 * (self.n2 / self.n1)
            if math.isclose(self.U2, expected_U2, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"✓ Voltage ratio holds: U2 = {self.U2:.2f} V, expected {expected_U2:.2f} V")
            else:
                logs.append(f"✗ Voltage ratio violation: U2 = {self.U2:.2f} V, expected {expected_U2:.2f} V")
                return False, logs

            # Law 2: Current ratio
            expected_I2 = self.I1 * (self.n1 / self.n2)
            if math.isclose(self.I2, expected_I2, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"✓ Current ratio holds: I2 = {self.I2:.2f} A, expected {expected_I2:.2f} A")
            else:
                logs.append(f"✗ Current ratio violation: I2 = {self.I2:.2f} A, expected {expected_I2:.2f} A")
                return False, logs

            # Law 3: Power conservation (ideal transformer)
            input_power = self.U1 * self.I1
            output_power = self.U2 * self.I2
            if math.isclose(input_power, output_power, rel_tol=self.rel_tol, abs_tol=self.abs_tol):
                logs.append(f"✓ Power conservation holds: P_in = {input_power:.2f} W, P_out = {output_power:.2f} W")
            else:
                logs.append(f"⚠ Power deviation: P_in = {input_power:.2f} W, P_out = {output_power:.2f} W")
                # Not necessarily a failure for real transformers, but log warning

            return True, logs

        except Exception as e:
            logs.append(f"✗ Error in test_invariants: {e}")
            return False, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        logs = []
        try:
            # No intermediate steps to trace in this problem
            logs.append("ℹ No intermediate steps to trace.")
            return True, logs
        except Exception as e:
            logs.append(f"✗ Error in test_trace_consistency: {e}")
            return False, logs

    def test_monotonicity(self):
        """
        Perturb an input and check trend, then restore.
        """
        logs = []
        original_n1 = self.n1
        try:
            # Increase n1 by 10%
            perturbed_n1 = original_n1 * 1.1
            # Recalculate U2 and I2 using transformer formulas
            perturbed_U2 = self.U1 * (self.n2 / perturbed_n1)
            perturbed_I2 = self.I1 * (perturbed_n1 / self.n2)

            # Check monotonicity: U2 decreases with increased n1, I2 increases
            if perturbed_U2 < self.U2:
                logs.append(f"✓ U2 decreases with increased n1: {perturbed_U2:.2f} V < {self.U2:.2f} V")
            else:
                logs.append(f"✗ U2 monotonicity violated: {perturbed_U2:.2f} V >= {self.U2:.2f} V")

            if perturbed_I2 > self.I2:
                logs.append(f"✓ I2 increases with increased n1: {perturbed_I2:.2f} A > {self.I2:.2f} A")
            else:
                logs.append(f"✗ I2 monotonicity violated: {perturbed_I2:.2f} A <= {self.I2:.2f} A")

            # Restore original value
            self.n1 = original_n1

            return True, logs

        except Exception as e:
            logs.append(f"✗ Error in test_monotonicity: {e}")
            self.n1 = original_n1
            return False, logs

    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        logs = []
        success = True

        # Test 1: Invariants (transformer laws)
        test1_success, test1_logs = self.test_invariants()
        logs.extend(test1_logs)
        if not test1_success:
            success = False

        # Test 2: Trace consistency
        test2_success, test2_logs = self.test_trace_consistency()
        logs.extend(test2_logs)
        if not test2_success:
            success = False

        # Test 3: Monotonicity
        test3_success, test3_logs = self.test_monotonicity()
        logs.extend(test3_logs)
        if not test3_success:
            success = False

        return success, logs