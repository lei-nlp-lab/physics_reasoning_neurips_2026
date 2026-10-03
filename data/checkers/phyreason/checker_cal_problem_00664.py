import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of input parameters
        self.U0 = getattr(solution_instance, 'U0', None)
        self.T0 = getattr(solution_instance, 'T0', None)
        self.m = getattr(solution_instance, 'm', None)
        self.q = getattr(solution_instance, 'q', None)

        # Safe extraction of results
        results = getattr(solution_instance, 'results', {})
        self.d = results.get('d', None)
        self.a1 = results.get('a_1', None)
        self.x1 = results.get('x_1', None)
        self.a2 = results.get('a_2', None)
        self.v = results.get('v', None)
        self.x2 = results.get('x_2', None)
        self.E1 = results.get('E_1', None)
        self.v1 = results.get('v_1', None)

        # Store logs
        self.logs = []

    def log(self, msg):
        self.logs.append(msg)

    def test_invariants(self):
        """Check consistency of physical laws."""
        try:
            # Use symbolic variables for clarity
            U0, T0, m, q, d, a1, v1, a2, v, x1, x2, E1 = sp.symbols('U0 T0 m q d a1 v1 a2 v x1 x2 E1', positive=True)

            # 1. Check electric field relation (step 1)
            # E1 should be 2*U0/d
            if self.E1 is not None and self.d is not None and self.U0 is not None:
                expected_E1 = 2 * self.U0 / self.d
                if not math.isclose(self.E1, expected_E1, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Invariant failed: E1 = {self.E1}, expected 2*U0/d = {expected_E1}")
                    return False

            # 2. Check Newton's second law for first interval (step 2-3)
            # a1 = E1 * q / m
            if self.a1 is not None and self.E1 is not None and self.m is not None and self.q is not None:
                expected_a1 = self.E1 * self.q / self.m
                if not math.isclose(self.a1, expected_a1, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Invariant failed: a1 = {self.a1}, expected E1*q/m = {expected_a1}")
                    return False

            # 3. Check kinematics for first interval (step 4)
            # v1 = a1 * T0
            if self.v1 is not None and self.a1 is not None and self.T0 is not None:
                expected_v1 = self.a1 * self.T0
                if not math.isclose(self.v1, expected_v1, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Invariant failed: v1 = {self.v1}, expected a1*T0 = {expected_v1}")
                    return False

            # 4. Check Newton's second law for second interval (step 5)
            # a2 = U0 * q / (d * m)
            if self.a2 is not None and self.U0 is not None and self.q is not None and self.d is not None and self.m is not None:
                expected_a2 = self.U0 * self.q / (self.d * self.m)
                if not math.isclose(self.a2, expected_a2, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Invariant failed: a2 = {self.a2}, expected U0*q/(d*m) = {expected_a2}")
                    return False

            # 5. Check kinematics for second interval (step 6)
            # v = v1 + a2 * T0
            if self.v is not None and self.v1 is not None and self.a2 is not None and self.T0 is not None:
                expected_v = self.v1 + self.a2 * self.T0
                if not math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Invariant failed: v = {self.v}, expected v1 + a2*T0 = {expected_v}")
                    return False

            # 6. Check displacement in first interval (step 8)
            # x1 = 0.5 * a1 * T0^2
            if self.x1 is not None and self.a1 is not None and self.T0 is not None:
                expected_x1 = 0.5 * self.a1 * self.T0**2
                if not math.isclose(self.x1, expected_x1, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Invariant failed: x1 = {self.x1}, expected 0.5*a1*T0^2 = {expected_x1}")
                    return False

            # 7. Check displacement in second interval (step 9)
            # x2 = v1 * T0 + 0.5 * a2 * T0^2
            if self.x2 is not None and self.v1 is not None and self.a2 is not None and self.T0 is not None:
                expected_x2 = self.v1 * self.T0 + 0.5 * self.a2 * self.T0**2
                if not math.isclose(self.x2, expected_x2, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Invariant failed: x2 = {self.x2}, expected v1*T0 + 0.5*a2*T0^2 = {expected_x2}")
                    return False

            # 8. Check total distance (step 10)
            # d = x1 + x2
            if self.d is not None and self.x1 is not None and self.x2 is not None:
                if not math.isclose(self.d, self.x1 + self.x2, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Invariant failed: d = {self.d}, expected x1 + x2 = {self.x1 + self.x2}")
                    return False

            self.log("All invariants passed.")
            return True

        except Exception as e:
            self.log(f"Error in test_invariants: {traceback.format_exc()}")
            return False

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Final answers from steps 7 and 11
            # v = 3 * sqrt(2 * U0 * q / (7 * m))
            # d = sqrt(7 * U0 * q * T0**2 / (2 * m))
            if self.U0 is None or self.T0 is None or self.m is None or self.q is None:
                self.log("Missing input parameters for trace consistency.")
                return False

            # Calculate expected v and d from formulas
            expected_v = 3 * math.sqrt(2 * self.U0 * self.q / (7 * self.m))
            expected_d = math.sqrt(7 * self.U0 * self.q * self.T0**2 / (2 * self.m))

            # Check against provided v and d
            if self.v is not None:
                if not math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Trace consistency failed: v = {self.v}, expected 3*sqrt(2*U0*q/(7*m)) = {expected_v}")
                    return False
            if self.d is not None:
                if not math.isclose(self.d, expected_d, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Trace consistency failed: d = {self.d}, expected sqrt(7*U0*q*T0^2/(2*m)) = {expected_d}")
                    return False

            # Additionally, check that intermediate results lead to these finals
            # Re-derive d from x1 and x2 formulas
            if self.a1 is not None and self.T0 is not None:
                x1_calc = 0.5 * self.a1 * self.T0**2
                if self.x1 is not None and not math.isclose(self.x1, x1_calc, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Trace consistency: x1 mismatch with a1 and T0.")
                    return False

            if self.v1 is not None and self.a2 is not None and self.T0 is not None:
                x2_calc = self.v1 * self.T0 + 0.5 * self.a2 * self.T0**2
                if self.x2 is not None and not math.isclose(self.x2, x2_calc, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"Trace consistency: x2 mismatch with v1, a2, T0.")
                    return False

            self.log("Trace consistency passed.")
            return True

        except Exception as e:
            self.log(f"Error in test_trace_consistency: {traceback.format_exc()}")
            return False

    def test_monotonicity(self):
        """Perturb mass m and check trends."""
        try:
            if None in [self.U0, self.T0, self.m, self.q]:
                self.log("Missing input parameters for monotonicity test.")
                return False

            # Perturb m by +10%
            m_perturbed = self.m * 1.10

            # Recalculate d and v using the derived formulas
            d_perturbed = math.sqrt(7 * self.U0 * self.q * self.T0**2 / (2 * m_perturbed))
            v_perturbed = 3 * math.sqrt(2 * self.U0 * self.q / (7 * m_perturbed))

            # Expected trends: if m increases, d and v should decrease (inverse relationship)
            if self.d is not None:
                if not (d_perturbed < self.d):
                    self.log(f"Monotonicity failed: d does not decrease when m increases. d_original={self.d}, d_perturbed={d_perturbed}")
                    return False

            if self.v is not None:
                if not (v_perturbed < self.v):
                    self.log(f"Monotonicity failed: v does not decrease when m increases. v_original={self.v}, v_perturbed={v_perturbed}")
                    return False

            self.log("Monotonicity test passed.")
            return True

        except Exception as e:
            self.log(f"Error in test_monotonicity: {traceback.format_exc()}")
            return False

    def run_all_tests(self):
        """Execute all tests and return success and logs."""
        self.logs = []  # Reset logs
        success = True

        # Run each test
        if not self.test_invariants():
            success = False
        if not self.test_trace_consistency():
            success = False
        if not self.test_monotonicity():
            success = False

        return success, self.logs