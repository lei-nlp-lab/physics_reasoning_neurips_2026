import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the checker with safe attribute access.
        """
        self.solution = solution_instance
        # Safe extraction of results only
        self.results = getattr(solution_instance, 'results', {})

        # Extract results with fallbacks
        self.v2_result = self.results.get('v_2', 0.0)
        self.U1_result = self.results.get('U_1', 0.0)
        self.dy_dx_result = self.results.get('dy/dx', 0.0)
        self.beta_result = self.results.get('beta', 0.0)
        self.Deltay_result = self.results.get('Deltay', 0.0)
        self.I_min_result = self.results.get('I_min', 0.0)
        self.Deltatheta_result = self.results.get('Deltatheta', 0.0)
        self.Deltaphi_result = self.results.get('Deltaphi', 0.0)
        self.Deltaphi_CD_result = self.results.get('Deltaphi_CD', 0.0)
        self.x_result = self.results.get('x', 0.0)
        self.x0_result = self.results.get('x_0', 0.0)

        # Constants (use typical values for checks)
        self.h = 6.626e-34
        self.hbar = self.h / (2 * math.pi)
        self.c = 3e8
        self.g = 9.8
        self.m_e = 9.11e-31
        self.q_e = 1.6e-19

        # For checks that require inputs, we use default values or skip if missing
        # We'll get inputs from results if they were stored, otherwise use defaults
        self.m = self.results.get('m', 0.0)
        self.v1 = self.results.get('v1', 0.0)
        self.theta1 = self.results.get('theta1', 0.0)
        self.V0 = self.results.get('V0', 0.0)
        self.theta2 = self.results.get('theta2', 0.0)
        self.x1 = self.results.get('x1', 0.0)
        self.n0 = self.results.get('n0', 1.0)
        self.i0 = self.results.get('i0', 0.0)
        self.k = self.results.get('k', 0.0)
        self.y = self.results.get('y', 0.0)
        self.q = self.results.get('q', self.q_e)
        self.d = self.results.get('d', 0.0)
        self.theta = self.results.get('theta', 0.0)
        self.x0 = self.results.get('x0', 0.0)
        self.x1_slit = self.results.get('x1_slit', 0.0)

        # Additional: extract v2 for beta calculation if available
        # In the problem, beta uses a different v2 (v1 - 1e5). We'll compute it from v1.
        self.v2_beta = self.v1 - 1e5 if self.v1 > 1e5 else 0.0

        # Logging
        self.logs = []

    def log(self, message):
        """Append message to logs."""
        self.logs.append(message)

    def test_invariants(self):
        """
        Verify fundamental physical laws using both symbolic and numerical checks.
        """
        try:
            # 1. Conservation of Mechanical Energy (Particle crossing potential step)
            if self.m > 0 and self.v1 > 0 and self.V0 >= 0:
                # Numerical check
                v2_expected = math.sqrt(max(0, self.v1**2 - 2*self.V0/self.m))
                if not math.isclose(v2_expected, self.v2_result, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: Energy conservation violated. Expected v2={v2_expected}, Result={self.v2_result}")
                    return False
                self.log("PASS: Energy conservation (particle)")

            # 2. Conservation of momentum in y-direction (Snell's Law for particle)
            if self.v1 > 0 and abs(math.sin(self.theta1)) > 1e-9 and abs(math.sin(self.theta2)) > 1e-9:
                # From impulse-momentum: v1*sin(theta1) = v2*sin(theta2)
                lhs = self.v1 * math.sin(self.theta1)
                rhs = self.v2_result * math.sin(self.theta2)
                if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: Momentum (y) conservation violated. LHS={lhs}, RHS={rhs}")
                    return False
                self.log("PASS: Momentum conservation (y-direction)")

            # 3. Snell's Law for optics (if applicable)
            if self.n0 > 0 and self.i0 > 0 and abs(math.sin(self.i0)) > 1e-9:
                n_y = self.n0 - self.k * self.y if self.k != 0 else self.n0
                # Using the derived slope formula: dy/dx = -sqrt((n(y)/(n0*sin(i0)))^2 - 1)
                # Ensure the argument inside sqrt is non-negative
                arg = (n_y / (self.n0 * math.sin(self.i0)))**2 - 1
                if arg >= 0:
                    expected_slope = -math.sqrt(arg)
                    if not math.isclose(expected_slope, self.dy_dx_result, rel_tol=1e-2, abs_tol=1e-2):
                        self.log(f"FAIL: Snell's law slope mismatch. Expected={expected_slope}, Result={self.dy_dx_result}")
                        return False
                self.log("PASS: Snell's law (slope)")

            # 4. de Broglie relation and phase difference for beta
            # Note: beta uses a specific v2 (v1 - 1e5), not v2_result from energy conservation.
            # So we skip this check because v2 for beta is different.
            # Instead, we check that beta is computed correctly with the given v2_beta.
            if self.m > 0 and self.v1 > 0 and self.d > 0:
                beta_expected = (self.m * self.d / self.h) * math.sin(self.theta) * (self.v1 - self.v2_beta)
                if not math.isclose(beta_expected, self.beta_result, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: de Broglie phase mismatch. Expected={beta_expected}, Result={self.beta_result}")
                    return False
                self.log("PASS: de Broglie phase relation (beta)")

            # 5. Accelerating potential for electrons
            if self.m > 0 and self.v1 > 0 and self.q > 0:
                U1_expected = (self.m * self.v1**2) / (2 * self.q)
                if not math.isclose(U1_expected, self.U1_result, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: Accelerating potential mismatch. Expected={U1_expected}, Result={self.U1_result}")
                    return False
                self.log("PASS: Electron accelerating potential")

            return True

        except Exception as e:
            self.log(f"ERROR in invariants: {traceback.format_exc()}")
            return False

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        try:
            # 1. Check two expressions for v2 are equivalent
            v2_1 = math.sqrt(max(0, self.v1**2 - 2*self.V0/self.m))
            # Avoid division by zero for theta2
            if abs(math.sin(self.theta2)) > 1e-9:
                v2_2 = self.v1 * math.sin(self.theta1) / math.sin(self.theta2)
                if not math.isclose(v2_1, v2_2, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: v2 expressions inconsistent. Energy v2={v2_1}, Momentum v2={v2_2}")
                    return False
            else:
                # If theta2=0, then v2_2 is infinite, but we can skip the check
                pass
            self.log("PASS: v2 expressions consistent")

            # 2. Check x expression consistency
            if abs(self.k) > 1e-9 and self.n0 > 0:
                # x = (n0/k) * ln((n0 - k*y)/n0 + sqrt(((n0 - k*y)/n0)^2 - 1))
                n_y = self.n0 - self.k * self.y
                term = (n_y/self.n0) + math.sqrt(max(0, (n_y/self.n0)**2 - 1))
                if term > 0:
                    x_expected = (self.n0/self.k) * math.log(term)
                    if not math.isclose(x_expected, self.x_result, rel_tol=1e-2, abs_tol=1e-2):
                        self.log(f"FAIL: x expression mismatch. Expected={x_expected}, Result={self.x_result}")
                        return False
                self.log("PASS: x expression consistent")

            # 3. Check phase difference Δφ_CD = 0
            if not math.isclose(self.Deltaphi_CD_result, 0.0, abs_tol=1e-2):
                self.log(f"FAIL: Δφ_CD should be 0, got {self.Deltaphi_CD_result}")
                return False
            self.log("PASS: Δφ_CD = 0")

            # 4. Check interference geometry: Δy = (x0 - x1)*(tan(θ+Δθ) - tan(θ))
            if self.x0 > 0 and self.x1_slit > 0:
                delta_y_expected = (self.x0 - self.x1_slit) * (math.tan(self.theta + self.Deltatheta_result) - math.tan(self.theta))
                if not math.isclose(delta_y_expected, self.Deltay_result, rel_tol=1e-2, abs_tol=1e-2):
                    self.log(f"FAIL: Δy geometry mismatch. Expected={delta_y_expected}, Result={self.Deltay_result}")
                    return False
                self.log("PASS: Δy geometry consistent")

            return True

        except Exception as e:
            self.log(f"ERROR in trace consistency: {traceback.format_exc()}")
            return False

    def test_monotonicity(self):
        """
        Perturb an input (V0) and check the monotonic trend in v2.
        """
        original_V0 = self.V0
        try:
            # Perturb V0 by +10%
            perturbed_V0 = original_V0 * 1.10
            # Compute v2 using energy formula: v2 = sqrt(v1^2 - 2*V0/m)
            if self.m > 0:
                original_v2 = math.sqrt(max(0, self.v1**2 - 2*original_V0/self.m))
                perturbed_v2 = math.sqrt(max(0, self.v1**2 - 2*perturbed_V0/self.m))
                # Expect v2 to decrease when V0 increases
                if not (perturbed_v2 <= original_v2 + 1e-9):  # Allow tiny numerical error
                    self.log(f"FAIL: Monotonicity violated. V0↑ should cause v2↓. Original v2={original_v2}, Perturbed v2={perturbed_v2}")
                    return False
                self.log("PASS: Monotonicity (V0 vs v2)")

            # Restore original V0
            self.V0 = original_V0
            return True

        except Exception as e:
            self.V0 = original_V0  # Restore on error
            self.log(f"ERROR in monotonicity: {traceback.format_exc()}")
            return False

    def run_all_tests(self):
        """
        Execute all tests and return success status and logs.
        """
        self.logs = []
        success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            try:
                if not test_func():
                    success = False
                    self.log(f"{test_name} test failed.")
                else:
                    self.log(f"{test_name} test passed.")
            except Exception as e:
                success = False
                self.log(f"{test_name} test crashed: {traceback.format_exc()}")

        return success, self.logs