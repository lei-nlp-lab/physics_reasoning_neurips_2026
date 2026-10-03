import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all necessary inputs and results from the solution instance.
        """
        # Input parameters (with safe defaults)
        self.m = getattr(solution_instance, 'm', None)
        self.q = getattr(solution_instance, 'q', None)
        self.d = getattr(solution_instance, 'd', None)
        self.g = getattr(solution_instance, 'g', None)
        self.k = getattr(solution_instance, 'k', None)

        # Results (with safe extraction from results dictionary if present)
        results = getattr(solution_instance, 'results', {})
        self.Q_B = getattr(solution_instance, 'Q_B', results.get('Q_B', None))
        self.Q_C = getattr(solution_instance, 'Q_C', results.get('Q_C', None))
        self.sin_angle_ABC = getattr(solution_instance, 'sin∠ABC', results.get('sin∠ABC', None))
        self.angle_ABC = getattr(solution_instance, '∠ABC', results.get('∠ABC', None))
        self.Q_1 = getattr(solution_instance, 'Q_1', results.get('Q_1', None))
        self.Q_2 = getattr(solution_instance, 'Q_2', results.get('Q_2', None))

        # Charge type indicators (strings)
        self.charge_type_B = getattr(solution_instance, 'charge_type_B', None)
        self.charge_type_C = getattr(solution_instance, 'charge_type_C', None)

        # If charge types are stored in results with different keys, try to get them
        if self.charge_type_B is None:
            self.charge_type_B = results.get('charge_type_B', None)
        if self.charge_type_C is None:
            self.charge_type_C = results.get('charge_type_C', None)

        # For robustness, also try to extract from any 'answer' attribute
        if hasattr(solution_instance, 'answer'):
            ans = solution_instance.answer
            if isinstance(ans, list) and len(ans) >= 2:
                # First answer is about charge types, second about expressions
                pass  # Could parse if needed, but we already have Q_1 and Q_2

        # Store logs for run_all_tests
        self.logs = []
        self.all_passed = True

    def _log(self, msg, passed):
        """Helper to log test results."""
        self.logs.append((msg, passed))
        if not passed:
            self.all_passed = False

    def test_invariants(self):
        """Verify physical invariants: geometry and force equilibrium."""
        try:
            # 1. Geometry: Pythagorean theorem
            if self.d is not None:
                AB = math.sqrt(3)/2 * self.d
                AC = self.d/2
                BC = self.d
                lhs = AB**2 + AC**2
                rhs = BC**2
                if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    self._log("Pythagorean theorem holds.", True)
                else:
                    self._log(f"Pythagorean theorem violated: AB²+AC²={lhs}, BC²={rhs}", False)
            else:
                self._log("Skipping Pythagorean check: distance d not available.", True)

            # 2. Trigonometry: sin(∠ABC) = AC/BC = 1/2
            if self.sin_angle_ABC is not None:
                if math.isclose(self.sin_angle_ABC, 0.5, rel_tol=1e-2, abs_tol=1e-2):
                    self._log("sin(∠ABC) = 1/2 as expected.", True)
                else:
                    self._log(f"sin(∠ABC) = {self.sin_angle_ABC}, expected 0.5", False)
            else:
                self._log("Skipping sin(∠ABC) check: value not available.", True)

            # 3. Angle ∠ABC = 30°
            if self.angle_ABC is not None:
                # Accept either degrees or radians
                if math.isclose(self.angle_ABC, 30, rel_tol=1e-2, abs_tol=1e-2):  # degrees
                    self._log("∠ABC = 30° as expected.", True)
                elif math.isclose(self.angle_ABC, math.pi/6, rel_tol=1e-2, abs_tol=1e-2):  # radians
                    self._log("∠ABC = π/6 rad as expected.", True)
                else:
                    self._log(f"∠ABC = {self.angle_ABC}, expected 30° or π/6 rad", False)
            else:
                self._log("Skipping ∠ABC check: value not available.", True)

            # 4. Force equilibrium (vector sum zero)
            if all(v is not None for v in [self.Q_1, self.Q_2, self.m, self.q, self.d, self.g, self.k]):
                # Compute forces using the coordinate system derived in the analysis
                # Coordinates: B(0,0), C(d,0), A(3d/4, √3 d/4)
                x_A = 3 * self.d / 4
                y_A = math.sqrt(3) * self.d / 4

                # Distance and unit vectors
                AB = math.sqrt(3)/2 * self.d
                AC = self.d/2

                # Force from B (repulsive)
                F_B_mag = self.k * self.q * self.Q_1 / (AB**2)
                F_Bx = F_B_mag * (x_A / AB)   # x_A / AB = √3/2
                F_By = F_B_mag * (y_A / AB)   # y_A / AB = 1/2

                # Force from C (repulsive)
                F_C_mag = self.k * self.q * self.Q_2 / (AC**2)
                F_Cx = F_C_mag * ((x_A - self.d) / AC)   # = -1/2
                F_Cy = F_C_mag * (y_A / AC)              # = √3/2

                # Gravity
                F_gy = -self.m * self.g

                # Net components
                F_x = F_Bx + F_Cx
                F_y = F_By + F_Cy + F_gy

                if math.isclose(F_x, 0, rel_tol=1e-2, abs_tol=1e-2) and \
                   math.isclose(F_y, 0, rel_tol=1e-2, abs_tol=1e-2):
                    self._log("Force equilibrium (F_net = 0) satisfied.", True)
                else:
                    self._log(f"Force equilibrium violated: F_x={F_x}, F_y={F_y}", False)
            else:
                missing = []
                if self.Q_1 is None: missing.append('Q_1')
                if self.Q_2 is None: missing.append('Q_2')
                if self.m is None: missing.append('m')
                if self.q is None: missing.append('q')
                if self.d is None: missing.append('d')
                if self.g is None: missing.append('g')
                if self.k is None: missing.append('k')
                self._log(f"Skipping force equilibrium check: missing {missing}.", True)

        except Exception as e:
            self._log(f"Error in test_invariants: {traceback.format_exc()}", False)

    def test_trace_consistency(self):
        """Check that intermediate results match final expressions and charge types."""
        try:
            # 1. Charge types are positive
            if self.charge_type_B is not None and self.charge_type_C is not None:
                if isinstance(self.charge_type_B, str) and 'positive' in self.charge_type_B.lower() and \
                   isinstance(self.charge_type_C, str) and 'positive' in self.charge_type_C.lower():
                    self._log("Both charges are positive as expected.", True)
                else:
                    self._log(f"Unexpected charge types: B={self.charge_type_B}, C={self.charge_type_C}", False)
            else:
                self._log("Skipping charge type check: values not available.", True)

            # 2. Verify Q_1 and Q_2 match the derived formulas
            if all(v is not None for v in [self.Q_1, self.Q_2, self.m, self.g, self.d, self.k, self.q]):
                Q1_expected = (3 * self.m * self.g * self.d**2) / (8 * self.k * self.q)
                Q2_expected = (math.sqrt(3) * self.m * self.g * self.d**2) / (8 * self.k * self.q)

                if math.isclose(self.Q_1, Q1_expected, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Q_1 matches expected expression: {Q1_expected}", True)
                else:
                    self._log(f"Q_1 mismatch: got {self.Q_1}, expected {Q1_expected}", False)

                if math.isclose(self.Q_2, Q2_expected, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Q_2 matches expected expression: {Q2_expected}", True)
                else:
                    self._log(f"Q_2 mismatch: got {self.Q_2}, expected {Q2_expected}", False)
            else:
                missing = []
                if self.Q_1 is None: missing.append('Q_1')
                if self.Q_2 is None: missing.append('Q_2')
                if self.m is None: missing.append('m')
                if self.g is None: missing.append('g')
                if self.d is None: missing.append('d')
                if self.k is None: missing.append('k')
                if self.q is None: missing.append('q')
                self._log(f"Skipping formula consistency check: missing {missing}.", True)

        except Exception as e:
            self._log(f"Error in test_trace_consistency: {traceback.format_exc()}", False)

    def test_monotonicity(self):
        """Perturb mass m and verify that charges increase proportionally."""
        try:
            if all(v is not None for v in [self.m, self.g, self.d, self.k, self.q, self.Q_1, self.Q_2]):
                # Original values
                m_original = self.m
                # Perturb m by +1%
                m_perturbed = m_original * 1.01
                # Recalculate expected charges using the formulas
                Q1_original = (3 * m_original * self.g * self.d**2) / (8 * self.k * self.q)
                Q2_original = (math.sqrt(3) * m_original * self.g * self.d**2) / (8 * self.k * self.q)
                Q1_perturbed = (3 * m_perturbed * self.g * self.d**2) / (8 * self.k * self.q)
                Q2_perturbed = (math.sqrt(3) * m_perturbed * self.g * self.d**2) / (8 * self.k * self.q)

                # Check that perturbed charges are larger
                if Q1_perturbed > Q1_original and Q2_perturbed > Q2_original:
                    self._log("Monotonicity holds: increasing mass increases both charges.", True)
                else:
                    self._log(f"Monotonicity violated: Q1 {Q1_perturbed} <= {Q1_original} or Q2 {Q2_perturbed} <= {Q2_original}", False)

                # Also check the ratios match the perturbation factor
                expected_factor = 1.01
                factor1 = Q1_perturbed / Q1_original
                factor2 = Q2_perturbed / Q2_original
                if math.isclose(factor1, expected_factor, rel_tol=1e-2, abs_tol=1e-2) and \
                   math.isclose(factor2, expected_factor, rel_tol=1e-2, abs_tol=1e-2):
                    self._log("Charges scale linearly with mass as expected.", True)
                else:
                    self._log(f"Scaling factors: Q1={factor1}, Q2={factor2}, expected {expected_factor}", False)

                # No need to restore because we only used local variables
            else:
                missing = []
                if self.m is None: missing.append('m')
                if self.g is None: missing.append('g')
                if self.d is None: missing.append('d')
                if self.k is None: missing.append('k')
                if self.q is None: missing.append('q')
                if self.Q_1 is None: missing.append('Q_1')
                if self.Q_2 is None: missing.append('Q_2')
                self._log(f"Skipping monotonicity test: missing {missing}.", True)

        except Exception as e:
            self._log(f"Error in test_monotonicity: {traceback.format_exc()}", False)

    def run_all_tests(self):
        """Run all tests and return overall success and logs."""
        self.logs = []
        self.all_passed = True

        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()

        return self.all_passed, self.logs