import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        # Given constants from the problem (SI units)
        self.q_given = -1e-2  # C
        self.W_AB_given = 0.3  # J (work done by electric force from A to B)
        self.W_BC_given = -0.3  # J (work done by electric force from B to C)
        self.BC_given = 10 * math.sqrt(3) / 100  # m (10√3 cm converted to meters)
        self.theta_given = 30  # degrees

        # Extract student inputs with safe fallbacks
        self.q = getattr(solution_instance, 'q', self.q_given)
        self.W_AB = getattr(solution_instance, 'W_AB', self.W_AB_given)
        self.W_BC = getattr(solution_instance, 'W_BC', self.W_BC_given)
        self.BC = getattr(solution_instance, 'BC', self.BC_given)
        self.theta = getattr(solution_instance, 'theta', self.theta_given)

        # Extract results from the solution instance's results dictionary or attributes
        results = getattr(solution_instance, 'results', {})
        self.U_AB = getattr(solution_instance, 'U_AB', results.get('U_AB', None))
        self.U_BC = getattr(solution_instance, 'U_BC', results.get('U_BC', None))
        self.U_BD = getattr(solution_instance, 'U_BD', results.get('U_BD', None))
        self.d_BD = getattr(solution_instance, 'd_BD', results.get('d_BD', None))
        self.E = getattr(solution_instance, 'E', results.get('E', None))
        self.phi_relation = getattr(solution_instance, 'phi_relation', results.get('phi_A,phi_C', None))

        # Expected values computed from given constants
        self.U_AB_expected = self.W_AB_given / self.q_given
        self.U_BC_expected = self.W_BC_given / self.q_given
        self.d_BD_expected = self.BC_given * math.cos(math.radians(self.theta_given))
        self.U_BD_expected = self.U_BC_expected  # from step_8
        self.E_expected = self.U_BD_expected / self.d_BD_expected

        # Student values as a dictionary for convenience
        self.student = {
            'U_AB': self.U_AB,
            'U_BC': self.U_BC,
            'U_BD': self.U_BD,
            'd_BD': self.d_BD,
            'E': self.E,
            'BC': self.BC,
            'q': self.q,
            'W_AB': self.W_AB,
            'W_BC': self.W_BC,
            'theta': self.theta
        }

        # Expected values as a dictionary
        self.expected = {
            'U_AB': self.U_AB_expected,
            'U_BC': self.U_BC_expected,
            'U_BD': self.U_BD_expected,
            'd_BD': self.d_BD_expected,
            'E': self.E_expected
        }

    def compute_E_from_inputs(self, q, W_BC, BC, theta):
        """
        Compute the electric field magnitude from given inputs using the physical formulas.
        """
        U_BC = W_BC / q
        d_BD = BC * math.cos(math.radians(theta))
        E = U_BC / d_BD
        return E

    def test_invariants(self):
        """
        Check physical invariants and correctness of student results.
        """
        logs = []
        success = True

        # Helper for fuzzy comparison
        def is_close(a, b, name, rel_tol=1e-2, abs_tol=1e-2):
            if a is None or b is None:
                logs.append(f"  {name}: Missing value(s).")
                return False
            if math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
                logs.append(f"  {name}: PASS (expected {b:.3g}, got {a:.3g})")
                return True
            else:
                logs.append(f"  {name}: FAIL (expected {b:.3g}, got {a:.3g})")
                return False

        logs.append("Checking invariants:")

        # 1. Check that U_AB + U_BC = 0 (since φ_A = φ_C)
        if self.U_AB is not None and self.U_BC is not None:
            sum_AB_BC = self.U_AB + self.U_BC
            if math.isclose(sum_AB_BC, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("  U_AB + U_BC = 0: PASS")
            else:
                logs.append(f"  U_AB + U_BC = 0: FAIL (sum = {sum_AB_BC:.3g})")
                success = False
        else:
            logs.append("  U_AB + U_BC = 0: Missing values.")
            success = False

        # 2. Check that U_BD = U_BC (step_8)
        if self.U_BD is not None and self.U_BC is not None:
            if math.isclose(self.U_BD, self.U_BC, rel_tol=1e-2, abs_tol=1e-2):
                logs.append("  U_BD = U_BC: PASS")
            else:
                logs.append(f"  U_BD = U_BC: FAIL (U_BD = {self.U_BD:.3g}, U_BC = {self.U_BC:.3g})")
                success = False
        else:
            logs.append("  U_BD = U_BC: Missing values.")
            success = False

        # 3. Check that d_BD = BC * cos(theta) (step_9)
        if self.d_BD is not None and self.BC is not None:
            d_BD_calc = self.BC * math.cos(math.radians(self.theta))
            if math.isclose(self.d_BD, d_BD_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"  d_BD = BC * cos(theta): PASS (calculated {d_BD_calc:.3g})")
            else:
                logs.append(f"  d_BD = BC * cos(theta): FAIL (expected {d_BD_calc:.3g}, got {self.d_BD:.3g})")
                success = False
        else:
            logs.append("  d_BD = BC * cos(theta): Missing values.")
            success = False

        # 4. Check that E = U_BD / d_BD (step_7)
        if self.E is not None and self.U_BD is not None and self.d_BD is not None and self.d_BD != 0:
            E_calc = self.U_BD / self.d_BD
            if math.isclose(self.E, E_calc, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"  E = U_BD / d_BD: PASS (calculated {E_calc:.3g})")
            else:
                logs.append(f"  E = U_BD / d_BD: FAIL (expected {E_calc:.3g}, got {self.E:.3g})")
                success = False
        else:
            logs.append("  E = U_BD / d_BD: Missing values or d_BD=0.")
            success = False

        logs.append("Checking correctness against expected values:")
        # Compare student results to expected values from given constants
        for key in ['U_AB', 'U_BC', 'U_BD', 'd_BD', 'E']:
            if self.student[key] is not None:
                if not is_close(self.student[key], self.expected[key], key):
                    success = False
            else:
                logs.append(f"  {key}: Missing student value.")
                success = False

        return success, logs

    def test_trace_consistency(self):
        """
        Check consistency between intermediate results and final answer.
        """
        logs = []
        success = True

        logs.append("Checking trace consistency:")
        try:
            # Recompute E from student's intermediate results (if available)
            if all(v is not None for v in [self.U_BD, self.d_BD]):
                E_from_trace = self.U_BD / self.d_BD
                if self.E is not None and math.isclose(self.E, E_from_trace, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"  E from U_BD and d_BD: PASS (E = {E_from_trace:.3g})")
                else:
                    logs.append(f"  E from U_BD and d_BD: FAIL (computed {E_from_trace:.3g}, student's E = {self.E})")
                    success = False
            else:
                logs.append("  E from U_BD and d_BD: Missing intermediate values.")
                success = False

            # Check that U_AB and U_BC lead to φ_A = φ_C
            if self.U_AB is not None and self.U_BC is not None:
                sum_AB_BC = self.U_AB + self.U_BC
                if math.isclose(sum_AB_BC, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append("  φ_A = φ_C from U_AB + U_BC = 0: PASS")
                else:
                    logs.append(f"  φ_A = φ_C from U_AB + U_BC = 0: FAIL (sum = {sum_AB_BC:.3g})")
                    success = False
            else:
                logs.append("  φ_A = φ_C from U_AB + U_BC = 0: Missing U_AB or U_BC.")
                success = False

            # Check that d_BD is consistent with BC and theta
            if self.d_BD is not None and self.BC is not None:
                d_BD_calc = self.BC * math.cos(math.radians(self.theta))
                if math.isclose(self.d_BD, d_BD_calc, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"  d_BD consistent with BC and theta: PASS")
                else:
                    logs.append(f"  d_BD consistent with BC and theta: FAIL (calculated {d_BD_calc:.3g}, got {self.d_BD:.3g})")
                    success = False
            else:
                logs.append("  d_BD consistent with BC and theta: Missing values.")
                success = False

        except Exception as e:
            logs.append(f"  Exception in trace consistency: {traceback.format_exc()}")
            success = False

        return success, logs

    def test_monotonicity(self):
        """
        Perturb an input (BC) and verify that the electric field changes as expected.
        """
        logs = []
        success = True

        logs.append("Checking monotonicity (perturbing BC):")
        try:
            # Use student's input values to compute E
            inputs = {
                'q': self.student['q'],
                'W_BC': self.student['W_BC'],
                'BC': self.student['BC'],
                'theta': self.student['theta']
            }
            E_orig = self.compute_E_from_inputs(**inputs)
            # Perturb BC by +1%
            inputs_perturbed = inputs.copy()
            inputs_perturbed['BC'] *= 1.01
            E_pert = self.compute_E_from_inputs(**inputs_perturbed)

            # Check that E decreases when BC increases (since E ∝ 1/BC)
            if E_pert < E_orig:
                logs.append(f"  Monotonicity: PASS (E decreases from {E_orig:.3g} to {E_pert:.3g} when BC increases by 1%)")
            else:
                logs.append(f"  Monotonicity: FAIL (E does not decrease as expected: E_orig={E_orig:.3g}, E_pert={E_pert:.3g})")
                success = False

            # Also check that the student's E is close to the computed E_orig
            if self.E is not None and math.isclose(self.E, E_orig, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"  Student's E matches computed E from inputs: PASS (E = {E_orig:.3g})")
            else:
                logs.append(f"  Student's E matches computed E from inputs: FAIL (computed {E_orig:.3g}, student's {self.E})")
                success = False

        except Exception as e:
            logs.append(f"  Exception in monotonicity: {traceback.format_exc()}")
            success = False

        return success, logs

    def run_all_tests(self):
        """
        Run all tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True

        # Run each test
        tests = [self.test_invariants, self.test_trace_consistency, self.test_monotonicity]
        test_names = ["Invariants", "Trace Consistency", "Monotonicity"]

        for test, name in zip(tests, test_names):
            all_logs.append(f"\n=== {name} Test ===")
            try:
                success, logs = test()
                overall_success = overall_success and success
                all_logs.extend(logs)
            except Exception as e:
                all_logs.append(f"  Exception during test: {traceback.format_exc()}")
                overall_success = False

        return overall_success, all_logs