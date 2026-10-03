import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Safely extract inputs and results from solution instance."""
        self.solution = solution_instance
        self.inputs = {}
        self.results = {}

        # Extract inputs with fallbacks
        self.inputs['m'] = getattr(solution_instance, 'm', getattr(solution_instance, 'inputs', {}).get('m', 2.0))
        self.inputs['R'] = getattr(solution_instance, 'R', getattr(solution_instance, 'inputs', {}).get('R', 0.5))
        self.inputs['E_p'] = getattr(solution_instance, 'E_p', getattr(solution_instance, 'inputs', {}).get('E_p', 49.0))
        self.inputs['g'] = getattr(solution_instance, 'g', getattr(solution_instance, 'inputs', {}).get('g', 10.0))

        # Extract results with fallbacks (using KB mapping)
        results_dict = getattr(solution_instance, 'results', {})
        self.results['v_A'] = getattr(solution_instance, 'v_A', results_dict.get('v_A', 0.0))
        self.results['v_C_min'] = getattr(solution_instance, 'v_C', results_dict.get('v_C', 0.0))  # Minimum required
        self.results['v_C_actual'] = getattr(solution_instance, 'v_C_actual', 
                                            getattr(solution_instance, 'v_C_prime', 
                                                   results_dict.get('v_C_prime', results_dict.get('v_C\'', 0.0))))
        self.results['F_N'] = getattr(solution_instance, 'F_N', results_dict.get('F_N', 0.0))
        self.results['F_N_prime'] = getattr(solution_instance, 'F_N_prime', 
                                           getattr(solution_instance, 'F_N_prime', 
                                                  results_dict.get('F_N_prime', results_dict.get('F_N\'', 0.0))))

        # Derived quantities for checking
        self.logs = []

    def log(self, msg):
        self.logs.append(msg)

    def test_invariants(self):
        """Verify energy conservation and Newton's laws."""
        try:
            m, R, E_p, g = self.inputs['m'], self.inputs['R'], self.inputs['E_p'], self.inputs['g']
            v_A, v_C_min, v_C_actual, F_N = self.results['v_A'], self.results['v_C_min'], self.results['v_C_actual'], self.results['F_N']

            # 1. Energy conservation from spring to point A
            K_A = 0.5 * m * v_A**2
            if not math.isclose(E_p, K_A, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Energy invariant violation: Spring energy {E_p} != kinetic at A {K_A}")
                return False

            # 2. Energy conservation from A to C (including potential)
            K_C = 0.5 * m * v_C_actual**2
            U_C = m * g * 2 * R  # Height difference from A to C is 2R
            total_energy_at_C = K_C + U_C
            if not math.isclose(E_p, total_energy_at_C, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Energy invariant violation: Initial energy {E_p} != total at C {total_energy_at_C}")
                return False

            # 3. Newton's second law at point C: F_N + mg = m*v_C_actual^2 / R
            centripetal_force = m * v_C_actual**2 / R
            required_normal = centripetal_force - m * g
            if not math.isclose(F_N, required_normal, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Newton's 2nd law violation at C: F_N {F_N} != m*v^2/R - mg {required_normal}")
                return False

            # 4. Newton's third law: F_N' should be opposite to F_N
            if not math.isclose(self.results['F_N_prime'], -F_N, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Newton's 3rd law violation: F_N' {self.results['F_N_prime']} != -F_N {-F_N}")
                return False

            self.log("All invariants passed.")
            return True
        except Exception as e:
            self.log(f"Invariant test failed with error: {traceback.format_exc()}")
            return False

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            m, R, g = self.inputs['m'], self.inputs['R'], self.inputs['g']
            v_A, v_C_min, v_C_actual, F_N = self.results['v_A'], self.results['v_C_min'], self.results['v_C_actual'], self.results['F_N']

            # 1. v_A from spring energy
            expected_v_A = math.sqrt(2 * self.inputs['E_p'] / m)
            if not math.isclose(v_A, expected_v_A, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Trace inconsistency: v_A {v_A} != sqrt(2E_p/m) {expected_v_A}")
                return False

            # 2. Minimum v_C for circular motion
            expected_v_C_min = math.sqrt(g * R)
            if not math.isclose(v_C_min, expected_v_C_min, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Trace inconsistency: v_C_min {v_C_min} != sqrt(gR) {expected_v_C_min}")
                return False

            # 3. Actual v_C from energy conservation
            expected_v_C_actual = math.sqrt(v_A**2 - 4 * g * R)
            if not math.isclose(v_C_actual, expected_v_C_actual, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Trace inconsistency: v_C_actual {v_C_actual} != sqrt(v_A^2 - 4gR) {expected_v_C_actual}")
                return False

            # 4. Normal force F_N
            expected_F_N = m * (v_C_actual**2 / R - g)
            if not math.isclose(F_N, expected_F_N, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Trace inconsistency: F_N {F_N} != m*(v_C^2/R - g) {expected_F_N}")
                return False

            # 5. Check that actual v_C >= minimum v_C (ball can reach C)
            if v_C_actual < v_C_min - 0.1:  # Small tolerance for floating point
                self.log(f"Trace inconsistency: v_C_actual {v_C_actual} < v_C_min {v_C_min}, ball cannot reach C")
                return False

            self.log("All trace consistency checks passed.")
            return True
        except Exception as e:
            self.log(f"Trace consistency test failed with error: {traceback.format_exc()}")
            return False

    def test_monotonicity(self):
        """Perturb spring energy and check F_N trend."""
        try:
            original_E_p = self.inputs['E_p']
            m, R, g = self.inputs['m'], self.inputs['R'], self.inputs['g']
            original_v_A = math.sqrt(2 * original_E_p / m)
            original_v_C_actual = math.sqrt(original_v_A**2 - 4 * g * R)
            original_F_N = m * (original_v_C_actual**2 / R - g)

            # Increase E_p by 10%
            perturbed_E_p = original_E_p * 1.1
            perturbed_v_A = math.sqrt(2 * perturbed_E_p / m)
            perturbed_v_C_actual = math.sqrt(perturbed_v_A**2 - 4 * g * R)
            perturbed_F_N = m * (perturbed_v_C_actual**2 / R - g)

            # F_N should increase with increased E_p
            if not perturbed_F_N > original_F_N:
                self.log(f"Monotonicity violation: F_N did not increase when E_p increased. Original F_N: {original_F_N}, Perturbed F_N: {perturbed_F_N}")
                return False

            # Restore original value (conceptually)
            self.log(f"Monotonicity check passed: F_N increases with E_p (ΔF_N = {perturbed_F_N - original_F_N:.3f})")
            return True
        except Exception as e:
            self.log(f"Monotonicity test failed with error: {traceback.format_exc()}")
            return False

    def run_all_tests(self):
        """Execute all tests and return overall success and logs."""
        self.logs = []
        success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            self.log(f"\n--- Running {test_name} Test ---")
            if not test_func():
                success = False
                self.log(f"{test_name} test FAILED")
            else:
                self.log(f"{test_name} test PASSED")

        return success, self.logs