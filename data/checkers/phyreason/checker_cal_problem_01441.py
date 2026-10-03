import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize with safe attribute access to avoid AttributeError in sandbox.
        """
        self.solution = solution_instance
        self.logs = []

        # Safely extract inputs and results with fallbacks
        self.v = getattr(solution_instance, 'v', getattr(solution_instance, 'results', {}).get('v', 0.0))
        self.a = getattr(solution_instance, 'a', getattr(solution_instance, 'results', {}).get('a', 0.0))
        self.t = getattr(solution_instance, 't', getattr(solution_instance, 'results', {}).get('t', 0.0))
        self.x_A = getattr(solution_instance, 'x_A', getattr(solution_instance, 'results', {}).get('x_A', 0.0))
        self.x_B = getattr(solution_instance, 'x_B', getattr(solution_instance, 'results', {}).get('x_B', 0.0))
        self.delta_x = getattr(solution_instance, 'delta_x', getattr(solution_instance, 'results', {}).get('delta_x', 0.0))
        
        # Expected values from steps analysis
        self.expected_v = 10.0
        self.expected_a = 2.0
        self.expected_t_catchup = 10.0
        self.expected_max_distance = 25.0

    def log(self, msg, success=None):
        """Helper for logging test results."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {msg}")
        else:
            self.logs.append(f"INFO: {msg}")

    def test_invariants(self):
        """
        Verify physical invariants:
        1. Uniform motion of Car A: x_A = v * t (at any time)
        2. Uniform accelerated motion of Car B: x_B = 0.5 * a * t^2 (starting from rest)
        3. Equality of speeds condition: v = a * t1 -> t1 = v / a
        4. Maximum distance occurs when speeds are equal (relative velocity zero)
        """
        try:
            # Test 1: Check Car A speed from step 1
            if not math.isclose(self.v, self.expected_v, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Car A speed mismatch: got {self.v}, expected {self.expected_v}", False)
                return False
            self.log(f"Car A speed v = {self.v} m/s matches expected {self.expected_v} m/s", True)

            # Test 2: Check Car B acceleration from step 2
            if not math.isclose(self.a, self.expected_a, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Car B acceleration mismatch: got {self.a}, expected {self.expected_a}", False)
                return False
            self.log(f"Car B acceleration a = {self.a} m/s² matches expected {self.expected_a} m/s²", True)

            # Test 3: Symbolic check of motion equations
            t_sym = sp.symbols('t', positive=True)
            v_sym, a_sym = sp.symbols('v a')
            x_A_formula = v_sym * t_sym
            x_B_formula = 0.5 * a_sym * t_sym**2
            # Equality condition for catch-up: x_A = x_B -> v*t = 0.5*a*t^2 -> t = 2*v/a
            catchup_condition = sp.Eq(x_A_formula, x_B_formula)
            catchup_solution = sp.solve(catchup_condition, t_sym)
            # Should get t = 0 and t = 2*v/a
            if not catchup_solution:
                self.log("Symbolic catch-up condition failed to solve", False)
                return False
            # Evaluate numerically
            t_catchup_expr = 2 * self.v / self.a
            if not math.isclose(self.t, t_catchup_expr, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Catch-up time mismatch: got {self.t}, expected {t_catchup_expr} from v={self.v}, a={self.a}", False)
                return False
            self.log(f"Catch-up time t = {self.t} s matches theoretical 2v/a = {t_catchup_expr} s", True)

            # Test 4: Maximum distance condition (speeds equal)
            t1 = self.v / self.a  # time when speeds equal
            x_A_at_t1 = self.v * t1
            x_B_at_t1 = 0.5 * self.a * t1**2
            max_distance_calc = x_A_at_t1 - x_B_at_t1
            if not math.isclose(self.delta_x, max_distance_calc, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Max distance mismatch: got {self.delta_x}, expected {max_distance_calc} from v={self.v}, a={self.a}", False)
                return False
            self.log(f"Max distance Δx = {self.delta_x} m matches theoretical v²/(2a) = {max_distance_calc} m", True)

            return True
        except Exception as e:
            self.log(f"Invariant test failed with error: {str(e)}", False)
            return False

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        Check that computed displacements at t=10s and at speed-equal time are consistent.
        """
        try:
            # Check catch-up time consistency with displacements
            x_A_catchup = self.v * self.t
            x_B_catchup = 0.5 * self.a * self.t**2
            if not math.isclose(x_A_catchup, x_B_catchup, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Displacements at catch-up time mismatch: x_A={x_A_catchup}, x_B={x_B_catchup}", False)
                return False
            self.log(f"At t={self.t} s, x_A={x_A_catchup} m = x_B={x_B_catchup} m (catch-up condition satisfied)", True)

            # Check maximum distance consistency with speed-equal time
            t1 = self.v / self.a
            x_A_t1 = self.v * t1
            x_B_t1 = 0.5 * self.a * t1**2
            delta_x_calc = x_A_t1 - x_B_t1
            if not math.isclose(self.delta_x, delta_x_calc, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Max distance mismatch with intermediate calc: got {self.delta_x}, expected {delta_x_calc}", False)
                return False
            self.log(f"Max distance {self.delta_x} m consistent with x_A({t1}s)-x_B({t1}s) = {delta_x_calc} m", True)

            # Verify against target answers
            if not math.isclose(self.t, self.expected_t_catchup, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Catch-up time {self.t} s does not match target {self.expected_t_catchup} s", False)
                return False
            if not math.isclose(self.delta_x, self.expected_max_distance, rel_tol=0.02, abs_tol=0.02):
                self.log(f"Max distance {self.delta_x} m does not match target {self.expected_max_distance} m", False)
                return False
            self.log(f"All results match target answers: t={self.t} s, Δx={self.delta_x} m", True)
            return True
        except Exception as e:
            self.log(f"Trace consistency test failed: {str(e)}", False)
            return False

    def test_monotonicity(self):
        """
        Perturb input parameter (Car A speed v) and check monotonic trends:
        - Increasing v should increase catch-up time (t = 2v/a)
        - Increasing v should increase maximum distance (Δx = v²/(2a))
        Restore original v after perturbation.
        """
        original_v = self.v
        perturb_factors = [0.9, 1.1]  # -10%, +10%
        try:
            for factor in perturb_factors:
                v_perturbed = original_v * factor
                # Recalculate expected outcomes using formulas
                t_perturbed = 2 * v_perturbed / self.a
                delta_x_perturbed = (v_perturbed**2) / (2 * self.a)

                # Check monotonicity: if v increases, both t and Δx should increase
                if factor > 1:
                    if not (t_perturbed > self.t and delta_x_perturbed > self.delta_x):
                        self.log(f"Monotonicity failed for v increase: t {t_perturbed} vs {self.t}, Δx {delta_x_perturbed} vs {self.delta_x}", False)
                        return False
                else:
                    if not (t_perturbed < self.t and delta_x_perturbed < self.delta_x):
                        self.log(f"Monotonicity failed for v decrease: t {t_perturbed} vs {self.t}, Δx {delta_x_perturbed} vs {self.delta_x}", False)
                        return False

            self.log("Monotonicity test passed: increasing v increases catch-up time and max distance", True)
            return True
        except Exception as e:
            self.log(f"Monotonicity test failed: {str(e)}", False)
            return False
        finally:
            # Restore original value (though self.v is just a local copy)
            self.v = original_v

    def run_all_tests(self):
        """
        Run all tests and return overall success and logs.
        """
        self.logs = []
        overall_success = True

        tests = [
            ("Physical Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            self.log(f"Running {test_name}...")
            try:
                success = test_func()
                overall_success = overall_success and success
            except Exception as e:
                self.log(f"Test {test_name} crashed: {traceback.format_exc()}", False)
                overall_success = False

        return overall_success, self.logs