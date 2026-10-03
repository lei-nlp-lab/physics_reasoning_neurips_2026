import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Use getattr and .get() to avoid AttributeError.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Known constants from the problem statement
        self.x_given = 36.0  # displacement in 3 seconds
        self.v0_given = 18.0  # initial velocity (m/s)
        self.t_given = 3.0    # time (s)

        # Extract results with safe fallbacks
        self.v0 = getattr(solution_instance, 'v_0', self.results.get('v_0', self.v0_given))
        self.t = getattr(solution_instance, 't', self.results.get('t', self.t_given))
        self.a = getattr(solution_instance, 'a', self.results.get('a', None))
        self.t0 = getattr(solution_instance, 't_0', self.results.get('t_0', None))
        self.x_prime = getattr(solution_instance, 'x_prime', self.results.get("x'", None))
        self.x_m = getattr(solution_instance, 'x_m', self.results.get('x_m', None))

        # Convert to float if they are sympy objects
        for attr in ['v0', 't', 'a', 't0', 'x_prime', 'x_m']:
            val = getattr(self, attr)
            if val is not None and not isinstance(val, (int, float)):
                try:
                    setattr(self, attr, float(val))
                except:
                    pass

        # Define symbols for symbolic checks (optional)
        self.v0_sym = sp.symbols('v0')
        self.t_sym = sp.symbols('t')
        self.a_sym = sp.symbols('a')
        self.x_sym = sp.symbols('x')
        self.t0_sym = sp.symbols('t0')
        self.x_prime_sym = sp.symbols('x_prime')

        self.logs = []

    def log(self, message, success):
        """Helper to log test results."""
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {message}")
        return success

    def test_invariants(self):
        """Verify kinematic equations and conservation principles."""
        tests_passed = True

        # 1. Displacement equation: x = v0*t + 0.5*a*t^2
        if self.a is not None:
            x_calc = self.v0 * self.t + 0.5 * self.a * self.t**2
            if not math.isclose(x_calc, self.x_given, rel_tol=1e-2, abs_tol=1e-2):
                tests_passed = self.log(
                    f"Displacement equation mismatch: computed {x_calc}, expected {self.x_given}", False
                ) and tests_passed
            else:
                tests_passed = self.log("Displacement equation satisfied", True) and tests_passed

            # 2. Velocity-time relation: v1 = v0 + a*t
            v1 = self.v0 + self.a * self.t
            # Average velocity from two formulas: v_avg = (v0+v1)/2 = x/t
            v_avg1 = (self.v0 + v1) / 2
            v_avg2 = self.x_given / self.t
            if not math.isclose(v_avg1, v_avg2, rel_tol=1e-2, abs_tol=1e-2):
                tests_passed = self.log(
                    f"Average velocity mismatch: (v0+v1)/2 = {v_avg1}, x/t = {v_avg2}", False
                ) and tests_passed
            else:
                tests_passed = self.log("Average velocity consistency satisfied", True) and tests_passed

        # 3. Stopping time: t0 = -v0 / a (since a < 0)
        if self.a is not None and self.t0 is not None:
            t0_calc = -self.v0 / self.a
            if not math.isclose(t0_calc, self.t0, rel_tol=1e-2, abs_tol=1e-2):
                tests_passed = self.log(
                    f"Stopping time mismatch: computed {t0_calc}, result {self.t0}", False
                ) and tests_passed
            else:
                tests_passed = self.log("Stopping time equation satisfied", True) and tests_passed

        # 4. Displacement within stopping time: x' = (v0/2) * t0
        if self.t0 is not None and self.x_prime is not None:
            x_prime_calc = (self.v0 / 2) * self.t0
            if not math.isclose(x_prime_calc, self.x_prime, rel_tol=1e-2, abs_tol=1e-2):
                tests_passed = self.log(
                    f"Stopping displacement mismatch: computed {x_prime_calc}, result {self.x_prime}", False
                ) and tests_passed
            else:
                tests_passed = self.log("Stopping displacement equation satisfied", True) and tests_passed

        # 5. Check x_m (displacement if car stopped in t seconds)
        if self.x_m is not None:
            x_m_calc = (self.v0 / 2) * self.t
            if not math.isclose(x_m_calc, self.x_m, rel_tol=1e-2, abs_tol=1e-2):
                tests_passed = self.log(
                    f"x_m mismatch: computed {x_m_calc}, result {self.x_m}", False
                ) and tests_passed
            else:
                tests_passed = self.log("x_m (assuming stop in t) consistent", True) and tests_passed

        return tests_passed

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        tests_passed = True

        # 1. Acceleration from displacement equation
        if self.a is not None:
            a_calc = 2 * (self.x_given - self.v0 * self.t) / self.t**2
            if not math.isclose(a_calc, self.a, rel_tol=1e-2, abs_tol=1e-2):
                tests_passed = self.log(
                    f"Acceleration mismatch: computed {a_calc}, result {self.a}", False
                ) and tests_passed
            else:
                tests_passed = self.log("Acceleration consistent with displacement", True) and tests_passed

        # 2. Stopping time from acceleration
        if self.a is not None and self.t0 is not None:
            t0_calc = -self.v0 / self.a
            if not math.isclose(t0_calc, self.t0, rel_tol=1e-2, abs_tol=1e-2):
                tests_passed = self.log(
                    f"Stopping time mismatch: computed {t0_calc}, result {self.t0}", False
                ) and tests_passed
            else:
                tests_passed = self.log("Stopping time consistent with acceleration", True) and tests_passed

        # 3. x' from stopping time
        if self.t0 is not None and self.x_prime is not None:
            x_prime_calc = (self.v0 / 2) * self.t0
            if not math.isclose(x_prime_calc, self.x_prime, rel_tol=1e-2, abs_tol=1e-2):
                tests_passed = self.log(
                    f"x' mismatch: computed {x_prime_calc}, result {self.x_prime}", False
                ) and tests_passed
            else:
                tests_passed = self.log("x' consistent with stopping time", True) and tests_passed

        return tests_passed

    def test_monotonicity(self):
        """
        Perturb inputs and check if outputs change as expected.
        We test two perturbations:
        1. Increase v0 -> acceleration becomes more negative.
        2. Increase displacement x -> acceleration becomes less negative.
        """
        tests_passed = True

        if self.a is None:
            self.log("Cannot test monotonicity: acceleration not available", False)
            return False

        original_v0 = self.v0
        original_x = self.x_given

        # 1. Perturb v0 up by 1%
        perturb_v0 = original_v0 * 1.01
        a_v0_perturbed = 2 * (original_x - perturb_v0 * self.t) / self.t**2
        if not a_v0_perturbed < self.a:  # expect more negative
            tests_passed = self.log(
                f"Monotonicity w.r.t v0 failed: a({perturb_v0}) = {a_v0_perturbed}, expected < {self.a}", False
            ) and tests_passed
        else:
            tests_passed = self.log(
                f"Monotonicity w.r.t v0 passed: a decreases when v0 increases", True
            ) and tests_passed

        # 2. Perturb x up by 1%
        perturb_x = original_x * 1.01
        a_x_perturbed = 2 * (perturb_x - original_v0 * self.t) / self.t**2
        if not a_x_perturbed > self.a:  # expect less negative
            tests_passed = self.log(
                f"Monotonicity w.r.t x failed: a({perturb_x}) = {a_x_perturbed}, expected > {self.a}", False
            ) and tests_passed
        else:
            tests_passed = self.log(
                f"Monotonicity w.r.t x passed: a increases when x increases", True
            ) and tests_passed

        return tests_passed

    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        self.logs = []
        overall_success = True

        try:
            if not self.test_invariants():
                overall_success = False
        except Exception as e:
            self.logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
            overall_success = False

        try:
            if not self.test_trace_consistency():
                overall_success = False
        except Exception as e:
            self.logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
            overall_success = False

        try:
            if not self.test_monotonicity():
                overall_success = False
        except Exception as e:
            self.logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
            overall_success = False

        return overall_success, self.logs