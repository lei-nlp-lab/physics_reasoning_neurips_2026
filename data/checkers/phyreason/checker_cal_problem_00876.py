import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract input parameters
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', 1.0))
        self.q = getattr(solution_instance, 'q', solution_instance.results.get('q', 1.0))
        self.v = getattr(solution_instance, 'v', solution_instance.results.get('v', 1.0))
        self.a = getattr(solution_instance, 'a', solution_instance.results.get('a', 1.0))

        # Safely extract results using the provided key mapping
        self.R = getattr(solution_instance, 'R', solution_instance.results.get('R', 0.0))
        self.B = getattr(solution_instance, 'B', solution_instance.results.get('B', 0.0))
        self.T = getattr(solution_instance, 'T', solution_instance.results.get('T', 0.0))
        self.t = getattr(solution_instance, 't', solution_instance.results.get('t', 0.0))

        # Constants
        self.sin60 = math.sin(math.radians(60))
        self.angle_degrees = 120
        self.angle_fraction = self.angle_degrees / 360.0

        # For symbolic checks
        self.symbols = sp.symbols('m q v a R B T t')
        self.m_sym, self.q_sym, self.v_sym, self.a_sym, self.R_sym, self.B_sym, self.T_sym, self.t_sym = self.symbols

    def test_invariants(self):
        logs = []
        success = True

        try:
            # 1. Geometric invariant: sin(60°) = a/R
            expected_R = self.a / self.sin60
            if not math.isclose(self.R, expected_R, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Geometric invariant. R expected {expected_R}, got {self.R}")
                success = False
            else:
                logs.append("PASS: Geometric invariant (R = a / sin60)")

            # 2. Lorentz force as centripetal force: q*v*B = m*v^2/R -> B = m*v/(q*R)
            # Symbolic check
            eq_lorentz = sp.Eq(self.q_sym * self.v_sym * self.B_sym, self.m_sym * self.v_sym**2 / self.R_sym)
            sol_B = sp.solve(eq_lorentz, self.B_sym)[0]
            expected_B_sym = sol_B.subs({
                self.m_sym: self.m,
                self.v_sym: self.v,
                self.q_sym: self.q,
                self.R_sym: self.R
            })
            # Numerical check
            expected_B_num = (self.m * self.v) / (self.q * self.R)
            if not (math.isclose(self.B, float(expected_B_sym), rel_tol=1e-2, abs_tol=1e-2) and
                    math.isclose(self.B, expected_B_num, rel_tol=1e-2, abs_tol=1e-2)):
                logs.append(f"FAIL: Lorentz force invariant. B expected ~{expected_B_num}, got {self.B}")
                success = False
            else:
                logs.append("PASS: Lorentz force invariant (q*v*B = m*v^2/R)")

            # 3. Period invariant: T = 2*pi*R/v
            expected_T = 2 * math.pi * self.R / self.v
            if not math.isclose(self.T, expected_T, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Period invariant. T expected {expected_T}, got {self.T}")
                success = False
            else:
                logs.append("PASS: Period invariant (T = 2πR/v)")

            # 4. Speed constancy (energy conservation): speed remains v throughout
            # Since magnetic force does no work, speed is constant. We assume the solution uses the same v.
            # This is inherent in the formulas, so we just note it.
            logs.append("INFO: Speed is constant (magnetic force does no work)")

        except Exception as e:
            logs.append(f"ERROR in test_invariants: {str(e)}")
            success = False

        return success, logs

    def test_trace_consistency(self):
        logs = []
        success = True

        try:
            # Check that B is consistent with R and the Lorentz force
            expected_B_from_R = (self.m * self.v) / (self.q * self.R)
            if not math.isclose(self.B, expected_B_from_R, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: B inconsistent with R. B from R: {expected_B_from_R}, got {self.B}")
                success = False
            else:
                logs.append("PASS: B consistent with R (via Lorentz force)")

            # Check that t is consistent with T and the angle (120°)
            expected_t_from_T = self.angle_fraction * self.T
            if not math.isclose(self.t, expected_t_from_T, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: t inconsistent with T. t from T: {expected_t_from_T}, got {self.t}")
                success = False
            else:
                logs.append("PASS: t consistent with T (t = (120/360) * T)")

            # Check that R, T, B, t are all positive (physical)
            for name, val in [('R', self.R), ('T', self.T), ('B', self.B), ('t', self.t)]:
                if val <= 0:
                    logs.append(f"FAIL: {name} must be positive, got {val}")
                    success = False

        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {str(e)}")
            success = False

        return success, logs

    def test_monotonicity(self):
        logs = []
        success = True

        try:
            # Perturb input 'a' by +10%
            a_original = self.a
            a_perturbed = a_original * 1.1

            # Recalculate using formulas from steps (without modifying instance)
            R_perturbed = a_perturbed / self.sin60
            B_perturbed = (self.m * self.v) / (self.q * R_perturbed)
            T_perturbed = 2 * math.pi * R_perturbed / self.v
            t_perturbed = self.angle_fraction * T_perturbed

            # Check trends
            if not (R_perturbed > self.R):
                logs.append(f"FAIL: Monotonicity - R should increase when a increases. Original R: {self.R}, perturbed R: {R_perturbed}")
                success = False
            else:
                logs.append("PASS: Monotonicity - R increases with a")

            if not (B_perturbed < self.B):
                logs.append(f"FAIL: Monotonicity - B should decrease when a increases. Original B: {self.B}, perturbed B: {B_perturbed}")
                success = False
            else:
                logs.append("PASS: Monotonicity - B decreases with a (since R increases)")

            if not (T_perturbed > self.T):
                logs.append(f"FAIL: Monotonicity - T should increase when a increases. Original T: {self.T}, perturbed T: {T_perturbed}")
                success = False
            else:
                logs.append("PASS: Monotonicity - T increases with a (since R increases)")

            if not (t_perturbed > self.t):
                logs.append(f"FAIL: Monotonicity - t should increase when a increases. Original t: {self.t}, perturbed t: {t_perturbed}")
                success = False
            else:
                logs.append("PASS: Monotonicity - t increases with a (since T increases)")

        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {str(e)}")
            success = False

        return success, logs

    def run_all_tests(self):
        all_logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            all_logs.append(f"\n=== Running {test_name} Test ===")
            try:
                success, logs = test_func()
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"CRITICAL ERROR in {test_name}: {traceback.format_exc()}")
                overall_success = False

        all_logs.insert(0, f"Overall success: {overall_success}")
        return overall_success, all_logs