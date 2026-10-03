import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the PhysicsChecker with safe attribute access.
        """
        self.logs = []
        self.success = True

        # Extract input parameters safely
        self.m = getattr(solution_instance, 'm', getattr(solution_instance, 'results', {}).get('m', 1.0))
        self.g = getattr(solution_instance, 'g', getattr(solution_instance, 'results', {}).get('g', 9.8))
        self.R = getattr(solution_instance, 'R', getattr(solution_instance, 'results', {}).get('R', 1.0))
        self.qE = math.sqrt(2) * self.m * self.g  # Given: electric force magnitude = √2 * m*g

        # Extract results safely with fallback
        self.results = getattr(solution_instance, 'results', {})
        self.E_p = getattr(solution_instance, 'E_p', self.results.get('E_p', 0.0))
        self.v_O = getattr(solution_instance, 'v_O', self.results.get('v_O', 0.0))
        self.a_x = getattr(solution_instance, 'a_x', self.results.get('a_x', 0.0))
        self.a_y = getattr(solution_instance, 'a_y', self.results.get('a_y', 0.0))
        self.x = getattr(solution_instance, 'x', self.results.get('x', 0.0))
        self.y = getattr(solution_instance, 'y', self.results.get('y', 0.0))

        # For symbolic computations
        self.m_sym, self.g_sym, self.R_sym, self.qE_sym = sp.symbols('m g R qE', positive=True)
        self.substitutions = {self.m_sym: self.m, self.g_sym: self.g, self.R_sym: self.R, 
                              self.qE_sym: self.qE}

    def _log(self, msg, is_error=False):
        """Helper to log messages and update success flag."""
        self.logs.append(msg)
        if is_error:
            self.success = False

    def test_invariants(self):
        """Check energy conservation and Newton's laws based on KB."""
        try:
            # 1. Energy conservation from A to B (Step 1)
            v_B_expected = math.sqrt(self.g * self.R)
            E_p_expected = 0.5 * self.m * self.g * self.R
            if not math.isclose(self.E_p, E_p_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Energy invariant failed: E_p = {self.E_p}, expected ≈ {E_p_expected}", True)

            # 2. Work-energy from B to O (Step 2-3)
            v_B = v_B_expected
            ke_change = 0.5 * self.m * self.v_O**2 - 0.5 * self.m * v_B**2
            work_total = self.m * self.g * self.R  # mgR
            if not math.isclose(ke_change, work_total, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Work-energy failed: ΔKE = {ke_change}, expected ≈ {work_total}", True)

            # 3. Newton's second law at point O (Step 4-6)
            # a_x = (qE cos45°)/m = (√2 mg * √2/2)/m = g
            a_x_expected = self.g
            if not math.isclose(self.a_x, a_x_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Newton's law (x) failed: a_x = {self.a_x}, expected ≈ {a_x_expected}", True)

            # a_y = (qE sin45° - mg)/m = (√2 mg * √2/2 - mg)/m = (mg - mg)/m = 0
            a_y_expected = 0.0
            if not math.isclose(self.a_y, a_y_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Newton's law (y) failed: a_y = {self.a_y}, expected ≈ {a_y_expected}", True)

            # Symbolic fallback check
            a_x_expr = (self.qE_sym * sp.sqrt(2)/2) / self.m_sym
            a_y_expr = (self.qE_sym * sp.sqrt(2)/2 - self.m_sym * self.g_sym) / self.m_sym
            a_x_val = float(a_x_expr.subs(self.substitutions))
            a_y_val = float(a_y_expr.subs(self.substitutions))
            if not (math.isclose(a_x_val, self.a_x, rel_tol=1e-2) and 
                    math.isclose(a_y_val, self.a_y, rel_tol=1e-2)):
                self._log("Symbolic Newton check discrepancy", False)

        except Exception as e:
            self._log(f"Error in test_invariants: {traceback.format_exc()}", True)

    def test_trace_consistency(self):
        """Cross-check intermediate results with final trajectory."""
        try:
            # Check v_O from energy (Step 3)
            v_O_expected = math.sqrt(3 * self.g * self.R)
            if not math.isclose(self.v_O, v_O_expected, rel_tol=1e-2, abs_tol=1e-2):
                self._log(f"Trace consistency failed: v_O = {self.v_O}, expected ≈ {v_O_expected}", True)

            # Check trajectory equation y^2 = 6R x (Step 9)
            if self.x != 0 and self.y != 0:  # Avoid division by zero
                lhs = self.y**2
                rhs = 6 * self.R * self.x
                if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Trajectory consistency failed: y^2 = {lhs}, 6R x = {rhs}", True)

            # Check kinematic equations (Step 7-8): x = 0.5 * a_x * t^2, y = v_O * t
            # Eliminate t: t = y / v_O, substitute into x equation
            if self.v_O != 0:
                t = self.y / self.v_O
                x_from_kinematic = 0.5 * self.a_x * t**2
                if not math.isclose(self.x, x_from_kinematic, rel_tol=1e-2, abs_tol=1e-2):
                    self._log(f"Kinematic consistency failed: x = {self.x}, from kinematics = {x_from_kinematic}", True)

        except Exception as e:
            self._log(f"Error in test_trace_consistency: {traceback.format_exc()}", True)

    def test_monotonicity(self):
        """Perturb R and verify trend in trajectory, then restore."""
        original_R = self.R
        try:
            # Perturb R by +1%
            self.R *= 1.01
            self.substitutions[self.R_sym] = self.R

            # Recompute expected v_O and trajectory coefficient
            v_O_new = math.sqrt(3 * self.g * self.R)
            # For fixed x, y should scale with sqrt(R) from y^2 = 6R x
            # Test with a sample x value (e.g., x = 1.0)
            test_x = 1.0
            y_original = math.sqrt(6 * original_R * test_x)
            y_new = math.sqrt(6 * self.R * test_x)

            # y should increase when R increases
            if not (y_new > y_original):
                self._log(f"Monotonicity failed: y did not increase with R", True)

            # Also v_O should increase
            v_O_original = math.sqrt(3 * self.g * original_R)
            if not (v_O_new > v_O_original):
                self._log(f"Monotonicity failed: v_O did not increase with R", True)

        except Exception as e:
            self._log(f"Error in test_monotonicity: {traceback.format_exc()}", True)
        finally:
            # Restore original R
            self.R = original_R
            self.substitutions[self.R_sym] = original_R

    def run_all_tests(self):
        """Execute all tests and return success status and logs."""
        self.logs = []
        self.success = True

        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()

        return self.success, self.logs