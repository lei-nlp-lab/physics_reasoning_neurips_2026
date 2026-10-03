import math
import sympy as sp
from sympy.core.relational import Equality

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all required inputs and results without direct attribute access.
        """
        # Extract inputs with safe defaults
        self.m = getattr(solution_instance, 'm', 0.60)
        self.h1 = getattr(solution_instance, 'h1', 1.8)
        self.h2 = getattr(solution_instance, 'h2', 1.2)
        self.h3 = getattr(solution_instance, 'h3', 1.5)
        self.h4 = 1.5  # Given in problem statement
        self.t = getattr(solution_instance, 't', 0.20)
        self.g = getattr(solution_instance, 'g', 10.0)

        # Extract results dictionary safely
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}

        # Map expected keys with fallbacks
        self.E1 = results.get('E_1', results.get('E_{1}', 0.0))
        self.E2 = results.get('E_2', results.get('E_{2}', 0.0))
        self.E3 = results.get('E_3', results.get('E_{3}', 0.0))
        self.E4 = results.get('E_4', results.get('E_{4}', 0.0))
        self.W = results.get('W', 0.0)
        self.F = results.get('F', 0.0)
        self.x = results.get('x', 0.0)
        self.F_plus_mg = results.get('F+mg', 0.0)
        self.ratio = results.get('fracE_2E_1=fracE_4E_3', 0.0)

        # Store logs
        self.logs = []

    def _log(self, message, success=None):
        """Helper to log messages with optional success flag."""
        if success is not None:
            status = "PASS" if success else "FAIL"
            self.logs.append(f"{status}: {message}")
        else:
            self.logs.append(f"INFO: {message}")

    def _fuzzy_eq(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy equality check with configurable tolerance."""
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        # Handle symbolic comparisons
        try:
            diff = float(abs(a - b))
            return math.isclose(diff, 0.0, rel_tol=rel_tol, abs_tol=abs_tol)
        except:
            return False

    def test_invariants(self):
        """
        Test fundamental physics invariants:
        1. Energy conservation during free fall and rebound
        2. Newton's second law for force calculation
        3. Ratio consistency
        """
        success = True
        g, m = self.g, self.m

        # 1. Energy definitions from work-energy theorem (steps 1-4)
        # E1 = m*g*h1
        expected_E1 = m * g * self.h1
        if not self._fuzzy_eq(self.E1, expected_E1):
            self._log(f"E1 mismatch: got {self.E1}, expected {expected_E1}", False)
            success = False
        else:
            self._log(f"E1 correct: {self.E1} J", True)

        # E2 = m*g*h2
        expected_E2 = m * g * self.h2
        if not self._fuzzy_eq(self.E2, expected_E2):
            self._log(f"E2 mismatch: got {self.E2}, expected {expected_E2}", False)
            success = False
        else:
            self._log(f"E2 correct: {self.E2} J", True)

        # E4 = m*g*h4
        expected_E4 = m * g * self.h4
        if not self._fuzzy_eq(self.E4, expected_E4):
            self._log(f"E4 mismatch: got {self.E4}, expected {expected_E4}", False)
            success = False
        else:
            self._log(f"E4 correct: {self.E4} J", True)

        # 2. Ratio consistency (step 5): E2/E1 = E4/E3
        # This should hold exactly if calculations are correct
        if self.E1 != 0 and self.E3 != 0:
            ratio1 = self.E2 / self.E1
            ratio2 = self.E4 / self.E3
            if not self._fuzzy_eq(ratio1, ratio2):
                self._log(f"Ratio mismatch: E2/E1={ratio1}, E4/E3={ratio2}", False)
                success = False
            else:
                self._log(f"Ratio consistent: {ratio1:.3f} = {ratio2:.3f}", True)

        # 3. Newton's second law: F_net = m*a (step 7)
        # From solution: F + mg = m*a, where a is acceleration during force application
        # We can check this indirectly via kinematics and work
        if self.F > 0 and self.x > 0:
            # From kinematics: x = 0.5*a*t^2 => a = 2*x/(t^2)
            a_calc = 2 * self.x / (self.t ** 2) if self.t != 0 else 0
            F_net_calc = m * a_calc
            # F_net should equal F + m*g (both forces downward)
            if not self._fuzzy_eq(self.F_plus_mg, F_net_calc):
                self._log(f"Newton's law mismatch: F+mg={self.F_plus_mg}, m*a={F_net_calc}", False)
                success = False
            else:
                self._log(f"Newton's law consistent: F+mg = m*a = {F_net_calc} N", True)

        return success

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers:
        1. W from energy method vs. from work definition (W = F*x)
        2. F from simultaneous equations
        """
        success = True

        # 1. Verify W calculation from energy steps (step 6)
        # From steps: E3 = W + m*g*h3, and ratio gives E3 = E4 * E1 / E2
        if self.E1 != 0 and self.E2 != 0:
            E3_calc = self.E4 * self.E1 / self.E2
            W_calc = E3_calc - self.m * self.g * self.h3
            if not self._fuzzy_eq(self.W, W_calc):
                self._log(f"W mismatch: got {self.W}, expected {W_calc}", False)
                success = False
            else:
                self._log(f"W consistent: {self.W} J", True)

        # 2. Verify F calculation (steps 10-11)
        # From work definition: W = F*x (step 9)
        # From kinematics: x = 0.5*a*t^2, and a = (F + m*g)/m
        # Combine: x = 0.5 * ((F + m*g)/m) * t^2
        # Then W = F * [0.5 * ((F + m*g)/m) * t^2]
        # This is a quadratic in F: W = (F * t^2 / (2*m)) * (F + m*g)
        # Solve symbolically and check against given F
        F, t, m, g, W = sp.symbols('F t m g W')
        eq = sp.Eq(W, F * t**2 / (2*m) * (F + m*g))
        # Substitute known values
        eq_sub = eq.subs({t: self.t, m: self.m, g: self.g, W: self.W})
        # Solve for F
        solutions = sp.solve(eq_sub, F)
        valid_solutions = [float(sol) for sol in solutions if sol.is_real and sol > 0]
        if valid_solutions:
            F_calc = valid_solutions[0]
            if not self._fuzzy_eq(self.F, F_calc):
                self._log(f"F mismatch: got {self.F}, expected {F_calc}", False)
                success = False
            else:
                self._log(f"F consistent: {self.F} N", True)
        else:
            self._log("Could not solve for F from work-kinematics equations", False)
            success = False

        # 3. Check that x is positive and reasonable
        if self.x <= 0:
            self._log(f"Displacement x must be positive, got {self.x}", False)
            success = False
        elif self.x > 1.0:  # Rough sanity: 0.2s force application shouldn't move ball >1m
            self._log(f"Displacement x={self.x} m seems too large", False)
            success = False
        else:
            self._log(f"Displacement x={self.x} m is reasonable", True)

        return success

    def test_monotonicity(self):
        """
        Perturb an input parameter and verify expected trend in output.
        We'll perturb mass m and check that both W and F scale appropriately.
        """
        success = True
        original_m = self.m

        # Perturbation factor
        delta = 0.1  # 10% increase
        m_perturbed = original_m * (1 + delta)

        # Recalculate expected changes using physics formulas
        # For W: from energy method, W = E4*E1/E2 - m*g*h3
        # E1, E2, E4 are all proportional to m, so ratio E4*E1/E2 is proportional to m
        # Thus W should increase with m if (E4*E1/E2 - m*g*h3) increases with m
        # Compute derivative dW/dm = (E4*E1/E2)/m - g*h3
        # With given numbers: E1=10.8, E2=7.2, E4=9.0 (for m=0.6)
        # So dW/dm = (9*10.8/7.2)/0.6 - 10*1.5 = (13.5)/0.6 - 15 = 22.5 - 15 = 7.5 > 0
        # Therefore W should increase with m.

        # For F: from W = (F * t^2 / (2*m)) * (F + m*g)
        # This is implicit. We can check numerically by solving the quadratic for both masses.
        g, h1, h2, h3, h4, t = self.g, self.h1, self.h2, self.h3, self.h4, self.t

        # Helper function to compute W and F for a given mass
        def compute_W_F(mass):
            E1 = mass * g * h1
            E2 = mass * g * h2
            E4 = mass * g * h4
            E3 = E4 * E1 / E2
            W_val = E3 - mass * g * h3
            # Solve quadratic for F: W = (F * t^2 / (2*mass)) * (F + mass*g)
            F_sym = sp.symbols('F_sym')
            eq = sp.Eq(W_val, F_sym * t**2 / (2*mass) * (F_sym + mass * g))
            solutions = sp.solve(eq, F_sym)
            valid = [float(s) for s in solutions if s.is_real and s > 0]
            return W_val, valid[0] if valid else None

        # Compute for original and perturbed mass
        W_orig, F_orig = compute_W_F(original_m)
        W_pert, F_pert = compute_W_F(m_perturbed)

        # Check monotonicity: both should increase with mass (positive derivatives)
        if not (W_pert > W_orig):
            self._log(f"W does not increase with mass: {W_orig} -> {W_pert}", False)
            success = False
        else:
            self._log(f"W increases with mass: {W_orig:.2f} -> {W_pert:.2f} J", True)

        if F_pert is None or not (F_pert > F_orig):
            self._log(f"F does not increase with mass: {F_orig} -> {F_pert}", False)
            success = False
        else:
            self._log(f"F increases with mass: {F_orig:.2f} -> {F_pert:.2f} N", True)

        return success

    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        self.logs.clear()
        self.logs.append("=== PhysicsChecker Started ===")

        test_results = []
        # Run each test and collect results
        test_results.append(("Invariants", self.test_invariants()))
        test_results.append(("Trace Consistency", self.test_trace_consistency()))
        test_results.append(("Monotonicity", self.test_monotonicity()))

        overall_success = all(result for _, result in test_results)

        # Summarize
        self.logs.append("\n=== Summary ===")
        for test_name, result in test_results:
            status = "PASS" if result else "FAIL"
            self.logs.append(f"{test_name}: {status}")

        self.logs.append(f"\nOverall: {'PASS' if overall_success else 'FAIL'}")
        return overall_success, self.logs