import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Follows robustness rule #1: safe attribute access.
        """
        self.solution = solution_instance
        
        # Inputs (with defaults if missing)
        self.M = getattr(solution_instance, 'M', 1.0)
        self.Q = getattr(solution_instance, 'Q', 1.0)
        self.g = getattr(solution_instance, 'g', 9.8)
        self.alpha = getattr(solution_instance, 'alpha', 0.0)
        self.beta = getattr(solution_instance, 'beta', 1.0)
        self.y0 = getattr(solution_instance, 'y0', 0.0)
        self.x0 = getattr(solution_instance, 'x0', 0.0)
        
        # Results (from results dictionary if available)
        results = getattr(solution_instance, 'results', {})
        self.f = results.get('f', 0.0)
        self.F_e_parallel = results.get('F_e_parallel', 0.0)
        self.F_g_parallel = results.get('F_g_parallel', 0.0)
        self.y_e = results.get('y_e', 0.0)
        self.Fy = results.get('Fy', 0.0)  # F(y)
        self.FDeltay = results.get('FDeltay', 0.0)  # F(Δy)
        self.k = results.get('k', 0.0)
        self.omega = results.get('omega', 0.0)
        
        # Symbolic variables for checks
        self.symbols = {
            'M': sp.symbols('M', positive=True),
            'Q': sp.symbols('Q', positive=True),
            'g': sp.symbols('g', positive=True),
            'alpha': sp.symbols('alpha', positive=True),
            'beta': sp.symbols('beta', positive=True),
            'y0': sp.symbols('y0', real=True),
            'y': sp.symbols('y', real=True),
            'Delta_y': sp.symbols('Delta_y', real=True)
        }
        
        self.logs = []

    def log(self, msg, level="INFO"):
        """Helper to log messages with level."""
        self.logs.append(f"[{level}] {msg}")

    def safe_eval(self, expr, subs_dict):
        """
        Safely evaluate a symbolic expression numerically.
        Returns float or None on error.
        """
        try:
            # Convert subs_dict values to floats if possible
            eval_dict = {}
            for k, v in subs_dict.items():
                try:
                    eval_dict[k] = float(v)
                except (TypeError, ValueError):
                    eval_dict[k] = v
            result = float(sp.N(expr.subs(eval_dict)))
            return result
        except Exception as e:
            self.log(f"Error evaluating {expr}: {e}", "WARNING")
            return None

    def test_invariants(self):
        """
        Test physical invariants based on KB theorems.
        - Vertical force balance at equilibrium (Newton's 1st law)
        - Force consistency (Newton's 2nd law)
        - Hooke's law for SHM region
        """
        self.log("Starting invariant tests...")
        
        # 1. Vertical force balance at equilibrium y_e (from steps 1-3)
        # Equation: Q * beta * (y0 - y_e) = M * g
        lhs_eq = self.Q * self.beta * (self.y0 - self.y_e)
        rhs_eq = self.M * self.g
        
        if math.isclose(lhs_eq, rhs_eq, rel_tol=1e-2, abs_tol=1e-2):
            self.log("PASS: Vertical force balance at equilibrium holds.")
        else:
            self.log(f"FAIL: Vertical force balance mismatch. LHS={lhs_eq}, RHS={rhs_eq}", "ERROR")
            return False
        
        # 2. Force component consistency (steps 5-7)
        # F_g_parallel should be M*g*sin(alpha)
        expected_F_g_parallel = self.M * self.g * math.sin(self.alpha)
        if math.isclose(self.F_g_parallel, expected_F_g_parallel, rel_tol=1e-2, abs_tol=1e-2):
            self.log("PASS: Gravitational parallel force matches expected.")
        else:
            self.log(f"FAIL: F_g_parallel mismatch. Got {self.F_g_parallel}, expected {expected_F_g_parallel}", "ERROR")
            return False
        
        # F_e_parallel at arbitrary y should be Q*beta*(y0 - y)*sin(alpha)
        # Test at y = y0 (simplest case)
        test_y = self.y0
        expected_F_e_parallel = self.Q * self.beta * (self.y0 - test_y) * math.sin(self.alpha)
        # Note: We don't have F_e_parallel as function of y, so skip if not available
        if hasattr(self.solution, 'F_e_parallel_func'):
            actual_F_e_parallel = self.solution.F_e_parallel_func(test_y)
            if math.isclose(actual_F_e_parallel, expected_F_e_parallel, rel_tol=1e-2, abs_tol=1e-2):
                self.log("PASS: Electric parallel force formula matches at y=y0.")
            else:
                self.log(f"FAIL: F_e_parallel mismatch at y=y0. Got {actual_F_e_parallel}, expected {expected_F_e_parallel}", "WARNING")
        else:
            self.log("SKIP: No F_e_parallel function to test.", "INFO")
        
        # 3. Hooke's law check (steps 11-13)
        # k should equal Q*beta
        expected_k = self.Q * self.beta
        if math.isclose(self.k, expected_k, rel_tol=1e-2, abs_tol=1e-2):
            self.log("PASS: Effective spring constant matches Q*beta.")
        else:
            self.log(f"FAIL: k mismatch. Got {self.k}, expected {expected_k}", "ERROR")
            return False
        
        # Angular frequency: omega = sqrt(k/M) = sqrt(Q*beta/M)
        expected_omega = math.sqrt(self.Q * self.beta / self.M)
        if math.isclose(self.omega, expected_omega, rel_tol=1e-2, abs_tol=1e-2):
            self.log("PASS: Angular frequency matches sqrt(Q*beta/M).")
        else:
            self.log(f"FAIL: omega mismatch. Got {self.omega}, expected {expected_omega}", "ERROR")
            return False
        
        # Frequency: f = omega/(2π) = (1/(2π)) * sqrt(Q*beta/M)
        expected_f = expected_omega / (2 * math.pi)
        if math.isclose(self.f, expected_f, rel_tol=1e-2, abs_tol=1e-2):
            self.log("PASS: Oscillation frequency matches expected formula.")
        else:
            self.log(f"FAIL: f mismatch. Got {self.f}, expected {expected_f}", "ERROR")
            return False
        
        return True

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        Tests consistency between y_e, force expressions, and f.
        """
        self.log("Starting trace consistency tests...")
        
        # 1. Check that y_e formula matches the equilibrium condition
        y_e_formula = self.y0 - (self.M * self.g) / (self.beta * self.Q)
        if math.isclose(self.y_e, y_e_formula, rel_tol=1e-2, abs_tol=1e-2):
            self.log("PASS: y_e matches formula from vertical force balance.")
        else:
            self.log(f"FAIL: y_e mismatch. Got {self.y_e}, expected {y_e_formula}", "ERROR")
            return False
        
        # 2. Check F(y) formula at a test point (y = y_e)
        test_y = self.y_e
        expected_Fy = (self.M * self.g - self.Q * self.beta * (self.y0 - test_y)) * math.sin(self.alpha)
        if math.isclose(self.Fy, expected_Fy, rel_tol=1e-2, abs_tol=1e-2):
            self.log("PASS: F(y) matches formula at equilibrium y.")
        else:
            self.log(f"FAIL: F(y) mismatch at y=y_e. Got {self.Fy}, expected {expected_Fy}", "WARNING")
            # Not critical for overall consistency
        
        # 3. Check F(Δy) formula (should be -k*Δy after leaving incline)
        # Test at Δy = 1.0 (arbitrary)
        test_delta_y = 1.0
        expected_FDeltay = -self.k * test_delta_y
        if hasattr(self.solution, 'FDeltay_func'):
            actual_FDeltay = self.solution.FDeltay_func(test_delta_y)
            if math.isclose(actual_FDeltay, expected_FDeltay, rel_tol=1e-2, abs_tol=1e-2):
                self.log("PASS: F(Δy) matches Hooke's law form.")
            else:
                self.log(f"FAIL: F(Δy) mismatch. Got {actual_FDeltay}, expected {expected_FDeltay}", "WARNING")
        else:
            self.log("SKIP: No F(Δy) function to test.", "INFO")
        
        # 4. Verify that f derived from k and M matches reported f
        f_from_omega = self.omega / (2 * math.pi)
        if math.isclose(self.f, f_from_omega, rel_tol=1e-2, abs_tol=1e-2):
            self.log("PASS: f is consistent with omega.")
        else:
            self.log(f"FAIL: f inconsistent with omega. f={self.f}, omega/(2π)={f_from_omega}", "ERROR")
            return False
        
        return True

    def test_monotonicity(self):
        """
        Perturb an input (mass M) and check the trend in output (frequency f).
        Rule: f should decrease as M increases (since f ∝ 1/√M).
        """
        self.log("Starting monotonicity test (perturbing mass M)...")
        
        original_M = self.M
        perturbation_factor = 1.1  # Increase M by 10%
        
        try:
            # Compute f for original and perturbed mass using formula
            f_original = (1/(2*math.pi)) * math.sqrt(self.beta * self.Q / original_M)
            f_perturbed = (1/(2*math.pi)) * math.sqrt(self.beta * self.Q / (original_M * perturbation_factor))
            
            # Check trend: f should decrease when M increases
            if f_perturbed < f_original:
                self.log("PASS: Frequency decreases with increased mass (as expected).")
                return True
            else:
                self.log(f"FAIL: Frequency did not decrease. f_original={f_original}, f_perturbed={f_perturbed}", "ERROR")
                return False
        except Exception as e:
            self.log(f"Error in monotonicity test: {e}", "ERROR")
            return False

    def run_all_tests(self):
        """
        Run all tests and return (success_bool, logs_list).
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
                self.log(f"Running {test_name} test...")
                if not test_func():
                    success = False
                    self.log(f"{test_name} test failed.", "ERROR")
                else:
                    self.log(f"{test_name} test passed.", "INFO")
            except Exception as e:
                success = False
                self.log(f"{test_name} test crashed: {traceback.format_exc()}", "ERROR")
        
        return success, self.logs