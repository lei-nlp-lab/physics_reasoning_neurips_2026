import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        """
        self.solution = solution_instance
        
        # Input parameters: directly get from solution_instance attributes
        # Use defaults only if not present
        self.M = getattr(solution_instance, 'M', 1.0)
        self.m = getattr(solution_instance, 'm', 1.0)
        self.a = getattr(solution_instance, 'a', 1.0)
        self.b = getattr(solution_instance, 'b', 1.0)
        self.g = getattr(solution_instance, 'g', 9.8)
        
        # Results (with safe defaults)
        self.results = getattr(solution_instance, 'results', {})
        self.v2 = self.results.get('v_2', 0.0)   # velocity of groove at lowest point
        self.x2 = self.results.get('x_2', 0.0)   # horizontal displacement of groove at lowest point
        self.v3 = self.results.get('v_3', 0.0)   # velocity of ball at height b/2
        
        # Optional: trajectory equation (if stored as string or numeric)
        self.traj_eq = self.results.get('trajectory_eq', None)
        
        # Convert to floats if they are sympy expressions or strings
        for attr in ['M', 'm', 'a', 'b', 'g', 'v2', 'x2', 'v3']:
            val = getattr(self, attr)
            if isinstance(val, (int, float)):
                continue
            try:
                # Try to convert to float if possible
                setattr(self, attr, float(val))
            except (TypeError, ValueError):
                # If it's a string that can't be converted, keep as is
                pass
        
        # Tolerance for numerical comparisons
        self.rel_tol = 1e-2
        self.abs_tol = 1e-2
        
        # Logs
        self.logs = []

    def log(self, message):
        """Helper to append log messages."""
        self.logs.append(message)

    def is_close(self, a, b):
        """Fuzzy numerical comparison."""
        return math.isclose(a, b, rel_tol=self.rel_tol, abs_tol=self.abs_tol)

    def test_invariants(self):
        """
        Check conservation of horizontal momentum and mechanical energy.
        """
        try:
            # 1. At the lowest point: momentum conservation
            # From step_1: 0 = m*v1 - M*v2, so v1 = (M/m)*v2
            v1 = (self.M / self.m) * self.v2
            # Total horizontal momentum should be zero
            p_total = self.m * v1 + self.M * (-self.v2)   # v2 is in opposite direction
            if not self.is_close(p_total, 0.0):
                self.log(f"FAIL: Horizontal momentum not conserved at lowest point. p_total = {p_total}")
                return False
            self.log("PASS: Horizontal momentum conserved at lowest point.")

            # 2. Energy conservation at lowest point
            # Initial potential energy: m*g*b (descended height b)
            # Final kinetic energy: (1/2)*m*v1^2 + (1/2)*M*v2^2
            delta_E = self.m * self.g * self.b - (0.5 * self.m * v1**2 + 0.5 * self.M * self.v2**2)
            if not self.is_close(delta_E, 0.0):
                self.log(f"FAIL: Energy not conserved at lowest point. delta_E = {delta_E}")
                return False
            self.log("PASS: Energy conserved at lowest point.")

            # 3. Displacement relations from momentum (step_4 and step_5)
            # x1 = displacement of ball = a - x2 (from step_5)
            x1 = self.a - self.x2
            # Momentum conservation in displacements: m*x1 = M*x2
            if not self.is_close(self.m * x1, self.M * self.x2):
                self.log(f"FAIL: Displacement momentum relation not satisfied. m*x1 = {self.m*x1}, M*x2 = {self.M*self.x2}")
                return False
            self.log("PASS: Displacement momentum relation holds.")

            # 4. Check trajectory equation (if provided)
            if self.traj_eq is not None:
                # The trajectory equation from step_8:
                # [x*(M+m) - m*a]^2 / (M^2 * a^2) + y^2 / b^2 = 1
                # Check that the initial point (a,0) satisfies it.
                x, y = self.a, 0.0
                lhs = ( (x*(self.M+self.m) - self.m*self.a)**2 ) / (self.M**2 * self.a**2) + (y**2) / (self.b**2)
                if not self.is_close(lhs, 1.0):
                    self.log(f"FAIL: Initial point (a,0) does not satisfy trajectory equation. lhs = {lhs}")
                    return False
                # Check the lowest point (x2, -b)
                x, y = self.x2, -self.b
                lhs = ( (x*(self.M+self.m) - self.m*self.a)**2 ) / (self.M**2 * self.a**2) + (y**2) / (self.b**2)
                if not self.is_close(lhs, 1.0):
                    self.log(f"FAIL: Lowest point (x2,-b) does not satisfy trajectory equation. lhs = {lhs}")
                    return False
                self.log("PASS: Trajectory equation holds for initial and lowest points.")

            # 5. Check the condition for v3 (if M/m = b/(a-b))
            condition = self.M / self.m - self.b / (self.a - self.b)
            if not self.is_close(condition, 0.0):
                self.log(f"WARNING: Condition M/m = b/(a-b) not satisfied. Difference = {condition}. v3 expression may not be valid.")
            else:
                # At height b/2: energy and momentum
                # From step_10: angle = 60°, so cos60 = 0.5
                # Momentum: 0 = m*v3*cos60 - M*v4  => v4 = (m/M)*v3*0.5
                v4 = (self.m / self.M) * self.v3 * 0.5
                # Energy: m*g*(b/2) = (1/2)*m*v3^2 + (1/2)*M*v4^2
                delta_E_half = self.m * self.g * (self.b/2) - (0.5 * self.m * self.v3**2 + 0.5 * self.M * v4**2)
                if not self.is_close(delta_E_half, 0.0):
                    self.log(f"FAIL: Energy not conserved at height b/2. delta_E = {delta_E_half}")
                    return False
                self.log("PASS: Energy conserved at height b/2 (under condition M/m = b/(a-b)).")

            return True
        except Exception as e:
            self.log(f"ERROR in test_invariants: {e}\n{traceback.format_exc()}")
            return False

    def test_trace_consistency(self):
        """
        Check that intermediate steps lead to the given results.
        """
        try:
            # 1. Check v2 from step_1 and step_2
            # From momentum: v1 = (M/m)*v2
            # From energy: m*g*b = 0.5*m*v1^2 + 0.5*M*v2^2
            # Substitute v1: m*g*b = 0.5*m*(M^2/m^2)*v2^2 + 0.5*M*v2^2
            # => v2^2 = (2*m*g*b) / (M^2/m + M) = (2*m^2*g*b) / (M^2 + M*m)
            expected_v2_sq = (2 * self.m**2 * self.g * self.b) / (self.M**2 + self.M * self.m)
            if not self.is_close(self.v2**2, expected_v2_sq):
                self.log(f"FAIL: v2 does not match derived expression. Expected v2^2 = {expected_v2_sq}, got {self.v2**2}")
                return False
            self.log("PASS: v2 consistent with momentum and energy equations.")

            # 2. Check x2 from step_4 and step_5
            # From momentum: m*x1 = M*x2, and x1 + x2 = a => x1 = a - x2
            # => m*(a - x2) = M*x2 => x2 = m*a/(M+m)
            expected_x2 = self.m * self.a / (self.M + self.m)
            if not self.is_close(self.x2, expected_x2):
                self.log(f"FAIL: x2 does not match derived expression. Expected {expected_x2}, got {self.x2}")
                return False
            self.log("PASS: x2 consistent with displacement relations.")

            # 3. Check v3 (if condition holds)
            condition = self.M / self.m - self.b / (self.a - self.b)
            if self.is_close(condition, 0.0):
                # From step_12: v3 = 2b * sqrt(g/(a+3b))
                expected_v3 = 2 * self.b * math.sqrt(self.g / (self.a + 3 * self.b))
                if not self.is_close(self.v3, expected_v3):
                    self.log(f"FAIL: v3 does not match given expression. Expected {expected_v3}, got {self.v3}")
                    return False
                self.log("PASS: v3 matches the expression under condition M/m = b/(a-b).")
            else:
                self.log("SKIP: v3 trace check because condition M/m = b/(a-b) not met.")

            return True
        except Exception as e:
            self.log(f"ERROR in test_trace_consistency: {e}\n{traceback.format_exc()}")
            return False

    def test_monotonicity(self):
        """
        Perturb mass m and check that v2 and x2 change in the expected direction.
        """
        try:
            # Store original values
            m_orig = self.m
            M_orig = self.M
            a_orig = self.a
            b_orig = self.b
            g_orig = self.g
            v2_orig = self.v2
            x2_orig = self.x2
            
            perturb_factor = 1.01  # 1% increase
            
            # Perturb m
            self.m = m_orig * perturb_factor
            
            # Recompute v2 and x2 using the formulas (not from the solution)
            v2_new = math.sqrt(2 * self.m**2 * self.g * self.b / (self.M**2 + self.M * self.m))
            x2_new = self.m * self.a / (self.M + self.m)
            
            # Restore original values
            self.m = m_orig
            self.M = M_orig
            self.a = a_orig
            self.b = b_orig
            self.g = g_orig
            
            # Check monotonicity: both should increase with m
            if v2_new <= v2_orig:
                self.log(f"FAIL: v2 does not increase when m increases. Original {v2_orig}, new {v2_new}")
                return False
            if x2_new <= x2_orig:
                self.log(f"FAIL: x2 does not increase when m increases. Original {x2_orig}, new {x2_new}")
                return False
            
            self.log("PASS: v2 and x2 increase with m (monotonicity).")
            return True
        except Exception as e:
            self.log(f"ERROR in test_monotonicity: {e}\n{traceback.format_exc()}")
            return False

    def run_all_tests(self):
        """
        Run all tests and return success flag and logs.
        """
        self.logs = []
        success = True
        
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]
        
        for name, test_func in tests:
            self.log(f"Running {name}...")
            if not test_func():
                success = False
                self.log(f"{name} failed.")
            else:
                self.log(f"{name} passed.")
        
        return success, self.logs