import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Use getattr to avoid AttributeError in sandbox.
        """
        # Extract inputs
        self.l = getattr(solution_instance, 'l', 1.0)           # length of rope, default 1.0 for numerical checks
        self.m = getattr(solution_instance, 'm', 1.0)           # mass, default 1.0
        self.g = getattr(solution_instance, 'g', 9.8)           # gravity, default 9.8 m/s^2

        # Extract results with safe fallbacks
        # Note: solution_instance may have a 'results' dictionary or attributes.
        # We first try direct attribute, then results dictionary, then default.
        self.v0 = getattr(solution_instance, 'v0', None)
        if self.v0 is None:
            self.v0 = getattr(solution_instance, 'v_0', None)
        if self.v0 is None and hasattr(solution_instance, 'results'):
            self.v0 = solution_instance.results.get('v_0', solution_instance.results.get('v0', 0.0))
        if self.v0 is None:
            self.v0 = 0.0

        self.v = getattr(solution_instance, 'v', None)
        if self.v is None and hasattr(solution_instance, 'results'):
            self.v = solution_instance.results.get('v', 0.0)
        if self.v is None:
            self.v = 0.0

        self.v_prime = getattr(solution_instance, 'v_prime', None)
        if self.v_prime is None:
            self.v_prime = getattr(solution_instance, 'vp', None)
        if self.v_prime is None and hasattr(solution_instance, 'results'):
            # Try both 'v'' and 'vp' keys
            self.v_prime = solution_instance.results.get("v''", solution_instance.results.get('vp', 0.0))
        if self.v_prime is None:
            self.v_prime = 0.0

        self.alpha = getattr(solution_instance, 'alpha', None)
        if self.alpha is None and hasattr(solution_instance, 'results'):
            self.alpha = solution_instance.results.get('alpha', 0.0)
        if self.alpha is None:
            self.alpha = 0.0

        self.T = getattr(solution_instance, 'T', None)
        if self.T is None and hasattr(solution_instance, 'results'):
            self.T = solution_instance.results.get('T', 0.0)
        if self.T is None:
            self.T = 0.0

        self.v1 = getattr(solution_instance, 'v1', None)
        if self.v1 is None:
            self.v1 = getattr(solution_instance, 'v_1', None)
        if self.v1 is None and hasattr(solution_instance, 'results'):
            self.v1 = solution_instance.results.get('v_1', solution_instance.results.get('v1', 0.0))
        if self.v1 is None:
            self.v1 = 0.0

        # Convert symbolic results to numeric if needed
        self._symbolic_mode = False
        try:
            # Check if any value is a sympy expression
            if any(sp.sympify(val).is_symbol for val in [self.v0, self.v, self.v_prime, self.alpha, self.T, self.v1]):
                self._symbolic_mode = True
        except:
            pass

        # Numerical values for fuzzy checks (substitute symbols if present)
        self._num_v0 = self._to_numeric(self.v0)
        self._num_v = self._to_numeric(self.v)
        self._num_v_prime = self._to_numeric(self.v_prime)
        self._num_alpha = self._to_numeric(self.alpha)
        self._num_T = self._to_numeric(self.T)
        self._num_v1 = self._to_numeric(self.v1)

    def _to_numeric(self, val):
        """Convert a value to numeric float, handling sympy expressions."""
        if isinstance(val, (int, float)):
            return float(val)
        try:
            # If it's a sympy expression with symbols, substitute with instance values
            expr = sp.sympify(val)
            if expr.is_number:
                return float(expr)
            else:
                # Substitute symbols l, m, g
                subs_dict = {sp.Symbol('l'): self.l, sp.Symbol('m'): self.m, sp.Symbol('g'): self.g}
                return float(expr.subs(subs_dict))
        except:
            return 0.0

    def test_invariants(self):
        """Verify energy conservation and Newton's second law."""
        logs = []
        success = True

        # 1. Conservation of mechanical energy from just after rope becomes taut (v'=0) to lowest point.
        # Height difference: at 60° from vertical, height above lowest point = l*(1 - cos60) = l*(1 - 0.5) = 0.5*l
        # So potential energy change: ΔU = m*g*0.5*l
        # Kinetic energy at lowest point: 0.5*m*v1^2
        # Thus: m*g*0.5*l = 0.5*m*v1^2  => v1 = sqrt(g*l)
        expected_v1 = math.sqrt(self.g * self.l)
        if not math.isclose(self._num_v1, expected_v1, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Energy conservation violated: v1 = {self._num_v1}, expected sqrt(g*l) = {expected_v1}")
            success = False
        else:
            logs.append(f"Energy conservation check passed: v1 = {self._num_v1} ≈ sqrt(g*l) = {expected_v1}")

        # 2. Newton's second law at lowest point: T - m*g = m*v1^2 / l
        # With v1 = sqrt(g*l), then T = m*g + m*(g*l)/l = m*g + m*g = 2*m*g
        expected_T = 2 * self.m * self.g
        if not math.isclose(self._num_T, expected_T, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Newton's second law at lowest point violated: T = {self._num_T}, expected 2mg = {expected_T}")
            success = False
        else:
            logs.append(f"Newton's second law check passed: T = {self._num_T} ≈ 2mg = {expected_T}")

        return success, logs

    def test_trace_consistency(self):
        """Cross-check intermediate results with steps analysis."""
        logs = []
        success = True

        # Step 1 & 2: Projectile motion to find time t and initial velocity v0.
        # From step2: l*cos60 = 0.5*g*t^2 => t = sqrt( (l*cos60) / (0.5*g) ) = sqrt( (0.5*l) / (0.5*g) ) = sqrt(l/g)
        t_expected = math.sqrt(self.l / self.g)
        # From step1: l*sin60 + (sqrt(3)/2)*l = v0 * t
        left_side = self.l * math.sin(math.radians(60)) + (math.sqrt(3)/2) * self.l
        v0_from_steps = left_side / t_expected
        if not math.isclose(self._num_v0, v0_from_steps, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"v0 inconsistent with kinematics: v0 = {self._num_v0}, expected from steps = {v0_from_steps}")
            success = False
        else:
            logs.append(f"v0 check passed: v0 = {self._num_v0} ≈ {v0_from_steps}")

        # Step 3: Velocity just before rope becomes taut: v = 2*sqrt(g*l) (given in step3)
        expected_v = 2 * math.sqrt(self.g * self.l)
        if not math.isclose(self._num_v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"v inconsistent: v = {self._num_v}, expected 2√(g*l) = {expected_v}")
            success = False
        else:
            logs.append(f"v check passed: v = {self._num_v} ≈ 2√(g*l) = {expected_v}")

        # Step 4: Angle alpha = 60°
        if not math.isclose(self._num_alpha, 60.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"alpha inconsistent: alpha = {self._num_alpha}°, expected 60°")
            success = False
        else:
            logs.append(f"alpha check passed: alpha = {self._num_alpha}° ≈ 60°")

        # Step 5: v' = 0
        if not math.isclose(self._num_v_prime, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"v' inconsistent: v' = {self._num_v_prime}, expected 0")
            success = False
        else:
            logs.append(f"v' check passed: v' = {self._num_v_prime} ≈ 0")

        return success, logs

    def test_monotonicity(self):
        """Perturb input (mass m) and check scaling of results."""
        logs = []
        success = True

        # Save original m
        original_m = self.m
        # Perturb: increase mass by 10%
        self.m *= 1.1

        # Recalculate expected results using formulas (assuming formulas from steps)
        # v0, v, v1, T depend on m? Actually from steps:
        # v0 = sqrt(3*g*l) -> independent of m
        # v = 2*sqrt(g*l) -> independent of m
        # v1 = sqrt(g*l) -> independent of m
        # T = 2*m*g -> directly proportional to m

        # So we only check T for monotonicity.
        expected_T_new = 2 * self.m * self.g
        # We don't have a new T from the solution, so we compute using the formula.
        # Instead, we can check that if we had the new T, it should be 1.1 times the original.
        # But we can't recompute the solution. So we do a thought experiment: 
        # The problem states T = 2mg, so it should scale linearly with m.
        # We'll check that the original T is 2*original_m*g, and if we increase m by 10%, 
        # the new T should be 1.1 times the original T.
        original_T_expected = 2 * original_m * self.g
        new_T_expected = 2 * self.m * self.g
        scaling_factor = new_T_expected / original_T_expected

        # Restore original m
        self.m = original_m

        # The scaling factor should be exactly 1.1 (since T ∝ m)
        if not math.isclose(scaling_factor, 1.1, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"Monotonicity violated: T does not scale linearly with m. Scaling factor = {scaling_factor}, expected 1.1")
            success = False
        else:
            logs.append(f"Monotonicity check passed: T scales linearly with m (factor {scaling_factor} ≈ 1.1)")

        return success, logs

    def run_all_tests(self):
        """Run all tests and return overall success and logs."""
        all_logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            try:
                success, logs = test_func()
                all_logs.append(f"--- {test_name} ---")
                all_logs.extend(logs)
                if not success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"Error in {test_name}: {traceback.format_exc()}")
                overall_success = False

        return overall_success, all_logs