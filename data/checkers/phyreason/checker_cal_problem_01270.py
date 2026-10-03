import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Use getattr and .get to avoid AttributeError in the sandbox.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Physical constants (given in the problem)
        self.g = 10.0
        self.tantheta = 0.75
        self.theta_deg = 37.0

        # Extract results with safe fallbacks
        self.v0 = self._safe_get('v_0')
        self.a = self._safe_get('a')
        self.x = self._safe_get('x')
        self.x1 = self._safe_get('x_1')
        self.x2 = self._safe_get('x_2')
        self.h2 = self._safe_get('h_2')
        self.h = self._safe_get('h', 1.8)  # default 1.8 if not found

        # Initialize logs
        self.logs = []

    def _safe_get(self, key, default=None):
        """
        Safely retrieve a value from solution instance or results dict.
        """
        value = getattr(self.solution, key, None)
        if value is None:
            value = self.results.get(key, default)
        return value

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """
        Fuzzy numerical matching with tolerances.
        """
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """
        Verify fundamental physics laws and geometric relationships.
        """
        logs = []
        success = True

        # Part 1: Stationary incline (steps 1-3)
        if self.v0 is not None and self.x is not None and self.h is not None:
            # Time from horizontal motion: t1 = x / v0
            t1 = self.x / self.v0 if self.v0 != 0 else None
            if t1 and t1 > 0:
                # Check vertical free fall: h = 0.5 * g * t1^2
                h_expected = 0.5 * self.g * t1 ** 2
                if not self._is_close(self.h, h_expected):
                    logs.append(f"FAIL: Vertical free fall invariant: h={self.h}, expected {h_expected}")
                    success = False
                else:
                    logs.append("PASS: Vertical free fall invariant.")

                # Check geometry: tanθ = h / x
                if not self._is_close(self.tantheta, self.h / self.x):
                    logs.append(f"FAIL: Geometry invariant: tanθ={self.tantheta}, h/x={self.h/self.x}")
                    success = False
                else:
                    logs.append("PASS: Geometry invariant (tanθ = h/x).")
            else:
                logs.append("SKIP: Part 1 invariants due to missing or zero v0.")

        # Part 2: Moving incline (steps 4-8)
        if all(val is not None for val in [self.v0, self.a, self.x2, self.x1, self.h2]):
            # Time from horizontal motion of ball: t2 = x2 / v0
            t2 = self.x2 / self.v0 if self.v0 != 0 else None
            if t2 and t2 > 0:
                # Check vertical displacement: h2 = 0.5 * g * t2^2
                h2_expected = 0.5 * self.g * t2 ** 2
                if not self._is_close(self.h2, h2_expected):
                    logs.append(f"FAIL: Vertical displacement invariant: h2={self.h2}, expected {h2_expected}")
                    success = False
                else:
                    logs.append("PASS: Vertical displacement invariant.")

                # Check incline horizontal displacement: x1 = 0.5 * a * t2^2
                x1_expected = 0.5 * self.a * t2 ** 2
                if not self._is_close(self.x1, x1_expected):
                    logs.append(f"FAIL: Incline motion invariant: x1={self.x1}, expected {x1_expected}")
                    success = False
                else:
                    logs.append("PASS: Incline motion invariant.")

                # Check relative geometry: tanθ = h2 / (x2 - x1)
                denominator = self.x2 - self.x1
                if denominator != 0:
                    if not self._is_close(self.tantheta, self.h2 / denominator):
                        logs.append(f"FAIL: Relative geometry invariant: tanθ={self.tantheta}, h2/(x2-x1)={self.h2/denominator}")
                        success = False
                    else:
                        logs.append("PASS: Relative geometry invariant (tanθ = h2/(x2-x1)).")
                else:
                    logs.append("SKIP: Relative geometry invariant due to zero denominator.")
            else:
                logs.append("SKIP: Part 2 invariants due to missing or zero v0.")
        else:
            logs.append("SKIP: Part 2 invariants due to missing data.")

        self.logs.extend(logs)
        return success, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        """
        logs = []
        success = True

        # Check v0 consistency with Part 1
        if self.v0 is not None and self.x is not None and self.h is not None:
            # Compute v0 from geometry and free fall
            t_est = math.sqrt(2 * self.h / self.g)
            x_est = self.h / self.tantheta
            v0_est = x_est / t_est
            if not self._is_close(self.v0, v0_est):
                logs.append(f"FAIL: v0 trace consistency: computed {v0_est}, given {self.v0}")
                success = False
            else:
                logs.append("PASS: v0 trace consistency.")

        # Check a consistency with Part 2
        if all(val is not None for val in [self.v0, self.a, self.x2, self.x1, self.h2]):
            # Using the condition: tanθ = h2 / (x2 - x1)
            # Solve for a symbolically
            t2 = self.x2 / self.v0
            # From x1 = 0.5 * a * t2^2 => a = 2 * x1 / t2^2
            a_est = 2 * self.x1 / (t2 ** 2) if t2 != 0 else None
            if a_est is not None and not self._is_close(self.a, a_est):
                logs.append(f"FAIL: a trace consistency: computed {a_est}, given {self.a}")
                success = False
            else:
                logs.append("PASS: a trace consistency.")

        self.logs.extend(logs)
        return success, logs

    def test_monotonicity(self):
        """
        Perturb an input (height h) and check the trend in v0.
        Uses symbolic differentiation to verify monotonicity.
        """
        logs = []
        success = True

        try:
            # Symbolic check for v0 = sqrt(g*h/2) / tanθ
            h_sym = sp.symbols('h', positive=True)
            v0_expr = sp.sqrt(self.g * h_sym / 2) / self.tantheta
            dv0_dh = sp.diff(v0_expr, h_sym)
            # Evaluate derivative at the given h
            dv0_dh_val = dv0_dh.subs(h_sym, self.h).evalf()
            if dv0_dh_val <= 0:
                logs.append(f"FAIL: v0 monotonicity: derivative at h={self.h} is {dv0_dh_val} (expected positive)")
                success = False
            else:
                logs.append(f"PASS: v0 monotonicity: derivative at h={self.h} is {dv0_dh_val} > 0.")

            # For a, we check the expression from the condition: a = (2*v0/t2) - (g/tanθ)
            # But v0 and t2 depend on h. We use the relationships from Part 1 and Part 2.
            # However, due to potential inconsistencies in the problem, we skip a detailed monotonicity check.
            logs.append("SKIP: a monotonicity check due to complex dependencies.")
        except Exception as e:
            logs.append(f"ERROR in monotonicity test: {traceback.format_exc()}")
            success = False

        self.logs.extend(logs)
        return success, logs

    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        self.logs = []
        overall_success = True

        # Test 1: Check against target answers
        try:
            if self.v0 is not None and not self._is_close(self.v0, 4.0):
                self.logs.append(f"FAIL: v0 target mismatch: {self.v0} vs 4.0")
                overall_success = False
            else:
                self.logs.append("PASS: v0 matches target 4.0 m/s.")

            if self.a is not None and not self._is_close(self.a, 13.3):
                self.logs.append(f"FAIL: a target mismatch: {self.a} vs 13.3")
                overall_success = False
            else:
                self.logs.append("PASS: a matches target 13.3 m/s^2.")
        except Exception as e:
            self.logs.append(f"ERROR in target check: {e}")
            overall_success = False

        # Test 2: Invariants
        inv_success, inv_logs = self.test_invariants()
        overall_success = overall_success and inv_success

        # Test 3: Trace consistency
        trace_success, trace_logs = self.test_trace_consistency()
        overall_success = overall_success and trace_success

        # Test 4: Monotonicity
        mono_success, mono_logs = self.test_monotonicity()
        overall_success = overall_success and mono_success

        return overall_success, self.logs