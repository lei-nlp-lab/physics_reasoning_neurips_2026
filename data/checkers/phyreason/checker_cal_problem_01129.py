import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        Use getattr and dict.get to avoid AttributeError in sandbox.
        """
        self.solution_instance = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Input parameters (with safe defaults)
        self.L = getattr(solution_instance, 'L', 1.0)
        self.v0 = getattr(solution_instance, 'v0', 1.0)
        self.m = getattr(solution_instance, 'm', 1.0)
        self.q = getattr(solution_instance, 'q', 1.0)

        # Results (expected keys)
        self.x1 = self.results.get('x_1', 0.0)
        self.E = self.results.get('E', 0.0)
        self.B = self.results.get('B', 0.0)
        self.fracEB = self.results.get('fracEB', 0.0)  # E/B ratio

        # Constants
        self.sin53 = 0.8
        self.cos53 = 0.6

        # Derived quantities (computed for checking)
        self.v_cross = (self.v0 * self.sin53) / self.cos53  # v at x-axis from step_2: v = (sin53/cos53) * v0

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """
        Fuzzy numerical matching using math.isclose.
        Handles both numeric and sympy expressions (converts to float).
        """
        try:
            a_val = float(a) if hasattr(a, 'evalf') else a
            b_val = float(b) if hasattr(b, 'evalf') else b
            return math.isclose(a_val, b_val, rel_tol=rel_tol, abs_tol=abs_tol)
        except Exception:
            return False

    def test_invariants(self):
        """
        Verify fundamental physics laws:
        1. Work-energy theorem in electric field region.
        2. Impulse-momentum theorem in vertical direction.
        3. Lorentz force as centripetal force in magnetic field.
        4. Geometric relationship for circular arc.
        """
        logs = []
        success = True

        try:
            # 1. Work-energy theorem (step_4)
            # qEL = 0.5*m*(v^2 - v0^2)
            lhs_we = self.q * self.E * self.L
            rhs_we = 0.5 * self.m * (self.v_cross**2 - self.v0**2)
            if not self._is_close(lhs_we, rhs_we):
                logs.append(f"Work-energy theorem mismatch: qEL={lhs_we}, ΔK={rhs_we}")
                success = False
            else:
                logs.append("Work-energy theorem holds.")

            # 2. Impulse-momentum theorem (step_3)
            # qEt = m*v*sin53 + m*v0*cos53
            # First compute time t from horizontal motion: x1 = v0*sin53 * t -> t = x1/(v0*sin53)
            t = self.x1 / (self.v0 * self.sin53)
            lhs_im = self.q * self.E * t
            rhs_im = self.m * self.v_cross * self.sin53 + self.m * self.v0 * self.cos53
            if not self._is_close(lhs_im, rhs_im):
                logs.append(f"Impulse-momentum mismatch: qEt={lhs_im}, Δp_y={rhs_im}")
                success = False
            else:
                logs.append("Impulse-momentum theorem holds.")

            # 3. Lorentz force as centripetal force (step_7)
            # B*q*v = m*v^2 / R  -> B = (m*v)/(q*R)
            # Geometric relationship (step_8): 2R*sin53 = x1 -> R = x1/(2*sin53)
            R = self.x1 / (2 * self.sin53)
            lhs_lf = self.B * self.q * self.v_cross
            rhs_lf = self.m * self.v_cross**2 / R
            if not self._is_close(lhs_lf, rhs_lf):
                logs.append(f"Lorentz force mismatch: Bqv={lhs_lf}, mv^2/R={rhs_lf}")
                success = False
            else:
                logs.append("Lorentz force as centripetal force holds.")

            # 4. Geometric relationship (step_8) already used above; verify consistency
            if not self._is_close(2 * R * self.sin53, self.x1):
                logs.append(f"Geometry mismatch: 2R*sin53={2*R*self.sin53}, x1={self.x1}")
                success = False
            else:
                logs.append("Geometric relationship holds.")

        except Exception as e:
            logs.append(f"Error in test_invariants: {traceback.format_exc()}")
            success = False

        return success, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results with final answers.
        Specifically, verify that the computed velocity at x-axis and the ratio E/B
        are consistent with the derived expressions.
        """
        logs = []
        success = True

        try:
            # Check velocity at x-axis: v = (4/3) v0 (from step_2)
            expected_v = (4/3) * self.v0
            if not self._is_close(self.v_cross, expected_v):
                logs.append(f"Velocity mismatch: v_cross={self.v_cross}, expected (4/3)v0={expected_v}")
                success = False
            else:
                logs.append(f"Velocity at x-axis is consistent: v = (4/3)v0 = {expected_v}")

            # Check E/B ratio (step_11): E/B = (5/8) v0
            if self.B != 0:
                computed_ratio = self.E / self.B
                expected_ratio = (5/8) * self.v0
                if not self._is_close(computed_ratio, expected_ratio):
                    logs.append(f"E/B ratio mismatch: computed={computed_ratio}, expected (5/8)v0={expected_ratio}")
                    success = False
                else:
                    logs.append(f"E/B ratio is consistent: E/B = (5/8)v0 = {expected_ratio}")
            else:
                logs.append("Skipping E/B ratio check because B is zero.")

            # Check x1 from magnetic geometry vs. expected (24/7)L
            expected_x1 = (24/7) * self.L
            if not self._is_close(self.x1, expected_x1):
                logs.append(f"x1 mismatch: computed={self.x1}, expected (24/7)L={expected_x1}")
                success = False
            else:
                logs.append(f"x1 is consistent: x1 = (24/7)L = {expected_x1}")

        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {traceback.format_exc()}")
            success = False

        return success, logs

    def test_monotonicity(self):
        """
        Perturb an input (e.g., L) and verify that the outputs change as expected.
        Uses the derived formulas to recompute results, then restores the original.
        """
        logs = []
        success = True

        try:
            # Save original values
            L_orig = self.L
            # Perturb L by +10%
            self.L = L_orig * 1.1

            # Recompute expected results using formulas from steps
            # x1 = (24/7) * L
            x1_new = (24/7) * self.L
            # E = (7*m*v0^2) / (18*q*L)
            E_new = (7 * self.m * self.v0**2) / (18 * self.q * self.L)
            # B = (28*m*v0) / (45*q*L)
            B_new = (28 * self.m * self.v0) / (45 * self.q * self.L)
            # fracEB = (5/8) * v0
            fracEB_new = (5/8) * self.v0

            # Check trends: x1 should increase, E and B should decrease, fracEB unchanged
            if not (x1_new > self.x1 and E_new < self.E and B_new < self.B):
                logs.append(f"Monotonicity fail after L increase: x1={x1_new} (should increase), E={E_new} (should decrease), B={B_new} (should decrease)")
                success = False
            else:
                logs.append("Monotonicity holds: x1 increases, E and B decrease with increased L.")

            # Restore original L
            self.L = L_orig

        except Exception as e:
            logs.append(f"Error in test_monotonicity: {traceback.format_exc()}")
            success = False
            # Ensure L is restored even on error
            self.L = L_orig if 'L_orig' in locals() else self.L

        return success, logs

    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True

        # Run each test
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            success, logs = test_func()
            all_logs.append(f"=== {test_name} ===")
            all_logs.extend(logs)
            if not success:
                overall_success = False
                all_logs.append(f"FAILED: {test_name}")
            else:
                all_logs.append(f"PASSED: {test_name}")

        return overall_success, all_logs