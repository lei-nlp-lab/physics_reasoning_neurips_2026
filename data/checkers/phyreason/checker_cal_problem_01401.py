import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access for inputs
        self.m = getattr(solution_instance, 'm', solution_instance.results.get('m', None))
        self.g = getattr(solution_instance, 'g', solution_instance.results.get('g', 9.8))
        self.h = getattr(solution_instance, 'h', solution_instance.results.get('h', None))
        self.q = getattr(solution_instance, 'q', solution_instance.results.get('q', None))
        self.Q = getattr(solution_instance, 'Q', solution_instance.results.get('Q', None))

        # Safe attribute access for results
        self.W = getattr(solution_instance, 'W', solution_instance.results.get('W', None))
        self.U_AC = getattr(solution_instance, 'U_AC', solution_instance.results.get('U_AC', None))
        self.W_BC = getattr(solution_instance, 'W_BC', solution_instance.results.get('W_BC', None))
        self.W_AC = getattr(solution_instance, 'W_AC', solution_instance.results.get('W_AC', None))

        # Derived quantities
        self.v_B = math.sqrt(3 * self.g * self.h) if (self.g is not None and self.h is not None) else None
        self.logs = []

    def _isclose(self, a, b):
        """Robust fuzzy comparison for numbers or symbolic expressions."""
        if a is None or b is None:
            return False
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)
        try:
            # Try symbolic equality with simplification
            diff = sp.simplify(a - b)
            return diff == 0 or (isinstance(diff, sp.Number) and abs(float(diff)) < 1e-2)
        except:
            return False

    def test_invariants(self):
        """Check fundamental physics laws."""
        logs = []
        try:
            # 1. Work-energy theorem from A to B
            if all(v is not None for v in [self.m, self.g, self.h, self.W, self.v_B]):
                lhs = self.m * self.g * self.h + self.W
                rhs = 0.5 * self.m * self.v_B**2
                if self._isclose(lhs, rhs):
                    logs.append("PASS: Work-energy theorem holds (A->B).")
                else:
                    logs.append(f"FAIL: Work-energy theorem mismatch. lhs={lhs}, rhs={rhs}")
            else:
                logs.append("SKIP: Missing data for work-energy theorem.")

            # 2. Equipotential condition B->C
            if self.W_BC is not None:
                if self._isclose(self.W_BC, 0):
                    logs.append("PASS: W_BC = 0 (B and C equipotential).")
                else:
                    logs.append(f"FAIL: W_BC should be zero, got {self.W_BC}")
            else:
                logs.append("SKIP: W_BC not provided.")

        except Exception as e:
            logs.append(f"ERROR in test_invariants: {traceback.format_exc()}")
        self.logs.extend(logs)
        return all("PASS" in log or "SKIP" in log for log in logs)

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        try:
            # Check W_AC = W + W_BC
            if all(v is not None for v in [self.W_AC, self.W, self.W_BC]):
                if self._isclose(self.W_AC, self.W + self.W_BC):
                    logs.append("PASS: W_AC = W + W_BC.")
                else:
                    logs.append(f"FAIL: W_AC != W + W_BC. W_AC={self.W_AC}, W+W_BC={self.W+self.W_BC}")
            else:
                logs.append("SKIP: Missing W_AC, W, or W_BC.")

            # Check U_AC = -W_AC / q
            if all(v is not None for v in [self.U_AC, self.W_AC, self.q]):
                if self._isclose(self.U_AC, -self.W_AC / self.q):
                    logs.append("PASS: U_AC = -W_AC / q.")
                else:
                    logs.append(f"FAIL: U_AC != -W_AC/q. U_AC={self.U_AC}, -W_AC/q={-self.W_AC/self.q}")
            else:
                logs.append("SKIP: Missing U_AC, W_AC, or q.")

            # Check W = 0.5 * m * g * h
            if all(v is not None for v in [self.W, self.m, self.g, self.h]):
                expected_W = 0.5 * self.m * self.g * self.h
                if self._isclose(self.W, expected_W):
                    logs.append("PASS: W matches 0.5*m*g*h.")
                else:
                    logs.append(f"FAIL: W mismatch. Expected {expected_W}, got {self.W}")
            else:
                logs.append("SKIP: Missing W, m, g, or h.")

        except Exception as e:
            logs.append(f"ERROR in test_trace_consistency: {traceback.format_exc()}")
        self.logs.extend(logs)
        return all("PASS" in log or "SKIP" in log for log in logs)

    def test_monotonicity(self):
        """Perturb input h and check that W and U_AC scale as expected."""
        logs = []
        try:
            if None in [self.m, self.g, self.h, self.q]:
                logs.append("SKIP: Missing inputs for monotonicity test.")
                self.logs.extend(logs)
                return True

            # Perturb h by +10%
            h_perturbed = self.h * 1.1
            # Recalculate using the derived formulas
            W_perturbed = 0.5 * self.m * self.g * h_perturbed
            U_AC_perturbed = -W_perturbed / self.q

            # Check monotonicity: W increases with h, U_AC becomes more negative (if q>0)
            if self._isclose(W_perturbed, 1.1 * self.W):  # linear scaling
                logs.append("PASS: W scales linearly with h.")
            else:
                logs.append(f"FAIL: W does not scale linearly. W_perturbed={W_perturbed}, expected {1.1*self.W}")

            if self.q > 0:
                if U_AC_perturbed < self.U_AC:  # more negative
                    logs.append("PASS: U_AC becomes more negative with increased h (q>0).")
                else:
                    logs.append(f"FAIL: U_AC monotonicity violated. U_AC={self.U_AC}, U_AC_perturbed={U_AC_perturbed}")
            else:
                logs.append("SKIP: q sign unknown, cannot check U_AC monotonicity.")

        except Exception as e:
            logs.append(f"ERROR in test_monotonicity: {traceback.format_exc()}")
        self.logs.extend(logs)
        return all("PASS" in log or "SKIP" in log for log in logs)

    def run_all_tests(self):
        """Execute all checks and return overall success and logs."""
        self.logs = []
        success = True
        try:
            success &= self.test_invariants()
            success &= self.test_trace_consistency()
            success &= self.test_monotonicity()
        except Exception as e:
            self.logs.append(f"CRITICAL: Checker crashed: {traceback.format_exc()}")
            success = False
        return success, self.logs