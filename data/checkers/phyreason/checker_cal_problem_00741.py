import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.solution = solution_instance
        self.logs = []

        # Safe attribute extraction using getattr and results dict fallback
        self.n_ratio = self._get_value('n_1/n_2')
        self.I = self._get_value('I')
        self.U1 = self._get_value('U_1')
        self.U2 = self._get_value('U_2')

    def _get_value(self, key):
        """Safely retrieve a value from solution instance."""
        val = getattr(self.solution, key, None)
        if val is None:
            results = getattr(self.solution, 'results', {})
            val = results.get(key, None)
        return val

    def test_invariants(self):
        """Check fundamental physics laws: transformer relation and Ohm's law."""
        logs = []
        success = True

        # 1. Transformer voltage relation: n1/n2 = U1 / U2
        if self.n_ratio is not None and self.U1 is not None and self.U2 is not None:
            if not math.isclose(self.n_ratio, self.U1 / self.U2, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Transformer relation violated. n1/n2 = {self.n_ratio}, U1/U2 = {self.U1 / self.U2}")
                success = False
            else:
                logs.append(f"PASS: Transformer relation holds: n1/n2 = U1/U2 = {self.n_ratio}")
        else:
            logs.append("SKIP: Transformer relation check due to missing data.")
            success = False

        # 2. Ohm's law on secondary side: I = U2 / R, ensure positive resistance
        if self.I is not None and self.U2 is not None:
            if self.I <= 0:
                logs.append(f"FAIL: Secondary current I must be positive, got {self.I}")
                success = False
            else:
                R_bulb = self.U2 / self.I
                if R_bulb <= 0:
                    logs.append(f"FAIL: Computed bulb resistance R = U2/I = {R_bulb} <= 0")
                    success = False
                else:
                    logs.append(f"PASS: Ohm's law gives positive R_bulb = {R_bulb:.2f} Ω")
        else:
            logs.append("SKIP: Ohm's law check due to missing data.")
            success = False

        self.logs.extend(logs)
        return success

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers (if step data is available)."""
        logs = []
        # In this problem, step data is not provided in the checker, so we only note that.
        logs.append("INFO: Trace consistency check requires step-wise data, which is not provided.")
        self.logs.extend(logs)
        return True  # Not a failure, just informational

    def test_monotonicity(self):
        """Perturb inputs (turns ratio and bulb resistance) and verify expected trends."""
        logs = []
        success = True

        if all(v is not None for v in [self.n_ratio, self.U1, self.U2, self.I]):
            R_bulb = self.U2 / self.I

            # Perturb turns ratio n1/n2 upward by 10%
            n_new = self.n_ratio * 1.1
            U2_new = self.U1 / n_new
            I_new = U2_new / R_bulb
            if not I_new < self.I:
                logs.append(f"FAIL: Increasing n should decrease I. n: {self.n_ratio} -> {n_new}, I: {self.I} -> {I_new}")
                success = False
            else:
                logs.append(f"PASS: n↑ → I↓ (I_new={I_new:.3f} A < I={self.I} A)")

            # Perturb bulb resistance upward by 10%
            R_new = R_bulb * 1.1
            I_new2 = self.U2 / R_new
            if not I_new2 < self.I:
                logs.append(f"FAIL: Increasing R_bulb should decrease I. R: {R_bulb:.2f} -> {R_new:.2f} Ω, I: {self.I} -> {I_new2}")
                success = False
            else:
                logs.append(f"PASS: R_bulb↑ → I↓ (I_new={I_new2:.3f} A < I={self.I} A)")
        else:
            logs.append("SKIP: Monotonicity check due to missing data.")
            success = False

        self.logs.extend(logs)
        return success

    def run_all_tests(self):
        """Execute all checks and return overall success and log messages."""
        self.logs = []
        success = True

        if not self.test_invariants():
            success = False
        if not self.test_trace_consistency():
            success = False
        if not self.test_monotonicity():
            success = False

        return success, self.logs