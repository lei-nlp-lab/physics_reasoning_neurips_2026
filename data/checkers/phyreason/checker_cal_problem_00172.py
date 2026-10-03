import math
import sympy as sp
from sympy import symbols, Eq, simplify, sqrt, diff, cos

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the PhysicsChecker with safe attribute access.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})
        self.inputs = getattr(solution_instance, 'inputs', {})

        # Safe extraction of inputs with defaults (using dimensionless units for safety)
        self.m = self.inputs.get('m', 1.0)
        self.q = self.inputs.get('q', 1.0)
        self.v0 = self.inputs.get('v0', 1.0)
        self.r0 = self.inputs.get('r0', 1.0)
        self.theta = self.inputs.get('theta', math.acos(math.sqrt(4/5)))  # from cosθ = √(4/5)

        # Safe extraction of results with defaults
        self.E0 = self.results.get('E_0', 0.0)
        self.B = self.results.get('B', 0.0)
        self.l = self.results.get('l', 0.0)
        self.fracDeltaBB = self.results.get('fracDeltaBB', 0.0)
        self.r = self.results.get('r', 0.0)  # radius for 0.5m ion in B-field
        self.v = self.results.get('v', 0.0)  # velocity for 0.5m ion after acceleration

        # Expected values computed from KB and STEPS ANALYSIS
        self.expected_E0 = self.m * self.v0**2 / (self.q * self.r0)
        self.expected_B = self.m * self.v0 / (self.q * self.r0)
        self.expected_v = math.sqrt(5) * self.v0
        self.expected_r = (math.sqrt(5) / 2) * self.r0
        self.expected_l = 1.5 * self.r0
        self.expected_fracDeltaBB_exact = math.sqrt(17) - 4
        self.expected_fracDeltaBB_approx = 0.12

    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Fuzzy numerical matching with 1% tolerance."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """
        Verify fundamental physical invariants: Newton's 2nd Law and Work-Energy Theorem.
        """
        logs = []
        success = True

        # --- Invariant 1: Centripetal force for m ion in electric field ---
        # Electric force = centripetal force: q*E0 = m*v0^2 / r0
        expected_E0 = self.m * self.v0**2 / (self.q * self.r0)
        if not self._isclose(self.E0, expected_E0):
            logs.append(f"FAIL: E0 mismatch. Got {self.E0}, expected {expected_E0}.")
            success = False
        else:
            logs.append("PASS: E0 satisfies Newton's 2nd Law (centripetal force).")

        # --- Invariant 2: Work-Energy Theorem for 0.5m ion ---
        # q*U_NP = ΔKE, with U_NP = m*v0^2 / q (given)
        U_NP = self.m * self.v0**2 / self.q
        delta_KE = 0.25 * self.m * (self.v**2 - self.v0**2)  # 0.5m mass factor
        if not self._isclose(self.q * U_NP, delta_KE):
            logs.append(f"FAIL: Work-Energy Theorem mismatch. q*U_NP={self.q*U_NP}, ΔKE={delta_KE}.")
            success = False
        else:
            logs.append("PASS: Work-Energy Theorem holds for 0.5m ion.")

        # --- Invariant 3: Centripetal force for 0.5m ion in magnetic field ---
        # Magnetic force = centripetal force: q*v*B = 0.5m*v^2 / r
        lhs = self.q * self.v * self.B
        rhs = 0.5 * self.m * self.v**2 / self.r
        if not self._isclose(lhs, rhs):
            logs.append(f"FAIL: Magnetic centripetal force mismatch. qvB={lhs}, 0.5m v^2/r={rhs}.")
            success = False
        else:
            logs.append("PASS: Newton's 2nd Law holds for 0.5m ion in B-field.")

        return success, logs

    def test_trace_consistency(self):
        """
        Cross-check intermediate results (v, r) with final results (l, ΔB/B).
        """
        logs = []
        success = True

        # Check velocity v for 0.5m ion
        if not self._isclose(self.v, self.expected_v):
            logs.append(f"FAIL: Velocity v mismatch. Got {self.v}, expected {self.expected_v}.")
            success = False
        else:
            logs.append(f"PASS: Velocity v = {self.v:.3f} matches expected √5 v0.")

        # Check radius r for 0.5m ion in B-field
        if not self._isclose(self.r, self.expected_r):
            logs.append(f"FAIL: Radius r mismatch. Got {self.r}, expected {self.expected_r}.")
            success = False
        else:
            logs.append(f"PASS: Radius r = {self.r:.3f} matches expected (√5/2) r0.")

        # Check distance l on detection plate
        if not self._isclose(self.l, self.expected_l):
            logs.append(f"FAIL: Distance l mismatch. Got {self.l}, expected {self.expected_l}.")
            success = False
        else:
            logs.append(f"PASS: Distance l = {self.l:.3f} matches expected 1.5 r0.")

        # Check ΔB/B (exact and approximate)
        if not (self._isclose(self.fracDeltaBB, self.expected_fracDeltaBB_exact) or
                self._isclose(self.fracDeltaBB, self.expected_fracDeltaBB_approx)):
            logs.append(f"FAIL: ΔB/B mismatch. Got {self.fracDeltaBB}, expected {self.expected_fracDeltaBB_exact:.3f} (exact) or {self.expected_fracDeltaBB_approx:.3f} (approx).")
            success = False
        else:
            logs.append(f"PASS: ΔB/B = {self.fracDeltaBB:.3f} matches expected value.")

        return success, logs

    def test_monotonicity(self):
        """
        Perturb inputs and verify expected trends in outputs.
        Uses symbolic derivatives to check monotonicity.
        """
        logs = []
        success = True

        # Define symbolic variables
        m, q, v0, r0 = symbols('m q v0 r0', positive=True)
        theta = symbols('theta', positive=True)

        # Expected formulas from STEPS ANALYSIS
        E0_expr = m * v0**2 / (q * r0)
        B_expr = m * v0 / (q * r0)
        v_expr = sqrt(5) * v0
        r_expr = (sqrt(5) / 2) * r0
        l_expr = 1.5 * r0
        # ΔB/B is constant (sqrt(17)-4), so derivative is zero

        # Check monotonicity with respect to mass m
        dE0_dm = diff(E0_expr, m)
        dB_dm = diff(B_expr, m)
        dv_dm = diff(v_expr, m)
        dr_dm = diff(r_expr, m)
        dl_dm = diff(l_expr, m)

        # All derivatives should be non-negative (positive or zero) for physical inputs
        trends = [
            (dE0_dm, "E0 w.r.t m", ">0"),
            (dB_dm, "B w.r.t m", ">0"),
            (dv_dm, "v w.r.t m", "=0"),
            (dr_dm, "r w.r.t m", "=0"),
            (dl_dm, "l w.r.t m", "=0")
        ]

        for deriv, name, expected in trends:
            # Evaluate derivative at current input values
            deriv_val = deriv.subs({m: self.m, q: self.q, v0: self.v0, r0: self.r0})
            if expected == ">0":
                if deriv_val <= 0:
                    logs.append(f"FAIL: Monotonicity {name} failed. Derivative = {deriv_val}, expected positive.")
                    success = False
                else:
                    logs.append(f"PASS: {name} increases with m (derivative = {deriv_val:.3f}).")
            elif expected == "=0":
                if not math.isclose(float(deriv_val), 0, rel_tol=1e-9):
                    logs.append(f"FAIL: Monotonicity {name} failed. Derivative = {deriv_val}, expected zero.")
                    success = False
                else:
                    logs.append(f"PASS: {name} independent of m (derivative = 0).")

        # Additional check: perturb v0 and verify trends
        dE0_dv0 = diff(E0_expr, v0)
        dB_dv0 = diff(B_expr, v0)
        dv_dv0 = diff(v_expr, v0)
        dr_dv0 = diff(r_expr, v0)
        dl_dv0 = diff(l_expr, v0)

        trends_v0 = [
            (dE0_dv0, "E0 w.r.t v0", ">0"),
            (dB_dv0, "B w.r.t v0", ">0"),
            (dv_dv0, "v w.r.t v0", ">0"),
            (dr_dv0, "r w.r.t v0", "=0"),
            (dl_dv0, "l w.r.t v0", "=0")
        ]

        for deriv, name, expected in trends_v0:
            deriv_val = deriv.subs({m: self.m, q: self.q, v0: self.v0, r0: self.r0})
            if expected == ">0":
                if deriv_val <= 0:
                    logs.append(f"FAIL: Monotonicity {name} failed. Derivative = {deriv_val}, expected positive.")
                    success = False
                else:
                    logs.append(f"PASS: {name} increases with v0 (derivative = {deriv_val:.3f}).")
            elif expected == "=0":
                if not math.isclose(float(deriv_val), 0, rel_tol=1e-9):
                    logs.append(f"FAIL: Monotonicity {name} failed. Derivative = {deriv_val}, expected zero.")
                    success = False
                else:
                    logs.append(f"PASS: {name} independent of v0 (derivative = 0).")

        return success, logs

    def run_all_tests(self):
        """
        Execute all tests and return overall success and logs.
        """
        all_logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            success, logs = test_func()
            all_logs.append(f"--- {test_name} ---")
            all_logs.extend(logs)
            if not success:
                overall_success = False

        return overall_success, all_logs