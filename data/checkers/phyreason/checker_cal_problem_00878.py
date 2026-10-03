import sympy as sp
import math
import traceback
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Initialize the checker with safe attribute access.
        """
        # Safe extraction of inputs
        self.m = getattr(solution_instance, 'm', None)
        self.q = getattr(solution_instance, 'q', None)
        self.B = getattr(solution_instance, 'B', None)
        self.L = getattr(solution_instance, 'L', None)
        self.alpha = getattr(solution_instance, 'alpha', 30.0)  # degrees

        # Safe extraction of results
        self.results = getattr(solution_instance, 'results', {})
        self.beta = self.results.get('β', None)
        self.T = self.results.get('T', None)
        self.t_over_5 = self.results.get('t/5', None)
        self.t = self.results.get('t', None)
        self.theta = self.results.get('theta', None)
        self.r_3 = self.results.get('r_3', None)
        self.r_1 = self.results.get('r_1', None)
        self.v_3 = self.results.get('v_3', None)
        self.v_1 = self.results.get('v_1', None)

    def _isclose(self, a: float, b: float) -> bool:
        """
        Fuzzy numerical comparison with 1% tolerance.
        """
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

    def test_invariants(self) -> Tuple[bool, List[str]]:
        """
        Verify fundamental physical laws: Lorentz force equals centripetal force,
        period formula, and time-angle relationships.
        """
        logs = []
        success = True

        # 1. Lorentz force for v1 and r1
        if None not in (self.q, self.v_1, self.B, self.m, self.r_1):
            try:
                left = self.q * self.v_1 * self.B
                right = self.m * self.v_1**2 / self.r_1
                if not self._isclose(left, right):
                    logs.append(f"Lorentz force invariant failed for v1: left={left}, right={right}")
                    success = False
            except Exception as e:
                logs.append(f"Error in Lorentz force for v1: {e}")
                success = False
        else:
            logs.append("Missing values for Lorentz force check (v1)")

        # 2. Lorentz force for v3 and r3
        if None not in (self.q, self.v_3, self.B, self.m, self.r_3):
            try:
                left = self.q * self.v_3 * self.B
                right = self.m * self.v_3**2 / self.r_3
                if not self._isclose(left, right):
                    logs.append(f"Lorentz force invariant failed for v3: left={left}, right={right}")
                    success = False
            except Exception as e:
                logs.append(f"Error in Lorentz force for v3: {e}")
                success = False
        else:
            logs.append("Missing values for Lorentz force check (v3)")

        # 3. Period formula T = 2πm/(qB)
        if None not in (self.m, self.q, self.B, self.T):
            try:
                expected_T = 2 * math.pi * self.m / (self.q * self.B)
                if not self._isclose(self.T, expected_T):
                    logs.append(f"Period invariant failed: T={self.T}, expected={expected_T}")
                    success = False
            except Exception as e:
                logs.append(f"Error in period formula: {e}")
                success = False
        else:
            logs.append("Missing values for period check")

        # 4. Time and central angle for motion with v2: t = (β/2π) * T
        if None not in (self.t, self.beta, self.T):
            try:
                beta_rad = self.beta * math.pi / 180.0
                expected_t = (beta_rad / (2 * math.pi)) * self.T
                if not self._isclose(self.t, expected_t):
                    logs.append(f"Time-angle invariant failed for t: t={self.t}, expected={expected_t}")
                    success = False
            except Exception as e:
                logs.append(f"Error in time-angle for t: {e}")
                success = False
        else:
            logs.append("Missing values for time-angle check (t)")

        # 5. Time and central angle for motion with v3: t/5 = (θ/2π) * T
        if None not in (self.t_over_5, self.theta, self.T):
            try:
                theta_rad = self.theta * math.pi / 180.0
                expected_t_over_5 = (theta_rad / (2 * math.pi)) * self.T
                if not self._isclose(self.t_over_5, expected_t_over_5):
                    logs.append(f"Time-angle invariant failed for t/5: t/5={self.t_over_5}, expected={expected_t_over_5}")
                    success = False
            except Exception as e:
                logs.append(f"Error in time-angle for t/5: {e}")
                success = False
        else:
            logs.append("Missing values for time-angle check (t/5)")

        return success, logs

    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """
        Cross-check intermediate results with formulas from the steps.
        """
        logs = []
        success = True

        # 1. Check r1 = 2L/(2 + √3)
        if None not in (self.L, self.r_1):
            expected_r1 = 2 * self.L / (2 + math.sqrt(3))
            if not self._isclose(self.r_1, expected_r1):
                logs.append(f"r_1 inconsistent: {self.r_1} vs expected {expected_r1}")
                success = False

        # 2. Check v1 = 2qBL/((2+√3)m)
        if None not in (self.q, self.B, self.L, self.m, self.v_1):
            expected_v1 = 2 * self.q * self.B * self.L / ((2 + math.sqrt(3)) * self.m)
            if not self._isclose(self.v_1, expected_v1):
                logs.append(f"v_1 inconsistent: {self.v_1} vs expected {expected_v1}")
                success = False

        # 3. Check β = 300°
        if self.beta is not None and not self._isclose(self.beta, 300.0):
            logs.append(f"β inconsistent: {self.beta} vs expected 300")
            success = False

        # 4. Check T = 2πm/(qB)
        if None not in (self.m, self.q, self.B, self.T):
            expected_T = 2 * math.pi * self.m / (self.q * self.B)
            if not self._isclose(self.T, expected_T):
                logs.append(f"T inconsistent: {self.T} vs expected {expected_T}")
                success = False

        # 5. Check t = 5πm/(3qB)
        if None not in (self.m, self.q, self.B, self.t):
            expected_t = 5 * math.pi * self.m / (3 * self.q * self.B)
            if not self._isclose(self.t, expected_t):
                logs.append(f"t inconsistent: {self.t} vs expected {expected_t}")
                success = False

        # 6. Check t/5 = πm/(3qB)
        if None not in (self.m, self.q, self.B, self.t_over_5):
            expected_t_over_5 = math.pi * self.m / (3 * self.q * self.B)
            if not self._isclose(self.t_over_5, expected_t_over_5):
                logs.append(f"t/5 inconsistent: {self.t_over_5} vs expected {expected_t_over_5}")
                success = False

        # 7. Check θ = 60°
        if self.theta is not None and not self._isclose(self.theta, 60.0):
            logs.append(f"θ inconsistent: {self.theta} vs expected 60")
            success = False

        # 8. Check r3 = 2√3 L / 3
        if None not in (self.L, self.r_3):
            expected_r3 = 2 * math.sqrt(3) * self.L / 3
            if not self._isclose(self.r_3, expected_r3):
                logs.append(f"r_3 inconsistent: {self.r_3} vs expected {expected_r3}")
                success = False

        # 9. Check v3 = 2√3 qBL/(3m)
        if None not in (self.q, self.B, self.L, self.m, self.v_3):
            expected_v3 = 2 * math.sqrt(3) * self.q * self.B * self.L / (3 * self.m)
            if not self._isclose(self.v_3, expected_v3):
                logs.append(f"v_3 inconsistent: {self.v_3} vs expected {expected_v3}")
                success = False

        return success, logs

    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Perturb inputs and verify expected trends using formulas from steps.
        """
        logs = []
        success = True

        # 1. v1 proportional to B (v1 ∝ B)
        if None not in (self.m, self.q, self.B, self.L, self.v_1):
            try:
                B_perturbed = self.B * 1.01
                v1_perturbed = 2 * self.q * B_perturbed * self.L / ((2 + math.sqrt(3)) * self.m)
                expected_v1_perturbed = self.v_1 * 1.01
                if not self._isclose(v1_perturbed, expected_v1_perturbed):
                    logs.append(f"Monotonicity v1(B) failed: v1_perturbed={v1_perturbed}, expected={expected_v1_perturbed}")
                    success = False
            except Exception as e:
                logs.append(f"Error in monotonicity test for v1 and B: {e}")
                success = False
        else:
            logs.append("Missing values for v1(B) monotonicity test")

        # 2. v3 proportional to B (v3 ∝ B)
        if None not in (self.m, self.q, self.B, self.L, self.v_3):
            try:
                B_perturbed = self.B * 1.01
                v3_perturbed = 2 * math.sqrt(3) * self.q * B_perturbed * self.L / (3 * self.m)
                expected_v3_perturbed = self.v_3 * 1.01
                if not self._isclose(v3_perturbed, expected_v3_perturbed):
                    logs.append(f"Monotonicity v3(B) failed: v3_perturbed={v3_perturbed}, expected={expected_v3_perturbed}")
                    success = False
            except Exception as e:
                logs.append(f"Error in monotonicity test for v3 and B: {e}")
                success = False
        else:
            logs.append("Missing values for v3(B) monotonicity test")

        # 3. T proportional to m (T ∝ m)
        if None not in (self.m, self.q, self.B, self.T):
            try:
                m_perturbed = self.m * 1.01
                T_perturbed = 2 * math.pi * m_perturbed / (self.q * self.B)
                expected_T_perturbed = self.T * 1.01
                if not self._isclose(T_perturbed, expected_T_perturbed):
                    logs.append(f"Monotonicity T(m) failed: T_perturbed={T_perturbed}, expected={expected_T_perturbed}")
                    success = False
            except Exception as e:
                logs.append(f"Error in monotonicity test for T and m: {e}")
                success = False
        else:
            logs.append("Missing values for T(m) monotonicity test")

        # 4. T inversely proportional to q (T ∝ 1/q)
        if None not in (self.m, self.q, self.B, self.T):
            try:
                q_perturbed = self.q * 1.01
                T_perturbed = 2 * math.pi * self.m / (q_perturbed * self.B)
                expected_T_perturbed = self.T / 1.01
                if not self._isclose(T_perturbed, expected_T_perturbed):
                    logs.append(f"Monotonicity T(q) failed: T_perturbed={T_perturbed}, expected={expected_T_perturbed}")
                    success = False
            except Exception as e:
                logs.append(f"Error in monotonicity test for T and q: {e}")
                success = False
        else:
            logs.append("Missing values for T(q) monotonicity test")

        return success, logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all tests and return overall success and aggregated logs.
        """
        all_logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_method in tests:
            try:
                test_success, logs = test_method()
                all_logs.append(f"{test_name}: {'PASS' if test_success else 'FAIL'}")
                all_logs.extend(logs)
                if not test_success:
                    overall_success = False
            except Exception as e:
                all_logs.append(f"ERROR in {test_name}: {e}")
                all_logs.append(traceback.format_exc())
                overall_success = False

        return overall_success, all_logs