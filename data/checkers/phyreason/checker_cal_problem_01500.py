import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with defaults from problem statement
        self.m = getattr(solution_instance, 'm', 1.0)
        self.L = getattr(solution_instance, 'L', 0.5)
        self.T_max = getattr(solution_instance, 'T_max', 42.0)
        self.h = getattr(solution_instance, 'h', 5.5)
        self.g = getattr(solution_instance, 'g', 10.0)

        # Extract results safely; if not present, set to None
        results = getattr(solution_instance, 'results', {})
        self.v = getattr(solution_instance, 'v', results.get('v', None))
        self.v_1 = getattr(solution_instance, 'v_1', results.get('v_1', None))
        self.t = getattr(solution_instance, 't', results.get('t', None))
        self.x = getattr(solution_instance, 'x', results.get('x', None))

        # Store symbolic variables for invariant checks
        self._m_sym = sp.symbols('m')
        self._L_sym = sp.symbols('L')
        self._g_sym = sp.symbols('g')
        self._T_sym = sp.symbols('T')
        self._v_sym = sp.symbols('v')
        self._v1_sym = sp.symbols('v1')
        self._t_sym = sp.symbols('t')
        self._h_sym = sp.symbols('h')

    def test_invariants(self) -> Tuple[bool, List[str]]:
        logs = []
        success = True

        # 1. Check Newton's second law at the highest point (tension = 0)
        if self.v is not None:
            # Symbolic: m * v^2 / L = m * g
            lhs_top = self.m * self.v**2 / self.L
            rhs_top = self.m * self.g
            if not math.isclose(lhs_top, rhs_top, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Top point circular motion: m*v²/L = {lhs_top} != m*g = {rhs_top}")
                success = False
            else:
                logs.append("PASS: Top point circular motion satisfies Newton's second law.")
        else:
            logs.append("SKIP: v not provided for top point check.")

        # 2. Check Newton's second law at the lowest point (with T_max)
        if self.v_1 is not None:
            # Symbolic: T_max - m*g = m * v_1^2 / L
            lhs_bottom = self.T_max - self.m * self.g
            rhs_bottom = self.m * self.v_1**2 / self.L
            if not math.isclose(lhs_bottom, rhs_bottom, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Bottom point circular motion: T_max - m*g = {lhs_bottom} != m*v1²/L = {rhs_bottom}")
                success = False
            else:
                logs.append("PASS: Bottom point circular motion satisfies Newton's second law.")
        else:
            logs.append("SKIP: v_1 not provided for bottom point check.")

        # 3. Check kinematic free-fall from height (h-L)
        if self.t is not None:
            # Symbolic: h - L = 0.5 * g * t^2
            fall_height = self.h - self.L
            expected_t = math.sqrt(2 * fall_height / self.g)
            if not math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Free-fall time: t = {self.t} != sqrt(2*(h-L)/g) = {expected_t}")
                success = False
            else:
                logs.append("PASS: Free-fall time matches kinematics.")
        else:
            logs.append("SKIP: t not provided for free-fall check.")

        return success, logs

    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        logs = []
        success = True

        # Check consistency of each step with the given formulas
        # Step 1: v = sqrt(g*L)
        if self.v is not None:
            expected_v = math.sqrt(self.g * self.L)
            if not math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v = {self.v} != sqrt(g*L) = {expected_v}")
                success = False
            else:
                logs.append(f"PASS: v = {self.v} matches sqrt(g*L) ≈ {expected_v}.")

        # Step 3: v_1 = sqrt((T_max - m*g)*L / m)
        if self.v_1 is not None:
            expected_v1 = math.sqrt((self.T_max - self.m * self.g) * self.L / self.m)
            if not math.isclose(self.v_1, expected_v1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: v_1 = {self.v_1} != sqrt((T_max - m*g)*L/m) = {expected_v1}")
                success = False
            else:
                logs.append(f"PASS: v_1 = {self.v_1} matches expected {expected_v1}.")

        # Step 4: t = sqrt(2*(h-L)/g)
        if self.t is not None:
            expected_t = math.sqrt(2 * (self.h - self.L) / self.g)
            if not math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: t = {self.t} != sqrt(2*(h-L)/g) = {expected_t}")
                success = False
            else:
                logs.append(f"PASS: t = {self.t} matches expected {expected_t}.")

        # Step 6: x = v_1 * t
        if self.x is not None and self.v_1 is not None and self.t is not None:
            expected_x = self.v_1 * self.t
            if not math.isclose(self.x, expected_x, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: x = {self.x} != v_1 * t = {expected_x}")
                success = False
            else:
                logs.append(f"PASS: x = {self.x} matches v_1 * t = {expected_x}.")

        return success, logs

    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        logs = []
        success = True

        # Perturb mass m by +10%
        original_m = self.m
        perturbed_m = original_m * 1.1
        self.m = perturbed_m

        # Recompute expected values using formulas (not relying on solution methods)
        v_pert = math.sqrt(self.g * self.L)  # unchanged from original
        v1_pert = math.sqrt((self.T_max - self.m * self.g) * self.L / self.m)
        t_pert = math.sqrt(2 * (self.h - self.L) / self.g)  # unchanged
        x_pert = v1_pert * t_pert

        # Restore original mass
        self.m = original_m

        # Check monotonicity: if m increases, v1 and x should decrease (since T_max is fixed)
        if self.v_1 is not None and self.x is not None:
            if v1_pert < self.v_1:
                logs.append(f"PASS: v1 decreases from {self.v_1} to {v1_pert} when m increases (as expected).")
            else:
                logs.append(f"FAIL: v1 did not decrease when m increased (from {self.v_1} to {v1_pert}).")
                success = False

            if x_pert < self.x:
                logs.append(f"PASS: x decreases from {self.x} to {x_pert} when m increases (as expected).")
            else:
                logs.append(f"FAIL: x did not decrease when m increased (from {self.x} to {x_pert}).")
                success = False
        else:
            logs.append("SKIP: v_1 or x not available for monotonicity check.")

        return success, logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        all_logs = []
        overall_success = True

        # Run each test and collect results
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for test_name, test_func in tests:
            all_logs.append(f"--- {test_name} Test ---")
            try:
                success, logs = test_func()
                all_logs.extend(logs)
                if not success:
                    overall_success = False
                    all_logs.append(f"Result: FAILED")
                else:
                    all_logs.append(f"Result: PASSED")
            except Exception as e:
                all_logs.append(f"ERROR during {test_name}: {e}")
                overall_success = False
                all_logs.append(f"Result: FAILED")
            all_logs.append("")

        return overall_success, all_logs