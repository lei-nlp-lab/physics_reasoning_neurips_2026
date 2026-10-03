import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.v0 = self._safe_get(solution_instance, 'v_0')
        self.v1 = self._safe_get(solution_instance, 'v_1')
        self.v2 = self._safe_get(solution_instance, 'v_2')
        self.v3 = self._safe_get(solution_instance, 'v_3')
        self.x = self._safe_get(solution_instance, 'x')

        # Constants from the problem statement
        self.m = 1.0  # kg
        self.g = 10.0  # m/s^2
        self.theta = 37.0  # degrees
        self.sin_theta = 0.6
        self.cos_theta = 0.8
        self.mu = 0.5
        self.R = 1.0  # m
        self.L = 3.0  # m

    def _safe_get(self, obj, key):
        """Safely retrieve attribute or dictionary key."""
        val = getattr(obj, key, None)
        if val is None:
            val = getattr(obj, 'results', {}).get(key, 0.0)
        return float(val) if val is not None else 0.0

    def test_invariants(self) -> List[str]:
        """Check fundamental physics laws."""
        logs = []
        # 1. v1^2 = 4 * mu * g * L * cos(theta)  (from step2)
        lhs1 = self.v1**2
        rhs1 = 4 * self.mu * self.g * self.L * self.cos_theta
        if not math.isclose(lhs1, rhs1, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: v1^2 = {lhs1:.3f} ≠ 4μ g L cosθ = {rhs1:.3f}")
        else:
            logs.append(f"PASS: v1^2 = 4μ g L cosθ within 1%")

        # 2. v2 = sqrt(g * R)  (from step3)
        lhs2 = self.v2
        rhs2 = math.sqrt(self.g * self.R)
        if not math.isclose(lhs2, rhs2, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: v2 = {lhs2:.3f} ≠ √(gR) = {rhs2:.3f}")
        else:
            logs.append(f"PASS: v2 = √(gR) within 1%")

        # 3. g*(R - cosθ) = 0.5*(v3^2 - v2^2)  (from step4)
        lhs3 = self.g * (self.R - self.cos_theta)
        rhs3 = 0.5 * (self.v3**2 - self.v2**2)
        if not math.isclose(lhs3, rhs3, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: g(R-cosθ) = {lhs3:.3f} ≠ 0.5(v3^2 - v2^2) = {rhs3:.3f}")
        else:
            logs.append(f"PASS: Energy balance circular→incline within 1%")

        # 4. m g L sinθ + 0.5 m v3^2 = μ m g x cosθ  (from step7)
        lhs4 = self.m * self.g * self.L * self.sin_theta + 0.5 * self.m * self.v3**2
        rhs4 = self.mu * self.m * self.g * self.x * self.cos_theta
        if not math.isclose(lhs4, rhs4, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: m g L sinθ + 0.5 m v3^2 = {lhs4:.3f} ≠ μ m g x cosθ = {rhs4:.3f}")
        else:
            logs.append(f"PASS: Work-energy after rebound within 1%")

        return logs

    def test_trace_consistency(self) -> List[str]:
        """Cross-check intermediate results with final answer."""
        logs = []
        # Sanity checks on ordering (based on physics intuition)
        if self.v0 <= 0 or self.v1 <= 0 or self.v2 <= 0 or self.v3 <= 0 or self.x <= 0:
            logs.append("WARNING: Non-positive speed or distance.")
        if self.v0 < self.v1:
            logs.append(f"NOTE: v0 ({self.v0:.3f}) < v1 ({self.v1:.3f}) – potential energy gain?")
        if self.v1 < self.v3:
            logs.append(f"NOTE: v1 ({self.v1:.3f}) < v3 ({self.v3:.3f}) – check energy flow.")
        if self.v3 < self.v2:
            logs.append(f"NOTE: v3 ({self.v3:.3f}) < v2 ({self.v2:.3f}) – unusual.")
        if self.x < self.L:
            logs.append(f"NOTE: x ({self.x:.3f}) < L ({self.L:.3f}) – less than incline length.")

        # Numerical consistency with expected target answers (optional)
        target_v0 = 8.0
        target_v1 = 4 * math.sqrt(3)  # ≈ 6.928
        target_x = 6.25
        if math.isclose(self.v0, target_v0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"PASS: v0 matches target {target_v0} m/s")
        else:
            logs.append(f"NOTE: v0 = {self.v0:.3f} vs target {target_v0} m/s")

        if math.isclose(self.v1, target_v1, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"PASS: v1 matches target {target_v1:.3f} m/s")
        else:
            logs.append(f"NOTE: v1 = {self.v1:.3f} vs target {target_v1:.3f} m/s")

        if math.isclose(self.x, target_x, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"PASS: x matches target {target_x} m")
        else:
            logs.append(f"NOTE: x = {self.x:.3f} vs target {target_x} m")

        return logs

    def test_monotonicity(self) -> List[str]:
        """Perturb μ and verify expected trends."""
        logs = []
        original_mu = self.mu
        perturbed_mu = original_mu * 1.01  # +1% perturbation

        # Recompute v1 and x using formulas from steps
        v1_new = math.sqrt(4 * perturbed_mu * self.g * self.L * self.cos_theta)
        x_new = (self.g * self.L * self.sin_theta + 0.5 * self.v3**2) / (perturbed_mu * self.g * self.cos_theta)

        # Check trends
        if v1_new > self.v1:
            logs.append(f"PASS: v1 increases with μ (from {self.v1:.3f} to {v1_new:.3f})")
        else:
            logs.append(f"FAIL: v1 does not increase with μ")

        if x_new < self.x:
            logs.append(f"PASS: x decreases with μ (from {self.x:.3f} to {x_new:.3f})")
        else:
            logs.append(f"FAIL: x does not decrease with μ")

        # Restore μ (not strictly needed as we used local copy)
        return logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all tests and return overall success and logs."""
        logs = []
        logs.append("=== PhysicsChecker Report ===")

        # Run tests
        inv_logs = self.test_invariants()
        trace_logs = self.test_trace_consistency()
        mono_logs = self.test_monotonicity()

        logs.extend(["--- Invariants ---"] + inv_logs)
        logs.extend(["--- Trace Consistency ---"] + trace_logs)
        logs.extend(["--- Monotonicity ---"] + mono_logs)

        # Determine overall success: no "FAIL" in logs
        success = not any("FAIL" in log for log in logs)
        logs.append(f"=== Overall: {'PASS' if success else 'FAIL'} ===")
        return success, logs