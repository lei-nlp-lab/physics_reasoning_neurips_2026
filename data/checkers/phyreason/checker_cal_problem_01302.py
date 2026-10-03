import math
import sympy as sp
from typing import Tuple, List, Any, Dict

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Safely extract inputs and results from the solution instance.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Inputs with fallback to problem defaults
        self.m = self._safe_get('m', 2.0)               # kg
        self.k1 = self._safe_get('k1', 200.0)          # N/m
        self.ab = self._safe_get('ab', 0.6)            # m
        self.g = self._safe_get('g', 10.0)             # m/s^2

        # Results (using keys from mapping)
        self.x1 = self._safe_get('x_1')
        self.x2 = self._safe_get('x_2')
        self.h = self._safe_get('h')
        self.x3 = self._safe_get('x_3')
        self.k2 = self._safe_get('k_2')

        # Derived quantities
        self.mg = self.m * self.g

    def _safe_get(self, key: str, default: float = None) -> float:
        """Safely retrieve attribute from instance or results dict."""
        val = getattr(self.solution, key, None)
        if val is None:
            val = self.results.get(key, default)
        return float(val) if val is not None else None

    def _is_close(self, a: float, b: float) -> bool:
        """Fuzzy comparison with 2% tolerance."""
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Check fundamental physics laws: Hooke's law, equilibrium, geometry."""
        logs = []
        success = True

        # 1. Initial deformation of spring B (Hooke's law)
        if self.x1 is not None:
            expected_x1 = self.mg / self.k1
            if not self._is_close(self.x1, expected_x1):
                logs.append(f"FAIL: x1 = {self.x1} ≠ mg/k1 = {expected_x1}")
                success = False
            else:
                logs.append(f"PASS: x1 = mg/k1 = {expected_x1}")
        else:
            logs.append("MISSING: x1 not provided")
            success = False

        # 2. Final deformation of spring B (Hooke's law, same magnitude)
        if self.x2 is not None:
            expected_x2 = self.mg / self.k1
            if not self._is_close(self.x2, expected_x2):
                logs.append(f"FAIL: x2 = {self.x2} ≠ mg/k1 = {expected_x2}")
                success = False
            else:
                logs.append(f"PASS: x2 = mg/k1 = {expected_x2}")
        else:
            logs.append("MISSING: x2 not provided")
            success = False

        # 3. Height risen (geometry)
        if all(v is not None for v in [self.x1, self.x2, self.h]):
            expected_h = self.x1 + self.x2
            if not self._is_close(self.h, expected_h):
                logs.append(f"FAIL: h = {self.h} ≠ x1 + x2 = {expected_h}")
                success = False
            else:
                logs.append(f"PASS: h = x1 + x2 = {expected_h}")
        else:
            logs.append("MISSING: x1, x2, or h not provided for geometry check")
            success = False

        # 4. Elongation of spring C (geometry)
        if all(v is not None for v in [self.x1, self.x2, self.x3]):
            expected_x3 = self.ab - self.x1 - self.x2
            if not self._is_close(self.x3, expected_x3):
                logs.append(f"FAIL: x3 = {self.x3} ≠ ab - x1 - x2 = {expected_x3}")
                success = False
            else:
                logs.append(f"PASS: x3 = ab - x1 - x2 = {expected_x3}")
        else:
            logs.append("MISSING: x1, x2, or x3 not provided for spring C elongation")
            success = False

        # 5. Force equilibrium on A (Newton's 2nd law, static)
        if all(v is not None for v in [self.x2, self.x3, self.k2]):
            # Final balance: k2*x3 = mg + k1*x2 (from steps)
            lhs = self.k2 * self.x3
            rhs = self.mg + self.k1 * self.x2
            if not self._is_close(lhs, rhs):
                logs.append(f"FAIL: k2*x3 = {lhs} ≠ mg + k1*x2 = {rhs}")
                success = False
            else:
                logs.append(f"PASS: Force equilibrium k2*x3 = mg + k1*x2 holds")
        else:
            logs.append("MISSING: x2, x3, or k2 not provided for force balance")
            success = False

        return success, logs

    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Cross-check that intermediate results lead to the final k2."""
        logs = []
        success = True

        # Recompute k2 from x2, x3, and inputs
        if all(v is not None for v in [self.x2, self.x3]):
            computed_k2 = (self.mg + self.k1 * self.x2) / self.x3
            if self.k2 is not None:
                if not self._is_close(self.k2, computed_k2):
                    logs.append(f"FAIL: Reported k2 = {self.k2} ≠ computed (mg + k1*x2)/x3 = {computed_k2}")
                    success = False
                else:
                    logs.append(f"PASS: k2 consistent with intermediates")
            else:
                logs.append("MISSING: k2 not provided")
                success = False
        else:
            logs.append("MISSING: x2 or x3 not provided for k2 check")
            success = False

        # Check that x1 equals x2 (both = mg/k1)
        if self.x1 is not None and self.x2 is not None:
            if not self._is_close(self.x1, self.x2):
                logs.append(f"FAIL: x1 = {self.x1} ≠ x2 = {self.x2} (both should equal mg/k1)")
                success = False
            else:
                logs.append(f"PASS: x1 = x2 = {self.x1}")

        return success, logs

    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Perturb mass m and verify that k2 changes in the expected direction.
        Does not modify the original solution instance.
        """
        logs = []
        success = True

        # Only run if all needed values are present
        if None in [self.x1, self.x2, self.x3, self.k2]:
            logs.append("MISSING: One or more results missing; skipping monotonicity test")
            return False, logs

        # Original k2 from results
        k2_original = self.k2

        # Perturb mass by +10%
        m_perturbed = self.m * 1.1
        # Ensure perturbation doesn't make x3 negative: ab - 2*m_perturbed*g/k1 > 0
        if self.ab - 2 * m_perturbed * self.g / self.k1 <= 0:
            logs.append("SKIP: Perturbed mass too large, would make x3 ≤ 0")
            return True, logs  # Not a failure, just skip

        # Compute new values using the physics formulas
        x2_new = m_perturbed * self.g / self.k1
        x3_new = self.ab - 2 * x2_new  # because x1_new = x2_new
        k2_new = 2 * m_perturbed * self.g / x3_new

        # Check monotonicity: increasing m should increase k2
        if k2_new <= k2_original:
            logs.append(f"FAIL: Increasing m from {self.m} to {m_perturbed} did not increase k2 ({k2_original} -> {k2_new})")
            success = False
        else:
            logs.append(f"PASS: k2 increases with mass (perturbation test)")

        # Also test a negative perturbation (-5%)
        m_perturbed2 = self.m * 0.95
        x2_new2 = m_perturbed2 * self.g / self.k1
        x3_new2 = self.ab - 2 * x2_new2
        k2_new2 = 2 * m_perturbed2 * self.g / x3_new2

        if k2_new2 >= k2_original:
            logs.append(f"FAIL: Decreasing m from {self.m} to {m_perturbed2} did not decrease k2 ({k2_original} -> {k2_new2})")
            success = False
        else:
            logs.append(f"PASS: k2 decreases with mass (negative perturbation test)")

        return success, logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all audit tests and return overall success and logs."""
        all_logs = []
        overall_success = True

        # Test 1: Invariants
        success, logs = self.test_invariants()
        all_logs.append("=== Invariants Test ===")
        all_logs.extend(logs)
        overall_success = overall_success and success

        # Test 2: Trace Consistency
        success, logs = self.test_trace_consistency()
        all_logs.append("=== Trace Consistency Test ===")
        all_logs.extend(logs)
        overall_success = overall_success and success

        # Test 3: Monotonicity
        success, logs = self.test_monotonicity()
        all_logs.append("=== Monotonicity Test ===")
        all_logs.extend(logs)
        overall_success = overall_success and success

        return overall_success, all_logs