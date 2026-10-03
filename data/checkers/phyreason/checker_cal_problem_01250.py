import math
import sympy as sp
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Safely extracts inputs and results from the solution instance.
        """
        self.solution = solution_instance
        self.results = getattr(solution_instance, 'results', {})

        # Constants from problem statement
        self.M = getattr(solution_instance, 'M', 0.5)  # kg
        self.R = getattr(solution_instance, 'R', 0.8)  # m
        self.mu = getattr(solution_instance, 'mu', 0.5)
        self.theta = getattr(solution_instance, 'theta', 37)  # degrees
        self.L = getattr(solution_instance, 'L', 0.9)  # m
        self.v0 = getattr(solution_instance, 'v0', 6.0)  # m/s
        self.g = 10.0  # m/s², inferred from steps

        # Trigonometric values
        self.sin37 = 0.6
        self.cos37 = 0.8
        self.tan37 = self.sin37 / self.cos37  # 0.75

        # Extracted results with safe fallbacks
        self.v_B = self._safe_get('v_B')
        self.v_D = self._safe_get('v_D')
        self.v_y = self._safe_get('v_y')
        self.t = self._safe_get('t')
        self.a = self._safe_get('a')
        self.s = self._safe_get('s')
        self.v_all = self._safe_get('v_all')
        self.v_plane = self._safe_get('v_plane')
        self.v_v = self._safe_get('v_v')
        self.a_prime = self._safe_get('a\'') or self._safe_get('a_prime')
        self.N = self._safe_get('N')

        # For collision time Δt, not given, but derived from a' and v_v
        if self.a_prime is not None and self.v_v is not None and self.a_prime > 0:
            self.dt = self.v_v / self.a_prime  # implied collision duration
        else:
            self.dt = None

    def _safe_get(self, key: str) -> float:
        """Safely retrieve a result value, defaulting to None if missing."""
        val = getattr(self.solution, key, self.results.get(key, None))
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Robust numerical comparison with tolerances."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self) -> Tuple[bool, List[str]]:
        """Check fundamental physics laws and relationships."""
        logs = []
        success = True

        # 1. Centripetal force at point B: F_B = 3Mg, and F_B - Mg = M v_B² / R
        if self.v_B is not None:
            expected_vB_sq = 2 * self.g * self.R  # from 3Mg - Mg = M v_B² / R
            if not self._is_close(self.v_B**2, expected_vB_sq):
                logs.append(f"FAIL: v_B² = {self.v_B**2} ≠ 2gR = {expected_vB_sq}")
                success = False
            else:
                logs.append("PASS: Centripetal condition at B holds.")
        else:
            logs.append("SKIP: v_B missing.")

        # 2. Acceleration on conveyor belt: a = μ g
        if self.a is not None:
            expected_a = self.mu * self.g
            if not self._is_close(self.a, expected_a):
                logs.append(f"FAIL: a = {self.a} ≠ μ g = {expected_a}")
                success = False
            else:
                logs.append("PASS: Acceleration on belt matches μ g.")
        else:
            logs.append("SKIP: a missing.")

        # 3. Kinematics on belt: v_D² = v_B² + 2 a L
        if all(x is not None for x in [self.v_D, self.v_B, self.a]):
            expected_vD_sq = self.v_B**2 + 2 * self.a * self.L
            if not self._is_close(self.v_D**2, expected_vD_sq):
                logs.append(f"FAIL: v_D² = {self.v_D**2} ≠ v_B² + 2aL = {expected_vD_sq}")
                success = False
            else:
                logs.append("PASS: Kinematic relation on belt holds.")
        else:
            logs.append("SKIP: v_D, v_B, or a missing.")

        # 4. Projectile motion: t = (2 v_D tan37) / g
        if all(x is not None for x in [self.t, self.v_D]):
            expected_t = (2 * self.v_D * self.tan37) / self.g
            if not self._is_close(self.t, expected_t):
                logs.append(f"FAIL: t = {self.t} ≠ (2 v_D tan37)/g = {expected_t}")
                success = False
            else:
                logs.append("PASS: Projectile time of flight correct.")
        else:
            logs.append("SKIP: t or v_D missing.")

        # 5. Vertical velocity at impact: v_y = g t
        if all(x is not None for x in [self.v_y, self.t]):
            expected_vy = self.g * self.t
            if not self._is_close(self.v_y, expected_vy):
                logs.append(f"FAIL: v_y = {self.v_y} ≠ g t = {expected_vy}")
                success = False
            else:
                logs.append("PASS: Vertical velocity matches g t.")
        else:
            logs.append("SKIP: v_y or t missing.")

        # 6. Resultant velocity magnitude: v_all = √(v_D² + v_y²)
        if all(x is not None for x in [self.v_all, self.v_D, self.v_y]):
            expected_vall = math.sqrt(self.v_D**2 + self.v_y**2)
            if not self._is_close(self.v_all, expected_vall):
                logs.append(f"FAIL: v_all = {self.v_all} ≠ √(v_D²+v_y²) = {expected_vall}")
                success = False
            else:
                logs.append("PASS: Resultant velocity magnitude correct.")
        else:
            logs.append("SKIP: v_all, v_D, or v_y missing.")

        # 7. Velocity along the plane after collision: v_plane = v_D cos37 + v_y sin37
        if all(x is not None for x in [self.v_plane, self.v_D, self.v_y]):
            expected_vplane = self.v_D * self.cos37 + self.v_y * self.sin37
            if not self._is_close(self.v_plane, expected_vplane):
                logs.append(f"FAIL: v_plane = {self.v_plane} ≠ v_D cos37 + v_y sin37 = {expected_vplane}")
                success = False
            else:
                logs.append("PASS: Velocity along plane after collision correct.")
        else:
            logs.append("SKIP: v_plane, v_D, or v_y missing.")

        # 8. Normal velocity component before collision: v_v = v_y cos37 - v_D sin37
        if all(x is not None for x in [self.v_v, self.v_D, self.v_y]):
            expected_vv = self.v_y * self.cos37 - self.v_D * self.sin37
            if not self._is_close(self.v_v, expected_vv):
                logs.append(f"FAIL: v_v = {self.v_v} ≠ v_y cos37 - v_D sin37 = {expected_vv}")
                success = False
            else:
                logs.append("PASS: Normal velocity component correct.")
        else:
            logs.append("SKIP: v_v, v_D, or v_y missing.")

        # 9. Normal force during collision: N = M g cos37 + M a'
        if all(x is not None for x in [self.N, self.a_prime]):
            expected_N = self.M * self.g * self.cos37 + self.M * self.a_prime
            if not self._is_close(self.N, expected_N):
                logs.append(f"FAIL: N = {self.N} ≠ M g cos37 + M a' = {expected_N}")
                success = False
            else:
                logs.append("PASS: Normal force relation holds.")
        else:
            logs.append("SKIP: N or a' missing.")

        return success, logs

    def test_trace_consistency(self) -> Tuple[bool, List[str]]:
        """Verify that intermediate results are consistent with final answers."""
        logs = []
        success = True

        # Target answers from problem (numerical approximations)
        targets = {
            'v_B': 4.0,
            'v_all': (5 * math.sqrt(13)) / 2,  # ≈ 9.014
            'v_plane': 8.5,
            'N': 7.0
        }

        for key, target in targets.items():
            val = getattr(self, key, None)
            if val is not None:
                if not self._is_close(val, target):
                    logs.append(f"FAIL: {key} = {val} ≠ target {target}")
                    success = False
                else:
                    logs.append(f"PASS: {key} matches target.")
            else:
                logs.append(f"SKIP: {key} missing.")

        # Additional internal consistency: s = (v0² - v_B²) / (2a)
        if all(x is not None for x in [self.s, self.v_B, self.a]):
            expected_s = (self.v0**2 - self.v_B**2) / (2 * self.a)
            if not self._is_close(self.s, expected_s):
                logs.append(f"FAIL: s = {self.s} ≠ (v0² - v_B²)/(2a) = {expected_s}")
                success = False
            else:
                logs.append("PASS: Distance to reach belt speed consistent.")
        else:
            logs.append("SKIP: s, v_B, or a missing.")

        # Check that s > L (object does not reach belt speed before D)
        if self.s is not None and self.L is not None:
            if self.s <= self.L:
                logs.append(f"WARNING: s = {self.s} ≤ L = {self.L}, but should be > L.")
            else:
                logs.append("PASS: s > L, object doesn't reach v0 on belt.")
        else:
            logs.append("SKIP: s or L missing.")

        return success, logs

    def test_monotonicity(self) -> Tuple[bool, List[str]]:
        """
        Perturb an input (coefficient of friction μ) and verify that 
        v_D changes in the expected direction. Restores original μ.
        """
        logs = []
        success = True

        if self.v_D is None or self.a is None:
            logs.append("SKIP: v_D or a missing for monotonicity test.")
            return False, logs

        original_mu = self.mu
        perturbed_mu = original_mu * 1.1  # Increase μ by 10%
        perturbed_a = perturbed_mu * self.g

        # Recompute v_D² = v_B² + 2 * a * L (assuming v_B unchanged)
        if self.v_B is not None:
            perturbed_vD_sq = self.v_B**2 + 2 * perturbed_a * self.L
            perturbed_vD = math.sqrt(max(0, perturbed_vD_sq))

            if perturbed_vD > self.v_D:
                logs.append(f"PASS: Increasing μ from {original_mu} to {perturbed_mu} increases v_D "
                            f"({self.v_D} -> {perturbed_vD}).")
            else:
                logs.append(f"FAIL: Increasing μ did not increase v_D as expected.")
                success = False
        else:
            logs.append("SKIP: v_B missing for monotonicity test.")
            success = False

        # Restore original μ (though we only used a local copy)
        self.mu = original_mu
        return success, logs

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all checks and return overall success and logs."""
        all_logs = ["=== PhysicsChecker Started ==="]
        overall_success = True

        # Test 1: Invariants
        success, logs = self.test_invariants()
        all_logs.append("\n--- Invariants Test ---")
        all_logs.extend(logs)
        overall_success = overall_success and success

        # Test 2: Trace Consistency
        success, logs = self.test_trace_consistency()
        all_logs.append("\n--- Trace Consistency Test ---")
        all_logs.extend(logs)
        overall_success = overall_success and success

        # Test 3: Monotonicity
        success, logs = self.test_monotonicity()
        all_logs.append("\n--- Monotonicity Test ---")
        all_logs.extend(logs)
        overall_success = overall_success and success

        all_logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, all_logs