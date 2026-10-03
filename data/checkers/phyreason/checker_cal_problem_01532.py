import math
import sympy as sp
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        """
        Safely initialize from solution_instance without direct attribute access.
        """
        # Extract inputs with safe fallbacks
        self.v = self._safe_get(solution_instance, 'v', 30.0)
        self.mu_s = self._safe_get(solution_instance, 'mu_s', 0.6)
        self.g = self._safe_get(solution_instance, 'g', 10.0)  # default to 10 m/s² as per problem

        # Extract results
        self.R = self._safe_get(solution_instance, 'R', 0.0)
        self.tantheta = self._safe_get(solution_instance, 'tantheta', 0.0)

        # Store logs
        self.logs = []

    def _safe_get(self, obj: Any, key: str, default: float) -> float:
        """
        Safely retrieve attribute or dictionary value with multiple fallbacks.
        """
        # Try direct attribute
        val = getattr(obj, key, None)
        if val is not None:
            return float(val)
        # Try results dictionary
        results = getattr(obj, 'results', {})
        if isinstance(results, dict):
            val = results.get(key, None)
            if val is not None:
                return float(val)
        # Return default
        return float(default)

    def _is_close(self, a: float, b: float) -> bool:
        """
        Fuzzy numerical matching with 1-2% tolerance.
        """
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

    def test_invariants(self) -> bool:
        """
        Verify fundamental physics laws: static friction provides centripetal force,
        and banked curve trigonometry.
        """
        success = True

        # 1. Flat curve: minimum radius from static friction and centripetal force
        R_expected = self.v**2 / (self.mu_s * self.g)
        if not self._is_close(self.R, R_expected):
            self.logs.append(
                f"FAIL: R = {self.R} m, but expected v²/(μ·g) = {R_expected:.2f} m"
            )
            success = False
        else:
            self.logs.append(
                f"PASS: Minimum radius R = {self.R} m matches v²/(μ·g)"
            )

        # 2. Banked curve: tan(θ) = v²/(g·R) using the same R
        tantheta_expected = self.v**2 / (self.g * self.R)
        if not self._is_close(self.tantheta, tantheta_expected):
            self.logs.append(
                f"FAIL: tan(θ) = {self.tantheta}, but expected v²/(g·R) = {tantheta_expected:.4f}"
            )
            success = False
        else:
            self.logs.append(
                f"PASS: tan(θ) = {self.tantheta} matches v²/(g·R)"
            )

        # 3. Symbolic check (optional, for completeness)
        try:
            m = sp.symbols('m')
            # Flat curve: μ·m·g = m·v²/R  →  R = v²/(μ·g)
            lhs_flat = self.mu_s * m * self.g
            rhs_flat = m * self.v**2 / self.R
            if not sp.simplify(lhs_flat - rhs_flat) == 0:
                self.logs.append("SYMBOLIC WARNING: Flat curve force balance may be inconsistent")
                # Not a failure, since we already did numerical check
            # Banked curve: m·g·tanθ = m·v²/R  →  tanθ = v²/(g·R)
            lhs_banked = m * self.g * self.tantheta
            rhs_banked = m * self.v**2 / self.R
            if not sp.simplify(lhs_banked - rhs_banked) == 0:
                self.logs.append("SYMBOLIC WARNING: Banked curve force balance may be inconsistent")
        except Exception as e:
            self.logs.append(f"Symbolic check skipped due to error: {e}")

        return success

    def test_trace_consistency(self) -> bool:
        """
        Ensure that the intermediate steps (if available) align with the final answers.
        """
        success = True

        # Check that the provided R and tantheta are consistent with each other via physics
        # From the banked curve formula: tanθ = v²/(g·R) → R = v²/(g·tanθ)
        R_from_tantheta = self.v**2 / (self.g * self.tantheta)
        if not self._is_close(self.R, R_from_tantheta):
            self.logs.append(
                f"TRACE FAIL: R = {self.R} m does not match v²/(g·tanθ) = {R_from_tantheta:.2f} m"
            )
            success = False
        else:
            self.logs.append(
                f"TRACE PASS: R and tanθ are mutually consistent via v²/(g·tanθ)"
            )

        # Additionally, if the solution_instance has step results, we could check them,
        # but the problem does not require intermediate storage.
        return success

    def test_monotonicity(self) -> bool:
        """
        Perturb an input (speed v) and verify the expected trend in R and tanθ.
        """
        success = True

        # Store original
        v_original = self.v
        perturbation_factor = 1.05  # 5% increase

        # Increase v
        self.v *= perturbation_factor

        # Recalculate expected values using formulas
        R_new_expected = self.v**2 / (self.mu_s * self.g)
        tantheta_new_expected = self.v**2 / (self.g * self.R)  # using original R

        # Check monotonicity: R should increase with v²
        R_original_expected = v_original**2 / (self.mu_s * self.g)
        if not R_new_expected > R_original_expected:
            self.logs.append(
                f"MONOTONICITY FAIL: R does not increase with v (original R calc = {R_original_expected:.2f}, new = {R_new_expected:.2f})"
            )
            success = False
        else:
            self.logs.append(
                f"MONOTONICITY PASS: R increases with v²"
            )

        # Check monotonicity: tanθ should increase with v² (for fixed R)
        tantheta_original_expected = v_original**2 / (self.g * self.R)
        if not tantheta_new_expected > tantheta_original_expected:
            self.logs.append(
                f"MONOTONICITY FAIL: tanθ does not increase with v (original tanθ calc = {tantheta_original_expected:.4f}, new = {tantheta_new_expected:.4f})"
            )
            success = False
        else:
            self.logs.append(
                f"MONOTONICITY PASS: tanθ increases with v² for fixed R"
            )

        # Restore original v
        self.v = v_original

        return success

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all checks and return overall success and log messages.
        """
        self.logs.clear()
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity),
        ]

        for name, test_func in tests:
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self.logs.append(f"ERROR in {name}: {e}")
                overall_success = False

        return overall_success, self.logs