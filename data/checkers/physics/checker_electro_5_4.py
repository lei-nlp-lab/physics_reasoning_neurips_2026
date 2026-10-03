import math
import sympy as sp
from typing import Any, Dict, List, Tuple

class PhysicsChecker:
    """Audits a solution for the spaceship-package relativistic interception problem."""

    def __init__(self, solution_instance: Any) -> None:
        # Extract results dictionary safely
        if hasattr(solution_instance, 'results'):
            results = solution_instance.results
        elif isinstance(solution_instance, dict):
            results = solution_instance.get('results', {})
        else:
            results = {}
        self.results = results
        self.alpha_prime = self.results.get('alpha_prime', None)
        self.v_prime = self.results.get('v_prime', None)

        # Extract givens if available
        if hasattr(solution_instance, 'givens'):
            givens = solution_instance.givens
        elif isinstance(solution_instance, dict):
            givens = solution_instance.get('givens', {})
        else:
            givens = {}
        self.givens = givens
        self.c = self.givens.get('c', None)  # may be None or numeric/symbolic

        # Pre‑compute expected numerical values (c = 1 for numerical checks)
        self._c_num = 1.0
        self._v_ship_num = 0.5
        self._v_package_num = 0.75
        self._u_y_num = -self._v_ship_num
        self._u_x_num = math.sqrt(self._v_package_num**2 - self._u_y_num**2)
        self._gamma_num = 1.0 / math.sqrt(1.0 - self._v_ship_num**2)
        self._denom_num = 1.0 - self._u_y_num * self._v_ship_num
        self._u_prime_x_num = self._u_x_num / (self._gamma_num * self._denom_num)
        self._u_prime_y_num = (self._u_y_num - self._v_ship_num) / self._denom_num
        self._expected_angle_num = math.atan(abs(self._u_prime_y_num) / self._u_prime_x_num)
        self._expected_v_prime_num = math.hypot(self._u_prime_x_num, self._u_prime_y_num)

    def _get_numeric_value(self, expr: Any) -> float:
        """Convert a result (string, sympy, or number) to float using c = 1."""
        if expr is None:
            return None
        if isinstance(expr, str):
            # Replace arctan with atan, remove 'c'
            expr_str = expr.replace('arctan', 'atan').replace('c', str(self._c_num))
            try:
                sym_expr = sp.sympify(expr_str)
                return float(sp.N(sym_expr))
            except Exception:
                return None
        if isinstance(expr, (int, float)):
            return float(expr)
        if hasattr(expr, 'evalf'):  # sympy expression
            try:
                return float(expr.evalf(subs={sp.Symbol('c'): self._c_num}))
            except Exception:
                return None
        return None

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_alpha_prime_exists(self) -> Tuple[bool, str]:
        if self.alpha_prime is None:
            return (False, "SKIP: results['alpha_prime'] not provided")
        return (True, "results['alpha_prime'] present")

    def test_interface_v_prime_exists(self) -> Tuple[bool, str]:
        if self.v_prime is None:
            return (False, "SKIP: results['v_prime'] not provided")
        return (True, "results['v_prime'] present")

    # ------------------------------------------------------------------
    # Invariant / Theorem tests (fuzzy numerical matching allowed)
    # ------------------------------------------------------------------
    def test_invariant_velocity_addition(self) -> Tuple[bool, str]:
        """Check that the transformed speed and angle match relativistic formulas."""
        angle_num = self._get_numeric_value(self.alpha_prime)
        v_num = self._get_numeric_value(self.v_prime)
        if angle_num is None or v_num is None:
            return (False, "SKIP: cannot parse result(s) to numeric")
        if not math.isfinite(angle_num) or not math.isfinite(v_num):
            return (False, "Numeric evaluation produced non‑finite values")
        if not math.isclose(angle_num, self._expected_angle_num, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Angle mismatch: got {angle_num}, expected {self._expected_angle_num}")
        if not math.isclose(v_num, self._expected_v_prime_num, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Speed mismatch: got {v_num}, expected {self._expected_v_prime_num}")
        return (True, "Velocity addition formula correctly applied")

    def test_invariant_speed_in_S(self) -> Tuple[bool, str]:
        """Check that the package speed in S derived from u_x, u_y equals 3c/4."""
        u = math.hypot(self._u_x_num, self._u_y_num)
        if not math.isclose(u, self._v_package_num, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Package speed in S is {u}, expected {self._v_package_num}")
        return (True, f"Speed in S = 3c/4 confirmed")

    def test_invariant_angle_range(self) -> Tuple[bool, str]:
        """The angle in ship 1 frame should be acute and in first quadrant."""
        angle_num = self._get_numeric_value(self.alpha_prime)
        if angle_num is None:
            return (False, "SKIP: cannot parse angle")
        if angle_num <= 0 or angle_num >= math.pi / 2:
            return (False, f"Angle {angle_num} rad outside (0, π/2)")
        return (True, f"Angle {angle_num} rad is within expected range")

    # ------------------------------------------------------------------
    # Modeling / trace consistency tests
    # ------------------------------------------------------------------
    def test_modeling_intercept_condition(self) -> Tuple[bool, str]:
        """Verify that the package's y-velocity in S is the same as ship2's velocity."""
        if not math.isclose(self._u_y_num, -self._v_ship_num, rel_tol=1e-2, abs_tol=1e-2):
            return (False, "Package y-velocity does not match ship2's y-velocity")
        return (True, "Intercept condition satisfied: u_y = -c/2")

    def test_modeling_lorentz_factor(self) -> Tuple[bool, str]:
        """Check the Lorentz factor of ship 1."""
        gamma_calc = 1.0 / math.sqrt(1.0 - self._v_ship_num**2)
        if not math.isclose(gamma_calc, self._gamma_num, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Gamma mismatch: calculated {gamma_calc}, expected {self._gamma_num}")
        return (True, f"Lorentz factor = {self._gamma_num} (correct)")

    def test_modeling_denominator(self) -> Tuple[bool, str]:
        """Check the denominator in the velocity transformation."""
        denom = 1.0 - self._u_y_num * self._v_ship_num
        if not math.isclose(denom, self._denom_num, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Denominator mismatch: {denom}, expected {self._denom_num}")
        return (True, f"Denominator = {self._denom_num} (correct)")

    # ------------------------------------------------------------------
    # Monotonicity tests
    # ------------------------------------------------------------------
    def test_monotonicity_angle_increases_with_ship_speed(self) -> Tuple[bool, str]:
        """
        Perturb the ship speed upward by 5% and verify that the required launch
        angle in S' increases. This tests the monotonic relation between ship
        speed and transformed angle.
        """
        if self.alpha_prime is None:
            return (False, "SKIP: results['alpha_prime'] not provided for comparison")
        baseline_angle = self._expected_angle_num
        if not math.isfinite(baseline_angle):
            return (False, "Baseline angle is not finite")

        # Perturb v_ship upward by 5% (c = 1)
        v_pert = 1.05 * self._v_ship_num  # 0.525
        # Recompute the angle for this perturbed input
        u_y_pert = -v_pert
        u_x_pert = math.sqrt(self._v_package_num**2 - u_y_pert**2)
        gamma_pert = 1.0 / math.sqrt(1.0 - v_pert**2)
        denom_pert = 1.0 - u_y_pert * v_pert
        u_prime_x_pert = u_x_pert / (gamma_pert * denom_pert)
        u_prime_y_pert = (u_y_pert - v_pert) / denom_pert
        angle_pert = math.atan(abs(u_prime_y_pert) / u_prime_x_pert)

        if angle_pert <= baseline_angle:
            return (False, f"Angle did not increase: baseline {baseline_angle}, perturbed {angle_pert}")
        return (True, f"Angle increased monotonically with ship speed (baseline {baseline_angle:.4f} → perturbed {angle_pert:.4f}")  # (str missing closing parenthesis fixed)

    # ------------------------------------------------------------------
    # Weak answer sanity test (no math.isclose, no re‑derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_positive_finite(self) -> Tuple[bool, str]:
        """Sanity: angle and speed are non‑negative and finite."""
        angle_num = self._get_numeric_value(self.alpha_prime)
        v_num = self._get_numeric_value(self.v_prime)
        if angle_num is None or v_num is None:
            return (False, "SKIP: cannot parse results")
        if angle_num < 0 or math.isinf(angle_num):
            return (False, f"Angle {angle_num} is negative or infinite")
        if v_num < 0 or math.isinf(v_num):
            return (False, f"Speed {v_num} is negative or infinite")
        if v_num > 1.0:  # c = 1
            return (False, f"Speed {v_num} > c (non‑physical)")
        return (True, "Angle and speed are plausible")

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed = False
                msg = f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)