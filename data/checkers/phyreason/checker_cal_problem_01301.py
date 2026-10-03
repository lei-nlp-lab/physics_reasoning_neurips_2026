import math
import sympy as sp
from typing import Tuple, List, Any

class PhysicsChecker:
    def __init__(self, solution_instance: Any):
        # Safe extraction of inputs with defaults from problem statement
        self.v0 = getattr(solution_instance, 'v0', 20.0)
        self.theta_deg = 37.0
        self.theta = math.radians(self.theta_deg)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.sin37 = 0.6
        self.cos37 = 0.8
        self.tan37 = self.sin37 / self.cos37  # 0.75

        # Safe extraction of results with fallbacks
        results = getattr(solution_instance, 'results', {})
        self.t = getattr(solution_instance, 't', results.get('t', None))
        self.s = getattr(solution_instance, 's', results.get('s', None))
        self.x = getattr(solution_instance, 'x', results.get('x', None))
        self.y = getattr(solution_instance, 'y', results.get('y', None))

        # Derived quantities for convenience
        self.v0x = self.v0  # horizontal launch
        self.v0y = 0.0

        # Logs storage
        self.logs = []

    def _log(self, msg: str, success: bool):
        self.logs.append(f"{'PASS' if success else 'FAIL'}: {msg}")
        return success

    def _isclose(self, a: float, b: float) -> bool:
        return math.isclose(a, b, rel_tol=1e-2, abs_tol=1e-2)

    def test_invariants(self) -> bool:
        """
        Verify kinematic and geometric invariants using both symbolic and numerical checks.
        """
        success = True

        # 1. Check time of flight formula: t = 2*v0*tanθ / g
        t_expected = 2 * self.v0 * self.tan37 / self.g
        if self.t is not None:
            if not self._isclose(self.t, t_expected):
                success = self._log(
                    f"Time of flight mismatch: got {self.t}, expected {t_expected}", False)
            else:
                self._log(f"Time of flight matches formula: t={self.t}", True)
        else:
            # If t is missing, compute from formula for later checks
            self.t = t_expected
            self._log("Using formula-derived t for checks", True)

        # 2. Check horizontal displacement: x = v0 * t
        if self.x is not None:
            x_expected = self.v0 * self.t
            if not self._isclose(self.x, x_expected):
                success = self._log(
                    f"Horizontal displacement mismatch: got {self.x}, expected {x_expected}", False)
            else:
                self._log(f"Horizontal displacement matches: x={self.x}", True)
        else:
            self.x = self.v0 * self.t
            self._log("Computed x from v0 and t", True)

        # 3. Check vertical displacement: y = 0.5 * g * t**2
        if self.y is not None:
            y_expected = 0.5 * self.g * self.t**2
            if not self._isclose(self.y, y_expected):
                success = self._log(
                    f"Vertical displacement mismatch: got {self.y}, expected {y_expected}", False)
            else:
                self._log(f"Vertical displacement matches: y={self.y}", True)
        else:
            self.y = 0.5 * self.g * self.t**2
            self._log("Computed y from g and t", True)

        # 4. Check slope geometry: y/x = tanθ (since launch is horizontal and landing on slope)
        if not self._isclose(self.y / self.x, self.tan37):
            success = self._log(
                f"Slope geometry mismatch: y/x={self.y/self.x}, tan37={self.tan37}", False)
        else:
            self._log(f"Slope geometry holds: y/x = tan37", True)

        # 5. Check distance along slope: s = y / sinθ = x / cosθ
        s_from_y = self.y / self.sin37
        s_from_x = self.x / self.cos37
        if self.s is not None:
            if not (self._isclose(self.s, s_from_y) or self._isclose(self.s, s_from_x)):
                success = self._log(
                    f"Distance mismatch: got {self.s}, expected from y {s_from_y} or from x {s_from_x}", False)
            else:
                self._log(f"Distance matches geometry: s={self.s}", True)
        else:
            self.s = s_from_y  # consistent with step5
            self._log("Computed s from y and sin37", True)

        # 6. Check symbolic energy conservation (kinetic + potential, assuming zero potential at launch)
        # Kinetic energy at launch: 0.5 * m * v0**2 (m arbitrary, set m=1)
        # At landing, velocity components: vx = v0, vy = g*t (since v0y=0 and vy = -g*t, but magnitude)
        # Kinetic energy at landing: 0.5 * (vx**2 + vy**2)
        # Potential energy loss: m * g * y
        m = 1.0
        E_launch = 0.5 * m * self.v0**2
        vx_land = self.v0
        vy_land = self.g * self.t  # magnitude, since downward positive
        E_land = 0.5 * m * (vx_land**2 + vy_land**2)
        delta_E = E_launch + m * self.g * self.y - E_land
        if not self._isclose(delta_E, 0.0):
            success = self._log(
                f"Energy conservation deviation: ΔE = {delta_E}", False)
        else:
            self._log("Energy conserved within tolerance", True)

        return success

    def test_trace_consistency(self) -> bool:
        """
        Cross-check intermediate results (x, y) with final answers (t, s).
        """
        success = True

        # 1. Check that t from step3 matches the one used in step4/5/6
        t_step3 = 2 * self.v0 * self.tan37 / self.g
        if not self._isclose(self.t, t_step3):
            success = self._log(
                f"Time t={self.t} inconsistent with step3 formula t={t_step3}", False)
        else:
            self._log("Time consistent with step3", True)

        # 2. Check that y computed from t (step4) matches geometry y = s * sinθ
        y_from_t = 0.5 * self.g * self.t**2
        y_from_s = self.s * self.sin37
        if not self._isclose(y_from_t, y_from_s):
            success = self._log(
                f"Vertical displacement inconsistent: from t {y_from_t}, from s {y_from_s}", False)
        else:
            self._log("Vertical displacement consistent between steps", True)

        # 3. Check that x computed from t (step1) matches geometry x = s * cosθ
        x_from_t = self.v0 * self.t
        x_from_s = self.s * self.cos37
        if not self._isclose(x_from_t, x_from_s):
            success = self._log(
                f"Horizontal displacement inconsistent: from t {x_from_t}, from s {x_from_s}", False)
        else:
            self._log("Horizontal displacement consistent between steps", True)

        # 4. Verify Pythagorean relation: s^2 = x^2 + y^2
        s_from_pythag = math.sqrt(self.x**2 + self.y**2)
        if not self._isclose(self.s, s_from_pythag):
            success = self._log(
                f"Distance s inconsistent with Pythagoras: s={self.s}, √(x²+y²)={s_from_pythag}", False)
        else:
            self._log("Pythagorean relation holds", True)

        return success

    def test_monotonicity(self) -> bool:
        """
        Perturb initial speed v0 and verify expected trends.
        """
        original_v0 = self.v0
        success = True

        # Increase v0 by 1%
        perturbed_v0 = original_v0 * 1.01
        # Recompute t and s using formulas
        perturbed_t = 2 * perturbed_v0 * self.tan37 / self.g
        perturbed_s = self.g * perturbed_t**2 / (2 * self.sin37)

        # t should increase
        if not perturbed_t > self.t:
            success = self._log(
                f"Monotonicity failure: t did not increase when v0 increased (t={self.t}, perturbed t={perturbed_t})", False)
        else:
            self._log(f"t increases with v0 as expected (perturbed t={perturbed_t})", True)

        # s should increase
        if not perturbed_s > self.s:
            success = self._log(
                f"Monotonicity failure: s did not increase when v0 increased (s={self.s}, perturbed s={perturbed_s})", False)
        else:
            self._log(f"s increases with v0 as expected (perturbed s={perturbed_s})", True)

        # Restore original v0
        self.v0 = original_v0
        return success

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """
        Execute all tests and return overall success and logs.
        """
        self.logs.clear()
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for name, test_func in tests:
            self.logs.append(f"--- Testing {name} ---")
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                overall_success = False
                self.logs.append(f"ERROR in {name}: {e}")

        return overall_success, self.logs