import math
import sympy as sp
from typing import Tuple, List

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe attribute access with fallbacks
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract results with defaults (0.0 for safety, but None to detect missing)
        self.y = self._safe_get('y', 20.0)          # vertical displacement (m)
        self.t = self._safe_get('t')                # time of flight (s)
        self.v_y = self._safe_get('v_y')            # vertical velocity at A (m/s, positive downward)
        self.v_0 = self._safe_get('v_0')            # horizontal throw velocity (m/s)
        
        # Constants (with defaults from problem statement)
        self.g = getattr(solution_instance, 'g', 10.0)
        self.brick_thickness = 0.1  # m
        self.n_bricks = 200
        self.theta_deg = 37.0
        # Use the exact trigonometric values provided in the problem
        self.sin_theta = 0.6
        self.cos_theta = 0.8
        
        # Expected derived quantities (using exact trig values)
        self.y_expected = self.n_bricks * self.brick_thickness  # 20.0 m
        self.t_expected = math.sqrt(2 * self.y_expected / self.g) if self.g != 0 else None
        self.v_y_expected = self.g * self.t_expected if self.t_expected is not None else None
        self.v_0_expected = self.v_y_expected * (self.sin_theta / self.cos_theta) if self.v_y_expected is not None else None
        
        # Test logs
        self.logs = []
        self.success = True

    def _safe_get(self, key, default=None):
        """Safely retrieve attribute from solution_instance or results dict."""
        # First try direct attribute, then results dict
        val = getattr(self, f'_{key}', None)
        if val is None:
            val = self.results.get(key, default)
        return val

    def _log(self, message, is_ok):
        """Append log message and update success flag."""
        status = "PASS" if is_ok else "FAIL"
        self.logs.append(f"{status}: {message}")
        if not is_ok:
            self.success = False

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """Check fundamental physics laws using pure numeric arithmetic."""
        # 1. Vertical displacement from bricks
        if self._is_close(self.y, self.y_expected):
            self._log(f"Vertical displacement y={self.y} m matches brick count.", True)
        else:
            self._log(f"Vertical displacement y={self.y} m disagrees with expected {self.y_expected} m.", False)

        # 2. Time from free fall (initial v_y=0)
        if self.t is not None and self.t_expected is not None:
            if self._is_close(self.t, self.t_expected):
                self._log(f"Time t={self.t} s matches free-fall sqrt(2y/g).", True)
            else:
                self._log(f"Time t={self.t} s disagrees with expected {self.t_expected} s.", False)

        # 3. Vertical velocity from constant acceleration
        if self.v_y is not None and self.t is not None:
            v_y_calc = self.g * self.t
            if self._is_close(self.v_y, v_y_calc):
                self._log(f"Vertical velocity v_y={self.v_y} m/s matches g*t.", True)
            else:
                self._log(f"Vertical velocity v_y={self.v_y} m/s disagrees with g*t={v_y_calc} m/s.", False)

        # 4. Horizontal velocity from perpendicular condition
        if self.v_0 is not None and self.v_y is not None:
            v_0_calc = self.v_y * (self.sin_theta / self.cos_theta)
            if self._is_close(self.v_0, v_0_calc):
                self._log(f"Horizontal velocity v_0={self.v_0} m/s matches v_y*tan(θ).", True)
            else:
                self._log(f"Horizontal velocity v_0={self.v_0} m/s disagrees with v_y*tan(θ)={v_0_calc} m/s.", False)

        # 5. Energy conservation: (1/2) v_y^2 = g * y
        if self.v_y is not None:
            energy_left = 0.5 * self.v_y ** 2
            energy_right = self.g * self.y
            if self._is_close(energy_left, energy_right):
                self._log(f"Energy conserved: (1/2)v_y^2={energy_left} ≈ g*y={energy_right}.", True)
            else:
                self._log(f"Energy violation: (1/2)v_y^2={energy_left} ≠ g*y={energy_right}.", False)

        # 6. Perpendicularity condition using exact numeric values
        if self.v_0 is not None and self.v_y is not None:
            # Correct condition: v_0 * cosθ = v_y * sinθ
            # So check if v_0 * cosθ - v_y * sinθ ≈ 0
            dot_product = self.v_0 * self.cos_theta - self.v_y * self.sin_theta
            if abs(dot_product) < 1e-2:  # numeric threshold
                self._log(f"Velocity perpendicular to slope: dot product ≈ {dot_product}.", True)
            else:
                self._log(f"Velocity not perpendicular to slope: dot product = {dot_product}.", False)

    def test_trace_consistency(self):
        """Verify intermediate steps lead to final results using numeric calculations."""
        # Recompute using the step-by-step formulas
        y_step = self.y_expected
        t_step = math.sqrt(2 * y_step / self.g) if self.g != 0 else None
        v_y_step = self.g * t_step if t_step is not None else None
        v_0_step = v_y_step * (self.sin_theta / self.cos_theta) if v_y_step is not None else None

        # Compare with stored results
        if self._is_close(self.y, y_step):
            self._log(f"Step consistency: y={self.y} m matches brick-derived {y_step} m.", True)
        else:
            self._log(f"Step inconsistency: y={self.y} m vs brick-derived {y_step} m.", False)

        if t_step is not None and self._is_close(self.t, t_step):
            self._log(f"Step consistency: t={self.t} s matches sqrt(2y/g)={t_step} s.", True)
        elif t_step is None:
            self._log("Step consistency: t_step cannot be computed (g=0?).", False)
        else:
            self._log(f"Step inconsistency: t={self.t} s vs sqrt(2y/g)={t_step} s.", False)

        if v_y_step is not None and self._is_close(self.v_y, v_y_step):
            self._log(f"Step consistency: v_y={self.v_y} m/s matches g*t={v_y_step} m/s.", True)
        elif v_y_step is None:
            self._log("Step consistency: v_y_step cannot be computed.", False)
        else:
            self._log(f"Step inconsistency: v_y={self.v_y} m/s vs g*t={v_y_step} m/s.", False)

        if v_0_step is not None and self._is_close(self.v_0, v_0_step):
            self._log(f"Step consistency: v_0={self.v_0} m/s matches v_y*tan(θ)={v_0_step} m/s.", True)
        elif v_0_step is None:
            self._log("Step consistency: v_0_step cannot be computed.", False)
        else:
            self._log(f"Step inconsistency: v_0={self.v_0} m/s vs v_y*tan(θ)={v_0_step} m/s.", False)

    def test_monotonicity(self):
        """Perturb input (brick count) and check output trends using pure numeric calculations."""
        original_y = self.y
        perturb_factor = 1.01  # 1% increase
        y_perturbed = original_y * perturb_factor
        
        # Recompute with perturbed y
        t_perturbed = math.sqrt(2 * y_perturbed / self.g)
        v_y_perturbed = self.g * t_perturbed
        v_0_perturbed = v_y_perturbed * (self.sin_theta / self.cos_theta)
        
        # Check monotonic increases
        if y_perturbed > original_y:
            if t_perturbed > self.t:
                self._log(f"Monotonicity: t increases with y (t={self.t} -> {t_perturbed}).", True)
            else:
                self._log(f"Monotonicity violation: t does not increase with y.", False)
            
            if v_y_perturbed > self.v_y:
                self._log(f"Monotonicity: v_y increases with y (v_y={self.v_y} -> {v_y_perturbed}).", True)
            else:
                self._log(f"Monotonicity violation: v_y does not increase with y.", False)
            
            if v_0_perturbed > self.v_0:
                self._log(f"Monotonicity: v_0 increases with y (v_0={self.v_0} -> {v_0_perturbed}).", True)
            else:
                self._log(f"Monotonicity violation: v_0 does not increase with y.", False)
        else:
            self._log("Monotonicity test skipped: perturbation did not increase y.", False)

    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Execute all checks and return (success, logs)."""
        self.logs = []
        self.success = True
        
        self.logs.append("=== Physics Checker Started ===")
        self.logs.append(f"Loaded: y={self.y}, t={self.t}, v_y={self.v_y}, v_0={self.v_0}")
        
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        self.logs.append(f"=== Overall Status: {'PASS' if self.success else 'FAIL'} ===")
        return self.success, self.logs