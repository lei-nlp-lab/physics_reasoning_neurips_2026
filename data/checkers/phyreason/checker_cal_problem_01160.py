import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract inputs safely
        self.m = getattr(solution_instance, 'm', 0.03)  # 30 g -> 0.03 kg
        self.h = getattr(solution_instance, 'h', 45.0)   # height in m
        self.g = getattr(solution_instance, 'g', 10.0)   # gravity in m/s²

        # Extract results safely with fallback
        self.t = self._safe_get(solution_instance, 't')
        self.v = self._safe_get(solution_instance, 'v')
        self.overlinev = self._safe_get(solution_instance, 'overlinev')
        self.x = self._safe_get(solution_instance, 'x')
        self.t_1 = self._safe_get(solution_instance, 't_1')

        # Initialize logs
        self.logs = []

    def _safe_get(self, solution_instance, key):
        """Safe attribute access with multiple fallbacks."""
        # First try direct attribute, then results dict
        val = getattr(solution_instance, key, None)
        if val is None:
            # Try to get from results dict if it exists
            results = getattr(solution_instance, 'results', {})
            val = results.get(key, 0.0)
        # Ensure numeric type
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust numerical comparison."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    def test_invariants(self):
        """Check fundamental physical laws."""
        logs = []
        success = True

        # 1. Energy conservation: mgh = 0.5*m*v^2 -> v = sqrt(2gh)
        expected_v = math.sqrt(2 * self.g * self.h)
        if not self._is_close(self.v, expected_v):
            logs.append(f"Energy conservation failed: v={self.v}, sqrt(2gh)={expected_v}")
            success = False
        else:
            logs.append("Energy conservation OK.")

        # 2. Free-fall time: h = 0.5*g*t^2 -> t = sqrt(2h/g)
        expected_t = math.sqrt(2 * self.h / self.g)
        if not self._is_close(self.t, expected_t):
            logs.append(f"Free-fall time invariant failed: t={self.t}, sqrt(2h/g)={expected_t}")
            success = False
        else:
            logs.append("Free-fall time invariant OK.")

        # 3. Velocity-time relation: v = g*t
        expected_v_from_t = self.g * self.t
        if not self._is_close(self.v, expected_v_from_t):
            logs.append(f"Velocity-time relation failed: v={self.v}, g*t={expected_v_from_t}")
            success = False
        else:
            logs.append("Velocity-time relation OK.")

        # 4. Last-second displacement consistency
        # Method 1: x = h - 0.5*g*t_1^2
        expected_x = self.h - 0.5 * self.g * (self.t_1 ** 2)
        # Method 2: x = 0.5*g*(t^2 - t_1^2)
        alt_expected_x = 0.5 * self.g * (self.t ** 2 - self.t_1 ** 2)
        if not (self._is_close(self.x, expected_x) and self._is_close(self.x, alt_expected_x)):
            logs.append(f"Last-second displacement failed: x={self.x}, expected={expected_x}, alt={alt_expected_x}")
            success = False
        else:
            logs.append("Last-second displacement invariant OK.")

        # 5. Average speed in last second: overlinev = x / 1 (since time interval is 1s)
        expected_overlinev = self.x  # because delta_t = 1
        if not self._is_close(self.overlinev, expected_overlinev):
            logs.append(f"Average speed invariant failed: overlinev={self.overlinev}, x={self.x}")
            success = False
        else:
            logs.append("Average speed invariant OK.")

        self.logs.extend(logs)
        return success

    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True

        # 1. t_1 = t - 1
        expected_t_1 = self.t - 1.0
        if not self._is_close(self.t_1, expected_t_1):
            logs.append(f"Trace consistency t_1 failed: t_1={self.t_1}, t-1={expected_t_1}")
            success = False
        else:
            logs.append("Trace t_1 = t-1 OK.")

        # 2. x and overlinev: overlinev = x (since time=1s)
        if not self._is_close(self.overlinev, self.x):
            logs.append(f"Trace consistency overlinev vs x failed: overlinev={self.overlinev}, x={self.x}")
            success = False
        else:
            logs.append("Trace overlinev = x OK.")

        # 3. x from h and t: x = h - 0.5*g*(t-1)^2
        x_from_t = self.h - 0.5 * self.g * (self.t - 1) ** 2
        if not self._is_close(self.x, x_from_t):
            logs.append(f"Trace consistency x from t failed: x={self.x}, computed={x_from_t}")
            success = False
        else:
            logs.append("Trace x from t OK.")

        self.logs.extend(logs)
        return success

    def test_monotonicity(self):
        """Perturb inputs and verify expected trends."""
        logs = []
        success = True

        original_h = self.h
        original_g = self.g

        # Perturbation 1: Increase h by 10%
        perturbed_h = original_h * 1.10
        # Recompute using free-fall formulas
        t_pert = math.sqrt(2 * perturbed_h / original_g)
        v_pert = original_g * t_pert
        t_1_pert = t_pert - 1.0
        x_pert = perturbed_h - 0.5 * original_g * (t_1_pert ** 2)
        overlinev_pert = x_pert  # time interval = 1s

        # Check trends: t and v should increase with h
        if not (t_pert > self.t and v_pert > self.v):
            logs.append(f"Monotonicity (h increase) failed: t_pert={t_pert}, t={self.t}; v_pert={v_pert}, v={self.v}")
            success = False
        else:
            logs.append("Monotonicity (h increase) OK: t and v increase.")

        # Perturbation 2: Increase g by 10%
        perturbed_g = original_g * 1.10
        # Recompute with original h
        t_pert_g = math.sqrt(2 * original_h / perturbed_g)
        v_pert_g = perturbed_g * t_pert_g
        t_1_pert_g = t_pert_g - 1.0
        x_pert_g = original_h - 0.5 * perturbed_g * (t_1_pert_g ** 2)
        overlinev_pert_g = x_pert_g

        # Check trends: t should decrease, v should increase
        if not (t_pert_g < self.t and v_pert_g > self.v):
            logs.append(f"Monotonicity (g increase) failed: t_pert={t_pert_g}, t={self.t}; v_pert={v_pert_g}, v={self.v}")
            success = False
        else:
            logs.append("Monotonicity (g increase) OK: t decreases, v increases.")

        self.logs.extend(logs)
        return success

    def run_all_tests(self):
        """Execute all checks and return (success, logs)."""
        self.logs = []  # Reset logs
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for name, test_func in tests:
            try:
                if not test_func():
                    overall_success = False
            except Exception as e:
                self.logs.append(f"{name} test crashed: {traceback.format_exc()}")
                overall_success = False

        return overall_success, self.logs