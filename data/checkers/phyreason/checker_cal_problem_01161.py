import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safe extraction of inputs and results
        self.g = 10.0  # Given constant
        self.H = 500.0  # Given constant
        self.v0 = 0.0  # Released from rest

        # Extract results with fallbacks - use correct keys from interface contract
        self.t = getattr(solution_instance, 't', solution_instance.results.get('t', None))
        self.h1 = getattr(solution_instance, 'h_1', solution_instance.results.get('h_1', None))
        self.h5 = getattr(solution_instance, 'h_5', solution_instance.results.get('h_5', None))
        self.h10 = getattr(solution_instance, 'h_10', solution_instance.results.get('h_10', None))

        # Store for monotonicity perturbation
        self._original_H = self.H

    def test_invariants(self):
        logs = []
        success = True

        # 1. Time from free-fall: H = 0.5 * g * t^2
        if self.t is not None:
            expected_t = math.sqrt(2 * self.H / self.g)
            if not math.isclose(self.t, expected_t, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: t = {self.t}, expected from H = 0.5*g*t^2 gives {expected_t}")
                success = False
            else:
                logs.append("PASS: Total fall time consistent with H = 0.5*g*t^2")
        else:
            logs.append("FAIL: t not found")
            success = False

        # 2. Displacement in first second: h1 = 0.5 * g * 1^2
        if self.h1 is not None:
            expected_h1 = 0.5 * self.g * 1**2
            if not math.isclose(self.h1, expected_h1, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: h1 = {self.h1}, expected {expected_h1}")
                success = False
            else:
                logs.append("PASS: Displacement in first second correct")
        else:
            logs.append("FAIL: h1 not found")
            success = False

        # 3. Displacement at half time: h5 = 0.5 * g * 5^2
        if self.h5 is not None:
            expected_h5 = 0.5 * self.g * 5**2
            if not math.isclose(self.h5, expected_h5, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: h5 = {self.h5}, expected {expected_h5}")
                success = False
            else:
                logs.append("PASS: Displacement at half time correct")
        else:
            logs.append("FAIL: h5 not found")
            success = False

        # 4. Displacement in last second: h10 = 0.5 * g * t^2 - 0.5 * g * (t-1)^2
        if self.h10 is not None and self.t is not None:
            expected_h10 = 0.5 * self.g * (self.t**2 - (self.t - 1)**2)
            if not math.isclose(self.h10, expected_h10, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: h10 = {self.h10}, expected {expected_h10}")
                success = False
            else:
                logs.append("PASS: Displacement in last second correct")
        elif self.h10 is None:
            logs.append("FAIL: h10 not found")
            success = False

        # 5. Energy conservation: m*g*H = 0.5*m*v_final^2 => g*H = 0.5*(g*t)^2
        if self.t is not None:
            left = self.g * self.H
            right = 0.5 * (self.g * self.t)**2
            if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: Energy conservation violated: gH = {left}, 0.5*(g*t)^2 = {right}")
                success = False
            else:
                logs.append("PASS: Energy conservation holds")

        return success, logs

    def test_trace_consistency(self):
        logs = []
        success = True

        # Expected values from steps analysis
        expected_values = {
            't': 10.0,
            'h1': 5.0,
            'h5': 125.0,
            'h10': 95.0
        }

        for key, expected in expected_values.items():
            actual = getattr(self, key, None)
            if actual is None:
                logs.append(f"FAIL: {key} not found in results")
                success = False
            elif not math.isclose(actual, expected, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: {key} = {actual}, expected {expected}")
                success = False
            else:
                logs.append(f"PASS: {key} matches step analysis")

        return success, logs

    def test_monotonicity(self):
        logs = []
        success = True

        # Perturb height H by +10%
        delta_H = 0.1 * self.H
        new_H = self.H + delta_H

        # Recompute using free-fall formulas
        new_t = math.sqrt(2 * new_H / self.g)
        new_h_last = 0.5 * self.g * (2 * new_t - 1)  # displacement in last second

        # Check monotonicity: if H increases, t should increase and last-second displacement should increase
        if self.t is not None and self.h10 is not None:
            if not (new_t > self.t):
                logs.append(f"FAIL: Increasing H did not increase t (new_t={new_t}, old_t={self.t})")
                success = False
            else:
                logs.append("PASS: t increases with H")

            if not (new_h_last > self.h10):
                logs.append(f"FAIL: Last-second displacement did not increase (new={new_h_last}, old={self.h10})")
                success = False
            else:
                logs.append("PASS: Last-second displacement increases with H")
        else:
            logs.append("FAIL: t or h10 missing, cannot test monotonicity")
            success = False

        # Restore original H (though we never changed the attribute)
        # This is just symbolic; we used a local variable new_H.
        logs.append("Monotonicity test completed, original H restored.")

        return success, logs

    def run_all_tests(self):
        all_logs = []
        overall_success = True

        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for name, test_func in tests:
            success, logs = test_func()
            all_logs.append(f"=== {name} ===")
            all_logs.extend(logs)
            if not success:
                overall_success = False

        return overall_success, all_logs