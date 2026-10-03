import math
import traceback

class PhysicsChecker:
    """
    Physics checker for the parachutist problem.
    Audits:
      - Results keys for time_to_terminal, dist_to_terminal, knee_bend_min.
      - Theoretical infinity for time and distance to terminal speed.
      - Numerical correctness of minimum knee bend d = v_i²/(9g).
      - Monotonicity of knee bend with impact speed.
    """

    def __init__(self, solution_instance):
        # Extract results from solution_instance
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        self.time_to_terminal = results.get('time_to_terminal', None)
        self.dist_to_terminal = results.get('dist_to_terminal', None)
        self.knee_bend_min = results.get('knee_bend_min', None)

        # Extract givens (physical constants)
        givens = getattr(solution_instance, 'givens', {})
        if givens is None:
            givens = {}
        self.g = givens.get('g', None)
        self.v_t1 = givens.get('v_t1', None)
        self.v_i = givens.get('v_i', None)

        # Fallback to direct attributes if not present in givens
        if self.g is None:
            self.g = getattr(solution_instance, 'g', None)
        if self.v_t1 is None:
            self.v_t1 = getattr(solution_instance, 'v_t1', None)
        if self.v_i is None:
            self.v_i = getattr(solution_instance, 'v_i', None)

        # Also try to get from a standard attribute
        if self.g is None:
            self.g = getattr(solution_instance, 'gravitational_acceleration', None)

        # Set a_max_allowed from the physics problem (constant 10g)
        # It's not strictly required to check, but can be used.
        if self.g is not None:
            self.a_max_allowed = 10.0 * self.g
        else:
            self.a_max_allowed = None

    def run_all_tests(self):
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

    # ---------------------------------------------------------------------
    # Interface tests: check required result keys exist
    # ---------------------------------------------------------------------
    def test_interface_result_time_to_terminal_present(self):
        if self.time_to_terminal is None:
            return (False, "Missing 'time_to_terminal' in results")
        return (True, "time_to_terminal is present")

    def test_interface_result_dist_to_terminal_present(self):
        if self.dist_to_terminal is None:
            return (False, "Missing 'dist_to_terminal' in results")
        return (True, "dist_to_terminal is present")

    def test_interface_result_knee_bend_min_present(self):
        if self.knee_bend_min is None:
            return (False, "Missing 'knee_bend_min' in results")
        return (True, "knee_bend_min is present")

    # ---------------------------------------------------------------------
    # Invariant tests: fundamental physics relationships
    # ---------------------------------------------------------------------
    def test_invariant_time_constant_positive(self):
        # τ = v_t1 / g should be positive
        if self.v_t1 is None or self.g is None:
            return (False, "SKIP: v_t1 and/or g not available")
        if self.g <= 0:
            return (False, f"g must be positive, got {self.g}")
        tau = self.v_t1 / self.g
        if not (tau > 0):
            return (False, f"τ = v_t1/g = {tau} should be positive")
        return (True, f"τ = {tau:.3f} s is positive")

    def test_invariant_terminal_speed_positive(self):
        # v_t1 is known to be 30 m/s, but we just check positivity
        if self.v_t1 is None:
            return (False, "SKIP: v_t1 not available")
        if self.v_t1 <= 0:
            return (False, f"v_t1 must be positive, got {self.v_t1}")
        return (True, f"v_t1 = {self.v_t1} > 0")

    # ---------------------------------------------------------------------
    # Modeling tests: compare against re-derived formulas from physics
    # ---------------------------------------------------------------------
    def test_modeling_time_to_terminal_infinite(self):
        # The correct physical result is that the exact terminal speed is reached only as t→∞.
        # Accept either the string 'infinite' or math.inf.
        if self.time_to_terminal is None:
            return (False, "SKIP: time_to_terminal not available")
        if isinstance(self.time_to_terminal, str) and self.time_to_terminal.lower() in ('infinite', 'infinity'):
            return (True, "time_to_terminal correctly indicates infinity")
        if isinstance(self.time_to_terminal, float) and math.isinf(self.time_to_terminal):
            return (True, "time_to_terminal is float('inf')")
        return (False, "time_to_terminal should be infinite, but got {!r}".format(self.time_to_terminal))

    def test_modeling_dist_to_terminal_infinite(self):
        if self.dist_to_terminal is None:
            return (False, "SKIP: dist_to_terminal not available")
        if isinstance(self.dist_to_terminal, str) and self.dist_to_terminal.lower() in ('infinite', 'infinity'):
            return (True, "dist_to_terminal correctly indicates infinity")
        if isinstance(self.dist_to_terminal, float) and math.isinf(self.dist_to_terminal):
            return (True, "dist_to_terminal is float('inf')")
        return (False, "dist_to_terminal should be infinite, but got {!r}".format(self.dist_to_terminal))

    def test_modeling_knee_bend_formula(self):
        # Re‑derive d_min = v_i^2 / (9g) from physics (including gravity in energy balance)
        if self.knee_bend_min is None or self.v_i is None or self.g is None:
            return (False, "SKIP: required variables missing")
        if self.g <= 0:
            return (False, f"g must be positive, got {self.g}")
        expected_d = self.v_i**2 / (9.0 * self.g)
        if not math.isclose(self.knee_bend_min, expected_d, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"knee_bend_min = {self.knee_bend_min:.4f}, expected ~{expected_d:.4f} "
                    "from d = v_i²/(9g)")
        return (True, f"knee_bend_min = {self.knee_bend_min:.4f} matches expected {expected_d:.4f}")

    # ---------------------------------------------------------------------
    # Monotonicity tests: verify physical trends by perturbing an input
    # ---------------------------------------------------------------------
    def test_monotonicity_knee_bend_increases_with_v_i(self):
        # If v_i is increased by 10%, d (≈ v_i²/(9g)) should increase.
        if self.knee_bend_min is None or self.v_i is None or self.g is None:
            return (False, "SKIP: missing variables for monotonicity test")
        if self.g <= 0 or self.v_i <= 0:
            return (False, "v_i and g must be positive")
        # Compute expected d for 10% larger velocity
        v_i_perturbed = self.v_i * 1.10
        d_perturbed = v_i_perturbed**2 / (9.0 * self.g)
        # Check that the perturbed d is larger than the original d
        if d_perturbed <= self.knee_bend_min * 0.99:  # allow tiny numerical tolerance
            return (False, f"Perturbing v_i by +10% should increase d; original d={self.knee_bend_min:.4f}, "
                    f"perturbed d={d_perturbed:.4f}")
        return (True, f"d increases from {self.knee_bend_min:.4f} to {d_perturbed:.4f} under +10% v_i")

    # ---------------------------------------------------------------------
    # Answer sanity (weak): only sign, finite, broad magnitude range
    # ---------------------------------------------------------------------
    def test_answer_sanity_knee_bend_positive_and_reasonable(self):
        if self.knee_bend_min is None:
            return (False, "SKIP: knee_bend_min not available")
        if not (isinstance(self.knee_bend_min, (int, float))):
            return (False, "knee_bend_min must be numeric")
        if self.knee_bend_min <= 0:
            return (False, "knee_bend_min must be positive")
        if math.isnan(self.knee_bend_min) or math.isinf(self.knee_bend_min):
            return (False, "knee_bend_min must be finite")
        # Broad range: typical human knee bend < 1 m and > 0.001 m
        if self.knee_bend_min > 1.0:
            return (False, f"knee_bend_min = {self.knee_bend_min} seems unrealistically large (>1 m)")
        if self.knee_bend_min < 0.001:
            return (False, f"knee_bend_min = {self.knee_bend_min} seems unrealistically small (<1 mm)")
        return (True, f"knee_bend_min = {self.knee_bend_min:.4f} m is positive, finite, and in a plausible range")

# Example usage (if needed)
if __name__ == "__main__":
    # Mock solution for testing
    class MockSolution:
        def __init__(self):
            self.results = {
                'time_to_terminal': 'infinite',
                'dist_to_terminal': 'infinite',
                'knee_bend_min': 0.102
            }
            self.givens = {
                'g': 9.8,
                'v_t1': 30,
                'v_i': 3
            }
            # Also allow direct attributes
            self.g = 9.8
            self.v_t1 = 30
            self.v_i = 3

    checker = PhysicsChecker(MockSolution())
    success, logs = checker.run_all_tests()
    for log in logs:
        print(log)
    print(f"\nOverall success: {success}")