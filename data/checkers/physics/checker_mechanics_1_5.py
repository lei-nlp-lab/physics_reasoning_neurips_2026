import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results from solution_instance (expected key 'x')
        results = getattr(solution_instance, 'results', None)
        if results is None:
            results = {}
        elif not isinstance(results, dict):
            try:
                results = dict(results)
            except Exception:
                results = {}
        self.results = results
        self.x = self.results.get('x', None)

        # Extract givens from solution_instance; try dictionary first, then attributes
        givens = getattr(solution_instance, 'givens', None)
        if givens is None or not isinstance(givens, dict):
            givens = {}
        self.givens = givens

        # Individual givens (with fallback to attributes if not in dict)
        self.theta = givens.get('theta', getattr(solution_instance, 'theta', None))
        self.mu = givens.get('mu', getattr(solution_instance, 'mu', None))
        self.v0 = givens.get('v0', getattr(solution_instance, 'v0', None))
        self.g = givens.get('g', getattr(solution_instance, 'g', None))
        self.t_target = givens.get('t_target', getattr(solution_instance, 't_target', None))

        # Explanation steps (optional)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Derived quantities will be computed on demand
        self._sin_theta = None
        self._cos_theta = None
        self._a_up = None
        self._a_down = None

    def _compute_derived(self):
        """Compute trigonometric values and accelerations if givens are present."""
        if self.theta is not None and self.mu is not None and self.g is not None:
            theta_rad = math.radians(self.theta)
            self._sin_theta = math.sin(theta_rad)
            self._cos_theta = math.cos(theta_rad)
            self._a_up = -self.g * (self._sin_theta + self.mu * self._cos_theta)
            self._a_down = -self.g * (self._sin_theta - self.mu * self._cos_theta)

    def _compute_displacement(self, mu, v0, theta, g, t_target):
        """Re‑compute the total displacement for given inputs using two‑phase kinematics.
           Assumes positive direction up the incline.
        """
        theta_rad = math.radians(theta)
        sin_t = math.sin(theta_rad)
        cos_t = math.cos(theta_rad)
        a_up = -g * (sin_t + mu * cos_t)
        a_down = -g * (sin_t - mu * cos_t)
        # Phase 1: upward until stop
        t_stop = -v0 / a_up   # a_up negative, so t_stop positive
        x_stop = v0 * t_stop + 0.5 * a_up * t_stop**2
        # Phase 2: downward from rest
        t2 = t_target - t_stop
        if t2 < 0:
            # Brick never stops within t_target
            x_final = v0 * t_target + 0.5 * a_up * t_target**2
            return x_final
        x_final = x_stop + 0.5 * a_down * t2**2
        return x_final

    # ------------------------------------------------------------------
    # Interface tests: check presence and type of required keys
    # ------------------------------------------------------------------
    def test_interface_result_key(self):
        """Check that results['x'] exists and is a numeric type."""
        if self.x is None:
            return (False, "FAIL: results['x'] is missing or None")
        if not isinstance(self.x, (int, float)):
            return (False, f"FAIL: results['x'] has unsupported type {type(self.x).__name__}")
        if math.isnan(self.x) or math.isinf(self.x):
            return (False, f"FAIL: results['x'] is NaN or Inf")
        return (True, f"PASS: results['x'] = {self.x} is numeric and finite")

    def test_interface_given_theta(self):
        """Check that given 'theta' is provided."""
        if self.theta is None:
            return (False, "SKIP: theta not provided")
        if not isinstance(self.theta, (int, float)):
            return (False, f"FAIL: theta has unsupported type {type(self.theta).__name__}")
        return (True, f"PASS: theta = {self.theta}")

    def test_interface_given_mu(self):
        """Check that given 'mu' is provided."""
        if self.mu is None:
            return (False, "SKIP: mu not provided")
        if not isinstance(self.mu, (int, float)):
            return (False, f"FAIL: mu has unsupported type {type(self.mu).__name__}")
        return (True, f"PASS: mu = {self.mu}")

    def test_interface_given_v0(self):
        """Check that given 'v0' is provided."""
        if self.v0 is None:
            return (False, "SKIP: v0 not provided")
        if not isinstance(self.v0, (int, float)):
            return (False, f"FAIL: v0 has unsupported type {type(self.v0).__name__}")
        return (True, f"PASS: v0 = {self.v0}")

    def test_interface_given_g(self):
        """Check that given 'g' is provided."""
        if self.g is None:
            return (False, "SKIP: g not provided")
        if not isinstance(self.g, (int, float)):
            return (False, f"FAIL: g has unsupported type {type(self.g).__name__}")
        return (True, f"PASS: g = {self.g}")

    def test_interface_given_t_target(self):
        """Check that given 't_target' is provided."""
        if self.t_target is None:
            return (False, "SKIP: t_target not provided")
        if not isinstance(self.t_target, (int, float)):
            return (False, f"FAIL: t_target has unsupported type {type(self.t_target).__name__}")
        return (True, f"PASS: t_target = {self.t_target}")

    # ------------------------------------------------------------------
    # Invariant tests: theorem-based physical sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_acceleration_signs(self):
        """Check that accelerations (if derivable) are negative (since positive up)."""
        self._compute_derived()
        if self._a_up is None or self._a_down is None:
            return (False, "SKIP: insufficient givens to compute accelerations")
        if self._a_up >= 0:
            return (False, f"FAIL: computed a_up = {self._a_up} should be negative")
        if self._a_down >= 0:
            return (False, f"FAIL: computed a_down = {self._a_down} should be negative")
        return (True, f"PASS: a_up = {self._a_up} and a_down = {self._a_down} are both negative")

    def test_invariant_a_up_physics(self):
        """Check that a_up conforms to Newton's 2nd law: a_up = -g*(sinθ + μ cosθ)."""
        self._compute_derived()
        if self._a_up is None:
            return (False, "SKIP: cannot compute a_up")
        a_up_sol = getattr(self, 'a_up', None) or self.results.get('a_up', None)
        if a_up_sol is None:
            return (False, "SKIP: solution did not provide a_up")
        if math.isclose(a_up_sol, self._a_up, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"PASS: a_up solution {a_up_sol} matches derived {self._a_up}")
        else:
            return (False, f"FAIL: a_up solution {a_up_sol} differs from derived {self._a_up}")

    def test_invariant_a_down_physics(self):
        """Check that a_down conforms to Newton's 2nd law: a_down = -g*(sinθ - μ cosθ)."""
        self._compute_derived()
        if self._a_down is None:
            return (False, "SKIP: cannot compute a_down")
        a_down_sol = getattr(self, 'a_down', None) or self.results.get('a_down', None)
        if a_down_sol is None:
            return (False, "SKIP: solution did not provide a_down")
        if math.isclose(a_down_sol, self._a_down, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"PASS: a_down solution {a_down_sol} matches derived {self._a_down}")
        else:
            return (False, f"FAIL: a_down solution {a_down_sol} differs from derived {self._a_down}")

    def test_invariant_stop_time_physics(self):
        """Check that t_stop (if provided) satisfies v0 + a_up*t_stop = 0."""
        self._compute_derived()
        if self._a_up is None or self.v0 is None:
            return (False, "SKIP: cannot compute expected stop time")
        t_stop_expected = -self.v0 / self._a_up
        t_stop_sol = getattr(self, 't_stop', None) or self.results.get('t_stop', None)
        if t_stop_sol is None:
            return (False, "SKIP: solution did not provide t_stop")
        if math.isclose(t_stop_sol, t_stop_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"PASS: t_stop solution {t_stop_sol} matches expected {t_stop_expected}")
        else:
            return (False, f"FAIL: t_stop solution {t_stop_sol} differs from expected {t_stop_expected}")

    def test_invariant_displacement_at_stop(self):
        """Check that x_stop (if provided) satisfies kinematics: x = v0*t + 0.5*a*t^2."""
        self._compute_derived()
        if self._a_up is None or self.v0 is None:
            return (False, "SKIP: cannot compute expected displacement at stop")
        t_stop_sol = getattr(self, 't_stop', None) or self.results.get('t_stop', None)
        if t_stop_sol is None:
            return (False, "SKIP: solution did not provide t_stop")
        x_stop_sol = getattr(self, 'x_stop', None) or self.results.get('x_stop', None)
        if x_stop_sol is None:
            return (False, "SKIP: solution did not provide x_stop")
        x_stop_expected = self.v0 * t_stop_sol + 0.5 * self._a_up * t_stop_sol**2
        if math.isclose(x_stop_sol, x_stop_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"PASS: x_stop solution {x_stop_sol} matches expected {x_stop_expected}")
        else:
            return (False, f"FAIL: x_stop solution {x_stop_sol} differs from expected {x_stop_expected}")

    # ------------------------------------------------------------------
    # Modeling/trace tests: check consistency of phases
    # ------------------------------------------------------------------
    def test_modeling_phases_present(self):
        """Check that the solution uses two phases (up then down). 
           We check that t_stop < t_target if both are available."""
        t_stop_sol = getattr(self, 't_stop', None) or self.results.get('t_stop', None)
        t_target = self.t_target
        if t_stop_sol is None or t_target is None:
            return (False, "SKIP: t_stop or t_target not available for phase check")
        if t_stop_sol < t_target:
            return (True, f"PASS: t_stop={t_stop_sol} < t_target={t_target}, two phases needed")
        else:
            return (False, f"FAIL: t_stop={t_stop_sol} >= t_target={t_target}, motion may be single phase, but expected reversal")

    def test_modeling_final_displacement_via_phases(self):
        """If the solution provides x_stop and a_down, and t_stop is available,
           check that final x = x_stop + 0.5*a_down*(t_target - t_stop)^2."""
        x_stop_sol = getattr(self, 'x_stop', None) or self.results.get('x_stop', None)
        a_down_sol = getattr(self, 'a_down', None) or self.results.get('a_down', None)
        t_stop_sol = getattr(self, 't_stop', None) or self.results.get('t_stop', None)
        t_target = self.t_target
        x_final = self.x
        if any(v is None for v in [x_stop_sol, a_down_sol, t_stop_sol, t_target, x_final]):
            return (False, "SKIP: missing one or more required values for phase composition")
        t2 = t_target - t_stop_sol
        x_expected = x_stop_sol + 0.5 * a_down_sol * t2**2
        if math.isclose(x_final, x_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"PASS: final x {x_final} matches two‑phase composition {x_expected}")
        else:
            return (False, f"FAIL: final x {x_final} differs from two‑phase composition {x_expected}")

    # ------------------------------------------------------------------
    # Monotonicity test: explicitly perturb an input and compare with baseline
    # ------------------------------------------------------------------
    def test_monotonicity_increasing_friction_reduces_displacement(self):
        """Perturb μ upward by 10% and verify that the computed displacement strictly decreases.
           Baseline displacement is taken from the solution (original μ).
           If solution's x is not available, skip.
        """
        if self.x is None:
            return (False, "SKIP: results['x'] not available for baseline")
        if any(v is None for v in [self.theta, self.mu, self.v0, self.g, self.t_target]):
            return (False, "SKIP: insufficient givens to compute perturbed displacement")
        # Compute displacement for original μ using the solution's x as baseline
        baseline_x = self.x

        # Perturb μ upward by 10%
        mu_perturbed = self.mu * 1.10
        # Compute expected displacement for perturbed μ using the same two‑phase model
        perturbed_x = self._compute_displacement(
            mu_perturbed, self.v0, self.theta, self.g, self.t_target
        )

        # Monotonic expectation: larger friction => smaller (positive) displacement
        if baseline_x > perturbed_x:
            return (True, f"PASS: baseline x={baseline_x:.4f} > perturbed x={perturbed_x:.4f} (μ increased by 10%)")
        else:
            return (False, f"FAIL: baseline x={baseline_x:.4f} <= perturbed x={perturbed_x:.4f} (expected decrease)")

    # ------------------------------------------------------------------
    # Weak answer sanity (must NOT use math.isclose or derive expected values)
    # ------------------------------------------------------------------
    def test_answer_sanity_bounds(self):
        """Check that final displacement x is positive and not unreasonably large.
           Use lenient bounds: 0 < x < 2 ft (since initial speed 5 ft/s, max
           travel in 0.5s without friction is about 1.25 ft, but with friction it's less.
           A safe upper bound is 2 ft to account for possible errors.)"""
        if self.x is None:
            return (False, "SKIP: results['x'] not available")
        if not isinstance(self.x, (int, float)):
            return (False, f"FAIL: x is not numeric: {self.x}")
        if math.isnan(self.x) or math.isinf(self.x):
            return (False, "FAIL: x is NaN or Inf")
        if self.x <= 0:
            return (False, f"FAIL: x = {self.x} should be positive (brick moves up before sliding back)")
        if self.x >= 2.0:
            return (False, f"FAIL: x = {self.x} is >= 2 ft, seems too large")
        return (True, f"PASS: x = {self.x} is positive and within reasonable bound")

    # ------------------------------------------------------------------
    # Main test runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        """Run all test_* methods and return (all_pass, logs)."""
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)