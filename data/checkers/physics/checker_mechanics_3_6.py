import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for the spaceship radio signal problem.
    Audits outputs for a solution instance: Earth frame time, ship frame time,
    Earth frame location, and ship frame location.
    """
    def __init__(self, solution_instance):
        # Extract givens and results safely; use None for missing values.
        # The solution_instance is expected to have attributes/results describing the problem.
        # We support both attribute access and .results dict.
        if hasattr(solution_instance, 'results'):
            results = solution_instance.results
        else:
            results = {}
        self.results = results

        # Fundamental constants (if available from instance, else None)
        self.c = getattr(solution_instance, 'c', None) or results.get('c', None)
        self.v = getattr(solution_instance, 'v', None) or results.get('v', None)
        self.d = getattr(solution_instance, 'd', None) or results.get('d', None)  # distance in km as given
        # If d is missing, try d_km
        if self.d is None:
            self.d = results.get('d_km', None)

        # Derived quantities (computed if givens exist)
        if self.d is not None and self.c is not None and self.v is not None:
            self.d_m = self.d * 1000.0          # convert km -> m
            self.gamma = 1.0 / math.sqrt(1.0 - (self.v**2 / self.c**2))
        else:
            self.d_m = None
            self.gamma = None

        # Output results (always None-safe)
        self.t_ship = results.get('t_ship', None)
        self.t_earth = results.get('t_earth', None)
        self.x_earth_km = results.get('x_earth_km', None)
        self.x_ship = results.get('x_ship', None)

    # ------------------------------------------------------------------
    # Interface checks: required keys and basic types
    # ------------------------------------------------------------------
    def test_interface_required_keys_present(self):
        """Check that all expected result keys exist."""
        required = ['t_ship', 't_earth', 'x_earth_km', 'x_ship']
        missing = [key for key in required if key not in self.results]
        if missing:
            return False, f"Missing keys: {missing}"
        return True, "All required keys present"

    def test_interface_values_are_numbers(self):
        """Check that each result is a finite number (int/float)."""
        keys = ['t_ship', 't_earth', 'x_earth_km', 'x_ship']
        for key in keys:
            val = self.results.get(key)
            if val is None:
                return False, f"{key} is None"
            if not isinstance(val, (int, float)):
                return False, f"{key} is not a number, got {type(val)}"
            if not math.isfinite(val):
                return False, f"{key} is not finite: {val}"
        return True, "All values are finite numbers"

    # ------------------------------------------------------------------
    # Invariant (theorem-based) checks: use math.isclose
    # ------------------------------------------------------------------
    def test_invariant_earth_frame_propagation(self):
        """Check Earth frame: x_earth_km ≈ c * t_earth / 1000."""
        if None in (self.t_earth, self.x_earth_km, self.c):
            return False, "SKIP: missing t_earth, x_earth_km, or c"
        expected_km = self.c * self.t_earth / 1000.0
        if not math.isclose(self.x_earth_km, expected_km, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"x_earth_km={self.x_earth_km} differs from c*t_earth/1000={expected_km:.6e}"
        return True, "Earth frame location consistent with light travel"

    def test_invariant_ship_at_rest(self):
        """Check that ship frame location is 0."""
        if self.x_ship is None:
            return False, "SKIP: x_ship not available"
        if not math.isclose(self.x_ship, 0.0, abs_tol=1e-2):
            return False, f"x_ship should be 0, got {self.x_ship}"
        return True, "Ship at rest in its own frame (x'=0)"

    def test_invariant_lorentz_transform(self):
        """Check t_ship = gamma*(t_earth - v*x_earth_m/c^2)."""
        if None in (self.t_ship, self.t_earth, self.x_earth_km, self.v, self.c, self.gamma):
            return False, "SKIP: missing required quantities for Lorentz transform"
        x_earth_m = self.x_earth_km * 1000.0
        expected_t_ship = self.gamma * (self.t_earth - self.v * x_earth_m / (self.c**2))
        if not math.isclose(self.t_ship, expected_t_ship, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"t_ship={self.t_ship} differs from Lorentz transform {expected_t_ship:.6e}"
        return True, "Lorentz transform consistency verified"

    def test_invariant_lorentz_factor(self):
        """Check gamma = 1/sqrt(1-v^2/c^2)."""
        if None in (self.v, self.c, self.gamma):
            return False, "SKIP: missing v, c, or gamma"
        expected_gamma = 1.0 / math.sqrt(1.0 - (self.v**2 / self.c**2))
        if not math.isclose(self.gamma, expected_gamma, rel_tol=1e-2):
            return False, f"gamma={self.gamma} differs from computed {expected_gamma}"
        return True, "Lorentz factor correct"

    # ------------------------------------------------------------------
    # Modeling checks (based on explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_earth_time_derivation(self):
        """Check that t_earth equals d/(c-v) when d, c, v are known."""
        if None in (self.t_earth, self.d, self.c, self.v):
            return False, "SKIP: missing parameters for Earth time derivation"
        d_m = self.d * 1000.0
        expected_t = d_m / (self.c - self.v)
        if not math.isclose(self.t_earth, expected_t, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"t_earth={self.t_earth} differs from d/(c-v)={expected_t:.6e}"
        return True, "Earth time formula t = d/(c-v) verified"

    def test_modeling_lorentz_transform_uses_correct_coordinates(self):
        """Check that t_ship derived from Earth coordinates and gamma is consistent with independent computation."""
        if None in (self.t_ship, self.t_earth, self.x_earth_km, self.v, self.c, self.gamma):
            return False, "SKIP: missing required values"
        x_earth_m = self.x_earth_km * 1000.0
        t_ship_computed = self.gamma * (self.t_earth - self.v * x_earth_m / (self.c**2))
        if not math.isclose(self.t_ship, t_ship_computed, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"t_ship does not match Lorentz transform (computed {t_ship_computed:.6e})"
        return True, "Ship frame time consistent with Lorentz transform"

    def test_modeling_ship_frame_x_is_zero(self):
        """Check that x_ship is 0 (ship at origin in its own frame)."""
        if self.x_ship is None:
            return False, "SKIP: x_ship not available"
        if not math.isclose(self.x_ship, 0.0, abs_tol=1e-2):
            return False, f"x_ship should be 0, got {self.x_ship}"
        return True, "Ship frame location is 0"

    # ------------------------------------------------------------------
    # Monotonicity checks: perturb an input and observe directional trend
    # ------------------------------------------------------------------
    def test_monotonicity_distance_earth_time(self):
        """If distance d increases, t_earth should increase."""
        if None in (self.t_earth, self.d, self.c, self.v):
            return False, "SKIP: missing parameters for monotonicity test"
        # Use a small perturbation (1%) and recompute t_earth from formula
        d_base = self.d * 1000.0
        t_base = d_base / (self.c - self.v)
        d_perturbed = d_base * 1.01
        t_perturbed = d_perturbed / (self.c - self.v)
        if not (t_perturbed > t_base):
            return False, f"Monotonicity failed: increasing d did not increase t_earth (base={t_base}, perturbed={t_perturbed})"
        return True, "Monotonicity: Earth time increases with distance"

    def test_monotonicity_distance_ship_time(self):
        """If distance d increases, t_ship should increase (via Lorentz transform)."""
        if None in (self.t_ship, self.d, self.c, self.v, self.gamma):
            return False, "SKIP: missing parameters for ship time monotonicity"
        # Recompute t_ship for perturbed d (1% increase)
        d_base = self.d * 1000.0
        t_earth_base = d_base / (self.c - self.v)
        x_earth_base = self.c * t_earth_base
        t_ship_base = self.gamma * (t_earth_base - self.v * x_earth_base / (self.c**2))

        d_perturbed = d_base * 1.01
        t_earth_pert = d_perturbed / (self.c - self.v)
        x_earth_pert = self.c * t_earth_pert
        t_ship_pert = self.gamma * (t_earth_pert - self.v * x_earth_pert / (self.c**2))

        if not (t_ship_pert > t_ship_base):
            return False, f"Monotonicity failed: increasing d did not increase t_ship (base={t_ship_base}, perturbed={t_ship_pert})"
        return True, "Monotonicity: Ship frame time increases with distance"

    def test_monotonicity_velocity_earth_time(self):
        """If ship speed v increases, t_earth should increase (since denominator decreases)."""
        if None in (self.t_earth, self.d, self.c, self.v):
            return False, "SKIP: missing parameters for velocity monotonicity"
        d_m = self.d * 1000.0
        v_base = self.v
        t_base = d_m / (self.c - v_base)
        v_perturbed = v_base * 1.01
        t_perturbed = d_m / (self.c - v_perturbed)
        if not (t_perturbed > t_base):
            return False, f"Monotonicity failed: increasing v did not increase t_earth (base={t_base}, perturbed={t_perturbed})"
        return True, "Monotonicity: Earth time increases with ship speed"

    # ------------------------------------------------------------------
    # Single weak answer sanity method (combined checks)
    # ------------------------------------------------------------------
    def test_answer_sanity_weak_checks(self):
        """
        Weak sanity: check that t_ship < t_earth (time dilation),
        x_earth_km > d (ship moves away), and all values finite/non-negative.
        No hardcoded reference numbers.
        """
        # Ordering check
        if None in (self.t_ship, self.t_earth):
            return False, "SKIP: missing times for ordering"
        if not (self.t_ship < self.t_earth):
            return False, f"t_ship ({self.t_ship}) should be less than t_earth ({self.t_earth})"
        if None not in (self.x_earth_km, self.d):
            if not (self.x_earth_km > self.d):
                return False, f"x_earth_km ({self.x_earth_km}) should be > d ({self.d})"
        # Finiteness/positivity
        keys = ['t_ship', 't_earth', 'x_earth_km']
        for key in keys:
            val = self.results.get(key)
            if val is None:
                continue
            if not math.isfinite(val):
                return False, f"{key} is not finite: {val}"
            if val < 0:
                return False, f"{key} is negative: {val}"
        # x_ship should be 0 (or close)
        if self.x_ship is not None:
            if not math.isfinite(self.x_ship) or self.x_ship < -1e-9:
                return False, f"x_ship is not zero/positive: {self.x_ship}"
        return True, "Weak sanity: ordering and positivity pass"

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)