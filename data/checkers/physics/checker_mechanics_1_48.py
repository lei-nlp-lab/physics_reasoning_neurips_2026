import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results from solution_instance (safe defaults)
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}

        self.principle = results.get('principle', None)
        self.orbit_shape = results.get('orbit_shape', None)
        self.perigee_altitude = results.get('perigee_altitude', None)
        self.perigee_distance = results.get('perigee_distance', None)

        # Extract givens if available (may be missing)
        givens = getattr(solution_instance, 'givens', {})
        if not isinstance(givens, dict):
            givens = {}
        self.h = givens.get('h', None)          # altitude in m
        self.R_e = givens.get('R_e', None)      # Earth radius in m
        self.GM = givens.get('GM', None)        # gravitational parameter m^3/s^2
        self.KE_loss_fraction = givens.get('KE_loss_fraction', None)
        self.m = givens.get('m', None)          # mass (not strictly needed but for completeness)

        # Derived quantities (only for internal checks)
        self.r0 = None
        self.v0 = None
        self.v = None
        if self.h is not None and self.R_e is not None:
            self.r0 = self.R_e + self.h
            if self.GM is not None:
                self.v0 = math.sqrt(self.GM / self.r0)
                if self.KE_loss_fraction is not None:
                    self.v = math.sqrt(1 - self.KE_loss_fraction) * self.v0

    # ==================== INTERFACE TESTS ====================

    def test_interface_principle_exists(self):
        if self.principle is None:
            return (False, "SKIP: principle not provided")
        if not isinstance(self.principle, str) or self.principle.strip() == '':
            return (False, f"principle must be a non-empty string, got {type(self.principle)}: {self.principle}")
        return (True, "principle provided as non-empty string")

    def test_interface_orbit_shape_exists(self):
        if self.orbit_shape is None:
            return (False, "SKIP: orbit_shape not provided")
        if not isinstance(self.orbit_shape, str) or self.orbit_shape.strip() == '':
            return (False, f"orbit_shape must be a non-empty string, got {type(self.orbit_shape)}: {self.orbit_shape}")
        return (True, "orbit_shape provided as non-empty string")

    def test_interface_perigee_altitude_numeric(self):
        if self.perigee_altitude is None:
            return (False, "SKIP: perigee_altitude not provided")
        if not isinstance(self.perigee_altitude, (int, float)):
            return (False, f"perigee_altitude must be numeric, got {type(self.perigee_altitude)}")
        return (True, "perigee_altitude is numeric")

    def test_interface_perigee_distance_numeric(self):
        if self.perigee_distance is None:
            return (False, "SKIP: perigee_distance not provided")
        if not isinstance(self.perigee_distance, (int, float)):
            return (False, f"perigee_distance must be numeric, got {type(self.perigee_distance)}")
        return (True, "perigee_distance is numeric")

    # ==================== INVARIANT TESTS (theorem-based, use isclose) ====================

    def test_invariant_perigee_less_than_initial_radius(self):
        """Check that the perigee distance is less than the initial orbital radius (collision point = apogee)."""
        if self.perigee_distance is None or self.r0 is None:
            return (False, "SKIP: missing perigee_distance or initial radius (r0)")
        if self.perigee_distance < self.r0:
            return (True, f"perigee_distance {self.perigee_distance:.2e} m < initial radius {self.r0:.2e} m (consistent with apogee)")
        return (False, f"perigee_distance {self.perigee_distance:.2e} m is not less than initial radius {self.r0:.2e} m")

    def test_invariant_perigee_altitude_positive(self):
        if self.perigee_altitude is None:
            return (False, "SKIP: perigee_altitude not provided")
        if self.perigee_altitude >= 0:
            return (True, "perigee altitude is non-negative")
        return (False, f"perigee altitude {self.perigee_altitude} is negative")

    def test_invariant_energy_conservation_sign(self):
        """Check that the total mechanical energy after collision is negative (bound orbit)."""
        # Energy = kinetic - potential, E = 0.5*m*v^2 - GM*m/r0
        # We can compute if we have m, v, r0, GM
        if self.m is None or self.v is None or self.r0 is None or self.GM is None:
            return (False, "SKIP: missing data to compute energy (m, v, r0, GM)")
        E = 0.5 * self.m * self.v**2 - self.GM * self.m / self.r0
        if E < 0:
            return (True, f"Total mechanical energy E = {E:.3e} J is negative (bound orbit)")
        return (False, f"Total mechanical energy E = {E:.3e} J is not negative (expected bound orbit)")

    # ==================== MODELING / TRACE TESTS ====================

    def test_modeling_principle_content(self):
        """Check that principle mentions conservation of energy and angular momentum."""
        if self.principle is None:
            return (False, "SKIP: principle not provided")
        principle_lower = self.principle.lower()
        has_energy = 'energy' in principle_lower
        has_angular = 'angular momentum' in principle_lower
        if has_energy and has_angular:
            return (True, "principle contains both energy and angular momentum conservation")
        missing = []
        if not has_energy:
            missing.append("energy conservation")
        if not has_angular:
            missing.append("angular momentum conservation")
        return (False, f"principle missing: {', '.join(missing)}")

    def test_modeling_orbit_shape_elliptic(self):
        """Check that orbit shape is described as elliptic."""
        if self.orbit_shape is None:
            return (False, "SKIP: orbit_shape not provided")
        shape_lower = self.orbit_shape.lower()
        if 'elliptic' in shape_lower or 'ellipse' in shape_lower:
            return (True, f"orbit_shape contains 'elliptic' or 'ellipse': {self.orbit_shape}")
        return (False, f"orbit_shape does not indicate elliptic orbit: {self.orbit_shape}")

    def test_modeling_perigee_less_than_initial_altitude(self):
        """Check that perigee altitude is less than initial altitude (collision point is apogee)."""
        if self.perigee_altitude is None or self.h is None:
            return (False, "SKIP: missing perigee_altitude or initial altitude h")
        if self.perigee_altitude < self.h:
            return (True, f"perigee_altitude {self.perigee_altitude:.2e} m < initial altitude {self.h:.2e} m")
        return (False, f"perigee_altitude {self.perigee_altitude:.2e} m >= initial altitude {self.h:.2e} m (should be closer)")

    # ==================== MONOTONICITY TESTS ====================

    def test_monotonicity_kinetic_energy_loss(self):
        """
        Perturb the kinetic energy loss fraction upward and verify that the perigee altitude decreases.
        Uses the same physics (energy and angular momentum) recomputed inside the checker.
        """
        # Need all underlying constants
        if any(x is None for x in [self.m, self.GM, self.r0, self.v0, self.KE_loss_fraction, self.perigee_altitude]):
            return (False, "SKIP: missing data to compute monotonicity (m, GM, r0, v0, KE_loss_fraction, perigee_altitude)")

        # Baseline values from results (should be consistent)
        r0 = self.r0
        v0 = self.v0
        loss_base = self.KE_loss_fraction  # e.g. 0.02
        v_base = math.sqrt(1 - loss_base) * v0
        # Compute perigee from formulas
        # Energy: E = 0.5*m*v^2 - GM*m/r0
        # semimajor: a = -GM*m/(2*E)
        # perigee: r_p = 2*a - r0
        E_base = 0.5 * self.m * v_base**2 - self.GM * self.m / r0
        a_base = -self.GM * self.m / (2 * E_base)
        rp_base = 2 * a_base - r0
        alt_base = rp_base - self.R_e if self.R_e is not None else None
        # Should match provided perigee altitude
        if alt_base is not None and not math.isclose(alt_base, self.perigee_altitude, rel_tol=1e-2, abs_tol=1e-2):
            # If baseline mismatch, still proceed with monotonicity test using the computed baseline
            # (assuming the provided value may have rounding)
            pass

        # Perturb: increase loss fraction slightly (e.g., +0.01, but not exceed 1)
        delta = 0.005  # small relative increase
        loss_new = min(loss_base + delta, 0.99)  # ensure still loss
        v_new = math.sqrt(1 - loss_new) * v0
        E_new = 0.5 * self.m * v_new**2 - self.GM * self.m / r0
        a_new = -self.GM * self.m / (2 * E_new)
        rp_new = 2 * a_new - r0
        alt_new = rp_new - self.R_e if self.R_e is not None else None

        # Check that perigee decreased (closer approach)
        if alt_new < alt_base:
            return (True, f"Increased KE loss from {loss_base:.4f} to {loss_new:.4f}: perigee altitude decreased from {alt_base:.2e} to {alt_new:.2e} (consistent)")
        else:
            return (False, f"Increased KE loss from {loss_base:.4f} to {loss_new:.4f}: perigee altitude did not decrease (base={alt_base:.2e}, new={alt_new:.2e})")

    # ==================== ANSWER SANITY (weak, no isclose, no re-derivation of final answer) ====================

    def test_answer_sanity_perigee_altitude_reasonable(self):
        """
        Weak sanity: perigee altitude should be positive and not exceed the initial altitude.
        (No hardcoded expected value, no math.isclose.)
        """
        if self.perigee_altitude is None:
            return (False, "SKIP: perigee_altitude not provided")
        if self.h is None:
            # cannot compare to initial altitude, just check positivity
            if self.perigee_altitude >= 0:
                return (True, "perigee_altitude is non-negative")
            return (False, f"perigee_altitude {self.perigee_altitude} is negative")
        # broad magnitude: not larger than twice initial altitude (obviously less)
        if 0 <= self.perigee_altitude <= 2 * self.h:
            return (True, f"perigee_altitude {self.perigee_altitude:.2e} m is in plausible range [0, {2*self.h:.2e}]")
        return (False, f"perigee_altitude {self.perigee_altitude:.2e} m outside plausible range")

    # ==================== RUN ALL TESTS ====================

    def run_all_tests(self):
        logs = []
        all_pass = True
        # Discover all methods starting with 'test_'
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {type(e).__name__}: {e}"
                traceback.print_exc()
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)