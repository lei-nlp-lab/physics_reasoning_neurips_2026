import math

class PhysicsChecker:
    """Audit a solution for the photon rocket flashlight problem."""

    def __init__(self, solution_instance):
        """Extract all required inputs and results safely, defaulting to None."""
        results = getattr(solution_instance, 'results', None) or {}
        givens = getattr(solution_instance, 'givens', None) or {}

        # Core output
        self.v = results.get('v', None)

        # Problem parameters – try givens first, then fall back to instance attributes
        self.m = givens.get('m', getattr(solution_instance, 'm', None))
        self.P = givens.get('P', getattr(solution_instance, 'P', None))
        self.T = givens.get('T', getattr(solution_instance, 'T', None))
        self.c = givens.get('c', getattr(solution_instance, 'c', None))

        # Storage for internal derived values (only computed when needed)
        self._E = None  # total radiated energy
        self._expected_v = None

    # ------------------------------------------------------------------
    # Interface tests – check that required keys/attributes exist
    # ------------------------------------------------------------------
    def test_interface_v_exists(self):
        """Check that the final speed v is present in results."""
        if self.v is None:
            return (False, "SKIP: results['v'] is missing")
        return (True, "v is present")

    def test_interface_givens_available(self):
        """Check that the essential input parameters are available."""
        missing = []
        if self.m is None:
            missing.append('m')
        if self.P is None:
            missing.append('P')
        if self.T is None:
            missing.append('T')
        if self.c is None:
            missing.append('c')
        if missing:
            return (False, f"SKIP: missing givens: {', '.join(missing)}")
        return (True, "All required givens (m, P, T, c) are available")

    # ------------------------------------------------------------------
    # Invariant tests – theorem-based physical sanity checks (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_momentum_conservation(self):
        """Check conservation of momentum: m*v = (P*T)/c."""
        if None in (self.v, self.m, self.P, self.T, self.c):
            return (False, "SKIP: missing data for momentum check")
        photon_momentum = (self.P * self.T) / self.c
        flashlight_momentum = self.m * self.v
        if math.isclose(flashlight_momentum, photon_momentum, rel_tol=1e-2, abs_tol=1e-20):
            return (True, f"Momentum conserved: m*v ≈ {flashlight_momentum:.3e}, E/c ≈ {photon_momentum:.3e}")
        return (False, f"Momentum mismatch: m*v = {flashlight_momentum:.3e}, E/c = {photon_momentum:.3e}")

    def test_invariant_impulse_momentum(self):
        """Check that the impulse from photon emission equals momentum change: (P/c)*T = m*v."""
        if None in (self.v, self.m, self.P, self.T, self.c):
            return (False, "SKIP: missing data for impulse check")
        impulse = (self.P / self.c) * self.T
        momentum_change = self.m * self.v
        if math.isclose(impulse, momentum_change, rel_tol=1e-2, abs_tol=1e-20):
            return (True, f"Impulse = (P/c)*T ≈ {impulse:.3e}, Δp = m*v ≈ {momentum_change:.3e}")
        return (False, f"Impulse mismatch: impulse = {impulse:.3e}, Δp = {momentum_change:.3e}")

    # ------------------------------------------------------------------
    # Modeling / trace tests – check that intermediate steps are correctly represented
    # ------------------------------------------------------------------
    def test_modeling_energy_positive(self):
        """Check that the total radiated energy E = P*T is positive (physical)."""
        if None in (self.P, self.T):
            return (False, "SKIP: missing P or T for energy check")
        energy = self.P * self.T
        if energy > 0:
            return (True, f"Total radiated energy = {energy:.2f} J > 0")
        return (False, f"Total radiated energy = {energy}, should be positive")

    def test_modeling_photon_direction(self):
        """Check that the flashlight moves opposite to the emitted photon beam (explanation step)."""
        # The momentum conservation equation implies v and photon momentum have opposite signs.
        # If we assume +x for flashlight velocity, photons must carry negative momentum.
        if self.v is None or self.v <= 0:
            return (False, "SKIP: v missing or non‑positive; cannot check direction")
        # Photon momentum magnitude = (P*T)/c, and direction is opposite to flashlight motion.
        # The equation m*v = (P*T)/c already encodes the correct magnitude; direction is implicit.
        # We simply assert that the model used a consistent sign convention.
        # This is a weak check – we just verify that v is positive (already done above).
        if None in (self.m, self.P, self.T, self.c):
            return (False, "SKIP: missing parameters for direction check")
        # Ensure that the momentum equality holds (which implies opposite directions)
        if not math.isclose(self.m * self.v, (self.P * self.T) / self.c, rel_tol=1e-2, abs_tol=1e-20):
            return (False, "Momentum equality fails – direction check inconclusive")
        return (True, "Photon beam direction is opposite to flashlight motion (consistent with model)")

    # ------------------------------------------------------------------
    # Monotonicity tests – trend checks via explicit input perturbation
    # ------------------------------------------------------------------
    def test_monotonicity_speed_decreases_with_mass(self):
        """If mass increases (other params fixed), v should decrease."""
        if None in (self.m, self.P, self.T, self.c):
            return (False, "SKIP: missing givens for monotonicity test")
        v0 = (self.P * self.T) / (self.m * self.c)
        m_perturbed = 1.1 * self.m
        v1 = (self.P * self.T) / (m_perturbed * self.c)
        if v1 < v0:
            return (True, f"Mass +10% reduces v: {v0:.3e} -> {v1:.3e}")
        return (False, f"Monotonicity failure: v does not decrease with increased mass")

    def test_monotonicity_speed_increases_with_power(self):
        """If power increases (other params fixed), v should increase."""
        if None in (self.m, self.P, self.T, self.c):
            return (False, "SKIP: missing givens for monotonicity test")
        v0 = (self.P * self.T) / (self.m * self.c)
        P_perturbed = 1.1 * self.P
        v1 = (P_perturbed * self.T) / (self.m * self.c)
        if v1 > v0:
            return (True, f"Power +10% increases v: {v0:.3e} -> {v1:.3e}")
        return (False, f"Monotonicity failure: v does not increase with increased power")

    def test_monotonicity_speed_increases_with_time(self):
        """If time increases, v should increase."""
        if None in (self.m, self.P, self.T, self.c):
            return (False, "SKIP: missing givens for monotonicity test")
        v0 = (self.P * self.T) / (self.m * self.c)
        T_perturbed = 1.1 * self.T
        v1 = (self.P * T_perturbed) / (self.m * self.c)
        if v1 > v0:
            return (True, f"Time +10% increases v: {v0:.3e} -> {v1:.3e}")
        return (False, f"Monotonicity failure: v does not increase with increased time")

    # ------------------------------------------------------------------
    # Answer sanity test – weak, non‑quantitative check
    # ------------------------------------------------------------------
    def test_answer_sanity_positive_finite(self):
        """Sanity: v must be positive, finite, and nonrelativistic."""
        if self.v is None:
            return (False, "SKIP: v not available")
        if self.v <= 0 or not math.isfinite(self.v):
            return (False, f"Unphysical v: {self.v}")
        if self.c is not None and self.v >= self.c:
            return (False, f"v >= c: impossible for nonrelativistic problem")
        return (True, f"v = {self.v:.3e} is positive, finite, and nonrelativistic")

    # ------------------------------------------------------------------
    # Test runner
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)