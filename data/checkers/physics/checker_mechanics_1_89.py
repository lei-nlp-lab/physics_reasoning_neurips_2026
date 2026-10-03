import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Auditor for a simple harmonic motion problem where a stone is dropped into a tunnel through Earth.
    The goal is to find the initial velocity v0 such that the stone reaches the opposite side in T1/2 time.
    """

    def __init__(self, solution_instance):
        # Extract givens: omega0 and R (try attributes first, then within a 'givens' dict)
        self.omega0 = getattr(solution_instance, 'omega0', None)
        self.R = getattr(solution_instance, 'R', None)
        if self.omega0 is None:
            givens = getattr(solution_instance, 'givens', {})
            self.omega0 = givens.get('omega0', None)
        if self.R is None:
            givens = getattr(solution_instance, 'givens', {})
            self.R = givens.get('R', None)

        # Extract result: v0 from results dict (pseudocode emit key)
        results = getattr(solution_instance, 'results', {})
        self.v0 = results.get('v0', None)

    # ----------------------------------------------------------------------
    # Helper: compute the first time t>0 when x(t) = -R for given v0, omega, R
    # ----------------------------------------------------------------------
    @staticmethod
    def _time_to_opposite_side(v0, omega, R):
        """
        Compute the time when stone first reaches x = -R.
        Uses the analytic formula from the phase representation:
            x(t) = A cos(ω t + φ)
        with A = sqrt(R^2 + (v0/ω)^2), φ = arctan2(v0/ω, R)
        Then solve cos(ω t + φ) = -R/A.
        First positive solution: ω t = arccos(-R/A) - φ   (must be in (0, π) range)
        """
        A = math.sqrt(R**2 + (v0/omega)**2)
        phi = math.atan2(v0/omega, R)   # in (0, pi/2)
        cos_theta = -R / A
        # Clamp to [-1,1] to avoid numerical issues
        if cos_theta < -1:
            cos_theta = -1
        if cos_theta > 1:
            cos_theta = 1
        theta = math.acos(cos_theta)    # in [0, π]
        omega_t = theta - phi
        if omega_t <= 0:
            # Should not happen for positive v0, but fallback to next branch
            omega_t += 2 * math.pi
        return omega_t / omega

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_givens(self):
        """Check that required givens (omega0, R) are present and numeric."""
        if self.omega0 is None:
            return (False, "SKIP: omega0 not provided")
        if self.R is None:
            return (False, "SKIP: R not provided")
        if not isinstance(self.omega0, (int, float)):
            return (False, f"omega0 must be numeric, got {type(self.omega0)}")
        if not isinstance(self.R, (int, float)):
            return (False, f"R must be numeric, got {type(self.R)}")
        if self.omega0 <= 0:
            return (False, f"omega0 must be positive, got {self.omega0}")
        if self.R <= 0:
            return (False, f"R must be positive, got {self.R}")
        return (True, "omega0 and R present and numeric")

    def test_interface_results(self):
        """Check that result v0 is present and numeric."""
        if self.v0 is None:
            return (False, "SKIP: v0 not provided in results")
        if not isinstance(self.v0, (int, float)):
            return (False, f"v0 must be numeric, got {type(self.v0)}")
        return (True, "v0 present and numeric")

    # ----------------------------------------------------------------------
    # Physics invariant tests (theorem-based)
    # ----------------------------------------------------------------------
    def test_invariant_shm_ode(self):
        """Verify that the given solution satisfies the SHM ODE: d²x/dt² + ω0² x = 0."""
        if self.omega0 is None or self.R is None or self.v0 is None:
            return (False, "SKIP: insufficient data to check ODE")
        t = sp.Symbol('t', real=True)
        omega = sp.symbols('omega', real=True, positive=True)
        x_expr = self.R * sp.cos(omega * t) - (self.v0 / omega) * sp.sin(omega * t)
        d2x = sp.diff(x_expr, t, 2)
        lhs = d2x + omega**2 * x_expr
        simplified = sp.simplify(lhs)
        test_times = [0.0, self.R / self.omega0, 1.0 / self.omega0]
        for t_val in test_times:
            numeric_lhs = simplified.subs({omega: self.omega0, t: t_val}).evalf()
            if not math.isclose(float(numeric_lhs), 0.0, abs_tol=1e-9):
                return (False, f"ODE not satisfied at t={t_val}: LHS = {numeric_lhs}")
        return (True, "x(t) satisfies d²x/dt² + ω0² x = 0")

    def test_invariant_energy_conservation(self):
        """Check conservation of mechanical energy (kinetic + potential) at multiple times."""
        if self.omega0 is None or self.R is None or self.v0 is None:
            return (False, "SKIP: insufficient data for energy check")
        # E0 = v² + ω0² r² at t=0: r=R, v=-v0
        E0 = self.v0**2 + self.omega0**2 * self.R**2
        T1_over_4 = math.pi / (4 * self.omega0)
        t = T1_over_4
        r_t = self.R * math.cos(self.omega0 * t) - (self.v0 / self.omega0) * math.sin(self.omega0 * t)
        v_t = -self.R * self.omega0 * math.sin(self.omega0 * t) - self.v0 * math.cos(self.omega0 * t)
        E_t = v_t**2 + self.omega0**2 * r_t**2
        if not math.isclose(E0, E_t, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Energy not conserved: t=0 energy {E0}, t=T1/4 energy {E_t}")
        return (True, "Energy conserved (within tolerance)")

    def test_invariant_amplitude_relation(self):
        """Check that the amplitude sqrt(R² + (v0/ω0)²) is consistent with the condition at T1/2."""
        if self.omega0 is None or self.R is None or self.v0 is None:
            return (False, "SKIP: insufficient data for amplitude check")
        A = math.sqrt(self.R**2 + (self.v0 / self.omega0)**2)
        # Condition from phase derivation: v0/ω0 = R
        if not math.isclose(self.v0 / self.omega0, self.R, rel_tol=1e-2):
            return (False, f"Required consistency v0/ω0 ≈ R fails: v0/ω0 = {self.v0/self.omega0}, R = {self.R}")
        return (True, "Amplitude and phase condition consistent with target time")

    # ----------------------------------------------------------------------
    # Modeling / trace checks
    # ----------------------------------------------------------------------
    def test_modeling_initial_position(self):
        """Check that at t=0 the position equals R (surface)."""
        if self.R is None:
            return (False, "SKIP: R not provided")
        if self.v0 is not None and self.omega0 is not None:
            x0 = self.R * math.cos(0) - (self.v0 / self.omega0) * math.sin(0)
            if not math.isclose(x0, self.R, rel_tol=1e-9):
                return (False, f"x(0) = {x0} != R = {self.R}")
        if self.R is not None and self.R <= 0:
            return (False, "R must be positive (surface)")
        return (True, "Initial position consistent")

    def test_modeling_initial_velocity(self):
        """Check that at t=0 the velocity equals -v0."""
        if self.v0 is None or self.omega0 is None or self.R is None:
            return (False, "SKIP: insufficient data for initial velocity check")
        v0_calc = -self.R * self.omega0 * math.sin(0) - self.v0 * math.cos(0)
        if not math.isclose(v0_calc, -self.v0, rel_tol=1e-9, abs_tol=1e-12):
            return (False, f"v(0) = {v0_calc} != -v0 = {-self.v0}")
        return (True, "Initial velocity matches -v0")

    def test_modeling_final_condition(self):
        """Check that at T2 = T1/2 = π/(2ω0) the position equals -R."""
        if self.omega0 is None or self.R is None or self.v0 is None:
            return (False, "SKIP: insufficient data for final condition")
        T2 = math.pi / (2 * self.omega0)
        x_T2 = self.R * math.cos(self.omega0 * T2) - (self.v0 / self.omega0) * math.sin(self.omega0 * T2)
        if not math.isclose(x_T2, -self.R, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"x(T2) = {x_T2} != -R = {-self.R}")
        return (True, "Reaches opposite surface at T2")

    def test_modeling_phase_condition(self):
        """Check the key derived condition: v0 = R ω0 (necessary for the given T2)."""
        if self.omega0 is None or self.R is None or self.v0 is None:
            return (False, "SKIP: insufficient data for phase condition")
        required_v0 = self.R * self.omega0
        if not math.isclose(self.v0, required_v0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"v0 = {self.v0} but requirement from phase condition: v0 = {required_v0}")
        return (True, "Phase condition satisfied: v0 = R ω0")

    # ----------------------------------------------------------------------
    # Monotonicity tests
    # ----------------------------------------------------------------------
    def test_monotonicity_time_increases_with_v0(self):
        """
        Verify that the time to reach the opposite side (-R) decreases as v0 increases.
        We compare baseline v0 with a perturbed v0 (10% larger) and check that the perturbed time
        is smaller than the baseline time (which should be T2 if v0 is correct).
        """
        if self.omega0 is None or self.R is None or self.v0 is None:
            return (False, "SKIP: insufficient data for monotonicity test")
        # Baseline time using the given v0
        t_base = self._time_to_opposite_side(self.v0, self.omega0, self.R)
        # Perturbed v0 larger by 10% (but must stay positive)
        v0_large = 1.1 * self.v0
        t_large = self._time_to_opposite_side(v0_large, self.omega0, self.R)
        # For large v0, time should be smaller (monotonic decreasing)
        if t_large >= t_base:
            return (False, f"Time for v0={v0_large:.3f} ({t_large:.3f}) should be less than time for v0={self.v0:.3f} ({t_base:.3f})")
        # For perturbed v0 smaller by 10%, time should be larger
        v0_small = 0.9 * self.v0
        t_small = self._time_to_opposite_side(v0_small, self.omega0, self.R)
        if t_small <= t_base:
            return (False, f"Time for v0={v0_small:.3f} ({t_small:.3f}) should be larger than time for v0={self.v0:.3f} ({t_base:.3f})")
        return (True, "Time decreases monotonically with v0 (correct direction)")

    # ----------------------------------------------------------------------
    # Weak answer sanity checks (only simple properties, no re-derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_v0_properties(self):
        """Sanity: v0 should be positive and finite."""
        if self.v0 is None:
            return (False, "SKIP: v0 not available")
        if not math.isfinite(self.v0):
            return (False, f"v0 is not finite: {self.v0}")
        if self.v0 <= 0:
            return (False, f"v0 must be positive (thrown downward), got {self.v0}")
        # Broad magnitude check relative to R*ω0 (if available)
        if self.omega0 is not None and self.R is not None:
            typical = self.R * self.omega0
            if typical > 0:
                ratio = self.v0 / typical
                if not (0.5 < ratio < 2.0):
                    return (False, f"v0 / (R ω0) = {ratio} far from 1, physically suspect")
        return (True, "v0 is positive, finite, and within plausible range")

    # ----------------------------------------------------------------------
    # The required run_all_tests method
    # ----------------------------------------------------------------------
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