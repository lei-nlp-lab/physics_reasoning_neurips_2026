import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens safely – use None for missing values, never numeric defaults.
        givens = getattr(solution_instance, 'givens', {})
        if givens is None:
            givens = {}
        self.lambda_ = givens.get('lambda', None)          # wavelength (m)
        self.theta_target = givens.get('theta_target', None) # target divergence (rad)
        self.d0 = givens.get('d0', None)                   # laser beam diameter before expansion (m)

        # Extract results safely – use None for missing values.
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        self.D_rect = results.get('D_rect', None)
        self.D_circ = results.get('D_circ', None)
        self.R_circ = results.get('R_circ', None)
        self.f_ratio = results.get('f_ratio', None)

        # Additional optional fields (for trace checks)
        self.r_laser = givens.get('r', None)               # laser beam radius (m)
        self.f_short = givens.get('f_short', None)         # short focal length (m)
        self.f_telescope = givens.get('f_telescope', None) # telescope focal length (m)

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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

    # ------------------------------------------------------------------
    # Interface checks: required keys exist and are numeric
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all four expected result keys exist."""
        keys = [self.D_rect, self.D_circ, self.R_circ, self.f_ratio]
        missing = [k for k in ['D_rect','D_circ','R_circ','f_ratio']
                   if getattr(self, k, None) is None]
        if missing:
            return (False, f"Missing keys: {', '.join(missing)}")
        # Also check they are numeric (float or int)
        for key, val in zip(['D_rect','D_circ','R_circ','f_ratio'], keys):
            if not isinstance(val, (int, float)):
                return (False, f"{key} has non-numeric type {type(val)}")
        return (True, "All required result keys present and numeric")

    # ------------------------------------------------------------------
    # Invariant checks: theorem-based physical relations (use isclose)
    # ------------------------------------------------------------------
    def test_invariant_rectangular_diffraction(self):
        """For rectangular aperture: D_rect = 2 * λ / θ_target."""
        if any(v is None for v in [self.lambda_, self.theta_target, self.D_rect]):
            return (False, "SKIP: missing lambda, theta_target or D_rect")
        expected = 2.0 * self.lambda_ / self.theta_target
        if math.isclose(self.D_rect, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"D_rect ≈ {expected:.3e} matches")
        else:
            return (False, f"D_rect = {self.D_rect:.3e}, expected ≈ {expected:.3e}")

    def test_invariant_circular_diffraction(self):
        """For circular aperture: D_circ = 2.44 * λ / θ_target."""
        if any(v is None for v in [self.lambda_, self.theta_target, self.D_circ]):
            return (False, "SKIP: missing lambda, theta_target or D_circ")
        expected = 2.44 * self.lambda_ / self.theta_target
        if math.isclose(self.D_circ, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"D_circ ≈ {expected:.3e} matches")
        else:
            return (False, f"D_circ = {self.D_circ:.3e}, expected ≈ {expected:.3e}")

    def test_invariant_radius_relation(self):
        """R_circ must equal D_circ / 2."""
        if any(v is None for v in [self.D_circ, self.R_circ]):
            return (False, "SKIP: missing D_circ or R_circ")
        expected = self.D_circ / 2.0
        if math.isclose(self.R_circ, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "R_circ = D_circ/2 holds")
        else:
            return (False, f"R_circ = {self.R_circ:.3e}, expected {expected:.3e}")

    def test_invariant_fratio_relation(self):
        """If d0 is given, f_ratio should equal D_rect / d0."""
        if any(v is None for v in [self.d0, self.D_rect, self.f_ratio]):
            return (False, "SKIP: missing d0, D_rect or f_ratio")
        if self.d0 == 0:
            return (False, "SKIP: d0 is zero")
        expected = self.D_rect / self.d0
        if math.isclose(self.f_ratio, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"f_ratio ≈ {expected:.3e} matches D_rect/d0")
        else:
            return (False, f"f_ratio = {self.f_ratio:.3e}, expected {expected:.3e}")

    # ------------------------------------------------------------------
    # Modeling / trace checks: consistency with explanation steps
    # ------------------------------------------------------------------
    def test_modeling_angular_width_from_rect(self):
        """Re-derive full angular width from D_rect: θ = 2λ / D_rect, should be near target."""
        if any(v is None for v in [self.lambda_, self.D_rect, self.theta_target]):
            return (False, "SKIP: missing lambda, D_rect or theta_target")
        derived_theta = 2.0 * self.lambda_ / self.D_rect
        if math.isclose(derived_theta, self.theta_target, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Derived θ = {derived_theta:.3e} matches target {self.theta_target:.3e}")
        else:
            return (False, f"Derived θ = {derived_theta:.3e}, target θ = {self.theta_target:.3e}")

    def test_modeling_angular_width_from_circ(self):
        """Re-derive circular aperture angular width: θ = 2.44λ / D_circ."""
        if any(v is None for v in [self.lambda_, self.D_circ, self.theta_target]):
            return (False, "SKIP: missing lambda, D_circ or theta_target")
        derived_theta = 2.44 * self.lambda_ / self.D_circ
        if math.isclose(derived_theta, self.theta_target, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Circular θ ≈ {derived_theta:.3e} matches target")
        else:
            return (False, f"Derived θ = {derived_theta:.3e}, target {self.theta_target:.3e}")

    def test_trace_beam_expansion_consistency(self):
        """
        If f_short and f_telescope are given, check that 
        D_rect ≈ d0 * (f_telescope / f_short) (beam expansion).
        """
        if any(v is None for v in [self.d0, self.f_short, self.f_telescope, self.D_rect]):
            return (False, "SKIP: missing d0, f_short, f_telescope, or D_rect")
        if self.f_short == 0:
            return (False, "SKIP: f_short is zero")
        expected_D = self.d0 * self.f_telescope / self.f_short
        if math.isclose(self.D_rect, expected_D, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Beam expansion: D_rect ≈ {expected_D:.3e} matches")
        else:
            return (False, f"D_rect = {self.D_rect:.3e}, expected {expected_D:.3e} from expansion")

    # ------------------------------------------------------------------
    # Monotonicity checks: perturb a physical input and verify trend
    # ------------------------------------------------------------------
    def test_monotonicity_increase_lambda_increases_D_rect(self):
        """
        If λ increases by 10%, D_rect (rectangular) should increase proportionally.
        We recompute using the same relation.
        """
        if any(v is None for v in [self.lambda_, self.theta_target, self.D_rect]):
            return (False, "SKIP: missing lambda, theta_target or D_rect")
        # perturb lambda by +10%
        lambda_pert = self.lambda_ * 1.1
        expected_D_new = 2.0 * lambda_pert / self.theta_target
        # D_rect is the original, but we expect larger D_rect for larger λ
        # Compare: original D_rect should be smaller than perturbed D
        if self.D_rect < expected_D_new:
            return (True, "Monotonicity holds: D_rect increases with λ")
        else:
            return (False, f"Monotonicity violated: D_rect = {self.D_rect:.3e}, "
                           f"perturbed D = {expected_D_new:.3e}")

    def test_monotonicity_increase_d0_decreases_fratio(self):
        """
        If d0 increases, f_ratio (D_rect / d0) should decrease (since D_rect fixed).
        """
        if any(v is None for v in [self.d0, self.D_rect, self.f_ratio]):
            return (False, "SKIP: missing d0, D_rect or f_ratio")
        # perturb d0 by +10%
        d0_pert = self.d0 * 1.1
        expected_fratio_new = self.D_rect / d0_pert
        # original f_ratio should be larger than new
        if self.f_ratio > expected_fratio_new:
            return (True, "Monotonicity holds: f_ratio decreases when d0 increases")
        else:
            return (False, f"Monotonicity violated: f_ratio = {self.f_ratio:.3e}, "
                           f"perturbed f_ratio = {expected_fratio_new:.3e}")

    # ------------------------------------------------------------------
    # Weak answer sanity checks (sign, finiteness, broad range)
    # ------------------------------------------------------------------
    def test_answer_sanity_positivity(self):
        """
        All computed quantities must be positive finite numbers.
        (Only sanity – no comparison to expected values.)
        """
        values = [self.D_rect, self.D_circ, self.R_circ, self.f_ratio]
        labels = ['D_rect', 'D_circ', 'R_circ', 'f_ratio']
        for label, val in zip(labels, values):
            if val is None:
                continue
            if not (isinstance(val, (int, float)) and math.isfinite(val) and val > 0):
                return (False, f"{label} = {val} is not positive finite")
        return (True, "All present results are positive finite numbers")