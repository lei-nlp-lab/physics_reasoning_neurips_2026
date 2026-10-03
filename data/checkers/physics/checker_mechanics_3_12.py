import math
import traceback

class PhysicsChecker:
    """
    Audits a relativistic radar problem solution.
    Expected result keys:
        distance_Earth_shipframe, beta, v, x_C_earth
    """

    def __init__(self, solution_instance):
        # Hard-coded givens from problem statement (allowed for re-derivation inside checker)
        self.delta_tau = 40.0               # seconds
        self.freq_ratio = 0.5               # f_r / f_t
        self.c = 3e8                        # m/s
        self.expected_beta = 1.0 / 3.0
        self.expected_gamma = 1.0 / math.sqrt(1.0 - self.expected_beta**2)

        # Safely extract results from the solution
        results = getattr(solution_instance, 'results', {})
        self.distance_Earth_shipframe = results.get('distance_Earth_shipframe', None)
        self.beta = results.get('beta', None)
        self.v = results.get('v', None)
        self.x_C_earth = results.get('x_C_earth', None)

    # ------------------------------------------------------------------
    # Interface checks (required keys exist and are numeric)
    # ------------------------------------------------------------------
    def test_interface_distance_key(self):
        if self.distance_Earth_shipframe is None:
            return (False, "distance_Earth_shipframe missing")
        if not isinstance(self.distance_Earth_shipframe, (int, float)):
            return (False, "distance_Earth_shipframe not numeric")
        return (True, "distance_Earth_shipframe present")

    def test_interface_beta_key(self):
        if self.beta is None:
            return (False, "beta missing")
        if not isinstance(self.beta, (int, float)):
            return (False, "beta not numeric")
        if not (0 < self.beta < 1):
            return (False, "beta out of (0,1) range")
        return (True, "beta present and in (0,1)")

    def test_interface_v_key(self):
        if self.v is None:
            return (False, "v missing")
        if not isinstance(self.v, (int, float)):
            return (False, "v not numeric")
        if self.v <= 0:
            return (False, "v should be positive")
        return (True, "v present and positive")

    def test_interface_x_C_key(self):
        if self.x_C_earth is None:
            return (False, "x_C_earth missing")
        if not isinstance(self.x_C_earth, (int, float)):
            return (False, "x_C_earth not numeric")
        if self.x_C_earth <= 0:
            return (False, "x_C_earth should be positive")
        return (True, "x_C_earth present and positive")

    # ------------------------------------------------------------------
    # Theorem-based invariant checks (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_doppler_receding_mirror(self):
        if self.beta is None:
            return (False, "SKIP: beta not available")
        # Relativistic Doppler for receding source with mirror: (1-β)/(1+β) = f_r/f_t
        lhs = (1.0 - self.beta) / (1.0 + self.beta)
        if math.isclose(lhs, self.freq_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Doppler ratio = {lhs:.4f} ≈ 0.5")
        else:
            return (False, f"Doppler ratio = {lhs:.4f}, expected 0.5")

    def test_invariant_round_trip_time(self):
        if self.distance_Earth_shipframe is None:
            return (False, "SKIP: distance not available")
        # In ship frame, pulse travels to Earth in Δτ/2 = 20 s → |x'| = 20c
        expected = self.c * self.delta_tau / 2.0
        if math.isclose(abs(self.distance_Earth_shipframe), expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"|distance| ≈ 20c = {expected:.3e} m")
        else:
            return (False, f"|distance| = {abs(self.distance_Earth_shipframe):.3e}, expected {expected:.3e}")

    def test_invariant_lorentz_x_C(self):
        """
        In ship frame, event C (reception) has t'=40 s, x'=0.
        Lorentz transform to Earth frame: x_C = γ (0 + v t') = γ v 40
        """
        if self.beta is None or self.v is None or self.x_C_earth is None:
            return (False, "SKIP: missing beta, v, or x_C_earth")
        gamma = 1.0 / math.sqrt(1.0 - self.beta**2)
        expected = gamma * self.v * self.delta_tau   # v = β c, so expected = γ β c Δτ
        if math.isclose(self.x_C_earth, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"x_C ≈ γ v Δτ = {expected:.3e} m")
        else:
            return (False, f"x_C = {self.x_C_earth:.3e}, expected {expected:.3e}")

    # ------------------------------------------------------------------
    # Modeling / trace consistency checks (use isclose)
    # ------------------------------------------------------------------
    def test_modeling_velocity(self):
        if self.beta is None or self.v is None:
            return (False, "SKIP: beta or v missing")
        expected_v = self.beta * self.c
        if math.isclose(self.v, expected_v, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"v = βc = {expected_v:.3e} m/s")
        else:
            return (False, f"v = {self.v:.3e}, expected {expected_v:.3e}")

    def test_modeling_reflection_distance_derivation(self):
        """
        The reflection event in the ship frame occurs at t' = Δτ/2 = 20 s.
        The distance to Earth at that moment is c * (Δτ/2) = 20c.
        Check that the solution's distance_Earth_shipframe matches this.
        """
        if self.distance_Earth_shipframe is None:
            return (False, "SKIP: distance not available")
        expected = self.c * self.delta_tau / 2.0
        if math.isclose(abs(self.distance_Earth_shipframe), expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Reflection distance = 20c = {expected:.3e} m (consistent with Δτ/2)")
        else:
            return (False, f"Reflection distance = {abs(self.distance_Earth_shipframe):.3e}, expected {expected:.3e}")

    def test_modeling_lorentz_reception(self):
        """
        The reception event in the ship frame is (t'=40, x'=0).
        Lorentz transform to Earth frame:
            t_C = γ t',   x_C = γ v t'
        This should match the solution's x_C_earth.
        """
        if self.beta is None or self.v is None or self.x_C_earth is None:
            return (False, "SKIP: missing beta, v, or x_C_earth")
        gamma = 1.0 / math.sqrt(1.0 - self.beta**2)
        expected_x_C = gamma * self.v * self.delta_tau   # = γ β c 40
        if math.isclose(self.x_C_earth, expected_x_C, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"x_C = γ v Δτ = {expected_x_C:.3e} m (Lorentz transform consistent)")
        else:
            return (False, f"x_C = {self.x_C_earth:.3e}, expected {expected_x_C:.3e}")

    # ------------------------------------------------------------------
    # Monotonicity checks (perturb an input and compare)
    # ------------------------------------------------------------------
    def test_monotonicity_distance_vs_delta_tau(self):
        """
        Baseline: Δτ = 40 s → distance_Earth_shipframe = 20c.
        If Δτ were larger (e.g., 44 s), distance should be larger.
        """
        if self.distance_Earth_shipframe is None:
            return (False, "SKIP: distance not available")
        baseline = abs(self.distance_Earth_shipframe)
        perturbed_delta_tau = self.delta_tau * 1.1  # 44 s
        expected_perturbed = self.c * perturbed_delta_tau / 2.0  # 22c
        if expected_perturbed > baseline:
            return (True, "distance increases with Δτ (monotonicity ok)")
        else:
            return (False, "distance does not increase with Δτ")

    def test_monotonicity_x_C_vs_delta_tau(self):
        """
        x_C_earth = 2 β c γ (Δτ).  Since all factors positive, larger Δτ → larger x_C.
        """
        if self.beta is None or self.x_C_earth is None:
            return (False, "SKIP: beta or x_C_earth missing")
        baseline = self.x_C_earth
        gamma = 1.0 / math.sqrt(1.0 - self.beta**2)
        perturbed_delta_tau = self.delta_tau * 1.1  # 44 s
        expected_perturbed = gamma * self.v * perturbed_delta_tau  # same formula
        if expected_perturbed > baseline:
            return (True, "x_C_earth increases with Δτ (monotonicity ok)")
        else:
            return (False, "x_C_earth does not increase with Δτ")

    def test_monotonicity_beta_vs_freq_ratio(self):
        """
        Baseline: freq_ratio = 0.5 → beta = 1/3.
        If freq_ratio were larger (e.g., 0.6), beta should be smaller.
        """
        if self.beta is None:
            return (False, "SKIP: beta not available")
        baseline_beta = self.beta
        perturbed_ratio = 0.6
        # (1-β)/(1+β) = perturbed_ratio  →  β = (1 - ratio)/(1 + ratio)
        expected_beta = (1.0 - perturbed_ratio) / (1.0 + perturbed_ratio)
        if baseline_beta > expected_beta:
            return (True, "beta decreases with increasing freq_ratio (monotonicity ok)")
        else:
            return (False, "beta does not decrease with increasing freq_ratio")

    # ------------------------------------------------------------------
    # Weak answer sanity check (broad range, sign, finite)
    # ------------------------------------------------------------------
    def test_answer_sanity_broad_range(self):
        issues = []
        # distance_Earth_shipframe ~ 6e9 m → allow ±1e9
        if self.distance_Earth_shipframe is not None:
            d = abs(self.distance_Earth_shipframe)
            if d < 5e9 or d > 7e9:
                issues.append(f"distance = {d:.2e} out of [5e9, 7e9]")
        # v ~ 1e8 m/s → allow factor of 2
        if self.v is not None:
            if self.v < 0.5e8 or self.v > 2e8:
                issues.append(f"v = {self.v:.2e} out of [0.5e8, 2e8]")
        # x_C_earth ~ 8.5e9 m → allow ±2e9
        if self.x_C_earth is not None:
            if self.x_C_earth < 7e9 or self.x_C_earth > 10e9:
                issues.append(f"x_C = {self.x_C_earth:.2e} out of [7e9, 10e9]")
        if not issues:
            return (True, "All broad ranges ok")
        else:
            return (False, "; ".join(issues))

    # ------------------------------------------------------------------
    # Master test runner
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