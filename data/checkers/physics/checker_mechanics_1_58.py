import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results dictionary safely
        self.results = getattr(solution_instance, 'results', None)
        if self.results is None:
            self.results = {}

        # Extract optional givens from solution_instance.givens or as top-level attributes
        givens = getattr(solution_instance, 'givens', None) or {}
        # Jump speed
        self.v_jump = (
            givens.get('v_jump') or
            getattr(solution_instance, 'v_jump', None) or
            None
        )
        # Density (could be under 'density' or 'rho')
        self.rho = (
            givens.get('rho') or
            givens.get('density') or
            getattr(solution_instance, 'rho', None) or
            getattr(solution_instance, 'density', None) or
            None
        )
        # Gravitational constant (standard physical constant, not from solution)
        self.G = 6.674e-11

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_R_max_present(self):
        """Check that 'R_max' key exists in results and is numeric."""
        if self.results is None:
            return (False, "FAIL: results dict is missing")
        if 'R_max' not in self.results:
            return (False, "FAIL: 'R_max' key missing in results")
        R = self.results['R_max']
        if not isinstance(R, (int, float)):
            return (False, f"FAIL: R_max is not numeric, got {type(R).__name__}")
        return (True, "PASS: R_max exists and is numeric")

    # ------------------------------------------------------------------
    # Theorem-based invariant tests (at least two)
    # ------------------------------------------------------------------
    def test_invariant_escape_condition(self):
        """Check that R_max satisfies the escape condition:
           v_jump = sqrt(8πGρ/3) * R_max (the combined energy‑mass relation).
        """
        R = self.results.get('R_max')
        if R is None:
            return (False, "SKIP: R_max not available")
        if self.v_jump is None or self.rho is None:
            return (False, "SKIP: v_jump or rho not provided for invariant test")
        # Compute coefficient k = sqrt(8πGρ/3)
        k = math.sqrt((8.0/3.0) * math.pi * self.G * self.rho)
        # Expected R from given v_jump
        R_expected = self.v_jump / k
        # Allow 10% relative tolerance and absolute 10 m
        if math.isclose(R, R_expected, rel_tol=0.1, abs_tol=10.0):
            return (True, f"PASS: R_max = {R} ≈ {R_expected:.2f}")
        else:
            return (False, f"FAIL: R_max = {R} does not match expected {R_expected:.2f} (rel diff {abs(R-R_expected)/R_expected:.2%})")

    def test_invariant_mass_energy_consistency(self):
        """Check that the asteroid mass derived from R_max and ρ satisfies
           the escape energy condition: ½mv² = GMm/R.
        """
        R = self.results.get('R_max')
        if R is None:
            return (False, "SKIP: R_max not available")
        if self.v_jump is None or self.rho is None:
            return (False, "SKIP: v_jump or rho missing for mass‑energy check")
        # Compute mass from assumed uniform sphere
        M = (4.0/3.0) * math.pi * (R ** 3) * self.rho
        # Escape velocity from surface
        v_esc = math.sqrt(2.0 * self.G * M / R)
        # Check that v_jump is approximately >= v_esc (within 10% relative)
        if v_esc > 0 and math.isclose(self.v_jump, v_esc, rel_tol=0.1, abs_tol=0.1):
            return (True, f"PASS: v_jump = {self.v_jump} ≈ v_esc = {v_esc:.2f}")
        elif self.v_jump >= v_esc * 0.9:  # allow slightly below due to rounding
            return (True, f"PASS: v_jump ({self.v_jump}) ≥ 0.9*v_esc ({v_esc:.2f})")
        else:
            return (False, f"FAIL: v_jump = {self.v_jump} not near v_esc = {v_esc:.2f}")

    # ------------------------------------------------------------------
    # Modeling/trace quality checks (at least two)
    # ------------------------------------------------------------------
    def test_modeling_positive_and_finite(self):
        """Check that R_max is positive and finite (basic Physical plausibility)."""
        R = self.results.get('R_max')
        if R is None:
            return (False, "SKIP: R_max not available")
        if not isinstance(R, (int, float)):
            return (False, f"FAIL: R_max type {type(R).__name__}")
        if not math.isfinite(R):
            return (False, "FAIL: R_max is infinite or NaN")
        if R <= 0:
            return (False, f"FAIL: R_max = {R} is not positive")
        return (True, f"PASS: R_max = {R} is positive and finite")

    def test_modeling_consistent_with_typical_params(self):
        """Check that R_max is within a factor 2 of the value obtained from
           typical literature parameters (v_jump = 2.7 m/s, ρ = 2000 kg/m³).
        """
        R = self.results.get('R_max')
        if R is None:
            return (False, "SKIP: R_max not available")
        # Typical values from problem statement
        v_typical = 2.7      # m/s
        rho_typical = 2000   # kg/m³
        k = math.sqrt((8.0/3.0) * math.pi * self.G * rho_typical)
        R_typical = v_typical / k   # ≈ 2.67e3 m
        if R_typical <= 0:
            return (False, "FAIL: computed typical R is zero or negative")
        ratio = R / R_typical
        # Accept ratio between 0.5 and 2 (i.e., within factor 2)
        if 0.5 <= ratio <= 2.0:
            return (True, f"PASS: R_max = {R} is within factor 2 of typical {R_typical:.2f} (ratio = {ratio:.2f})")
        else:
            return (False, f"FAIL: R_max = {R} is outside factor 2 of typical {R_typical:.2f} (ratio = {ratio:.2f})")

    def test_modeling_escape_velocity_plausibility(self):
        """Check that the escape velocity computed from R_max and ρ (if given)
           falls within the plausible human jump speed range (1–5 m/s).
        """
        R = self.results.get('R_max')
        if R is None:
            return (False, "SKIP: R_max not available")
        if self.rho is None:
            return (False, "SKIP: density ρ not provided; cannot compute v_esc")
        # Compute mass from uniform sphere
        M = (4.0/3.0) * math.pi * (R ** 3) * self.rho
        v_esc = math.sqrt(2.0 * self.G * M / R)
        if not math.isfinite(v_esc):
            return (False, "FAIL: v_esc is infinite or NaN")
        # Plausible human jump speed range: 1–5 m/s (allowing some margin)
        if 1.0 <= v_esc <= 5.0:
            return (True, f"PASS: v_esc = {v_esc:.2f} m/s is in plausible range [1, 5] m/s")
        else:
            return (False, f"FAIL: v_esc = {v_esc:.2f} m/s is outside plausible range [1, 5] m/s")

    # ------------------------------------------------------------------
    # Monotonicity test (one required)
    # ------------------------------------------------------------------
    def test_monotonicity_R_max_increases_with_v_jump(self):
        """Perturb v_jump upward by 10% and check that R_max increases."""
        R = self.results.get('R_max')
        if R is None:
            return (False, "SKIP: R_max not available")
        if self.v_jump is None or self.rho is None:
            return (False, "SKIP: v_jump/rho missing for monotonicity")
        # Compute coefficient k = sqrt(8πGρ/3)
        k = math.sqrt((8.0/3.0) * math.pi * self.G * self.rho)
        # Original R_max from results
        R_orig = R
        # New R_max with 10% larger v_jump
        v_new = self.v_jump * 1.1
        R_new = v_new / k
        # Expect R_new > R_orig
        if R_new > R_orig:
            return (True, f"PASS: Increasing v_jump by 10% gives R_new = {R_new:.2f} > R_orig = {R_orig:.2f}")
        else:
            return (False, f"FAIL: Monotonicity violated: R_new = {R_new:.2f} <= R_orig = {R_orig:.2f}")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one, no hardcoded target values)
    # ------------------------------------------------------------------
    def test_answer_sanity_magnitude(self):
        """Check that R_max is within plausible asteroid radius range (1 m .. 100 km)."""
        R = self.results.get('R_max')
        if R is None:
            return (False, "SKIP: R_max not available")
        if not isinstance(R, (int, float)):
            return (False, f"FAIL: R_max not numeric")
        if not math.isfinite(R):
            return (False, "FAIL: R_max infinite or NaN")
        if R < 1.0 or R > 1.0e5:
            return (False, f"FAIL: R_max = {R} is outside plausible range [1, 1e5] m")
        return (True, f"PASS: R_max = {R} is within 1 m .. 100 km")

    # ------------------------------------------------------------------
    # Runner: discover all test_* methods
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Iterate over all bound methods starting with 'test_'
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name, None)
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