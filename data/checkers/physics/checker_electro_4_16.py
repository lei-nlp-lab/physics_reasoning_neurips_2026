import math
import traceback

class PhysicsChecker:
    """
    Audits a solution for the problem of harmonic plane wave incident on a dielectric interface.
    Checks interface completeness, physical invariants, modeling consistency, monotonicity, and answer sanity.
    """

    def __init__(self, solution_instance):
        # Extract givens and results safely
        self.givens = getattr(solution_instance, 'givens', {}) or {}
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', []) or []

        # Givens
        self.nu = self.givens.get('nu', None)
        self.n1 = self.givens.get('n1', None)
        self.n2 = self.givens.get('n2', None)
        self.rho = self.givens.get('rho', None)
        self.E0 = self.givens.get('E0', None)
        self.c = self.givens.get('c', None)          # speed of light in vacuum

        # Results
        self.d_max = self.results.get('d_max', None)
        self.d_min = self.results.get('d_min', None)
        self.phase_change_B = self.results.get('phase_change_B', None)
        self.E2_avg_at_zero = self.results.get('E2_avg_at_zero', None)
        self.B2_avg_at_zero = self.results.get('B2_avg_at_zero', None)
        self.cause = self.results.get('cause', None)

        # Derived wavelength (if possible)
        if self.nu is not None and self.n1 is not None and self.c is not None:
            self.lambda1 = self.c / (self.nu * self.n1)
        else:
            self.lambda1 = None

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all expected result keys exist and are not None."""
        keys = ['d_max', 'd_min', 'phase_change_B', 'E2_avg_at_zero', 'B2_avg_at_zero', 'cause']
        missing = [k for k in keys if self.results.get(k) is None]
        if missing:
            return False, f"Missing or None results keys: {missing}"
        return True, "All required result keys present."

    # ----------------------------------------------------------------------
    # Invariant tests (theorem‑based physical sanity)
    # ----------------------------------------------------------------------
    def test_invariant_phase_change_B(self):
        """The magnetic field should have zero phase change upon reflection."""
        if self.phase_change_B is None:
            return False, "SKIP: phase_change_B not available"
        if self.phase_change_B != 0:
            return False, f"Expected phase_change_B = 0, got {self.phase_change_B}"
        return True, "phase_change_B is 0 (correct)."

    def test_invariant_cause(self):
        """The cause of darkening must be 'Electric field'."""
        if self.cause is None:
            return False, "SKIP: cause not available"
        if not isinstance(self.cause, str):
            return False, f"cause must be a string, got {type(self.cause)}"
        if self.cause.strip().lower() != "electric field":
            return False, f"Expected cause = 'Electric field', got '{self.cause}'"
        return True, "Cause correctly identified as electric field."

    def test_invariant_sign_E2_at_zero(self):
        """⟨E²⟩ at d=0 must be non‑negative."""
        if self.E2_avg_at_zero is None:
            return False, "SKIP: E2_avg_at_zero not available"
        if self.E2_avg_at_zero < 0:
            return False, f"E2_avg_at_zero is negative: {self.E2_avg_at_zero}"
        return True, f"E2_avg_at_zero = {self.E2_avg_at_zero} ≥ 0."

    def test_invariant_sign_B2_at_zero(self):
        """⟨B²⟩ at d=0 must be non‑negative."""
        if self.B2_avg_at_zero is None:
            return False, "SKIP: B2_avg_at_zero not available"
        if self.B2_avg_at_zero < 0:
            return False, f"B2_avg_at_zero is negative: {self.B2_avg_at_zero}"
        return True, f"B2_avg_at_zero = {self.B2_avg_at_zero} ≥ 0."

    def test_invariant_extrema_positions_positive(self):
        """d_max and d_min must be non‑negative (distance from interface)."""
        if self.d_max is not None and self.d_max < 0:
            return False, f"d_max is negative: {self.d_max}"
        if self.d_min is not None and self.d_min < 0:
            return False, f"d_min is negative: {self.d_min}"
        return True, "Extrema positions are non‑negative."

    def test_invariant_extrema_order(self):
        """First maximum should be farther from interface than first minimum (or equal if d_min=0)."""
        if self.d_max is None or self.d_min is None:
            return False, "SKIP: d_max or d_min missing"
        if self.d_max <= self.d_min:
            return False, f"Expected d_max > d_min, got d_max={self.d_max}, d_min={self.d_min}"
        return True, f"d_max > d_min: {self.d_max} > {self.d_min}."

    # ----------------------------------------------------------------------
    # Modeling tests (use math.isclose with tolerance)
    # ----------------------------------------------------------------------
    def test_modeling_d_max_distance(self):
        """For the first maximum, d_max ≈ λ₁/4."""
        if self.d_max is None or self.lambda1 is None:
            return False, "SKIP: d_max or λ₁ not computable"
        expected = self.lambda1 / 4.0
        if not math.isclose(self.d_max, expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"d_max = {self.d_max}, expected ≈ {expected} (λ₁/4)"
        return True, f"d_max ≈ λ₁/4: {self.d_max} ≈ {expected}."

    def test_modeling_d_min_zero(self):
        """The first minimum of ⟨E²⟩ is at d=0 (interface)."""
        if self.d_min is None:
            return False, "SKIP: d_min not available"
        # If d_min is exactly 0, that's fine; if not, allow small deviation (e.g., numerical rounding)
        if not math.isclose(self.d_min, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"d_min = {self.d_min}, expected ≈ 0 (first minimum at interface)"
        return True, f"d_min ≈ 0: {self.d_min}."

    def test_modeling_extrema_spacing(self):
        """The distance between a minimum and the next maximum should be λ₁/4."""
        if self.d_max is None or self.d_min is None or self.lambda1 is None:
            return False, "SKIP: extrema or λ₁ missing"
        gap = abs(self.d_max - self.d_min)
        expected = self.lambda1 / 4.0
        if not math.isclose(gap, expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"|d_max - d_min| = {gap}, expected ≈ {expected} (λ₁/4)"
        return True, f"Spacing between extremum = {gap} ≈ λ₁/4."

    def test_modeling_e2_at_zero_formula(self):
        """Verify ⟨E²⟩(d=0) = (E₀²/2)*(1 - √ρ)²."""
        if self.E2_avg_at_zero is None or self.E0 is None or self.rho is None:
            return False, "SKIP: E2_avg_at_zero, E0, or rho missing"
        expected = (self.E0**2 / 2.0) * (1.0 - math.sqrt(self.rho))**2
        if not math.isclose(self.E2_avg_at_zero, expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"E2_avg_at_zero = {self.E2_avg_at_zero}, expected {expected}"
        return True, f"E2_avg_at_zero matches formula: {self.E2_avg_at_zero} ≈ {expected}."

    def test_modeling_b2_at_zero_formula(self):
        """Verify ⟨B²⟩(d=0) = (n₁²E₀²/(2c²))*(1 + √ρ)²."""
        if self.B2_avg_at_zero is None or self.n1 is None or self.E0 is None or self.c is None or self.rho is None:
            return False, "SKIP: one or more required quantities missing"
        expected = (self.n1**2 * self.E0**2 / (2.0 * self.c**2)) * (1.0 + math.sqrt(self.rho))**2
        if not math.isclose(self.B2_avg_at_zero, expected, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"B2_avg_at_zero = {self.B2_avg_at_zero}, expected {expected}"
        return True, f"B2_avg_at_zero matches formula: {self.B2_avg_at_zero} ≈ {expected}."

    # ----------------------------------------------------------------------
    # Monotonicity tests (perturb input, check monotonic trend)
    # ----------------------------------------------------------------------
    def test_monotonicity_e2_with_rho(self):
        """⟨E²⟩ at d=0 decreases as ρ increases (derivative negative)."""
        if self.E2_avg_at_zero is None or self.rho is None or self.E0 is None:
            return False, "SKIP: E2_avg_at_zero, rho, or E0 missing"
        # Perturb rho upward by 10%
        rho_pert = self.rho * 1.1
        expected_pert = (self.E0**2 / 2.0) * (1.0 - math.sqrt(rho_pert))**2
        # check that perturbed value is less than original (monotonic decreasing for rho > 0)
        if self.rho == 0:
            # derivative zero at rho=0, but still decreasing for small rho? Actually at rho=0, sqrt(rho) not defined? For rho=0, sqrt(0)=0, then formula gives constant? Let's skip if rho==0.
            return False, "SKIP: rho=0 gives pathological sqrt; derivative zero, cannot test monotonicity"
        if not (expected_pert < self.E2_avg_at_zero):
            return False, f"Monotonicity fail: expected perturbed value {expected_pert} < original {self.E2_avg_at_zero}"
        return True, f"⟨E²⟩(0) decreases with ρ: {self.E2_avg_at_zero} -> {expected_pert} at ρ+10%."

    def test_monotonicity_b2_with_rho(self):
        """⟨B²⟩ at d=0 increases as ρ increases (derivative positive)."""
        if self.B2_avg_at_zero is None or self.rho is None or self.n1 is None or self.E0 is None or self.c is None:
            return False, "SKIP: required quantities missing"
        rho_pert = self.rho * 1.1
        expected_pert = (self.n1**2 * self.E0**2 / (2.0 * self.c**2)) * (1.0 + math.sqrt(rho_pert))**2
        if not (expected_pert > self.B2_avg_at_zero):
            return False, f"Monotonicity fail: expected perturbed value {expected_pert} > original {self.B2_avg_at_zero}"
        return True, f"⟨B²⟩(0) increases with ρ: {self.B2_avg_at_zero} -> {expected_pert} at ρ+10%."

    def test_monotonicity_d_max_with_nu(self):
        """d_max decreases as frequency ν increases (wavelength decreases)."""
        if self.d_max is None or self.nu is None or self.n1 is None or self.c is None:
            return False, "SKIP: d_max, ν, n1, or c missing"
        # Perturb nu upward by 10%
        nu_pert = self.nu * 1.1
        lambda1_pert = self.c / (nu_pert * self.n1)
        d_max_pert = lambda1_pert / 4.0   # first maximum
        if not (d_max_pert < self.d_max):
            return False, f"Monotonicity fail: expected d_max({nu_pert}) = {d_max_pert} < original {self.d_max}"
        return True, f"d_max decreases with ν: {self.d_max} -> {d_max_pert} at ν+10%."

    # ----------------------------------------------------------------------
    # Answer sanity test (weak, at most one)
    # ----------------------------------------------------------------------
    def test_answer_sanity_cause(self):
        """The cause must be a non‑empty string (weak sanity)."""
        if self.cause is None:
            return False, "SKIP: cause not available"
        if not isinstance(self.cause, str) or len(self.cause.strip()) == 0:
            return False, f"cause is not a non‑empty string: {self.cause}"
        return True, f"cause is a non‑empty string: '{self.cause}'."

    # ----------------------------------------------------------------------
    # Run all tests
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
        return all_pass, logs