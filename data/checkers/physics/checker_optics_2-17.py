import math
import traceback

class PhysicsChecker:
    """
    A unit-test-like checker for the soap film interference problem.
    Audits interface completeness, physical invariants, modeling quality,
    monotonicity, and weak answer sanity.
    """

    def __init__(self, solution_instance):
        """
        Extract all relevant data safely using None defaults.
        Never use numeric defaults for missing required quantities.
        """
        # --- Givens ---
        givens = getattr(solution_instance, 'givens', {})
        self.n = givens.get('n', None)
        self.lambda0 = givens.get('lambda0', None)
        if self.lambda0 is not None:
            try:
                self.lambda0 = float(self.lambda0)
            except (TypeError, ValueError):
                self.lambda0 = None

        # --- Results ---
        results = getattr(solution_instance, 'results', {})
        self.I_max_over_I0 = results.get('I_max_over_I0', None)
        self.I_min_over_I0 = results.get('I_min_over_I0', None)
        if self.I_max_over_I0 is not None:
            try:
                self.I_max_over_I0 = float(self.I_max_over_I0)
            except (TypeError, ValueError):
                self.I_max_over_I0 = None
        if self.I_min_over_I0 is not None:
            try:
                self.I_min_over_I0 = float(self.I_min_over_I0)
            except (TypeError, ValueError):
                self.I_min_over_I0 = None

        # --- Auxiliary fields ---
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'solving_pseudocode', None)

        # Precomputed reference (will be used in several tests)
        self.R_ref = 1.0 / 49.0
        self.I_max_ref = 4.0 * self.R_ref   # ~0.08163265306

    # ------------------------------------------------------------------
    # Interface tests: check that required keys exist and have correct types
    # ------------------------------------------------------------------
    def test_interface_I_max_over_I0_exists(self):
        """The I_max_over_I0 key must be present and numeric."""
        if self.I_max_over_I0 is None:
            return (False, "No I_max_over_I0 provided.")
        if not isinstance(self.I_max_over_I0, (int, float)):
            return (False, f"I_max_over_I0 is not numeric: {type(self.I_max_over_I0)}")
        return (True, f"I_max_over_I0 = {self.I_max_over_I0} is a valid number.")

    def test_interface_I_min_over_I0_exists(self):
        """The I_min_over_I0 key must be present and numeric."""
        if self.I_min_over_I0 is None:
            return (False, "No I_min_over_I0 provided.")
        if not isinstance(self.I_min_over_I0, (int, float)):
            return (False, f"I_min_over_I0 is not numeric: {type(self.I_min_over_I0)}")
        return (True, f"I_min_over_I0 = {self.I_min_over_I0} is a valid number.")

    def test_interface_n_given(self):
        """The refractive index n should be provided in givens."""
        if self.n is None:
            return (False, "Refractive index n is missing from givens.")
        try:
            n_val = float(self.n)
        except (TypeError, ValueError):
            return (False, f"n is not numeric: {self.n}")
        return (True, f"n = {n_val}")

    # ------------------------------------------------------------------
    # Invariant tests: theorem-based physical sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_reflectance(self):
        """If n is given, the Fresnel reflectance R = ((n-1)/(n+1))^2 must be 1/49."""
        if self.n is None:
            return (False, "SKIP: n not available.")
        try:
            n_val = float(self.n)
        except (TypeError, ValueError):
            return (False, f"n is not numeric: {self.n}")
        R_computed = ((n_val - 1) / (n_val + 1)) ** 2
        if not math.isclose(R_computed, self.R_ref, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"R = {R_computed:.6f}, expected ~{self.R_ref:.6f}")
        return (True, f"R = {R_computed:.6f} ≈ 1/49")

    def test_invariant_max_intensity_formula(self):
        """I_max / I0 should equal 4R."""
        if self.n is None:
            if self.I_max_over_I0 is None:
                return (False, "SKIP: neither n nor I_max_over_I0 available.")
            # Without n we cannot recompute R; just check I_max over I0 is positive and ≤ 1
            if self.I_max_over_I0 <= 0 or self.I_max_over_I0 >= 1:
                return (False, f"I_max_over_I0 = {self.I_max_over_I0} not in (0,1)")
            return (True, f"I_max_over_I0 ≈ {self.I_max_over_I0:.4f} (cannot verify formula without n).")
        try:
            n_val = float(self.n)
        except (TypeError, ValueError):
            return (False, f"n is not numeric: {self.n}")
        R_computed = ((n_val - 1) / (n_val + 1)) ** 2
        expected_max = 4.0 * R_computed
        if self.I_max_over_I0 is None:
            return (False, "I_max_over_I0 missing; cannot compare.")
        if not math.isclose(self.I_max_over_I0, expected_max, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"I_max/I0 = {self.I_max_over_I0:.6f}, expected {expected_max:.6f}")
        return (True, f"I_max/I0 = {self.I_max_over_I0:.6f} ≈ 4R = {expected_max:.6f}")

    def test_invariant_min_intensity_zero(self):
        """I_min/I0 should be zero (destructive interference)."""
        if self.I_min_over_I0 is None:
            return (False, "I_min_over_I0 not provided.")
        # Use isclose because numerical rounding may give tiny non-zero.
        if not math.isclose(self.I_min_over_I0, 0.0, abs_tol=1e-6):
            return (False, f"I_min/I0 = {self.I_min_over_I0}, expected 0")
        return (True, f"I_min/I0 = {self.I_min_over_I0} (≈0)")

    # ------------------------------------------------------------------
    # Modeling / trace tests: check the consistency of intermediate steps
    # ------------------------------------------------------------------
    def test_modeling_max_min_relation(self):
        """The maxima intensity must be strictly greater than the minima intensity."""
        if self.I_max_over_I0 is None or self.I_min_over_I0 is None:
            return (False, "SKIP: both I_max and I_min required.")
        if self.I_max_over_I0 <= self.I_min_over_I0:
            return (False, f"I_max ({self.I_max_over_I0}) not > I_min ({self.I_min_over_I0})")
        return (True, f"I_max = {self.I_max_over_I0} > I_min = {self.I_min_over_I0}")

    def test_modeling_phase_change_mentioned(self):
        """The explanation steps or pseudocode should mention a π phase change."""
        combined = ""
        if self.explanation_steps:
            if isinstance(self.explanation_steps, str):
                combined += self.explanation_steps
            elif isinstance(self.explanation_steps, list):
                combined = " ".join(str(step) for step in self.explanation_steps)
        if self.pseudocode:
            if isinstance(self.pseudocode, str):
                combined += self.pseudocode
            elif isinstance(self.pseudocode, list):
                combined += " ".join(str(item) for item in self.pseudocode)
        if not combined:
            return (False, "SKIP: no explanation or pseudocode to inspect.")
        if "π" in combined or "pi" in combined.lower() or "phase" in combined.lower():
            return (True, "Phase change (π) or phase is mentioned.")
        return (False, "Phase change (π) is not mentioned in explanation or pseudocode.")

    # ------------------------------------------------------------------
    # Monotonicity tests: perturb an input and check the trend
    # ------------------------------------------------------------------
    def test_monotonicity_I_max_with_n(self):
        """If n is given, increasing n should increase I_max (derivative positive for n>1)."""
        if self.n is None:
            return (False, "SKIP: n not available.")
        try:
            n0 = float(self.n)
        except (TypeError, ValueError):
            return (False, f"n is not numeric: {self.n}")
        if n0 <= 1.0:
            return (False, f"n = {n0} not >1; cannot test monotonicity.")

        # Baseline: compute I_max from formula using current n
        R0 = ((n0 - 1) / (n0 + 1)) ** 2
        I_max0 = 4.0 * R0

        # Perturb n upward by 10%
        n1 = n0 * 1.1
        R1 = ((n1 - 1) / (n1 + 1)) ** 2
        I_max1 = 4.0 * R1

        if I_max1 <= I_max0:
            return (False, f"Expected I_max increase with n, got I_max0={I_max0:.6f}, I_max1={I_max1:.6f}")
        return (True, f"I_max increases from {I_max0:.6f} to {I_max1:.6f} when n increases.")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one such method, no isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_values(self):
        """
        Weak check: maxima intensity should be positive and <=1,
        minima intensity should be nonnegative and <= maxima.
        """
        failures = []
        if self.I_max_over_I0 is not None:
            if not (0 < self.I_max_over_I0 <= 1):
                failures.append(f"I_max/I0 = {self.I_max_over_I0} not in (0,1]")
        if self.I_min_over_I0 is not None:
            if self.I_min_over_I0 < 0:
                failures.append(f"I_min/I0 = {self.I_min_over_I0} is negative")
            if self.I_max_over_I0 is not None and self.I_min_over_I0 > self.I_max_over_I0:
                failures.append(f"I_min ({self.I_min_over_I0}) > I_max ({self.I_max_over_I0})")
        if failures:
            return (False, "; ".join(failures))
        if self.I_max_over_I0 is None and self.I_min_over_I0 is None:
            return (False, "No intensity values provided at all.")
        return (True, "Sanity check passed: values are in plausible ranges.")

    # ------------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------------
    def run_all_tests(self):
        """
        Discover and run every method starting with 'test_'.
        Returns (all_pass, log_messages).
        """
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
                passed = False
                msg = f"CRASHED: {e}\n{traceback.format_exc()}"
            status = "PASS" if passed else "FAIL"
            logs.append(f"{status}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)