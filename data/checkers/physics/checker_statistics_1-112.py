import math
import traceback

class PhysicsChecker:
    """
    Audits the computed minimum work to freeze 1 kg of water from 25°C.
    Checks interface, physical invariants, modeling quality, monotonicity, and provides a weak sanity test.
    """

    def __init__(self, solution_instance):
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.W_min_J = results.get('W_min_J', None)

        # Attempt to extract givens from solution_instance
        # They might be stored as attributes or in a 'givens' dict.
        self.m = getattr(solution_instance, 'm', None)
        if self.m is None:
            givens = getattr(solution_instance, 'givens', {})
            self.m = givens.get('m', None)
        self.cp = getattr(solution_instance, 'cp', None)
        if self.cp is None:
            givens = getattr(solution_instance, 'givens', {})
            self.cp = givens.get('cp', None)
        self.L = getattr(solution_instance, 'L', None)
        if self.L is None:
            givens = getattr(solution_instance, 'givens', {})
            self.L = givens.get('L', None)
        self.T0 = getattr(solution_instance, 'T0', None)
        if self.T0 is None:
            givens = getattr(solution_instance, 'givens', {})
            self.T0 = givens.get('T0', None)
        self.Tf = getattr(solution_instance, 'Tf', None)
        if self.Tf is None:
            givens = getattr(solution_instance, 'givens', {})
            self.Tf = givens.get('Tf', None)
        self.cal_to_J = getattr(solution_instance, 'cal_to_J', None)
        if self.cal_to_J is None:
            givens = getattr(solution_instance, 'givens', {})
            self.cal_to_J = givens.get('cal_to_J', None)

        # Explanation steps for trace reference (optional)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ---- Interface checks ----

    def test_interface_W_min_J_exists(self):
        """Check that the required result key is present and is a number."""
        if self.W_min_J is None:
            return (False, "SKIP: W_min_J not provided in results")
        if not isinstance(self.W_min_J, (int, float)):
            return (False, f"W_min_J must be numeric, got {type(self.W_min_J).__name__}")
        if math.isnan(self.W_min_J) or math.isinf(self.W_min_J):
            return (False, "W_min_J is NaN or infinite")
        return (True, "W_min_J is present and finite")

    # ---- Modeling / sign conventions ----

    def test_modeling_W_min_positive(self):
        """Work required to freeze water must be positive (work input)."""
        if self.W_min_J is None:
            return (False, "SKIP: W_min_J not available")
        if self.W_min_J <= 0:
            return (False, f"W_min_J = {self.W_min_J} should be positive")
        return (True, "W_min_J is positive (work input)")

    def test_modeling_reasonable_magnitude(self):
        """Weak sanity: the minimum work should be on the order of 10^5 J for 1 kg of water."""
        if self.W_min_J is None:
            return (False, "SKIP: W_min_J not available")
        # Broad range: 10^4 to 10^6 J (covers both 3.5e4 and 3.5e5)
        if not (1e4 <= self.W_min_J <= 1e6):
            return (False, f"W_min_J = {self.W_min_J} outside expected magnitude range [1e4, 1e6] J")
        return (True, "W_min_J lies in a physically plausible range")

    # ---- Invariant checks (theorem-based) ----

    def test_invariant_temperature_ordering(self):
        """Initial temperature must be greater than freezing temperature."""
        if self.T0 is None or self.Tf is None:
            return (False, "SKIP: T0 or Tf not available")
        if not (self.T0 > self.Tf):
            return (False, f"T0 ({self.T0}) should be > Tf ({self.Tf})")
        return (True, "T0 > Tf holds")

    def test_invariant_total_work_formula(self):
        """
        Re-derive the minimum work using the thermodynamic formula and compare
        with the provided W_min_J.  Requires all givens.
        """
        # Check that all needed givens are present
        if any(x is None for x in [self.m, self.cp, self.L, self.T0, self.Tf, self.cal_to_J]):
            return (False, "SKIP: not all required givens available for re-derivation")

        # Convert to consistent units: use cal for internal; then convert to J
        # Sensible cooling work (cal)
        W_sensible = self.m * self.cp * ((self.Tf - self.T0) - self.T0 * math.log(self.Tf / self.T0))
        # Latent heat work (cal)
        W_latent = self.m * self.L * (self.T0 / self.Tf - 1)
        W_min_cal = W_sensible + W_latent
        expected_W_min_J = W_min_cal * self.cal_to_J

        # Use fuzzy matching because of rounding
        if not math.isclose(self.W_min_J, expected_W_min_J, rel_tol=1e-2, abs_tol=1e-2):
            return (False,
                    f"W_min_J {self.W_min_J} deviates from re-derived {expected_W_min_J} "
                    f"(sens={W_sensible:.2f} cal, lat={W_latent:.2f} cal, cal={W_min_cal:.2f} cal)")
        return (True, f"W_min_J matches re-derived value within 1% tolerance")

    # ---- Monotonicity checks ----

    def test_monotonicity_increasing_T0_increases_work(self):
        """
        Perturb T0 upward and verify that W_min increases.
        W_min depends monotonically on T0 because both sensible and latent terms increase with T0.
        """
        if any(x is None for x in [self.m, self.cp, self.L, self.T0, self.Tf, self.cal_to_J]):
            return (False, "SKIP: not all required givens available for monotonicity check")

        # Baseline W_min (cal) using given T0
        W_sens_base = self.m * self.cp * ((self.Tf - self.T0) - self.T0 * math.log(self.Tf / self.T0))
        W_lat_base = self.m * self.L * (self.T0 / self.Tf - 1)
        W_base_cal = W_sens_base + W_lat_base

        # Perturb T0 upward by 5%
        T0_perturbed = self.T0 * 1.05
        W_sens_pert = self.m * self.cp * ((self.Tf - T0_perturbed) - T0_perturbed * math.log(self.Tf / T0_perturbed))
        W_lat_pert = self.m * self.L * (T0_perturbed / self.Tf - 1)
        W_pert_cal = W_sens_pert + W_lat_pert

        if not (W_pert_cal > W_base_cal):
            return (False, f"Increasing T0 by 5% should increase total work: base {W_base_cal:.2f} cal -> perturbed {W_pert_cal:.2f} cal")
        return (True, f"Monotonicity OK: T0 +5% increases W_min (base {W_base_cal:.2f} cal, perturbed {W_pert_cal:.2f} cal)")

    def test_monotonicity_decreasing_Tf_increases_work(self):
        """
        Perturb Tf downward and verify that W_min increases.
        Lower freezing temperature requires more work (both sensible and latent terms increase).
        """
        if any(x is None for x in [self.m, self.cp, self.L, self.T0, self.Tf, self.cal_to_J]):
            return (False, "SKIP: not all required givens available for monotonicity check")

        # Baseline Tf
        W_sens_base = self.m * self.cp * ((self.Tf - self.T0) - self.T0 * math.log(self.Tf / self.T0))
        W_lat_base = self.m * self.L * (self.T0 / self.Tf - 1)
        W_base_cal = W_sens_base + W_lat_base

        # Perturb Tf downward by 2% (still above absolute zero)
        Tf_perturbed = self.Tf * 0.98
        W_sens_pert = self.m * self.cp * ((Tf_perturbed - self.T0) - self.T0 * math.log(Tf_perturbed / self.T0))
        W_lat_pert = self.m * self.L * (self.T0 / Tf_perturbed - 1)
        W_pert_cal = W_sens_pert + W_lat_pert

        if not (W_pert_cal > W_base_cal):
            return (False, f"Decreasing Tf by 2% should increase total work: base {W_base_cal:.2f} cal -> perturbed {W_pert_cal:.2f} cal")
        return (True, f"Monotonicity OK: Tf -2% increases W_min (base {W_base_cal:.2f} cal, perturbed {W_pert_cal:.2f} cal)")

    # ---- Weak answer sanity (secondary) ----

    def test_answer_sanity_sign(self):
        """Fallback sanity: work must be positive and finite."""
        if self.W_min_J is None:
            return (False, "SKIP: W_min_J not available")
        if not (math.isfinite(self.W_min_J) and self.W_min_J > 0):
            return (False, f"W_min_J = {self.W_min_J} is not finite and positive")
        return (True, "W_min_J is finite and positive")

    # ---- Main runner ----

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