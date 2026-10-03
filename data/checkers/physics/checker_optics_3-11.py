import math

class PhysicsChecker:
    """Audits a solution instance for the Doppler broadening and Doppler-free spectroscopy problem."""

    def __init__(self, solution_instance):
        # Extract givens safely (default None)
        self.nu0 = getattr(solution_instance, 'nu0', None)
        self.m = getattr(solution_instance, 'm', None)
        self.T = getattr(solution_instance, 'T', None)
        self.k = getattr(solution_instance, 'k', None)  # Boltzmann constant
        self.c = getattr(solution_instance, 'c', None)  # speed of light
        self.tau = getattr(solution_instance, 'tau', None)
        self.h = getattr(solution_instance, 'h', None)

        # Extract results safely
        results = getattr(solution_instance, 'results', {}) or {}
        self.dnu_D = results.get('dnu_D', None)
        self.dnu_nat = results.get('dnu_nat', None)
        self.wavevector_sum = results.get('wavevector_sum', None)

    def run_all_tests(self):
        """Discover and run all test_ methods."""
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

    # -------------------- Interface Checks --------------------
    def test_interface_has_doppler_width(self):
        if self.dnu_D is None:
            return (False, "Missing required result: dnu_D")
        return (True, "dnu_D present")

    def test_interface_has_natural_width(self):
        if self.dnu_nat is None:
            return (False, "Missing required result: dnu_nat")
        return (True, "dnu_nat present")

    def test_interface_has_wavevector_sum(self):
        if self.wavevector_sum is None:
            return (False, "Missing required result: wavevector_sum")
        return (True, "wavevector_sum present")

    # -------------------- Modeling Checks (numerical verification of emitted formulas) --------------------
    def test_modeling_doppler_width_formula(self):
        """Re-compute Doppler width from givens and compare."""
        if None in (self.nu0, self.m, self.T, self.k, self.c, self.dnu_D):
            return (False, "SKIP: insufficient data for Doppler width check")
        expected = self.nu0 * math.sqrt(8 * self.k * self.T * math.log(2) / (self.m * self.c**2))
        if math.isclose(self.dnu_D, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Doppler width matches formula: {self.dnu_D} ≈ {expected:.6e}")
        else:
            return (False, f"Doppler width mismatch: got {self.dnu_D}, expected {expected:.6e}")

    def test_modeling_natural_width_formula(self):
        """Re-compute natural width from lifetime and compare."""
        if None in (self.tau, self.dnu_nat):
            return (False, "SKIP: insufficient data for natural width check")
        expected = 1.0 / (2 * math.pi * self.tau)
        if math.isclose(self.dnu_nat, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Natural width matches formula: {self.dnu_nat} ≈ {expected:.6e}")
        else:
            return (False, f"Natural width mismatch: got {self.dnu_nat}, expected {expected:.6e}")

    def test_modeling_wavevector_sum_zero(self):
        """Check that wavevector_sum is zero (k1 = -k2 condition)."""
        if self.wavevector_sum is None:
            return (False, "SKIP: wavevector_sum not available")
        if isinstance(self.wavevector_sum, (int, float)):
            if math.isclose(self.wavevector_sum, 0, abs_tol=1e-2):
                return (True, "wavevector_sum ≈ 0 (Doppler-free condition)")
            else:
                return (False, f"wavevector_sum = {self.wavevector_sum}, expected ≈ 0")
        # If it's a list/array, check each element
        try:
            if all(math.isclose(v, 0, abs_tol=1e-2) for v in self.wavevector_sum):
                return (True, "All wavevector_sum components ≈ 0")
            else:
                return (False, f"wavevector_sum = {self.wavevector_sum}, expected all ≈ 0")
        except TypeError:
            return (False, f"wavevector_sum type unsupported: {type(self.wavevector_sum)}")

    # -------------------- Invariant Checks (physical positivity, basic ranges) --------------------
    def test_invariant_doppler_width_positive(self):
        if self.dnu_D is None:
            return (False, "SKIP: no dnu_D")
        if self.dnu_D > 0:
            return (True, f"Doppler width positive: {self.dnu_D}")
        else:
            return (False, f"Doppler width not positive: {self.dnu_D}")

    def test_invariant_natural_width_positive(self):
        if self.dnu_nat is None:
            return (False, "SKIP: no dnu_nat")
        if self.dnu_nat > 0:
            return (True, f"Natural width positive: {self.dnu_nat}")
        else:
            return (False, f"Natural width not positive: {self.dnu_nat}")

    def test_invariant_doppler_finite(self):
        if self.dnu_D is None:
            return (False, "SKIP: no dnu_D")
        if math.isfinite(self.dnu_D):
            return (True, "Doppler width finite")
        else:
            return (False, f"Doppler width not finite: {self.dnu_D}")

    def test_invariant_natural_finite(self):
        if self.dnu_nat is None:
            return (False, "SKIP: no dnu_nat")
        if math.isfinite(self.dnu_nat):
            return (True, "Natural width finite")
        else:
            return (False, f"Natural width not finite: {self.dnu_nat}")

    # -------------------- Monotonicity Checks (input perturbation) --------------------
    def test_monotonicity_doppler_increases_with_temperature(self):
        """Perturb temperature upward by 1% and verify Doppler width increases."""
        if None in (self.T, self.dnu_D):
            return (False, "SKIP: insufficient data for monotonicity check (T or dnu_D missing)")
        # Use formula to compute baseline from givens
        if None in (self.nu0, self.m, self.k, self.c):
            # If we can't recompute, use the stored value but we cannot perturb temperature without recomputing
            return (False, "SKIP: missing givens needed to recompute with perturbed T")
        # Baseline Doppler width computed from current T
        baseline = self.nu0 * math.sqrt(8 * self.k * self.T * math.log(2) / (self.m * self.c**2))
        # Perturb T upward by 1%
        T_high = self.T * 1.01
        dnu_high = self.nu0 * math.sqrt(8 * self.k * T_high * math.log(2) / (self.m * self.c**2))
        if dnu_high > baseline:
            return (True, f"Doppler width increases with T: baseline {baseline:.6e}, +1% T gives {dnu_high:.6e}")
        else:
            return (False, f"Monotonicity violation: baseline {baseline:.6e}, perturbed {dnu_high:.6e}")

    def test_monotonicity_natural_decreases_with_lifetime(self):
        """Perturb lifetime upward by 1% and verify natural width decreases."""
        if None in (self.tau, self.dnu_nat):
            return (False, "SKIP: insufficient data for monotonicity check (tau or dnu_nat missing)")
        # Baseline natural width computed from tau
        baseline = 1.0 / (2 * math.pi * self.tau)
        # Perturb tau upward by 1%
        tau_long = self.tau * 1.01
        dnu_long = 1.0 / (2 * math.pi * tau_long)
        if dnu_long < baseline:
            return (True, f"Natural width decreases with tau: baseline {baseline:.6e}, +1% tau gives {dnu_long:.6e}")
        else:
            return (False, f"Monotonicity violation: baseline {baseline:.6e}, perturbed {dnu_long:.6e}")

    # -------------------- Weak Answer Sanity (only broad checks, no re-derivation) --------------------
    def test_answer_sanity_broad_checks(self):
        """
        Weak sanity: both widths should be positive, finite, and within a reasonable range.
        Do not re-derive; only check sign/finiteness/order-of-magnitude.
        """
        problems = []
        if self.dnu_D is not None:
            if not (self.dnu_D > 0 and math.isfinite(self.dnu_D)):
                problems.append(f"Doppler width not valid: {self.dnu_D}")
            elif self.dnu_D > 1e20:  # far beyond typical values
                problems.append(f"Doppler width far too large: {self.dnu_D}")
        else:
            problems.append("dnu_D missing")
        if self.dnu_nat is not None:
            if not (self.dnu_nat > 0 and math.isfinite(self.dnu_nat)):
                problems.append(f"Natural width not valid: {self.dnu_nat}")
            elif self.dnu_nat < 1e-20 or self.dnu_nat > 1e10:
                problems.append(f"Natural width outside typical range: {self.dnu_nat}")
        else:
            problems.append("dnu_nat missing")
        if problems:
            return (False, "; ".join(problems))
        return (True, "Both widths appear reasonable (positive, finite, within plausible range)")