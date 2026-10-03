import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens safely – assume they are accessible as attributes or in a dict
        self.l = getattr(solution_instance, 'l', None)
        self.theta = getattr(solution_instance, 'theta', None)   # in radians
        self.t = getattr(solution_instance, 't', None)
        self.g = getattr(solution_instance, 'g', None)

        # Extract results – typically stored in a dictionary
        results = getattr(solution_instance, 'results', {})
        self.rho_ratio = results.get('rho_ratio', None)

        # If givens are inside a 'givens' dict, fallback to that
        if self.l is None:
            givens = getattr(solution_instance, 'givens', {})
            self.l = givens.get('l', None)
            self.theta = givens.get('theta', None)
            self.t = givens.get('t', None)
            self.g = givens.get('g', None)

        # Convert units if needed: l is given in km, but we'll work in meters (1e5 m)
        # Keep as given, but note that the derived formula uses l in meters.
        # The user should have already converted; if not, we can detect.

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Verify that all necessary givens and results exist."""
        missing = []
        if self.l is None:
            missing.append('l')
        if self.theta is None:
            missing.append('theta')
        if self.t is None:
            missing.append('t')
        if self.g is None:
            missing.append('g')
        if self.rho_ratio is None:
            missing.append('rho_ratio')
        if missing:
            return (False, f"Missing key(s): {', '.join(missing)}")
        return (True, "All required keys present")

    def test_interface_valid_types(self):
        """Check that numeric values are finite and positive (where expected)."""
        for name, val, pos in [('l', self.l, True),
                               ('theta', self.theta, True),
                               ('t', self.t, True),
                               ('g', self.g, True),
                               ('rho_ratio', self.rho_ratio, True)]:
            if val is None:
                continue
            if not isinstance(val, (int, float)):
                return (False, f"{name} is not numeric, got {type(val)}")
            if not math.isfinite(val):
                return (False, f"{name} is not finite")
            if pos and val <= 0:
                return (False, f"{name} is not positive")
        return (True, "All types and finiteness checks passed")

    # ------------------------------------------------------------------
    # Invariant checks (theorem‑based)
    # ------------------------------------------------------------------
    def test_invariant_small_angle_approx(self):
        """For θ = 0.5°, sin(θ) ≈ θ within 0.5%."""
        if self.theta is None:
            return (False, "SKIP: theta not available")
        # The approximation θ ≈ sinθ holds for small angles (error ~ θ^3/6)
        sin_theta = math.sin(self.theta)
        if not math.isclose(sin_theta, self.theta, rel_tol=1e-2, abs_tol=1e-6):
            return (False, f"Small angle approximation fails: sin({self.theta}) = {sin_theta}")
        return (True, "Small angle approximation satisfied (sinθ ≈ θ)")

    def test_invariant_derived_formula(self):
        """
        Re‑derive the density ratio from givens using the formula:
          rho_ratio = (g * t^2 * θ^3) / (5760 * π * l)
        (where l is in meters). Compare with reported result.
        This is an allowed exact derivation inside the checker.
        """
        if any(x is None for x in [self.l, self.theta, self.t, self.g, self.rho_ratio]):
            return (False, "SKIP: missing input")

        # It is important that l is in meters. The problem states l = 100 km = 1e5 m.
        # If the user passed l in km, we convert for our computation.
        # We try to detect: if l is around 100 (presumably km), we convert.
        l_meters = self.l
        if abs(l_meters - 100) < 1e-6 or abs(l_meters - 100e3) > 1000:  # heuristic: around 100 km if no conversion
            l_meters = self.l * 1e3   # assume user gave km -> convert to m
        # If it's already ~1e5, keep as is
        if abs(l_meters - 1e5) > 1e4:
            # not clearly meters or km; use as is (might be wrong, but we will note)
            pass

        denominator = 5760.0 * math.pi * l_meters
        numerator = self.g * (self.t ** 2) * (self.theta ** 3)
        computed_ratio = numerator / denominator

        if not math.isclose(computed_ratio, self.rho_ratio, rel_tol=2e-2, abs_tol=2e-2):
            return (False, f"Derived ratio {computed_ratio:.4f} differs from reported {self.rho_ratio:.4f}")
        return (True, f"Derived ratio matches reported ({computed_ratio:.4f} vs {self.rho_ratio:.4f})")

    # ------------------------------------------------------------------
    # Modeling quality checks
    # ------------------------------------------------------------------
    def test_modeling_units_consistent(self):
        """Check that the given l is appropriately scaled (100 km -> 1e5 m)."""
        if self.l is None:
            return (False, "SKIP: l not available")
        # Acceptable range: between 80,000 and 120,000 (if in meters)
        if 80000 <= self.l <= 120000:
            return (True, "l appears to be in meters (≈1e5)")
        # If l is around 100, plausibly in km
        if 80 <= self.l <= 120:
            return (True, "l appears to be in km (≈100)")
        return (False, f"Unusual l value: {self.l} (expected ~1e5 m or ~100 km)")

    def test_modeling_intermediate_expression(self):
        """Optional: check that Earth radius R_e computed from givens is consistent."""
        if self.l is None or self.rho_ratio is None:
            return (False, "SKIP: missing data for intermediate check")
        l_meters = self.l
        if abs(l_meters - 100) < 1e-6:
            l_meters = 1e5
        R_e = 180 * l_meters / math.pi
        # Earth radius ~6371 km -> expect ~6.37e6 m
        if not (6e6 < R_e < 7e6):
            return (False, f"Derived Earth radius {R_e/1e3:.1f} km seems off")
        return (True, f"Earth radius ~ {R_e/1e3:.0f} km, reasonable")

    # ------------------------------------------------------------------
    # Monotonicity checks (trend based on functional dependence)
    # ------------------------------------------------------------------
    def test_monotonicity_ratio_inverse_with_l(self):
        """
        The ratio ρ_e/ρ_s ∝ 1/l.
        If we increase l by 1%, the ratio should decrease.
        We compute both using the derived formula to avoid dependency on the solution's result.
        """
        if any(x is None for x in [self.l, self.theta, self.t, self.g]):
            return (False, "SKIP: missing input for monotonicity")

        l_meters = self.l
        if abs(l_meters - 100) < 1e-6:
            l_meters = 1e5

        denominator = 5760.0 * math.pi
        numerator = self.g * (self.t ** 2) * (self.theta ** 3)

        base_ratio = numerator / (denominator * l_meters)
        perturbed_ratio = numerator / (denominator * (l_meters * 1.01))

        if perturbed_ratio >= base_ratio:
            return (False, f"Ratio did not decrease when l increased: {perturbed_ratio:.4f} >= {base_ratio:.4f}")
        return (True, f"Monotonicity holds: l ↑ 1% → ratio ↓ from {base_ratio:.4f} to {perturbed_ratio:.4f}")

    def test_monotonicity_ratio_increase_with_t(self):
        """
        The ratio ∝ t^2.
        Increasing t by 1% should increase ratio by ~2%.
        """
        if any(x is None for x in [self.l, self.theta, self.t, self.g]):
            return (False, "SKIP: missing input")

        l_meters = self.l
        if abs(l_meters - 100) < 1e-6:
            l_meters = 1e5

        factor = (self.g * (self.theta ** 3)) / (5760.0 * math.pi * l_meters)
        base_ratio = factor * (self.t ** 2)
        perturbed_ratio = factor * ((self.t * 1.01) ** 2)

        if perturbed_ratio <= base_ratio:
            return (False, f"Ratio did not increase when t increased: {perturbed_ratio:.4f} <= {base_ratio:.4f}")
        return (True, f"Monotonicity holds: t ↑ 1% → ratio ↑ from {base_ratio:.4f} to {perturbed_ratio:.4f}")

    # ------------------------------------------------------------------
    # Weak answer sanity (broad magnitude only)
    # ------------------------------------------------------------------
    def test_answer_sanity_magnitude(self):
        """Check that the reported density ratio is plausible: between 2 and 5."""
        if self.rho_ratio is None:
            return (False, "SKIP: rho_ratio not available")
        if not (2.0 < self.rho_ratio < 5.0):
            return (False, f"rho_ratio {self.rho_ratio} outside expected range (2,5)")
        return (True, f"rho_ratio = {self.rho_ratio} is within [2,5]")

    # ------------------------------------------------------------------
    # Main test runner
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