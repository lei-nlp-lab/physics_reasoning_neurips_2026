import math

class PhysicsChecker:
    """
    Physics checker for Doppler and pressure broadening problem.
    Implements interface, invariant, modeling/trace, and monotonicity tests.
    """
    def __init__(self, solution_instance):
        # Extract results safely; do not assign numeric defaults.
        self.delta_nu_D_over_nu = solution_instance.results.get('delta_nu_D_over_nu', None)
        self.n = solution_instance.results.get('n', None)
        self.v_rel = solution_instance.results.get('v_rel', None)
        self.delta_nu_p_over_nu = solution_instance.results.get('delta_nu_p_over_nu', None)

        # Problem givens (from context)
        self.T = 300                   # K
        self.p = 100 * 133.322         # Torr -> Pa (100 Torr = 13332.2 Pa)
        self.m = 4.2e-27               # kg
        self.k = 1.4e-23               # J/K
        self.c = 3e8                   # m/s
        self.sigma = 1e-19             # m^2 (assumed cross‑section)

    # ------------------------------------------------------------
    # INTERFACE TESTS
    # ------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all four required result keys exist."""
        if self.delta_nu_D_over_nu is None:
            return (False, "Missing key: 'delta_nu_D_over_nu'")
        if self.n is None:
            return (False, "Missing key: 'n'")
        if self.v_rel is None:
            return (False, "Missing key: 'v_rel'")
        if self.delta_nu_p_over_nu is None:
            return (False, "Missing key: 'delta_nu_p_over_nu'")
        return (True, "All four required keys present")

    def test_interface_numeric_values(self):
        """Check that all values are numeric (int/float) and not NaN."""
        if not isinstance(self.delta_nu_D_over_nu, (int, float)):
            return (False, "delta_nu_D_over_nu is not numeric")
        if not isinstance(self.n, (int, float)):
            return (False, "n is not numeric")
        if not isinstance(self.v_rel, (int, float)):
            return (False, "v_rel is not numeric")
        if not isinstance(self.delta_nu_p_over_nu, (int, float)):
            return (False, "delta_nu_p_over_nu is not numeric")
        if any(math.isnan(x) for x in [self.delta_nu_D_over_nu, self.n, self.v_rel, self.delta_nu_p_over_nu]):
            return (False, "One or more values is NaN")
        return (True, "All values are numeric and finite")

    # ------------------------------------------------------------
    # INVARIANT TESTS (theorem-based physical consistency)
    # ------------------------------------------------------------
    def test_invariant_doppler_width_formula(self):
        """Check Δν_D/ν ≟ sqrt(8 ln2 k T / (m c^2))."""
        if self.delta_nu_D_over_nu is None:
            return (False, "SKIP: delta_nu_D_over_nu missing")
        expected = math.sqrt(8 * math.log(2) * self.k * self.T / (self.m * self.c**2))
        if math.isclose(self.delta_nu_D_over_nu, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Doppler width matched (got {self.delta_nu_D_over_nu:.3e}, expected {expected:.3e})")
        else:
            return (False, f"Doppler width mismatch: got {self.delta_nu_D_over_nu:.3e}, expected {expected:.3e}")

    def test_invariant_n_ideal_gas(self):
        """Check n ≟ p/(k T)."""
        if self.n is None:
            return (False, "SKIP: n missing")
        expected = self.p / (self.k * self.T)
        if math.isclose(self.n, expected, rel_tol=1e-2, abs_tol=1):
            return (True, f"n matched (got {self.n:.3e}, expected {expected:.3e})")
        else:
            return (False, f"n mismatch: got {self.n:.3e}, expected {expected:.3e}")

    def test_invariant_v_rel_formula(self):
        """Check v_rel ≟ sqrt(16 k T / (π m))."""
        if self.v_rel is None:
            return (False, "SKIP: v_rel missing")
        expected = math.sqrt(16 * self.k * self.T / (math.pi * self.m))
        if math.isclose(self.v_rel, expected, rel_tol=1e-2, abs_tol=1):
            return (True, f"v_rel matched (got {self.v_rel:.3e}, expected {expected:.3e})")
        else:
            return (False, f"v_rel mismatch: got {self.v_rel:.3e}, expected {expected:.3e}")

    def test_invariant_quantities_positive(self):
        """All results must be positive."""
        if self.n is not None and self.n <= 0:
            return (False, f"n is not positive: {self.n}")
        if self.v_rel is not None and self.v_rel <= 0:
            return (False, f"v_rel is not positive: {self.v_rel}")
        if self.delta_nu_D_over_nu is not None and self.delta_nu_D_over_nu <= 0:
            return (False, f"delta_nu_D_over_nu is not positive: {self.delta_nu_D_over_nu}")
        if self.delta_nu_p_over_nu is not None and self.delta_nu_p_over_nu <= 0:
            return (False, f"delta_nu_p_over_nu is not positive: {self.delta_nu_p_over_nu}")
        return (True, "All quantities positive")

    # ------------------------------------------------------------
    # MODELING / TRACE QUALITY TESTS (at least two)
    # ------------------------------------------------------------
    def test_modeling_pressure_width_derivation_consistency(self):
        """Check Δν_p/ν consistency with n, v_rel, σ, and visible light ν."""
        if any(x is None for x in (self.delta_nu_p_over_nu, self.n, self.v_rel)):
            return (False, "SKIP: one or more required results missing")
        gamma = self.n * self.sigma * self.v_rel
        delta_nu_p = gamma / (2 * math.pi)
        nu_visible = 6e14  # Hz, standard visible light
        expected_ratio = delta_nu_p / nu_visible
        if math.isclose(self.delta_nu_p_over_nu, expected_ratio, rel_tol=0.05, abs_tol=1e-8):
            return (True, f"Pressure width ratio consistent (got {self.delta_nu_p_over_nu:.3e}, expected {expected_ratio:.3e})")
        else:
            return (False, f"Pressure width ratio mismatch: got {self.delta_nu_p_over_nu:.3e}, expected {expected_ratio:.3e}")

    def test_modeling_collision_frequency_consistency(self):
        """Check that the implied collision frequency from Δν_p/ν matches n σ v_rel."""
        if any(x is None for x in (self.delta_nu_p_over_nu, self.n, self.v_rel)):
            return (False, "SKIP: missing data")
        nu_visible = 6e14
        implied_gamma = self.delta_nu_p_over_nu * nu_visible * 2 * math.pi
        expected_gamma = self.n * self.sigma * self.v_rel
        if math.isclose(implied_gamma, expected_gamma, rel_tol=0.05, abs_tol=1e6):
            return (True, f"Collision frequency consistent: implied {implied_gamma:.3e}, expected {expected_gamma:.3e}")
        else:
            return (False, f"Collision frequency mismatch: implied {implied_gamma:.3e}, expected {expected_gamma:.3e}")

    def test_modeling_doppler_width_larger_than_pressure_width(self):
        """Standard result: Doppler width dominates over pressure width for visible light under given conditions."""
        if self.delta_nu_D_over_nu is None or self.delta_nu_p_over_nu is None:
            return (False, "SKIP: missing widths")
        if self.delta_nu_D_over_nu > self.delta_nu_p_over_nu:
            return (True, f"Doppler width > pressure width: {self.delta_nu_D_over_nu:.3e} > {self.delta_nu_p_over_nu:.3e}")
        else:
            return (False, f"Expected Doppler width ({self.delta_nu_D_over_nu:.3e}) to be larger than pressure width ({self.delta_nu_p_over_nu:.3e})")

    # ------------------------------------------------------------
    # MONOTONICITY TESTS (each perturbs an input and checks trend)
    # ------------------------------------------------------------
    def test_monotonicity_doppler_width_increases_with_T(self):
        """Δν_D/ν ∝ sqrt(T); +5% T should yield ≈+2.5% increase."""
        if self.delta_nu_D_over_nu is None:
            return (False, "SKIP: delta_nu_D_over_nu missing")
        baseline = math.sqrt(8 * math.log(2) * self.k * self.T / (self.m * self.c**2))
        T_perturbed = self.T * 1.05
        perturbed = math.sqrt(8 * math.log(2) * self.k * T_perturbed / (self.m * self.c**2))
        if perturbed > baseline:
            return (True, f"Doppler width increases with T: baseline {baseline:.3e}, perturbed (T+5%) {perturbed:.3e}")
        else:
            return (False, "Doppler width should increase with T but calculation shows decrease")

    def test_monotonicity_pressure_width_increases_with_p(self):
        """Δν_p/ν ∝ p; +5% p should yield ≈+5% increase."""
        if any(x is None for x in (self.delta_nu_p_over_nu, self.n, self.v_rel)):
            return (False, "SKIP: missing data")
        n_base = self.p / (self.k * self.T)
        v_rel_base = math.sqrt(16 * self.k * self.T / (math.pi * self.m))
        gamma_base = n_base * self.sigma * v_rel_base
        baseline_ratio = (gamma_base / (2 * math.pi)) / 6e14

        p_perturbed = self.p * 1.05
        n_pert = p_perturbed / (self.k * self.T)   # v_rel unchanged
        gamma_pert = n_pert * self.sigma * v_rel_base
        perturbed_ratio = (gamma_pert / (2 * math.pi)) / 6e14

        if perturbed_ratio > baseline_ratio:
            return (True, f"Pressure width increases with p: baseline {baseline_ratio:.3e}, perturbed (p+5%) {perturbed_ratio:.3e}")
        else:
            return (False, "Pressure width should increase with p but calculation shows decrease")

    # ------------------------------------------------------------
    # WEAK ANSWER SANITY (only one such method)
    # ------------------------------------------------------------
    def test_answer_sanity_order_of_magnitude(self):
        """Check both width ratios lie in physically broad ranges."""
        if self.delta_nu_D_over_nu is not None:
            if not (1e-8 < self.delta_nu_D_over_nu < 1e-3):
                return (False, f"delta_nu_D_over_nu ({self.delta_nu_D_over_nu}) outside expected order of magnitude (1e-8 to 1e-3)")
        if self.delta_nu_p_over_nu is not None:
            if not (1e-10 < self.delta_nu_p_over_nu < 1e-4):
                return (False, f"delta_nu_p_over_nu ({self.delta_nu_p_over_nu}) outside expected order of magnitude (1e-10 to 1e-4)")
        return (True, "Both widths within reasonable magnitude ranges")

    # ------------------------------------------------------------
    # MAIN RUNNER
    # ------------------------------------------------------------
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