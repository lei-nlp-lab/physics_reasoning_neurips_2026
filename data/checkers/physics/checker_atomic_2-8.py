import math
import traceback

class PhysicsChecker:
    """
    Auditor for nuclear density calculation.
    Checks:
    - Interface: result key 'rho_toncm3' present and numeric.
    - Modeling: density computed via standard formula matches within tolerance.
    - Invariant: density is independent of A (not directly testable here).
    - Monotonicity: density decreases with R0 increase, increases with u increase.
    - Sanity: result is finite, positive, and roughly in 1e9 order.
    """

    # Standard physical constants (from problem statement)
    U_STANDARD = 1.66e-27  # atomic mass unit in kg
    R0_STANDARD = 1.2e-15  # nuclear radius parameter in meters
    TON_PER_KG = 1e-3     # 1 ton = 1000 kg
    CM3_PER_M3 = 1e6      # 1 m^3 = 1e6 cm^3
    CONVERSION_FACTOR = TON_PER_KG / CM3_PER_M3  # = 1e-9

    def __init__(self, solution_instance):
        # Extract givens safely (prefer solution_instance attributes, else use standard)
        self.u = getattr(solution_instance, 'u', None) or self.U_STANDARD
        self.R0 = getattr(solution_instance, 'R0', None) or self.R0_STANDARD

        # Extract result keys safely
        # results may be an attribute or a dict; handle both
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        # Allow direct attribute access for rho_toncm3
        self.rho_toncm3 = getattr(solution_instance, 'rho_toncm3', None)
        if self.rho_toncm3 is None:
            self.rho_toncm3 = results.get('rho_toncm3', None)

    def _compute_density_toncm3(self, u=None, R0=None):
        """Compute nuclear density in ton/cm^3 from given u and R0."""
        if u is None:
            u = self.u
        if R0 is None:
            R0 = self.R0
        if u is None or R0 is None:
            return None
        rho_kgm3 = (3 * u) / (4 * math.pi * R0**3)
        return rho_kgm3 * self.CONVERSION_FACTOR

    def run_all_tests(self):
        """Discover and run all test_* methods, return (overall_pass, logs)."""
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
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

    # ---------- Interface tests ----------
    def test_interface_result_key_exists(self):
        """Check that the result key 'rho_toncm3' is present and numeric."""
        if self.rho_toncm3 is None:
            return (False, "rho_toncm3 is missing")
        if not isinstance(self.rho_toncm3, (int, float)):
            return (False, f"rho_toncm3 is not numeric, got {type(self.rho_toncm3)}")
        return (True, "rho_toncm3 present and numeric")

    def test_interface_givens_available(self):
        """Check that givens u and R0 are available (either from instance or defaults)."""
        if self.u is None or self.R0 is None:
            return (False, f"u={self.u}, R0={self.R0} - cannot compute reference")
        return (True, f"u={self.u}, R0={self.R0} available")

    # ---------- Modeling/trace tests ----------
    def test_modeling_density_formula(self):
        """Check that the provided density matches the standard formula within 2% tolerance."""
        if self.rho_toncm3 is None:
            return (False, "SKIP: rho_toncm3 not available")
        expected = self._compute_density_toncm3()
        if expected is None:
            return (False, "Cannot compute expected density")
        if math.isclose(self.rho_toncm3, expected, rel_tol=0.02, abs_tol=1e-2):
            return (True, f"rho_toncm3={self.rho_toncm3:.4e} matches expected={expected:.4e} within 2%")
        else:
            return (False, f"rho_toncm3={self.rho_toncm3:.4e} deviates from expected={expected:.4e} by more than 2%")

    def test_modeling_unit_conversion_factor(self):
        """Check that the given rho_toncm3 is consistent with the conversion from kg/m^3 times 1e-9."""
        if self.rho_toncm3 is None:
            return (False, "SKIP: rho_toncm3 not available")
        # Recompute rho_kgm3 from provided rho_toncm3 by inverse conversion
        rho_kgm3_from_provided = self.rho_toncm3 / self.CONVERSION_FACTOR
        # Now compute expected rho_kgm3 using formula
        expected_rho_kgm3 = (3 * self.u) / (4 * math.pi * self.R0**3)
        if math.isclose(rho_kgm3_from_provided, expected_rho_kgm3, rel_tol=0.02, abs_tol=1e-2):
            return (True, "Unit conversion factor (1e-9) is correctly applied")
        else:
            return (False, f"Back-computed rho_kgm3={rho_kgm3_from_provided:.4e} diverges from formula value={expected_rho_kgm3:.4e}")

    # ---------- Invariant tests ----------
    def test_invariant_positive_density(self):
        """Nuclear density must be positive."""
        if self.rho_toncm3 is None:
            return (False, "SKIP: rho_toncm3 not available")
        if self.rho_toncm3 > 0:
            return (True, "Density is positive")
        else:
            return (False, f"Density is non-positive: {self.rho_toncm3}")

    def test_invariant_density_independent_of_A(self):
        """
        The model assumes density independent of mass number A.
        If the solver had provided multiple nuclei, they should have same density.
        We cannot check without multiple data, so we check that the formula derivation is consistent.
        This test passes by construction if the formula is used.
        """
        # This is a knowledge check: the formula ρ = 3u/(4πR0^3) is A-independent.
        # We can verify that the result is close to this standard value.
        # Essentially same as test_modeling_density_formula.
        expected = self._compute_density_toncm3()
        if expected is None:
            return (False, "Cannot compute expected density")
        if self.rho_toncm3 is not None and math.isclose(self.rho_toncm3, expected, rel_tol=0.02):
            return (True, "Density is consistent with A-independent formula")
        else:
            return (False, "Density not consistent with A-independent formula")

    # ---------- Monotonicity tests ----------
    def test_monotonicity_density_decreases_with_R0(self):
        """If R0 increases by 1%, density should decrease."""
        if self.u is None or self.R0 is None:
            return (False, "SKIP: missing givens for perturbation")
        # Baseline density
        base_rho = self._compute_density_toncm3()
        # Perturb R0: +1%
        perturbed_R0 = self.R0 * 1.01
        perturbed_rho = self._compute_density_toncm3(R0=perturbed_R0)
        if perturbed_rho is None or base_rho is None:
            return (False, "Cannot compute perturbed density")
        # Density should decrease (since R0 in denominator, cubic)
        if perturbed_rho < base_rho:
            return (True, f"Perturbed density {perturbed_rho:.4e} < base {base_rho:.4e} as expected")
        else:
            return (False, f"Perturbed density {perturbed_rho:.4e} not less than base {base_rho:.4e}")

    def test_monotonicity_density_increases_with_u(self):
        """If atomic mass unit increases by 1%, density should increase."""
        if self.u is None or self.R0 is None:
            return (False, "SKIP: missing givens for perturbation")
        base_rho = self._compute_density_toncm3()
        perturbed_u = self.u * 1.01
        perturbed_rho = self._compute_density_toncm3(u=perturbed_u)
        if perturbed_rho is None or base_rho is None:
            return (False, "Cannot compute perturbed density")
        if perturbed_rho > base_rho:
            return (True, f"Perturbed density {perturbed_rho:.4e} > base {base_rho:.4e} as expected")
        else:
            return (False, f"Perturbed density {perturbed_rho:.4e} not greater than base {base_rho:.4e}")

    # ---------- Weak answer sanity test (only one, secondary) ----------
    def test_answer_sanity_order_of_magnitude(self):
        """
        Weak sanity: density should be positive and in the ballpark of 1e9 ton/cm^3.
        Accept within one order of magnitude: 1e8 to 1e10.
        """
        if self.rho_toncm3 is None:
            return (False, "SKIP: rho_toncm3 not available")
        if self.rho_toncm3 <= 0:
            return (False, f"Density is non-positive: {self.rho_toncm3}")
        if not math.isfinite(self.rho_toncm3):
            return (False, "Density is not finite (NaN or Inf)")
        # Check order of magnitude: between 1e8 and 1e10
        if 1e8 <= self.rho_toncm3 <= 1e10:
            return (True, f"Density {self.rho_toncm3:.4e} is within plausible order of magnitude")
        else:
            return (False, f"Density {self.rho_toncm3:.4e} is far from the expected 1e9 ton/cm^3")