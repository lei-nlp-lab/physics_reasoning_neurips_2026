import math
import traceback

class PhysicsChecker:
    """
    Audits a solution instance for the magnet design problem.
    Expected result keys: 'P', 'm', 'F'.
    Givens (if available): B, l_g, A_g, mu0, J, rho, d, l_turn.
    """

    def __init__(self, solution_instance):
        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.P = results.get('P', None)
        self.m = results.get('m', None)
        self.F = results.get('F', None)

        # Extract givens safely (may be None)
        self.B = getattr(solution_instance, 'B', None)
        self.l_g = getattr(solution_instance, 'l_g', None)
        self.A_g = getattr(solution_instance, 'A_g', None)
        self.mu0 = getattr(solution_instance, 'mu0', None)
        self.J = getattr(solution_instance, 'J', None)
        self.rho = getattr(solution_instance, 'rho', None)
        self.d = getattr(solution_instance, 'd', None)
        self.l_turn = getattr(solution_instance, 'l_turn', None)

        # Use standard constants if givens are missing (for fallback checks)
        self._default_mu0 = 4e-7 * math.pi
        self._default_J = 1e7               # A/m^2
        self._default_rho = 2e-8            # Ohm*m
        self._default_d = 8000              # kg/m^3
        self._default_l_turn = 6.0          # m (perimeter of 1m x 2m pole)

    def _get_or_default(self, attr, default):
        value = getattr(self, attr, None)
        return value if value is not None else default

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

    # ------------------------------------------------------------------
    # INTERFACE TESTS
    # ------------------------------------------------------------------
    def test_interface_required_keys_present(self):
        """Check that all three required result keys exist and are not None."""
        if self.P is None or self.m is None or self.F is None:
            return (False, f"Missing result keys: P={self.P}, m={self.m}, F={self.F}")
        return (True, "All three result keys present")

    def test_interface_numeric_types(self):
        """Check that results are numeric (int or float)."""
        for name, val in [('P', self.P), ('m', self.m), ('F', self.F)]:
            if val is not None and not isinstance(val, (int, float)):
                return (False, f"{name} is not numeric: {type(val)}")
        return (True, "All results have numeric type")

    # ------------------------------------------------------------------
    # INVARIANT / THEOREM-BASED SANITY CHECKS
    # ------------------------------------------------------------------
    def test_invariant_force_positive_definite(self):
        """Force must be positive (attraction)."""
        if self.F is None:
            return (False, "SKIP: F not available")
        if not self.F > 0:
            return (False, f"F <= 0: {self.F}")
        return (True, f"F positive: {self.F}")

    def test_invariant_mass_positive(self):
        if self.m is None:
            return (False, "SKIP: m not available")
        if not self.m > 0:
            return (False, f"m <= 0: {self.m}")
        return (True, f"m positive: {self.m}")

    def test_invariant_power_positive(self):
        if self.P is None:
            return (False, "SKIP: P not available")
        if not self.P > 0:
            return (False, f"P <= 0: {self.P}")
        return (True, f"P positive: {self.P}")

    # ------------------------------------------------------------------
    # MODELING QUALITY CHECKS (re-derive from givens)
    # ------------------------------------------------------------------
    def test_modeling_force_attraction(self):
        """Check F = B^2 * A_g / (2 * mu0)."""
        B = self._get_or_default('B', None)
        A_g = self._get_or_default('A_g', None)
        mu0 = self._get_or_default('mu0', self._default_mu0)
        if B is None or A_g is None:
            return (False, "SKIP: B or A_g missing")
        expected = (B ** 2 * A_g) / (2 * mu0)
        if self.F is None:
            return (False, "SKIP: F missing")
        if not math.isclose(self.F, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"F {self.F} not close to expected {expected}")
        return (True, f"F matches magnetic pressure formula")

    def test_modeling_power_derived(self):
        """Check P = J^2 * rho * V_cu, with V_cu derived from NI, etc."""
        B = self._get_or_default('B', None)
        l_g = self._get_or_default('l_g', None)
        mu0 = self._get_or_default('mu0', self._default_mu0)
        J = self._get_or_default('J', self._default_J)
        rho = self._get_or_default('rho', self._default_rho)
        l_turn = self._get_or_default('l_turn', self._default_l_turn)
        if any(x is None for x in [B, l_g, J, rho, l_turn]):
            return (False, "SKIP: needed givens missing")
        NI = B * l_g / mu0
        A_cu = NI / J
        V_cu = A_cu * l_turn
        expected = J**2 * rho * V_cu
        if self.P is None:
            return (False, "SKIP: P missing")
        if not math.isclose(self.P, expected, rel_tol=2e-2, abs_tol=1e-2):
            return (False, f"P {self.P} not close to expected {expected}")
        return (True, f"P matches derived formula")

    def test_modeling_mass_derived(self):
        """Check m = d * V_cu."""
        B = self._get_or_default('B', None)
        l_g = self._get_or_default('l_g', None)
        mu0 = self._get_or_default('mu0', self._default_mu0)
        J = self._get_or_default('J', self._default_J)
        d = self._get_or_default('d', self._default_d)
        l_turn = self._get_or_default('l_turn', self._default_l_turn)
        if any(x is None for x in [B, l_g, J, d, l_turn]):
            return (False, "SKIP: needed givens missing")
        NI = B * l_g / mu0
        A_cu = NI / J
        V_cu = A_cu * l_turn
        expected = d * V_cu
        if self.m is None:
            return (False, "SKIP: m missing")
        if not math.isclose(self.m, expected, rel_tol=2e-2, abs_tol=1e-2):
            return (False, f"m {self.m} not close to expected {expected}")
        return (True, f"m matches derived formula")

    def test_modeling_power_from_force_independent(self):
        """Sanity: P and F should follow different dependencies (not testing equality)."""
        # Just ensure both are finite and positive (already covered)
        if self.P is None or self.F is None:
            return (False, "SKIP: P or F missing")
        if not math.isfinite(self.P) or not math.isfinite(self.F):
            return (False, f"Non-finite P={self.P} or F={self.F}")
        return (True, "Both P and F are finite")

    # ------------------------------------------------------------------
    # MONOTONICITY CHECKS (perturb B)
    # ------------------------------------------------------------------
    def test_monotonicity_B_increases_P(self):
        """If B increases, expected P should increase. Use result B if available."""
        B = self._get_or_default('B', None)
        if B is None:
            return (False, "SKIP: B missing")
        # Perturb B by +10%
        B_pert = B * 1.1
        # Compute expected P with perturbed B (using same other givens)
        l_g = self._get_or_default('l_g', None)
        mu0 = self._get_or_default('mu0', self._default_mu0)
        J = self._get_or_default('J', self._default_J)
        rho = self._get_or_default('rho', self._default_rho)
        l_turn = self._get_or_default('l_turn', self._default_l_turn)
        if any(x is None for x in [l_g, J, rho, l_turn]):
            return (False, "SKIP: needed givens missing")
        NI_orig = B * l_g / mu0
        NI_pert = B_pert * l_g / mu0
        A_cu_orig = NI_orig / J
        A_cu_pert = NI_pert / J
        V_cu_orig = A_cu_orig * l_turn
        V_cu_pert = A_cu_pert * l_turn
        P_orig = J**2 * rho * V_cu_orig
        P_pert = J**2 * rho * V_cu_pert
        # Check that P_pert > P_orig
        if not P_pert > P_orig:
            return (False, f"Perturbed P {P_pert} not greater than original P {P_orig}")
        return (True, f"B increase leads to P increase: {P_orig} -> {P_pert}")

    def test_monotonicity_B_increases_F(self):
        """If B increases, expected F should increase."""
        B = self._get_or_default('B', None)
        A_g = self._get_or_default('A_g', None)
        mu0 = self._get_or_default('mu0', self._default_mu0)
        if B is None or A_g is None:
            return (False, "SKIP: B or A_g missing")
        B_pert = B * 1.1
        F_orig = (B**2 * A_g) / (2 * mu0)
        F_pert = (B_pert**2 * A_g) / (2 * mu0)
        if not F_pert > F_orig:
            return (False, f"Perturbed F {F_pert} not greater than original F {F_orig}")
        return (True, f"B increase leads to F increase: {F_orig} -> {F_pert}")

    # ------------------------------------------------------------------
    # WEAK ANSWER SANITY (no isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_broad_ranges(self):
        """Check that results are within plausible broad ranges."""
        fail_msgs = []
        # Power: typical electromagnet ~10^4 to 10^6 W
        if self.P is not None:
            if not (1e4 <= self.P <= 1e6):
                fail_msgs.append(f"P={self.P} not in [1e4, 1e6]")
        else:
            fail_msgs.append("P missing")
        # Mass of copper: a few hundred kg
        if self.m is not None:
            if not (100 <= self.m <= 1000):
                fail_msgs.append(f"m={self.m} not in [100, 1000]")
        else:
            fail_msgs.append("m missing")
        # Force: ~10^5 to 10^6 N
        if self.F is not None:
            if not (1e5 <= self.F <= 1e6):
                fail_msgs.append(f"F={self.F} not in [1e5, 1e6]")
        else:
            fail_msgs.append("F missing")
        if fail_msgs:
            return (False, "; ".join(fail_msgs))
        return (True, "All results in broad plausible ranges")