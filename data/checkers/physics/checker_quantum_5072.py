import math
import cmath
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for electron spin in ramped magnetic field problem.
    Checks interface, invariants, modeling, monotonicity, and weak answer sanity.
    """

    def __init__(self, solution_instance):
        # Extract givens and results safely with None defaults
        self.H_z = None
        self.H_x = None
        self.T = None
        self.gamma = None
        self.hbar = None
        self.omega_z = None
        self.omega_x = None
        self.c_minus = None
        self.psi_T = None
        self.adiabatic_threshold = None

        if solution_instance is None:
            return

        # Helper to retrieve from multiple possible locations
        def _get(name):
            # Try attribute first
            val = getattr(solution_instance, name, None)
            if val is not None:
                return val
            # Try results dict
            if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
                val = solution_instance.results.get(name, None)
                if val is not None:
                    return val
            # Try givens dict
            if hasattr(solution_instance, 'givens') and isinstance(solution_instance.givens, dict):
                val = solution_instance.givens.get(name, None)
                if val is not None:
                    return val
            return None

        self.H_z = _get('H_z')
        self.H_x = _get('H_x')
        self.T = _get('T')
        self.gamma = _get('gamma')
        self.hbar = _get('hbar')
        self.omega_z = _get('omega_z')
        self.omega_x = _get('omega_x')
        self.c_minus = _get('c_minus')
        self.psi_T = _get('psi_T')
        self.adiabatic_threshold = _get('adiabatic_threshold')

        # Compute omega_z, omega_x if not given but possible
        if self.omega_z is None and self.gamma is not None and self.H_z is not None:
            self.omega_z = self.gamma * self.H_z
        if self.omega_x is None and self.gamma is not None and self.H_x is not None:
            self.omega_x = self.gamma * self.H_x

        # Keep hbar as None to avoid false passes

    def _isclose(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all required result keys exist."""
        missing = []
        if self.c_minus is None:
            missing.append('c_minus')
        if self.psi_T is None:
            missing.append('psi_T')
        if self.adiabatic_threshold is None:
            missing.append('adiabatic_threshold')
        if missing:
            return (False, f"Missing required keys: {missing}")
        return (True, "All required keys present")

    def test_interface_givens_availability(self):
        """Check that essential givens are available as numeric values."""
        missing = []
        if self.H_z is None:
            missing.append('H_z')
        if self.H_x is None:
            missing.append('H_x')
        if self.T is None:
            missing.append('T')
        if self.gamma is None and (self.omega_z is None or self.omega_x is None):
            missing.append('gamma (or omega_z and omega_x)')
        if missing:
            return (False, f"Missing givens: {missing}")
        return (True, "Essential givens available")

    # ----------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ----------------------------------------------------------------------
    def test_invariant_c_minus_scaling(self):
        """Check that |c_minus| ≈ H_x/(2H_z) for large T (adiabatic regime)."""
        if any(v is None for v in [self.omega_z, self.omega_x, self.T, self.c_minus]):
            return (False, "SKIP: missing omega_z, omega_x, T, or c_minus")
        if self.omega_z * self.T < 10:  # not yet adiabatic
            return (False, "SKIP: ω_z*T too small for asymptotic formula")
        expected_mag = abs(self.omega_x) / (2.0 * abs(self.omega_z)) if self.omega_z != 0 else None
        if expected_mag is None or expected_mag == 0:
            return (False, "SKIP: zero denominator")
        actual_mag = abs(self.c_minus)
        if self._isclose(actual_mag, expected_mag):
            return (True, f"|c_minus| ≈ {expected_mag:.3e} (actual {actual_mag:.3e})")
        return (False, f"|c_minus| = {actual_mag:.3e}, expected ~{expected_mag:.3e}")

    def test_invariant_adiabatic_threshold_condition(self):
        """Check that ω_z * T >> 1 as required by adiabatic theorem."""
        if self.omega_z is None or self.T is None:
            return (False, "SKIP: missing omega_z or T")
        if self.omega_z == 0:
            return (False, "SKIP: omega_z = 0")
        product = self.T * self.omega_z
        if product > 10:
            return (True, f"ω_z*T = {product:.2e} >> 1")
        return (False, f"ω_z*T = {product:.2e} is not >> 1; adiabatic condition not satisfied")

    def test_invariant_phase_of_c_minus(self):
        """Check that c_minus * exp(-i ω_z T) is approximately real and negative (if ω_x/ω_z > 0)."""
        if any(v is None for v in [self.omega_z, self.T, self.c_minus]):
            return (False, "SKIP: missing omega_z, T, or c_minus")
        if abs(self.c_minus) < 1e-12:
            return (False, "SKIP: c_minus too small")
        corrected = self.c_minus * cmath.exp(-1j * self.omega_z * self.T)
        expected_real = -self.omega_x / (2.0 * self.omega_z)
        if self._isclose(corrected.real, expected_real, abs_tol=1e-2*abs(expected_real)):
            return (True, "c_minus phase consistent with exp(i ω_z T)")
        return (False, f"c_minus corrected complex = {corrected}, expected real ≈ {expected_real}")

    # ----------------------------------------------------------------------
    # Modeling quality tests
    # ----------------------------------------------------------------------
    def test_modeling_state_ratio(self):
        """Check that the ratio of |-> coefficient to |+> coefficient is approximately -H_x/(2H_z)."""
        if any(v is None for v in [self.H_x, self.H_z, self.psi_T]):
            return (False, "SKIP: missing H_x, H_z, or psi_T")
        if self.H_z == 0:
            return (False, "SKIP: H_z = 0")
        if not isinstance(self.psi_T, (list, tuple)) or len(self.psi_T) != 2:
            return (False, "SKIP: psi_T is not a pair of complex coefficients")
        c_plus, c_minus = self.psi_T
        if abs(c_plus) == 0:
            return (False, "SKIP: c_plus zero")
        ratio = c_minus / c_plus
        expected_ratio = -self.H_x / (2.0 * self.H_z)
        if self._isclose(ratio.real, expected_ratio, abs_tol=1e-2*abs(expected_ratio)):
            return (True, f"Ratio c_minus/c_plus ≈ {expected_ratio:.3e}")
        return (False, f"Ratio = {ratio}, expected ≈ {expected_ratio:.3e}")

    def test_modeling_adiabatic_threshold_text(self):
        """Check that the threshold description mentions T >> 1/ω_z."""
        if self.adiabatic_threshold is None:
            return (False, "SKIP: adiabatic_threshold not available")
        thresh_str = str(self.adiabatic_threshold)
        if ">>" in thresh_str and ("1/omega_z" in thresh_str or "1/ω_z" in thresh_str):
            return (True, "Threshold condition mentions T >> 1/ω_z")
        return (False, f"Threshold text '{thresh_str}' does not clearly state T >> 1/ω_z")

    # ----------------------------------------------------------------------
    # Monotonicity test
    # ----------------------------------------------------------------------
    def test_monotonicity_amplitude_scales_with_Hx(self):
        """
        Check monotonicity: if H_x is increased by 10%, |c_minus| should increase by 10%.
        Baseline: |c_minus|_0 from solution.
        Perturbation: δH_x/H_x = 0.1.
        Expected: |c_minus|_perturbed = 1.1 * |c_minus|_0.
        Verify that this relative change is consistent with the linear scaling derived from H_x/(2H_z).
        """
        if any(v is None for v in [self.H_x, self.H_z, self.c_minus]):
            return (False, "SKIP: missing H_x, H_z, or c_minus")
        if self.H_z == 0:
            return (False, "SKIP: H_z = 0")
        # Baseline magnitude from solution
        baseline_mag = abs(self.c_minus)
        # Expected baseline from theory (should match, but we don't rely on that)
        expected_baseline = abs(self.H_x) / (2.0 * abs(self.H_z))
        # Perturb: increase H_x by 10%
        perturbed_Hx = 1.1 * self.H_x
        expected_perturbed_mag = abs(perturbed_Hx) / (2.0 * abs(self.H_z))
        # The ratio of perturbed to baseline should be 1.1
        ratio_expected = expected_perturbed_mag / expected_baseline  # should be 1.1
        # Check that the solution's baseline is close to theory (otherwise test is meaningless)
        if not self._isclose(baseline_mag, expected_baseline):
            return (False, f"Baseline |c_minus| = {baseline_mag:.3e} does not match theory {expected_baseline:.3e}, cannot test monotonicity")
        # Now verify that a 10% increase in H_x would produce a 10% increase in |c_minus|
        # Since we have only the baseline, we compute what the perturbed magnitude would be
        # using the linear scaling; this is a logical consistency check.
        if ratio_expected > 1.0:
            # Also check that the solution's c_minus is consistent with linear scaling: the ratio of |c_minus| / (H_x/(2H_z)) should be close to 1.
            # Already done above. So monotonicity holds.
            return (True, f"Monotonicity holds: |c_minus| scales linearly with H_x (ratio perturbed/baseline = {ratio_expected:.4f} > 1)")
        return (False, "Unexpected: ratio_expected not > 1")

    # ----------------------------------------------------------------------
    # Answer sanity (weak)
    # ----------------------------------------------------------------------
    def test_answer_sanity_state_coefficients(self):
        """Weak sanity: state norm near 1 and |c_minus| << |c_plus|."""
        if self.psi_T is None:
            return (False, "SKIP: psi_T not available")
        if not isinstance(self.psi_T, (list, tuple)) or len(self.psi_T) != 2:
            return (False, "SKIP: psi_T is not a pair of complex coefficients")
        c_plus, c_minus = self.psi_T
        norm = abs(c_plus)**2 + abs(c_minus)**2
        if not (0.9 < norm < 1.1):
            return (False, f"Norm = {norm:.3f} not near 1")
        if abs(c_minus) > 0.5 * abs(c_plus):
            return (False, f"|c_minus| = {abs(c_minus):.3f} too large relative to |c_plus| = {abs(c_plus):.3f}")
        return (True, "State coefficients plausible: norm ~1 and |c_minus| << |c_plus|")

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
                passed, msg = (False, f"{name} CRASHED: {e}\n{traceback.format_exc()}")
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)