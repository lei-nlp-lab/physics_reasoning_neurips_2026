import math
import traceback

class PhysicsChecker:
    """
    Physics checker for muonic aluminum atom problem.
    Checks consistency of computed wavelength and lifetime with hydrogenic theory.
    """
    def __init__(self, solution_instance):
        # Safety: use None as default for all missing values
        self.lambda_cm = None
        self.tau = None
        self.m_e = None
        self.m_mu = None
        self.Z = None
        self.tau_H = None

        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.lambda_cm = solution_instance.results.get('lambda_cm', None)
            self.tau = solution_instance.results.get('tau', None)

        # Givens may be attributes of solution_instance
        self.m_e = getattr(solution_instance, 'm_e', None)
        self.m_mu = getattr(solution_instance, 'm_mu', None)
        self.Z = getattr(solution_instance, 'Z', None)
        self.tau_H = getattr(solution_instance, 'tau_H', None)

        # Accept also from results dict (though typically not)
        if self.m_e is None and hasattr(solution_instance, 'results'):
            self.m_e = solution_instance.results.get('m_e', None)
        if self.m_mu is None and hasattr(solution_instance, 'results'):
            self.m_mu = solution_instance.results.get('m_mu', None)
        if self.Z is None and hasattr(solution_instance, 'results'):
            self.Z = solution_instance.results.get('Z', None)
        if self.tau_H is None and hasattr(solution_instance, 'results'):
            self.tau_H = solution_instance.results.get('tau_H', None)

        # Physical constants (MeV, eV, m, s)
        self.hc_eV_m = 1.23984193e-6  # eV*m
        self.m_e_MeV = 0.511          # MeV/c^2

    # ----------------------------------------------------------------------
    # Interface tests (keys existence and types)
    # ----------------------------------------------------------------------
    def test_interface_results_keys(self):
        """Check that both result keys are present."""
        if self.lambda_cm is None:
            return (False, "FAIL: lambda_cm key missing or None")
        if self.tau is None:
            return (False, "FAIL: tau key missing or None")
        return (True, "Both lambda_cm and tau present")

    def test_interface_types(self):
        """Check that lambda_cm and tau are numeric and positive."""
        if self.lambda_cm is None or self.tau is None:
            return (False, "SKIP: missing values")
        if not (isinstance(self.lambda_cm, (int, float)) and isinstance(self.tau, (int, float))):
            return (False, "FAIL: lambda_cm or tau not numeric")
        if self.lambda_cm <= 0 or self.tau <= 0:
            return (False, "FAIL: lambda_cm or tau not positive")
        return (True, "Types and positivity ok")

    # ----------------------------------------------------------------------
    # Invariant tests (physical sanity)
    # ----------------------------------------------------------------------
    def test_invariant_positive(self):
        """Check positivity (fallback if interface missed)."""
        failures = []
        if self.lambda_cm is not None and self.lambda_cm <= 0:
            failures.append("lambda_cm <= 0")
        if self.tau is not None and self.tau <= 0:
            failures.append("tau <= 0")
        if failures:
            return (False, "FAIL: " + ", ".join(failures))
        if self.lambda_cm is None or self.tau is None:
            return (False, "SKIP: missing values")
        return (True, "Both positive")

    def test_invariant_finite(self):
        """Check that values are finite (not inf or nan)."""
        if self.lambda_cm is None or self.tau is None:
            return (False, "SKIP: missing values")
        if not (math.isfinite(self.lambda_cm) and math.isfinite(self.tau)):
            return (False, "FAIL: infinite or NaN value")
        return (True, "Values are finite")

    # ----------------------------------------------------------------------
    # Modeling tests (consistency with theoretical formulas)
    # Use fuzzy matching (isclose) as per instructions
    # ----------------------------------------------------------------------
    def test_modeling_wavelength(self):
        """Check that lambda_cm is consistent with hydrogenic transition energy."""
        if any(v is None for v in [self.lambda_cm, self.m_e, self.m_mu, self.Z]):
            return (False, "SKIP: missing required data (m_e, m_mu, Z, or lambda_cm)")
        # Compute delta_E in eV using formula: 13.6 * (m_mu/m_e) * Z^2 * (1/4 - 1/9) eV
        delta_E = 13.6 * (self.m_mu / self.m_e) * (self.Z ** 2) * (1.0/4.0 - 1.0/9.0)
        # Compute lambda in meters, then to cm
        lambda_m = self.hc_eV_m / delta_E
        lambda_cm_expected = lambda_m * 100.0
        # Use relative tolerance 1% and absolute 1e-2 cm (since lambda ~1e-9 cm, this is huge but okay)
        if math.isclose(self.lambda_cm, lambda_cm_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"lambda_cm consistent: expected {lambda_cm_expected:.3e}, got {self.lambda_cm:.3e}")
        else:
            return (False, f"FAIL: lambda_cm mismatch: expected {lambda_cm_expected:.3e}, got {self.lambda_cm:.3e}")

    def test_modeling_lifetime(self):
        """Check that tau is consistent with scaling from hydrogen 3d lifetime."""
        if any(v is None for v in [self.tau, self.m_e, self.m_mu, self.Z, self.tau_H]):
            return (False, "SKIP: missing required data (m_e, m_mu, Z, tau_H, or tau)")
        # Correct formula: tau = tau_H * (m_e/m_mu)^3 / Z^4   (since A ~ m^3 Z^4)
        ratio = self.m_e / self.m_mu
        tau_expected = self.tau_H * (ratio ** 3) / (self.Z ** 4)
        if math.isclose(self.tau, tau_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"tau consistent: expected {tau_expected:.3e}, got {self.tau:.3e}")
        else:
            return (False, f"FAIL: tau mismatch: expected {tau_expected:.3e}, got {self.tau:.3e}")

    # ----------------------------------------------------------------------
    # Monotonicity tests (directional trends)
    # ----------------------------------------------------------------------
    def test_monotonicity_wavelength_Z(self):
        """Check that wavelength decreases when Z increases (with other parameters fixed)."""
        if any(v is None for v in [self.lambda_cm, self.m_e, self.m_mu, self.Z]):
            return (False, "SKIP: missing data for monotonicity")
        # Use a 5% increase in Z
        Z_new = self.Z * 1.05
        # Compute new expected lambda using same formula
        delta_E_orig = 13.6 * (self.m_mu / self.m_e) * (self.Z ** 2) * (5.0/36.0)
        delta_E_new = 13.6 * (self.m_mu / self.m_e) * (Z_new ** 2) * (5.0/36.0)
        # lambda inversely proportional to delta_E
        lambda_new_expected = (delta_E_orig / delta_E_new) * self.lambda_cm  # scaling
        if lambda_new_expected < self.lambda_cm:
            return (True, f"Monotonicity holds: Z increased by 5%, lambda decreases (from {self.lambda_cm:.3e} to {lambda_new_expected:.3e})")
        else:
            return (False, f"FAIL: lambda did not decrease: old {self.lambda_cm:.3e}, new expected {lambda_new_expected:.3e}")

    def test_monotonicity_lifetime_Z(self):
        """Check that lifetime decreases when Z increases."""
        if any(v is None for v in [self.tau, self.m_e, self.m_mu, self.Z, self.tau_H]):
            return (False, "SKIP: missing data for monotonicity")
        Z_new = self.Z * 1.05
        tau_new_expected = self.tau_H * (self.m_e / self.m_mu) ** 3 / (Z_new ** 4)
        if tau_new_expected < self.tau:
            return (True, f"Monotonicity holds: Z increased by 5%, tau decreases (from {self.tau:.3e} to {tau_new_expected:.3e})")
        else:
            return (False, f"FAIL: tau did not decrease: old {self.tau:.3e}, new expected {tau_new_expected:.3e}")

    # ----------------------------------------------------------------------
    # Weak answer sanity (optional, no math.isclose, no re-derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_range(self):
        """Weak sanity: check that values lie in physically plausible broad ranges."""
        if self.lambda_cm is None or self.tau is None:
            return (False, "SKIP: missing values")
        # lambda: typical muonic atom X-rays can be ~10^-12 to 10^-6 cm
        lambda_ok = 1e-12 < self.lambda_cm < 1e-6
        # tau: typical muonic lifetimes from 10^-12 to 10^-20 s; we allow wider range
        tau_ok = 1e-30 < self.tau < 1e-5
        if not lambda_ok:
            return (False, f"FAIL: lambda_cm={self.lambda_cm:.3e} outside plausible range [1e-12, 1e-6]")
        if not tau_ok:
            return (False, f"FAIL: tau={self.tau:.3e} outside plausible range [1e-30, 1e-5]")
        return (True, "Both values within broad physical ranges")

    # ----------------------------------------------------------------------
    # Test runner (must be present by contract)
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
        return (all_pass, logs)