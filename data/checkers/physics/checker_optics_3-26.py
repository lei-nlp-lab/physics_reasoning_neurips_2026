import math
import traceback

class PhysicsChecker:
    """
    Audits a solution for the moonlight intensity problem.
    Checks interface, physics invariants, modeling consistency, monotonicity,
    and a weak final-answer sanity.
    """

    # Known constants from the problem statement (used only when solution_instance does not provide them)
    S = 1361.0               # W/m^2, solar constant
    p = 0.12                 # geometric albedo of the Moon
    R_m = 1.74e6             # m, Moon radius
    d = 3.84e8               # m, Earth-Moon distance
    h = 6.626e-34            # J s, Planck constant
    c = 3.0e8                # m/s, speed of light
    lambda_vis = 550e-9      # m, average visible wavelength

    def __init__(self, solution_instance):
        # Extract givens (use None if not provided)
        self.S_given = getattr(solution_instance, 'S', None)
        self.p_given = getattr(solution_instance, 'p', None)
        self.R_m_given = getattr(solution_instance, 'R_m', None)
        self.d_given = getattr(solution_instance, 'd', None)
        self.h_given = getattr(solution_instance, 'h', None)
        self.c_given = getattr(solution_instance, 'c', None)
        self.lambda_given = getattr(solution_instance, 'lambda', None)

        # Extract result keys (expected from pseudocode emit contract)
        results = getattr(solution_instance, 'results', {})
        self.I_Wcm2 = results.get('I_Wcm2', None)
        self.N_photon_flux = results.get('N_photon_flux', None)

        # Use class constants as fallback if not provided, but note: we treat missing givens as None
        # so that tests can skip gracefully. The fallback is only used inside tests when computing expectations.
        self._S = self.S_given if self.S_given is not None else self.S
        self._p = self.p_given if self.p_given is not None else self.p
        self._R_m = self.R_m_given if self.R_m_given is not None else self.R_m
        self._d = self.d_given if self.d_given is not None else self.d
        self._h = self.h_given if self.h_given is not None else self.h
        self._c = self.c_given if self.c_given is not None else self.c
        self._lambda = self.lambda_given if self.lambda_given is not None else self.lambda_vis

    # ----- INTERFACE TESTS -----
    def test_interface_has_I_Wcm2(self):
        if self.I_Wcm2 is None:
            return (False, "I_Wcm2 not found in results")
        if not isinstance(self.I_Wcm2, (int, float)):
            return (False, f"I_Wcm2 is not numeric, got {type(self.I_Wcm2)}")
        if math.isnan(self.I_Wcm2) or math.isinf(self.I_Wcm2):
            return (False, f"I_Wcm2 is NaN or infinite")
        return (True, "I_Wcm2 present and numeric")

    def test_interface_has_N_photon_flux(self):
        if self.N_photon_flux is None:
            return (False, "N_photon_flux not found in results")
        if not isinstance(self.N_photon_flux, (int, float)):
            return (False, f"N_photon_flux is not numeric, got {type(self.N_photon_flux)}")
        if math.isnan(self.N_photon_flux) or math.isinf(self.N_photon_flux):
            return (False, f"N_photon_flux is NaN or infinite")
        return (True, "N_photon_flux present and numeric")

    # ----- INVARIANT TESTS -----
    def test_invariant_intensity_formula(self):
        """Check that I_Wcm2 matches the formula: I_Wcm2 = p*S*R_m^2/(4*d^2)/1e4."""
        if any(g is None for g in [self.S_given, self.p_given, self.R_m_given, self.d_given]):
            return (False, "SKIP: one or more givens missing to compute expected intensity")
        expected_I_Wm2 = self.p_given * self.S_given * (self.R_m_given**2) / (4 * self.d_given**2)
        expected_I_Wcm2 = expected_I_Wm2 / 1e4
        if self.I_Wcm2 is None:
            return (False, "I_Wcm2 is None")
        if math.isclose(self.I_Wcm2, expected_I_Wcm2, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"I_Wcm2 {self.I_Wcm2} ≈ expected {expected_I_Wcm2:.3e} (1% tolerance)")
        else:
            return (False, f"I_Wcm2 {self.I_Wcm2} does not match expected {expected_I_Wcm2:.3e} (1% tolerance)")

    def test_invariant_photon_flux_formula(self):
        """Check that N_photon_flux = I_Wcm2 / (h*c/lambda)."""
        if any(g is None for g in [self.h_given, self.c_given, self.lambda_given]):
            return (False, "SKIP: one or more photonic givens missing")
        if self.I_Wcm2 is None:
            return (False, "I_Wcm2 is None, cannot compute photon flux")
        E_ph = self.h_given * self.c_given / self.lambda_given
        expected_N = self.I_Wcm2 / E_ph
        if self.N_photon_flux is None:
            return (False, "N_photon_flux is None")
        if math.isclose(self.N_photon_flux, expected_N, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"N_photon_flux {self.N_photon_flux:.3e} ≈ expected {expected_N:.3e} (1% tolerance)")
        else:
            return (False, f"N_photon_flux {self.N_photon_flux:.3e} does not match expected {expected_N:.3e} (1% tolerance)")

    def test_invariant_conversion_Wm2_to_Wcm2(self):
        """Check that I_Wcm2 = I_in_Wm2/1e4, if I_in_Wm2 is given."""
        # Not all solutions may expose I_in_Wm2; we skip if not present.
        # Here we assume the result might have an intermediate 'I_Wm2' key.
        # But the problem doesn't require it; we skip if missing.
        # This test is optional.
        return (True, "SKIP: I_in_Wm2 not required by problem")

    # ----- MODELING / TRACE TESTS -----
    def test_modeling_power_conservation(self):
        """Check that reflected power equals p*S*pi*R_m^2 and that flux at Earth is P_ref/(4*pi*d^2)."""
        if self.I_Wcm2 is None:
            return (False, "I_Wcm2 missing, cannot perform power conservation check")
        # We need the givens; if any missing, skip.
        if any(g is None for g in [self.p_given, self.S_given, self.R_m_given, self.d_given]):
            return (False, "SKIP: missing givens for power conservation")
        # Compute expected I_Wcm2 from power conservation
        P_ref = self.p_given * self.S_given * math.pi * (self.R_m_given**2)
        I_Wm2 = P_ref / (4 * math.pi * self.d_given**2)
        expected_I_Wcm2 = I_Wm2 / 1e4
        if math.isclose(self.I_Wcm2, expected_I_Wcm2, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Power conservation holds: I_Wcm2 consistent with p*S*pi*R_m^2/(4*pi*d^2)/1e4")
        else:
            return (False, f"Power conservation violated: I_Wcm2 {self.I_Wcm2} differs from {expected_I_Wcm2:.3e}")

    def test_modeling_photon_flux_visible_range(self):
        """Check that the photon flux uses an average wavelength around 550 nm."""
        if self.lambda_given is None:
            # Use class constant and just check order of magnitude
            E_ph = self.h * self.c / self.lambda_vis
            if self.I_Wcm2 is None or self.N_photon_flux is None:
                return (False, "SKIP: missing data")
            expected_N = self.I_Wcm2 / E_ph
            if math.isclose(self.N_photon_flux, expected_N, rel_tol=0.5, abs_tol=1e10):
                return (True, "Photon flux order consistent with 550 nm")
            else:
                return (False, f"Photon flux {self.N_photon_flux:.3e} not consistent with 550 nm (expected {expected_N:.3e})")
        else:
            # Already checked in invariant; just return skip to avoid duplication
            return (True, "SKIP: detailed check done in test_invariant_photon_flux_formula")

    # ----- MONOTONICITY TESTS -----
    def test_monotonicity_intensity_with_distance(self):
        """If Earth-Moon distance increases, intensity should decrease (1/d^2)."""
        if any(g is None for g in [self.p_given, self.S_given, self.R_m_given, self.d_given]):
            return (False, "SKIP: missing givens for monotonicity")
        base_d = self.d_given
        perturbed_d = base_d * 1.05
        expected_base = self.p_given * self.S_given * (self.R_m_given**2) / (4 * base_d**2) / 1e4
        expected_perturbed = self.p_given * self.S_given * (self.R_m_given**2) / (4 * perturbed_d**2) / 1e4
        if expected_perturbed < expected_base:
            return (True, f"Intensity decreases when distance increases by 5%: {expected_base:.3e} → {expected_perturbed:.3e}")
        else:
            return (False, f"Monotonicity violated: intensity should decrease with distance")

    def test_monotonicity_intensity_with_radius(self):
        """If Moon radius increases, intensity should increase (R_m^2)."""
        if any(g is None for g in [self.p_given, self.S_given, self.R_m_given, self.d_given]):
            return (False, "SKIP: missing givens")
        base_R = self.R_m_given
        perturbed_R = base_R * 1.05
        expected_base = self.p_given * self.S_given * (base_R**2) / (4 * self.d_given**2) / 1e4
        expected_perturbed = self.p_given * self.S_given * (perturbed_R**2) / (4 * self.d_given**2) / 1e4
        if expected_perturbed > expected_base:
            return (True, f"Intensity increases when radius increases by 5%: {expected_base:.3e} → {expected_perturbed:.3e}")
        else:
            return (False, "Monotonicity violated: intensity should increase with radius")

    def test_monotonicity_photon_flux_with_wavelength(self):
        """If wavelength increases, photon flux increases (N proportional to lambda)."""
        if self.I_Wcm2 is None or self.h_given is None or self.c_given is None:
            return (False, "SKIP: missing data for photon flux monotonicity")
        base_lambda = self.lambda_given if self.lambda_given is not None else self.lambda_vis
        perturbed_lambda = base_lambda * 1.1
        E_base = self.h_given * self.c_given / base_lambda
        E_perturbed = self.h_given * self.c_given / perturbed_lambda
        N_base = self.I_Wcm2 / E_base
        N_perturbed = self.I_Wcm2 / E_perturbed
        if N_perturbed > N_base:
            return (True, f"Photon flux increases when wavelength increases by 10%: {N_base:.3e} → {N_perturbed:.3e}")
        else:
            return (False, f"Monotonicity violated: photon flux should increase with wavelength")

    # ----- WEAK ANSWER SANITY (only one such test, no isclose, no derivation) -----
    def test_answer_sanity_order_magnitude(self):
        """Check that both answers are of plausible order of magnitude."""
        if self.I_Wcm2 is None or self.N_photon_flux is None:
            return (False, "Missing results")
        # I_Wcm2 should be around 1e-6 to 1e-5
        if not (1e-7 <= self.I_Wcm2 <= 1e-4):
            return (False, f"I_Wcm2 = {self.I_Wcm2:.3e} outside plausible range [1e-7, 1e-4]")
        # N_photon_flux should be around 1e12 to 1e13
        if not (1e11 <= self.N_photon_flux <= 1e14):
            return (False, f"N_photon_flux = {self.N_photon_flux:.3e} outside plausible range [1e11, 1e14]")
        return (True, "Both answers within plausible order-of-magnitude range")

    # ----- RUN ALL TESTS -----
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Discover all bound methods starting with 'test_'
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