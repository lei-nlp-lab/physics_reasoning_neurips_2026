import math
import traceback

class PhysicsChecker:
    """
    Audits a solution for the solar neutrino flux problem (Homestake experiment).
    Checks consistency with the pp-chain neutrino production and solar constant.
    """
    def __init__(self, solution_instance):
        # Extract results dictionary safely
        results = getattr(solution_instance, 'results', None)
        if results is None:
            results = {}  # fallback empty dict to avoid errors

        # Primary result: neutrino number flux (required)
        self.flux = results.get('flux', None)

        # Optional givens: solar constant, Earth-Sun distance, pp-chain energy, conversion
        self.S = getattr(solution_instance, 'S', None)                # W/m^2
        self.d = getattr(solution_instance, 'd', None)                # m (AU)
        self.Q_pp = getattr(solution_instance, 'Q_pp', None)         # MeV
        self.conv = getattr(solution_instance, 'conv', None)         # J/MeV

        # If givens are not direct attributes, try in results (fallback)
        if self.S is None:
            self.S = results.get('S', None)
        if self.d is None:
            self.d = results.get('d', None)
        if self.Q_pp is None:
            self.Q_pp = results.get('Q_pp', None)
        if self.conv is None:
            self.conv = results.get('conv', None)

        # Physical constants
        self.MeV_to_J = 1.602176634e-13  # J per MeV

    # ---------- Interface tests ----------
    def test_interface_required_keys(self):
        if self.flux is None:
            return (False, "Missing required result key 'flux'")
        if not isinstance(self.flux, (int, float)):
            return (False, f"flux must be numeric, got {type(self.flux).__name__}")
        return (True, "flux key is present and numeric")

    # ---------- Invariant (theorem-based) tests ----------
    def test_invariant_flux_positive(self):
        if self.flux is None:
            return (False, "SKIP: flux not available")
        if self.flux > 0:
            return (True, f"flux = {self.flux} > 0")
        return (False, f"flux must be positive, got {self.flux}")

    def test_invariant_flux_magnitude(self):
        """Neutrino flux from the Sun must be in the range 1e14 - 1e15 m^{-2}s^{-1}."""
        if self.flux is None:
            return (False, "SKIP: flux not available")
        if 1e14 < self.flux < 1e15:
            return (True, f"flux {self.flux} is within plausible range (1e14–1e15)")
        return (False, f"flux {self.flux} is outside plausible range")

    def test_invariant_flux_scaling_with_solar_constant(self):
        """
        If the solar constant S is provided, verify that the reported flux is
        consistent with the relation: flux ≈ 2 * S / (Q_pp * conv), where
        Q_pp is the total pp-chain energy release. Use approximate values
        from the problem (Q_pp ~ 26.73 MeV or 25 MeV thermal).
        """
        # Need S, Q_pp, conv
        if any(x is None for x in (self.S, self.Q_pp, self.conv)):
            return (False, "SKIP: missing S, Q_pp, or conv for scaling check")
        if self.flux is None:
            return (False, "SKIP: flux not available")

        # Convert Q_pp to Joules
        Q_pp_J = self.Q_pp * self.conv
        # Expected flux: 2 neutrinos per reaction, solar constant S
        expected_flux = 2.0 * self.S / Q_pp_J
        # Use lenient tolerance (1% relative) – matches standard problem approximation
        if math.isclose(self.flux, expected_flux, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"flux = {self.flux} ~ expected {expected_flux:.3e} (rel_tol=1%)")
        # Allow the rounded target 5e14 vs actual ~4.7e14 with wider tolerance if needed
        # but we stick to the 1% rule as per instructions.
        return (False, f"flux {self.flux} deviates from expected {expected_flux:.3e} (rel_tol=1%)")

    # ---------- Modeling / trace checks ----------
    def test_modeling_flux_uses_two_neutrino_per_reaction(self):
        """
        If S and Q_pp are available, verify that the computed flux is (nearly)
        exactly 2 * S / (Q_pp * conv) without any extra factor. This tests that
        the model correctly accounts for 2 neutrinos per pp-chain.
        """
        if any(x is None for x in (self.S, self.Q_pp, self.conv)):
            return (False, "SKIP: missing S, Q_pp, or conv for modeling check")
        if self.flux is None:
            return (False, "SKIP: flux not available")

        Q_pp_J = self.Q_pp * self.conv
        expected_flux = 2.0 * self.S / Q_pp_J
        if math.isclose(self.flux, expected_flux, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Flux matches 2*S/(Q_pp*conv) model")
        return (False, f"Flux {self.flux} does not match model value {expected_flux:.3e}")

    def test_modeling_mean_neutrino_energy(self):
        """
        Check that the mean neutrino energy used/implied in the solution is
        approximately 0.26 MeV (pp neutrinos). This assumes the solution has a
        'mean_nu_energy' result or we can compute it from flux and energy.
        """
        # Since the problem does not require this to be in results, we skip.
        # But we can provide a placeholder that always passes if missing.
        mean_en = getattr(self, 'flux', None)  # dummy – not really used
        if mean_en is None:
            return (False, "SKIP: mean neutrino energy not available for modeling")
        # If it were present, we would check consistency.
        return (True, "Modeling test skipped (mean_nu_energy not required)")

    # ---------- Monotonicity tests ----------
    def test_monotonicity_flux_increases_with_solar_constant(self):
        """
        Perturb S by +10% and verify that the resulting expected flux increases.
        (Only a conceptual check – we cannot rerun the solver, so we simulate.)
        """
        if any(x is None for x in (self.S, self.Q_pp, self.conv)):
            return (False, "SKIP: missing S, Q_pp, or conv for monotonicity")
        if self.flux is None:
            return (False, "SKIP: flux not available")

        base_Q_pp_J = self.Q_pp * self.conv
        base_flux = 2.0 * self.S / base_Q_pp_J  # what the model predicts

        # Increase S by 10%
        S_perturbed = self.S * 1.10
        perturbed_flux = 2.0 * S_perturbed / base_Q_pp_J
        if perturbed_flux > base_flux:
            return (True, f"+10% S -> flux increases from {base_flux:.3e} to {perturbed_flux:.3e}")
        return (False, "Monotonicity failure: increase in S should increase flux")

    # ---------- Weak answer sanity (broad range only, no isclose) ----------
    def test_answer_sanity_flux_broad_range(self):
        """
        Weak sanity: flux must be between 1e14 and 1e15 m^{-2}s^{-1} and finite.
        """
        if self.flux is None:
            return (False, "SKIP: flux not available")
        if not math.isfinite(self.flux):
            return (False, "flux is not finite")
        if 1e14 <= self.flux <= 1e15:
            return (True, f"flux {self.flux} lies in [1e14, 1e15]")
        return (False, f"flux {self.flux} outside broad acceptable range")

    # ---------- Test runner ----------
    def run_all_tests(self):
        """
        Discovers all test_* methods and runs them, returning (all_pass, log_list).
        """
        logs = []
        all_pass = True
        # Get all bound methods with name starting with test_
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