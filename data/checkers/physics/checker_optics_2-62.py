import math
import sys

class PhysicsChecker:
    """
    Auditor for telescope resolution problem.
    Checks consistency with Rayleigh criterion, unit conversion,
    and that the answer is physically plausible.
    """
    def __init__(self, solution_instance):
        # Extract givens – use None for missing values
        self.D = getattr(solution_instance, 'D', None)          # cm
        # wavelength: look for various common names
        wavelength = None
        for attr in ['lambda_cm', 'wavelength_cm', 'lambda_angstrom', 'wavelength_angstrom']:
            val = getattr(solution_instance, attr, None)
            if val is not None:
                wavelength = val
                if 'angstrom' in attr.lower():
                    # convert to cm: 1 Å = 1e-8 cm
                    wavelength = val * 1e-8
                break
        if wavelength is None and hasattr(solution_instance, 'givens'):
            givens = solution_instance.givens
            if isinstance(givens, dict):
                wavelength = givens.get('lambda', None)
                if wavelength is None:
                    wavelength = givens.get('wavelength', None)
                if isinstance(wavelength, (int, float)):
                    # assume it's in angstroms (typical)
                    wavelength = wavelength * 1e-8
        self.lambda_cm = wavelength

        # result key
        self.theta_arcsec = None
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.theta_arcsec = solution_instance.results.get('theta_arcsec', None)

        # optional but given in problem
        self.focal_length = getattr(solution_instance, 'focal_length', None)   # m (or cm)
        self.magnification = getattr(solution_instance, 'magnification', None)

        # fundamental conversion factor
        self.conversion_factor = 206265.0

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_theta_arcsec_key(self):
        """Check that result key 'theta_arcsec' is present and numeric."""
        if self.theta_arcsec is None:
            return (False, "Missing 'theta_arcsec' in results")
        if not isinstance(self.theta_arcsec, (int, float)):
            return (False, f"'theta_arcsec' is not numeric: {type(self.theta_arcsec).__name__}")
        return (True, "theta_arcsec key present and numeric")

    def test_interface_diameter_available(self):
        """Check that the objective diameter D is provided."""
        if self.D is None:
            return (False, "Missing D (objective diameter)")
        if not isinstance(self.D, (int, float)) or self.D <= 0:
            return (False, f"D is not a positive number: {self.D}")
        return (True, f"D = {self.D} cm")

    def test_interface_lambda_available(self):
        """Check that wavelength is provided."""
        if self.lambda_cm is None:
            return (False, "Missing wavelength (lambda_cm)")
        if not isinstance(self.lambda_cm, (int, float)) or self.lambda_cm <= 0:
            return (False, f"Wavelength is not a positive number: {self.lambda_cm}")
        return (True, f"λ = {self.lambda_cm:.3e} cm")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_rayleigh_criterion(self):
        """
        Check that theta_arcsec satisfies the Rayleigh criterion:
        θ = 1.22 λ / D  -->  θ_arcsec = 1.22 * λ / D * 206265
        """
        if any(v is None for v in [self.D, self.lambda_cm, self.theta_arcsec]):
            return (False, "SKIP: missing D, λ, or theta_arcsec")
        # compute expected value
        expected_theta_arcsec = 1.22 * self.lambda_cm / self.D * self.conversion_factor
        if not math.isclose(self.theta_arcsec, expected_theta_arcsec, rel_tol=1e-2, abs_tol=1e-2):
            msg = f"theta_arcsec = {self.theta_arcsec:.4f} but Rayleigh criterion gives {expected_theta_arcsec:.4f}"
            return (False, msg)
        return (True, f"θ_arcsec = {self.theta_arcsec:.4f} consistent with Rayleigh criterion")

    def test_invariant_conversion_rad_to_arcsec(self):
        """Check that the value is positive and conversion factor is used."""
        if self.theta_arcsec is None:
            return (False, "SKIP: no theta_arcsec")
        # compute equivalent radian value
        theta_rad = self.theta_arcsec / self.conversion_factor
        # check that it's small (diffraction limit ~ 1e-5 rad)
        if theta_rad <= 0 or theta_rad > 1e-3:
            return (False, f"Implied θ_rad = {theta_rad:.2e} not in plausible range for telescope diffraction")
        return (True, f"θ_rad ~ {theta_rad:.2e} (valid)")

    def test_invariant_sign_and_finite(self):
        """Check that theta_arcsec is positive and finite."""
        if self.theta_arcsec is None:
            return (False, "SKIP: no theta_arcsec")
        if not math.isfinite(self.theta_arcsec):
            return (False, "theta_arcsec is not finite")
        if self.theta_arcsec <= 0:
            return (False, "theta_arcsec is not positive")
        return (True, "theta_arcsec is positive and finite")

    # ------------------------------------------------------------------
    # Modeling / trace tests
    # ------------------------------------------------------------------
    def test_modeling_independence_of_focal_length(self):
        """
        Verify that focal length is not used in the final answer.
        (The diffraction limit depends only on λ and D.)
        """
        if self.focal_length is None:
            return (True, "SKIP: focal_length not available")
        # If focal length were used, the answer would likely change.
        # We can't know, but we can check that the answer is compatible
        # with the Rayleigh criterion only.
        if any(v is None for v in [self.D, self.lambda_cm, self.theta_arcsec]):
            return (False, "SKIP: missing D, λ, or theta_arcsec")
        expected = 1.22 * self.lambda_cm / self.D * self.conversion_factor
        if math.isclose(self.theta_arcsec, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Answer matches pure diffraction limit; focal length not used")
        else:
            return (False, "Answer deviates from pure diffraction limit; possible use of focal length")

    def test_modeling_magnification_not_affecting(self):
        """
        Check that magnification is not used in the final answer.
        (The diffraction limit sets the resolution, not the eyepiece magnification.)
        """
        if self.magnification is None:
            return (True, "SKIP: magnification not available")
        # same as above
        if any(v is None for v in [self.D, self.lambda_cm, self.theta_arcsec]):
            return (False, "SKIP: missing D, λ, or theta_arcsec")
        expected = 1.22 * self.lambda_cm / self.D * self.conversion_factor
        if math.isclose(self.theta_arcsec, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "Answer matches diffraction limit; magnification not affecting")
        else:
            return (False, "Answer deviates from diffraction limit; possible effect of magnification")

    # ------------------------------------------------------------------
    # Monotonicity test (diameter perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_diameter_increase_decreases_theta(self):
        """
        For a fixed wavelength, increasing D should decrease the angular resolution.
        We use the given answer and artificially compute what a larger D would give.
        """
        if any(v is None for v in [self.D, self.lambda_cm, self.theta_arcsec]):
            return (False, "SKIP: missing D, λ, or theta_arcsec")
        baseline = self.theta_arcsec
        # Simulate a 10% increase in D
        D_new = self.D * 1.1
        expected_new = 1.22 * self.lambda_cm / D_new * self.conversion_factor
        if expected_new >= baseline:
            return (False, f"Increasing D from {self.D} to {D_new:.1f} gave expected θ={expected_new:.4f} which is not less than baseline {baseline:.4f}")
        return (True, f"Monotonicity holds: D+10% → θ decreases from {baseline:.4f} to {expected_new:.4f}")

    # ------------------------------------------------------------------
    # Answer sanity (weak)
    # ------------------------------------------------------------------
    def test_answer_sanity_theta_arcsec(self):
        """
        Very weak sanity: the answer should be positive and of order a few arcseconds.
        """
        if self.theta_arcsec is None:
            return (False, "SKIP: no theta_arcsec")
        if not isinstance(self.theta_arcsec, (int, float)):
            return (False, "theta_arcsec is not numeric")
        if self.theta_arcsec <= 0:
            return (False, "theta_arcsec <= 0, but angular resolution must be positive")
        # typical telescopic diffraction limit: for 8 cm, λ=6000 Å, θ≈2″
        if self.theta_arcsec > 100:   # far too large
            return (False, f"theta_arcsec = {self.theta_arcsec:.1f} is unreasonably large")
        if self.theta_arcsec < 0.1:   # far too small
            return (False, f"theta_arcsec = {self.theta_arcsec:.4f} is unrealistically small")
        return (True, f"theta_arcsec = {self.theta_arcsec:.2f} is within plausible range (0.1″ - 100″)")

    # ------------------------------------------------------------------
    # Test runner (mandatory)
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # discover all test methods
        for attr_name in sorted(dir(self)):
            if not attr_name.startswith("test_"):
                continue
            method = getattr(self, attr_name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{attr_name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {attr_name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

# ------------------------------------------------------------------
# Example usage (only for testing, not part of submission)
# ------------------------------------------------------------------
if __name__ == "__main__":
    # dummy solution instance
    class DummySolution:
        def __init__(self):
            self.D = 8.0
            self.lambda_cm = 6e-5
            self.focal_length = 150.0   # 1.5 m = 150 cm
            self.magnification = 80
            self.results = {'theta_arcsec': 2.0}
    sol = DummySolution()
    checker = PhysicsChecker(sol)
    success, logs = checker.run_all_tests()
    for l in logs:
        print(l)
    print(f"\nOverall: {'PASS' if success else 'FAIL'}")