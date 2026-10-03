import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens safely (try multiple sources)
        self.nu0 = getattr(solution_instance, 'nu0', None)
        self.lambda_C = getattr(solution_instance, 'lambda_C', None)
        self.c = getattr(solution_instance, 'c', None)

        # Fallback to givens dict if present
        if self.nu0 is None and hasattr(solution_instance, 'givens'):
            givens = solution_instance.givens
            self.nu0 = givens.get('nu0', None)
            self.lambda_C = givens.get('lambda_C', None)
            self.c = givens.get('c', None)

        # Extract result (final frequency)
        self.nu_prime = None
        if hasattr(solution_instance, 'results'):
            self.nu_prime = solution_instance.results.get('nu_prime', None)
        if self.nu_prime is None:
            self.nu_prime = getattr(solution_instance, 'nu_prime', None)

        # Use standard values as fallback only for c (accepted physical constant)
        if self.c is None:
            self.c = 3.0e8  # m/s, only constant with no ambiguity
        # For consistency, if nu0 and c are known, compute initial wavelength
        self.lambda0 = None
        if self.nu0 is not None and self.nu0 > 0:
            self.lambda0 = self.c / self.nu0
        # Compute expected scattered wavelength from Compton formula (θ=90°)
        self.lambda_prime_expected = None
        if self.lambda0 is not None and self.lambda_C is not None:
            self.lambda_prime_expected = self.lambda0 + self.lambda_C
        # Compute expected final frequency from expected wavelength
        self.nu_prime_expected = None
        if self.lambda_prime_expected is not None and self.lambda_prime_expected > 0:
            self.nu_prime_expected = self.c / self.lambda_prime_expected

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

    # ------------------ Interface Tests ------------------
    def test_interface_nu_prime_exists(self):
        """Check that the final frequency is provided."""
        if self.nu_prime is None:
            return (False, "nu_prime is None")
        return (True, "nu_prime is present")

    def test_interface_nu_prime_positive(self):
        """Check that the final frequency is positive."""
        if self.nu_prime is None:
            return (False, "SKIP: nu_prime not available")
        if self.nu_prime <= 0:
            return (False, f"nu_prime = {self.nu_prime} is not positive")
        return (True, "nu_prime > 0")

    def test_interface_givens_present(self):
        """Check that essential givens (nu0, lambda_C, c) are present."""
        missing = []
        if self.nu0 is None:
            missing.append("nu0")
        if self.lambda_C is None:
            missing.append("lambda_C")
        if self.c is None:
            missing.append("c")
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All givens present")

    # ------------------ Invariant Tests (theorem-based) ------------------
    def test_invariant_positive_derived_quantities(self):
        """Check that derived quantities (lambda0, lambda_prime, nu_prime) are >0 when computable."""
        if self.lambda0 is not None and self.lambda0 <= 0:
            return (False, f"lambda0 = {self.lambda0} is not positive")
        if self.lambda_prime_expected is not None and self.lambda_prime_expected <= 0:
            return (False, f"lambda_prime_expected = {self.lambda_prime_expected} is not positive")
        if self.nu_prime_expected is not None and self.nu_prime_expected <= 0:
            return (False, f"nu_prime_expected = {self.nu_prime_expected} is not positive")
        return (True, "All derived quantities positive or not computable")

    def test_invariant_compton_shift(self):
        """Verify that scattered wavelength of provided answer matches Compton formula."""
        if self.nu_prime is None or self.lambda0 is None or self.lambda_C is None:
            return (False, "SKIP: insufficient data")
        # Compute wavelength from provided frequency
        lambda_prime_actual = self.c / self.nu_prime
        expected = self.lambda0 + self.lambda_C
        if math.isclose(lambda_prime_actual, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"lambda' (actual) ≈ {lambda_prime_actual:.3e}, expected {expected:.3e}")
        else:
            return (False, f"lambda' (actual) = {lambda_prime_actual:.3e}, expected {expected:.3e}")

    def test_invariant_energy_decrease(self):
        """Check that scattered frequency is lower than initial (photon loses energy)."""
        if self.nu0 is None or self.nu_prime is None:
            return (False, "SKIP: missing nu0 or nu_prime")
        if self.nu_prime < self.nu0:
            return (True, f"nu' = {self.nu_prime:.3e} < nu0 = {self.nu0:.3e}")
        else:
            return (False, f"nu' = {self.nu_prime:.3e} >= nu0 = {self.nu0:.3e} (not decreased)")

    # ------------------ Modeling Tests ------------------
    def test_modeling_frequency_derivation(self):
        """Re-derive final frequency from givens using Compton formula and compare with provided answer."""
        if self.nu_prime_expected is None or self.nu_prime is None:
            return (False, "SKIP: cannot compute expected nu_prime")
        if math.isclose(self.nu_prime, self.nu_prime_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"nu_prime provided = {self.nu_prime:.3e} matches re-derived = {self.nu_prime_expected:.3e}")
        else:
            return (False, f"nu_prime provided = {self.nu_prime:.3e} does not match re-derived = {self.nu_prime_expected:.3e}")

    def test_modeling_wavelength_shift_equals_compton(self):
        """Check that the amount of wavelength shift equals Compton wavelength (θ=90°)."""
        if self.lambda0 is None or self.nu_prime is None or self.lambda_C is None:
            return (False, "SKIP: missing data")
        lambda_prime_actual = self.c / self.nu_prime
        shift = lambda_prime_actual - self.lambda0
        if math.isclose(shift, self.lambda_C, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Δλ = {shift:.3e} ≈ λ_C = {self.lambda_C:.3e}")
        else:
            return (False, f"Δλ = {shift:.3e} != λ_C = {self.lambda_C:.3e}")

    # ------------------ Monotonicity Tests ------------------
    def test_monotonicity_frequency_increases_with_nu0(self):
        """Perturb initial frequency upward and verify scattered frequency also increases."""
        if self.nu0 is None or self.lambda_C is None:
            return (False, "SKIP: missing nu0 or lambda_C")
        # Use tolerances for finite difference
        baseline = self.c / (self.c / self.nu0 + self.lambda_C)  # λ' = c/nu0 + λ_C, then ν' = c/λ'
        perturbation_factor = 1.01
        perturbed_nu0 = self.nu0 * perturbation_factor
        perturbed = self.c / (self.c / perturbed_nu0 + self.lambda_C)
        if perturbed > baseline:
            return (True, f"nu0 increased by 1%: baseline ν' = {baseline:.3e}, perturbed ν' = {perturbed:.3e}")
        else:
            return (False, f"nu0 increased but ν' decreased: baseline {baseline:.3e}, perturbed {perturbed:.3e}")

    # ------------------ Answer Sanity Test (weak) ------------------
    def test_answer_sanity_nu_prime_range(self):
        """Check that the final frequency is in a physically plausible range (2e19 – 3e19 Hz)."""
        if self.nu_prime is None:
            return (False, "SKIP: nu_prime not available")
        if 2.0e19 <= self.nu_prime <= 3.0e19:
            return (True, f"nu_prime = {self.nu_prime:.3e} is in [2e19, 3e19]")
        else:
            return (False, f"nu_prime = {self.nu_prime:.3e} is outside expected range")