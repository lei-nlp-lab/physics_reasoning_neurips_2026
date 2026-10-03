import math

class PhysicsChecker:
    """
    Audits a solution instance for the problem:
    "Hydrogen atom in first excited (2p) state in a cavity: find temperature T
     where spontaneous and induced emission probabilities are equal."
    """

    def __init__(self, solution_instance):
        # Safely extract givens, results, and other attributes
        # Expected givens: h, k, c, lambda_lya (or can be computed from them)
        # Expected results: T

        # Results dictionary
        results = getattr(solution_instance, 'results', {}) or {}
        # Givens dictionary
        givens = getattr(solution_instance, 'givens', {}) or {}

        # Store the extracted values (None if missing)
        self.T = results.get('T', None)

        # Physical constants (SI units)
        self.h = givens.get('h', None) or results.get('h', None)  # Planck constant
        self.k = givens.get('k', None) or results.get('k', None)  # Boltzmann constant
        self.c = givens.get('c', None) or results.get('c', None)  # speed of light
        # Lyman-alpha wavelength (2p -> 1s)
        self.lambda_lya = givens.get('lambda_lya', None) or results.get('lambda_lya', None)

        # Optionally store explanation steps and pseudocode for context
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

    def run_all_tests(self):
        """Discover and run all test_ prefixed methods, return (success, logs)."""
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

    # -----------------------------------------------------------------
    # Interface checks – ensure required keys and types exist
    # -----------------------------------------------------------------
    def test_interface_required_results(self):
        """Check that results key 'T' is present and is a number."""
        if self.T is None:
            return (False, "Missing key 'T' in results")
        if not isinstance(self.T, (int, float)):
            return (False, "T is not numeric")
        return (True, "T present and numeric")

    def test_interface_givens_present(self):
        """Check that required givens (h, k, c, lambda_lya) are available."""
        missing = []
        for name, val in [('h', self.h), ('k', self.k), ('c', self.c), ('lambda_lya', self.lambda_lya)]:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        # Also check they are positive numbers
        for name, val in [('h', self.h), ('k', self.k), ('c', self.c), ('lambda_lya', self.lambda_lya)]:
            if not isinstance(val, (int, float)) or val <= 0:
                return (False, f"Given {name} is not a positive number")
        return (True, "All required givens present and positive")

    # -----------------------------------------------------------------
    # Invariant tests – theorem-based physical sanity (use math.isclose)
    # -----------------------------------------------------------------
    def test_invariant_T_positive(self):
        """T must be positive (absolute temperature)."""
        if self.T is None:
            return (False, "T not available")
        if self.T <= 0:
            return (False, f"T = {self.T} is not positive")
        return (True, f"T = {self.T} > 0")

    def test_invariant_einstein_condition(self):
        """
        Check that exp(h*nu/(k*T)) - 1 = 1, i.e., exp = 2.
        Since the condition derived from equality of spontaneous and induced emissions
        requires this relation. Uses the reported T and computed nu from lambda_lya.
        """
        if any(v is None for v in [self.T, self.h, self.k, self.lambda_lya, self.c]):
            return (False, "Missing data for Einstein condition check")
        nu = self.c / self.lambda_lya
        exponent = self.h * nu / (self.k * self.T)
        # exp(exponent) should equal 2
        if not math.isclose(math.exp(exponent), 2.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"exp(hν/(kT)) = {math.exp(exponent):.4f} not close to 2")
        return (True, "Einstein condition satisfied: exp(hν/(kT)) ≈ 2")

    def test_invariant_T_formula(self):
        """
        Re-derive T from givens and compare with reported T.
        Uses formula T = h*nu/(k*ln2) with nu = c/lambda_lya.
        (This is a direct consequence of the physics.)
        """
        if any(v is None for v in [self.T, self.h, self.k, self.lambda_lya, self.c]):
            return (False, "Missing data for T formula check")
        nu = self.c / self.lambda_lya
        expected_T = self.h * nu / (self.k * math.log(2))
        if not math.isclose(self.T, expected_T, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Reported T = {self.T:6e} K, expected T = {expected_T:6e} K")
        return (True, f"T matches formula: {self.T:.3e} K ≈ {expected_T:.3e} K")

    def test_invariant_frequency_positive(self):
        """Transition frequency must be positive and consistent with Lyman-alpha."""
        if any(v is None for v in [self.c, self.lambda_lya]):
            return (False, "Missing c or lambda_lya")
        nu = self.c / self.lambda_lya
        if nu <= 0:
            return (False, f"Frequency nu = {nu} is not positive")
        # Frequency should be about 2.466e15 Hz for Lyman-alpha
        if not (2.0e15 < nu < 3.0e15):
            return (False, f"Frequency nu = {nu:.3e} Hz seems off for Lyman-alpha")
        return (True, f"Frequency nu = {nu:.3e} Hz is reasonable for 2p->1s")

    # -----------------------------------------------------------------
    # Modeling / trace checks – verify intermediate relations
    # -----------------------------------------------------------------
    def test_modeling_planck_denominator(self):
        """
        If we have T, compute the Planck denominator u = exp(hν/(kT))-1.
        The condition A = B u(ν) implies u = 1, so denominator = 1.
        Check using the reported T.
        """
        if any(v is None for v in [self.T, self.h, self.k, self.lambda_lya, self.c]):
            return (False, "Missing data for Planck denominator check")
        nu = self.c / self.lambda_lya
        denom = math.exp(self.h * nu / (self.k * self.T)) - 1.0
        if not math.isclose(denom, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Denominator = {denom:.4f} not close to 1")
        return (True, "Planck denominator ≈ 1, consistent with equality condition")

    def test_modeling_einstein_AB_ratio(self):
        """
        Check that the ratio A/B computed from the formula equals the
        Planck energy density at the transition frequency and reported T.
        A/B = 8πhν³/c³, and u(ν) = A/B * 1/(exp(hν/kT)-1).
        Since A/B = (8πhν³/c³), we can check order-of-magnitude consistency.
        (This is a weak check, but uses the reported T and nu.)
        """
        if any(v is None for v in [self.T, self.h, self.k, self.lambda_lya, self.c]):
            return (False, "Missing data for Einstein ratio check")
        nu = self.c / self.lambda_lya
        # Compute A/B from formula
        AB_ratio = 8.0 * math.pi * self.h * (nu**3) / (self.c**3)
        # Compute u(ν) from Planck's law
        u_nu = AB_ratio / (math.exp(self.h * nu / (self.k * self.T)) - 1.0)
        # Condition: u(ν) should equal A/B (since spontaneous = induced implies u = A/B)
        # Actually condition: A = B u(ν) => u(ν) = A/B, so they should be equal.
        if not math.isclose(u_nu, AB_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"u(ν) = {u_nu:.3e} not close to A/B = {AB_ratio:.3e}")
        return (True, "u(ν) ≈ A/B, consistent with equality condition")

    # -----------------------------------------------------------------
    # Monotonicity tests – perturb input and check output direction
    # -----------------------------------------------------------------
    def test_monotonicity_T_with_nu(self):
        """
        Perturb frequency nu by +5% and check that the derived T (using the
        same relation T = hν/(k ln2) ) increases.
        Since we can't change the stored T, we simulate the relation.
        """
        if any(v is None for v in [self.T, self.h, self.k, self.lambda_lya, self.c]):
            return (False, "Missing data for monotonicity nu test")
        nu_base = self.c / self.lambda_lya
        # Compute T from formula for base and perturbed
        T_base = self.h * nu_base / (self.k * math.log(2))
        nu_pert = nu_base * 1.05
        T_pert = self.h * nu_pert / (self.k * math.log(2))
        if T_pert <= T_base:
            return (False, f"T should increase with nu, but T_pert={T_pert:.3e} <= T_base={T_base:.3e}")
        return (True, "T increases when nu increases (monotonic)")

    def test_monotonicity_T_with_h(self):
        """
        Perturb Planck constant h by +5% and check that T increases.
        """
        if any(v is None for v in [self.T, self.h, self.k, self.lambda_lya, self.c]):
            return (False, "Missing data for monotonicity h test")
        nu = self.c / self.lambda_lya
        T_base = self.h * nu / (self.k * math.log(2))
        h_pert = self.h * 1.05
        T_pert = h_pert * nu / (self.k * math.log(2))
        if T_pert <= T_base:
            return (False, f"T should increase with h, but T_pert={T_pert:.3e} <= T_base={T_base:.3e}")
        return (True, "T increases when h increases (monotonic)")

    def test_monotonicity_T_with_lambda(self):
        """
        Perturb wavelength lambda_lya by +5% (i.e., decrease nu) and check that T decreases.
        """
        if any(v is None for v in [self.T, self.h, self.k, self.lambda_lya, self.c]):
            return (False, "Missing data for monotonicity lambda test")
        nu_base = self.c / self.lambda_lya
        T_base = self.h * nu_base / (self.k * math.log(2))
        lambda_pert = self.lambda_lya * 1.05
        nu_pert = self.c / lambda_pert
        T_pert = self.h * nu_pert / (self.k * math.log(2))
        if T_pert >= T_base:
            return (False, f"T should decrease when lambda increases, but T_pert={T_pert:.3e} >= T_base={T_base:.3e}")
        return (True, "T decreases when lambda increases (monotonic)")

    # -----------------------------------------------------------------
    # Answer sanity – only weak checks, no isclose, no re-derivation
    # -----------------------------------------------------------------
    def test_answer_sanity_T_range(self):
        """Weak sanity: T should be positive and in a reasonable range for this physics."""
        if self.T is None:
            return (False, "T not available")
        if self.T <= 0:
            return (False, f"T = {self.T} is not positive")
        # Typical expected value ~1.7e5 K, allow two orders of magnitude
        if self.T < 1e3 or self.T > 1e7:
            return (False, f"T = {self.T:.2e} K outside expected range 1e3 - 1e7 K")
        return (True, f"T is in reasonable range ({self.T:.3e} K)")