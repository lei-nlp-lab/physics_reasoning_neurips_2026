import math

class PhysicsChecker:
    """
    Audits a solution to the Scotchlite retro-reflector problem.
    Checks existence, physical invariants, modeling quality, and monotonicity.
    """

    def __init__(self, solution_instance):
        # Safely extract results dict and set defaults to None
        self.results = getattr(solution_instance, 'results', {})
        self.n = self.results.get('n', None)
        self.eta = self.results.get('eta', None)

    # ------------------------------------------------------------------
    # Interface tests: check that required keys exist and are valid types
    # ------------------------------------------------------------------
    def test_interface_n_exists(self):
        if self.n is None:
            return (False, "Missing: 'n' not found in results")
        return (True, "n is present")

    def test_interface_eta_exists(self):
        if self.eta is None:
            return (False, "Missing: 'eta' not found in results")
        return (True, "eta is present")

    def test_interface_n_is_number(self):
        if self.n is None:
            return (False, "SKIP: n is missing")
        if not isinstance(self.n, (int, float)):
            return (False, f"n has type {type(self.n)}, expected numeric")
        return (True, "n is numeric")

    def test_interface_eta_is_number(self):
        if self.eta is None:
            return (False, "SKIP: eta is missing")
        if not isinstance(self.eta, (int, float)):
            return (False, f"eta has type {type(self.eta)}, expected numeric")
        return (True, "eta is numeric")

    # ------------------------------------------------------------------
    # Invariant tests: theorem-based physical sanity (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_n_greater_than_one(self):
        if self.n is None:
            return (False, "SKIP: n is missing")
        if self.n <= 1:
            return (False, f"n = {self.n} <= 1, violates the condition n > 1 for a refracting medium")
        return (True, f"n = {self.n} > 1, OK")

    def test_invariant_eta_between_zero_and_one(self):
        if self.eta is None:
            return (False, "SKIP: eta is missing")
        if not (0 <= self.eta <= 1):
            return (False, f"eta = {self.eta} is not between 0 and 1")
        return (True, "eta in [0,1]")

    def test_invariant_retro_reflection_condition(self):
        """
        For perfect retro-reflection the focal length equals the sphere radius.
        Condition: n/(2*(n-1)) = 1.  Check with fuzzy tolerance.
        """
        if self.n is None:
            return (False, "SKIP: n is missing")
        if math.isclose(self.n, 1, rel_tol=1e-9):
            return (False, "n = 1 leads to infinite focal length, not retro-reflection")
        expected_ratio = 1.0
        actual_ratio = self.n / (2 * (self.n - 1))
        if math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"n/(2(n-1)) = {actual_ratio:.4f} ≈ 1, retro-reflection condition holds")
        else:
            return (False, f"n/(2(n-1)) = {actual_ratio:.4f} ≠ 1, retro-reflection condition violated")

    def test_invariant_fresnel_reflectance_positive(self):
        """
        Fresnel reflectance R_fresnel = ((n-1)/(n+1))^2 must be < 1.
        """
        if self.n is None:
            return (False, "SKIP: n is missing")
        r = ((self.n - 1) / (self.n + 1)) ** 2
        if r < 0 or r >= 1:
            return (False, f"Fresnel reflectance R = {r:.4f} is out of expected range [0,1)")
        return (True, f"Fresnel reflectance R = {r:.4f} is valid")

    # ------------------------------------------------------------------
    # Modeling / trace tests: check consistency with explanation steps
    # ------------------------------------------------------------------
    def test_modeling_transmission_formula(self):
        """
        Transmission through one surface: T = 4n/(n+1)^2 for normal incidence.
        Check that using self.n yields a finite positive number <= 1.
        """
        if self.n is None:
            return (False, "SKIP: n is missing")
        t = 4 * self.n / ((self.n + 1) ** 2)
        if not (0 < t <= 1):
            return (False, f"T = 4n/(n+1)^2 = {t:.4f} is not in (0,1]")
        return (True, f"T = {t:.4f} is physically plausible")

    def test_modeling_efficiency_ideal_bound(self):
        """
        Ideal efficiency (perfect back reflector) = T^2, where T = 4n/(n+1)^2.
        The given eta should not exceed this ideal value.
        """
        if self.n is None or self.eta is None:
            return (False, "SKIP: n or eta missing")
        t = 4 * self.n / ((self.n + 1) ** 2)
        ideal = t * t  # T^2 * R_back (R_back = 1)
        if self.eta > ideal + 1e-6:
            return (False, f"eta = {self.eta:.4f} exceeds ideal efficiency T^2 = {ideal:.4f}")
        return (True, f"eta = {self.eta:.4f} ≤ ideal T^2 = {ideal:.4f}, consistent")

    def test_modeling_efficiency_lower_bound(self):
        """
        Efficiency cannot be negative; for n≈2 and realistic coatings,
        typical eta ~0.75, so eta should be > 0.5 as a weak lower bound.
        """
        if self.eta is None:
            return (False, "SKIP: eta is missing")
        if self.eta <= 0.0:
            return (False, f"eta = {self.eta:.4f} is not positive")
        if self.eta < 0.5:
            return (False, f"eta = {self.eta:.4f} seems too low for typical retro-reflectors")
        return (True, f"eta = {self.eta:.4f} is plausible (>0.5)")

    # ------------------------------------------------------------------
    # Monotonicity test: explicitly perturb an input and compare outputs
    # ------------------------------------------------------------------
    def test_monotonicity_n_increases_T_decreases(self):
        """
        For n>1, the normal‑incidence transmission T = 4n/(n+1)^2 is monotonically decreasing.
        Perturb n upward by 5% and verify that the perturbed T is smaller than the baseline T.
        """
        if self.n is None:
            return (False, "SKIP: n is missing")

        baseline_n = self.n
        # Use a relative perturbation of 5%
        perturbed_n = baseline_n * 1.05

        baseline_T = 4 * baseline_n / ((baseline_n + 1) ** 2)
        perturbed_T = 4 * perturbed_n / ((perturbed_n + 1) ** 2)

        # Since derivative is negative for n>1, perturbed_T must be strictly smaller
        if perturbed_T >= baseline_T - 1e-9:
            return (False, f"Monotonicity violation: T({perturbed_n:.4f}) = {perturbed_T:.6f} "
                           f"should be < T({baseline_n:.4f}) = {baseline_T:.6f}")
        return (True, f"T decreases when n increases: {baseline_T:.6f} -> {perturbed_T:.6f}")

    # ------------------------------------------------------------------
    # Weak answer sanity checks (only one allowed; no isclose, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_n_and_eta_positive(self):
        """
        Minimal sanity: both n and eta are positive finite numbers.
        """
        if self.n is None or self.eta is None:
            return (False, "SKIP: n or eta missing")
        if not (math.isfinite(self.n) and self.n > 0):
            return (False, f"n = {self.n} is not a positive finite number")
        if not (math.isfinite(self.eta) and self.eta > 0):
            return (False, f"eta = {self.eta} is not a positive finite number")
        return (True, "n and eta are positive finite numbers")

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
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