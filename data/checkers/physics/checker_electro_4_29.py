import math
import traceback

class PhysicsChecker:
    """
    Auditor for X-ray reflection from metal surface (Drude model, Snell's law,
    Fresnel s-polarization).
    """

    def __init__(self, solution_instance):
        # Extract givens safely (use None for missing)
        givens = getattr(solution_instance, 'givens', {})
        self.omega = givens.get('omega', None)
        self.omega_p = givens.get('omega_p', None)
        self.theta = givens.get('theta', None)  # incident angle in radians

        # Extract results safely
        results = getattr(solution_instance, 'results', {})
        self.theta_0 = results.get('theta_0', None)
        self.R = results.get('R', None)

        # Derived quantities (for internal calculations)
        self.n = None
        if self.omega is not None and self.omega_p is not None:
            if self.omega > self.omega_p:
                self.n = math.sqrt(1.0 - (self.omega_p / self.omega) ** 2)

    # ------------------------------------------------------------------
    # Helper numerical computations (based on physics relationships)
    # ------------------------------------------------------------------
    def _compute_n(self, omega, omega_p):
        """Refractive index from Drude model (real, omega > omega_p)."""
        if omega is None or omega_p is None or omega <= omega_p:
            return None
        return math.sqrt(1.0 - (omega_p / omega) ** 2)

    def _compute_theta_0(self, omega, omega_p):
        """Critical angle from sin(theta0) = n."""
        n = self._compute_n(omega, omega_p)
        if n is None or n > 1.0:
            return None
        return math.asin(n)

    def _compute_R(self, omega, omega_p, theta):
        """Reflectivity for s‑polarization (theta < theta0)."""
        n = self._compute_n(omega, omega_p)
        if n is None:
            return None
        sin_theta_t = math.sin(theta) / n
        if abs(sin_theta_t) > 1.0:
            # Total reflection would occur (R = 1), but here we only expect partial
            return 1.0
        cos_theta_t = math.sqrt(1.0 - sin_theta_t * sin_theta_t)
        r_s = (math.cos(theta) - n * cos_theta_t) / (math.cos(theta) + n * cos_theta_t)
        return r_s * r_s

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_theta_0_exists(self):
        if self.theta_0 is None:
            return (False, "results['theta_0'] is missing (None)")
        return (True, "results['theta_0'] exists")

    def test_interface_R_exists(self):
        if self.R is None:
            return (False, "results['R'] is missing (None)")
        return (True, "results['R'] exists")

    def test_interface_theta_0_type(self):
        if self.theta_0 is None:
            return (False, "SKIP: theta_0 not available")
        if not isinstance(self.theta_0, (int, float)):
            return (False, f"theta_0 has type {type(self.theta_0)}, expected numeric")
        return (True, "theta_0 is numeric")

    def test_interface_R_type(self):
        if self.R is None:
            return (False, "SKIP: R not available")
        if not isinstance(self.R, (int, float)):
            return (False, f"R has type {type(self.R)}, expected numeric")
        return (True, "R is numeric")

    # ------------------------------------------------------------------
    # Invariant tests (theorem‑based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_critical_angle_relation(self):
        """sin(theta_0) must equal sqrt(1 - omega_p^2/omega^2)."""
        if self.theta_0 is None or self.omega is None or self.omega_p is None:
            return (False, "SKIP: missing givens or theta_0")
        n_ref = self._compute_n(self.omega, self.omega_p)
        if n_ref is None:
            return (False, "omega <= omega_p, cannot compute n")
        sin_theta0 = math.sin(self.theta_0)
        if not math.isclose(sin_theta0, n_ref, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"sin(theta_0)={sin_theta0:.5f} != n={n_ref:.5f}")
        return (True, f"sin(theta_0) matches n ({n_ref:.5f})")

    def test_invariant_refractive_index_consistency(self):
        """If n is given or computed, it should agree with Drude model."""
        if self.omega is None or self.omega_p is None:
            return (False, "SKIP: missing omega or omega_p")
        n_drude = self._compute_n(self.omega, self.omega_p)
        if n_drude is None:
            return (False, "omega <= omega_p, n not real")
        # If the solution also emitted n (optional), we could check; but we only infer from theta_0.
        # Instead, check that if theta_0 is available, sin(theta_0) ≈ n_drude
        if self.theta_0 is not None:
            sin_theta0 = math.sin(self.theta_0)
            if not math.isclose(sin_theta0, n_drude, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"sin(theta_0)={sin_theta0:.5f} != n={n_drude:.5f}")
        return (True, f"Drude n = {n_drude:.5f} is consistent with theta_0")

    def test_invariant_total_reflection(self):
        """For theta > theta_0, R should be 1 (total reflection)."""
        if self.theta is None or self.theta_0 is None or self.R is None:
            return (False, "SKIP: missing theta, theta_0, or R")
        if self.theta > self.theta_0:
            if not math.isclose(self.R, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                return (False, f"Total reflection expected but R={self.R:.5f} != 1")
            return (True, f"Total reflection (R≈1) for theta > theta_0")
        else:
            return (True, "theta <= theta_0, skipping total reflection check")

    # ------------------------------------------------------------------
    # Modeling / trace tests (check local relations from explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_snell_law_component(self):
        """Check that cos(theta_t) used in Fresnel coefficient is correctly derived."""
        if (self.theta is None or self.omega is None or self.omega_p is None
                or self.R is None):
            return (False, "SKIP: missing givens or R")
        # For partial reflection, theta must be < theta_0
        n = self._compute_n(self.omega, self.omega_p)
        if n is None:
            return (False, "omega <= omega_p, n not real")
        # Compute theta_t from results indirectly: we can attempt to back‑compute from R.
        # But better: use the R formula to verify that R matches the Fresnel expression.
        expected_R = self._compute_R(self.omega, self.omega_p, self.theta)
        if expected_R is None:
            return (False, "Could not compute expected R from givens")
        if not math.isclose(self.R, expected_R, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"R from result ({self.R:.5f}) != expected from Fresnel ({expected_R:.5f})")
        return (True, "R matches Fresnel formula for s‑polarization")

    def test_modeling_fresnel_positive_denominator(self):
        """Check denominator cos(theta) + n*cos(theta_t) > 0 (physical)."""
        if self.theta is None or self.n is None:
            return (False, "SKIP: missing theta or n")
        n = self.n
        sin_theta_t = math.sin(self.theta) / n
        if abs(sin_theta_t) >= 1.0:
            return (True, "Total reflection, denominator not applicable")
        cos_theta_t = math.sqrt(1.0 - sin_theta_t * sin_theta_t)
        denom = math.cos(self.theta) + n * cos_theta_t
        if denom <= 0:
            return (False, f"Denominator {denom:.5f} <= 0, unphysical")
        return (True, f"Denominator positive ({denom:.5f})")

    # ------------------------------------------------------------------
    # Monotonicity tests (trend checks via perturbed calculations)
    # ------------------------------------------------------------------
    def test_monotonicity_R_increases_with_theta(self):
        """For theta < theta_0, R should increase when theta increases (towards theta_0)."""
        if self.theta is None or self.omega is None or self.omega_p is None:
            return (False, "SKIP: missing givens for perturbation")
        n = self._compute_n(self.omega, self.omega_p)
        if n is None:
            return (False, "omega <= omega_p")
        # Compute baseline R at current theta
        R_base = self._compute_R(self.omega, self.omega_p, self.theta)
        # Perturb theta by +1% (ensure still < theta_0)
        theta_perturbed = self.theta * 1.01
        # Check that perturbed is still less than theta_0; if not, skip.
        # Compute theta_0 from n:
        theta_0_local = math.asin(n) if n <= 1 else None
        if theta_0_local is None or theta_perturbed >= theta_0_local:
            return (True, "Perturbed theta exceeds critical angle, skipping")
        R_pert = self._compute_R(self.omega, self.omega_p, theta_perturbed)
        if R_pert is None or R_base is None:
            return (False, "Could not compute reflectivities")
        if not (R_pert >= R_base - 1e-9):  # allow tiny numerical error
            return (False, f"R decreased: base {R_base:.5f}, perturbed {R_pert:.5f}")
        return (True, f"R increased as theta increased: {R_base:.5f} → {R_pert:.5f}")

    def test_monotonicity_theta_0_increases_with_omega(self):
        """Critical angle theta_0 increases when omega increases (n→1)."""
        if self.omega is None or self.omega_p is None:
            return (False, "SKIP: missing omega or omega_p")
        # Compute baseline theta_0
        theta_0_base = self._compute_theta_0(self.omega, self.omega_p)
        # Perturb omega by +1%
        omega_pert = self.omega * 1.01
        theta_0_pert = self._compute_theta_0(omega_pert, self.omega_p)
        if theta_0_base is None or theta_0_pert is None:
            return (False, "Could not compute theta_0 for baseline or perturbed")
        if not (theta_0_pert >= theta_0_base - 1e-9):
            return (False, f"theta_0 decreased: base {theta_0_base:.5f}, perturbed {theta_0_pert:.5f}")
        return (True, f"theta_0 increased with omega: {theta_0_base:.5f} → {theta_0_pert:.5f}")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one method allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_basic_checks(self):
        """
        Basic sanity: outputs must be finite and in physically plausible ranges.
        This is the only weak answer sanity test.
        """
        any_checked = False
        if self.theta_0 is not None:
            any_checked = True
            if not math.isfinite(self.theta_0):
                return (False, f"theta_0 is not finite: {self.theta_0}")
            if not (0.0 <= self.theta_0 <= math.pi/2 + 1e-9):
                return (False, f"theta_0 = {self.theta_0:.5f} outside [0, π/2]")
        if self.R is not None:
            any_checked = True
            if not math.isfinite(self.R):
                return (False, f"R is not finite: {self.R}")
            if not (0.0 <= self.R <= 1.0 + 1e-9):
                return (False, f"R = {self.R:.5f} outside [0,1]")
        if not any_checked:
            return (False, "SKIP: no outputs available to check")
        return (True, "All available outputs are finite and in valid ranges")

    # ------------------------------------------------------------------
    # Run all tests
    # ------------------------------------------------------------------
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
                passed, msg = False, f"CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)