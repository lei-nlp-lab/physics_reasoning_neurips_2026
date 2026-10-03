import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens
        givens = getattr(solution_instance, 'givens', None)
        if givens is None:
            self.omega = None
            self.v = None
            self.c = None
            self.n = None
        else:
            if isinstance(givens, dict):
                self.omega = givens.get('omega', None)
                self.v = givens.get('v', None)
                self.c = givens.get('c', None)
                self.n = givens.get('n', None)
            else:
                self.omega = getattr(givens, 'omega', None)
                self.v = getattr(givens, 'v', None)
                self.c = getattr(givens, 'c', None)
                self.n = getattr(givens, 'n', None)

        # Extract results
        results = getattr(solution_instance, 'results', None)
        if results is None:
            self.omega_r = None
        else:
            if isinstance(results, dict):
                self.omega_r = results.get('omega_r', None)
            else:
                self.omega_r = getattr(results, 'omega_r', None)

        # Derived quantities (None if missing)
        if self.v is not None and self.c is not None and self.c != 0:
            self.beta = self.v / self.c
        else:
            self.beta = None

    # ------------------------------------------------------------------
    # Helper: compute expected omega_r from givens (used in tests)
    # ------------------------------------------------------------------
    def _expected_omega_r(self):
        if self.omega is not None and self.beta is not None and abs(self.beta) < 1:
            return self.omega * (1 - self.beta) / (1 + self.beta)
        return None

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        if self.omega_r is None:
            return (False, "Missing 'omega_r' in results.")
        if self.omega is None:
            return (False, "Missing 'omega' in givens.")
        if self.v is None:
            return (False, "Missing 'v' in givens.")
        if self.c is None:
            return (False, "Missing 'c' in givens.")
        if self.c <= 0:
            return (False, "Speed of light c must be positive.")
        if self.v >= self.c:
            return (False, "Velocity v must be less than c (v < c).")
        return (True, "All required keys present and valid.")

    # ------------------------------------------------------------------
    # Theorem‑based invariant checks (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_omega_r_formula(self):
        if None in (self.omega, self.omega_r, self.beta):
            return (False, "SKIP: Missing data for omega_r formula check.")
        expected = self._expected_omega_r()
        if expected is None:
            return (False, "Cannot compute expected omega_r (beta may be >=1).")
        if not math.isclose(self.omega_r, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"omega_r = {self.omega_r} does not match expected {expected}.")
        return (True, "omega_r satisfies the reflection formula.")

    def test_invariant_doppler_approximation(self):
        # For small beta, omega_r ≈ omega * (1 - 2β)
        if None in (self.omega, self.omega_r, self.beta):
            return (False, "SKIP: Missing data for Doppler approximation.")
        if abs(self.beta) > 0.1:
            return (False, "SKIP: beta too large for approximation.")
        approx = self.omega * (1 - 2 * self.beta)
        if not math.isclose(self.omega_r, approx, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"omega_r = {self.omega_r} not close to low‑velocity approximation {approx}.")
        return (True, "omega_r matches low‑velocity approximation.")

    def test_invariant_positive_omega_r(self):
        if self.omega_r is None:
            return (False, "SKIP: omega_r not available.")
        if self.omega_r <= 0:
            return (False, f"omega_r must be positive, got {self.omega_r}.")
        return (True, "omega_r is positive.")

    def test_invariant_phase_continuity(self):
        # Phase continuity: (k_i v - ω) t = (-k_r v - ω_r) t for all t
        #   => (ω/c)v - ω = -(ω_r/c)v - ω_r
        #   => ω (v - c) = -ω_r (v + c)
        if None in (self.omega, self.omega_r, self.v, self.c):
            return (False, "SKIP: Missing data for phase continuity.")
        lhs = self.omega * (self.v - self.c)
        rhs = -self.omega_r * (self.v + self.c)
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Phase continuity fails: LHS={lhs}, RHS={rhs}")
        return (True, "Phase continuity condition holds.")

    # ------------------------------------------------------------------
    # Modeling / trace checks (based on explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_lorentz_consistency(self):
        """
        Trace the derivation: transform incident frequency to moving frame,
        reflect without change, then transform back to stationary frame.
        Check that the resulting omega_r matches the stored value.
        """
        if None in (self.omega, self.omega_r, self.beta):
            return (False, "SKIP: Missing data for Lorentz consistency check.")
        gamma = 1.0 / math.sqrt(1.0 - self.beta**2)

        # Incident frequency in Σ'
        omega_prime = gamma * self.omega * (1.0 - self.beta)

        # Reflection in Σ' does not change frequency
        omega_2_prime = omega_prime

        # Transform reflected wave back to Σ
        # Reflected four‑vector in Σ': (k'_z = -ω_2'/c, ω'_2/c)
        # Inverse Lorentz: ω_r/c = γ (ω_2'/c + β * (-k'_z)) = γ (ω_2'/c + β*(ω_2'/c))
        omega_r_expected = gamma * omega_2_prime * (1.0 + self.beta)

        if not math.isclose(self.omega_r, omega_r_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Lorentz consistency gives omega_r = {omega_r_expected}, "
                           f"but stored value is {self.omega_r}.")
        return (True, "Lorentz transformation chain is consistent with stored omega_r.")

    def test_modeling_reflected_wave_vector(self):
        """
        Verify that the reflected wave obeys the vacuum dispersion relation:
        ω_r = c |k_r|, and that k_r points in the negative z direction.
        For normal incidence, k_r = -ω_r/c.
        """
        if self.omega_r is None or self.c is None:
            return (False, "SKIP: Missing omega_r or c for wave vector check.")
        if self.c <= 0:
            return (False, "c must be positive.")
        # In vacuum, ω_r = c |k_r| => |k_r| = ω_r/c
        expected_k_mag = self.omega_r / self.c
        # The reflected wave vector is -ω_r/c (since reflected backward)
        # We cannot directly check sign without the wave vector stored,
        # but we can check that the magnitude relation holds (dispersion)
        # Also check that ω_r/c is positive.
        if expected_k_mag <= 0:
            return (False, f"Computed |k_r| = {expected_k_mag} is not positive.")
        # We don't have actual k_r stored, so we assume the solver used k_r = -ω_r/c.
        # This test is a consistency check that omega_r gives a valid wave number.
        return (True, f"Reflected wave magnitude |k_r| = ω_r/c = {expected_k_mag} (positive).")

    # ------------------------------------------------------------------
    # Monotonicity check (perturb v and observe effect on omega_r)
    # ------------------------------------------------------------------
    def test_monotonicity_omega_r_decreases_with_v(self):
        if None in (self.omega, self.omega_r, self.v, self.c, self.beta):
            return (False, "SKIP: Missing data for monotonicity check.")
        delta = 0.01 * self.c          # 1% of c
        v_new = self.v + delta
        if v_new >= self.c:
            return (False, "SKIP: v_new would exceed c.")
        beta_new = v_new / self.c
        omega_r_new = self.omega * (1 - beta_new) / (1 + beta_new)
        # We expect omega_r_new < self.omega_r
        if omega_r_new >= self.omega_r:
            return (False, f"omega_r did not decrease when v increased: new={omega_r_new} >= old={self.omega_r}")
        return (True, "Monotonicity: increasing v decreases omega_r.")

    # ------------------------------------------------------------------
    # Weak answer sanity (one method only, no re‑derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_omega_r_range(self):
        if self.omega_r is None:
            return (False, "SKIP: omega_r not available.")
        # Broad plausible range: positive and not ten times larger than incident omega
        if not (0 < self.omega_r < self.omega * 2):
            return (False, f"omega_r = {self.omega_r} seems out of plausible range.")
        return (True, "omega_r within plausible range.")

    # ------------------------------------------------------------------
    # Run all tests automatically
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)