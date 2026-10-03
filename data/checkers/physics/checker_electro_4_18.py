import math
import traceback

class PhysicsChecker:
    """Auditor for electromagnetic wave polarization and plasma problem."""

    def __init__(self, solution_instance):
        # Extract result keys safely (primarily from results dict)
        self.results = {}
        if hasattr(solution_instance, 'results'):
            self.results = solution_instance.results
        # Get givens if available
        self.givens = {}
        if hasattr(solution_instance, 'givens'):
            self.givens = solution_instance.givens

        # Result keys expected by contract
        self.E_R = self.results.get('E_R', None)
        self.E_L = self.results.get('E_L', None)
        self.phi = self.results.get('phi', None)
        self.n = self.results.get('n', None)
        self.beta = self.results.get('beta', None)
        self.high_freq_cond = self.results.get('high_freq_cond', None)

        # Additional variables from givens or results (optional)
        self.omega = self.givens.get('omega', None)
        self.c = self.givens.get('c', None)
        self.omega_p = self.givens.get('omega_p', None)
        self.omega_c = self.givens.get('omega_c', None)
        self.n0 = self.givens.get('n0', None)
        self.e = self.givens.get('e', None)
        self.m = self.givens.get('m', None)
        self.epsilon0 = self.givens.get('epsilon0', None)
        self.B0 = self.givens.get('B0', None)

        # Also allow n_plus/n_minus if stored in results (not required)
        self.n_plus = self.results.get('n_plus', None)
        self.n_minus = self.results.get('n_minus', None)

        # Explanation steps (for guidance only)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', [])

    def run_all_tests(self):
        """Discover and run all test_* methods, return (bool, list of logs)."""
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
                passed = False
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)

    # ----------------------------------------------------------------------
    # INTERFACE TESTS
    # ----------------------------------------------------------------------
    def test_interface_required_E_R(self):
        if self.E_R is None:
            return (False, "SKIP: E_R is missing")
        if not isinstance(self.E_R, str) and not hasattr(self.E_R, 'free_symbols'):
            return (False, "E_R is not a string or symbolic expression")
        return (True, "E_R present and has plausible type")

    def test_interface_required_E_L(self):
        if self.E_L is None:
            return (False, "SKIP: E_L is missing")
        if not isinstance(self.E_L, str) and not hasattr(self.E_L, 'free_symbols'):
            return (False, "E_L is not a string or symbolic expression")
        return (True, "E_L present")

    def test_interface_required_phi(self):
        if self.phi is None:
            return (False, "SKIP: phi is missing")
        ok = isinstance(self.phi, (str, int, float)) or hasattr(self.phi, 'free_symbols')
        if not ok:
            return (False, "phi is neither string, number, nor symbolic expression")
        return (True, "phi present")

    def test_interface_required_n(self):
        if self.n is None:
            return (False, "SKIP: n is missing")
        ok = isinstance(self.n, (str, int, float)) or hasattr(self.n, 'free_symbols')
        if not ok:
            return (False, "n has unexpected type")
        return (True, "n present")

    def test_interface_required_beta(self):
        if self.beta is None:
            return (False, "SKIP: beta is missing")
        ok = isinstance(self.beta, (str, int, float)) or hasattr(self.beta, 'free_symbols')
        if not ok:
            return (False, "beta has unexpected type")
        return (True, "beta present")

    def test_interface_required_high_freq_cond(self):
        if self.high_freq_cond is None:
            return (False, "SKIP: high_freq_cond is missing")
        if not isinstance(self.high_freq_cond, str):
            return (False, "high_freq_cond is not a string")
        return (True, "high_freq_cond present as string")

    # ----------------------------------------------------------------------
    # INVARIANT TESTS (theorem-based, allow fuzzy match)
    # ----------------------------------------------------------------------
    def test_invariant_rotation_angle_from_wavenumbers(self):
        """
        If numeric phi, beta, omega, c are available, check that phi = beta*omega*z/c
        for some z (use default z=1 if not given).
        """
        if self.phi is None or self.beta is None or self.omega is None or self.c is None:
            return (False, "SKIP: missing numeric givens (omega, c, beta, phi)")
        if not all(isinstance(x, (int, float)) for x in [self.omega, self.c, self.beta, self.phi]):
            return (False, "SKIP: one of values is not numeric")
        z = 1  # default; we could also try to extract from results, but not required
        expected_phi = self.beta * self.omega * z / self.c
        if math.isclose(self.phi, expected_phi, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "phi matches beta*ω*z/c within tolerance")
        else:
            return (False, f"phi = {self.phi}, expected approx {expected_phi}")

    def test_invariant_n_and_beta_from_nplus_nminus(self):
        """
        If n_plus and n_minus are provided, check that n = (n_+ + n_-)/2 and beta = (n_+ - n_-)/2.
        """
        if self.n_plus is None or self.n_minus is None:
            if self.n is not None and self.beta is not None:
                self.n_plus = self.n + self.beta
                self.n_minus = self.n - self.beta
            else:
                return (False, "SKIP: neither n_plus/n_minus nor n/beta available")
        n_plus_num = self._to_float(self.n_plus)
        n_minus_num = self._to_float(self.n_minus)
        n_num = self._to_float(self.n)
        beta_num = self._to_float(self.beta)
        if any(x is None for x in [n_plus_num, n_minus_num, n_num, beta_num]):
            return (False, "SKIP: non-numeric values for n_plus/n_minus or n/beta")
        if math.isclose((n_plus_num + n_minus_num)/2, n_num, rel_tol=1e-2, abs_tol=1e-2) and \
           math.isclose((n_plus_num - n_minus_num)/2, beta_num, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "n and beta are consistent with n_plus, n_minus")
        return (False, "n or beta inconsistent with n_plus/n_minus")

    def test_invariant_plasma_dispersion_right_hand(self):
        """
        For right circular polarization, check n_+^2 ≈ 1 - ω_p^2 / (ω(ω - ω_c)).
        """
        if self.n_plus is None or self.omega is None or self.omega_p is None or self.omega_c is None:
            return (False, "SKIP: missing n_plus or plasma parameters")
        n_plus_num = self._to_float(self.n_plus)
        if n_plus_num is None:
            return (False, "SKIP: non-numeric n_plus")
        if self.omega <= self.omega_c:
            return (False, "SKIP: ω not > ω_c")
        expected_n_sq = 1 - self.omega_p**2 / (self.omega * (self.omega - self.omega_c))
        actual_n_sq = n_plus_num ** 2
        if math.isclose(actual_n_sq, expected_n_sq, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "n_+^2 matches cold plasma dispersion")
        return (False, f"n_+^2 = {actual_n_sq}, expected {expected_n_sq}")

    def test_invariant_plasma_dispersion_left_hand(self):
        """
        For left circular polarization, check n_-^2 ≈ 1 - ω_p^2 / (ω(ω + ω_c)).
        """
        if self.n_minus is None or self.omega is None or self.omega_p is None or self.omega_c is None:
            return (False, "SKIP: missing n_minus or plasma parameters")
        n_minus_num = self._to_float(self.n_minus)
        if n_minus_num is None:
            return (False, "SKIP: non-numeric n_minus")
        expected_n_sq = 1 - self.omega_p**2 / (self.omega * (self.omega + self.omega_c))
        actual_n_sq = n_minus_num ** 2
        if math.isclose(actual_n_sq, expected_n_sq, rel_tol=1e-2, abs_tol=1e-2):
            return (True, "n_-^2 matches cold plasma dispersion")
        return (False, f"n_-^2 = {actual_n_sq}, expected {expected_n_sq}")

    # ----------------------------------------------------------------------
    # MODELING TESTS (based on explanation steps)
    # ----------------------------------------------------------------------
    def test_modeling_high_freq_condition_contains_plasma_cyclotron(self):
        """
        The high_freq_cond must reference both plasma frequency and cyclotron frequency.
        Should state that ω >> max(ω_p, ω_c).
        """
        if self.high_freq_cond is None:
            return (False, "SKIP: high_freq_cond missing")
        cond_str = str(self.high_freq_cond).lower()
        has_omega_p = any(s in cond_str for s in ['omega_p', 'ω_p', 'wp'])
        has_omega_c = any(s in cond_str for s in ['omega_c', 'ω_c', 'wc', 'omegab'])
        has_cyclotron = 'cyclotron' in cond_str
        if has_omega_p and (has_omega_c or has_cyclotron):
            return (True, "high_freq_cond includes both plasma and cyclotron frequencies")
        return (False, "high_freq_cond does not clearly mention both frequencies")

    def test_modeling_phi_contains_beta_omega_z_c(self):
        """
        The rotation angle expression should be of the form phi = beta*omega*z/c or equivalent.
        """
        if self.phi is None:
            return (False, "SKIP: phi missing")
        phi_str = str(self.phi).lower()
        required = ['beta', 'omega', 'z', 'c']
        if all(r in phi_str for r in required):
            return (True, "phi string contains beta, omega, z, c")
        return (False, f"phi string missing one of {required}")

    # ----------------------------------------------------------------------
    # MONOTONICITY TESTS (explicit perturbation of input and comparison)
    # ----------------------------------------------------------------------
    def test_monotonicity_phi_with_beta(self):
        """
        Verify that phi increases when beta increases, using the relationship
        phi = beta * omega * z / c.  This checks monotonicity of the physical model,
        not the specific phi value given by the solution.
        """
        # Need numeric omega, c, beta to compute baseline and perturbed outputs.
        if not all(isinstance(x, (int, float)) for x in [self.beta, self.omega, self.c]):
            return (False, "SKIP: missing numeric values for beta, omega, c")
        # Use a fixed z=1 (unit distance)
        z = 1
        baseline_phi = self.beta * self.omega * z / self.c
        # Perturb beta upward by 10%
        beta_new = self.beta * 1.1
        perturbed_phi = beta_new * self.omega * z / self.c
        if perturbed_phi > baseline_phi:
            return (True, "phi increases with beta (monotonic)")
        else:
            return (False, "phi does not monotonically increase with beta")

    def test_monotonicity_beta_with_omega(self):
        """
        Verify that beta decreases when omega increases, using the plasma model
        beta = omega_p^2 * omega_c / (2 * omega^3).
        """
        if not all(isinstance(x, (int, float)) for x in [self.beta, self.omega, self.omega_p, self.omega_c]):
            return (False, "SKIP: missing numeric values for beta, omega, omega_p, omega_c")
        # Compute baseline beta using the formula (not relying on solution's beta)
        baseline_beta = self.omega_p**2 * self.omega_c / (2 * self.omega**3)
        # Perturb omega upward by 10%
        omega_new = self.omega * 1.1
        perturbed_beta = self.omega_p**2 * self.omega_c / (2 * omega_new**3)
        if perturbed_beta < baseline_beta:
            return (True, "beta decreases with increasing omega (monotonic)")
        else:
            return (False, "beta does not monotonically decrease with omega")

    # ----------------------------------------------------------------------
    # ANSWER SANITY (weak, only one allowed)
    # ----------------------------------------------------------------------
    def test_answer_sanity_phi_positive_or_zero(self):
        """
        Rotation angle should be non-negative if beta>0 (Faraday rotation sign).
        Simple sign check.
        """
        if self.phi is None:
            return (False, "SKIP: phi missing")
        phi_val = self._to_float(self.phi)
        if phi_val is None:
            return (False, "SKIP: phi not numeric")
        if phi_val >= -1e-9:  # allow small negative due to numerical noise
            return (True, "phi is non-negative")
        return (False, f"phi = {phi_val} is negative")

    # ----------------------------------------------------------------------
    # Helper methods
    # ----------------------------------------------------------------------
    def _to_float(self, x):
        """Try to convert x to float; if fails, return None."""
        if isinstance(x, (int, float)):
            return float(x)
        if hasattr(x, 'evalf'):
            try:
                return float(x.evalf())
            except:
                return None
        # string: risky, skip
        return None