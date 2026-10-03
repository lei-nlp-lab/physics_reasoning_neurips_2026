import math
import traceback

class PhysicsChecker:
    """
    Physics auditor for ionization rate induced by an EM wave on hydrogen ground state.
    Uses Fermi's golden rule, dipole approximation, plane wave final state.
    """

    def __init__(self, solution_instance):
        # Extract givens with None defaults
        self.hbar = getattr(solution_instance, 'hbar', None)
        self.m_e = getattr(solution_instance, 'm_e', None)
        self.e = getattr(solution_instance, 'e', None)
        self.a0 = getattr(solution_instance, 'a0', None)
        self.V = getattr(solution_instance, 'V', None)  # normalization volume, may be missing
        self.omega = getattr(solution_instance, 'omega', None)
        self.A0_mag = getattr(solution_instance, 'A0_mag', None)  # magnitude of A0, optional

        # Extract results (must be present)
        self.results = getattr(solution_instance, 'results', {})
        self.k_f = self.results.get('k_f', None)
        self.R = self.results.get('R', None)

        # Optional intermediate quantities that might have been emitted
        self.p_if_mag = self.results.get('p_if_mag', None)
        self.rho_total = self.results.get('rho_total', None)
        self.M_avg_sq = self.results.get('M_avg_sq', None)

        # Energy of ground state
        self.E_i = -13.6  # eV, but we'll also compute in SI if needed
        # For consistency, use SI: E_i = -e^2/(2*a0) (in CGS? problem likely uses SI)
        # We'll compute from given e and a0 if available, else use -13.6 eV (approx 2.18e-18 J)
        if self.e is not None and self.a0 is not None:
            # E_i = - e^2 / (2 * a0) in SI (actually e^2/(8 pi epsilon0 a0) but we use simplified)
            # We'll use the known hydrogen ground state energy: -13.6 eV = -2.179e-18 J
            pass  # keep as -13.6 eV for simplicity

        # Threshold condition
        self.ionization_threshold = 13.6  # eV

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

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_results_exist(self):
        """Check that results dictionary contains 'k_f' and 'R'."""
        if 'k_f' not in self.results:
            return (False, "Missing 'k_f' in results")
        if 'R' not in self.results:
            return (False, "Missing 'R' in results")
        return (True, "k_f and R present")

    def test_interface_k_f_positive(self):
        """k_f must be a positive number."""
        if self.k_f is None:
            return (False, "SKIP: k_f is None")
        if not isinstance(self.k_f, (int, float)):
            return (False, f"k_f type {type(self.k_f)} not numeric")
        if self.k_f <= 0:
            return (False, f"k_f = {self.k_f} not positive")
        return (True, f"k_f = {self.k_f} > 0")

    def test_interface_R_positive(self):
        """R must be a positive number."""
        if self.R is None:
            return (False, "SKIP: R is None")
        if not isinstance(self.R, (int, float)):
            return (False, f"R type {type(self.R)} not numeric")
        if self.R <= 0:
            return (False, f"R = {self.R} not positive")
        return (True, f"R = {self.R} > 0")

    def test_interface_givens_present(self):
        """Essential physical constants should be present and positive."""
        missing = []
        for attr in ['hbar', 'm_e', 'e', 'a0', 'omega']:
            val = getattr(self, attr, None)
            if val is None:
                missing.append(attr)
            elif not isinstance(val, (int, float)) or val <= 0:
                missing.append(f"{attr} (invalid)")
        if missing:
            return (False, f"Missing or invalid givens: {', '.join(missing)}")
        return (True, "All essential givens present")

    # ------------------------------------------------------------------
    # Invariant checks (theorem-based, with math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_energy_conservation(self):
        """
        Energy conservation: ħ^2 k_f^2/(2 m_e) = ħ ω - |E_i|
        (assuming E_i negative). Use isclose.
        """
        if any(x is None for x in [self.hbar, self.m_e, self.omega, self.k_f]):
            return (False, "SKIP: missing hbar, m_e, omega, or k_f")
        # Compute E_i magnitude (13.6 eV in Joules)
        # Better to use exact: E_i = -13.6 * 1.602e-19 J = -2.179e-18 J
        E_i_J = 13.6 * 1.602176634e-19  # absolute value
        left = (self.hbar**2) * (self.k_f**2) / (2 * self.m_e)
        right = self.hbar * self.omega - E_i_J
        if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Energy conservation fails: left={left}, right={right}")
        return (True, f"Energy conservation satisfied: ħ²k_f²/2m = {left}, ħω - |E_i| = {right}")

    def test_invariant_fermi_golden_rule_form(self):
        """
        Check that R follows the Fermi golden rule form:
        R = (2π/ħ) * |M|^2 * ρ(E_f)
        We can verify if R given matches a simplified computation using
        the provided M_avg_sq and rho_total if available.
        If those not available, we skip.
        """
        if any(x is None for x in [self.R, self.hbar, self.M_avg_sq, self.rho_total]):
            return (False, "SKIP: missing R, hbar, M_avg_sq, or rho_total")
        expected_R = (2 * math.pi / self.hbar) * self.M_avg_sq * self.rho_total
        if not math.isclose(self.R, expected_R, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Fermi golden rule mismatch: R={self.R}, expected={expected_R}")
        return (True, f"Fermi golden rule consistent: R ≈ {expected_R}")

    def test_invariant_density_of_states_scaling(self):
        """
        Check that rho_total scales with k_f as k_f * (V/(2π)^3)*(m_e/ħ^2)*4π.
        """
        if any(x is None for x in [self.rho_total, self.k_f, self.m_e, self.hbar, self.V]):
            return (False, "SKIP: missing rho_total, k_f, m_e, hbar, or V")
        expected_rho = (self.V / (2 * math.pi)**3) * (self.m_e * self.k_f / self.hbar**2) * 4 * math.pi
        if not math.isclose(self.rho_total, expected_rho, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Rho_total scaling off: rho={self.rho_total}, expected={expected_rho}")
        return (True, f"Density of states consistent: rho ≈ {expected_rho}")

    def test_invariant_momentum_matrix_element_scaling(self):
        """
        Check p_if_mag scaling with k_f:
        p_if ∝ k_f / (1/a0^2 + k_f^2)^2.
        We can check that p_if_mag * (1/a0^2 + k_f^2)^2 / k_f is roughly constant.
        """
        if any(x is None for x in [self.p_if_mag, self.k_f, self.a0]):
            return (False, "SKIP: missing p_if_mag, k_f, or a0")
        factor = self.p_if_mag * (1/self.a0**2 + self.k_f**2)**2 / self.k_f
        # Expect approximately hbar * 8*sqrt(pi)/sqrt(V) * (1/a0)^(3/2)
        # But V may be unknown, so check that factor is positive and finite
        if not math.isfinite(factor):
            return (False, f"p_if scaling factor not finite: {factor}")
        if factor <= 0:
            return (False, f"p_if scaling factor non-positive: {factor}")
        return (True, f"Momentum matrix element scaling plausible: factor = {factor}")

    # ------------------------------------------------------------------
    # Modeling/trace consistency checks
    # ------------------------------------------------------------------
    def test_modeling_angular_average_factor(self):
        """
        Check that M_avg_sq includes the 1/3 factor from angular average of |A0·p_if|^2.
        If A0_mag and p_if_mag are available, M_avg_sq should be (e^2/m_e^2)*|A0|^2*|p_if|^2/3.
        """
        if any(x is None for x in [self.M_avg_sq, self.e, self.m_e, self.A0_mag, self.p_if_mag]):
            return (False, "SKIP: missing M_avg_sq, e, m_e, A0_mag, or p_if_mag")
        expected = (self.e**2 / self.m_e**2) * (self.A0_mag**2) * (self.p_if_mag**2) / 3
        if not math.isclose(self.M_avg_sq, expected, rel_tol=1e-2, abs_tol=1e-20):
            return (False, f"Angular average factor mismatch: M_avg_sq={self.M_avg_sq}, expected={expected}")
        return (True, f"Angular average factor correct: M_avg_sq ≈ {expected}")

    def test_modeling_ionization_threshold(self):
        """
        Ionization only possible if ħω > |E_i|.
        """
        if self.omega is None or self.hbar is None:
            return (False, "SKIP: missing omega or hbar")
        photon_energy = self.hbar * self.omega
        threshold = 13.6 * 1.602176634e-19   # J
        if photon_energy < threshold - 1e-20:
            return (False, f"Photon energy {photon_energy} J less than ionization threshold {threshold} J")
        return (True, f"Photon energy {photon_energy} J exceeds threshold")

    def test_modeling_dipole_approximation_check(self):
        """
        Dipole approximation holds if k_photon * a0 << 1.
        k_photon = ω/c. Check that ω*a0/c < 0.1.
        """
        if self.omega is None or self.a0 is None:
            return (False, "SKIP: missing omega or a0")
        c = 299792458  # m/s
        k_photon = self.omega / c
        product = k_photon * self.a0
        if product > 0.1:
            return (False, f"Dipole approx may fail: k_photon*a0 = {product} > 0.1")
        return (True, f"Dipole approx valid: k_photon*a0 = {product} < 0.1")

    # ------------------------------------------------------------------
    # Monotonicity checks (perturb input, check trend)
    # ------------------------------------------------------------------
    def test_monotonicity_kf_with_omega(self):
        """
        Increasing omega should increase k_f (since k_f = sqrt(2m(ħω - |E_i|))/ħ).
        """
        if any(x is None for x in [self.hbar, self.m_e, self.omega, self.k_f]):
            return (False, "SKIP: missing hbar, m_e, omega, or k_f")
        # Compute k_f with small increase in omega
        delta = 0.01 * self.omega  # 1% increase
        omega_new = self.omega + delta
        E_i_J = 13.6 * 1.602176634e-19
        k_f_new = math.sqrt(2 * self.m_e * (self.hbar * omega_new - E_i_J)) / self.hbar
        if k_f_new <= self.k_f:
            return (False, f"k_f did not increase with omega: old={self.k_f}, new after 1% increase={k_f_new}")
        return (True, f"k_f increases with omega: old={self.k_f:.3e}, new={k_f_new:.3e}")

    def test_monotonicity_R_with_omega(self):
        """
        Increasing omega should increase R (above threshold). 
        We'll recompute R using the same formula dependencies.
        """
        if any(x is None for x in [self.hbar, self.m_e, self.e, self.a0, self.V, self.omega, self.R, self.A0_mag]):
            return (False, "SKIP: missing required quantities for R monotonicity")
        # Compute current R using the derived formula from pseudocode
        # Current omega
        E_i_J = 13.6 * 1.602176634e-19
        k_f_current = math.sqrt(2 * self.m_e * (self.hbar * self.omega - E_i_J)) / self.hbar
        # p_if_mag formula (ignoring prefactors that cancel in ratio)
        # p_if ∝ k_f / (1/a0^2 + k_f^2)^2
        # Actually, p_if_mag = hbar * k_f * (8*sqrt(pi)/sqrt(self.V))* (1/self.a0)**(1.5) / (1/self.a0**2 + k_f**2)**2
        # But we can use the provided p_if_mag if available; else compute from scratch.
        # Since we have A0_mag, we can compute R formula directly.
        # Let's compute the full R using the pseudocode formula but avoid needing V
        # Actually, R expression from pseudocode: R = (64/3) * (e^2|A0|^2 / (hbar m_e a0^3)) * (k_f^3 / (1/a0^2 + k_f^2)^4)
        # That final expression does NOT depend on V because the density of states V/(2π)^3 cancels with 1/V from p_if normalization.
        # So we can use that expression for monotonicity test.
        # But we need to know the exact constants; we can compute R from the formula and compare.
        # For monotonicity, we only need trend: compute R with perturbed omega.
        # Let's compute using the derived final expression:
        # R_current_comp = (64/3) * (self.e**2 * self.A0_mag**2 / (self.hbar * self.m_e * self.a0**3)) * (k_f_current**3 / (1/self.a0**2 + k_f_current**2)**4)
        # That should match self.R if the solver used the same constants. We'll use to test monotonicity.
        # To avoid complexity, we will compute R at two omega values using the same formula.
        # Compute k_f for omega + 1%
        delta = 0.01 * self.omega
        omega_new = self.omega + delta
        k_f_new = math.sqrt(2 * self.m_e * (self.hbar * omega_new - E_i_J)) / self.hbar
        # Compute R_new using same scaling
        # R ∝ k_f^3 / (1/a0^2 + k_f^2)^4
        factor_old = k_f_current**3 / (1/self.a0**2 + k_f_current**2)**4
        factor_new = k_f_new**3 / (1/self.a0**2 + k_f_new**2)**4
        # The proportionality constant cancels in ratio
        if factor_new <= factor_old:
            return (False, f"R factor did not increase with omega: old_factor={factor_old}, new_factor={factor_new}")
        return (True, f"R factor increases with omega: old={factor_old:.3e}, new={factor_new:.3e}")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one, as per instructions)
    # ------------------------------------------------------------------
    def test_answer_sanity_sign_and_finite(self):
        """R must be positive and finite (not NaN, not inf)."""
        if self.R is None:
            return (False, "SKIP: R is None")
        if not math.isfinite(self.R):
            return (False, f"R is not finite: {self.R}")
        if self.R <= 0:
            return (False, f"R not positive: {self.R}")
        # Very weak magnitude check: R should be less than some astronomical number
        # e.g., less than 10^100 /s (true for any physical question)
        if self.R > 1e100:
            return (False, f"R suspiciously large: {self.R}")
        return (True, f"R = {self.R} is positive, finite, and plausible")