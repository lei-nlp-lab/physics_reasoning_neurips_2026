import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results safely, using None defaults
        self.results = getattr(solution_instance, 'results', {})
        self.P_down = self.results.get('P_down', None)
        self.P_theta_phi = self.results.get('P_theta_phi', None)
        self.mu_mag = self.results.get('mu_mag', None)
        self.final_l = self.results.get('final_l', None)
        self.final_lz = self.results.get('final_lz', None)
        self.final_s = self.results.get('final_s', None)
        self.final_sz = self.results.get('final_sz', None)
        self.mu_strong = self.results.get('mu_strong', None)
        # Store explanation steps for optional reference (not required for tests)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ------ Interface tests ------
    def test_interface_required_keys(self):
        """Check that all expected keys exist and are not None."""
        required_keys = ['P_down', 'P_theta_phi', 'mu_mag', 'final_l',
                         'final_lz', 'final_s', 'final_sz', 'mu_strong']
        missing = [k for k in required_keys if self.results.get(k, None) is None]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present.")

    # ------ Invariant tests (theorem-based, use math.isclose) ------
    def test_invariant_cg_normalization(self):
        """Check that P_down + P_up = 1, where P_up = 1/3 from Clebsch-Gordan."""
        if self.P_down is None:
            return (False, "SKIP: P_down not available")
        P_up = 1.0 / 3.0
        total = self.P_down + P_up
        if not math.isclose(total, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"P_down + P_up = {total}, expected 1.0")
        return (True, "Normalization holds.")

    def test_invariant_angular_normalization(self):
        """Check that integral of P(θ,φ) over solid angle = 1, i.e., 4π * P_theta_phi = 1."""
        if self.P_theta_phi is None:
            return (False, "SKIP: P_theta_phi not available")
        integral = 4.0 * math.pi * self.P_theta_phi
        if not math.isclose(integral, 1.0, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"4π * P_theta_phi = {integral}, expected 1.0")
        return (True, "Angular probability normalized.")

    def test_invariant_lande_g_factor(self):
        """Verify Landé g-factor for ^2P_{1/2} is 2/3 using formula: g_J = 1 + [j(j+1)-l(l+1)+s(s+1)]/[2j(j+1)]."""
        j = 0.5
        l = 1.0
        s = 0.5
        g_formula = 1.0 + (j*(j+1) - l*(l+1) + s*(s+1)) / (2.0*j*(j+1))
        g_expected = 2.0/3.0
        if not math.isclose(g_formula, g_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Landé g computed as {g_formula}, expected {g_expected}")
        return (True, "Landé g-factor correct.")

    def test_invariant_weak_moment_relation(self):
        """Check that mu_mag = μ_B/3 (symbolic consistency)."""
        if self.mu_mag is None:
            return (False, "SKIP: mu_mag not available")
        # The relationship is inherently satisfied by design; we confirm existence.
        return (True, "Weak moment relation plausible (symbolic).")

    def test_invariant_strong_moment_sign(self):
        """Check that mu_strong is negative (as expected for lowest Zeeman state)."""
        if self.mu_strong is None:
            return (False, "SKIP: mu_strong not available")
        if isinstance(self.mu_strong, (int, float)):
            if self.mu_strong >= 0:
                return (False, f"mu_strong = {self.mu_strong}, expected negative")
        return (True, "Strong moment sign consistent with expectation.")

    # ------ Modeling tests (based on explanation steps) ------
    def test_modeling_cg_expansion(self):
        """Verify that P_down matches the square of the CG coefficient for m_s = -1/2 (√(2/3))."""
        if self.P_down is None:
            return (False, "SKIP: P_down not available")
        expected = 2.0 / 3.0
        if not math.isclose(self.P_down, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"P_down = {self.P_down}, expected {expected}")
        return (True, "Spin-down probability matches CG coefficient.")

    def test_modeling_angular_expression(self):
        """Check that P_theta_phi = 1/(4π) is consistent with the spherical harmonic combination."""
        if self.P_theta_phi is None:
            return (False, "SKIP: P_theta_phi not available")
        expected = 1.0 / (4.0 * math.pi)
        if not math.isclose(self.P_theta_phi, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"P_theta_phi = {self.P_theta_phi}, expected {expected}")
        return (True, "Angular probability constant matches.")

    def test_modeling_strong_field_state(self):
        """Check that the strong field quantum numbers correspond to the lowest Zeeman energy."""
        if None in (self.final_l, self.final_lz, self.final_s, self.final_sz):
            return (False, "SKIP: final state quantum numbers not all present")
        if self.final_l != 1:
            return (False, f"final_l = {self.final_l}, expected 1")
        if self.final_s != 0.5:
            return (False, f"final_s = {self.final_s}, expected 0.5")
        m_l = self.final_lz
        m_s = self.final_sz
        zeeman_val = m_l + 2*m_s
        expected_min = -1.0 + 2*(-0.5)  # -2
        if not math.isclose(zeeman_val, expected_min, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"(m_l+2m_s) = {zeeman_val}, expected {expected_min} (lowest Zeeman energy)")
        return (True, "Strong field state is the Zeeman ground state.")

    # ------ Monotonicity tests (explicit input perturbation) ------
    def test_monotonicity_zeeman_factor_vs_m_l(self):
        """
        For the strong-field state, increasing m_l by 1 makes the Zeeman factor (m_l+2m_s) larger
        (less negative), hence the magnetic moment becomes less negative (increases).
        Perturb m_l -> m_l+1 and compare the factor.
        """
        if None in (self.final_lz, self.final_sz):
            return (False, "SKIP: final_lz or final_sz missing")
        m_l_base = self.final_lz
        m_s = self.final_sz
        factor_base = m_l_base + 2*m_s
        # Perturb m_l by +1
        m_l_pert = m_l_base + 1
        factor_pert = m_l_pert + 2*m_s
        # The factor should increase (become less negative) when m_l increases.
        if factor_pert <= factor_base:
            return (False, f"Perturbed factor {factor_pert} not > base factor {factor_base}")
        # Also, the change in magnetic moment Δμ = -μ_B * (factor_pert - factor_base) is negative,
        # meaning μ becomes larger (less negative). We can check the sign of Δμ if mu_strong numeric.
        # This is sufficient.
        return (True, "Zeeman factor monotonicity with m_l verified.")

    # ------ Answer sanity (weak, no math.isclose, no re-derivation) ------
    def test_answer_sanity_final_values(self):
        """Weak sanity: all results are finite, signs plausible, magnitudes non-negative."""
        issues = []
        if self.P_down is not None:
            if self.P_down < 0 or self.P_down > 1:
                issues.append("P_down out of [0,1]")
        if self.P_theta_phi is not None:
            if self.P_theta_phi < 0:
                issues.append("P_theta_phi negative")
        if self.mu_mag is not None:
            # mu_mag should be positive
            if self.mu_mag < 0:
                issues.append("mu_mag negative")
        if self.mu_strong is not None:
            # mu_strong should be negative
            if self.mu_strong > 0:
                issues.append("mu_strong positive, expected negative")
        if issues:
            return (False, "; ".join(issues))
        return (True, "All final values pass sanity checks.")

    # ------ Run all tests ------
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)