import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        # givens (solution_instance should have these attributes)
        self.hbar = getattr(solution_instance, 'hbar', None)
        self.c = getattr(solution_instance, 'c', None)  # cm/s
        self.m_rho = getattr(solution_instance, 'm_rho', None)
        self.Gamma_rho = getattr(solution_instance, 'Gamma_rho', None)
        self.m_p = getattr(solution_instance, 'm_p', None)
        self.m_n = getattr(solution_instance, 'm_n', None)
        self.m_pi = getattr(solution_instance, 'm_pi', None)
        self.E_rho_lab = getattr(solution_instance, 'E_rho_lab', None)
        self.sigma = getattr(solution_instance, 'sigma', None)  # cm^2
        self.L_target = getattr(solution_instance, 'L_target', None)
        self.rho_H2 = getattr(solution_instance, 'rho_H2', None)
        self.M_H = getattr(solution_instance, 'M_H', None)
        self.N_A = getattr(solution_instance, 'N_A', None)

        # results (solution_instance.results dict)
        results = getattr(solution_instance, 'results', {})
        self.tau = results.get('tau', None)
        self.d = results.get('d', None)
        self.E_pi_th = results.get('E_pi_th', None)
        self.N = results.get('N', None)
        self.theta_min = results.get('theta_min', None)

    # -----------------------------------------------------------------
    # Interface tests
    # -----------------------------------------------------------------
    def test_interface_results_keys(self):
        """Check all required result keys exist."""
        missing = []
        for k in ['tau', 'd', 'E_pi_th', 'N', 'theta_min']:
            if getattr(self, k) is None:
                missing.append(k)
        if missing:
            return (False, f"SKIP: missing keys: {missing}")
        return (True, "All required result keys present")

    def test_interface_tau_positive(self):
        if self.tau is None:
            return (False, "SKIP: tau not available")
        if self.tau <= 0:
            return (False, f"tau = {self.tau} is not positive")
        return (True, "tau > 0")

    # -----------------------------------------------------------------
    # Invariant tests (use math.isclose with rel_tol=1e-2, abs_tol=1e-2)
    # -----------------------------------------------------------------
    def test_invariant_lifetime_formula(self):
        """τ = ħ / Γ"""
        if None in (self.hbar, self.Gamma_rho, self.tau):
            return (False, "SKIP: missing data for lifetime formula")
        expected = self.hbar / self.Gamma_rho
        if math.isclose(self.tau, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"tau={self.tau:e} matches hbar/Gamma={expected:e}")
        return (False, f"tau={self.tau:e} != hbar/Gamma={expected:e}")

    def test_invariant_boost_distance(self):
        """d = βγ c τ (using computed gamma)"""
        if None in (self.tau, self.E_rho_lab, self.m_rho, self.c, self.d):
            return (False, "SKIP: missing data for decay distance")
        gamma = self.E_rho_lab / self.m_rho
        if gamma < 1:
            return (False, f"gamma={gamma} < 1 unphysical")
        beta = math.sqrt(1 - 1/(gamma*gamma))
        expected = beta * gamma * self.c * self.tau
        if math.isclose(self.d, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"d={self.d:e} matches beta gamma c tau = {expected:e}")
        return (False, f"d={self.d:e} != expected {expected:e}")

    def test_invariant_threshold_energy(self):
        """Threshold energy from invariant mass condition."""
        if None in (self.m_p, self.m_rho, self.m_pi, self.E_pi_th):
            return (False, "SKIP: missing data for threshold energy")
        # E_pi_th = ((m_p + m_rho)^2 - m_pi^2 - m_p^2) / (2*m_p)
        # Use m_n = m_rho? Actually threshold uses ρ mass, but final state has ρ⁰ + n.
        # For threshold we set (m_target + m_projectile) = m_ρ + m_n? 
        # But the explanation used (m_p + m_ρ)^2. Let's follow that.
        expected = ((self.m_p + self.m_rho)**2 - self.m_pi**2 - self.m_p**2) / (2 * self.m_p)
        if math.isclose(self.E_pi_th, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_pi_th={self.E_pi_th:.1f} matches expected {expected:.1f}")
        return (False, f"E_pi_th={self.E_pi_th:.1f} != expected {expected:.1f}")

    def test_invariant_number_density(self):
        """N = σ * n * L where n = ρ N_A / M_H"""
        if None in (self.sigma, self.rho_H2, self.M_H, self.N_A, self.L_target, self.N):
            return (False, "SKIP: missing data for number check")
        n = self.rho_H2 * self.N_A / self.M_H
        expected = self.sigma * n * self.L_target
        if math.isclose(self.N, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"N={self.N:e} matches expected {expected:e}")
        return (False, f"N={self.N:e} != expected {expected:e}")

    def test_invariant_theta_min_zero(self):
        """For high boost both pions go forward => theta_min = 0."""
        if self.theta_min is None:
            return (False, "SKIP: theta_min not available")
        if self.theta_min != 0:
            return (False, f"theta_min={self.theta_min} should be 0")
        return (True, "theta_min == 0")

    # -----------------------------------------------------------------
    # Modeling / trace tests (single local relation each)
    # -----------------------------------------------------------------
    def test_modeling_gamma_beta(self):
        """Check gamma = E_rho / m_rho, beta derived correctly."""
        if None in (self.E_rho_lab, self.m_rho):
            return (False, "SKIP: missing E_rho_lab / m_rho")
        gamma = self.E_rho_lab / self.m_rho
        if gamma < 1:
            return (False, f"gamma = {gamma} < 1 invalid")
        beta = math.sqrt(1 - 1/(gamma*gamma))
        if 0 <= beta < 1:
            return (True, f"gamma={gamma:.2f}, beta={beta:.2f} in valid range")
        return (False, f"beta = {beta} out of [0,1)")

    def test_modeling_rest_frame_momentum(self):
        """Compute p* = sqrt((m_rho/2)^2 - m_pi^2) and check > 0."""
        if None in (self.m_rho, self.m_pi):
            return (False, "SKIP: missing masses for p*")
        p_star = math.sqrt((self.m_rho/2)**2 - self.m_pi**2)
        if p_star > 0:
            return (True, f"p*={p_star:.1f} MeV/c positive")
        return (False, f"p*={p_star} should be positive")

    def test_modeling_forward_condition(self):
        """Check that beta * E_star > p_star ⇒ both pions forward."""
        if None in (self.E_rho_lab, self.m_rho, self.m_pi, self.theta_min):
            return (False, "SKIP: missing for forward condition")
        gamma = self.E_rho_lab / self.m_rho
        beta = math.sqrt(1 - 1/(gamma*gamma))
        p_star = math.sqrt((self.m_rho/2)**2 - self.m_pi**2)
        E_star = math.sqrt(p_star**2 + self.m_pi**2)
        both_forward = (beta * E_star > p_star)
        if both_forward and self.theta_min == 0:
            return (True, "Forward condition satisfied and theta_min=0")
        if not both_forward and self.theta_min > 0:
            return (True, "Forward condition not met, expected non-zero theta_min")
        return (False, f"Forward condition (beta*E* > p*) = {both_forward} inconsistent with theta_min={self.theta_min}")

    # -----------------------------------------------------------------
    # Monotonicity tests (explicit perturbation)
    # -----------------------------------------------------------------
    def test_monotonicity_decay_distance_energy(self):
        """Increase E_rho => d should increase (since gamma increases)."""
        if None in (self.E_rho_lab, self.tau, self.c, self.m_rho, self.d):
            return (False, "SKIP: missing data for monotonicity d vs E")
        # Perturb E_rho up by 10%
        E_pert = self.E_rho_lab * 1.10
        gamma_pert = E_pert / self.m_rho
        beta_pert = math.sqrt(1 - 1/(gamma_pert*gamma_pert))
        d_pert = beta_pert * gamma_pert * self.c * self.tau
        if d_pert > self.d:
            return (True, f"d increased from {self.d:e} to {d_pert:e} when E increased")
        return (False, f"d did not increase (original {self.d:e}, perturbed {d_pert:e})")

    def test_monotonicity_N_cross_section(self):
        """Increase sigma => N increases (proportional)."""
        if None in (self.sigma, self.rho_H2, self.M_H, self.N_A, self.L_target, self.N):
            return (False, "SKIP: missing for N monotonicity")
        # Perturb sigma up by 5%
        sigma_pert = self.sigma * 1.05
        n = self.rho_H2 * self.N_A / self.M_H
        N_pert = sigma_pert * n * self.L_target
        if N_pert > self.N:
            return (True, f"N increased from {self.N:e} to {N_pert:e} when sigma increased")
        return (False, f"N did not increase (original {self.N:e}, perturbed {N_pert:e})")

    # -----------------------------------------------------------------
    # Single weak answer sanity (no hardcoded target values, no isclose)
    # -----------------------------------------------------------------
    def test_answer_sanity_orders(self):
        """Check orders of magnitude (sign, finite, broad range)."""
        checks = []
        if self.tau is not None:
            if not (1e-25 < self.tau < 1e-22):
                checks.append(f"tau={self.tau} outside typical range")
        if self.d is not None:
            if not (1e-14 < self.d < 1e-11):
                checks.append(f"d={self.d} outside typical range")
        if self.E_pi_th is not None:
            if not (200 < self.E_pi_th < 5000):
                checks.append(f"E_pi_th={self.E_pi_th} outside expected range")
        if self.N is not None:
            if not (0 < self.N < 1):
                checks.append(f"N={self.N} outside (0,1)")
        if self.theta_min is not None:
            if self.theta_min < 0 or self.theta_min > math.pi:
                checks.append(f"theta_min={self.theta_min} outside [0,pi]")
        if checks:
            return (False, "; ".join(checks))
        return (True, "All answers pass broad sanity checks")

    # -----------------------------------------------------------------
    # Run all tests
    # -----------------------------------------------------------------
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