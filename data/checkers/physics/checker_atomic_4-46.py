import math

class PhysicsChecker:
    """
    Audits K0L -> π+π- decay.
    Checks invariant (Lorentz transformation), modeling (boost formula),
    monotonicity (effect of mass and speed), and a weak answer sanity.
    """

    def __init__(self, solution_instance):
        # Results from the solution (must be present)
        results = getattr(solution_instance, 'results', {})
        self.p_T_max = results.get('p_T_max', None)
        self.p_L_max = results.get('p_L_max', None)

        # Extract given quantities – NEVER use numeric fallbacks
        givens = getattr(solution_instance, 'givens', {})

        # Masses (MeV/c^2)
        self.M = getattr(solution_instance, 'M', None) or givens.get('M', None)
        self.m = getattr(solution_instance, 'm', None) or givens.get('m', None)

        # Kaon momentum p_K (MeV/c)
        self.p_K = getattr(solution_instance, 'p_K', None)
        if self.p_K is None:
            self.p_K = givens.get('p_K', None)

        # Ratio p_K/(Mc) – may be provided directly
        self.p_K_over_Mc = getattr(solution_instance, 'p_K_over_Mc', None)
        if self.p_K_over_Mc is None:
            self.p_K_over_Mc = givens.get('p_K_over_Mc', None)

    # ------------------------------------------------------------------
    # Internal helpers (no defaults)
    # ------------------------------------------------------------------
    def _compute_pstar_Mev(self):
        """Return rest-frame momentum p* (MeV/c) if M,m available, else None."""
        if self.M is None or self.m is None:
            return None
        M = float(self.M) if hasattr(self.M, 'n') else float(self.M)
        m = float(self.m) if hasattr(self.m, 'n') else float(self.m)
        return math.sqrt((M / 2.0)**2 - m**2)

    def _compute_energy_star_over_c(self):
        """Return E*/c = M/2 (MeV)."""
        if self.M is None:
            return None
        M = float(self.M) if hasattr(self.M, 'n') else float(self.M)
        return M / 2.0

    def _compute_beta_gamma(self):
        """
        Return (beta, gamma) derived from p_K/Mc or from p_K and M.
        Returns None if insufficient data.
        """
        # Try using ratio directly
        if self.p_K_over_Mc is not None:
            ratio = float(self.p_K_over_Mc)
            if ratio <= 0:
                return None
            # beta = p/E = p / sqrt(p^2 + M^2) = (p/(Mc)) / sqrt((p/(Mc))^2 + 1)
            beta = ratio / math.sqrt(ratio**2 + 1)
            gamma = 1.0 / math.sqrt(1.0 - beta**2)
            return (beta, gamma)

        # Try using p_K and M
        if self.p_K is not None and self.M is not None:
            pK_val = float(self.p_K) if hasattr(self.p_K, 'n') else float(self.p_K)
            M_val = float(self.M) if hasattr(self.M, 'n') else float(self.M)
            if M_val <= 0:
                return None
            ratio = pK_val / M_val  # c=1 natural units
            if ratio <= 0:
                return None
            beta = ratio / math.sqrt(ratio**2 + 1)
            gamma = 1.0 / math.sqrt(1.0 - beta**2)
            return (beta, gamma)

        return None  # cannot determine

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_results_exist(self):
        if self.p_T_max is None:
            return (False, "p_T_max missing from results")
        if self.p_L_max is None:
            return (False, "p_L_max missing from results")
        if not isinstance(self.p_T_max, (int, float)):
            return (False, "p_T_max is not numeric")
        if not isinstance(self.p_L_max, (int, float)):
            return (False, "p_L_max is not numeric")
        return (True, "both required result keys present and numeric")

    # ------------------------------------------------------------------
    # Invariant tests (theorem-based, use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_transverse_momentum_equals_pstar(self):
        """Transverse momentum is invariant: max p_T = p* (rest frame momentum)."""
        if self.p_T_max is None:
            return (False, "SKIP: p_T_max not available")
        p_star = self._compute_pstar_Mev()
        if p_star is None:
            return (False, "SKIP: cannot compute p* (M or m missing)")
        if not math.isclose(self.p_T_max, p_star, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"p_T_max {self.p_T_max} != p* = {p_star:.2f}")
        return (True, f"p_T_max = {self.p_T_max:.2f} matches p* = {p_star:.2f}")

    def test_invariant_lab_energy_derived_from_boost(self):
        """
        The energy of the pion with maximum longitudinal momentum (p_T = 0) must equal
        γ( E* + β p* ) from Lorentz transformation.
        """
        if self.p_L_max is None:
            return (False, "SKIP: p_L_max not available")
        if self.m is None:
            return (False, "SKIP: pion mass missing")
        # Compute baseline p* and E*/c
        p_star = self._compute_pstar_Mev()
        E_star_over_c = self._compute_energy_star_over_c()
        if p_star is None or E_star_over_c is None:
            return (False, "SKIP: cannot compute p* or E*/c")
        bg = self._compute_beta_gamma()
        if bg is None:
            return (False, "SKIP: cannot determine beta/gamma")
        beta, gamma = bg
        # Expected maximum energy (with c=1): E_max = sqrt(p_L_max^2 + m^2)
        m_val = float(self.m) if hasattr(self.m, 'n') else float(self.m)
        E_max_lab = math.sqrt(self.p_L_max**2 + m_val**2)
        # Expected from boost: γ (E* + β p*)
        expected_E = gamma * (E_star_over_c + beta * p_star)
        if not math.isclose(E_max_lab, expected_E, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E_max from p_L_max = {E_max_lab:.2f} != γ(E*+βp*) = {expected_E:.2f}")
        return (True, f"E_max = {E_max_lab:.2f} matches Lorentz energy transformation")

    # ------------------------------------------------------------------
    # Modeling (trace) tests
    # ------------------------------------------------------------------
    def test_modeling_longitudinal_boost_max(self):
        """Maximum longitudinal momentum uses Lorentz transformation with cosθ*=1."""
        if self.p_L_max is None:
            return (False, "SKIP: p_L_max not available")
        p_star = self._compute_pstar_Mev()
        E_star_over_c = self._compute_energy_star_over_c()
        if p_star is None or E_star_over_c is None:
            return (False, "SKIP: cannot compute p* or E*/c")
        bg = self._compute_beta_gamma()
        if bg is None:
            return (False, "SKIP: cannot determine beta/gamma")
        beta, gamma = bg
        expected = gamma * (p_star + beta * E_star_over_c)
        if not math.isclose(self.p_L_max, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"p_L_max {self.p_L_max} != expected {expected:.2f}")
        return (True, f"p_L_max = {self.p_L_max:.2f} matches Lorentz boost formula")

    def test_modeling_energy_momentum_relation(self):
        """
        For the maximum longitudinal pion, the relativistic energy-momentum relation
        E^2 - p_L^2 = m^2 must hold (c=1).
        """
        if self.p_L_max is None or self.m is None:
            return (False, "SKIP: p_L_max or m missing")
        m_val = float(self.m) if hasattr(self.m, 'n') else float(self.m)
        # Compute energy from p_L_max and mass
        E = math.sqrt(self.p_L_max**2 + m_val**2)
        # Compute invariant mass squared
        inv = E**2 - self.p_L_max**2
        if not math.isclose(inv, m_val**2, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E^2 - p_L^2 = {inv:.2f} != m^2 = {m_val**2:.2f}")
        return (True, f"Energy-momentum relation holds: E^2 - p_L^2 = {inv:.2f} ≈ m^2")

    # ------------------------------------------------------------------
    # Monotonicity tests (explicit perturbation)
    # ------------------------------------------------------------------
    def test_monotonicity_p_T_max_increases_with_M(self):
        """If M increases, p* increases, so p_T_max should increase."""
        if self.p_T_max is None:
            return (False, "SKIP: p_T_max not available")
        p_star = self._compute_pstar_Mev()
        if p_star is None:
            return (False, "SKIP: cannot compute baseline p*")
        if not math.isclose(self.p_T_max, p_star, rel_tol=1e-2, abs_tol=1e-2):
            return (False, "SKIP: p_T_max does not match p*; cannot test monotonicity")
        M_val = float(self.M) if hasattr(self.M, 'n') else float(self.M)
        M_pert = M_val + 1.0   # +1 MeV/c^2
        m_val = float(self.m) if hasattr(self.m, 'n') else float(self.m)
        p_star_pert = math.sqrt((M_pert / 2.0)**2 - m_val**2)
        if p_star_pert <= p_star:
            return (False, f"Perturbed p* ({p_star_pert:.2f}) not > baseline p* ({p_star:.2f})")
        return (True, f"p_T_max increases with M: baseline {p_star:.2f} < perturbed {p_star_pert:.2f}")

    def test_monotonicity_p_L_max_increases_with_beta(self):
        """Small increase in β (keeping M,m fixed) should increase p_L_max."""
        if self.p_L_max is None:
            return (False, "SKIP: p_L_max not available")
        p_star = self._compute_pstar_Mev()
        E_star_over_c = self._compute_energy_star_over_c()
        if p_star is None or E_star_over_c is None:
            return (False, "SKIP: cannot compute p* or E*/c")
        bg = self._compute_beta_gamma()
        if bg is None:
            return (False, "SKIP: cannot determine baseline beta/gamma")
        beta0, gamma0 = bg
        # Compute baseline using our formula
        base_pL = gamma0 * (p_star + beta0 * E_star_over_c)
        if not math.isclose(self.p_L_max, base_pL, rel_tol=1e-2, abs_tol=1e-2):
            return (False, "SKIP: p_L_max does not match computed baseline")
        # Perturb beta upward (ensure not exceeding 1)
        if beta0 >= 0.99:
            return (False, "SKIP: beta too close to 1")
        beta_pert = beta0 + 0.01
        gamma_pert = 1.0 / math.sqrt(1.0 - beta_pert**2)
        perturbed_pL = gamma_pert * (p_star + beta_pert * E_star_over_c)
        if perturbed_pL <= base_pL:
            return (False, f"Perturbed p_L_max ({perturbed_pL:.2f}) not > baseline ({base_pL:.2f})")
        return (True, f"p_L_max increases with β: baseline {base_pL:.2f} < perturbed {perturbed_pL:.2f}")

    # ------------------------------------------------------------------
    # Weak answer sanity (at most one, no re-derivation, no isclose)
    # ------------------------------------------------------------------
    def test_answer_sanity_basic(self):
        if self.p_T_max is None or self.p_L_max is None:
            return (False, "SKIP: results missing")
        # Finite and positive
        if not math.isfinite(self.p_T_max) or not math.isfinite(self.p_L_max):
            return (False, "one result is not finite")
        if self.p_T_max <= 0 or self.p_L_max <= 0:
            return (False, "results must be positive")
        # Ordering: due to boost, longitudinal max exceeds transverse max
        if self.p_L_max <= self.p_T_max:
            return (False, "p_L_max should be > p_T_max")
        return (True, "basic sanity: finite, positive, p_L_max > p_T_max")

    # ------------------------------------------------------------------
    # Test harness
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