import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Extract all needed data from solution_instance safely.
        Use None defaults for missing attributes/keys.
        """
        # Givens (masses in GeV)
        self.m_pi = getattr(solution_instance, 'm_pi', None)
        self.m_p = getattr(solution_instance, 'm_p', None)
        self.m_K = getattr(solution_instance, 'm_K', None)
        self.m_n = getattr(solution_instance, 'm_n', None)

        # Results
        if hasattr(solution_instance, 'results'):
            self.E_pi = solution_instance.results.get('E_pi', None)
        else:
            self.E_pi = None

        # Derived threshold mass for final state
        self.M_final = None
        if all(x is not None for x in [self.m_K, self.m_n]):
            self.M_final = 2 * self.m_K + self.m_n

        # Derived pion momentum (optional, used in an invariant test)
        self.p_pi = None
        if self.E_pi is not None and self.m_pi is not None:
            if self.E_pi >= self.m_pi:
                self.p_pi = math.sqrt(self.E_pi**2 - self.m_pi**2)

    # ---------- Interface checks ----------
    def test_interface_E_pi_exists(self):
        """Check that results['E_pi'] is present and numeric."""
        if self.E_pi is None:
            return (False, "E_pi not found in results")
        if not isinstance(self.E_pi, (int, float)):
            return (False, f"E_pi is not numeric, got {type(self.E_pi)}")
        return (True, "E_pi exists and is numeric")

    def test_interface_givens_present(self):
        """Check that all required mass givens are provided."""
        missing = []
        for name, val in [('m_pi', self.m_pi), ('m_p', self.m_p),
                          ('m_K', self.m_K), ('m_n', self.m_n)]:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {', '.join(missing)}")
        return (True, "All required mass givens present")

    # ---------- Invariant/theorem checks (at least two) ----------
    def test_invariant_mass_conservation_formula(self):
        """
        Check the algebraic relation:
        m_pi^2 + m_p^2 + 2 m_p E_pi = (2 m_K + m_n)^2.
        """
        if any(x is None for x in [self.m_pi, self.m_p, self.E_pi, self.M_final]):
            return (False, "SKIP: missing data for invariant mass formula check")
        lhs = self.m_pi**2 + self.m_p**2 + 2 * self.m_p * self.E_pi
        rhs = self.M_final**2
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Invariant mass formula holds: lhs={lhs:.4f}, rhs={rhs:.4f}")
        else:
            return (False, f"Invariant mass formula fails: lhs={lhs:.4f}, rhs={rhs:.4f}")

    def test_invariant_4momentum_squared(self):
        """
        Check the invariant computed from the initial 4-momentum directly:
        S = (E_pi + m_p)^2 - p_pi^2 should equal (2 m_K + m_n)^2.
        This uses the same physics but a different numerical computation.
        """
        if any(x is None for x in [self.m_pi, self.m_p, self.E_pi, self.p_pi, self.M_final]):
            return (False, "SKIP: missing data for 4-momentum invariant check")
        S_initial = (self.E_pi + self.m_p)**2 - self.p_pi**2
        rhs = self.M_final**2
        if math.isclose(S_initial, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Initial 4-momentum invariant S={S_initial:.4f} matches threshold={rhs:.4f}")
        else:
            return (False, f"Initial 4-momentum invariant S={S_initial:.4f} != threshold={rhs:.4f}")

    # ---------- Modeling / trace checks ----------
    def test_modeling_threshold_derivation(self):
        """
        Re-derive E_pi from the invariant mass formula and compare
        with the result.
        """
        if any(x is None for x in [self.m_pi, self.m_p, self.M_final, self.E_pi]):
            return (False, "SKIP: missing data for threshold derivation check")
        E_pi_calc = (self.M_final**2 - self.m_pi**2 - self.m_p**2) / (2 * self.m_p)
        if math.isclose(self.E_pi, E_pi_calc, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Derived E_pi={E_pi_calc:.4f} matches result E_pi={self.E_pi:.4f}")
        else:
            return (False, f"Derived E_pi={E_pi_calc:.4f} != result E_pi={self.E_pi:.4f}")

    def test_modeling_positive_energy(self):
        """E_pi must be greater than rest mass of pion (physical)."""
        if self.E_pi is None or self.m_pi is None:
            return (False, "SKIP: missing data for positivity check")
        if self.E_pi > self.m_pi:
            return (True, f"E_pi={self.E_pi:.4f} > m_pi={self.m_pi:.4f}")
        else:
            return (False, f"E_pi={self.E_pi:.4f} not greater than m_pi={self.m_pi:.4f}")

    # ---------- Monotonicity checks ----------
    def test_monotonicity_threshold_increases_with_kaon_mass(self):
        """
        If m_K increases by 1%, the computed threshold E_pi should increase.
        """
        if any(x is None for x in [self.m_pi, self.m_p, self.m_K, self.m_n]):
            return (False, "SKIP: missing data for monotonicity check (kaon mass)")
        # Baseline threshold
        M_final_base = 2 * self.m_K + self.m_n
        E_pi_base = (M_final_base**2 - self.m_pi**2 - self.m_p**2) / (2 * self.m_p)

        # Perturb m_K upward by 1%
        m_K_pert = self.m_K * 1.01
        M_final_pert = 2 * m_K_pert + self.m_n
        E_pi_pert = (M_final_pert**2 - self.m_pi**2 - self.m_p**2) / (2 * self.m_p)

        if E_pi_pert > E_pi_base:
            return (True, f"Increasing m_K increases threshold: base={E_pi_base:.4f}, perturbed={E_pi_pert:.4f}")
        else:
            return (False, f"Monotonicity violated: base={E_pi_base:.4f}, perturbed={E_pi_pert:.4f}")

    def test_monotonicity_threshold_decreases_with_pion_mass(self):
        """
        If m_pi increases by 5%, the computed threshold E_pi should decrease.
        """
        if any(x is None for x in [self.m_pi, self.m_p, self.m_K, self.m_n]):
            return (False, "SKIP: missing data for monotonicity check (pion mass)")
        M_final = 2 * self.m_K + self.m_n
        E_pi_base = (M_final**2 - self.m_pi**2 - self.m_p**2) / (2 * self.m_p)

        m_pi_pert = self.m_pi * 1.05
        E_pi_pert = (M_final**2 - m_pi_pert**2 - self.m_p**2) / (2 * self.m_p)

        if E_pi_pert < E_pi_base:
            return (True, f"Increasing m_pi decreases threshold: base={E_pi_base:.4f}, perturbed={E_pi_pert:.4f}")
        else:
            return (False, f"Monotonicity violated: base={E_pi_base:.4f}, perturbed={E_pi_pert:.4f}")

    # ---------- Answer sanity (weak) ----------
    def test_answer_sanity_positive_finite(self):
        """Final sanity: E_pi is positive, finite, and in a broad plausible range."""
        if self.E_pi is None:
            return (False, "SKIP: E_pi not provided")
        if not math.isfinite(self.E_pi):
            return (False, "E_pi is not finite")
        if self.E_pi <= 0:
            return (False, "E_pi is not positive")
        # Broad plausible range: threshold for K- production > sum of masses involved?
        # but we keep it loose: 0.5 < E_pi < 10 GeV
        if 0.5 <= self.E_pi <= 10.0:
            return (True, f"E_pi={self.E_pi:.4f} GeV is plausible")
        else:
            return (False, f"E_pi={self.E_pi:.4f} GeV is outside plausible range [0.5,10]")

    # ---------- Run all tests ----------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Discover all methods starting with 'test_'
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