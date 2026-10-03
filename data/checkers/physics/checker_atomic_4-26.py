import math
import traceback
import sympy as sp

class PhysicsChecker:
    """
    Audits a solution for the muon decay and proton conversion problem.
    Checks interface completeness, physical invariants, modeling quality,
    and monotonicity trends.
    """

    def __init__(self, solution_instance):
        """
        Extract all required data from solution_instance safely using None defaults.
        Assumes solution_instance has:
          - .results (dict)         -> keys: 'reaction', 'E_nue_bar_max', 'E_threshold'
          - .givens (dict, optional) -> keys: 'm_mu', 'm_e', 'm_p', 'm_n' (MeV/c^2)
        """
        # Results from the solution
        self.results = getattr(solution_instance, 'results', {})
        self.reaction = self.results.get('reaction', None)
        self.E_nue_bar_max = self.results.get('E_nue_bar_max', None)
        self.E_threshold = self.results.get('E_threshold', None)

        # Givens (physics constants) – use None if missing
        self.givens = getattr(solution_instance, 'givens', {})
        self.m_mu = self.givens.get('m_mu', None)
        self.m_e  = self.givens.get('m_e', None)
        self.m_p  = self.givens.get('m_p', None)
        self.m_n  = self.givens.get('m_n', None)

        # Pre‑compute expected values if masses are available (for invariant checks)
        self._expected_E_nue_bar_max = None
        self._expected_E_threshold = None
        if self.m_mu is not None and self.m_e is not None:
            self._expected_E_nue_bar_max = self._compute_E_nue_bar_max(self.m_mu, self.m_e)
        if self.m_p is not None and self.m_n is not None and self.m_e is not None:
            self._expected_E_threshold = self._compute_E_threshold(self.m_p, self.m_n, self.m_e)

    # ----------------------------------------------------------------------
    # Internal helpers (do not start with "test_")
    # ----------------------------------------------------------------------
    def _compute_E_nue_bar_max(self, m_mu, m_e):
        """Maximum antineutrino energy from muon decay (natural units, c=1)."""
        return (m_mu**2 - m_e**2) / (2 * m_mu)

    def _compute_E_threshold(self, m_p, m_n, m_e):
        """Threshold energy for inverse beta decay ν̅_e + p → n + e⁺ (natural units)."""
        return ((m_n + m_e)**2 - m_p**2) / (2 * m_p)

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_reaction_key(self):
        """Check that the reaction key exists and is a non‑empty string."""
        if self.reaction is None:
            return (False, "Missing 'reaction' key in results.")
        if not isinstance(self.reaction, str) or len(self.reaction.strip()) == 0:
            return (False, f"Reaction is not a non‑empty string: {self.reaction}")
        return (True, f"Reaction present: '{self.reaction}'")

    def test_interface_E_nue_bar_max_key(self):
        """Check that E_nue_bar_max exists and is a number."""
        if self.E_nue_bar_max is None:
            return (False, "Missing 'E_nue_bar_max' key in results.")
        if not isinstance(self.E_nue_bar_max, (int, float)):
            return (False, f"E_nue_bar_max is not numeric: {self.E_nue_bar_max}")
        return (True, f"E_nue_bar_max = {self.E_nue_bar_max}")

    def test_interface_E_threshold_key(self):
        """Check that E_threshold exists and is a number."""
        if self.E_threshold is None:
            return (False, "Missing 'E_threshold' key in results.")
        if not isinstance(self.E_threshold, (int, float)):
            return (False, f"E_threshold is not numeric: {self.E_threshold}")
        return (True, f"E_threshold = {self.E_threshold}")

    # ----------------------------------------------------------------------
    # Invariant / theorem‑based sanity tests (use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_max_antineutrino_energy(self):
        """Verify that E_nue_bar_max matches the theoretical expression (if masses available)."""
        if self._expected_E_nue_bar_max is None:
            return (False, "SKIP: missing required masses (m_mu, m_e) in givens.")
        if self.E_nue_bar_max is None:
            return (False, "SKIP: results['E_nue_bar_max'] is missing.")
        if not math.isclose(self.E_nue_bar_max, self._expected_E_nue_bar_max,
                            rel_tol=1e-2, abs_tol=1e-2):
            return (False,
                    f"E_nue_bar_max mismatch: got {self.E_nue_bar_max}, "
                    f"expected ≈ {self._expected_E_nue_bar_max:.4f} MeV.")
        return (True, f"E_nue_bar_max = {self.E_nue_bar_max} ≈ {self._expected_E_nue_bar_max:.4f} MeV (OK)")

    def test_invariant_threshold_energy(self):
        """Verify that E_threshold matches the theoretical expression (if masses available)."""
        if self._expected_E_threshold is None:
            return (False, "SKIP: missing required masses (m_p, m_n, m_e) in givens.")
        if self.E_threshold is None:
            return (False, "SKIP: results['E_threshold'] is missing.")
        if not math.isclose(self.E_threshold, self._expected_E_threshold,
                            rel_tol=1e-2, abs_tol=1e-2):
            return (False,
                    f"E_threshold mismatch: got {self.E_threshold}, "
                    f"expected ≈ {self._expected_E_threshold:.4f} MeV.")
        return (True, f"E_threshold = {self.E_threshold} ≈ {self._expected_E_threshold:.4f} MeV (OK)")

    def test_invariant_energy_sufficiency(self):
        """Check that the maximum antineutrino energy exceeds the threshold (sufficiency)."""
        if self.E_nue_bar_max is None or self.E_threshold is None:
            return (False, "SKIP: one or both energies missing.")
        if self.E_nue_bar_max <= self.E_threshold:
            return (False,
                    f"Insufficient energy: E_nue_bar_max = {self.E_nue_bar_max} MeV "
                    f"≤ E_threshold = {self.E_threshold} MeV.")
        return (True, f"E_nue_bar_max ({self.E_nue_bar_max}) > E_threshold ({self.E_threshold}) – sufficient.")

    def test_invariant_positivity(self):
        """All energies must be non‑negative (physical)."""
        if self.E_nue_bar_max is not None and self.E_nue_bar_max < 0:
            return (False, f"E_nue_bar_max negative: {self.E_nue_bar_max}")
        if self.E_threshold is not None and self.E_threshold < 0:
            return (False, f"E_threshold negative: {self.E_threshold}")
        return (True, "All available energies are non‑negative.")

    # ----------------------------------------------------------------------
    # Modeling / trace tests (based on explanation steps)
    # ----------------------------------------------------------------------
    def test_modeling_reaction_contains_correct_particles(self):
        """Reaction string must contain muon, electron, antineutrino, and neutrino symbols."""
        if self.reaction is None:
            return (False, "SKIP: reaction not available.")
        # standard notations: mu⁻, e⁻, ν̅_e, ν_μ or similar
        required = [r'\mu', r'e', r'\nu', r'\overline']  # common in LaTeX
        missing = [p for p in required if p not in self.reaction]
        # also accept plain text 'mu-', etc.
        alt_required = ['mu', 'e', 'nu', 'bar']
        missing_alt = [p for p in alt_required if p not in self.reaction]
        if missing and missing_alt:
            # allow either notation
            return (False,
                    f"Reaction missing required particles: {missing} or {missing_alt}.")
        return (True, f"Reaction includes all required particles: {self.reaction}")

    def test_modeling_arrow_present(self):
        """Reaction must contain an arrow '->' or '→'."""
        if self.reaction is None:
            return (False, "SKIP: reaction not available.")
        if '->' not in self.reaction and '→' not in self.reaction:
            return (False, "Reaction missing arrow.")
        return (True, "Reaction contains arrow.")

    # ----------------------------------------------------------------------
    # Monotonicity tests (perturb one input and check trend)
    # ----------------------------------------------------------------------
    def test_monotonicity_max_energy_increases_with_m_mu(self):
        """If m_mu increases, E_nue_bar_max should increase (positive derivative)."""
        if self.m_mu is None or self.m_e is None or self.E_nue_bar_max is None:
            return (False, "SKIP: missing masses or E_nue_bar_max.")
        # perturb m_mu up by 1% (small)
        delta = 0.01 * self.m_mu
        m_mu_perturbed = self.m_mu + delta
        E_perturbed = self._compute_E_nue_bar_max(m_mu_perturbed, self.m_e)
        if E_perturbed <= self.E_nue_bar_max:
            return (False,
                    f"Monotonicity violation: increasing m_mu from {self.m_mu} to "
                    f"{m_mu_perturbed} gave E_max = {E_perturbed} (was {self.E_nue_bar_max}).")
        return (True, f"E_nue_bar_max increases with m_mu (perturbed: {E_perturbed:.4f} > {self.E_nue_bar_max:.4f})")

    def test_monotonicity_max_energy_decreases_with_m_e(self):
        """If m_e increases, E_nue_bar_max should decrease (negative derivative)."""
        if self.m_mu is None or self.m_e is None or self.E_nue_bar_max is None:
            return (False, "SKIP: missing masses or E_nue_bar_max.")
        delta = 0.01 * self.m_e
        m_e_perturbed = self.m_e + delta
        E_perturbed = self._compute_E_nue_bar_max(self.m_mu, m_e_perturbed)
        if E_perturbed >= self.E_nue_bar_max:
            return (False,
                    f"Monotonicity violation: increasing m_e from {self.m_e} to "
                    f"{m_e_perturbed} gave E_max = {E_perturbed} (was {self.E_nue_bar_max}).")
        return (True, f"E_nue_bar_max decreases with m_e (perturbed: {E_perturbed:.4f} < {self.E_nue_bar_max:.4f})")

    def test_monotonicity_threshold_increases_with_m_n(self):
        """If m_n increases, E_threshold should increase (positive derivative)."""
        if self.m_p is None or self.m_n is None or self.m_e is None or self.E_threshold is None:
            return (False, "SKIP: missing masses or E_threshold.")
        delta = 0.01 * self.m_n
        m_n_perturbed = self.m_n + delta
        E_perturbed = self._compute_E_threshold(self.m_p, m_n_perturbed, self.m_e)
        if E_perturbed <= self.E_threshold:
            return (False,
                    f"Monotonicity violation: increasing m_n from {self.m_n} to "
                    f"{m_n_perturbed} gave E_thr = {E_perturbed} (was {self.E_threshold}).")
        return (True, f"E_threshold increases with m_n (perturbed: {E_perturbed:.4f} > {self.E_threshold:.4f})")

    def test_monotonicity_threshold_decreases_with_m_p(self):
        """If m_p increases, E_threshold should decrease (negative derivative)."""
        if self.m_p is None or self.m_n is None or self.m_e is None or self.E_threshold is None:
            return (False, "SKIP: missing masses or E_threshold.")
        delta = 0.01 * self.m_p
        m_p_perturbed = self.m_p + delta
        # Threshold formula: ((m_n+m_e)^2 - m_p^2)/(2 m_p) -> derivative w.r.t m_p is negative for typical masses
        E_perturbed = self._compute_E_threshold(m_p_perturbed, self.m_n, self.m_e)
        if E_perturbed >= self.E_threshold:
            return (False,
                    f"Monotonicity violation: increasing m_p from {self.m_p} to "
                    f"{m_p_perturbed} gave E_thr = {E_perturbed} (was {self.E_threshold}).")
        return (True, f"E_threshold decreases with m_p (perturbed: {E_perturbed:.4f} < {self.E_threshold:.4f})")

    # ----------------------------------------------------------------------
    # Weak answer sanity test (do not use isclose or rederive exact values)
    # ----------------------------------------------------------------------
    def test_answer_sanity_energies_positive_and_finite(self):
        """Both energies should be positive and finite (no NaN/Inf)."""
        for name, val in [('E_nue_bar_max', self.E_nue_bar_max), ('E_threshold', self.E_threshold)]:
            if val is None:
                continue
            if not math.isfinite(val):
                return (False, f"{name} is not finite: {val}")
            if val <= 0:
                return (False, f"{name} is not positive: {val}")
        return (True, "All present energies are positive and finite.")

    # ----------------------------------------------------------------------
    # Main runner
    # ----------------------------------------------------------------------
    def run_all_tests(self):
        """Discover and run all test_* methods, collect logs."""
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