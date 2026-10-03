import sys
import math

class PhysicsChecker:
    def __init__(self, solution_instance):
        self.results = {}
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.results = solution_instance.results
        elif isinstance(solution_instance, dict) and 'results' in solution_instance and isinstance(solution_instance['results'], dict):
            self.results = solution_instance['results']
        # Use None defaults for all required keys
        self.mb_stat = self.results.get('mb_stat', None)
        self.fd_stat = self.results.get('fd_stat', None)
        self.be_stat = self.results.get('be_stat', None)
        self.T_deg = self.results.get('T_deg', None)
        self.T_F = self.results.get('T_F', None)

    # ---------- Interface checks ----------
    def test_interface_mb_stat(self):
        if self.mb_stat is None:
            return (False, "SKIP: mb_stat not provided")
        if not isinstance(self.mb_stat, str):
            return (False, "mb_stat must be a string")
        return (True, "mb_stat present and is string")

    def test_interface_fd_stat(self):
        if self.fd_stat is None:
            return (False, "SKIP: fd_stat not provided")
        if not isinstance(self.fd_stat, str):
            return (False, "fd_stat must be a string")
        return (True, "fd_stat present and is string")

    def test_interface_be_stat(self):
        if self.be_stat is None:
            return (False, "SKIP: be_stat not provided")
        if not isinstance(self.be_stat, str):
            return (False, "be_stat must be a string")
        return (True, "be_stat present and is string")

    def test_interface_T_deg(self):
        if self.T_deg is None:
            return (False, "SKIP: T_deg not provided")
        if not isinstance(self.T_deg, str):
            return (False, "T_deg must be a string")
        return (True, "T_deg present and is string")

    def test_interface_T_F(self):
        if self.T_F is None:
            return (False, "SKIP: T_F not provided")
        if not isinstance(self.T_F, (int, float)):
            return (False, "T_F must be a number")
        return (True, "T_F present and is numeric")

    # ---------- Theorem-based invariant checks ----------
    def test_invariant_statistics_denominator(self):
        """Fermi-Dirac has +1, Bose-Einstein has -1, Maxwell-Boltzmann has neither."""
        if self.fd_stat is None or self.be_stat is None or self.mb_stat is None:
            return (False, "SKIP: one or more statistics formulas missing")
        if '+1' not in self.fd_stat:
            return (False, "fd_stat should contain '+1' in denominator")
        if '-1' not in self.be_stat:
            return (False, "be_stat should contain '-1' in denominator")
        if '+1' in self.mb_stat or '-1' in self.mb_stat:
            return (False, "mb_stat should NOT contain '+1' or '-1'")
        return (True, "Denominator signs correct for FD (+1) and BE (-1); MB has no +/-1")

    def test_invariant_statistics_exponential_structure(self):
        """All three formulas contain 'exp' and 'w_l'."""
        for name, formula in [('mb_stat', self.mb_stat), ('fd_stat', self.fd_stat), ('be_stat', self.be_stat)]:
            if formula is None:
                return (False, f"SKIP: {name} not available")
            if 'w_l' not in formula:
                return (False, f"{name} missing 'w_l'")
            if 'exp' not in formula and 'e^' not in formula:
                return (False, f"{name} missing exponential (exp or e^)")
        return (True, "All three formulas contain 'w_l' and an exponential")

    def test_invariant_statistics_exponential_argument(self):
        """All formulas contain 'alpha' and 'beta epsilon_l' in the exponent."""
        for name, formula in [('mb_stat', self.mb_stat), ('fd_stat', self.fd_stat), ('be_stat', self.be_stat)]:
            if formula is None:
                return (False, f"SKIP: {name} not available")
            if 'alpha' not in formula:
                return (False, f"{name} missing 'alpha'")
            if 'beta' not in formula or 'epsilon_l' not in formula:
                return (False, f"{name} missing 'beta epsilon_l'")
        return (True, "All formulas contain 'alpha' and 'beta epsilon_l'")

    # ---------- Modeling/trace checks ----------
    def test_modeling_classical_limit_formula(self):
        """Check that T_deg formula contains essential physical constants and exponents."""
        if self.T_deg is None:
            return (False, "SKIP: T_deg not available")
        s = self.T_deg.lower()
        required = ['h^2', 'n^{2/3}', '2pi', 'm_n', 'k_b']
        for ch in required:
            if ch not in s:
                return (False, f"T_deg missing '{ch}'")
        return (True, "T_deg formula contains all required physical symbols")

    def test_modeling_fermi_temperature_numeric(self):
        """Re-derive T_F from constants and compare with provided value (theory-based check)."""
        if self.T_F is None:
            return (False, "SKIP: T_F not available")
        try:
            hbar = 1.054571817e-34
            m_n = 1.67492749804e-27
            k_B = 1.380649e-23
            n_2d = 1e16
            expected_T_F = math.pi * hbar**2 * n_2d / (m_n * k_B)
            if not math.isclose(self.T_F, expected_T_F, rel_tol=0.02, abs_tol=1e-4):
                return (False, f"T_F = {self.T_F} deviates from re-derived {expected_T_F:.6e}")
            return (True, f"T_F matches re-derived value {expected_T_F:.6e}")
        except Exception as e:
            return (False, f"T_F re-derivation error: {e}")

    # ---------- Monotonicity checks ----------
    def test_monotonicity_fermi_temperature_increases_with_density(self):
        """Perturb density upward and check that the resulting Fermi temperature increases."""
        if self.T_F is None:
            return (False, "SKIP: T_F not available")
        try:
            hbar = 1.054571817e-34
            m_n = 1.67492749804e-27
            k_B = 1.380649e-23
            base_n = 1e16
            base_T_F = math.pi * hbar**2 * base_n / (m_n * k_B)
            perturbed_n = base_n * 1.05
            perturbed_T_F = math.pi * hbar**2 * perturbed_n / (m_n * k_B)
            if not (perturbed_T_F > base_T_F):
                return (False, "Monotonicity violated: T_F should increase with density")
            # Provided T_F should be positive and finite (consistency)
            if not math.isfinite(self.T_F) or self.T_F <= 0:
                return (False, "Provided T_F is not positive and finite")
            return (True, f"Monotonicity holds: T_F increases by factor {perturbed_n/base_n}; provided T_F = {self.T_F}")
        except Exception as e:
            return (False, f"Monotonicity test error: {e}")

    # ---------- Weak answer sanity (one allowed) ----------
    def test_answer_sanity_T_F(self):
        """Broad sanity check: T_F must be positive, finite, and of order ~0.01 K."""
        if self.T_F is None:
            return (False, "SKIP: T_F not available")
        if not math.isfinite(self.T_F) or self.T_F <= 0:
            return (False, "T_F must be positive and finite")
        if self.T_F > 0.1:
            return (False, f"T_F = {self.T_F} is surprisingly large; expected ~0.015 K")
        return (True, "T_F positive and within expected order of magnitude")

    # ---------- Run all tests ----------
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