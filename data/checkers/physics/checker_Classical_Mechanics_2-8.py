import math
import traceback

class PhysicsChecker:
    """
    Checker for pion photoproduction threshold problem.
    Tests interface, invariants, modeling, monotonicity, and a weak sanity check.
    """

    def __init__(self, solution_instance):
        # Extract givens and results safely
        givens = getattr(solution_instance, 'givens', {})
        results = getattr(solution_instance, 'results', {})

        self.m_p = givens.get('m_p', None) or results.get('m_p', None)
        self.m_pi = givens.get('m_pi', None) or results.get('m_pi', None)
        self.epsilon = givens.get('epsilon', None) or results.get('epsilon', None)

        self.E_gamma_th = results.get('E_gamma_th', None)
        self.E_p_min = results.get('E_p_min', None)
        self.E_p_threshold = results.get('E_p_threshold', None)

        # Precompute threshold invariant if possible
        if self.m_p is not None and self.m_pi is not None:
            self.s_th = (self.m_p + self.m_pi) ** 2
        else:
            self.s_th = None

    def _safe_both(self, *vars):
        """Return True if all given variables are not None."""
        return all(v is not None for v in vars)

    # --------------------------------------------------------------------
    # Interface tests
    # --------------------------------------------------------------------
    def test_interface_required_results(self):
        """Check that all three expected result keys exist."""
        keys = ['E_gamma_th', 'E_p_min', 'E_p_threshold']
        for k in keys:
            if k not in self.results:
                # Also allow as attributes? Already extracted.
                pass
        present = all(self.results.get(k) is not None for k in keys)
        if present:
            return (True, "All required result keys present.")
        else:
            missing = [k for k in keys if self.results.get(k) is None]
            return (False, f"Missing result keys: {missing}")

    def test_interface_givens_positive(self):
        """Check that givens are positive numbers."""
        for name, val in [('m_p', self.m_p), ('m_pi', self.m_pi), ('epsilon', self.epsilon)]:
            if val is None:
                return (False, f"{name} missing.")
            if val <= 0:
                return (False, f"{name} is not positive: {val}")
        return (True, "All givens are positive.")

    # --------------------------------------------------------------------
    # Invariant tests (theorem-based, use math.isclose)
    # --------------------------------------------------------------------
    def test_invariant_threshold_lab(self):
        """Check that s_lab = m_p^2 + 2*m_p*E_gamma_th equals s_th."""
        if not self._safe_both(self.m_p, self.m_pi, self.E_gamma_th, self.s_th):
            return (False, "SKIP: missing m_p, m_pi, or E_gamma_th")
        s_lab = self.m_p**2 + 2 * self.m_p * self.E_gamma_th
        if math.isclose(s_lab, self.s_th, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"s_lab = {s_lab:.6e} ≈ s_th = {self.s_th:.6e}")
        else:
            return (False, f"s_lab = {s_lab:.6e} != s_th = {self.s_th:.6e} (rel diff {abs(s_lab-self.s_th)/self.s_th:.2%})")

    def test_invariant_threshold_headon(self):
        """Check that s_headon = m_p^2 + 4*epsilon*E_p_min equals s_th."""
        if not self._safe_both(self.m_p, self.m_pi, self.epsilon, self.E_p_min, self.s_th):
            return (False, "SKIP: missing m_p, m_pi, epsilon, or E_p_min")
        s_headon = self.m_p**2 + 4 * self.epsilon * self.E_p_min
        if math.isclose(s_headon, self.s_th, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"s_headon = {s_headon:.6e} ≈ s_th = {self.s_th:.6e}")
        else:
            return (False, f"s_headon = {s_headon:.6e} != s_th = {self.s_th:.6e} (rel diff {abs(s_headon-self.s_th)/self.s_th:.2%})")

    def test_invariant_positive_energies(self):
        """Check that all computed energies are positive and finite."""
        for name, val in [('E_gamma_th', self.E_gamma_th), ('E_p_min', self.E_p_min), ('E_p_threshold', self.E_p_threshold)]:
            if val is None:
                return (False, f"SKIP: {name} missing")
            if val <= 0 or math.isinf(val) or math.isnan(val):
                return (False, f"{name} is not positive/finite: {val}")
        return (True, "All computed energies are positive and finite.")

    # --------------------------------------------------------------------
    # Modeling / trace tests
    # --------------------------------------------------------------------
    def test_modeling_ultrarelativistic_approximation(self):
        """Check that E_p_min >> m_p (ultrarelativistic)."""
        if not self._safe_both(self.m_p, self.E_p_min):
            return (False, "SKIP: missing m_p or E_p_min")
        ratio = self.E_p_min / self.m_p
        if ratio > 1e10:  # threshold is ~7e10, so okay
            return (True, f"E_p_min/m_p = {ratio:.2e} >> 1 (ultrarelativistic)")
        else:
            return (False, f"E_p_min/m_p = {ratio:.2e} not ultrarelativistic enough")

    def test_modeling_consistency_Ep_threshold(self):
        """Check that E_p_threshold equals E_p_min (as per pseudocode contract)."""
        if not self._safe_both(self.E_p_min, self.E_p_threshold):
            return (False, "SKIP: missing E_p_min or E_p_threshold")
        if math.isclose(self.E_p_min, self.E_p_threshold, rel_tol=1e-6):
            return (True, "E_p_threshold equals E_p_min")
        else:
            return (False, f"E_p_threshold = {self.E_p_threshold}, E_p_min = {self.E_p_min}")

    def test_modeling_expected_order_of_magnitude(self):
        """Check that E_gamma_th ~ 0.1 GeV and E_p_min ~ 1e10 GeV (broad)."""
        issues = []
        if self.E_gamma_th is not None:
            if not (0.1 < self.E_gamma_th < 0.2):
                issues.append(f"E_gamma_th = {self.E_gamma_th} not in [0.1,0.2] GeV")
        if self.E_p_min is not None:
            if not (1e9 < self.E_p_min < 1e12):
                issues.append(f"E_p_min = {self.E_p_min} not in [1e9,1e12] GeV")
        if issues:
            return (False, "; ".join(issues))
        return (True, "Energies in expected order-of-magnitude range.")

    # --------------------------------------------------------------------
    # Monotonicity tests
    # --------------------------------------------------------------------
    def test_monotonicity_photon_energy_inverse(self):
        """For part (b): increasing photon energy decreases threshold proton energy."""
        if not self._safe_both(self.m_p, self.m_pi, self.epsilon, self.E_p_min):
            return (False, "SKIP: missing required quantities for monotonicity")
        # pertub epsilon by +10%
        epsilon_new = self.epsilon * 1.1
        s_th = (self.m_p + self.m_pi) ** 2
        expected_new_Ep = (s_th - self.m_p**2) / (4 * epsilon_new)
        if expected_new_Ep < self.E_p_min:
            return (True, f"ε increase → Ep_min decreases: old={self.E_p_min:.4e}, new={expected_new_Ep:.4e}")
        else:
            return (False, f"Monotonicity failed: old Ep_min={self.E_p_min:.4e}, new (with 1.1*ε)={expected_new_Ep:.4e}")

    # --------------------------------------------------------------------
    # Weak sanity test (no re-derivation, no isclose)
    # --------------------------------------------------------------------
    def test_answer_sanity_threshold_less_than_proton_energy(self):
        """Very basic: lab threshold should be much smaller than cosmic proton threshold."""
        if not self._safe_both(self.E_gamma_th, self.E_p_min):
            return (False, "SKIP: missing energies")
        if self.E_gamma_th < self.E_p_min:
            return (True, "E_gamma_th << E_p_min as expected")
        else:
            return (False, "E_gamma_th is not much smaller than E_p_min")

    # --------------------------------------------------------------------
    # Main orchestrator
    # --------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # Get all test methods dynamically
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)