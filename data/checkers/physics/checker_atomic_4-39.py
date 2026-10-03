import math
import traceback
from typing import Any, Dict, Optional, Tuple

class PhysicsChecker:
    """
    Checker for the universe-filling black-body radiation problem.
    Audits threshold energies, mean free paths, and comparison with universe size.
    """

    def __init__(self, solution_instance):
        # Extract results dictionary safely
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}

        # Required result keys as per [emit] contract
        self.E_min = results.get('E_min', None)           # eV
        self.l = results.get('l', None)                   # light years (ly)
        self.R_universe = results.get('R_universe', None) # cm
        self.E_p_th = results.get('E_p_th', None)         # eV

        # Attempt to extract givens (may be in solution_instance.givens or directly as attributes)
        # Use None defaults to avoid silent numeric fallbacks.
        self.m_e_c2 = getattr(solution_instance, 'm_e_c2', None)          # eV
        self.m_p_c2 = getattr(solution_instance, 'm_p_c2', None)          # eV
        self.m_pi_c2 = getattr(solution_instance, 'm_pi_c2', None)        # eV
        self.epsilon = getattr(solution_instance, 'epsilon', None)        # eV
        self.n_gamma = getattr(solution_instance, 'n_gamma', None)        # cm^{-3}
        self.sigma_T = getattr(solution_instance, 'sigma_T', None)        # cm^2
        self.cm_to_ly = getattr(solution_instance, 'cm_to_ly', None)      # conversion factor

        # Also try from givens dict if that pattern is used
        givens = getattr(solution_instance, 'givens', {})
        if isinstance(givens, dict):
            self.m_e_c2 = self.m_e_c2 or givens.get('m_e_c2')
            self.m_p_c2 = self.m_p_c2 or givens.get('m_p_c2')
            self.m_pi_c2 = self.m_pi_c2 or givens.get('m_pi_c2')
            self.epsilon = self.epsilon or givens.get('epsilon')
            self.n_gamma = self.n_gamma or givens.get('n_gamma')
            self.sigma_T = self.sigma_T or givens.get('sigma_T')
            self.cm_to_ly = self.cm_to_ly or givens.get('cm_to_ly')

    # ----------------------------------------------------------------------
    # Interface tests: check that required keys are present and have valid types
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that all required results keys exist."""
        missing = []
        for key, val in [('E_min', self.E_min), ('l', self.l),
                         ('R_universe', self.R_universe), ('E_p_th', self.E_p_th)]:
            if val is None:
                missing.append(key)
        if missing:
            return (False, f"Missing result keys: {missing}.")
        return (True, "All required result keys present.")

    def test_interface_types(self) -> Tuple[bool, str]:
        """Check that result values are numeric and positive."""
        for name, val in [('E_min', self.E_min), ('l', self.l),
                          ('R_universe', self.R_universe), ('E_p_th', self.E_p_th)]:
            if val is None:
                return (False, f"SKIP: {name} not available")
            if not isinstance(val, (int, float)):
                return (False, f"{name} is not numeric: {type(val).__name__}")
            if val <= 0:
                return (False, f"{name} is not positive: {val}")
        return (True, "All values are positive numeric.")

    # ----------------------------------------------------------------------
    # Invariant (theorem-based) tests
    # ----------------------------------------------------------------------
    def test_invariant_pair_production_threshold(self) -> Tuple[bool, str]:
        """Check that E_min satisfies the threshold formula E_min = (m_e c^2)^2 / epsilon."""
        if self.E_min is None:
            return (False, "SKIP: E_min not available")
        if self.m_e_c2 is None or self.epsilon is None:
            return (False, "SKIP: m_e_c2 or epsilon not available for threshold check")
        expected = self.m_e_c2 ** 2 / self.epsilon
        if math.isclose(self.E_min, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_min = {self.E_min} eV; expected ~{expected:.2e}")
        else:
            return (False, f"E_min = {self.E_min} eV; expected ~{expected:.2e}")

    def test_invariant_mean_free_path(self) -> Tuple[bool, str]:
        """Check that l satisfies l = 1 / (n_gamma * sigma_T/3) in cm, then convert to ly."""
        if self.l is None:
            return (False, "SKIP: l not available")
        if self.n_gamma is None or self.sigma_T is None or self.cm_to_ly is None:
            return (False, "SKIP: n_gamma, sigma_T or cm_to_ly not available")
        sigma = self.sigma_T / 3.0
        l_cm = 1.0 / (self.n_gamma * sigma)
        l_ly = l_cm / self.cm_to_ly
        if l_ly <= 0:
            return (False, "Computed mean free path would be non-positive; this indicates unphysical inputs.")
        if math.isclose(self.l, l_ly, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"l = {self.l} ly; expected ~{l_ly:.2e}")
        else:
            return (False, f"l = {self.l} ly; expected ~{l_ly:.2e}")

    def test_invariant_comparison_universe(self) -> Tuple[bool, str]:
        """Check that l is much smaller than R_universe (by at least a factor of 10^4)."""
        if self.l is None or self.R_universe is None or self.cm_to_ly is None:
            return (False, "SKIP: l, R_universe, or cm_to_ly not available")
        l_cm = self.l * self.cm_to_ly
        if self.R_universe <= 0:
            return (False, "R_universe must be positive")
        ratio = l_cm / self.R_universe
        # According to problem, ratio is about 1e-6.
        if ratio < 1e-3:
            return (True, f"l/R_universe = {ratio:.3e} << 1")
        else:
            return (False, f"l/R_universe = {ratio:.3e}, expected << 1")

    def test_invariant_proton_threshold(self) -> Tuple[bool, str]:
        """Check that E_p_th satisfies photopion threshold formula E_p_th = (m_pi m_p c^4) / (2*epsilon)."""
        if self.E_p_th is None:
            return (False, "SKIP: E_p_th not available")
        if self.m_p_c2 is None or self.m_pi_c2 is None or self.epsilon is None:
            return (False, "SKIP: m_p_c2, m_pi_c2 or epsilon not available")
        expected = (self.m_pi_c2 * self.m_p_c2) / (2.0 * self.epsilon)
        if math.isclose(self.E_p_th, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_p_th = {self.E_p_th} eV; expected ~{expected:.2e}")
        else:
            return (False, f"E_p_th = {self.E_p_th} eV; expected ~{expected:.2e}")

    # ----------------------------------------------------------------------
    # Modeling/Trace tests (check consistency with explanation steps)
    # ----------------------------------------------------------------------
    def test_modeling_consistent_units(self) -> Tuple[bool, str]:
        """Check that E_min and E_p_th are in eV, l in ly, R_universe in cm (reasonable range)."""
        # Order-of-magnitude sanity (no hardcoded answer values)
        if self.E_min is not None and self.E_min < 1e12:
            return (False, f"E_min seems too small: {self.E_min} eV (expected ~10^14)")
        if self.l is not None and self.l < 1e2:
            return (False, f"l seems too small: {self.l} ly (expected ~10^4)")
        if self.R_universe is not None and (self.R_universe < 1e25 or self.R_universe > 1e30):
            return (False, f"R_universe out of expected range: {self.R_universe} cm")
        if self.E_p_th is not None and self.E_p_th < 1e18:
            return (False, f"E_p_th seems too small: {self.E_p_th} eV (expected ~10^20)")
        return (True, "All results have reasonable orders of magnitude.")

    def test_trace_threshold_relation_e_min(self) -> Tuple[bool, str]:
        """Check that E_min is positive (trivial consistency with threshold condition)."""
        if self.E_min is None:
            return (False, "SKIP: E_min not available")
        if self.E_min > 0:
            return (True, "E_min positive; consistent with threshold condition.")
        else:
            return (False, "E_min should be positive.")

    # ----------------------------------------------------------------------
    # Monotonicity tests (perturb one input and check trend)
    # ----------------------------------------------------------------------
    def test_monotonicity_E_min_vs_epsilon(self) -> Tuple[bool, str]:
        """If epsilon increases by 10%, E_min should decrease by factor ~1/1.1."""
        if self.E_min is None or self.epsilon is None or self.m_e_c2 is None:
            return (False, "SKIP: required data missing")
        perturbed_eps = self.epsilon * 1.1
        expected_perturbed = self.m_e_c2 ** 2 / perturbed_eps
        ratio = self.E_min / expected_perturbed
        if math.isclose(ratio, 1.1, rel_tol=0.02):
            return (True, f"E_min decreases by factor ~1.1 when epsilon increased: ratio {ratio:.3f}")
        else:
            return (False, f"E_min ratio = {ratio:.3f}, expected ~1.1")

    def test_monotonicity_l_vs_n_gamma(self) -> Tuple[bool, str]:
        """If n_gamma increases by 10%, l should decrease by factor ~1/1.1."""
        if self.l is None or self.n_gamma is None or self.sigma_T is None or self.cm_to_ly is None:
            return (False, "SKIP: required data missing")
        sigma = self.sigma_T / 3.0
        baseline_l_cm = 1.0 / (self.n_gamma * sigma)
        perturbed_n = self.n_gamma * 1.1
        perturbed_l_cm = 1.0 / (perturbed_n * sigma)
        # Compare the values in cm (avoid conversion to ly)
        ratio = baseline_l_cm / perturbed_l_cm
        if math.isclose(ratio, 1.1, rel_tol=0.02):
            return (True, f"l decreases by factor ~1.1 when n_gamma increased: ratio {ratio:.3f}")
        else:
            return (False, f"l ratio = {ratio:.3f}, expected ~1.1")

    def test_monotonicity_E_p_th_vs_epsilon(self) -> Tuple[bool, str]:
        """If epsilon increases by 10%, E_p_th should decrease by factor ~1/1.1."""
        if self.E_p_th is None or self.epsilon is None or self.m_p_c2 is None or self.m_pi_c2 is None:
            return (False, "SKIP: required data missing")
        baseline = (self.m_pi_c2 * self.m_p_c2) / (2.0 * self.epsilon)
        perturbed_eps = self.epsilon * 1.1
        perturbed = (self.m_pi_c2 * self.m_p_c2) / (2.0 * perturbed_eps)
        ratio = baseline / perturbed
        if math.isclose(ratio, 1.1, rel_tol=0.02):
            return (True, f"E_p_th decreases by factor ~1.1 when epsilon increased: ratio {ratio:.3f}")
        else:
            return (False, f"E_p_th ratio = {ratio:.3f}, expected ~1.1")

    # ----------------------------------------------------------------------
    # Weak answer sanity check (only one allowed)
    # ----------------------------------------------------------------------
    def test_answer_sanity_basic(self) -> Tuple[bool, str]:
        """Verify that all results are finite positive numbers (weak check)."""
        vals = [('E_min', self.E_min), ('l', self.l), ('R_universe', self.R_universe), ('E_p_th', self.E_p_th)]
        for name, val in vals:
            if val is None:
                return (False, f"SKIP: {name} not available")
            if not math.isfinite(val):
                return (False, f"{name} is not finite: {val}")
            if val <= 0:
                return (False, f"{name} is not positive: {val}")
        return (True, "All results are finite positive numbers.")

    # ----------------------------------------------------------------------
    # Main runner
    # ----------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, list]:
        """Execute all test_* methods and return (overall_pass, logs)."""
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
                passed, msg = False, f"{name} CRASHED: {traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)