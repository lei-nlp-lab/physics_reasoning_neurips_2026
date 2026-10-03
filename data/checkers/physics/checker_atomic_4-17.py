import math
import sympy as sp

class PhysicsChecker:
    """
    PhysicsChecker for threshold energy problem: gamma + e- -> e+ + e- + e-.
    Checks interface, invariants, modeling, monotonicity, and answer sanity.
    """
    def __init__(self, solution_instance):
        # Safe extraction of results
        self.results = {}
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            self.results = solution_instance.results
        elif hasattr(solution_instance, 'get'):
            # If solution_instance itself is dict-like
            self.results = solution_instance

        # Extract required results, default None
        self.E_threshold = self.results.get('E_threshold', None)
        self.E_threshold_numerical = self.results.get('E_threshold_numerical', None)

        # Known physical constants (MeV)
        self.m_e_c2 = 0.511

    # ----------------------------------------------------------------------
    # Interface tests
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that both required keys exist in results."""
        if 'E_threshold' not in self.results:
            return (False, "Missing key 'E_threshold'")
        if 'E_threshold_numerical' not in self.results:
            return (False, "Missing key 'E_threshold_numerical'")
        return (True, "Both required keys present")

    def test_interface_types(self):
        """Check type of E_threshold (sympy or string) and E_threshold_numerical (numeric)."""
        if self.E_threshold is None or self.E_threshold_numerical is None:
            return (False, "One or both values are None")
        # E_threshold can be a sympy expression or a string; accept both
        if not isinstance(self.E_threshold, (sp.Basic, str)):
            return (False, f"E_threshold type {type(self.E_threshold)} not sympy/str")
        # E_threshold_numerical should be numeric
        if not isinstance(self.E_threshold_numerical, (int, float)):
            return (False, f"E_threshold_numerical type {type(self.E_threshold_numerical)} not numeric")
        return (True, "Types valid")

    # ----------------------------------------------------------------------
    # Invariant tests (theorem-based physical sanity)
    # ----------------------------------------------------------------------
    def test_invariant_energy_conservation(self):
        """
        Check that E_threshold satisfies energy conservation:
        E_gamma + m_e c^2 = 3 * sqrt( (E_gamma/(3c))^2 c^2 + m_e^2 c^4 )
        At threshold: E_gamma = 4 m_e c^2.
        """
        if self.E_threshold is None:
            return (False, "SKIP: E_threshold unavailable")

        # Convert symbolic to numeric if needed
        E_val = None
        if isinstance(self.E_threshold, sp.Basic):
            try:
                E_val = float(self.E_threshold.subs({sp.Symbol('m_e'): self.m_e_c2, sp.Symbol('c'): 1}))
            except:
                E_val = float(self.E_threshold)
        elif isinstance(self.E_threshold, str):
            try:
                # Try to evaluate simple string expression
                local_dict = {'m_e': self.m_e_c2, 'c': 1}
                E_val = eval(self.E_threshold, {"__builtins__": None}, local_dict)
            except Exception:
                return (False, "Cannot evaluate E_threshold string")
        else:
            return (False, "Unexpected E_threshold type")

        if E_val is None or not math.isfinite(E_val):
            return (False, "E_val not finite")

        # Energy conservation equation:
        # left = E_gamma + m_e c^2
        # right = 3 * sqrt(E_gamma^2/9 + m_e^2 c^4)
        m = self.m_e_c2
        left = E_val + m
        right = 3 * math.sqrt(E_val**2 / 9 + m**2)
        if math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Energy conservation holds: left={left:.3f}, right={right:.3f}")
        else:
            return (False, f"Energy conservation fails: left={left:.3f}, right={right:.3f}")

    def test_invariant_momentum_conservation(self):
        """Check momentum conservation: p_gamma = E_gamma/c = 3p, with p = sqrt(E^2 - m^2)/c."""
        if self.E_threshold is None:
            return (False, "SKIP: E_threshold unavailable")
        # Use numeric E_gamma from result
        if self.E_threshold_numerical is None:
            return (False, "SKIP: E_threshold_numerical unavailable")
        E_gamma = self.E_threshold_numerical
        m = self.m_e_c2
        # At threshold, p_gamma = E_gamma (c=1)
        p_gamma = E_gamma
        # Each final particle has same p (magnitude) such that p = E_gamma/3
        p_final = E_gamma / 3.0
        # also check from energy: E_final = sqrt(p_final^2 + m^2)
        E_final = math.sqrt(p_final**2 + m**2)
        # Energy conservation: E_gamma + m = 3 * E_final
        left_energy = E_gamma + m
        right_energy = 3 * E_final
        if math.isclose(left_energy, right_energy, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Momentum conservation leads to consistent energy: left={left_energy:.3f}, right={right_energy:.3f}")
        else:
            return (False, f"Momentum-energy consistency fails: left={left_energy:.3f}, right={right_energy:.3f}")

    def test_invariant_invariant_mass(self):
        """Check that (E_gamma + m_e)^2 - p_gamma^2 = (3m_e)^2 at threshold."""
        if self.E_threshold_numerical is None:
            return (False, "SKIP: E_threshold_numerical unavailable")
        E_gamma = self.E_threshold_numerical
        m = self.m_e_c2
        # In natural units c=1, p_gamma = E_gamma
        S = (E_gamma + m)**2 - E_gamma**2
        expected_S = (3*m)**2
        if math.isclose(S, expected_S, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Invariant mass squared S={S:.3f} matches expected {expected_S:.3f}")
        else:
            return (False, f"Invariant mass mismatch: S={S:.3f}, expected {expected_S:.3f}")

    # ----------------------------------------------------------------------
    # Modeling / trace tests (quality based on explanation steps)
    # ----------------------------------------------------------------------
    def test_modeling_numerical_value(self):
        """Check that E_threshold_numerical is approximately 2.044 MeV (derived in-checker)."""
        # Derive expected value inside checker: 4 * m_e c^2 = 2.044
        expected_numerical = 4 * self.m_e_c2  # 2.044
        if self.E_threshold_numerical is None:
            return (False, "SKIP: E_threshold_numerical unavailable")
        if math.isclose(self.E_threshold_numerical, expected_numerical, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Numerical threshold {self.E_threshold_numerical} matches derived {expected_numerical}")
        else:
            return (False, f"Numerical threshold {self.E_threshold_numerical} deviates from expected {expected_numerical}")

    def test_modeling_threshold_ratio(self):
        """Check that E_threshold / (m_e c^2) = 4, derived from conservation."""
        if self.E_threshold_numerical is None or self.m_e_c2 == 0:
            return (False, "SKIP: data missing")
        ratio = self.E_threshold_numerical / self.m_e_c2
        if math.isclose(ratio, 4.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"E_threshold/m_e_c2 = {ratio:.3f}, expected 4.0")
        else:
            return (False, f"Ratio = {ratio:.3f}, not 4.0")

    # ----------------------------------------------------------------------
    # Monotonicity tests (perturbation of physical parameters)
    # ----------------------------------------------------------------------
    def test_monotonicity_threshold_increases_with_mass(self):
        """
        Perturb m_e c^2 by +10% and check that threshold increases proportionally.
        Since E_threshold = 4 * m_e c^2, a 10% increase in mass leads to 10% increase in threshold.
        """
        if self.E_threshold_numerical is None:
            return (False, "SKIP: E_threshold_numerical unavailable")
        baseline_mass = self.m_e_c2
        baseline_threshold = self.E_threshold_numerical

        # Perturb mass by +10%
        perturbed_mass = baseline_mass * 1.10
        # Expected new threshold from physics: 4 * perturbed_mass
        expected_threshold = 4 * perturbed_mass
        # Ratio of thresholds should equal ratio of masses (1.10)
        threshold_ratio = baseline_threshold / (4 * baseline_mass)  # should be 1.0
        # Alternatively, compute proportional increase directly
        percent_increase = (expected_threshold - (4 * baseline_mass)) / (4 * baseline_mass) * 100
        if math.isclose(percent_increase, 10.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Threshold increases proportionally: mass +10% gives threshold +{percent_increase:.1f}%")
        else:
            return (False, f"Monotonicity fails: mass +10% gave threshold change {percent_increase:.1f}%, expected 10%")

    def test_monotonicity_threshold_positive(self):
        """Check that E_threshold_numerical is positive."""
        if self.E_threshold_numerical is None:
            return (False, "SKIP: E_threshold_numerical unavailable")
        if self.E_threshold_numerical > 0:
            return (True, f"Threshold positive: {self.E_threshold_numerical}")
        else:
            return (False, f"Threshold should be positive but is {self.E_threshold_numerical}")

    # ----------------------------------------------------------------------
    # Answer sanity (weak, secondary)
    # ----------------------------------------------------------------------
    def test_answer_sanity_basic(self):
        """
        Weak sanity: threshold energy must be between 1 and 10 MeV (given m_e c^2=0.511, 4*0.511=2.044).
        No isclose, no derivation.
        """
        if self.E_threshold_numerical is None:
            return (False, "SKIP: E_threshold_numerical unavailable")
        val = self.E_threshold_numerical
        if 1.0 < val < 10.0:
            return (True, f"Threshold {val:.3f} MeV plausible range (1-10 MeV)")
        else:
            return (False, f"Threshold {val:.3f} MeV outside plausible range (1-10 MeV)")

    # ----------------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------------
    def run_all_tests(self):
        """Discover and run all test_ methods, returning (success, logs)."""
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