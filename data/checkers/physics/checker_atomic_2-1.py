import math
import traceback

class PhysicsChecker:
    """
    A robust auditor for the 'Arguments Against Electrons in the Nucleus' problem.
    Checks interface, physical invariants, and modeling consistency.
    """
    def __init__(self, solution_instance):
        # Extract results safely – defaults to None
        results = getattr(solution_instance, 'results', {})
        if not isinstance(results, dict):
            results = {}
        self.E_kin_MeV = results.get('E_kin_MeV', None)
        self.spin_expected = results.get('spin_expected', None)
        self.spin_observed = results.get('spin_observed', None)
        self.ratio = results.get('ratio', None)
        self.tau = results.get('tau', None)

        # Optional: extract givens for recalculations
        self.givens = getattr(solution_instance, 'givens', None)
        if self.givens is None:
            self.givens = {}

    # ------------------------------------------------------------------
    # Interface tests: required keys present and correct types
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """All five emit keys must exist."""
        if None in [self.E_kin_MeV, self.spin_expected, self.spin_observed, self.ratio, self.tau]:
            missing = []
            if self.E_kin_MeV is None: missing.append('E_kin_MeV')
            if self.spin_expected is None: missing.append('spin_expected')
            if self.spin_observed is None: missing.append('spin_observed')
            if self.ratio is None: missing.append('ratio')
            if self.tau is None: missing.append('tau')
            return (False, f"Missing result keys: {missing}")
        return (True, "All required keys present")

    def test_interface_spin_expected_type(self):
        """spin_expected must be the string 'half-integer'."""
        if self.spin_expected is None:
            return (False, "SKIP: spin_expected not available")
        if not isinstance(self.spin_expected, str):
            return (False, "spin_expected must be a string")
        if self.spin_expected != 'half-integer':
            return (False, f"Expected 'half-integer', got '{self.spin_expected}'")
        return (True, "spin_expected is 'half-integer'")

    def test_interface_spin_observed_type(self):
        """spin_observed must be the string 'integer'."""
        if self.spin_observed is None:
            return (False, "SKIP: spin_observed not available")
        if not isinstance(self.spin_observed, str):
            return (False, "spin_observed must be a string")
        if self.spin_observed != 'integer':
            return (False, f"Expected 'integer', got '{self.spin_observed}'")
        return (True, "spin_observed is 'integer'")

    def test_interface_ratio_type(self):
        """ratio must be numeric."""
        if self.ratio is None:
            return (False, "SKIP: ratio not available")
        if not isinstance(self.ratio, (int, float)):
            return (False, f"ratio must be numeric, got {type(self.ratio)}")
        return (True, f"ratio is numeric: {self.ratio}")

    def test_interface_tau_type(self):
        """tau must be numeric."""
        if self.tau is None:
            return (False, "SKIP: tau not available")
        if not isinstance(self.tau, (int, float)):
            return (False, f"tau must be numeric, got {type(self.tau)}")
        return (True, f"tau is numeric: {self.tau}")

    # ------------------------------------------------------------------
    # Invariant tests: theorem-based physical sanity (use isclose only)
    # ------------------------------------------------------------------
    def test_invariant_E_kin_MeV_positive(self):
        """Kinetic energy must be positive and at least of order 1 MeV."""
        if self.E_kin_MeV is None:
            return (False, "SKIP: E_kin_MeV not available")
        if self.E_kin_MeV <= 0:
            return (False, f"E_kin_MeV must be positive, got {self.E_kin_MeV}")
        # The uncertainty principle gives ~9.5 MeV, but any positive value is plausible
        return (True, f"E_kin_MeV = {self.E_kin_MeV} > 0")

    def test_invariant_E_kin_MeV_exceeds_binding(self):
        """Kinetic energy should be larger than typical nuclear binding (~8 MeV)."""
        if self.E_kin_MeV is None:
            return (False, "SKIP: E_kin_MeV not available")
        if self.E_kin_MeV < 5:  # allow some margin
            return (False, f"E_kin_MeV = {self.E_kin_MeV} < 5 MeV, unlikely for uncertainty argument")
        return (True, f"E_kin_MeV = {self.E_kin_MeV} >= 5 MeV, consistent")

    def test_invariant_spin_contradiction(self):
        """Expected and observed spins must differ (contradiction)."""
        if self.spin_expected is None or self.spin_observed is None:
            return (False, "SKIP: spin values not available")
        if self.spin_expected == self.spin_observed:
            return (False, f"No contradiction: both are '{self.spin_expected}'")
        return (True, f"Spin contradiction: expected '{self.spin_expected}' vs observed '{self.spin_observed}'")

    def test_invariant_ratio_order(self):
        """Ratio mu_B/mu_N should be of order 1000 (1836)."""
        if self.ratio is None:
            return (False, "SKIP: ratio not available")
        if not isinstance(self.ratio, (int, float)):
            return (False, f"ratio not numeric: {self.ratio}")
        if not (100 < self.ratio < 10000):
            return (False, f"ratio = {self.ratio} not in expected range (100,10000)")
        return (True, f"ratio = {self.ratio} is in order-of-magnitude range")

    def test_invariant_tau_extremely_short(self):
        """Lifetime must be extremely short (<< 1 second)."""
        if self.tau is None:
            return (False, "SKIP: tau not available")
        if not isinstance(self.tau, (int, float)):
            return (False, f"tau not numeric: {self.tau}")
        if self.tau <= 0:
            return (False, f"tau must be positive, got {self.tau}")
        if self.tau > 1e-10:
            return (False, f"tau = {self.tau} far too large for radiative collapse")
        return (True, f"tau = {self.tau} << 1 s, consistent")

    # ------------------------------------------------------------------
    # Modeling / trace tests: check consistency with explanation steps
    # ------------------------------------------------------------------
    def test_modeling_spin_argument_consistency(self):
        """The spin argument: for N-14, expected half-integer, observed integer."""
        if self.spin_expected is None or self.spin_observed is None:
            return (False, "SKIP: spin values not available")
        # The explanation explicitly says N-14 yields half-integer expected, integer observed.
        # Basic check: they are opposite
        if self.spin_expected == 'half-integer' and self.spin_observed == 'integer':
            return (True, "Spin contradiction matches N-14 example")
        else:
            return (False, f"Spin values ({self.spin_expected}, {self.spin_observed}) do not match the N-14 argument")

    def test_modeling_ratio_magnitude(self):
        """The ratio mu_B/mu_N should be about 1836 (mass ratio)."""
        if self.ratio is None:
            return (False, "SKIP: ratio not available")
        # Recompute from fundamental constants if we have them
        # Otherwise, just check order of magnitude
        # Fundamental constants (SI)
        # We'll use a cheap approximate: m_p/m_e * (g-factor ratio) ~ 1836 * 1
        # Actually mu_B = e*hbar/(2*m_e), mu_N = e*hbar/(2*m_p) => ratio = m_p/m_e ≈ 1836
        # Since constants are available in Python, we can compute exactly
        try:
            hbar = 1.054571817646e-34  # J s
            e = 1.602176634e-19        # C
            m_e = 9.1093837015e-31     # kg
            m_p = 1.67262192369e-27    # kg
            mu_B = e * hbar / (2.0 * m_e)
            mu_N = e * hbar / (2.0 * m_p)
            expected_ratio = mu_B / mu_N  # should be m_p/m_e ≈ 1836.152673
            if math.isclose(self.ratio, expected_ratio, rel_tol=0.1):
                return (True, f"ratio = {self.ratio} matches m_p/m_e ≈ {expected_ratio:.2f}")
            else:
                # fallback: still check order of magnitude
                if 1500 < self.ratio < 2500:
                    return (True, f"ratio = {self.ratio} within reasonable range (1500-2500)")
                else:
                    return (False, f"ratio = {self.ratio} outside expected range (1500-2500)")
        except Exception:
            # If constants not available, just do order-of-magnitude
            if 1000 < self.ratio < 3000:
                return (True, f"ratio = {self.ratio} in order-of-magnitude range (1000-3000)")
            else:
                return (False, f"ratio = {self.ratio} unexpected, should be ~1800")

    def test_modeling_kinetic_energy_from_uncertainty(self):
        """If R is given in solution_instance.givens, recompute E_kin and compare."""
        # Check if we have R in givens
        R = self.givens.get('R', None)
        if R is None:
            return (False, "SKIP: no radius 'R' in givens to recompute")
        if self.E_kin_MeV is None:
            return (False, "SKIP: E_kin_MeV not available for comparison")
        # Attempt to recompute using uncertainty principle
        try:
            hbar = 1.054571817646e-34
            m_e = 9.1093837015e-31
            delta_p = hbar / (2.0 * R)          # minimum uncertainty
            E_kin_J = (delta_p**2) / (2.0 * m_e)
            E_kin_MeV_computed = E_kin_J / 1.602176634e-13  # convert J to MeV
            if math.isclose(self.E_kin_MeV, E_kin_MeV_computed, rel_tol=0.5):  # allow 50% uncertainty
                return (True, f"Computed E_kin = {E_kin_MeV_computed:.2f} MeV matches provided {self.E_kin_MeV:.2f} MeV")
            else:
                return (False, f"Provided E_kin = {self.E_kin_MeV} MeV, but uncertainty principle gives {E_kin_MeV_computed:.2f} MeV")
        except Exception as e:
            return (False, f"Recalculation crashed: {e}")

    # ------------------------------------------------------------------
    # Monotonicity tests (only if R provided in givens)
    # ------------------------------------------------------------------
    def test_monotonicity_kinetic_energy_with_radius(self):
        """If radius decreases, kinetic energy should increase (inverse square)."""
        R = self.givens.get('R', None)
        if R is None or self.E_kin_MeV is None:
            return (False, "SKIP: need radius and E_kin for monotonicity")
        try:
            # Perturb R by +5%
            R_perturbed = R * 1.05
            hbar = 1.054571817646e-34
            m_e = 9.1093837015e-31
            delta_p_pert = hbar / (2.0 * R_perturbed)
            E_kin_J_pert = (delta_p_pert**2) / (2.0 * m_e)
            E_kin_pert_MeV = E_kin_J_pert / 1.602176634e-13
            # Baseline from provided value
            baseline = self.E_kin_MeV
            if E_kin_pert_MeV < baseline * 0.9:  # should be smaller (since R increased)
                return (True, f"Monotonicity holds: R increased 5%, E_kin decreased from {baseline:.2f} to {E_kin_pert_MeV:.2f}")
            else:
                return (False, f"Monotonicity fails: R increased 5%, E_kin went from {baseline:.2f} to {E_kin_pert_MeV:.2f}")
        except Exception as e:
            return (False, f"Monotonicity test crashed: {e}")

    # ------------------------------------------------------------------
    # Weak answer sanity (one test max, no re-derivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_positive_finite(self):
        """All numeric results must be positive and finite (no NaN/Inf)."""
        for name, val in [('E_kin_MeV', self.E_kin_MeV),
                          ('ratio', self.ratio),
                          ('tau', self.tau)]:
            if val is None:
                continue
            if not isinstance(val, (int, float)):
                continue
            if val <= 0:
                return (False, f"Answer sanity: {name} = {val} is not positive")
            if math.isinf(val) or math.isnan(val):
                return (False, f"Answer sanity: {name} is inf or nan")
        # Check spin strings are sensible
        if self.spin_expected and self.spin_expected not in ['half-integer', 'integer']:
            return (False, f"Unknown spin_expected: {self.spin_expected}")
        if self.spin_observed and self.spin_observed not in ['half-integer', 'integer']:
            return (False, f"Unknown spin_observed: {self.spin_observed}")
        return (True, "All provided results are positive, finite, and sensible")

    # ------------------------------------------------------------------
    # Runner: discover and execute all test methods
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)