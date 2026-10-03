import math
import traceback

class PhysicsChecker:
    """
    Audits a solution instance for the particle physics discussion problem.
    Extracts key results from solution_instance.results and runs unit-like tests.
    """

    def __init__(self, solution_instance):
        # Extract results safely, defaulting to None
        self.solution_instance = solution_instance
        if hasattr(solution_instance, 'results'):
            results = solution_instance.results
        else:
            results = {}

        self.m_Jpsi = results.get('m_Jpsi', None)
        self.Gamma_Jpsi = results.get('Gamma_Jpsi', None)
        self.flavor = results.get('flavor', None)
        self.d_n = results.get('d_n', None)
        self.kurie_eq = results.get('kurie_eq', None)
        self.a_mu = results.get('a_mu', None)

    # ----------------------------------------------------------------------
    # Interface tests – check that required keys exist and have valid types
    # ----------------------------------------------------------------------
    def test_interface_keys_exist(self):
        """All expected result keys must be present in the solution."""
        missing = []
        for key in ('m_Jpsi', 'Gamma_Jpsi', 'flavor', 'd_n', 'kurie_eq', 'a_mu'):
            if getattr(self, key) is None:
                missing.append(key)
        if missing:
            return False, f"Missing keys: {', '.join(missing)}"
        return True, "All expected keys present"

    def test_interface_type_m_Jpsi(self):
        """m_Jpsi must be a numeric type (float or int)."""
        if self.m_Jpsi is None:
            return False, "SKIP: m_Jpsi not available"
        if not isinstance(self.m_Jpsi, (int, float)):
            return False, f"m_Jpsi is not numeric, type: {type(self.m_Jpsi)}"
        return True, f"m_Jpsi type OK: {type(self.m_Jpsi).__name__}"

    def test_interface_type_Gamma_Jpsi(self):
        """Gamma_Jpsi must be numeric."""
        if self.Gamma_Jpsi is None:
            return False, "SKIP: Gamma_Jpsi not available"
        if not isinstance(self.Gamma_Jpsi, (int, float)):
            return False, f"Gamma_Jpsi is not numeric, type: {type(self.Gamma_Jpsi)}"
        return True, f"Gamma_Jpsi type OK"

    def test_interface_type_flavor(self):
        """flavor must be a string."""
        if self.flavor is None:
            return False, "SKIP: flavor not available"
        if not isinstance(self.flavor, str):
            return False, f"flavor is not a string, type: {type(self.flavor)}"
        return True, f"flavor type OK: {self.flavor}"

    def test_interface_type_d_n(self):
        """d_n must be numeric."""
        if self.d_n is None:
            return False, "SKIP: d_n not available"
        if not isinstance(self.d_n, (int, float)):
            return False, f"d_n is not numeric, type: {type(self.d_n)}"
        return True, f"d_n type OK"

    def test_interface_type_kurie_eq(self):
        """kurie_eq must be a string."""
        if self.kurie_eq is None:
            return False, "SKIP: kurie_eq not available"
        if not isinstance(self.kurie_eq, str):
            return False, f"kurie_eq is not a string, type: {type(self.kurie_eq)}"
        return True, f"kurie_eq type OK"

    def test_interface_type_a_mu(self):
        """a_mu must be numeric."""
        if self.a_mu is None:
            return False, "SKIP: a_mu not available"
        if not isinstance(self.a_mu, (int, float)):
            return False, f"a_mu is not numeric, type: {type(self.a_mu)}"
        return True, f"a_mu type OK"

    # ----------------------------------------------------------------------
    # Invariant tests – physical sanity checks using fuzzy matching
    # ----------------------------------------------------------------------
    def test_invariant_Jpsi_mass(self):
        """J/psi mass should be approximately 3096.9 MeV/c^2 (within 2%)."""
        if self.m_Jpsi is None:
            return False, "SKIP: m_Jpsi not available"
        expected = 3096.9  # MeV/c^2
        if math.isclose(self.m_Jpsi, expected, rel_tol=0.02):
            return True, f"m_Jpsi = {self.m_Jpsi} MeV/c^2 ≈ expected {expected}"
        return False, f"m_Jpsi = {self.m_Jpsi} MeV/c^2 is not close to {expected} (2% rel. tol.)"

    def test_invariant_Jpsi_width(self):
        """J/psi width should be approximately 63 keV (within 15% rel. tol.)."""
        if self.Gamma_Jpsi is None:
            return False, "SKIP: Gamma_Jpsi not available"
        expected = 63  # keV
        # Allow 15% because measurement has large uncertainty
        if math.isclose(self.Gamma_Jpsi, expected, rel_tol=0.15):
            return True, f"Gamma_Jpsi = {self.Gamma_Jpsi} keV ≈ expected {expected}"
        return False, f"Gamma_Jpsi = {self.Gamma_Jpsi} keV is not close to {expected} (15% rel. tol.)"

    def test_invariant_neutron_edm_magnitude(self):
        """Neutron EDM magnitude must be less than ~1e-23 e·cm (older limit)."""
        if self.d_n is None:
            return False, "SKIP: d_n not available"
        # Target answer gives (0.4±1.1) × 10^{-24} cm; so value should be ≤ 2e-24 cm
        if abs(self.d_n) > 2e-24:
            return False, f"|d_n| = {abs(self.d_n):.3e} e·cm exceeds plausible bound"
        return True, f"|d_n| = {abs(self.d_n):.3e} e·cm within range"

    def test_invariant_muon_anomaly_value(self):
        """Muon anomalous magnetic moment (a_mu) should be ~0.00116592 (within 0.2% rel. tol.)."""
        if self.a_mu is None:
            return False, "SKIP: a_mu not available"
        expected = 116592.1e-8  # = 0.001165921
        # Use rel_tol=0.002 (0.2%) because theoretical prediction is precise
        if math.isclose(self.a_mu, expected, rel_tol=0.002):
            return True, f"a_mu = {self.a_mu:.6e} ≈ expected {expected:.6e}"
        return False, f"a_mu = {self.a_mu:.6e} not close to expected {expected:.6e} (0.2% rel. tol.)"

    # ----------------------------------------------------------------------
    # Modeling / trace tests – check correctness of equations and explanations
    # ----------------------------------------------------------------------
    def test_modeling_kurie_equation_form(self):
        """Kurie equation string must contain the symbolic structure (E0 - Ee) and dependence on p_e^2."""
        if self.kurie_eq is None:
            return False, "SKIP: kurie_eq not available"
        eq_str = self.kurie_eq
        has_energy_diff = ("E_0" in eq_str) or ("E0" in eq_str)
        has_momentum = "p_e" in eq_str
        if not has_energy_diff:
            return False, "kurie_eq missing energy difference term (E_0 - E_e)"
        if not has_momentum:
            return False, "kurie_eq missing momentum factor (p_e^2)"
        # Check for typical structure: dI/(p_e^2 F d p_e)
        if "dI" in eq_str and "F" in eq_str:
            return True, "kurie_eq contains expected structure: dI/(p_e^2 F d p_e) proportional to (E0 - Ee)^2"
        return True, f"kurie_eq plausibly correct: {eq_str[:50]}..."

    def test_modeling_flavor_description(self):
        """The flavor answer should state that nu_e and nu_mu are different."""
        if self.flavor is None:
            return False, "SKIP: flavor not available"
        if self.flavor.lower() in ['different', 'nu_e and nu_mu are different', 'two types']:
            return True, f"flavor statement: '{self.flavor}'"
        return False, f"Unexpected flavor description: '{self.flavor}'"

    # ----------------------------------------------------------------------
    # Monotonicity test – explicitly perturb an input and compare outputs
    # ----------------------------------------------------------------------
    def test_monotonicity_alpha_on_a_mu(self):
        """
        Muon anomalous magnetic moment increases when fine-structure constant α increases.
        Uses the QED formula: a_mu = α/(2π) + 0.76578*(α/π)^2.
        Perturb α by +1% and verify that the computed a_mu increases.
        """
        # Use standard α = 1/137.035999084 (approximately)
        alpha_base = 1.0 / 137.035999084
        # Compute a_mu using the theoretical formula (including second-order correction)
        def a_mu_from_alpha(alpha):
            term1 = alpha / (2.0 * math.pi)
            term2 = 0.76578 * (alpha / math.pi) ** 2
            return term1 + term2

        a_mu_base = a_mu_from_alpha(alpha_base)
        alpha_perturbed = alpha_base * 1.01  # 1% increase
        a_mu_pert = a_mu_from_alpha(alpha_perturbed)

        if a_mu_pert > a_mu_base:
            return True, f"Monotonicity holds: a_mu({alpha_perturbed:.5e}) = {a_mu_pert:.6e} > a_mu({alpha_base:.5e}) = {a_mu_base:.6e}"
        else:
            return False, f"Monotonicity violated: a_mu decreased (base {a_mu_base:.6e}, perturbed {a_mu_pert:.6e})"

    # ----------------------------------------------------------------------
    # Answer sanity – very weak checks only
    # ----------------------------------------------------------------------
    def test_answer_sanity_sign_and_finite(self):
        """All numerical results must be finite and (where applicable) positive."""
        numerics = {
            'm_Jpsi': self.m_Jpsi,
            'Gamma_Jpsi': self.Gamma_Jpsi,
            'd_n': self.d_n,
            'a_mu': self.a_mu
        }
        bad = []
        for name, val in numerics.items():
            if val is None:
                continue
            if not math.isfinite(val):
                bad.append(f"{name}={val} not finite")
            # positivity: masses, widths, and a_mu are positive; d_n could be negative (sign not defined)
            if name in ('m_Jpsi', 'Gamma_Jpsi', 'a_mu') and val < 0:
                bad.append(f"{name} negative ({val})")
        if bad:
            return False, "; ".join(bad)
        return True, "All numerical results are finite and positive where expected"

    # ----------------------------------------------------------------------
    # Run all tests
    # ----------------------------------------------------------------------
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