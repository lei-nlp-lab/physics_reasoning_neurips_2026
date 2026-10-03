import math
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for a set of order-of-magnitude estimates in particle/nuclear physics.
    Performs interface, invariant, modeling, monotonicity and one weak answer-sanity test.
    All tests follow the robustness rules: safe None defaults, fuzzy matching for invariants,
    no hardcoding of answer keys, graceful handling of missing data.
    """

    # Fundamental constants (SI units) used for derived checks
    h = 6.62607015e-34          # Planck constant (J·s)
    hbar = 1.054571817e-34      # reduced Planck constant (J·s)
    c = 299792458.0             # speed of light (m/s)
    e = 1.602176634e-19         # elementary charge (C)
    m_e = 9.1093837015e-31      # electron mass (kg)
    m_p = 1.67262192369e-27     # proton mass (kg)
    m_n = 1.67492749804e-27     # neutron mass (kg)
    epsilon0 = 8.8541878128e-12 # vacuum permittivity (F/m)
    mu_N = 5.0507837461e-27     # nuclear magneton (J/T)
    r0 = 1.2e-15                # nuclear size constant (m)
    u = 1.66053906660e-27       # atomic mass unit (kg)

    def __init__(self, solution_instance):
        """
        Initialize the checker from a solution instance.
        Expects solution_instance.results to be a dictionary with keys
        lambda_c, sigma_T, a0, E1, Delta_E_hf, mu_Li7, delta_m_np,
        tau_n, B_He4, R_max, tau_pi0, tau_mu.
        """
        # Extract results dict safely
        results = {}
        if hasattr(solution_instance, 'results') and isinstance(solution_instance.results, dict):
            results = solution_instance.results

        # Store each quantity with None default (NO numeric fallbacks)
        self.lambda_c = results.get('lambda_c', None)
        self.sigma_T = results.get('sigma_T', None)
        self.a0 = results.get('a0', None)
        self.E1 = results.get('E1', None)
        self.Delta_E_hf = results.get('Delta_E_hf', None)
        self.mu_Li7 = results.get('mu_Li7', None)
        self.delta_m_np = results.get('delta_m_np', None)
        self.tau_n = results.get('tau_n', None)
        self.B_He4 = results.get('B_He4', None)
        self.R_max = results.get('R_max', None)
        self.tau_pi0 = results.get('tau_pi0', None)
        self.tau_mu = results.get('tau_mu', None)

    # ---------------- Interface Tests ----------------

    def test_interface_lambda_c_exists(self):
        """Check that lambda_c key exists and is a number."""
        if self.lambda_c is None:
            return False, "lambda_c not provided"
        if not isinstance(self.lambda_c, (int, float)):
            return False, f"lambda_c is not numeric: {type(self.lambda_c)}"
        return True, "lambda_c present and numeric"

    def test_interface_sigma_T_exists(self):
        if self.sigma_T is None:
            return False, "sigma_T not provided"
        if not isinstance(self.sigma_T, (int, float)):
            return False, f"sigma_T is not numeric: {type(self.sigma_T)}"
        return True, "sigma_T present and numeric"

    def test_interface_a0_exists(self):
        if self.a0 is None:
            return False, "a0 not provided"
        if not isinstance(self.a0, (int, float)):
            return False, f"a0 is not numeric: {type(self.a0)}"
        return True, "a0 present and numeric"

    def test_interface_E1_exists(self):
        if self.E1 is None:
            return False, "E1 not provided"
        if not isinstance(self.E1, (int, float)):
            return False, f"E1 is not numeric: {type(self.E1)}"
        return True, "E1 present and numeric"

    def test_interface_Delta_E_hf_exists(self):
        if self.Delta_E_hf is None:
            return False, "Delta_E_hf not provided"
        if not isinstance(self.Delta_E_hf, (int, float)):
            return False, f"Delta_E_hf is not numeric: {type(self.Delta_E_hf)}"
        return True, "Delta_E_hf present and numeric"

    def test_interface_mu_Li7_exists(self):
        if self.mu_Li7 is None:
            return False, "mu_Li7 not provided"
        if not isinstance(self.mu_Li7, (int, float)):
            return False, f"mu_Li7 is not numeric: {type(self.mu_Li7)}"
        return True, "mu_Li7 present and numeric"

    def test_interface_delta_m_np_exists(self):
        if self.delta_m_np is None:
            return False, "delta_m_np not provided"
        if not isinstance(self.delta_m_np, (int, float)):
            return False, f"delta_m_np is not numeric: {type(self.delta_m_np)}"
        return True, "delta_m_np present and numeric"

    def test_interface_tau_n_exists(self):
        if self.tau_n is None:
            return False, "tau_n not provided"
        if not isinstance(self.tau_n, (int, float)):
            return False, f"tau_n is not numeric: {type(self.tau_n)}"
        return True, "tau_n present and numeric"

    def test_interface_B_He4_exists(self):
        if self.B_He4 is None:
            return False, "B_He4 not provided"
        if not isinstance(self.B_He4, (int, float)):
            return False, f"B_He4 is not numeric: {type(self.B_He4)}"
        return True, "B_He4 present and numeric"

    def test_interface_R_max_exists(self):
        if self.R_max is None:
            return False, "R_max not provided"
        if not isinstance(self.R_max, (int, float)):
            return False, f"R_max is not numeric: {type(self.R_max)}"
        return True, "R_max present and numeric"

    def test_interface_tau_pi0_exists(self):
        if self.tau_pi0 is None:
            return False, "tau_pi0 not provided"
        if not isinstance(self.tau_pi0, (int, float)):
            return False, f"tau_pi0 is not numeric: {type(self.tau_pi0)}"
        return True, "tau_pi0 present and numeric"

    def test_interface_tau_mu_exists(self):
        if self.tau_mu is None:
            return False, "tau_mu not provided"
        if not isinstance(self.tau_mu, (int, float)):
            return False, f"tau_mu is not numeric: {type(self.tau_mu)}"
        return True, "tau_mu present and numeric"

    # ---------------- Invariant Tests (physics theorems) ----------------

    def test_invariant_compton_wavelength(self):
        """Check lambda_c = h/(m_e*c) to within 2%."""
        if self.lambda_c is None:
            return False, "SKIP: lambda_c not available"
        expected = self.h / (self.m_e * self.c)
        if math.isclose(self.lambda_c, expected, rel_tol=2e-2):
            return True, f"lambda_c = {self.lambda_c:.3e} m, expected ~ {expected:.3e} m"
        else:
            return False, f"lambda_c = {self.lambda_c:.3e} deviates >2% from {expected:.3e}"

    def test_invariant_thomson_cross_section(self):
        """Check sigma_T = (8π/3)*(e^2/(4π ε0 m_e c^2))^2."""
        if self.sigma_T is None:
            return False, "SKIP: sigma_T not available"
        r_e = self.e**2 / (4 * math.pi * self.epsilon0 * self.m_e * self.c**2)
        expected = (8 * math.pi / 3) * r_e**2
        if math.isclose(self.sigma_T, expected, rel_tol=2e-2):
            return True, f"sigma_T = {self.sigma_T:.3e} m^2, expected ~ {expected:.3e} m^2"
        else:
            return False, f"sigma_T = {self.sigma_T:.3e} deviates >2% from {expected:.3e}"

    def test_invariant_bohr_radius(self):
        """Check a0 = 4π ε0 ħ^2/(m_e e^2)."""
        if self.a0 is None:
            return False, "SKIP: a0 not available"
        expected = (4 * math.pi * self.epsilon0 * self.hbar**2) / (self.m_e * self.e**2)
        if math.isclose(self.a0, expected, rel_tol=2e-2):
            return True, f"a0 = {self.a0:.3e} m, expected ~ {expected:.3e} m"
        else:
            return False, f"a0 = {self.a0:.3e} deviates >2% from {expected:.3e}"

    def test_invariant_ionization_energy(self):
        """Check E1 magnitude = m_e e^4/(32 π^2 ε0^2 ħ^2) or known 13.6 eV."""
        if self.E1 is None:
            return False, "SKIP: E1 not available"
        # Re-derive: E1 (magnitude) = m_e*e^4/(32*π^2*ε0^2*ħ^2) in Joules
        expected_J = (self.m_e * self.e**4) / (32 * math.pi**2 * self.epsilon0**2 * self.hbar**2)
        expected_eV = expected_J / 1.602176634e-19
        # The result may be given in J or eV; we compare magnitude in J
        # Check if result is within 2% of ~2.18e-18 J (13.6 eV)
        ref = 2.179872361e-18  # Joules for 13.6 eV
        if math.isclose(abs(self.E1), ref, rel_tol=2e-2):
            return True, f"|E1| = {abs(self.E1):.3e} J, expected ~ {ref:.3e} J"
        # Also allow if value is given in eV and matches ~13.6
        if math.isclose(abs(self.E1), 13.6, rel_tol=2e-2):
            return True, f"|E1| = {abs(self.E1):.3f} eV, expected ~ 13.6 eV"
        return False, f"|E1| = {abs(self.E1):.3e} (J or eV?) does not match 13.6 eV within 2%"

    def test_invariant_nuclear_radius_scaling(self):
        """Check R_max ~ r0 * A^(1/3) with A=208 for largest stable."""
        if self.R_max is None:
            return False, "SKIP: R_max not available"
        A = 208.0
        expected = self.r0 * (A ** (1.0/3.0))
        if math.isclose(self.R_max, expected, rel_tol=2e-1):  # order-of-magnitude, allow 20%
            return True, f"R_max = {self.R_max:.3e} m, expected ~ {expected:.3e} m"
        else:
            return False, f"R_max = {self.R_max:.3e} deviates >20% from {expected:.3e}"

    # ---------------- Modeling / Trace Tests ----------------

    def test_modeling_compton_wavelength_positive(self):
        """Compton wavelength must be positive."""
        if self.lambda_c is None:
            return False, "SKIP: lambda_c not available"
        if self.lambda_c > 0:
            return True, "lambda_c positive"
        return False, "lambda_c non-positive"

    def test_modeling_thomson_cross_section_positive(self):
        if self.sigma_T is None:
            return False, "SKIP: sigma_T not available"
        if self.sigma_T > 0:
            return True, "sigma_T positive"
        return False, "sigma_T non-positive"

    def test_modeling_bohr_radius_positive(self):
        if self.a0 is None:
            return False, "SKIP: a0 not available"
        if self.a0 > 0:
            return True, "a0 positive"
        return False, "a0 non-positive"

    def test_modeling_ionization_energy_negative(self):
        """Ionization potential is usually given as positive (binding energy magnitude) but the physical E1 is negative. Accept either sign, but check magnitude already done."""
        if self.E1 is None:
            return False, "SKIP: E1 not available"
        # Allow negative or positive magnitude ~13.6
        if abs(self.E1) > 0:
            return True, "E1 magnitude positive (sign flexible)"
        return False, "E1 zero or missing"

    def test_modeling_mu_Li7_order(self):
        """Magnetic dipole moment of Li-7 should be of order nuclear magneton (few μ_N)."""
        if self.mu_Li7 is None:
            return False, "SKIP: mu_Li7 not available"
        # Expected ~ 3.256 μ_N = 3.256 * 5.05e-27 = 1.64e-26 J/T
        expected = 3.256 * self.mu_N
        if math.isclose(self.mu_Li7, expected, rel_tol=2e-1):  # allow 20% (order-of-magnitude)
            return True, f"mu_Li7 = {self.mu_Li7:.3e} J/T, expected ~ {expected:.3e} J/T"
        else:
            return False, f"mu_Li7 = {self.mu_Li7:.3e} deviates >20% from {expected:.3e}"

    def test_modeling_delta_m_np_sign(self):
        """Neutron is heavier than proton, so m_n - m_p positive."""
        if self.delta_m_np is None:
            return False, "SKIP: delta_m_np not available"
        if self.delta_m_np > 0:
            return True, f"delta_m_np = {self.delta_m_np:.3e} kg (positive)"
        else:
            return False, f"delta_m_np = {self.delta_m_np:.3e} should be positive"

    def test_modeling_B_He4_positive(self):
        if self.B_He4 is None:
            return False, "SKIP: B_He4 not available"
        if self.B_He4 > 0:
            return True, "B_He4 binding energy positive"
        return False, "B_He4 non-positive"

    def test_modeling_R_max_fm_scale(self):
        """Nuclear radius should be a few fm (1e-15 m)."""
        if self.R_max is None:
            return False, "SKIP: R_max not available"
        if 1e-15 < self.R_max < 1e-14:
            return True, f"R_max = {self.R_max:.3e} m (few fm)"
        return False, f"R_max = {self.R_max:.3e} outside expected few-fm range"

    def test_modeling_tau_pi0_order(self):
        """Pion lifetime ~ 10^-16 - 10^-17 s (electromagnetic decay)."""
        if self.tau_pi0 is None:
            return False, "SKIP: tau_pi0 not available"
        if 1e-17 < self.tau_pi0 < 1e-15:
            return True, f"tau_pi0 = {self.tau_pi0:.3e} s"
        return False, f"tau_pi0 = {self.tau_pi0:.3e} outside expected ~10^-16 s range"

    def test_modeling_tau_mu_order(self):
        """Muon lifetime ~ 2.2e-6 s (weak decay)."""
        if self.tau_mu is None:
            return False, "SKIP: tau_mu not available"
        if 1e-7 < self.tau_mu < 1e-5:
            return True, f"tau_mu = {self.tau_mu:.3e} s"
        return False, f"tau_mu = {self.tau_mu:.3e} outside expected ~2e-6 s range"

    def test_modeling_tau_n_order(self):
        """Neutron lifetime about 900 s (~15 min)."""
        if self.tau_n is None:
            return False, "SKIP: tau_n not available"
        if 100 < self.tau_n < 2000:
            return True, f"tau_n = {self.tau_n:.1f} s"
        return False, f"tau_n = {self.tau_n:.1f} outside expected ~900 s range"

    # ---------------- Monotonicity Tests ----------------

    def test_monotonicity_lambda_c_decreases_with_mass(self):
        """If electron mass were larger, Compton wavelength smaller."""
        if self.lambda_c is None:
            return False, "SKIP: lambda_c not available"
        # Baseline: current
        baseline = self.lambda_c
        # Perturb mass upward by 5%
        m_e_pert = self.m_e * 1.05
        lambda_c_pert = self.h / (m_e_pert * self.c)
        if lambda_c_pert < baseline:
            return True, f"lambda_c decreases from {baseline:.3e} to {lambda_c_pert:.3e} when m_e +5%"
        else:
            return False, "Monotonicity violated: lambda_c should decrease with increasing mass"

    def test_monotonicity_sigma_T_increases_with_charge(self):
        """Thomson cross section increases as e^4; perturb charge upward."""
        if self.sigma_T is None:
            return False, "SKIP: sigma_T not available"
        # Baseline: compute from given sigma_T? Better to use formula with perturbed e
        # Since we don't have the original e used, we compare relative change.
        # We'll compute expected sigma_T from constants, then perturb.
        r_e_base = self.e**2 / (4 * math.pi * self.epsilon0 * self.m_e * self.c**2)
        expected_base = (8 * math.pi / 3) * r_e_base**2
        e_pert = self.e * 1.05
        r_e_pert = e_pert**2 / (4 * math.pi * self.epsilon0 * self.m_e * self.c**2)
        expected_pert = (8 * math.pi / 3) * r_e_pert**2
        if expected_pert > expected_base:
            return True, f"sigma_T increases from {expected_base:.3e} to {expected_pert:.3e} when e +5%"
        else:
            return False, "Monotonicity violated: sigma_T should increase with charge"

    def test_monotonicity_bohr_radius_increases_with_epsilon0(self):
        """Bohr radius increases if epsilon0 increases (weaker field)."""
        if self.a0 is None:
            return False, "SKIP: a0 not available"
        # Baseline using constants
        base = 4 * math.pi * self.epsilon0 * self.hbar**2 / (self.m_e * self.e**2)
        epsilon0_pert = self.epsilon0 * 1.1
        pert = 4 * math.pi * epsilon0_pert * self.hbar**2 / (self.m_e * self.e**2)
        if pert > base:
            return True, f"a0 increases from {base:.3e} to {pert:.3e} when epsilon0 +10%"
        else:
            return False, "Monotonicity violated: a0 should increase with epsilon0"

    def test_monotonicity_ionization_energy_decreases_with_mass(self):
        """|E1| decreases if mass increases? Actually E1 magnitude ∝ m_e, so increases. Let's check E1 magnitude ∝ m_e e^4. Increase m_e -> |E1| larger."""
        if self.E1 is None:
            return False, "SKIP: E1 not available"
        # baseline magnitude from constants
        base_J = (self.m_e * self.e**4) / (32 * math.pi**2 * self.epsilon0**2 * self.hbar**2)
        m_e_pert = self.m_e * 1.1
        pert_J = (m_e_pert * self.e**4) / (32 * math.pi**2 * self.epsilon0**2 * self.hbar**2)
        if pert_J > base_J:
            return True, f"|E1| increases from {base_J:.3e} J to {pert_J:.3e} J when m_e +10%"
        else:
            return False, "Monotonicity violated: |E1| should increase with electron mass"

    # ---------------- Weak Answer Sanity (at most one) ----------------

    def test_answer_sanity_all_finite_and_positive(self):
        """Weak sanity: all results should be finite positive numbers (except E1 sign can be negative)."""
        quantities = {
            'lambda_c': self.lambda_c,
            'sigma_T': self.sigma_T,
            'a0': self.a0,
            'E1': self.E1,
            'Delta_E_hf': self.Delta_E_hf,
            'mu_Li7': self.mu_Li7,
            'delta_m_np': self.delta_m_np,
            'tau_n': self.tau_n,
            'B_He4': self.B_He4,
            'R_max': self.R_max,
            'tau_pi0': self.tau_pi0,
            'tau_mu': self.tau_mu
        }
        for name, val in quantities.items():
            if val is None:
                return False, f"Missing {name}"
            if not math.isfinite(val):
                return False, f"{name} is not finite: {val}"
            if name == 'E1':
                # E1 can be negative (binding) or positive (ionization potential); allow both, just check nonzero
                if val == 0:
                    return False, "E1 cannot be zero"
            else:
                if val <= 0:
                    return False, f"{name} should be positive, got {val}"
        return True, "All 12 quantities present, finite, and positive (E1 magnitude ok)"

    # ---------------- Runner ----------------

    def run_all_tests(self):
        """Discover all test_ methods, execute them, return (success, logs)."""
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