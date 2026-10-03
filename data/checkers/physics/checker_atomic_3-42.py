import math
import traceback

class PhysicsChecker:
    """
    PhysicsChecker for a neutrino problem set covering:
    - Number of neutrino types, spin
    - Lepton number conservation, missing particle identification
    - Magnetic moment and helicity for massless neutrinos
    - Velocity of a non-relativistic neutrino using equipartition
    """

    def __init__(self, solution_instance):
        # Safely extract results dict; default to empty dict if missing
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}

        # Extract each expected key with None default
        self.num_types = results.get('num_types', None)
        self.spin = results.get('spin', None)
        self.missing_particle = results.get('missing_particle', None)
        self.mu = results.get('mu', None)
        self.neutrino_helicity = results.get('neutrino_helicity', None)
        self.antineutrino_helicity = results.get('antineutrino_helicity', None)
        self.v = results.get('v', None)

        # Physical constants from the problem statement (hardcoded as given)
        self.k_B = 8.617333262e-5   # eV/K
        self.T = 3.0               # K
        self.m_nu = 0.1            # eV/c^2
        self.c = 3.0e8             # m/s

    # ----------------------------------------------------------------------
    # Interface tests: ensure all required keys exist and have expected types
    # ----------------------------------------------------------------------
    def test_interface_num_types_present(self):
        if self.num_types is None:
            return (False, "num_types is missing or None")
        try:
            int(self.num_types)
        except (ValueError, TypeError):
            return (False, f"num_types has invalid type/value: {self.num_types}")
        return (True, "num_types exists and is convertible to integer")

    def test_interface_spin_present(self):
        if self.spin is None:
            return (False, "spin is missing or None")
        if not isinstance(self.spin, str) or '/' not in self.spin:
            return (False, f"spin should be a fraction string like '1/2', got {self.spin}")
        return (True, "spin is present and is a fraction string")

    def test_interface_missing_particle_present(self):
        if self.missing_particle is None:
            return (False, "missing_particle is missing or None")
        if not isinstance(self.missing_particle, str):
            return (False, "missing_particle should be a string")
        return (True, "missing_particle is present and is a string")

    def test_interface_mu_present(self):
        if self.mu is None:
            return (False, "mu (magnetic moment) is missing or None")
        try:
            float(self.mu)
        except (ValueError, TypeError):
            return (False, f"mu cannot be converted to float: {self.mu}")
        return (True, "mu is present and numeric")

    def test_interface_neutrino_helicity_present(self):
        if self.neutrino_helicity is None:
            return (False, "neutrino_helicity is missing or None")
        try:
            int(self.neutrino_helicity) if isinstance(self.neutrino_helicity, str) else float(self.neutrino_helicity)
        except (ValueError, TypeError):
            return (False, f"neutrino_helicity is not numeric: {self.neutrino_helicity}")
        return (True, "neutrino_helicity exists")

    def test_interface_antineutrino_helicity_present(self):
        if self.antineutrino_helicity is None:
            return (False, "antineutrino_helicity is missing or None")
        try:
            int(self.antineutrino_helicity) if isinstance(self.antineutrino_helicity, str) else float(self.antineutrino_helicity)
        except (ValueError, TypeError):
            return (False, f"antineutrino_helicity is not numeric: {self.antineutrino_helicity}")
        return (True, "antineutrino_helicity exists")

    def test_interface_v_present(self):
        if self.v is None:
            return (False, "v (velocity) is missing or None")
        try:
            float(self.v)
        except (ValueError, TypeError):
            return (False, f"v cannot be converted to float: {self.v}")
        return (True, "v is present and numeric")

    # ----------------------------------------------------------------------
    # Invariant tests: physics-based checks (use math.isclose where appropriate)
    # ----------------------------------------------------------------------
    def test_invariant_neutrino_types_count(self):
        if self.num_types is None:
            return (False, "SKIP: num_types not available")
        expected = 3
        if int(self.num_types) != expected:
            return (False, f"num_types = {self.num_types}, expected {expected}")
        return (True, f"num_types correctly = {expected}")

    def test_invariant_neutrino_spin(self):
        if self.spin is None:
            return (False, "SKIP: spin not available")
        if self.spin.strip() != '1/2':
            return (False, f"spin = {self.spin}, expected '1/2'")
        return (True, "spin is 1/2")

    def test_invariant_lepton_number_conservation(self):
        if self.missing_particle is None:
            return (False, "SKIP: missing_particle not available")
        valid = {'nu_e', '\\nu_e', r'\nu_e', 'ν_e', 'electron neutrino', 'electron-neutrino'}
        if self.missing_particle.strip() not in valid:
            return (False, f"missing_particle = {self.missing_particle}, expected something like ν_e")
        return (True, "missing_particle correctly is ν_e (based on problem expectation)")

    def test_invariant_magnetic_moment_zero_for_massless(self):
        if self.mu is None:
            return (False, "SKIP: mu not available")
        mu_val = float(self.mu)
        if mu_val != 0.0:
            return (False, f"mu = {mu_val}, should be exactly 0 for massless neutrino")
        return (True, "mu = 0, consistent with massless neutrino")

    def test_invariant_neutrino_helicity(self):
        if self.neutrino_helicity is None:
            return (False, "SKIP: neutrino_helicity not available")
        helicity = int(self.neutrino_helicity) if isinstance(self.neutrino_helicity, str) else self.neutrino_helicity
        if helicity != -1:
            return (False, f"neutrino_helicity = {helicity}, expected -1")
        return (True, "neutrino helicity = -1 (left-handed)")

    def test_invariant_antineutrino_helicity(self):
        if self.antineutrino_helicity is None:
            return (False, "SKIP: antineutrino_helicity not available")
        helicity = int(self.antineutrino_helicity) if isinstance(self.antineutrino_helicity, str) else self.antineutrino_helicity
        if helicity != +1:
            return (False, f"antineutrino_helicity = {helicity}, expected +1")
        return (True, "antineutrino helicity = +1 (right-handed)")

    def test_invariant_nonrelativistic_regime(self):
        kT = self.k_B * self.T   # eV
        if kT > 0.1 * self.m_nu:
            return (False, f"Regime might be relativistic: kT = {kT:.2e} eV, m = {self.m_nu} eV, ratio {kT/self.m_nu:.2f}")
        return (True, f"Non‑relativistic regime confirmed: kT = {kT:.2e} eV << m_nu = {self.m_nu} eV")

    # ----------------------------------------------------------------------
    # Modeling / trace tests: check derived formulas and reasoning using isclose
    # ----------------------------------------------------------------------
    def test_modeling_velocity_equipartition(self):
        if self.v is None:
            return (False, "SKIP: v not available")
        v_expected = math.sqrt(3 * self.k_B * self.T / self.m_nu) * self.c
        v_reported = float(self.v)
        if not math.isclose(v_reported, v_expected, rel_tol=2e-2, abs_tol=1.0):
            return (False, f"v = {v_reported:.3e}, expected ≈ {v_expected:.3e} (relative error {abs(v_reported-v_expected)/v_expected:.2%})")
        return (True, f"v = {v_reported:.3e} matches equipartition prediction {v_expected:.3e} within 2%")

    def test_modeling_missing_particle_total_L(self):
        """
        Model the missing particle identification using total lepton number conservation.
        (Standard lepton number L: particle +1, antiparticle -1.)
        Reaction: ν_μ + e^- → μ^- + X
        Initial L_total: ν_μ(+1) + e^- (+1) = +2
        Final L_total: μ^- (+1) + L(X) = +2 → L(X) = +1 → X is a neutrino (not antineutrino).
        The problem's expected answer is ν_e.
        """
        if self.missing_particle is None:
            return (False, "SKIP: missing_particle not available")
        # Compute expected L values
        # Known particles: nu_mu L=+1, e- L=+1, mu- L=+1, antineutrino_e L=-1, etc.
        # We only need to know that X must have L=+1 (neutrino).
        # Check that reported missing_particle is a neutrino symbol (nu_e or similar)
        neutrino_symbols = {'nu_e', '\\nu_e', r'\nu_e', 'ν_e', 'electron neutrino', 'electron-neutrino',
                            'nu_mu', '\\nu_\\mu', r'\nu_\mu', 'ν_μ', 'muon neutrino',
                            'nu_tau', '\\nu_\\tau', r'\nu_\tau', 'ν_τ', 'tau neutrino'}
        if self.missing_particle.strip() not in neutrino_symbols:
            return (False, f"missing_particle = {self.missing_particle}, but from total lepton number conservation it must be a neutrino (L=+1). Expected ν_e or similar.")
        # Additionally, the problem expects ν_e specifically.
        expected_particles = {'nu_e', '\\nu_e', r'\nu_e', 'ν_e', 'electron neutrino', 'electron-neutrino'}
        if self.missing_particle.strip() not in expected_particles:
            return (False, f"missing_particle = {self.missing_particle}, but from the problem context it should be ν_e (electron neutrino).")
        return (True, "missing_particle is an electron neutrino, consistent with total lepton number conservation.")

    # ----------------------------------------------------------------------
    # Monotonicity tests: perturb an input and check that output changes in expected direction
    # ----------------------------------------------------------------------
    def test_monotonicity_velocity_increases_with_temperature(self):
        """
        Perturb temperature upward by 10% and compute new v_perturbed from equipartition.
        Check that the reported velocity is less than v_perturbed (since v ~ sqrt(T)).
        """
        if self.v is None:
            return (False, "SKIP: v not available")
        v_reported = float(self.v)
        T_base = self.T
        T_perturbed = T_base * 1.1   # 10% increase
        v_perturbed = math.sqrt(3 * self.k_B * T_perturbed / self.m_nu) * self.c
        if v_reported >= v_perturbed:
            return (False, f"v_reported = {v_reported:.3e} is NOT less than v_perturbed for T +10% = {v_perturbed:.3e}, violating monotonicity (v ~ sqrt(T))")
        return (True, f"v_reported = {v_reported:.3e} < v_perturbed = {v_perturbed:.3e}, consistent with v increasing with T")

    def test_monotonicity_velocity_decreases_with_mass(self):
        """
        Perturb mass upward by 5% and compute new v_perturbed from equipartition.
        Check that the reported velocity is greater than v_perturbed (since v ~ 1/sqrt(m)).
        """
        if self.v is None:
            return (False, "SKIP: v not available")
        v_reported = float(self.v)
        m_base = self.m_nu
        m_perturbed = m_base * 1.05   # 5% increase
        v_perturbed = math.sqrt(3 * self.k_B * self.T / m_perturbed) * self.c
        if v_reported <= v_perturbed:
            return (False, f"v_reported = {v_reported:.3e} is NOT greater than v_perturbed for m +5% = {v_perturbed:.3e}, violating monotonicity (v ~ 1/sqrt(m))")
        return (True, f"v_reported = {v_reported:.3e} > v_perturbed = {v_perturbed:.3e}, consistent with v decreasing with m")

    # ----------------------------------------------------------------------
    # Weak answer sanity (at most one such test)
    # ----------------------------------------------------------------------
    def test_answer_sanity_weak(self):
        if self.v is None:
            return (False, "SKIP: v not available")
        v = float(self.v)
        if v <= 0:
            return (False, f"v = {v}, must be positive")
        if v >= self.c:
            return (False, f"v = {v} >= c = {self.c}, unphysical")
        if v < 2.6e6 or v > 2.6e8:
            return (False, f"v = {v:.2e} seems far from expected order 2.6e7 m/s")
        return (True, f"v = {v:.2e} is positive, < c, and within range")

    # ----------------------------------------------------------------------
    # Test runner
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
                passed, msg = (False, f"{name} CRASHED: {traceback.format_exc()}")
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)